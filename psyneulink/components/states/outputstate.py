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
Note that its `variable <OutputState.variable>` must be compatible (in number and type of elements) with the item of
its owner's `value <Mechanism_Base.value>` specified by the OutputState's `index <OutputState.index>` attribute. If
its **owner* is not specified, `initialization is deferred.

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

OutputStates can also be added* to a Mechanism, using the Mechanism's `add_states <Mechanism_Base.add_methods>` method.
Unlike specification in the constructor, this **does not** replace any OutputStates already assigned to the Mechanism.
Doing so appends them to the list of OutputStates in the Mechanism's `output_states <Mechanism_Base.output_states>`
attribute, and their values are appended to its `output_values <Mechanism_Base.output_values>` attribute.  If the name
of an OutputState added to a Mechanism is the same as one that already exists, its name is suffixed with a numerical
index (incremented for each OutputState with that name; see `Naming`), and the OutputState is added to the list (that
is, it does *not* replace ones that were already created).


.. _OutputState_Variable_and_Value:

*OutputState's* `variable <OutputState.variable>`, `value <OutputState.value>`, *and Mechanism's* `value
<Mechanism_Base.value>`

Each OutputState created with or assigned to a Mechanism must reference one or more items of the Mechanism's
`value <Mechanism_Base.value>` attribute, that it uses to generate its own `value <OutputState.value>`.  The item(s)
referenced are specified by its `index <OutputState.index>` attribute.  The OutputState's `variable
<OutputState.variable>` must be compatible (in number and type of elements) with the item(s) of the Mechanism's
`value <Mechanism_Base.value>` referenced by its `index <OutputState.index>`;  by default this is '0', referring to
the first item of the Mechanism's `value <Mechanism_Base.value>`.  The OutputState's `variable
<OutputState.variable>` is used as the input to its `function <OutputState.function>`, which may modify the value
under the influence of a `GatingSignal`; the result may be further modified by the OutputState's `calculate
<OutputState.calculate>` function (e.g., to combine, compare, or otherwise evaluate the index items of the Mechanism's
`value <Mechanism_Base.value>`), before being assigned to its `value <OutputState.value>` (see
`OutputState_Customization` for additional details).  The OutputState's `value <OutputState.value>` must, in turn,
be compatible with any Projections that are assigned to it, or `used to specify it
<OutputState_Projection_Destination_Specification>`.

The `value <OutputState.value>` of each OutputState of a Mechanism is assigned to a corresponding item of the
Mechanism's `output_values <Mechanism_Base.output_values>` attribute, in the order in which they are assigned in the
**output_states** argument of its constructor, and listed in its `output_states <Mechanism_Base.output_states>`
attribute.  Note that the `output_values <Mechanism_Base.output_values>` attribute of a Mechanism is **not the same**
as its `value <Mechanism_Base.value>` attribute, which contains the full and unmodified results of the Mechanism's
`function <Mechanism_Base.function>` (since, as noted above, OutputStates  may modify the item of the Mechanism`s
`value <Mechanism_Base.value>` to which they refer).


.. _OutputState_Forms_of_Specification:

Forms of Specification
^^^^^^^^^^^^^^^^^^^^^^

OutputStates can be specified in a variety of ways, that fall into three broad categories:  specifying an OutputState
directly; use of a `State specification dictionary <State_Specification>`; or by specifying one or more Components to
which it should project. Each of these is described below:

    .. _OutputState_Direct_Specification:

    **Direct Specification of an OutputState**

    * existing **OutputState object** or the name of one -- it cannot belong to another Mechanism, and the format of
      its `variable <OutputState.variable>` must be compatible with the `indexed item <OutputState_Variable_and_Value>`
      of the owner Mechanism's `value <Mechanism_Base.value>`.
    ..
    * **OutputState class**, **keyword** *OUTPUT_STATE*, or a **string** -- creates a default OutputState that is
      assigned an `index <OutputState.index>` of '0', uses the first item of the owner Mechanism's `value
      <Mechanism_Base.value>` to format the OutputState's `variable <OutputState.variable>`, and assigns it as the
      `primary OutputState <OutputState_Primary>` for the Mechanism. If the class name or *INPUT_STATE* keyword is used,
      a default name is assigned to the State; if a string is specified, it is used as the `name <OutputState.name>` of
      the OutputState  (see `Naming`).

    .. _OutputState_Specification_by_Value:

    * **value** -- creates a default OutputState using the specified value as the OutputState's `variable
      <OutputState.variable>`.  This must be compatible with (have the same number and type of elements as)
      the item of the owner Mechanism's `value <Mechanism_Base.value>` to which the OutputState is assigned
      (the first item by default, or the one designated by its `index <OutputState.index>` attribute).  A default
      name is assigned based on the name of the Mechanism (see `Naming`).
    ..
    .. _OutputState_Specification_Dictionary:

    **OutputState Specification Dictionary**

    * **OutputState specification dictionary** -- this can be used to specify the attributes of an OutputState,
      using any of the entries that can be included in a `State specification dictionary <State_Specification>`
      (see `examples <State_Specification_Dictionary_Examples>` in State).  If the dictionary includes a *VARIABLE*
      entry, its value must be compatible with the item of the owner Mechanism's `value <Mechanism_Base.value>`
      specified for its `index <OutputState.index>` attribute ('0' by default; see
      `above <OutputState_Variable_and_Value>`)

      The *PROJECTIONS* or *MECHANISMS* entry can be used to specify one or more efferent `MappingProjections
      <MappingProjection>` from the OutputState, and/or `ModulatoryProjections <ModulatoryProjection>` for it to
      receive; however, this may be constrained by or have consequences for the OutState's `variable
      <InputState.variable>` and/or its `value <OutputState.value>` `OutputState_Compatability_and_Constraints`).

      In addition to the standard entries of a `State specification dictionary <State_Specification>`, the dictionary
      can also include either or both of the following entries specific to OutputStates:

      * *INDEX*:<int> - specifies the OutputState's `index <OutputState.index>` attribute; if this is not included,
        the first item of the owner Mechanism's `value <Mechanism_Base.value>` is assigned as the the OutputState's
        `variable <OutputState.variable>` (see `description below <OutputState_Index>` for additional details).
      |
      * *CALCULATE*:<function> - specifies the function assigned as the OutputState's `calculate
        <OutputState.calculate>` attribute;  if this is not included, an identity function is used to assign the
        OutputState's `variable <OutputState.variable>` as its `value <OutputState.value>` (see `description below
        <OutputState_Calculate>` for additional details).

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
        * **3-item tuple:** *(<value, State spec, or list of State specs>, index, Projection specification)* -- this
          allows the specification of State(s) to which the OutputState should project, together with a
          specification of its `index <OutputState.index>` attribute, and (optionally) parameters of the
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
            * **index** -- must be an integer; specifies the `index <OutputState.index>` for the OutputState.
            |
            * **Projection specification** (optional) -- `specifies a Projection <Projection_Specification>` that
              must be compatible with the State specification(s) in the 1st item; if there is more than one
              State specified, and the Projection specification is used, all of the States
              must be of the same type (i.e.,either InputStates or GatingSignals), and the `Projection
              Specification <Projection_Specification>` cannot be an instantiated Projection (since a
              Projection cannot be assigned more than one `receiver <Projection_Base.receiver>`).

.. _OutputState_Compatability_and_Constraints:

OutputState `value <OutputState.value>`: Compatibility and Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `variable <OutputState.variable>` of an OutputState must be compatible with the item of its owner Mechanism's
`value <Mechanism_Base.value>` referenced by its `index <Mechanism_Base.index>` attribute.  This may have consequences that must be taken into account when `specifying an OutputState by
Components to which it projects <_OutputState_Projection_Destination_Specification>`.  These depend on the context in
which the specification is made, and possibly the value of other specifications.  These considerations and how they
are handled are described below, starting with constraints that are given the highest precedence:

  * **OutputState specified in a Mechanism's constructor** -- the item of the Mechanism's `value <Mechanism_Base.value>`
    `indexed by the OutputState <OutputState_Index>`)>` is used to determine the OutputState's
    `variable <OutputState.variable>`.  This, together with the OutputState's `function <OutputState.function>` and
    possibly its `calculate <OutputState.calculate>` attribute, determine the OutputState's `value <OutputState.value>`
    (see `above <OutputState_Variable_and_Value>`).  Therefore, any specifications of the OutputState relevant to its
    `value <OutputState.value>` must be compatible with these factors (for example, `specifying it by value
    <OutputState_Specification_by_Value>` or by a `MappingProjection` or an `InputState` to which it should project
    (see `above <OutputState_Projection_Destination_Specification>`).

    COMMENT:
    ***XXX EXAMPLE HERE
    COMMENT
  ..
  * **OutputState specified on its own** -- any direct specification of the OutputState's `variable
    <OutputState.variable>` is used to determine its format (e.g., `specifying it by value
    <OutputState_Specification_by_Value>`, or a *VARIABLE* entry in an `OutputState specification dictionary
    <OutputState_Specification_Dictionary>`.  In this case, the value of any `Components used to specify the
    OutputState <OutputState_Projection_Destination_Specification>` must be compatible with the specification of its
    `variable <OutputState.variable>` and the consequences this has for its `value <OutputState.value>` (see below).

    COMMENT:
    ***XXX EXAMPLE HERE
    COMMENT
  ..
  * **OutputState's** `value <OutputState.value>` **not constrained by any of the conditions above** -- then its
    `variable <OutputState.variable>` is determined by the default for an OutputState (1d array of length 1).
    If it is `specified to project to any other Components <OutputState_Projection_Destination_Specification>`,
    then if the Component is a:

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

The default OutputState uses the first (and usually only) item of the owner Mechanism's `value <Mechanism_Base.value>`
as its value.  However, this can be modified in two ways, using the OutputState's `index <OutputState.index>` and
`calculate <OutputState.calculate>` attributes (see `OutputState_Structure` below). If the Mechanism's `function
<Mechanism_Base.function>` returns a value with more than one item (i.e., a list of lists, or a 2d np.array), then an
OutputState can be assigned to any of those items by specifying its `index <OutputState.index>` attribute. An
OutputState can also be configured to transform the value of the item, by specifying a function for its `calculate
<OutputState.calculate>` attribute; the result will then be assigned as the OutputState's `value <OutputState.value>`.
An OutputState's `index <OutputState.index>` and `calculate <OutputState.calculate>` attributes can be assigned when
the OutputState is assigned to a Mechanism, by including *INDEX* and *CALCULATE* entries in a `specification dictionary
<OutputState_Specification>` for the OutputState, as in the following example::


    >>> my_mech = pnl.DDM(function=pnl.BogaczEtAl(),
    ...                   output_states=[pnl.DDM_OUTPUT.DECISION_VARIABLE,
    ...                                  pnl.DDM_OUTPUT.PROBABILITY_UPPER_THRESHOLD,
    ...                                  {pnl.NAME: 'DECISION ENTROPY',
    ...                                   pnl.INDEX: 2,
    ...                                   pnl.CALCULATE: pnl.Stability(metric=pnl.ENTROPY).function }])

COMMENT:
   ADD VERSION IN WHICH INDEX IS SPECIFICED USING DDM_standard_output_states
COMMENT

In this example, ``my_mech`` is configured with three OutputStates.  The first two are `Standard OutputStates
<OutputState_Standard>` that represent the decision variable of the DDM and the probability of it crossing of the
upper (vs. lower) threshold. The third is a custom OutputState, that computes the entropy of the probability of
crossing the upper threshold. It uses the `Entropy` Function for its `calculate <OutputState.calculate>` attribute,
and *INDEX* is assigned ``2`` to reference the third item of the DDM's `value <DDM.value>` attribute (items are
indexed starting with 0), which contains the probability of crossing the upper threshold.  The three OutputStates
will be assigned to the `output_states <Mechanism_Base.output_states>` attribute of ``my_mech``, and their values
will be assigned as items in its `output_values <Mechanism_Base.output_values>` attribute, in the order in which they
are listed in the **output_states** argument of the constructor for ``my_mech``.

Custom OutputStates can also be created on their own, and separately assigned or added to a Mechanism.  For example,
the ``DECISION ENTROPY`` OutputState could be created as follows::

    >>> decision_entropy_output_state = pnl.OutputState(name='DECISION ENTROPY',
    ...                                                 index=2,
    ...                                                 calculate=pnl.Stability(metric=pnl.ENTROPY).function)

and then assigned either as::

    >>> my_mech = pnl.DDM(function=pnl.BogaczEtAl(),
    ...                   output_states=[pnl.DDM_OUTPUT.DECISION_VARIABLE,
    ...                                  pnl.DDM_OUTPUT.PROBABILITY_UPPER_THRESHOLD,
    ...                                  decision_entropy_output_state])

or::

    >>> another_decision_entropy_output_state = pnl.OutputState(name='DECISION ENTROPY',
    ...                                                index=2,
    ...                                                calculate=pnl.Stability(metric=pnl.ENTROPY).function)
    >>> my_mech2 = pnl.DDM(function=pnl.BogaczEtAl(),
    ...                    output_states=[pnl.DDM_OUTPUT.DECISION_VARIABLE,
    ...                                   pnl.DDM_OUTPUT.PROBABILITY_UPPER_THRESHOLD])

    >>> my_mech2.add_states(another_decision_entropy_output_state) # doctest: +SKIP

COMMENT:
The line after the last command is the `add_state <Mecanism_Base.add_states>` method returning the list of States
added to the Mechanism. Note, also, that another new OutputState had to be used for the second example, as trying to
add the first one created for ``my_mech`` to ``my_mech2`` would have produce an error (since a State already
belonging to one Mechanism can't be added to another.
COMMENT

Note that another new OutputState had to be used for the second example, as trying to
add the first one created for ``my_mech`` to ``my_mech2`` would have produce an error (since a State already
belonging to one Mechanism can't be added to another.


.. _OutputState_Structure:

Structure
---------

Every OutputState is owned by a `Mechanism <Mechanism>`. It can send one or more `MappingProjections
<MappingProjection>` to other Mechanisms.  If its owner is a `TERMINAL` Mechanism of a Process and/or System, then the
OutputState will also be treated as the output of that `Process <Process_Input_And_Output>` and/or of a System.  It has
the following attributes, that includes ones specific to, and that can be used to `customize, the OutputState
<OutputState_Customization>`:

.. _OutputState_Index:

* `index <OutputState.index>`: this determines the item of its owner Mechanism's `value <Mechanism_Base.value>` to
  which it is assigned.  By default, this is set to 0, which assigns it to the first item of the Mechanism's `value
  <Mechanism_Base.value>`.  The `index <Mechanism_Base.index>` must be equal to or less than one minus the number of
  OutputStates listed in the Mechanism's `output_states <Mechanism_Base.output_states>` attribute.  The `variable
  <OutputState.variable>` of the OutputState must also match (in the number and type of its elements) the item of the
  Mechanism's `value <Mechanism_Base.value>` designated by the `index <OutputState.index>`.

.. _OutputState_Calculate:

* `calculate <OutputState.calculate>`:  this specifies a function used to convert the item of the owner Mechanism's
  `value <Mechanism_Base.value>` (designated by the OutputState's `index <OutputState.index>` attribute), before
  providing it to the OutputState's `function <OutputState.function>`.  The `calculate <OutputState.calculate>`
  attribute can be assigned any function that accept the OutputState's `variable <OutputState.variable>` as its input,
  and that generates a result that can be used the input for the OutputState's `function <OutputState.function>`.
  The default is an identity function (`Linear` with **slope**\\ =1 and **intercept**\\ =0), that simply assigns the
  specified item of the Mechanism's `value <Mechanism_Base.value>` unmodified as the input for OutputState's
  `function <OutputState.function>`.
..
* `variable <OutputState.variable>` --  the value provided as the input to the OutputState's `function
  <OutputState.function>`; it must match the value of the item of its owner Mechanism's `value  <Mechanism_Base.value>`
  to which it is assigned (designated by its `index <OutputState.index>` attribute), both in the number and types of
  its elements)

* `function <OutputState.function>` -- takes the OutputState's `variable <OutputState.variable>` as its input, and
  generates the OutpuState's `value <OutputState.value>` as its result.  The default function is `Linear` that simply
  assigns the OutputState's `variable <OutputState.variable>` as its `value <OutputState.value>`.  However, the
  parameters of the `function <OutputState.function>` -- and thus the `value <OutputState.value>` of the OutputState --
  can be modified by any `GatingProjections <GatingProjection>` received by the OutputState (listed in its
  `mod_afferents <OutputState.mod_afferents>` attribute.  A custom function can also be specified, so long as it can
  take as its input a value that is compatiable with the OutputState's `variable <OutputState.variable>`.

* `projections <OutputState.projections>` -- all of the `Projections <Projection>` sent and received by the OutputState;

.. _OutputState_Effent_and_Modulatory_Projections:

* `efferents <OutputState.path_afferents>` -- `MappingProjections <MappingProjection>` that project from the
  OutputState.

* `mod_afferents <OutputState.mod_afferents>` -- `GatingProjections <GatingProjection>` that project to the OutputState,
  the `value <GatingProjection.value>` of which can modify the OutputState's `value <InputState.value>` (see the
  descriptions of Modulation under `ModulatorySignals <ModulatorySignal_Modulation>` and `GatingSignals
  <GatingSignal_Modulation>` for additional details).  If the OutputState receives more than one GatingProjection,
  their values are combined before they are used to modify the `value <OutputState.value>` of the OutputState.
..
* `value <OutputState.value>`:  assigned the result of the function specified by the
  `calculate <OutputState.calculate>` attribute, possibly modified by the result of the OutputState`s
  `function <OutputState.function>` and any `GatingProjections <GatingProjection>` received by the OutputState.
  It is used as the input to any projections that the OutputStatue sends.


.. _OutputState_Execution:

Execution
---------

An OutputState cannot be executed directly.  It is executed when the Mechanism to which it belongs is executed.
When the Mechanism is executed, it places the results of its execution in its `value <Mechanism_Base.value>`
attribute. The OutputState's `index <OutputState.index>` attribute designates the item of the Mechanism's
`value <Mechanism_Base.value>` for use by the OutputState.  The OutputState is updated by calling the function
specified by its `calculate <OutputState_Calculate>` attribute with the designated item of the Mechanism's
`value <Mechanism_Base.value>` as its input.  This is used by the Mechanism's
`function <Mechanism_Base.function>`, modified by any `GatingProjections <GatingProjection>` it receives (listed in
its `mod_afferents <OutputState.mod_afferents>` attribute), to generate the `value <OutputState.value>` of the
OutputState.  This is assigned to a corresponding item of the Mechanism's `output_values
<Mechanism_Base.output_values>` attribute, and is used as the input to any projections for which the
OutputState is the `sender <Projection_Base.sender>`.

.. _OutputState_Class_Reference:

Class Reference
---------------


"""

import numbers

import numpy as np
import typecheck as tc

from psyneulink.components.component import Component, InitStatus
from psyneulink.components.functions.function import Linear, is_function_type
from psyneulink.components.shellclasses import Mechanism, Projection
from psyneulink.components.states.state import State_Base, _instantiate_state_list, state_type_keywords, ADD_STATES
from psyneulink.globals.keywords import \
    PROJECTION, PROJECTIONS, PROJECTION_TYPE, MAPPING_PROJECTION, INPUT_STATE, INPUT_STATES, RECEIVER, GATING_SIGNAL, \
    COMMAND_LINE, STATE, OUTPUT_STATE, OUTPUT_STATES, OUTPUT_STATE_PARAMS, RESULT, INDEX, PARAMS, REFERENCE_VALUE,\
    CALCULATE, MEAN, MEDIAN, NAME, STANDARD_DEVIATION, STANDARD_OUTPUT_STATES, VARIANCE, ALL, MECHANISM_VALUE
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.globals.utilities import UtilitiesError, iscompatible, type_match, is_numeric

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

# This is a convenience class that provides list of standard_output_state names in IDE
class OUTPUTS():
    RESULT=RESULT
    MEAN=MEAN
    MEDIAN=MEDIAN
    STANDARD_DEVIATION=STANDARD_DEVIATION
    VARIANCE=VARIANCE

standard_output_states = [{NAME: RESULT},
                          {NAME:MEAN,
                           CALCULATE:lambda x: np.mean(x)},
                          {NAME:MEDIAN,
                           CALCULATE:lambda x: np.median(x)},
                          {NAME:STANDARD_DEVIATION,
                           CALCULATE:lambda x: np.std(x)},
                          {NAME:VARIANCE,
                           CALCULATE:lambda x: np.var(x)},
                          {NAME: MECHANISM_VALUE,
                           INDEX: ALL}
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
    calculate=Linear,     \
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
        a template that specifies the format of the item of the owner Mechanism's
        `value <Mechanism_Base.value>` attribute to which the OutputState will be assigned (specified by
        the **index** argument).  This must match (in number and type of elements) the OutputState's
        **variable** argument.  It is used to insure the compatibility of the source of the
        input for the OutputState with its `variable <OutputState.variable>`.

    variable : number, list or np.ndarray
        specifies the template for the OutputState's `variable <OutputState.variable>`.

    size : int, list or ndarray of ints
        specifies variable as array(s) of zeros if **variable** is not passed as an argument;
        if **variable** is specified, it takes precedence over the specification of **size**.
        As an example, the following mechanisms are equivalent::
            T1 = TransferMechanism(size = [3, 2])
            T2 = TransferMechanism(default_variable = [[0, 0, 0], [0, 0]])

    function : Function, function, or method : default Linear
        specifies the function used to transform the item of the owner Mechanism's `value <Mechanism_Base.value>`
        designated by the OutputState's `index <OutputState.index>` attribute, under the possible influence of
        `GatingProjections <GatingProjection>` received by the OutputState.

    index : int : default PRIMARY
        specifies the item of the owner Mechanism's `value <Mechanism_Base.value>` used as input for the
        function specified by the OutputState's `calculate <OutputState.calculate>` attribute, to determine the
        OutputState's `value <OutputState.value>`.

    calculate : Function, function, or method : default Linear
        specifies the function used to convert the designated item of the owner Mechanism's
        `value <Mechanism_Base.value>` (specified by the OutputState's :keyword:`index` attribute),
        before it is assigned as the OutputState's `value <OutputState.value>`.  The function must accept a value that
        has the same format (number and type of elements) as the item of the Mechanism's
        `value <Mechanism_Base.value>`.

    projections : list of Projection specifications
        species the `MappingProjection(s) <MappingProjection>` to be sent by the OutputState, and/or
        `GatingProjections(s) <GatingProjection>` to be received (see `OutputState_Projections` for additional details);
        these will be listed in its `efferents <OutputState.efferents>` and `mod_afferents <InputState.mod_afferents>`
        attributes, respectively (see `OutputState_Projections` for additional details).

    params : Dict[param keyword, param value] : default None
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
        assigned the item of the owner Mechanism's `value <Mechanism_Base.value>` specified by the
        OutputState's `index <OutputState.index>` attribute.

    index : int
        the item of the owner Mechanism's `value <Mechanism_Base.value>` used as input for the function specified by
        its `calculate <OutputState.calculate>` attribute (see `index <OutputState_Index>` for additional details).

    calculate : function or method : default Linear(slope=1, intercept=0))
        function used to convert the item of the owner Mechanism's `value <Mechanism_Base.value>` specified by
        the OutputState's `index <OutputState.index>` attribute.  The result is combined with the result of the
        OutputState's `function <OutputState.function>` to determine both the `value <OutputState.value>` of the
        OutputState, as well as the value of the corresponding item of the owner Mechanism's `output_values
        <Mechanism_Base.output_values>`. The default (`Linear`) transfers the value unmodified  (see `calculate
        <OutputState_Calculate>` for additional details)

    function : TransferFunction : default Linear(slope=1, intercept=0))
        function used to assign the result of the OutputState's `calculate <OutputState.calculate>` function,
        under the possible influence of `GatingProjections <GatingProjection>` received by the OutputState,
        to its `value <OutputState.value>`, as well as to the corresponding item of the owner's `output_values
        <Mechanism_Base.output_values>` attribute.

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

    stateAttributes = State_Base.stateAttributes | {INDEX, CALCULATE}

    connectsWith = [INPUT_STATE, GATING_SIGNAL]
    connectsWithAttribute = [INPUT_STATES]
    projectionSocket = RECEIVER
    modulators = [GATING_SIGNAL]

    class ClassDefaults(State_Base.ClassDefaults):
        variable = None

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'OutputStateCustomClassPreferences',
    #     kp<pref>: <setting>...}

    paramClassDefaults = State_Base.paramClassDefaults.copy()
    paramClassDefaults.update({PROJECTION_TYPE: MAPPING_PROJECTION})
    #endregion

    @tc.typecheck
    def __init__(self,
                 owner=None,
                 reference_value=None,
                 variable=None,
                 size=None,
                 function=Linear(),
                 index=PRIMARY,
                 calculate:is_function_type=Linear,
                 projections=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        if context is None:
            context = COMMAND_LINE
        else:
            context = self

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(index=index,
                                                  calculate=calculate,
                                                  function=function,
                                                  params=params)

        # If owner or reference_value has not been assigned, defer init to State._instantiate_projection()
        # if owner is None or reference_value is None:
        if owner is None:
            # Temporarily name OutputState
            self._assign_deferred_init_name(name, context)
            # Store args for deferred initialization
            self.init_args = locals().copy()
            self.init_args['context'] = context
            self.init_args['name'] = name
            self.init_args['projections'] = projections

            # Flag for deferred initialization
            self.init_status = InitStatus.DEFERRED_INITIALIZATION
            return

        self.reference_value = reference_value

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
                         context=context)

    def _validate_variable(self, variable, context=None):
        """Insure variable is compatible with output component of owner.function relevant to this State

        Validate variable against component of owner's value (output of owner's function)
             that corresponds to this OutputState (since that is what is used as the input to OutputState);
             this should have been provided as reference_value in the call to OutputState__init__()

        Note:
        * This method is called only if the parameterValidationPref is True

        :param variable: (anything but a dict) - variable to be validated:
        :param context: (str)
        :return none:
        """
        variable = self._update_variable(super(OutputState, self)._validate_variable(variable, context))

        self.instance_defaults.variable = self.reference_value

        # Insure that variable is compatible with (relevant item of) output value of owner's function
        # if not iscompatible(variable, self.reference_value):
        if (variable is not None
            and self.reference_value is not None
            and not iscompatible(variable, self.reference_value)):
            raise OutputStateError("Variable ({}) of OutputState for {} is not compatible with "
                                           "the output ({}) of its function".
                                           format(variable,
                                                  self.owner.name,
                                                  self.reference_value))
        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate index and calculate parameters

        Validate that index is within the range of the number of items in the owner Mechanism's ``value``,
        and that the corresponding item is a valid input to the calculate function


        """

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        if INDEX in target_set:
            # If INDEX specification is SEQUENTIAL:
            #    - can't yet determine relationship to default_value
            #    - can't yet evaluate calculate function (below)
            # so just return
            if target_set[INDEX] in {ALL, SEQUENTIAL}:
                return
            else:
                try:
                    self.owner.default_value[target_set[INDEX]]
                except IndexError:
                    raise OutputStateError("Value of \'{}\' argument for {} ({}) is greater than the number "
                                           "of items in the output_values ({}) for its owner Mechanism ({})".
                                           format(INDEX, self.name, target_set[INDEX], self.owner.default_value,
                                                  self.owner.name))

        # IMPLEMENT: VALIDATE THAT CALCULATE FUNCTION ACCEPTS VALUE CONSISTENT WITH
        #            CORRESPONDING ITEM OF OWNER MECHANISM'S VALUE
        if CALCULATE in target_set:

            try:
                if isinstance(target_set[CALCULATE], type):
                    function = target_set[CALCULATE]().function
                else:
                    function = target_set[CALCULATE]
                try:
                    index = target_set[INDEX]
                except KeyError:
                    # Assign default value for index if it was not specified
                    index = self.index
                # Default index is an index keyword (e.g., SEQUENTIAL) so can't evaluate at present
                if isinstance(index, str):
                    if not index in StandardOutputStates.keywords:
                        raise OutputStateError("Illegal keyword ({}) found in specification of index for {} of {}".
                                               format(index, self.name, self.owner.name))
                    return

                default_value_item_str = self.owner.default_value[index] if isinstance(index, int) else index
                error_msg = ("Item {} of value for {} ({}) is not compatible with "
                             "the function specified for the {} parameter of {} ({})".
                             format(index,
                                    self.owner.name,
                                    default_value_item_str,
                                    CALCULATE,
                                    self.name,
                                    target_set[CALCULATE]))
                try:
                    function(self.owner.default_value[index], context=context)
                except TypeError:
                    try:
                        function(self.owner.default_value[index])
                    except:
                        raise OutputStateError(error_msg)
                # except IndexError:
                #     # This handles cases in which index has not yet been assigned
                #     pass
                except:
                    raise OutputStateError(error_msg)
            except KeyError:
                pass

    def _validate_against_reference_value(self, reference_value):
        """Validate that State.variable is compatible with the reference_value

        reference_value is the value of the Mechanism to which the OutputState is assigned
        """
        if reference_value is not None and not iscompatible(reference_value, self.instance_defaults.variable):
            name = self.name or ""
            raise OutputStateError("Value specified for {} {} of {} ({}) is not compatible "
                                   "with its expected format ({})".
                                   format(name, self.componentName, self.owner.name, self.instance_defaults.variable, reference_value))

    # MODIFIED 11/15/17 NEW:
    def _instantiate_attributes_before_function(self, context=None):
        if self.variable is None and self.reference_value is None:
            self.instance_defaults.variable = self.owner.default_value[0]
    # MODIFIED 11/15/17 END

    def _instantiate_attributes_after_function(self, context=None):
        """Instantiate calculate function
        """
        super()._instantiate_attributes_after_function(context=context)

        if isinstance(self.calculate, type):
            self.calculate = self.calculate().function

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

        """
        from psyneulink.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base
        from psyneulink.components.states.modulatorysignals.modulatorysignal import \
            ModulatorySignal, _is_modulatory_spec
        from psyneulink.components.mechanisms.adaptive.adaptivemechanism import AdaptiveMechanism_Base
        from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
        from psyneulink.components.projections.projection import ProjectionTuple


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

    def _execute(self, function_params, context):
        """Call self.function with owner's value as variable
        """

        # Most common case is OutputState has index, so assume that for efficiency
        try:
            # Get indexed item of owner's value
            owner_val = self.owner.value[self.index]
        except IndexError:
            # Index is ALL, so use owner's entire value
            if self.index is ALL:
                owner_val = self.owner.value
            else:
                raise IndexError

        # IMPLEMENTATION NOTE: OutputStates don't currently receive PathwayProjections,
        #                      so there is no need to use their value (as do InputStates)
        value = self.function(variable=owner_val,
                                params=function_params,
                                context=context)

        return type_match(self.calculate(owner_val), type(value))

    def _get_primary_state(self, mechanism):
        return mechanism.output_state

    @tc.typecheck
    def _parse_state_specific_specs(self, owner, state_dict, state_specific_spec):
        """Get index and/or connections specified in an OutputState specification tuple

        Tuple specification can be:
            (state_spec, connections)
            (state_spec, index, connections)

        See State._parse_state_specific_spec for additional info.

        Returns:
             - state_spec:  1st item of tuple
             - params dict with INDEX and/or PROJECTIONS entries if either of them was specified

        """
        # FIX: ADD FACILITY TO SPECIFY WEIGHTS AND/OR EXPONENTS FOR INDIVIDUAL OutputState SPECS
        #      CHANGE EXPECTATION OF *PROJECTIONS* ENTRY TO BE A SET OF TUPLES WITH THE WEIGHT AND EXPONENT FOR IT
        #      THESE CAN BE USED BY THE InputState's LinearCombination Function
        #          (AKIN TO HOW THE MECHANISM'S FUNCTION COMBINES InputState VALUES)
        #      THIS WOULD ALLOW FULLY GENEREAL (HIEARCHICALLY NESTED) ALGEBRAIC COMBINATION OF INPUT VALUES
        #      TO A MECHANISM
        from psyneulink.components.projections.projection import _parse_connection_specs, ProjectionTuple
        from psyneulink.components.system import MonitoredOutputStatesOption

        params_dict = {}
        state_spec = state_specific_spec

        if isinstance(state_specific_spec, dict):
            return None, state_specific_spec

        elif isinstance(state_specific_spec, ProjectionTuple):
            # MODIFIED 11/25/17 NEW:
            state_spec = None
            # MODIFIED 11/25/17 END:
            params_dict[PROJECTIONS] = _parse_connection_specs(self,
                                                               owner=owner,
                                                               connections=[state_specific_spec])

        elif isinstance(state_specific_spec, tuple):

            tuple_spec = state_specific_spec
            state_spec = None
            INDEX_INDEX = 1

            # MODIFIED 11/23/17 NEW:
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
                # # MODIFIED 11/28/17 OLD:
                projection_spec = tuple_spec[1]
                # MODIFIED 11/28/17 NEW:
                # projection_spec =
                # MODIFIED 11/28/17 END:
            # MODIFIED 11/23/17 END

            # MODIFIED 11/23/17 NEW: ADDED ELSE AND INDENTED
            else:
                projection_spec = state_specific_spec if len(state_specific_spec)==2 else (state_specific_spec[0],
                                                                                           state_specific_spec[-1])

            if not len(tuple_spec) in {2,3} :
                raise OutputStateError("Tuple provided in {} specification dictionary for {} ({}) must have "
                                       "either 2 ({} and {}) or 3 (optional additional {}) items, "
                                       "or must be a {}".
                                       format(OutputState.__name__, owner.name, tuple_spec,
                                              STATE, PROJECTION, INDEX, ProjectionTuple.__name__))


            params_dict[PROJECTIONS] = _parse_connection_specs(connectee_state_type=self,
                                                               owner=owner,
                                                               connections=projection_spec)


            # Get INDEX specification from (state_spec, index, connections) tuple:
            if len(tuple_spec) == 3:

                index = tuple_spec[INDEX_INDEX]

                if index is not None and not isinstance(index, numbers.Number):
                    raise OutputStateError("The {} (2nd) item of the {} specification tuple for {} ({}) "
                                           "must be a number".format(INDEX, OutputState.__name__, owner.name, index))
                try:
                    owner.default_value[index]
                except IndexError:
                    raise OutputStateError("The {0} (2nd) item of the {1} specification tuple for {2} ({3}) is out "
                                           "of bounds for the number of items in {4}'s value ({5}, max index: {6})".
                                           format(INDEX, OutputState.__name__, owner.name, index,
                                                  owner.name, owner.default_value, len(owner.default_value)-1))
                params_dict[INDEX] = index

        elif state_specific_spec is not None:
            raise OutputStateError("PROGRAM ERROR: Expected tuple or dict for {}-specific params but, got: {}".
                                  format(self.__class__.__name__, state_specific_spec))

        return state_spec, params_dict

    @property
    def pathway_projections(self):
        return self.efferents

    @pathway_projections.setter
    def pathway_projections(self, assignment):
        self.efferents = assignment


def _instantiate_output_states(owner, output_states=None, context=None):
    """Call State._instantiate_state_list() to instantiate ContentAddressableList of OutputState(s)

    Create ContentAddressableList of OutputState(s) specified in paramsCurrent[OUTPUT_STATES]

    If output_states is not specified:
        - use owner.output_states as list of OutputState specifications
        - if owner.output_states is empty, use owner.value to create a default OutputState

    For each OutputState:
         check for index param:
             if it is a State, get from index attribute
             if it is dict, look for INDEX entry
             if it is anything else, assume index is PRIMARY
         get indexed value from output.value
         append the indexed value to reference_value
             so that it matches specification of OutputStates (by # and function return values)
         instantiate Calculate function if specified

    When completed:
        - self.output_states contains a ContentAddressableList of one or more OutputStates;
        - self.output_state contains first or only OutputState in list;
        - paramsCurrent[OUTPUT_STATES] contains the same ContentAddressableList (of one or more OutputStates)
        - each OutputState corresponds to an item in the output of the owner's function
        - if there is only one OutputState, it is assigned the full value

    (See State._instantiate_state_list() for additional details)

    IMPLEMENTATION NOTE:
        default(s) for self.paramsCurrent[OUTPUT_STATES] (self.value) are assigned here
        rather than in _validate_params, as it requires function to have been instantiated first

    Returns list of instantiated OutputStates
    """

    reference_value = []

    # Get owner.value
    # IMPLEMENTATION NOTE:  ?? IS THIS REDUNDANT WITH SAME TEST IN Mechanism.execute ?  JUST USE RETURN VALUE??
    owner_value = owner.default_value

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
            owner_value = owner.default_value
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

              # parse output_state
            from psyneulink.components.states.state import _parse_state_spec
            output_state = _parse_state_spec(state_type=OutputState, owner=owner, state_spec=output_state)

            # Default is PRIMARY
            index = PRIMARY
            output_state_value = owner_value[index]

            # OutputState object, so get its index attribute
            if isinstance(output_state, OutputState):
                index = output_state.index
                output_state_value = owner_value[index]

            # OutputState specification dictionary, so get attributes
            elif isinstance(output_state, dict):

                # If OutputState's name matches the name entry of a dict in standard_output_states,
                #    use the named Standard OutputState
                if output_state[NAME] and hasattr(owner, STANDARD_OUTPUT_STATES):
                    std_output_state = owner.standard_output_states.get_state_dict(output_state[NAME])
                    if std_output_state is not None:
                        # If any params were specified for the OutputState, add them to std_output_state
                        if PARAMS in output_state and output_state[PARAMS] is not None:
                            std_output_state.update(output_state[PARAMS])
                        output_states[i] = std_output_state

                if output_state[PARAMS]:
                    # If OutputState's index is specified, use it
                    if INDEX in output_state[PARAMS]:
                        index = output_state[PARAMS][INDEX]

                    # If OutputState's calculate function is specified, use it to determine OutputState's vaue
                    if CALCULATE in output_state[PARAMS]:
                        output_state_value = output_state[PARAMS][CALCULATE](owner_value[index], context=context)
                    else:
                        output_state_value = owner_value[index]

            else:
                if not isinstance(output_state, str):
                    raise OutputStateError("PROGRAM ERROR: unrecognized item ({}) in output_states specification for {}"
                                           .format(output_state, owner.name))

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
    if any(keyword in context for keyword in {COMMAND_LINE, ADD_STATES}):
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
    """Collection of OutputState specification dictionaries for `standard
    OutputStates <OutputState_Standard>` of a class

    Arguments
    ---------
    owner : Component
        the Component to which this OutputState belongs

    output_state_dicts : list of dicts
        list of dictionaries specifying OutputStates for the Component specified by `owner`

    indices : PRIMARY,
    SEQUENTIAL, list of ints
        specifies how to assign the INDEX entry for each dict listed in `output_state_dicts`;

        The effects of each value of indices are as follows:

            * *PRIMARY* -- assigns the INDEX for the owner's primary OutputState to all output_states
              for which an INDEX entry is not already specified;

            * *SEQUENTIAL* -- assigns sequentially incremented int to each INDEX entry,
              ignoring any INDEX entries previously specified for individual OutputStates;

            * list of ints -- assigns each int to the corresponding entry in `output_state_dicts`;
              ignoring any INDEX entries previously specified for individual OutputStates;

            * None -- assigns `None` to INDEX entries for all OutputStates for which it is not already specified.

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

        # Validate that all items in output_state_dicts are dicts
        for item in output_state_dicts:
            if not isinstance(item, dict):
                raise StandardOutputStatesError("All items of {} for {} must be dicts (but {} is not)".
                                     format(self.__class__.__name__, owner.componentName, item))
        self.data = output_state_dicts.copy()

        # Assign indices

        # List was provided, so check that:
        # - it has the appropriate number of items
        # - they are all ints
        # and then assign each int to the INDEX entry in the corresponding dict in output_state_dicts
        # OutputState
        if isinstance(indices, list):
            if len(indices) != len(output_state_dicts):
                raise StandardOutputStatesError("Length of the list of indices provided to {} for {} ({}) "
                                       "must equal the number of OutputStates dicts provided ({})"
                                       "length".format(self.__class__.__name__,
                                                       owner.name,
                                                       len(indices),
                                                       len(output_state_dicts)))

            if not all(isinstance(item, int) for item in indices):
                raise StandardOutputStatesError("All the items in the list of indices provided to {} for {} of {}) "
                                               "must be ints".
                                               format(self.__class__.__name__, self.name, owner.name))

            for index, state_dict in zip(indices, self.data):
                state_dict[INDEX] = index

        # Assign indices sequentially based on order of items in output_state_dicts arg
        elif indices is SEQUENTIAL:
            for index, state_dict in enumerate(self.data):
                state_dict[INDEX] = index

        # Assign PRIMARY as INDEX for all OutputStates in output_state_dicts that don't already have an index specified
        elif indices is PRIMARY:
            for state_dict in self.data:
                if INDEX in state_dict:
                    continue
                state_dict[INDEX] = PRIMARY

        # No indices specification, so assign None to INDEX for all OutputStates in output_state_dicts
        #  that don't already have an index specified
        else:
            for state_dict in self.data:
                if INDEX in state_dict:
                    continue
                state_dict[INDEX] = None


        # Add names of each OutputState as property of the owner's class that returns its name string
        for state in self.data:
            setattr(owner.__class__, state[NAME], make_readonly_property(state[NAME]))

        # Add <NAME_INDEX> of each OutputState as property of the owner's class, that returns its index
        for state in self.data:
            setattr(owner.__class__, state[NAME]+'_INDEX', make_readonly_property(state[INDEX]))

    @tc.typecheck
    def get_state_dict(self, name:str):
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

    @property
    def indices(self):
        return [item[INDEX] for item in self.data]


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
