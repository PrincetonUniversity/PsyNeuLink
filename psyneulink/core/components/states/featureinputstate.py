# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *******************************************  FeatureInputState *****************************************************
#
"""

Overview
--------

The purpose of a FeatureInputState is to receive and combine inputs to a `Mechanism`, allow them to be modified, and provide
them to the Mechanism's `function <Mechanism_Base.function>`. A FeatureInputState receives input to a `Mechanism`
provided by the `Projections <Projection>` to that Mechanism from others in a `Process` or `System`.  If the
FeatureInputState belongs to an `ORIGIN` Mechanism (see `role of Mechanisms in Processes and Systems
<Mechanism_Role_In_Processes_And_Systems>`), then it receives the input specified when that Process or System is
`run <Run>`.  The `PathwayProjections <PathWayProjection>` received by a FeatureInputState are listed in its `path_afferents
<FeatureInputState.path_afferents>`, and its `ModulatoryProjections <ModulatoryProjection>` in its `mod_afferents
<FeatureInputState.mod_afferents>` attribute.  Its `function <FeatureInputState.function>` combines the values received from its
PathWayProjections, modifies the combined value according to value(s) any ModulatoryProjections it receives, and
provides the result to the assigned item of its owner Mechanism's `variable <Mechanism_Base.variable>` and
`input_values <Mechanism_Base.input_values>` attributes (see `below` and `Mechanism FeatureInputStates <Mechanism_FeatureInputStates>`
for additional details about the role of FeatureInputStates in Mechanisms, and their assignment to the items of a Mechanism's
`variable <Mechanism_Base.variable>` attribute).

.. _FeatureInputState_Creation:

Creating a FeatureInputState
----------------------

A FeatureInputState can be created by calling its constructor, but in general this is not necessary as a `Mechanism` can
usually automatically create the FeatureInputState(s) it needs when it is created.  For example, if the Mechanism is
being created within the `pathway <Process.pathway>` of a `Process`, its FeatureInputState is created and  assigned as the
`receiver <MappingProjection.receiver>` of a `MappingProjection` from the  preceding Mechanism in the `pathway
<Process.pathway>`.  FeatureInputStates can also be specified in the **input_states** argument of a Mechanism's
constructor (see `below <FeatureInputState_Specification>`).

The `variable <FeatureInputState.variable>` of a FeatureInputState can be specified using the **variable** or **size** arguments of
its constructor.  It can also be specified using the **projections** argument, if neither **variable** nor **size** is
specified.  The **projections** argument is used to `specify Projections <State_Projections>` to the FeatureInputState. If
neither the **variable** nor **size** arguments is specified, then the value of the `Projections(s) <Projection>` or
their `sender <Projection_Base.sender>`\\s (all of which must be the same length) is used to determine the `variable
<FeatureInputState.variable>` of the FeatureInputState.

If a FeatureInputState is created using its constructor, and a Mechanism is specified in the **owner** argument,
it is automatically assigned to that Mechanism.  Note that its `value <FeatureInputState.value>` (generally determined
by the size of its `variable <FeatureInputState.variable>` -- see `below <FeatureInputState_Variable_and_Value>`) must
be compatible (in number and type of elements) with the item of its owner's `variable <Mechanism_Base.variable>` to
which it is assigned (see `below <FeatureInputState_Variable_and_Value>` and `Mechanism <Mechanism_Variable_and_FeatureInputStates>`).
If the **owner** argument is not specified, `initialization <State_Deferred_Initialization>` is deferred.

.. _FeatureInputState_Deferred_Initialization:

*Owner Assignment and Deferred Initialization*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A FeatureInputState must be owned by a `Mechanism <Mechanism>`.  When FeatureInputState is specified in the constructor for a
Mechanism (see `below <FeatureInputState_Specification>`), it is automatically assigned to that Mechanism as its owner. If
the FeatureInputState is created on its own, its `owner <FeatureInputState.owner>` can specified in the **owner**  argument of its
constructor, in which case it is assigned to that Mechanism. If its **owner** argument is not specified, its
initialization is `deferred <State_Deferred_Initialization>` until
COMMENT:
TBI: its `owner <State_Base.owner>` attribute is assigned or
COMMENT
the FeatureInputState is assigned to a Mechanism using the Mechanism's `add_states <Mechanism_Base.add_states>` method.

.. _FeatureInputState_Primary:

*Primary FeatureInputState*
~~~~~~~~~~~~~~~~~~~~~

Every Mechanism has at least one FeatureInputState, referred to as its *primary FeatureInputState*.  If FeatureInputStates are not
`explicitly specified <FeatureInputState_Specification>` for a Mechanism, a primary FeatureInputState is automatically created
and assigned to its `input_state <Mechanism_Base.input_state>` attribute (note the singular), and also to the first
entry of the Mechanism's `input_states <Mechanism_Base.input_states>` attribute (note the plural).  The `value
<FeatureInputState.value>` of the primary FeatureInputState is assigned as the first (and often only) item of the Mechanism's
`variable <Mechanism_Base.variable>` and `input_values <Mechanism_Base.input_values>` attributes.

.. _FeatureInputState_Specification:

*FeatureInputState Specification*
~~~~~~~~~~~~~~~~~~~~~~~~~~

Specifying FeatureInputStates when a Mechanism is created
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FeatureInputStates can be specified for a `Mechanism <Mechanism>` when it is created, in the **input_states** argument of the
Mechanism's constructor (see `examples <State_Constructor_Argument_Examples>` in State), or in an *INPUT_STATES* entry
of a parameter dictionary assigned to the constructor's **params** argument.  The latter takes precedence over the
former (that is, if an *INPUT_STATES* entry is included in the parameter dictionary, any specified in the
**input_states** argument are ignored).

    .. _FeatureInputState_Replace_Default_Note:

    .. note::
       Assigning FeatureInputStates to a Mechanism in its constructor **replaces** any that are automatically generated for
       that Mechanism (i.e., those that it creates for itself by default).  If any of those are needed, they must be
       explicitly specified in the list assigned to the **input_states** argument, or the *INPUT_STATES* entry of the
       parameter dictionary in the **params** argument.  The number of FeatureInputStates specified must also be equal to
       the number of items in the Mechanism's `variable <Mechanism_Base.variable>` attribute.

.. _FeatureInputState_Variable_and_Value:

*FeatureInputState's* `variable <FeatureInputState.variable>`, `value <FeatureInputState.value>` *and Mechanism's* `variable <Mechanism_Base.variable>`
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Each FeatureInputState specified in the **input_states** argument of a Mechanism's constructor must correspond to an item of
the Mechanism's `variable <Mechanism_Base.variable>` attribute (see `Mechanism <Mechanism_Variable_and_FeatureInputStates>`),
and the `value <FeatureInputState.value>` of the FeatureInputState must be compatible with that item (that is, have the same number
and type of elements).  By default, this is also true of the FeatureInputState's `variable <FeatureInputState.variable>` attribute,
since the default `function <FeatureInputState.function>` for a FeatureInputState is a `LinearCombination`, the purpose of which
is to combine the inputs it receives and possibly modify the combined value (under the influence of any
`ModulatoryProjections <ModulatoryProjection>` it receives), but **not mutate its form**. Therefore, under most
circumstances, both the `variable <FeatureInputState.variable>` of a FeatureInputState and its `value <FeatureInputState.value>` should
match the item of its owner's `variable <Mechanism_Base.variable>` to which the FeatureInputState is assigned.

The format of a FeatureInputState's `variable <FeatureInputState.variable>` can be specified in a variety of ways.  The most
straightforward is in the **variable** argument of its constructor.  More commonly, however, it is determined by
the context in which it is being created, such as the specification for its owner Mechanism's `variable
<Mechanism_Base.variable>` or for the FeatureInputState in the Mechanism's **input_states** argument (see `below
<FeatureInputState_Forms_of_Specification>` and `Mechanism FeatureInputState specification <Mechanism_FeatureInputState_Specification>`
for details).


Adding FeatureInputStates to a Mechanism after it is created
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FeatureInputStates can also be **added** to a Mechanism, either by creating the FeatureInputState on its own, and specifying the
Mechanism in the FeatureInputState's **owner** argument, or by using the Mechanism's `add_states <Mechanism_Base.add_states>`
method (see `examples <State_Create_State_Examples>` in State).

    .. _FeatureInputState_Add_State_Note:

    .. note::
       Adding FeatureInputStates *does not replace* any that the Mechanism generates by default;  rather they are added to the
       Mechanism, and appended to the list of FeatureInputStates in its `input_states <Mechanism_Base>` attribute. Importantly,
       the Mechanism's `variable <Mechanism_Base.variable>` attribute is extended with items that correspond to the
       `value <FeatureInputState.value>` attribute of each added FeatureInputState.  This may affect the relationship of the
       Mechanism's `variable <Mechanism_Base.variable>` to its `function <Mechanism_Base.function>`, as well as the
       number of its `OutputStates <OutputState>` (see `note <Mechanism_Add_FeatureInputStates_Note>`).

If the name of a FeatureInputState added to a Mechanism is the same as one that already exists, its name is suffixed with a
numerical index (incremented for each FeatureInputState with that name; see `Naming`), and the FeatureInputState is added to the
list (that is, it will *not* replace ones that already exist).

.. _FeatureInputState_Forms_of_Specification:

Forms of Specification
^^^^^^^^^^^^^^^^^^^^^^

FeatureInputStates can be specified in a variety of ways, that fall into three broad categories:  specifying a FeatureInputState
directly; use of a `State specification dictionary <State_Specification>`; or by specifying one or more Components that
should project to the FeatureInputState. Each of these is described below:

    .. _FeatureInputState_Direct_Specification:

    **Direct Specification of a FeatureInputState**

    * existing **FeatureInputState object** or the name of one -- it can not already belong to another Mechanism and, if used
      to specify a FeatureInputState in the constructor for a Mechanism, its `value <FeatureInputState.value>` must be compatible
      with the corresponding item of the owner Mechanism's `variable <Mechanism_Base.variable>` (see `Mechanism
      FeatureInputState specification <Mechanism_FeatureInputState_Specification>` and `FeatureInputState_Compatability_and_Constraints`
      below).
    ..
    * **FeatureInputState class**, **keyword** *INPUT_STATE*, or a **string** -- this creates a default FeatureInputState; if used
      to specify a FeatureInputState in the constructor for a Mechanism, the item of the owner Mechanism's `variable
      <Mechanism_Base.variable>` to which the FeatureInputState is assigned is used as the format for the FeatureInputState`s
      `variable <FeatureInputState.variable>`; otherwise, the default for the FeatureInputState is used.  If a string is specified,
      it is used as the `name <FeatureInputState.name>` of the FeatureInputState (see `example
      <State_Constructor_Argument_Examples>`).

    .. _FeatureInputState_Specification_by_Value:

    * **value** -- this creates a default FeatureInputState using the specified value as the FeatureInputState's `variable
      <FeatureInputState.variable>`; if used to specify a FeatureInputState in the constructor for a Mechanism, the format must be
      compatible with the corresponding item of the owner Mechanism's `variable <Mechanism_Base.variable>` (see
      `Mechanism FeatureInputState specification <Mechanism_FeatureInputState_Specification>`, `example
      <State_Value_Spec_Example>`, and discussion `below <FeatureInputState_Compatability_and_Constraints>`).

    .. _FeatureInputState_Specification_Dictionary:

    **FeatureInputState Specification Dictionary**

    * **FeatureInputState specification dictionary** -- this can be used to specify the attributes of a FeatureInputState, using
      any of the entries that can be included in a `State specification dictionary <State_Specification>` (see
      `examples <State_Specification_Dictionary_Examples>` in State).  If the dictionary is used to specify an
      FeatureInputState in the constructor for a Mechanism, and it includes a *VARIABLE* and/or *VALUE* or entry, the value
      must be compatible with the item of the owner Mechanism's `variable <Mechanism_Base.variable>` to which the
      FeatureInputState is assigned (see `Mechanism FeatureInputState specification <Mechanism_FeatureInputState_Specification>`).

      The *PROJECTIONS* entry can include specifications for one or more States, Mechanisms and/or Projections that
      should project to the FeatureInputState (including both `MappingProjections <MappingProjection>` and/or
      `ModulatoryProjections <ModulatoryProjection>`; however, this may be constrained by or have consequences for the
      FeatureInputState's `variable <FeatureInputState.variable>` (see `FeatureInputState_Compatability_and_Constraints`).

      In addition to the standard entries of a `State specification dictionary <State_Specification>`, the dictionary
      can also include either or both of the following entries specific to FeatureInputStates:

      * *WEIGHT*:<number>
          the value must be an integer or float, and is assigned as the value of the FeatureInputState's `weight
          <FeatureInputState.weight>` attribute (see `weight and exponent <FeatureInputState_Weights_And_Exponents>`);
          this takes precedence over any specification in the **weight** argument of the FeatureInputState's constructor.
      |
      * *EXPONENT*:<number>
          the value must be an integer or float, and is assigned as the value of the FeatureInputState's `exponent
          <FeatureInputState.exponent>` attribute (see `weight and exponent <FeatureInputState_Weights_And_Exponents>`);
          this takes precedence over any specification in the **exponent** argument of the FeatureInputState's constructor.

    .. _FeatureInputState_Projection_Source_Specification:

    **Specification of a FeatureInputState by Components that Project to It**

    COMMENT:
    `examples
      <State_Projections_Examples>` in State)
    COMMENT

    COMMENT:
    ?? PUT IN ITS OWN SECTION ABOVE OR BELOW??
    Projections to a FeatureInputState can be specified either as attributes, in the constructor for an
    FeatureInputState (in its **projections** argument or in the *PROJECTIONS* entry of an `FeatureInputState specification dictionary
    <FeatureInputState_Specification_Dictionary>`), or used to specify the FeatureInputState itself (using one of the
    `FeatureInputState_Forms_of_Specification` described above. See `State Projections <State_Projections>` for additional
    details concerning the specification of
    Projections when creating a State.
    COMMENT

    A FeatureInputState can also be specified by specifying one or more States, Mechanisms or Projections that should project
    to it, as described below.  Specifying a FeatureInputState in this way creates both the FeatureInputState and any of the
    specified or implied Projection(s) to it (if they don't already exist). `MappingProjections <MappingProjection>`
    are assigned to the FeatureInputState's `path_afferents <FeatureInputState.path_afferents>` attribute, and `GatingProjections
    <GatingProjection>` to its `mod_afferents <FeatureInputState.mod_afferents>` attribute. Any of the following can be used
    to specify a FeatureInputState by the Components that projection to it (see `below
    <FeatureInputState_Compatability_and_Constraints>` for an explanation of the relationship between the `value` of these
    Components and the FeatureInputState's `variable <FeatureInputState.variable>`):

    * **OutputState, GatingSignal, Mechanism, or list with any of these** -- creates a FeatureInputState with Projection(s)
      to it from the specified State(s) or Mechanism(s).  For each Mechanism specified, its `primary OutputState
      <OutputState_Primary>` (or GatingSignal) is used.
    ..
    * **Projection** -- any form of `Projection specification <Projection_Specification>` can be
      used;  creates a FeatureInputState and assigns it as the Projection's `receiver <Projection_Base.receiver>`.

    .. _FeatureInputState_Tuple_Specification:

    * **FeatureInputState specification tuples** -- these are convenience formats that can be used to compactly specify an
      FeatureInputState and Projections to it any of the following ways:

        .. _FeatureInputState_State_Mechanism_Tuple:

        * **2-item tuple:** *(<State name or list of State names>, <Mechanism>)* -- 1st item must be the name of an
          `OutputState` or `ModulatorySignal`, or a list of such names, and the 2nd item must be the Mechanism to
          which they all belong.  Projections of the relevant types are created for each of the specified States
          (see `State 2-item tuple <State_2_Item_Tuple>` for additional details).
        |
        * **2-item tuple:** *(<value, State specification, or list of State specs>, <Projection specification>)* --
          this is a contracted form of the 4-item tuple described below;
        |
        * **4-item tuple:** *(<value, State spec, or list of State specs>, weight, exponent, Projection specification)*
          -- this allows the specification of State(s) that should project to the FeatureInputState, together with a
          specification of the FeatureInputState's `weight <FeatureInputState.weight>` and/or `exponent <FeatureInputState.exponent>`
          attributes of the FeatureInputState, and (optionally) the Projection(s) to it.  This can be used to compactly
          specify a set of States that project the FeatureInputState, while using the 4th item to determine its variable
          (e.g., using the matrix of the Projection specification) and/or attributes of the Projection(s) to it. Each
          tuple must have at least the following first three items (in the order listed), and can include the fourth:

            |
            * **value, State specification, or list of State specifications** -- specifies either the `variable
              <FeatureInputState.variable>` of the FeatureInputState, or one or more States that should project to it.  The State
              specification(s) can be a (State name, Mechanism) tuple (see above), and/or include Mechanisms (in which
              case their `primary OutputState <OutputStatePrimary>` is used.  All of the State specifications must be
              consistent with (that is, their `value <State_Base.value>` must be compatible with the `variable
              <Projection_Base.variable>` of) the Projection specified in the fourth item if that is included;
            |
            * **weight** -- must be an integer or a float; multiplies the `value <FeatureInputState.value>` of the FeatureInputState
              before it is combined with others by the Mechanism's `function <Mechanism.function>` (see
              ObjectiveMechanism for `examples <ObjectiveMechanism_Weights_and_Exponents_Example>`);
            |
            * **exponent** -- must be an integer or float; exponentiates the `value <FeatureInputState.value>` of the
              FeatureInputState before it is combined with others by the ObjectiveMechanism's `function
              <ObjectiveMechanism.function>` (see ObjectiveMechanism for `examples
              <ObjectiveMechanism_Weights_and_Exponents_Example>`);
            |
            * **Projection specification** (optional) -- `specifies a Projection <Projection_Specification>` that
              must be compatible with the State specification(s) in the 1st item; if there is more than one State
              specified, and the Projection specification is used, all of the States
              must be of the same type (i.e.,either OutputStates or GatingSignals), and the `Projection
              Specification <Projection_Specification>` cannot be an instantiated Projection (since a
              Projection cannot be assigned more than one `sender <Projection_Base.sender>`).

.. _FeatureInputState_Compatability_and_Constraints:

FeatureInputState `variable <FeatureInputState.variable>`: Compatibility and Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `variable <FeatureInputState.variable>` of a FeatureInputState must be compatible with the item of its owner Mechanism's
`variable <Mechanism_Base.variable>` to which it is assigned (see `Mechanism_Variable_and_FeatureInputStates>`). This may
have consequences that must be taken into account when `specifying a FeatureInputState by Components that project to it
<FeatureInputState_Projection_Source_Specification>`.  These depend on the context in which the specification is made, and
possibly the value of other specifications.  These considerations and how they are handled are described below,
starting with constraints that are given the highest precedence:

  *  **FeatureInputState is** `specified in a Mechanism's constructor <Mechanism_FeatureInputState_Specification>` and the
    **default_variable** argument for the Mechanism is also specified -- the item of the variable to which the
    `FeatureInputState is assigned <Mechanism_Variable_and_FeatureInputStates>` is used to determine the FeatureInputState's `variable must
    <FeatureInputState.variable>`.  Any other specifications of the FeatureInputState relevant to its `variable <FeatureInputState.variable>`
    must be compatible with this (for example, `specifying it by value <FeatureInputState_Specification_by_Value>` or by a
    `MappingProjection` or `OutputState` that projects to it (see `above <FeatureInputState_Projection_Source_Specification>`).

    COMMENT:
    ***XXX EXAMPLE HERE
    COMMENT
  ..
  * **FeatureInputState is specified on its own**, or the **default_variable** argument of its Mechanism's constructor
    is not specified -- any direct specification of the FeatureInputState's `variable <FeatureInputState.variable>` is used to
    determine its format (e.g., `specifying it by value <FeatureInputState_Specification_by_Value>`, or a *VARIABLE* entry
    in an `FeatureInputState specification dictionary <FeatureInputState_Specification_Dictionary>`.  In this case, the value of any
    `Components used to specify the FeatureInputState <FeatureInputState_Projection_Source_Specification>` that are relevant to its
    `variable <FeatureInputState.variable>` must be compatible with it (see below).

    COMMENT:
    ***XXX EXAMPLE HERE
    COMMENT
  ..
  * If the FeatureInputState's `variable <FeatureInputState.variable>` is not constrained by any of the conditions above,
    then its format is determined by the `specification of Components that project to it
    <FeatureInputState_Projection_Source_Specification>`:

    * **More than one Component is specified with the same :ref:`value` format** -- that format is used to determine
      the format of the FeatureInputState's `variable <FeatureInputState.variable>`.
    |
    * **More than one Component is specified with different :ref:`value` formats** -- the FeatureInputState's `variable
      <FeatureInputState.variable>` is determined by item of the default `variable <Mechanism_Base.variable>` for
      the class of its owner Mechanism.
    |
    * **A single Component is specified** -- its :ref:`value` is used to determine the format of the FeatureInputState's
      `variable <FeatureInputState.variable>`;  if the Component is a(n):

      * **MappingProjection** -- can be specified by its class, an existing MappingProjection, or a matrix:

        * `MappingProjection` **class** -- a default value is used both the for the FeatureInputState's `variable
          <FeatureInputState.variable>` and the Projection's `value <Projection_Base.value>` (since the Projection's
          `sender <Projection_Base.sender>` is unspecified, its `initialization is deferred
          <Projection_Deferred_Initialization>`.
        |
        * **Existing MappingProjection** -- then its `value <Projection_Base.value>` determines the
          FeatureInputState's `variable <FeatureInputState.variable>`.
        |
        * `Matrix specification <Mapping_Matrix_Specification>` -- its receiver dimensionality determines the format
          of the FeatureInputState's `variable <FeatureInputState.variable>`. For a standard 2d "weight" matrix (i.e., one that maps
          a 1d array from its `sender <Projection_Base.sender>` to a 1d array of its `receiver
          <Projection_Base.receiver>`), the receiver dimensionality is its outer dimension (axis 1, or its number of
          columns).  However, if the `sender <Projection_Base.sender>` has more than one dimension, then the
          dimensionality of the receiver (used for the FeatureInputState's `variable <FeatureInputState.variable>`) is the
          dimensionality of the matrix minus the dimensionality of the sender's `value <OutputState.value>`
          (see `matrix dimensionality <Mapping_Matrix_Dimensionality>`).
      |
      * **OutputState or ProcessingMechanism** -- the `value <OutputState.value>` of the OutputState (if it is a
        Mechanism, then its `primary OutputState <OutputState_Primary>`) determines the format of the FeatureInputState's
        `variable <FeatureInputState.variable>`, and a MappingProjection is created from the OutputState to the FeatureInputState
        using an `IDENTITY_MATRIX`.  If the FeatureInputState's `variable <FeatureInputState.variable>` is constrained (as in some
        of the cases above), then a `FULL_CONNECTIVITY_MATRIX` is used which maps the shape of the OutputState's `value
        <OutputState.value>` to that of the FeatureInputState's `variable <FeatureInputState.variable>`.
      |
      * **GatingProjection, GatingSignal or GatingMechanism** -- any of these can be used to specify a FeatureInputState;
        their `value` does not need to be compatible with the FeatureInputState's `variable <FeatureInputState.variable>`, however
        it does have to be compatible with the `modulatory parameter <Function_Modulatory_Params>` of the FeatureInputState's
        `function <FeatureInputState.function>`.

.. _FeatureInputState_Structure:

Structure
---------

Every FeatureInputState is owned by a `Mechanism <Mechanism>`. It can receive one or more `MappingProjections
<MappingProjection>` from other Mechanisms, as well as from the Process or System to which its owner belongs (if it
is the `ORIGIN` Mechanism for that Process or System).  It has the following attributes, that includes ones specific
to, and that can be used to customize the FeatureInputState:

* `projections <OutputState.projections>` -- all of the `Projections <Projection>` received by the FeatureInputState.

.. _FeatureInputState_Afferent_Projections:

* `path_afferents <FeatureInputState.path_afferents>` -- `MappingProjections <MappingProjection>` that project to the
  FeatureInputState, the `value <MappingProjection.value>`\\s of which are combined by the FeatureInputState's `function
  <FeatureInputState.function>`, possibly modified by its `mod_afferents <FeatureInputState_mod_afferents>`, and assigned to the
  corresponding item of the owner Mechanism's `variable <Mechanism_Base.variable>`.
..
* `mod_afferents <FeatureInputState_mod_afferents>` -- `GatingProjections <GatingProjection>` that project to the FeatureInputState,
  the `value <GatingProjection.value>` of which can modify the FeatureInputState's `value <FeatureInputState.value>` (see the
  descriptions of Modulation under `ModulatorySignals <ModulatorySignal_Modulation>` and `GatingSignals
  <GatingSignal_Modulation>` for additional details).  If the FeatureInputState receives more than one GatingProjection,
  their values are combined before they are used to modify the `value <FeatureInputState.value>` of FeatureInputState.

.. _FeatureInputState_Variable:

* `variable <FeatureInputState.variable>` -- serves as the template for the `value <Projection_Base.value>` of the
  `Projections <Projection>` received by the FeatureInputState:  each must be compatible with (that is, match both the
  number and type of elements of) the FeatureInputState's `variable <FeatureInputState.variable>`. In general, this must also be
  compatible with the item of the owner Mechanism's `variable <Mechanism_Base.variable>` to which the FeatureInputState is
  assigned (see `above <FeatureInputState_Variable_and_Value>` and `Mechanism FeatureInputState
  specification <Mechanism_FeatureInputState_Specification>`).

.. _FeatureInputState_Function:

* `function <FeatureInputState.function>` -- aggregates the `value <Projection_Base.value>` of all of the
  `Projections <Projection>` received by the FeatureInputState, and assigns the result to the FeatureInputState's `value
  <FeatureInputState.value>` attribute.  The default function is `LinearCombination` that performs an elementwise (Hadamard)
  sums the values. However, the parameters of the `function <FeatureInputState.function>` -- and thus the `value
  <FeatureInputState.value>` of the FeatureInputState -- can be modified by any `GatingProjections <GatingProjection>` received by
  the FeatureInputState (listed in its `mod_afferents <FeatureInputState.mod_afferents>` attribute.  A custom function can also be
  specified, so long as it generates a result that is compatible with the item of the Mechanism's `variable
  <Mechanism_Base.variable>` to which the `FeatureInputState is assigned <Mechanism_FeatureInputStates>`.

.. _FeatureInputState_Value:

* `value <FeatureInputState.value>` -- the result returned by its `function <FeatureInputState.function>`,
  after aggregating the value of the `PathProjections <PathwayProjection>` it receives, possibly modified by any
  `GatingProjections <GatingProjection>` received by the FeatureInputState. It must be compatible with the
  item of the owner Mechanism's `variable <Mechanism_Base.variable>` to which the `FeatureInputState has been assigned
  <Mechanism_FeatureInputStates>` (see `above <FeatureInputState_Variable_and_Value>` and `Mechanism FeatureInputState specification
  <Mechanism_FeatureInputState_Specification>`).

.. _FeatureInputState_Weights_And_Exponents:

* `weight <FeatureInputState.weight>` and `exponent <FeatureInputState.exponent>` -- these can be used by the Mechanism to which the
  FeatureInputState belongs when that combines the `value <FeatureInputState.value>`\\s of its States (e.g., an ObjectiveMechanism
  uses the weights and exponents assigned to its FeatureInputStates to determine how the values it monitors are combined by
  its `function <ObjectiveMechanism>`).  The value of each must be an integer or float, and the default is 1 for both.

.. _FeatureInputState_Execution:

Execution
---------

A FeatureInputState cannot be executed directly.  It is executed when the Mechanism to which it belongs is executed.
When this occurs, the FeatureInputState executes any `Projections <Projection>` it receives, calls its `function
<FeatureInputState.function>` to aggregate the values received from any `MappingProjections <MappingProjection>` it receives
(listed in its its `path_afferents  <FeatureInputState.path_afferents>` attribute) and modulate them in response to any
`GatingProjections <GatingProjection>` (listed in its `mod_afferents <FeatureInputState.mod_afferents>` attribute),
and then assigns the result to the FeatureInputState's `value <FeatureInputState.value>` attribute. This, in turn, is assigned to
the item of the Mechanism's `variable <Mechanism_Base.variable>` and `input_values <Mechanism_Base.input_values>`
attributes  corresponding to that FeatureInputState (see `Mechanism Variable and FeatureInputStates
<Mechanism_Variable_and_FeatureInputStates>` for additional details).

.. _FeatureInputState_Class_Reference:

Class Reference
---------------

"""
import numbers
import warnings

import collections
import numpy as np
import typecheck as tc

from psyneulink.core.components.functions.combinationfunctions import LinearCombination
from psyneulink.core.components.states.inputstate import InputState
from psyneulink.core.components.states.outputstate import OutputState
from psyneulink.core.components.states.state import StateError, State_Base, _instantiate_state_list, state_type_keywords
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import EXPONENT, FEATURE_INPUT_STATE, GATING_SIGNAL, INPUT_STATE, INPUT_STATE_PARAMS, LEARNING_SIGNAL, MAPPING_PROJECTION, MATRIX, MECHANISM, OPERATION, OUTPUT_STATE, OUTPUT_STATES, PROCESS_INPUT_STATE, PRODUCT, PROJECTIONS, PROJECTION_TYPE, REFERENCE_VALUE, SENDER, SIZE, SUM, SYSTEM_INPUT_STATE, VALUE, VARIABLE, WEIGHT
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import append_type_to_name, is_numeric, iscompatible

__all__ = [
    'FeatureInputState', 'FeatureInputStateError', 'state_type_keywords',
]

state_type_keywords = state_type_keywords.update({FEATURE_INPUT_STATE})

WEIGHT_INDEX = 1
EXPONENT_INDEX = 2

DEFER_VARIABLE_SPEC_TO_MECH_MSG = "FeatureInputState variable not yet defined, defer to Mechanism"

class FeatureInputStateError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class FeatureInputState(InputState):
    """
    FeatureInputState(                                    \
        owner=None,                                \
        variable=None,                             \
        size=None,                                 \
        function=LinearCombination(operation=SUM), \
        combine=None,                              \
        projections=None,                          \
        weight=None,                               \
        exponent=None,                             \
        internal_only=False,                       \
        params=None,                               \
        name=None,                                 \
        prefs=None)

    Subclass of `InputState <InputState>` that receives only one `PathwayProjection <PathwayProjection>`, and may apply
    any `Function` to its variable.

    Arguments
    ---------

    owner : Mechanism
        the Mechanism to which the FeatureInputState belongs;  it must be specified or determinable from the context in which
        the FeatureInputState is created.

    reference_value : number, list or np.ndarray
        the value of the item of the owner Mechanism's `variable <Mechanism_Base.variable>` attribute to which
        the FeatureInputState is assigned; used as the template for the FeatureInputState's `value <FeatureInputState.value>` attribute.

    variable : number, list or np.ndarray
        specifies the template for the FeatureInputState's `variable <FeatureInputState.variable>` attribute.

    size : int, list or np.ndarray of ints
        specifies variable as array(s) of zeros if **variable** is not passed as an argument;
        if **variable** is specified, it takes precedence over the specification of **size**.

    function : Function or method : default LinearCombination(operation=SUM)
        specifies the function used to aggregate the `values <Projection_Base.value>` of the `Projections <Projection>`
        received by the FeatureInputState, under the possible influence of `GatingProjections <GatingProjection>` received
        by the FeatureInputState.  It must produce a result that has the same format (number and type of elements) as the
        item of its owner Mechanism's `variable <Mechanism_Base.variable>` to which the FeatureInputState has been assigned.

    combine : SUM or PRODUCT : default None
        specifies the **operation** argument used by the default `LinearCombination` function, wnich determines how the
        `value <Projection.value>` of the FeatureInputState's `projections <FeatureInputState.projections>` are combined.  This is a
        convenience argument, that **operation** to be specified without having to specify the function's constructor;
        accordingly, it assumes that LinearCombination (the default) is used as the FeatureInputState's function -- if it
        conflicts with a specification of **function** an error is generated.

    projections : list of Projection specifications
        specifies the `MappingProjection(s) <MappingProjection>` and/or `GatingProjection(s) <GatingProjection>` to be
        received by the FeatureInputState, and that are listed in its `path_afferents <FeatureInputState.path_afferents>` and
        `mod_afferents <FeatureInputState.mod_afferents>` attributes, respectively (see
        `FeatureInputState_Compatability_and_Constraints` for additional details).  If **projections** but neither
        **variable** nor **size** are specified, then the `value <Projection.value>` of the Projection(s) or their
        `senders <Projection.sender>` specified in **projections** argument are used to determine the FeatureInputState's
        `variable <FeatureInputState.variable>`.

    weight : number : default 1
        specifies the value of the `weight <FeatureInputState.weight>` attribute of the FeatureInputState.

    exponent : number : default 1
        specifies the value of the `exponent <FeatureInputState.exponent>` attribute of the FeatureInputState.

    internal_only : bool : False
        specifies whether external input is required by the FeatureInputState's `owner <FeatureInputState.owner>` if its `role
        <Mechanism_Role_In_Processes_And_Systems>` is *EXTERNAL_INPUT*  (see `internal_only <FeatureInputState.internal_only>`
        for details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the FeatureInputState or its function, and/or a custom function and its parameters. Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default see `name <FeatureInputState.name>`
        specifies the name of the FeatureInputState; see FeatureInputState `name <FeatureInputState.name>` for details.

    prefs : PreferenceSet or specification dict : default State.classPreferences
        specifies the `PreferenceSet` for the FeatureInputState; see `prefs <FeatureInputState.prefs>` for details.


    Attributes
    ----------

    owner : Mechanism
        the Mechanism to which the FeatureInputState belongs.

    path_afferents : List[MappingProjection]
        `MappingProjections <MappingProjection>` that project to the FeatureInputState
        (i.e., for which it is a `receiver <Projection_Base.receiver>`).

    mod_afferents : List[GatingProjection]
        `GatingProjections <GatingProjection>` that project to the FeatureInputState.

    projections : List[Projection]
        all of the `Projections <Projection>` received by the FeatureInputState.

    variable : value, list or np.ndarray
        the template for the `value <Projection_Base.value>` of each Projection that the FeatureInputState receives,
        each of which must match the format (number and types of elements) of the FeatureInputState's
        `variable <FeatureInputState.variable>`.  If neither the **variable** or **size** argument is specified, and
        **projections** is specified, then `variable <FeatureInputState.variable>` is assigned the `value
        <Projection.value>` of the Projection(s) or its `sender <Projection.sender>`.

    function : CombinationFunction : default LinearCombination(operation=SUM))
        performs an element-wise (Hadamard) aggregation of the `value <Projection_Base.value>` of each Projection
        received by the FeatureInputState, under the possible influence of any `GatingProjections <GatingProjection>` received
        by the FeatureInputState.

    value : value or ndarray
        the output of the FeatureInputState's `function <FeatureInputState.function>`, which is the aggregated value of the
        `PathwayProjections <PathwayProjection>` (e.g., `MappingProjections <MappingProjection>`) received by the
        FeatureInputState (and listed in its `path_afferents <FeatureInputState.path_afferents>` attribute), possibly `modulated
        <ModulatorySignal_Modulation>` by any `GatingProjections <GatingProjection>` it receives (listed in its
        `mod_afferents <FeatureInputState.mod_afferents>` attribute).  The result (whether a value or an ndarray) is
        assigned to an item of the owner Mechanism's `variable <Mechanism_Base.variable>`.

    label : string or number
        the string label that represents the current `value <FeatureInputState.value>` of the FeatureInputState, according to the
        owner mechanism's `input_labels_dict <Mechanism.input_labels_dict>`. If the current `value <FeatureInputState.value>`
        of the FeatureInputState does not have a corresponding label, then the numeric `value <FeatureInputState.value>` is returned.

    weight : number
        see `weight and exponent <FeatureInputState_Weights_And_Exponents>` for description.

    exponent : number
        see `weight and exponent <FeatureInputState_Weights_And_Exponents>` for description.

    internal_only : bool
        determines whether input is required for this FeatureInputState from `Run` or another `Composition` when the
        FeatureInputState's `owner <FeatureInputState.owner>` is executed, and its `role <Mechanism_Role_In_Processes_And_Systems>`
        is designated as *EXTERNAL_INPUT*;  if `True`, external input is *not* required or allowed;  otherwise,
        external input is required.

    name : str
        the name of the FeatureInputState; if it is not specified in the **name** argument of the constructor, a default is
        assigned by the FeatureInputStateRegistry of the Mechanism to which the FeatureInputState belongs.  Note that some Mechanisms
        automatically create one or more non-default FeatureInputStates, that have pre-specified names.  However, if any
        FeatureInputStates are specified in the **input_states** argument of the Mechanism's constructor, those replace those
        FeatureInputStates (see `note <Mechanism_Default_State_Suppression_Note>`), and `standard naming conventions <Naming>`
        apply to the FeatureInputStates specified, as well as any that are added to the Mechanism once it is created.

        .. note::
            Unlike other PsyNeuLink components, State names are "scoped" within a Mechanism, meaning that States with
            the same name are permitted in different Mechanisms.  However, they are *not* permitted in the same
            Mechanism: States within a Mechanism with the same base name are appended an index in the order of their
            creation.

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the FeatureInputState; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).


    """

    #region CLASS ATTRIBUTES

    componentType = INPUT_STATE
    paramsType = INPUT_STATE_PARAMS

    stateAttributes = State_Base.stateAttributes | {WEIGHT, EXPONENT}

    connectsWith = [OUTPUT_STATE,
                    PROCESS_INPUT_STATE,
                    SYSTEM_INPUT_STATE,
                    LEARNING_SIGNAL,
                    GATING_SIGNAL]
    connectsWithAttribute = [OUTPUT_STATES]
    projectionSocket = SENDER
    modulators = [GATING_SIGNAL]


    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'FeatureInputStateCustomClassPreferences',
    #     kp<pref>: <setting>...}

    # Note: the following enforce encoding as 1D np.ndarrays (one variable/value array per state)
    variableEncodingDim = 1
    valueEncodingDim = 1

    class Parameters(State_Base.Parameters):
        """
            Attributes
            ----------

                exponent
                    see `exponent <FeatureInputState.exponent>`

                    :default value: None
                    :type:

                function
                    see `function <FeatureInputState.function>`

                    :default value: `LinearCombination`(offset=0.0, operation=sum, scale=1.0)
                    :type: `Function`

                internal_only
                    see `internal_only <FeatureInputState.internal_only>`

                    :default value: False
                    :type: bool

                weight
                    see `weight <FeatureInputState.weight>`

                    :default value: None
                    :type:

        """
        function = Parameter(LinearCombination(operation=SUM), stateful=False, loggable=False)
        weight = Parameter(None, modulable=True)
        exponent = Parameter(None, modulable=True)
        combine = None
        internal_only = Parameter(False, stateful=False, loggable=False)

    paramClassDefaults = State_Base.paramClassDefaults.copy()
    paramClassDefaults.update({PROJECTION_TYPE: MAPPING_PROJECTION,
                               MECHANISM: None,     # These are used to specifiy FeatureInputStates by projections to them
                               OUTPUT_STATES: None  # from the OutputStates of a particular Mechanism (see docs)
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
                 combine:tc.optional(tc.enum(SUM,PRODUCT))=None,
                 weight=None,
                 exponent=None,
                 internal_only:bool=False,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        super(FeatureInputState, self).__init__(owner,
                                                reference_value=reference_value,
                                                variable=variable,
                                                size=size,
                                                function=function,
                                                projections=projections,
                                                combine=combine,
                                                weight=weight,
                                                exponent=exponent,
                                                internal_only=internal_only,
                                                params=params,
                                                name=name,
                                                prefs=prefs,
                                                context=context)

    def _validate_function(self, function):
        pass

    @tc.typecheck
    def _parse_state_specific_specs(self, owner, state_dict, state_specific_spec):
        """Get weights, exponents and/or any connections specified in a FeatureInputState specification tuple

        Tuple specification can be:
            (state_spec, connections)
            (state_spec, weights, exponents, connections)

        See State._parse_state_specific_spec for additional info.
.
        Returns:
             - state_spec:  1st item of tuple if it is a numeric value;  otherwise None
             - params dict with WEIGHT, EXPONENT and/or PROJECTIONS entries if any of these was specified.

        """
        # FIX: ADD FACILITY TO SPECIFY WEIGHTS AND/OR EXPONENTS FOR INDIVIDUAL OutputState SPECS
        #      CHANGE EXPECTATION OF *PROJECTIONS* ENTRY TO BE A SET OF TUPLES WITH THE WEIGHT AND EXPONENT FOR IT
        #      THESE CAN BE USED BY THE FeatureInputState's LinearCombination Function
        #          (AKIN TO HOW THE MECHANISM'S FUNCTION COMBINES FeatureInputState VALUES)
        #      THIS WOULD ALLOW AN ADDITIONAL HIERARCHICAL LEVEL FOR NESTING ALGEBRAIC COMBINATION OF INPUT VALUES
        #      TO A MECHANISM
        from psyneulink.core.components.projections.projection import Projection, _parse_connection_specs
        from psyneulink.core.components.mechanisms.mechanism import Mechanism

        params_dict = {}
        state_spec = state_specific_spec

        if isinstance(state_specific_spec, dict):
            # FIX: 10/3/17 - CHECK HERE THAT, IF MECHANISM ENTRY IS USED, A VARIABLE, WEIGHT AND/OR EXPONENT ENTRY
            # FIX:                       IS APPLIED TO ALL THE OutputStates SPECIFIED IN OUTPUT_STATES
            # FIX:                       UNLESS THEY THEMSELVES USE A State specification dict WITH ANY OF THOSE ENTRIES
            # FIX:           USE ObjectiveMechanism EXAMPLES
            # if MECHANISM in state_specific_spec:
            #     if OUTPUT_STATES in state_specific_spec
            if SIZE in state_specific_spec:
                if (VARIABLE in state_specific_spec or
                        any(key in state_dict and state_dict[key] is not None for key in {VARIABLE, SIZE})):
                    raise FeatureInputStateError("PROGRAM ERROR: SIZE specification found in state_specific_spec dict "
                                          "for {} specification of {} when SIZE or VARIABLE is already present in its "
                                          "state_specific_spec dict or state_dict".format(self.__name__, owner.name))
                state_dict.update({VARIABLE:np.zeros(state_specific_spec[SIZE])})
                del state_specific_spec[SIZE]
                return state_dict, state_specific_spec
            return None, state_specific_spec

        elif isinstance(state_specific_spec, tuple):

            # GET STATE_SPEC AND ASSIGN PROJECTIONS_SPEC **********************************************************

            tuple_spec = state_specific_spec

            # 2-item tuple specification
            if len(tuple_spec) == 2:

                # 1st item is a value, so treat as State spec (and return to _parse_state_spec to be parsed)
                #   and treat 2nd item as Projection specification
                if is_numeric(tuple_spec[0]):
                    state_spec = tuple_spec[0]
                    reference_value = state_dict[REFERENCE_VALUE]
                    # Assign value so sender_dim is skipped below
                    # (actual assignment is made in _parse_state_spec)
                    if reference_value is None:
                        state_dict[REFERENCE_VALUE]=state_spec
                    elif  not iscompatible(state_spec, reference_value):
                        raise StateError("Value in first item of 2-item tuple specification for {} of {} ({}) "
                                         "is not compatible with its {} ({})".
                                         format(FeatureInputState.__name__, owner.name, state_spec,
                                                REFERENCE_VALUE, reference_value))
                    projections_spec = tuple_spec[1]

                # Tuple is Projection specification that is used to specify the State,
                else:
                    # return None in state_spec to suppress further, recursive parsing of it in _parse_state_spec
                    state_spec = None
                    if tuple_spec[0] != self:
                        # If 1st item is not the current state (self), treat as part of the projection specification
                        projections_spec = tuple_spec
                    else:
                        # Otherwise, just use 2nd item as projection spec
                        state_spec = None
                        projections_spec = tuple_spec[1]

            # 3- or 4-item tuple specification
            elif len(tuple_spec) in {3,4}:
                # Tuple is projection specification that is used to specify the State,
                #    so return None in state_spec to suppress further, recursive parsing of it in _parse_state_spec
                state_spec = None
                # Reduce to 2-item tuple Projection specification
                projection_item = tuple_spec[3] if len(tuple_spec)==4 else None
                projections_spec = (tuple_spec[0],projection_item)

            # GET PROJECTIONS IF SPECIFIED *************************************************************************

            try:
                projections_spec
            except UnboundLocalError:
                pass
            else:
                try:
                    params_dict[PROJECTIONS] = _parse_connection_specs(self,
                                                                       owner=owner,
                                                                       connections=projections_spec)
                    # Parse the value of all of the Projections to get/validate variable for FeatureInputState
                    variable = []
                    for projection_spec in params_dict[PROJECTIONS]:
                        # FIX: 10/3/17 - PUTTING THIS HERE IS A HACK...
                        # FIX:           MOVE TO _parse_state_spec UNDER PROCESSING OF ProjectionTuple SPEC
                        # FIX:           USING _get_state_for_socket
                        # from psyneulink.core.components.projections.projection import _parse_projection_spec

                        # Try to get matrix for projection
                        try:
                            sender_dim = np.array(projection_spec.state.value).ndim
                        except AttributeError as e:
                            # KDM 10/23/18: if DEFERRED_INIT is set, it will be set on the non-stateful .context
                            # attr so these should be ok
                            if (isinstance(projection_spec.state, type) or
                                     projection_spec.state.context.initialization_status==ContextFlags.DEFERRED_INIT):
                                continue
                            else:
                                raise StateError("PROGRAM ERROR: indeterminate value for {} "
                                                 "specified to project to {} of {}".
                                                 format(projection_spec.state.name, self.__name__, owner.name))

                        projection = projection_spec.projection
                        if isinstance(projection, dict):
                            # Don't try to get MATRIX from projection without checking,
                            #    since projection is a defaultDict,
                            #    which will add a matrix entry and assign it to None if it is not there
                            if MATRIX in projection:
                                matrix = projection[MATRIX]
                            else:
                                matrix = None
                        elif isinstance(projection, Projection):
                            if projection.context.initialization_status == ContextFlags.DEFERRED_INIT:
                                continue
                            # possible needs to be projection.defaults.matrix?
                            matrix = projection.matrix
                        else:
                            raise FeatureInputStateError("Unrecognized Projection specification for {} of {} ({})".
                                                  format(self.name, owner.name, projection_spec))

                        # Determine length of value of projection
                        if matrix is None:
                            # If a reference_value has been specified, it presumably represents the item of the
                            #    owner Mechanism's default_variable to which the FeatureInputState corresponds,
                            #    so use that to constrain the FeatureInputState's variable
                            if state_dict[REFERENCE_VALUE] is not None:
                                variable.append(state_dict[REFERENCE_VALUE])
                                continue
                            # If matrix has not been specified, no worries;
                            #    variable_item can be determined by value of sender
                            sender_shape = np.array(projection_spec.state.value).shape
                            variable_item = np.zeros(sender_shape)
                            # If variable_item HASN'T been specified, or it is same shape as any previous ones,
                            #     use sender's value
                            if ((VARIABLE not in state_dict or state_dict[VARIABLE] is None) and
                                    (not variable or variable_item.shape == variable[0].shape)):
                                # state_dict[VARIABLE] = variable
                                variable.append(variable_item)
                            # If variable HAS been assigned, make sure value is the same for this sender
                            elif np.array(state_dict[VARIABLE]).shape != variable_item.shape:
                                # If values for senders differ, assign None so that State's default is used
                                variable = None
                                # No need to check any more Projections
                                break

                        # Remove dimensionality of sender OutputState, and assume that is what receiver will receive
                        else:
                            proj_val_shape = matrix.shape[sender_dim :]
                            # state_dict[VARIABLE] = np.zeros(proj_val_shape)
                            variable.append(np.zeros(proj_val_shape))
                    # Sender's value has not been defined or senders have values of different lengths,
                    if not variable:
                        # Try to assign number of items = number of projections, each with default variable length
                        try:
                            state_dict[VARIABLE] = np.zeros_like(self.defaults.variable) * \
                                                   len(params_dict[PROJECTIONS])
                        # FeatureInputState's default variable has not yet been defined
                        except AttributeError:
                            # If reference_value was provided, use that as the FeatureInputState's variable
                            #    (i.e., assume its function won't transform it)
                            if REFERENCE_VALUE in state_dict and state_dict[REFERENCE_VALUE] is not None:
                                state_dict[VARIABLE] = state_dict[REFERENCE_VALUE]
                            # Nothing to use as variable, so raise exception and allow it to be handled "above"
                            else:
                                raise AttributeError(DEFER_VARIABLE_SPEC_TO_MECH_MSG)
                    else:
                        state_dict[VARIABLE] = variable

                except FeatureInputStateError:
                    raise FeatureInputStateError("Tuple specification in {} specification dictionary "
                                          "for {} ({}) is not a recognized specification for one or more "
                                          "{}s, {}s, or {}s that project to it".
                                          format(FeatureInputState.__name__,
                                                 owner.name,
                                                 projections_spec,
                                                 'Mechanism',
                                                 OutputState.__name__,
                                                 Projection.__name__))

            # GET WEIGHT AND EXPONENT IF SPECIFIED ***************************************************************

            if len(tuple_spec) == 2:
                pass

            # Tuple is (spec, weights, exponents<, afferent_source_spec>),
            #    for specification of weights and exponents,  + connection(s) (afferent projection(s)) to FeatureInputState
            elif len(tuple_spec) in {3, 4}:

                weight = tuple_spec[WEIGHT_INDEX]
                exponent = tuple_spec[EXPONENT_INDEX]

                if weight is not None and not isinstance(weight, numbers.Number):
                    raise FeatureInputStateError("Specification of the weight ({}) in tuple of {} specification dictionary "
                                          "for {} must be a number".format(weight, FeatureInputState.__name__, owner.name))
                params_dict[WEIGHT] = weight

                if exponent is not None and not isinstance(exponent, numbers.Number):
                    raise FeatureInputStateError("Specification of the exponent ({}) in tuple of {} specification dictionary "
                                          "for {} must be a number".format(exponent, FeatureInputState.__name__, owner.name))
                params_dict[EXPONENT] = exponent

            else:
                raise StateError("Tuple provided as state_spec for {} of {} ({}) must have either 2, 3 or 4 items".
                                 format(FeatureInputState.__name__, owner.name, tuple_spec))

        elif state_specific_spec is not None:
            raise FeatureInputStateError("PROGRAM ERROR: Expected tuple or dict for {}-specific params but, got: {}".
                                  format(self.__class__.__name__, state_specific_spec))

        return state_spec, params_dict

    @staticmethod
    def _state_spec_allows_override_variable(spec):
        '''
        Returns
        -------
            True - if **spec** outlines a spec for creating a FeatureInputState whose variable can be
                overridden by a default_variable or size argument
            False - otherwise

            ex: specifying a FeatureInputState with a Mechanism allows overriding
        '''
        from psyneulink.core.components.mechanisms.mechanism import Mechanism

        if isinstance(spec, Mechanism):
            return True
        if isinstance(spec, collections.Iterable):
            # generally 2-4 tuple spec, but allows list spec
            for item in spec:
                if isinstance(item, Mechanism):
                    return True
                # handles tuple spec where first item of tuple is itself a (name, Mechanism) tuple
                elif (
                    isinstance(item, collections.Iterable)
                    and len(item) >= 2
                    and isinstance(item[1], Mechanism)
                ):
                    return True

        return False

    @property
    def pathway_projections(self):
        return self.path_afferents

    @pathway_projections.setter
    def pathway_projections(self, assignment):
        self.path_afferents = assignment

    @property
    def socket_width(self):
        return self.defaults.variable.shape[-1]

    @property
    def socket_template(self):
        return np.zeros(self.socket_width)

    @property
    def label(self):
        return self.get_label()

    def get_label(self, execution_context=None):
        try:
            label_dictionary = self.owner.input_labels_dict
        except AttributeError:
            label_dictionary = {}
        return self._get_value_label(label_dictionary, self.owner.input_states, execution_context=execution_context)

    @property
    def position_in_mechanism(self):
        if hasattr(self, "owner"):
            if self.owner is not None:
                return self.owner.get_input_state_position(self)
            else:
                return None
        return None

    @staticmethod
    def _get_state_function_value(owner, function, variable):
        """Put FeatureInputState's variable in a list if its function is LinearCombination and variable is >=2d

        FeatureInputState variable must be embedded in a list so that LinearCombination (its default function)
        returns a variable that is >=2d intact (rather than as arrays to be combined);
        this is normally done in State.update() (and in State._instantiate-function), but that
        can't be called by _parse_state_spec since the FeatureInputState itself may not yet have been instantiated.

        """
        import inspect

        if (
                (
                    (inspect.isclass(function) and issubclass(function, LinearCombination))
                    or isinstance(function, LinearCombination)
                )
                and isinstance(variable, np.matrix)
        ):
            variable = [variable]

        # if function is None, use State's default function
        function = function or FeatureInputState.defaults.function

        return State_Base._get_state_function_value(owner=owner, function=function, variable=variable)


def _instantiate_input_states(owner, input_states=None, reference_value=None, context=None):
    """Call State._instantiate_state_list() to instantiate ContentAddressableList of FeatureInputState(s)

    Create ContentAddressableList of FeatureInputState(s) specified in paramsCurrent[INPUT_STATES]

    If input_states is not specified:
        - use owner.input_states as list of FeatureInputState specifications
        - if owner.input_states is empty, user owner.defaults.variable to create a default FeatureInputState

    When completed:
        - self.input_states contains a ContentAddressableList of one or more input_states
        - self.input_state contains the `primary FeatureInputState <FeatureInputState_Primary>`:  first or only one in input_states
        - paramsCurrent[INPUT_STATES] contains the same ContentAddressableList (of one or more input_states)
        - each FeatureInputState corresponds to an item in the variable of the owner's function
        - the value of all of the input_states is stored in a list in input_value
        - if there is only one FeatureInputState, it is assigned the full value

    Note: State._instantiate_state_list()
              parses self.defaults.variable (2D np.array, passed in reference_value)
              into individual 1D arrays, one for each input state

    (See State._instantiate_state_list() for additional details)

    Returns list of instantiated FeatureInputStates
    """

    # This allows method to be called by Mechanism.add_input_states() with set of user-specified input_states,
    #    while calls from init_methods continue to use owner.input_states (i.e., FeatureInputState specifications
    #    assigned in the **input_states** argument of the Mechanism's constructor)
    input_states = input_states or owner.input_states
    state_list = _instantiate_state_list(owner=owner,
                                         state_list=input_states,
                                         state_type=FeatureInputState,
                                         state_param_identifier=INPUT_STATE,
                                         reference_value=reference_value if reference_value is not None
                                                                         else owner.defaults.variable,
                                         # reference_value=reference_value,
                                         reference_value_name=VALUE,
                                         context=context)

    # Call from Mechanism.add_states, so add to rather than assign input_states (i.e., don't replace)
    if context & (ContextFlags.METHOD | ContextFlags.COMMAND_LINE):
        owner.input_states.extend(state_list)
    else:
        owner._input_states = state_list

    # Check that number of input_states and their variables are consistent with owner.defaults.variable,
    #    and adjust the latter if not
    variable_item_is_OK = False
    for i, input_state in enumerate(owner.input_states):
        try:
            variable_item_is_OK = iscompatible(owner.defaults.variable[i], input_state.value)
            if not variable_item_is_OK:
                break
        except IndexError:
            variable_item_is_OK = False
            break

    if not variable_item_is_OK:
        old_variable = owner.defaults.variable
        owner.defaults.variable = owner._handle_default_variable(default_variable=[state.value for state in owner.input_states])

        if owner.verbosePref:
            warnings.warn(
                "Variable for {} ({}) has been adjusted to match number and format of its input_states: ({})".format(
                    old_variable,
                    append_type_to_name(owner),
                    owner.defaults.variable,
                )
            )

    return state_list
