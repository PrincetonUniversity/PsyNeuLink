# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


#  *********************************************  State ********************************************************

"""

Overview
--------

A State provides an interface to one or more `Projections <Projection>`, and receives the `value(s) <Projection>`
provided by them.  The value of a State can be modulated by a `ModulatoryProjection <ModulatoryProjection>`. There are
three primary types of States (InputStates, ParameterStates and OutputStates) as well as one subtype (ModulatorySignal,
used to send ModulatoryProjections), as summarized in the table below:

.. _State_Types_Table:

.. table:: **State Types and Associated Projection Types**
   :align: left

   +-------------------+--------------------+------------------------+-----------------+-------------------------------+
   | *State Type*      | *Owner*            |      *Description*     | *Modulated by*  |       *Specification*         |
   +===================+====================+========================+=================+===============================+
   | `InputState`      |  `Mechanism        |receives input from     | `GatingSignal`  |`InputState` constructor;      |
   |                   |  <Mechanism>`      |`MappingProjection`     |                 |`Mechanism <Mechanism>`        |
   |                   |                    |                        |                 |constructor or its             |
   |                   |                    |                        |                 |`add_states` method            |
   +-------------------+--------------------+------------------------+-----------------+-------------------------------+
   |`ParameterState`   |  `Mechanism        |represents parameter    | `LearningSignal`|Implicitly whenever a          |
   |                   |  <Mechanism>` or   |value for a `Component  | and/or          |parameter value is             |
   |                   |  `Projection       |<Component>`            | `ControlSignal` |`specified                     |
   |                   |  <Projection>`     |or its `function        |                 |<ParameterState_Specification>`|
   |                   |                    |<Component.function>`   |                 |                               |
   +-------------------+--------------------+------------------------+-----------------+-------------------------------+
   | `OutputState`     |  `Mechanism        |provides output to      | `GatingSignal`  |`OutputState` constructor;     |
   |                   |  <Mechanism>`      |`MappingProjection`     |                 |`Mechanism <Mechanism>`        |
   |                   |                    |                        |                 |constructor or its             |
   |                   |                    |                        |                 |`add_states` method            |
   +-------------------+--------------------+------------------------+-----------------+-------------------------------+
   |`ModulatorySignal  |`AdaptiveMechanism  |provides value for      |                 |`AdaptiveMechanism             |
   |<ModulatorySignal>`|<AdaptiveMechanism>`|`ModulatoryProjection   |                 |<AdaptiveMechanism>`           |
   |                   |                    |<ModulatoryProjection>` |                 |constructor; tuple in State    |
   |                   |                    |                        |                 |or parameter specification     |
   +-------------------+--------------------+------------------------+-----------------+-------------------------------+

COMMENT:

* `InputState`:
    used by a Mechanism to receive input from `MappingProjections <MappingProjection>`;
    its value can be modulated by a `GatingSignal`.

* `ParameterState`:
    * used by a Mechanism to represent the value of one of its parameters, or a parameter of its
      `function <Mechanism_Base.function>`, that can be modulated by a `ControlSignal`;
    * used by a `MappingProjection` to represent the value of its `matrix <MappingProjection.MappingProjection.matrix>`
      parameter, that can be modulated by a `LearningSignal`.

* `OutputState`:
    used by a Mechanism to send its value to any efferent projections.  For
    `ProcessingMechanisms <ProcessingMechanism>` these are `PathwayProjections <PathwayProjection>`, most commonly
    `MappingProjections <MappingProjection>`.  For `ModulatoryMechanisms <ModulatoryMechanism>`, these are
    `ModulatoryProjections <ModulatoryProjection>` as described below. The `value <OutputState.value>` of an
    OutputState can be modulated by a `GatingSignal`.

* `ModulatorySignal <ModulatorySignal>`:
    a subclass of `OutputState` used by `AdaptiveMechanisms <AdaptiveMechanism>` to modulate the value of the primary
    types of States listed above.  There are three types of ModulatorySignals:

    * `LearningSignal`, used by a `LearningMechanism` to modulate the *MATRIX* ParameterState of a `MappingProjection`;
    * `ControlSignal`, used by a `ControlMechanism <ControlMechanism>` to modulate the `ParameterState` of a `Mechanism
      <Mechanism>`;
    * `GatingSignal`, used by a `GatingMechanism` to modulate the `InputState` or `OutputState` of a `Mechanism
       <Mechanism>`.
    Modulation is discussed further `below <State_Modulation>`, and described in detail under
    `ModulatorySignals <ModulatorySignal_Modulation>`.

COMMENT

.. _State_Creation:

Creating a State
----------------

In general, States are created automatically by the objects to which they belong (their `owner <State_Owner>`),
or by specifying the State in the constructor for its owner.  For example, unless otherwise specified, when a
`Mechanism <Mechanism>` is created it creates a default `InputState` and `OutputState` for itself, and whenever any
Component is created, it automatically creates a `ParameterState` for each of its `configurable parameters
<Component_Configurable_Attributes>` and those of its `function <Component_Function>`. States are also created in
response to explicit specifications.  For example, InputStates and OutputStates can be specified in the constructor for
a Mechanism (see `Mechanism_State_Specification`); and ParameterStates are specified in effect when the value of a
parameter for any Component or its `function <Component.function>` is specified in the constructor for that Component
or function.  InputStates and OutputStates (but *not* ParameterStates) can also be created directly using their
constructors, and then assigned to a Mechanism using the Mechanism's `add_states <Mechanism_Base.add_states>` method;
however, this should be done with caution as the State must be compatible with other attributes of its owner (such as
its OutputStates) and its `function <Mechanism_Base.function>` (for example, see `note <Mechanism_Add_InputStates_Note>`
regarding InputStates). Parameter States **cannot** on their own; they are always and only created when the Component
to which a parameter belongs is created.

.. _State_Specification:

Specifying a State
~~~~~~~~~~~~~~~~~~

A State can be specified using any of the following:

    * existing **State** object;
    ..
    * name of a **State subclass** (`InputState`, `ParameterState`, or `OutputState`) -- creates a default State of the
      specified type, using a default value for the State that is determined by the context in which it is specified.
    ..
    * **value** -- creates a default State using the specified value as its default `value <State_Base.value>`.

    .. _State_Specification_Dictionary:

    * **State specification dictionary** -- can use the following: *KEY*:<value> entries, in addition to those
      specific to the State's type (see documentation for each type):

      * *STATE_TYPE*:<State type>
          specifies type of State to create (necessary if it cannot be determined from the
          the context of the other entries or in which it is being created).
      ..
      * *NAME*:<str>
          the string is used as the name of the State.
      ..
      * *VALUE*:<value>
          the value is used as the default value of the State.

      A State specification dictionary can also be used to specify one or more `Projections <Projection>' to or from
      the State, including `ModulatoryProjection(s) <ModulatoryProjection>` used to modify the `value
      <State_Base.value>` of the State.  The type of Projection(s) created depend on the type of State specified and
      context of the specification (see `examples <State_Specification_Dictionary_Examples>`).  This can be done using
      any of the following entries, each of which can contain any of the forms used to `specify a Projection
      <Projection_Specification>`:

      * *PROJECTIONS*:List[<`projection specification <Projection_Specification>`>,...]
          the list must contain a one or more `Projection specifications <Projection_Specification>` to or from
          the State, and/or `ModulatorySignals <ModulatorySignal>` from which it should receive projections (see
          `State_Projections` below).

      .. _State_State_Name_Entry:

      * *<str>*:List[<`projection specification <Projection_Specification>`>,...]
          this must be the only entry in the dictionary, and the string cannot be a PsyNeuLink
          keyword;  it is used as the name of the State, and the list must contain one or more `Projection
          specifications <Projection_Specification>`.

      .. _State_MECHANISM_STATES_Entries:

      * *MECHANISM*:Mechanism
          this can be used to specify one or more Projections to or from the specified Mechanism.  If the entry appears
          without any accompanying State specification entries (see below), the Projection is assumed to be a
          `MappingProjection` to the Mechanism's `primary InputState <InputState_Primary>` or from its `primary
          OutputState <OutputState_Primary>`, depending upon the type of Mechanism and context of specification.  It
          can also be accompanied by one or more State specification entries described below, to create one or more
          Projections to/from those States (see `examples <State_State_Name_Entry_Example>`).
      ..
      * *<STATES_KEYWORD>:List[<str or State.name>,...]
         this must accompany a *MECHANISM* entry (described above), and is used to specify its State(s) by name.
         Each entry must use one of the following keywords as its key, and there can be no more than one of each:
            - *INPUT_STATES*
            - *OUTPUT_STATES*
            - *PARAMETER_STATES*
            - *LEARNING_SIGNAL*
            - *CONTROL_SIGNAL*
            - *GATING_SIGNAL*.
         Each entry must contain a list States of the specified type, all of which belong to the Mechanism specified in
         the *MECHANISM* entry;  each item in the list must be the name of one the Mechanism's States, or a
         `ProjectionTuple <State_ProjectionTuple>` the first item of which is the name of a State. The types of
         States that can be specified in this manner depends on the type of the Mechanism and context of the
         specification (see `examples <State_State_Name_Entry_Example>`).

    * **State, Mechanism, or list of these** -- creates a default State with Projection(s) to/from the specified
      States;  the type of State being created determines the type and directionality of the Projection(s) and,
      if Mechanism(s) are specified, which of their primary States are used (see State subclasses for specifics).

   .. _State_Tuple_Specification:

    * **Tuple specifications** -- these are convenience formats that can be used to compactly specify a State
      by specifying other Components with which it should be connected by Projection(s). Different States support
      different forms, but all support the following two forms:

      .. _State_2_Item_Tuple:

      * **2-item tuple: (State name or list of State names, Mechanism)** - 1st item is the name of a State or list of
        State names, and the 2nd item is the Mechanism to which they belong; a Projection is created to or from each
        of the States specified.  The type of Projection depends on the type of State being created, and the type of
        States specified in the tuple  (see `Projection_Table`).  For example, if the State being created is an
        InputState, and the States specified in the tuple are OutputStates, then `MappingProjections
        <MappingProjection>` are used; if `ModulatorySignals <ModulatorySignal>` are specified, then the corresponding
        type of `ModulatoryProjections <ModulatoryProjection>` are created.  See State subclasses for additional
        details and compatibility requirements.
      |
      .. _State_ProjectionTuple:
      * `ProjectionTuple <Projection_ProjectionTuple>` -- a 4-item tuple that specifies one or more `Projections
        <Projection>` to or from other State(s), along with a weight and/or exponent for each.

.. _State_Projections:

Projections
~~~~~~~~~~~

When a State is created, it can be assigned one or more `Projections <Projection>`, in either the **projections**
argument of its constructor, or a *PROJECTIONS* entry of a `State specification dictionary
<State_Specification_Dictionary>` (or a dictionary assigned to the **params** argument of the State's constructor).
The following types of Projections can be specified for each type of State:

    .. _State_Projections_Table:

    .. table:: **Specifiable Projections for State Types**
        :align: left

        +------------------+-------------------------------+-------------------------------------+
        | *State Type*     || *PROJECTIONS* specification  || *Assigned to Attribute*            |
        +==================+===============================+=====================================+
        |`InputState`      || `PathwayProjection(s)        || `path_afferents                    |
        |                  |   <PathwayProjection>`        |   <InputState.path_afferents>`      |
        |                  || `GatingProjection(s)         || `mod_afferents                     |
        |                  |   <GatingProjection>`         |   <InputState.mod_afferents>`       |
        +------------------+-------------------------------+-------------------------------------+
        |`ParameterState`  || `ControlProjection(s)        || `mod_afferents                     |
        |                  |   <ControlProjection>`        |   <ParameterState.mod_afferents>`   |
        +------------------+-------------------------------+-------------------------------------+
        |`OutputState`     || `PathwayProjection(s)        || `efferents                         |
        |                  |   <PathwayProjection>`        |   <OutputState.efferents>`          |
        |                  || `GatingProjection(s)         || `mod_afferents                     |
        |                  |   <GatingProjection>`         |   <OutputState.mod_afferents>`      |
        +------------------+-------------------------------+-------------------------------------+
        |`ModulatorySignal`|| `ModulatoryProjection(s)     || `efferents                         |
        |                  |   <ModulatoryProjection>`     |   <ModulatorySignal.efferents>`     |
        +------------------+-------------------------------+-------------------------------------+

Projections must be specified in a list.  Each entry must be either a `specification for a projection
<Projection_Specification>`, or for a `sender <Projection_Base.sender>` or `receiver <Projection_Base.receiver>` of
one, in which case the appropriate type of Projection is created.  A sender or receiver can be specified as a `State
<State>` or a `Mechanism <Mechanism>`. If a Mechanism is specified, its primary `InputState <InputState_Primary>` or
`OutputState <OutputState_Primary>`  is used, as appropriate.  When a sender or receiver is used to specify the
Projection, the type of Projection created is inferred from the State and the type of sender or receiver specified,
as illustrated in the `examples <State_Projections_Examples>` below.  Note that the State must be `assigned to an
owner <State_Creation>` in order to be functional, irrespective of whether any `Projections <Projection>` have been
assigned to it.


.. _State_Deferred_Initialization:

Deferred Initialization
~~~~~~~~~~~~~~~~~~~~~~~

If a State is created on its own, and its `owner <State_Owner>` Mechanism is specified, it is assigned to that
Mechanism; if its owner not specified, then its initialization is `deferred <State_Deferred_Initialization>`.
Its initialization is completed automatically when it is assigned to an owner `Mechanism <Mechanism_Base>` using the
owner's `add_states <Mechanism_Base.add_states>` method.  If the State is not assigned to an owner, it will not be
functional (i.e., used during the execution of `Mechanisms <Mechanism_Base_Execution>` and/or `Compositions
<Composition_Execution>`, irrespective of whether it has any `Projections <Projection>` assigned to it.


.. _State_Structure:

Structure
---------

.. _State_Owner:

Owner
~~~~~

Every State has an `owner <State_Base.owner>`.  For `InputStates <InputState>` and `OutputStates <OutputState>`, the
owner must be a `Mechanism <Mechanism>`.  For `ParameterStates <ParameterState>` it can be a Mechanism or a
`PathwayProjection <PathwayProjection>`. For `ModulatorySignals <ModulatorySignal>`, it must be an `AdaptiveMechanism
<AdaptiveMechanism>`. When a State is created as part of another Component, its `owner <State_Base.owner>` is
assigned automatically to that Component.  It is also assigned automatically when the State is assigned to a
`Mechanism <Mechanism>` using that Mechanism's `add_states <Mechanism_Base.add_states>` method.  Otherwise, it must be
specified explicitly in the **owner** argument of the constructor for the State (in which case it is immediately
assigned to the specified Mechanism).  If the **owner** argument is not specified, the State's initialization is
`deferred <State_Deferred_Initialization>` until it has been assigned to an owner using the owner's `add_states
<Mechanism_Base.add_states>` method.

Projections
~~~~~~~~~~~

Every State has attributes that lists the `Projections <Projection>` it sends and/or receives.  These depend on the
type of State, listed below (and shown in the `table <State_Projections_Table>`):

.. table::  State Projection Attributes
   :align: left

   ============================================ ============================================================
   *Attribute*                                  *Projection Type and State(s)*
   ============================================ ============================================================
   `path_afferents <State_Base.path_afferents>` `MappingProjections <MappingProjection>` to `InputState`
   `mod_afferents <State_Base.mod_afferents>`   `ModulatoryProjections <ModulatoryProjection>` to any State
   `efferents <State_Base.efferents>`           `MappingProjections <MappingProjection>` from `OutputState`
   ============================================ ============================================================

In addition to these attributes, all of the Projections sent and received by a State are listed in its `projections
<State_Base.projections>` attribute.


Variable, Function and Value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition, like all PsyNeuLink Components, it also has the three following core attributes:

    * `variable <State_Base.variable>`:  for an `InputState` and `ParameterState`,
      the value of this is determined by the value(s) of the Projection(s) that it receives (and that are listed in
      its `path_afferents <State_Base.path_afferents>` attribute).  For an `OutputState`, it is the item of the owner
      Mechanism's `value <Mechanism_Base.value>` to which the OutputState is assigned (specified by the OutputState's
      `index <OutputState_Index>` attribute.
    ..
    * `function <State_Base.function>`:  for an `InputState` this aggregates the values of the Projections that the
      State receives (the default is `LinearCombination` that sums the values), under the potential influence of a
      `GatingSignal`;  for a `ParameterState`, it determines the value of the associated parameter, under the potential
      influence of a `ControlSignal` (for a `Mechanism <Mechanism>`) or a `LearningSignal` (for a `MappingProjection`);
      for an OutputState, it conveys the result  of the Mechanism's function to its `output_values
      <Mechanism_Base.output_values>` attribute, under the potential influence of a `GatingSignal`.  See
      `ModulatorySignals <ModulatorySignal_Structure>` and the `AdaptiveMechanism <AdaptiveMechanism>` associated with
      each type for a description of how they can be used to modulate the `function <State_Base.function>` of a State.
    ..
    * `value <State_Base.value>`:  for an `InputState` this is the aggregated value of the `PathwayProjections` it
      receives;  for a `ParameterState`, this represents the value of the parameter that will be used by the State's
      owner or its `function <Component.function>`; for an `OutputState`, it is the item of the  owner Mechanism's
      `value <Mechanisms.value>` to which the OutputState is assigned, possibly modified by its `calculate
      <OutputState_Calculate>` attribute and/or a `GatingSignal`, and used as the `value <Projection_Base.value>` of
      the Projections listed in its `efferents <OutputState.path_efferents>` attribute.

.. _State_Modulation:

Modulation
~~~~~~~~~~

Every type of State has a `mod_afferents <State_Base.mod_afferents>` attribute, that lists the `ModulatoryProjections
<ModulatoryProjection>` it receives.  Each ModulatoryProjection comes from a `ModulatorySignal <ModulatorySignal>`
that specifies how it should modulate the State's `value <State_Base.value>` when the State is updated (see
`ModulatorySignal_Modulation` and `ModulatorySignal_Anatomy_Figure`).  In most cases, a ModulatorySignal uses the
State's `function <State_Base.function>` to modulate its `value <State_Base.value>`.  The function of every State
assigns one of its parameters as its *ADDITIVE_PARAM* and another as its *MULTIPLICATIVE_PARAM*. The
`modulation <ModulatorySignal.modulation>` attribute of a ModulatorySignal determines which of these to modify when the
State uses it `function <State_Base.function>` to calculate its `value  <State_Base.value>`.  However, the
ModulatorySignal can also be configured to override the State's `value <State_Base.value>` (i.e., assign it directly),
or to disable modulation, using one of the values of `ModulationParam` for its
`modulation <ModulatorySignal.modulation>` attribute (see `ModulatorySignal_Modulation` for a more detailed discussion).

.. _State_Execution:

Execution
---------

States cannot be executed.  They are updated when the Component to which they belong is executed.  InputStates and
ParameterStates belonging to a Mechanism are updated before the Mechanism's function is called.  OutputStates are
updated after the Mechanism's function is called.  When a State is updated, it executes any Projections that project
to it (listed in its `all_afferents <State_Base.all_afferents>` attribute.  It uses the values it receives from any
`PathWayProjections` (listed in its `path_afferents` attribute) as the variable for its
`function <State_Base.function>`. It then executes all of the ModulatoryProjections it receives.  Different
ModulatorySignals may call for different forms of modulation (see `ModulatorySignal_Modulation`).  Accordingly,
it separately sums the values specified by any ModulatorySignals for the *MULTIPLICATIVE_PARAM* of its
`function <State_Base.function>`, and similarly for the *ADDITIVE_PARAM*.  It then applies the summed value for each
to the corresponding parameter of its `function <State_Base.function>`.  If any of the ModulatorySignals specifies
*OVERRIDE*, then the value of that ModulatorySignal is used as the State's `value <State_Base.value>`. Finally,
the State calls its `function <State_Base.function>` to determine its `value <State_Base.value>`.

.. note::
   The change in the value of a `State <State>` does not occur until the Mechanism to which the State belongs is next
   executed; This conforms to a "lazy evaluation" protocol (see :ref:`Lazy Evaluation <LINK>` for an explanation
   of "lazy" updating).

.. _State_Examples:

Examples
========

.. _State_Constructor_Examples:

Usually, States are created automatically by the Mechanism to which they belong.  For example, creating a
TransferMechanism::

    my_mech = pnl.TransferMechanism()

automatically creates an InputState, ParameterStates for its parameters, including the `slope <Linear.slope>` and
`intercept <Linear.intercept>` parameters of its `Linear` `Function <Function>` (its default `function
<TransferMechanism.function>`), and an OutputState (named *RESULT*)::

    print(my_mech.input_states)
    > [(InputState InputState-0)]
    print(my_mech.parameter_states)
    > [(ParameterState intercept), (ParameterState slope), (ParameterState noise), (ParameterState time_constant)]
    print(my_mech.output_states)
    > [(OutputState RESULT)]

.. _State_Constructor_Argument_Examples:

*Using the* **input_states** *argument of a Mechanism constructor.*

When States are specified explicitly, it is usually in an argument of the constructor for the Mechanism to which they
belong.  For example, the following specifies that ``my_mech`` should have an InputState named 'MY INPUT`::

    my_mech = pnl.TransferMechanism(input_states=['MY INPUT'])
    print(my_mech.input_states)
    > [(InputState 'MY INPUT')]

The InputState was specified by a string (for its name) in the **input_states** argument.  It can also be specified in
a variety of other ways, as described `above <State_Specification>` and illustrated in the examples below.
Note that when one or more States is specified in the argument of a Mechanism's constructor, it replaces any defaults
States created by the Mechanism when none are specified (see `note <Mechanism_Default_State_Suppression_Note>`.

.. _State_Value_Spec_Example:

For example, the following specifies the InputState by a value to use as its `default_variable
<InputState.default_variable>` attribute::

    my_mech = pnl.TransferMechanism(input_states=[[0,0])

The value is also used to format the InputState's `value <InputState.value>`, as well as the first (and, in this case,
only) item of the Mechanism's `variable <Mechanism_Base>` (i.e., the one to which the InputState is assigned), as
show below::

    print(my_mech.input_states[0].variable)
    > [0 0]
    print (my_mech.input_state.value)
    > [ 0.  0.]
    print (my_mech.variable)
    > [[0 0]]

Note that in the first print state, the InputState was referenced as the first one in the `input_states
<Mechanism_Base.input_states>` attribute of ``my_mech``;  the second print state references it directly,
as the `primary InputState <Input_State.primary>` of ``my_mech``, using its `input_state <Mechanism_Base.input_state>`
attribute (note the singular).

.. _State_Multiple_InputSates_Example:

*Multiple InputStates*

The **input_states** argument can also be used to create more than one InputState::

    my_mech = pnl.TransferMechanism(input_states=['MY FIRST INPUT', 'MY SECOND INPUT'])
    print(my_mech.input_states)
    > [(InputState MY FIRST INPUT), (InputState MY SECOND INPUT)]

Here, the print statement uses the `input_states <Mechanism_Base.input_states>` attribute, since there is now more
than one InputState.  OutputStates can be specified in a similar way, using the **output_states** argument.

    .. note::
        Although InputStates and OutputStates can be specified in a Mechanism's constructor, ParameterStates cannot;
        those are created automatically when the Mechanism is created, for each of its `user configurable parameters
        <Component_User_Params>`  and those of its `function <Mechanism_Base.function>`.  However, the `value
        <ParameterState.value>` can be specified when the Mechanism is created, or its `function
        <Mechanism_Base.function>` is assigned, and can be accessed and subsequently modified, as described under
        `ParameterState_Specification>`.

.. _State_Standard_OutputStates_Example:

*OutputStates*

The following example specifies two OutputStates for ``my_mech``, using its `Standard OutputStates
<OutputState_Standard>`::

    my_mech = pnl.TransferMechanism(output_states=['RESULT', 'MEAN'])

As with InputStates, specification of OutputStates in the **output_states** argument suppresses the creation of any
default OutPutStates that would have been created if no OutputStates were specified (see `note
<Mechanism_Default_State_Suppression_Note>` above).  For example, TransferMechanisms create a *RESULT* OutputState
by default, that contains the result of their `function <OutputState.function>`.  This default behavior is suppressed
by any specifications in its **output_states** argument.  Therefore, to retain a *RESULTS* OutputState,
it must be included in the **output_states** argument along with any others that are specified, as in the example
above.  If the name of a specified OutputState matches the name of a Standard OutputState <OutputState_Standard>` for
the type of Mechanism, then that is used (as is the case for both of the OutputStates specified for the
`TransferMechanism` in the example above); otherwise, a new OutputState is created.

.. _State_Specification_Dictionary_Examples:

*State specification dictionary*

States can be specified in greater detail using a `State specification dictionary
<State_Specification_Dictionary>`. In the example below, this is used to specify the variable and name of an
InputState::

    my_mech = pnl.TransferMechanism(input_states=[{STATE_TYPE: InputState,
                                                   NAME: 'MY INPUT',
                                                   VARIABLE: [0,0]})

The *STATE_TYPE* entry is included here for completeness, but is not actually needed when the State specification
dicationary is used in **input_states** or **output_states** argument of a Mechanism, since the State's type
is clearly determined by the context of the specification;  however, where that is not clear, then the *STATE_TYPE*
entry must be included.

.. _State_Projections_Examples:

*Projections*

A State specification dictionary can also be used to specify projections to or from the State, also in
a number of different ways.  The most straightforward is to include them in a *PROJECTIONS* entry.  For example, the
following specifies that the InputState of ``my_mech`` receive two Projections,  one from ``source_mech_1`` and another
from ``source_mech_2``, and that its OutputState send one to ``destination_mech``::

    source_mech_1 = pnl.TransferMechanism(name='SOURCE_1')
    source_mech_2 = pnl.TransferMechanism(name='SOURCE_2')
    destination_mech = pnl.TransferMechanism(name='DEST')
    my_mech = pnl.TransferMechanism(name='MY_MECH',
                                    input_states=[{pnl.NAME: 'MY INPUT',
                                                   pnl.PROJECTIONS:[source_mech_1, source_mech_2]}],
                                    output_states=[{pnl.NAME: 'RESULT',
                                                    pnl.PROJECTIONS:[destination_mech]}])

    # Print names of the Projections:
    for projection in my_mech.input_state.path_afferents:
        print(projection.name)
    > MappingProjection from SOURCE_1[RESULT] to MY_MECH[MY INPUT]
    > MappingProjection from SOURCE_2[RESULT] to MY_MECH[MY INPUT]
    for projection in my_mech.output_state.efferents:
        print(projection.name)
    > MappingProjection from MY_MECH[RESULT] to DEST[InputState]


A *PROJECTIONS* entry can contain any of the forms used to `specify a Projection <Projection_Specification>`.
Here, Mechanisms are used, which creates Projections from the `primary InputState <InputState_Primary>` of
``source_mech``, and to the `primary OutputState <OutputState_Primary>` of ``destination_mech``.  Note that
MappingProjections are created, since the Projections specified are between InputStates and OutputStates.
`ModulatoryProjections` can also be specified in a similar way.  The following creates a `GatingMechanism`, and
specifies that the InputState of ``my_mech`` should receive a `GatingProjection` from it::

    my_gating_mech = pnl.GatingMechanism()
    my_mech = pnl.TransferMechanism(name='MY_MECH',
                                    input_states=[{pnl.NAME: 'MY INPUT',
                                                   pnl.PROJECTIONS:[my_gating_mech]}])


.. _State_Modulatory_Projections_Examples:

Conversely, ModulatoryProjections can also be specified from a Mechanism to one or more States that it modulates.  In
the following example, a `ControlMechanism` is created that sends `ControlProjections <ControlProjection>` to the
`drift_rate <BogaczEtAl.drift_rate>` and `threshold <BogaczEtAl.threshold>` ParameterStates of a `DDM` Mechanism::

    my_mech = pnl.DDM(name='MY DDM')
    my_ctl_mech = pnl.ControlMechanism(control_signals=[{pnl.NAME: 'MY DDM DRIFT RATE AND THREHOLD CONTROL SIGNAL',
                                                         pnl.PROJECTIONS: [my_mech.parameter_states[pnl.DRIFT_RATE],
                                                                           my_mech.parameter_states[pnl.THRESHOLD]]}])
    # Print ControlSignals and their ControlProjections
    for control_signal in my_ctl_mech.control_signals:
        print(control_signal.name)
        for control_projection in control_signal.efferents:
            print("\t{}: {}".format(control_projection.receiver.owner.name, control_projection.receiver))
    > MY DDM DRIFT RATE AND THREHOLD CONTROL SIGNAL
    >     MY DDM: (ParameterState drift_rate)
    >     MY DDM: (ParameterState threshold)

Note that a ControlMechanism uses a **control_signals** argument in place of an **output_states** argument (since it
uses `ControlSignal <ControlSignals>` for its `OutputStates <OutputState>`.  In the example above,
both ControlProjections are assigned to a single ControlSignal.  However, they could each be assigned to their own by
specifying them in separate itesm of the **control_signals** argument::

    my_mech = pnl.DDM(name='MY DDM')
    my_ctl_mech = pnl.ControlMechanism(control_signals=[{pnl.NAME: 'DRIFT RATE CONTROL SIGNAL',
                                                         pnl.PROJECTIONS: [my_mech.parameter_states[pnl.DRIFT_RATE]]},
                                                        {pnl.NAME: 'THRESHOLD RATE CONTROL SIGNAL',
                                                         pnl.PROJECTIONS: [my_mech.parameter_states[pnl.THRESHOLD]]}])
    # Print ControlSignals and their ControlProjections...
    > DRIFT RATE CONTROL SIGNAL
    >     MY DDM: (ParameterState drift_rate)
    > THRESHOLD RATE CONTROL SIGNAL
    >     MY DDM: (ParameterState threshold)

Specifying Projections in a State specification dictionary affords flexibility -- for example, naming the State
and/or specifying other attributes.  However, if this is not necessary, the Projections can be used to specify
States directly.  For example, the following, which is much simpler, produces the same result as the previous
example (sans the custom name; though as the printout below shows, the default names are usually pretty clear)::

    my_ctl_mech = pnl.ControlMechanism(control_signals=[my_mech.parameter_states[pnl.DRIFT_RATE],
                                                        my_mech.parameter_states[pnl.THRESHOLD]])
    # Print ControlSignals and their ControlProjections...
    > MY DDM drift_rate ControlSignal
    >    MY DDM: (ParameterState drift_rate)
    > MY DDM threshold ControlSignal
    >    MY DDM: (ParameterState threshold)

.. _State_State_Name_Entry_Example:

*Convenience formats*

There are two convenience formats for specifying States and their Projections in a State specification
dictionary.  The `first <State_State_Name_Entry>` is to use the name of the State as the key for its entry,
and then a list of , as in the following example::

    source_mech_1 = pnl.TransferMechanism()
    source_mech_2 = pnl.TransferMechanism()
    destination_mech = pnl.TransferMechanism()
    my_mech_C = pnl.TransferMechanism(input_states=[{'MY INPUT':[source_mech_1, source_mech_2]}],
                                      output_states=[{'RESULT':[destination_mech]}])

This produces the same result as the first example under `State specification dictionary <State_Projections_Examples>`
above, but it is simpler and easier to read.

The second convenience format is used to specify one or more Projections to/from the States of a single Mechanism
by their name.  It uses the keyword *MECHANISM* to specify the Mechanism, coupled with a State-specific entry to
specify Projections to its States.  This can be useful when a Mechanism must send Projections to several States
of another Mechanism, such as a ControlMechanism that sends ControlProjections to several parameters of a
given Mechanism, as in the following example::

    my_mech = pnl.DDM(name='MY DDM')
    my_ctl_mech = pnl.ControlMechanism(control_signals=[{pnl.MECHANISM: my_mech,
                                                         pnl.PARAMETER_STATES: [pnl.DRIFT_RATE, pnl.THRESHOLD]}])

This produces the same result as the `earlier example <State_Modulatory_Projections_Examples>` of ControlProjections,
once again in a simpler and easier to read form.  However, it be used only to specify Projections for a State to or
from the States of a single Mechanism;  Projections involving other Mechanisms must be assigned to other States.

.. _State_Create_State_Examples:

*Create and then assign a state*

Finally, a State can be created directly using its constructor, and then assigned to a Mechanism.
The following creates an InputState ``my_input_state`` with a `MappingProjection` to it from the
`primary OutputState <OutputState_Primary>` of ``mech_A`` and assigns it to ``mech_B``::

    mech_A = pnl.TransferMechanism()
    my_input_state = pnl.InputState(name='MY INPUT STATE',
                                    projections=[mech_A])
    mech_B = pnl.TransferMechanism(input_states=[my_input_state])
    print(mech_B.input_states)
    > [(InputState MY INPUT STATE)]

The InputState ``my_input_state`` could also have been assigned to ``mech_B`` in one of two other ways:
by explicity adding it using ``mech_B``\\'s `add_states <Mechanism_Base.add_states>` method::

    mech_A = pnl.TransferMechanism()
    my_input_state = pnl.InputState(name='MY INPUT STATE',
                                    projections=[mech_A])
    mech_B = pnl.TransferMechanism()
    mech_B.add_states([my_input_state])

or by constructing it after ``mech_B`` and assigning ``mech_B`` as its owner::

    mech_A = pnl.TransferMechanism()
    mech_B = pnl.TransferMechanism()
    my_input_state = pnl.InputState(name='MY INPUT STATE',
                                    owner=mech_B,
                                    projections=[mech_A])

Note that, in both cases, adding the InputState to ``mech_B`` does not replace its the default InputState generated
when it was created, as shown by printing the `input_states <Mechanism_Base.input_states>` for ``mech_B``::

    print(mech_B.input_states)
    > [(InputState InputState-0), (InputState MY INPUT STATE)]
    > [(InputState InputState-0), (InputState MY INPUT STATE)]

As a consequence, ``my_input_state`` is  **not** the `primary InputState <InputState_Primary>` for ``mech_B`` (i.e.,
input_states[0]), but rather its second InputState (input_states[1]). This is differs from specifying the InputState
as part of the constructor for the Mechanism, which suppresses generation of the default InputState,
as in the first example above (see `note <Mechanism_Default_State_Suppression_Note>`).

COMMENT:

*** ??ADD THESE TO EXAMPLES, HERE OR IN Projection??

    def test_mapping_projection_with_mech_and_state_name_specs(self):
         R1 = pnl.TransferMechanism(output_states=['OUTPUT_1', 'OUTPUT_2'])
         R2 = pnl.TransferMechanism(default_variable=[[0],[0]],
                                    input_states=['INPUT_1', 'INPUT_2'])
         T = pnl.TransferMechanism(input_states=[{pnl.MECHANISM: R1,
                                                  pnl.OUTPUT_STATES: ['OUTPUT_1', 'OUTPUT_2']}],
                                   output_states=[{pnl.MECHANISM:R2,
                                                   pnl.INPUT_STATES: ['INPUT_1', 'INPUT_2']}])

   def test_transfer_mech_input_states_specification_dict_spec(self):
        R1 = TransferMechanism(output_states=['FIRST', 'SECOND'])
        T = TransferMechanism(default_variable=[[0],[0]],
                                      input_states=[{NAME: 'FROM DECISION',
                                                     PROJECTIONS: [R1.output_states['FIRST']]},
                                                    {NAME: 'FROM RESPONSE_TIME',
                                                     PROJECTIONS: R1.output_states['SECOND']}])

   def test_transfer_mech_input_states_projection_in_specification_dict_spec(self):
        R1 = TransferMechanism(output_states=['FIRST', 'SECOND'])
        T = TransferMechanism(input_states=[{NAME: 'My InputState with Two Projections',
                                             PROJECTIONS:[R1.output_states['FIRST'],
                                                          R1.output_states['SECOND']]}])

    def test_transfer_mech_input_states_mech_output_state_in_specification_dict_spec(self):
        R1 = TransferMechanism(output_states=['FIRST', 'SECOND'])
        T = TransferMechanism(input_states=[{MECHANISM: R1,
                                             OUTPUT_STATES: ['FIRST', 'SECOND']}])
        assert len(T.input_states)==1
        for input_state in T.input_states:
            for projection in input_state.path_afferents:
                assert projection.sender.owner is R1

creates a `GatingSignal` with
`GatingProjections <GatingProjection>` to ``mech_B`` and ``mech_C``, and assigns it to ``my_gating_mech``::

    my_gating_signal = pnl.GatingSignal(projections=[mech_B, mech_C])
    my_gating_mech = GatingMechanism(gating_signals=[my_gating_signal]

The `GatingMechanism` created will gate the `primary InputStates <InputState_Primary>` of ``mech_B`` and ``mech_C``.

The following creates

   def test_multiple_modulatory_projections_with_mech_and_state_name_specs(self):

        M = pnl.DDM(name='MY DDM')
        C = pnl.ControlMechanism(control_signals=[{pnl.MECHANISM: M,
                                                   pnl.PARAMETER_STATES: [pnl.DRIFT_RATE, pnl.THRESHOLD]}])
        G = pnl.GatingMechanism(gating_signals=[{pnl.MECHANISM: M,
                                                 pnl.OUTPUT_STATES: [pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME]}])


        M = pnl.DDM(name='MY DDM')
        C = pnl.ControlMechanism(control_signals=[{'DECISION_CONTROL':[M.parameter_states[pnl.DRIFT_RATE],
                                                                       M.parameter_states[pnl.THRESHOLD]]}])
        G = pnl.GatingMechanism(gating_signals=[{'DDM_OUTPUT_GATE':[M.output_states[pnl.DECISION_VARIABLE],
                                                                    M.output_states[pnl.RESPONSE_TIME]]}])

COMMENT

.. _State_Class_Reference:

Class Reference
---------------

"""

import inspect
import numbers
import warnings
from collections import Iterable

import numpy as np
import typecheck as tc

from psyneulink.components.component import Component, ComponentError, InitStatus, component_keywords, function_type
from psyneulink.components.functions.function import LinearCombination, ModulationParam, \
    _get_modulated_param, get_param_value_for_keyword
from psyneulink.components.shellclasses import Mechanism, Process_Base, Projection, State
from psyneulink.globals.keywords import DEFERRED_INITIALIZATION, \
    CONTEXT, COMMAND_LINE, CONTROL_PROJECTION_PARAMS, CONTROL_SIGNAL_SPECS, EXECUTING, FUNCTION, FUNCTION_PARAMS, \
    GATING_PROJECTION_PARAMS, GATING_SIGNAL_SPECS, INITIALIZING, \
    LEARNING, LEARNING_PROJECTION_PARAMS, LEARNING_SIGNAL_SPECS, INPUT_STATES, PARAMETER_STATES, OUTPUT_STATES,\
    MAPPING_PROJECTION_PARAMS, MECHANISM, MATRIX, AUTO_ASSIGN_MATRIX, WEIGHT, EXPONENT,\
    MODULATORY_PROJECTIONS, MODULATORY_SIGNAL, NAME, OWNER, PARAMS, PATHWAY_PROJECTIONS, \
    PREFS_ARG, PROJECTIONS, PROJECTION_PARAMS, PROJECTION_TYPE, RECEIVER, REFERENCE_VALUE, REFERENCE_VALUE_NAME, \
    SENDER, SIZE, STANDARD_OUTPUT_STATES, STATE, STATE_PARAMS, STATE_TYPE, STATE_VALUE, VALUE, VARIABLE, \
    kwAssign, kwStateComponentCategory, kwStateContext, kwStateName, kwStatePrefs
from psyneulink.globals.log import LogEntry, LogLevel
from psyneulink.globals.preferences.componentpreferenceset import kpVerbosePref
from psyneulink.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.globals.registry import register_category
from psyneulink.globals.utilities import ContentAddressableList, MODULATION_OVERRIDE, Modulation, convert_to_np_array, \
    get_args, get_class_attributes, is_value_spec, iscompatible, merge_param_dicts, type_match, is_numeric
from psyneulink.scheduling.timescale import CurrentTime, TimeScale

__all__ = [
    'State_Base', 'state_keywords', 'state_type_keywords', 'StateError', 'StateRegistry',
]

state_keywords = component_keywords.copy()
state_keywords.update({MECHANISM,
                       STATE_TYPE,
                       STATE_VALUE,
                       STATE_PARAMS,
                       PATHWAY_PROJECTIONS,
                       MODULATORY_PROJECTIONS,
                       PROJECTION_TYPE,
                       LEARNING_PROJECTION_PARAMS,
                       LEARNING_SIGNAL_SPECS,
                       CONTROL_PROJECTION_PARAMS,
                       CONTROL_SIGNAL_SPECS,
                       GATING_PROJECTION_PARAMS,
                       GATING_SIGNAL_SPECS
                       })
state_type_keywords = {STATE_TYPE}

STANDARD_STATE_ARGS = {STATE_TYPE, OWNER, REFERENCE_VALUE, VARIABLE, NAME, PARAMS, PREFS_ARG}
STATE_SPEC = 'state_spec'
ADD_STATES = 'ADD_STATES'

def _is_state_class(spec):
    if inspect.isclass(spec) and issubclass(spec, State):
        return True
    return False



# Note:  This is created only for assignment of default projection types for each State subclass (see .__init__.py)
#        Individual stateRegistries (used for naming) are created for each Mechanism
StateRegistry = {}

class StateError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


    def __str__(self):
        return repr(self.error_value)


# State factory method:
# def state(name=NotImplemented, params=NotImplemented, context=None):
#         """Instantiates default or specified subclass of State
#
#        If called w/o arguments or 1st argument=NotImplemented, instantiates default subclass (ParameterState)
#         If called with a name string:
#             - if registered in owner Mechanism's state_registry as name of a subclass, instantiates that class
#             - otherwise, uses it as the name for an instantiation of the default subclass, and instantiates that
#         If a params dictionary is included, it is passed to the subclass
#
#         :param name:
#         :param param_defaults:
#         :return:
#         """
#
#         # Call to instantiate a particular subclass, so look up in MechanismRegistry
#         if name in Mechanism's _stateRegistry:
#             return _stateRegistry[name].mechanismSubclass(params)
#         # Name is not in MechanismRegistry or is not provided, so instantiate default subclass
#         else:
#             # from Components.Defaults import DefaultState
#             return DefaultState(name, params)



# DOCUMENT:  INSTANTATION CREATES AN ATTIRBUTE ON THE OWNER MECHANISM WITH THE STATE'S NAME + kwValueSuffix
#            THAT IS UPDATED BY THE STATE'S value setter METHOD (USED BY LOGGING OF MECHANISM ENTRIES)
class State_Base(State):
    """
    State_Base(        \
    owner,             \
    variable=None,     \
    size=None,         \
    projections=None,  \
    params=None,       \
    name=None,         \
    prefs=None)

    Base class for State.

    .. note::
       State is an abstract class and should NEVER be instantiated by a call to its constructor.
       It should be instantiated using the constructor for a `subclass <State_Subtypes>`.


    COMMENT:
        Description
        -----------
            Represents and updates the value of the input, output or parameter of a Mechanism
                - receives inputs from projections (self.path_afferents, PROJECTIONS)
                - input_states and parameterStates: combines inputs from all projections (mapping, control or learning)
                    and uses this as variable of function to update the value attribute
                - output_states: represent values of output of function
            Value attribute:
                 - is updated by the execute method (which calls State's function)
                 - can be used as sender (input) to one or more projections
                 - can be accessed by KVO
            Constraints:
                - value must be compatible with variable of function
                - value must be compatible with receiver.value for all projections it receives

            Subclasses:
                Must implement:
                    componentType
                    ParamClassDefaults with:
                        + FUNCTION (or <subclass>.function
                        + FUNCTION_PARAMS (optional)
                        + PROJECTION_TYPE - specifies type of projection to use for instantiation of default subclass
                Standard subclasses and constraints:
                    InputState - used as input State for Mechanism;  additional constraint:
                        - value must be compatible with variable of owner's function method
                    OutputState - used as output State for Mechanism;  additional constraint:
                        - value must be compatible with the output of the owner's function
                    MechanismsParameterState - used as State for Mechanism parameter;  additional constraint:
                        - output of function must be compatible with the parameter's value

        Class attributes
        ----------------
            + componentCategory = kwStateFunctionCategory
            + className = STATE
            + suffix
            + classPreference (PreferenceSet): StatePreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.CATEGORY
            + ClassDefaults.variable (value): [0]
            + requiredParamClassDefaultTypes = {FUNCTION_PARAMS : [dict],    # Subclass function params
                                               PROJECTION_TYPE: [str, Projection]})   # Default projection type
            + paramClassDefaults (dict): {PROJECTIONS: []}             # Projections to States
            + owner (Mechansim)
            + FUNCTION (Function class or object, or method)

        Class methods
        -------------
            - set_value(value) -
                validates and assigns value, and updates observers
                returns None
            - update_state(context) -
                updates self.value by combining all projections and using them to compute new value
                return None

        StateRegistry
        -------------
            Used by .__init__.py to assign default projection types to each State subclass
            Note:
            * All States that belong to a given owner are registered in the owner's _stateRegistry,
                which maintains a dict for each State type that it uses, a count for all instances of that type,
                and a dictionary of those instances;  NONE of these are registered in the StateRegistry
                This is so that the same name can be used for instances of a State type by different owners
                    without adding index suffixes for that name across owners,
                    while still indexing multiple uses of the same base name within an owner

        Arguments
        ---------
        - value (value) - establishes type of value attribute and initializes it (default: [0])
        - owner(Mechanism) - assigns State to Mechanism (default: NotImplemented)
        - params (dict):  (if absent, default State is implemented)
            + FUNCTION (method)         |  Implemented in subclasses; used in update()
            + FUNCTION_PARAMS (dict) |
            + PROJECTIONS:<projection specification or list of ones>
                if absent, no projections will be created
                projection specification can be: (see Projection for details)
                    + Projection object
                    + Projection class
                    + specification dict
                    + a list containing any or all of the above
                    if dict, must contain entries specifying a projection:
                        + PROJECTION_TYPE:<Projection class>: must be a subclass of Projection
                        + PROJECTION_PARAMS:<dict>? - must be dict of params for PROJECTION_TYPE
        - name (str): if it is not specified, a default based on the class is assigned in register_category,
                            of the form: className+n where n is the n'th instantiation of the class
        - prefs (PreferenceSet or specification dict):
             if it is omitted, a PreferenceSet will be constructed using the classPreferences for the subclass
             dict entries must have a preference keyPath as their key, and a PreferenceEntry or setting as their value
             (see Description under PreferenceSet for details)
        - context (str): must be a reference to a subclass, or an exception will be raised
    COMMENT

    Attributes
    ----------

    owner : Mechanism or Projection
        object to which the State belongs (see `State_Owner` for additional details).

    base_value : number, list or np.ndarray
        value with which the State was initialized.

    all_afferents : Optional[List[Projection]]
        list of all Projections received by the State (i.e., for which it is a `receiver <Projection_Base.receiver>`.

    path_afferents : Optional[List[Projection]]
        list all `PathwayProjections <PathwayProjection>` received by the State;
        note:  only `InputStates <InputState>` have path_afferents;  the list is empty for other types of States.

    mod_afferents : Optional[List[GatingProjection]]
        list of all `ModulatoryProjections <ModulatoryProjection>` received by the State.

    projections : List[Projection]
        list of all of the `Projections <Projection>` sent or received by the State.

    efferents : Optional[List[Projection]]
        list of outgoing Projections from the State (i.e., for which is a `sender <Projection_Base.sender>`;
        note:  only `OutputStates <OutputState>`, and members of its `ModulatoryProjection <ModulatoryProjection>`
        subclass (`LearningProjection, ControlProjection and GatingProjection) have efferents;  the list is empty for
        InputStates and ParameterStates.

    function : TransferFunction : default determined by type
        used to determine the State's own value from the value of the Projection(s) it receives;  the parameters that
        the TransferFunction identifies as ADDITIVE and MULTIPLICATIVE are subject to modulation by a
        `ModulatoryProjection <ModulatoryProjection_Structure>`.

    value : number, list or np.ndarray
        current value of the State (updated by `update <State_Base.update>` method).

    name : str
        the name of the State. If the State's `initialization has been deferred <State_Deferred_Initialization>`,
        it is assigned a temporary name (indicating its deferred initialization status) until initialization is
        completed, at which time it is assigned its designated name.  If that is the name of an existing State,
        it is appended with an indexed suffix, incremented for each State with the same base name (see `Naming`). If
        the name is not  specified in the **name** argument of its constructor, a default name is assigned by the
        subclass (see subclass for details).

        .. _State_Naming_Note:

        .. note::
            Unlike other PsyNeuLink Components, States names are "scoped" within a Mechanism, meaning that States with
            the same name are permitted in different Mechanisms.  However, they are *not* permitted in the same
            Mechanism: States within a Mechanism with the same base name are appended an index in the order of their
            creation).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the State; if it is not specified in the **prefs** argument of the constructor,
        a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet <LINK>` for
        details).

    """

    componentCategory = kwStateComponentCategory
    className = STATE
    suffix = " " + className
    paramsType = None

    stateAttributes = {FUNCTION, FUNCTION_PARAMS, PROJECTIONS}

    class ClassDefaults(State.ClassDefaults):
        variable = [0]

    registry = StateRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY

    requiredParamClassDefaultTypes = Component.requiredParamClassDefaultTypes.copy()
    requiredParamClassDefaultTypes.update({FUNCTION_PARAMS : [dict],
                                           PROJECTION_TYPE: [str, Projection]})   # Default projection type
    paramClassDefaults = Component.paramClassDefaults.copy()
    paramClassDefaults.update({STATE_TYPE: None})

    @tc.typecheck
    def __init__(self,
                 owner:tc.any(Mechanism, Projection),
                 variable=None,
                 size=None,
                 projections=None,
                 params=None,
                 name=None,
                 prefs=None,
                 context=None,
                 **kargs):
        """Initialize subclass that computes and represents the value of a particular State of a Mechanism

        This is used by subclasses to implement the InputState(s), OutputState(s), and ParameterState(s) of a Mechanism.

        Arguments:
            - owner (Mechanism):
                 Mechanism with which State is associated (default: NotImplemented)
                 this argument is required, as can't instantiate a State without an owning Mechanism
            - variable (value): value of the State:
                must be list or tuple of numbers, or a number (in which case it will be converted to a single-item list)
                must match input and output of State's update function, and any sending or receiving projections
            - size (int or array/list of ints):
                Sets variable to be array(s) of zeros, if **variable** is not specified as an argument;
                if **variable** is specified, it takes precedence over the specification of **size**.
            - params (dict):
                + if absent, implements default State determined by PROJECTION_TYPE param
                + if dict, can have the following entries:
                    + PROJECTIONS:<Projection object, Projection class, dict, or list of either or both>
                        if absent, no projections will be created
                        if dict, must contain entries specifying a projection:
                            + PROJECTION_TYPE:<Projection class> - must be a subclass of Projection
                            + PROJECTION_PARAMS:<dict> - must be dict of params for PROJECTION_TYPE
            - name (str): string with name of State (default: name of owner + suffix + instanceIndex)
            - prefs (dict): dictionary containing system preferences (default: Prefs.DEFAULTS)
            - context (str)
            - **kargs (dict): dictionary of arguments using the following keywords for each of the above kargs:
                # + STATE_VALUE = value
                + VARIABLE = variable
                + STATE_PARAMS = params
                + kwStateName = name
                + kwStatePrefs = prefs
                + kwStateContext = context
                NOTES:
                    * these are used for dictionary specification of a State in param declarations
                    * they take precedence over arguments specified directly in the call to __init__()
        """
        if kargs:
            try:
                variable = self._update_variable(kargs[VARIABLE])
            except (KeyError, NameError):
                pass
            try:
                size = kargs[SIZE]
            except (KeyError, NameError):
                pass
            try:
                projections = kargs[PROJECTIONS]
            except (KeyError, NameError):
                pass
            try:
                name = kargs[kwStateName]
            except (KeyError, NameError):
                pass
            try:
                prefs = kargs[kwStatePrefs]
            except (KeyError, NameError):
                pass
            try:
                context = kargs[kwStateContext]
            except (KeyError, NameError):
                pass

        # Enforce that only called from subclass
        if (not isinstance(context, State_Base) and
                not any(key in context for key in {INITIALIZING, ADD_STATES, COMMAND_LINE})):
            raise StateError("Direct call to abstract class State() is not allowed; "
                                      "use state() or one of the following subclasses: {0}".
                                      format(", ".join("{!s}".format(key) for (key) in StateRegistry.keys())))

        # Enforce that subclass must implement and _execute method
        if not hasattr(self, '_execute'):
            raise StateError("{}, as a subclass of {}, must implement an _execute() method".
                             format(self.__class__.__name__, STATE))

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(projections=projections,
                                                  params=params)

        self.owner = owner

        # If name is not specified, assign default name
        if name is not None and DEFERRED_INITIALIZATION in name:
            name = self._assign_default_state_name(context=name)



        # Register State with StateRegistry of owner (Mechanism to which the State is being assigned)
        register_category(entry=self,
                          base_class=State_Base,
                          name=name,
                          registry=owner._stateRegistry,
                          # sub_group_attr='owner',
                          context=context)

        self.path_afferents = []
        self.mod_afferents = []
        self.efferents = []
        self._stateful = False

        self._path_proj_values = []
        # Create dict with entries for each ModualationParam and initialize - used in update()
        self._mod_proj_values = {}
        for attrib, value in get_class_attributes(ModulationParam):
            self._mod_proj_values[getattr(ModulationParam,attrib)] = []

        # VALIDATE VARIABLE, PARAM_SPECS, AND INSTANTIATE self.function
        super(State_Base, self).__init__(default_variable=variable,
                                         size=size,
                                         param_defaults=params,
                                         name=name,
                                         prefs=prefs,
                                         context=context.__class__.__name__)

        # IMPLEMENTATION NOTE:  MOVE TO COMPOSITION ONCE THAT IS IMPLEMENTED
        # INSTANTIATE PROJECTIONS SPECIFIED IN projections ARG OR params[PROJECTIONS:<>]
        if PROJECTIONS in self.paramsCurrent and self.paramsCurrent[PROJECTIONS]:
            self._instantiate_projections(self.paramsCurrent[PROJECTIONS], context=context)
        else:
            # No projections specified, so none will be created here
            # IMPLEMENTATION NOTE:  This is where a default projection would be implemented
            #                       if params = NotImplemented or there is no param[PROJECTIONS]
            pass

        self.projections = self.path_afferents + self.mod_afferents + self.efferents

        if context is COMMAND_LINE:
            state_list = getattr(owner, owner.stateListAttr[self.__class__])
            if state_list and not self in state_list:
                owner.add_states(self)


    def _handle_size(self, size, variable):
        """Overwrites the parent method in Component.py, because the variable of a State
            is generally 1D, rather than 2D as in the case of Mechanisms"""
        if size is not NotImplemented:

            def checkAndCastInt(x):
                if not isinstance(x, numbers.Number):
                    raise StateError("Size ({}) is not a number.".format(x))
                if x < 1:
                    raise StateError("Size ({}) is not a positive number.".format(x))
                try:
                    int_x = int(x)
                except:
                    raise StateError(
                        "Failed to convert size argument ({}) for {} {} to an integer. For States, size "
                        "should be a number, which is an integer or can be converted to integer.".
                        format(x, type(self), self.name))
                if int_x != x:
                    if hasattr(self, 'prefs') and hasattr(self.prefs, kpVerbosePref) and self.prefs.verbosePref:
                        warnings.warn("When size ({}) was cast to integer, its value changed to {}.".format(x, int_x))
                return int_x

            # region Convert variable to a 1D array, cast size to an integer
            if size is not None:
                size = checkAndCastInt(size)
            try:
                if variable is not None:
                    variable = self._update_variable(np.atleast_1d(variable))
            except:
                raise StateError("Failed to convert variable (of type {}) to a 1D array.".format(type(variable)))
            # endregion

            # region if variable is None and size is not None, make variable a 1D array of zeros of length = size
            if variable is None and size is not None:
                try:
                    variable = self._update_variable(np.zeros(size))
                except:
                    raise ComponentError("variable (perhaps default_variable) was not specified, but PsyNeuLink "
                                         "was unable to infer variable from the size argument, {}. size should be"
                                         " an integer or able to be converted to an integer. Either size or "
                                         "variable must be specified.".format(size))
            #endregion

            if variable is not None and size is not None:  # try tossing this "if" check
                # If they conflict, raise exception
                if size != len(variable):
                    raise StateError("The size arg of {} ({}) conflicts with the length of its variable arg ({})".
                                     format(self.name, size, variable))

        return variable

    def _validate_variable(self, variable, context=None):
        """Validate variable and return validated variable

        Sets self.base_value = self.value = variable
        Insures that it is a number of list or tuple of numbers

        This overrides the class method, to perform more detailed type checking
        See explanation in class method.
        Note:  this method (or the class version) is called only if the parameter_validation attribute is True
        """

        variable = self._update_variable(super(State, self)._validate_variable(variable, context))

        if not context:
            context = kwAssign + ' Base Value'
        else:
            context = context + kwAssign + ' Base Value'

        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """validate projection specification(s)

        Call super (Component._validate_params()
        Validate following params:
            + PROJECTIONS:  <entry or list of entries>; each entry must be one of the following:
                + Projection object
                + Projection class
                + specification dict, with the following entries:
                    + PROJECTION_TYPE:<Projection class> - must be a subclass of Projection
                    + PROJECTION_PARAMS:<dict> - must be dict of params for PROJECTION_TYPE
            # IMPLEMENTATION NOTE: TBI - When learning projection is implemented
            # + FUNCTION_PARAMS:  <dict>, every entry of which must be one of the following:
            #     ParameterState, projection, 2-item tuple or value
        """
        # FIX: PROJECTION_REFACTOR
        #      SHOULD ADD CHECK THAT PROJECTION_TYPE IS CONSISTENT WITH TYPE SPECIFIED BY THE
        #      RECEIVER/SENDER SOCKET SPECIFICATIONS OF CORRESPONDING PROJECTION TYPES (FOR API)

        if PROJECTIONS in request_set and request_set[PROJECTIONS] is not None:
            # if projection specification is an object or class reference, needs to be wrapped in a list
            #    to be consistent with paramClassDefaults and for consistency of treatment below
            projections = request_set[PROJECTIONS]
            if not isinstance(projections, list):
                projections = [projections]
                request_set[PROJECTIONS] = projections
        else:
            # If no projections, ignore (none will be created)
            projections = None

        super(State, self)._validate_params(request_set, target_set, context=context)

        if projections:
            # Validate projection specs in list
            for projection in projections:
                try:
                    issubclass(projection, Projection)
                except TypeError:
                    if (isinstance(projection, Projection) or iscompatible(projection, dict)):
                        continue
                    else:
                        if self.prefs.verbosePref:
                            print("{0} in {1} is not a projection, projection type, or specification dict; "
                                  "{2} will be used to create default {3} for {4}".
                                format(projection,
                                       self.__class__.__name__,
                                       target_set[PROJECTION_TYPE],
                                       self.owner.name))

    def _instantiate_function(self, context=None):
        """Insure that output of function (self.value) is compatible with its input (self.instance_defaults.variable)

        This constraint reflects the role of State functions:
            they simply update the value of the State;
            accordingly, their variable and value must be compatible
        """

        var_is_matrix = False
        # If variable is a matrix (e.g., for the MATRIX ParameterState of a MappingProjection),
        #     it needs to be embedded in a list so that it is properly handled by LinearCombination
        #     (i.e., solo matrix is returned intact, rather than treated as arrays to be combined);
        # Notes:
        #     * this is not a problem when LinearCombination is called in State.update(), since that puts
        #         projection values in a list before calling LinearCombination to combine them
        #     * it is removed from the list below, after calling _instantiate_function
        # FIX: UPDATE WITH MODULATION_MODS REMOVE THE FOLLOWING COMMENT:
        #     * no change is made to PARAMETER_MODULATION_FUNCTION here (matrices may be multiplied or added)
        #         (that is handled by the individual State subclasses (e.g., ADD is enforced for MATRIX ParameterState)
        if (
            (
                (inspect.isclass(self.function) and issubclass(self.function, LinearCombination))
                or isinstance(self.function, LinearCombination)
            )
            and (
                isinstance(self.instance_defaults.variable, np.matrix)
                or (
                    isinstance(self.instance_defaults.variable, np.ndarray)
                    and self.instance_defaults.variable.ndim >= 2
                )
            )
        ):
            self.instance_defaults.variable = [self.instance_defaults.variable]
            var_is_matrix = True

        super()._instantiate_function(context=context)

        # If it is a matrix, remove from list in which it was embedded after instantiating and evaluating function
        if var_is_matrix:
            self.instance_defaults.variable = self.instance_defaults.variable[0]

        # Ensure that output of the function (self.value) is compatible with (same format as) its input (self.instance_defaults.variable)
        #     (this enforces constraint that State functions should only combine values from multiple projections,
        #     but not transform them in any other way;  so the format of its value should be the same as its variable).
        if not iscompatible(self.instance_defaults.variable, self.value):
            raise StateError(
                "Output ({0}: {1}) of function ({2}) for {3} {4} of {5}"
                " must be the same format as its input ({6}: {7})".format(
                    type(self.value).__name__,
                    self.value,
                    self.function.__self__.componentName,
                    self.name,
                    self.__class__.__name__,
                    self.owner.name,
                    self.instance_defaults.variable.__class__.__name__,
                    self.instance_defaults.variable
                )
            )

    # FIX: PROJECTION_REFACTOR
    #      - MOVE THESE TO Projection, WITH self (State) AS ADDED ARG
    #          BOTH _instantiate_projections_to_state AND _instantiate_projections_from_state
    #          CAN USE self AS connectee STATE, since _parse_connection_specs USES SOCKET TO RESOLVE
    #      - ALTERNATIVE: BREAK STATE FIELD OF ProjectionTuple INTO sender AND receiver FIELDS, THEN COMBINE
    #          _instantiate_projections_to_state AND _instantiate_projections_to_state INTO ONE METHOD
    #          MAKING CORRESPONDING ASSIGNMENTS TO send AND receiver FIELDS (WOULD BE CLEARER)

    def _instantiate_projections(self, projections, context=None):
        """Implement any Projection(s) to/from State specified in PROJECTIONS entry of params arg

        Must be implemented by subclasss, to handle interpretation of projection specification(s)
        in a class-appropriate manner:
            PathwayProjections:
              InputState: _instantiate_projections_to_state (.path_afferents)
              ParameterState: disallowed
              OutputState: _instantiate_projections_from_state (.efferents)
              ModulatorySignal: disallowed
            ModulatoryProjections:
              InputState, OutputState and ParameterState:  _instantiate_projections_to_state (mod_afferents)
              ModulatorySignal: _instantiate_projections_from_state (.efferents)
        """

        raise StateError("{} must implement _instantiate_projections (called for {})".
                         format(self.__class__.__name__,
                                self.name))

    # IMPLEMENTATION NOTE:  MOVE TO COMPOSITION ONCE THAT IS IMPLEMENTED
    def _instantiate_projections_to_state(self, projections, context=None):
        """Instantiate projections to a State and assign them to self.path_afferents

        Parses specifications in projection_list into ProjectionTuples

        For projection_spec in ProjectionTuple:
            - if it is a Projection specifiction dicionatry, instantiate it
            - assign self as receiver
            - assign sender
            - if deferred_init and sender is instantiated, complete initialization
            - assign to path_afferents or mod_afferents
            - if specs fail, instantiates a default Projection of type specified by self.params[PROJECTION_TYPE]

        Notes:
            Calls _parse_connection_specs() to parse projection_list into a list of ProjectionTuples;
                 _parse_connection_specs, in turn, calls _parse_projection_spec for each spec in projection_list,
                 which returns either a Projection object or Projection specification dictionary for each spec;
                 that is placed in projection_spec entry of ProjectionTuple (State, weight, exponent, projection_spec).
            When the Projection is instantiated, it assigns itself to
               its receiver's .path_afferents attribute (in Projection_Base._instantiate_receiver) and
               its sender's .efferents attribute (in Projection_Base._instantiate_sender);
               so, need to test for prior assignment to avoid duplicates.
        """

        from psyneulink.components.projections.pathway.pathwayprojection import PathwayProjection_Base
        from psyneulink.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base
        from psyneulink.components.projections.projection import _parse_connection_specs

        default_projection_type = self.paramClassDefaults[PROJECTION_TYPE]

        # If specification is not a list, wrap it in one for consistency of treatment below
        # (since specification can be a list, so easier to treat any as a list)
        projection_list = projections
        if not isinstance(projection_list, list):
            projection_list = [projection_list]

        # Parse each Projection specification in projection_list using self as connectee_state:
        # - calls _parse_projection_spec for each projection_spec in list
        # - validates that Projection specification is compatible with its sender and self
        # - returns ProjectionTuple with Projection specification dictionary for projection_spec
        projection_tuples = _parse_connection_specs(self.__class__, self.owner, projection_list)

        # For Projection in each ProjectionTuple:
        # - instantiate the Projection if necessary, and initialize if possible
        # - insure its value is compatible with self.value FIX: ??and variable is compatible with sender's value
        # - assign it to self.path_afferents or .mod_afferents
        for connection in projection_tuples:

            # Get sender State, weight, exponent and projection for each projection specification
            #    note: weight and exponent for connection have been assigned to Projection in _parse_connection_specs
            state, weight, exponent, projection_spec = connection

            # GET Projection --------------------------------------------------------

            # Projection object
            if isinstance(projection_spec, Projection):
                projection = projection_spec
                projection_type = projection.__class__

            # Projection specification dictionary:
            elif isinstance(projection_spec, dict):
                # Instantiate Projection
                projection_spec[WEIGHT]=weight
                projection_spec[EXPONENT]=exponent
                projection_type = projection_spec.pop(PROJECTION_TYPE, None) or default_projection_type
                projection = projection_type(**projection_spec)

            else:
                raise StateError("PROGRAM ERROR: Unrecognized {} specification ({}) returned "
                                 "from _parse_connection_specs for connection from {} to {} of {}".
                                 format(Projection.__name__,projection_spec,sender.__name__,self.name,self.owner.name))

            # ASSIGN PARAMS

            # Deferred init
            if projection.init_status is InitStatus.DEFERRED_INITIALIZATION:

                proj_sender = projection.init_args[SENDER]
                proj_receiver = projection.init_args[RECEIVER]

                # validate receiver
                if proj_receiver is not None and proj_receiver != self:
                    raise StateError("Projection ({}) assigned to {} of {} already has a receiver ({})".
                                     format(projection_type.__name__, self.name, self.owner.name, proj_receiver.name))
                projection.init_args[RECEIVER] = self

                # parse/validate sender
                if proj_sender:
                    # If the Projection already has State as its sender,
                    #    it must be the same as the one specified in the connection spec
                    if isinstance(proj_sender, State) and proj_sender != state:
                        raise StateError("Projection assigned to {} of {} from {} already has a sender ({})".
                                         format(self.name, self.owner.name, state.name, proj_sender.name))
                    # If the Projection has a Mechanism specified as its sender:
                    elif isinstance(state, State):
                        #    Connection spec (state) is specified as a State,
                        #    so validate that State belongs to Mechanism and is of the correct type
                        sender = _get_state_for_socket(owner=self.owner,
                                                       mech=proj_sender,
                                                       state_spec=state,
                                                       state_types=state.__class__,
                                                       projection_socket=SENDER)
                    elif isinstance(proj_sender, Mechanism) and inspect.isclass(state) and issubclass(state, State):
                        #    Connection spec (state) is specified as State type
                        #    so try to get that State type for the Mechanism
                        sender = _get_state_for_socket(owner=self.owner,
                                                       state_spec=proj_sender,
                                                       state_types=state)
                    else:
                        sender = proj_sender
                else:
                    sender = state
                projection.init_args[SENDER] = sender

                # Construct and assign name
                if isinstance(sender, State):
                    if sender.init_status is InitStatus.DEFERRED_INITIALIZATION:
                        sender_name = sender.init_args[NAME]
                    else:
                        sender_name = sender.name
                    sender_name = sender_name or sender.__class__.__name__
                elif inspect.isclass(sender) and issubclass(sender, State):
                    sender_name = sender.__name__
                else:
                    raise StateError("SENDER of {} to {} of {} is neither a State or State class".
                                     format(projection_type.__name__, self.name, self.owner.name))
                projection._assign_default_projection_name(state=self,
                                                           sender_name=sender_name,
                                                           receiver_name=self.name)

                # If sender has been instantiated, try to complete initialization
                # If not, assume it will be handled later (by Mechanism or Composition)
                if isinstance(sender, State) and sender.init_status is InitStatus.INITIALIZED:
                    projection._deferred_init()


            # VALIDATE (if initialized)

            if projection.init_status is InitStatus.INITIALIZED:

                # FIX: 10/3/17 - VERIFY THE FOLLOWING:
                # IMPLEMENTATION NOTE:
                #     Assume that validation of Projection's variable (i.e., compatibility with sender)
                #         has already been handled by instantiation of the Projection and/or its sender

                # Validate value:
                #    - check that output of projection's function (projection_spec.value) is compatible with
                #        self.variable;  if it is not, raise exception:
                #        the buck stops here; can't modify projection's function to accommodate the State,
                #        or there would be an unmanageable regress of reassigning projections,
                #        requiring reassignment or modification of sender OutputStates, etc.

                # PathwayProjection:
                #    - check that projection's value is compatible with the State's variable
                if isinstance(projection, PathwayProjection_Base):
                    if not iscompatible(projection.value, self.instance_defaults.variable):
                        raise StateError("Output of function for {} ({}) is not compatible with value of {} ({}).".
                                         format(projection.name, projection.value, self.name, self.value))

                # ModualatoryProjection:
                #    - check that projection's value is compatible with value of the function param being modulated
                elif isinstance(projection, ModulatoryProjection_Base):
                    function_param_value = _get_modulated_param(self, projection).function_param_val
                    # Match the projection's value with the value of the function parameter
                    mod_proj_spec_value = type_match(projection.value, type(function_param_value))
                    if (function_param_value is not None
                        and not iscompatible(function_param_value, mod_proj_spec_value)):
                        raise StateError("Output of function for {} ({}) is not compatible with value of {} ({}).".
                                             format(projection.name, projection.value, self.name, self.value))

                # projection._assign_default_projection_name(state=self,
                #                                            sender_name=projection.sender.name,
                #                                            receiver_name=self.name)


            # ASSIGN TO STATE

            # Avoid duplicates, since instantiation of projection may have already called this method
            #    and assigned Projection to self.path_afferents or mod_afferents lists
            if isinstance(projection, PathwayProjection_Base) and not projection in self.path_afferents:
                self.path_afferents.append(projection)
            elif isinstance(projection, ModulatoryProjection_Base) and not projection in self.mod_afferents:
                self.mod_afferents.append(projection)


    def _instantiate_projection_from_state(self, projection_spec, receiver=None, context=None):
        """Instantiate outgoing projection from a State and assign it to self.efferents

        Instantiate Projections specified in projection_list and, for each:
            - insure its self.value is compatible with the projection's function variable
            - place it in self.efferents:

        Notes:
            # LIST VERSION:
            # If receivers is not specified, they must be assigned to projection specs in projection_list
            # Calls _parse_connection_specs() to parse projection_list into a list of ProjectionTuples;
            #    _parse_connection_specs, in turn, calls _parse_projection_spec for each spec in projection_list,
            #    which returns either a Projection object or Projection specification dictionary for each spec;
            #    that is placed in projection_spec entry of ProjectionTuple (State, weight, exponent, projection_spec).

            Calls _parse_connection_specs() to parse projection into a ProjectionTuple;
               _parse_connection_specs, in turn, calls _parse_projection_spec for the projection_spec,
               which returns either a Projection object or Projection specification dictionary for the spec;
               that is placed in projection_spec entry of ProjectionTuple (State, weight, exponent, projection_spec).

            When the Projection is instantiated, it assigns itself to
               its self.path_afferents or .mod_afferents attribute (in Projection_Base._instantiate_receiver) and
               its sender's .efferents attribute (in Projection_Base._instantiate_sender);
               so, need to test for prior assignment to avoid duplicates.

        Returns instantiated Projection
        """
        from psyneulink.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base
        from psyneulink.components.projections.pathway.pathwayprojection import PathwayProjection_Base
        from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
        from psyneulink.components.projections.modulatory.gatingprojection import GatingProjection
        from psyneulink.components.projections.projection import ProjectionTuple, _parse_connection_specs

        # FIX: 10/3/17 THIS NEEDS TO BE MADE SPECIFIC TO EFFERENT PROJECTIONS (I.E., FOR WHICH IT CAN BE A SENDER)
        # default_projection_type = ProjectionRegistry[self.paramClassDefaults[PROJECTION_TYPE]].subclass
        default_projection_type = self.paramClassDefaults[PROJECTION_TYPE]

        # projection_object = None # flags whether projection object has been instantiated; doesn't store object
        # projection_type = None   # stores type of projection to instantiate
        # projection_params = {}


        # IMPLEMENTATION NOTE:  THE FOLLOWING IS WRITTEN AS A LOOP IN PREP FOR GENERALINZING METHOD
        #                       TO HANDLE PROJECTION LIST (AS PER _instantiate_projection_to_state())

        # If projection_spec and/or receiver is not in a list, wrap it in one for consistency of treatment below
        # (since specification can be a list, so easier to treat any as a list)
        projection_list = projection_spec
        if not isinstance(projection_list, list):
            projection_list = [projection_list]

        if not receiver:
            receiver_list = [None] * len(projection_list)
        elif not isinstance(receiver, list):
            receiver_list = [receiver]

        # Parse Projection specification using self as connectee_state:
        # - calls _parse_projection_spec for projection_spec;
        # - validates that Projection specification is compatible with its receiver and self
        # - returns ProjectionTuple with Projection specification dictionary for projection_spec
        projection_tuples = _parse_connection_specs(self.__class__, self.owner, receiver_list)

        # For Projection in ProjectionTuple:
        # - instantiate the Projection if necessary, and initialize if possible
        # - insure its variable is compatible with self.value and its value is compatible with receiver's variable
        # - assign it to self.path_efferents

        for connection, receiver in zip(projection_tuples, receiver_list):

            # VALIDATE CONNECTION AND RECEIVER SPECS

            # Validate that State to be connected to specified in receiver is same as any one specified in connection
            def _get_receiver_state(spec):
                """Get state specification from ProjectionTuple, which itself may be a ProjectionTuple"""
                if isinstance(spec, ProjectionTuple):
                    spec = _parse_connection_specs(connectee_state_type=self.__class__,
                                                   owner=self.owner,
                                                   connections=receiver)
                    return _get_receiver_state(spec[0].state)
                elif isinstance(spec, Projection):
                    return spec.receiver
                # FIX: 11/25/17 -- NEEDS TO CHECK WHETHER PRIMARY SHOULD BE INPUT_STATE OR PARAMETER_STATE
                elif isinstance(spec, Mechanism):
                    return spec.input_state
                return spec
            receiver_state = _get_receiver_state(receiver)
            connection_receiver_state = _get_receiver_state(connection)
            if receiver_state != connection_receiver_state:
                raise StateError("PROGRAM ERROR: State specified as receiver ({}) should "
                                 "be the same as the one specified in the connection {}.".
                                 format(receiver_state, connection_receiver_state))

            if (not isinstance(connection, ProjectionTuple)
                and receiver
                and not isinstance(receiver, (State, Mechanism))
                and not (inspect.isclass(receiver) and issubclass(receiver, (State, Mechanism)))):
                raise StateError("Receiver ({}) of {} from {} must be a {}, {}, a class of one, or a {}".
                                 format(receiver, projection_spec, self.name,
                                        State.__name__, Mechanism.__name__, ProjectionTuple.__name__))

            if isinstance(receiver, Mechanism):
                from psyneulink.components.states.inputstate import InputState
                from psyneulink.components.states.parameterstate import ParameterState

                # If receiver is a Mechanism and Projection is a MappingProjection,
                #    use primary InputState (and warn if verbose is set)
                if isinstance(default_projection_type, (MappingProjection, GatingProjection)):
                    if self.owner.verbosePref:
                        warnings.warn("Receiver {} of {} from {} is a {} and {} is a {}, "
                                      "so its primary {} will be used".
                                      format(receiver, projection_spec, self.name, Mechanism.__name__,
                                             Projection.__name__, default_projection_type.__name__,
                                             InputState.__name__))
                    receiver = receiver.input_state

                    raise StateError("Receiver {} of {} from {} is a {}, but the specified {} is a {} so "
                                     "target {} can't be determined".
                                      format(receiver, projection_spec, self.name, Mechanism.__name__,
                                             Projection.__name__, default_projection_type.__name__,
                                             ParameterState.__name__))


            # GET Projection --------------------------------------------------------

            # Get sender State, weight, exponent and projection for each projection specification
            #    note: weight and exponent for connection have been assigned to Projection in _parse_connection_specs
            connection_receiver, weight, exponent, projection_spec = connection

            # Parse projection_spec and receiver specifications
            #    - if one is assigned and the other is not, assign the one to the other
            #    - if both are assigned, validate they are the same
            #    - if projection_spec is None and receiver is specified, use the latter to construct default Projection

            # Projection object
            if isinstance(projection_spec, Projection):
                projection = projection_spec
                projection_type = projection.__class__

                if projection.init_status is InitStatus.DEFERRED_INITIALIZATION:
                    projection.init_args[RECEIVER] = projection.init_args[RECEIVER] or receiver
                    proj_recvr = projection.init_args[RECEIVER]
                else:
                    projection.receiver = projection.receiver or receiver
                    proj_recvr = projection.receiver
                projection._assign_default_projection_name(state=self,
                                                           sender_name=self.name,
                                                           receiver_name=proj_recvr.name)

            # Projection specification dictionary or None:
            elif isinstance(projection_spec, (dict, None)):

                # Instantiate Projection from specification dict
                projection_type = projection_spec.pop(PROJECTION_TYPE, None) or default_projection_type
                # If Projection was not specified, create default Projection specification dict
                if not (projection_spec or len(projection_spec)):
                    projection_spec = {SENDER: self, RECEIVER: receiver_state}
                projection = projection_type(**projection_spec)
                try:
                    projection.receiver = projection.receiver
                except AttributeError:
                    projection.receiver = receiver
                proj_recvr = projection.receiver

            else:
                rcvr_str = ""
                if receiver:
                    if isinstance(receiver, State):
                        rcvr_str = " to {}".format(receiver.name)
                    else:
                        rcvr_str = " to {}".format(receiver.__name__)
                raise StateError("PROGRAM ERROR: Unrecognized {} specification ({}) returned "
                                 "from _parse_connection_specs for connection from {} of {}{}".
                                 format(Projection.__name__,projection_spec,self.name,self.owner.name,rcvr_str))

            # Validate that receiver and projection_spec receiver are now the same
            receiver = proj_recvr or receiver  # If receiver was not specified, assign it receiver from projection_spec
            if proj_recvr and receiver and not proj_recvr is receiver:
                # Note: if proj_recvr is None, it will be assigned under handling of deferred_init below
                raise StateError("Receiver ({}) specified for Projection ({}) "
                                 "is not the same as the one specified in {} ({})".
                                 format(proj_recvr, projection.name, ProjectionTuple.__name__, receiver))

            # ASSIGN REMAINING PARAMS

            # Deferred init
            if projection.init_status is InitStatus.DEFERRED_INITIALIZATION:

                projection.init_args[SENDER] = self

                # Construct and assign name
                if isinstance(receiver, State):
                    if receiver.init_status is InitStatus.DEFERRED_INITIALIZATION:
                        receiver_name = receiver.init_args[NAME]
                    else:
                        receiver_name = receiver.name
                elif inspect.isclass(receiver) and issubclass(receiver, State):
                    receiver_name = receiver.__name__
                else:
                    raise StateError("RECEIVER of {} to {} of {} is neither a State or State class".
                                     format(projection_type.__name__, self.name, self.owner.name))
                if isinstance(projection, PathwayProjection_Base):
                    projection_name = projection_type.__name__ + " from " + self.owner.name + " to " + receiver_name

                else:
                    if isinstance(receiver, State):
                        receiver_name = receiver.owner.name + " " + receiver.name
                    else:
                        receiver_name = receiver.__name__
                    projection_name = projection_type.__name__ + " for " + receiver_name
                # projection.init_args[NAME] = projection.init_args[NAME] or projection_name
                projection._assign_default_projection_name(state=self,
                                                           sender_name=self.name,
                                                           receiver_name=receiver_name)


                # If receiver has been instantiated, try to complete initialization
                # If not, assume it will be handled later (by Mechanism or Composition)
                if isinstance(receiver, State) and receiver.init_status is InitStatus.INITIALIZED:
                    projection._deferred_init()

            # VALIDATE (if initialized or being initialized (UNSET))

            if projection.init_status in {InitStatus.INITIALIZED, InitStatus.UNSET}:

                # If still being initialized, then assign sender and receiver as necessary
                if projection.init_status is InitStatus.UNSET:
                    if not isinstance(projection.sender, State):
                        projection.sender = self
                        projection.instance_defaults.variable = self.value

                    if not isinstance(projection.receiver, State):
                        projection.receiver = receiver_state

                    # FIX: 10/3/17 -- ??IS THE FOLLOWING LEGIT? (IT IS DONE TO PASS TESTS BELOW)
                    # FIX:            [HASN'T BEEN ASSIGNED IN __init__ YET BECAUSE CALL CAN BE FROM
                    # FIX:            LearningProjection._instantiate_sender() WHICH IS ITSELF CALLED
                    # FIX:            FROM _instantiate_attributes_before_function
                    projection.value = receiver.variable

                # Validate variable
                #    - check that input to Projection is compatible with self.value
                if not iscompatible(self.value, projection.instance_defaults.variable):
                    raise StateError("Input to {} ({}) is not compatible with the value ({}) of "
                                     "the State from which it is supposed to project ({})".
                                     format(projection.name, projection.variable, self.value, self.name))

                # Validate value:
                #    - check that output of projection's function (projection_spec.value) is compatible with
                #        variable of the State to which it projects;  if it is not, raise exception:
                #        the buck stops here; can't modify projection's function to accommodate the State,
                #        or there would be an unmanageable regress of reassigning projections,
                #        requiring reassignment or modification of sender OutputStates, etc.

                # PathwayProjection:
                #    - check that projection's value is compatible with the receiver's variable
                if isinstance(projection, PathwayProjection_Base):
                    if not iscompatible(projection.value, receiver.instance_defaults.variable):
                        raise StateError("Output of {} ({}) is not compatible with the variable ({}) of "
                                         "the State to which it is supposed to project ({}).".
                                         format(projection.name, projection.value,
                                                receiver.instance_defaults.variable, receiver.name, ))

                # ModualatoryProjection:
                #    - check that projection's value is compatible with value of the function param being modulated
                elif isinstance(projection, ModulatoryProjection_Base):
                    function_param_value = _get_modulated_param(receiver, projection).function_param_val
                    # Match the projection's value with the value of the function parameter
                    mod_proj_spec_value = type_match(projection.value, type(function_param_value))
                    if (function_param_value is not None
                        and not iscompatible(function_param_value, mod_proj_spec_value)):
                        raise StateError("Output of {} ({}) is not compatible with the value of {} ({}).".
                                         format(projection.name,mod_proj_spec_value,receiver.name,function_param_value))

                projection._assign_default_projection_name(state=self,
                                                           sender_name=self.name,
                                                           receiver_name=projection.receiver.name)


            # ASSIGN TO STATE

            # Avoid duplicates, since instantiation of projection may have already called this method
            #    and assigned Projection to self.efferents
            if not projection in self.efferents:
                self.efferents.append(projection)

            return projection

    def _get_primary_state(self, mechanism):
        raise StateError("PROGRAM ERROR: {} does not implement _get_primary_state method".
                         format(self.__class__.__name__))

    def _parse_state_specific_specs(self, owner, state_dict, state_specific_spec):
        """Parse parameters in State specification tuple specific to each subclass

        Called by _parse_state_spec()
        state_dict contains standard args for State constructor passed to _parse_state_spec
        state_specific_spec is either a:
            - tuple containing a specification for the State and/or Projections to/from it
            - a dict containing state-specific parameters to be processed

         Returns two values:
         - state_spec:  specification for the State;
                          - can be None (this is usually the case when state_specific_spec
                            is a tuple specifying a Projection that will be used to specify the state)
                          - if a value is returned, that is used by _parse_state_spec in a recursive call to
                            parse the specified value as the State specification
         - params: state-specific parameters that will be included in the PARAMS entry of the State specification dict
         """
        raise StateError("PROGRAM ERROR: {} does not implement _parse_state_specific_specs method".
                         format(self.__class__.__name__))

    def update(self, params=None, time_scale=TimeScale.TRIAL, context=None):
        """Update each projection, combine them, and assign return result

        Call update for each projection in self.path_afferents (passing specified params)
        Note: only update LearningSignals if context == LEARNING; otherwise, just get their value
        Call self.function (default: LinearCombination function) to combine their values
        Returns combined values of
    """

        # region SET UP ------------------------------------------------------------------------------------------------
        # Get State-specific param_specs
        try:
            # Get State params
            self.stateParams = params[self.paramsType]
        except (KeyError, TypeError):
            self.stateParams = {}
        except (AttributeError):
            raise StateError("PROGRAM ERROR: paramsType not specified for {}".format(self.name))
        #endregion

        # Flag format of input
        if isinstance(self.value, numbers.Number):
            # Treat as single real value
            value_is_number = True
        else:
            # Treat as vector (list or np.array)
            value_is_number = False

        # region AGGREGATE INPUT FROM PROJECTIONS -----------------------------------------------------------------------------

        # Get type-specific params from PROJECTION_PARAMS
        mapping_params = merge_param_dicts(self.stateParams, MAPPING_PROJECTION_PARAMS, PROJECTION_PARAMS)
        learning_projection_params = merge_param_dicts(self.stateParams, LEARNING_PROJECTION_PARAMS, PROJECTION_PARAMS)
        control_projection_params = merge_param_dicts(self.stateParams, CONTROL_PROJECTION_PARAMS, PROJECTION_PARAMS)
        gating_projection_params = merge_param_dicts(self.stateParams, GATING_PROJECTION_PARAMS, PROJECTION_PARAMS)
        #endregion

        #For each projection: get its params, pass them to it, get the projection's value, and append to relevant list
        self._path_proj_values = []
        for value in self._mod_proj_values:
            self._mod_proj_values[value] = []

        from psyneulink.components.process import ProcessInputState
        from psyneulink.components.projections.pathway.pathwayprojection import PathwayProjection_Base
        from psyneulink.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base
        from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
        from psyneulink.components.projections.modulatory.learningprojection import LearningProjection
        from psyneulink.components.projections.modulatory.controlprojection import ControlProjection
        from psyneulink.components.projections.modulatory.gatingprojection import GatingProjection

        # If owner is a Mechanism, get its execution_id
        if isinstance(self.owner, (Mechanism, Process_Base)):
            self_id = self.owner._execution_id
        # If owner is a MappingProjection, get it's sender's execution_id
        elif isinstance(self.owner, MappingProjection):
            self_id = self.owner.sender.owner._execution_id
        else:
            raise StateError("PROGRAM ERROR: Object ({}) of type {} has a {}, but this is only allowed for "
                             "Mechanisms and MappingProjections".
                             format(self.owner.name, self.owner.__class__.__name__, self.__class__.__name__,))


        modulatory_override = False

        # Get values of all Projections
        for projection in self.all_afferents:

            # Only update if sender has also executed in this round
            #     (i.e., has same execution_id as owner)
            # Get sender's execution id
            if hasattr(projection, 'sender'):
                sender = projection.sender
            else:
                if self.verbosePref:
                    warnings.warn("{} to {} {} of {} ignored [has no sender]".format(projection.__class__.__name__,
                                                                                     self.name,
                                                                                     self.__class__.__name__,
                                                                                     self.owner.name))
                continue

            sender_id = sender.owner._execution_id
            if isinstance(sender.owner, Mechanism):
                if (not sender.owner.ignore_execution_id) and (sender_id != self_id):
                    continue
            else:
                if sender_id != self_id:
                    continue

            # Only accept projections from a Process to which the owner Mechanism belongs
            if isinstance(sender, ProcessInputState):
                if not sender.owner in self.owner.processes.keys():
                    continue

            # Merge with relevant projection type-specific params
            if isinstance(projection, MappingProjection):
                projection_params = merge_param_dicts(self.stateParams, projection.name, mapping_params, )
            elif isinstance(projection, LearningProjection):
                projection_params = merge_param_dicts(self.stateParams, projection.name, learning_projection_params)
            elif isinstance(projection, ControlProjection):
                projection_params = merge_param_dicts(self.stateParams, projection.name, control_projection_params)
            elif isinstance(projection, GatingProjection):
                projection_params = merge_param_dicts(self.stateParams, projection.name, gating_projection_params)
            if not projection_params:
                projection_params = None

            # FIX: UPDATE FOR LEARNING
            # Update LearningSignals only if context == LEARNING;  otherwise, assign zero for projection_value
            # Note: done here rather than in its own method in order to exploit parsing of params above
            if isinstance(projection, LearningProjection) and not LEARNING in context:
                # projection_value = projection.value
                projection_value = projection.value * 0.0
            else:
                projection_value = projection.execute(params=projection_params,
                                                      time_scale=time_scale,
                                                      context=context)

            # If this is initialization run and projection initialization has been deferred, pass
            if INITIALIZING in context and projection.init_status is InitStatus.DEFERRED_INITIALIZATION:
                continue

            if isinstance(projection, PathwayProjection_Base):
                # Add projection_value to list of TransmissiveProjection values (for aggregation below)
                self._path_proj_values.append(projection_value)

            # If it is a ModulatoryProjection, add its value to the list in the dict entry for the relevant mod_param
            elif isinstance(projection, ModulatoryProjection_Base):
                # Get the meta_param to be modulated from modulation attribute of the  projection's ModulatorySignal
                #    and get the function parameter to be modulated to type_match the projection value below
                mod_meta_param, mod_param_name, mod_param_value = _get_modulated_param(self, projection)
                # If meta_param is DISABLE, ignore the ModulatoryProjection
                if mod_meta_param is Modulation.DISABLE:
                    continue
                if mod_meta_param is Modulation.OVERRIDE:
                    # If paramValidationPref is set, allow all projections to be processed
                    #    to be sure there are no other conflicting OVERRIDES assigned
                    if self.owner.paramValidationPref:
                        if modulatory_override:
                            raise StateError("Illegal assignment of {} to more than one {} ({} and {})".
                                             format(MODULATION_OVERRIDE, MODULATORY_SIGNAL,
                                                    projection.name, modulatory_override[2]))
                        modulatory_override = (MODULATION_OVERRIDE, projection_value, projection)
                        continue
                    # Otherwise, for efficiency, assign OVERRIDE value to State here and return
                    else:
                        self.value = type_match(projection_value, type(self.value))
                        return
                else:
                    mod_value = type_match(projection_value, type(mod_param_value))
                self._mod_proj_values[mod_meta_param].append(mod_value)

        # Handle ModulatoryProjection OVERRIDE
        #    if there is one and it wasn't been handled above (i.e., if paramValidation is set)
        if modulatory_override:
            self.value = type_match(modulatory_override[1], type(self.value))
            return

        # AGGREGATE ModulatoryProjection VALUES  -----------------------------------------------------------------------

        # For each modulated parameter of the State's function,
        #    combine any values received from the relevant projections into a single modulation value
        #    and assign that to the relevant entry in the params dict for the State's function.
        for mod_param, value_list in self._mod_proj_values.items():
            if value_list:
                aggregated_mod_val = mod_param.reduce(value_list)
                function_param = self.function_object.params[mod_param.attrib_name]
                if not FUNCTION_PARAMS in self.stateParams:
                    self.stateParams[FUNCTION_PARAMS] = {function_param: aggregated_mod_val}
                else:
                    self.stateParams[FUNCTION_PARAMS].update({function_param: aggregated_mod_val})

        # CALL STATE'S function TO GET ITS VALUE  ----------------------------------------------------------------------
        try:
            # pass only function params (which implement the effects of any ModulatoryProjections)
            function_params = self.stateParams[FUNCTION_PARAMS]
        except (KeyError, TypeError):
            function_params = None
        self.value = self._execute(function_params=function_params, context=context)

    def execute(self, input=None, time_scale=None, params=None, context=None):
        return self.function(variable=input, params=params, time_scale=time_scale, context=context)

    @property
    def owner(self):
        return self._owner

    @owner.setter
    def owner(self, assignment):
        self._owner = assignment

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, assignment):

        # MODIFIED 7/8/17 OLD:
        # from math import isnan
        # if isinstance(assignment, np.ndarray) and assignment.ndim == 2 and isnan(assignment[0][0]):
        #             TEST = True
        # MODIFIED 7/8/17 END

        self._value = assignment

        # Store value in log if specified
        # Get logPref
        if self.prefs:
            log_pref = self.prefs.logPref

        # Get context
        try:
            curr_frame = inspect.currentframe()
            prev_frame = inspect.getouterframes(curr_frame, 2)
            context = inspect.getargvalues(prev_frame[1][0]).locals['context']
        except KeyError:
            context = ""

        # If context is consistent with log_pref, record value to log
        if (log_pref is LogLevel.ALL_ASSIGNMENTS or
                (log_pref is LogLevel.EXECUTION and EXECUTING in context) or
                (log_pref is LogLevel.VALUE_ASSIGNMENT and (EXECUTING in context and kwAssign in context))):
            self.owner.log.entries[self.name] = LogEntry(CurrentTime(), context, assignment)
            # self.owner.log.entries[self.name] = LogEntry(CentralClock, context, assignment)

    @property
    def projections(self):
        return self._projections

    @projections.setter
    def projections(self, assignment):
        self._projections = assignment

    @property
    def all_afferents(self):
        return self.path_afferents + self.mod_afferents

    def _assign_default_state_name(self, context=None):
        return False


def _instantiate_state_list(owner,
                            state_list,              # list of State specs, (state_spec, params) tuples, or None
                            state_type,              # StateType subclass
                            state_param_identifier,  # used to specify state_type State(s) in params[]
                            reference_value,         # value(s) used as default for State and to check compatibility
                            reference_value_name,    # name of reference_value type (e.g. variable, output...)
                            context=None):
    """Instantiate and return a ContentAddressableList of States specified in state_list

    Arguments:
    - state_type (class): State class to be instantiated
    - state_list (list): List of State specifications (generally from owner.paramsCurrent[kw<State>]),
                             each item of which must be a:
                                 string (used as name)
                                 number (used as constraint value)
                                 dict (key=name, value=reference_value or param dict)
                         if None, instantiate a single default State using reference_value as state_spec
    - state_param_identifier (str): kw used to identify set of States in params;  must be one of:
        - INPUT_STATE
        - OUTPUT_STATE
    - reference_value (2D np.array): set of 1D np.ndarrays used as default values and
        for compatibility testing in instantiation of State(s):
        - INPUT_STATE: self.instance_defaults.variable
        - OUTPUT_STATE: self.value
        ?? ** Note:
        * this is ignored if param turns out to be a dict (entry value used instead)
    - reference_value_name (str):  passed to State._instantiate_state(), used in error messages
    - context (str)

    If state_list is None:
        - instantiate a default State using reference_value,
        - place as the single entry of the list returned.
    Otherwise, if state_list is:
        - a single value:
            instantiate it (if necessary) and place as the single entry in an OrderedDict
        - a list:
            instantiate each item (if necessary) and place in a ContentAddressableList
    In each case, generate a ContentAddressableList with one or more entries, assigning:
        # the key for each entry the name of the State if provided,
        #     otherwise, use MECHANISM<state_type>States-n (incrementing n for each additional entry)
        # the State value for each entry to the corresponding item of the Mechanism's state_type State's value
        # the dict to both self.<state_type>States and paramsCurrent[MECHANISM<state_type>States]
        # self.<state_type>State to self.<state_type>States[0] (the first entry of the dict)
    Notes:
        * if there is only one State, but the value of the Mechanism's state_type has more than one item:
            assign it to the sole State, which is assumed to have a multi-item value
        * if there is more than one State:
            the number of States must match length of Mechanisms state_type value or an exception is raised
    """

    # If no States were passed in, instantiate a default state_type using reference_value
    if not state_list:
        # assign reference_value as single item in a list, to be used as state_spec below
        state_list = reference_value

        # issue warning if in VERBOSE mode:
        if owner.prefs.verbosePref:
            print("No {0} specified for {1}; default will be created using {2} "
                  "of function ({3}) as its value".format(state_param_identifier,
                                                          owner.__class__.__name__,
                                                          reference_value_name,
                                                          reference_value))

    # States should be either in a list, or possibly an np.array (from reference_value assignment above):
    if not isinstance(state_list, (ContentAddressableList, list, np.ndarray)):
        # This shouldn't happen, as items of state_list should be validated to be one of the above in _validate_params
        raise StateError("PROGRAM ERROR: {} for {} is not a recognized \'{}\' specification for {}; "
                         "it should have been converted to a list in Mechanism._validate_params)".
                         format(state_list, owner.name, state_param_identifier, owner.__class__.__name__))


    # VALIDATE THAT NUMBER OF STATES IS COMPATIBLE WITH NUMBER OF ITEMS IN reference_values

    num_states = len(state_list)
    # Check that reference_value is an indexable object, the items of which are the constraints for each State
    # Notes
    # * generally, this will be a list or an np.ndarray (either >= 2D np.array or with a dtype=object)
    # * for OutputStates, this should correspond to its value
    try:
        # Insure that reference_value is an indexible item (list, >=2D np.darray, or otherwise)
        num_constraint_items = len(reference_value)
    except:
        raise StateError("PROGRAM ERROR: reference_value ({0}) for {1} of {2}"
                             " must be an indexable object (e.g., list or np.ndarray)".
                             format(reference_value, reference_value_name, state_type.__name__))
    # If number of States does not equal the number of items in reference_value, raise exception
    if num_states != num_constraint_items:
        if num_states > num_constraint_items:
            comparison_string = 'more'
        else:
            comparison_string = 'fewer'
        raise StateError("There are {} {}s specified ({}) than the number of items ({}) "
                             "in the {} of the function for \'{}\'".
                             format(comparison_string,
                                    state_param_identifier,
                                    num_states,
                                    num_constraint_items,
                                    reference_value_name,
                                    owner.name))


    # INSTANTIATE EACH STATE

    states = ContentAddressableList(component_type=State_Base,
                                    name=owner.name+' ContentAddressableList of ' + state_param_identifier)
    # For each state, pass state_spec and the corresponding item of reference_value to _instantiate_state

    for index, state_spec in enumerate(state_list):

        # # Get name of state, and use as index to assign to states ContentAddressableList
        # default_name = state_type._assign_default_state_name(state_type)
        # name = default_name or None

        state = _instantiate_state(state_type=state_type,
                                   owner=owner,
                                   reference_value=reference_value[index],
                                   reference_value_name=reference_value_name,
                                   state_spec=state_spec,
                                   # name=name,
                                   context=context)
        # # Get name of state, and use as index to assign to states ContentAddressableList
        # default_name = state._assign_default_state_name()
        # if default_name:
        #      state_name = default_name
        # elif state.init_status is InitStatus.DEFERRED_INITIALIZATION:
        #     state_name = state.init_args[NAME]
        # else:
        #     state_name = state.name
        #
        # states[state_name] = state

        states[state.name] = state

    return states


@tc.typecheck
def _instantiate_state(state_type:_is_state_class,           # State's type
                       owner:tc.any(Mechanism, Projection),  # State's owner
                       reference_value,                      # constraint for State's value and default for variable
                       name:tc.optional(str)=None,           # state's name if specified
                       variable=None,                        # used as default value for state if specified
                       params=None,                          # state-specific params
                       prefs=None,
                       context=None,
                       **state_spec):                        # captures *state_spec* arg and any other non-standard ones
    """Instantiate a State of specified type, with a value that is compatible with reference_value

    This is the interface between the various ways in which a state can be specified and the State's constructor
        (see list below, and `State_Specification` in docstring above).
    It calls _parse_state_spec to:
        create a State specification dictionary (the canonical form) that can be passed to the State's constructor;
        place any State subclass-specific params in the params entry;
        call _parse_state_specific_specs to parse and validate the values of those params
    It checks that the State's value is compatible with the reference value and/or any projection specifications

    # Constraint value must be a number or a list or tuple of numbers
    # (since it is used as the variable for instantiating the requested state)

    If state_spec is a:
    + State class:
        implement default using reference_value
    + State object:
        check compatibility of value with reference_value
        check owner is owner (if not, user is given options in _check_state_ownership)
    + 2-item tuple:
        assign first item to state_spec
        assign second item to STATE_PARAMS{PROJECTIONS:<projection>}
    + Projection object:
        assign reference_value to value
        assign projection to STATE_PARAMS{PROJECTIONS:<projection>}
    + Projection class (or keyword string constant for one):
        assign reference_value to value
        assign projection class spec to STATE_PARAMS{PROJECTIONS:<projection>}
    + specification dict for State
        check compatibility of STATE_VALUE with reference_value

    Returns a State or None
    """


    # Parse reference value to get actual value (in case it is, itself, a specification dict)
    reference_value_dict = _parse_state_spec(owner=owner,
                                             state_type=state_type,
                                             state_spec=reference_value,
                                             value=None,
                                             params=None)
    # Its value is assigned to the VARIABLE entry (including if it was originally just a value)
    reference_value = reference_value_dict[VARIABLE]

    parsed_state_spec = _parse_state_spec(state_type=state_type,
                                          owner=owner,
                                          reference_value=reference_value,
                                          name=name,
                                          variable=variable,
                                          params=params,
                                          prefs=prefs,
                                          context=context,
                                          **state_spec)

    # STATE SPECIFICATION IS A State OBJECT ***************************************
    # Validate and return

    # - check that its value attribute matches the reference_value
    # - check that it doesn't already belong to another owner
    # - if either fails, assign default State

    if isinstance(parsed_state_spec, State):

        state = parsed_state_spec

        # State initialization was deferred (owner or reference_value was missing), so
        #    assign owner, variable, and/or reference_value
        #    if they were not specified in call to _instantiate_state
        if state.init_status is InitStatus.DEFERRED_INITIALIZATION:
            if not state.init_args[OWNER]:
                state.init_args[OWNER] = owner
            if not VARIABLE in state.init_args or state.init_args[VARIABLE] is None:
                state.init_args[VARIABLE] = owner.instance_defaults.variable[0]
            if not hasattr(state, REFERENCE_VALUE):
                if REFERENCE_VALUE in state.init_args and state.init_args[REFERENCE_VALUE] is not None:
                    state.reference_value = state.init_args[REFERENCE_VALUE]
                else:
                    # state.reference_value = owner.instance_defaults.variable[0]
                    state.reference_value = state.init_args[VARIABLE]
            state.init_args[CONTEXT]=context
            state._deferred_init()

        if variable:
                state.instance_defaults.variable = variable

        # # FIX: 10/3/17 - CHECK THE FOLLOWING BY CALLING STATE-SPECIFIC METHOD?
        # # FIX: DO THIS IN _parse_connection_specs?
        # # *reference_value* arg should generally be a constraint for the value of the State;  however,
        # #     if state_spec is a Projection, and method is being called from:
        # #         InputState, reference_value should be the projection's value;
        # #         ParameterState, reference_value should be the projection's value;
        # #         OutputState, reference_value should be the projection's variable
        # # variable:
        # #   InputState: set of projections it receives
        # #   ParameterState: value of its sender
        # #   OutputState: value[INDEX] of its owner
        # # FIX: ----------------------------------------------------------

        # FIX: THIS SHOULD ONLY APPLY TO InputState AND ParameterState; WHAT ABOUT OutputState?
        # State's assigned value is incompatible with its reference_value (presumably its owner Mechanism's variable)
        reference_value = reference_value if reference_value is not None else state.reference_value
        if not iscompatible(state.value, reference_value):
            raise StateError("{}'s value attribute ({}) is incompatible with the {} ({}) of its owner ({})".
                             format(state.name, state.value, REFERENCE_VALUE, state.reference_value, owner.name))

        # State has already been assigned to an owner
        if state.owner is not None and not state.owner is owner:
            raise StateError("State {} does not belong to the owner for which it is specified ({})".
                             format(state.name, owner.name))
        return state

    # STATE SPECIFICATION IS A State specification dictionary ***************************************
    #    so, call constructor to instantiate State

    state_spec_dict = parsed_state_spec

    state_spec_dict.pop(VALUE, None)

    #  Convert reference_value to np.array to match state_variable (which, as output of function, will be an np.array)
    if state_spec_dict[REFERENCE_VALUE] is None:
        state_spec_dict[REFERENCE_VALUE] = state_spec_dict[VARIABLE]
    state_spec_dict[REFERENCE_VALUE] = convert_to_np_array(state_spec_dict[REFERENCE_VALUE],1)

    # INSTANTIATE STATE:

    # IMPLEMENTATION NOTE:
    # - setting prefs=NotImplemented causes TypeDefaultPreferences to be assigned (from ComponentPreferenceSet)
    # - alternative would be prefs=owner.prefs, causing state to inherit the prefs of its owner;
    state_type = state_spec_dict.pop(STATE_TYPE, None)
    if REFERENCE_VALUE_NAME in state_spec_dict:
        del state_spec_dict[REFERENCE_VALUE_NAME]
    if state_spec_dict[PARAMS] and  REFERENCE_VALUE_NAME in state_spec_dict[PARAMS]:
        del state_spec_dict[PARAMS][REFERENCE_VALUE_NAME]

    # Implement default State
    state = state_type(**state_spec_dict, context=context)

# FIX LOG: ADD NAME TO LIST OF MECHANISM'S VALUE ATTRIBUTES FOR USE BY LOGGING ENTRIES
    # This is done here to register name with Mechanism's stateValues[] list
    # It must be consistent with value setter method in State
# FIX LOG: MOVE THIS TO MECHANISM STATE __init__ (WHERE IT CAN BE KEPT CONSISTENT WITH setter METHOD??
#      OR MAYBE JUST REGISTER THE NAME, WITHOUT SETTING THE
# FIX: 2/17/17:  COMMENTED THIS OUT SINCE IT CREATES AN ATTRIBUTE ON OWNER THAT IS NAMED <state.name.value>
#                NOT SURE WHAT THE PURPOSE IS
#     setattr(owner, state.name+'.value', state.value)

    state._validate_against_reference_value(reference_value)

    return state


def _parse_state_type(owner, state_spec):
    """Determine State type for state_spec and return type

    Determine type from context and/or type of state_spec if the latter is not a `State <State>` or `Mechanism
    <Mechanism>`.
    """

    # State class reference
    if isinstance(state_spec, State):
        return type(state_spec)

    # keyword for a State or name of a standard_output_state or of State itself
    if isinstance(state_spec, str):

        # State keyword
        if state_spec in state_type_keywords:
            import sys
            return getattr(sys.modules['PsyNeuLink.Components.States.'+state_spec], state_spec)

        # Try as name of State
        for state_attr in [INPUT_STATES, PARAMETER_STATES, OUTPUT_STATES]:
            state_list = getattr(owner, state_attr)
            try:
                state = state_list[state_spec]
                return state.__class__
            except TypeError:
                pass

        # standard_output_state
        if hasattr(owner, STANDARD_OUTPUT_STATES):
            # check if string matches the name entry of a dict in standard_output_states
            # item = next((item for item in owner.standard_output_states.names if state_spec is item), None)
            # if item is not None:
            #     # assign dict to owner's output_state list
            #     return owner.standard_output_states.get_dict(state_spec)
            # from PsyNeuLink.Components.States.OutputState import StandardOutputStates
            if owner.standard_output_states.get_state_dict(state_spec):
                from psyneulink.components.States.OutputState import OutputState
                return OutputState

    # State specification dict
    if isinstance(state_spec, dict):
        if STATE_TYPE in state_spec:
            if not inspect.isclass(state_spec[STATE_TYPE]) and issubclass(state_spec[STATE_TYPE], State):
                raise StateError("STATE entry of state specification for {} ({})"
                                 "is not a State or type of State".
                                 format(owner.name, state_spec[STATE]))
            return state_spec[STATE_TYPE]

    raise StateError("{} is not a legal State specification for {}".format(state_spec, owner.name))



STATE_SPEC_INDEX = 0

# FIX: CHANGE EXPECTATION OF *PROJECTIONS* ENTRY TO BE A SET OF TUPLES WITH THE WEIGHT AND EXPONENT FOR IT
#          THESE CAN BE USED BY THE InputState's LinearCombination Function
#          (AKIN TO HOW THE MECHANISM'S FUNCTION COMBINES InputState VALUES)
#          THIS WOULD ALLOW FULLY GENEREAL (HIEARCHICALLY NESTED) ALGEBRAIC COMBINATION OF INPUT VALUES TO A MECHANISM
@tc.typecheck
def _parse_state_spec(state_type=None,
                      owner=None,
                      reference_value=None,
                      name=None,
                      variable=None,
                      value=None,
                      params=None,
                      prefs=None,
                      context=None,
                      **state_spec):

    """Parse State specification and return either State object or State specification dictionary

    If state_spec is or resolves to a State object, returns State object.
    Otherwise, return State specification dictionary using any arguments provided as defaults
    Warn if variable is assigned the default value, and verbosePref is set on owner.
    *value* arg should generally be a constraint for the value of the State;  however,
        if state_spec is a Projection, and method is being called from:
            InputState, value should be the projection's value;
            ParameterState, value should be the projection's value;
            OutputState, value should be the projection's variable

    If a State specification dictionary is specified in the *state_specs* argument,
       its entries are moved to standard_args, replacing any that are there, and they are deleted from state_specs;
       any remaining entries are passed to _parse_state_specific_specs and placed in params.
    This gives precedence to values of standard args specified in a State specification dictionary
       (e.g., by the user) over any explicitly specified in the call to _instantiate_state.
    The standard arguments (from standard_args and/or a State specification dictonary in state_specs)
        are placed assigned to state_dict, as defaults for the State specification dictionary returned by this method.
    Any item in *state_specs* OTHER THAN a State specification dictionary is placed in state_spec_arg
       is parsed and/or validated by this method.
    Values in standard_args (i.e., specified in the call to _instantiate_state) are used to validate a state specified
       in state_spec_arg;
       - if the State is an existing one, the standard_arg values are assigned to it;
       - if state_spec_arg specifies a new State, the values in standard_args are used as defaults;  any specified
          in the state_spec_arg specification are used
    Any arguments to _instantiate_states that are not standard arguments (in standard_args) or a state_specs_arg
       generate a warning and are ignored.

    Return either State object or State specification dictionary
    """
    from psyneulink.components.projections.projection \
        import _is_projection_spec, _parse_projection_spec, _parse_connection_specs, ProjectionTuple
    from psyneulink.components.states.modulatorysignals.modulatorysignal import _is_modulatory_spec
    from psyneulink.components.mechanisms.adaptive.adaptivemechanism import AdaptiveMechanism_Base


    # Get all of the standard arguments passed from _instantiate_state (i.e., those other than state_spec) into a dict
    standard_args = get_args(inspect.currentframe())

    STATE_SPEC_ARG = 'state_spec'
    state_specification = None
    state_specific_args = {}

    # If there is a state_specs arg passed from _instantiate_state:
    if STATE_SPEC_ARG in state_spec:

        # If it is a State specification dictionary
        if isinstance(state_spec[STATE_SPEC_ARG], dict):
            # Use the value of any standard args specified in the State specification dictionary
            #    to replace those explicitly specified in the call to _instantiate_state (i.e., passed in standard_args)
            #    (use copy so that items in state_spec dict are not deleted when called from _validate_params)
            state_specific_args = state_spec[STATE_SPEC_ARG].copy()
            standard_args.update({key: state_specific_args[key]
                                  for key in state_specific_args
                                  if key in standard_args})
            # Delete them from the State specification dictionary, leaving only state-specific items there
            for key in standard_args:
                state_specific_args.pop(key, None)

        else:
            state_specification = state_spec[STATE_SPEC_ARG]

        # Delete the State specification dictionary from state_spec
        del state_spec[STATE_SPEC_ARG]

    if REFERENCE_VALUE_NAME in state_spec:
        del state_spec[REFERENCE_VALUE_NAME]

    if state_spec:
        if owner.verbosePref:
            print('Args other than standard args and state_spec were in _instantiate_state ({})'.
                  format(state_spec))
        state_specific_args.update(state_spec)

    state_dict = standard_args
    context = state_dict.pop(CONTEXT, None)
    owner = state_dict[OWNER]
    state_type = state_dict[STATE_TYPE]
    reference_value = state_dict[REFERENCE_VALUE]
    variable = state_dict[VARIABLE]
    params = state_specific_args

    #  Convert reference_value to np.array to match state_variable (which, as output of function, will be an np.array)
    if isinstance(reference_value, numbers.Number):
        reference_value = convert_to_np_array(reference_value,1)

    # Validate that state_type is a State class
    if not inspect.isclass(state_type) or not issubclass(state_type, State):
        raise StateError("\'state_type\' arg ({}) must be a sublcass of {}".format(state_type,
                                                                                   State.__name__))
    state_type_name = state_type.__name__

    # EXISTING STATES

    # Determine whether specified State is one to be instantiated or to be connected with,
    #    and validate that it is consistent with any standard_args specified in call to _instantiate_state

    # function; try to resolve to a value
    if isinstance(state_specification, function_type):
        state_specification = state_specification()

    # ModulatorySpecification of some kind
    if _is_modulatory_spec(state_specification):
        # If it is an AdaptiveMechanism specification, get its ModulatorySignal class
        # (so it is recognized by _is_projection_spec below (Mechanisms are not for secondary reasons)
        if isinstance(state_specification, type) and issubclass(state_specification, AdaptiveMechanism_Base):
            state_specification = state_specification.outputStateType
        projection = state_type

    # State or Mechanism object specification:
    if isinstance(state_specification, (Mechanism, State)):

        projection = None

        # Mechanism object:
        if isinstance(state_specification, Mechanism):
            mech = state_specification
            # Instantiating State of specified Mechanism, so get primary State of state_type
            if mech is owner:
                state_specification = state_type._get_primary_state(state_type, mech)
            # mech used to specify State to be connected with:
            else:
                state_specification = mech
                projection = state_type

        # Specified State is one with which connectee can connect, so assume it is a Projection specification
        if state_specification.__class__.__name__ in state_type.connectsWith + state_type.modulators:
            projection = state_type

        # Specified State is same as connectee's type (state_type),
        #    so assume it is a reference to the State itself that is being (or has been) instantiated
        elif isinstance(state_specification, state_type):
            # Make sure that the specified State belongs to the Mechanism passed in the owner arg
            if state_specification.init_status is InitStatus.DEFERRED_INITIALIZATION:
                state_owner = state_specification.init_args[OWNER]
            else:
                state_owner = state_specification.owner
            if owner is not None and state_owner is not None and state_owner is not owner:
                raise StateError("Attempt to assign a {} ({}) to {} that belongs to another {} ({})".
                                 format(State.__name__, state_specification.name, owner.name,
                                        Mechanism.__name__, state_owner.name))
            return state_specification

        # Re-process with Projection specified
        state_dict = _parse_state_spec(state_type=state_type,
                                       owner=owner,
                                       variable=variable,
                                       value=value,
                                       reference_value=reference_value,
                                       params=params,
                                       prefs=prefs,
                                       context=context,
                                       state_spec=ProjectionTuple(state=state_specification,
                                                                  weight=None,
                                                                  exponent=None,
                                                                  projection=projection))

    # # State class
    # elif (inspect.isclass(state_specification) and issubclass(state_specification, State)):
    #     # Specified type of State is same as connectee's type (state_type),
    #     #    so assume it is a reference to the State itself to be instantiated
    #     if state_specification is not state_type:
    #         raise StateError("Specification of {} for {} (\'{}\') is insufficient to instantiate the {}".
    #             format(state_type_name, owner.name, state_specification.__name__, State.__name__))

    # # MODIFIED 11/25/17 NEW:
    # # State class
    # if (inspect.isclass(state_specification) and issubclass(state_specification, State)):
    #     try:
    #         state_specification = (owner.paramClassDefaults[name], state_specification)
    #     except:
    #         pass
    #     else:
    #         state_dict = _parse_state_spec(state_spec=state_specification,
    #                                        **state_dict)
    # # MODIFIED 11/25/17 END:


    # Projection specification (class, object, or matrix value (matrix keyword processed below):
    elif _is_projection_spec(state_specification, include_matrix_spec=False):

        # FIX: 11/12/17 - HANDLE SITUATION IN WHICH projection_spec IS A MATRIX (AND SENDER IS SOMEHOW KNOWN)
        # Parse to determine whether Projection's value is specified
        projection_spec = _parse_projection_spec(state_specification, owner=owner, state_type=state_dict[STATE_TYPE])

        projection_value=None
        sender=None
        matrix=None

        # Projection has been instantiated
        if isinstance(projection_spec, Projection):
            if projection_spec.init_status is InitStatus.INITIALIZED:
                projection_value = projection_spec.value
            # If deferred_init, need to get sender and matrix to determine value
            else:
                try:
                    sender = projection_spec.init_args[SENDER]
                    matrix = projection_spec.init_args[PARAMS][FUNCTION_PARAMS][MATRIX]
                except KeyError:
                    pass
        # Projection specification dict:
        else:
            # Need to get sender and matrix to determine value
            sender = projection_spec[SENDER]
            matrix = projection_spec[MATRIX]

        if sender is not None and matrix is not None and matrix is not AUTO_ASSIGN_MATRIX:
            sender = _get_state_for_socket(owner=owner,state_spec=sender,state_types=state_dict[STATE_TYPE])
            projection_value = np.zeros(matrix.shape[sender.value.ndim :])

        reference_value = state_dict[REFERENCE_VALUE]
        # If State's reference_value is not specified, but Projection's value is, use projection_spec's value
        if reference_value is None and projection_value is not None:
            state_dict[REFERENCE_VALUE] = projection_value
        # If State's reference_value has been specified, check for compatibility with projection_spec's value
        elif (reference_value is not None and projection_value is not None
            and not iscompatible(reference_value, projection_value)):
            raise StateError("{} of {} ({}) is not compatible with {} of {} ({}) for {}".
                             format(VALUE, Projection.__name__, projection_value, REFERENCE_VALUE,
                                    state_dict[STATE_TYPE].__name__, reference_value, owner.name))

        # Move projection_spec to PROJECTIONS entry of params specification dict (for instantiation of Projection)
        if state_dict[PARAMS] is None:
            state_dict[PARAMS] = {}
        state_dict[PARAMS].update({PROJECTIONS:[state_specification]})

    # string (keyword or name specification)
    elif isinstance(state_specification, str):
        # Check if it is a keyword
        spec = get_param_value_for_keyword(owner, state_specification)
        # A value was returned, so use value of keyword as reference_value
        if spec is not None:
            state_dict[REFERENCE_VALUE] = spec
            # NOTE: (7/26/17 CW) This warning below may not be appropriate, since this routine is run if the
            # matrix parameter is specified as a keyword, which may be intentional.
            if owner.prefs.verbosePref:
                print("{} not specified for {} of {};  reference value ({}) will be used".
                      format(VARIABLE, state_type, owner.name, state_dict[REFERENCE_VALUE]))
        # It is not a keyword, so treat string as the name for the state
        else:
            state_dict[NAME] = state_specification

    # # function; try to resolve to a value
    # elif isinstance(state_specification, function_type):
    #     state_dict[REFERENCE_VALUE] = get_param_value_for_function(owner, state_specification)
    #     if state_dict[REFERENCE_VALUE] is None:
    #         raise StateError("PROGRAM ERROR: state_spec for {} of {} is a function ({}), but failed to return a value".
    #                          format(state_type_name, owner.name, state_specification))

    # FIX: THIS SHOULD REALLY BE PARSED IN A STATE-SPECIFIC WAY:
    #      FOR InputState: variable
    #      FOR ParameterState: default (base) parameter value
    #      FOR OutputState: index
    #      FOR ModulatorySignal: default value of ModulatorySignal (e.g, allocation or gating policy)
    # value, so use as variable of State
    elif is_value_spec(state_specification):
        state_dict[REFERENCE_VALUE] = np.atleast_1d(state_specification)

    elif isinstance(state_specification, Iterable) or state_specification is None:

        # Standard state specification dict
        # Warn if VARIABLE was not in dict
        if VARIABLE not in state_dict and owner.prefs.verbosePref:
            print("{} missing from specification dict for {} of {};  "
                  "will be inferred from context or the default ({}) will be used".
                  format(VARIABLE, state_type, owner.name, state_dict))

        if isinstance(state_specification, (list, set)):
            state_specific_specs = ProjectionTuple(state=state_specification,
                                              weight=None,
                                              exponent=None,
                                              projection=state_type)

        # State specification is a tuple
        elif isinstance(state_specification, tuple):

            # 1st item of tuple is a tuple (presumably a (State name, Mechanism) tuple),
            #    so parse to get specified State (any projection spec should be included as 4th item of outer tuple)
            if isinstance(state_specification[0],tuple):
                proj_spec = _parse_connection_specs(connectee_state_type=state_type,
                                                    owner=owner,
                                                    connections=state_specification[0])
                state_specification = (proj_spec[0].state,) + state_specification[1:]

            # Reassign tuple for handling by _parse_state_specific_specs
            state_specific_specs = state_specification

        # Otherwise, just pass params to State subclass
        else:
            state_specific_specs = params

        if state_specific_specs:
            state_spec, params = state_type._parse_state_specific_specs(state_type,
                                                                         owner=owner,
                                                                         state_dict=state_dict,
                                                                         state_specific_spec = state_specific_specs)
            # State subclass returned a state_spec, so call _parse_state_spec to parse it
            if state_spec:
                state_dict = _parse_state_spec(context=context, state_spec=state_spec, **standard_args)

            # Move PROJECTIONS entry to params
            if PROJECTIONS in state_dict:
                if not isinstance(state_dict[PROJECTIONS], list):
                    state_dict[PROJECTIONS] = [state_dict[PROJECTIONS]]
                params[PROJECTIONS].append(state_dict[PROJECTIONS])

            # MECHANISM entry specifies Mechanism; <STATES> entry has names of its States
            #           MECHANISM: <Mechanism>, <STATES>:[<State.name>, ...]}
            if MECHANISM in state_specific_args:

                if not PROJECTIONS in params:
                    params[PROJECTIONS] = []

                mech = state_specific_args[MECHANISM]
                if not isinstance(mech, Mechanism):
                    raise StateError("Value of the {} entry ({}) in the specification dictionary "
                                     "for {} of {} is not a {}".
                                     format(MECHANISM, mech, state_type.__name__, owner.name, Mechanism.__name__))

                # For States with which the one being specified can connect:
                for STATES in state_type.connectsWithAttribute:

                   if STATES in state_specific_args:
                        state_specs = state_specific_args[STATES]
                        state_specs = state_specs if isinstance(state_specs, list) else [state_specs]
                        for state_spec in state_specs:
                            # If State is a tuple, get its first item as state
                            state = state_spec[0] if isinstance(state_spec, tuple) else state_spec
                            try:
                                state_attr = getattr(mech, STATES)
                                state = state_attr[state]
                            except:
                                name = owner.name if not 'unnamed' in owner.name else 'a ' + owner.__class__.__name__
                                raise StateError("Unrecognized name ({}) for {} of {} in specification of {} for {}".
                                                 format(state, STATES, mech.name, state_type.__name__, name))
                            # If state_spec was a tuple, put state back in as its first item and use as projection spec
                            if isinstance(state_spec, tuple):
                                state = (state,) + state_spec[1:]
                            params[PROJECTIONS].append(state)
                        # Delete <STATES> entry as it is not a parameter of a State
                        del state_specific_args[STATES]

                # Delete MECHANISM entry as it is not a parameter of a State
                del state_specific_args[MECHANISM]

            # FIX: 11/4/17 - MAY STILL NEED WORK:
            # FIX:   PROJECTIONS FROM UNRECOGNIZED KEY ENTRY MAY BE REDUNDANT OR CONFLICT WITH ONE ALREADY IN PARAMS
            # FIX:   NEEDS TO BE BETTER COORDINATED WITH _parse_state_specific_specs
            # FIX:   REGARDING WHAT IS IN state_specific_args VS params (see REF_VAL_NAME BRANCH)
            # FIX:   ALSO, ??DOES PROJECTIONS ENTRY BELONG IN param OR state_dict?
            # Check for single unrecognized key in params, used for {<STATE_NAME>:[<projection_spec>,...]} format
            unrecognized_keys = [key for key in state_specific_args if not key in state_type.stateAttributes]
            if unrecognized_keys:
                if len(unrecognized_keys)==1:
                    key = unrecognized_keys[0]
                    state_dict[NAME] = key
                    params[PROJECTIONS] = state_specific_args[key]
                    del state_specific_args[key]
                else:
                    raise StateError("There is more than one entry of the {} specification dictionary for {} ({}) "
                                     "that is not a keyword; there should be only one (used to name the State, "
                                     "with a list of Projection specifications".
                                     format(state_type.__name__, owner.name,
                                            ", ".join([s for s in list(state_specific_args.keys())])))

            for param in state_type.stateAttributes:
                if param in state_specific_args:
                    params[param] = state_specific_args[param]

            if PROJECTIONS in params and params[PROJECTIONS] is not None:
                #       (E.G., WEIGHTS AND EXPONENTS FOR InputState AND INDEX FOR OutputState)
                # Get and parse projection specifications for the State
                params[PROJECTIONS] = _parse_connection_specs(state_type, owner, params[PROJECTIONS])

            # Update state_dict[PARAMS] with params
            if state_dict[PARAMS] is None:
                state_dict[PARAMS] = {}
            state_dict[PARAMS].update(params)

    else:
        # if owner.verbosePref:
        #     warnings.warn("PROGRAM ERROR: state_spec for {} of {} is an unrecognized specification ({})".
        #                  format(state_type_name, owner.name, state_spec))
        # return
        raise StateError("PROGRAM ERROR: state_spec for {} of {} is an unrecognized specification ({})".
                         format(state_type_name, owner.name, state_specification))

    # If variable is none, use value:
    if state_dict[VARIABLE] is None:
        if state_dict[VALUE] is not None:
            state_dict[VARIABLE] = state_dict[VALUE]
        else:
            state_dict[VARIABLE] = state_dict[REFERENCE_VALUE]
    elif state_dict[REFERENCE_VALUE] is not None:
        if not iscompatible(state_dict[VARIABLE], state_dict[REFERENCE_VALUE]):
            if context is not None and context in '_parse_arg_input_states':
                from psyneulink.components.mechanisms.mechanism import MechanismError
                raise MechanismError('default variable determined from the specified input_states spec ({0}) '
                                     'is not compatible with the specified default variable ({1})'.
                                     format(state_dict[VARIABLE], state_dict[REFERENCE_VALUE]))
            else:
                name = state_dict[NAME] or state_type.__name__
                raise StateError("Specification of {} for {} of {} ({}) is not compatible with its {} ({})".
                                 format(VARIABLE, name, owner.name,
                                        state_dict[VARIABLE], REFERENCE_VALUE, state_dict[REFERENCE_VALUE]))

    return state_dict


# FIX: REPLACE mech_state_attribute WITH DETERMINATION FROM state_type
# FIX:          ONCE STATE CONNECTION CHARACTERISTICS HAVE BEEN IMPLEMENTED IN REGISTRY
@tc.typecheck
def _get_state_for_socket(owner,
                          state_spec=None,
                          state_types:tc.optional(tc.any(list, _is_state_class))=None,
                          mech:tc.optional(Mechanism)=None,
                          mech_state_attribute:tc.optional(tc.any(str, list))=None,
                          projection_socket:tc.optional(tc.any(str, set))=None):
    """Take some combination of Mechanism, state name (string), Projection, and projection_socket, and return
    specified State(s)

    If state_spec is:
        State name (str), then *mech* and *mech_state_attribute* args must be specified
        Mechanism, then *state_type* must be specified; primary State is returned
        Projection, *projection_socket* arg must be specified;
                    Projection must be instantiated or in deferred_init, with projection_socket attribute assigned

    IMPLEMENTATION NOTES:
    Currently does not support State specification dict (referenced State must be instantiated)
    Currently does not support Projection specification using class or Projection specification dict
        (Projection must be instantiated, or in deferred_init status with projection_socket assigned)

    Returns a State if it can be resolved, or list of allowed State types if not.
    """
    from psyneulink.components.projections.projection import \
        _is_projection_spec, _validate_connection_request, _parse_projection_spec
    from psyneulink.globals.utilities import is_matrix
    from collections import Iterable

    # # If the mech_state_attribute specified has more than one item, get the primary one
    # if isinstance(mech_state_attribute, list):
    #     mech_state_attribute = mech_state_attribute[0]

    # state_types should be a list, and state_type its first (or only) item
    if isinstance(state_types, list):
        state_type = state_types[0]
    else:
        state_type = state_types
        state_types = [state_types]

    state_type_names = ", ".join([s.__name__ for s in state_types])

    # Return State itself if it is an instantiated State
    if isinstance(state_spec, State):
        return state_spec

    # Return state_type (Class) if state_spec is:
    #    - an allowable State type for the projection_socket
    #    - a projection keyword (e.g., 'LEARNING' or 'CONTROL', and it is consistent with projection_socket
    # Otherwise, return list of allowable State types for projection_socket (if state_spec is a Projection type)
    if _is_projection_spec(state_spec):

        # # MODIFIED 11/25/17 OLD:
        # # These specifications require that a particular State be specified to assign its default Projection type
        # if ((is_matrix(state_spec) or (isinstance(state_spec, dict) and not PROJECTION_TYPE in state_spec))
        #     and state_type is 'MULTIPLE'):
        #     raise StateError("PROGRAM ERROR: Projection specified ({}) for object "
        #                      "that has multiple possible States {}) for the specified socket ({}).".
        #                      format(state_spec, state_types, projection_socket))
        # proj_spec = _parse_projection_spec(state_spec, owner=owner, state_type=state_type)
        # if isinstance(proj_spec, Projection):
        #     proj_type = proj_spec.__class__
        # else:
        #     proj_type = proj_spec[PROJECTION_TYPE]
        # MODIFIED 11/25/17 NEW:
        # These specifications require that a particular State be specified to assign its default Projection type
        if ((is_matrix(state_spec) or (isinstance(state_spec, dict) and not PROJECTION_TYPE in state_spec))):
            for st in state_types:
                try:
                    proj_spec = _parse_projection_spec(state_spec, owner=owner, state_type=st)
                    if isinstance(proj_spec, Projection):
                        proj_type = proj_spec.__class__
                    else:
                        proj_type = proj_spec[PROJECTION_TYPE]
                except:
                    continue
        else:
            proj_spec = _parse_projection_spec(state_spec, owner=owner, state_type=state_type)
            if isinstance(proj_spec, Projection):
                proj_type = proj_spec.__class__
            else:
                proj_type = proj_spec[PROJECTION_TYPE]
        # MODIFIED 11/25/17 END:

        # Get State type if it is appropriate for the specified socket of the Projection's type
        s = next((s for s in state_types if s.__name__ in getattr(proj_type.sockets, projection_socket)), None)
        if s:
            try:
                # Return State associated with projection_socket if proj_spec is an actual Projection
                state = getattr(proj_spec, projection_socket)
                return state
            except AttributeError:
                # Otherwise, return first state_type (s)
                return s

        # FIX: 10/3/17 - ??IS THE FOLLOWING NECESSARY?  ??HOW IS IT DIFFERENT FROM ABOVE?
        # Otherwise, get State types that are allowable for that projection_socket
        elif inspect.isclass(proj_type) and issubclass(proj_type, Projection):
            projection_socket_state_names = getattr(proj_type.sockets, projection_socket)
            projection_socket_state_types = [StateRegistry[name].subclass for name in projection_socket_state_names]
            return projection_socket_state_types
        else:
            assert False
            # return state_type

    # Get state by name
    if isinstance(state_spec, str):

        if mech is None:
            raise StateError("PROGRAM ERROR: A {} must be specified to specify its {} ({}) by name".
                             format(Mechanism.__name__, State.__name__, state_spec))
        if mech_state_attribute is None:
            raise StateError("PROGRAM ERROR: The attribute of {} that holds the requested State ({}) must be specified".
                             format(mech.name, state_spec))
        for attr in mech_state_attribute:
            try:
                stateListAttribute = getattr(mech, attr)
                state = stateListAttribute[state_spec]
            except AttributeError:
                stateListAttribute = None
            except (KeyError, TypeError):
                state = None
            else:
                break
        if stateListAttribute is None:
            raise StateError("PROGRAM ERROR: {} attribute(s) not found on {}'s type ({})".
                             format(mech_state_attribute, mech.name, mech.__class__.__name__))
        if state is None:
            if len(mech_state_attribute)==1:
                attr_name = mech_state_attribute[0] + " attribute."
            else:
                attr_name = " or ".join(", ".format(attr) for (attr) in mech_state_attribute) + " attributes."
            raise StateError("{} does not have a {} named \'{}\' in its {}".
                             format(mech.name, State.__name__, state_spec, attr_name))

    # Get primary State of specified type
    elif isinstance(state_spec, Mechanism):

        if state_type is None:
            raise StateError("PROGRAM ERROR: The type of State requested for {} must be specified "
                             "to get its primary State".format(state_spec.name))
        try:
            state = state_type._get_primary_state(state_type, state_spec)
        except StateError:
            if mech_state_attribute:
                try:
                    state = getattr(state_spec, mech_state_attribute)[0]
                except:
                    raise StateError("{} does not seem to have an {} attribute"
                                     .format(state_spec.name, mech_state_attribute))
            for attr in mech_state_attribute:
                try:
                    state = getattr(state_spec, attr)[0]
                except :
                    state = None
                else:
                    break
                if state is None:
                    raise StateError("PROGRAM ERROR: {} attribute(s) not found on {}'s type ({})".
                                     format(mech_state_attribute, mech.name, mech.__class__.__name__))

    # # Get
    # elif isinstance(state_spec, type) and issubclass(state_spec, Mechanism):


    # Get state from Projection specification (exclude matrix spec in test as it can't be used to determine the state)
    elif _is_projection_spec(state_spec, include_matrix_spec=False):
        _validate_connection_request(owner=owner,
                                     connect_with_states=state_type,
                                     projection_spec=state_spec,
                                     projection_socket=projection_socket)
        if isinstance(state_spec, Projection):
            state = state_spec.socket_assignments[projection_socket]
            if state is None:
                state = state_type
        else:
            return state_spec

    else:
        if state_spec is None:
            raise StateError("PROGRAM ERROR: Missing state specification for {}".format(owner.name))
        else:
            raise StateError("Unrecognized state specification: {} for {}".format(state_spec, owner.name))

    return state


def _is_legal_state_spec_tuple(owner, state_spec, state_type_name=None):

    from psyneulink.components.projections.projection import _is_projection_spec

    state_type_name = state_type_name or STATE

    if len(state_spec) != 2:
        raise StateError("Tuple provided as state_spec for {} of {} ({}) must have exactly two items".
                         format(state_type_name, owner.name, state_spec))
    if not (_is_projection_spec(state_spec[1]) or
                # IMPLEMENTATION NOTE: Mechanism or State allowed as 2nd item of tuple or
                #                      string (parameter name) as 1st and Mechanism as 2nd
                #                      to accommodate specification of param for ControlSignal
                isinstance(state_spec[1], (Mechanism, State))
                           or (isinstance(state_spec[0], Mechanism) and
                                       state_spec[1] in state_spec[0]._parameter_states)):

        raise StateError("2nd item of tuple in state_spec for {} of {} ({}) must be a specification "
                         "for a Mechanism, State, or Projection".
                         format(state_type_name, owner.__class__.__name__, state_spec[1]))
