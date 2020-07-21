# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *******************************************  InputPort *****************************************************
#
"""
Contents
--------

* `InputPort_Overview`
* `InputPort_Creation`
    - `InputPort_Deferred_Initialization`
    - `InputPort_Primary`
    - `InputPort_Specification`
        • `Forms of Specification <InputPort_Forms_of_Specification>`
        • `Variable, Value and Mechanism <InputPort_Variable_and_Value>`
        • `Variable: Compatability and Constaints <InputPort_Compatability_and_Constraints>`
* `InputPort_Structure`
    - `Afferent Projections <InputPort_Afferent_Projections>`
    - `Variable <InputPort_Variable>`
    - `Function <InputPort_Function>`
    - `Value <InputPort_Value>`
    - `Weights ane Exponents <InputPort_Weights_And_Exponents>`
* `InputPort_Execution`
* `InputPort_Class_Reference`

.. _InputPort_Overview:

Overview
--------

The purpose of an InputPort is to receive and combine inputs to a `Mechanism <Mechanism>`, allow them to be modified,
and provide them to the Mechanism's `function <Mechanism_Base.function>`. An InputPort receives input to a Mechanism
provided by the `Projections <Projection>` to that Mechanism from others in a `Composition`.  If the InputPort belongs
to an `ORIGIN` Mechanism (see `Mechanism_Role_In_Compositions`), then it receives the input specified when that
Composition is `run <Run>`.  The `PathwayProjections <PathWayProjection>` received by an InputPort are listed in its
`path_afferents <Port.path_afferents>`, and its `ModulatoryProjections <ModulatoryProjection>` in its `mod_afferents
<Port.mod_afferents>` attribute.  Its `function <InputPort.function>` combines the values received from its
PathWayProjections, modifies the combined value according to value(s) any ModulatoryProjections it receives, and
provides the result to the assigned item of its owner Mechanism's `variable <Mechanism_Base.variable>` and
`input_values <Mechanism_Base.input_values>` attributes (see `below` and `Mechanism InputPorts <Mechanism_InputPorts>`
for additional details about the role of InputPorts in Mechanisms, and their assignment to the items of a Mechanism's
`variable <Mechanism_Base.variable>` attribute).

.. _InputPort_Creation:

Creating an InputPort
----------------------

An InputPort can be created by calling its constructor, but in general this is not necessary as a `Mechanism
<Mechanism>` can usually automatically create the InputPort(s) it needs when it is created.  For example, if the
Mechanism isbeing created within the `pathway <Process.pathway>` of a `Process`, its InputPort is created and assigned
as the `receiver <MappingProjection.receiver>` of a `MappingProjection` from the  preceding Mechanism in the `pathway
<Process.pathway>`.  InputPorts can also be specified in the **input_ports** argument of a Mechanism's constructor
(see `below <InputPort_Specification>`).

The `variable <InputPort.variable>` of an InputPort can be specified using the **variable** or **size** arguments of
its constructor.  It can also be specified using the **projections** argument, if neither **variable** nor **size** is
specified.  The **projections** argument is used to `specify Projections <Port_Projections>` to the InputPort. If
neither the **variable** nor **size** arguments is specified, then the value of the `Projections(s) <Projection>` or
their `sender <Projection_Base.sender>`\\s (all of which must be the same length) is used to determine the `variable
<InputPort.variable>` of the InputPort.

If an InputPort is created using its constructor, and a Mechanism is specified in the **owner** argument,
it is automatically assigned to that Mechanism.  Note that its `value <InputPort.value>` (generally determined
by the size of its `variable <InputPort.variable>` -- see `below <InputPort_Variable_and_Value>`) must
be compatible (in number and type of elements) with the item of its owner's `variable <Mechanism_Base.variable>` to
which it is assigned (see `below <InputPort_Variable_and_Value>` and `Mechanism <Mechanism_Variable_and_InputPorts>`).
If the **owner** argument is not specified, `initialization <Port_Deferred_Initialization>` is deferred.

.. _InputPort_Deferred_Initialization:

*Owner Assignment and Deferred Initialization*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An InputPort must be owned by a `Mechanism <Mechanism>`.  When InputPort is specified in the constructor for a
Mechanism (see `below <InputPort_Specification>`), it is automatically assigned to that Mechanism as its owner. If
the InputPort is created on its own, its `owner <Port.owner>` can specified in the **owner**  argument of its
constructor, in which case it is assigned to that Mechanism. If its **owner** argument is not specified, its
initialization is `deferred <Port_Deferred_Initialization>` until
COMMENT:
TBI: its `owner <Port_Base.owner>` attribute is assigned or
COMMENT
the InputPort is assigned to a Mechanism using the Mechanism's `add_ports <Mechanism_Base.add_ports>` method.

.. _InputPort_Primary:

*Primary InputPort*
~~~~~~~~~~~~~~~~~~~

Every Mechanism has at least one InputPort, referred to as its *primary InputPort*.  If InputPorts are not
`explicitly specified <InputPort_Specification>` for a Mechanism, a primary InputPort is automatically created
and assigned to its `input_port <Mechanism_Base.input_port>` attribute (note the singular), and also to the first
entry of the Mechanism's `input_ports <Mechanism_Base.input_ports>` attribute (note the plural).  The `value
<InputPort.value>` of the primary InputPort is assigned as the first (and often only) item of the Mechanism's
`variable <Mechanism_Base.variable>` and `input_values <Mechanism_Base.input_values>` attributes.

.. _InputPort_Specification:

*InputPort Specification*
~~~~~~~~~~~~~~~~~~~~~~~~~~

Specifying InputPorts when a Mechanism is created
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

InputPorts can be specified for a `Mechanism <Mechanism>` when it is created, in the **input_ports** argument of the
Mechanism's constructor (see `examples <Port_Constructor_Argument_Examples>` in Port), or in an *INPUT_PORTS* entry
of a parameter dictionary assigned to the constructor's **params** argument.  The latter takes precedence over the
former (that is, if an *INPUT_PORTS* entry is included in the parameter dictionary, any specified in the
**input_ports** argument are ignored).

    .. _InputPort_Replace_Default_Note:

    .. note::
       Assigning InputPorts to a Mechanism in its constructor **replaces** any that are automatically generated for
       that Mechanism (i.e., those that it creates for itself by default).  If any of those are needed, they must be
       explicitly specified in the list assigned to the **input_ports** argument, or the *INPUT_PORTS* entry of the
       parameter dictionary in the **params** argument.  The number of InputPorts specified must also be equal to
       the number of items in the Mechanism's `variable <Mechanism_Base.variable>` attribute.

.. _InputPort_Variable_and_Value:

*InputPort's* `variable <InputPort.variable>`, `value <InputPort.value>` *and Mechanism's* `variable <Mechanism_Base.variable>`
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Each InputPort specified in the **input_ports** argument of a Mechanism's constructor must correspond to an item of
the Mechanism's `variable <Mechanism_Base.variable>` attribute (see `Mechanism <Mechanism_Variable_and_InputPorts>`),
and the `value <InputPort.value>` of the InputPort must be compatible with that item (that is, have the same number
and type of elements).  By default, this is also true of the InputPort's `variable <InputPort.variable>` attribute,
since the default `function <InputPort.function>` for an InputPort is a `LinearCombination`, the purpose of which
is to combine the inputs it receives and possibly modify the combined value (under the influence of any
`ModulatoryProjections <ModulatoryProjection>` it receives), but **not mutate its form**. Therefore, under most
circumstances, both the `variable <InputPort.variable>` of an InputPort and its `value <InputPort.value>` should
match the item of its owner's `variable <Mechanism_Base.variable>` to which the InputPort is assigned.

The format of an InputPort's `variable <InputPort.variable>` can be specified in a variety of ways.  The most
straightforward is in the **variable** argument of its constructor.  More commonly, however, it is determined by
the context in which it is being created, such as the specification for its owner Mechanism's `variable
<Mechanism_Base.variable>` or for the InputPort in the Mechanism's **input_ports** argument (see `below
<InputPort_Forms_of_Specification>` and `Mechanism InputPort specification <Mechanism_InputPort_Specification>`
for details).


Adding InputPorts to a Mechanism after it is created
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

InputPorts can also be **added** to a Mechanism, either by creating the InputPort on its own, and specifying the
Mechanism in the InputPort's **owner** argument, or by using the Mechanism's `add_ports <Mechanism_Base.add_ports>`
method (see `examples <Port_Create_Port_Examples>` in Port).

    .. _InputPort_Add_Port_Note:

    .. note::
       Adding InputPorts *does not replace* any that the Mechanism generates by default;  rather they are added to the
       Mechanism, and appended to the list of InputPorts in its `input_ports <Mechanism_Base.input_ports>` attribute.
       Importantly, the Mechanism's `variable <Mechanism_Base.variable>` attribute is extended with items that
       correspond to the `value <InputPort.value>` attribute of each added InputPort.  This may affect the
       relationship of the Mechanism's `variable <Mechanism_Base.variable>` to its `function
       <Mechanism_Base.function>`, as well as the number of its `OutputPorts <OutputPort>` (see `note
       <Mechanism_Add_InputPorts_Note>`).

If the name of an InputPort added to a Mechanism is the same as one that already exists, its name is suffixed with a
numerical index (incremented for each InputPort with that name; see `Registry_Naming`), and the InputPort is added to
the list (that is, it will *not* replace ones that already exist).

.. _InputPort_Forms_of_Specification:

Forms of Specification
^^^^^^^^^^^^^^^^^^^^^^

InputPorts can be specified in a variety of ways, that fall into three broad categories:  specifying an InputPort
directly; use of a `Port specification dictionary <Port_Specification>`; or by specifying one or more Components that
should project to the InputPort. Each of these is described below:

    .. _InputPort_Direct_Specification:

    **Direct Specification of an InputPort**

    * existing **InputPort object** or the name of one -- If this is used to specify an InputPort in the
      constructor for a Mechanism, its `value <InputPort.value>` must be compatible with the corresponding item of
      the owner Mechanism's `variable <Mechanism_Base.variable>` (see `Mechanism InputPort specification
      <Mechanism_InputPort_Specification>` and `InputPort_Compatability_and_Constraints` below).  If the InputPort
      belongs to another Mechanism, then an InputPort is created along with Projections(s) that `shadow the inputs
      <InputPort_Shadow_Inputs>` to the specified InputPort.
    ..
    * **InputPort class**, **keyword** *INPUT_PORT*, or a **string** -- this creates a default InputPort; if used
      to specify an InputPort in the constructor for a Mechanism, the item of the owner Mechanism's `variable
      <Mechanism_Base.variable>` to which the InputPort is assigned is used as the format for the InputPort`s
      `variable <InputPort.variable>`; otherwise, the default for the InputPort is used.  If a string is specified,
      it is used as the `name <InputPort.name>` of the InputPort (see `example <Port_Constructor_Argument_Examples>`).

    .. _InputPort_Specification_by_Value:

    * **value** -- this creates a default InputPort using the specified value as the InputPort's `variable
      <InputPort.variable>`; if used to specify an InputPort in the constructor for a Mechanism, the format must be
      compatible with the corresponding item of the owner Mechanism's `variable <Mechanism_Base.variable>` (see
      `Mechanism InputPort specification <Mechanism_InputPort_Specification>`, `example
      <port_value_Spec_Example>`, and discussion `below <InputPort_Compatability_and_Constraints>`).

    .. _InputPort_Specification_Dictionary:

    **InputPort Specification Dictionary**

    * **InputPort specification dictionary** -- this can be used to specify the attributes of an InputPort, using
      any of the entries that can be included in a `Port specification dictionary <Port_Specification>` (see
      `examples <Port_Specification_Dictionary_Examples>` in Port).  If the dictionary is used to specify an
      InputPort in the constructor for a Mechanism, and it includes a *VARIABLE* and/or *VALUE* or entry, the value
      must be compatible with the item of the owner Mechanism's `variable <Mechanism_Base.variable>` to which the
      InputPort is assigned (see `Mechanism InputPort specification <Mechanism_InputPort_Specification>`).

      The *PROJECTIONS* entry can include specifications for one or more Ports, Mechanisms and/or Projections that
      should project to the InputPort (including both `MappingProjections <MappingProjection>` and/or
      `ModulatoryProjections <ModulatoryProjection>`; however, this may be constrained by or have consequences for the
      InputPort's `variable <InputPort.variable>` (see `InputPort_Compatability_and_Constraints`).

      In addition to the standard entries of a `Port specification dictionary <Port_Specification>`, the dictionary
      can also include either or both of the following entries specific to InputPorts:

      * *WEIGHT*:<number>
          the value must be an integer or float, and is assigned as the value of the InputPort's `weight
          <InputPort.weight>` attribute (see `weight and exponent <InputPort_Weights_And_Exponents>`);
          this takes precedence over any specification in the **weight** argument of the InputPort's constructor.

      * *EXPONENT*:<number>
          the value must be an integer or float, and is assigned as the value of the InputPort's `exponent
          <InputPort.exponent>` attribute (see `weight and exponent <InputPort_Weights_And_Exponents>`);
          this takes precedence over any specification in the **exponent** argument of the InputPort's constructor.

    .. _InputPort_Projection_Source_Specification:

    **Specification of an InputPort by Components that Project to It**

    COMMENT:
    `examples
      <Port_Projections_Examples>` in Port)
    COMMENT

    COMMENT:
    ?? PUT IN ITS OWN SECTION ABOVE OR BELOW??
    Projections to an InputPort can be specified either as attributes, in the constructor for an
    InputPort (in its **projections** argument or in the *PROJECTIONS* entry of an `InputPort specification dictionary
    <InputPort_Specification_Dictionary>`), or used to specify the InputPort itself (using one of the
    `InputPort_Forms_of_Specification` described above. See `Port Projections <Port_Projections>` for additional
    details concerning the specification of
    Projections when creating a Port.
    COMMENT

    An InputPort can also be specified by specifying one or more Ports, Mechanisms or Projections that should project
    to it, as described below.  Specifying an InputPort in this way creates both the InputPort and any of the
    specified or implied Projection(s) to it (if they don't already exist). `MappingProjections <MappingProjection>`
    are assigned to the InputPort's `path_afferents <Port.path_afferents>` attribute, while `ControlProjections
    <ControlProjection>` and `GatingProjections <GatingProjection>` to its `mod_afferents <Port.mod_afferents>`
    attribute. Any of the following can be used to specify an InputPort by the Components that projection to it (see
    `below <InputPort_Compatability_and_Constraints>` for an explanation of the relationship between the `value` of
    these Components and the InputPort's `variable <InputPort.variable>`):

    * **OutputPort, GatingSignal, Mechanism, or list with any of these** -- creates an InputPort with Projection(s)
      to it from the specified Port(s) or Mechanism(s).  For each Mechanism specified, its `primary OutputPort
      <OutputPort_Primary>` (or GatingSignal) is used.
    ..
    * **Projection** -- any form of `Projection specification <Projection_Specification>` can be
      used;  creates an InputPort and assigns it as the Projection's `receiver <Projection_Base.receiver>`.

    .. _InputPort_Tuple_Specification:

    * **InputPort specification tuples** -- these are convenience formats that can be used to compactly specify an
      InputPort and Projections to it any of the following ways:

        .. _InputPort_Port_Mechanism_Tuple:

        * **2-item tuple:** *(<Port name or list of Port names>, <Mechanism>)* -- 1st item must be the name of an
          `OutputPort` or `ModulatorySignal`, or a list of such names, and the 2nd item must be the Mechanism to
          which they all belong.  Projections of the relevant types are created for each of the specified Ports
          (see `Port 2-item tuple <Port_2_Item_Tuple>` for additional details).

        * **2-item tuple:** *(<value, Port specification, or list of Port specs>, <Projection specification>)* --
          this is a contracted form of the 4-item tuple described below;

        * **3 or 4-item tuple:** *(<value, Port spec, or list of Port specs>, weight, exponent, Projection
          specification)* -- this allows the specification of Port(s) that should project to the InputPort, together
          with a specification of the InputPort's `weight <InputPort.weight>` and/or `exponent <InputPort.exponent>`
          attributes of the InputPort, and (optionally) the Projection(s) to it.  This can be used to compactly
          specify a set of Ports that project the InputPort, while using the 4th item to determine its variable
          (e.g., using the matrix of the Projection specification) and/or attributes of the Projection(s) to it. Each
          tuple must have at least the following first three items (in the order listed), and can include the fourth:

            * **value, Port specification, or list of Port specifications** -- specifies either the `variable
              <InputPort.variable>` of the InputPort, or one or more Ports that should project to it.  The Port
              specification(s) can be a (Port name, Mechanism) tuple (see above), and/or include Mechanisms (in which
              case their `primary OutputPort <OutputPortPrimary>` is used.  All of the Port specifications must be
              consistent with (that is, their `value <Port_Base.value>` must be compatible with the `variable
              <Projection_Base.variable>` of) the Projection specified in the fourth item if that is included;

            * **weight** -- must be an integer or a float; multiplies the `value <InputPort.value>` of the InputPort
              before it is combined with others by the Mechanism's `function <Mechanism.function>` (see
              ObjectiveMechanism for `examples <ObjectiveMechanism_Weights_and_Exponents_Example>`);

            * **exponent** -- must be an integer or float; exponentiates the `value <InputPort.value>` of the
              InputPort before it is combined with others by the ObjectiveMechanism's `function
              <ObjectiveMechanism.function>` (see ObjectiveMechanism for `examples
              <ObjectiveMechanism_Weights_and_Exponents_Example>`);

            * **Projection specification** (optional) -- `specifies a Projection <Projection_Specification>` that
              must be compatible with the Port specification(s) in the 1st item; if there is more than one Port
              specified, and the Projection specification is used, all of the Ports
              must be of the same type (i.e.,either OutputPorts or GatingSignals), and the `Projection
              Specification <Projection_Specification>` cannot be an instantiated Projection (since a
              Projection cannot be assigned more than one `sender <Projection_Base.sender>`).

    .. _InputPort_Shadow_Inputs:

    * **InputPorts of Mechanisms to shadow** -- either of the following can be used to create InputPorts that
      receive the same inputs as ("shadow") the ones specified:

      * *InputPort or [InputPort, ...]* -- each InputPort must belong to an existing Mechanism; creates a new
        InputPort for each one specified, along with Projections to it that parallel those of the one specified
        (see below).

      * *{SHADOW_INPUTS: <InputPort or Mechanism or [<InputPort or Mechanism>,...]>}* -- any InputPorts specified
        must belong to an existing Mechanism;  creates a new InputPort for each one specified, and for each of the
        InputPorts belonging to any Mechanisms specified, along with Projections to them that parallel those of the
        one(s) specified (see below).

      For each InputPort specified, and all of the InputPorts belonging to any Mechanisms specified, a new InputPort
      is created along with Projections to it that parallel those received by the corresponding InputPort in the
      list.  In other words, for each InputPort specified, a new one is created that receives exactly the same inputs
      from the same `senders  <Projection_Base.sender>` as the ones specified.

.. _InputPort_Compatability_and_Constraints:

InputPort `variable <InputPort.variable>`: Compatibility and Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `variable <InputPort.variable>` of an InputPort must be compatible with the item of its owner Mechanism's
`variable <Mechanism_Base.variable>` to which it is assigned (see `Mechanism_Variable_and_InputPorts`). This may
have consequences that must be taken into account when `specifying an InputPort by Components that project to it
<InputPort_Projection_Source_Specification>`.  These depend on the context in which the specification is made, and
possibly the value of other specifications.  These considerations and how they are handled are described below,
starting with constraints that are given the highest precedence:

  *  **InputPort is** `specified in a Mechanism's constructor <Mechanism_InputPort_Specification>` and the
    **default_variable** argument for the Mechanism is also specified -- the item of the variable to which the
    `InputPort is assigned <Mechanism_Variable_and_InputPorts>` is used to determine the InputPort's `variable must
    <InputPort.variable>`.  Any other specifications of the InputPort relevant to its `variable <InputPort.variable>`
    must be compatible with this (for example, `specifying it by value <InputPort_Specification_by_Value>` or by a
    `MappingProjection` or `OutputPort` that projects to it (see `above <InputPort_Projection_Source_Specification>`).

    COMMENT:
    ***XXX EXAMPLE HERE
    COMMENT
  ..
  * **InputPort is specified on its own**, or the **default_variable** argument of its Mechanism's constructor
    is not specified -- any direct specification of the InputPort's `variable <InputPort.variable>` is used to
    determine its format (e.g., `specifying it by value <InputPort_Specification_by_Value>`, or a *VARIABLE* entry
    in an `InputPort specification dictionary <InputPort_Specification_Dictionary>`.  In this case, the value of any
    `Components used to specify the InputPort <InputPort_Projection_Source_Specification>` that are relevant to its
    `variable <InputPort.variable>` must be compatible with it (see below).

    COMMENT:
    ***XXX EXAMPLE HERE
    COMMENT
  ..
  * If the InputPort's `variable <InputPort.variable>` is not constrained by any of the conditions above,
    then its format is determined by the `specification of Components that project to it
    <InputPort_Projection_Source_Specification>`:

    * **More than one Component is specified with the same `value` format** -- that format is used to determine
      the format of the InputPort's `variable <InputPort.variable>`.

    * **More than one Component is specified with different `value` formats** -- the InputPort's `variable
      <InputPort.variable>` is determined by item of the default `variable <Mechanism_Base.variable>` for
      the class of its owner Mechanism.

    * **A single Component is specified** -- its `value` is used to determine the format of the InputPort's
      `variable <InputPort.variable>`;  if the Component is a(n):

      * **MappingProjection** -- can be specified by its class, an existing MappingProjection, or a matrix:

        * `MappingProjection` **class** -- a default value is used both the for the InputPort's `variable
          <InputPort.variable>` and the Projection's `value <Projection_Base.value>` (since the Projection's
          `sender <Projection_Base.sender>` is unspecified, its `initialization is deferred
          <Projection_Deferred_Initialization>`.

        * **Existing MappingProjection** -- then its `value <Projection_Base.value>` determines the
          InputPort's `variable <InputPort.variable>`.

        * `Matrix specification <MappingProjection_Matrix_Specification>` -- its receiver dimensionality determines the
          format of the InputPort's `variable <InputPort.variable>`. For a standard 2d "weight" matrix (i.e., one that
          maps a 1d array from its `sender <Projection_Base.sender>` to a 1d array of its `receiver
          <Projection_Base.receiver>`), the receiver dimensionality is its outer dimension (axis 1, or its number of
          columns).  However, if the `sender <Projection_Base.sender>` has more than one dimension, then the
          dimensionality of the receiver (used for the InputPort's `variable <InputPort.variable>`) is the
          dimensionality of the matrix minus the dimensionality of the sender's `value <OutputPort.value>`
          (see `matrix dimensionality <Mapping_Matrix_Dimensionality>`).

      * **OutputPort or ProcessingMechanism** -- the `value <OutputPort.value>` of the OutputPort (if it is a
        Mechanism, then its `primary OutputPort <OutputPort_Primary>`) determines the format of the InputPort's
        `variable <InputPort.variable>`, and a MappingProjection is created from the OutputPort to the InputPort
        using an `IDENTITY_MATRIX`.  If the InputPort's `variable <InputPort.variable>` is constrained (as in some
        of the cases above), then a `FULL_CONNECTIVITY_MATRIX` is used which maps the shape of the OutputPort's `value
        <OutputPort.value>` to that of the InputPort's `variable <InputPort.variable>`.

      * **GatingProjection, GatingSignal or GatingMechanism** -- any of these can be used to specify an InputPort;
        their `value` does not need to be compatible with the InputPort's `variable <InputPort.variable>`, however
        it does have to be compatible with the `modulatory parameter <Function_Modulatory_Params>` of the InputPort's
        `function <InputPort.function>`.

.. _InputPort_Structure:

Structure
---------

Every InputPort is owned by a `Mechanism <Mechanism>`. It can receive one or more `MappingProjections
<MappingProjection>` from other Mechanisms, as well as from the Process or System to which its owner belongs (if it
is the `ORIGIN` Mechanism for that Process or System).  It has the following attributes, that includes ones specific
to, and that can be used to customize the InputPort:

* `projections <Port.projections>` -- all of the `Projections <Projection>` received by the InputPort.

.. _InputPort_Afferent_Projections:

* `path_afferents <Port.path_afferents>` -- `MappingProjections <MappingProjection>` that project to the InputPort,
  the `value <Projection_Base.value>`\\s of which are combined by the InputPort's `function <InputPort.function>`,
  possibly modified by its `mod_afferents <InputPort_mod_afferents>`, and assigned to the corresponding item of the
  owner Mechanism's `variable <Mechanism_Base.variable>`.

* `mod_afferents <InputPort_mod_afferents>` -- `GatingProjections <GatingProjection>` that project to the InputPort,
  the `value <GatingProjection.value>` of which can modify the InputPort's `value <InputPort.value>` (see the
  descriptions of Modulation under `ModulatorySignals <ModulatorySignal_Modulation>` and `GatingSignals
  <GatingSignal_Modulation>` for additional details).  If the InputPort receives more than one GatingProjection,
  their values are combined before they are used to modify the `value <InputPort.value>` of InputPort.

.. _InputPort_Variable:

* `variable <InputPort.variable>` -- serves as the template for the `value <Projection_Base.value>` of the
  `Projections <Projection>` received by the InputPort:  each must be compatible with (that is, match both the
  number and type of elements of) the InputPort's `variable <InputPort.variable>` (see `Mapping_Matrix` for additonal
  details). In general, this must also be compatible with the item of the owner Mechanism's `variable
  <Mechanism_Base.variable>` to which the InputPort is assigned (see `above <InputPort_Variable_and_Value>` and
  `Mechanism InputPort specification <Mechanism_InputPort_Specification>`).

.. _InputPort_Function:

* `function <InputPort.function>` -- combines the `value <Projection_Base.value>` of all of the
  `Projections <Projection>` received by the InputPort, and assigns the result to the InputPort's `value
  <InputPort.value>` attribute.  The default function is `LinearCombination` that performs an elementwise (Hadamard)
  sums the values. However, the parameters of the `function <InputPort.function>` -- and thus the `value
  <InputPort.value>` of the InputPort -- can be modified by any `GatingProjections <GatingProjection>` received by
  the InputPort (listed in its `mod_afferents <Port.mod_afferents>` attribute.  A custom function can also be
  specified, so long as it generates a result that is compatible with the item of the Mechanism's `variable
  <Mechanism_Base.variable>` to which the `InputPort is assigned <Mechanism_InputPorts>`.

.. _InputPort_Value:

* `value <InputPort.value>` -- the result returned by its `function <InputPort.function>`,
  after aggregating the value of the `PathProjections <PathwayProjection>` it receives, possibly modified by any
  `GatingProjections <GatingProjection>` received by the InputPort. It must be compatible with the
  item of the owner Mechanism's `variable <Mechanism_Base.variable>` to which the `InputPort has been assigned
  <Mechanism_InputPorts>` (see `above <InputPort_Variable_and_Value>` and `Mechanism InputPort specification
  <Mechanism_InputPort_Specification>`).

.. _InputPort_Weights_And_Exponents:

* `weight <InputPort.weight>` and `exponent <InputPort.exponent>` -- these can be used by the Mechanism to which the
  InputPort belongs when that combines the `value <InputPort.value>`\\s of its Ports (e.g., an ObjectiveMechanism
  uses the weights and exponents assigned to its InputPorts to determine how the values it monitors are combined by
  its `function <ObjectiveMechanism>`).  The value of each must be an integer or float, and the default is 1 for both.

.. _InputPort_Execution:

Execution
---------

An InputPort cannot be executed directly.  It is executed when the Mechanism to which it belongs is executed.
When this occurs, the InputPort executes any `Projections <Projection>` it receives, calls its `function
<InputPort.function>` to combines the values received from any `MappingProjections <MappingProjection>` it receives
(listed in its its `path_afferents  <Port.path_afferents>` attribute) and modulate them in response to any
`GatingProjections <GatingProjection>` (listed in its `mod_afferents <Port.mod_afferents>` attribute),
and then assigns the result to the InputPort's `value <InputPort.value>` attribute. This, in turn, is assigned to
the item of the Mechanism's `variable <Mechanism_Base.variable>` and `input_values <Mechanism_Base.input_values>`
attributes  corresponding to that InputPort (see `Mechanism Variable and InputPorts
<Mechanism_Variable_and_InputPorts>` for additional details).

.. _InputPort_Class_Reference:

Class Reference
---------------

"""
import inspect
import numbers
import warnings

import collections
import numpy as np
import typecheck as tc

from psyneulink.core.components.component import DefaultsFlexibility
from psyneulink.core.components.functions.function import Function
from psyneulink.core.components.functions.transferfunctions import Linear
from psyneulink.core.components.functions.combinationfunctions import CombinationFunction, LinearCombination, Reduce
from psyneulink.core.components.functions.statefulfunctions.memoryfunctions import Buffer
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.components.ports.port import PortError, Port_Base, _instantiate_port_list, port_type_keywords
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import \
    COMBINE, CONTROL_SIGNAL, EXPONENT, FUNCTION, GATING_SIGNAL, INPUT_PORT, INPUT_PORTS, INPUT_PORT_PARAMS, \
    LEARNING_SIGNAL, MAPPING_PROJECTION, MATRIX, NAME, OPERATION, OUTPUT_PORT, OUTPUT_PORTS, OWNER,\
    PARAMS, PRODUCT, PROJECTIONS, REFERENCE_VALUE, \
    SENDER, SHADOW_INPUTS, SHADOW_INPUT_NAME, SIZE, PORT_TYPE, SUM, VALUE, VARIABLE, WEIGHT
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import append_type_to_name, convert_to_np_array, is_numeric, iscompatible, kwCompatibilityLength

__all__ = [
    'InputPort', 'InputPortError', 'port_type_keywords', 'SHADOW_INPUTS',
]

port_type_keywords = port_type_keywords.update({INPUT_PORT})

# InputPortPreferenceSet = BasePreferenceSet(log_pref=logPrefTypeDefault,
#                                                          reportOutput_pref=reportOutputPrefTypeDefault,
#                                                          verbose_pref=verbosePrefTypeDefault,
#                                                          param_validation_pref=paramValidationTypeDefault,
#                                                          level=PreferenceLevel.TYPE,
#                                                          name='InputPortClassPreferenceSet')

WEIGHT_INDEX = 1
EXPONENT_INDEX = 2

DEFER_VARIABLE_SPEC_TO_MECH_MSG = "InputPort variable not yet defined, defer to Mechanism"

class InputPortError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class InputPort(Port_Base):
    """
    InputPort(                                     \
        variable=None,                             \
        reference_value=None,                      \
        function=LinearCombination(operation=SUM), \
        combine=None,                              \
        projections=None,                          \
        weight=None,                               \
        exponent=None,                             \
        internal_only=False)

    Subclass of `Port <Port>` that calculates and represents the input to a `Mechanism <Mechanism>` from one or more
    `PathwayProjections <PathwayProjection>`.  See `Port_Class_Reference` for additional arguments and attributes.

    COMMENT:

    PortRegistry
    -------------
        All InputPorts are registered in PortRegistry, which maintains an entry for the subclass,
        a count for all instances of it, and a dictionary of those instances

    COMMENT

    Arguments
    ---------

    reference_value : number, list or np.ndarray
        the value of the item of the owner Mechanism's `variable <Mechanism_Base.variable>` attribute to which
        the InputPort is assigned; used as the template for the InputPort's `value <InputPort.value>` attribute.

    variable : number, list or np.ndarray
        specifies the shape of the  InputPort's `variable <InputPort.variable>`, which may be used to define the
        shape of the `matrix <MappingProjection.matrix>` parameter of the `MappingProjection` that projects to the
        Inputport (see `InputPort_Variable` for additional details).

    function : Function or method : default LinearCombination(operation=SUM)
        specifies the function applied to the variable. The default value combines the `values
        <Projection_Base.value>` of the `Projections <Projection>` received by the InputPort.  Any function
        can be assigned, however:  a) it must produce a result that has the same format (number and type of elements)
        as the item of its owner Mechanism's `variable <Mechanism_Base.variable>` to which the InputPort has been
        assigned;  b) if it is not a CombinationFunction, it may produce unpredictable results if the InputPort
        receives more than one Projection (see `function <InputPort.function>`.

    combine : SUM or PRODUCT : default None
        specifies the **operation** argument used by the default `LinearCombination` function, which determines how the
        `value <Projection_Base.value>` of the InputPort's `projections <Port.projections>` are combined.  This is a
        convenience argument, that allows the **operation** to be specified without having to specify the
        LinearCombination function; it assumes that LinearCombination (the default) is used as the InputPort's function
        -- if it conflicts with a specification of **function** an error is generated.

    projections : list of Projection specifications
        specifies the `MappingProjection(s) <MappingProjection>`, `ControlProjection(s) <ControlProjection>` and/or
        `GatingProjection(s) <GatingProjection>` to be received by the InputPort, and that are listed in its
        `path_afferents <Port.path_afferents>` and `mod_afferents <Port.mod_afferents>` attributes,
        respectively (see `InputPort_Compatability_and_Constraints` for additional details).  If **projections** but
        neither **variable** nor **size** are specified, then the `value <Projection_Base.value>` of the Projection(s)
        or their `senders <Projection_Base.sender>` specified in **projections** argument are used to determine the
        InputPort's `variable <InputPort.variable>`.

    weight : number : default 1
        specifies the value of the `weight <InputPort.weight>` attribute of the InputPort.

    exponent : number : default 1
        specifies the value of the `exponent <InputPort.exponent>` attribute of the InputPort.

    internal_only : bool : False
        specifies whether the InputPort requires external input when its `owner <Port.owner>` is the `INPUT`
        `Node <Composition_Nodes>` of a `Composition (see `internal_only <InputPort.internal_only>` for details).

    Attributes
    ----------

    variable : value, list or np.ndarray
        the template for the `value <Projection_Base.value>` of each Projection that the InputPort receives,
        each of which must match the format (number and types of elements) of the InputPort's
        `variable <InputPort.variable>`.  If neither the **variable** or **size** argument is specified, and
        **projections** is specified, then `variable <InputPort.variable>` is assigned the `value
        <Projection_Base.value>` of the Projection(s) or its `sender <Projection_Base.sender>`.

    function : Function
        If it is a `CombinationFunction`, it combines the `values <Projection_Base.value>` of the `PathwayProjections
        <PathwayProjection>` (e.g., `MappingProjections <MappingProjection>`) received by the InputPort  (listed in
        its `path_afferents <Port.path_afferents>` attribute), under the possible influence of `GatingProjections
        <GatingProjection>` received by the InputPort (listed in its `mod_afferents <Port.mod_afferents>` attribute).
        The result is assigned to the InputPort's `value <InputPort.value>` attribute. For example, the default
        (`LinearCombination` with *SUM* as it **operation**) performs an element-wise (Hadamard) sum of its Projection
        `values <Projection_Base.value>`, and assigns to `value <InputPort.value>` an array that is of the same length
        as each of the Projection `values <Projection_Base.value>`.  If the InputPort receives only one Projection,
        then any other function can be applied and it will generate a value that is the same length as the Projection's
        `value <Projection_Base.value>`. However, if the InputPort receives more than one Projection and uses a function
        other than a CombinationFunction, a warning is generated and only the `value <Projection_Base.value>` of the
        first Projection list in `path_afferents <Port.path_afferents>` is used by the function, which may generate
        unexpected results when executing the Mechanism or Composition to which it belongs.

    value : value or ndarray
        the output of the InputPort's `function <InputPort.function>`, that is assigned to an item of the owner
        Mechanism's `variable <Mechanism_Base.variable>` attribute.

    label : string or number
        the string label that represents the current `value <InputPort.value>` of the InputPort, according to the
        owner mechanism's `input_labels_dict <Mechanism.input_labels_dict>`. If the current `value <InputPort.value>`
        of the InputPort does not have a corresponding label, then the numeric `value <InputPort.value>` is returned.

    weight : number
        see `weight and exponent <InputPort_Weights_And_Exponents>` for description.

    exponent : number
        see `weight and exponent <InputPort_Weights_And_Exponents>` for description.

    internal_only : bool
        determines whether `input from a Composition <Composition_Execution_Input>` must be specified for this
        InputPort from a Composition's `execution method <Composition_Execution_Method>` if the InputPort's `owner
        <Port.owner>` is an `INPUT` `Node <Composition_Nodes>` of that Composition; if `True`, external input is
        *not* required or allowed.

    name : str
        the name of the InputPort; if it is not specified in the **name** argument of the constructor, a default is
        assigned by the InputPortRegistry of the Mechanism to which the InputPort belongs.  Note that some Mechanisms
        automatically create one or more non-default InputPorts, that have pre-specified names.  However, if any
        InputPorts are specified in the **input_ports** argument of the Mechanism's constructor, those replace those
        InputPorts (see `note <Mechanism_Default_Port_Suppression_Note>`), and `standard naming conventions
        <Registry_Naming>` apply to the InputPorts specified, as well as any that are added to the Mechanism once it
        is created (see `note <Port_Naming_Note>`).

    """

    #region CLASS ATTRIBUTES

    componentType = INPUT_PORT
    paramsType = INPUT_PORT_PARAMS

    portAttributes = Port_Base.portAttributes | {WEIGHT, EXPONENT}

    connectsWith = [OUTPUT_PORT,
                    LEARNING_SIGNAL,
                    GATING_SIGNAL,
                    CONTROL_SIGNAL
                    ]
    connectsWithAttribute = [OUTPUT_PORTS]
    projectionSocket = SENDER
    modulators = [GATING_SIGNAL, CONTROL_SIGNAL]
    canReceive = modulators + [MAPPING_PROJECTION]
    projection_type = MAPPING_PROJECTION

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TYPE_DEFAULT_PREFERENCES
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     PREFERENCE_SET_NAME: 'InputPortCustomClassPreferences',
    #     PREFERENCE_KEYWORD<pref>: <setting>...}

    # Note: the following enforce encoding as 1D np.ndarrays (one variable/value array per port)
    variableEncodingDim = 1
    valueEncodingDim = 1

    class Parameters(Port_Base.Parameters):
        """
            Attributes
            ----------

                combine
                    see `combine <InputPort.combine>`

                    :default value: None
                    :type:

                exponent
                    see `exponent <InputPort.exponent>`

                    :default value: None
                    :type:

                function
                    see `function <InputPort_Function>`

                    :default value: `LinearCombination`
                    :type: `Function`

                internal_only
                    see `internal_only <InputPort.internal_only>`

                    :default value: False
                    :type: ``bool``

                shadow_inputs
                    specifies whether InputPort shadows inputs of another InputPort;
                    if not None, must be assigned another InputPort

                    :default value: None
                    :type:
                    :read only: True

                weight
                    see `weight <InputPort.weight>`

                    :default value: None
                    :type:
        """
        function = Parameter(LinearCombination(operation=SUM), stateful=False, loggable=False)
        weight = Parameter(None, modulable=True)
        exponent = Parameter(None, modulable=True)
        combine = None
        internal_only = Parameter(False, stateful=False, loggable=False, pnl_internal=True)
        shadow_inputs = Parameter(None, stateful=False, loggable=False, read_only=True, pnl_internal=True, structural=True)

    #endregion

    @handle_external_context()
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
                 internal_only: tc.optional(bool) = None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None,
                 **kwargs):

        if variable is None and size is None and projections is not None:
            variable = self._assign_variable_from_projection(variable, size, projections)

        # If combine argument is specified, save it along with any user-specified function for _validate_params()
        if combine:
            self.combine_function_args = (combine, function)

        # If owner or reference_value has not been assigned, defer init to Port._instantiate_projection()
        # if owner is None or (variable is None and reference_value is None and projections is None):
        if owner is None:
            # Temporarily name InputPort
            self._assign_deferred_init_name(name, context)
            # Store args for deferred initialization
            self._store_deferred_init_args(**locals())

            # Flag for deferred initialization
            self.initialization_status = ContextFlags.DEFERRED_INIT
            return

        self.reference_value = reference_value

        # Validate sender (as variable) and params, and assign to variable
        # Note: pass name of owner (to override assignment of componentName in super.__init__)
        super(InputPort, self).__init__(
            owner,
            variable=variable,
            size=size,
            projections=projections,
            function=function,
            weight=weight,
            exponent=exponent,
            internal_only=internal_only,
            shadow_inputs=None,
            params=params,
            name=name,
            prefs=prefs,
            context=context,
        )

        if self.name is self.componentName or self.componentName + '-' in self.name:
            self._assign_default_port_Name(context=context)

    def _assign_variable_from_projection(self, variable, size, projections):
        """Assign variable to value of Projection in projections
        """
        from psyneulink.core.components.projections.projection import \
            Projection, _parse_connection_specs

        if not isinstance(projections, list):
            projections = [projections]

        # Use only first specification in the list returned, and assume any others are the same size
        #     (which they must be); leave validation of this to _instantiate_projections_to_port
        proj_spec = _parse_connection_specs(InputPort, self, projections)[0]

        if isinstance(proj_spec.projection, Projection):
            variable = proj_spec.projection.defaults.value
        elif isinstance(proj_spec.port, OutputPort):
            variable = proj_spec.port.defaults.value
        else:
            raise InputPortError(f"Unrecognized specification for \'{PROJECTIONS}\' arg of {self.name}.")

        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate weights and exponents

        This needs to be done here
            (so that they can be ignored if not specified here or in the function)
        """

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        # Make sure **combine** and **function** args specified in constructor don't conflict
        if hasattr(self, 'combine_function_args'):
            combine, function = self.combine_function_args
            if function:
                owner_name = ""
                if self.owner:
                    owner_name = f" for InputPort of {self.owner.name}."
                if isinstance(function, LinearCombination):
                    # specification of combine conflicts with operation specified for LinearCombination in function arg
                    if function.operation != combine:
                        raise InputPortError(f"Specification of {repr(COMBINE)} argument ({combine.upper()}) "
                                             f"conflicts with specification of {repr(OPERATION)} "
                                             f"({function.operation.upper()}) for LinearCombination in "
                                             f"{repr(FUNCTION)} argument{owner_name}.")
                    else:
                        # LinearFunction has been specified with same operation as specified for combine,
                        # so delete combine_function_args attribute so it is not seen in _instantiate_function
                        # in order to leave function intact (as it may have other parameters specified by user)
                        del self.combine_function_args
                # combine assumes LinearCombination, but Function other than LinearCombination specified for function
                elif isinstance(function, Function):
                    raise InputPortError(f"Specification of {repr(COMBINE)} argument ({combine.upper()}) "
                                         f"conflicts with Function specified in {repr(FUNCTION)} argument "
                                         f"({function.name}){owner_name}.")
                # combine assumes LinearCombination, but class other than LinearCombination specified for function
                elif isinstance(function, type):
                    if not issubclass(function, LinearCombination):
                        raise InputPortError(f"Specification of {repr(COMBINE)} argument ({combine.upper()}) "
                                             f"conflicts with Function specified in {repr(FUNCTION)} argument "
                                             f"({function.__name__}){owner_name}.")
                else:
                    raise InputPortError(f"PROGRAM ERROR: unrecognized specification for function argument "
                                         f"({function}){owner_name}.")

        if WEIGHT in target_set and target_set[WEIGHT] is not None:
            if not isinstance(target_set[WEIGHT], (int, float)):
                raise InputPortError(f"'{WEIGHT}' parameter of {self.name} for {self.owner.name} "
                                     f"({target_set[WEIGHT]}) must be an int or float.")

        if EXPONENT in target_set and target_set[EXPONENT] is not None:
            if not isinstance(target_set[EXPONENT], (int, float)):
                raise InputPortError(f"'{EXPONENT}' parameter of {self.name} for {self.owner.name}"
                                     f"({ target_set[EXPONENT]}) must be an int or float.")

    def _validate_against_reference_value(self, reference_value):
        """Validate that Port.value is compatible with reference_value

        reference_value is the item of the owner Mechanism's variable to which the InputPort is assigned
        """
        match_len_option = {kwCompatibilityLength:False}
        if reference_value is not None and not iscompatible(reference_value, self.defaults.value, **match_len_option):
            name = self.name or ""
            raise InputPortError(f"Value specified for {name} {self.componentName} of {self.owner.name} "
                                 f"({self.defaults.value}) is not compatible with its expected format "
                                 f"({reference_value}).")

    def _instantiate_function(self, function, function_params=None, context=None):
        """If combine option was specified in constructor, assign as operation argument of LinearCombination function"""
        if hasattr(self, 'combine_function_args'):
            function = LinearCombination(operation=self.combine_function_args[0])
            del self.combine_function_args
        super()._instantiate_function(function=function, context=context)
        self._use_1d_variable = False
        if not isinstance(self.function, CombinationFunction):
            self._use_1d_variable = True
            self.function._variable_shape_flexibility = DefaultsFlexibility.RIGID
        else:
            self.function._variable_shape_flexibility = DefaultsFlexibility.FLEXIBLE

    def _instantiate_projections(self, projections, context=None):
        """Instantiate Projections specified in PROJECTIONS entry of params arg of Port's constructor

        Call _instantiate_projections_to_port to assign:
            PathwayProjections to .path_afferents
            ModulatoryProjections to .mod_afferents
        """
        self._instantiate_projections_to_port(projections=projections, context=context)

    def _check_for_duplicate_projections(self, projection):
        """Check if projection is redundant with one in path_afferents of InputPort

        Check for any instantiated projection in path_afferents with the same sender as projection
        or one in deferred_init status with sender specification that is the same type as projection.

        Returns redundant Projection if found, otherwise False.
        """

        try:
            self.path_afferents
        except:
            if self.initialization_status == ContextFlags.DEFERRED_INIT:
                raise InputPortError(f"Attempt to assign Projection ('{projection}') "
                                     f"to InputPort ('{self.name}') that is in deferred init")
            else:
                raise InputPortError(f"No 'path_afferents' for {self.name}")

        # FIX: 7/22/19 - CHECK IF SENDER IS SPECIFIED AS MECHANISM AND, IF SO, CHECK ITS PRIMARY_OUTPUT_PORT
        duplicate = next(iter([proj for proj in self.path_afferents
                               if ((proj.sender == projection.sender and proj != projection)
                                   or (proj.initialization_status == ContextFlags.DEFERRED_INIT
                                       and proj._init_args[SENDER] == type(projection.sender)))]), None)
        if duplicate and self.verbosePref or self.owner.verbosePref:
            from psyneulink.core.components.projections.projection import Projection
            warnings.warn(f'{Projection.__name__} from {projection.sender.name}  {projection.sender.__class__.__name__}'
                          f' of {projection.sender.owner.name} to {self.name} {self.__class__.__name__} of '
                          f'{self.owner.name} already exists; will ignore additional one specified ({projection.name}).')
        return duplicate

    def _parse_function_variable(self, variable, context=None):
        variable = super()._parse_function_variable(variable, context)
        try:
            if self._use_1d_variable and variable.ndim > 1:
                return np.array(variable[0])
        except AttributeError:
            pass
        return variable

    def _get_variable_from_projections(self, context=None):
        """
            Call self.function with self._path_proj_values

            If variable is None there are no active PathwayProjections in the Composition being run,
            return None so that it is ignored in execute method (i.e., not combined with base_value)
        """
        # Check for Projections that are active in the Composition being run
        path_proj_values = [
            proj.parameters.value._get(context)
            for proj in self.path_afferents
            if self.afferents_info[proj].is_active_in_composition(context.composition)
        ]

        if len(path_proj_values) > 0:
            return convert_to_np_array(path_proj_values)
        else:
            return None

    def _get_primary_port(self, mechanism):
        return mechanism.input_port

    @tc.typecheck
    def _parse_port_specific_specs(self, owner, port_dict, port_specific_spec):
        """Get weights, exponents and/or any connections specified in an InputPort specification tuple

        Tuple specification can be:
            (port_spec, connections)
            (port_spec, weights, exponents, connections)

        See Port._parse_port_specific_spec for additional info.
.
        Returns:
             - port_spec:  1st item of tuple if it is a numeric value;  otherwise None
             - params dict with WEIGHT, EXPONENT and/or PROJECTIONS entries if any of these was specified.

        """
        # FIX: ADD FACILITY TO SPECIFY WEIGHTS AND/OR EXPONENTS FOR INDIVIDUAL OutputPort SPECS
        #      CHANGE EXPECTATION OF *PROJECTIONS* ENTRY TO BE A SET OF TUPLES WITH THE WEIGHT AND EXPONENT FOR IT
        #      THESE CAN BE USED BY THE InputPort's LinearCombination Function
        #          (AKIN TO HOW THE MECHANISM'S FUNCTION COMBINES InputPort VALUES)
        #      THIS WOULD ALLOW AN ADDITIONAL HIERARCHICAL LEVEL FOR NESTING ALGEBRAIC COMBINATION OF INPUT VALUES
        #      TO A MECHANISM
        from psyneulink.core.components.projections.projection import Projection, _parse_connection_specs

        params_dict = {}
        port_spec = port_specific_spec

        if isinstance(port_specific_spec, dict):
            # FIX: 10/3/17 - CHECK HERE THAT, IF MECHANISM ENTRY IS USED, A VARIABLE, WEIGHT AND/OR EXPONENT ENTRY
            # FIX:                       IS APPLIED TO ALL THE OutputPorts SPECIFIED IN OUTPUT_PORTS
            # FIX:                       UNLESS THEY THEMSELVES USE A Port specification dict WITH ANY OF THOSE ENTRIES
            # FIX:           USE ObjectiveMechanism EXAMPLES
            # if MECHANISM in port_specific_spec:
            #     if OUTPUT_PORTS in port_specific_spec
            if SIZE in port_specific_spec:
                if (VARIABLE in port_specific_spec or
                        any(key in port_dict and port_dict[key] is not None for key in {VARIABLE, SIZE})):
                    raise InputPortError(f"PROGRAM ERROR: SIZE specification found in port_specific_spec dict "
                                         f"for {self.__name__} specification of {owner.name} when SIZE or VARIABLE "
                                         f"is already present in its port_specific_spec dict or port_dict.")
                port_dict.update({VARIABLE:np.zeros(port_specific_spec[SIZE])})
                del port_specific_spec[SIZE]
                return port_dict, port_specific_spec
            return None, port_specific_spec

        elif isinstance(port_specific_spec, tuple):

            # GET PORT_SPEC AND ASSIGN PROJECTIONS_SPEC **********************************************************

            tuple_spec = port_specific_spec

            # 2-item tuple specification
            if len(tuple_spec) == 2:

                # 1st item is a value, so treat as Port spec (and return to _parse_port_spec to be parsed)
                #   and treat 2nd item as Projection specification
                if is_numeric(tuple_spec[0]):
                    port_spec = tuple_spec[0]
                    reference_value = port_dict[REFERENCE_VALUE]
                    # Assign value so sender_dim is skipped below
                    # (actual assignment is made in _parse_port_spec)
                    if reference_value is None:
                        port_dict[REFERENCE_VALUE]=port_spec
                    elif not iscompatible(port_spec, reference_value):
                        raise PortError(f"Value in first item of 2-item tuple specification {InputPort.__name__} of "
                                        f"{owner.name} ({port_spec}) is not compatible with its {REFERENCE_VALUE} "
                                        f"({reference_value}).")
                    projections_spec = tuple_spec[1]

                # Tuple is Projection specification that is used to specify the Port,
                else:
                    # return None in port_spec to suppress further, recursive parsing of it in _parse_port_spec
                    port_spec = None
                    if tuple_spec[0] != self:
                        # If 1st item is not the current port (self), treat as part of the projection specification
                        projections_spec = tuple_spec
                    else:
                        # Otherwise, just use 2nd item as projection spec
                        port_spec = None
                        projections_spec = tuple_spec[1]

            # 3- or 4-item tuple specification
            elif len(tuple_spec) in {3,4}:
                # Tuple is projection specification that is used to specify the Port,
                #    so return None in port_spec to suppress further, recursive parsing of it in _parse_port_spec
                port_spec = None
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
                    # Parse the value of all of the Projections to get/validate variable for InputPort
                    variable = []
                    for projection_spec in params_dict[PROJECTIONS]:
                        # FIX: 10/3/17 - PUTTING THIS HERE IS A HACK...
                        # FIX:           MOVE TO _parse_port_spec UNDER PROCESSING OF ProjectionTuple SPEC
                        # FIX:           USING _get_port_for_socket
                        # from psyneulink.core.components.projections.projection import _parse_projection_spec

                        # Try to get matrix for projection
                        try:
                            sender_dim = np.array(projection_spec.port.value).ndim
                        except AttributeError as e:
                            if (isinstance(projection_spec.port, type) or
                                     projection_spec.port.initialization_status == ContextFlags.DEFERRED_INIT):
                                continue
                            else:
                                raise PortError(f"PROGRAM ERROR: indeterminate value for {projection_spec.port.name} "
                                                f"specified to project to {self.__name__} of {owner.name}.")

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
                            if projection.initialization_status == ContextFlags.DEFERRED_INIT:
                                continue
                            # possible needs to be projection.defaults.matrix?
                            matrix = projection.matrix
                        else:
                            raise InputPortError(f"Unrecognized Projection specification for {self.name} of "
                                                 f"{owner.name} ({projection_spec}).")

                        # Determine length of value of projection
                        if matrix is None:
                            # If a reference_value has been specified, it presumably represents the item of the
                            #    owner Mechanism's default_variable to which the InputPort corresponds,
                            #    so use that to constrain the InputPort's variable
                            if port_dict[REFERENCE_VALUE] is not None:
                                variable.append(port_dict[REFERENCE_VALUE])
                                continue
                            # If matrix has not been specified, no worries;
                            #    variable_item can be determined by value of sender
                            sender_shape = np.array(projection_spec.port.value).shape
                            variable_item = np.zeros(sender_shape)
                            # If variable_item HASN'T been specified, or it is same shape as any previous ones,
                            #     use sender's value
                            if ((VARIABLE not in port_dict or port_dict[VARIABLE] is None) and
                                    (not variable or variable_item.shape == variable[0].shape)):
                                # port_dict[VARIABLE] = variable
                                variable.append(variable_item)
                            # If variable HAS been assigned, make sure value is the same for this sender
                            elif np.array(port_dict[VARIABLE]).shape != variable_item.shape:
                                # If values for senders differ, assign None so that Port's default is used
                                variable = None
                                # No need to check any more Projections
                                break

                        # Remove dimensionality of sender OutputPort, and assume that is what receiver will receive
                        else:
                            proj_val_shape = matrix.shape[sender_dim :]
                            # port_dict[VARIABLE] = np.zeros(proj_val_shape)
                            variable.append(np.zeros(proj_val_shape))
                    # Sender's value has not been defined or senders have values of different lengths,
                    if not variable:
                        # If reference_value was provided, use that as the InputPort's variable
                        #    (i.e., assume its function won't transform it)
                        if REFERENCE_VALUE in port_dict and port_dict[REFERENCE_VALUE] is not None:
                            port_dict[VARIABLE] = port_dict[REFERENCE_VALUE]
                        # Nothing to use as variable, so raise exception and allow it to be handled "above"
                        else:
                            raise AttributeError(DEFER_VARIABLE_SPEC_TO_MECH_MSG)
                    else:
                        port_dict[VARIABLE] = variable

                except InputPortError:
                    raise InputPortError(f"Tuple specification in {InputPort.__name__} specification dictionary for "
                                         f"{owner.name} ({projections_spec}) is not a recognized specification for "
                                         f"one or more Mechanisms, {OutputPort.__name__}s, or {Projection.__name__}s "
                                         f"that project to it.")

            # GET WEIGHT AND EXPONENT IF SPECIFIED ***************************************************************

            if len(tuple_spec) == 2:
                pass

            # Tuple is (spec, weights, exponents<, afferent_source_spec>),
            #    for specification of weights and exponents,  + connection(s) (afferent projection(s)) to InputPort
            elif len(tuple_spec) in {3, 4}:

                weight = tuple_spec[WEIGHT_INDEX]
                exponent = tuple_spec[EXPONENT_INDEX]

                if weight is not None and not isinstance(weight, numbers.Number):
                    raise InputPortError(f"Specification of the weight ({weight}) in tuple of {InputPort.__name__} "
                                         f"specification dictionary for {owner.name} must be a number.")
                params_dict[WEIGHT] = weight

                if exponent is not None and not isinstance(exponent, numbers.Number):
                    raise InputPortError(f"Specification of the exponent ({exponent}) in tuple of {InputPort.__name__} "
                                         f"specification dictionary for {owner.name} must be a number.")
                params_dict[EXPONENT] = exponent

            else:
                raise PortError(f"Tuple provided as port_spec for {InputPort.__name__} of {owner.name} "
                                f"({tuple_spec}) must have either 2, 3 or 4 items.")

        elif port_specific_spec is not None:
            raise InputPortError(f"PROGRAM ERROR: Expected tuple or dict for {self.__class__.__name__}-specific params "
                                 f"but, got: {port_specific_spec}.")

        return port_spec, params_dict

    def _parse_self_port_type_spec(self, owner, input_port, context=None):
        """Return InputPort specification dictionary with projections that shadow inputs to input_port

        Called by _parse_port_spec if InputPort specified for a Mechanism belongs to a different Mechanism
        """

        if not isinstance(input_port, InputPort):
            raise InputPortError(f"PROGRAM ERROR: InputPort._parse_self_port_type called "
                                 f"with non-InputPort specification ({input_port}).")

        sender_output_ports = [p.sender for p in input_port.path_afferents]
        port_spec = {NAME: SHADOW_INPUT_NAME + input_port.owner.name,
                      VARIABLE: np.zeros_like(input_port.variable),
                      PORT_TYPE: InputPort,
                      PROJECTIONS: sender_output_ports,
                      PARAMS: {SHADOW_INPUTS: input_port},
                      OWNER: owner}
        return port_spec

    @staticmethod
    def _port_spec_allows_override_variable(spec):
        """
        Returns
        -------
            True - if **spec** outlines a spec for creating an InputPort whose variable can be
                overridden by a default_variable or size argument
            False - otherwise

            ex: specifying an InputPort with a Mechanism allows overriding
        """
        from psyneulink.core.components.mechanisms.mechanism import Mechanism

        if isinstance(spec, Mechanism):
            return True
        if isinstance(spec, collections.abc.Iterable):
            # generally 2-4 tuple spec, but allows list spec
            for item in spec:
                if isinstance(item, Mechanism):
                    return True
                # handles tuple spec where first item of tuple is itself a (name, Mechanism) tuple
                elif (
                    isinstance(item, collections.abc.Iterable)
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

    def get_label(self, context=None):
        try:
            label_dictionary = self.owner.input_labels_dict
        except AttributeError:
            label_dictionary = {}
        return self._get_value_label(label_dictionary, self.owner.input_ports, context=context)

    @property
    def position_in_mechanism(self):
        if hasattr(self, "owner"):
            if self.owner is not None:
                return self.owner.get_input_port_position(self)
            else:
                return None
        return None

    @staticmethod
    def _get_port_function_value(owner, function, variable):
        """Put InputPort's variable in a list if its function is LinearCombination and variable is >=2d

        InputPort variable must be embedded in a list so that LinearCombination (its default function)
        returns a variable that is >=2d intact (rather than as arrays to be combined);
        this is normally done in port._update() (and in Port._instantiate-function), but that
        can't be called by _parse_port_spec since the InputPort itself may not yet have been instantiated.

        """

        if (
                (
                    (inspect.isclass(function) and issubclass(function, LinearCombination))
                    or isinstance(function, LinearCombination)
                )
                and isinstance(variable, np.matrix)
        ):
            variable = [variable]

        # if function is None, use Port's default function
        function = function or InputPort.defaults.function

        return Port_Base._get_port_function_value(owner=owner, function=function, variable=variable)


def _instantiate_input_ports(owner, input_ports=None, reference_value=None, context=None):
    """Call Port._instantiate_port_list() to instantiate ContentAddressableList of InputPort(s)

    Create ContentAddressableList of InputPort(s) specified in self.input_ports

    If input_ports is not specified:
        - use owner.input_ports as list of InputPort specifications
        - if owner.input_ports is empty, user owner.defaults.variable to create a default InputPort

    When completed:
        - self.input_ports contains a ContentAddressableList of one or more input_ports
        - self.input_port contains the `primary InputPort <InputPort_Primary>`:  first or only one in input_ports
        - self.input_ports contains the same ContentAddressableList (of one or more input_ports)
        - each InputPort corresponds to an item in the variable of the owner's function
        - the value of all of the input_ports is stored in a list in input_value
        - if there is only one InputPort, it is assigned the full value

    Note: Port._instantiate_port_list()
              parses self.defaults.variable (2D np.array, passed in reference_value)
              into individual 1D arrays, one for each InputPort

    (See Port._instantiate_port_list() for additional details)

    Returns list of instantiated InputPorts
    """

    # This allows method to be called by Mechanism.add_input_ports() with set of user-specified input_ports,
    #    while calls from init_methods continue to use owner.input_ports (i.e., InputPort specifications
    #    assigned in the **input_ports** argument of the Mechanism's constructor)
    input_ports = input_ports or owner.input_ports

    # Parse any SHADOW_INPUTS specs into actual InputPorts to be shadowed
    if input_ports is not None:
        input_ports = _parse_shadow_inputs(owner, input_ports)

    port_list = _instantiate_port_list(owner=owner,
                                         port_list=input_ports,
                                         port_types=InputPort,
                                         port_Param_identifier=INPUT_PORT,
                                         reference_value=reference_value if reference_value is not None
                                                                         else owner.defaults.variable,
                                         reference_value_name=VALUE,
                                         context=context)

    # Call from Mechanism.add_ports, so add to rather than assign input_ports (i.e., don't replace)
    if context.source & (ContextFlags.METHOD | ContextFlags.COMMAND_LINE):
        owner.input_ports.extend(port_list)
    else:
        owner.input_ports = port_list

    # Assign value of require_projection_in_composition
    for port in owner.input_ports:
        # Assign True for owner's primary InputPort and the value has not already been set in InputPort constructor
        if port.require_projection_in_composition is None and owner.input_port == port:
            port.parameters.require_projection_in_composition._set(True, context)

    # Check that number of input_ports and their variables are consistent with owner.defaults.variable,
    #    and adjust the latter if not
    variable_item_is_OK = False
    for i, input_port in enumerate(owner.input_ports):
        try:
            variable_item_is_OK = iscompatible(owner.defaults.variable[i], input_port.value)
            if not variable_item_is_OK:
                break
        except IndexError:
            variable_item_is_OK = False
            break

    if not variable_item_is_OK:
        old_variable = owner.defaults.variable
        owner.defaults.variable = owner._handle_default_variable(default_variable=[port.value
                                                                                   for port in owner.input_ports])

        if owner.verbosePref:
            warnings.warn(f"Variable for {old_variable} ({append_type_to_name(owner)}) has been adjusted to match "
                          f"number and format of its input_ports: ({owner.defaults.variable}).")

    return port_list

def _parse_shadow_inputs(owner, input_ports):
    """Parses any {SHADOW_INPUTS:[InputPort or Mechaism,...]} items in input_ports into InputPort specif. dict."""

    input_ports_to_shadow_specs=[]
    for spec_idx, spec in enumerate(input_ports):
        # If {SHADOW_INPUTS:[InputPort or Mechaism,...]} is found:
        if isinstance(spec, dict) and SHADOW_INPUTS in spec:
            input_ports_to_shadow_in_spec=[]
            # For each item in list of items to shadow specified in that entry:
            for item in list(spec[SHADOW_INPUTS]):
                from psyneulink.core.components.mechanisms.mechanism import Mechanism
                # If an InputPort was specified, just used that
                if isinstance(item, InputPort):
                    input_ports_to_shadow_in_spec.append(item)
                # If Mechanism was specified, use all of its InputPorts
                elif isinstance(item, Mechanism):
                    input_ports_to_shadow_in_spec.extend(item.input_ports)
                else:
                    raise InputPortError(f"Specification of {repr(SHADOW_INPUTS)} in for {repr(INPUT_PORTS)} arg of "
                                         f"{owner.name} must be a {Mechanism.__name__} or {InputPort.__name__}.")
            input_ports_to_shadow_specs.append((spec_idx, input_ports_to_shadow_in_spec))

    # If any SHADOW_INPUTS specs were found in input_ports, replace them with actual InputPorts to be shadowed
    if input_ports_to_shadow_specs:
        for item in input_ports_to_shadow_specs:
            idx = item[0]
            del input_ports[idx]
            input_ports[idx:idx] = item[1]
        # Update owner's variable based on full set of InputPorts specified
        owner.defaults.variable, _ = owner._handle_arg_input_ports(input_ports)

    return input_ports
