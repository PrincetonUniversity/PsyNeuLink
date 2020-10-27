# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


#  *********************************************  Port ********************************************************

"""
Contents
--------

  * `Port_Overview`
  * `Port_Creation`
      - `Port_Specification`
      - `Port_Projections`
      - `Port_Deferred_Initialization`
  * `Port_Structure`
      - `Port_Strucure_Owner`
      - `Port_Structure_Projections`
      - `Port_Structure_Variable_Function_Value`
      - `Port_Modulation`
  * `Port_Execution`
  * `Port_Examples`
  * `Port_Class_Reference`


.. _Port_Overview:

Overview
--------

A Port provides an interface to one or more `Projections <Projection>`, and receives the `value(s) <Projection>`
provided by them.  The value of a Port can be modulated by a `ModulatoryProjection <ModulatoryProjection>`. There are
three primary types of Ports (InputPorts, ParameterPorts and OutputPorts) as well as one subtype (ModulatorySignal,
used to send ModulatoryProjections), as summarized in the table below:

.. _Port_types_Table:

.. table:: **Port Types and Associated Projection Types**
   :align: left

   +-------------------+----------------------+-----------------------+-----------------+------------------------------+
   | *Port Type*       | *Owner*              |      *Description*    | *Modulated by*  |       *Specification*        |
   +===================+======================+=======================+=================+==============================+
   | `InputPort`       |  `Mechanism          |receives input from    |`ControlSignal`  |`InputPort` constructor;      |
   |                   |  <Mechanism>`        |`MappingProjection`    |or `GatingSignal`|`Mechanism <Mechanism>`       |
   |                   |                      |                       |                 |constructor or its            |
   |                   |                      |                       |                 |`add_ports` method            |
   +-------------------+----------------------+-----------------------+-----------------+------------------------------+
   |`ParameterPort`    |  `Mechanism          |represents parameter   |`ControlSignal`  |Implicitly whenever a         |
   |                   |  <Mechanism>` or     |value for a `Component |and/or           |parameter value is            |
   |                   |  `Projection         |<Component>`           |`LearningSignal` |`specified                    |
   |                   |  <Projection>`       |or its `function       |                 |<ParameterPort_Specification>`|
   |                   |                      |<Component.function>`  |                 |                              |
   +-------------------+----------------------+-----------------------+-----------------+------------------------------+
   | `OutputPort`      |  `Mechanism          |provides output to     |`ControlSignal`  |`OutputPort` constructor;     |
   |                   |  <Mechanism>`        |`MappingProjection`    |or `GatingSignal`|`Mechanism <Mechanism>`       |
   |                   |                      |                       |                 |constructor or its            |
   |                   |                      |                       |                 |`add_ports` method            |
   +-------------------+----------------------+-----------------------+-----------------+------------------------------+
   |`ModulatorySignal  |`ModulatoryMechanism  |provides value for     |                 |`ModulatoryMechanism          |
   |<ModulatorySignal>`|<ModulatoryMechanism>`|`ModulatoryProjection  |                 |<ModulatoryMechanism>`        |
   |                   |                      |<ModulatoryProjection>`|                 |constructor; tuple in Port    |
   |                   |                      |                       |                 |or parameter specification    |
   +-------------------+----------------------+-----------------------+-----------------+------------------------------+

COMMENT:

* `InputPort`:
    used by a Mechanism to receive input from `MappingProjections <MappingProjection>`;
    its value can be modulated by a `ControlSignal` or a `GatingSignal`.

* `ParameterPort`:
    * used by a Mechanism to represent the value of one of its parameters, or a parameter of its
      `function <Mechanism_Base.function>`, that can be modulated by a `ControlSignal`;
    * used by a `MappingProjection` to represent the value of its `matrix <MappingProjection.MappingProjection.matrix>`
      parameter, that can be modulated by a `LearningSignal`.

* `OutputPort`:
    used by a Mechanism to send its value to any efferent projections.  For
    `ProcessingMechanisms <ProcessingMechanism>` these are `PathwayProjections <PathwayProjection>`, most commonly
    `MappingProjections <MappingProjection>`.  For `ModulatoryMechanisms <ModulatoryMechanism>`, these are
    `ModulatoryProjections <ModulatoryProjection>` as described below. The `value <OutputPort.value>` of an
    OutputPort can be modulated by a `ControlSignal` or a `GatingSignal`.

* `ModulatorySignal <ModulatorySignal>`:
    a subclass of `OutputPort` used by `ModulatoryMechanisms <ModulatoryMechanism>` to modulate the value of the primary
    types of Ports listed above.  There are three types of ModulatorySignals:

    * `LearningSignal`, used by a `LearningMechanism` to modulate the *MATRIX* ParameterPort of a `MappingProjection`;
    * `ControlSignal`, used by a `ControlMechanism <ControlMechanism>` to modulate the `ParameterPort` of a `Mechanism
      <Mechanism>`;
    * `GatingSignal`, used by a `GatingMechanism` to modulate the `InputPort` or `OutputPort` of a `Mechanism
       <Mechanism>`.
    Modulation is discussed further `below <Port_Modulation>`, and described in detail under
    `ModulatorySignals <ModulatorySignal_Modulation>`.

COMMENT

.. _Port_Creation:

Creating a Port
----------------

In general, Ports are created automatically by the objects to which they belong (their `owner <Port_Strucure_Owner>`),
or by specifying the Port in the constructor for its owner.  For example, unless otherwise specified, when a
`Mechanism <Mechanism>` is created it creates a default `InputPort` and `OutputPort` for itself, and whenever any
Component is created, it automatically creates a `ParameterPort` for each of its `configurable parameters
<Component_Structural_Attributes>` and those of its `function <Component_Function>`. Ports are also created in
response to explicit specifications.  For example, InputPorts and OutputPorts can be specified in the constructor for
a Mechanism (see `Mechanism_Port_Specification`); and ParameterPorts are specified in effect when the value of a
parameter for any Component or its `function <Component.function>` is specified in the constructor for that Component
or function.  InputPorts and OutputPorts (but *not* ParameterPorts) can also be created directly using their
constructors, and then assigned to a Mechanism using the Mechanism's `add_ports <Mechanism_Base.add_ports>` method;
however, this should be done with caution as the Port must be compatible with other attributes of its owner (such as
its OutputPorts) and its `function <Mechanism_Base.function>` (for example, see `note <Mechanism_Add_InputPorts_Note>`
regarding InputPorts). Parameter Ports **cannot** on their own; they are always and only created when the Component
to which a parameter belongs is created.

COMMENT:
    IMPLEMENTATION NOTE:
    If the constructor for a Port is called programmatically other than on the command line (e.g., within a method)
    the **context** argument must be specified (by convention, as ContextFlags.METHOD); otherwise, it is assumed that
    it is being created on the command line.  This is taken care of when it is created automatically (e.g., as part
    of the construction of a Mechanism or Projection) by the _instantiate_port method that specifies a context
    when it calls the relevant Port constructor methods.
COMMENT

.. _Port_Specification:

*Specifying a Port*
~~~~~~~~~~~~~~~~~~~~

A Port can be specified using any of the following:

    * existing **Port** object;
    ..
    * name of a **Port subclass** (`InputPort`, `ParameterPort`, or `OutputPort`) -- creates a default Port of the
      specified type, using a default value for the Port that is determined by the context in which it is specified.
    ..
    * **value** -- creates a default Port using the specified value as its default `value <Port_Base.value>`.

    .. _Port_Specification_Dictionary:

    * **Port specification dictionary** -- can use the following: *KEY*:<value> entries, in addition to those
      specific to the Port's type (see documentation for each Port type):

      * *PORT_TYPE*:<Port type>
          specifies type of Port to create (necessary if it cannot be determined from
          the context of the other entries or in which it is being created).
      ..
      * *NAME*:<str>
          the string is used as the name of the Port.
      ..
      * *VALUE*:<value>
          the value is used as the default value of the Port.

      A Port specification dictionary can also be used to specify one or more `Projections <Projection>' to or from
      the Port, including `ModulatoryProjection(s) <ModulatoryProjection>` used to modify the `value
      <Port_Base.value>` of the Port.  The type of Projection(s) created depend on the type of Port specified and
      context of the specification (see `examples <Port_Specification_Dictionary_Examples>`).  This can be done using
      any of the following entries, each of which can contain any of the forms used to `specify a Projection
      <Projection_Specification>`:

      * *PROJECTIONS*:List[<`projection specification <Projection_Specification>`>,...]
          the list must contain a one or more `Projection specifications <Projection_Specification>` to or from
          the Port, and/or `ModulatorySignals <ModulatorySignal>` from which it should receive projections (see
          `Port_Projections` below).

      .. _Port_Port_Name_Entry:

      * *<str>*:List[<`projection specification <Projection_Specification>`>,...]
          this must be the only entry in the dictionary, and the string cannot be a PsyNeuLink
          keyword;  it is used as the name of the Port, and the list must contain one or more `Projection
          specifications <Projection_Specification>`.

      .. _Port_MECHANISM_PORTS_Entries:

      * *MECHANISM*:Mechanism
          this can be used to specify one or more Projections to or from the specified Mechanism.  If the entry appears
          without any accompanying Port specification entries (see below), the Projection is assumed to be a
          `MappingProjection` to the Mechanism's `primary InputPort <InputPort_Primary>` or from its `primary
          OutputPort <OutputPort_Primary>`, depending upon the type of Mechanism and context of specification.  It
          can also be accompanied by one or more Port specification entries described below, to create one or more
          Projections to/from those specific Ports (see `examples <Port_Port_Name_Entry_Example>`).
      ..
      * <PORTS_KEYWORD>:List[<str or Port.name>,...]
         this must accompany a *MECHANISM* entry (described above), and is used to specify its Port(s) by name.
         Each entry must use one of the following keywords as its key, and there can be no more than one of each:
            - *INPUT_PORTS*
            - *OUTPUT_PORTS*
            - *PARAMETER_PORTS*
            - *LEARNING_SIGNAL*
            - *CONTROL_SIGNAL*
            - *GATING_SIGNAL*.
         Each entry must contain a list Ports of the specified type, all of which belong to the Mechanism specified in
         the *MECHANISM* entry;  each item in the list must be the name of one the Mechanism's Ports, or a
         `ProjectionTuple <Port_ProjectionTuple>` the first item of which is the name of a Port. The types of
         Ports that can be specified in this manner depends on the type of the Mechanism and context of the
         specification (see `examples <Port_Port_Name_Entry_Example>`).

    * **Port, Mechanism, or list of these** -- creates a default Port with Projection(s) to/from the specified
      Ports;  the type of Port being created determines the type and directionality of the Projection(s) and,
      if Mechanism(s) are specified, which of their primary Ports are used (see Port subclasses for specifics).

   .. _Port_Tuple_Specification:

    * **Tuple specifications** -- these are convenience formats that can be used to compactly specify a Port
      by specifying other Components with which it should be connected by Projection(s). Different Ports support
      different forms, but all support the following two forms:

      .. _Port_2_Item_Tuple:

      * **2-item tuple:** *(<Port name or list of Port names>, <Mechanism>)* -- 1st item is the name of a Port or
        list of them, and the 2nd item is the Mechanism to which they belong; a Projection is created to or from each
        of the Ports specified.  The type of Projection depends on the type of Port being created, and the type of
        Ports specified in the tuple  (see `Projection_Table`).  For example, if the Port being created is an
        InputPort, and the Ports specified in the tuple are OutputPorts, then `MappingProjections
        <MappingProjection>` are used; if `ModulatorySignals <ModulatorySignal>` are specified, then the corresponding
        type of `ModulatoryProjections <ModulatoryProjection>` are created.  See Port subclasses for additional
        details and compatibility requirements.
      |
      .. _Port_ProjectionTuple:
      * `ProjectionTuple <Projection_ProjectionTuple>` -- a 4-item tuple that specifies one or more `Projections
        <Projection>` to or from other Port(s), along with a weight and/or exponent for each.

.. _Port_Projections:

*Projections*
~~~~~~~~~~~~~

When a Port is created, it can be assigned one or more `Projections <Projection>`, in either the **projections**
argument of its constructor, or a *PROJECTIONS* entry of a `Port specification dictionary
<Port_Specification_Dictionary>` (or a dictionary assigned to the **params** argument of the Port's constructor).
The following types of Projections can be specified for each type of Port:

    .. _Port_Projections_Table:

    .. table:: **Specifiable Projections for Port Types**
        :align: left

        +------------------+-------------------------------+-------------------------------------+
        | *Port Type*      | *PROJECTIONS* specification   | *Assigned to Attribute*             |
        +==================+===============================+=====================================+
        |`InputPort`       | `PathwayProjection(s)         | `path_afferents                     |
        |                  | <PathwayProjection>`          | <Port_Base.path_afferents>`         |
        |                  |                               |                                     |
        |                  | `ControlProjection(s)         | `mod_afferents                      |
        |                  | <ControlProjection>`          | <Port_Base.mod_afferents>`          |
        |                  |                               |                                     |
        |                  | `GatingProjection(s)          | `mod_afferents                      |
        |                  | <GatingProjection>`           | <Port_Base.mod_afferents>`          |
        +------------------+-------------------------------+-------------------------------------+
        |`ParameterPort`   | `ControlProjection(s)         | `mod_afferents                      |
        |                  | <ControlProjection>`          | <ParameterPort.mod_afferents>`      |
        +------------------+-------------------------------+-------------------------------------+
        |`OutputPort`      | `PathwayProjection(s)         | `efferents                          |
        |                  | <PathwayProjection>`          | <Port_Base.efferents>`              |
        |                  |                               |                                     |
        |                  | `ControlProjection(s)         | `mod_afferents                      |
        |                  | <ControlProjection>`          | <Port_Base.mod_afferents>`          |
        |                  |                               |                                     |
        |                  | `GatingProjection(s)          | `mod_afferents                      |
        |                  | <GatingProjection>`           | <Port_Base.mod_afferents>`          |
        +------------------+-------------------------------+-------------------------------------+
        |`ModulatorySignal`|  `ModulatoryProjection(s)     | `efferents                          |
        |                  |  <ModulatoryProjection>`      | <ModulatorySignal.efferents>`       |
        +------------------+-------------------------------+-------------------------------------+

Projections must be specified in a list.  Each entry must be either a `specification for a projection
<Projection_Specification>`, or for a `sender <Projection_Base.sender>` or `receiver <Projection_Base.receiver>` of
one, in which case the appropriate type of Projection is created.  A sender or receiver can be specified as a `Port
<Port>` or a `Mechanism <Mechanism>`. If a Mechanism is specified, its primary `InputPort <InputPort_Primary>` or
`OutputPort <OutputPort_Primary>`  is used, as appropriate.  When a sender or receiver is used to specify the
Projection, the type of Projection created is inferred from the Port and the type of sender or receiver specified,
as illustrated in the `examples <Port_Projections_Examples>` below.  Note that the Port must be `assigned to an
owner <Port_Creation>` in order to be functional, irrespective of whether any `Projections <Projection>` have been
assigned to it.


.. _Port_Deferred_Initialization:

*Deferred Initialization*
~~~~~~~~~~~~~~~~~~~~~~~~~

If a Port is created on its own, and its `owner <Port_Strucure_Owner>` Mechanism is specified, it is assigned to that
Mechanism; if its owner not specified, then its initialization is `deferred <Port_Deferred_Initialization>`.
Its initialization is completed automatically when it is assigned to an owner `Mechanism <Mechanism>` using the
owner's `add_ports <Mechanism_Base.add_ports>` method.  If the Port is not assigned to an owner, it will not be
functional (i.e., used during the execution of `Mechanisms <Mechanism_Execution>` and/or `Compositions
<Composition_Execution>`, irrespective of whether it has any `Projections <Projection>` assigned to it.


.. _Port_Structure:

Structure
---------

.. _Port_Strucure_Owner:

*Owner*
~~~~~~~

Every Port has an `owner <Port_Base.owner>`.  For `InputPorts <InputPort>` and `OutputPorts <OutputPort>`, the
owner must be a `Mechanism <Mechanism>`.  For `ParameterPorts <ParameterPort>` it can be a Mechanism or a
`PathwayProjection <PathwayProjection>`. For `ModulatorySignals <ModulatorySignal>`, it must be a `ModulatoryMechanism
<ModulatoryMechanism>`. When a Port is created as part of another Component, its `owner <Port_Base.owner>` is
assigned automatically to that Component.  It is also assigned automatically when the Port is assigned to a
`Mechanism <Mechanism>` using that Mechanism's `add_ports <Mechanism_Base.add_ports>` method.  Otherwise, it must be
specified explicitly in the **owner** argument of the constructor for the Port (in which case it is immediately
assigned to the specified Mechanism).  If the **owner** argument is not specified, the Port's initialization is
`deferred <Port_Deferred_Initialization>` until it has been assigned to an owner using the owner's `add_ports
<Mechanism_Base.add_ports>` method.

.. _Port_Structure_Projections:

*Projections*
~~~~~~~~~~~~~

Every Port has attributes that lists the `Projections <Projection>` it sends and/or receives.  These depend on the
type of Port, listed below (and shown in the `table <Port_Projections_Table>`):

.. table::  Port Projection Attributes
   :align: left

   ============================================ ============================================================
   *Attribute*                                  *Projection Type and Port(s)*
   ============================================ ============================================================
   `path_afferents <Port_Base.path_afferents>` `MappingProjections <MappingProjection>` to `InputPort`
   `mod_afferents <Port_Base.mod_afferents>`   `ModulatoryProjections <ModulatoryProjection>` to any Port
   `efferents <Port_Base.efferents>`           `MappingProjections <MappingProjection>` from `OutputPort`
   ============================================ ============================================================

In addition to these attributes, all of the Projections sent and received by a Port are listed in its `projections
<Port_Base.projections>` attribute.


.. _Port_Structure_Variable_Function_Value:

*Variable, Function and Value*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition, like all PsyNeuLink Components, it also has the three following core attributes:

    * `variable <Port_Base.variable>`:  for an `InputPort` and `ParameterPort`,
      the value of this is determined by the value(s) of the Projection(s) that it receives (and that are listed in
      its `path_afferents <Port_Base.path_afferents>` attribute).  For an `OutputPort`, it is the item of the owner
      Mechanism's `value <Mechanism_Base.value>` to which the OutputPort is assigned (specified by the OutputPort's
      `index <OutputPort_Index>` attribute.
    ..
    * `function <Port_Base.function>`:  for an `InputPort` this combines the values of the Projections that the
      Port receives (the default is `LinearCombination` that sums the values), under the potential influence of a
      `GatingSignal`;  for a `ParameterPort`, it determines the value of the associated parameter, under the potential
      influence of a `ControlSignal` (for a `Mechanism <Mechanism>`) or a `LearningSignal` (for a `MappingProjection`);
      for an OutputPort, it conveys the result  of the Mechanism's function to its `output_values
      <Mechanism_Base.output_values>` attribute, under the potential influence of a `GatingSignal`.  See
      `ModulatorySignals <ModulatorySignal_Structure>` and the `ModulatoryMechanism <ModulatoryMechanism>` associated
      with each type for a description of how they can be used to modulate the `function <Port_Base.function>` of a
      Port.
    ..
    * `value <Port_Base.value>`:  for an `InputPort` this is the combined value of the `PathwayProjections` it
      receives;  for a `ParameterPort`, this represents the value of the parameter that will be used by the Port's
      owner or its `function <Component.function>`; for an `OutputPort`, it is the item of the  owner Mechanism's
      `value <Mechanisms.value>` to which the OutputPort is assigned, possibly modified by its `assign
      <OutputPort_Assign>` attribute and/or a `GatingSignal`, and used as the `value <Projection_Base.value>` of
      the Projections listed in its `efferents <OutputPort.path_efferents>` attribute.

.. _Port_Modulation:

*Modulation*
~~~~~~~~~~~~

Every type of Port has a `mod_afferents <Port_Base.mod_afferents>` attribute, that lists the `ModulatoryProjections
<ModulatoryProjection>` it receives.  Each ModulatoryProjection comes from a `ModulatorySignal <ModulatorySignal>`
that specifies how it should modulate the Port's `value <Port_Base.value>` when the Port is updated (see
`ModulatorySignal_Modulation` and `ModulatorySignal_Anatomy_Figure`).  In most cases, a ModulatorySignal uses the
Port's `function <Port_Base.function>` to modulate its `value <Port_Base.value>`.  The function of every Port
assigns one of its parameters as its *ADDITIVE_PARAM* and another as its *MULTIPLICATIVE_PARAM*. The
`modulation <ModulatorySignal.modulation>` attribute of a ModulatorySignal determines which of these to modify when the
Port uses it `function <Port_Base.function>` to calculate its `value  <Port_Base.value>`.  However, the
ModulatorySignal can also be configured to override the Port's `value <Port_Base.value>` (i.e., assign it directly),
or to disable modulation, using either the keyword *OVERRIDE* or *DSIABLE*, respectively, to specify the value for its
`modulation <ModulatorySignal.modulation>` attribute (see `ModulatorySignal_Modulation` for a more detailed discussion).

.. _Port_Execution:

Execution
---------

Ports cannot be executed.  They are updated when the Component to which they belong is executed.  InputPorts and
ParameterPorts belonging to a Mechanism are updated before the Mechanism's function is called.  OutputPorts are
updated after the Mechanism's function is called.  When a Port is updated, it executes any Projections that project
to it (listed in its `all_afferents <Port_Base.all_afferents>` attribute.  It uses the values it receives from any
`PathWayProjections` (listed in its `path_afferents` attribute) as the variable for its
`function <Port_Base.function>`. It then executes all of the ModulatoryProjections it receives.  Different
ModulatorySignals may call for different forms of modulation (see `ModulatorySignal_Modulation`).  Accordingly,
it separately sums the values specified by any ModulatorySignals for the *MULTIPLICATIVE_PARAM* of its
`function <Port_Base.function>`, and similarly for the *ADDITIVE_PARAM*.  It then applies the summed value for each
to the corresponding parameter of its `function <Port_Base.function>`.  If any of the ModulatorySignals specifies
*OVERRIDE*, then the value of that ModulatorySignal is used as the Port's `value <Port_Base.value>`. Finally,
the Port calls its `function <Port_Base.function>` to determine its `value <Port_Base.value>`.

.. note::
   The change in the value of a `Port <Port>` does not occur until the Mechanism to which the Port belongs is next
   executed; This conforms to a "lazy evaluation" protocol (see `Lazy Evaluation <Component_Lazy_Updating>` for an
   explanation of "lazy" updating).

.. _Port_Examples:

Examples
========

.. _Port_Constructor_Examples:

Usually, Ports are created automatically by the Mechanism to which they belong.  For example, creating a
TransferMechanism::

    my_mech = pnl.TransferMechanism()

automatically creates an InputPort, ParameterPorts for its parameters, including the `slope <Linear.slope>` and
`intercept <Linear.intercept>` parameters of its `Linear` `Function <Function>` (its default `function
<Mechanism_Base.function>`), and an OutputPort (named *RESULT*)::

    print(my_mech.input_ports)
    > [(InputPort InputPort-0)]
    print(my_mech.parameter_ports)
    > [(ParameterPort intercept), (ParameterPort slope), (ParameterPort noise), (ParameterPort integration_rate)]
    print(my_mech.output_ports)
    > [(OutputPort RESULT)]

.. _Port_Constructor_Argument_Examples:

*Using the* **input_ports** *argument of a Mechanism constructor.*

When Ports are specified explicitly, it is usually in an argument of the constructor for the Mechanism to which they
belong.  For example, the following specifies that ``my_mech`` should have an InputPort named 'MY INPUT`::

    my_mech = pnl.TransferMechanism(input_ports=['MY INPUT'])
    print(my_mech.input_ports)
    > [(InputPort 'MY INPUT')]

The InputPort was specified by a string (for its name) in the **input_ports** argument.  It can also be specified in
a variety of other ways, as described `above <Port_Specification>` and illustrated in the examples below.
Note that when one or more Ports is specified in the argument of a Mechanism's constructor, it replaces any defaults
Ports created by the Mechanism when none are specified (see `note <Mechanism_Default_Port_Suppression_Note>`.

.. _port_value_Spec_Example:

For example, the following specifies the InputPort by a value to use as its `default_variable
<InputPort.default_variable>` attribute::

    my_mech = pnl.TransferMechanism(input_ports=[[0,0])

The value is also used to format the InputPort's `value <InputPort.value>`, as well as the first (and, in this case,
only) item of the Mechanism's `variable <Mechanism_Base.variable>` (i.e., the one to which the InputPort is
assigned), as show below::

    print(my_mech.input_ports[0].variable)
    > [0 0]
    print (my_mech.input_port.value)
    > [ 0.  0.]
    print (my_mech.variable)
    > [[0 0]]

Note that in the first print port, the InputPort was referenced as the first one in the `input_ports
<Mechanism_Base.input_ports>` attribute of ``my_mech``;  the second print port references it directly,
as the `primary InputPort <Input_port.primary>` of ``my_mech``, using its `input_port <Mechanism_Base.input_port>`
attribute (note the singular).

.. _Port_Multiple_InputSates_Example:

*Multiple InputPorts*

The **input_ports** argument can also be used to create more than one InputPort::

    my_mech = pnl.TransferMechanism(input_ports=['MY FIRST INPUT', 'MY SECOND INPUT'])
    print(my_mech.input_ports)
    > [(InputPort MY FIRST INPUT), (InputPort MY SECOND INPUT)]

Here, the print statement uses the `input_ports <Mechanism_Base.input_ports>` attribute, since there is now more
than one InputPort.  OutputPorts can be specified in a similar way, using the **output_ports** argument.

    .. note::
        Although InputPorts and OutputPorts can be specified in a Mechanism's constructor, ParameterPorts cannot;
        those are created automatically when the Mechanism is created, for each of its `user configurable parameters
        <Component_User_Params>`  and those of its `function <Mechanism_Base.function>`.  However, the `value
        <ParameterPort.value>` can be specified when the Mechanism is created, or its `function
        <Mechanism_Base.function>` is assigned, and can be accessed and subsequently modified, as described under
        `ParameterPort_Specification>`.

.. _Port_Standard_OutputPorts_Example:

*OutputPorts*

The following example specifies two OutputPorts for ``my_mech``, using its `Standard OutputPorts
<OutputPort_Standard>`::

    my_mech = pnl.TransferMechanism(output_ports=['RESULT', 'MEAN'])

As with InputPorts, specification of OutputPorts in the **output_ports** argument suppresses the creation of any
default OutputPorts that would have been created if no OutputPorts were specified (see `note
<Mechanism_Default_Port_Suppression_Note>` above).  For example, TransferMechanisms create a *RESULT* OutputPort
by default, that contains the result of their `function <OutputPort.function>`.  This default behavior is suppressed
by any specifications in its **output_ports** argument.  Therefore, to retain a *RESULT* OutputPort,
it must be included in the **output_ports** argument along with any others that are specified, as in the example
above.  If the name of a specified OutputPort matches the name of a Standard OutputPort <OutputPort_Standard>` for
the type of Mechanism, then that is used (as is the case for both of the OutputPorts specified for the
`TransferMechanism` in the example above); otherwise, a new OutputPort is created.

.. _Port_Specification_Dictionary_Examples:

*Port specification dictionary*

Ports can be specified in greater detail using a `Port specification dictionary
<Port_Specification_Dictionary>`. In the example below, this is used to specify the variable and name of an
InputPort::

    my_mech = pnl.TransferMechanism(input_ports=[{PORT_TYPE: InputPort,
                                                   NAME: 'MY INPUT',
                                                   VARIABLE: [0,0]})

The *PORT_TYPE* entry is included here for completeness, but is not actually needed when the Port specification
dicationary is used in **input_ports** or **output_ports** argument of a Mechanism, since the Port's type
is clearly determined by the context of the specification;  however, where that is not clear, then the *PORT_TYPE*
entry must be included.

.. _Port_Projections_Examples:

*Projections*

A Port specification dictionary can also be used to specify projections to or from the Port, also in
a number of different ways.  The most straightforward is to include them in a *PROJECTIONS* entry.  For example, the
following specifies that the InputPort of ``my_mech`` receive two Projections,  one from ``source_mech_1`` and another
from ``source_mech_2``, and that its OutputPort send one to ``destination_mech``::

    source_mech_1 = pnl.TransferMechanism(name='SOURCE_1')
    source_mech_2 = pnl.TransferMechanism(name='SOURCE_2')
    destination_mech = pnl.TransferMechanism(name='DEST')
    my_mech = pnl.TransferMechanism(name='MY_MECH',
                                    input_ports=[{pnl.NAME: 'MY INPUT',
                                                   pnl.PROJECTIONS:[source_mech_1, source_mech_2]}],
                                    output_ports=[{pnl.NAME: 'RESULT',
                                                    pnl.PROJECTIONS:[destination_mech]}])

    # Print names of the Projections:
    for projection in my_mech.input_port.path_afferents:
        print(projection.name)
    > MappingProjection from SOURCE_1[RESULT] to MY_MECH[MY INPUT]
    > MappingProjection from SOURCE_2[RESULT] to MY_MECH[MY INPUT]
    for projection in my_mech.output_port.efferents:
        print(projection.name)
    > MappingProjection from MY_MECH[RESULT] to DEST[InputPort]


A *PROJECTIONS* entry can contain any of the forms used to `specify a Projection <Projection_Specification>`.
Here, Mechanisms are used, which creates Projections from the `primary InputPort <InputPort_Primary>` of
``source_mech``, and to the `primary OutputPort <OutputPort_Primary>` of ``destination_mech``.  Note that
MappingProjections are created, since the Projections specified are between InputPorts and OutputPorts.
`ModulatoryProjections` can also be specified in a similar way.  The following creates a `GatingMechanism`, and
specifies that the InputPort of ``my_mech`` should receive a `GatingProjection` from it::

    my_gating_mech = pnl.GatingMechanism()
    my_mech = pnl.TransferMechanism(name='MY_MECH',
                                    input_ports=[{pnl.NAME: 'MY INPUT',
                                                   pnl.PROJECTIONS:[my_gating_mech]}])


.. _Port_Modulatory_Projections_Examples:

Conversely, ModulatoryProjections can also be specified from a Mechanism to one or more Ports that it modulates.  In
the following example, a `ControlMechanism` is created that sends `ControlProjections <ControlProjection>` to the
`drift_rate <DriftDiffusionAnalytical.drift_rate>` and `threshold <DriftDiffusionAnalytical.threshold>`
ParameterPorts of a `DDM` Mechanism::

    my_mech = pnl.DDM(name='MY DDM')
    my_ctl_mech = pnl.ControlMechanism(control_signals=[{pnl.NAME: 'MY DDM DRIFT RATE AND THREHOLD CONTROL SIGNAL',
                                                         pnl.PROJECTIONS: [my_mech.parameter_ports[pnl.DRIFT_RATE],
                                                                           my_mech.parameter_ports[pnl.THRESHOLD]]}])
    # Print ControlSignals and their ControlProjections
    for control_signal in my_ctl_mech.control_signals:
        print(control_signal.name)
        for control_projection in control_signal.efferents:
            print("\t{}: {}".format(control_projection.receiver.owner.name, control_projection.receiver))
    > MY DDM DRIFT RATE AND THREHOLD CONTROL SIGNAL
    >     MY DDM: (ParameterPort drift_rate)
    >     MY DDM: (ParameterPort threshold)

Note that a ControlMechanism uses a **control_signals** argument in place of an **output_ports** argument (since it
uses `ControlSignal <ControlSignals>` for its `OutputPorts <OutputPort>`.  In the example above,
both ControlProjections are assigned to a single ControlSignal.  However, they could each be assigned to their own by
specifying them in separate itesm of the **control_signals** argument::

    my_mech = pnl.DDM(name='MY DDM')
    my_ctl_mech = pnl.ControlMechanism(control_signals=[{pnl.NAME: 'DRIFT RATE CONTROL SIGNAL',
                                                         pnl.PROJECTIONS: [my_mech.parameter_ports[pnl.DRIFT_RATE]]},
                                                        {pnl.NAME: 'THRESHOLD RATE CONTROL SIGNAL',
                                                         pnl.PROJECTIONS: [my_mech.parameter_ports[pnl.THRESHOLD]]}])
    # Print ControlSignals and their ControlProjections...
    > DRIFT RATE CONTROL SIGNAL
    >     MY DDM: (ParameterPort drift_rate)
    > THRESHOLD RATE CONTROL SIGNAL
    >     MY DDM: (ParameterPort threshold)

Specifying Projections in a Port specification dictionary affords flexibility -- for example, naming the Port
and/or specifying other attributes.  However, if this is not necessary, the Projections can be used to specify
Ports directly.  For example, the following, which is much simpler, produces the same result as the previous
example (sans the custom name; though as the printout below shows, the default names are usually pretty clear)::

    my_ctl_mech = pnl.ControlMechanism(control_signals=[my_mech.parameter_ports[pnl.DRIFT_RATE],
                                                        my_mech.parameter_ports[pnl.THRESHOLD]])
    # Print ControlSignals and their ControlProjections...
    > MY DDM drift_rate ControlSignal
    >    MY DDM: (ParameterPort drift_rate)
    > MY DDM threshold ControlSignal
    >    MY DDM: (ParameterPort threshold)

.. _Port_Port_Name_Entry_Example:

*Convenience formats*

There are two convenience formats for specifying Ports and their Projections in a Port specification
dictionary.  The `first <Port_Port_Name_Entry>` is to use the name of the Port as the key for its entry,
and then a list of , as in the following example::

    source_mech_1 = pnl.TransferMechanism()
    source_mech_2 = pnl.TransferMechanism()
    destination_mech = pnl.TransferMechanism()
    my_mech_C = pnl.TransferMechanism(input_ports=[{'MY INPUT':[source_mech_1, source_mech_2]}],
                                      output_ports=[{'RESULT':[destination_mech]}])

This produces the same result as the first example under `Port specification dictionary <Port_Projections_Examples>`
above, but it is simpler and easier to read.

The second convenience format is used to specify one or more Projections to/from the Ports of a single Mechanism
by their name.  It uses the keyword *MECHANISM* to specify the Mechanism, coupled with a Port-specific entry to
specify Projections to its Ports.  This can be useful when a Mechanism must send Projections to several Ports
of another Mechanism, such as a ControlMechanism that sends ControlProjections to several parameters of a
given Mechanism, as in the following example::

    my_mech = pnl.DDM(name='MY DDM')
    my_ctl_mech = pnl.ControlMechanism(control_signals=[{pnl.MECHANISM: my_mech,
                                                         pnl.PARAMETER_PORTS: [pnl.DRIFT_RATE, pnl.THRESHOLD]}])

This produces the same result as the `earlier example <Port_Modulatory_Projections_Examples>` of ControlProjections,
once again in a simpler and easier to read form.  However, it be used only to specify Projections for a Port to or
from the Ports of a single Mechanism;  Projections involving other Mechanisms must be assigned to other Ports.

.. _Port_Create_Port_Examples:

*Create and then assign a port*

Finally, a Port can be created directly using its constructor, and then assigned to a Mechanism.
The following creates an InputPort ``my_input_port`` with a `MappingProjection` to it from the
`primary OutputPort <OutputPort_Primary>` of ``mech_A`` and assigns it to ``mech_B``::

    mech_A = pnl.TransferMechanism()
    my_input_port = pnl.InputPort(name='MY INPUTPORT',
                                    projections=[mech_A])
    mech_B = pnl.TransferMechanism(input_ports=[my_input_port])
    print(mech_B.input_ports)
    > [(InputPort MY INPUTPORT)]

The InputPort ``my_input_port`` could also have been assigned to ``mech_B`` in one of two other ways:
by explicity adding it using ``mech_B``\\'s `add_ports <Mechanism_Base.add_ports>` method::

    mech_A = pnl.TransferMechanism()
    my_input_port = pnl.InputPort(name='MY INPUTPORT',
                                    projections=[mech_A])
    mech_B = pnl.TransferMechanism()
    mech_B.add_ports([my_input_port])

or by constructing it after ``mech_B`` and assigning ``mech_B`` as its owner::

    mech_A = pnl.TransferMechanism()
    mech_B = pnl.TransferMechanism()
    my_input_port = pnl.InputPort(name='MY INPUTPORT',
                                    owner=mech_B,
                                    projections=[mech_A])

Note that, in both cases, adding the InputPort to ``mech_B`` does not replace its the default InputPort generated
when it was created, as shown by printing the `input_ports <Mechanism_Base.input_ports>` for ``mech_B``::

    print(mech_B.input_ports)
    > [(InputPort InputPort-0), (InputPort MY INPUTPORT)]
    > [(InputPort InputPort-0), (InputPort MY INPUTPORT)]

As a consequence, ``my_input_port`` is  **not** the `primary InputPort <InputPort_Primary>` for ``mech_B`` (i.e.,
input_ports[0]), but rather its second InputPort (input_ports[1]). This is differs from specifying the InputPort
as part of the constructor for the Mechanism, which suppresses generation of the default InputPort,
as in the first example above (see `note <Mechanism_Default_Port_Suppression_Note>`).

COMMENT:

*** ??ADD THESE TO EXAMPLES, HERE OR IN Projection??

    def test_mapping_projection_with_mech_and_port_Name_specs(self):
         R1 = pnl.TransferMechanism(output_ports=['OUTPUT_1', 'OUTPUT_2'])
         R2 = pnl.TransferMechanism(default_variable=[[0],[0]],
                                    input_ports=['INPUT_1', 'INPUT_2'])
         T = pnl.TransferMechanism(input_ports=[{pnl.MECHANISM: R1,
                                                  pnl.OUTPUT_PORTS: ['OUTPUT_1', 'OUTPUT_2']}],
                                   output_ports=[{pnl.MECHANISM:R2,
                                                   pnl.INPUT_PORTS: ['INPUT_1', 'INPUT_2']}])

   def test_transfer_mech_input_ports_specification_dict_spec(self):
        R1 = TransferMechanism(output_ports=['FIRST', 'SECOND'])
        T = TransferMechanism(default_variable=[[0],[0]],
                                      input_ports=[{NAME: 'FROM DECISION',
                                                     PROJECTIONS: [R1.output_ports['FIRST']]},
                                                    {NAME: 'FROM RESPONSE_TIME',
                                                     PROJECTIONS: R1.output_ports['SECOND']}])

   def test_transfer_mech_input_ports_projection_in_specification_dict_spec(self):
        R1 = TransferMechanism(output_ports=['FIRST', 'SECOND'])
        T = TransferMechanism(input_ports=[{NAME: 'My InputPort with Two Projections',
                                             PROJECTIONS:[R1.output_ports['FIRST'],
                                                          R1.output_ports['SECOND']]}])

    def test_transfer_mech_input_ports_mech_output_port_in_specification_dict_spec(self):
        R1 = TransferMechanism(output_ports=['FIRST', 'SECOND'])
        T = TransferMechanism(input_ports=[{MECHANISM: R1,
                                             OUTPUT_PORTS: ['FIRST', 'SECOND']}])
        assert len(T.input_ports)==1
        for input_port in T.input_ports:
            for projection in input_port.path_afferents:
                assert projection.sender.owner is R1

creates a `GatingSignal` with
`GatingProjections <GatingProjection>` to ``mech_B`` and ``mech_C``, and assigns it to ``my_gating_mech``::

    my_gating_signal = pnl.GatingSignal(projections=[mech_B, mech_C])
    my_gating_mech = GatingMechanism(gating_signals=[my_gating_signal]

The `GatingMechanism` created will gate the `primary InputPorts <InputPort_Primary>` of ``mech_B`` and ``mech_C``.

The following creates

   def test_multiple_modulatory_projections_with_mech_and_port_Name_specs(self):

        M = pnl.DDM(name='MY DDM')
        C = pnl.ControlMechanism(control_signals=[{pnl.MECHANISM: M,
                                                   pnl.PARAMETER_PORTS: [pnl.DRIFT_RATE, pnl.THRESHOLD]}])
        G = pnl.GatingMechanism(gating_signals=[{pnl.MECHANISM: M,
                                                 pnl.OUTPUT_PORTS: [pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME]}])


        M = pnl.DDM(name='MY DDM')
        C = pnl.ControlMechanism(control_signals=[{'DECISION_CONTROL':[M.parameter_ports[pnl.DRIFT_RATE],
                                                                       M.parameter_ports[pnl.THRESHOLD]]}])
        G = pnl.GatingMechanism(gating_signals=[{'DDM_OUTPUT_GATE':[M.output_ports[pnl.DECISION_VARIABLE],
                                                                    M.output_ports[pnl.RESPONSE_TIME]]}])

COMMENT

.. _Port_Class_Reference:

Class Reference
---------------

"""

import abc
import inspect
import itertools
import numbers
import sys
import types
import warnings

from collections.abc import Iterable
from collections import defaultdict

import numpy as np
import typecheck as tc

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.component import ComponentError, DefaultsFlexibility, component_keywords
from psyneulink.core.components.functions.combinationfunctions import CombinationFunction, LinearCombination
from psyneulink.core.components.functions.function import Function, get_param_value_for_keyword, is_function_type
from psyneulink.core.components.functions.transferfunctions import Linear
from psyneulink.core.components.shellclasses import Mechanism, Projection, Port
from psyneulink.core.globals.context import Context, ContextFlags
from psyneulink.core.globals.keywords import \
    ADDITIVE, ADDITIVE_PARAM, AUTO_ASSIGN_MATRIX, \
    CONTEXT, CONTROL_PROJECTION_PARAMS, CONTROL_SIGNAL_SPECS, DEFERRED_INITIALIZATION, DISABLE, EXPONENT, \
    FUNCTION, FUNCTION_PARAMS, GATING_PROJECTION_PARAMS, GATING_SIGNAL_SPECS, INPUT_PORTS, \
    LEARNING_PROJECTION_PARAMS, LEARNING_SIGNAL_SPECS, \
    MATRIX, MECHANISM, MODULATORY_PROJECTION, MODULATORY_PROJECTIONS, MODULATORY_SIGNAL, \
    MULTIPLICATIVE, MULTIPLICATIVE_PARAM, \
    NAME, OUTPUT_PORTS, OVERRIDE, OWNER, \
    PARAMETER_PORTS, PARAMS, PATHWAY_PROJECTIONS, PREFS_ARG, \
    PROJECTION_DIRECTION, PROJECTIONS, PROJECTION_PARAMS, PROJECTION_TYPE, \
    RECEIVER, REFERENCE_VALUE, REFERENCE_VALUE_NAME, SENDER, STANDARD_OUTPUT_PORTS, \
    PORT, PORT_COMPONENT_CATEGORY, PORT_CONTEXT, Port_Name, port_params, PORT_PREFS, PORT_TYPE, port_value, \
    VALUE, VARIABLE, WEIGHT
from psyneulink.core.globals.parameters import Parameter, ParameterAlias
from psyneulink.core.globals.preferences.basepreferenceset import VERBOSE_PREF
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.registry import register_category
from psyneulink.core.globals.socket import ConnectionInfo
from psyneulink.core.globals.utilities import \
    ContentAddressableList, convert_to_np_array, get_args, is_value_spec, iscompatible, \
    MODULATION_OVERRIDE, type_match

__all__ = [
    'Port_Base', 'port_keywords', 'port_type_keywords', 'PortError', 'PortRegistry', 'PORT_SPEC'
]

port_keywords = component_keywords.copy()
port_keywords.update({MECHANISM,
                      PORT_TYPE,
                      port_value,
                      port_params,
                      PATHWAY_PROJECTIONS,
                      MODULATORY_PROJECTIONS,
                      PROJECTION_TYPE,
                      PROJECTION_PARAMS,
                      LEARNING_PROJECTION_PARAMS,
                      LEARNING_SIGNAL_SPECS,
                      CONTROL_PROJECTION_PARAMS,
                      CONTROL_SIGNAL_SPECS,
                      GATING_PROJECTION_PARAMS,
                      GATING_SIGNAL_SPECS})

port_type_keywords = {PORT_TYPE}

PORT_SPECIFIC_PARAMS = 'PORT_SPECIFIC_PARAMS'
PROJECTION_SPECIFIC_PARAMS = 'PROJECTION_SPECIFIC_PARAMS'


STANDARD_PORT_ARGS = {PORT_TYPE, OWNER, REFERENCE_VALUE, VARIABLE, NAME, PARAMS, PREFS_ARG}
PORT_SPEC = 'port_spec'
REMOVE_PORTS = 'REMOVE_PORTS'

def _is_port_class(spec):
    if inspect.isclass(spec) and issubclass(spec, Port):
        return True
    return False


# Note:  This is created only for assignment of default projection types for each Port subclass (see .__init__.py)
#        Individual portRegistries (used for naming) are created for each Mechanism
PortRegistry = {}

class PortError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


    def __str__(self):
        return repr(self.error_value)


# DOCUMENT:  INSTANTATION CREATES AN ATTIRBUTE ON THE OWNER MECHANISM WITH THE PORT'S NAME + VALUE_SUFFIX
#            THAT IS UPDATED BY THE PORT'S value setter METHOD (USED BY LOGGING OF MECHANISM ENTRIES)
class Port_Base(Port):
    """
    Port_Base(     \
        owner=None \
        )

    Base class for Port.

    The arguments below can be used in the constructor for any subclass of Port.
    See `Component <Component_Class_Reference>` for additional arguments and attributes.

    .. note::
       Port is an abstract class and should *never* be instantiated by a call to its constructor. It should be created
       by calling the constructor for a `subclass <Port_Subtypes>`, or by using any of the other methods for `specifying
       a Port <Port_Specification>`.

    .. technical_note::

        PortRegistry
        -------------
            Used by .__init__.py to assign default Projection types to each Port subclass

            .. note::
              All Ports that belong to a given owner are registered in the owner's _portRegistry, which maintains a
              dict for each Port type that it uses, a count for all instances of that type, and a dictionary of those
              instances;  **none** of these are registered in the PortRegistry This. is so that the same name can be
              used for instances of a Port type by different owners without adding index suffixes for that name across
              owners, while still indexing multiple uses of the same base name within an owner

    Arguments
    ---------

    owner : Mechanism : default None
        the Mechanism to which the Port belongs;  if it is not specified or determinable from the context in which
        the Port is created, the Port's initialization is `deferred <Port_Deferred_Initialization>`.

    Attributes
    ----------

    variable : number, list or np.ndarray
        the Port's input, provided as the `variable <Port_Base.variable>` to its `function <Port_Base.function>`.

    owner : Mechanism or Projection
        object to which the Port belongs (see `Port_Strucure_Owner` for additional details).

    base_value : number, list or np.ndarray
        value with which the Port was initialized.

    all_afferents : Optional[List[Projection]]
        list of all Projections received by the Port (i.e., for which it is a `receiver <Projection_Base.receiver>`.

    path_afferents : Optional[List[Projection]]
        list of all `PathwayProjections <PathwayProjection>` received by the Port (i.e., for which it is the
        receiver <Projection_Base.receiver>` (note:  only `InputPorts <InputPort>` have path_afferents;  the list is
        empty for other types of Ports).

    mod_afferents : Optional[List[GatingProjection]]
        list of all `ModulatoryProjections <ModulatoryProjection>` received by the Port.

    projections : List[Projection]
        list of all of the `Projections <Projection>` sent or received by the Port.

    efferents : Optional[List[Projection]]
        list of outgoing Projections from the Port (i.e., for which is a `sender <Projection_Base.sender>`;
        note:  only `OutputPorts <OutputPort>`, and members of its `ModulatoryProjection <ModulatoryProjection>`
        subclass (`LearningProjection, ControlProjection and GatingProjection) have efferents;  the list is empty for
        InputPorts and ParameterPorts.

    function : TransferFunction : default determined by type
        used to determine the Port's `value <Port_Base.value>` from the `value <Projection_Base.value>` of the
        `Projection(s) <Projection>` it receives;  the parameters that the TransferFunction identifies as *ADDITIVE*
        and *MULTIPLICATIVE* are subject to modulation by a `ModulatorySignal <ModulatorySignal>`.

    value : number, list or np.ndarray
        current value of the Port.

    name : str
        the name of the Port. If the Port's `initialization has been deferred <Port_Deferred_Initialization>`, it is
        assigned a temporary name (indicating its deferred initialization status) until initialization is completed,
        at which time it is assigned its designated name.  If that is the name of an existing Port, it is appended
        with an indexed suffix, incremented for each Port with the same base name (see `Registry_Naming`). If the name
        is not  specified in the **name** argument of its constructor, a default name is assigned by the subclass
        (see subclass for details).

        .. _Port_Naming_Note:

        .. note::
            Unlike other PsyNeuLink Components, Ports names are "scoped" within a Mechanism, meaning that Ports with
            the same name are permitted in different Mechanisms.  However, they are *not* permitted in the same
            Mechanism: Ports within a Mechanism with the same base name are appended an index in the order of their
            creation).

    full_name : str
        the name of the Port with its owner if that is assigned: <owner.name>[<self.name>] if owner is not None;
        otherwise same as `name <Port.name>`.

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the Port; if it is not specified in the **prefs** argument of the constructor,
        a default is assigned using `classPreferences` defined in __init__.py (see `Preferences` for details).

    """

    componentCategory = PORT_COMPONENT_CATEGORY
    className = PORT
    suffix = " " + className
    paramsType = None

    class Parameters(Port.Parameters):
        """
            Attributes
            ----------

                function
                    see `function <Port_Base.function>`

                    :default value: `Linear`
                    :type: `Function`

                projections
                    see `projections <Port_Base.projections>`

                    :default value: None
                    :type:

                require_projection_in_composition
                    specifies whether the InputPort requires a projection when instantiated in a Composition;
                    if so, but none exists, a warning is issued.

                    :default value: True
                    :type: ``bool``
                    :read only: True
        """
        function = Parameter(Linear, stateful=False, loggable=False)
        projections = Parameter(
            None,
            structural=True,
            stateful=False,
            loggable=False
        )
        require_projection_in_composition = Parameter(True, stateful=False, loggable=False, read_only=True, pnl_internal=True)

    portAttributes = {FUNCTION, FUNCTION_PARAMS, PROJECTIONS}

    registry = PortRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY

    @tc.typecheck
    @abc.abstractmethod
    def __init__(self,
                 owner:tc.any(Mechanism, Projection),
                 variable=None,
                 size=None,
                 projections=None,
                 function=None,
                 params=None,
                 name=None,
                 prefs=None,
                 context=None,
                 **kwargs):
        """Initialize subclass that computes and represents the value of a particular Port of a Mechanism

        This is used by subclasses to implement the InputPort(s), OutputPort(s), and ParameterPort(s) of a Mechanism.

        Arguments:
            - owner (Mechanism):
                 Mechanism with which Port is associated (default: NotImplemented)
                 this argument is required, as can't instantiate a Port without an owning Mechanism
            - variable (value): value of the Port:
                must be list or tuple of numbers, or a number (in which case it will be converted to a single-item list)
                must match input and output of Port's _update method, and any sending or receiving projections
            - size (int or array/list of ints):
                Sets variable to be array(s) of zeros, if **variable** is not specified as an argument;
                if **variable** is specified, it takes precedence over the specification of **size**.
            - params (dict):
                + if absent, implements default Port determined by PROJECTION_TYPE param
                + if dict, can have the following entries:
                    + PROJECTIONS:<Projection object, Projection class, dict, or list of either or both>
                        if absent, no projections will be created
                        if dict, must contain entries specifying a projection:
                            + PROJECTION_TYPE:<Projection class> - must be a subclass of Projection
                            + PROJECTION_PARAMS:<dict> - must be dict of params for PROJECTION_TYPE
            - name (str): string with name of Port (default: name of owner + suffix + instanceIndex)
            - prefs (dict): dictionary containing preferences (default: Prefs.DEFAULTS)
            - context (str)
            - **kwargs (dict): dictionary of arguments using the following keywords for each of the above kwargs:
                # port_params is not handled here like the others are
                + port_params = params
                + Port_Name = name
                + PORT_PREFS = prefs
                + PORT_CONTEXT = context
                NOTES:
                    * these are used for dictionary specification of a Port in param declarations
                    * they take precedence over arguments specified directly in the call to __init__()
        """
        if kwargs:
            try:
                name = kwargs[Port_Name]
            except (KeyError, NameError):
                pass
            try:
                prefs = kwargs[PORT_PREFS]
            except (KeyError, NameError):
                pass
            try:
                context = kwargs[PORT_CONTEXT]
            except (KeyError, NameError):
                pass

        # Enforce that subclass must implement and _execute method
        if not hasattr(self, '_execute'):
            raise PortError("{}, as a subclass of {}, must implement an _execute() method".
                             format(self.__class__.__name__, PORT))

        self.owner = owner

        # If name is not specified, assign default name
        if name is not None and DEFERRED_INITIALIZATION in name:
            name = self._assign_default_port_Name(context=context)


        # Register Port with PortRegistry of owner (Mechanism to which the Port is being assigned)
        register_category(entry=self,
                          base_class=Port_Base,
                          name=name,
                          registry=owner._portRegistry,
                          # sub_group_attr='owner',
                          context=context)

        # VALIDATE VARIABLE, PARAM_SPECS, AND INSTANTIATE self.function
        super(Port_Base, self).__init__(
            default_variable=variable,
            size=size,
            function=function,
            projections=projections,
            param_defaults=params,
            name=name,
            prefs=prefs,
            **kwargs
        )

        self.path_afferents = []
        self.mod_afferents = []

        # IMPLEMENTATION NOTE:  MOVE TO COMPOSITION ONCE THAT IS IMPLEMENTED
        # INSTANTIATE PROJECTIONS SPECIFIED IN projections ARG OR params[PROJECTIONS:<>]
        if self.projections is not None:
            self._instantiate_projections(self.projections, context=context)
        else:
            # No projections specified, so none will be created here
            # IMPLEMENTATION NOTE:  This is where a default projection would be implemented
            #                       if params = NotImplemented or there is no param[PROJECTIONS]
            pass

        self.projections = self.path_afferents + self.mod_afferents + self.efferents

        if context.source == ContextFlags.COMMAND_LINE:
            owner.add_ports([self])

    def _handle_size(self, size, variable):
        """Overwrites the parent method in Component.py, because the variable of a Port
            is generally 1D, rather than 2D as in the case of Mechanisms
        """
        if size is not NotImplemented:

            def checkAndCastInt(x):
                if not isinstance(x, numbers.Number):
                    raise PortError("Size ({}) is not a number.".format(x))
                if x < 1:
                    raise PortError("Size ({}) is not a positive number.".format(x))
                try:
                    int_x = int(x)
                except:
                    raise PortError(
                        "Failed to convert size argument ({}) for {} {} to an integer. For Ports, size "
                        "should be a number, which is an integer or can be converted to integer.".
                        format(x, type(self), self.name))
                if int_x != x:
                    if hasattr(self, 'prefs') and hasattr(self.prefs, VERBOSE_PREF) and self.prefs.verbosePref:
                        warnings.warn("When size ({}) was cast to integer, its value changed to {}.".format(x, int_x))
                return int_x

            # region Convert variable to a 1D array, cast size to an integer
            if size is not None:
                size = checkAndCastInt(size)
            try:
                if variable is not None:
                    variable = np.atleast_1d(variable)
            except:
                raise PortError("Failed to convert variable (of type {}) to a 1D array.".format(type(variable)))
            # endregion

            # region if variable is None and size is not None, make variable a 1D array of zeros of length = size
            if variable is None and size is not None:
                try:
                    variable = np.zeros(size)
                except:
                    raise ComponentError("variable (perhaps default_variable) was not specified, but PsyNeuLink "
                                         "was unable to infer variable from the size argument, {}. size should be"
                                         " an integer or able to be converted to an integer. Either size or "
                                         "variable must be specified.".format(size))
            #endregion

            if variable is not None and size is not None:  # try tossing this "if" check
                # If they conflict, raise exception
                if size != len(variable):
                    raise PortError("The size arg of {} ({}) conflicts with the length of its variable arg ({})".
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

        variable = super(Port, self)._validate_variable(variable, context)

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
            #     ParameterPort, projection, 2-item tuple or value
        """
        # FIX: PROJECTION_REFACTOR
        #      SHOULD ADD CHECK THAT PROJECTION_TYPE IS CONSISTENT WITH TYPE SPECIFIED BY THE
        #      RECEIVER/SENDER SOCKET SPECIFICATIONS OF CORRESPONDING PROJECTION TYPES (FOR API)

        if PROJECTIONS in request_set and request_set[PROJECTIONS] is not None:
            # if projection specification is an object or class reference, needs to be wrapped in a list
            #    to be consistent with defaults and for consistency of treatment below
            projections = request_set[PROJECTIONS]
            if not isinstance(projections, list):
                projections = [projections]
                request_set[PROJECTIONS] = projections
        else:
            # If no projections, ignore (none will be created)
            projections = None

        super(Port, self)._validate_params(request_set, target_set, context=context)

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

    def _instantiate_function(self, function, function_params=None, context=None):

        var_is_matrix = False
        # If variable is a 2d array or matrix (e.g., for the MATRIX ParameterPort of a MappingProjection),
        #     it needs to be embedded in a list so that it is properly handled by LinearCombination
        #     (i.e., solo matrix is returned intact, rather than treated as arrays to be combined);
        # Notes:
        #     * this is not a problem when LinearCombination is called in port._update(), since that puts
        #         projection values in a list before calling LinearCombination to combine them
        #     * it is removed from the list below, after calling _instantiate_function
        # FIX: UPDATE WITH MODULATION_MODS REMOVE THE FOLLOWING COMMENT:
        #     * no change is made to PARAMETER_MODULATION_FUNCTION here (matrices may be multiplied or added)
        #         (that is handled by the individual Port subclasses (e.g., ADD is enforced for MATRIX ParameterPort)
        if (
            (
                (inspect.isclass(function) and issubclass(function, LinearCombination))
                or isinstance(function, LinearCombination)
            )
            and isinstance(self.defaults.variable, np.matrix)
        ):
            self.defaults.variable = [self.defaults.variable]
            var_is_matrix = True

        super()._instantiate_function(function=function, function_params=function_params, context=context)

        # If it is a matrix, remove from list in which it was embedded after instantiating and evaluating function
        if var_is_matrix:
            self.defaults.variable = self.defaults.variable[0]

    # FIX: PROJECTION_REFACTOR
    #      - MOVE THESE TO Projection, WITH self (Port) AS ADDED ARG
    #          BOTH _instantiate_projections_to_port AND _instantiate_projections_from_port
    #          CAN USE self AS connectee PORT, since _parse_connection_specs USES SOCKET TO RESOLVE
    #      - ALTERNATIVE: BREAK PORT FIELD OF ProjectionTuple INTO sender AND receiver FIELDS, THEN COMBINE
    #          _instantiate_projections_to_port AND _instantiate_projections_to_port INTO ONE METHOD
    #          MAKING CORRESPONDING ASSIGNMENTS TO send AND receiver FIELDS (WOULD BE CLEARER)

    def _instantiate_projections(self, projections, context=None):
        """Implement any Projection(s) to/from Port specified in PROJECTIONS entry of params arg

        Must be implemented by subclasss, to handle interpretation of projection specification(s)
        in a class-appropriate manner:
            PathwayProjections:
              InputPort: _instantiate_projections_to_port (.path_afferents)
              ParameterPort: disallowed
              OutputPort: _instantiate_projections_from_port (.efferents)
              ModulatorySignal: disallowed
            ModulatoryProjections:
              InputPort, OutputPort and ParameterPort:  _instantiate_projections_to_port (mod_afferents)
              ModulatorySignal: _instantiate_projections_from_port (.efferents)
        """

        raise PortError("{} must implement _instantiate_projections (called for {})".
                         format(self.__class__.__name__,
                                self.name))

    # FIX: MOVE TO InputPort AND ParameterPort OR...
    # IMPLEMENTATION NOTE:  MOVE TO COMPOSITION ONCE THAT IS IMPLEMENTED
    def _instantiate_projections_to_port(self, projections, context=None):
        """Instantiate projections to a Port and assign them to self.path_afferents

        Parses specifications in projection_list into ProjectionTuples

        For projection_spec in ProjectionTuple:
            - if it is a Projection specifiction dicionatry, instantiate it
            - assign self as receiver
            - assign sender
            - if deferred_init and sender is instantiated, complete initialization
            - assign to path_afferents or mod_afferents
            - if specs fail, instantiates a default Projection of type specified by self.projection_type

        Notes:
            Calls _parse_connection_specs() to parse projection_list into a list of ProjectionTuples;
                 _parse_connection_specs, in turn, calls _parse_projection_spec for each spec in projection_list,
                 which returns either a Projection object or Projection specification dictionary for each spec;
                 that is placed in projection_spec entry of ProjectionTuple (Port, weight, exponent, projection_spec).
            When the Projection is instantiated, it assigns itself to
               its receiver's .path_afferents attribute (in Projection_Base._instantiate_receiver) and
               its sender's .efferents attribute (in Projection_Base._instantiate_sender);
               so, need to test for prior assignment to avoid duplicates.
        """

        from psyneulink.core.components.projections.pathway.pathwayprojection import PathwayProjection_Base
        from psyneulink.core.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base
        from psyneulink.core.components.projections.projection import _parse_connection_specs

        default_projection_type = self.projection_type

        # If specification is not a list, wrap it in one for consistency of treatment below
        # (since specification can be a list, so easier to treat any as a list)
        projection_list = projections
        if not isinstance(projection_list, list):
            projection_list = [projection_list]

        # return a list of the newly created projections
        new_projections = []

        # Parse each Projection specification in projection_list using self as connectee_port:
        # - calls _parse_projection_spec for each projection_spec in list
        # - validates that Projection specification is compatible with its sender and self
        # - returns ProjectionTuple with Projection specification dictionary for projection_spec
        projection_tuples = _parse_connection_specs(self.__class__, self.owner, projection_list)

        # For Projection in each ProjectionTuple:
        # - instantiate the Projection if necessary, and initialize if possible
        # - insure its value is compatible with self.value FIX: ??and variable is compatible with sender's value
        # - assign it to self.path_afferents or .mod_afferents
        for connection in projection_tuples:

            # Get sender Port, weight, exponent and projection for each projection specification
            #    note: weight and exponent for connection have been assigned to Projection in _parse_connection_specs
            port, weight, exponent, projection_spec = connection

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
                raise PortError("PROGRAM ERROR: Unrecognized {} specification ({}) returned "
                                 "from _parse_connection_specs for connection from {} to {} of {}".
                                 format(Projection.__name__,projection_spec,sender.__name__,self.name,self.owner.name))

            # ASSIGN PARAMS

            # Deferred init
            if projection.initialization_status == ContextFlags.DEFERRED_INIT:

                proj_sender = projection._init_args[SENDER]
                proj_receiver = projection._init_args[RECEIVER]

                # validate receiver
                if proj_receiver is not None and proj_receiver != self:
                    raise PortError("Projection ({}) assigned to {} of {} already has a receiver ({})".
                                     format(projection_type.__name__, self.name, self.owner.name, proj_receiver.name))
                projection._init_args[RECEIVER] = self


                # parse/validate sender
                if proj_sender:
                    # If the Projection already has Port as its sender,
                    #    it must be the same as the one specified in the connection spec
                    if isinstance(proj_sender, Port) and proj_sender != port:
                        raise PortError("Projection assigned to {} of {} from {} already has a sender ({})".
                                         format(self.name, self.owner.name, port.name, proj_sender.name))
                    # If the Projection has a Mechanism specified as its sender:
                    elif isinstance(port, Port):
                        #    Connection spec (port) is specified as a Port,
                        #    so validate that Port belongs to Mechanism and is of the correct type
                        sender = _get_port_for_socket(owner=self.owner,
                                                       mech=proj_sender,
                                                       port_spec=port,
                                                       port_types=port.__class__,
                                                       projection_socket=SENDER)
                    elif isinstance(proj_sender, Mechanism) and inspect.isclass(port) and issubclass(port, Port):
                        #    Connection spec (port) is specified as Port type
                        #    so try to get that Port type for the Mechanism
                        sender = _get_port_for_socket(owner=self.owner,
                                                       connectee_port_type=self.__class__,
                                                       port_spec=proj_sender,
                                                       port_types=port)
                    else:
                        sender = proj_sender
                else:
                    sender = port
                projection._init_args[SENDER] = sender

                projection.sender = sender
                projection.receiver = projection._init_args[RECEIVER]
                projection.receiver.afferents_info[projection] = ConnectionInfo()

                # Construct and assign name
                if isinstance(sender, Port):
                    if sender.initialization_status == ContextFlags.DEFERRED_INIT:
                        sender_name = sender._init_args[NAME]
                    else:
                        sender_name = sender.name
                    sender_name = sender_name or sender.__class__.__name__
                elif inspect.isclass(sender) and issubclass(sender, Port):
                    sender_name = sender.__name__
                else:
                    raise PortError("SENDER of {} to {} of {} is neither a Port or Port class".
                                     format(projection_type.__name__, self.name, self.owner.name))
                projection._assign_default_projection_name(port=self,
                                                           sender_name=sender_name,
                                                           receiver_name=self.name)

                # If sender has been instantiated, try to complete initialization
                # If not, assume it will be handled later (by Mechanism or Composition)
                if isinstance(sender, Port) and sender.initialization_status == ContextFlags.INITIALIZED:
                    projection._deferred_init(context=context)


            # VALIDATE (if initialized)

            if projection.initialization_status == ContextFlags.INITIALIZED:

                # FIX: 10/3/17 - VERIFY THE FOLLOWING:
                # IMPLEMENTATION NOTE:
                #     Assume that validation of Projection's variable (i.e., compatibility with sender)
                #         has already been handled by instantiation of the Projection and/or its sender

                # Validate value:
                #    - check that output of projection's function (projection_spec.value) is compatible with
                #        self.variable;  if it is not, raise exception:
                #        the buck stops here; can't modify projection's function to accommodate the Port,
                #        or there would be an unmanageable regress of reassigning projections,
                #        requiring reassignment or modification of sender OutputPorts, etc.

                # PathwayProjection:
                #    - check that projection's value is compatible with the Port's variable
                if isinstance(projection, PathwayProjection_Base):
                    if not iscompatible(projection.defaults.value, self.defaults.variable[0]):
                    # if len(projection.value) != self.defaults.variable.shape[-1]:
                        raise PortError("Output of function for {} ({}) is not compatible with value of {} ({}).".
                                         format(projection.name, projection.value, self.name, self.defaults.value))

                # ModualatoryProjection:
                #    - check that projection's value is compatible with value of the function param being modulated
                elif isinstance(projection, ModulatoryProjection_Base):
                    mod_spec, mod_param_name, mod_param_value = self._get_modulated_param(projection, context=context)
                    # Match the projection's value with the value of the function parameter
                    mod_proj_spec_value = type_match(projection.defaults.value, type(mod_param_value))
                    if (mod_param_value is not None
                        and not iscompatible(mod_param_value, mod_proj_spec_value)):
                        raise PortError(f"Output of function for {projection.name} ({projection.defaults.value}) "
                                        f"is not compatible with value of {self.name} ({self.defaults.value}).")

            # ASSIGN TO PORT

            # Avoid duplicates, since instantiation of projection may have already called this method
            #    and assigned Projection to self.path_afferents or mod_afferents lists
            if self._check_for_duplicate_projections(projection):
                continue

            # reassign default variable shape to this port and its function
            if isinstance(projection, PathwayProjection_Base) and projection not in self.path_afferents:
                projs = self.path_afferents
                variable = self.defaults.variable
                projs.append(projection)
                new_projections.append(projection)
                if len(projs) > 1:
                    # KDM 5/16/18: Why are we casting this to 2d? I expect this to make the InputPort variable
                    # 2d, so its owner's 3d, but that does not appear to be happening.
                    # Removing this cast can cause an AutoAssignMatrix to interpret the entire InputPort's variable
                    # as its target - ex: two incoming projections -> [0, 0]; third sees socket_width of len 2, so
                    # creates a projection with value length 2, so variable becomes [0, 0, 0, 0]
                    if variable.ndim == 1:
                        variable = np.atleast_2d(variable)
                    self.defaults.variable = np.append(variable, np.atleast_2d(projection.defaults.value), axis=0)

                # assign identical default variable to function if it can be modified
                if self.function._variable_shape_flexibility is DefaultsFlexibility.FLEXIBLE:
                    self.function.defaults.variable = self.defaults.variable.copy()
                elif (
                    self.function._variable_shape_flexibility is DefaultsFlexibility.INCREASE_DIMENSION
                    and np.array([self.function.defaults.variable]).shape == self.defaults.variable.shape
                ):
                    self.function.defaults.variable = np.array([self.defaults.variable])
                elif self.function.defaults.variable.shape != self.defaults.variable.shape:
                    from psyneulink.core.compositions.composition import Composition
                    warnings.warn('A {} from {} is being added to an {} of {} ({}) that already receives other '
                                  'Projections, but does not use a {}; unexpected results may occur when the {} '
                                  'or {} to which it belongs is executed.'.
                                  format(Projection.__name__, projection.sender.owner.name, self.__class__.__name__,
                                         self.owner.name, self.name, CombinationFunction.__name__, Mechanism.__name__,
                                         Composition.__name__))
                            # f'A {Projection.__name__} from {projection.sender.owner.name} is being added ' \
                            #     f'to an {self.__class__.__name__} of {self.owner.name} ({self.name}) ' \
                            #     f'that already receives other Projections, ' \
                            #     f'but does not use a {CombinationFunction.__name__}; ' \
                            #     f'unexpected results may occur when the {Mechanism.__name__} ' \
                            #     f'or {Composition.__name__} to which it belongs is executed.')

            elif isinstance(projection, ModulatoryProjection_Base) and projection not in self.mod_afferents:
                self.mod_afferents.append(projection)
                new_projections.append(projection)

            self.owner._projection_added(projection, context)

        return new_projections

    # FIX: MOVE TO OutputPort or...
    # IMPLEMENTATION NOTE:  MOVE TO COMPOSITION ONCE THAT IS IMPLEMENTED
    def _instantiate_projection_from_port(self, projection_spec, receiver=None, feedback=False, context=None):
        """Instantiate outgoing projection from a Port and assign it to self.efferents

        Instantiate Projections specified in projection_list and, for each:
            - insure its self.value is compatible with the projection's function variable
            - place it in self.efferents:

        Notes:
            # LIST VERSION:
            # If receivers is not specified, they must be assigned to projection specs in projection_list
            # Calls _parse_connection_specs() to parse projection_list into a list of ProjectionTuples;
            #    _parse_connection_specs, in turn, calls _parse_projection_spec for each spec in projection_list,
            #    which returns either a Projection object or Projection specification dictionary for each spec;
            #    that is placed in projection_spec entry of ProjectionTuple (Port, weight, exponent, projection_spec).

            Calls _parse_connection_specs() to parse projection into a ProjectionTuple;
               _parse_connection_specs, in turn, calls _parse_projection_spec for the projection_spec,
               which returns either a Projection object or Projection specification dictionary for the spec;
               that is placed in projection_spec entry of ProjectionTuple (Port, weight, exponent, projection_spec).

            When the Projection is instantiated, it assigns itself to
               its self.path_afferents or .mod_afferents attribute (in Projection_Base._instantiate_receiver) and
               its sender's .efferents attribute (in Projection_Base._instantiate_sender);
               so, need to test for prior assignment to avoid duplicates.

        Returns instantiated Projection
        """
        from psyneulink.core.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base
        from psyneulink.core.components.projections.pathway.pathwayprojection import PathwayProjection_Base
        from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
        from psyneulink.core.components.projections.modulatory.gatingprojection import GatingProjection
        from psyneulink.core.components.projections.projection import ProjectionTuple, _parse_connection_specs

        # FIX: 10/3/17 THIS NEEDS TO BE MADE SPECIFIC TO EFFERENT PROJECTIONS (I.E., FOR WHICH IT CAN BE A SENDER)
        # default_projection_type = ProjectionRegistry[self.projection_type].subclass
        default_projection_type = self.projection_type

        # projection_object = None # flags whether projection object has been instantiated; doesn't store object
        # projection_type = None   # stores type of projection to instantiate
        # projection_params = {}


        # IMPLEMENTATION NOTE:  THE FOLLOWING IS WRITTEN AS A LOOP IN PREP FOR GENERALINZING METHOD
        #                       TO HANDLE PROJECTION LIST (AS PER _instantiate_projections_to_port())

        # If projection_spec and/or receiver is not in a list, wrap it in one for consistency of treatment below
        # (since specification can be a list, so easier to treat any as a list)
        projection_list = projection_spec
        if not isinstance(projection_list, list):
            projection_list = [projection_list]

        if not receiver:
            receiver_list = [None] * len(projection_list)
        elif not isinstance(receiver, list):
            receiver_list = [receiver]

        # Parse Projection specification using self as connectee_port:
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

            # Validate that Port to be connected to specified in receiver is same as any one specified in connection
            def _get_receiver_port(spec):
                """Get port specification from ProjectionTuple, which itself may be a ProjectionTuple"""
                if isinstance(spec, (tuple, ProjectionTuple)):
                    spec = _parse_connection_specs(connectee_port_type=self.__class__,
                                                   owner=self.owner,
                                                   connections=receiver)
                    return _get_receiver_port(spec[0].port)
                elif isinstance(spec, Projection):
                    return spec.receiver
                # FIX: 11/25/17 -- NEEDS TO CHECK WHETHER PRIMARY SHOULD BE INPUT_PORT OR PARAMETER_PORT
                elif isinstance(spec, Mechanism):
                    return spec.input_port
                return spec
            receiver_port = _get_receiver_port(receiver)
            connection_receiver_port = _get_receiver_port(connection)
            if receiver_port != connection_receiver_port:
                raise PortError("PROGRAM ERROR: Port specified as receiver ({}) should "
                                 "be the same as the one specified in the connection {}.".
                                 format(receiver_port, connection_receiver_port))

            if (not isinstance(connection, ProjectionTuple)
                and receiver
                and not isinstance(receiver, (Port, Mechanism))
                and not (inspect.isclass(receiver) and issubclass(receiver, (Port, Mechanism)))):
                raise PortError("Receiver ({}) of {} from {} must be a {}, {}, a class of one, or a {}".
                                 format(receiver, projection_spec, self.name,
                                        Port.__name__, Mechanism.__name__, ProjectionTuple.__name__))

            if isinstance(receiver, Mechanism):
                from psyneulink.core.components.ports.inputport import InputPort
                from psyneulink.core.components.ports.parameterport import ParameterPort

                # If receiver is a Mechanism and Projection is a MappingProjection,
                #    use primary InputPort (and warn if verbose is set)
                if isinstance(default_projection_type, (MappingProjection, GatingProjection)):
                    if self.owner.verbosePref:
                        warnings.warn("Receiver {} of {} from {} is a {} and {} is a {}, "
                                      "so its primary {} will be used".
                                      format(receiver, projection_spec, self.name, Mechanism.__name__,
                                             Projection.__name__, default_projection_type.__name__,
                                             InputPort.__name__))
                    receiver = receiver.input_port

                    raise PortError("Receiver {} of {} from {} is a {}, but the specified {} is a {} so "
                                     "target {} can't be determined".
                                     format(receiver, projection_spec, self.name, Mechanism.__name__,
                                            Projection.__name__, default_projection_type.__name__,
                                            ParameterPort.__name__))


            # GET Projection --------------------------------------------------------

            # Get sender Port, weight, exponent and projection for each projection specification
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

                if projection.initialization_status == ContextFlags.DEFERRED_INIT:
                    projection._init_args[RECEIVER] = projection._init_args[RECEIVER] or receiver
                    proj_recvr = projection._init_args[RECEIVER]
                else:
                    projection.receiver = projection.receiver or receiver
                    proj_recvr = projection.receiver
                projection._assign_default_projection_name(port=self,
                                                           sender_name=self.name,
                                                           receiver_name=proj_recvr.name)

            # Projection specification dictionary or None:
            elif isinstance(projection_spec, (dict, None)):

                # Instantiate Projection from specification dict
                projection_type = projection_spec.pop(PROJECTION_TYPE, None) or default_projection_type
                # If Projection was not specified, create default Projection specification dict
                if not (projection_spec or len(projection_spec)):
                    projection_spec = {SENDER: self, RECEIVER: receiver_port}
                projection = projection_type(**projection_spec)
                try:
                    projection.receiver = projection.receiver
                except AttributeError:
                    projection.receiver = receiver
                proj_recvr = projection.receiver

            else:
                rcvr_str = ""
                if receiver:
                    if isinstance(receiver, Port):
                        rcvr_str = " to {}".format(receiver.name)
                    else:
                        rcvr_str = " to {}".format(receiver.__name__)
                raise PortError("PROGRAM ERROR: Unrecognized {} specification ({}) returned "
                                 "from _parse_connection_specs for connection from {} of {}{}".
                                 format(Projection.__name__,projection_spec,self.name,self.owner.name,rcvr_str))

            # Validate that receiver and projection_spec receiver are now the same
            receiver = proj_recvr or receiver  # If receiver was not specified, assign it receiver from projection_spec
            if proj_recvr and receiver and proj_recvr is not receiver:
                # Note: if proj_recvr is None, it will be assigned under handling of deferred_init below
                raise PortError("Receiver ({}) specified for Projection ({}) "
                                 "is not the same as the one specified in {} ({})".
                                 format(proj_recvr, projection.name, ProjectionTuple.__name__, receiver))

            # ASSIGN REMAINING PARAMS

            # Deferred init
            if projection.initialization_status == ContextFlags.DEFERRED_INIT:
                projection._init_args[SENDER] = self
                if isinstance(receiver, Port) and receiver.initialization_status == ContextFlags.INITIALIZED:
                    projection._deferred_init(context=context)

            # VALIDATE (if initialized or being initialized)

            if projection.initialization_status & (ContextFlags.INITIALIZED | ContextFlags.INITIALIZING):

                # If still being initialized, then assign sender and receiver as necessary
                if projection.initialization_status == ContextFlags.INITIALIZING:
                    if not isinstance(projection.sender, Port):
                        projection.sender = self

                    if not isinstance(projection.receiver, Port):
                        projection.receiver = receiver_port

                    projection._assign_default_projection_name(
                        port=self,
                        sender_name=self.name,
                        receiver_name=projection.receiver.name
                    )

                # when this is called during initialization, doesn't make sense to validate here
                # because the projection values are set later to the values they're being validated against here
                else:
                    # Validate variable
                    #    - check that input to Projection is compatible with self.value
                    if not iscompatible(self.defaults.value, projection.defaults.variable):
                        raise PortError(f"Input to {projection.name} ({projection.defaults.variable}) "
                                        f"is not compatible with the value ({self.defaults.value}) of "
                                        f"the Port from which it is supposed to project ({self.name}).")

                    # Validate value:
                    #    - check that output of projection's function (projection_spec.value) is compatible with
                    #        variable of the Port to which it projects;  if it is not, raise exception:
                    #        the buck stops here; can't modify projection's function to accommodate the Port,
                    #        or there would be an unmanageable regress of reassigning projections,
                    #        requiring reassignment or modification of sender OutputPorts, etc.

                    # PathwayProjection:
                    #    - check that projection's value is compatible with the receiver's variable
                    if isinstance(projection, PathwayProjection_Base):
                        if not iscompatible(projection.value, receiver.socket_template):
                            raise PortError(f"Output of {projection.name} ({projection.value}) "
                                            f"is not compatible with the variable ({receiver.defaults.variable}) of "
                                            f"the Port to which it is supposed to project ({receiver.name}).")

                    # ModualatoryProjection:
                    #    - check that projection's value is compatible with value of the function param being modulated
                    elif isinstance(projection, ModulatoryProjection_Base):
                        mod_spec, mod_param_name, mod_param_value = self._get_modulated_param(projection,
                                                                                              receiver=receiver,
                                                                                              context=context)
                        # Match the projection's value with the value of the function parameter
                        # should be defaults.value?
                        mod_proj_spec_value = type_match(projection.value, type(mod_param_value))
                        if (mod_param_value is not None
                            and not iscompatible(mod_param_value, mod_proj_spec_value)):
                            raise PortError(f"Output of {projection.name} ({mod_proj_spec_value}) is not compatible "
                                            f"with the value of {receiver.name} ({mod_param_value}).")


            # ASSIGN TO PORT

            # Avoid duplicates, since instantiation of projection may have already called this method
            #    and assigned Projection to self.efferents
            if self._check_for_duplicate_projections(projection):
                continue

            # FIX: MODIFIED FEEDBACK - CHECK THAT THAT THIS IS STILL NEEDED (RE: ASSIGNMENT IN ModulatorySignal)
            # FIX: 9/14/19 - NOTE:  IT *IS* NEEDED FOR CONTROLPROJECTIONS
            #                       SPECIFIED FOR PARAMETER IN CONSTRUCTOR OF A MECHANISM
            if isinstance(projection, ModulatoryProjection_Base):
                self.owner.aux_components.append((projection, feedback))
            return projection

    def _remove_projection_from_port(self, projection, context=None):
        """Remove Projection entry from Port.efferents."""
        del self.efferents[self.efferents.index(projection)]

    def _remove_projection_to_port(self, projection, context=None):
        """
        If projection is in mod_afferents, remove that projection from self.mod_afferents.
        Else, Remove Projection entry from Port.path_afferents and reshape variable accordingly.
        """
        if projection in self.mod_afferents:
            del self.mod_afferents[self.mod_afferents.index(projection)]
        else:
            shape = list(self.defaults.variable.shape)
            # Reduce outer dimension by one
            shape[0]-=1
            self.defaults.variable = np.resize(self.defaults.variable, shape)
            self.function.defaults.variable = np.resize(self.function.defaults.variable, shape)
            del self.path_afferents[self.path_afferents.index(projection)]

    def _get_primary_port(self, mechanism):
        raise PortError("PROGRAM ERROR: {} does not implement _get_primary_port method".
                         format(self.__class__.__name__))

    def _parse_port_specific_specs(self, owner, port_dict, port_specific_spec):
        """Parse parameters in Port specification tuple specific to each subclass

        Called by _parse_port_spec()
        port_dict contains standard args for Port constructor passed to _parse_port_spec
        port_specific_spec is either a:
            - tuple containing a specification for the Port and/or Projections to/from it
            - a dict containing port-specific parameters to be processed

         Returns two values:
         - port_spec:  specification for the Port;
                          - can be None (this is usually the case when port_specific_spec
                            is a tuple specifying a Projection that will be used to specify the port)
                          - if a value is returned, that is used by _parse_port_spec in a recursive call to
                            parse the specified value as the Port specification
         - params: port-specific parameters that will be included in the PARAMS entry of the Port specification dict
         """
        raise PortError("PROGRAM ERROR: {} does not implement _parse_port_specific_specs method".
                         format(self.__class__.__name__))

    def _update(self, params=None, context=None):
        """Update each projection, combine them, and assign return result

        Assign any runtime_params specified for Port, its function, and any of its afferent projections;
          - assumes that type-specific sub-dicts have been created for each Projection type for which there are params
          - and that specifications for individual Projections have been put in their own PROJECTION-SPECIFIC sub-dict
          - specifications for individual Projections are removed from that as used (for check by Mechanism at end)
        Call _update for each projection in self.path_afferents (passing specified params)
        Note: only update LearningSignals if context == LEARNING; otherwise, just get their value
        Call self.function (default: LinearCombination function) to combine their values
        Returns combined values of projections, modulated by any mod_afferents
        """
        from psyneulink.core.components.projections.projection import \
            projection_param_keywords, projection_param_keyword_mapping

        # Skip execution and set value directly if function is identity_function and no runtime_params were passed
        if (
            len(self.all_afferents) == 0
            and self.function._is_identity(context)
            and not params
        ):
            variable = self._parse_function_variable(self._get_variable_from_projections(context))
            self.parameters.variable._set(variable, context)
            # FIX: below conversion really should not be happening ultimately, but it is
            # in _validate_variable. Should be removed eventually
            variable = convert_to_np_array(variable, 1)
            self.parameters.value._set(variable, context)
            self.most_recent_context = context
            self.function.most_recent_context = context
            return

        # GET RUNTIME PARAMS FOR PORT AND ITS PROJECTIONS ---------------------------------------------------------

        # params (ones passed from Mechanism that should be kept intact for other Ports):
        #  - remove any found in PORT_SPECIFIC_PARAMS specific to this port
        #      (Mechanism checks for errant ones after all ports have been executed)
        #  - remove any found in PROJECTION_SPECIFIC for Projections to this port
        #      (Mechanism checks for errant ones after all ports have been executed)

        # local_params (ones that will be passed to Port's execute method):
        #  - copy of ones params outer dict
        #  - ones for this Port from params PARAMS_SPECIFIC_DICT (override "general" ones from outer dict)
        #  - mod_params (from _execute_afferent_projections)

        # projection_params (ones that will passed to _execute_afferent_projections() method):
        #  - copy of ones from <PROJECTION_TYPE>_PROJECTION_PARAMS
        #  - ones from PROJECTION_SPECIFIC_PARAMS relevant to Projections to this Port
        #    (override more "general" ones specified for the type of Projection)

        # params = params or {}
        params = defaultdict(lambda:{}, params or {})

        # FIX 5/8/20 [JDC]:
        #    ADD IF STATEMENT HERE TO ASSIGN EMPTY DICTS TO local_params AND projection_params IF params is EMPTY

        # Move any params specified for Port's function in FUNCTION_PARAMS dict into runtime_port_params
        # Do this on params so it holds for all subsequents ports processed
        if FUNCTION_PARAMS in params:
            params.update(params.pop(FUNCTION_PARAMS))

        # Copy all items in outer level of params to local_params (i.e., excluding its subdicts)
        local_params = defaultdict(lambda:{}, {k:v for k,v in params.items() if not isinstance(v,dict)})
        # Get rid of items in params specific to this Port
        for entry in params[PORT_SPECIFIC_PARAMS].copy():
            if entry in {self, self.name}:
                # Move param from params to local_params
                local_params.update(params[PORT_SPECIFIC_PARAMS].pop(entry))

        # Put copy of all type-specific Projection dicts from params into local_params
        # FIX: ON FIRST PASS ALSO CREATES THOSE DICTS IN params IF THEY DON'T ALREADY EXIST
        projection_params = defaultdict(lambda:{}, {proj_type:params[proj_type].copy()
                                                    for proj_type in projection_param_keywords()})

        for entry in params[PROJECTION_SPECIFIC_PARAMS].copy():
            if self.all_afferents and entry in self.all_afferents + [p.name for p in self.all_afferents]:
                if isinstance(entry, str):
                    projection_type = next(p for p in self.all_afferents if p.name ==entry).componentType
                else:
                    projection_type = entry.componentType
                # Get key for type-specific dict in params in which to place it in local_params
                projection_param_type = projection_param_keyword_mapping()[projection_type]
                # Move from params into relevant type-specific dict in local_params
                projection_params[projection_param_type].update(params[PROJECTION_SPECIFIC_PARAMS].pop(entry))

        # Note:  having removed all Port- and Projection-specific entries on params, no longer concerned with it;
        #        now only care about local_params and projection_params

        # If either the Port's variable or value is specified in runtime_params, skip executing Projections
        if local_params and any(var_or_val in local_params for var_or_val in {VARIABLE, VALUE}):
            mod_params = {}
        else:
            # Otherwise, execute afferent Projections
            mod_params = self._execute_afferent_projections(projection_params, context)
            if mod_params == OVERRIDE:
                return
        local_params.update(mod_params)

        # EXECUTE PORT  -------------------------------------------------------------------------------------

        self._validate_and_assign_runtime_params(local_params, context=context)
        variable = local_params.pop(VARIABLE, None)
        self.execute(variable, context=context, runtime_params=local_params)

    def _execute_afferent_projections(self, projection_params, context):
        """Execute all afferent Projections for Port

        Returns
        -------
        mod_params : dict or OVERRIDE

        """
        from psyneulink.core.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base
        from psyneulink.core.components.projections.modulatory.learningprojection import LearningProjection
        from psyneulink.library.components.projections.pathway.maskedmappingprojection import MaskedMappingProjection
        from psyneulink.core.components.projections.projection import projection_param_keyword_mapping

        def set_projection_value(projection, value, context):
            """Manually set Projection value"""
            projection.parameters.value._set(value, context)
            # KDM 8/14/19: a caveat about the dot notation/most_recent_context here!
            # should these be manually set despite it not actually being executed?
            # explicitly getting/setting based on context will be more clear
            projection.most_recent_context = context
            projection.function.most_recent_context = context
            for pport in projection.parameter_ports:
                pport.most_recent_context = context
                pport.function.most_recent_context = context

        # EXECUTE AFFERENT PROJECTIONS ------------------------------------------------------------------------------

        modulatory_override = False
        mod_proj_values = {}

        # For each projection: get its params, pass them to it, get the projection's value, and append to relevant list
        for projection in self.all_afferents:

            if not self.afferents_info[projection].is_active_in_composition(context.composition):
                continue

            if hasattr(projection, 'sender'):
                sender = projection.sender
            else:
                if self.verbosePref:
                    warnings.warn(f"{projection.__class__.__name__} to {self.name} {self.__class__.__name__} "
                                  f"of {self.owner.name} ignored [has no sender].")
                continue

            # Get type-specific params that apply for type of current
            projection_params_keyword = projection_param_keyword_mapping()[projection.componentType]
            projection_type_params = projection_params[projection_params_keyword].copy()

            # Get Projection's variable and/or value if specified in runtime_port_params
            projection_variable = projection_type_params.pop(VARIABLE, None)
            projection_value = projection_type_params.pop(VALUE, None)

            # Projection value specifed in runtime_port_params, so just assign its value
            if projection_value:
                set_projection_value(projection, projection_value, context)

            # ----------------------

            # Handle LearningProjection
            #  - update LearningSignals only if context == LEARNING;  otherwise, assign zero for projection_value
            # IMPLEMENTATION NOTE: done here rather than in its own method in order to exploit parsing of params above
            elif (isinstance(projection, LearningProjection) and ContextFlags.LEARNING not in context.execution_phase):
                projection_value = projection.defaults.value * 0.0
            elif (
                # learning projections add extra behavior in _execute that invalidates identity function
                not isinstance(projection, LearningProjection)
                # masked mapping projections apply a mask separate from their function - consider replacing it
                # with a masked linear matrix and removing this special class?
                and not isinstance(projection, MaskedMappingProjection)
                and projection.function._is_identity(context)
                # has no parameter ports with afferents (these can modulate parameters and make it non-identity)
                and len(list(itertools.chain.from_iterable([p.all_afferents for p in projection.parameter_ports]))) == 0
                # matrix ParameterPort may be a non identity Accumulator integrator
                and all(pport.function._is_identity(context) for pport in projection.parameter_ports)
            ):
                if projection_variable is None:
                    projection_variable = projection.sender.parameters.value._get(context)
                    # KDM 8/14/19: this fallback seems to always happen on the first execution
                    # of the Projection's function (LinearMatrix). Unsure if this is intended or not
                    if projection_variable is None:
                        projection_variable = projection.function.defaults.value
                projection.parameters.variable._set(projection_variable, context)
                projection_value = projection._parse_function_variable(projection_variable)
                set_projection_value(projection,projection_value, context)

            # Actually execute Projection to get its value
            else:
                if projection_variable is None:
                    projection_variable = projection.sender.parameters.value._get(context)
                projection_value = projection.execute(variable=projection_variable,
                                                      context=context,
                                                      runtime_params=projection_type_params,
                                                      )

            # If this is initialization run and projection initialization has been deferred, pass
            try:
                if projection.initialization_status == ContextFlags.DEFERRED_INIT:
                    continue
            except AttributeError:
                pass

            # # KDM 6/20/18: consider moving handling of Modulatory projections into separate method
            # If it is a ModulatoryProjection, add its value to the list in the dict entry for the relevant mod_param
            if isinstance(projection, ModulatoryProjection_Base):
                # Get the meta_param to be modulated from modulation attribute of the  projection's ModulatorySignal
                #    and get the function parameter to be modulated to type_match the projection value below
                mod_spec, mod_param_name, mod_param_value = self._get_modulated_param(projection, context=context)
                # If meta_param is DISABLE, ignore the ModulatoryProjection
                if mod_spec == DISABLE:
                    continue
                if mod_spec == OVERRIDE:
                    # If paramValidationPref is set, allow all projections to be processed
                    #    to be sure there are no other conflicting OVERRIDES assigned
                    if self.owner.paramValidationPref:
                        if modulatory_override:
                            raise PortError(f"Illegal assignment of {MODULATION_OVERRIDE} to more than one "
                                             f"{MODULATORY_SIGNAL} ({projection.name} and {modulatory_override[2]}).")
                        modulatory_override = (MODULATION_OVERRIDE, projection_value, projection)
                        continue
                    # Otherwise, for efficiency, assign first OVERRIDE value encountered and return
                    else:
                        # FIX 5/8/20 [JDC]: SHOULD THIS USE set_projection_value()??
                        self.parameters.value._set(type_match(projection_value, type(self.defaults.value)), context)
                        return OVERRIDE
                else:
                    try:
                        mod_value = type_match(projection_value, type(mod_param_value))
                    except TypeError:
                        # if type_match fails, assume that the computation is
                        # valid further down the line. This was implicitly true
                        # before adding this catch block by manually setting the
                        # modulated param value from None to a default
                        mod_value = projection_value

                    if mod_param_name not in mod_proj_values.keys():
                        mod_proj_values[mod_param_name]=[mod_value]
                    else:
                        mod_proj_values[mod_param_name].append(mod_value)

        # Handle ModulatoryProjection OVERRIDE
        #    if there is one and it wasn't been handled above (i.e., if paramValidation is set)
        if modulatory_override:
            # KDM 6/20/18: consider defining exactly when and how type_match occurs, now it seems
            # a bit handwavy just to make stuff work
            # FIX 5/8/20 [JDC]: SHOULD THIS USE set_projection_value()??
            self.parameters.value._set(type_match(modulatory_override[1], type(self.defaults.value)), context)
            return OVERRIDE

        # AGGREGATE ModulatoryProjection VALUES  -----------------------------------------------------------------------

        mod_params = {}
        for mod_param_name, value_list in mod_proj_values.items():
            param = getattr(self.function.parameters, mod_param_name)
            # If the param has a single modulatory value, use that
            if len(value_list)==1:
                mod_val = value_list[0]
            # If the param has multiple modulatory values, combine them
            else:
                mod_val = self._get_combined_mod_val(mod_param_name, value_list)

            # FIX: SHOULD THIS REALLY BE GETTING SET HERE??
            # Set modulatory parameter's value
            param._set(mod_val, context)
            # Add mod_param and its value to port_params for Port's function
            mod_params.update({mod_param_name: mod_val})
        return mod_params

    def _execute(self, variable=None, context=None, runtime_params=None):
        if variable is None:
            variable = self._get_variable_from_projections(context)

            # if the fallback is also None
            # return None, so that this port is ignored
            # KDM 8/2/19: double check the relevance of this branch
            if variable is None:
                return None

        return super()._execute(
            variable,
            context=context,
            runtime_params=runtime_params,
        )

    def _get_modulated_param(self, mod_proj, receiver=None, context=None):
        """Return modulation specification from ModulatoryProjection, and name and value of param modulated."""

        from psyneulink.core.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base

        receiver = receiver or self

        if not isinstance(mod_proj, ModulatoryProjection_Base):
            raise PortError(f'Specification of {MODULATORY_PROJECTION} to {receiver.full_name} ({mod_proj}) '
                                f'is not a {ModulatoryProjection_Base.__name__}')

        # Get modulation specification from the Projection sender's modulation attribute
        mod_spec = mod_proj.sender.parameters.modulation._get(context)

        if mod_spec in {OVERRIDE, DISABLE}:
            mod_param_name = mod_proj.receiver.name
            mod_param_value = mod_proj.sender.parameters.value.get(context)
        else:
            mod_param = getattr(receiver.function.parameters, mod_spec)
            try:
                mod_param_name = mod_param.source.name
            except:
                mod_param_name = mod_param.name

            # Get the value of the modulated parameter
            mod_param_value = getattr(receiver.function.parameters, mod_spec).get(context)

        return mod_spec, mod_param_name, mod_param_value

    def _get_combined_mod_val(self, mod_param_name, values):
        """Combine the modulatory values received by ModulatoryProjections to mod_param_name
        Uses function specified by modulation_combination_function attribute of param,
        or MULTIPLICATIVE if not specified
        """
        comb_fct = getattr(self.function.parameters, mod_param_name).modulation_combination_function or MULTIPLICATIVE
        aliases = getattr(self.function.parameters, mod_param_name).aliases

        if comb_fct==MULTIPLICATIVE or any(mod_spec in aliases for mod_spec in {MULTIPLICATIVE, MULTIPLICATIVE_PARAM}):
            return np.product(np.array(values), axis=0)
        if comb_fct==ADDITIVE or any(mod_spec in aliases for mod_spec in {MULTIPLICATIVE, ADDITIVE_PARAM}):
            return np.sum(np.array(values), axis=0)
        elif isinstance(comb_fct, is_function_type):
            return comb_fct(values)
        else:
            assert False, f'PROGRAM ERROR: modulation_combination_function not properly specified ' \
                          f'for {mod_param_name} {Parameter.__name__} of {self.name}'

    @abc.abstractmethod
    def _get_variable_from_projections(self, context=None):
        """
            Return a variable to be used for self.execute when the variable passed in is None
        """
        pass

    def _get_value_label(self, labels_dict, all_ports, context=None):
        subdicts = False
        if labels_dict != {}:
            if isinstance(list(labels_dict.values())[0], dict):
                subdicts = True

        if not subdicts:    # Labels are specified at the mechanism level - not individual ports
            # label dict only applies to index 0 port
            if all_ports.index(self) == 0:
                for label in labels_dict:
                    if np.allclose(labels_dict[label], self.parameters.value.get(context)):
                        return label
            # if this isn't the index 0 port OR a label was not found then just return the original value
            return self.parameters.value.get(context)

        for port in labels_dict:
            if port is self:
                return self.find_label_value_match(port, labels_dict, context)
            elif port == self.name:
                return self.find_label_value_match(self.name, labels_dict, context)
            elif port == all_ports.index(self):
                return self.find_label_value_match(all_ports.index(self), labels_dict, context)

        return self.parameters.value.get(context)

    def find_label_value_match(self, key, labels_dict, context=None):
        for label in labels_dict[key]:
            if np.allclose(labels_dict[key][label], self.parameters.value.get(context)):
                return label
        return self.parameters.value.get(context)

    @property
    def owner(self):
        return self._owner

    @owner.setter
    def owner(self, assignment):
        self._owner = assignment

    @property
    def all_afferents(self):
        return self.path_afferents + self.mod_afferents

    @property
    def afferents_info(self):
        try:
            return self._afferents_info
        except AttributeError:
            self._afferents_info = {}
            return self._afferents_info

    @property
    def efferents(self):
        try:
            return self._efferents
        except:
            self._efferents = []
            return self._efferents

    @efferents.setter
    def efferents(self, proj):
        assert False, f"Illegal attempt to directly assign {repr('efferents')} attribute of {self.name}"

    @property
    def full_name(self):
        """Return name relative to owner as:  <owner.name>[<self.name>]"""
        if self.owner:
            return f'{self.owner.name}[{self.name}]'
        else:
            return self.name

    def _assign_default_port_Name(self, context=None):
        return False

    def _get_input_struct_type(self, ctx):
        # Use function input type. The shape should be the same,
        # however, some functions still need input shape workarounds.
        func_input_type = ctx.get_input_struct_type(self.function)
        # MODIFIED 4/4/20 NEW: [PER JAN]
        if len(self.path_afferents) > 0:
            assert len(func_input_type) == len(self.path_afferents), \
                "{} shape mismatch: {}\nport:\n\t{}\n\tfunc: {}\npath_afferents: {}".format(
                    self, func_input_type, self.defaults.variable,
                    self.function.defaults.variable, len(self.path_afferents))
        # MODIFIED 4/4/20 END
        input_types = [func_input_type]
        # Add modulation
        for mod in self.mod_afferents:
            input_types.append(ctx.get_output_struct_type(mod))
        return pnlvm.ir.LiteralStructType(input_types)

    def _gen_llvm_function_body(self, ctx, builder, params, state, arg_in, arg_out, *, tags:frozenset):
        state_f = ctx.import_llvm_function(self.function)

        # Create a local copy of the function parameters
        base_params = pnlvm.helpers.get_param_ptr(builder, self, params,
                                                  "function")
        f_params = builder.alloca(state_f.args[0].type.pointee)
        builder.store(builder.load(base_params), f_params)

        # FIXME: Handle and combine multiple afferents
        assert len(self.mod_afferents) <= 1

        # Apply modulation
        for idx, afferent in enumerate(self.mod_afferents):
            # The first input is function data input
            # Modulatory projections are ordered after that

            # Get the modulation value
            f_mod_ptr = builder.gep(arg_in, [ctx.int32_ty(0),
                                             ctx.int32_ty(idx + 1)])

            # Get name of the modulated parameter
            if afferent.sender.modulation == MULTIPLICATIVE:
                name = self.function.parameters.multiplicative_param.source.name
            elif afferent.sender.modulation == ADDITIVE:
                name = self.function.parameters.additive_param.source.name
            elif afferent.sender.modulation == DISABLE:
                name = None
            elif afferent.sender.modulation == OVERRIDE:
                # Directly store the value in the output array
                if f_mod_ptr.type != arg_out.type:
                    assert len(f_mod_ptr.type.pointee) == 1
                    warnings.warn("Shape mismatch: Overriding modulation should match parameter port output: {} vs. {}".format(
                                  afferent.defaults.value, self.defaults.value))
                    f_mod_ptr = builder.gep(f_mod_ptr, [ctx.int32_ty(0), ctx.int32_ty(0)])
                builder.store(builder.load(f_mod_ptr), arg_out)
                return builder
            else:
                assert False, "Unsupported modulation parameter: {}".format(afferent.sender.modulation)

            # Replace base param with the modulation value
            if name is not None:
                f_mod_param_ptr = pnlvm.helpers.get_param_ptr(builder,
                                                              self.function,
                                                              f_params, name)
                if f_mod_param_ptr.type != f_mod_ptr.type:
                    warnings.warn("Shape mismatch: Modulation vs. modulated parameter: {} vs. {}".format(
                                  afferent.defaults.value,
                                  getattr(self.function.parameters, name).get(None)))
                    param_val = pnlvm.helpers.load_extract_scalar_array_one(builder, f_mod_ptr)
                else:
                    param_val = builder.load(f_mod_ptr)
                builder.store(param_val, f_mod_param_ptr)

        # OutputPort returns 1D array even for scalar functions
        if arg_out.type != state_f.args[3].type:
            assert len(arg_out.type.pointee) == 1
            arg_out = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(0)])
        # Extract the data part of input
        f_input = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(0)])
        f_state = pnlvm.helpers.get_state_ptr(builder, self, state, "function")
        builder.call(state_f, [f_params, f_state, f_input, arg_out])
        return builder

    @staticmethod
    def _get_port_function_value(owner, function, variable):
        """Execute the function of a Port and return its value
        # FIX: CONSIDER INTEGRATING THIS INTO _EXECUTE FOR PORT?

        This is a stub, that a Port subclass can override to treat its function in a Port-specific manner.
        Used primarily during validation, when the function may not have been fully instantiated yet
        (e.g., InputPort must sometimes embed its variable in a list-- see InputPort._get_port_function_value).
        """
        return function.execute(variable, context=Context(source=ContextFlags.UNSET, execution_id=None))

    @property
    def _dependent_components(self):
        return list(itertools.chain(
            super()._dependent_components,
            self.efferents,
        ))

    @property
    def _dict_summary(self):
        return {
            **super()._dict_summary,
            **{
                'shape': str(self.defaults.variable.shape),
                'dtype': str(self.defaults.variable.dtype)
            }
        }


def _instantiate_port_list(owner,
                            port_list,              # list of Port specs, (port_spec, params) tuples, or None
                            port_types,             # PortType subclass
                            port_Param_identifier,  # used to specify port_type Port(s) in params[]
                            reference_value,         # value(s) used as default for Port and to check compatibility
                            reference_value_name,    # name of reference_value type (e.g. variable, output...)
                            context=None):
    """Instantiate and return a ContentAddressableList of Ports specified in port_list

    Arguments:
    - port_type (class): Port class to be instantiated
    - port_list (list): List of Port specifications (generally from owner.kw<port>),
                             each item of which must be a:
                                 string (used as name)
                                 number (used as constraint value)
                                 dict (key=name, value=reference_value or param dict)
                         if None, instantiate a single default Port using reference_value as port_spec
    - port_Param_identifier (str): used to identify set of Ports in params;  must be one of:
        - INPUT_PORT
        - OUTPUT_PORT
        (note: this is not a list, even if port_types is, since it is about the attribute to which the
               ports will be assigned)
    - reference_value (2D np.array): set of 1D np.ndarrays used as default values and
        for compatibility testing in instantiation of Port(s):
        - INPUT_PORT: self.defaults.variable
        - OUTPUT_PORT: self.value
        ?? ** Note:
        * this is ignored if param turns out to be a dict (entry value used instead)
    - reference_value_name (str):  passed to Port._instantiate_port(), used in error messages
    - context (str)

    If port_list is None:
        - instantiate a default Port using reference_value,
        - place as the single entry of the list returned.
    Otherwise, if port_list is:
        - a single value:
            instantiate it (if necessary) and place as the single entry in an OrderedDict
        - a list:
            instantiate each item (if necessary) and place in a ContentAddressableList
    In each case, generate a ContentAddressableList with one or more entries, assigning:
        # the key for each entry the name of the Port if provided,
        #     otherwise, use MECHANISM<port_type>Ports-n (incrementing n for each additional entry)
        # the Port value for each entry to the corresponding item of the Mechanism's port_type Port's value
        # the dict to self.<port_type>Ports
        # self.<port_type>Port to self.<port_type>Ports[0] (the first entry of the dict)
    Notes:
        * if there is only one Port, but the value of the Mechanism's port_type has more than one item:
            assign it to the sole Port, which is assumed to have a multi-item value
        * if there is more than one Port:
            the number of Ports must match length of Mechanisms port_type value or an exception is raised
    """

    # If no Ports were passed in, instantiate a default port_type using reference_value
    if not port_list:
        # assign reference_value as single item in a list, to be used as port_spec below
        port_list = reference_value

        # issue warning if in VERBOSE mode:
        if owner.prefs.verbosePref:
            print(f"No {port_Param_identifier} specified for {owner.__class__.__name__}; "
                  f"default will be created using {reference_value_name} "
                  f"of function ({reference_value}) as its value.")

    # Ports should be either in a list, or possibly an np.array (from reference_value assignment above):
    # KAM 6/21/18 modified to include tuple as an option for port_list
    if not isinstance(port_list, (ContentAddressableList, list, np.ndarray, tuple)):
        # This shouldn't happen, as items of port_list should be validated to be one of the above in _validate_params
        raise PortError("PROGRAM ERROR: {} for {} is not a recognized \'{}\' specification for {}; "
                         "it should have been converted to a list in Mechanism._validate_params)".
                         format(port_list, owner.name, port_Param_identifier, owner.__class__.__name__))


    # VALIDATE THAT NUMBER OF PORTS IS COMPATIBLE WITH NUMBER OF ITEMS IN reference_values

    num_ports = len(port_list)
    # Check that reference_value is an indexable object, the items of which are the constraints for each Port
    # Notes
    # * generally, this will be a list or an np.ndarray (either >= 2D np.array or with a dtype=object)
    # * for OutputPorts, this should correspond to its value
    try:
        # Insure that reference_value is an indexible item (list, >=2D np.darray, or otherwise)
        num_constraint_items = len(reference_value)
    except:
        raise PortError(f"PROGRAM ERROR: reference_value ({reference_value}) for {reference_value_name} of "
                         f"{[s.__name__ for s in port_types]} must be an indexable object (e.g., list or np.ndarray).")
    # If number of Ports does not equal the number of items in reference_value, raise exception
    if num_ports != num_constraint_items:
        if num_ports > num_constraint_items:
            comparison_string = 'more'
        else:
            comparison_string = 'fewer'
        raise PortError(f"There are {comparison_string} {port_Param_identifier}s specified ({num_ports}) "
                         f"than the number of items ({num_constraint_items}) in the {reference_value_name} "
                         f"of the function for {repr(owner.name)}.")

    # INSTANTIATE EACH PORT

    ports = ContentAddressableList(component_type=Port_Base,
                                   name=owner.name + ' ContentAddressableList of ' + port_Param_identifier)
    # For each port, pass port_spec and the corresponding item of reference_value to _instantiate_port

    if not isinstance(port_types, list):
        port_types = [port_types] * len(port_list)
    if len(port_types) != len(port_list):
        port_types = [port_types[0]] * len(port_list)
    # for index, port_spec, port_type in enumerate(zip(port_list, port_types)):
    for index, port_spec, port_type in zip(list(range(len(port_list))), port_list, port_types):
        # # Get name of port, and use as index to assign to ports ContentAddressableList
        # default_name = port_type._assign_default_port_Name(port_type)
        # name = default_name or None

        port = _instantiate_port(port_type=port_type,
                                   owner=owner,
                                   reference_value=reference_value[index],
                                   reference_value_name=reference_value_name,
                                   port_spec=port_spec,
                                   # name=name,
                                   context=context)
        # automatically generate projections (e.g. when an InputPort is specified by the OutputPort of another mech)
        for proj in port.path_afferents:
            owner.aux_components.append(proj)

        # KDM 12/3/19: this depends on name setting for InputPorts that
        # ensures there are no duplicates. If duplicates exist, ports
        # will be overwritten
        # be careful of:
        #   test_rumelhart_semantic_network_sequential
        #   test_mix_and_match_input_sources
        ports[port.name] = port

    return ports

@tc.typecheck
def _instantiate_port(port_type:_is_port_class,           # Port's type
                       owner:tc.any(Mechanism, Projection),  # Port's owner
                       reference_value,                      # constraint for Port's value and default for variable
                       name:tc.optional(str)=None,           # port's name if specified
                       variable=None,                        # used as default value for port if specified
                       params=None,                          # port-specific params
                       prefs=None,
                       context=None,
                       **port_spec):                        # captures *port_spec* arg and any other non-standard ones
    """Instantiate a Port of specified type, with a value that is compatible with reference_value

    This is the interface between the various ways in which a port can be specified and the Port's constructor
        (see list below, and `Port_Specification` in docstring above).
    It calls _parse_port_spec to:
        create a Port specification dictionary (the canonical form) that can be passed to the Port's constructor;
        place any Port subclass-specific params in the params entry;
        call _parse_port_specific_specs to parse and validate the values of those params
    It checks that the Port's value is compatible with the reference value and/or any projection specifications

    # Constraint value must be a number or a list or tuple of numbers
    # (since it is used as the variable for instantiating the requested port)

    If port_spec is a:
    + Port class:
        implement default using reference_value
    + Port object:
        check compatibility of value with reference_value
        check owner is owner; if not, raise exception
    + 2-item tuple:
        assign first item to port_spec
        assign second item to port_params{PROJECTIONS:<projection>}
    + Projection object:
        assign reference_value to value
        assign projection to port_params{PROJECTIONS:<projection>}
    + Projection class (or keyword string constant for one):
        assign reference_value to value
        assign projection class spec to port_params{PROJECTIONS:<projection>}
    + specification dict for Port
        check compatibility of port_value with reference_value

    Returns a Port or None
    """

    # Parse reference value to get actual value (in case it is, itself, a specification dict)
    from psyneulink.core.globals.utilities import is_numeric
    if not is_numeric(reference_value):
        reference_value_dict = _parse_port_spec(owner=owner,
                                                 port_type=port_type,
                                                 port_spec=reference_value,
                                                 value=None,
                                                 params=None)
        # Its value is assigned to the VARIABLE entry (including if it was originally just a value)
        reference_value = reference_value_dict[VARIABLE]

    parsed_port_spec = _parse_port_spec(port_type=port_type,
                                          owner=owner,
                                          reference_value=reference_value,
                                          name=name,
                                          variable=variable,
                                          params=params,
                                          prefs=prefs,
                                          context=context,
                                          **port_spec)

    # PORT SPECIFICATION IS A Port OBJECT ***************************************
    # Validate and return

    # - check that its value attribute matches the reference_value
    # - check that it doesn't already belong to another owner
    # - if either fails, assign default Port

    if isinstance(parsed_port_spec, Port):

        port = parsed_port_spec

        # Port initialization was deferred (owner or reference_value was missing), so
        #    assign owner, variable, and/or reference_value if they were not already specified
        if port.initialization_status == ContextFlags.DEFERRED_INIT:
            if not port._init_args[OWNER]:
                port._init_args[OWNER] = owner
            # If variable was not specified by user or Port's constructor:
            if VARIABLE not in port._init_args or port._init_args[VARIABLE] is None:
                # If call to _instantiate_port specified variable, use that
                if variable is not None:
                    port._init_args[VARIABLE] = variable
                # Otherwise, use Port's owner's default variable as default if it has one
                elif len(owner.defaults.variable):
                    port._init_args[VARIABLE] = owner.defaults.variable[0]
                # If all else fails, use Port's own defaults.variable
                else:
                    port._init_args[VARIABLE] = port.defaults.variable
            if not hasattr(port, REFERENCE_VALUE):
                if REFERENCE_VALUE in port._init_args and port._init_args[REFERENCE_VALUE] is not None:
                    port.reference_value = port._init_args[REFERENCE_VALUE]
                else:
                    # port.reference_value = owner.defaults.variable[0]
                    port.reference_value = port._init_args[VARIABLE]
            port._deferred_init(context=context)

        # # FIX: 10/3/17 - CHECK THE FOLLOWING BY CALLING PORT-SPECIFIC METHOD?
        # # FIX: DO THIS IN _parse_connection_specs?
        # # *reference_value* arg should generally be a constraint for the value of the Port;  however,
        # #     if port_spec is a Projection, and method is being called from:
        # #         InputPort, reference_value should be the projection's value;
        # #         ParameterPort, reference_value should be the projection's value;
        # #         OutputPort, reference_value should be the projection's variable
        # # variable:
        # #   InputPort: set of projections it receives
        # #   ParameterPort: value of its sender
        # #   OutputPort: _parse_output_port_variable()
        # # FIX: ----------------------------------------------------------

        # FIX: THIS SHOULD ONLY APPLY TO InputPort AND ParameterPort; WHAT ABOUT OutputPort?
        # Port's assigned value is incompatible with its reference_value (presumably its owner Mechanism's variable)
        reference_value = reference_value if reference_value is not None else port.reference_value
        if not iscompatible(port.defaults.value, reference_value):
            raise PortError("{}'s value attribute ({}) is incompatible with the {} ({}) of its owner ({})".
                             format(port.name, port.defaults.value, REFERENCE_VALUE, reference_value, owner.name))

        # Port has already been assigned to an owner
        if port.owner is not None and port.owner is not owner:
            raise PortError("Port {} does not belong to the owner for which it is specified ({})".
                             format(port.name, owner.name))
        return port

    # PORT SPECIFICATION IS A Port specification dictionary ***************************************
    #    so, call constructor to instantiate Port

    port_spec_dict = parsed_port_spec

    port_spec_dict.pop(VALUE, None)

    # FIX: 2/25/18  GET REFERENCE_VALUE FROM REFERENCE_DICT?
    # Get reference_value
    if port_spec_dict[REFERENCE_VALUE] is None:
        port_spec_dict[REFERENCE_VALUE] = reference_value
        if reference_value is None:
            port_spec_dict[REFERENCE_VALUE] = port_spec_dict[VARIABLE]

    #  Convert reference_value to np.array to match port_variable (which, as output of function, will be an np.array)
    if port_spec_dict[REFERENCE_VALUE] is not None:
        port_spec_dict[REFERENCE_VALUE] = convert_to_np_array(port_spec_dict[REFERENCE_VALUE], 1)

    # INSTANTIATE PORT:

    # IMPLEMENTATION NOTE:
    # - setting prefs=NotImplemented causes TYPE_DEFAULT_PREFERENCES to be assigned (from BasePreferenceSet)
    # - alternative would be prefs=owner.prefs, causing port to inherit the prefs of its owner;
    port_type = port_spec_dict.pop(PORT_TYPE, None)
    if REFERENCE_VALUE_NAME in port_spec_dict:
        del port_spec_dict[REFERENCE_VALUE_NAME]
    if port_spec_dict[PARAMS] and REFERENCE_VALUE_NAME in port_spec_dict[PARAMS]:
        del port_spec_dict[PARAMS][REFERENCE_VALUE_NAME]

    # Implement default Port
    port = port_type(**port_spec_dict, context=context)

# FIX LOG: ADD NAME TO LIST OF MECHANISM'S VALUE ATTRIBUTES FOR USE BY LOGGING ENTRIES
    # This is done here to register name with Mechanism's portValues[] list
    # It must be consistent with value setter method in Port
# FIX LOG: MOVE THIS TO MECHANISM PORT __init__ (WHERE IT CAN BE KEPT CONSISTENT WITH setter METHOD??
#      OR MAYBE JUST REGISTER THE NAME, WITHOUT SETTING THE
# FIX: 2/17/17:  COMMENTED THIS OUT SINCE IT CREATES AN ATTRIBUTE ON OWNER THAT IS NAMED <port.name.value>
#                NOT SURE WHAT THE PURPOSE IS
#     setattr(owner, port.name+'.value', port.value)

    port._validate_against_reference_value(reference_value)

    return port


def _parse_port_type(owner, port_spec):
    """Determine Port type for port_spec and return type

    Determine type from context and/or type of port_spec if the latter is not a `Port <Port>` or `Mechanism
    <Mechanism>`.
    """

    # Port class reference
    if isinstance(port_spec, Port):
        return type(port_spec)

    # keyword for a Port or name of a standard_output_port or of Port itself
    if isinstance(port_spec, str):

        # Port keyword
        if port_spec in port_type_keywords:
            return getattr(sys.modules['PsyNeuLink.Components.Ports.' + port_spec], port_spec)

        # Try as name of Port
        for port_attr in [INPUT_PORTS, PARAMETER_PORTS, OUTPUT_PORTS]:
            port_list = getattr(owner, port_attr)
            try:
                port = port_list[port_spec]
                return port.__class__
            except TypeError:
                pass

        # standard_output_port
        if hasattr(owner, STANDARD_OUTPUT_PORTS):
            # check if string matches the name entry of a dict in standard_output_ports
            # item = next((item for item in owner.standard_output_ports.names if port_spec is item), None)
            # if item is not None:
            #     # assign dict to owner's output_port list
            #     return owner.standard_output_ports.get_dict(port_spec)
            # from psyneulink.core.Components.Ports.OutputPort import StandardOutputPorts
            if owner.standard_output_ports.get_port_dict(port_spec):
                from psyneulink.core.components.Ports.OutputPort import OutputPort
                return OutputPort

    # Port specification dict
    if isinstance(port_spec, dict):
        if PORT_TYPE in port_spec:
            if not inspect.isclass(port_spec[PORT_TYPE]) and issubclass(port_spec[PORT_TYPE], Port):
                raise PortError("PORT entry of port specification for {} ({})"
                                 "is not a Port or type of Port".
                                 format(owner.name, port_spec[PORT]))
            return port_spec[PORT_TYPE]

    raise PortError("{} is not a legal Port specification for {}".format(port_spec, owner.name))


PORT_SPEC_INDEX = 0

# FIX: CHANGE EXPECTATION OF *PROJECTIONS* ENTRY TO BE A SET OF TUPLES WITH THE WEIGHT AND EXPONENT FOR IT
#          THESE CAN BE USED BY THE InputPort's LinearCombination Function
#          (AKIN TO HOW THE MECHANISM'S FUNCTION COMBINES InputPort VALUES)
#          THIS WOULD ALLOW FULLY GENEREAL (HIEARCHICALLY NESTED) ALGEBRAIC COMBINATION OF INPUT VALUES TO A MECHANISM
@tc.typecheck
def _parse_port_spec(port_type=None,
                      owner=None,
                      reference_value=None,
                      name=None,
                      variable=None,
                      value=None,
                      params=None,
                      prefs=None,
                      context=None,
                      **port_spec):
    """Parse Port specification and return either Port object or Port specification dictionary

    If port_spec is or resolves to a Port object, returns Port object.
    Otherwise, return Port specification dictionary using any arguments provided as defaults
    Warn if variable is assigned the default value, and verbosePref is set on owner.
    *value* arg should generally be a constraint for the value of the Port;  however,
        if port_spec is a Projection, and method is being called from:
            InputPort, value should be the projection's value;
            ParameterPort, value should be the projection's value;
            OutputPort, value should be the projection's variable

    If a Port specification dictionary is specified in the *port_specs* argument,
       its entries are moved to standard_args, replacing any that are there, and they are deleted from port_specs;
       any remaining entries are passed to _parse_port_specific_specs and placed in params.
    This gives precedence to values of standard args specified in a Port specification dictionary
       (e.g., by the user) over any explicitly specified in the call to _instantiate_port.
    The standard arguments (from standard_args and/or a Port specification dictonary in port_specs)
        are placed assigned to port_dict, as defaults for the Port specification dictionary returned by this method.
    Any item in *port_specs* OTHER THAN a Port specification dictionary is placed in port_spec_arg
       is parsed and/or validated by this method.
    Values in standard_args (i.e., specified in the call to _instantiate_port) are used to validate a port specified
       in port_spec_arg;
       - if the Port is an existing one, the standard_arg values are assigned to it;
       - if port_spec_arg specifies a new Port, the values in standard_args are used as defaults;  any specified
          in the port_spec_arg specification are used
    Any arguments to _instantiate_ports that are not standard arguments (in standard_args) or a port_specs_arg
       generate a warning and are ignored.

    Return either Port object or Port specification dictionary
    """
    from psyneulink.core.components.projections.projection \
        import _is_projection_spec, _parse_projection_spec, _parse_connection_specs, ProjectionTuple, ProjectionError
    from psyneulink.core.components.ports.modulatorysignals.modulatorysignal import _is_modulatory_spec
    from psyneulink.core.components.mechanisms.modulatory.modulatorymechanism import ModulatoryMechanism_Base
    from psyneulink.core.components.projections.projection import _get_projection_value_shape
    from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection


    # Get all of the standard arguments passed from _instantiate_port (i.e., those other than port_spec) into a dict
    standard_args = get_args(inspect.currentframe())

    PORT_SPEC_ARG = 'port_spec'
    port_specification = None
    port_specific_args = {}

    # If there is a port_specs arg passed from _instantiate_port:
    if PORT_SPEC_ARG in port_spec:

        # If it is a Port specification dictionary
        if isinstance(port_spec[PORT_SPEC_ARG], dict):

            # If the Port specification is a Projection that has a sender already assigned,
            #    then return that Port with the Projection assigned to it
            #    (this occurs, for example, if an instantiated ControlSignal is used to specify a parameter
            try:
                assert len(port_spec[PORT_SPEC_ARG][PROJECTIONS])==1
                projection = port_spec[PORT_SPEC_ARG][PROJECTIONS][0]
                port = projection.sender
                if port.initialization_status == ContextFlags.DEFERRED_INIT:
                    port._init_args[PROJECTIONS] = projection
                else:
                    port._instantiate_projections_to_port(projections=projection, context=context)
                return port
            except:
                pass

            # Use the value of any standard args specified in the Port specification dictionary
            #    to replace those explicitly specified in the call to _instantiate_port (i.e., passed in standard_args)
            #    (use copy so that items in port_spec dict are not deleted when called from _validate_params)
            port_specific_args = port_spec[PORT_SPEC_ARG].copy()
            standard_args.update({key: port_specific_args[key]
                                  for key in port_specific_args
                                  if key in standard_args and port_specific_args[key] is not None})
            # Delete them from the Port specification dictionary, leaving only port-specific items there
            for key in standard_args:
                port_specific_args.pop(key, None)

            try:
                spec = port_spec[PORT_SPEC_ARG]
                port_tuple = [spec[PORT_SPEC_ARG], spec[WEIGHT], spec[EXPONENT]]
                try:
                    port_tuple.append(spec[PROJECTIONS])
                except KeyError:
                    pass
                port_specification = tuple(port_tuple)
            except KeyError:
                pass

        else:
            port_specification = port_spec[PORT_SPEC_ARG]

        # Delete the Port specification dictionary from port_spec
        del port_spec[PORT_SPEC_ARG]

    if REFERENCE_VALUE_NAME in port_spec:
        del port_spec[REFERENCE_VALUE_NAME]

    if port_spec:
        if owner.verbosePref:
            print(f'Args other than standard args and port_spec were in _instantiate_port ({port_spec}).')
        port_spec.update(port_specific_args)
        port_specific_args = port_spec

    port_dict = standard_args
    context = port_dict.pop(CONTEXT, None)
    owner = port_dict[OWNER]
    port_type = port_dict[PORT_TYPE]
    reference_value = port_dict[REFERENCE_VALUE]
    variable = port_dict[VARIABLE]
    params = port_specific_args

    # Validate that port_type is a Port class
    if isinstance(port_type, str):
        try:
            port_type = PortRegistry[port_type].subclass
        except KeyError:
            raise PortError("{} specified as a string (\'{}\') must be the name of a sublcass of {}".
                             format(PORT_TYPE, port_type, Port.__name__))
        port_dict[PORT_TYPE] = port_type
    elif not inspect.isclass(port_type) or not issubclass(port_type, Port):
        raise PortError("\'port_type\' arg ({}) must be a sublcass of {}".format(port_type,
                                                                                   Port.__name__))
    port_type_name = port_type.__name__

    # EXISTING PORTS

    # Determine whether specified Port is one to be instantiated or to be connected with,
    #    and validate that it is consistent with any standard_args specified in call to _instantiate_port

    # function; try to resolve to a value
    if isinstance(port_specification, types.FunctionType):
        port_specification = port_specification()

    # ModulatorySpecification of some kind
    if _is_modulatory_spec(port_specification):
        # If it is a ModulatoryMechanism specification, get its ModulatorySignal class
        # (so it is recognized by _is_projection_spec below (Mechanisms are not for secondary reasons)
        if isinstance(port_specification, type) and issubclass(port_specification, ModulatoryMechanism_Base):
            port_specification = port_specification.outputPortTypes
            # IMPLEMENTATION NOTE:  The following is to accomodate GatingSignals on ControlMechanism
            # FIX: TRY ELIMINATING SIMILAR HANDLING IN Projection (and OutputPort?)
            # FIX: AND ANY OTHER PLACES WHERE LISTS ARE DEALT WITH
            if isinstance(port_specification, list):
                # If modulatory projection is specified as a Mechanism that allows more than one type of OutputPort
                #   (e.g., ModulatoryMechanism allows either ControlSignals or GatingSignals together with standard
                #   OutputPorts) make sure that only one of these is appropriate for port to be modulated
                #   (port_type.connectswith), otherwise it is ambiguous which to assign as Port_Specification
                specs = [s for s in port_specification if s.__name__ in port_type.connectsWith]
                try:
                    port_specification, = specs
                except ValueError:
                    assert False, \
                        f"PROGRAM ERROR:  More than one {Port.__name__} type found ({specs})" \
                            f"that can be specificied as a modulatory {Projection.__name__} to {port_type}"

        projection = port_type

    # Port or Mechanism object specification:
    if isinstance(port_specification, (Mechanism, Port)):

        projection = None

        # Mechanism object:
        if isinstance(port_specification, Mechanism):
            mech = port_specification
            # Instantiating Port of specified Mechanism, so get primary Port of port_type
            if mech is owner:
                port_specification = port_type._get_primary_port(port_type, mech)
            # mech used to specify Port to be connected with:
            else:
                port_specification = mech
                projection = port_type

        if port_specification.__class__ == port_type:
            # Make sure that the specified Port belongs to the Mechanism passed in the owner arg
            if port_specification.initialization_status == ContextFlags.DEFERRED_INIT:
                port_owner = port_specification._init_args[OWNER]
            else:
                port_owner = port_specification.owner
            if owner is not None and port_owner is not None and port_owner is not owner:
                try:
                    new_port_specification = port_type._parse_self_port_type_spec(port_type,
                                                                                     owner,
                                                                                     port_specification,
                                                                                     context)
                    port_specification = _parse_port_spec(port_type=port_type,
                                                            owner=owner,
                                                            port_spec=new_port_specification)
                    assert True
                except AttributeError:
                    raise PortError("Attempt to assign a {} ({}) to {} that belongs to another {} ({})".
                                     format(Port.__name__, port_specification.name, owner.name,
                                            Mechanism.__name__, port_owner.name))
            return port_specification

        # Specication is a Port with which connectee can connect, so assume it is a Projection specification
        elif port_specification.__class__.__name__ in port_type.connectsWith + port_type.modulators:
            projection = port_type

        # Re-process with Projection specified
        port_dict = _parse_port_spec(port_type=port_type,
                                       owner=owner,
                                       variable=variable,
                                       value=value,
                                       reference_value=reference_value,
                                       params=params,
                                       prefs=prefs,
                                       context=context,
                                       port_spec=ProjectionTuple(port=port_specification,
                                                                  weight=None,
                                                                  exponent=None,
                                                                  projection=projection))

    # Projection specification (class, object, or matrix value (matrix keyword processed below):
    elif _is_projection_spec(port_specification, include_matrix_spec=False):

        # FIX: 11/12/17 - HANDLE SITUATION IN WHICH projection_spec IS A MATRIX (AND SENDER IS SOMEHOW KNOWN)
        # Parse to determine whether Projection's value is specified
        projection_spec = _parse_projection_spec(port_specification, owner=owner, port_type=port_dict[PORT_TYPE])

        projection_value=None
        sender=None
        matrix=None

        # Projection has been instantiated
        if isinstance(projection_spec, Projection):
            if projection_spec.initialization_status == ContextFlags.INITIALIZED:
            # if projection_spec.initialization_status != ContextFlags.DEFERRED_INIT:
                projection_value = projection_spec.value
            # If deferred_init, need to get sender and matrix to determine value
            else:
                try:
                    sender = projection_spec._init_args[SENDER]
                    matrix = projection_spec._init_args[MATRIX]
                except (KeyError, TypeError):
                    pass
        # Projection specification dict:
        else:
            # Need to get sender and matrix to determine value
            sender = projection_spec[SENDER]
            matrix = projection_spec[MATRIX]

        if sender is not None and matrix is not None and matrix is not AUTO_ASSIGN_MATRIX:
            # Get sender of Projection to determine its value
            from psyneulink.core.components.ports.outputport import OutputPort
            sender = _get_port_for_socket(owner=owner,
                                           connectee_port_type=port_type,
                                           port_spec=sender,
                                           port_types=[OutputPort])
            projection_value = _get_projection_value_shape(sender, matrix)

        reference_value = port_dict[REFERENCE_VALUE]
        # If Port's reference_value is not specified, but Projection's value is, use projection_spec's value
        if reference_value is None and projection_value is not None:
            port_dict[REFERENCE_VALUE] = projection_value
        # If Port's reference_value has been specified, check for compatibility with projection_spec's value
        elif (reference_value is not None and projection_value is not None
            and not iscompatible(reference_value, projection_value)):
            raise PortError("{} of {} ({}) is not compatible with {} of {} ({}) for {}".
                             format(VALUE, Projection.__name__, projection_value, REFERENCE_VALUE,
                                    port_dict[PORT_TYPE].__name__, reference_value, owner.name))

        # Move projection_spec to PROJECTIONS entry of params specification dict (for instantiation of Projection)
        if port_dict[PARAMS] is None:
            port_dict[PARAMS] = {}
        port_dict[PARAMS].update({PROJECTIONS:[port_specification]})

    # string (keyword or name specification)
    elif isinstance(port_specification, str):
        # Check if it is a keyword
        spec = get_param_value_for_keyword(owner, port_specification)
        # A value was returned, so use value of keyword as reference_value
        if spec is not None:
            port_dict[REFERENCE_VALUE] = spec
            # NOTE: (7/26/17 CW) This warning below may not be appropriate, since this routine is run if the
            # matrix parameter is specified as a keyword, which may be intentional.
            if owner.prefs.verbosePref:
                print("{} not specified for {} of {};  reference value ({}) will be used".
                      format(VARIABLE, port_type, owner.name, port_dict[REFERENCE_VALUE]))
        # It is not a keyword, so treat string as the name for the port
        else:
            port_dict[NAME] = port_specification

    # # function; try to resolve to a value
    # elif isinstance(Port_Specification, types.FunctionType):
    #     port_dict[REFERENCE_VALUE] = get_param_value_for_function(owner, Port_Specification)
    #     if port_dict[REFERENCE_VALUE] is None:
    #         raise PortError("PROGRAM ERROR: port_spec for {} of {} is a function ({}), but failed to return a value".
    #                          format(port_type_name, owner.name, Port_Specification))

    # FIX: THIS SHOULD REALLY BE PARSED IN A PORT-SPECIFIC WAY:
    #      FOR InputPort: variable
    #      FOR ParameterPort: default (base) parameter value
    #      FOR OutputPort: index
    #      FOR ModulatorySignal: default value of ModulatorySignal (e.g, allocation or gating policy)
    # value, so use as variable of Port
    elif is_value_spec(port_specification):
        port_dict[REFERENCE_VALUE] = np.atleast_1d(port_specification)

    elif isinstance(port_specification, Iterable) or port_specification is None:

        # Standard port specification dict
        # Warn if VARIABLE was not in dict
        if ((VARIABLE not in port_dict or port_dict[VARIABLE] is None)
                and hasattr(owner, 'prefs') and owner.prefs.verbosePref):
            print("{} missing from specification dict for {} of {};  "
                  "will be inferred from context or the default ({}) will be used".
                  format(VARIABLE, port_type, owner.name, port_dict))

        if isinstance(port_specification, (list, set)):
            port_specific_specs = ProjectionTuple(port=port_specification,
                                              weight=None,
                                              exponent=None,
                                              projection=port_type)

        # Port specification is a tuple
        elif isinstance(port_specification, tuple):

            # 1st item of tuple is a tuple (presumably a (Port name, Mechanism) tuple),
            #    so parse to get specified Port (any projection spec should be included as 4th item of outer tuple)
            if isinstance(port_specification[0],tuple):
                proj_spec = _parse_connection_specs(connectee_port_type=port_type,
                                                    owner=owner,
                                                    connections=port_specification[0])
                port_specification = (proj_spec[0].port,) + port_specification[1:]

            # Reassign tuple for handling by _parse_port_specific_specs
            port_specific_specs = port_specification

        # Otherwise, just pass params to Port subclass
        else:
            port_specific_specs = params

        if port_specific_specs:
            port_spec, params = port_type._parse_port_specific_specs(port_type,
                                                                         owner=owner,
                                                                         port_dict=port_dict,
                                                                         port_specific_spec = port_specific_specs)
            # Port subclass returned a port_spec, so call _parse_port_spec to parse it
            if port_spec is not None:
                port_dict = _parse_port_spec(context=context, port_spec=port_spec, **standard_args)

            # Move PROJECTIONS entry to params
            if PROJECTIONS in port_dict:
                if not isinstance(port_dict[PROJECTIONS], list):
                    port_dict[PROJECTIONS] = [port_dict[PROJECTIONS]]
                params[PROJECTIONS].append(port_dict[PROJECTIONS])

            # MECHANISM entry specifies Mechanism; <PORTS> entry has names of its Ports
            #           MECHANISM: <Mechanism>, <PORTS>:[<Port.name>, ...]}
            if MECHANISM in port_specific_args:

                if PROJECTIONS not in params:
                    if NAME in spec:
                        # substitute into tuple spec
                        params[PROJECTIONS] = (spec[NAME], params[MECHANISM])
                    else:
                        params[PROJECTIONS] = []

                mech = port_specific_args[MECHANISM]
                if not isinstance(mech, Mechanism):
                    raise PortError("Value of the {} entry ({}) in the "
                                     "specification dictionary for {} of {} is "
                                     "not a {}".format(MECHANISM,
                                                       mech,
                                                       port_type.__name__,
                                                       owner.name,
                                                       Mechanism.__name__))

                # For Ports with which the one being specified can connect:
                for PORTS in port_type.connectsWithAttribute:

                    if PORTS in port_specific_args:
                        port_specs = port_specific_args[PORTS]
                        port_specs = port_specs if isinstance(port_specs, list) else [port_specs]
                        for port_spec in port_specs:
                            # If Port is a tuple, get its first item as port
                            port = port_spec[0] if isinstance(port_spec, tuple) else port_spec
                            try:
                                port_attr = getattr(mech, PORTS)
                                port = port_attr[port]
                            except:
                                name = owner.name if 'unnamed' not in owner.name else 'a ' + owner.__class__.__name__
                                raise PortError("Unrecognized name ({}) for {} "
                                                 "of {} in specification of {} "
                                                 "for {}".format(port,
                                                                 PORTS,
                                                                 mech.name,
                                                                 port_type.__name__,
                                                                 name))
                            # If port_spec was a tuple, put port back in as its first item and use as projection spec
                            if isinstance(port_spec, tuple):
                                port = (port,) + port_spec[1:]
                            params[PROJECTIONS].append(port)
                        # Delete <PORTS> entry as it is not a parameter of a Port
                        del port_specific_args[PORTS]

                # Delete MECHANISM entry as it is not a parameter of a Port
                del port_specific_args[MECHANISM]

            # FIX: 11/4/17 - MAY STILL NEED WORK:
            # FIX:   PROJECTIONS FROM UNRECOGNIZED KEY ENTRY MAY BE REDUNDANT OR CONFLICT WITH ONE ALREADY IN PARAMS
            # FIX:   NEEDS TO BE BETTER COORDINATED WITH _parse_port_specific_specs
            # FIX:   REGARDING WHAT IS IN port_specific_args VS params (see REF_VAL_NAME BRANCH)
            # FIX:   ALSO, ??DOES PROJECTIONS ENTRY BELONG IN param OR port_dict?
            # Check for single unrecognized key in params, used for {<Port_Name>:[<projection_spec>,...]} format
            unrecognized_keys = [key for key in port_specific_args if key not in port_type.portAttributes]
            if unrecognized_keys:
                if len(unrecognized_keys)==1:
                    key = unrecognized_keys[0]
                    port_dict[NAME] = key
                    # KDM 12/24/19: in some cases, params[PROJECTIONS] is
                    # already parsed into a ProjectionTuple, and this assignment
                    # will replace it with an unparsed ndarray which causes a
                    # ProjectionError in _parse_connection_specs
                    if (
                        PROJECTIONS not in params
                        or not any([
                            isinstance(x, ProjectionTuple)
                            for x in params[PROJECTIONS]
                        ])
                    ):
                        params[PROJECTIONS] = port_specific_args[key]
                        del port_specific_args[key]
                else:
                    raise PortError("There is more than one entry of the {} "
                                     "specification dictionary for {} ({}) "
                                     "that is not a keyword; there should be "
                                     "only one (used to name the Port, with a "
                                     "list of Projection specifications".
                                     format(port_type.__name__,
                                            owner.name,
                                            ", ".join([s for s in list(port_specific_args.keys())])))

            for param in port_type.portAttributes:
                # KDM 12/24/19: below is meant to skip overwriting an already
                # parsed ProjectionTuple just as in the section above
                if param in port_specific_args:
                    try:
                        param_value_is_tuple = any([
                            isinstance(x, ProjectionTuple)
                            for x in params[param]
                        ])
                    except TypeError:
                        param_value_is_tuple = isinstance(
                            params[param],
                            ProjectionTuple
                        )
                    except KeyError:
                        # param may not be in port_specific_args, and in this
                        # case use the default/previous behavior
                        param_value_is_tuple = False

                    if param not in params or not param_value_is_tuple:
                        params[param] = port_specific_args[param]

            if PROJECTIONS in params and params[PROJECTIONS] is not None:
                #       (E.G., WEIGHTS AND EXPONENTS FOR InputPort AND INDEX FOR OutputPort)
                # Get and parse projection specifications for the Port
                params[PROJECTIONS] = _parse_connection_specs(port_type, owner, params[PROJECTIONS])

            # Update port_dict[PARAMS] with params
            if port_dict[PARAMS] is None:
                port_dict[PARAMS] = {}
            port_dict[PARAMS].update(params)

    else:
        # if owner.verbosePref:
        #     warnings.warn("PROGRAM ERROR: port_spec for {} of {} is an unrecognized specification ({})".
        #                  format(port_type_name, owner.name, port_spec))
        # return
        raise PortError("PROGRAM ERROR: port_spec for {} of {} is an unrecognized specification ({})".
                         format(port_type_name, owner.name, port_specification))

    # If variable is none, use value:
    if port_dict[VARIABLE] is None:
        if port_dict[VALUE] is not None:
            # TODO: be careful here - if the port spec has a function that
            # changes the shape of its variable, this will be incorrect
            port_dict[VARIABLE] = port_dict[VALUE]
        else:
            port_dict[VARIABLE] = port_dict[REFERENCE_VALUE]

    # get the Port's value from the spec function if it exists,
    # otherwise we can assume there is a default function that does not
    # affect the shape, so it matches variable
    # FIX: JDC 2/21/18 PROBLEM IS THAT, IF IT IS AN InputPort, THEN EITHER _update MUST BE CALLED
    # FIX:    OR VARIABLE MUST BE WRAPPED IN A LIST, ELSE LINEAR COMB MAY TREAT A 2D ARRAY
    # FIX:    AS TWO ITEMS TO BE COMBINED RATHER THAN AS A 2D ARRAY
    # KDM 6/7/18: below this can end up assigning to the port a variable of the same shape as a default function
    #   (because when calling the function, _check_args is called and if given None, will fall back to instance or
    #   class defaults)
    try:
        spec_function = port_dict[PARAMS][FUNCTION]
        # if isinstance(spec_function, Function):
        if isinstance(spec_function, (Function, types.FunctionType, types.MethodType)):
            spec_function_value = port_type._get_port_function_value(owner, spec_function, port_dict[VARIABLE])
        elif inspect.isclass(spec_function) and issubclass(spec_function, Function):
            try:
                spec_function = spec_function(**port_dict[PARAMS][FUNCTION_PARAMS])
            except (KeyError, TypeError):
                spec_function = spec_function()
            spec_function_value = port_type._get_port_function_value(owner, spec_function, port_dict[VARIABLE])
        else:
            raise PortError('port_spec value for FUNCTION ({0}) must be a function, method, '
                             'Function class or instance of one'.
                             format(spec_function))
    except (KeyError, TypeError):
        spec_function_value = port_type._get_port_function_value(owner, None, port_dict[VARIABLE])
        spec_function = port_type.class_defaults.function


    # Assign value based on variable if not specified
    if port_dict[VALUE] is None:
        port_dict[VALUE] = spec_function_value
    # Otherwise, make sure value returned by spec function is same as one specified for Port's value
    # else:
    #     if not np.asarray(port_dict[VALUE]).shape == np.asarray(spec_function_value).shape:
    #         port_Name = port_dict[NAME] or 'unnamed'
    #         raise PortError('port_spec value ({}) specified for {} {} of {} is not compatible with '
    #                          'the value ({}) computed from the port_spec function ({})'.
    #                          format(port_dict[VALUE], port_Name, port_type.__name__,
    #                                 port_dict[OWNER].name, spec_function_value, spec_function))

    if port_dict[REFERENCE_VALUE] is not None and not iscompatible(port_dict[VALUE], port_dict[REFERENCE_VALUE]):
        raise PortError("Port value ({}) does not match reference_value ({}) for {} of {})".
                         format(port_dict[VALUE], port_dict[REFERENCE_VALUE], port_type.__name__, owner.name))

    return port_dict


# FIX: REPLACE mech_port_attribute WITH DETERMINATION FROM port_type
# FIX:          ONCE PORT CONNECTION CHARACTERISTICS HAVE BEEN IMPLEMENTED IN REGISTRY
@tc.typecheck
def _get_port_for_socket(owner,
                          connectee_port_type:tc.optional(_is_port_class)=None,
                          port_spec=None,
                          port_types:tc.optional(tc.any(list, _is_port_class))=None,
                          mech:tc.optional(Mechanism)=None,
                          mech_port_attribute:tc.optional(tc.any(str, list))=None,
                          projection_socket:tc.optional(tc.any(str, set))=None):
    """Take some combination of Mechanism, port name (string), Projection, and projection_socket, and return
    specified Port(s)

    If port_spec is:
        Port name (str), then *mech* and *mech_port_attribute* args must be specified
        Mechanism, then *port_type* must be specified; primary Port is returned
        Projection, *projection_socket* arg must be specified;
                    Projection must be instantiated or in deferred_init, with projection_socket attribute assigned

    IMPLEMENTATION NOTES:
    Currently does not support Port specification dict (referenced Port must be instantiated)
    Currently does not support Projection specification using class or Projection specification dict
        (Projection must be instantiated, or in deferred_init status with projection_socket assigned)

    Returns a Port if it can be resolved, or list of allowed Port types if not.
    """
    from psyneulink.core.components.projections.projection import \
        _is_projection_spec, _validate_connection_request, _parse_projection_spec
    from psyneulink.core.globals.utilities import is_matrix

    # # If the mech_port_attribute specified has more than one item, get the primary one
    # if isinstance(mech_port_attribute, list):
    #     mech_port_attribute = mech_port_attribute[0]

    # port_types should be a list, and port_type its first (or only) item
    if isinstance(port_types, list):
        port_type = port_types[0]
    else:
        port_type = port_types
        port_types = [port_types]

    port_type_names = ", ".join([s.__name__ for s in port_types])

    # Return Port itself if it is an instantiated Port
    if isinstance(port_spec, Port):
        return port_spec

    # Return port_type (Class) if port_spec is:
    #    - an allowable Port type for the projection_socket
    #    - a projection keyword (e.g., 'LEARNING' or 'CONTROL', and it is consistent with projection_socket
    # Otherwise, return list of allowable Port types for projection_socket (if port_spec is a Projection type)
    if _is_projection_spec(port_spec):

        # These specifications require that a particular Port be specified to assign its default Projection type
        if ((is_matrix(port_spec) or (isinstance(port_spec, dict) and PROJECTION_TYPE not in port_spec))):
            for st in port_types:
                try:
                    proj_spec = _parse_projection_spec(port_spec, owner=owner, port_type=st)
                    if isinstance(proj_spec, Projection):
                        proj_type = proj_spec.__class__
                    else:
                        proj_type = proj_spec[PROJECTION_TYPE]
                except:
                    continue
        else:
            proj_spec = _parse_projection_spec(port_spec, owner=owner, port_type=port_type)
            if isinstance(proj_spec, Projection):
                proj_type = proj_spec.__class__
            else:
                proj_type = proj_spec[PROJECTION_TYPE]

        # Get Port type if it is appropriate for the specified socket of the
        #  Projection's type
        s = next((s for s in port_types if
                  s.__name__ in getattr(proj_type.sockets, projection_socket)),
                 None)
        if s:
            try:
                # Return Port associated with projection_socket if proj_spec is an actual Projection
                port = getattr(proj_spec, projection_socket)
                return port
            except AttributeError:
                # Otherwise, return first port_type (s)
                return s

        # FIX: 10/3/17 - ??IS THE FOLLOWING NECESSARY?  ??HOW IS IT DIFFERENT FROM ABOVE?
        # Otherwise, get Port types that are allowable for that projection_socket
        elif inspect.isclass(proj_type) and issubclass(proj_type, Projection):
            projection_socket_port_Names = getattr(proj_type.sockets, projection_socket)
            projection_socket_port_types = [PortRegistry[name].subclass for name in projection_socket_port_Names]
            return projection_socket_port_types
        else:
            assert False
            # return port_type

    # Get port by name
    if isinstance(port_spec, str):

        if mech is None:
            raise PortError("PROGRAM ERROR: A {} must be specified to specify its {} ({}) by name".
                             format(Mechanism.__name__, Port.__name__, port_spec))
        if mech_port_attribute is None:
            raise PortError("PROGRAM ERROR: The attribute of {} that holds the requested Port ({}) must be specified".
                             format(mech.name, port_spec))
        for attr in mech_port_attribute:
            try:
                portListAttribute = getattr(mech, attr)
                port = portListAttribute[port_spec]
            except AttributeError:
                portListAttribute = None
            except (KeyError, TypeError):
                port = None
            else:
                break
        if portListAttribute is None:
            raise PortError("PROGRAM ERROR: {} attribute(s) not found on {}'s type ({})".
                             format(mech_port_attribute, mech.name, mech.__class__.__name__))
        if port is None:
            if len(mech_port_attribute)==1:
                attr_name = mech_port_attribute[0] + " attribute"
            else:
                attr_name = " or ".join(f"{repr(attr)}" for (attr) in mech_port_attribute) + " attributes"
            raise PortError(f"{mech.name} does not have a {Port.__name__} named \'{port_spec}\' in its {attr_name}.")

    # Get primary Port of specified type
    elif isinstance(port_spec, Mechanism):

        if port_type is None:
            raise PortError("PROGRAM ERROR: The type of Port requested for {} must be specified "
                             "to get its primary Port".format(port_spec.name))
        try:
            port = port_type._get_primary_port(port_type, port_spec)
            # Primary Port for Mechanism specified in port_spec is not compatible
            # with owner's Port for which a connection is being specified
            if port.__class__.__name__ not in connectee_port_type.connectsWith:
                from psyneulink.core.components.projections.projection import ProjectionError
                raise ProjectionError(f"Primary {port_type.__name__} of {port_spec.name} ({port.name}) cannot be "
                                      f"used "
                                      f"as a {projection_socket} of a {Projection.__name__} "
                                      f"{PROJECTION_DIRECTION[projection_socket]} {connectee_port_type.__name__} of "
                                      f"{owner.name}")
        except PortError:
            if mech_port_attribute:
                try:
                    port = getattr(port_spec, mech_port_attribute)[0]
                except:
                    raise PortError("{} does not seem to have an {} attribute"
                                     .format(port_spec.name, mech_port_attribute))
            for attr in mech_port_attribute:
                try:
                    port = getattr(port_spec, attr)[0]
                except :
                    port = None
                else:
                    break
                if port is None:
                    raise PortError("PROGRAM ERROR: {} attribute(s) not found on {}'s type ({})".
                                     format(mech_port_attribute, mech.name, mech.__class__.__name__))

    # # Get
    # elif isinstance(port_spec, type) and issubclass(port_spec, Mechanism):


    # Get port from Projection specification (exclude matrix spec in test as it can't be used to determine the port)
    elif _is_projection_spec(port_spec, include_matrix_spec=False):
        _validate_connection_request(owner=owner,
                                     connect_with_ports=port_type,
                                     projection_spec=port_spec,
                                     projection_socket=projection_socket)
        if isinstance(port_spec, Projection):
            port = port_spec.socket_assignments[projection_socket]
            if port is None:
                port = port_type
        else:
            return port_spec

    else:
        if port_spec is None:
            raise PortError("PROGRAM ERROR: Missing port specification for {}".format(owner.name))
        else:
            raise PortError("Unrecognized port specification: {} for {}".format(port_spec, owner.name))

    return port


def _is_legal_port_spec_tuple(owner, port_spec, port_type_name=None):

    from psyneulink.core.components.projections.projection import _is_projection_spec

    port_type_name = port_type_name or PORT

    if len(port_spec) != 2:
        raise PortError("Tuple provided as port_spec for {} of {} ({}) must have exactly two items".
                         format(port_type_name, owner.name, port_spec))
    if not (_is_projection_spec(port_spec[1]) or
                # IMPLEMENTATION NOTE: Mechanism or Port allowed as 2nd item of tuple or
                #                      string (parameter name) as 1st and Mechanism as 2nd
                #                      to accommodate specification of param for ControlSignal
                isinstance(port_spec[1], (Mechanism, Port))
                           or (isinstance(port_spec[0], Mechanism) and
                                       port_spec[1] in port_spec[0]._parameter_ports)):
        raise PortError("2nd item of tuple in port_spec for {} of {} ({}) must be a specification "
                         "for a Mechanism, Port, or Projection".
                         format(port_type_name, owner.__class__.__name__, port_spec[1]))


def _merge_param_dicts(source, specific, general, remove_specific=True, remove_general=False):
    """Search source dict for specific and general dicts, merge specific with general, and return merged

    Used to merge a subset of dicts in runtime_params (that may have several dicts);
        for example: only the MAPPING_PROJECTION_PARAMS (specific) with PROJECTION_PARAMS (general)
    Allows dicts to be referenced by name (e.g., paramName) rather than by object
    Searches source dict for specific and general dicts
    - if both are found, merges them, with entries from specific overwriting any duplicates in general
    - if only one is found, returns just that dict
    - if neither are found, returns empty dict
    remove_specific and remove_general args specify whether to remove those from source;
        if specific and/or general are specified by name (i.e., as keys in source the value of which are subdicts),
        then the entyr with the subdict is removed from source;  if they are specified as dicts,
        then the corresponding entires  are removed from source.

    Arguments
    _________

    source : dict
        container dict (entries are dicts); search entries for specific and general dicts

    specific : dict or str)
        if str, use as key to look for specific dict in source, and check that it is a dict

    general : dict or str
        if str, use as key to look for general dict in source, and check that it is a dict

    Returns
    -------

    merged: dict
    """

    # Validate source as dict
    if not source:
        return {}
    if not isinstance(source, dict):
        raise PortError("merge_param_dicts: source {0} must be a dict".format(source))

    # Get specific and make sure it is a dict
    if isinstance(specific, str):
        try:
            specific_name = specific
            specific = source[specific]
        except (KeyError, TypeError):
            specific = {}
    if not isinstance(specific, dict):
        raise PortError("merge_param_dicts: specific {specific} must be dict or the name of one in {source}.")

    # Get general and make sure it is a dict
    if isinstance(general, str):
        try:
            general_name = general
            general = source[general]
        except (KeyError, TypeError):
            general = {}
    if not isinstance(general, dict):
        raise PortError("merge_param_dicts: general {general} must be dict or the name of one in {source}.")

    general.update(specific)

    if remove_specific:
        try:
            source.pop(specific_name, None)
        except ValueError:
            for entry in specific:
                source.pop(entry, None)
    if remove_general:
        try:
            source.pop(general_name, None)
        except ValueError:
            for entry in general:
                source.pop(entry, None)

    return general
