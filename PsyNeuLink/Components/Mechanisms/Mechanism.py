# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ****************************************  MECHANISM MODULE ***********************************************************

"""
..
    * :ref:`Mechanism_Overview`
    * :ref:`Mechanism_Creation`
    * :ref:`Mechanism_Structure`
     * :ref:`Mechanism_Function`
     * :ref:`Mechanism_States`
        * :ref:`Mechanism_InputStates`
        * :ref:`Mechanism_ParameterStates`
        * :ref:`Mechanism_OutputStates`
     * :ref:`Mechanism_Attributes`
     * :ref:`Mechanism_Role_In_Processes_And_Systems`
    * :ref:`Mechanism_Execution`
     * :ref:`Mechanism_Runtime_Parameters`
    * :ref:`Mechanism_Class_Reference`


.. _Mechanism_Overview:

Overview
--------

A Mechanism takes an input, transforms it in some way, and makes the result available as its output.  There are two
types of Mechanisms in PsyNeuLink:

    * `ProcessingMechanisms <ProcessingMechanism>` aggregrate the input they receive from other Mechanisms, and/or the
      input to the `Process` or `System` to which they belong, transform it in some way, and
      provide the result as input to other Mechanisms in the Process or System, or as the output for a Process or
      System itself.  There are a variety of different types of ProcessingMechanism, that accept various forms of
      input and transform them in different ways (see `ProcessingMechanisms <ProcessingMechanism>` for a list).
    ..
    * `AdaptiveMechanisms <AdaptiveMechanism>` monitor the output of one or more other Mechanisms, and use this
      to modulate the parameters of other Mechanisms or Projections.  There are three basic AdaptiveMechanisms:

      * `LearningMechanisms <LearningMechanism>` - these receive training (target) values, and compare them with the
        output of a Mechanism to generate `LearningSignals <LearningSignal>` that are used to modify `MappingProjections
        <MappingProjection>` (see `learning <Process_Learning>`).
      |
      * `ControlMechanisms <ControlMechanism>` - these evaluate the output of a specified set of Mechanisms, and
        generate `ControlSignals <ControlSignal>` used to modify the parameters of those or other Mechanisms.
      |
      * `GatingMechanisms <GatingMechanism>` - these use their input(s) to determine whether and how to modify the
        `value <State_Base.value>` of the `InputState(s) <InputState>` and/or `OutputState(s) <OutputState>` of other
        Mechanisms.
      |
      Each type of AdaptiveMechanism is associated with a corresponding type of `ModulatorySignal` (a type of
      `OutputState` specialized for use with the AdaptiveMechanism) and `ModulatoryProjection`.

Every Mechanism is made up of four fundamental components:

    * `InputState(s) <InputState>` used to receive and represent its input(s);
    ..
    * `Function <Function>` used to transform its input(s) into its output(s);
    ..
    * `ParameterState(s) <ParameterState>` used to represent the parameters of its Function (and/or any
      parameters that are specific to the Mechanism itself);
    ..
    * `OutputState(s) <OutputState>` used to represent its output(s)

These are described in the sections on `Mechanism_Function` and `Mechanism_States` (`Mechanism_InputStates`,
`Mechanism_ParameterStates`, and `Mechanism_OutputStates`), and shown graphically in a `figure <Mechanism_Figure>`,
under `Mechanism_Structure` below.

.. _Mechanism_Creation:

Creating a Mechanism
--------------------

Mechanisms can be created in several ways.  The simplest is to call the constructor for the desired type of Mechanism.
Alternatively, the `mechanism` command can be used to create a specific type of Mechanism or an instance of
`default_mechanism <Mechanism_Base.default_mechanism>`. Mechanisms can also be specified "in context," for example in
the `pathway <Process_Base.pathway>` attribute of a `Process`; the Mechanism can be specified in either of the ways
mentioned above, or using one of the following:

  * the name of an **existing Mechanism**;
  ..
  * the name of a **Mechanism type** (subclass);
  ..
  * a **specification dictionary** -- this can contain an entry specifying the type of Mechanism,
    and/or entries specifying the value of parameters used to instantiate it.
    These should take the following form:

      * *MECHANISM_TYPE*: <name of a Mechanism type>
          if this entry is absent, a `default_mechanism <Mechanism_Base.default_mechanism>` will be created.

      * *NAME*: <str>
          the string will be used as the `name <Mechanism_Base.name>` of the Mechanism;  if this entry is absent,
          the name will be the name of the Mechanism's type, suffixed with an index if there are any others of the
          same type for which a default name has been assigned.

      * <name of parameter>:<value>
          this can contain any of the `standard parameters <Mechanism_Attributes>` for instantiating a Mechanism
          or ones specific to a particular type of Mechanism (see documentation for the type).  The key must be
          the name of the argument used to specify the parameter in the Mechanism's constructor, and the value must
          be a legal value for that parameter, using any of the ways allowed for `specifying a parameter
          <ParameterState_Specification>`. The parameter values specified will be used to instantiate the Mechanism.
          These can be overridden during execution by specifying `Mechanism_Runtime_Parameters`, either when calling
          the Mechanism's `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` method, or where it is
          specified in the `pathway <Process_Base.pathway>` attribute of a `Process`.

  * **automatically** -- PsyNeuLink automatically creates one or more Mechanisms under some circumstances.
    For example, an `ObjectiveMechanism` and `LearningMechanisms <LearningMechanism>` are created automatically when
    `learning <Process_Learning>` is specified for a Process; and an `ObjectiveMechanism` and `ControlMechanism`
    are created when the `controller <System_Base.controller>` is specified for a `System`.

.. _Mechanism_State_Specification:

Specifying States
~~~~~~~~~~~~~~~~~

Every Mechanism has one or more `InputStates <InputState>`, `ParameterStates <ParameterState>`, and `OutputStates
<OutputState>` (described `below <Mechanism_States>`) that allow it to receive and send `Projections <Projection>`,
and to execute its `function <Mechanism_Base.function>`).  When a Mechanism is created, it automatically creates the
ParameterStates it needs to represent its parameters, including those of its `function <Mechanism_Base.function>`. It
also creates any InputStates and OutputStates required for the Projections it has been assigned. InputStates and
OutputStates, and corresponding Projections (including those from `ModulatorySignals <ModulatorySignal>`) can also be
specified explicitly in the **input_states** and **output_states** arguments of the Mechanism's constructor (see `first
example <Mechanism_Example_1>` below), or in a `parameter specification dictionary <ParameterState_Specification>
assigned to its **params** argument using entries with the keys *INPUT_STATES* and *OUTPUT_STATES*, respectively (see
`second example <Mechanism_Example_2>` below).  While specifying in the arguments directly is simpler and more
convenient, the dictionary format allows parameter sets to be created elsewhere and/or re-used.  The value of each
entry can be any of the allowable forms for `specifying a state <State_Creation>`. InputStates and OutputStates can
also be added to an existing Mechanism using its `add_states` method, although this is generally not needed and
therefore not recommended.

Examples
^^^^^^^^

.. _Mechanism_Example_1:

The following example creates an instance of a TransferMechanism that names the default InputState ``MY_INPUT``,
and assigns three `standard OutputStates <OutputState_Standard>`::

     my_mech = TransferMechanism(input_states=['MY_INPUT'],
                                 output_states=[RESULT, MEAN, VARIANCE])

.. _Mechanism_Example_2:

This shows how the same Mechanism can be specified using a dictionary assigned to the **params** argument::

     my_mech = TransferMechanism(params={INPUT_STATES: ['MY_INPUT'],
                                         OUTPUT_STATES: [RESULT, MEAN, VARIANCE]})

.. _Mechanism_Parameter_Specification:

Specifying Parameters
~~~~~~~~~~~~~~~~~~~~~

As described `below <Mechanism_ParameterStates>`, Mechanisms have `ParameterStates <ParameterState>` that provide the
current value of a parameter used by the Mechanism and/or its `function <Mechanism_Base.function>` when it is `executed
<Mechanism_Execution>`. These can also be used by a `ControlMechanism` to control the parameters of the Mechanism and/or
it `function <Mechanism_Base.function>`.  The value of any of these, and their control, can be specified in the
corresponding argument of the constructor for the Mechanism and/or its `function <Mechanism_Base.function>`,  or in a
parameter specification dictionary assigned to the **params** argument of its constructor, as described under
`ParameterState_Specification`.


.. _Mechanism_Structure:

Structure
---------

.. _Mechanism_Function:

Function
~~~~~~~~

The core of every Mechanism is its function, which transforms its input to generate its output.  The function is
specified by the Mechanism's `function <Mechanism_Base.function>` attribute.  Every type of Mechanism has at least one
(primary) function, and some have additional (auxiliary) ones (for example, `TransferMechanism` and `EVCMechanism`).
Mechanism functions are generally from the PsyNeuLink `Function` class.  Most Mechanisms allow their function to be
specified, using the `function` argument of the Mechanism's constructor.  The function can be specified using the
name of `Function <Function>` class, or its constructor (including arguments that specify its parameters).  For
example, the `function <TransferMechanism.function>` of a `TransferMechanism`, which is `Linear` by default, can be
specified to be the `Logistic` function as follows::

    my_mechanism = TransferMechanism(function=Logistic(gain=1.0, bias=-4))

Notice that the parameters of the :keyword:`function` (in this case, `gain` and `bias`) can be specified by including
them in its constructor.  Some Mechanisms support only a single function.  In that case, the :keyword:`function`
argument is not available in the Mechanism's constructor, but it does include arguments for the function's
parameters.  For example, the :keyword:`function` of a `ComparatorMechanism` is always the `LinearCombination` function,
so the Mechanisms' constructor does not have a :keyword:`function` argument.  However, it does have a
**comparison_operation** argument, that is used to set the LinearCombination function's `operation` parameter.

The parameters for a Mechanism's primary function can also be specified as entries in a *FUNCTION_PARAMS* entry of a
`parameter specification dictionary <ParameterState_Specification>` in the **params** argument of the Mechanism's
constructor.  For example, the parameters of the `Logistic` function in the example above can
also be assigned as follows::

    my_mechanism = TransferMechanism(function=Logistic
                                     params={FUNCTION_PARAMS: {GAIN:1.0,
                                                               BIAS=-4.0})

Again, while not as simple as specifying these as arguments in the function's construtor, this format is more flexible.
Any values specified in the parameter dictionary will **override** any specified within the constructor for the function
itself (see `DDM_Parameters` for an example).

.. _Mechanism_Function_Object:

function_object Attribute
^^^^^^^^^^^^^^^^^^^^^^^^^

The `Function <Function>` Component assigned as the primary function of a Mechanism is assigned to the Mechanism's
`function_object <Component.function_object>` attribute, and its `function <Function_Base.function>` is assigned
to the Mechanism's `function <Mechanism_Base.function>` attribute.

.. note::
   It is important to recognize the distinction between a `Function <Function>` and its `function
   <Function_Base.function>` attribute (note the difference in capitalization).  A *Function* is a PsyNeuLink
   `Component`, that can be created using a constructor; a *function* is an attribute that contains a callable method
   belonging to a Function, and that is executed when the Component to which the Function belongs is executed.
   Functions are used to assign, store, and apply parameter values associated with their function (see `Function
   <Function_Overview> for a more detailed explanation).

The parameters of a Mechanism's `function <Mechanism_Base.function>` are attributes of its `function_object
<Component.function_object>`, and can be accessed using standard "dot" notation for that object.  For
example, the `gain <Logistic.gain>` and `bias <Logistic.bias>` parameters of the `Logistic` function in the example
above can be access as ``my_mechanism.function_object.gain`` and ``my_mechanism.function_object.bias``.  They are
also assigned to a dictionary in the Mechanism's `function_params <Mechanism_Base.function_params>` attribute,
and can be  accessed using the parameter's name as the key for its entry in the dictionary.  For example,
the parameters in the  example above could also be accessed as ``my_mechanism.function_params[GAIN]`` and
``my_mechanism.function_params[GAIN]``

Some Mechanisms have auxiliary functions that are inherent (i.e., not made available as arguments in the Mechanism's
constructor;  e.g., the `integrator_function <TransferMechanism.integrator_function>` of a `TransferMechanism`);
however, the Mechanism may include parameters for those functions in its constructor (e.g., the **noise** argument in
the constructor for a `TransferMechanism` is used as the `noise <AdaptiveIntegrator.noise>` parameter of the
`AdaptiveIntegrator` assigned to the TransferMechanism's `integrator_function <TransferMechanism.integrator_function>`).

COMMENT:
NOT CURRENTLY IMPLEMENTED
For Mechanisms that offer a selection of functions for the primary function (such as the `TransferMechanism`), if all
of the functions use the same parameters, then those parameters can also be specified as entries in a `parameter
specification dictionary <ParameterState_Specification>` as described above;  however, any parameters that are unique
to a particular function must be specified in a constructor for that function.  For Mechanisms that have additional,
auxiliary functions, those must be specified in arguments for them in the Mechanism's constructor, and their parameters
must be specified in constructors for those functions unless documented otherwise.
COMMENT

COMMENT:
    FOR DEVELOPERS:
    + FUNCTION : function or method :  method used to transform Mechanism input to its output;
        This must be implemented by the subclass, or an exception will be raised;
        each item in the variable of this method must be compatible with a corresponding InputState;
        each item in the output of this method must be compatible  with the corresponding OutputState;
        for any parameter of the method that has been assigned a ParameterState,
        the output of the ParameterState's own execute method must be compatible with
        the value of the parameter with the same name in params[FUNCTION_PARAMS] (EMP)
    + FUNCTION_PARAMS (dict):
        NOTE: function parameters can be specified either as arguments in the Mechanism's __init__ method,
        or by assignment of the function_params attribute for paramClassDefaults.
        Only one of these methods should be used, and should be chosen using the following principle:
        - if the Mechanism implements one function, then its parameters should be provided as arguments in the __init__
        - if the Mechanism implements several possible functions and they do not ALL share the SAME parameters,
            then the function should be provided as an argument but not they parameters; they should be specified
            as arguments in the specification of the function
        each parameter is instantiated as a ParameterState
        that will be placed in <Mechanism>._parameter_states;  each parameter is also referenced in
        the <Mechanism>.function_params dict, and assigned its own attribute (<Mechanism>.<param>).
COMMENT


.. _Mechanism_Custom_Function:

Custom Functions
^^^^^^^^^^^^^^^^

A Mechanism's `function <Mechanism_Base.function>` can be customized by assigning a user-defined function (e.g.,
a lambda function), so long as it takes arguments and returns values that are compatible with those of the
Mechanism's default for that function.  This is also true for auxiliary functions that appear as arguments in a
Mechanism's constructor (e.g., the `EVCMechanism_Auxiliary_Functions` of an EVC Mechanmism). A user-defined function
can be assigned using the Mechanism's `assign_params` method (the safest means) or by assigning it directly to the
corresponding attribute of the Mechanism (for its primary function, its `function <Mechanism_Base.function>` attribute).
It is *strongly advised* that auxiliary functions that are inherent to a Mechanism (i.e., ones that do *not* appear
as an argument in the Mechanism's constructor, such as the `integrator_function <TransferMechanism.integrator_function>`
of a `TransferMechanism`) *not* be assigned custom functions;  this is because their parameters are included as
arguments in the constructor for the Mechanism, and thus changing the function could produce confusing and/or
unpredictable effects.


COMMENT:
    When a custom function is specified,
    the function itself is assigned to the Mechanism's designated attribute.  At the same time, PsyNeuLink automatically
    creates a `UserDefinedFunction` object, and assigns the custom function to its
    `function <UserDefinedFunction.function>` attribute.
COMMENT

variable and value Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The input to a Mechanism's `function <Mechanism_Base.function>` is provided by the Mechanism's
`variable <Mechanism_Base.variable>` attribute.  This is a 2d array with one item for each of the Mechanism's
`Input_states <Mechanism_InputStates>`.  The result of the `function <Mechanism_Base.function>` is placed in the
Mechanism's `value <Mechanism_Base.value>` attribute, which is also a 2d array with one or more items.  The
Mechanism's `value <Mechanism_Base.value>` is used by its `OutputStates <Mechanism_OutputStates>` to generate their
`value <OutputState.value>` attributes, each of which is assigned as the value of an item of the list in the Mechanism's
`output_values <Mechanism_Base.output_values>` attribute (see `Mechanism_OutputStates` below).

.. note::
   The input to a Mechanism is not necessarily the same as the input to its `function <Mechanism_Base.function>`. The
   input to a Mechanism is first processed by its `InputState(s) <Mechanism_InputStates>`, and then assigned to the
   Mechanism's `variable <Mechanism_Base>` attribute, which is used as the input to its `function
   <Mechanism_Base.function>`. Similarly, the result of a Mechanism's function is not necessarily the same as the
   Mechanism's output.  The result of the `function <Mechanism_Base.function>` is assigned to the Mechanism's  `value
   <Mechanism_Base.value>` attribute, which is then used by its `OutputState(s) <Mechanism_OutputStates>` to assign
   items to its `output_values <Mechanism_Base.output_values>` attribute.

.. _Mechanism_States:

States
~~~~~~

Every Mechanism has one or more of each of three types of States:  `InputState(s) <InputState>`,
`ParameterState(s) <ParameterState>`, `and OutputState(s) <OutputState>`.  Generally, these are created automatically
when the Mechanism is created.  InputStates and OutputStates (but not ParameterStates) can also be specified explicitly
for a Mechanism, or added to an existing Mechanism using its `add_states` method, as described
`above <Mechanism_State_Specification>`).

.. _Mechanism_Figure:

The three types of States are shown schematically in the figure below, and described briefly in the following sections.

.. figure:: _static/Mechanism_States_fig.svg
   :alt: Mechanism States
   :scale: 75 %
   :align: left

   **Schematic of a Mechanism showing its three types of States** (input, parameter and output). Every Mechanism has at
   least one `InputState` (its `primary InputState <InputState_Primary>` ) and one `OutputState`
   (its `primary OutputState <OutputState_Primary>`), and can have additional ones of each.  It also has one
   `ParameterState` for each of its parameters and the parameters of its `function <Mechanism_Base.function>`.

.. _Mechanism_InputStates:

InputStates
^^^^^^^^^^^

These receive and represent the input to a Mechanism. A Mechanism usually has only one (`primary <InputState_Primary>`)
`InputState`, identified by its `input_state <Mechanism_Base.input_state>`, attribute.  However some Mechanisms have
more  than one InputState. For example, a `ComparatorMechanism` has one InputState for its **SAMPLE** and another for
its **TARGET** input. If a Mechanism has more than one InputState, they are listed in the Mechanism's `input_states
<Mechanism_Base.input_states>` attribute (note the plural).  The `input_states <Mechanism_Base.input_states>` attribute
is a ContentAddressableList -- a PsyNeuLink-defined subclass of the Python class
`UserList <https://docs.python.org/3.6/library/collections.html?highlight=userlist#collections.UserList>`_ --
that allows a specific InputState in the list to be accessed using its name as the index for the list (e.g.,
``my_mechanism['InputState name']``).

COMMENT:
[TBI:]
If the InputState are created automatically, or are not assigned a name when specified, then each is named
using the following template: [TBI]
COMMENT

Each InputState of a Mechanism can receive one or more `Projections <Projection>` from other Mechanisms.  The
`PathwayProjections <PathwayProjection>` (e.g., `MappingProjections <MappingProjection>`) it receives are listed in
its `path_afferents <InputState.path_afferents>` attribute.  If the Mechanism is an `ORIGIN` Mechanism of a
`Process`, this includes a Projection from the `ProcessInputState <Process_Input_And_Output>` for that Process.  Any
`GatingProjections <GatingProjection>` it receives are listed in its `mod_afferents <InputState.mod_afferents>`
attribute.  The InputState's `function <InputState.InputState.function>` aggregates  the values received from its
`path_afferents <InputState.path_afferents>`, `modulates <ModulatorySignal_Modulation>` this in response to any
GatingProjections it receives, and assigns the result to the InputState's `value <InputState.value>` attribute.

.. _Mechanism_Variable:

The `value <InputState.value>` of each InputState for a Mechanism is assigned to a different item of the Mechanism's
`variable <Mechanism_Base.variable>` attribute (a 2d np.array), as well as to a corresponding item of its `input_values
<Mechanism_Base.input_values>` attribute (a list).  The `variable <Mechanism_Base.variable>` provides the input to the
Mechanism's `function <Mechanism_Base.function>`, while its `input_values <Mechanism_Base.input_values>` provides a
more convenient way of accessing the value of its individual items.  Because there is a one-to-one correspondence
between a Mechanism's InputStates and the items of its `variable <Mechanism_Base.variable>`, their lengths must be
equal;  that is, the number of items in the Mechanism's `variable <Mechanism_Base.variable>` attribute (its size
along axis 0) must equal the number of InputStates in its `input_states <Mechanism_Base.input_states>` attribute.
Therefore, if any InputStates are specified in the constructor, the number of them must match the number of items in
`variable <Mechanism_Base.variable>`.  However, if InputStates are added using the Mechanism's `add_states` method,
then its `variable <Mechanism_Base.variable>` is extended to accommodate the number of InputStates added (note that
this must be coordinated with the Mechanism's `function <Mechanism_Base.function>`, which takes the Mechanism's
`variable <Mechanism_Base.variable>` as its input -- see `InputState documentation
<InputStates_Mechanism_Variable_and_Function>` for a more detailed explanation). The order in which InputStates are
specified in Mechanism's constructor, and/or added using its `add_states` method, determines the order of the items
to which they are assigned assigned in he Mechanism's `variable <Mechanism_Base.variable>`, and are listed in its
`input_values <Mechanism_Base.input_values>` attribute.


COMMENT:

If more InputStates are specified than there are items in `variable <Mechanism_Base.variable>, the latter is extended
to  match the former (see `InputState <InputStates_Mechanism_Variable_and_Function>` for a more detailed explanation).


The number of input_states for the Mechanism must match the number of items specified for the Mechanism's
``variable`` (that is, its size along its first dimension, axis 0).  An exception is if the Mechanism's `variable``
has more than one item, but only a single InputState;  in that case, the ``value`` of that InputState must have the
same number of items as the Mechanisms's ``variable``.
COMMENT

.. _Mechanism_ParameterStates:

ParameterStates and Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`ParameterStates <ParameterState>` provide the value for each parameter of a Mechanism and its `function
<Mechanism_Base.function>`.  One ParameterState is assigned to each of the parameters of the Mechanism and/or its
`function <Mechanism_Base.function>` (corresponding to the arguments in their constructors). The ParameterState takes
the value specified for a parameter (see `below <Mechanism_Parameter_Value_Specification>`) as its `variable
<ParameterState.variable>`, and uses it as the input to the ParameterState's `function <ParameterState.function>`,
which `modulates <ModulatorySignal_Modulation>` it in response to any `ControlProjections <ControlProjection>` received
by the ParameterState (specified in its `mod_afferents <ParameterState.mod_afferents>` attribute), and assigns the
result to the ParameterState's `value <ParameterState.value>`.  This is the value used by the Mechanism or its
`function <Mechanism_Base.function>` when the Mechanism `executes <Mechanism_Execution>`.  Accordingly, when the value
of a parameter is accessed (e.g., using "dot" notation, such as ``my_mech.my_param``), it is actually the
*ParameterState's* `value <ParameterState.value>` that is returned (thereby accurately reflecting the value used
during the last execution of the Mechanism or its `function <Mechanism_Base.function>`).

.. _Mechanism_Parameter_Value_Specification:

The "base" value of a parameter (i.e., the unmodulated value used as the ParameterState's `variable
<ParameterState.variable>` and the input to its `function <ParameterState.function>`) can specified when a Mechanism
and/or its `function <Mechanism_Base.function>` are first created,  using the corresponding arguments of their
constructors (see `Mechanism_Function` above).  Parameter values can also be specified later, by direct assignment of a
value to the attribute for the parameter, or by using the Mechanism's `assign_param` method (the recommended means;
see `ParameterState_Specification`).  Note that the attributes for the parameters of a Mechanism's `function
<Mechanism_Base.function>` usually belong to the `Function <Function_Overview>` referenced in its `function_object
<Component.function_object>` attribute, not the Mechanism itself, and therefore must be assigned to the Function
Component (see `Mechanism_Function_Object` above).

All of the Mechanism's parameters are listed in a dictionary in its `user_params` attribute; that dictionary contains
a *FUNCTION_PARAMS* entry that contains a sub-dictionary with the parameters of the Mechanism's `function
<Mechanism_Base.function>`.  The *FUNCTION_PARAMS* sub-dictionary is also accessible directly from the Mechanism's
`function_params <Mechanism_Base.function_params>` attribute.

.. _Mechanism_OutputStates:

OutputStates
^^^^^^^^^^^^
These represent the output(s) of a Mechanism. A Mechanism can have several `OutputStates <OutputState>`, and each can
send Projections that transmit its value to other Mechanisms and/or as the output of the `Process` or `System` to which
the Mechanism belongs.  Every Mechanism has at least one OutputState, referred to as its `primary OutputState
<OutputState_Primary>`.  If OutputStates are not explicitly specified for a Mechanism, a primary OutputState is
automatically created and assigned to its `output_state <Mechanism_Base.output_state>` attribute (note the singular),
and also to the first entry of the Mechanism's `output_states <Mechanism_Base.output_states>` attribute (note the
plural).  The `value <OutputState.value>` of the primary OutputState is assigned as the first (and often only) item
of the Mechanism's `value <Mechanism_Base.value>` attribute, which is the result of the Mechanism's `function
<Mechanism_Base.function>`.  Additional OutputStates can be assigned to represent values corresponding to other items
of the Mechanism's `value <Mechanism_Base.value>` (if there are any) and/or values derived from any or all of those
items. `Standard OutputStates <OutputState_Standard>` are available for each type of Mechanism, and custom ones can
be configured (see `OutputState Specification <OutputState_Specification>`. These can be assigned in the
**output_states** argument of the Mechanism's constructor.

All of a Mechanism's OutputStates (including the primary one) are listed in its `output_states
<Mechanism_Base.output_states>` attribute (note the plural). The `output_states <Mechanism_Base.output_states>`
attribute is a ContentAddressableList -- a PsyNeuLink-defined subclass of the Python class
`UserList <https://docs.python.org/3.6/library/collections.html?highlight=userlist#collections.UserList>`_ -- that
allows a specific OutputState in the list to be accessed using its name as the index for the list (e.g.,
``my_mechanism['OutputState name']``).  This list can also be used to assign additional OutputStates to the Mechanism
after it has been created.

The `value <OutputState.value>` of each of the Mechanism's OutputStates is assigned as an item in the Mechanism's
`output_values <Mechanism_Base.output_values>` attribute, in the same order in which they are listed in its
`output_states <Mechanism_Base.output_states>` attribute.  Note, that the `output_values <Mechanism_Base.output_values>`
attribute of a Mechanism is distinct from its `value <Mechanism_Base.value>` attribute, which contains the full and
unmodified results of its `function <Mechanism_Base.function>` (this is because OutputStates can modify the item of
the Mechanism`s `value <Mechanism_Base.value>` to which they refer -- see `OutputStates <OutputState_Customization>`).


.. _Mechanism_Attributes:

Additional Attributes
~~~~~~~~~~~~~~~~~~~~~

In addition to the `standard attributes <Component_Structure>` of any `Component`, Mechanisms have a set of
Mechanism-specific attributes (listed below). These can be specified in arguments of the Mechanism's constructor,
in a `parameter specification dictionary <ParameterState_Specification>` assigned to the **params** argument of the
Mechanism's constructor, by direct reference to the corresponding attribute of the Mechanisms after it has been
constructed (e.g., ``my_mechanism.param``), or using the Mechanism's `assign_params` method. The Mechanism-specific
attributes are listed below by their argument names / keywords, along with a description of how they are specified:

COMMENT:
    * **input_states** / *INPUT_STATES* - a list specifying specialized input_states used by a Mechanism type
      (see :ref:`InputState specification <InputState_Creation>` for details of specification).
    ..
    * **output_states** / *OUTPUT_STATES* - specifies specialized OutputStates required by a Mechanism subclass
      (see :ref:`OutputStates_Creation` for details of specification).
COMMENT
    ..
    * **monitor_for_control** / *MONITOR_FOR_CONTROL* - specifies which of the Mechanism's OutputStates is monitored by
      the `controller` for the System to which the Mechanism belongs (see :ref:`specifying monitored OutputStates
      <ControlMechanism_Monitored_OutputStates>` for details of specification).
    ..
    * **monitor_for_learning** / *MONITOR_FOR_LEARNING* - specifies which of the Mechanism's OutputStates is used for
      learning (see `Learning <LearningMechanism_Activation_Output>` for details of specification).


.. _Mechanism_Role_In_Processes_And_Systems:

Role in Processes and Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mechanisms that are part of one or more `Processes <Process>` are assigned designations that indicate the
`role <Process_Mechanisms>` they play in those Processes, and similarly for `role <System_Mechanisms>` they play in
any `Systems <System>` to which they belong. These designations are listed in the Mechanism's `processes
<Mechanism_Base.processes>` and `systems <Mechanism_Base.systems>` attributes, respectively.  Any Mechanism
designated as `ORIGIN` receives a `MappingProjection` to its `primary InputState <InputState_Primary>` from the
Process(es) to which it belongs.  Accordingly, when the Process (or System of which the Process is a part) is
executed, those Mechanisms receive the input provided to the Process (or System).  The `output_values
<Mechanism_Base.output_values>` of any Mechanism designated as the `TERMINAL` Mechanism for a Process is assigned as
the `output <Process_Base.output>` of that Process, and similarly for any System to which it belongs.

.. note::
   A Mechanism that is the `ORIGIN` or `TERMINAL` of a Process does not necessarily have the same role in the
   System(s) to which the Mechanism or Process belongs (see `example <LearningProjection_Target_vs_Terminal_Figure>`).


.. _Mechanism_Execution:

Execution
---------

A Mechanism can be executed using its `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` methods.  This
can be useful in testing a Mechanism and/or debugging.  However, more typically, Mechanisms are executed as part of a
`Process <Process_Execution>` or `System <System_Execution>`.  For either of these, the Mechanism must be included in
the `pathway <Process_Base.pathway>` of a Process.  There, it can be specified on its own, or as the first item of a
tuple that also has an optional set of `runtime parameters <Mechanism_Runtime_Parameters>` (see `Process Mechanisms
<Process_Mechanisms>` for additional details about specifying a Mechanism in a Process `pathway
<Process_Base.pathway>`).

.. _Mechanism_Runtime_Parameters:

Runtime Parameters
~~~~~~~~~~~~~~~~~~

.. note::
   This is an advanced feature, and is generally not required for most applications.

The parameters of a Mechanism are usually specified when the Mechanism is `created <Mechanism_Creation>`.  However,
these can be overridden when it `executed <Mechanism_Base.execution>`.  This can be done in a `parameter specification
dictionary <ParameterState_Specification>` assigned to the **runtime_param** argument of the Mechanism's `execute
<Mechanism_Base.execute>` method, or in a `tuple with the Mechanism <Process_Mechanism_Specification>` in the `pathway`
of a `Process`.  Any value assigned to a parameter in a **runtime_params** dictionary will override the current value of
that parameter for the (and *only* the) current execution of the Mechanism; the value will return to its previous value
following that execution, unless the `runtimeParamStickyAssignmentPref` is set for the component to which the parameter
belongs.

The runtime parameters for a Mechanism are specified using a dictionary that contains one or more entries, each of which
is for a parameter of the Mechanism or its  `function <Mechanism_Base.function>`, or for one of the `Mechanism's States
<Mechanism_States>`. Entries for parameters of the Mechanism or its `function <Mechanism_Base.function>` use the
standard format for `parameter specification dictionaries <ParameterState_Specification>`. Entries for the Mechanism's
States can be used to specify runtime parameters of the corresponding State, its `function <State_Base.function>`, or
any of the `Projections to that state <State_Projections>`. Each entry for the parameters of a State uses a key
corresponding to the type of State (*INPUT_STATE_PARAMS*, *OUTPUT_STATE_PARAMS* or *PARAMETER_STATE_PARAMS*), and a
value that is a sub-dictionary containing a dictionary with the runtime  parameter specifications for all States of that
type). Within that sub-dictionary, specification of parameters for the State or its `function <State_Base.function>` use
the  standard format for a `parameter specification dictionary <ParameterState_Specification>`.  Parameters for all of
the `State's Projections <State_Projections>` can be specified in an entry with the key *PROJECTION_PARAMS*, and a
sub-dictionary that contains the parameter specifications;  parameters for Projections of a particular type can be
placed in an entry with a key specifying the type (*MAPPING_PROJECTION_PARAMS*, *LEARNING_PROJECTION_PARAMS*,
*CONTROL_PROJECTION_PARAMS*, or *GATING_PROJECTION_PARAMS*; and parameters for a specific Projection can be placed in
an entry with a key specifying the name of the Projection and a sub-dictionary with the specifications.

COMMENT:
    ADD EXAMPLE(S) HERE
COMMENT

COMMENT:
?? DO PROJECTION DICTIONARIES PERTAIN TO INCOMING OR OUTGOING PROJECTIONS OR BOTH??
?? CAN THE KEY FOR A STATE DICTIONARY REFERENCE A SPECIFIC STATE BY NAME, OR ONLY STATE-TYPE??

State keyword: dict for State's params
    Function or Projection keyword: dict for Funtion or Projection's params
        parameter keyword: vaue of param

    dict: can be one (or more) of the following:
        + INPUT_STATE_PARAMS:<dict>
        + PARAMETER_STATE_PARAMS:<dict>
   [TBI + OUTPUT_STATE_PARAMS:<dict>]
        - each dict will be passed to the corresponding State
        - params can be any permissible executeParamSpecs for the corresponding State
        - dicts can contain the following embedded dicts:
            + FUNCTION_PARAMS:<dict>:
                 will be passed the State's execute method,
                     overriding its paramInstanceDefaults for that call
            + PROJECTION_PARAMS:<dict>:
                 entry will be passed to all of the State's Projections, and used by
                 by their execute methods, overriding their paramInstanceDefaults for that call
            + MAPPING_PROJECTION_PARAMS:<dict>:
                 entry will be passed to all of the State's MappingProjections,
                 along with any in a PROJECTION_PARAMS dict, and override paramInstanceDefaults
            + LEARNING_PROJECTION_PARAMS:<dict>:
                 entry will be passed to all of the State's LearningProjections,
                 along with any in a PROJECTION_PARAMS dict, and override paramInstanceDefaults
            + CONTROL_PROJECTION_PARAMS:<dict>:
                 entry will be passed to all of the State's ControlProjections,
                 along with any in a PROJECTION_PARAMS dict, and override paramInstanceDefaults
            + GATING_PROJECTION_PARAMS:<dict>:
                 entry will be passed to all of the State's GatingProjections,
                 along with any in a PROJECTION_PARAMS dict, and override paramInstanceDefaults
            + <ProjectionName>:<dict>:
                 entry will be passed to the State's Projection with the key's name,
                 along with any in the PROJECTION_PARAMS and MappingProjection or ControlProjection dicts
COMMENT

.. _Mechanism_Class_Reference:

Class Reference
---------------

"""

import inspect
import logging

from collections import OrderedDict
from inspect import isclass

import numpy as np
import typecheck as tc

from PsyNeuLink.Components.Component import Component, ExecutionStatus, function_type, method_type
from PsyNeuLink.Components.ShellClasses import Function, Mechanism, Projection
from PsyNeuLink.Globals.Defaults import timeScaleSystemDefault
from PsyNeuLink.Globals.Keywords import CHANGED, COMMAND_LINE, DDM_MECHANISM, EVC_SIMULATION, EXECUTING, FUNCTION_PARAMS, INITIALIZING, INIT_FUNCTION_METHOD_ONLY, INIT__EXECUTE__METHOD_ONLY, INPUT_STATES, INPUT_STATE_PARAMS, MECHANISM_TIME_SCALE, MONITOR_FOR_CONTROL, MONITOR_FOR_LEARNING, NO_CONTEXT, OUTPUT_STATES, OUTPUT_STATE_PARAMS, PARAMETER_STATE, PARAMETER_STATE_PARAMS, PROCESS_INIT, SEPARATOR_BAR, SET_ATTRIBUTE, SYSTEM_INIT, TIME_SCALE, UNCHANGED, VALIDATE, kwMechanismComponentCategory, kwMechanismExecuteFunction, kwMechanismType, kwProcessDefaultMechanism
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceLevel
from PsyNeuLink.Globals.Registry import register_category
from PsyNeuLink.Globals.Utilities import AutoNumber, ContentAddressableList, append_type_to_name, convert_to_np_array, iscompatible, kwCompatibilityNumeric
from PsyNeuLink.Scheduling.TimeScale import CentralClock, TimeScale

logger = logging.getLogger(__name__)
MechanismRegistry = {}

class MonitoredOutputStatesOption(AutoNumber):
    """Specifies outputStates to be monitored by a `ControlMechanism` (see `ControlMechanism_Monitored_OutputStates
    for a more complete description of their meanings."""
    ONLY_SPECIFIED_OUTPUT_STATES = ()
    """Only monitor explicitly specified Outputstates."""
    PRIMARY_OUTPUT_STATES = ()
    """Monitor only the `primary OutputState <OutputState_Primary>` of a Mechanism."""
    ALL_OUTPUT_STATES = ()
    """Monitor all OutputStates <Mechanism_Base.outputStates>` of a Mechanism."""
    NUM_MONITOR_STATES_OPTIONS = ()


class MechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


def mechanism(mech_spec=None, params=None, context=None):
    """Factory method for Mechanism; returns the type of Mechanism specified or a `default_mechanism`.
    If called with no arguments, returns the `default_mechanism <Mechanism_Base.default_mechanism>`.

    Arguments
    ---------

    mech_spec : Optional[Mechanism subclass, str, or dict]
        specification for the Mechanism to create.
        If it is the name of a Mechanism subclass, a default instance of that subclass is returned.
        If it is string that is the name of a Mechanism subclass registered in the `MechanismRegistry`,
        an instance of a default Mechanism for *that class* is returned; otherwise, the string is used to name an
        instance of the `default_mechanism <Mechanism_Base.default_mechanism>.  If it is a dict, it must be a
        `Mechanism specification dictionary <`Mechanism_Creation>`. If it is `None` or not specified, an instance of
        the `default Mechanism <Mechanism_Base.default_mechanism>` is returned;
        the nth instance created will be named by using the Mechanism's `componentType <Mechanism_Base.componentType>`
        attribute as the base for the name and adding an indexed suffix:  componentType-n.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the Mechanism and/or its function, and/or a custom function and its parameters.  It is passed to the relevant
        subclass to instantiate the Mechanism. Its entries can be used to specify any attributes relevant to a
        `Mechanism <Mechanism_Structure>` and/or defined specifically by the subclass being created.  Values specified
        for parameters in the dictionary override any assigned to those parameters in arguments of the constructor.

    COMMENT:
        context : str
            if it is the keyword VALIDATE, returns `True` if specification would return a valid
            subclass object; otherwise returns :keyword:`False`.
    COMMENT

    Returns
    -------

    Instance of specified type of Mechanism or None : Mechanism
    """

    # Called with a keyword
    if mech_spec in MechanismRegistry:
        return MechanismRegistry[mech_spec].mechanismSubclass(params=params, context=context)

    # Called with a string that is not in the Registry, so return default type with the name specified by the string
    elif isinstance(mech_spec, str):
        return Mechanism_Base.default_mechanism(name=mech_spec, params=params, context=context)

    # Called with a Mechanism type, so return instantiation of that type
    elif isclass(mech_spec) and issubclass(mech_spec, Mechanism):
        return mech_spec(params=params, context=context)

    # Called with Mechanism specification dict (with type and params as entries within it), so:
    #    - get mech_type from kwMechanismType entry in dict
    #    - pass all other entries as params
    elif isinstance(mech_spec, dict):
        # Get Mechanism type from kwMechanismType entry of specification dict
        try:
            mech_spec = mech_spec[kwMechanismType]
        # kwMechanismType config_entry is missing (or mis-specified), so use default (and warn if in VERBOSE mode)
        except (KeyError, NameError):
            if Mechanism.classPreferences.verbosePref:
                print("{0} entry missing from mechanisms dict specification ({1}); default ({2}) will be used".
                      format(kwMechanismType, mech_spec, Mechanism_Base.default_mechanism))
            return Mechanism_Base.default_mechanism(name=kwProcessDefaultMechanism, context=context)
        # Instantiate Mechanism using mech_spec dict as arguments
        else:
            return mech_spec(context=context, **mech_spec)

    # Called without a specification, so return default type
    elif mech_spec is None:
        return Mechanism_Base.default_mechanism(name=kwProcessDefaultMechanism, context=context)

    # Can't be anything else, so return empty
    else:
        return None


class Mechanism_Base(Mechanism):
    """Base class for Mechanism.

    .. note::
       Mechanism is an abstract class and should **never** be instantiated by a direct call to its constructor.
       It should be instantiated using the :class:`mechanism` command (see it for description of parameters),
       by calling the constructor for a subclass, or using other methods for specifying a Mechanism in context
       (see `Mechanism_Creation`).

    COMMENT:
        Description
        -----------
            Mechanism is a Category of the Component class.
            A Mechanism is associated with a name and:
            - one or more input_states:
                two ways to get multiple input_states, if supported by Mechanism subclass being instantiated:
                    • specify 2d variable for Mechanism (i.e., without explicit InputState specifications)
                        once the variable of the Mechanism has been converted to a 2d array, an InputState is assigned
                        for each item of axis 0, and the corresponding item is assigned as the InputState's variable
                    • explicitly specify input_states in params[*INPUT_STATES*] (each with its own variable
                        specification); those variables will be concantenated into a 2d array to create the Mechanism's
                        variable
                if both methods are used, they must generate the same sized variable for the mechanims
                ?? WHERE IS THIS CHECKED?  WHICH TAKES PRECEDENCE: InputState SPECIFICATION (IN _instantiate_state)??
            - an execute method:
                coordinates updating of input_states, _parameter_states (and params), execution of the function method
                implemented by the subclass, (by calling its _execute method), and updating of the OutputStates
            - one or more parameters, each of which must be (or resolve to) a reference to a ParameterState
                these determine the operation of the function of the Mechanism subclass being instantiated
            - one or more OutputStates:
                the variable of each receives the corresponding item in the output of the Mechanism's function
                the value of each is passed to corresponding MappingProjections for which the Mechanism is a sender
                * Notes:
                    by default, a Mechanism has only one OutputState, assigned to <Mechanism>.outputState;  however:
                    if params[OUTPUT_STATES] is a list (of names) or specification dict (of MechanismOuput State
                    specs), <Mechanism>.outputStates (note plural) is created and contains a dict of outputStates,
                    the first of which points to <Mechanism>.outputState (note singular)
                [TBI * each OutputState maintains a list of Projections for which it serves as the sender]

        Constraints
        -----------
            - the number of input_states must correspond to the length of the variable of the Mechanism's execute method
            - the value of each InputState must be compatible with the corresponding item in the
                variable of the Mechanism's execute method
            - the value of each ParameterState must be compatible with the corresponding parameter of  the Mechanism's
                 execute method
            - the number of outputStates must correspond to the length of the output of the Mechanism's execute method,
                (self.value)
            - the value of each OutputState must be compatible with the corresponding item of the self.value
                 (the output of the Mechanism's execute method)

        Class attributes
        ----------------
            + componentCategory = kwMechanismFunctionCategory
            + className = componentCategory
            + suffix = " <className>"
            + className (str): kwMechanismFunctionCategory
            + suffix (str): " <className>"
            + registry (dict): MechanismRegistry
            + classPreference (PreferenceSet): Mechanism_BasePreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.CATEGORY
            + variableClassDefault (list)
            + paramClassDefaults (dict):
                + MECHANISM_TIME_SCALE (TimeScale): TimeScale.TRIAL (timeScale at which Mechanism executes)
                + [TBI: kwMechanismExecutionSequenceTemplate (list of States):
                    specifies order in which types of States are executed;  used by self.execute]
            + default_mechanism (str): Currently DDM_MECHANISM (class reference resolved in __init__.py)

        Class methods
        -------------
            - _validate_variable(variable, context)
            - _validate_params(request_set, target_set, context)
            - update_states_and_execute(time_scale, params, context):
                updates input, param values, executes <subclass>.function, returns outputState.value
            - terminate_execute(self, context=None): terminates execution of Mechanism (for TimeScale = time_step)
            - adjust(params, context)
                modifies specified Mechanism params (by calling Function._instantiate_defaults)
                returns output
            - plot(): generates a plot of the Mechanism's function using the specified parameter values

        MechanismRegistry
        -----------------
            All Mechanisms are registered in MechanismRegistry, which maintains a dict for each subclass,
              a count for all instances of that type, and a dictionary of those instances
    COMMENT

    Attributes
    ----------

    variable : ndarray : default variableInstanceDefault
        used as input to the Mechanism's `function <Mechanism_Base.function>`.  It is always at least a 2d np.array,
        with each item of axis 0 corresponding to a `value <InputState.value>` of one of the Mechanism's `InputStates
        <InputState>` (in the order they are listed in its `input_states <Mechanism_Base.input_states>` attribute), and
        the first item (i.e., item 0) corresponding to the `value <InputState.value>` of the `primary InputState
        <InputState_Primary>`.  When specified in the **variable** argument of the constructor for the Mechanism,
        it is used as a template to define the format (shape and type of elements) of the input the Mechanism's
        `function <Mechanism_Base.function>`.

        .. _receivesProcessInput (bool): flags if Mechanism (as first in Pathway) receives Process input Projection

    input_state : InputState : default default InputState
        `primary InputState <InputState_Primary>` for the Mechanism;  same as first entry of its `input_states
        <Mechanism_Base.input_states>` attribute.  Its `value <InputState.value>` is assigned as the first item of the
        Mechanism's `variable <Mechanism_Base.variable>`.

    input_states : ContentAddressableList[str, InputState]
        a list of the Mechanism's `InputStates <Mechanism_InputStates>`. The first (and possibly only) entry is always
        the Mechanism's `primary InputState <InputState_Primary>` (i.e., the one in the its `input_state
        <Mechanism_Base.input_state>` attribute).

    input_values : List[List or 1d np.array] : default variableInstanceDefault
        each item in the list corresponds to the `value <InputState.value>` of one of the Mechanism's `InputStates
        <Mechanism_InputStates>` listed in its `input_states <Mechanism_Base.input_states>` attribute.  The value of
        each item is the same as the corresponding item in the Mechanism's `variable <Mechanism_Base.variable>`
        attribute.  The latter is a 2d np.array; the `input_values <Mechanism_Base.input_values>` attribute provides
        this information in a simpler list format.

    _parameter_states : ContentAddressableList[str, ParameterState]
        a list of the Mechanism's `ParameterStates <Mechanism_ParameterStates>`, one for each of its specifiable
        parameters and those of its `function <Mechanism_Base.function>` (i.e., the ones for which there are
        arguments in their constructors).  The value of the parameters of the Mechanism are also accessible as
        attributes of the Mechanism (using the name of the parameter); the function parameters are listed in the
        Mechanism's `function_params <Mechanism_Base.function_params>` attribute, and as attributes of the `Function`
        assigned to its `function_object <Component.function_object>` attribute.

    COMMENT:
       MOVE function and function_params (and add user_params) to Component docstring
    COMMENT

    function : Function, function or method
        the primary function for the Mechanism, called when it is `executed <Mechanism_Execution>`.  It takes the
        Mechanism's `variable <Mechanism_Base.variable>` attribute as its input, and its result is assigned to the
        Mechanism's `value <Mechanism_Base.value>` attribute.

    function_params : Dict[str, value]
        contains the parameters for the Mechanism's `function <Mechanism_Base.function>`.  The key of each entry is the
        name of a parameter of the function, and its value is the parameter's value.

    value : ndarray : default None
        output of the Mechanism's `function <Mechanism_Base.function>`.  It is always at least a 2d np.array, with the
        items of axis 0 corresponding to the values referenced by the corresponding `index <OutputState.index>`
        attribute of the Mechanism's `OutputStates <OutputState>`.  The first item is generally referenced by the
        Mechanism's `primary OutputState <OutputState_Primary>` (i.e., the one in the its `output_state
        <Mechanism_Base.output_state>` attribute).  The `value <Mechanism_Base.value>` is `None` until the Mechanism
        has been executed at least once.

        .. note::
           the `value <Mechanism_Base.value>` of a Mechanism is not necessarily the same as its
           `output_values <Mechanism_Base.output_values>` attribute, which lists the `value <OutputState.value>`\\s
           of its `OutputStates <Mechanism_Base.output_states>`.

    default_value : ndarray : default None
        set equal to the `value <Mechanism_Base.value>` attribute when the Mechanism is first initialized; maintains
        its value even when `value <Mechanism_Base.value>` is reset to None when (re-)initialized prior to execution.

    output_state : OutputState : default default OutputState
        `primary OutputState <OutputState_Primary>` for the Mechanism;  same as first entry of its `output_states
        <Mechanism_Base.output_states>` attribute.

    output_states : ContentAddressableList[str, OutputState]
        a list of the Mechanism's `OutputStates <Mechanism_OutputStates>`.

        There is always
        at least one entry, which identifies the Mechanism's `primary OutputState <OutputState_Primary>`.

        a list of the Mechanism's `OutputStates <Mechanism_OutputStates>`. The first (and possibly only) entry is always
        the Mechanism's `primary OutputState <OutputState_Primary>` (i.e., the one in the its `output_state
        <Mechanism_Base.output_state>` attribute).

    output_values : List[value] : default Mechanism_Base.function(variableInstanceDefault)
        each item in the list corresponds to the `value <OutputState.value>` of one of the Mechanism's `OutputStates
        <Mechanism_OutputStates>` listed in its `output_states <Mechanism_Base.output_states>` attribute.

        .. note:: The `output_values <Mechanism_Base.output_values>` of a Mechanism is not necessarily the same as its
                  `value <Mechanism_Base.value>` attribute, since an OutputState's
                  `function <OutputState.OutputState.function>` and/or its `calculate <Mechanism_Base.calculate>`
                  attribute may use the Mechanism's `value <Mechanism_Base.value>` to generate a derived quantity for
                  the `value <OutputState.OutputState.value>` of that OutputState (and its corresponding item in the
                  the Mechanism's `output_values <Mechanism_Base.output_values>` attribute).

        COMMENT:
            EXAMPLE HERE
        COMMENT

        .. _outputStateValueMapping : Dict[str, int]:
               contains the mappings of OutputStates to their indices in the output_values list
               The key of each entry is the name of an OutputState, and the value is its position in the
                    :py:data:`OutputStates <Mechanism_Base.outputStates>` OrderedDict.
               Used in ``_update_output_states`` to assign the value of each OutputState to the correct item of
                   the Mechanism's ``value`` attribute.
               Any Mechanism with a function that returns a value with more than one item (i.e., len > 1) MUST implement
                   self.execute rather than just use the params[FUNCTION].  This is so that _outputStateValueMapping
                   can be implemented.
               TBI: if the function of a Mechanism is specified only by params[FUNCTION]
                   (i.e., it does not implement self.execute) and it returns a value with len > 1
                   it MUST also specify kwFunctionOutputStateValueMapping.

    is_finished : bool : default False
        set by a Mechanism to signal completion of its `execution <Mechanism_Execution>`; used by `Component-based
        Conditions <Conditions_Component_Based>` to predicate the execution of one or more other Components on the
        Mechanism.

    COMMENT:
        phaseSpec : int or float :  default 0
            determines the `TIME_STEP`\ (s) at which the Mechanism is executed as part of a System
            (see :ref:`Process_Mechanisms` for specification, and :ref:`System Phase <System_Execution_Phase>`
            for how phases are used).
    COMMENT

    processes : Dict[Process, str]:
        a dictionary of the `Processes <Process>` to which the Mechanism belongs, that designates its  `role
        <Mechanism_Role_In_Processes_And_Systems>` in each.  The key of each entry is a Process to which the Mechansim
        belongs, and its value is the Mechanism's `role in that Process <Process_Mechanisms>`.

    systems : Dict[System, str]:
        a dictionary of the `Systems <System>` to which the Mechanism belongs, that designates its `role
        <Mechanism_Role_In_Processes_And_Systems>` in each. The key of each entry is a System to which the Mechanism
        belongs, and its value is the Mechanism's `role in that System <System_Mechanisms>`.

    time_scale : TimeScale : default TimeScale.TRIAL
        determines the default value of the `TimeScale` used by the Mechanism when `executed <Mechanism_Execution>`.

    default_mechanism : Mechanism : default DDM
        type of Mechanism instantiated when the `mechanism` command is called without a specification for its
        **mech_spec** argument.

    name : str : default <Mechanism subclass>-<index>
        the name of the Mechanism.
        Specified in the **name** argument of the constructor for the Mechanism;  if not is specified,
        a default is assigned by `MechanismRegistry` based on the Mechanism's subclass
        (see `Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Mechanism.classPreferences
        the `PreferenceSet` for the Mechanism.
        Specified in the **prefs** argument of the constructor for the Mechanism;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

        .. _stateRegistry : Registry
               registry containing dicts for each State type (InputState, OutputState and ParameterState) with instance
               dicts for the instances of each type and an instance count for each State type in the Mechanism.
               Note: registering instances of State types with the Mechanism (rather than in the StateRegistry)
                     allows the same name to be used for instances of a State type belonging to different Mechanisms
                     without adding index suffixes for that name across Mechanisms
                     while still indexing multiple uses of the same base name within a Mechanism.

    """

    #region CLASS ATTRIBUTES
    componentCategory = kwMechanismComponentCategory
    className = componentCategory
    suffix = " " + className

    registry = MechanismRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY
    # Any preferences specified below will override those specified in CategoryDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to CATEGORY automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'MechanismCustomClassPreferences',
    #     kp<pref>: <setting>...}


    #FIX:  WHEN CALLED BY HIGHER LEVEL OBJECTS DURING INIT (e.g., PROCESS AND SYSTEM), SHOULD USE FULL Mechanism.execute
    # By default, init only the _execute method of Mechanism subclass objects when their execute method is called;
    #    that is, DO NOT run the full Mechanism execute Process, since some components may not yet be instantiated
    #    (such as outputStates)
    initMethod = INIT__EXECUTE__METHOD_ONLY

    # IMPLEMENTATION NOTE: move this to a preference
    default_mechanism = DDM_MECHANISM


    variableClassDefault = [0.0]
    # Note:  the following enforce encoding as 2D np.ndarrays,
    #        to accomodate multiple States:  one 1D np.ndarray per state
    variableEncodingDim = 2
    valueEncodingDim = 2

    # Category specific defaults:
    paramClassDefaults = Component.paramClassDefaults.copy()
    paramClassDefaults.update({
        INPUT_STATES:None,
        OUTPUT_STATES:None,
        MONITOR_FOR_CONTROL: NotImplemented,  # This has to be here to "register" it as a valid param for the class
                                              # but is set to NotImplemented so that it is ignored if it is not
                                              # assigned;  setting it to None actively disallows assignment
                                              # (see EVCMechanism_instantiate_input_states for more details)
        MONITOR_FOR_LEARNING: None,
        # TBI - kwMechanismExecutionSequenceTemplate: [
        #     Components.States.InputState.InputState,
        #     Components.States.ParameterState.ParameterState,
        #     Components.States.OutputState.OutputState]
        MECHANISM_TIME_SCALE: TimeScale.TRIAL
        })

    # def __new__(cls, *args, **kwargs):
    # def __new__(cls, name=NotImplemented, params=NotImplemented, context=None):
    #endregion

    @tc.typecheck
    def __init__(self,
                 variable=None,
                 size=None,
                 input_states=None,
                 output_states=None,
                 params=None,
                 name=None,
                 prefs=None,
                 context=None):
        """Assign name, category-level preferences, register Mechanism, and enforce category methods

        This is an abstract class, and can only be called from a subclass;
           it must be called by the subclass with a context value

        NOTES:
        * Since Mechanism is a subclass of Component, it calls super.__init__
            to validate size and default_variable and param_defaults, and assign params to paramInstanceDefaults;
            it uses INPUT_STATE as the default_variable
        * registers Mechanism with MechanismRegistry

        """

        # Forbid direct call to base class constructor
        if context is None or (not isinstance(context, type(self)) and not VALIDATE in context):
            # raise MechanismError("Direct call to abstract class Mechanism() is not allowed; "
                                 # "use mechanism() or one of the following subclasses: {0}".
                                 # format(", ".join("{!s}".format(key) for (key) in MechanismRegistry.keys())))
                                 # format(", ".join("{!s}".format(key) for (key) in MechanismRegistry.keys())))
            raise MechanismError("Direct call to abstract class Mechanism() is not allowed; "
                                 "use Mechanism() or a subclass")

        # IMPLEMENT **kwargs (PER State)


        # Ensure that all input_states and output_states, whether from paramClassDefaults or constructor arg,
        #    have been included in user_params and implemented as properties
        #    (in case the subclass did not include one and/or the other as an argument in its constructor)

        kwargs = {}

        # input_states = []
        # if INPUT_STATES in self.paramClassDefaults and self.paramClassDefaults[INPUT_STATES]:
        #     input_states.extend(self.paramClassDefaults[INPUT_STATES])
        # if INPUT_STATES in self.user_params and self.user_params[INPUT_STATES]:
        #     input_states.extend(self.user_params[INPUT_STATES])
        # if input_states:
        #     kwargs[INPUT_STATES] = input_states
        #
        # output_states = []
        # if OUTPUT_STATES in self.paramClassDefaults and self.paramClassDefaults[OUTPUT_STATES]:
        #     output_states.extend(self.paramClassDefaults[OUTPUT_STATES])
        # if OUTPUT_STATES in self.user_params and self.user_params[OUTPUT_STATES]:
        #     output_states.extend(self.user_params[OUTPUT_STATES])
        # if output_states:
        #     kwargs[OUTPUT_STATES] = output_states
        #
        # kwargs[PARAMS] = params
        #
        # params = self._assign_args_to_param_dicts(**kwargs)

        self._execution_id = None
        self.is_finished = False

        # Register with MechanismRegistry or create one
        if not context is VALIDATE:
            register_category(entry=self,
                              base_class=Mechanism_Base,
                              name=name,
                              registry=MechanismRegistry,
                              context=context)

        # Create Mechanism's _stateRegistry and state type entries
        from PsyNeuLink.Components.States.State import State_Base
        self._stateRegistry = {}

        # InputState
        from PsyNeuLink.Components.States.InputState import InputState
        register_category(entry=InputState,
                          base_class=State_Base,
                          registry=self._stateRegistry,
                          context=context)
        # ParameterState
        from PsyNeuLink.Components.States.ParameterState import ParameterState
        register_category(entry=ParameterState,
                          base_class=State_Base,
                          registry=self._stateRegistry,
                          context=context)
        # OutputState
        from PsyNeuLink.Components.States.OutputState import OutputState
        register_category(entry=OutputState,
                          base_class=State_Base,
                          registry=self._stateRegistry,
                          context=context)

        # Mark initialization in context
        if not context or isinstance(context, object) or inspect.isclass(context):
            context = INITIALIZING + self.name + SEPARATOR_BAR + self.__class__.__name__
        else:
            context = context + SEPARATOR_BAR + INITIALIZING + self.name

        super(Mechanism_Base, self).__init__(default_variable=variable,
                                             size=size,
                                             param_defaults=params,
                                             prefs=prefs,
                                             name=name,
                                             context=context)

        # FUNCTIONS:

# IMPLEMENTATION NOTE:  REPLACE THIS WITH ABC (ABSTRACT CLASS)
        # Assign class functions
        self.classMethods = {
            kwMechanismExecuteFunction: self.execute,
            # kwMechanismAdjustFunction: self.adjust_function,
            # kwMechanismTerminateFunction: self.terminate_execute
        }
        self.classMethodNames = self.classMethods.keys()

        #  Validate class methods:
        #    make sure all required ones have been implemented in (i.e., overridden by) subclass
        for name, method in self.classMethods.items():
            try:
                method
            except (AttributeError):
                raise MechanismError("{0} is not implemented in Mechanism class {1}".
                                     format(name, self.name))

        try:
            self._default_value = self.value.copy()
        except AttributeError:
            self._default_value = self.value
        self.value = self._old_value = None
        self._status = INITIALIZING
        self._receivesProcessInput = False
        self.phaseSpec = None
        self.processes = {}
        self.systems = {}


    def _validate_variable(self, variable, context=None):
        """Convert variableClassDefault and self.variable to 2D np.array: one 1D value for each InputState

        # VARIABLE SPECIFICATION:                                        ENCODING:
        # Simple value variable:                                         0 -> [array([0])]
        # Single state array (vector) variable:                         [0, 1] -> [array([0, 1])
        # Multiple state variables, each with a single value variable:  [[0], [0]] -> [array[0], array[0]]

        :param variable:
        :param context:
        :return:
        """

        super(Mechanism_Base, self)._validate_variable(variable, context)

        # Force Mechanism variable specification to be a 2D array (to accomodate multiple InputStates - see above):
        # Note: _instantiate_input_states (below) will parse into 1D arrays, one for each InputState
        self.variableClassDefault = convert_to_np_array(self.variableClassDefault, 2)
        self.variable = convert_to_np_array(self.variable, 2)

    def _filter_params(self, params):
        """Add rather than override INPUT_STATES and/or OUTPUT_STATES

        Allows specification of INPUT_STATES or OUTPUT_STATES in params dictionary to be added to,
        rather than override those in paramClassDefaults (the default behavior)
        """

        import copy

        # INPUT_STATES:

        # Check if input_states is in params (i.e., was specified in arg of constructor)
        if not INPUT_STATES in params or params[INPUT_STATES] is None:
            # If it wasn't, assign from paramClassDefaults (even if it is None) to force creation of input_states attrib
            if self.paramClassDefaults[INPUT_STATES] is not None:
                params[INPUT_STATES] = copy.deepcopy(self.paramClassDefaults[INPUT_STATES])
            else:
                params[INPUT_STATES] = None
        # Convert input_states_spec to list if it is not one
        if params[INPUT_STATES] is not None and not isinstance(params[INPUT_STATES], (list, dict)):
            params[INPUT_STATES] = [params[INPUT_STATES]]
        self.user_params.__additem__(INPUT_STATES, params[INPUT_STATES])

        # OUTPUT_STATES:

        # Check if OUTPUT_STATES is in params (i.e., was specified in arg of contructor)
        if not OUTPUT_STATES in params or params[OUTPUT_STATES] is None:
            if self.paramClassDefaults[OUTPUT_STATES] is not None:
                params[OUTPUT_STATES] = copy.deepcopy(self.paramClassDefaults[OUTPUT_STATES])
            else:
                params[OUTPUT_STATES] = None
        # Convert OUTPUT_STATES_spec to list if it is not one
        if params[OUTPUT_STATES] is not None and not isinstance(params[OUTPUT_STATES], (list, dict)):
            params[OUTPUT_STATES] = [params[OUTPUT_STATES]]
        self.user_params.__additem__(OUTPUT_STATES, params[OUTPUT_STATES])

        # try:
        #     input_states_spec = params[INPUT_STATES]
        # except KeyError:
        #     pass
        # else:
        #     # Convert input_states_spec to list if it is not one
        #     if not isinstance(input_states_spec, list):
        #         input_states_spec = [input_states_spec]
        #     # # Get input_states specified in paramClassDefaults
        #     # if self.paramClassDefaults[INPUT_STATES] is not None:
        #     #     default_input_states = self.paramClassDefaults[INPUT_STATES].copy()
        #     # else:
        #     #     default_input_states = None
        #     # # Convert input_states from paramClassDefaults to a list if it is not one
        #     # if default_input_states is not None and not isinstance(default_input_states, list):
        #     #     default_input_states = [default_input_states]
        #     # # Add InputState specified in params to those in paramClassDefaults
        #     # #    Note: order is important here;  new ones should be last, as paramClassDefaults defines the
        #     # #          the primary InputState which must remain first for the input_states ContentAddressableList
        #     # default_input_states.extend(input_states_spec)
        #     # # Assign full set back to params_arg
        #     # params[INPUT_STATES] = default_input_states
        #
        #     # Get inputStates specified in paramClassDefaults
        #     if self.paramClassDefaults[INPUT_STATES] is not None:
        #         default_input_states = self.paramClassDefaults[INPUT_STATES].copy()
        #         # Convert inputStates from paramClassDefaults to a list if it is not one
        #         if not isinstance(default_input_states, list):
        #             default_input_states = [default_input_states]
        #         # Add input_states specified in params to those in paramClassDefaults
        #         #    Note: order is important here;  new ones should be last, as paramClassDefaults defines the
        #         #          the primary InputState which must remain first for the input_states ContentAddressableList
        #         default_input_states.extend(input_states_spec)
        #         # Assign full set back to params_arg
        #         params[INPUT_STATES] = default_input_states

        # # OUTPUT_STATES:
        # try:
        #     output_states_spec = params[OUTPUT_STATES]
        # except KeyError:
        #     pass
        # else:
        #     # Convert output_states_spec to list if it is not one
        #     if not isinstance(output_states_spec, list):
        #         output_states_spec = [output_states_spec]
        #     # Get outputStates specified in paramClassDefaults
        #     default_output_states = self.paramClassDefaults[OUTPUT_STATES].copy()
        #     # Convert outputStates from paramClassDefaults to a list if it is not one
        #     if not isinstance(default_output_states, list):
        #         default_output_states = [default_output_states]
        #     # Add output_states specified in params to those in paramClassDefaults
        #     #    Note: order is important here;  new ones should be last, as paramClassDefaults defines the
        #     #          the primary OutputState which must remain first for the output_states ContentAddressableList
        #     default_output_states.extend(output_states_spec)
        #     # Assign full set back to params_arg
        #     params[OUTPUT_STATES] = default_output_states

    def _validate_params(self, request_set, target_set=None, context=None):
        """validate TimeScale, INPUT_STATES, FUNCTION_PARAMS, OUTPUT_STATES and MONITOR_FOR_CONTROL

        Go through target_set params (populated by Component._validate_params) and validate values for:
            + TIME_SCALE:  <TimeScale>
            + INPUT_STATES:
                <MechanismsInputState or Projection object or class,
                specification dict for one, 2-item tuple, or numeric value(s)>;
                if it is missing or not one of the above types, it is set to self.variable
            + FUNCTION_PARAMS:  <dict>, every entry of which must be one of the following:
                ParameterState or Projection object or class, specification dict for one, 2-item tuple, or numeric
                value(s);
                if invalid, default (from paramInstanceDefaults or paramClassDefaults) is assigned
            + OUTPUT_STATES:
                <MechanismsOutputState object or class, specification dict, or numeric value(s);
                if it is missing or not one of the above types, it is set to None here;
                    and then to default value of self.value (output of execute method) in instantiate_output_state
                    (since execute method must be instantiated before self.value is known)
                if OUTPUT_STATES is a list or OrderedDict, it is passed along (to instantiate_output_states)
                if it is a OutputState class ref, object or specification dict, it is placed in a list
            + MONITORED_STATES:
                ** DOCUMENT

        Note: PARAMETER_STATES are validated separately -- ** DOCUMENT WHY

        TBI - Generalize to go through all params, reading from each its type (from a registry),
                                   and calling on corresponding subclass to get default values (if param not found)
                                   (as PROJECTION_TYPE and PROJECTION_SENDER are currently handled)
        """

        # Perform first-pass validation in Function.__init__():
        # - returns full set of params based on subclass paramClassDefaults
        super(Mechanism, self)._validate_params(request_set,target_set,context)

        params = target_set

        #VALIDATE TIME SCALE
        try:
            param_value = params[TIME_SCALE]
        except KeyError:
            if any(context_string in context for context_string in {COMMAND_LINE, SET_ATTRIBUTE}):
                pass
            else:
                self.timeScale = timeScaleSystemDefault
        else:
            if isinstance(param_value, TimeScale):
                self.timeScale = params[TIME_SCALE]
            else:
                if self.prefs.verbosePref:
                    print("Value for {0} ({1}) param of {2} must be of type {3};  default will be used: {4}".
                          format(TIME_SCALE, param_value, self.name, type(TimeScale), timeScaleSystemDefault))

        # VALIDATE INPUT STATE(S)

        # INPUT_STATES is specified, so validate:
        if INPUT_STATES in params and params[INPUT_STATES] is not None:

            param_value = params[INPUT_STATES]

            # If it is a single item or a non-OrderedDict, place in a list (for use here and in instantiate_inputState)
            if not isinstance(param_value, (list, OrderedDict, ContentAddressableList)):
                param_value = [param_value]
            # Validate each item in the list or OrderedDict
            # Note:
            # * number of input_states is validated against length of the owner Mechanism's variable
            #     in instantiate_inputState, where an input_state is assigned to each item of variable
            i = 0
            for key, item in param_value if isinstance(param_value, dict) else enumerate(param_value):
                from PsyNeuLink.Components.States.InputState import InputState
                # If not valid...
                if not ((isclass(item) and (issubclass(item, InputState) or # InputState class ref
                                                issubclass(item, Projection))) or    # Project class ref
                            isinstance(item, InputState) or      # InputState object
                            isinstance(item, dict) or            # InputState specification dict
                            isinstance(item, str) or             # Name (to be used as key in input_states dict)
                            iscompatible(item, **{kwCompatibilityNumeric: True})):   # value
                    # set to None, so it is set to default (self.variable) in instantiate_inputState
                    param_value[key] = None
                    if self.prefs.verbosePref:
                        print("Item {0} of {1} param ({2}) in {3} is not a"
                              " InputState, specification dict or value, nor a list of dict of them; "
                              "variable ({4}) of execute method for {5} will be used"
                              " to create a default OutputState for {3}".
                              format(i,
                                     INPUT_STATES,
                                     param_value,
                                     self.__class__.__name__,
                                     self.variable,
                                     self.execute.__self__.name))
                i += 1
            params[INPUT_STATES] = param_value

        # INPUT_STATES is not specified
        else:
            # pass if call is from assign_params (i.e., not from an init method)
            if any(context_string in context for context_string in {COMMAND_LINE, SET_ATTRIBUTE}):
                pass
            else:
                # INPUT_STATES not specified:
                # - set to None, so that it is set to default (self.variable) in instantiate_inputState
                # - if in VERBOSE mode, warn in instantiate_inputState, where default value is known
                params[INPUT_STATES] = None

        # VALIDATE FUNCTION_PARAMS
        try:
            function_param_specs = params[FUNCTION_PARAMS]
        except KeyError:
            if any(context_string in context for context_string in {COMMAND_LINE, SET_ATTRIBUTE}):
                pass
            elif self.prefs.verbosePref:
                print("No params specified for {0}".format(self.__class__.__name__))
        else:
            if not (isinstance(function_param_specs, dict)):
                raise MechanismError("{0} in {1} must be a dict of param specifications".
                                     format(FUNCTION_PARAMS, self.__class__.__name__))
            # Validate params
            from PsyNeuLink.Components.States.ParameterState import ParameterState
            for param_name, param_value in function_param_specs.items():
                try:
                    default_value = self.paramInstanceDefaults[FUNCTION_PARAMS][param_name]
                except KeyError:
                    raise MechanismError("{0} not recognized as a param of execute method for {1}".
                                         format(param_name, self.__class__.__name__))
                if not ((isclass(param_value) and
                             (issubclass(param_value, ParameterState) or
                                  issubclass(param_value, Projection))) or
                        isinstance(param_value, ParameterState) or
                        isinstance(param_value, Projection) or
                        isinstance(param_value, dict) or
                        iscompatible(param_value, default_value)):
                    params[FUNCTION_PARAMS][param_name] = default_value
                    if self.prefs.verbosePref:
                        print("{0} param ({1}) for execute method {2} of {3} is not a ParameterState, "
                              "projection, tuple, or value; default value ({4}) will be used".
                              format(param_name,
                                     param_value,
                                     self.execute.__self__.componentName,
                                     self.__class__.__name__,
                                     default_value))

        # VALIDATE OUTPUT STATE(S)

        # OUTPUT_STATES is specified, so validate:
        if OUTPUT_STATES in params and params[OUTPUT_STATES] is not None:

            param_value = params[OUTPUT_STATES]

            # If it is a single item or a non-OrderedDict, place in list (for use here and in instantiate_output_state)
            if not isinstance(param_value, (ContentAddressableList, list, OrderedDict)):
                param_value = [param_value]
            # Validate each item in the list or OrderedDict
            i = 0
            for key, item in param_value if isinstance(param_value, dict) else enumerate(param_value):
                from PsyNeuLink.Components.States.OutputState import OutputState
                # If not valid...
                if not ((isclass(item) and issubclass(item, OutputState)) or # OutputState class ref
                            isinstance(item, OutputState) or   # OutputState object
                            isinstance(item, dict) or                   # OutputState specification dict
                            isinstance(item, str) or                    # Name (to be used as key in outputStates dict)
                            iscompatible(item, **{kwCompatibilityNumeric: True})):  # value
                    # set to None, so it is set to default (self.value) in instantiate_output_state
                    param_value[key] = None
                    if self.prefs.verbosePref:
                        print("Item {0} of {1} param ({2}) in {3} is not a"
                              " OutputState, specification dict or value, nor a list of dict of them; "
                              "output ({4}) of execute method for {5} will be used"
                              " to create a default OutputState for {3}".
                              format(i,
                                     OUTPUT_STATES,
                                     param_value,
                                     self.__class__.__name__,
                                     self.value,
                                     self.execute.__self__.name))
                i += 1
            params[OUTPUT_STATES] = param_value

        # OUTPUT_STATES is not specified
        else:
            # pass if call is from assign_params (i.e., not from an init method)
            if any(context_string in context for context_string in {COMMAND_LINE, SET_ATTRIBUTE}):
                pass
            else:
                # OUTPUT_STATES not specified:
                # - set to None, so that it is set to default (self.value) in instantiate_output_state
                # Notes:
                # * if in VERBOSE mode, warning will be issued in instantiate_output_state, where default value is known
                # * number of outputStates is validated against length of owner Mechanism's execute method output (EMO)
                #     in instantiate_output_state, where an OutputState is assigned to each item (value) of the EMO
                params[OUTPUT_STATES] = None

    def _validate_inputs(self, inputs=None):
        # Only ProcessingMechanism supports run() method of Function;  ControlMechanism and LearningMechanism do not
        raise MechanismError("{} does not support run() method".format(self.__class__.__name__))

    def _instantiate_attributes_before_function(self, context=None):

        self._instantiate_input_states(context=context)
        self._instantiate_parameter_states(context=context)
        super()._instantiate_attributes_before_function(context=context)

    def _instantiate_function(self, context=None):
        """Assign weights and exponents if specified in input_states
        """

        super()._instantiate_function(context=context)

        if self.input_states and any(input_state.weight is not None for input_state in self.input_states):

            # Construct defaults:
            #    from function_object.weights if specified else 1's
            try:
                default_weights = self.function_object.weights
            except AttributeError:
                default_weights = None
            if default_weights is None:
                default_weights = default_weights or [1.0] * len(self.input_states)

            # Assign any weights specified in input_state spec
            weights = [[input_state.weight if input_state.weight is not None else default_weight]
                       for input_state, default_weight in zip(self.input_states, default_weights)]
            self.function_object._weights = weights

        if self.input_states and any(input_state.exponent is not None for input_state in self.input_states):

            # Construct defaults:
            #    from function_object.weights if specified else 1's
            try:
                default_exponents = self.function_object.exponents
            except AttributeError:
                default_exponents = None
            if default_exponents is None:
                default_exponents = default_exponents or [1.0] * len(self.input_states)

            # Assign any exponents specified in input_state spec
            exponents = [[input_state.exponent if input_state.exponent is not None else default_exponent]
                       for input_state, default_exponent in zip(self.input_states, default_exponents)]
            self.function_object._exponents = exponents

    def _instantiate_attributes_after_function(self, context=None):

        self._instantiate_output_states(context=context)
        super()._instantiate_attributes_after_function(context=context)

    def _instantiate_input_states(self, context=None):
        """Call State._instantiate_input_states to instantiate orderedDict of InputState(s)

        This is a stub, implemented to allow Mechanism subclasses to override _instantiate_input_states
            or process InputStates before and/or after call to _instantiate_input_states
        """
        from PsyNeuLink.Components.States.InputState import _instantiate_input_states
        _instantiate_input_states(owner=self, input_states=self.input_states, context=context)
        _instantiate_input_states(owner=self, context=context)

    def _instantiate_parameter_states(self, context=None):
        """Call State._instantiate_parameter_states to instantiate a ParameterState for each parameter in user_params

        This is a stub, implemented to allow Mechanism subclasses to override _instantiate_parameter_states
            or process InputStates before and/or after call to _instantiate_parameter_states
        """

        from PsyNeuLink.Components.States.ParameterState import _instantiate_parameter_states
        _instantiate_parameter_states(owner=self, context=context)

    def _instantiate_output_states(self, context=None):
        """Call State._instantiate_output_states to instantiate orderedDict of OutputState(s)

        This is a stub, implemented to allow Mechanism subclasses to override _instantiate_output_states
            or process InputStates before and/or after call to _instantiate_output_states
        """
        from PsyNeuLink.Components.States.OutputState import _instantiate_output_states
        _instantiate_output_states(owner=self, output_states=self.output_states, context=context)

    def _add_projection_to_mechanism(self, state, projection, context=None):

        from PsyNeuLink.Components.Projections.Projection import _add_projection_to
        _add_projection_to(receiver=self, state=state, projection_spec=projection, context=context)

    def _add_projection_from_mechanism(self, receiver, state, projection, context=None):
        """Add projection to specified state
        """
        from PsyNeuLink.Components.Projections.Projection import _add_projection_from
        _add_projection_from(sender=self, state=state, projection_spec=projection, receiver=receiver, context=context)

    def execute(self,
                input=None,
                runtime_params=None,
                clock=CentralClock,
                time_scale=TimeScale.TRIAL,
                ignore_execution_id = False,
                context=None):
        """Carry out a single `execution <Mechanism_Execution>` of the Mechanism.


        COMMENT:
            Update InputState(s) and parameter(s), call subclass _execute, update OutputState(s), and assign self.value

            Execution sequence:
            - Call self.input_state.execute() for each entry in self.input_states:
                + execute every self.input_state.path_afferents.[<Projection>.execute()...]
                + aggregate results and/or gate state using self.input_state.function()
                + assign the result in self.input_state.value
            - Call every self.params[<ParameterState>].execute(); for each:
                + execute self.params[<ParameterState>].mod_afferents.[<Projection>.execute()...]
                    (usually this is just a single ControlProjection)
                + aggregate results for each ModulationParam or assign value from an OVERRIDE specification
                + assign the result to self.params[<ParameterState>].value
            - Call subclass' self.execute(params):
                - use self.input_state.value as its variable,
                - use self.params[<ParameterState>].value for each param of subclass' self.function
                - call self._update_output_states() to assign the output to each self.output_states[<OutputState>].value
                Note:
                * if execution is occurring as part of initialization, each output_state is reset to 0
                * otherwise, their values are left as is until the next update
        COMMENT

        Arguments
        ---------

        input : List[value] or ndarray : default variableInstanceDefault
            input to use for execution of the Mechanism.
            This must be consistent with the format of the Mechanism's `InputState(s) <Mechanism_InputStates>`:
            the number of items in the  outermost level of the list, or axis 0 of the ndarray, must equal the number
            of the Mechanism's `input_states  <Mechanism_Base.input_states>`, and each item must be compatible with the
            format (number and type of elements) of the `variable <InputState.InputState.variable>` of the
            corresponding InputState (see `Run Inputs <Run_Inputs>` for details of input
            specification formats).

        runtime_params : Optional[Dict[str, Dict[str, Dict[str, value]]]]:
            a dictionary that can include any of the parameters used as arguments to instantiate the Mechanism,
            its function, or `Projection(s) to any of its States <State_Projections>`.  Any value assigned to a
            parameter will override the current value of that parameter for the (and only the current) execution of
            the Mechanism, and will return to its previous value following execution (unless the
            `runtimeParamStickyAssignmentPref` is set for the Component to which the parameter belongs).  See
            `runtime_params <Mechanism_Runtime_Parameters>` above for details concerning specification.

        time_scale : TimeScale :  default TimeScale.TRIAL
            specifies whether the Mechanism is executed for a single `TIME_STEP` or a `TRIAL`.

        Returns
        -------

        Mechanism's output_values : List[value]
            list with the `value <OutputState.value>` of each of the Mechanism's `OutputStates
            <Mechanism_OutputStates>` after either one `TIME_STEP` or a `TRIAL`.

        """
        self.ignore_execution_id = ignore_execution_id
        context = context or NO_CONTEXT

        # IMPLEMENTATION NOTE: Re-write by calling execute methods according to their order in functionDict:
        #         for func in self.functionDict:
        #             self.functionsDict[func]()

        # Limit init to scope specified by context
        if INITIALIZING in context:
            if PROCESS_INIT in context or SYSTEM_INIT in context:
                # Run full execute method for init of Process and System
                pass
            # Only call subclass' _execute method and then return (do not complete the rest of this method)
            elif self.initMethod is INIT__EXECUTE__METHOD_ONLY:
                return_value =  self._execute(variable=self.variable,
                                                 runtime_params=runtime_params,
                                                 clock=clock,
                                                 time_scale=time_scale,
                                                 context=context)

                # # # MODIFIED 3/3/17 OLD:
                # # return np.atleast_2d(return_value)
                # # MODIFIED 3/3/17 NEW:
                # converted_to_2d = np.atleast_2d(return_value)
                # MODIFIED 3/7/17 NEWER:
                # IMPLEMENTATION NOTE:  THIS IS HERE BECAUSE IF return_value IS A LIST, AND THE LENGTH OF ALL OF ITS
                #                       ELEMENTS ALONG ALL DIMENSIONS ARE EQUAL (E.G., A 2X2 MATRIX PAIRED WITH AN
                #                       ARRAY OF LENGTH 2), np.array (AS WELL AS np.atleast_2d) GENERATES A ValueError
                if (isinstance(return_value, list) and
                    (all(isinstance(item, np.ndarray) for item in return_value) and
                        all(
                                all(item.shape[i]==return_value[0].shape[0]
                                    for i in range(len(item.shape)))
                                for item in return_value))):

                        return return_value
                else:
                    converted_to_2d = np.atleast_2d(return_value)
                # If return_value is a list of heterogenous elements, return as is
                #     (satisfies requirement that return_value be an array of possibly multidimensional values)
                if converted_to_2d.dtype == object:
                    return return_value
                # Otherwise, return value converted to 2d np.array
                else:
                    return converted_to_2d
                # MODIFIED 3/3/17 END

            # Call only subclass' function during initialization (not its full _execute method nor rest of this method)
            elif self.initMethod is INIT_FUNCTION_METHOD_ONLY:
                return_value = self.function(variable=self.variable,
                                             params=runtime_params,
                                             time_scale=time_scale,
                                             context=context)
                return np.atleast_2d(return_value)


        #region VALIDATE RUNTIME PARAMETER SETS
        # Insure that param set is for a States:
        if self.prefs.paramValidationPref:
            if runtime_params:
                # runtime_params can have entries for any of the the Mechanism's params, or
                #    one or more state keys, each of which should be for a params dictionary for the corresponding
                #    state type, and each of can contain only parameters relevant to that state
                state_keys = [INPUT_STATE_PARAMS, PARAMETER_STATE_PARAMS, OUTPUT_STATE_PARAMS]
                param_names = list({**self.user_params, **self.function_params})
                if not all(key in state_keys + param_names for key in runtime_params):
                        raise MechanismError("There is an invalid specification for a runtime parameter of {}".
                                             format(self.name))
                # for state_key in runtime_params:
                for state_key in [entry for entry in runtime_params if entry in state_keys]:
                    state_dict = runtime_params[state_key]
                    if not isinstance(state_dict, dict):
                        raise MechanismError("runtime_params entry for {} is not a dict".
                                             format(self.name, state_key))
                    for param_name in state_dict:
                        if not param_name in param_names:
                            raise MechanismError("{} entry in runtime_params for {} "
                                                 "contains an unrecognized parameter: {}".
                                                 format(state_key, self.name, param_name))

        #endregion

        # FIX: ??MAKE CONDITIONAL ON self.prefs.paramValidationPref??
        #region VALIDATE INPUT STATE(S) AND RUNTIME PARAMS
        self._check_args(variable=self.variable,
                        params=runtime_params,
                        target_set=runtime_params)
        #endregion

        #region UPDATE INPUT STATE(S)

        # Executing or simulating Process or System, get input by updating input_states

        if input is None and (EXECUTING in context or EVC_SIMULATION in context) and (self.input_state.path_afferents != []):
            self._update_input_states(runtime_params=runtime_params, time_scale=time_scale, context=context)

        # Direct call to execute Mechanism with specified input, so assign input to Mechanism's input_states
        else:
            if context is NO_CONTEXT:
                context = EXECUTING + ' ' + append_type_to_name(self)
                self.execution_status = ExecutionStatus.EXECUTING
            if input is None:
                input = self.variableInstanceDefault
            self._assign_input(input)

        #endregion

        #region UPDATE PARAMETER STATE(S)
        self._update_parameter_states(runtime_params=runtime_params, time_scale=time_scale, context=context)
        #endregion

        #region CALL SUBCLASS _execute method AND ASSIGN RESULT TO self.value

        self.value = self._execute(variable=self.variable,
                                   runtime_params=runtime_params,
                                   clock=clock,
                                   time_scale=time_scale,
                                   context=context)

        # # MODIFIED 3/3/17 OLD:
        # self.value = np.atleast_2d(self.value)
        # # MODIFIED 3/3/17 NEW:
        # converted_to_2d = np.atleast_2d(self.value)
        # # If self.value is a list of heterogenous elements, leave as is;
        # # Otherwise, use converted value (which is a genuine 2d array)
        # if converted_to_2d.dtype != object:
        #     self.value = converted_to_2d
        # MODIFIED 3/8/17 NEWER:
        # IMPLEMENTATION NOTE:  THIS IS HERE BECAUSE IF return_value IS A LIST, AND THE LENGTH OF ALL OF ITS
        #                       ELEMENTS ALONG ALL DIMENSIONS ARE EQUAL (E.G., A 2X2 MATRIX PAIRED WITH AN
        #                       ARRAY OF LENGTH 2), np.array (AS WELL AS np.atleast_2d) GENERATES A ValueError
        if (isinstance(self.value, list) and
            (all(isinstance(item, np.ndarray) for item in self.value) and
                all(
                        all(item.shape[i]==self.value[0].shape[0]
                            for i in range(len(item.shape)))
                        for item in self.value))):
                # return self.value
                pass
        else:
            converted_to_2d = np.atleast_2d(self.value)
            # If return_value is a list of heterogenous elements, return as is
            #     (satisfies requirement that return_value be an array of possibly multidimensional values)
            if converted_to_2d.dtype == object:
                # return self.value
                pass
            # Otherwise, return value converted to 2d np.array
            else:
                # return converted_to_2d
                self.value = converted_to_2d
        # MODIFIED 3/3/17 END

        # Set status based on whether self.value has changed
        self.status = self.value

        #endregion


        #region UPDATE OUTPUT STATE(S)
        self._update_output_states(runtime_params=runtime_params, time_scale=time_scale, context=context)
        #endregion

        #region REPORT EXECUTION
        if self.prefs.reportOutputPref and context and EXECUTING in context:
            self._report_mechanism_execution(self.input_values, self.user_params, self.output_state.value)
        #endregion

        #region RE-SET STATE_VALUES AFTER INITIALIZATION
        # If this is (the end of) an initialization run, restore state values to initial condition
        if '_init_' in context:
            for state in self.input_states:
                self.input_states[state].value = self.input_states[state].variable
            for state in self._parameter_states:
                self._parameter_states[state].value =  getattr(self, '_'+state)
            for state in self.output_states:
                # Zero outputStates in case of recurrence:
                #    don't want any non-zero values as a residuum of initialization runs to be
                #    transmittted back via recurrent Projections as initial inputs
                self.output_states[state].value = self.output_states[state].value * 0.0
        #endregion

        #endregion

        return self.value

    def run(self,
            inputs,
            num_trials=None,
            call_before_execution=None,
            call_after_execution=None,
            time_scale=None):
        """Run a sequence of `executions <Mechanism_Execution>`.

        COMMENT:
            Call execute method for each in a sequence of executions specified by the `inputs` argument.
        COMMENT

        Arguments
        ---------

        inputs : List[input] or ndarray(input) : default default_variable
            the inputs used for each in a sequence of executions of the Mechanism (see `Run_Inputs` for a detailed
            description of formatting requirements and options).

        num_trials: int
            number of trials to execute.

        call_before_execution : function : default None
            called before each execution of the Mechanism.

        call_after_execution : function : default None
            called after each execution of the Mechanism.

        time_scale : TimeScale : default TimeScale.TRIAL
            specifies whether the Mechanism is executed for a single `TIME_STEP` or a `TRIAL`.

        Returns
        -------

        Mechanism's output_values : List[value]
            list with the `value <OutputState.value>` of each of the Mechanism's `OutputStates
            <Mechanism_OutputStates>` for each execution of the Mechanism.

        """
        from PsyNeuLink.Globals.Run import run
        return run(self,
                   inputs=inputs,
                   num_trials=num_trials,
                   call_before_trial=call_before_execution,
                   call_after_trial=call_after_execution,
                   time_scale=time_scale)

    def _assign_input(self, input):

        input = np.atleast_2d(input)
        num_inputs = np.size(input,0)
        num_input_states = len(self.input_states)
        if num_inputs != num_input_states:
            # Check if inputs are of different lengths (indicated by dtype == np.dtype('O'))
            num_inputs = np.size(input)
            if isinstance(input, np.ndarray) and input.dtype is np.dtype('O') and num_inputs == num_input_states:
                pass
            else:
                num_inputs = np.size(input, 0)  # revert num_inputs to its previous value, when printing the error
                raise SystemError("Number of inputs ({0}) to {1} does not match "
                                  "its number of input_states ({2})".
                                  format(num_inputs, self.name,  num_input_states ))
        for i in range(num_input_states):
            # input_state = list(self.input_states.values())[i]
            input_state = self.input_states[i]
            # input_item = np.ndarray(input[i])
            input_item = input[i]

            if len(input_state.variable) == len(input_item):
                input_state.value = input_item
            else:
                raise MechanismError("Length ({}) of input ({}) does not match "
                                     "required length ({}) for input to {} of {}".
                                     format(len(input_item),
                                            input[i],
                                            len(input_state.variable),
                                            input_state.name,
                                            append_type_to_name(self)))
        self.variable = np.array(self.input_values)

    def _update_input_states(self, runtime_params=None, time_scale=None, context=None):
        """ Update value for each InputState in self.input_states:

        Call execute method for all (MappingProjection) Projections in InputState.path_afferents
        Aggregate results (using InputState execute method)
        Update InputState.value
        """
        for i in range(len(self.input_states)):
            state = self.input_states[i]
            state.update(params=runtime_params, time_scale=time_scale, context=context)
        self.variable = np.array(self.input_values)

    def _update_parameter_states(self, runtime_params=None, time_scale=None, context=None):

        for state in self._parameter_states:

            state.update(params=runtime_params, time_scale=time_scale, context=context)

    def _update_output_states(self, runtime_params=None, time_scale=None, context=None):
        """Execute function for each OutputState and assign result of each to corresponding item of self.output_values

        """
        for state in self.output_states:
            state.update(params=runtime_params, time_scale=time_scale, context=context)

    def initialize(self, value):
        """Assign an initial value to the Mechanism's `value <Mechanism_Base.value>` attribute and update its
        `OutputStates <Mechanism_OutputStates>`.

        COMMENT:
            Takes a number or 1d array and assigns it to the first item of the Mechanism's
            `value <Mechanism_Base.value>` attribute.
        COMMENT

        Arguments
        ---------

        value : List[value] or 1d ndarray
            value used to initialize the first item of the Mechanism's `value <Mechanism_Base.value>` attribute.

        """
        if self.paramValidationPref:
            if not iscompatible(value, self.value):
                raise MechanismError("Initialization value ({}) is not compatiable with value of {}".
                                     format(value, append_type_to_name(self)))
        self.value[0] = value
        self._update_output_states()

    def _execute(self,
                    variable=None,
                    runtime_params=None,
                    clock=CentralClock,
                    time_scale=None,
                    context=None):
        return self.function(variable=variable, params=runtime_params, time_scale=time_scale, context=context)

    def _report_mechanism_execution(self, input_val=None, params=None, output=None):

        if input_val is None:
            input_val = self.input_values
        if output is None:
            output = self.output_state.value
        params = params or self.user_params

        import re
        if 'mechanism' in self.name or 'Mechanism' in self.name:
            mechanism_string = ' '
        else:
            mechanism_string = ' mechanism'

        # kmantel: previous version would fail on anything but iterables of things that can be cast to floats
        #   if you want more specific output, you can add conditional tests here
        try:
            input_string = [float("{:0.3}".format(float(i))) for i in input_val].__str__().strip("[]")
        except TypeError:
            input_string = input_val

        print ("\n\'{}\'{} executed:\n- input:  {}".
               format(self.name,
                      mechanism_string,
                      input_string))

        if params:
            print("- params:")
            # Sort for consistency of output
            params_keys_sorted = sorted(params.keys())
            for param_name in params_keys_sorted:
                # No need to report:
                #    function_params here, as they will be reported for the function itself below;
                #    input_states or output_states, as these are not really params
                if param_name in {FUNCTION_PARAMS, INPUT_STATES, OUTPUT_STATES}:
                    continue
                param_is_function = False
                param_value = params[param_name]
                if isinstance(param_value, Function):
                    param = param_value.name
                    param_is_function = True
                elif isinstance(param_value, type(Function)):
                    param = param_value.__name__
                    param_is_function = True
                elif isinstance(param_value, (function_type, method_type)):
                    param = param_value.__self__.__class__.__name__
                    param_is_function = True
                else:
                    param = param_value
                print ("\t{}: {}".format(param_name, str(param).__str__().strip("[]")))
                if param_is_function:
                    # Sort for consistency of output
                    func_params_keys_sorted = sorted(self.function_object.user_params.keys())
                    for fct_param_name in func_params_keys_sorted:
                        print ("\t\t{}: {}".
                               format(fct_param_name,
                                      str(self.function_object.user_params[fct_param_name]).__str__().strip("[]")))

        # kmantel: previous version would fail on anything but iterables of things that can be cast to floats
        #   if you want more specific output, you can add conditional tests here
        try:
            output_string = re.sub(r'[\[,\],\n]', '', str([float("{:0.3}".format(float(i))) for i in output]))
        except TypeError:
            output_string = output

        print("- output: {}".format(output_string))


    def plot(self, x_range=None):
        """Generate a plot of the Mechanism's `function <Mechanism_Base.function>` using the specified parameter values
        (see `DDM.plot <DDM.plot>` for details of the animated DDM plot).

        Arguments
        ---------

        x_range : List
            specify the range over which the `function <Mechanism_Base.function>` should be plotted. x_range must be
            provided as a list containing two floats: lowest value of x and highest value of x.  Default values
            depend on the Mechanism's `function <Mechanism_Base.function>`.

            - Logistic Function: default x_range = [-5.0, 5.0]
            - Exponential Function: default x_range = [0.1, 5.0]
            - All Other Functions: default x_range = [-10.0, 10.0]

        Returns
        -------
        Plot of Mechanism's `function <Mechanism_Base.function>` : Matplotlib window
            Matplotlib window of the Mechanism's `function <Mechanism_Base.function>` plotted with specified parameters
            over the specified x_range

        """

        import matplotlib.pyplot as plt

        if not x_range:
            if "Logistic" in str(self.function):
                x_range= [-5.0, 5.0]
            elif "Exponential" in str(self.function):
                x_range = [0.1, 5.0]
            else:
                x_range = [-10.0, 10.0]
        x_space = np.linspace(x_range[0],x_range[1])
        plt.plot(x_space, self.function(x_space), lw=3.0, c='r')
        plt.show()

    @tc.typecheck
    def add_states(self, states, context=COMMAND_LINE):
        """
        add_states(states)

        Add one or more `States <State>` to the Mechanism.  Only `InputStates <InputState> and `OutputStates
        <OutputState>` can be added; `ParameterStates <ParameterState>` cannot be added to a Mechanism after it has
        been constructed.

        If the `owner <State_Base.owner>` of a State specified in the **states** argument is not the same as the
        Mechanism to which it is being added, the user is given the option of reassigning the State to the `owner
        <State_Base.owner>`, making a copy of the State and assigning that to the `owner <State_Base.owner>`, or
        aborting.  If the name of a specified State is the same as an existing one with the same name, an index is
        appended to its name, and incremented for each State subsequently added with the same name
        (see :ref:`naming conventions <LINK>`).

        .. note::
            Adding States to a Mechanism changes the size of its `variable <Mechanism_Base.variable>` attribute,
            which may produce an incompatibility with its `function <Mechanism_Base.function>` (see
            `InputStates <InputStates_Mechanism_Variable_and_Function>` for a more detailed explanation).

        Arguments
        ---------

        states : State or List[State]
            one more `InputStates <InputState>` or `OutputStates <OutputState>` to be added to the Mechanism.
            State specification(s) can be an InputState or OutputState object, class reference, class keyword, or
            `State specification dictionary <State_Specification>` (the latter must have a *STATE_TYPE* entry
            specifying the class or keyword for InputState or OutputState).

        """
        from PsyNeuLink.Components.States.State import _parse_state_type
        from PsyNeuLink.Components.States.InputState import InputState, _instantiate_input_states
        from PsyNeuLink.Components.States.OutputState import OutputState, _instantiate_output_states

        # Put in list to standardize treatment below
        if not isinstance(states, list):
            states = [states]

        input_states = []
        output_states = []

        for state in states:
            state_type = _parse_state_type(self, state)
            if (isinstance(state_type, InputState) or
                    (inspect.isclass(state_type) and issubclass(state_type, InputState))):
                input_states.append(state)
            elif (isinstance(state_type, OutputState) or
                    (inspect.isclass(state_type) and issubclass(state_type, OutputState))):
                output_states.append(state)

        # _instantiate_state_list(self, input_states, InputState)
        if input_states:
            _instantiate_input_states(self, input_states, context=context)
        if output_states:
            _instantiate_output_states(self, output_states, context=context)

    def _get_mechanism_param_values(self):
        """Return dict with current value of each ParameterState in paramsCurrent
        :return: (dict)
        """
        from PsyNeuLink.Components.States.ParameterState import ParameterState
        return dict((param, value.value) for param, value in self.paramsCurrent.items()
                    if isinstance(value, ParameterState) )

    def _get_primary_state(self, state_type):
        from PsyNeuLink.Components.States.InputState import InputState
        from PsyNeuLink.Components.States.ParameterState import ParameterState
        from PsyNeuLink.Components.States.OutputState import OutputState
        if issubclass(state_type, InputState):
            return self.input_state
        if issubclass(state_type, OutputState):
            return self.output_state
        if issubclass(state_type, ParameterState):
            raise Mechanism("PROGRAM ERROR:  illegal call to {} for a primary {}".format(self.name, PARAMETER_STATE))

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, assignment):
        self._value = assignment

        # # MODIFIED 1/28/17 NEW: [COPIED FROM State]
        # # Store value in log if specified
        # # Get logPref
        # if self.prefs:
        #     log_pref = self.prefs.logPref
        #
        # # Get context
        # try:
        #     curr_frame = inspect.currentframe()
        #     prev_frame = inspect.getouterframes(curr_frame, 2)
        #     context = inspect.getargvalues(prev_frame[1][0]).locals['context']
        # except KeyError:
        #     context = ""
        #
        # # If context is consistent with log_pref, record value to log
        # if (log_pref is LogLevel.ALL_ASSIGNMENTS or
        #         (log_pref is LogLevel.EXECUTION and EXECUTING in context) or
        #         (log_pref is LogLevel.VALUE_ASSIGNMENT and (EXECUTING in context and kwAssign in context))):
        #     self.log.entries[self.name] = LogEntry(CurrentTime(), context, assignment)
        # # MODIFIED 1/28/17 END

    @property
    def default_value(self):
        return self._default_value

    @property
    def input_state(self):
        return self.input_states[0]

    @property
    def input_values(self):
        try:
            return self.input_states.values
        except (TypeError, AttributeError):
            return None

    @property
    def output_state(self):
        return self.output_states[0]

    @property
    def output_values(self):
        return self.output_states.values

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, current_value):
        # if current_value != self._old_value:
        try:
            if np.array_equal(current_value, self._old_value):
                self._status = UNCHANGED
            else:
                self._status = CHANGED
                self._old_value = current_value
        # FIX:  CATCHES ELEMENTWISE COMPARISON DEPRECATION WARNING/ERROR -- NEEDS TO BE FIXED AT SOME POINT
        except:
            self._status = CHANGED


def _is_mechanism_spec(spec):
    """Evaluate whether spec is a valid Mechanism specification

    Return true if spec is any of the following:
    + Mechanism class
    + Mechanism object:
    Otherwise, return :keyword:`False`

    Returns: (bool)
    """
    if inspect.isclass(spec) and issubclass(spec, Mechanism):
        return True
    if isinstance(spec, Mechanism):
        return True
    return False

# MechanismTuple indices
# OBJECT_ITEM = 0
# # PARAMS_ITEM = 1
# # PHASE_ITEM = 2
#
# MechanismTuple = namedtuple('MechanismTuple', 'mechanism')

from collections import UserList
class MechanismList(UserList):
    """Provides access to items and their attributes in a list of :class:`MechanismTuples` for an owner.

    :class:`MechanismTuples` are of the form: (Mechanism object, runtime_params dict, phaseSpec int).

    Attributes
    ----------
    mechanisms : list of Mechanism objects

    names : list of strings
        each item is a Mechanism.name

    values : list of values
        each item is a Mechanism_Base.value

    outputStateNames : list of strings
        each item is an OutputState.name

    outputStateValues : list of values
        each item is an OutputState.value
    """

    def __init__(self, owner, components_list:list):
        super().__init__()
        self.mechs = components_list
        self.owner = owner
        # for item in components_list:
        #     if not isinstance(item, MechanismTuple):
        #         raise MechanismError("The following item in the components_list arg of MechanismList()"
        #                              " is not a MechanismTuple: {}".format(item))

        self.process_tuples = components_list

    def __getitem__(self, item):
        """Return specified Mechanism in MechanismList
        """
        # return list(self.mechs[item])[MECHANISM]
        return self.mechs[item]

    def __setitem__(self, key, value):
        raise ("MechanismList is read only ")

    def __len__(self):
        return (len(self.mechs))

    # def _get_tuple_for_mech(self, mech):
    #     """Return first Mechanism tuple containing specified Mechanism from the list of mechs
    #     """
    #     if list(item for item in self.mechs).count(mech):
    #         if self.owner.verbosePref:
    #             print("PROGRAM ERROR:  {} found in more than one object_item in {} in {}".
    #                   format(append_type_to_name(mech), self.__class__.__name__, self.owner.name))
    #     return next((object_item for object_item in self.mechs if object_item is mech), None)

    @property
    def mechs_sorted(self):
        """Return list of mechs sorted by Mechanism name"""
        return sorted(self.mechs, key=lambda object_item: object_item.name)

    @property
    def mechanisms(self):
        """Return list of all mechanisms in MechanismList"""
        return list(self)

    @property
    def names(self):
        """Return names of all mechanisms in MechanismList"""
        return list(item.name for item in self.mechanisms)

    @property
    def values(self):
        """Return values of all mechanisms in MechanismList"""
        return list(item.value for item in self.mechanisms)

    @property
    def outputStateNames(self):
        """Return names of all outputStates for all mechanisms in MechanismList"""
        names = []
        for item in self.mechanisms:
            for output_state in item.output_states:
                names.append(output_state.name)
        return names

    @property
    def outputStateValues(self):
        """Return values of outputStates for all mechanisms in MechanismList"""
        values = []
        for item in self.mechanisms:
            for output_state in item.output_states:
                values.append(output_state.value)
        return values
