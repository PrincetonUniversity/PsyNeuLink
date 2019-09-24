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
     * :ref:`Mechanism_Additional_Attributes`
     * :ref:`Mechanism_Role_In_Processes_And_Systems`
    * :ref:`Mechanism_Execution`
     * :ref:`Mechanism_Runtime_Parameters`
    * :ref:`Mechanism_Class_Reference`


.. _Mechanism_Overview:

Overview
--------

A Mechanism takes an input, transforms it in some way, and makes the result available as its output.  There are two
types of Mechanisms in PsyNeuLink:

    * `ProcessingMechanisms <ProcessingMechanism>` aggregate the input they receive from other Mechanisms, and/or the
      input to the `Process` or `System` to which they belong, transform it in some way, and
      provide the result as input to other Mechanisms in the Process or System, or as the output for a Process or
      System itself.  There are a variety of different types of ProcessingMechanism, that accept various forms of
      input and transform them in different ways (see `ProcessingMechanisms <ProcessingMechanism>` for a list).
    ..
    * `AdaptiveMechanisms <AdaptiveMechanism>` monitor the output of one or more other Mechanisms, and use this
      to modulate the parameters of other Mechanisms or Projections.  There are three basic AdaptiveMechanisms:

      * `LearningMechanism <LearningMechanism>` - these receive training (target) values, and compare them with the
        output of a Mechanism to generate `LearningSignals <LearningSignal>` that are used to modify `MappingProjections
        <MappingProjection>` (see `learning <Process_Execution_Learning>`).

      * `ControlMechanism <ControlMechanism>` - these evaluate the output of a specified set of Mechanisms, and
        generate `ControlSignals <ControlSignal>` used to modify the parameters of those or other Mechanisms.

      * `GatingMechanism <GatingMechanism>` - these use their input(s) to determine whether and how to modify the
        `value <State_Base.value>` of the `InputState(s) <InputState>` and/or `OutputState(s) <OutputState>` of other
        Mechanisms.

      Each type of AdaptiveMechanism is associated with a corresponding type of `ModulatorySignal <ModulatorySignal>`
      (a type of `OutputState` specialized for use with the AdaptiveMechanism) and `ModulatoryProjection
      <ModulatoryProjection>`.

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
the `pathway <Process.pathway>` attribute of a `Process`; the Mechanism can be specified in either of the ways
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
          this can contain any of the `standard parameters <Mechanism_Additional_Attributes>` for instantiating a
          Mechanism or ones specific to a particular type of Mechanism (see documentation for the type).  The key must
          be the name of the argument used to specify the parameter in the Mechanism's constructor, and the value must
          be a legal value for that parameter, using any of the ways allowed for `specifying a parameter
          <ParameterState_Specification>`. The parameter values specified will be used to instantiate the Mechanism.
          These can be overridden during execution by specifying `Mechanism_Runtime_Parameters`, either when calling
          the Mechanism's `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` method, or where it is
          specified in the `pathway <Process.pathway>` attribute of a `Process`.

  * **automatically** -- PsyNeuLink automatically creates one or more Mechanisms under some circumstances. For example,
    a `ComparatorMechanism` and `LearningMechanism <LearningMechanism>` are created automatically when `learning is
    specified <Process_Learning_Sequence>` for a Process; and an `ObjectiveMechanism` and `ControlMechanism
    <ControlMechanism>` are created when the `controller <System.controller>` is specified for a `System`.

.. _Mechanism_State_Specification:

*Specifying States*
~~~~~~~~~~~~~~~~~~~

Every Mechanism has one or more `InputStates <InputState>`, `ParameterStates <ParameterState>`, and `OutputStates
<OutputState>` (described `below <Mechanism_States>`) that allow it to receive and send `Projections <Projection>`,
and to execute its `function <Mechanism_Base.function>`).  When a Mechanism is created, it automatically creates the
ParameterStates it needs to represent its parameters, including those of its `function <Mechanism_Base.function>`.
It also creates any InputStates and OutputStates required for the Projections it has been assigned. InputStates and
OutputStates, and corresponding Projections (including those from `ModulatorySignals <ModulatorySignal>`) can also be
specified explicitly in the **input_states** and **output_states** arguments of the Mechanism's constructor (see
`Mechanism_InputStates` and `Mechanism_OutputStates`, respectively, as well as the `first example <Mechanism_Example_1>`
below, and `State_Examples`).  They can also be specified in a `parameter specification dictionary
<ParameterState_Specification>` assigned to the Mechanism's **params** argument, using entries with the keys
*INPUT_STATES* and *OUTPUT_STATES*, respectively (see `second example <Mechanism_Example_2>` below).  While
specifying the **input_states** and **output_states** arguments directly is simpler and more convenient,
the dictionary format allows parameter sets to be created elsewhere and/or re-used.  The value of each entry can be
any of the allowable forms for `specifying a state <State_Specification>`. InputStates and OutputStates can also be
added to an existing Mechanism using its `add_states <Mechanism_Base.add_states>` method, although this is generally
not needed and can have consequences that must be considered (e.g., see `note <Mechanism_Add_InputStates_Note>`),
and therefore is not recommended.

.. _Mechanism_Default_State_Suppression_Note:

    .. note::
       When States are specified in the **input_states** or **output_states** arguments of a Mechanism's constructor,
       they replace any default States generated by the Mechanism when it is created (if no States were specified).
       This is particularly relevant for OutputStates, as most Mechanisms create one or more `Standard OutputStates
       <OutputState_Standard>` by default, that have useful properties.  To retain those States if any are specified in
       the **output_states** argument, they must be included along with those states in the **output_states** argument
       (see `examples <State_Standard_OutputStates_Example>`).  The same is true for default InputStates and the
       **input_states** argument.

       This behavior differs from adding a State once the Mechanism is created.  States added to Mechanism using the
       Mechanism's `add_states <Mechanism_Base.add_states>` method, or by assigning the Mechanism in the **owner**
       argument of the State's constructor, are added to the Mechanism without replacing any of its existing States,
       including any default States that may have been generated when the Mechanism was created (see `examples
       <State_Create_State_Examples>` in State).


Examples
^^^^^^^^

.. _Mechanism_Example_1:

The following example creates an instance of a TransferMechanism that names the default InputState ``MY_INPUT``,
and assigns three `Standard OutputStates <OutputState_Standard>`::

    >>> import psyneulink as pnl
    >>> my_mech = pnl.TransferMechanism(input_states=['MY_INPUT'],
    ...                                 output_states=[pnl.RESULT, pnl.OUTPUT_MEAN, pnl.OUTPUT_VARIANCE])


.. _Mechanism_Example_2:

This shows how the same Mechanism can be specified using a dictionary assigned to the **params** argument::

     >>> my_mech = pnl.TransferMechanism(params={pnl.INPUT_STATES: ['MY_INPUT'],
     ...                                         pnl.OUTPUT_STATES: [pnl.RESULT, pnl.OUTPUT_MEAN, pnl.OUTPUT_VARIANCE]})

See `State <State_Examples>` for additional examples of specifying the States of a Mechanism.

.. _Mechanism_Parameter_Specification:

*Specifying Parameters*
~~~~~~~~~~~~~~~~~~~~~~~

As described `below <Mechanism_ParameterStates>`, Mechanisms have `ParameterStates <ParameterState>` that provide the
current value of a parameter used by the Mechanism and/or its `function <Mechanism_Base.function>` when it is `executed
<Mechanism_Execution>`. These can also be used by a `ControlMechanism <ControlMechanism>` to control the parameters of
the Mechanism and/or it `function <Mechanism_Base.function>`.  The value of any of these, and their control, can be
specified in the corresponding argument of the constructor for the Mechanism and/or its `function
<Mechanism_Base.function>`,  or in a parameter specification dictionary assigned to the **params** argument of its
constructor, as described under `ParameterState_Specification`.


.. _Mechanism_Structure:

Structure
---------

.. _Mechanism_Function:

*Function*
~~~~~~~~~~

The core of every Mechanism is its function, which transforms its input to generate its output.  The function is
specified by the Mechanism's `function <Mechanism_Base.function>` attribute.  Every type of Mechanism has at least one
(primary) function, and some have additional (auxiliary) ones (for example, `TransferMechanism` and
`EVCControlMechanism`). Mechanism functions are generally from the PsyNeuLink `Function` class.  Most Mechanisms
allow their function to be specified, using the `function` argument of the Mechanism's constructor.  The function can
be specified using the name of `Function <Function>` class, or its constructor (including arguments that specify its
parameters).  For example, the `function <TransferMechanism.function>` of a `TransferMechanism`, which is `Linear` by
default, can be specified to be the `Logistic` function as follows::

    >>> my_mechanism = pnl.TransferMechanism(function=pnl.Logistic(gain=1.0, bias=-4))

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

    >>> my_mechanism = pnl.TransferMechanism(function=pnl.Logistic,
    ...                                      params={pnl.FUNCTION_PARAMS: {pnl.GAIN: 1.0, pnl.BIAS: -4.0}})

Again, while not as simple as specifying these as arguments in the function's constructor, this format is more flexible.
Any values specified in the parameter dictionary will **override** any specified within the constructor for the function
itself (see `DDM <DDM_Creation>` for an example).

.. _Mechanism_Function:

`function <Mechanism_Base.function>` Attribute
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `Function <Function>` Component assigned as the primary function of a Mechanism is assigned to the Mechanism's
`function <Component.function>` attribute, and its `function <Function_Base.function>` is assigned
to the Mechanism's `function <Mechanism_Base.function>` attribute.

.. note::
   It is important to recognize the distinction between a `Function <Function>` and its `function
   <Function_Base.function>` attribute (note the difference in capitalization).  A *Function* is a PsyNeuLink `Component
   <Component>`, that can be created using a constructor; a *function* is an attribute that contains a callable method
   belonging to a Function, and that is executed when the Component to which the Function belongs is executed.
   Functions are used to assign, store, and apply parameter values associated with their function (see `Function
   <Function_Overview> for a more detailed explanation).

The parameters of a Mechanism's `function <Mechanism_Base.function>` are attributes of its `function
<Component.function>`, and can be accessed using standard "dot" notation for that object.  For
example, the `gain <Logistic.gain>` and `bias <Logistic.bias>` parameters of the `Logistic` function in the example
above can be access as ``my_mechanism.function.gain`` and ``my_mechanism.function.bias``.  They are
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
        that will be placed in <Mechanism_Base>._parameter_states;  each parameter is also referenced in
        the <Mechanism>.function_params dict, and assigned its own attribute (<Mechanism>.<param>).
COMMENT


.. _Mechanism_Custom_Function:

Custom Functions
^^^^^^^^^^^^^^^^

A Mechanism's `function <Mechanism_Base.function>` can be customized by assigning a user-defined function (e.g.,
a lambda function), so long as it takes arguments and returns values that are compatible with those of the
Mechanism's defaults for that function.  This is also true for auxiliary functions that appear as arguments in a
Mechanism's constructor (e.g., the `EVCControlMechanism`).  A user-defined function can be assigned using the Mechanism's
`assign_params` method (the safest means) or by assigning it directly to the corresponding attribute of the Mechanism
(for its primary function, its `function <Mechanism_Base.function>` attribute). When a user-defined function is
specified, it is automatically converted to a `UserDefinedFunction`.

.. note::
   It is *strongly advised* that auxiliary functions that are inherent to a Mechanism
   (i.e., ones that do *not* appear as an argument in the Mechanism's constructor,
   such as the `integrator_function <TransferMechanism.integrator_function>` of a
   `TransferMechanism`) *not* be assigned custom functions;  this is because their
   parameters are included as arguments in the constructor for the Mechanism,
   and thus changing the function could produce confusing and/or unpredictable effects.


COMMENT:
    When a custom function is specified,
    the function itself is assigned to the Mechanism's designated attribute.  At the same time, PsyNeuLink automatically
    creates a `UserDefinedFunction` object, and assigns the custom function to its
    `function <UserDefinedFunction.function>` attribute.
COMMENT

.. _Mechanism_Variable_and_Value:

`variable <Mechanism_Base.variable>` and `value <Mechanism_Base.value>` Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The input to a Mechanism's `function <Mechanism_Base.function>` is provided by the Mechanism's `variable
<Mechanism_Base.variable>` attribute.  This is an ndarray that is at least 2d, with one item of its outermost
dimension (axis 0) for each of the Mechanism's `input_states <Mechanism_Base.input_states>` (see
`below <Mechanism_InputStates>`).  The result of the  `function <Mechanism_Base.function>` is placed in the
Mechanism's `value <Mechanism_Base.value>` attribute which is  also at least a 2d array.  The Mechanism's `value
<Mechanism_Base.value>` is referenced by its `OutputStates <Mechanism_OutputStates>` to generate their own `value
<OutputState.value>` attributes, each of which is assigned as the value of an item of the list in the Mechanism's
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

*States*
~~~~~~~~

Every Mechanism has one or more of each of three types of States:  `InputState(s) <InputState>`,
`ParameterState(s) <ParameterState>`, `and OutputState(s) <OutputState>`.  Generally, these are created automatically
when the Mechanism is created.  InputStates and OutputStates (but not ParameterStates) can also be specified explicitly
for a Mechanism, or added to an existing Mechanism using its `add_states <Mechanism_Base.add_states>` method, as
described `above <Mechanism_State_Specification>`).

.. _Mechanism_Figure:

The three types of States are shown schematically in the figure below, and described briefly in the following sections.

.. figure:: _static/Mechanism_States_fig.svg
   :alt: Mechanism States
   :scale: 75 %
   :align: left

   **Schematic of a Mechanism showing its three types of States** (`InputState`, `ParameterState` and `OutputState`).
   Every Mechanism has at least one (`primary <InputState_Primary>`) InputState and one (`primary
   <OutputState_Primary>`) OutputState, but can have additional states of each type.  It also has one
   `ParameterState` for each of its parameters and the parameters of its `function <Mechanism_Base.function>`.
   The `value <InputState.value>` of each InputState is assigned as an item of the Mechanism's `variable
   <Mechanism_Base.variable>`, and the result of its `function <Mechanism_Base.function>` is assigned as the Mechanism's
   `value <Mechanism_Base.value>`, the items of which are referenced by its OutputStates to determine their own
   `value <OutputState.value>`\\s (see `Mechanism_Variable_and_Value` above, and more detailed descriptions below).

.. _Mechanism_InputStates:

InputStates
^^^^^^^^^^^

These receive, potentially combine, and represent the input to a Mechanism, and provide this to the Mechanism's
`function <Mechanism_Base.function>`. Usually, a Mechanism has only one (`primary <InputState_Primary>`) `InputState`,
identified in its `input_state <Mechanism_Base.input_state>` attribute. However some Mechanisms have more than one
InputState. For example, a `ComparatorMechanism` has one InputState for its **SAMPLE** and another for its **TARGET**
input. All of the Mechanism's InputStates (including its primary InputState <InputState_Primary>` are listed in its
`input_states <Mechanism_Base.input_states>` attribute (note the plural).  The `input_states
<Mechanism_Base.input_states>` attribute is a ContentAddressableList -- a PsyNeuLink-defined subclass of the Python
class `UserList <https://docs.python.org/3.6/library/collections.html?highlight=userlist#collections.UserList>`_ --
that allows a specific InputState in the list to be accessed using its name as the index for the list (e.g.,
``my_mechanism['InputState name']``).

.. _Mechanism_Variable_and_InputStates:

The `value <InputState.value>` of each InputState for a Mechanism is assigned to a different item of the Mechanism's
`variable <Mechanism_Base.variable>` attribute (a 2d np.array), as well as to a corresponding item of its `input_values
<Mechanism_Base.input_values>` attribute (a list).  The `variable <Mechanism_Base.variable>` provides the input to the
Mechanism's `function <Mechanism_Base.function>`, while its `input_values <Mechanism_Base.input_values>` provides a
convenient way of accessing the value of its individual items.  Because there is a one-to-one correspondence between
a Mechanism's InputStates and the items of its `variable <Mechanism_Base.variable>`, their size along their outermost
dimension (axis 0) must be equal; that is, the number of items in the Mechanism's `variable <Mechanism_Base.variable>`
attribute must equal the number of InputStates in its `input_states <Mechanism_Base.input_states>` attribute. A
Mechanism's constructor does its best to insure this:  if its **default_variable** and/or its **size** argument is
specified, it constructs a number of InputStates (and each with a `value <InputState.value>`) corresponding to the
items specified for the Mechanism's `variable <Mechanism_Base.variable>`, as in the examples below::

    my_mech_A = pnl.TransferMechanism(default_variable=[[0],[0,0]])
    print(my_mech_A.input_states)
    > [(InputState InputState-0), (InputState InputState-1)]
    print(my_mech_A.input_states[0].value)
    > [ 0.]
    print(my_mech_A.input_states[1].value)
    > [ 0.  0.]

    my_mech_B = pnl.TransferMechanism(default_variable=[[0],[0],[0]])
    print(my_mech_B.input_states)
    > [(InputState InputState-0), (InputState InputState-1), (InputState InputState-2)]

Conversely, if the **input_states** argument is used to specify InputStates for the Mechanism, they are used to format
the Mechanism's variable::

    my_mech_C = pnl.TransferMechanism(input_states=[[0,0], 'Hello'])
    print(my_mech_C.input_states)
    > [(InputState InputState-0), (InputState Hello)]
    print(my_mech_C.variable)
    > [array([0, 0]) array([0])]

If both the **default_variable** (or **size**) and **input_states** arguments are specified, then the number and format
of their respective items must be the same (see `State <State_Examples>` for additional examples of specifying States).

If InputStates are added using the Mechanism's `add_states <Mechanism_Base.add_states>` method, then its
`variable <Mechanism_Base.variable>` is extended to accommodate the number of InputStates added (note that this must
be coordinated with the Mechanism's `function <Mechanism_Base.function>`, which takes the Mechanism's `variable
<Mechanism_Base.variable>` as its input (see `note <Mechanism_Add_InputStates_Note>`).

The order in which `InputStates are specified <Mechanism_InputState_Specification>` in the Mechanism's constructor,
and/or `added <Mechanism_Add_InputStates>` using its `add_states <Mechanism_Base.add_states>` method,  determines the
order of the items to which they are assigned assigned in he Mechanism's `variable  <Mechanism_Base.variable>`,
and are listed in its `input_states <Mechanism_Base.input_states>` and `input_values <Mechanism_Base.input_values>`
attribute.  Note that a Mechanism's `input_values <Mechanism_Base.input_values>` attribute has the same information as
the Mechanism's `variable <Mechanism_Base.variable>`, but in the form of a list rather than an ndarray.

.. _Mechanism_InputState_Specification:

**Specifying InputStates and a Mechanism's** `variable <Mechanism_Base.variable>` **Attribute**

When a Mechanism is created, the number and format of the items in its `variable <Mechanism_Base.variable>`
attribute, as well as the number of InputStates it has and their `variable <InputState.variable>` and `value
<InputState.value>` attributes, are determined by one of the following arguments in the Mechanism's constructor:

* **default_variable** (at least 2d ndarray) -- determines the number and format of the items of the Mechanism's
  `variable <Mechanism_Base>` attribute.  The number of items in its outermost dimension (axis 0) determines the
  number of InputStates created for the Mechanism, and the format of each item determines the format for the
  `variable <InputState.variable>` and `value  <InputState.value>` attributes of the corresponding InputState.
  If any InputStates are specified in the **input_states** argument or an *INPUT_STATES* entry of
  a specification dictionary assigned to the **params** argument of the Mechanism's constructor, then the number
  must match the number of items in **default_variable**, or an error is generated.  The format of the items in
  **default_variable** are used to specify the format of the `variable <InputState.variable>` or `value
  <InputState.value>` of the corresponding InputStates for any that are not explicitly specified in the
  **input_states** argument or *INPUT_STATES* entry (see below).
..
* **size** (int, list or ndarray) -- specifies the number and length of items in the Mechanism's variable,
  if **default_variable** is not specified. For example, the following mechanisms are equivalent::
    T1 = TransferMechanism(size = [3, 2])
    T2 = TransferMechanism(default_variable = [[0, 0, 0], [0, 0]])
  The relationship to any specifications in the **input_states** argument or
  *INPUT_STATES* entry of a **params** dictionary is the same as for the **default_variable** argument,
  with the latter taking precedence (see above).
..
* **input_states** (list) -- this can be used to explicitly `specify the InputStates <InputState_Specification>`
  created for the Mechanism. Each item must be an `InputState specification <InputState_Specification>`, and the number
  of items must match the number of items in the **default_variable** argument or **size** argument
  if either of those is specified.  If the `variable <InputState.variable>` and/or `value <InputState.value>`
  is `explicitly specified for an InputState <InputState_Variable_and_Value>` in the **input_states** argument or
  *INPUT_STATES* entry of a **params** dictionary, it must be compatible with the value of the corresponding
  item of **default_variable**; otherwise, the format of the item in **default_variable** corresponding to the
  InputState is used to specify the format of the InputState's `variable <InputState.variable>` (e.g., the InputState is
  `specified using an OutputState <InputState_Projection_Source_Specification>` to project to it;).  If
  **default_variable** is not specified, a default value is specified by the Mechanism.  InputStates can also be
  specifed that `shadow the inputs <InputState_Shadow_Inputs>` of other InputStates and/or Mechanisms; that is, receive
  Projections from all of the same `senders <Projection_Base.sender>` as those specified.

COMMENT:
*** ADD SOME EXAMPLES HERE (see `examples <XXX>`)
COMMENT

COMMENT:
*** ADD THESE TO ABOVE WHEN IMPLEMENTED:
    If more InputStates are specified than there are items in `variable <Mechanism_Base.variable>,
        the latter is extended to  match the former.
    If the Mechanism's `variable <Mechanism_Base.variable>` has more than one item, it may still be assigned
        a single InputState;  in that case, the `value <InputState.value>` of that InputState must have the same
        number of items as the Mechanisms's `variable <Mechanism_Base.variable>`.
COMMENT
..
* *INPUT_STATES* entry of a params dict (list) -- specifications are treated in the same manner as those in the
  **input_states** argument, and take precedence over those.

.. _Mechanism_Add_InputStates:

**Adding InputStates**

InputStates can be added to a Mechanism using its `add_states <Mechanism_Base.add_states>` method;  this extends its
`variable <Mechanism_Base.variable>` by a number of items equal to the number of InputStates added, and each new item
is assigned a format compatible with the `value <InputState.value>` of the corresponding InputState added;  if the
InputState's `variable <InputState.variable>` is not specified, it is assigned the default format for an item of the
owner's `variable <Mechanism_Base.variable>` attribute. The InputStates are appended to the end of the list in the
Mechanism's `input_states <Mechanism_Base.input_states>` attribute.  Adding in States in this manner does **not**
replace any existing States, including any default States generated when the Mechanism was constructed (this is
contrast to States specified in a Mechanism's constructor which **do** `replace any default State(s) of the same type
<Mechanism_Default_State_Suppression_Note>`).

.. _Mechanism_Add_InputStates_Note:

.. note::
    Adding InputStates to a Mechanism using its `add_states <Mechanism_Base.add_states>` method may introduce an
    incompatibility with the Mechanism's `function <Mechanism_Base.function>`, which takes the Mechanism's `variable
    <Mechanism_Base.variable>` as its input; such an incompatibility will generate an error.  It may also influence
    the number of OutputStates created for the Mechanism. It is the user's responsibility to ensure that the
    assignment of InputStates to a Mechanism using the `add_states <Mechanism_Base.add_states>` is coordinated with
    the specification of its `function <Mechanism_Base.function>`, so that the total number of InputStates (listed
    in the Mechanism's `input_states <Mechanism_Base.input_states>` attribute matches the number of items expected
    for the input to the function specified in the Mechanism's `function <Mechanism_Base.function>` attribute
    (i.e., its length along axis 0).

.. _Mechanism_InputState_Projections:

**Projections to InputStates**

Each InputState of a Mechanism can receive one or more `Projections <Projection>` from other Mechanisms.  When a
Mechanism is created, a `MappingProjection` is created automatically for any OutputStates or Projections from them
that are in its `InputState specification <InputState_Specification>`, using `AUTO_ASSIGN_MATRIX` as the Projection's
`matrix specification <Mapping_Matrix_Specification>`.  However, if a specification in the **input_states** argument
or an *INPUT_STATES* entry of a **params** dictionary cannot be resolved to an instantiated OutputState at the time the
Mechanism is created, no MappingProjection is assigned to the InputState, and this must be done by some other means;
any specifications in the Mechanism's `input_states <Mechanism_Base.input_states>` attribute that are not
associated with an instantiated OutputState at the time the Mechanism is executed are ignored.

The `PathwayProjections <PathwayProjection>` (e.g., `MappingProjections <MappingProjection>`) it receives are listed
in its `path_afferents <InputState.path_afferents>` attribute.  If the Mechanism is an `ORIGIN` Mechanism of a
`Process`, this includes a Projection from the `ProcessInputState <Process_Input_And_Output>` for that Process.  Any
`GatingProjections <GatingProjection>` it receives are listed in its `mod_afferents <InputState.mod_afferents>`
attribute.


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
during the last execution of the Mechanism or its `function <Mechanism_Base.function>`).  The ParameterStates for a
Mechanism are listed in its `parameter_states <Mechanism_Base.parameter_states>` attribute.

.. _Mechanism_Parameter_Value_Specification:

The "base" value of a parameter (i.e., the unmodulated value used as the ParameterState's `variable
<ParameterState.variable>` and the input to its `function <ParameterState.function>`) can specified when a Mechanism
and/or its `function <Mechanism_Base.function>` are first created,  using the corresponding arguments of their
constructors (see `Mechanism_Function` above).  Parameter values can also be specified later, by direct assignment of a
value to the attribute for the parameter, or by using the Mechanism's `assign_param` method (the recommended means;
see `ParameterState_Specification`).  Note that the attributes for the parameters of a Mechanism's `function
<Mechanism_Base.function>` usually belong to the `Function <Function_Overview>` referenced in its `function
<Component.function>` attribute, not the Mechanism itself, and therefore must be assigned to the Function
Component (see `Mechanism_Function` above).

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


.. _Mechanism_Additional_Attributes:

*Additional Attributes*
~~~~~~~~~~~~~~~~~~~~~~~

.. _Mechanism_Constructor_Arguments:

Additional Constructor Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to the `standard attributes <Component_Structure>` of any `Component <Component>`, Mechanisms have a set of
Mechanism-specific attributes (listed below). These can be specified in arguments of the Mechanism's constructor,
in a `parameter specification dictionary <ParameterState_Specification>` assigned to the **params** argument of the
Mechanism's constructor, by direct reference to the corresponding attribute of the Mechanisms after it has been
constructed (e.g., ``my_mechanism.param``), or using the Mechanism's `assign_params` method. The Mechanism-specific
attributes are listed below by their argument names / keywords, along with a description of how they are specified:

    * **input_states** / *INPUT_STATES* - a list specifying the Mechanism's input_states
      (see `InputState_Specification` for details of specification).
    ..
    * **output_states** / *OUTPUT_STATES* - specifies specialized OutputStates required by a Mechanism subclass
      (see `OutputState_Specification` for details of specification).
    ..
    * **monitor_for_control** / *MONITOR_FOR_CONTROL* - specifies which of the Mechanism's OutputStates is monitored by
      the `controller` for the System to which the Mechanism belongs (see `specifying monitored OutputStates
      <ObjectiveMechanism_Monitor>` for details of specification).
    ..
    * **monitor_for_learning** / *MONITOR_FOR_LEARNING* - specifies which of the Mechanism's OutputStates is used for
      learning (see `Learning <LearningMechanism_Activation_Output>` for details of specification).

.. _Mechanism_Convenience_Properties:

Projection Convenience Properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Mechanism also has several convenience properties, listed below, that list its `Projections <Projection>` and the
Mechanisms that send/receive these:

    * `projections <Mechanism_Base.projections>` -- all of the Projections sent or received by the Mechanism;
    * `afferents <Mechanism_Base.afferents>` -- all of the Projections received by the Mechanism;
    * `path_afferents <Mechanism_Base.afferents>` -- all of the PathwayProjections received by the Mechanism;
    * `mod_afferents <Mechanism_Base.afferents>` -- all of the ModulatoryProjections received by the Mechanism;
    * `efferents <Mechanism_Base.efferents>` -- all of the Projections sent by the Mechanism;
    * `senders <Mechanism_Base.senders>` -- all of the Mechanisms that send a Projection to the Mechanism
    * `modulators <Mechanism_Base.modulators>` -- all of the AdaptiveMechanisms that send a ModulatoryProjection to the
      Mechanism
    * `receivers <Mechanism_Base.receivers>` -- all of the Mechanisms that receive a Projection from the Mechanism

Each of these is a `ContentAddressableList`, which means that the names of the Components in each list can be listed by
appending ``.names`` to the property.  For examples, the names of all of the Mechanisms that receive a Projection from
``my_mech`` can be accessed by ``my_mech.receivers.names``.


.. _Mechanism_Labels_Dicts:

Value Label Dictionaries
^^^^^^^^^^^^^^^^^^^^^^^^

*Overview*

Mechanisms have two attributes that can be used to specify labels for the values of its InputState(s) and
OutputState(s):

    * *INPUT_LABELS_DICT* -- used to specify labels for values of the InputState(s) of the Mechanism;  if specified,
      the dictionary is contained in the Mechanism's `input_labels_dict <Mechanism_Base.input_labels_dict>` attribute.

    * *OUTPUT_LABELS_DICT* -- used to specify labels for values of the OutputState(s) of the Mechanism;  if specified,
      the dictionary is contained in the Mechanism's `output_labels_dict <Mechanism_Base.output_labels_dict>` attribute.

The labels specified in these dictionaries can be used to:

    - specify items in the `inputs <Composition_Run_Inputs>` and `targets <Run_Targets>` arguments of the `run <System.run>` method
      of a `System`
    - report the values of the InputState(s) and OutputState(s) of a Mechanism
    - visualize the inputs and outputs of the System's Mechanisms

*Specifying Label Dictionaries*

Label dictionaries can only be specified in a parameters dictionary assigned to the **params** argument of the
Mechanism's constructor, using the keywords described above.  A standard label dictionary contains key:value pairs of
the following form:

    * *<state name or index>:<sub-dictionary>* -- this is used to specify labels that are specific to individual States
      of the type corresponding to the dictionary;
        - *key* - either the name of a State of that type, or its index in the list of States of that type (i.e,
          `input_states <Mechanism_Base.input_states>` or `output_states <Mechanism_Base.output_states>`);
        - *value* - a dictionary containing *label:value* entries to be used for that State, where the label is a string
          and the shape of the value matches the shape of the `InputState value <InputState.value>` or `OutputState
          value <OutputState.value>` for which it is providing a *label:value* mapping.

      For example, if a Mechanism has two InputStates, named *SAMPLE* and *TARGET*, then *INPUT_LABELS_DICT* could be
      assigned two entries, *SAMPLE*:<dict> and *TARGET*:<dict> or, correspondingly, 0:<dict> and 1:<dict>, in which
      each dictionary contains separate *label:value* entries for the *SAMPLE* and *TARGET* InputStates.

>>> input_labels_dictionary = {pnl.SAMPLE: {"red": [0],
...                                         "green": [1]},
...                            pnl.TARGET: {"red": [0],
...                                         "green": [1]}}

In the following two cases, a shorthand notation is allowed:

    - a Mechanism has only one state of a particular type (only one InputState or only one OutputState)
    - only the index zero InputState or index zero OutputState needs labels

In these cases, a label dictionary for that type of state may simply contain the *label:value* entries described above.
The *label:value* mapping will **only** apply to the index zero state of the state type for which this option is used.
Any additional states of that type will not have value labels. For example, if the input_labels_dictionary below were
applied to a Mechanism with multiple InputState, only the index zero InputState would use the labels "red" and "green".

>>> input_labels_dictionary = {"red": [0],
...                            "green": [1]}

*Using Label Dictionaries*

When using labels to specify items in the `inputs <Composition_Run_Inputs>` arguments of the `run <System.run>` method, labels may
directly replace any or all of the `InputState values <InputState.value>` in an input specification dictionary. Keep in
mind that each label must be specified in the `input_labels_dict <Mechanism_Base.input_labels_dict>` of the Origin
Mechanism to which inputs are being specified, and must map to a value that would have been valid in that position of
the input dictionary.

        >>> import psyneulink as pnl
        >>> input_labels_dict = {"red": [[1, 0, 0]],
        ...                      "green": [[0, 1, 0]],
        ...                      "blue": [[0, 0, 1]]}
        >>> M = pnl.ProcessingMechanism(default_variable=[[0, 0, 0]],
        ...                             params={pnl.INPUT_LABELS_DICT: input_labels_dict})
        >>> P = pnl.Process(pathway=[M])
        >>> S = pnl.System(processes=[P])
        >>> input_dictionary = {M: ['red', 'green', 'blue', 'red']}
        >>> # (equivalent to {M: [[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]], [[1, 0, 0]]]}, which is a valid input specification)
        >>> results = S.run(inputs=input_dictionary)

The same general rules apply when using labels to specify `target values <Run_Targets>` for a pathway with learning.
With target values, however, the labels must be included in the `output_labels_dict <Mechanism_Base.output_labels_dict>`
of the Mechanism that projects to the `TARGET` Mechanism (see `TARGET Mechanisms <LearningMechanism_Targets>`), or in
other words, the last Mechanism in the `learning sequence <Process_Learning_Sequence>`. This is the same Mechanism used
to specify target values for a particular learning sequence in the `targets dictionary <Run_Targets>`.

        >>> input_labels_dict_M1 = {"red": [[1]],
        ...                         "green": [[0]]}
        >>> output_labels_dict_M2 = {"red": [1],
        ...                         "green": [0]}
        >>> M1 = pnl.ProcessingMechanism(params={pnl.INPUT_LABELS_DICT: input_labels_dict_M1})
        >>> M2 = pnl.ProcessingMechanism(params={pnl.OUTPUT_LABELS_DICT: output_labels_dict_M2})
        >>> P = pnl.Process(pathway=[M1, M2],
        ...                 learning=pnl.ENABLED,
        ...                 learning_rate=0.25)
        >>> S = pnl.System(processes=[P])
        >>> input_dictionary = {M1: ['red', 'green', 'green', 'red']}
        >>> # (equivalent to {M1: [[[1]], [[0]], [[0]], [[1]]]}, which is a valid input specification)
        >>> target_dictionary = {M2: ['red', 'green', 'green', 'red']}
        >>> # (equivalent to {M2: [[1], [0], [0], [1]]}, which is a valid target specification)
        >>> results = S.run(inputs=input_dictionary,
        ...                 targets=target_dictionary)

Several attributes are available for viewing the labels for the current value(s) of a Mechanism's InputState(s) and
OutputState(s).

    - The `label <InputState.label>` attribute of an InputState or OutputState returns the current label of
      its value, if one exists, and its value otherwise.

    - The `input_labels <Mechanism_Base.input_labels>` and `output_labels <Mechanism_Base.output_labels>` attributes of
      Mechanisms return a list containing the labels corresponding to the value(s) of the InputState(s) or
      OutputState(s) of the Mechanism, respectively. If the current value of a state does not have a corresponding
      label, then its numeric value is used instead.

>>> output_labels_dict = {"red": [1, 0, 0],
...                      "green": [0, 1, 0],
...                      "blue": [0, 0, 1]}
>>> M = pnl.ProcessingMechanism(default_variable=[[0, 0, 0]],
...                             params={pnl.OUTPUT_LABELS_DICT: output_labels_dict})
>>> P = pnl.Process(pathway=[M])
>>> S = pnl.System(processes=[P])
>>> input_dictionary =  {M: [[1, 0, 0]]}
>>> results = S.run(inputs=input_dictionary)
>>> M.get_output_labels(S)
['red']
>>> M.output_states[0].get_label(S)
'red'

Labels may be used to visualize the input and outputs of Mechanisms in a System via the **show_structure** option of the
System's `show_graph <System.show_graph>` method with the keyword **LABELS**.

        >>> S.show_graph(show_mechanism_structure=pnl.LABELS)

.. note::

    A given label dictionary only applies to the Mechanism to which it belongs, and a given label only applies to its
    corresponding InputState. For example, the label 'red', may translate to different values on different InputStates
    of the same Mechanism, and on different Mechanisms of a System.

.. Mechanism_Attribs_Dicts:

Attribute Dictionary
^^^^^^^^^^^^^^^^^^^^

A Mechanism has an `attributes_dict` attribute containing a dictionary of its attributes that can be used to
specify the `variable <OutputState.variable>` of its OutputStates (see `OutputState_Customization`).


.. _Mechanism_Role_In_Processes_And_Systems:

*Role in Processes and Systems*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mechanisms that are part of one or more `Processes <Process>` are assigned designations that indicate the
`role <Process_Mechanisms>` they play in those Processes, and similarly for `role <System_Mechanisms>` they play in
any `Systems <System>` to which they belong. These designations are listed in the Mechanism's `processes
<Mechanism_Base.processes>` and `systems <Mechanism_Base.systems>` attributes, respectively.  Any Mechanism
designated as `ORIGIN` receives a `MappingProjection` to its `primary InputState <InputState_Primary>` from the
Process(es) to which it belongs.  Accordingly, when the Process (or System of which the Process is a part) is
executed, those Mechanisms receive the input provided to the Process (or System).  The `output_values
<Mechanism_Base.output_values>` of any Mechanism designated as the `TERMINAL` Mechanism for a Process is assigned as
the `output <Process.output>` of that Process, and similarly for any System to which it belongs.

.. note::
   A Mechanism that is the `ORIGIN` or `TERMINAL` of a Process does not necessarily have the same role in the
   System(s) to which the Mechanism or Process belongs (see `example <LearningProjection_Output_vs_Terminal_Figure>`).


.. _Mechanism_Execution:

Execution
---------

A Mechanism can be executed using its `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` methods.  This
can be useful in testing a Mechanism and/or debugging.  However, more typically, Mechanisms are executed as part of a
`Process <Process_Execution>` or `System <System_Execution>`.  For either of these, the Mechanism must be included in
the `pathway <Process.pathway>` of a Process.  There, it can be specified on its own, or as the first item of a
tuple that also has an optional set of `runtime parameters <Mechanism_Runtime_Parameters>` (see `Process Mechanisms
<Process_Mechanisms>` for additional details about specifying a Mechanism in a Process `pathway
<Process.pathway>`).

.. _Mechanism_Runtime_Parameters:

*Runtime Parameters*
~~~~~~~~~~~~~~~~~~~~

.. note::
   This is an advanced feature, and is generally not required for most applications.

The parameters of a Mechanism are usually specified when the Mechanism is `created <Mechanism_Creation>`.  However,
these can be overridden when it `executed <Mechanism_Base.execution>`.  This can be done in a `parameter specification
dictionary <ParameterState_Specification>` assigned to the **runtime_param** argument of the Mechanism's `execute
<Mechanism_Base.execute>` method, or in a `tuple with the Mechanism <Process_Mechanism_Specification>` in the `pathway`
of a `Process`.  Any value assigned to a parameter in a **runtime_params** dictionary will override the current value of
the parameter for that (and *only* that) execution of the Mechanism; the value will return to its previous value
following that execution.

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

import abc
import inspect
import itertools
import logging
import warnings

from collections import OrderedDict
from inspect import isclass

import numpy as np
import typecheck as tc

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.component import Component, function_type, method_type
from psyneulink.core.components.functions.function import FunctionOutputType
from psyneulink.core.components.functions.transferfunctions import Linear
from psyneulink.core.components.shellclasses import Function, Mechanism, Projection, State
from psyneulink.core.components.states.inputstate import DEFER_VARIABLE_SPEC_TO_MECH_MSG, InputState
from psyneulink.core.components.states.modulatorysignals.modulatorysignal import _is_modulatory_spec
from psyneulink.core.components.states.outputstate import OutputState
from psyneulink.core.components.states.parameterstate import ParameterState
from psyneulink.core.components.states.state import REMOVE_STATES, _parse_state_spec
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import \
    ADDITIVE_PARAM, EXECUTION_PHASE, FUNCTION, FUNCTION_PARAMS, \
    INITIALIZING, INIT_EXECUTE_METHOD_ONLY, INIT_FUNCTION_METHOD_ONLY, \
    INPUT_LABELS_DICT, INPUT_STATE, INPUT_STATES, INPUT_STATE_VARIABLES, \
    MONITOR_FOR_CONTROL, MONITOR_FOR_LEARNING, MULTIPLICATIVE_PARAM, \
    OUTPUT_LABELS_DICT, OUTPUT_STATE, OUTPUT_STATES, OWNER_EXECUTION_COUNT, OWNER_EXECUTION_TIME, OWNER_VALUE, \
    PARAMETER_STATE, PARAMETER_STATES, PREVIOUS_VALUE, PROJECTIONS, REFERENCE_VALUE, \
    TARGET_LABELS_DICT, VALUE, VARIABLE, kwMechanismComponentCategory
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.scheduling.condition import Condition
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.registry import register_category, remove_instance_from_registry
from psyneulink.core.globals.utilities import ContentAddressableList, ReadOnlyOrderedDict, append_type_to_name, \
    convert_to_np_array, iscompatible, kwCompatibilityNumeric

__all__ = [
    'Mechanism_Base', 'MechanismError', 'MechanismRegistry'
]

logger = logging.getLogger(__name__)
MechanismRegistry = {}


class MechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


from collections import UserDict
class MechParamsDict(UserDict):
    """Subclass for validation of dicts used to pass Mechanism parameters to OutputState for variable specification."""
    pass


def _input_state_variables_getter(owning_component=None, context=None):
    try:
        return [input_state.parameters.variable._get(context) for input_state in owning_component.input_states]
    except TypeError:
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
                coordinates updating of input_states, parameter_states (and params), execution of the function method
                implemented by the subclass, (by calling its _execute method), and updating of the OutputStates
            - one or more parameters, each of which must be (or resolve to) a reference to a ParameterState
                these determine the operation of the function of the Mechanism subclass being instantiated
            - one or more OutputStates:
                the variable of each receives the corresponding item in the output of the Mechanism's function
                the value of each is passed to corresponding MappingProjections for which the Mechanism is a sender
                * Notes:
                    by default, a Mechanism has only one OutputState, assigned to <Mechanism>.outputState;  however:
                    if params[OUTPUT_STATES] is a list (of names) or specification dict (of MechanismOuput State
                    specs), <Mechanism>.output_states (note plural) is created and contains a list of OutputStates,
                    the first of which points to <Mechanism>.outputState (note singular)
                [TBI * each OutputState maintains a list of Projections for which it serves as the sender]

        Constraints
        -----------
            - the number of input_states must correspond to the length of the variable of the Mechanism's execute method
            - the value of each InputState must be compatible with the corresponding item in the
                variable of the Mechanism's execute method
            - the value of each ParameterState must be compatible with the corresponding parameter of  the Mechanism's
                 execute method
            - the number of OutputStates must correspond to the length of the output of the Mechanism's execute method,
                (self.defaults.value)
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
            + class_defaults.variable (list)
            + paramClassDefaults (dict):
                + [TBI: kwMechanismExecutionSequenceTemplate (list of States):
                    specifies order in which types of States are executed;  used by self.execute]
            + default_mechanism (str): Currently DDM_MECHANISM (class reference resolved in __init__.py)

        Class methods
        -------------
            - _validate_variable(variable, context)
            - _validate_params(request_set, target_set, context)
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

    variable : at least ndarray : default self.defaults.variable
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

    input_values : List[List or 1d np.array] : default self.defaults.variable
        each item in the list corresponds to the `value <InputState.value>` of one of the Mechanism's `InputStates
        <Mechanism_InputStates>` listed in its `input_states <Mechanism_Base.input_states>` attribute.  The value of
        each item is the same as the corresponding item in the Mechanism's `variable <Mechanism_Base.variable>`
        attribute.  The latter is a 2d np.array; the `input_values <Mechanism_Base.input_values>` attribute provides
        this information in a simpler list format.

    input_labels_dict : dict
        contains entries that are either label:value pairs, or sub-dictionaries containing label:value pairs,
        in which each label (key) specifies a string associated with a value for the InputState(s) of the
        Mechanism; see `Mechanism_Labels_Dicts` for additional details.

    input_labels : list
        contains the labels corresponding to the value(s) of the InputState(s) of the Mechanism. If the current value
        of an InputState does not have a corresponding label, then its numeric value is used instead.

    external_input_values : list
        same as `input_values <Mechanism_Base.input_values>`, but containing the `value <InputState.value>` only of
        InputStates that are not designated as `internal_only <InputState.internal_only>`.

    COMMENT:
    target_labels_dict : dict
        contains entries that are either label:value pairs, or sub-dictionaries containing label:value pairs,
        in which each label (key) specifies a string associated with a value for the InputState(s) of the
        Mechanism if it is the `TARGET` Mechanism for a System; see `Mechanism_Labels_Dicts` and
        `target mechanism <LearningMechanism_Targets>` for additional details.
    COMMENT

    parameter_states : ContentAddressableList[str, ParameterState]
        a read-only list of the Mechanism's `ParameterStates <Mechanism_ParameterStates>`, one for each of its
        `configurable parameters <ParameterState_Configurable_Parameters>`, including those of its `function
        <Mechanism_Base.function>`.  The value of the parameters of the Mechanism and its `function
        <Mechanism_Base.function>` are also accessible as (and can be modified using) attributes of the Mechanism
        (see `Mechanism_ParameterStates`).

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

    value : ndarray
        output of the Mechanism's `function <Mechanism_Base.function>`.  It is always at least a 2d np.array, with the
        items of axis 0 corresponding to the values referenced by the corresponding `index <OutputState.index>`
        attribute of the Mechanism's `OutputStates <OutputState>`.  The first item is generally referenced by the
        Mechanism's `primary OutputState <OutputState_Primary>` (i.e., the one in the its `output_state
        <Mechanism_Base.output_state>` attribute).  The `value <Mechanism_Base.value>` is `None` until the Mechanism
        has been executed at least once.

        .. note::
           the `value <Mechanism_Base.value>` of a Mechanism is not necessarily the same as its
           `output_values <Mechanism_Base.output_values>` attribute, which lists the `values <OutputState.value>`
           of its `OutputStates <Mechanism_Base.outputStates>`.

    output_state : OutputState
        `primary OutputState <OutputState_Primary>` for the Mechanism;  same as first entry of its `output_states
        <Mechanism_Base.output_states>` attribute.

    output_states : ContentAddressableList[str, OutputState]
        list of the Mechanism's `OutputStates <Mechanism_OutputStates>`.

        There is always
        at least one entry, which identifies the Mechanism's `primary OutputState <OutputState_Primary>`.

        a list of the Mechanism's `OutputStates <Mechanism_OutputStates>`. The first (and possibly only) entry is always
        the Mechanism's `primary OutputState <OutputState_Primary>` (i.e., the one in the its `output_state
        <Mechanism_Base.output_state>` attribute).

    output_values : List[value]
        each item in the list corresponds to the `value <OutputState.value>` of one of the Mechanism's `OutputStates
        <Mechanism_OutputStates>` listed in its `output_states <Mechanism_Base.output_states>` attribute.

        .. note:: The `output_values <Mechanism_Base.output_values>` of a Mechanism is not necessarily the same as its
                  `value <Mechanism_Base.value>` attribute, since an OutputState's
                  `function <OutputState.OutputState.function>` and/or its `assign <Mechanism_Base.assign>`
                  attribute may use the Mechanism's `value <Mechanism_Base.value>` to generate a derived quantity for
                  the `value <OutputState.OutputState.value>` of that OutputState (and its corresponding item in the
                  the Mechanism's `output_values <Mechanism_Base.output_values>` attribute).

        COMMENT:
            EXAMPLE HERE
        COMMENT

        .. _outputStateValueMapping : Dict[str, int]:
               contains the mappings of OutputStates to their indices in the output_values list
               The key of each entry is the name of an OutputState, and the value is its position in the
                    :py:data:`OutputStates <Mechanism_Base.output_states>` ContentAddressableList.
               Used in ``_update_output_states`` to assign the value of each OutputState to the correct item of
                   the Mechanism's ``value`` attribute.
               Any Mechanism with a function that returns a value with more than one item (i.e., len > 1) MUST implement
                   self.execute rather than just use the params[FUNCTION].  This is so that _outputStateValueMapping
                   can be implemented.
               TBI: if the function of a Mechanism is specified only by params[FUNCTION]
                   (i.e., it does not implement self.execute) and it returns a value with len > 1
                   it MUST also specify kwFunctionOutputStateValueMapping.

    output_labels_dict : dict
        contains entries that are either label:value pairs, or sub-dictionaries containing label:value pairs,
        in which each label (key) specifies a string associated with a value for the OutputState(s) of the
        Mechanism; see `Mechanism_Labels_Dicts` for additional details.

    output_labels : list
        contains the labels corresponding to the value(s) of the OutputState(s) of the Mechanism. If the current value
        of an OutputState does not have a corresponding label, then its numeric value is used instead.

    condition : Condition : None
        condition to be associated with the Mechanism in the `Scheduler` responsible for executing it in each
        `System` to which it is assigned;  if it is not specified (i.e., its value is `None`), the default
        Condition for a `Component` is used.  It can be overridden in a given `System` by assigning a Condition for
        the Mechanism directly to a Scheduler that is then assigned to the System.

    COMMENT:
        phaseSpec : int or float :  default 0
            determines the `TIME_STEP` (s) at which the Mechanism is executed as part of a System
            (see :ref:`Process_Mechanisms` for specification, and :ref:`System Phase <System_Execution_Phase>`
            for how phases are used).
    COMMENT

    states : ContentAddressableList
        a list of all of the Mechanism's `States <State>`, composed from its `input_states
        <Mechanism_Base.input_states>`, `parameter_states <Mechanism_Base.parameter_states>`, and
        `output_states <Mechanism_Base.output_states>` attributes.

    projections : ContentAddressableList
        a list of all of the Mechanism's `Projections <Projection>`, composed from the
        `path_afferents <InputStates.path_afferents>` of all of its `input_states <Mechanism_Base.input_states>`,
        the `mod_afferents` of all of its `input_states <Mechanism_Base.input_states>`,
        `parameter_states <Mechanism)Base.parameter_states>`, and `output_states <Mechanism_Base.output_states>`,
        and the `efferents <OutputState.efferents>` of all of its `output_states <Mechanism_Base.output_states>`.

    afferents : ContentAddressableList
        a list of all of the Mechanism's afferent `Projections <Projection>`, composed from the
        `path_afferents <InputStates.path_afferents>` of all of its `input_states <Mechanism_Base.input_states>`,
        and the `mod_afferents` of all of its `input_states <Mechanism_Base.input_states>`,
        `parameter_states <Mechanism)Base.parameter_states>`, and `output_states <Mechanism_Base.output_states>`.,

    path_afferents : ContentAddressableList
        a list of all of the Mechanism's afferent `PathwayProjections <PathwayProjection>`, composed from the
        `path_afferents <InputStates.path_afferents>` attributes of all of its `input_states
        <Mechanism_Base.input_states>`.

    mod_afferents : ContentAddressableList
        a list of all of the Mechanism's afferent `ModulatoryProjections <ModulatoryProjection>`, composed from the
        `mod_afferents` attributes of all of its `input_states <Mechanism_Base.input_states>`, `parameter_states
        <Mechanism)Base.parameter_states>`, and `output_states <Mechanism_Base.output_states>`.

    efferents : ContentAddressableList
        a list of all of the Mechanism's efferent `Projections <Projection>`, composed from the `efferents
        <OutputState.efferents>` attributes of all of its `output_states <Mechanism_Base.output_states>`.

    senders : ContentAddressableList
        a list of all of the Mechanisms that send `Projections <Projection>` to the Mechanism (i.e., the senders of
        its `afferents <Mechanism_Base.afferents>`; this includes both `ProcessingMechanisms <ProcessingMechanism>`
        (that send `MappingProjections <MappingProjection>` and `AdaptiveMechanisms <AdaptiveMechanism>` (that send
        `ModulatoryProjections <ModulatoryProjection>` (also see `modulators <Mechanism_Base.modulators>`).

    modulators : ContentAddressableList
        a list of all of the `AdapativeMechanisms <AdaptiveMechanism>` that send `ModulatoryProjections
        <ModulatoryProjection>` to the Mechanism (i.e., the senders of its `mod_afferents
        <Mechanism_Base.mod_afferents>` (also see `senders <Mechanism_Base.senders>`).

    receivers : ContentAddressableList
        a list of all of the Mechanisms that receive `Projections <Projection>` from the Mechanism (i.e.,
        the receivers of its `efferents <Mechanism_Base.efferents>`.

    processes : Dict[Process, str]
        a dictionary of the `Processes <Process>` to which the Mechanism belongs, that designates its  `role
        <Mechanism_Role_In_Processes_And_Systems>` in each.  The key of each entry is a Process to which the Mechansim
        belongs, and its value is the Mechanism's `role in that Process <Process_Mechanisms>`.

    systems : Dict[System, str]
        a dictionary of the `Systems <System>` to which the Mechanism belongs, that designates its `role
        <Mechanism_Role_In_Processes_And_Systems>` in each. The key of each entry is a System to which the Mechanism
        belongs, and its value is the Mechanism's `role in that System <System_Mechanisms>`.

    attributes_dict : Dict[keyword, value]
        a dictionary containing the attributes (and their current values) that can be used to specify the
        `variable <OutputState.variable>` of the Mechanism's `OutputState` (see `OutputState_Customization`).

    name : str
        the name of the Mechanism; if it is not specified in the **name** argument of the constructor, a default is
        assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the Mechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

        .. _stateRegistry : Registry
               registry containing dicts for each State type (InputState, OutputState and ParameterState) with instance
               dicts for the instances of each type and an instance count for each State type in the Mechanism.
               Note: registering instances of State types with the Mechanism (rather than in the StateRegistry)
                     allows the same name to be used for instances of a State type belonging to different Mechanisms
                     without adding index suffixes for that name across Mechanisms
                     while still indexing multiple uses of the same base name within a Mechanism.
    """

    # CLASS ATTRIBUTES
    componentCategory = kwMechanismComponentCategory
    className = componentCategory
    suffix = " " + className

    class Parameters(Mechanism.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <Mechanism_Base.variable>`

                    :default value: numpy.array([[0]])
                    :type: numpy.ndarray
                    :read only: True

                value
                    see `value <Mechanism_Base.value>`

                    :default value: numpy.array([[0]])
                    :type: numpy.ndarray
                    :read only: True

                function
                    see `function <Mechanism_Base.function>`

                    :default value: `Linear`
                    :type: `Function`

                previous_value
                    see `previous_value <Mechanism_Base.previous_value>`

                    :default value: None
                    :type:
                    :read only: True

        """
        variable = Parameter(np.array([[0]]), read_only=True)
        value = Parameter(np.array([[0]]), read_only=True)
        previous_value = Parameter(None, read_only=True)
        function = Linear

        input_state_variables = Parameter(None, read_only=True, user=False, getter=_input_state_variables_getter)

    registry = MechanismRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY
    # Any preferences specified below will override those specified in CategoryDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to CATEGORY automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'MechanismCustomClassPreferences',
    #     kp<pref>: <setting>...}

    # Class-specific loggable items
    @property
    def _loggable_items(self):
        # States, afferent Projections are loggable for a Mechanism
        #     - this allows the value of InputStates and OutputStates to be logged
        #     - for MappingProjections, this logs the value of the Projection's matrix parameter
        #     - for ModulatoryProjections, this logs the value of the Projection
        # IMPLEMENTATION NOTE: this needs to be a property as Projections may be added after instantiation
        try:
            # return list(self.states) + list(self.afferents)
            return list(self.states)
        except:
            return []

    #FIX:  WHEN CALLED BY HIGHER LEVEL OBJECTS DURING INIT (e.g., PROCESS AND SYSTEM), SHOULD USE FULL Mechanism.execute
    # By default, init only the _execute method of Mechanism subclass objects when their execute method is called;
    #    that is, DO NOT run the full Mechanism execute Process, since some components may not yet be instantiated
    #    (such as OutputStates)
    initMethod = INIT_EXECUTE_METHOD_ONLY

    # Note:  the following enforce encoding as 2D np.ndarrays,
    #        to accomodate multiple States:  one 1D np.ndarray per state
    variableEncodingDim = 2
    valueEncodingDim = 2

    stateListAttr = {InputState:INPUT_STATES,
                       ParameterState:PARAMETER_STATES,
                       OutputState:OUTPUT_STATES}

    # Category specific defaults:
    paramClassDefaults = Component.paramClassDefaults.copy()
    paramClassDefaults.update({
        INPUT_STATES:None,
        OUTPUT_STATES:None,
        MONITOR_FOR_CONTROL: NotImplemented,  # This has to be here to "register" it as a valid param for the class
                                              # but is set to NotImplemented so that it is ignored if it is not
                                              # assigned;  setting it to None actively disallows assignment
                                              # (see EVCControlMechanism_instantiate_input_states for more details)
        MONITOR_FOR_LEARNING: None,
        INPUT_LABELS_DICT: {},
        TARGET_LABELS_DICT: {},
        OUTPUT_LABELS_DICT: {}
        # TBI - kwMechanismExecutionSequenceTemplate: [
        #     Components.States.InputState.InputState,
        #     Components.States.ParameterState.ParameterState,
        #     Components.States.OutputState.OutputState]
        })

    # def __new__(cls, *args, **kwargs):
    # def __new__(cls, name=NotImplemented, params=NotImplemented, context=None):

    @tc.typecheck
    @abc.abstractmethod
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_states=None,
                 function=None,
                 output_states=None,
                 params=None,
                 name=None,
                 prefs=None,
                 context=None,
                 **kwargs
                 ):
        """Assign name, category-level preferences, and variable; register Mechanism; and enforce category methods

        This is an abstract class, and can only be called from a subclass;
           it must be called by the subclass with a context value

        NOTES:
        * Since Mechanism is a subclass of Component, it calls super.__init__
            to validate size and default_variable and param_defaults, and assign params to paramInstanceDefaults;
            it uses INPUT_STATE as the default_variable
        * registers Mechanism with MechanismRegistry

        """

        # IMPLEMENT **kwargs (PER State)

        self._is_finished = False
        self.processes = ReadOnlyOrderedDict() # Note: use _add_process method to add item to processes property
        self.systems = ReadOnlyOrderedDict() # Note: use _add_system method to add item to systems property
        self.aux_components = []
        # Register with MechanismRegistry or create one
        register_category(entry=self,
                          base_class=Mechanism_Base,
                          name=name,
                          registry=MechanismRegistry,
                          context=context)

        # Create Mechanism's _stateRegistry and state type entries
        from psyneulink.core.components.states.state import State_Base
        self._stateRegistry = {}

        # InputState
        from psyneulink.core.components.states.inputstate import InputState
        register_category(entry=InputState,
                          base_class=State_Base,
                          registry=self._stateRegistry,
                          context=context)

        # ParameterState
        from psyneulink.core.components.states.parameterstate import ParameterState
        register_category(entry=ParameterState,
                          base_class=State_Base,
                          registry=self._stateRegistry,
                          context=context)

        # OutputState
        from psyneulink.core.components.states.outputstate import OutputState
        register_category(entry=OutputState,
                          base_class=State_Base,
                          registry=self._stateRegistry,
                          context=context)

        default_variable = self._handle_default_variable(default_variable, size, input_states, function, params)

        super(Mechanism_Base, self).__init__(default_variable=default_variable,
                                             size=size,
                                             function=function,
                                             param_defaults=params,
                                             prefs=prefs,
                                             name=name,
                                             **kwargs)

        # FIX: 10/3/17 - IS THIS CORRECT?  SHOULD IT BE INITIALIZED??
        self._status = INITIALIZING
        self._receivesProcessInput = False
        self.phaseSpec = None

    # ------------------------------------------------------------------------------------------------------------------
    # Parsing methods
    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------
    # Argument parsers
    # ---------------------------------------------------------

    def _parse_arg_variable(self, variable):
        if variable is None:
            return None

        return super()._parse_arg_variable(convert_to_np_array(variable, dimension=2))

    # ------------------------------------------------------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------------------------------------------------------

    def _handle_default_variable(self, default_variable=None, size=None, input_states=None, function=None, params=None):
        """
            Finds whether default_variable can be determined using **default_variable** and **size**
            arguments.

            Returns
            -------
                a default variable if possible
                None otherwise
        """
        default_variable_from_input_states = None

        # handle specifying through params dictionary
        try:
            default_variable_from_input_states, input_states_variable_was_specified = self._handle_arg_input_states(params[INPUT_STATES])

            # updated here in case it was parsed in _handle_arg_input_states
            params[INPUT_STATES] = self.input_states
        except (TypeError, KeyError):
            pass
        except AttributeError as e:
            if DEFER_VARIABLE_SPEC_TO_MECH_MSG in e.args[0]:
                pass

        if default_variable_from_input_states is None:
            # fallback to standard arg specification
            try:
                default_variable_from_input_states, input_states_variable_was_specified = self._handle_arg_input_states(input_states)
            except AttributeError as e:
                if DEFER_VARIABLE_SPEC_TO_MECH_MSG in e.args[0]:
                    pass

        if default_variable_from_input_states is not None:
            if default_variable is None:
                if size is None:
                    default_variable = default_variable_from_input_states
                else:
                    if input_states_variable_was_specified:
                        size_variable = self._handle_size(size, None)
                        if iscompatible(size_variable, default_variable_from_input_states):
                            default_variable = default_variable_from_input_states
                        else:
                            raise MechanismError(
                                'default variable determined from the specified input_states spec ({0}) '
                                'is not compatible with the default variable determined from size parameter ({1})'.
                                    format(default_variable_from_input_states, size_variable,
                                )
                            )
                    else:
                        # do not pass input_states variable as default_variable, fall back to size specification
                        pass
            else:
                if input_states_variable_was_specified:
                    if not iscompatible(self._parse_arg_variable(default_variable), default_variable_from_input_states):
                        raise MechanismError(
                            'Default variable determined from the specified input_states spec ({0}) for {1} '
                            'is not compatible with its specified default variable ({2})'.format(
                                default_variable_from_input_states, self.name, default_variable
                            )
                        )
                else:
                    # do not pass input_states variable as default_variable, fall back to default_variable specification
                    pass

        return super()._handle_default_variable(default_variable=default_variable, size=size)

    def _handle_arg_input_states(self, input_states):
        """
        Takes user-inputted argument **input_states** and returns an defaults.variable-like
        object that it represents

        Returns
        -------
            A, B where
            A is an defaults.variable-like object
            B is True if **input_states** contained an explicit variable specification, False otherwise
        """

        if input_states is None:
            return None, False

        default_variable_from_input_states = []
        input_state_variable_was_specified = None

        if not isinstance(input_states, list):
            input_states = [input_states]
            # KDM 6/28/18: you can't set to self.input_states because this triggers
            # a check for validation pref, but self.prefs does not exist yet so this fails
            self._input_states = input_states

        for i, s in enumerate(input_states):


            try:
                parsed_input_state_spec = _parse_state_spec(owner=self,
                                                            state_type=InputState,
                                                            state_spec=s,
                )
            except AttributeError as e:
                if DEFER_VARIABLE_SPEC_TO_MECH_MSG in e.args[0]:
                    default_variable_from_input_states.append(InputState.defaults.variable)
                    continue
                else:
                    raise MechanismError("PROGRAM ERROR: Problem parsing {} specification ({}) for {}".
                                         format(InputState.__name__, s, self.name))

            mech_variable_item = None

            if isinstance(parsed_input_state_spec, dict):
                try:
                    mech_variable_item = parsed_input_state_spec[VALUE]
                    if parsed_input_state_spec[VARIABLE] is None:
                        input_state_variable_was_specified = False
                except KeyError:
                    pass
            elif isinstance(parsed_input_state_spec, (Projection, Mechanism, State)):
                if parsed_input_state_spec.initialization_status == ContextFlags.DEFERRED_INIT:
                    args = parsed_input_state_spec.init_args
                    if REFERENCE_VALUE in args and args[REFERENCE_VALUE] is not None:
                        mech_variable_item = args[REFERENCE_VALUE]
                    elif VALUE in args and args[VALUE] is not None:
                        mech_variable_item = args[VALUE]
                    elif VARIABLE in args and args[VARIABLE] is not None:
                        mech_variable_item = args[VARIABLE]
                else:
                    try:
                        mech_variable_item = parsed_input_state_spec.value
                    except AttributeError:
                        mech_variable_item = parsed_input_state_spec.defaults.mech_variable_item
            else:
                mech_variable_item = parsed_input_state_spec.defaults.mech_variable_item

            if mech_variable_item is None:
                mech_variable_item = InputState.defaults.variable
            elif input_state_variable_was_specified is None and not InputState._state_spec_allows_override_variable(s):
                input_state_variable_was_specified = True

            default_variable_from_input_states.append(mech_variable_item)

        return default_variable_from_input_states, input_state_variable_was_specified

    # ------------------------------------------------------------------------------------------------------------------
    # Validation methods
    # ------------------------------------------------------------------------------------------------------------------

    def _validate_variable(self, variable, context=None):
        """Convert variable to 2D np.array: one 1D value for each InputState

        # VARIABLE SPECIFICATION:                                        ENCODING:
        # Simple value variable:                                         0 -> [array([0])]
        # Single state array (vector) variable:                         [0, 1] -> [array([0, 1])
        # Multiple state variables, each with a single value variable:  [[0], [0]] -> [array[0], array[0]]

        :param variable:
        :param context:
        :return:
        """

        variable = super(Mechanism_Base, self)._validate_variable(variable, context)

        # Force Mechanism variable specification to be a 2D array (to accomodate multiple InputStates - see above):
        variable = convert_to_np_array(variable, 2)

        return variable

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
        #     # Get OutputStates specified in paramClassDefaults
        #     default_output_states = self.paramClassDefaults[OUTPUT_STATES].copy()
        #     # Convert OutputStates from paramClassDefaults to a list if it is not one
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
            + INPUT_STATES:
                <MechanismsInputState or Projection object or class,
                specification dict for one, 2-item tuple, or numeric value(s)>;
                if it is missing or not one of the above types, it is set to self.defaults.variable
            + FUNCTION_PARAMS:  <dict>, every entry of which must be one of the following:
                ParameterState or Projection object or class, specification dict for one, 2-item tuple, or numeric
                value(s);
                if invalid, default (from paramInstanceDefaults or paramClassDefaults) is assigned
            + OUTPUT_STATES:
                <MechanismsOutputState object or class, specification dict, or numeric value(s);
                if it is missing or not one of the above types, it is set to None here;
                    and then to default value of value (output of execute method) in instantiate_output_state
                    (since execute method must be instantiated before self.defaults.value is known)
                if OUTPUT_STATES is a list or OrderedDict, it is passed along (to instantiate_output_states)
                if it is a OutputState class ref, object or specification dict, it is placed in a list
            + MONITORED_STATES:
                ** DOCUMENT

        Note: PARAMETER_STATES are validated separately -- ** DOCUMENT WHY

        TBI - Generalize to go through all params, reading from each its type (from a registry),
                                   and calling on corresponding subclass to get default values (if param not found)
                                   (as PROJECTION_TYPE and PROJECTION_SENDER are currently handled)
        """

        from psyneulink.core.components.states.state import _parse_state_spec
        from psyneulink.core.components.states.inputstate import InputState

        # Perform first-pass validation in Function.__init__():
        # - returns full set of params based on subclass paramClassDefaults
        super(Mechanism, self)._validate_params(request_set,target_set,context)

        params = target_set

        # VALIDATE INPUT STATE(S)

        # INPUT_STATES is specified, so validate:
        if INPUT_STATES in params and params[INPUT_STATES] is not None:
            try:
                for state_spec in params[INPUT_STATES]:
                    _parse_state_spec(owner=self, state_type=InputState, state_spec=state_spec)
            except AttributeError as e:
                if DEFER_VARIABLE_SPEC_TO_MECH_MSG in e.args[0]:
                    pass

        # VALIDATE FUNCTION_PARAMS
        try:
            function_param_specs = params[FUNCTION_PARAMS]
        except KeyError:
            if context.source & (ContextFlags.COMMAND_LINE | ContextFlags.PROPERTY):
                pass
            elif self.prefs.verbosePref:
                print("No params specified for {0}".format(self.__class__.__name__))
        else:
            if not (isinstance(function_param_specs, dict)):
                raise MechanismError("{0} in {1} must be a dict of param specifications".
                                     format(FUNCTION_PARAMS, self.__class__.__name__))
            # Validate params

            from psyneulink.core.components.states.parameterstate import ParameterState
            for param_name, param_value in function_param_specs.items():
                try:
                    self.defaults.value = self.paramInstanceDefaults[FUNCTION_PARAMS][param_name]
                except KeyError:
                    raise MechanismError("{0} not recognized as a param of execute method for {1}".
                                         format(param_name, self.__class__.__name__))
                if not ((isclass(param_value) and
                             (issubclass(param_value, ParameterState) or
                                  issubclass(param_value, Projection))) or
                        isinstance(param_value, ParameterState) or
                        isinstance(param_value, Projection) or
                        isinstance(param_value, dict) or
                        iscompatible(param_value, self.defaults.value)):
                    params[FUNCTION_PARAMS][param_name] = self.defaults.value
                    if self.prefs.verbosePref:
                        print("{0} param ({1}) for execute method {2} of {3} is not a ParameterState, "
                              "projection, tuple, or value; default value ({4}) will be used".
                              format(param_name,
                                     param_value,
                                     self.execute.__self__.componentName,
                                     self.__class__.__name__,
                                     self.defaults.value))

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
                from psyneulink.core.components.states.outputstate import OutputState
                # If not valid...
                if not ((isclass(item) and issubclass(item, OutputState)) or # OutputState class ref
                            isinstance(item, OutputState) or            # OutputState object
                            isinstance(item, dict) or                   # OutputState specification dict
                            isinstance(item, str) or                    # Name (to be used as key in OutputStates list)
                            isinstance(item, tuple) or                  # Projection specification tuple
                            _is_modulatory_spec(item) or                # Modulatory specification for the OutputState
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

        def validate_labels_dict(lablel_dict, type):
            for label, value in labels_dict.items():
                if not isinstance(label,str):
                    raise MechanismError("Key ({}) in the {} for {} must be a string".
                                         format(label, type, self.name))
                if not isinstance(value,(list, np.ndarray)):
                    raise MechanismError("The value of {} ({}) in the {} for {} must be a list or array".
                                         format(label, value, type, self.name))
        def validate_subdict_key(state_type, key, dict_type):
            # IMPLEMENTATION NOTE:
            #    can't yet validate that string is a legit InputState name or that index is within
            #    bounds of the number of InputStates;  that is done in _get_state_value_labels()
            if not isinstance(key, (int, str)):
                raise MechanismError("Key ({}) for {} of {} must the name of an {} or the index for one".
                                     format(key, dict_type, self.name, state_type.__name__))

        if INPUT_LABELS_DICT in params and params[INPUT_LABELS_DICT]:
            labels_dict = params[INPUT_LABELS_DICT]
            if isinstance(list(labels_dict.values())[0], dict):
                for key, ld in labels_dict.values():
                    validate_subdict_key(InputState, key, INPUT_LABELS_DICT)
                    validate_labels_dict(ld, INPUT_LABELS_DICT)
            else:
                validate_labels_dict(labels_dict, INPUT_LABELS_DICT)

        if OUTPUT_LABELS_DICT in params and params[OUTPUT_LABELS_DICT]:
            labels_dict = params[OUTPUT_LABELS_DICT]
            if isinstance(list(labels_dict.values())[0], dict):
                for key, ld in labels_dict.values():
                    validate_subdict_key(OutputState, key, OUTPUT_LABELS_DICT)
                    validate_labels_dict(ld, OUTPUT_LABELS_DICT)
            else:
                validate_labels_dict(labels_dict, OUTPUT_LABELS_DICT)

        if TARGET_LABELS_DICT in params and params[TARGET_LABELS_DICT]:
            for label, value in params[TARGET_LABELS_DICT].items():
                if not isinstance(label,str):
                    raise MechanismError("Key ({}) in the {} for {} must be a string".
                                         format(label, TARGET_LABELS_DICT, self.name))
                if not isinstance(value,(list, np.ndarray)):
                    raise MechanismError("The value of {} ({}) in the {} for {} must be a list or array".
                                         format(label, value, TARGET_LABELS_DICT, self.name))

    def _validate_inputs(self, inputs=None):
        # Only ProcessingMechanism supports run() method of Function;  ControlMechanism and LearningMechanism do not
        raise MechanismError("{} does not support run() method".format(self.__class__.__name__))

    def _instantiate_attributes_before_function(self, function=None, context=None):
        self.parameters.previous_value._set(None, context)
        self._instantiate_input_states(context=context)
        self._instantiate_parameter_states(function=function, context=context)
        super()._instantiate_attributes_before_function(function=function, context=context)

        # Assign attributes to be included in attributes_dict
        #   keys are keywords exposed to user for assignment
        #   values are names of corresponding attributes
        self.attributes_dict_entries = dict(OWNER_VARIABLE = VARIABLE,
                                            OWNER_VALUE = VALUE,
                                            OWNER_EXECUTION_COUNT = OWNER_EXECUTION_COUNT,
                                            OWNER_EXECUTION_TIME = OWNER_EXECUTION_TIME)
        if hasattr(self, PREVIOUS_VALUE):
            self.attributes_dict_entries.update({'PREVIOUS_VALUE': PREVIOUS_VALUE})

    def _instantiate_function(self, function, function_params=None, context=None):
        """Assign weights and exponents if specified in input_states
        """

        super()._instantiate_function(function=function, function_params=function_params, context=context)

        if self.input_states and any(input_state.weight is not None for input_state in self.input_states):

            # Construct defaults:
            #    from function.weights if specified else 1's
            try:
                default_weights = self.function.weights
            except AttributeError:
                default_weights = None
            if default_weights is None:
                default_weights = default_weights or [1.0] * len(self.input_states)

            # Assign any weights specified in input_state spec
            weights = [[input_state.weight if input_state.weight is not None else default_weight]
                       for input_state, default_weight in zip(self.input_states, default_weights)]
            self.function._weights = weights

        if self.input_states and any(input_state.exponent is not None for input_state in self.input_states):

            # Construct defaults:
            #    from function.weights if specified else 1's
            try:
                default_exponents = self.function.exponents
            except AttributeError:
                default_exponents = None
            if default_exponents is None:
                default_exponents = default_exponents or [1.0] * len(self.input_states)

            # Assign any exponents specified in input_state spec
            exponents = [[input_state.exponent if input_state.exponent is not None else default_exponent]
                       for input_state, default_exponent in zip(self.input_states, default_exponents)]
            self.function._exponents = exponents

        # this may be removed when the restriction making all Mechanism values 2D np arrays is lifted
        # ignore warnings of certain Functions that disable conversion
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=UserWarning)
            self.function.output_type = FunctionOutputType.NP_2D_ARRAY
            self.function.enable_output_type_conversion = True

        self.function.initialization_status = ContextFlags.INITIALIZING
        self.function._instantiate_value(context)
        self.function.initialization_status = ContextFlags.INITIALIZED

    def _instantiate_attributes_after_function(self, context=None):
        from psyneulink.core.components.states.parameterstate import _instantiate_parameter_state

        self._instantiate_output_states(context=context)
        # instantiate parameter states from UDF custom parameters if necessary
        try:
            cfp = self.function.cust_fct_params
            udf_parameters_lacking_states = {param_name: cfp[param_name] for param_name in cfp if param_name not in self.parameter_states.names}

            _instantiate_parameter_state(self, FUNCTION_PARAMS, udf_parameters_lacking_states, context=context, function=self.function)
            self._parse_param_state_sources()
        except AttributeError:
            pass

        super()._instantiate_attributes_after_function(context=context)

    def _instantiate_input_states(self, input_states=None, reference_value=None, context=None):
        """Call State._instantiate_input_states to instantiate orderedDict of InputState(s)

        This is a stub, implemented to allow Mechanism subclasses to override _instantiate_input_states
            or process InputStates before and/or after call to _instantiate_input_states
        """
        from psyneulink.core.components.states.inputstate import _instantiate_input_states
        return _instantiate_input_states(owner=self,
                                         input_states=input_states or self.input_states,
                                         reference_value=reference_value,
                                         context=context)

    def _instantiate_parameter_states(self, function=None, context=None):
        """Call State._instantiate_parameter_states to instantiate a ParameterState for each parameter in user_params

        This is a stub, implemented to allow Mechanism subclasses to override _instantiate_parameter_states
            or process InputStates before and/or after call to _instantiate_parameter_states
            :param function:
        """
        from psyneulink.core.components.states.parameterstate import _instantiate_parameter_states
        _instantiate_parameter_states(owner=self, function=function, context=context)

    def _instantiate_output_states(self, context=None):
        """Call State._instantiate_output_states to instantiate orderedDict of OutputState(s)

        This is a stub, implemented to allow Mechanism subclasses to override _instantiate_output_states
            or process InputStates before and/or after call to _instantiate_output_states
        """
        from psyneulink.core.components.states.outputstate import _instantiate_output_states
        # self._update_parameter_states(context=context)
        self._update_attribs_dicts(context=context)
        _instantiate_output_states(owner=self, output_states=self.output_states, context=context)

    def _add_projection_to_mechanism(self, state, projection, context=None):
        from psyneulink.core.components.projections.projection import _add_projection_to
        _add_projection_to(receiver=self, state=state, projection_spec=projection, context=context)

    def _add_projection_from_mechanism(self, receiver, state, projection, context=None):
        """Add projection to specified state
        """
        from psyneulink.core.components.projections.projection import _add_projection_from
        _add_projection_from(sender=self, state=state, projection_spec=projection, receiver=receiver, context=context)

    def _projection_added(self, projection, context=None):
        """Stub that can be overidden by subclasses that need to know when a projection is added to the Mechanism"""
        pass

    @handle_external_context(execution_id=NotImplemented)
    def reinitialize(self, *args, context=None):
        """
            If the mechanism's `function <Mechanism.function>` is an `IntegratorFunction`, or if the mechanism has and
            `integrator_function <TransferMechanism.integrator_function>` (see `TransferMechanism`), this method
            effectively begins the function's accumulation over again at the specified value, and updates related
            attributes on the mechanism.  It also reassigns `previous_value <Mechanism.previous_value>` to None.

            If the mechanism's `function <Mechanism_Base.function>` is an `IntegratorFunction`, its `reinitialize
            <Mechanism_Base.reinitialize>` method:

                (1) Calls the function's own `reinitialize <IntegratorFunction.reinitialize>` method (see Note below for
                    details)

                (2) Sets the mechanism's `value <Mechanism_Base.value>` to the output of the function's
                    reinitialize method

                (3) Updates its `output states <Mechanism_Base.output_state>` based on its new `value
                    <Mechanism_Base.value>`

            If the mechanism has an `integrator_function <TransferMechanism.integrator_function>`, its `reinitialize
            <Mechanism_Base.reinitialize>` method::

                (1) Calls the `integrator_function's <TransferMechanism.integrator_function>` own `reinitialize
                    <IntegratorFunction.reinitialize>` method (see Note below for details)

                (2) Executes its `function <Mechanism_Base.function>` using the output of the `integrator_function's
                    <TransferMechanism.integrator_function>` `reinitialize <IntegratorFunction.reinitialize>` method as the
                    function's variable

                (3) Sets the mechanism's `value <Mechanism_Base.value>` to the output of its function

                (4) Updates its `output states <Mechanism_Base.output_state>` based on its new `value
                    <Mechanism_Base.value>`

        .. note::
                The reinitialize method of an IntegratorFunction Function typically resets the function's `previous_value
                <IntegratorFunction.previous_value>` (and any other `stateful_attributes <IntegratorFunction.stateful_attributes>`) and
                `value <IntegratorFunction.value>` to the quantity (or quantities) specified. If `reinitialize
                <Mechanism_Base.reinitialize>` is called without arguments, the `initializer <IntegratorFunction.initializer>`
                value (or the values of each of the attributes in `initializers <IntegratorFunction.initializers>`) is used
                instead. The `reinitialize <IntegratorFunction.reinitialize>` method may vary across different Integrators.
                See individual functions for details on their `stateful_attributes <IntegratorFunction.stateful_attributes>`,
                as well as other reinitialization steps that the reinitialize method may carry out.
        """
        from psyneulink.core.components.functions.statefulfunctions.statefulfunction import StatefulFunction
        from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import IntegratorFunction

        if context.execution_id is NotImplemented:
            context.execution_id = self.most_recent_context.execution_id

        # If the primary function of the mechanism is stateful:
        # (1) reinitialize it, (2) update value, (3) update output states
        if isinstance(self.function, StatefulFunction):
            new_value = self.function.reinitialize(*args, context=context)
            self.parameters.value._set(np.atleast_2d(new_value), context=context)
            self._update_output_states(context=context)

        # If the mechanism has an auxiliary integrator function:
        # (1) reinitialize it, (2) run the primary function with the new "previous_value" as input
        # (3) update value, (4) update output states
        elif hasattr(self, "integrator_function"):
            if isinstance(self.integrator_function, IntegratorFunction):
                new_input = self.integrator_function.reinitialize(*args, context=context)[0]
                self.parameters.value._set(
                    self.function.execute(variable=new_input, context=context),
                    context=context,
                    override=True
                )
                self._update_output_states(context=context)

            elif self.integrator_function is None or isinstance(self.integrator_function, type):
                if hasattr(self, "integrator_mode"):
                    raise MechanismError("Reinitializing {} is not allowed because this Mechanism is not stateful. "
                                         "(It does not have an integrator to reinitialize.) If this Mechanism "
                                         "should be stateful, try setting the integrator_mode argument to True. "
                                         .format(self.name))
                else:
                    raise MechanismError("Reinitializing {} is not allowed because this Mechanism is not stateful. "
                                         "(It does not have an integrator to reinitialize).".format(self.name))

            else:
                raise MechanismError("Reinitializing {} is not allowed because its integrator_function is not an "
                                     "IntegratorFunction type function, therefore the Mechanism does not have an integrator to"
                                     " reinitialize.".format(self.name))
        else:
            raise MechanismError("Reinitializing {} is not allowed because this Mechanism is not stateful. "
                                 "(It does not have an accumulator to reinitialize).".format(self.name))

    def get_current_mechanism_param(self, param_name, context=None):
        if param_name == "variable":
            raise MechanismError("The method 'get_current_mechanism_param' is intended for retrieving the current "
                                 "value of a mechanism parameter. 'variable' is not a mechanism parameter. If looking "
                                 "for {}'s default variable, try {}.defaults.variable."
                                 .format(self.name, self.name))
        try:
            return self._parameter_states[param_name].parameters.value._get(context)
        except (AttributeError, TypeError):
            return getattr(self.parameters, param_name)._get(context)

    # when called externally, ContextFlags.PROCESSING is not set. Maintain this behavior here
    # even though it will not update input states for example
    @handle_external_context(execution_phase=ContextFlags.IDLE)
    def execute(self,
                input=None,
                context=None,
                runtime_params=None,
                ):
        """Carry out a single `execution <Mechanism_Execution>` of the Mechanism.

        COMMENT:
            Update InputState(s) and parameter(s), call subclass _execute, update OutputState(s), and assign self.value

            Execution sequence:
            - Call self.input_state.execute() for each entry in self.input_states:
                + execute every self.input_state.path_afferents.[<Projection>.execute()...]
                + combine results and/or gate state using self.input_state.function()
                + assign the result in self.input_state.value
            - Call every self.params[<ParameterState>].execute(); for each:
                + execute self.params[<ParameterState>].mod_afferents.[<Projection>.execute()...]
                    (usually this is just a single ControlProjection)
                + combine results for each ModulationParam or assign value from an OVERRIDE specification
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

        input : List[value] or ndarray : default self.defaults.variable
            input to use for execution of the Mechanism.
            This must be consistent with the format of the Mechanism's `InputState(s) <Mechanism_InputStates>`:
            the number of items in the  outermost level of the list, or axis 0 of the ndarray, must equal the number
            of the Mechanism's `input_states  <Mechanism_Base.input_states>`, and each item must be compatible with the
            format (number and type of elements) of the `variable <InputState.InputState.variable>` of the
            corresponding InputState (see `Run Inputs <Composition_Run_Inputs>` for details of input
            specification formats).

        runtime_params : Optional[Dict[str, Dict[str, Dict[str, value]]]]:
            a dictionary that can include any of the parameters used as arguments to instantiate the Mechanism or
            its function. Any value assigned to a parameter will override the current value of that parameter for *only
            the current* execution of the Mechanism. When runtime_params are passed down from the `Composition` level
            `Run` method, parameters reset to their original values immediately following the execution during which
            runtime_params were used. When `execute <Mechanism.execute>` is called directly, (such as for debugging),
            runtime_params exhibit "lazy updating": parameter values will not reset to their original values until the
            beginning of the next execution.

        Returns
        -------

        Mechanism's output_values : List[value]
            list with the `value <OutputState.value>` of each of the Mechanism's `OutputStates
            <Mechanism_OutputStates>` after either one `TIME_STEP` or a `TRIAL`.

        """

        if self.initialization_status == ContextFlags.INITIALIZED:
            context.string = "{} EXECUTING {}: {}".format(context.source.name,self.name,
                                                               ContextFlags._get_context_string(
                                                                       context.flags, EXECUTION_PHASE))
        else:
            context.string = "{} INITIALIZING {}".format(context.source.name, self.name)

        # IMPLEMENTATION NOTE: Re-write by calling execute methods according to their order in functionDict:
        #         for func in self.functionDict:
        #             self.functionsDict[func]()

        # Limit init to scope specified by context
        if self.initialization_status == ContextFlags.INITIALIZING:
            if context.composition:
                # Run full execute method for init of Process and System
                pass
            # Only call subclass' _execute method and then return (do not complete the rest of this method)
            elif self.initMethod is INIT_EXECUTE_METHOD_ONLY:
                return_value =  self._execute(
                    variable=self.defaults.variable,
                    context=context,
                    runtime_params=runtime_params,
                )

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

            # Call only subclass' function during initialization (not its full _execute method nor rest of this method)
            elif self.initMethod is INIT_FUNCTION_METHOD_ONLY:
                return_value = super()._execute(
                    variable=self.defaults.variable,
                    context=context,
                    runtime_params=runtime_params,
                )
                return np.atleast_2d(return_value)

        # FIX: ??MAKE CONDITIONAL ON self.prefs.paramValidationPref??
        # VALIDATE INPUT STATE(S) AND RUNTIME PARAMS
        self._check_args(
            params=runtime_params,
            target_set=runtime_params,
            context=context,
        )

        self._update_previous_value(context)

        # UPDATE VARIABLE and INPUT STATE(S)

        # Executing or simulating Process, System or Composition, so get input by updating input_states

        if (input is None
            and (context.execution_phase is not ContextFlags.IDLE)
            and (self.input_state.path_afferents != [])):
            variable = self._update_input_states(context=context, runtime_params=runtime_params)

        # Direct call to execute Mechanism with specified input, so assign input to Mechanism's input_states
        else:
            if context.source & ContextFlags.COMMAND_LINE:
                context.execution_phase = ContextFlags.PROCESSING
            if input is None:
                input = self.defaults.variable
            #     FIX:  this input value is sent to input CIMs when compositions are nested
            #           variable should be based on afferent projections
            variable = self._get_variable_from_input(input, context)

        self.parameters.variable._set(variable, context=context)

        # UPDATE PARAMETER STATE(S)
        self._update_parameter_states(context=context, runtime_params=runtime_params)

        # EXECUTE MECHNISM BY CALLING SUBCLASS _execute method AND ASSIGN RESULT TO self.value

        # IMPLEMENTATION NOTE: use value as buffer variable until it has been fully processed
        #                      to avoid multiple calls to (and potential log entries for) self.value property
        value = self._execute(
            variable=variable,
            context=context,
            runtime_params=runtime_params,

        )

        # IMPLEMENTATION NOTE:  THIS IS HERE BECAUSE IF return_value IS A LIST, AND THE LENGTH OF ALL OF ITS
        #                       ELEMENTS ALONG ALL DIMENSIONS ARE EQUAL (E.G., A 2X2 MATRIX PAIRED WITH AN
        #                       ARRAY OF LENGTH 2), np.array (AS WELL AS np.atleast_2d) GENERATES A ValueError
        if (isinstance(value, list) and
            (all(isinstance(item, np.ndarray) for item in value) and
                all(
                        all(item.shape[i]==value[0].shape[0]
                            for i in range(len(item.shape)))
                        for item in value))):
                pass
        else:
            converted_to_2d = np.atleast_2d(value)
            # If return_value is a list of heterogenous elements, return as is
            #     (satisfies requirement that return_value be an array of possibly multidimensional values)
            if converted_to_2d.dtype == object:
                pass
            # Otherwise, return value converted to 2d np.array
            else:
                # return converted_to_2d
                value = converted_to_2d

        self.parameters.value._set(value, context=context)

        # UPDATE OUTPUT STATE(S)
        self._update_output_states(context=context, runtime_params=runtime_params)

        # REPORT EXECUTION
        if self.prefs.reportOutputPref and (context.execution_phase & ContextFlags.PROCESSING | ContextFlags.LEARNING):
            self._report_mechanism_execution(
                self.get_input_values(context),
                self.user_params,
                self.output_state.parameters.value._get(context),
                context=context
            )
        return value

    def run(
        self,
        inputs,
        num_trials=None,
        call_before_execution=None,
        call_after_execution=None,
    ):
        """Run a sequence of `executions <Mechanism_Execution>`.

        COMMENT:
            Call execute method for each in a sequence of executions specified by the `inputs` argument.
        COMMENT

        Arguments
        ---------

        inputs : List[input] or ndarray(input) : default default_variable
            the inputs used for each in a sequence of executions of the Mechanism (see `Composition_Run_Inputs` for a detailed
            description of formatting requirements and options).

        num_trials: int
            number of trials to execute.

        call_before_execution : function : default None
            called before each execution of the Mechanism.

        call_after_execution : function : default None
            called after each execution of the Mechanism.

        Returns
        -------

        Mechanism's output_values : List[value]
            list with the `value <OutputState.value>` of each of the Mechanism's `OutputStates
            <Mechanism_OutputStates>` for each execution of the Mechanism.

        """
        from psyneulink.core.globals.environment import run
        return run(
            self,
            inputs=inputs,
            num_trials=num_trials,
            call_before_trial=call_before_execution,
            call_after_trial=call_after_execution,
        )

    def _get_variable_from_input(self, input, context=None):
        input = np.atleast_2d(input)
        num_inputs = np.size(input, 0)
        num_input_states = len(self.input_states)
        if num_inputs != num_input_states:
            # Check if inputs are of different lengths (indicated by dtype == np.dtype('O'))
            num_inputs = np.size(input)
            if isinstance(input, np.ndarray) and input.dtype is np.dtype('O') and num_inputs == num_input_states:
                # Reduce input back down to sequence of arrays (to remove extra dim added by atleast_2d above)
                input = np.squeeze(input)
            else:
                num_inputs = np.size(input, 0)  # revert num_inputs to its previous value, when printing the error
                raise MechanismError("Number of inputs ({0}) to {1} does not match "
                                  "its number of input_states ({2})".
                                  format(num_inputs, self.name,  num_input_states ))
        for input_item, input_state in zip(input, self.input_states):
            if len(input_state.defaults.value) == len(input_item):
                input_state.parameters.value._set(input_item, context)
            else:
                raise MechanismError(f"Length ({len(input_item)}) of input ({input_item}) does not match "
                                     f"required length ({len(input_state.defaults.variable)}) for input "
                                     f"to {input_state.name} {InputState.__name__} of {self.name}.")

        return np.array(self.get_input_values(context))

    def _update_previous_value(self, context=None):
        self.parameters.previous_value._set(self.parameters.value._get(context), context)

    def _update_input_states(self, context=None, runtime_params=None):
        """Update value for each InputState in self.input_states:

        Call execute method for all (MappingProjection) Projections in InputState.path_afferents
        Aggregate results (using InputState execute method)
        Update InputState.value
        """

        for i in range(len(self.input_states)):
            state = self.input_states[i]
            state._update(context=context, params=runtime_params)
        return np.array(self.get_input_values(context))

    def _update_parameter_states(self, context=None, runtime_params=None):

        for state in self._parameter_states:
            state._update(context=context, params=runtime_params)
        self._update_attribs_dicts(context=context)

    def _get_parameter_state_deferred_init_control_specs(self):
        # FIX: 9/14/19 - THIS ASSUMES THAT ONLY CONTROLPROJECTIONS RELEVANT TO COMPOSITION ARE in DEFERRED INIT;
        #                BUT WHAT IF NODE SPECIFIED CONTROL BY AN EXISTING CONTROLMECHANISM NOT IN A COMPOSITION
        #                THAT WAS THEN ADDED;  COMPOSITION WOULD STILL NEED TO KNOW ABOUT IT TO ACTIVATE THE CTLPROJ
        ctl_specs = []
        for parameter_state in self._parameter_states:
            for proj in parameter_state.mod_afferents:
                if proj.initialization_status == ContextFlags.DEFERRED_INIT:
                    proj_control_signal_specs = proj.control_signal_params or {}
                    proj_control_signal_specs.update({PROJECTIONS: [proj]})
                    ctl_specs.append(proj_control_signal_specs)
        return ctl_specs

    def _update_attribs_dicts(self, context):
        from psyneulink.core.globals.keywords import NOISE
        for state in self._parameter_states:
            if NOISE in state.name and self.initialization_status == ContextFlags.INITIALIZING:
                continue
            if state.name in self.user_params:
                self.user_params.__additem__(state.name, state.value)
            if state.name in self.function_params:
                self.function_params.__additem__(state.name, state.value)

    def _update_output_states(self, context=None, runtime_params=None):
        """Execute function for each OutputState and assign result of each to corresponding item of self.output_values

        owner_value arg can be used to override existing (or absent) value of owner as variable for OutputStates
        and assign a specified (set of) value(s).

        """
        for i in range(len(self.output_states)):
            state = self.output_states[i]
            state._update(context=context, params=runtime_params)

    def initialize(self, value, context=None):
        """Assign an initial value to the Mechanism's `value <Mechanism_Base.value>` attribute and update its
        `OutputStates <Mechanism_OutputStates>`.

        Arguments
        ---------

        value : List[value] or 1d ndarray
            value used to initialize the first item of the Mechanism's `value <Mechanism_Base.value>` attribute.

        """
        if self.paramValidationPref:
            if not iscompatible(value, self.defaults.value):
                raise MechanismError("Initialization value ({}) is not compatiable with value of {}".
                                     format(value, append_type_to_name(self)))
        self.parameters.value.set(np.atleast_1d(value), context, override=True)
        self._update_output_states(context=context)

    def _get_states_param_struct_type(self, ctx):
        gen = (ctx.get_param_struct_type(s) for s in self.states)
        return pnlvm.ir.LiteralStructType(gen)

    def _get_function_param_struct_type(self, ctx):
        return ctx.get_param_struct_type(self.function)

    def _get_param_struct_type(self, ctx):
        states_param_struct = self._get_states_param_struct_type(ctx)
        function_param_struct = self._get_function_param_struct_type(ctx)
        param_list = [states_param_struct, function_param_struct]

        mech_params = self._get_mech_params_type(ctx)
        if mech_params is not None:
            param_list.append(mech_params)

        return pnlvm.ir.LiteralStructType(param_list)

    def _get_mech_params_type(self, ctx):
        pass


    def _get_states_state_struct_type(self, ctx):
        gen = (ctx.get_state_struct_type(s) for s in self.states)
        return pnlvm.ir.LiteralStructType(gen)

    def _get_function_state_struct_type(self, ctx):
        return ctx.get_state_struct_type(self.function)

    def _get_state_struct_type(self, ctx):
        states_state_struct = self._get_states_state_struct_type(ctx)
        function_state_struct = self._get_function_state_struct_type(ctx)
        context_list = [states_state_struct, function_state_struct]

        mech_context = self._get_mech_state_struct_type(ctx)
        if mech_context is not None:
            context_list.append(mech_context)

        return pnlvm.ir.LiteralStructType(context_list)

    def _get_mech_state_struct_type(self, ctx):
        pass

    def _get_output_struct_type(self, ctx):
        output_type_list = []
        for state in self.output_states:
            output_type_list.append(ctx.get_output_struct_type(state))
        return pnlvm.ir.LiteralStructType(output_type_list)

    def _get_input_struct_type(self, ctx):
        input_type_list = []
        for state in self.input_states:
            # Extract the non-modulation portion of input state input struct
            input_type_list.append(ctx.get_input_struct_type(state).elements[0])
        mod_input_type_list = []
        for proj in self.mod_afferents:
            mod_input_type_list.append(ctx.get_output_struct_type(proj))
        if len(mod_input_type_list) > 0:
            input_type_list.append(pnlvm.ir.LiteralStructType(mod_input_type_list))
        return pnlvm.ir.LiteralStructType(input_type_list)

    def _get_state_param_initializer(self, context):
        gen = (s._get_param_initializer(context) for s in self.states)
        return tuple(gen)

    def _get_function_param_initializer(self, context):
        return self.function._get_param_initializer(context)

    def _get_param_initializer(self, context):
        state_param_init = self._get_state_param_initializer(context)
        function_param_init = self._get_function_param_initializer(context)
        param_init_list = [state_param_init, function_param_init]

        mech_params_init = self._get_mech_params_init()
        if mech_params_init is not None:
            param_init_list.append(mech_params_init)

        return tuple(param_init_list)

    def _get_mech_params_init(self):
        pass

    def _get_states_state_initializer(self, context):
        gen = (s._get_state_initializer(context) for s in self.states)
        return tuple(gen)

    def _get_function_state_initializer(self, context):
        return self.function._get_state_initializer(context)

    def _get_state_initializer(self, context):
        states_state_init = self._get_states_state_initializer(context)
        function_state_init = self._get_function_state_initializer(context)
        state_init_list = [states_state_init, function_state_init]

        return tuple(state_init_list)

    def _gen_llvm_states(self, ctx, builder, states,
                         get_output_ptr, fill_input_data,
                         mech_params, mech_state, mech_input):
        # Avoid recreating combined list in every iteration
        # FIXME: This should be converted to more standard memoization approach
        all_states = self.states
        mod_afferents = self.mod_afferents
        for i, state in enumerate(states):
            s_function = ctx.get_llvm_function(state)

            # Find output location
            builder, s_output = get_output_ptr(builder, i)

            # Allocate the input structure (data + modulation)
            s_input = builder.alloca(s_function.args[2].type.pointee)

            # Copy input data to input structure
            builder = fill_input_data(builder, s_input, i)

            # Copy mod_afferent inputs
            for idx, s_mod in enumerate(state.mod_afferents):
                mech_mod_afferent_idx = mod_afferents.index(s_mod)
                mod_in_ptr = builder.gep(mech_input, [ctx.int32_ty(0),
                                                      ctx.int32_ty(len(self.input_states)),
                                                      ctx.int32_ty(mech_mod_afferent_idx)])
                mod_out_ptr = builder.gep(s_input, [ctx.int32_ty(0), ctx.int32_ty(1 + idx)])
                afferent_val = builder.load(mod_in_ptr)
                builder.store(afferent_val, mod_out_ptr)

            state_idx = all_states.index(state)
            s_params = builder.gep(mech_params, [ctx.int32_ty(0),
                                                 ctx.int32_ty(0),
                                                 ctx.int32_ty(state_idx)])
            s_state = builder.gep(mech_state, [ctx.int32_ty(0),
                                               ctx.int32_ty(0),
                                               ctx.int32_ty(state_idx)])

            builder.call(s_function, [s_params, s_state, s_input, s_output])

        return builder

    def _gen_llvm_input_states(self, ctx, builder,
                               mech_params, mech_state, mech_input):
        # Allocate temporary storage. We rely on the fact that series
        # of input state results should match the main function input.
        is_output_list = []
        for state in self.input_states:
            is_function = ctx.get_llvm_function(state)
            is_output_list.append(is_function.args[3].type.pointee)

        # Check if all elements are the same. Function input will be array type if yes.
        if len(set(is_output_list)) == 1:
            is_output_type = pnlvm.ir.ArrayType(is_output_list[0], len(is_output_list))
        else:
            is_output_type = pnlvm.ir.LiteralStructType(is_output_list)

        is_output = builder.alloca(is_output_type)
        def _get_output_ptr(b, i):
            ptr = b.gep(is_output, [ctx.int32_ty(0), ctx.int32_ty(i)])
            return b, ptr

        def _fill_input(b, s_input, i):
            is_in = builder.gep(mech_input, [ctx.int32_ty(0), ctx.int32_ty(i)])
            data_ptr = builder.gep(s_input, [ctx.int32_ty(0), ctx.int32_ty(0)])
            b.store(b.load(is_in), data_ptr)
            return b

        builder = self._gen_llvm_states(ctx, builder, self.input_states,
                                        _get_output_ptr, _fill_input,
                                        mech_params, mech_state, mech_input)

        return is_output, builder

    def _gen_llvm_param_states(self, func, f_params_in, ctx, builder,
                               mech_params, mech_state, mech_input):
        # Allocate a shadow structure to overload user supplied parameters
        f_params_out = builder.alloca(f_params_in.type.pointee)
        # Copy original values. This handles params without param states.
        # Few extra copies will be eliminated by the compiler.
        builder.store(builder.load(f_params_in), f_params_out)

        # Filter out param states without corresponding params for this function
        param_states = [p for p in self._parameter_states if p.name in func._get_param_ids()]

        def _get_output_ptr(b, i):
            ptr = ctx.get_param_ptr(func, b, f_params_out, param_states[i].name)
            return b, ptr

        def _fill_input(b, s_input, i):
            param_in_ptr = ctx.get_param_ptr(func, b, f_params_in, param_states[i].name)
            raw_ps_input = b.gep(s_input, [ctx.int32_ty(0), ctx.int32_ty(0)])
            b.store(b.load(param_in_ptr), raw_ps_input)
            return b

        builder = self._gen_llvm_states(ctx, builder, param_states,
                                        _get_output_ptr, _fill_input,
                                        mech_params, mech_state, mech_input)
        return f_params_out, builder

    def _gen_llvm_output_state_parse_variable(self, ctx, builder,
                                              mech_params, mech_state, value, state):
            os_in_spec = state._variable_spec
            if os_in_spec == OWNER_VALUE:
                return value
            elif isinstance(os_in_spec, tuple) and os_in_spec[0] == OWNER_VALUE:
                return builder.gep(value, [ctx.int32_ty(0), ctx.int32_ty(os_in_spec[1])])
            #FIXME: For some reason this can be wrapped in a list
            elif isinstance(os_in_spec, list) and len(os_in_spec) == 1 and isinstance(os_in_spec[0], tuple) and os_in_spec[0][0] == OWNER_VALUE:
                return builder.gep(value, [ctx.int32_ty(0), ctx.int32_ty(os_in_spec[0][1])])
            else:
                #TODO: support more spec options
                assert False, "Unsupported output state spec: {} ({})".format(os_in_spec, value.type)

    def _gen_llvm_output_states(self, ctx, builder, value,
                                mech_params, mech_state, mech_in, mech_out):
        def _get_output_ptr(b, i):
            ptr = b.gep(mech_out, [ctx.int32_ty(0), ctx.int32_ty(i)])
            return b, ptr

        def _fill_input(b, s_input, i):
            data_ptr = self._gen_llvm_output_state_parse_variable(ctx, b,
               mech_params, mech_state, value, self.output_states[i])
            input_ptr = builder.gep(s_input, [ctx.int32_ty(0), ctx.int32_ty(0)])
            b.store(b.load(data_ptr), input_ptr)
            return b

        builder = self._gen_llvm_states(ctx, builder, self.output_states,
                                        _get_output_ptr, _fill_input,
                                        mech_params, mech_state, mech_in)
        return builder

    def _gen_llvm_invoke_function(self, ctx, builder, function, params, context, variable):
        fun = ctx.get_llvm_function(function)
        fun_in, builder = self._gen_llvm_function_input_parse(builder, ctx, fun, variable)
        fun_out = builder.alloca(fun.args[3].type.pointee)

        builder.call(fun, [params, context, fun_in, fun_out])

        return fun_out, builder

    def _gen_llvm_function_body(self, ctx, builder, params, context, arg_in, arg_out):

        is_output, builder = self._gen_llvm_input_states(ctx, builder, params, context, arg_in)

        mf_params_ptr = builder.gep(params, [ctx.int32_ty(0), ctx.int32_ty(1)])
        mf_params, builder = self._gen_llvm_param_states(self.function, mf_params_ptr, ctx, builder, params, context, arg_in)

        mf_ctx = builder.gep(context, [ctx.int32_ty(0), ctx.int32_ty(1)])
        value, builder = self._gen_llvm_invoke_function(ctx, builder, self.function, mf_params, mf_ctx, is_output)

        ppval, builder = self._gen_llvm_function_postprocess(builder, ctx, value)

        builder = self._gen_llvm_output_states(ctx, builder, ppval, params, context, arg_in, arg_out)
        return builder

    def _gen_llvm_function_input_parse(self, builder, ctx, func, func_in):
        return func_in, builder

    def _gen_llvm_function_postprocess(self, builder, ctx, mf_out):
        return mf_out, builder

    def _report_mechanism_execution(self, input_val=None, params=None, output=None, context=None):

        if input_val is None:
            input_val = self.get_input_values(context)
        if output is None:
            output = self.output_state.parameters.value._get(context)
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
                    func_params_keys_sorted = sorted(self.function.user_params.keys())
                    for fct_param_name in func_params_keys_sorted:
                        print ("\t\t{}: {}".
                               format(fct_param_name,
                                      str(self.function.user_params[fct_param_name]).__str__().strip("[]")))

        # kmantel: previous version would fail on anything but iterables of things that can be cast to floats
        #   if you want more specific output, you can add conditional tests here
        try:
            output_string = re.sub(r'[\[,\],\n]', '', str([float("{:0.3}".format(float(i))) for i in output]))
        except TypeError:
            output_string = output

        print("- output: {}".format(output_string))

    @tc.typecheck
    def _show_structure(self,
                       show_functions:bool=False,
                       show_mech_function_params:bool=False,
                       show_state_function_params:bool=False,
                       show_values:bool=False,
                       use_labels:bool=False,
                       show_headers:bool=False,
                       show_roles:bool=False,
                       show_conditions:bool=False,
                       composition=None,
                       compact_cim:bool=False,
                       condition:tc.optional(Condition)=None,
                       node_border:str="1",
                       output_fmt:tc.enum('pdf','struct')='pdf'
                       ):
        """Generate a detailed display of a the structure of a Mechanism.

        .. note::
           This method relies on `graphviz <http://www.graphviz.org>`_, which must be installed and imported
           (standard with PsyNeuLink pip install)

        Displays the structure of a Mechanism using html table format and shape='plaintext'.
        This method is called by `Composition.show_graph` if its **show_mechanism_structure** argument is specified as
        `True` when it is called.

        Arguments
        ---------

        show_functions : bool : default False
            show the `function <Component.function>` of the Mechanism and each of its States.

        show_mech_function_params : bool : default False
            show the parameters of the Mechanism's `function <Component.function>` if **show_functions** is True.

        show_state_function_params : bool : default False
            show parameters for the `function <Component.function>` of the Mechanism's States if **show_functions** is
            True).

        show_values : bool : default False
            show the `value <Component.value>` of the Mechanism and each of its States (prefixed by "=").

        use_labels : bool : default False
            use labels for values if **show_values** is `True`; labels must be specified in the `input_labels_dict
            <Mechanism.input_labels_dict>` (for InputState values) and `output_labels_dict
            <Mechanism.output_labels_dict>` (for OutputState values), otherwise the value is used.

        show_headers : bool : default False
            show the Mechanism, InputState, ParameterState and OutputState headers.

            **composition** argument (if **composition** is not specified, show_roles is ignored).

        show_conditions : bool : default False
            show the `conditions <Condition>` used by `Composition` to determine whether/when to execute each Mechanism
            (if **composition** is not specified, show_conditions is ignored).

        composition : Composition : default None
            specifies the `Composition` (to which the Mechanism must belong) for which to show its role (see **roles**);
            if this is not specified, the **show_roles** argument is ignored.

        compact_cim : bool : default False
            specifies whether to suppress InputState fields for input_CIM and OutputState fields for output_CIM

        output_fmt : keyword : default 'pdf'
            'pdf': generate and open a pdf with the visualization;\n
            'jupyter': return the object (ideal for working in jupyter/ipython notebooks)\n
            'struct': return a string that specifies the structure of the Mechanism using html table format
            for use in a GraphViz node specification.

        Example HTML for structure:

        <<table border="1" cellborder="0" cellspacing="0" bgcolor="tan">          <- MAIN TABLE

        <tr>                                                                      <- BEGIN OUTPUTSTATES
            <td colspan="2"><table border="0" cellborder="0" BGCOLOR="bisque">    <- OUTPUTSTATES OUTER TABLE
                <tr>
                    <td colspan="1"><b>OutputStates</b></td>                      <- OUTPUTSTATES HEADER
                </tr>
                <tr>
                    <td><table border="0" cellborder="1">                         <- OUTPUTSTATE CELLS TABLE
                        <tr>
                            <td port="OutputStatePort1">OutputState 1<br/><i>function 1</i><br/><i>=value</i></td>
                            <td port="OutputStatePort2">OutputState 2<br/><i>function 2</i><br/><i>=value</i></td>
                        </tr>
                    </table></td>
                </tr>
            </table></td>
        </tr>

        <tr>                                                                      <- BEGIN MECHANISM & PARAMETERSTATES
            <td port="Mech name"><b>Mech name</b><br/><i>Roles</i></td>           <- MECHANISM CELL (OUTERMOST TABLE)
            <td><table border="0" cellborder="0" BGCOLOR="bisque">                <- PARAMETERSTATES OUTER TABLE
                <tr>
                    <td><b>ParameterStates</b></td>                               <- PARAMETERSTATES HEADER
                </tr>
                <tr>
                    <td><table border="0" cellborder="1">                         <- PARAMETERSTATE CELLS TABLE
                        <tr><td port="ParamPort1">Param 1<br/><i>function 1</i><br/><i>= value</i></td></tr>
                        <tr><td port="ParamPort1">Param 2<br/><i>function 2</i><br/><i>= value</i></td></tr>
                    </table></td>
                </tr>
            </table></td>
        </tr>

        <tr>                                                                      <- BEGIN INPUTSTATES
            <td colspan="2"><table border="0" cellborder="0" BGCOLOR="bisque">    <- INPUTSTATES OUTER TABLE
                <tr>
                    <td colspan="1"><b>InputStates</b></td>                       <- INPUTSTATES HEADER
                </tr>
                <tr>
                    <td><table border="0" cellborder="1">                         <- INPUTSTATE CELLS TABLE
                        <tr>
                            <td port="InputStatePort1">InputState 1<br/><i>function 1</i><br/><i>= value</i></td>
                            <td port="InputStatePort2">InputState 2<br/><i>function 2</i><br/><i>= value</i></td>
                        </tr>
                    </table></td>
                </tr>
            </table></td>
        </tr>

        </table>>

        """

        # Table / cell specifications:

        # Overall node table:                                               NEAR LIGHTYELLOW
        node_table_spec = f'<table border={repr(node_border)} cellborder="0" cellspacing="1" bgcolor="#FFFFF0">'

        # Header of Mechanism cell:
        mech_header = f'<b><i>{Mechanism.__name__}</i></b>:<br/>'

        # Outer State table:
        outer_table_spec = '<table border="0" cellborder="0" bgcolor="#FAFAD0">' # NEAR LIGHTGOLDENRODYELLOW

        # Header cell of outer State table:
        input_states_header =     f'<tr><td colspan="1" valign="middle"><b><i>{InputState.__name__}s</i></b></td></tr>'
        parameter_states_header = f'<tr><td rowspan="1" valign="middle"><b><i>{ParameterState.__name__}s</i></b></td>'
        output_states_header =    f'<tr><td colspan="1" valign="middle"><b><i>{OutputState.__name__}s</i></b></td></tr>'

        # Inner State table (i.e., that contains individual states in each cell):
        inner_table_spec = '<table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD">'

        def mech_cell():
            """Return html with name of Mechanism, possibly with function and/or value
            Inclusion of roles, function and/or value is determined by arguments of call to _show_structure()
            """
            header = ''
            if show_headers:
                header = mech_header
            mech_name = f'<b>{header}<font point-size="16" >{self.name}</font></b>'

            mech_roles = ''
            if composition and show_roles:
                from psyneulink.core.components.system import System
                if isinstance(composition, System):
                    try:
                        mech_roles = f'<br/>[{self.systems[composition]}]'
                    except KeyError:
                        # # mech_roles = r'\n[{}]'.format(self.system)
                        # mech_roles = r'\n[CONTROLLER]'
                        from psyneulink.core.components.mechanisms.adaptive.control.controlmechanism import ControlMechanism
                        from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
                        if isinstance(self, ControlMechanism) and hasattr(self, 'system'):
                            mech_roles = r'\n[CONTROLLER]'
                        elif isinstance(self, ObjectiveMechanism) and hasattr(self, '_role'):
                            mech_roles = f'\n[{self._role}]'
                        else:
                            mech_roles = ""
                else:
                    from psyneulink.core.compositions.composition import CompositionInterfaceMechanism, NodeRole
                    if self is composition.controller:
                        # mech_roles = f'<br/><i>{NodeRole.MODEL_BASED_OPTIMIZER.name}</i>'
                        mech_roles = f'<br/><i>CONTROLLER</i>'
                    elif not isinstance(self, CompositionInterfaceMechanism):
                        roles = [role.name for role in list(composition.nodes_to_roles[self])]
                        mech_roles = f'<br/><i>{",".join(roles)}</i>'
                    assert True

            mech_condition = ''
            if composition and show_conditions and condition:
                mech_condition = f'<br/><i>{str(condition)}</i>'

            mech_function = ''
            fct_params = ''
            if show_functions:
                if show_mech_function_params:
                    fct_params = []
                    for param in [param for param in self.function_parameters
                                  if param.modulable and param.name not in {ADDITIVE_PARAM, MULTIPLICATIVE_PARAM}]:
                        fct_params.append(f'{param.name}={param._get()}')
                    fct_params = ", ".join(fct_params)
                mech_function = f'<br/><i>{self.function.__class__.__name__}({fct_params})</i>'
            mech_value = ''
            if show_values:
                mech_value = f'<br/>={self.value}'
            # Mech cell should span full width if there are no ParameterStates
            cols = 1
            if not len(self.parameter_states):
                cols = 2
            return f'<td port="{self.name}" colspan="{cols}">' + \
                   mech_name + mech_roles + mech_condition + mech_function + mech_value + '</td>'

        @tc.typecheck
        def state_table(state_list:ContentAddressableList,
                        state_type:tc.enum(InputState, ParameterState, OutputState)):
            """Return html with table for each state in state_list, including functions and/or values as specified

            Each table has a header cell and and inner table with cells for each state in the list
            InputState and OutputState cells are aligned horizontally;  ParameterState cells are aligned vertically.
            Use show_functions, show_values and include_labels arguments from call to _show_structure()
            See _show_structure docstring for full template.
            """

            def state_cell(state, include_function:bool=False, include_value:bool=False, use_label:bool=False):
                """Return html for cell in state inner table
                Format:  <td port="StatePort">StateName<br/><i>function 1</i><br/><i>=value</i></td>
                """

                function = ''
                fct_params = ''
                if include_function:
                    if show_state_function_params:
                        fct_params = []
                        for param in [param for param in self.function_parameters
                                      if param.modulable and param.name not in {ADDITIVE_PARAM, MULTIPLICATIVE_PARAM}]:
                            fct_params.append(f'{param.name}={param._get()}')
                        fct_params = ", ".join(fct_params)
                    function = f'<br/><i>{state.function.__class__.__name__}({fct_params})</i>'
                value=''
                if include_value:
                    if use_label and not isinstance(state, ParameterState):
                        value = f'<br/>={state.label}'
                    else:
                        value = f'<br/>={state.value}'
                return f'<td port="{self._get_port_name(state)}"><b>{state.name}</b>{function}{value}</td>'


            # InputStates
            if state_type is InputState:
                if show_headers:
                    states_header = input_states_header
                else:
                    states_header = ''
                table = f'<td colspan="2"> {outer_table_spec} {states_header}<tr><td>{inner_table_spec}<tr>'
                for state in state_list:
                    table += state_cell(state, show_functions, show_values, use_labels)
                table += '</tr></table></td></tr></table></td>'

            # ParameterStates
            elif state_type is ParameterState:
                if show_headers:
                    states_header = parameter_states_header
                else:
                    states_header = '<tr>'
                table = f'<td> {outer_table_spec} {states_header} <td> {inner_table_spec}'
                for state in state_list:
                    table += '<tr>' + state_cell(state, show_functions, show_values, use_labels) + '</tr>'
                table += '</table></td></tr></table></td>'

            # OutputStates
            elif state_type is OutputState:
                if show_headers:
                    states_header = output_states_header
                else:
                    states_header = ''
                table = f'<td colspan="2"> {outer_table_spec} <tr><td>{inner_table_spec}<tr>'
                for state in state_list:
                    table += state_cell(state, show_functions, show_values, use_labels)
                table += f'</tr></table></td></tr> {states_header} </table></td>'

            return table


        # Construct InputStates table
        if len(self.input_states) and (not compact_cim or self is not composition.input_CIM):
            input_states_table = f'<tr>{state_table(self.input_states, InputState)}</tr>'

        else:
            input_states_table = ''

        # Construct ParameterStates table
        if len(self.parameter_states):
            parameter_states_table = state_table(self.parameter_states, ParameterState)
        else:
            parameter_states_table = ''

        # Construct OutputStates table
        if len(self.output_states) and (not compact_cim or self is not composition.output_CIM):
            output_states_table = f'<tr>{state_table(self.output_states, OutputState)}</tr>'

        else:
            output_states_table = ''

        # Construct full table
        m_node_struct = '<' + node_table_spec + \
                        output_states_table + \
                        '<tr>' + mech_cell() + parameter_states_table + '</tr>' + \
                        input_states_table + \
                        '</table>>'

        if output_fmt == 'struct':
            # return m.node
            return m_node_struct

        # Make node
        import graphviz as gv
        struct_shape = 'plaintext' # assumes html is used to specify structure in m_node_struct

        m = gv.Digraph(#'mechanisms',
                       #filename='mechanisms_revisited.gv',
                       node_attr={'shape': struct_shape},
                       )
        m.node(self.name, m_node_struct, shape=struct_shape)

        if output_fmt == 'pdf':
            m.view(self.name.replace(" ", "-"), cleanup=True)

        elif output_fmt == 'jupyter':
            return m

    @tc.typecheck
    def _get_port_name(self, state:State):
        if isinstance(state, InputState):
            state_type = InputState.__name__
        elif isinstance(state, ParameterState):
            state_type = ParameterState.__name__
        elif isinstance(state, OutputState):
            state_type = OutputState.__name__
        else:
            assert False, f'Mechanism._get_port_name() must be called with an ' \
                f'{InputState.__name__}, {ParameterState.__name__} or {OutputState.__name__}'
        return state_type + '-' + state.name

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
        plt.plot(x_space, self.function(x_space)[0], lw=3.0, c='r')
        plt.show()

    @tc.typecheck
    def add_states(self, states):
        """
        add_states(states)

        Add one or more `States <State>` to the Mechanism.  Only `InputStates <InputState>` and `OutputStates
        <OutputState>` can be added; `ParameterStates <ParameterState>` cannot be added to a Mechanism after it has
        been constructed.

        If the `owner <State_Base.owner>` of a State specified in the **states** argument is not the same as the
        Mechanism to which it is being added an error is generated.    If the name of a specified State is the same
        as an existing one with the same name, an index is appended to its name, and incremented for each State
        subsequently added with the same name (see :ref:`naming conventions <LINK>`).  If a specified State already
        belongs to the Mechanism, the request is ignored.

        .. note::
            Adding InputStates to a Mechanism changes the size of its `variable <Mechanism_Base.variable>` attribute,
            which may produce an incompatibility with its `function <Mechanism_Base.function>` (see
            `Mechanism InputStates <Mechanism_InputStates>` for a more detailed explanation).

        Arguments
        ---------

        states : State or List[State]
            one more `InputStates <InputState>` or `OutputStates <OutputState>` to be added to the Mechanism.
            State specification(s) can be an InputState or OutputState object, class reference, class keyword, or
            `State specification dictionary <State_Specification>` (the latter must have a *STATE_TYPE* entry
            specifying the class or keyword for InputState or OutputState).

        Returns a dictionary with two entries, containing the list of InputStates and OutputStates added.
        -------

        Dictionary with entries containing InputStates and/or OutputStates added

        """
        from psyneulink.core.components.states.state import _parse_state_type
        from psyneulink.core.components.states.inputstate import InputState, _instantiate_input_states
        from psyneulink.core.components.states.outputstate import OutputState, _instantiate_output_states

        context = Context(source=ContextFlags.METHOD)

        # Put in list to standardize treatment below
        if not isinstance(states, list):
            states = [states]

        input_states = []
        output_states = []
        instantiated_input_states = None
        instantiated_output_states = None

        for state in states:
            # FIX: 11/9/17: REFACTOR USING _parse_state_spec
            state_type = _parse_state_type(self, state)
            if (isinstance(state_type, InputState) or
                    (inspect.isclass(state_type) and issubclass(state_type, InputState))):
                input_states.append(state)

            elif (isinstance(state_type, OutputState) or
                    (inspect.isclass(state_type) and issubclass(state_type, OutputState))):
                output_states.append(state)

        if input_states:
            added_variable, added_input_state = self._handle_arg_input_states(input_states)
            if added_input_state:
                if not isinstance(self.defaults.variable, list):
                    old_variable = self.defaults.variable.tolist()
                else:
                    old_variable = self.defaults.variable
                old_variable.extend(added_variable)
                self.defaults.variable = np.array(old_variable)
            instantiated_input_states = _instantiate_input_states(self,
                                                                  input_states,
                                                                  added_variable,
                                                                  context=context)
            for state in instantiated_input_states:
                if state.name is state.componentName or state.componentName + '-' in state.name:
                        state._assign_default_state_name(context=context)
            # self._instantiate_function(function=self.function)
        if output_states:
            instantiated_output_states = _instantiate_output_states(self, output_states, context=context)

        self.defaults.variable = self.input_values

        return {INPUT_STATES: instantiated_input_states,
                OUTPUT_STATES: instantiated_output_states}

    @tc.typecheck
    def remove_states(self, states, context=REMOVE_STATES):
        """
        remove_states(states)

        Remove one or more `States <State>` from the Mechanism.  Only `InputStates <InputState> and `OutputStates
        <OutputState>` can be removed; `ParameterStates <ParameterState>` cannot be removed from a Mechanism.

        Each Specified state must be owned by the Mechanism, otherwise the request is ignored.

        .. note::
            Removing InputStates from a Mechanism changes the size of its `variable <Mechanism_Base.variable>`
            attribute, which may produce an incompatibility with its `function <Mechanism_Base.function>` (see
            `Mechanism InputStates <Mechanism_InputStates>` for more detailed information).

        Arguments
        ---------

        states : State or List[State]
            one more states to be removed from the Mechanism.
            State specification(s) can be an State object or the name of one.

        """
        # from psyneulink.core.components.states.inputstate import INPUT_STATE
        from psyneulink.core.components.states.outputstate import OutputState

        # Put in list to standardize treatment below
        if not isinstance(states, (list, ContentAddressableList)):
            states = [states]

        def delete_state_projections(proj_list):
            for proj in proj_list:
                type(proj)._delete_projection(proj)

        for state in states:

            delete_state_projections(state.mod_afferents)

            if state in self.input_states:
                if isinstance(state, str):
                    state = self.input_states[state]
                index = self.input_states.index(state)
                delete_state_projections(state.path_afferents)
                del self.input_states[index]
                # If state is subclass of OutputState:
                #    check if regsistry has category for that class, and if so, use that
                category = INPUT_STATE
                class_name = state.__class__.__name__
                if class_name != INPUT_STATE and class_name in self._stateRegistry:
                    category = class_name
                remove_instance_from_registry(registry=self._stateRegistry,
                                              category=category,
                                              component=state)
                old_variable = self.defaults.variable
                old_variable = np.delete(old_variable,index,0)
                self.defaults.variable = old_variable

            elif state in self.parameter_states:
                if isinstance(state, ParameterState):
                    index = self.parameter_states.index(state)
                else:
                    index = self.parameter_states.index(self.parameter_states[state])
                del self.parameter_states[index]
                remove_instance_from_registry(registry=self._stateRegistry,
                                              category=PARAMETER_STATE,
                                              component=state)

            elif state in self.output_states:
                if isinstance(state, OutputState):
                    index = self.output_states.index(state)
                else:
                    index = self.output_states.index(self.output_states[state])
                delete_state_projections(state.efferents)
                del self.output_values[index]
                del self.output_states[state]
                # If state is subclass of OutputState:
                #    check if regsistry has category for that class, and if so, use that
                category = OUTPUT_STATE
                class_name = state.__class__.__name__
                if class_name != OUTPUT_STATE and class_name in self._stateRegistry:
                    category = class_name
                remove_instance_from_registry(registry=self._stateRegistry,
                                              category=category,
                                              component=state)

        self.defaults.variable = self.input_values

    def _delete_mechanism(mechanism):
        mechanism.remove_states(mechanism.input_states)
        mechanism.remove_states(mechanism.parameter_states)
        mechanism.remove_states(mechanism.output_states)
        # del mechanism.function
        remove_instance_from_registry(MechanismRegistry, mechanism.__class__.__name__,
                                      component=mechanism)
        del mechanism


    def _get_mechanism_param_values(self):
        """Return dict with current value of each ParameterState in paramsCurrent
        :return: (dict)
        """
        from psyneulink.core.components.states.parameterstate import ParameterState
        return dict((param, value.value) for param, value in self.paramsCurrent.items()
                    if isinstance(value, ParameterState) )

    def get_input_state_position(self, state):
        if state in self.input_states:
            return self.input_states.index(state)
        raise MechanismError("{} is not an InputState of {}.".format(state.name, self.name))

    # @tc.typecheck
    # def _get_state_value_labels(self, state_type:tc.any(InputState, OutputState)):
    def _get_state_value_labels(self, state_type, context=None):
        """Return list of labels for the value of each State of specified state_type.
        If the labels_dict has subdicts (one for each State), get label for the value of each State from its subdict.
        If the labels dict does not have subdicts, then use the same dict for the only (or all) State(s)
        """

        if state_type is InputState:
            states = self.input_states

        elif state_type is OutputState:
            states = self.output_states

        labels = []
        for state in states:
            labels.append(state.get_label(context))
        return labels

    @tc.typecheck
    def _add_process(self, process, role:str):
        from psyneulink.core.components.process import Process
        if not isinstance(process, Process):
            raise MechanismError("PROGRAM ERROR: First argument of call to {}._add_process ({}) must be a {}".
                                 format(Mechanism.__name__, process, Process.__name__))
        self.processes.__additem__(process, role)

    @tc.typecheck
    def _add_system(self, system, role:str):
        from psyneulink.core.components.system import System
        if not isinstance(system, System):
            raise MechanismError("PROGRAM ERROR: First argument of call to {}._add_system ({}) must be a {}".
                                 format(Mechanism.__name__, system, System.__name__))
        self.systems.__additem__(system, role)

    def is_finished(self, context=None):
        """
            set by a Mechanism to signal completion of its `execution <Mechanism_Execution>` in a `trial`; used by
            `Component-based Conditions <Conditions_Component_Based>` to predicate the execution of one or more other
            Components on the Mechanism.
        """
        return self._is_finished

    @property
    def input_state(self):
        return self.input_states[0]

    @property
    def input_values(self):
        try:
            return self.input_states.values
        except (TypeError, AttributeError):
            return None

    def get_input_values(self, context=None):
        input_values = []
        for input_state in self.input_states:
            if "LearningSignal" in input_state.name:
                input_values.append(input_state.parameters.value.get(context).flatten())
            else:
                input_values.append(input_state.parameters.value.get(context))
        return input_values

    @property
    def external_input_states(self):
        try:
            return [input_state for input_state in self.input_states if not input_state.internal_only]
        except (TypeError, AttributeError):
            return None

    @property
    def external_input_values(self):
        try:
            return [input_state.value for input_state in self.input_states if not input_state.internal_only]
        except (TypeError, AttributeError):
            return None

    @property
    def default_external_input_values(self):
        try:
            return [input_state.defaults.value for input_state in self.input_states if not input_state.internal_only]
        except (TypeError, AttributeError):
            return None

    @property
    def input_labels(self):
        """
        Returns a list with as many items as there are InputStates of the Mechanism. Each list item represents the value
        of the corresponding InputState, and is populated by a string label (from the input_labels_dict) when one
        exists, and the numeric value otherwise.
        """
        return self.get_input_labels()

    def get_input_labels(self, context=None):
        if self.input_labels_dict:
            return self._get_state_value_labels(InputState, context)
        else:
            return self.get_input_values(context)

    @property
    def parameter_states(self):
        return self._parameter_states

    @parameter_states.setter
    def parameter_states(self, value):
        # This keeps parameter_states property readonly,
        #    but averts exception when setting paramsCurrent in Component (around line 850)
        pass

    @property
    def output_state(self):
        return self.output_states[0]

    @property
    def output_values(self):
        return self.output_states.values

    def get_output_values(self, context=None):
        return [output_state.parameters.value.get(context) for output_state in self.output_states]

    @property
    def output_labels(self):
        """
        Returns a list with as many items as there are OutputStates of the Mechanism. Each list item represents the
        value of the corresponding OutputState, and is populated by a string label (from the output_labels_dict) when
        one exists, and the numeric value otherwise.
        """
        return self.get_output_labels()

    def get_output_labels(self, context=None):
        if self.output_labels_dict:
            return self._get_state_value_labels(OutputState, context)
        else:
            return self.get_output_values(context)

    @property
    def states(self):
        """Return list of all of the Mechanism's States"""
        return ContentAddressableList(
                component_type=State,
                list=list(self.input_states) +
                     list(self.parameter_states) +
                     list(self.output_states))

    @property
    def path_afferents(self):
        """Return list of path_afferent Projections to all of the Mechanism's input_states"""
        projs = []
        for input_state in self.input_states:
            projs.extend(input_state.path_afferents)
        return ContentAddressableList(component_type=Projection, list=projs)

    @property
    def mod_afferents(self):
        """Return all of the Mechanism's afferent modulatory Projections"""
        projs = []
        for input_state in self.input_states:
            projs.extend(input_state.mod_afferents)
        for parameter_state in self.parameter_states:
            projs.extend(parameter_state.mod_afferents)
        for output_state in self.output_states:
            projs.extend(output_state.mod_afferents)
        return ContentAddressableList(component_type=Projection, list=projs)

    @property
    def afferents(self):
        """Return list of all of the Mechanism's afferent Projections"""
        return ContentAddressableList(component_type=Projection,
                                      list= list(self.path_afferents) + list(self.mod_afferents))

    @property
    def efferents(self):
        """Return list of all of the Mechanism's efferent Projections"""
        projs = []
        try:
            for output_state in self.output_states:
                projs.extend(output_state.efferents)
        except TypeError:
            # self.output_states might be None
            pass
        return ContentAddressableList(component_type=Projection, list=projs)

    @property
    def projections(self):
        """Return all Projections"""
        return ContentAddressableList(component_type=Projection,
                                      list=list(self.path_afferents) +
                                           list(self.mod_afferents) +
                                           list(self.efferents))

    @property
    def senders(self):
        """Return all Mechanisms that send Projections to self"""
        return ContentAddressableList(component_type=Mechanism,
                                      list=[p.sender.owner for p in self.afferents
                                            if isinstance(p.sender.owner, Mechanism_Base)])

    @property
    def receivers(self):
        """Return all Mechanisms that send Projections to self"""
        return ContentAddressableList(component_type=Mechanism,
                                      list=[p.receiver.owner for p in self.efferents
                                            if isinstance(p.sender.owner, Mechanism_Base)])

    @property
    def modulators(self):
        """Return all Mechanisms that send Projections to self"""
        return ContentAddressableList(component_type=Mechanism,
                                      list=[p.sender.owner for p in self.mod_afferents
                                            if isinstance(p.sender.owner, Mechanism_Base)])

    @property
    def attributes_dict(self):
        """Note: this needs to be updated each time it is called, as it must be able to report current values"""

        # Construct attributes_dict from entries specified in attributes_dict_entries
        #   (which is assigned in _instantiate_attributes_before_function)
        attribs_dict = MechParamsDict({key:getattr(self, value) for key,value in self.attributes_dict_entries.items()})
        attribs_dict.update({INPUT_STATE_VARIABLES: [input_state.variable for input_state in self.input_states]})

        attribs_dict.update(self.user_params)
        del attribs_dict[FUNCTION]
        try:
            del attribs_dict[FUNCTION_PARAMS]
        except KeyError:
            pass
        del attribs_dict[INPUT_STATES]
        del attribs_dict[OUTPUT_STATES]
        try:
            attribs_dict.update(self.function_params)
        except KeyError:
            pass
        return attribs_dict

    @property
    def _dependent_components(self):
        return list(itertools.chain(
            super()._dependent_components,
            self.input_states,
            self.output_states,
            self.parameter_states,
        ))


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
        self.data = self.mechs
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
        """Return names of all OutputStates for all mechanisms in MechanismList"""
        names = []
        for item in self.mechanisms:
            for output_state in item.output_states:
                names.append(output_state.name)
        return names

    @property
    def outputStateValues(self):
        """Return values of OutputStates for all mechanisms in MechanismList"""
        values = []
        for item in self.mechanisms:
            for output_state in item.output_states:
                values.append(output_state.value)
        return values

    def get_output_state_values(self, context):
        """Return values of OutputStates for all mechanisms in MechanismList for **context**"""
        values = []
        for item in self.mechanisms:
            for output_state in item.output_states:
                values.append(output_state.parameters.value.get(context))
        return values
