# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ****************************************  MECHANISM MODULE ***********************************************************

"""

Contents
--------

  * `Mechanism_Overview`
  * `Mechanism_Creation`
  * `Mechanism_Structure`
      - `Mechanism_Function`
      - `Mechanism_Ports`
          • `Mechanism_InputPorts`
          • `Mechanism_ParameterPorts`
          • `Mechanism_OutputPorts`
      - `Mechanism_Additional_Attributes`
      - `Mechanism_in_Composition`
  * `Mechanism_Execution`
      - `Mechanism_Execution_Composition`
      - `Mechanism_Runtime_Params`
  * `Mechanism_Class_Reference`


.. _Mechanism_Overview:

Overview
--------

A Mechanism takes an input, transforms it in some way, and makes the result available as its output.  There are two
types of Mechanisms in PsyNeuLink:

    * `ProcessingMechanisms <ProcessingMechanism>` aggregate the input they receive from other Mechanisms, and/or the
      input to the `Composition` to which they belong, transform it in some way, and provide the result as input to
      other Mechanisms in the Composition, or as the output for the Composition itself.  There are a variety of
      different types of ProcessingMechanism, that accept various forms of input and transform them in different ways
      (see `ProcessingMechanisms <ProcessingMechanism>` for a list).

      to modulate the parameters of other Mechanisms or Projections.  There are three basic ModulatoryMechanisms:

      * `LearningMechanism <LearningMechanism>` - these receive training (target) values, and compare them with the
        output of a Mechanism to generate `LearningSignals <LearningSignal>` that are used to modify `MappingProjections
        <MappingProjection>` (see `learning <Process_Execution_Learning>`).

      * `ControlMechanism <ControlMechanism>` - these evaluate the output of a specified set of Mechanisms, and
        generate `ControlSignals <ControlSignal>` used to modify the parameters of those or other Mechanisms.

      * `GatingMechanism <GatingMechanism>` - these use their input(s) to determine whether and how to modify the
        `value <Port_Base.value>` of the `InputPort(s) <InputPort>` and/or `OutputPort(s) <OutputPort>` of other
        Mechanisms.

      Each type of ModulatoryMechanism is associated with a corresponding type of `ModulatorySignal <ModulatorySignal>`
      (a type of `OutputPort` specialized for use with the ModulatoryMechanism) and `ModulatoryProjection
      <ModulatoryProjection>`.

Every Mechanism is made up of four fundamental components:

    * `InputPort(s) <InputPort>` used to receive and represent its input(s);

    * `Function <Function>` used to transform its input(s) into its output(s);

    * `ParameterPort(s) <ParameterPort>` used to represent the parameters of its Function (and/or any
      parameters that are specific to the Mechanism itself);

    * `OutputPort(s) <OutputPort>` used to represent its output(s)

These are described in the sections on `Mechanism_Function` and `Mechanism_Ports` (`Mechanism_InputPorts`,
`Mechanism_ParameterPorts`, and `Mechanism_OutputPorts`), and shown graphically in a `figure <Mechanism_Figure>`,
under `Mechanism_Structure` below.

.. _Mechanism_Creation:

Creating a Mechanism
--------------------

Mechanisms are created by calling the constructor for a particular type.  PsyNeuLink also automatically
creates one or more Mechanisms under some circumstances. For example, a `ComparatorMechanism` and `LearningMechanism
<LearningMechanism>` are created automatically when `learning is specified <Composition_Learning>` for a `Composition`;
and an `ObjectiveMechanism` may be created when a `ControlMechanism <ControlMechanism>` is created.

COMMENT:
Mechanisms can be created in several ways.  The simplest is to call the constructor for the desired type of Mechanism.
Alternatively, the `mechanism` command can be used to create a specific type of Mechanism or an instance of
`default_mechanism <Mechanism_Base.default_mechanism>`. Mechanisms can also be specified "in context," for example in
the `pathway <Composition.pathway>` attribute of a `Process`; the Mechanism can be specified in either of the ways
mentioned above, or using one of the following:

  * the name of an **existing Mechanism**;

  * the name of a **Mechanism type** (subclass);

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
          <ParameterPort_Specification>`. The parameter values specified will be used to instantiate the Mechanism.
          These can be overridden during execution by specifying `Mechanism_Runtime_Params`, either when calling
          the Mechanism's `execute <Mechanism_Base.execute>` method, or in the `execution method
          <Composition_Execution_Methods>` of a Composition.

  * **automatically** -- PsyNeuLink automatically creates one or more Mechanisms under some circumstances. For example,
    a `ComparatorMechanism` and `LearningMechanism <LearningMechanism>` are created automatically when `learning is
    specified <Composition_Learning>` for a Composition; and an `ObjectiveMechanism` may be created when a
    `ControlMechanism <ControlMechanism>` is created.
COMMENT

.. _Mechanism_Port_Specification:

*Specifying Ports*
~~~~~~~~~~~~~~~~~~~

Every Mechanism has one or more `InputPorts <InputPort>`, `ParameterPorts <ParameterPort>`, and `OutputPorts
<OutputPort>` (described `below <Mechanism_Ports>`) that allow it to receive and send `Projections <Projection>`,
and to execute its `function <Mechanism_Base.function>`).  When a Mechanism is created, it automatically creates the
ParameterPorts it needs to represent its parameters, including those of its `function <Mechanism_Base.function>`.
It also creates any InputPorts and OutputPorts required for the Projections it has been assigned. InputPorts and
OutputPorts, and corresponding Projections (including those from `ModulatorySignals <ModulatorySignal>`) can also be
specified explicitly in the **input_ports** and **output_ports** arguments of the Mechanism's constructor (see
`Mechanism_InputPorts` and `Mechanism_OutputPorts`, respectively, as well as the `first example <Mechanism_Example_1>`
below, and `Port_Examples`).  They can also be specified in a `parameter specification dictionary
<ParameterPort_Specification>` assigned to the Mechanism's **params** argument, using entries with the keys
*INPUT_PORTS* and *OUTPUT_PORTS*, respectively (see `second example <Mechanism_Example_2>` below).  While
specifying the **input_ports** and **output_ports** arguments directly is simpler and more convenient,
the dictionary format allows parameter sets to be created elsewhere and/or re-used.  The value of each entry can be
any of the allowable forms for `specifying a port <Port_Specification>`. InputPorts and OutputPorts can also be
added to an existing Mechanism using its `add_ports <Mechanism_Base.add_ports>` method, although this is generally
not needed and can have consequences that must be considered (e.g., see `note <Mechanism_Add_InputPorts_Note>`),
and therefore is not recommended.

.. _Mechanism_Default_Port_Suppression_Note:

    .. note::
       When Ports are specified in the **input_ports** or **output_ports** arguments of a Mechanism's constructor,
       they replace any default Ports generated by the Mechanism when it is created (if no Ports were specified).
       This is particularly relevant for OutputPorts, as most Mechanisms create one or more `Standard OutputPorts
       <OutputPort_Standard>` by default, that have useful properties.  To retain those Ports if any are specified in
       the **output_ports** argument, they must be included along with those ports in the **output_ports** argument
       (see `examples <Port_Standard_OutputPorts_Example>`).  The same is true for default InputPorts and the
       **input_ports** argument.

       This behavior differs from adding a Port once the Mechanism is created.  Ports added to Mechanism using the
       Mechanism's `add_ports <Mechanism_Base.add_ports>` method, or by assigning the Mechanism in the **owner**
       argument of the Port's constructor, are added to the Mechanism without replacing any of its existing Ports,
       including any default Ports that may have been generated when the Mechanism was created (see `examples
       <Port_Create_Port_Examples>` in Port).


Examples
^^^^^^^^

.. _Mechanism_Example_1:

The following example creates an instance of a TransferMechanism that names the default InputPort ``MY_INPUT``,
and assigns three `Standard OutputPorts <OutputPort_Standard>`::

    >>> import psyneulink as pnl
    >>> my_mech = pnl.TransferMechanism(input_ports=['MY_INPUT'],
    ...                                 output_ports=[pnl.RESULT, pnl.MEAN, pnl.VARIANCE])


.. _Mechanism_Example_2:

This shows how the same Mechanism can be specified using a dictionary assigned to the **params** argument::

     >>> my_mech = pnl.TransferMechanism(params={pnl.INPUT_PORTS: ['MY_INPUT'],
     ...                                         pnl.OUTPUT_PORTS: [pnl.RESULT, pnl.MEAN, pnl.VARIANCE]})

See `Port <Port_Examples>` for additional examples of specifying the Ports of a Mechanism.

.. _Mechanism_Parameter_Specification:

*Specifying Parameters*
~~~~~~~~~~~~~~~~~~~~~~~

As described `below <Mechanism_ParameterPorts>`, Mechanisms have `ParameterPorts <ParameterPort>` that provide the
current value of a parameter used by the Mechanism and/or its `function <Mechanism_Base.function>` when it is `executed
<Mechanism_Execution>`. These can also be used by a `ControlMechanism <ControlMechanism>` to control the parameters of
the Mechanism and/or it `function <Mechanism_Base.function>`.  The value of any of these, and their control, can be
specified in the corresponding argument of the constructor for the Mechanism and/or its `function
<Mechanism_Base.function>`,  or in a parameter specification dictionary assigned to the **params** argument of its
constructor, as described under `ParameterPort_Specification`.


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
parameters).  For example, the `function <Mechanism_Base.function>` of a `TransferMechanism`, which is `Linear` by
default, can be specified to be the `Logistic` function as follows::

    >>> my_mechanism = pnl.TransferMechanism(function=pnl.Logistic(gain=1.0, bias=-4))

Notice that the parameters of the :keyword:`function` (in this case, `gain` and `bias`) can be specified by including
them in its constructor.  Some Mechanisms support only a single function.  In that case, the :keyword:`function`
argument is not available in the Mechanism's constructor, but it does include arguments for the function's
parameters.  For example, the :keyword:`function` of a `ComparatorMechanism` is always the `LinearCombination` function,
so the Mechanisms' constructor does not have a :keyword:`function` argument.  However, it does have a
**comparison_operation** argument, that is used to set the LinearCombination function's `operation` parameter.

The parameters for a Mechanism's primary function can also be specified as entries in a *FUNCTION_PARAMS* entry of a
`parameter specification dictionary <ParameterPort_Specification>` in the **params** argument of the Mechanism's
constructor.  For example, the parameters of the `Logistic` function in the example above can
also be assigned as follows::

    >>> my_mechanism = pnl.TransferMechanism(function=pnl.Logistic,
    ...                                      params={pnl.FUNCTION_PARAMS: {pnl.GAIN: 1.0, pnl.BIAS: -4.0}})

Again, while not as simple as specifying these as arguments in the function's constructor, this format is more flexible.
Any values specified in the parameter dictionary will **override** any specified within the constructor for the function
itself (see `DDM <DDM_Creation>` for an example).

COMMENT:
.. _Mechanism_Function_Attribute:

`function <Mechanism_Base.function>` *attribute*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `Function <Function>` assigned as the primary function of a Mechanism is assigned to the Mechanism's
`function <Component.function>` attribute, and its `function <Function_Base.function>` is assigned
to the Mechanism's `function <Mechanism_Base.function>` attribute.
COMMENT

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
specification dictionary <ParameterPort_Specification>` as described above;  however, any parameters that are unique
to a particular function must be specified in a constructor for that function.  For Mechanisms that have additional,
auxiliary functions, those must be specified in arguments for them in the Mechanism's constructor, and their parameters
must be specified in constructors for those functions unless documented otherwise.
COMMENT


COMMENT:
    FOR DEVELOPERS:
    + FUNCTION : function or method :  method used to transform Mechanism input to its output;
        This must be implemented by the subclass, or an exception will be raised;
        each item in the variable of this method must be compatible with a corresponding InputPort;
        each item in the output of this method must be compatible  with the corresponding OutputPort;
        for any parameter of the method that has been assigned a ParameterPort,
        the output of the ParameterPort's own execute method must be compatible with
        the value of the parameter with the same name in params[FUNCTION_PARAMS] (EMP)
    + FUNCTION_PARAMS (dict):
        NOTE: function parameters can be specified either as arguments in the Mechanism's __init__ method,
        or by assignment of the function_params attribute.
        Only one of these methods should be used, and should be chosen using the following principle:
        - if the Mechanism implements one function, then its parameters should be provided as arguments in the __init__
        - if the Mechanism implements several possible functions and they do not ALL share the SAME parameters,
            then the function should be provided as an argument but not they parameters; they should be specified
            as arguments in the specification of the function
        each parameter is instantiated as a ParameterPort
        that will be placed in <Mechanism_Base>._parameter_ports;  each parameter is also referenced in
        the <Mechanism>.function_params dict, and assigned its own attribute (<Mechanism>.<param>).
COMMENT


.. _Mechanism_Custom_Function:

Custom Functions
^^^^^^^^^^^^^^^^

A Mechanism's `function <Mechanism_Base.function>` can be customized by assigning a user-defined function (e.g.,
a lambda function), so long as it takes arguments and returns values that are compatible with those of the
Mechanism's defaults for that function.  This is also true for auxiliary functions that appear as arguments in a
Mechanism's constructor (e.g., the `EVCControlMechanism`).  A user-defined function can be assigned
directly to the corresponding attribute of the Mechanism
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

`variable <Mechanism_Base.variable>` *and* `value <Mechanism_Base.value>` *attributes*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The input to a Mechanism's `function <Mechanism_Base.function>` is provided by the Mechanism's `variable
<Mechanism_Base.variable>` attribute.  This is an ndarray that is at least 2d, with one item of its outermost
dimension (axis 0) for each of the Mechanism's `input_ports <Mechanism_Base.input_ports>` (see
`below <Mechanism_InputPorts>`).  The result of the  `function <Mechanism_Base.function>` is placed in the
Mechanism's `value <Mechanism_Base.value>` attribute which is  also at least a 2d array.  The Mechanism's `value
<Mechanism_Base.value>` is referenced by its `OutputPorts <Mechanism_OutputPorts>` to generate their own `value
<OutputPort.value>` attributes, each of which is assigned as the value of an item of the list in the Mechanism's
`output_values <Mechanism_Base.output_values>` attribute (see `Mechanism_OutputPorts` below).

.. note::
   The input to a Mechanism is not necessarily the same as the input to its `function <Mechanism_Base.function>`. The
   input to a Mechanism is first processed by its `InputPort(s) <Mechanism_InputPorts>`, and then assigned to the
   Mechanism's `variable <Mechanism_Base.variable>` attribute, which is used as the input to its `function
   <Mechanism_Base.function>`. Similarly, the result of a Mechanism's function is not necessarily the same as the
   Mechanism's output.  The result of the `function <Mechanism_Base.function>` is assigned to the Mechanism's  `value
   <Mechanism_Base.value>` attribute, which is then used by its `OutputPort(s) <Mechanism_OutputPorts>` to assign
   items to its `output_values <Mechanism_Base.output_values>` attribute.

.. _Mechanism_Ports:

*Ports*
~~~~~~~~

Every Mechanism has one or more of each of three types of Ports:  `InputPort(s) <InputPort>`,
`ParameterPort(s) <ParameterPort>`, `and OutputPort(s) <OutputPort>`.  Generally, these are created automatically
when the Mechanism is created.  InputPorts and OutputPorts (but not ParameterPorts) can also be specified explicitly
for a Mechanism, or added to an existing Mechanism using its `add_ports <Mechanism_Base.add_ports>` method, as
described `above <Mechanism_Port_Specification>`).

.. _Mechanism_Figure:

The three types of Ports are shown schematically in the figure below, and described briefly in the following sections.

.. figure:: _static/Mechanism_Ports_fig.svg
   :alt: Mechanism Ports
   :scale: 75 %
   :align: left

   **Schematic of a Mechanism showing its three types of Ports** (`InputPort`, `ParameterPort` and `OutputPort`).
   Every Mechanism has at least one (`primary <InputPort_Primary>`) InputPort and one (`primary
   <OutputPort_Primary>`) OutputPort, but can have additional ports of each type.  It also has one
   `ParameterPort` for each of its parameters and the parameters of its `function <Mechanism_Base.function>`.
   The `value <InputPort.value>` of each InputPort is assigned as an item of the Mechanism's `variable
   <Mechanism_Base.variable>`, and the result of its `function <Mechanism_Base.function>` is assigned as the Mechanism's
   `value <Mechanism_Base.value>`, the items of which are referenced by its OutputPorts to determine their own
   `value <OutputPort.value>`\\s (see `Mechanism_Variable_and_Value` above, and more detailed descriptions below).

.. _Mechanism_InputPorts:

InputPorts
^^^^^^^^^^^

These receive, potentially combine, and represent the input to a Mechanism, and provide this to the Mechanism's
`function <Mechanism_Base.function>`. Usually, a Mechanism has only one (`primary <InputPort_Primary>`) `InputPort`,
identified in its `input_port <Mechanism_Base.input_port>` attribute. However some Mechanisms have more than one
InputPort. For example, a `ComparatorMechanism` has one InputPort for its **SAMPLE** and another for its **TARGET**
input. All of the Mechanism's InputPorts (including its primary InputPort <InputPort_Primary>` are listed in its
`input_ports <Mechanism_Base.input_ports>` attribute (note the plural).  The `input_ports
<Mechanism_Base.input_ports>` attribute is a ContentAddressableList -- a PsyNeuLink-defined subclass of the Python
class `UserList <https://docs.python.org/3.6/library/collections.html?highlight=userlist#collections.UserList>`_ --
that allows a specific InputPort in the list to be accessed using its name as the index for the list (e.g.,
``my_mechanism['InputPort name']``).

.. _Mechanism_Variable_and_InputPorts:

The `value <InputPort.value>` of each InputPort for a Mechanism is assigned to a different item of the Mechanism's
`variable <Mechanism_Base.variable>` attribute (a 2d np.array), as well as to a corresponding item of its `input_values
<Mechanism_Base.input_values>` attribute (a list).  The `variable <Mechanism_Base.variable>` provides the input to the
Mechanism's `function <Mechanism_Base.function>`, while its `input_values <Mechanism_Base.input_values>` provides a
convenient way of accessing the value of its individual items.  Because there is a one-to-one correspondence between
a Mechanism's InputPorts and the items of its `variable <Mechanism_Base.variable>`, their size along their outermost
dimension (axis 0) must be equal; that is, the number of items in the Mechanism's `variable <Mechanism_Base.variable>`
attribute must equal the number of InputPorts in its `input_ports <Mechanism_Base.input_ports>` attribute. A
Mechanism's constructor does its best to insure this:  if its **default_variable** and/or its **size** argument is
specified, it constructs a number of InputPorts (and each with a `value <InputPort.value>`) corresponding to the
items specified for the Mechanism's `variable <Mechanism_Base.variable>`, as in the examples below::

    my_mech_A = pnl.TransferMechanism(default_variable=[[0],[0,0]])
    print(my_mech_A.input_ports)
    > [(InputPort InputPort-0), (InputPort InputPort-1)]
    print(my_mech_A.input_ports[0].value)
    > [ 0.]
    print(my_mech_A.input_ports[1].value)
    > [ 0.  0.]

    my_mech_B = pnl.TransferMechanism(default_variable=[[0],[0],[0]])
    print(my_mech_B.input_ports)
    > [(InputPort InputPort-0), (InputPort InputPort-1), (InputPort InputPort-2)]

Conversely, if the **input_ports** argument is used to specify InputPorts for the Mechanism, they are used to format
the Mechanism's variable::

    my_mech_C = pnl.TransferMechanism(input_ports=[[0,0], 'Hello'])
    print(my_mech_C.input_ports)
    > [(InputPort InputPort-0), (InputPort Hello)]
    print(my_mech_C.variable)
    > [array([0, 0]) array([0])]

If both the **default_variable** (or **size**) and **input_ports** arguments are specified, then the number and format
of their respective items must be the same (see `Port <Port_Examples>` for additional examples of specifying Ports).

If InputPorts are added using the Mechanism's `add_ports <Mechanism_Base.add_ports>` method, then its
`variable <Mechanism_Base.variable>` is extended to accommodate the number of InputPorts added (note that this must
be coordinated with the Mechanism's `function <Mechanism_Base.function>`, which takes the Mechanism's `variable
<Mechanism_Base.variable>` as its input (see `note <Mechanism_Add_InputPorts_Note>`).

The order in which `InputPorts are specified <Mechanism_InputPort_Specification>` in the Mechanism's constructor,
and/or `added <Mechanism_Add_InputPorts>` using its `add_ports <Mechanism_Base.add_ports>` method,  determines the
order of the items to which they are assigned assigned in he Mechanism's `variable  <Mechanism_Base.variable>`,
and are listed in its `input_ports <Mechanism_Base.input_ports>` and `input_values <Mechanism_Base.input_values>`
attribute.  Note that a Mechanism's `input_values <Mechanism_Base.input_values>` attribute has the same information as
the Mechanism's `variable <Mechanism_Base.variable>`, but in the form of a list rather than an ndarray.

.. _Mechanism_InputPort_Specification:

**Specifying InputPorts and a Mechanism's** `variable <Mechanism_Base.variable>` **Attribute**

When a Mechanism is created, the number and format of the items in its `variable <Mechanism_Base.variable>`
attribute, as well as the number of InputPorts it has and their `variable <InputPort.variable>` and `value
<InputPort.value>` attributes, are determined by one of the following arguments in the Mechanism's constructor:

* **default_variable** (at least 2d ndarray) -- determines the number and format of the items of the Mechanism's
  `variable <Mechanism_Base.variable>` attribute.  The number of items in its outermost dimension (axis 0) determines
  the number of InputPorts created for the Mechanism, and the format of each item determines the format for the
  `variable <InputPort.variable>` and `value  <InputPort.value>` attributes of the corresponding InputPort.
  If any InputPorts are specified in the **input_ports** argument or an *INPUT_PORTS* entry of
  a specification dictionary assigned to the **params** argument of the Mechanism's constructor, then the number
  must match the number of items in **default_variable**, or an error is generated.  The format of the items in
  **default_variable** are used to specify the format of the `variable <InputPort.variable>` or `value
  <InputPort.value>` of the corresponding InputPorts for any that are not explicitly specified in the
  **input_ports** argument or *INPUT_PORTS* entry (see below).
..
* **size** (int, list or ndarray) -- specifies the number and length of items in the Mechanism's variable,
  if **default_variable** is not specified. For example, the following mechanisms are equivalent::
    T1 = TransferMechanism(size = [3, 2])
    T2 = TransferMechanism(default_variable = [[0, 0, 0], [0, 0]])
  The relationship to any specifications in the **input_ports** argument or
  *INPUT_PORTS* entry of a **params** dictionary is the same as for the **default_variable** argument,
  with the latter taking precedence (see above).
..
* **input_ports** (list) -- this can be used to explicitly `specify the InputPorts <InputPort_Specification>`
  created for the Mechanism. Each item must be an `InputPort specification <InputPort_Specification>`, and the number
  of items must match the number of items in the **default_variable** argument or **size** argument
  if either of those is specified.  If the `variable <InputPort.variable>` and/or `value <InputPort.value>`
  is `explicitly specified for an InputPort <InputPort_Variable_and_Value>` in the **input_ports** argument or
  *INPUT_PORTS* entry of a **params** dictionary, it must be compatible with the value of the corresponding
  item of **default_variable**; otherwise, the format of the item in **default_variable** corresponding to the
  InputPort is used to specify the format of the InputPort's `variable <InputPort.variable>` (e.g., the InputPort is
  `specified using an OutputPort <InputPort_Projection_Source_Specification>` to project to it;).  If
  **default_variable** is not specified, a default value is specified by the Mechanism.  InputPorts can also be
  specifed that `shadow the inputs <InputPort_Shadow_Inputs>` of other InputPorts and/or Mechanisms; that is, receive
  Projections from all of the same `senders <Projection_Base.sender>` as those specified.

COMMENT:
    *** ADD SOME EXAMPLES HERE (see `examples <XXX>`)
COMMENT

COMMENT:
    *** ADD THESE TO ABOVE WHEN IMPLEMENTED:
        If more InputPorts are specified than there are items in `variable <Mechanism_Base.variable>,
            the latter is extended to  match the former.
        If the Mechanism's `variable <Mechanism_Base.variable>` has more than one item, it may still be assigned
            a single InputPort;  in that case, the `value <InputPort.value>` of that InputPort must have the same
            number of items as the Mechanisms's `variable <Mechanism_Base.variable>`.
COMMENT
..
* *INPUT_PORTS* entry of a params dict (list) -- specifications are treated in the same manner as those in the
  **input_ports** argument, and take precedence over those.

.. _Mechanism_Add_InputPorts:

**Adding InputPorts**

InputPorts can be added to a Mechanism using its `add_ports <Mechanism_Base.add_ports>` method;  this extends its
`variable <Mechanism_Base.variable>` by a number of items equal to the number of InputPorts added, and each new item
is assigned a format compatible with the `value <InputPort.value>` of the corresponding InputPort added;  if the
InputPort's `variable <InputPort.variable>` is not specified, it is assigned the default format for an item of the
owner's `variable <Mechanism_Base.variable>` attribute. The InputPorts are appended to the end of the list in the
Mechanism's `input_ports <Mechanism_Base.input_ports>` attribute.  Adding in Ports in this manner does **not**
replace any existing Ports, including any default Ports generated when the Mechanism was constructed (this is
contrast to Ports specified in a Mechanism's constructor which **do** `replace any default Port(s) of the same type
<Mechanism_Default_Port_Suppression_Note>`).

.. _Mechanism_Add_InputPorts_Note:

.. note::
    Adding InputPorts to a Mechanism using its `add_ports <Mechanism_Base.add_ports>` method may introduce an
    incompatibility with the Mechanism's `function <Mechanism_Base.function>`, which takes the Mechanism's `variable
    <Mechanism_Base.variable>` as its input; such an incompatibility will generate an error.  It may also influence
    the number of OutputPorts created for the Mechanism. It is the user's responsibility to ensure that the
    assignment of InputPorts to a Mechanism using the `add_ports <Mechanism_Base.add_ports>` is coordinated with
    the specification of its `function <Mechanism_Base.function>`, so that the total number of InputPorts (listed
    in the Mechanism's `input_ports <Mechanism_Base.input_ports>` attribute matches the number of items expected
    for the input to the function specified in the Mechanism's `function <Mechanism_Base.function>` attribute
    (i.e., its length along axis 0).

.. _Mechanism_InputPort_Projections:

**Projections to InputPorts**

Each InputPort of a Mechanism can receive one or more `Projections <Projection>` from other Mechanisms.  When a
Mechanism is created, a `MappingProjection` is created automatically for any OutputPorts or Projections from them that
are in its `InputPort specification <InputPort_Specification>`, using `AUTO_ASSIGN_MATRIX` as the Projection's `matrix
specification <MappingProjection_Matrix_Specification>`.  However, if a specification in the **input_ports** argument
or an *INPUT_PORTS* entry of a **params** dictionary cannot be resolved to an instantiated OutputPort at the time the
Mechanism is created, no MappingProjection is assigned to the InputPort, and this must be done by some other means;
any specifications in the Mechanism's `input_ports <Mechanism_Base.input_ports>` attribute that are not
associated with an instantiated OutputPort at the time the Mechanism is executed are ignored.

The `PathwayProjections <PathwayProjection>` (e.g., `MappingProjections <MappingProjection>`) it receives are listed
in its `path_afferents <Port.path_afferents>` attribute.  If the Mechanism is an `ORIGIN` Mechanism of a
`Composition`, this includes a Projection from the Composition's `input_CIM <Composition.input_CIM>`.  Any
`ControlProjections <ControlProjection>` or `GatingProjections <GatingProjection>` it receives are listed in its
`mod_afferents <Port.mod_afferents>` attribute.


.. _Mechanism_ParameterPorts:

*ParameterPorts and Parameters*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`ParameterPorts <ParameterPort>` provide the value for each parameter of a Mechanism and its `function
<Mechanism_Base.function>`.  One ParameterPort is assigned to each of the parameters of the Mechanism and/or its
`function <Mechanism_Base.function>` (corresponding to the arguments in their constructors). The ParameterPort takes
the value specified for a parameter (see `below <Mechanism_Parameter_Value_Specification>`) as its `variable
<ParameterPort.variable>`, and uses it as the input to the ParameterPort's `function <ParameterPort.function>`,
which `modulates <ModulatorySignal_Modulation>` it in response to any `ControlProjections <ControlProjection>` received
by the ParameterPort (specified in its `mod_afferents <ParameterPort.mod_afferents>` attribute), and assigns the
result to the ParameterPort's `value <ParameterPort.value>`.  This is the value used by the Mechanism or its
`function <Mechanism_Base.function>` when the Mechanism `executes <Mechanism_Execution>`.  Accordingly, when the value
of a parameter is accessed (e.g., using "dot" notation, such as ``my_mech.my_param``), it is actually the
*ParameterPort's* `value <ParameterPort.value>` that is returned (thereby accurately reflecting the value used
during the last execution of the Mechanism or its `function <Mechanism_Base.function>`).  The ParameterPorts for a
Mechanism are listed in its `parameter_ports <Mechanism_Base.parameter_ports>` attribute.

.. _Mechanism_Parameter_Value_Specification:

The "base" value of a parameter (i.e., the unmodulated value used as the ParameterPort's `variable
<ParameterPort.variable>` and the input to its `function <ParameterPort.function>`) can specified when a Mechanism
and/or its `function <Mechanism_Base.function>` are first created,  using the corresponding arguments of their
constructors (see `Mechanism_Function` above).  Parameter values can also be specified later, by direct assignment of a
value to the attribute for the parameter, or by using the Mechanism's `assign_param` method (the recommended means;
see `ParameterPort_Specification`).  Note that the attributes for the parameters of a Mechanism's `function
<Mechanism_Base.function>` usually belong to the `Function <Function_Overview>` referenced in its `function
<Component.function>` attribute, not the Mechanism itself, and therefore must be assigned to the Function
Component (see `Mechanism_Function` above).

.. _Mechanism_OutputPorts:

OutputPorts
^^^^^^^^^^^^
These represent the output(s) of a Mechanism. A Mechanism can have several `OutputPorts <OutputPort>`, and each can
send Projections that transmit its value to other Mechanisms and/or as the output of the `Composition` to which
the Mechanism belongs.  Every Mechanism has at least one OutputPort, referred to as its `primary OutputPort
<OutputPort_Primary>`.  If OutputPorts are not explicitly specified for a Mechanism, a primary OutputPort is
automatically created and assigned to its `output_port <Mechanism_Base.output_port>` attribute (note the singular),
and also to the first entry of the Mechanism's `output_ports <Mechanism_Base.output_ports>` attribute (note the
plural).  The `value <OutputPort.value>` of the primary OutputPort is assigned as the first (and often only) item
of the Mechanism's `value <Mechanism_Base.value>` attribute, which is the result of the Mechanism's `function
<Mechanism_Base.function>`.  Additional OutputPorts can be assigned to represent values corresponding to other items
of the Mechanism's `value <Mechanism_Base.value>` (if there are any) and/or values derived from any or all of those
items. `Standard OutputPorts <OutputPort_Standard>` are available for each type of Mechanism, and custom ones can
be configured (see `OutputPort Specification <OutputPort_Specification>`. These can be assigned in the
**output_ports** argument of the Mechanism's constructor.

All of a Mechanism's OutputPorts (including the primary one) are listed in its `output_ports
<Mechanism_Base.output_ports>` attribute (note the plural). The `output_ports <Mechanism_Base.output_ports>`
attribute is a ContentAddressableList -- a PsyNeuLink-defined subclass of the Python class
`UserList <https://docs.python.org/3.6/library/collections.html?highlight=userlist#collections.UserList>`_ -- that
allows a specific OutputPort in the list to be accessed using its name as the index for the list (e.g.,
``my_mechanism['OutputPort name']``).  This list can also be used to assign additional OutputPorts to the Mechanism
after it has been created.

The `value <OutputPort.value>` of each of the Mechanism's OutputPorts is assigned as an item in the Mechanism's
`output_values <Mechanism_Base.output_values>` attribute, in the same order in which they are listed in its
`output_ports <Mechanism_Base.output_ports>` attribute.  Note, that the `output_values <Mechanism_Base.output_values>`
attribute of a Mechanism is distinct from its `value <Mechanism_Base.value>` attribute, which contains the full and
unmodified results of its `function <Mechanism_Base.function>` (this is because OutputPorts can modify the item of
the Mechanism`s `value <Mechanism_Base.value>` to which they refer -- see `OutputPorts <OutputPort_Customization>`).

.. _Mechanism_Additional_Attributes:

*Additional Attributes*
~~~~~~~~~~~~~~~~~~~~~~~

.. _Mechanism_Constructor_Arguments:

*Additional Constructor Arguments*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to the `standard attributes <Component_Structure>` of any `Component <Component>`, Mechanisms have a set of
Mechanism-specific attributes (listed below). These can be specified in arguments of the Mechanism's constructor,
in a `parameter specification dictionary <ParameterPort_Specification>` assigned to the **params** argument of the
Mechanism's constructor, by direct reference to the corresponding attribute of the Mechanisms after it has been
constructed (e.g., ``my_mechanism.param``). The Mechanism-specific
attributes are listed below by their argument names / keywords, along with a description of how they are specified:

    * **input_ports** / *INPUT_PORTS* - a list specifying the Mechanism's input_ports
      (see `InputPort_Specification` for details of specification).
    ..
    * **output_ports** / *OUTPUT_PORTS* - specifies specialized OutputPorts required by a Mechanism subclass
      (see `OutputPort_Specification` for details of specification).
    ..
    COMMENT:
    * **monitor_for_control** / *MONITOR_FOR_CONTROL* - specifies which of the Mechanism's OutputPorts is monitored by
      the `controller` for the Composition to which the Mechanism belongs (see `specifying monitored OutputPorts
      <ObjectiveMechanism_Monitor>` for details of specification).
    COMMENT
    ..
    * **monitor_for_learning** / *MONITOR_FOR_LEARNING* - specifies which of the Mechanism's OutputPorts is used for
      learning (see `Learning <LearningMechanism_Activation_Output>` for details of specification).

.. _Mechanism_Convenience_Properties:

*Projection Convenience Properties*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Mechanism also has several convenience properties, listed below, that list its `Projections <Projection>` and the
Mechanisms that send/receive these:

    * `projections <Mechanism_Base.projections>` -- all of the Projections sent or received by the Mechanism;
    * `afferents <Mechanism_Base.afferents>` -- all of the Projections received by the Mechanism;
    * `path_afferents <Mechanism_Base.afferents>` -- all of the PathwayProjections received by the Mechanism;
    * `mod_afferents <Mechanism_Base.afferents>` -- all of the ModulatoryProjections received by the Mechanism;
    * `efferents <Mechanism_Base.efferents>` -- all of the Projections sent by the Mechanism;
    * `senders <Mechanism_Base.senders>` -- all of the Mechanisms that send a Projection to the Mechanism
    * `modulators <Mechanism_Base.modulators>` -- all of the ModulatoryMechanisms that send a ModulatoryProjection to
      the Mechanism
    * `receivers <Mechanism_Base.receivers>` -- all of the Mechanisms that receive a Projection from the Mechanism

Each of these is a `ContentAddressableList`, which means that the names of the Components in each list can be listed by
appending ``.names`` to the property.  For examples, the names of all of the Mechanisms that receive a Projection from
``my_mech`` can be accessed by ``my_mech.receivers.names``.


.. _Mechanism_Labels_Dicts:

*Value Label Dictionaries*
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Overview*

Mechanisms have two attributes that can be used to specify labels for the values of its InputPort(s) and
OutputPort(s):

    * *INPUT_LABELS_DICT* -- used to specify labels for values of the InputPort(s) of the Mechanism;  if specified,
      the dictionary is contained in the Mechanism's `input_labels_dict <Mechanism_Base.input_labels_dict>` attribute.

    * *OUTPUT_LABELS_DICT* -- used to specify labels for values of the OutputPort(s) of the Mechanism;  if specified,
      the dictionary is contained in the Mechanism's `output_labels_dict <Mechanism_Base.output_labels_dict>` attribute.

The labels specified in these dictionaries can be used to:

    - specify items in the `inputs <Composition_Execution_Inputs>` argument of the `run <Composition.run>` method of a
      `Composition`, or the `targets <Composition_Targret_Inputs>` argument of its `learn <Composition.learn>` method.
    - report the values of the InputPort(s) and OutputPort(s) of a Mechanism
    - visualize the inputs and outputs of the Composition's Mechanisms

*Specifying Label Dictionaries*

Label dictionaries can only be specified in a parameters dictionary assigned to the **params** argument of the
Mechanism's constructor, using the keywords described above.  A standard label dictionary contains key:value pairs of
the following form:

    * *<port name or index>:<sub-dictionary>* -- this is used to specify labels that are specific to individual Ports
      of the type corresponding to the dictionary;
        - *key* - either the name of a Port of that type, or its index in the list of Ports of that type (i.e,
          `input_ports <Mechanism_Base.input_ports>` or `output_ports <Mechanism_Base.output_ports>`);
        - *value* - a dictionary containing *label:value* entries to be used for that Port, where the label is a string
          and the shape of the value matches the shape of the `InputPort value <InputPort.value>` or `OutputPort
          value <OutputPort.value>` for which it is providing a *label:value* mapping.

      For example, if a Mechanism has two InputPorts, named *SAMPLE* and *TARGET*, then *INPUT_LABELS_DICT* could be
      assigned two entries, *SAMPLE*:<dict> and *TARGET*:<dict> or, correspondingly, 0:<dict> and 1:<dict>, in which
      each dictionary contains separate *label:value* entries for the *SAMPLE* and *TARGET* InputPorts.

>>> input_labels_dictionary = {pnl.SAMPLE: {"red": [0],
...                                         "green": [1]},
...                            pnl.TARGET: {"red": [0],
...                                         "green": [1]}}

In the following two cases, a shorthand notation is allowed:

    - a Mechanism has only one port of a particular type (only one InputPort or only one OutputPort)
    - only the index zero InputPort or index zero OutputPort needs labels

In these cases, a label dictionary for that type of port may simply contain the *label:value* entries described above.
The *label:value* mapping will **only** apply to the index zero port of the port type for which this option is used.
Any additional ports of that type will not have value labels. For example, if the input_labels_dictionary below were
applied to a Mechanism with multiple InputPort, only the index zero InputPort would use the labels "red" and "green".

>>> input_labels_dictionary = {"red": [0],
...                            "green": [1]}

*Using Label Dictionaries*

When using labels to specify items in the `inputs <Composition_Execution_Inputs>` arguments of the `run
<Composition.run>` method, labels may directly replace any or all of the `InputPort values <InputPort.value>` in an
input specification dictionary. Keep in mind that each label must be specified in the `input_labels_dict
<Mechanism_Base.input_labels_dict>` of the `INPUT` Mechanism to which inputs are being specified, and must map to a
value that would have been valid in that position of the input dictionary.

        >>> import psyneulink as pnl
        >>> input_labels_dict = {"red": [[1, 0, 0]],
        ...                      "green": [[0, 1, 0]],
        ...                      "blue": [[0, 0, 1]]}
        >>> M = pnl.ProcessingMechanism(default_variable=[[0, 0, 0]],
        ...                             params={pnl.INPUT_LABELS_DICT: input_labels_dict})
        >>> C = pnl.Composition(pathways=[M])
        >>> input_dictionary = {M: ['red', 'green', 'blue', 'red']}
        >>> # (equivalent to {M: [[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]], [[1, 0, 0]]]}, which is a valid input specification)
        >>> results = C.run(inputs=input_dictionary)

The same general rules apply when using labels to specify `target values <Run_Targets>` for a pathway with learning.
With target values, however, the labels must be included in the `output_labels_dict <Mechanism_Base.output_labels_dict>`
of the Mechanism that projects to the `TARGET` Mechanism (see `TARGET Mechanisms <LearningMechanism_Targets>`), or in
other words, the last Mechanism in a `learning pathway <LearningMechanism_Multilayer_Learning>`. This is the same
Mechanism used to specify target values for a particular learning pathway in the `targets dictionary <Run_Targets>`.

        >>> input_labels_dict_M1 = {"red": [[1]],
        ...                         "green": [[0]]}
        >>> output_labels_dict_M2 = {"red": [1],
        ...                         "green": [0]}
        >>> M1 = pnl.ProcessingMechanism(params={pnl.INPUT_LABELS_DICT: input_labels_dict_M1})
        >>> M2 = pnl.ProcessingMechanism(params={pnl.OUTPUT_LABELS_DICT: output_labels_dict_M2})
        >>> C = pnl.Composition()
        >>> learning_pathway = C.add_backpropagation_learning_pathway(pathway=[M1, M2], learning_rate=0.25)
        >>> input_dictionary = {M1: ['red', 'green', 'green', 'red']}
        >>> # (equivalent to {M1: [[[1]], [[0]], [[0]], [[1]]]}, which is a valid input specification)
        >>> target_dictionary = {M2: ['red', 'green', 'green', 'red']}
        >>> # (equivalent to {M2: [[1], [0], [0], [1]]}, which is a valid target specification)
        >>> results = C.learn(inputs=input_dictionary,
        ...                   targets=target_dictionary)

Several attributes are available for viewing the labels for the current value(s) of a Mechanism's InputPort(s) and
OutputPort(s).

    - The `label <InputPort.label>` attribute of an InputPort or OutputPort returns the current label of
      its value, if one exists, and its value otherwise.

    - The `input_labels <Mechanism_Base.input_labels>` and `output_labels <Mechanism_Base.output_labels>` attributes of
      Mechanisms return a list containing the labels corresponding to the value(s) of the InputPort(s) or
      OutputPort(s) of the Mechanism, respectively. If the current value of a port does not have a corresponding
      label, then its numeric value is used instead.

        >>> output_labels_dict = {"red": [1, 0, 0],
        ...                      "green": [0, 1, 0],
        ...                      "blue": [0, 0, 1]}
        >>> M = pnl.ProcessingMechanism(default_variable=[[0, 0, 0]],
        ...                             params={pnl.OUTPUT_LABELS_DICT: output_labels_dict})
        >>> C = pnl.Composition(pathways=[M])
        >>> input_dictionary =  {M: [[1, 0, 0]]}
        >>> results = C.run(inputs=input_dictionary)
        >>> M.get_output_labels(C)
        ['red']
        >>> M.output_ports[0].get_label(C)
        'red'

Labels may be used to visualize the input and outputs of Mechanisms in a Composition with the **show_structure** option
of the Composition's `show_graph <Composition.show_graph>` method with the keyword **LABELS**.

        >>> C.show_graph(show_mechanism_structure=pnl.LABELS)  #doctest: +SKIP

.. note::

    A given label dictionary only applies to the Mechanism to which it belongs, and a given label only applies to its
    corresponding InputPort. For example, the label 'red', may translate to different values on different InputPorts
    of the same Mechanism, and on different Mechanisms of a Composition.

.. _Mechanism_in_Composition:

*Mechanisms in Compositions*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mechanisms are most commonly used as `Nodes <Composition_Nodes>` in a `Composition's graph <Composition_Graph>`,
where they are connected to other Nodes using `Projections <Projection>`.

.. _Mechanism_Role_In_Compositions:

*Role in Compositions*
^^^^^^^^^^^^^^^^^^^^^^

When a Mechanism is added to a `Composition_Addition_Methods>`, it is assigned as a `Node <Composition_Nodes>` in the
`graph <Composition_Graph>` of that Comopsition, and one or more `NodeRoles <NodeRole>` indicating the role(s) that
the Node play(s) in the Composition.  These can be listed by calling the `Comopsition's `get_roles_by_nodes
<Comopsition.get_roles_by_nodes>` with the Mechanism as its argument.  The NodeRoles assigned to a Mechanism can
be different for different Compositions.  If a Mechanism is designated as an `INPUT` Node, it receives a
`MappingProjection` to its `primary InputPort <InputPort_Primary>` from the Composition.  When the Composition is
`executed <Composition_Execution>`, that InputPort receives the input specified for the Mechanism in the `inputs
<Composition_Execution_Inputs>` argument of the Composition's `execution method <Composition_Execution_Methods>`; or,
if it is a `nested Composition <Composition_Nested>`, then the Mechanism gets its input from a Node that projects to
it from the outer Composition.  If a Mechanism is designated as an `OUTPUT` Node, its `output_values
<Mechanism_Base.output_values>` are included in the value returned by its `execution method
<Composition_Execution_Methods>` and the Composition's `results <Composition.results>` attribute.

.. _Mechanism_Execution:

Execution
---------

When a Mechanism executes, the following sequence of actions is carried out:

    - The Mechanism updates its `InputPort`\\(s) by executing the `function <InputPort.function>` of each.  The
      resulting `value <InputPort.value>`\\(s) are used to assemble the Mechanism's `variable<Mechanism_Base.variable>`.
      Each `value <InputPort.value>` is added to an outer array, such that each item of the Mechanism's `variable
      <Mechanism_Base.variable>` corresponds to an InputPort `value <InputPort.value>`.  The array is placed in
      the Mechanism's `input_values <Mechanism_Base.input_values>` attribute, and also passed as the input to the
      Mechanism's `function <Mechanism_Base.function>` after updating its `ParameterPorts <ParamterPorts>`.

    - The Mechanism updates its `ParameterPort`\\(s) by executing each of their `functions <ParameterPort.function>`,
      the results of which are assigned as the values used for the corresponding Parameters, which include those of the
      Mechanism's `function <Mechanism_Base.function>`.

    - The Mechanism's `variable <Mechanism_Base.variable>` is passed as the input to the its `function
      <Mechanism_Base.function>`, and the function is execute using the parameter values generating by the execution
      of its ParameterPorts. The result of the Mechanism's `function <Mechanism_Base.function>` is placed in the
      Mechanism's `value <Mechanism_Base.value>` attribute.

    - The Mechanism updates its `OutputPort`\\(s) are updated based on `value <Mechanism_Base.value>`, by executing the
      `function <OutputPort.function>` of each. The resulting `value <OutputPort.value>` for each Outport is placed
      in the Mechanism's `output_values <Mechanism_Base.output_values>` attribute.

A Mechanism may be executed by calling its `execute <Mechanism_Base.execute>` method directly:

    >>> my_simple_mechanism = pnl.ProcessingMechanism()      #doctest: +SKIP
    >>> my_simple_mechanism.execute(1.0)                     #doctest: +SKIP

This can be useful for testing a Mechanism and/or debugging.  However, more typically, Mechanisms are `executed as
part of a Composition <Composition_Execution>`.

.. _Mechanism_Execution_Composition:

*Execution in a Composition*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Mechanism can be assigned to one or more Compositions;  the values of its `parameters <Component_Parameters>`,
including its `variable <Mechanism_Base.variable>` and `value <Mechanism_Base.value>` attributes, are maintained
separately for each `context in which it is executed <Composition_Execution_Context>` which, by default, is distinct
for each Composition in which it is executed;  these execution-specific values can be accessed using the parameter's
`get <Parameter.get>` method. A parameter's value can also be accessed using standard `dot <Parameter_Dot_Notation>`,
which returns its most recenty assigned value, irrespective of the context (including Composition) in which it was
assigned.

.. _Mechanism_Runtime_Params:

*Runtime Parameters*
~~~~~~~~~~~~~~~~~~~~

.. note::
   This is an advanced feature, but is generally not required for most applications. It is included for convenience;
   similar functionality can be achieved by setting the values of `parameters <Component_Parameters>` programmatically
   before the Mechanism is executed and then resetting them afterwards.

The runtime parameter values are those assigned to a Mechanism and its Components (i.e., its `function
<Mechanism_Base.function>` and `Ports <Mechanism_Ports`) when they execute.  These are generally the values specified
in the corresponding constructors, assigned explicitly after construction (see `User_Modifiable_Parameters`), or the
default values.  However, these values can be overidden for a particular execution, by specifying the desired values
in the **runtime_params** argument of the Mechanism's `execute <Mechanism_Base.execute>` method (see `below
<Mechanism_Runtime_Param_Specification>`) or the `execution method <Composition_Execution_Methods>` of a `Composition`
to which it belongs (see `Composition_Runtime_Params`).  When assigned in the context of a Composition, `Conditions
<Condition>` can be specified that determine when the values apply. Any values assigned using **runtime_params**
that apply will override the current value of the parameter for that (and *only* that) execution (if the Mechanism's
`execute <Mechanism_Base.execute>` is used) or as long as its `Condition` applies (if executed in a Composition),
after which the value will return to its previous value.  The value of a parameter can be modified on a permanent
basis, either for a given `execution context <Composition_Execution_Context>` using a its `set <Parameter.set>`
method; or for all execution contexts, by setting its default value using the Mechanism's `defaults
<Component.defaults>` attribute.

.. _Mechanism_Runtime_Param_Specification:

*Runtime specification ditionary: parameters of a Mechanism and its function*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Runtime parameter values are specified in the **runtime_params** argument of a Mechanism's `execute
<Mechanism_Base.execute>` method using a dictionary, in which each entry contains the name of the
of a parameter (as the key) and the value to assign to it, as in the following example::

        >>> T = pnl.TransferMechanism(function=Linear)
        >>> T.function.slope  #doctest: +SKIP
        1.0   # Default for slope
        >>> T.clip #doctest: +SKIP
        None  # Default for clip is None
        >>> T.execute(2.0,
        ...          runtime_params={"slope": 3.0,
        ...                           "clip": (0,5)}) #doctest: +SKIP
        array([[5.]])  # = 2 (input) * 3 (slope) = 6, but clipped at 5
        >>> T.function.slope #doctest: +SKIP
        1.0   # slope is restored 1.0
        >>> T.clip     #doctest: +SKIP
        None  # clip is restored to None

Note that even though ``slope`` is a parameter of the Mechanism's `function <Mechanism_Base.function>` (in this case,
`Linear`), the function itself does not have to be specified in the key of the runtime_params dictionary (although it
does have to be used when accessing or assigning the parameter's value using `dot notation <Parameter_Dot_Notation>`,
as shown above).

If a parameter is assigned a new value before the execution, that value is restored after the execution;  that is,
the parameter is assigned its previous value and *not* its default, as shown below::

        >>> T.function.slope = 10
        >>> T.clip = (0,3)
        >>> T.function.slope
        10
        >>> T.clip
        (0, 3)
        >>> T.execute(3.0,
        ...          runtime_params={"slope": 4.0,
        ...                           "clip": (0,4)}) #doctest: +SKIP
        array([[4.]])  # = 3 (input) * 4 (slope) = 12, but clipped at 4
        >>> T.function.slope #doctest: +SKIP
        10      # slope is restored 10.0, its previously assigned value
        >>> T.clip #doctest: +SKIP
        (0, 3)  # clip is restored to (0,3), its previously assigned value

.. _Mechanism_Runtime_Port_and_Projection_Param_Specification:

*Runtime specification ditionary: parameters of a Mechanism's Ports and Projections*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Runtime values can also be assigned to the parameters of a Mechanism's `Ports <Port>` and/or their `afferent
Projections <Mechanism_Base.afferents>` in entries of a **runtime_params** dict,

*Ports*.  Runtime values are assigned to the parameters of Ports (and/or their `function <Port_Base.function>`\\s)
in entries with a key that designates the type of Port (*INPUT_PORT_PARAMS*, *OUTPUT_PORT_PARAMS* or
*PARAMETER_PORT_PARAMS*), and a sub-dictionary containing the specifications for that type of Port as the value.
The sub-dictionary can contain entries with specification that apply to *all* Ports of that type and/or individual
Ports. If the key of an entry is the name of a parameter of the Port (or its `function <Port_Base.function>`), the
specified value applies to *all* Ports of that type.  Parameters for individual Ports are specified using the Port
or its `name <Port_Base.name>` as the key, and a dictionary containing parameter specifications as its value.

  .. note::

     - If the `variable <Port_base.variable>` of a Port is specified as a runtime parameter, then its afferent
       Projections will not be executed (see `Lazy Evaluation <Component_Lazy_Updating>`), but its `function
       <Port_Base.function>` will be.

     - If the `value <Port_Base.value>` of a Port is specified, *neither its `afferent Projections <Port_Projections>`
       nor it `function <Port_Base.function>` will be executed.

     - If the `variable <Port_base.variable>` and/or `value <Port_Base.value>` is specified for *all* of the
       OutputPorts of a Mechanism, then it's function will not be executed, and the `value <Mechanism_Base.value>`
       will retain its previous value (again in accord with `Lazy Evaluation <Component_Lazy_Updating>), though its
       OutputPorts *will* be executed using the assigned values, and it's `execution_count <Component_Execution_Count>`
       and `num_executions <Component_Num_Executions>` attributes will be incremented (since the OutputPorts --
       Components of the Mechanism -- executed).

     - As expected, specifying `value <Port_Base.value>` supercedes any specification of `variable
       <Port_Base.variable>` or of the parameters of its `function <Port_Base.function>`.

COMMENT:
   FIX 5/8/20 [JDC]: GET EXAMPLES FROM test_runtime_params
COMMENT

*Projections*.  The sub-dictionary specifying the parameters of a Port can also contain specifications for parameters
of its afferent `Projections <Port_Projections>` Projections.  These are placed in entries with a key that designates
the type of Projection, and a sub-dictionary containing the specifications for that type of Port as the value.  The
key for each type of projecction is its `componentType <Component_Type>` appended with ``_PARAMS`` (e.g.,
*MAPPING_PROJECTION_PARAMS*, *CONTROL_PROJECTION_PARAMS*, etc.).  The sub-dictionary can contain specifications that
apply to *all* Projections of that type and/or individual Projections. If the key of an entryis the name of a parameter
of the Projection (or its `function <Port_Base.function>`), the specified value applies to *all* Projections of that
type. Parameters for individual Projections are specified using the Projections or its `name <Projection_Base.name>
as the key, and a dictionary containing parameter specifications as its value.

   .. note::
     If the `value <Projection_Base.value>` of a Projection is specified as a runtime parameter, then it will not be
     executed (see `Lazy Evaluation <Component_Lazy_Updating>`); accordingly, specifying `value <Port_Base.value>`
     supercedes any specification of `variable <Port_Base.variable>` or of the parameters of its `function
     <Projection_Base.function>.`

COMMENT:
   FIX 5/8/20 [JDC]: EXAMPLES HERE AND ADD CORRESPONDING TESTS
.. note::
    Runtime parameter values are subject to the same type, value, and shape requirements as the original parameter
    value.
COMMENT


COMMENT:
?? DO PROJECTION DICTIONARIES PERTAIN TO INCOMING OR OUTGOING PROJECTIONS OR BOTH??
?? CAN THE KEY FOR A PORT DICTIONARY REFERENCE A SPECIFIC PORT BY NAME, OR ONLY PORT-TYPE??

Port keyword: dict for Port's params
    Function or Projection keyword: dict for Funtion or Projection's params
        parameter keyword: vaue of param

    dict: can be one (or more) of the following:
        + INPUT_PORT_PARAMS:<dict>
        + PARAMETER_PORT_PARAMS:<dict>
   [TBI + OUTPUT_PORT_PARAMS:<dict>]
        - each dict will be passed to the corresponding Port
        - params can be any permissible executeParamSpecs for the corresponding Port
        - dicts can contain the following embedded dicts:
            + FUNCTION_PARAMS:<dict>:
                 will be passed the Port's execute method,
                     overriding its current values for that call
            + PROJECTION_PARAMS:<dict>:
                 entry will be passed to all of the Port's Projections, and used by
                 by their execute methods, overriding their current values for that call
            + MAPPING_PROJECTION_PARAMS:<dict>:
                 entry will be passed to all of the Port's MappingProjections,
                 along with any in a PROJECTION_PARAMS dict, and override current values
            + LEARNING_PROJECTION_PARAMS:<dict>:
                 entry will be passed to all of the Port's LearningProjections,
                 along with any in a PROJECTION_PARAMS dict, and override current values
            + CONTROL_PROJECTION_PARAMS:<dict>:
                 entry will be passed to all of the Port's ControlProjections,
                 along with any in a PROJECTION_PARAMS dict, and override current values
            + GATING_PROJECTION_PARAMS:<dict>:
                 entry will be passed to all of the Port's GatingProjections,
                 along with any in a PROJECTION_PARAMS dict, and override current values
            + <ProjectionName>:<dict>:
                 entry will be passed to the Port's Projection with the key's name,
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
import re
import types
import warnings

from collections import defaultdict, OrderedDict, UserDict, UserList
from inspect import isclass
from numbers import Number

import numpy as np
import typecheck as tc

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.component import Component
from psyneulink.core.components.functions.function import FunctionOutputType
from psyneulink.core.components.functions.transferfunctions import Linear
from psyneulink.core.components.shellclasses import Function, Mechanism, Projection, Port
from psyneulink.core.components.ports.inputport import DEFER_VARIABLE_SPEC_TO_MECH_MSG, InputPort
from psyneulink.core.components.ports.modulatorysignals.modulatorysignal import _is_modulatory_spec
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.components.ports.parameterport import ParameterPort
from psyneulink.core.components.ports.port import \
    REMOVE_PORTS, PORT_SPEC, _parse_port_spec, PORT_SPECIFIC_PARAMS, PROJECTION_SPECIFIC_PARAMS
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
# TODO: remove unused keywords
from psyneulink.core.globals.keywords import \
    ADDITIVE_PARAM, EXECUTION_PHASE, EXPONENT, FUNCTION, FUNCTION_PARAMS, \
    INITIALIZING, INIT_EXECUTE_METHOD_ONLY, INIT_FUNCTION_METHOD_ONLY, INPUT, \
    INPUT_LABELS_DICT, INPUT_PORT, INPUT_PORT_PARAMS, INPUT_PORTS, INPUT_PORT_VARIABLES, \
    MECHANISM, MECHANISM_VALUE, MECHANISM_COMPONENT_CATEGORY, MODEL_SPEC_ID_INPUT_PORTS, MODEL_SPEC_ID_OUTPUT_PORTS, \
    MONITOR_FOR_CONTROL, MONITOR_FOR_LEARNING, MULTIPLICATIVE_PARAM, \
    NAME, OUTPUT, OUTPUT_LABELS_DICT, OUTPUT_PORT, OUTPUT_PORT_PARAMS, OUTPUT_PORTS, OWNER_EXECUTION_COUNT, OWNER_EXECUTION_TIME, OWNER_VALUE, \
    PARAMETER_PORT, PARAMETER_PORT_PARAMS, PARAMETER_PORTS, PROJECTIONS, REFERENCE_VALUE, RESULT, \
    TARGET_LABELS_DICT, VALUE, VARIABLE, WEIGHT
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.scheduling.condition import Condition, TimeScale
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.registry import register_category, remove_instance_from_registry
from psyneulink.core.globals.utilities import \
    ContentAddressableList, append_type_to_name, convert_all_elements_to_np_array, convert_to_np_array, \
    iscompatible, kwCompatibilityNumeric

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


class MechParamsDict(UserDict):
    """Subclass for validation of dicts used to pass Mechanism parameters to OutputPort for variable specification."""
    pass


def _input_port_variables_getter(owning_component=None, context=None):
    try:
        return [input_port.parameters.variable._get(context) for input_port in owning_component.input_ports]
    except TypeError:
        return None


class Mechanism_Base(Mechanism):
    """
    Mechanism_Base(             \
        default_variable=None,  \
        size=None,              \
        input_ports,            \
        function,               \
        output_ports,           \
        )

    Base class for Mechanism.

    The arguments below can be used in the constructor for any subclass of Mechanism.
    See `Component <Component_Class_Reference>` and subclasses for additional arguments and attributes.

    .. note::
       Mechanism is an abstract class and should *never* be instantiated by a direct call to its constructor.
       Mechanisms should be instantiated by calling the constructor for the desired subclass, or using other methods
       for specifying a Mechanism in context (see `Mechanism_Creation`)

    COMMENT:
        Description
        -----------
            Mechanism is a Category of the Component class.
            A Mechanism is associated with a name and:
            - one or more input_ports:
                two ways to get multiple input_ports, if supported by Mechanism subclass being instantiated:
                    • specify 2d variable for Mechanism (i.e., without explicit InputPort specifications)
                        once the variable of the Mechanism has been converted to a 2d array, an InputPort is assigned
                        for each item of axis 0, and the corresponding item is assigned as the InputPort's variable
                    • explicitly specify input_ports in params[*INPUT_PORTS*] (each with its own variable
                        specification); those variables will be concantenated into a 2d array to create the Mechanism's
                        variable
                if both methods are used, they must generate the same sized variable for the mechanims
                ?? WHERE IS THIS CHECKED?  WHICH TAKES PRECEDENCE: InputPort SPECIFICATION (IN _instantiate_port)??
            - an execute method:
                coordinates updating of input_ports, parameter_ports (and params), execution of the function method
                implemented by the subclass, (by calling its _execute method), and updating of the OutputPorts
            - one or more parameters, each of which must be (or resolve to) a reference to a ParameterPort
                these determine the operation of the function of the Mechanism subclass being instantiated
            - one or more OutputPorts:
                the variable of each receives the corresponding item in the output of the Mechanism's function
                the value of each is passed to corresponding MappingProjections for which the Mechanism is a sender
                * Notes:
                    by default, a Mechanism has only one OutputPort, assigned to <Mechanism>.outputPort;  however:
                    if params[OUTPUT_PORTS] is a list (of names) or specification dict (of MechanismOuput Port
                    specs), <Mechanism>.output_ports (note plural) is created and contains a list of OutputPorts,
                    the first of which points to <Mechanism>.outputPort (note singular)
                [TBI * each OutputPort maintains a list of Projections for which it serves as the sender]

        Constraints
        -----------
            - the number of input_ports must correspond to the length of the variable of the Mechanism's execute method
            - the value of each InputPort must be compatible with the corresponding item in the
                variable of the Mechanism's execute method
            - the value of each ParameterPort must be compatible with the corresponding parameter of  the Mechanism's
                 execute method
            - the number of OutputPorts must correspond to the length of the output of the Mechanism's execute method,
                (self.defaults.value)
            - the value of each OutputPort must be compatible with the corresponding item of the self.value
                 (the output of the Mechanism's execute method)

        MechanismRegistry
        -----------------
            All Mechanisms are registered in MechanismRegistry, which maintains a dict for each subclass,
            a count for all instances of that type, and a dictionary of those instances
    COMMENT

    Arguments
    ---------

    default_variable : number, list or np.ndarray : default None
        specifies the input to the Mechanism to use if none is provided in a call to its `execute
        <Mechanism_Base.execute>` method; also serves as a template to specify the shape of the `variable
        <InputPort.variable>` for its `InputPorts <Mechanism_InputPorts>` and the `variable <Mechanism_Base.variable>`
        of its `function <Mechanism_Base.function>` if those are not specified.  If it is not specified, then a
        subclass-specific default is assigned (usually [[0]]).

    size : int, list or np.ndarray of ints : default None
        specifies default_variable as array(s) of zeros if **default_variable** is not passed as an argument;
        if **default_variable** is specified, it takes precedence over the specification of **size**.
        For example, the following Mechanisms are equivalent::
            my_mech = ProcessingMechanism(size = [3, 2])
            my_mech = ProcessingMechanism(default_variable = [[0, 0, 0], [0, 0]])

    input_ports : str, list, dict, or np.ndarray : default None
        specifies the InputPorts for the Mechanism; if it is not specified, a single InputPort is created
        using the value of default_variable as its `variable <InputPort.variable>`;  if more than one is specified,
        the number and, if specified, their values must be compatible with any specifications made for
        **default_variable** or **size** (see `Mechanism_InputPorts` for additional details).

    function : Function : default Linear
        specifies the function used to generate the Mechanism's `value <Mechanism_Base.value>`;
        can be a PsyNeuLink `Function` or a `UserDefinedFunction`;  it `value <Function.value>` is used to determine
        the shape of the `primary outputPort <OutputPort_Primary>` of the Mechanism.

    output_ports : str, list or np.ndarray : default None
        specifies the OutputPorts for the Mechanism; if it is not specified, a single OutputPort is created
        the `value <OutputPort.value>` of which is assigned the first item in the outermost dimension (axis 0) of the
        Mechanism's `value <Mechanism_Base.value>` (see `Mechanism_OutputPorts` for additional details).

    Attributes
    ----------

    variable : at least 2d array
        used as input to the Mechanism's `function <Mechanism_Base.function>`.  It is always at least a 2d np.array,
        with each item of axis 0 corresponding to a `value <InputPort.value>` of one of the Mechanism's `InputPorts
        <InputPort>` (in the order they are listed in its `input_ports <Mechanism_Base.input_ports>` attribute), and
        the first item (i.e., item 0) corresponding to the `value <InputPort.value>` of the `primary InputPort
        <InputPort_Primary>`.  When specified in the **variable** argument of the constructor for the Mechanism,
        it is used as a template to define the format (shape and type of elements) of the input the Mechanism's
        `function <Mechanism_Base.function>`.

    input_port : InputPort
        `primary InputPort <InputPort_Primary>` for the Mechanism;  same as first entry of its `input_ports
        <Mechanism_Base.input_ports>` attribute.  Its `value <InputPort.value>` is assigned as the first item of the
        Mechanism's `variable <Mechanism_Base.variable>`.

    input_ports : ContentAddressableList[str, InputPort]
        a list of the Mechanism's `InputPorts <Mechanism_InputPorts>`. The first (and possibly only) entry is always
        the Mechanism's `primary InputPort <InputPort_Primary>` (i.e., the one in the its `input_port
        <Mechanism_Base.input_port>` attribute).

    input_values : List[List or 1d np.array]
        each item in the list corresponds to the `value <InputPort.value>` of one of the Mechanism's `InputPorts
        <Mechanism_InputPorts>` listed in its `input_ports <Mechanism_Base.input_ports>` attribute.  The value of
        each item is the same as the corresponding item in the Mechanism's `variable <Mechanism_Base.variable>`
        attribute.  The latter is a 2d np.array; the `input_values <Mechanism_Base.input_values>` attribute provides
        this information in a simpler list format.

    input_labels_dict : dict
        contains entries that are either label:value pairs, or sub-dictionaries containing label:value pairs,
        in which each label (key) specifies a string associated with a value for the InputPort(s) of the
        Mechanism; see `Mechanism_Labels_Dicts` for additional details.

    input_labels : list[str]
        contains the labels corresponding to the value(s) of the InputPort(s) of the Mechanism. If the current value
        of an InputPort does not have a corresponding label, then its numeric value is used instead.

    external_input_ports : list[InputPort]
        list of the `input_ports <Mechanism_Base.input_ports>` for the Mechanism that are not designated as
        `internal_only <InputPort.internal_only>`;  these receive `inputs from a Composition
        <Composition_Execution_Inputs>` if the Mechanism is one of its `INPUT` `Nodes <Composition_Nodes>`.

    external_input_values : List[List or 1d np.array]
        list of the `value <InputPort.value>`\\s of the Mechanism's `external_input_ports
        <Mechanism_Base.external_input_ports>`.

    COMMENT:
    target_labels_dict : dict
        contains entries that are either label:value pairs, or sub-dictionaries containing label:value pairs,
        in which each label (key) specifies a string associated with a value for the InputPort(s) of the
        Mechanism if it is the `TARGET` Mechanism for a Composition; see `Mechanism_Labels_Dicts` and
        `target mechanism <LearningMechanism_Targets>` for additional details.
    COMMENT

    parameter_ports : ContentAddressableList[str, ParameterPort]
        a read-only list of the Mechanism's `ParameterPorts <Mechanism_ParameterPorts>`, one for each of its
        `modulable parameters <ParameterPort_Modulable_Parameters>`, including those of its `function
        <Mechanism_Base.function>`.  The value of the parameters of the Mechanism and its `function
        <Mechanism_Base.function>` are also accessible as (and can be modified using) attributes of the Mechanism
        (see `Mechanism_ParameterPorts`).

    function : Function, function or method
        the primary function for the Mechanism, called when it is `executed <Mechanism_Execution>`.  It takes the
        Mechanism's `variable <Mechanism_Base.variable>` attribute as its input, and its result is assigned to the
        Mechanism's `value <Mechanism_Base.value>` attribute (see `Component_Function` for additional details).

    function_params : Dict[str, value]
        contains the parameters for the Mechanism's `function <Mechanism_Base.function>`.  The key of each entry is the
        name of a parameter of the function, and its value is the parameter's value.

    value : 2d np.array [array(float64)]
        result of the Mechanism's `function <Mechanism_Base.function>`.  It is always at least a 2d np.array, with the
        items of axis 0 corresponding to the values referenced by the corresponding `index <OutputPort.index>`
        attribute of the Mechanism's `OutputPorts <OutputPort>`.  The first item is generally referenced by the
        Mechanism's `primary OutputPort <OutputPort_Primary>` (i.e., the one in the its `output_port
        <Mechanism_Base.output_port>` attribute), as well as the first item of `output_values
        <Mechanism_Base.output_values>`.  The `value <Mechanism_Base.value>` is `None` until the Mechanism
        has been executed at least once.

        .. note::
           the `value <Mechanism_Base.value>` of a Mechanism is not necessarily the same as its
           `output_values <Mechanism_Base.output_values>` attribute, which lists the `values <OutputPort.value>`
           of its `OutputPorts <Mechanism_Base.output_ports>`.

    output_port : OutputPort
        `primary OutputPort <OutputPort_Primary>` for the Mechanism;  same as first entry of its `output_ports
        <Mechanism_Base.output_ports>` attribute.

    output_ports : ContentAddressableList[str, OutputPort]
        list of the Mechanism's `OutputPorts <Mechanism_OutputPorts>`. The first (and possibly only) entry is always
        the Mechanism's `primary OutputPort <OutputPort_Primary>` (i.e., the one in the its `output_port
        <Mechanism_Base.output_port>` attribute).

    output_values : List[array(float64)]
        each item in the list corresponds to the `value <OutputPort.value>` of one of the Mechanism's `OutputPorts
        <Mechanism_OutputPorts>` listed in its `output_ports <Mechanism_Base.output_ports>` attribute.

        .. note:: The `output_values <Mechanism_Base.output_values>` of a Mechanism is not necessarily the same as its
                  `value <Mechanism_Base.value>` attribute, since an OutputPort's
                  `function <OutputPort.OutputPort.function>` and/or its `assign <Mechanism_Base.assign>`
                  attribute may use the Mechanism's `value <Mechanism_Base.value>` to generate a derived quantity for
                  the `value <OutputPort.OutputPort.value>` of that OutputPort (and its corresponding item in the
                  the Mechanism's `output_values <Mechanism_Base.output_values>` attribute).

    output_labels_dict : dict
        contains entries that are either label:value pairs, or sub-dictionaries containing label:value pairs,
        in which each label (key) specifies a string associated with a value for the OutputPort(s) of the
        Mechanism; see `Mechanism_Labels_Dicts` for additional details.

    output_labels : list
        contains the labels corresponding to the value(s) of the OutputPort(s) of the Mechanism. If the current value
        of an OutputPort does not have a corresponding label, then its numeric value is used instead.

    standard_output_ports : list[dict]
        list of the dictionary specifications for `StandardOutputPorts <OutputPort_Standard>` that can be assigned as
        `OutputPorts <OutputPort>`; subclasses may extend this list to include additional ones.

        *RESULT* : 1d np.array
          first item in the outermost dimension (axis 0) of the Mechanism's `value <Mechanism_Base.value>`.

        *OWNER_VALUE* : list
          Full ndarray of Mechanism's `value <Mechanism_Base.value>`.

        *MECHANISM_VALUE* : list
          Synonym for *OWNER_VALUE*.

    standard_output_port_names : list[str]
        list of the names of the `standard_output_ports <Mechanism_Base.standard_output_ports>` that can be used to
        specify a `StandardOutputPort <OutputPort_Standard>` in the **output_ports** argument of the Mechanism's
        constructor, and to reference it in Mechanism's list of `output_ports <Mechanism_Base.output_ports>`.

    ports : ContentAddressableList
        a list of all of the Mechanism's `Ports <Port>`, composed from its `input_ports
        <Mechanism_Base.input_ports>`, `parameter_ports <Mechanism_Base.parameter_ports>`, and
        `output_ports <Mechanism_Base.output_ports>` attributes.

    projections : ContentAddressableList
        a list of all of the Mechanism's `Projections <Projection>`, composed from the
        `path_afferents <InputPorts.path_afferents>` of all of its `input_ports <Mechanism_Base.input_ports>`,
        the `mod_afferents` of all of its `input_ports <Mechanism_Base.input_ports>`,
        `parameter_ports <Mechanism)Base.parameter_ports>`, and `output_ports <Mechanism_Base.output_ports>`,
        and the `efferents <Port.efferents>` of all of its `output_ports <Mechanism_Base.output_ports>`.

    afferents : ContentAddressableList
        a list of all of the Mechanism's afferent `Projections <Projection>`, composed from the
        `path_afferents <InputPorts.path_afferents>` of all of its `input_ports <Mechanism_Base.input_ports>`,
        and the `mod_afferents` of all of its `input_ports <Mechanism_Base.input_ports>`,
        `parameter_ports <Mechanism)Base.parameter_ports>`, and `output_ports <Mechanism_Base.output_ports>`.,

    path_afferents : ContentAddressableList
        a list of all of the Mechanism's afferent `PathwayProjections <PathwayProjection>`, composed from the
        `path_afferents <InputPorts.path_afferents>` attributes of all of its `input_ports
        <Mechanism_Base.input_ports>`.

    mod_afferents : ContentAddressableList
        a list of all of the Mechanism's afferent `ModulatoryProjections <ModulatoryProjection>`, composed from the
        `mod_afferents` attributes of all of its `input_ports <Mechanism_Base.input_ports>`, `parameter_ports
        <Mechanism)Base.parameter_ports>`, and `output_ports <Mechanism_Base.output_ports>`.

    efferents : ContentAddressableList
        a list of all of the Mechanism's efferent `Projections <Projection>`, composed from the `efferents
        <Port.efferents>` attributes of all of its `output_ports <Mechanism_Base.output_ports>`.

    senders : ContentAddressableList
        a list of all of the Mechanisms that send `Projections <Projection>` to the Mechanism (i.e., the senders of
        its `afferents <Mechanism_Base.afferents>`; this includes both `ProcessingMechanisms <ProcessingMechanism>`
        (that send `MappingProjections <MappingProjection>` and `ModulatoryMechanisms <ModulatoryMechanism>` (that send
        `ModulatoryProjections <ModulatoryProjection>` (also see `modulators <Mechanism_Base.modulators>`).

    modulators : ContentAddressableList
        a list of all of the `AdapativeMechanisms <ModulatoryMechanism>` that send `ModulatoryProjections
        <ModulatoryProjection>` to the Mechanism (i.e., the senders of its `mod_afferents
        <Mechanism_Base.mod_afferents>` (also see `senders <Mechanism_Base.senders>`).

    receivers : ContentAddressableList
        a list of all of the Mechanisms that receive `Projections <Projection>` from the Mechanism (i.e.,
        the receivers of its `efferents <Mechanism_Base.efferents>`.

    condition : Condition : None
        condition to be associated with the Mechanism in the `Scheduler` responsible for executing it in each
        `Composition` to which it is assigned;  if it is not specified (i.e., its value is `None`), the default
        Condition for a `Component` is used.  It can be overridden in a given `Composition` by assigning a Condition
        for the Mechanism directly to a Scheduler that is then assigned to the Composition.

    name : str
        the name of the Mechanism; if it is not specified in the **name** argument of the constructor, a default is
        assigned by MechanismRegistry (see `Registry_Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the Mechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see `Preferences`
        for details).
        .. _portRegistry : Registry
               registry containing dicts for each Port type (InputPort, OutputPort and ParameterPort) with instance
               dicts for the instances of each type and an instance count for each Port type in the Mechanism.
               Note: registering instances of Port types with the Mechanism (rather than in the PortRegistry)
                     allows the same name to be used for instances of a Port type belonging to different Mechanisms
                     without adding index suffixes for that name across Mechanisms
                     while still indexing multiple uses of the same base name within a Mechanism.
    """

    # CLASS ATTRIBUTES
    componentCategory = MECHANISM_COMPONENT_CATEGORY
    className = componentCategory
    suffix = " " + className

    registry = MechanismRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY
    # Any preferences specified below will override those specified in CategoryDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to CATEGORY automatically
    # classPreferences = {
    #     PREFERENCE_SET_NAME: 'MechanismCustomClassPreferences',
    #     PREFERENCE_KEYWORD<pref>: <setting>...}

    # Class-specific loggable items
    @property
    def _loggable_items(self):
        # Ports, afferent Projections are loggable for a Mechanism
        #     - this allows the value of InputPorts and OutputPorts to be logged
        #     - for MappingProjections, this logs the value of the Projection's matrix parameter
        #     - for ModulatoryProjections, this logs the value of the Projection
        # IMPLEMENTATION NOTE: this needs to be a property as Projections may be added after instantiation
        try:
            # return list(self.ports) + list(self.afferents)
            return list(self.ports)
        except:
            return []

    #FIX:  WHEN CALLED BY COMPOSITION, SHOULD USE FULL Mechanism.execute
    # By default, init only the _execute method of Mechanism subclass objects when their execute method is called;
    #    that is, DO NOT run the full Mechanism execute Process, since some components may not yet be instantiated
    #    (such as OutputPorts)
    initMethod = INIT_EXECUTE_METHOD_ONLY

    # Note:  the following enforce encoding as 2D np.ndarrays,
    #        to accomodate multiple Ports:  one 1D np.ndarray per port
    variableEncodingDim = 2
    valueEncodingDim = 2

    portListAttr = {InputPort:INPUT_PORTS,
                     ParameterPort:PARAMETER_PORTS,
                     OutputPort:OUTPUT_PORTS}

    # Category specific defaults:

    standard_output_ports = [{NAME: RESULT},
                             {NAME: MECHANISM_VALUE,
                              VARIABLE: OWNER_VALUE},
                             {NAME: OWNER_VALUE,
                              VARIABLE: OWNER_VALUE}]
    standard_output_port_names = [i['name'] for i in standard_output_ports]

    class Parameters(Mechanism.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <Mechanism_Base.variable>`

                    :default value: numpy.array([[0]])
                    :type: ``numpy.ndarray``
                    :read only: True

                value
                    see `value <Mechanism_Base.value>`

                    :default value: numpy.array([[0]])
                    :type: ``numpy.ndarray``
                    :read only: True

                function
                    see `function <Mechanism_Base.function>`

                    :default value: `Linear`
                    :type: `Function`

                input_labels_dict
                    see `input_labels_dict <Mechanism_Base.input_labels_dict>`

                    :default value: {}
                    :type: <class 'dict'>

                input_ports
                    see `input_ports <Mechanism_Base.input_ports>`

                    :default value: None
                    :type:
                    :read only: True

                output_labels_dict
                    see `output_labels_dict <Mechanism_Base.output_labels_dict>`

                    :default value: {}
                    :type: <class 'dict'>

                output_ports
                    see `output_ports <Mechanism_Base.output_ports>`

                    :default value: None
                    :type:
                    :read only: True
        """
        variable = Parameter(np.array([[0]]),
                             read_only=True, pnl_internal=True,
                             constructor_argument='default_variable')
        value = Parameter(np.array([[0]]), read_only=True, pnl_internal=True)
        function = Parameter(Linear, stateful=False, loggable=False)

        input_port_variables = Parameter(None, read_only=True, user=False,
                                         getter=_input_port_variables_getter,
                                         pnl_internal=True)
        input_labels_dict = Parameter(
            {},
            stateful=False,
            loggable=False,
            pnl_internal=True
        )
        output_labels_dict = Parameter(
            {},
            stateful=False,
            loggable=False,
            pnl_internal=True
        )
        input_ports = Parameter(
            None,
            stateful=False,
            loggable=False,
            read_only=True,
            structural=True,
        )
        output_ports = Parameter(
            None,
            stateful=False,
            loggable=False,
            read_only=True,
            structural=True,
        )

        def _parse_input_ports(self, input_ports):
            if input_ports is None:
                return input_ports
            elif not isinstance(input_ports, list):
                input_ports = [input_ports]

            spec_list = []

            for port in input_ports:
                # handle tuple specification only because it cannot
                # be translated to and from JSON (converts to list, which is
                # not accepted as a valid specification)
                if isinstance(port, tuple):
                    if len(port) == 2:
                        # allows case [(transfer_mech, None)] in
                        # TestInputPortSpec
                        if not isinstance(port[0], str):
                            # no parsing
                            spec = port
                        else:
                            spec = {
                                NAME: port[0],
                                MECHANISM: port[1]
                            }
                    elif len(port) > 2:
                        # nonstandard 4 item tuple
                        if isinstance(port[0], (list, tuple)):
                            spec = port
                        else:
                            # if port is assigned to an object,
                            # use a reference instead of name/value
                            if isinstance(port[0], Component):
                                spec = {PORT_SPEC: port[0]}

                            else:
                                spec = {
                                    NAME: port[0].name,
                                    VALUE: port[0].defaults.value,
                                }

                            spec[WEIGHT] = port[1]
                            spec[EXPONENT] = port[2]

                            try:
                                spec[PROJECTIONS] = port[3]
                            except IndexError:
                                pass

                    spec_list.append(spec)
                else:
                    spec_list.append(port)

            return spec_list

        def _parse_output_ports(self, output_ports):
            if output_ports is not None and not isinstance(output_ports, list):
                return [output_ports]
            else:
                return output_ports

    # def __new__(cls, *args, **kwargs):
    # def __new__(cls, name=NotImplemented, params=NotImplemented, context=None):

    @tc.typecheck
    @abc.abstractmethod
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_ports=None,
                 function=None,
                 output_ports=None,
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
            to validate size and default_variable and param_defaults;
            it uses INPUT_PORT as the default_variable
        * registers Mechanism with MechanismRegistry

        """

        # IMPLEMENT **kwargs (PER Port)

        self.aux_components = []
        self.monitor_for_learning = None
        # Register with MechanismRegistry or create one
        register_category(entry=self,
                          base_class=Mechanism_Base,
                          name=name,
                          registry=MechanismRegistry,
                          context=context)

        # Create Mechanism's _portRegistry and port type entries
        from psyneulink.core.components.ports.port import Port_Base
        self._portRegistry = {}

        # InputPort
        from psyneulink.core.components.ports.inputport import InputPort
        register_category(entry=InputPort,
                          base_class=Port_Base,
                          registry=self._portRegistry,
                          context=context)

        # ParameterPort
        from psyneulink.core.components.ports.parameterport import ParameterPort
        register_category(entry=ParameterPort,
                          base_class=Port_Base,
                          registry=self._portRegistry,
                          context=context)

        # OutputPort
        from psyneulink.core.components.ports.outputport import OutputPort
        register_category(entry=OutputPort,
                          base_class=Port_Base,
                          registry=self._portRegistry,
                          context=context)

        super(Mechanism_Base, self).__init__(
            default_variable=default_variable,
            size=size,
            function=function,
            param_defaults=params,
            prefs=prefs,
            name=name,
            input_ports=input_ports,
            output_ports=output_ports,
            **kwargs
        )

        # FIX: 10/3/17 - IS THIS CORRECT?  SHOULD IT BE INITIALIZED??
        self._status = INITIALIZING
        self._receivesProcessInput = False

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

    def _handle_default_variable(self, default_variable=None, size=None, input_ports=None, function=None, params=None):
        """
            Finds whether default_variable can be determined using **default_variable** and **size**
            arguments.

            Returns
            -------
                a default variable if possible
                None otherwise
        """
        default_variable_from_input_ports = None

        # handle specifying through params dictionary
        try:
            default_variable_from_input_ports, input_ports_variable_was_specified = \
                self._handle_arg_input_ports(params[INPUT_PORTS])
        except (TypeError, KeyError):
            pass
        except AttributeError as e:
            if DEFER_VARIABLE_SPEC_TO_MECH_MSG in e.args[0]:
                pass

        if default_variable_from_input_ports is None:
            # fallback to standard arg specification
            try:
                default_variable_from_input_ports, input_ports_variable_was_specified = \
                    self._handle_arg_input_ports(input_ports)
            except AttributeError as e:
                if DEFER_VARIABLE_SPEC_TO_MECH_MSG in e.args[0]:
                    pass

        if default_variable_from_input_ports is not None:
            if default_variable is None:
                if size is None:
                    default_variable = default_variable_from_input_ports
                else:
                    if input_ports_variable_was_specified:
                        size_variable = self._handle_size(size, None)
                        if iscompatible(size_variable, default_variable_from_input_ports):
                            default_variable = default_variable_from_input_ports
                        else:
                            raise MechanismError(
                                'default variable determined from the specified input_ports spec ({0}) '
                                'is not compatible with the default variable determined from size parameter ({1})'.
                                    format(default_variable_from_input_ports, size_variable,
                                )
                            )
                    else:
                        # do not pass input_ports variable as default_variable, fall back to size specification
                        pass
            else:
                if input_ports_variable_was_specified:
                    if not iscompatible(self._parse_arg_variable(default_variable), default_variable_from_input_ports):
                        raise MechanismError(
                            'Default variable determined from the specified input_ports spec ({0}) for {1} '
                            'is not compatible with its specified default variable ({2})'.format(
                                default_variable_from_input_ports, self.name, default_variable
                            )
                        )
                else:
                    # do not pass input_ports variable as default_variable, fall back to default_variable specification
                    pass

        return super()._handle_default_variable(default_variable=default_variable, size=size)

    def _handle_arg_input_ports(self, input_ports):
        """
        Takes user-inputted argument **input_ports** and returns an defaults.variable-like
        object that it represents

        Returns
        -------
            A, B where
            A is an defaults.variable-like object
            B is True if **input_ports** contained an explicit variable specification, False otherwise
        """

        if input_ports is None:
            return None, False

        default_variable_from_input_ports = []
        input_port_variable_was_specified = None

        if not isinstance(input_ports, list):
            input_ports = [input_ports]

        for i, s in enumerate(input_ports):


            try:
                parsed_input_port_spec = _parse_port_spec(owner=self,
                                                            port_type=InputPort,
                                                            port_spec=s,
                                                            )
            except AttributeError as e:
                if DEFER_VARIABLE_SPEC_TO_MECH_MSG in e.args[0]:
                    default_variable_from_input_ports.append(InputPort.defaults.variable)
                    continue
                else:
                    raise MechanismError("PROGRAM ERROR: Problem parsing {} specification ({}) for {}".
                                         format(InputPort.__name__, s, self.name))

            mech_variable_item = None

            if isinstance(parsed_input_port_spec, dict):
                try:
                    mech_variable_item = parsed_input_port_spec[VALUE]
                    if parsed_input_port_spec[VARIABLE] is None:
                        input_port_variable_was_specified = False
                except KeyError:
                    pass
            elif isinstance(parsed_input_port_spec, (Projection, Mechanism, Port)):
                if parsed_input_port_spec.initialization_status == ContextFlags.DEFERRED_INIT:
                    args = parsed_input_port_spec._init_args
                    if REFERENCE_VALUE in args and args[REFERENCE_VALUE] is not None:
                        mech_variable_item = args[REFERENCE_VALUE]
                    elif VALUE in args and args[VALUE] is not None:
                        mech_variable_item = args[VALUE]
                    elif VARIABLE in args and args[VARIABLE] is not None:
                        mech_variable_item = args[VARIABLE]
                else:
                    try:
                        mech_variable_item = parsed_input_port_spec.value
                    except AttributeError:
                        mech_variable_item = parsed_input_port_spec.defaults.mech_variable_item
            else:
                mech_variable_item = parsed_input_port_spec.defaults.mech_variable_item

            if mech_variable_item is None:
                mech_variable_item = InputPort.defaults.variable
            elif input_port_variable_was_specified is None and not InputPort._port_spec_allows_override_variable(s):
                input_port_variable_was_specified = True

            default_variable_from_input_ports.append(mech_variable_item)

        return default_variable_from_input_ports, input_port_variable_was_specified

    # ------------------------------------------------------------------------------------------------------------------
    # Validation methods
    # ------------------------------------------------------------------------------------------------------------------

    def _validate_variable(self, variable, context=None):
        """Convert variable to 2D np.array: one 1D value for each InputPort

        # VARIABLE SPECIFICATION:                                        ENCODING:
        # Simple value variable:                                         0 -> [array([0])]
        # Single port array (vector) variable:                         [0, 1] -> [array([0, 1])
        # Multiple port variables, each with a single value variable:  [[0], [0]] -> [array[0], array[0]]

        :param variable:
        :param context:
        :return:
        """

        variable = super(Mechanism_Base, self)._validate_variable(variable, context)

        # Force Mechanism variable specification to be a 2D array (to accomodate multiple InputPorts - see above):
        variable = convert_to_np_array(variable, 2)

        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """validate TimeScale, INPUT_PORTS, FUNCTION_PARAMS, OUTPUT_PORTS and MONITOR_FOR_CONTROL

        Go through target_set params (populated by Component._validate_params) and validate values for:
            + INPUT_PORTS:
                <MechanismsInputPort or Projection object or class,
                specification dict for one, 2-item tuple, or numeric value(s)>;
                if it is missing or not one of the above types, it is set to self.defaults.variable
            + FUNCTION_PARAMS:  <dict>, every entry of which must be one of the following:
                ParameterPort or Projection object or class, specification dict for one, 2-item tuple, or numeric
                value(s);
                if invalid, default is assigned
            + OUTPUT_PORTS:
                <MechanismsOutputPort object or class, specification dict, or numeric value(s);
                if it is missing or not one of the above types, it is set to None here;
                    and then to default value of value (output of execute method) in instantiate_output_port
                    (since execute method must be instantiated before self.defaults.value is known)
                if OUTPUT_PORTS is a list or OrderedDict, it is passed along (to instantiate_output_ports)
                if it is a OutputPort class ref, object or specification dict, it is placed in a list
            + MONITORED_PORTS:
                ** DOCUMENT

        Note: PARAMETER_PORTS are validated separately -- ** DOCUMENT WHY

        TBI - Generalize to go through all params, reading from each its type (from a registry),
                                   and calling on corresponding subclass to get default values (if param not found)
                                   (as PROJECTION_TYPE and PROJECTION_SENDER are currently handled)
        """

        from psyneulink.core.components.ports.port import _parse_port_spec
        from psyneulink.core.components.ports.inputport import InputPort
        from psyneulink.core.components.ports.outputport import OutputPort

        # Perform first-pass validation in Function.__init__():
        super(Mechanism, self)._validate_params(request_set,target_set,context)

        params = target_set

        # VALIDATE InputPort(S)

        # INPUT_PORTS is specified, so validate:
        if INPUT_PORTS in params and params[INPUT_PORTS] is not None:
            try:
                try:
                    for port_spec in params[INPUT_PORTS]:
                        _parse_port_spec(owner=self, port_type=InputPort, port_spec=port_spec)
                except TypeError:
                    _parse_port_spec(owner=self, port_type=InputPort, port_spec=params[INPUT_PORTS])
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

            from psyneulink.core.components.ports.parameterport import ParameterPort
            for param_name, param_value in function_param_specs.items():
                if not ((isclass(param_value) and
                             (issubclass(param_value, ParameterPort) or
                              issubclass(param_value, Projection))) or
                        isinstance(param_value, ParameterPort) or
                        isinstance(param_value, Projection) or
                        isinstance(param_value, dict) or
                        iscompatible(param_value, self.defaults.value)):
                    params[FUNCTION_PARAMS][param_name] = self.defaults.value
                    if self.prefs.verbosePref:
                        print("{0} param ({1}) for execute method {2} of {3} is not a ParameterPort, "
                              "projection, tuple, or value; default value ({4}) will be used".
                              format(param_name,
                                     param_value,
                                     self.execute.__self__.componentName,
                                     self.__class__.__name__,
                                     self.defaults.value))

        # VALIDATE OUTPUTPORT(S)

        # OUTPUT_PORTS is specified, so validate:
        if OUTPUT_PORTS in params and params[OUTPUT_PORTS] is not None:

            param_value = params[OUTPUT_PORTS]

            # If it is a single item or a non-OrderedDict, place in list (for use here and in instantiate_output_port)
            if not isinstance(param_value, (ContentAddressableList, list, OrderedDict)):
                param_value = [param_value]
            # Validate each item in the list or OrderedDict
            i = 0
            for key, item in param_value if isinstance(param_value, dict) else enumerate(param_value):
                # If not valid...
                if not ((isclass(item) and issubclass(item, OutputPort)) or  # OutputPort class ref
                        isinstance(item, OutputPort) or  # OutputPort object
                        isinstance(item, dict) or  # OutputPort specification dict
                        isinstance(item, str) or  # Name (to be used as key in OutputPorts list)
                        isinstance(item, tuple) or  # Projection specification tuple
                        _is_modulatory_spec(item) or  # Modulatory specification for the OutputPort
                        iscompatible(item, **{kwCompatibilityNumeric: True})):  # value
                    # set to None, so it is set to default (self.value) in instantiate_output_port
                    param_value[key] = None
                    if self.prefs.verbosePref:
                        print("Item {0} of {1} param ({2}) in {3} is not a"
                              " OutputPort, specification dict or value, nor a list of dict of them; "
                              "output ({4}) of execute method for {5} will be used"
                              " to create a default OutputPort for {3}".
                              format(i,
                                     OUTPUT_PORTS,
                                     param_value,
                                     self.__class__.__name__,
                                     self.value,
                                     self.execute.__self__.name))
                i += 1
            params[OUTPUT_PORTS] = param_value

        def validate_labels_dict(lablel_dict, type):
            for label, value in labels_dict.items():
                # KDM 11/26/19: allowed ints and dicts because they are
                # expected in test_3_input_ports_2_label_dicts
                if not isinstance(label, (str, int)):
                    raise MechanismError("Key ({}) in the {} for {} must be a string or int ".
                                         format(label, type, self.name))
                if not isinstance(value, (list, np.ndarray, dict, int)):
                    raise MechanismError("The value of {} ({}) in the {} for {} must be a list, dict, or array".
                                         format(label, value, type, self.name))
        def validate_subdict_key(port_type, key, dict_type):
            # IMPLEMENTATION NOTE:
            #    can't yet validate that string is a legit InputPort name or that index is within
            #    bounds of the number of InputPorts;  that is done in _get_port_value_labels()
            if not isinstance(key, (int, str)):
                raise MechanismError("Key ({}) for {} of {} must the name of an {} or the index for one".
                                     format(key, dict_type, self.name, port_type.__name__))

        if INPUT_LABELS_DICT in params and params[INPUT_LABELS_DICT]:
            labels_dict = params[INPUT_LABELS_DICT]
            if isinstance(list(labels_dict.values())[0], dict):
                for subdict in labels_dict.values():
                    for key, ld in subdict.items():
                        validate_subdict_key(InputPort, key, INPUT_LABELS_DICT)
                        validate_labels_dict(ld, INPUT_LABELS_DICT)
            else:
                validate_labels_dict(labels_dict, INPUT_LABELS_DICT)

        if OUTPUT_LABELS_DICT in params and params[OUTPUT_LABELS_DICT]:
            labels_dict = params[OUTPUT_LABELS_DICT]
            if isinstance(list(labels_dict.values())[0], dict):
                for key, ld in labels_dict.values():
                    validate_subdict_key(OutputPort, key, OUTPUT_LABELS_DICT)
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
        self._instantiate_input_ports(context=context)
        self._instantiate_parameter_ports(function=function, context=context)
        super()._instantiate_attributes_before_function(function=function, context=context)

    def _instantiate_function(self, function, function_params=None, context=None):
        """Assign weights and exponents if specified in input_ports
        """

        super()._instantiate_function(function=function, function_params=function_params, context=context)

        if self.input_ports and any(input_port.weight is not None for input_port in self.input_ports):

            # Construct defaults:
            #    from function.weights if specified else 1's
            try:
                default_weights = self.function.weights
            except AttributeError:
                default_weights = None
            if default_weights is None:
                default_weights = default_weights or [1.0] * len(self.input_ports)

            # Assign any weights specified in input_port spec
            weights = [[input_port.weight if input_port.weight is not None else default_weight]
                       for input_port, default_weight in zip(self.input_ports, default_weights)]
            self.function._weights = weights

        if self.input_ports and any(input_port.exponent is not None for input_port in self.input_ports):

            # Construct defaults:
            #    from function.weights if specified else 1's
            try:
                default_exponents = self.function.exponents
            except AttributeError:
                default_exponents = None
            if default_exponents is None:
                default_exponents = default_exponents or [1.0] * len(self.input_ports)

            # Assign any exponents specified in input_port spec
            exponents = [[input_port.exponent if input_port.exponent is not None else default_exponent]
                       for input_port, default_exponent in zip(self.input_ports, default_exponents)]
            self.function._exponents = exponents

        # this may be removed when the restriction making all Mechanism values 2D np arrays is lifted
        # ignore warnings of certain Functions that disable conversion
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=UserWarning)
            self.function.output_type = FunctionOutputType.NP_2D_ARRAY
            self.function.enable_output_type_conversion = True

        self.function._instantiate_value(context)

    def _instantiate_attributes_after_function(self, context=None):
        from psyneulink.core.components.ports.parameterport import _instantiate_parameter_port

        self._instantiate_output_ports(context=context)
        # instantiate parameter ports from UDF custom parameters if necessary
        try:
            for param_name, param_value in self.function.cust_fct_params.items():
                if param_name not in self.parameter_ports.names:
                    _instantiate_parameter_port(
                        self,
                        param_name,
                        param_value,
                        context=context,
                        function=self.function
                    )
            self._parse_param_port_sources()
        except AttributeError:
            pass

        super()._instantiate_attributes_after_function(context=context)

    def _instantiate_input_ports(self, input_ports=None, reference_value=None, context=None):
        """Call Port._instantiate_input_ports to instantiate orderedDict of InputPort(s)

        This is a stub, implemented to allow Mechanism subclasses to override _instantiate_input_ports
            or process InputPorts before and/or after call to _instantiate_input_ports
        """
        from psyneulink.core.components.ports.inputport import _instantiate_input_ports
        return _instantiate_input_ports(owner=self,
                                         input_ports=input_ports or self.input_ports,
                                         reference_value=reference_value,
                                         context=context)

    def _instantiate_parameter_ports(self, function=None, context=None):
        """Call Port._instantiate_parameter_ports to instantiate a ParameterPort
        for each parameter with modulable=True

        This is a stub, implemented to allow Mechanism subclasses to override _instantiate_parameter_ports
            or process InputPorts before and/or after call to _instantiate_parameter_ports
            :param function:
        """
        from psyneulink.core.components.ports.parameterport import _instantiate_parameter_ports
        _instantiate_parameter_ports(owner=self, function=function, context=context)

    def _instantiate_output_ports(self, context=None):
        """Call Port._instantiate_output_ports to instantiate orderedDict of OutputPort(s)

        This is a stub, implemented to allow Mechanism subclasses to override _instantiate_output_ports
            or process InputPorts before and/or after call to _instantiate_output_ports
        """
        from psyneulink.core.components.ports.outputport import _instantiate_output_ports
        _instantiate_output_ports(owner=self, output_ports=self.output_ports, context=context)

    def _add_projection_to_mechanism(self, port, projection, context=None):
        from psyneulink.core.components.projections.projection import _add_projection_to
        _add_projection_to(receiver=self, port=port, projection_spec=projection, context=context)

    def _add_projection_from_mechanism(self, receiver, port, projection, context=None):
        """Add projection to specified port
        """
        from psyneulink.core.components.projections.projection import _add_projection_from
        _add_projection_from(sender=self, port=port, projection_spec=projection, receiver=receiver, context=context)

    def _projection_added(self, projection, context=None):
        """Stub that can be overidden by subclasses that need to know when a projection is added to the Mechanism"""
        pass

    @handle_external_context(execution_id=NotImplemented)
    def reset(self, *args, force=False, context=None):
        """Reset `value <Mechanism_Base.value>` if Mechanisms is stateful.

        If the mechanism's `function <Mechanism.function>` is an `IntegratorFunction`, or if the mechanism has and
        `integrator_function <TransferMechanism.integrator_function>` (see `TransferMechanism`), this method
        effectively begins the function's accumulation over again at the specified value, and updates related
        attributes on the mechanism.  It also clears the
        `value <Mechanism.value>` `history <Parameter.history`, thus
        effectively setting the previous value to ``None``.

        If the mechanism's `function <Mechanism_Base.function>` is an `IntegratorFunction`, its `reset
        <Mechanism_Base.reset>` method:

            (1) Calls the function's own `reset <IntegratorFunction.reset>` method (see Note below for
                details)

            (2) Sets the mechanism's `value <Mechanism_Base.value>` to the output of the function's
                reset method

            (3) Updates its `output ports <Mechanism_Base.output_port>` based on its new `value
                <Mechanism_Base.value>`

        If the mechanism has an `integrator_function <TransferMechanism.integrator_function>`, its `reset
        <Mechanism_Base.reset>` method::

            (1) Calls the `integrator_function's <TransferMechanism.integrator_function>` own `reset
                <IntegratorFunction.reset>` method (see Note below for details)

            (2) Executes its `function <Mechanism_Base.function>` using the output of the `integrator_function's
                <TransferMechanism.integrator_function>` `reset <IntegratorFunction.reset>` method as
                the function's variable

            (3) Sets the mechanism's `value <Mechanism_Base.value>` to the output of its function

            (4) Updates its `output ports <Mechanism_Base.output_port>` based on its new `value
                <Mechanism_Base.value>`

        .. note::
                The reset method of an IntegratorFunction Function typically resets the function's
                `previous_value <IntegratorFunction.previous_value>` (and any other `stateful_attributes
                <IntegratorFunction.stateful_attributes>`) and `value <IntegratorFunction.value>` to the quantity (or
                quantities) specified. If `reset <Mechanism_Base.reset>` is called without arguments,
                the `initializer <IntegratorFunction.initializer>` value (or the values of each of the attributes in
                `initializers <IntegratorFunction.initializers>`) is used instead. The `reset
                <IntegratorFunction.reset>` method may vary across different Integrators. See individual
                functions for details on their `stateful_attributes <IntegratorFunction.stateful_attributes>`,
                as well as other reinitialization steps that the reset method may carry out.
        """
        from psyneulink.core.components.functions.statefulfunctions.statefulfunction import StatefulFunction
        from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import IntegratorFunction

        if context.execution_id is NotImplemented:
            context.execution_id = self.most_recent_context.execution_id

        # If the primary function of the mechanism is stateful:
        # (1) reset it, (2) update value, (3) update output ports
        if isinstance(self.function, StatefulFunction):
            new_value = self.function.reset(*args, context=context)
            self.parameters.value._set(np.atleast_2d(new_value), context=context)
            self._update_output_ports(context=context)

        # If the mechanism has an auxiliary integrator function:
        # (1) reset it, (2) run the primary function with the new "previous_value" as input
        # (3) update value, (4) update output ports
        elif hasattr(self, "integrator_function"):
            if not isinstance(self.integrator_function, IntegratorFunction):
                raise MechanismError(
                    f"Resetting '{self.name}' is not allowed because its integrator_function "
                    f"is not an IntegratorFunction type function, therefore the Mechanism "
                    f"does not have an integrator to reset."
                )

            if self.parameters.integrator_mode._get(context) or force:
                new_input = self.integrator_function.reset(*args, context=context)[0]
                self.parameters.value._set(
                    self.function.execute(variable=new_input, context=context),
                    context=context,
                    override=True
                )
                self._update_output_ports(context=context)

            elif hasattr(self, "integrator_mode"):
                    raise MechanismError(f"Resetting '{self.name}' is not allowed because this Mechanism "
                                         f"is not stateful; it does not have an integrator to reset. "
                                         f"If it should be stateful, try setting the integrator_mode argument to True.")
            else:
                raise MechanismError(f"Resetting '{self.name}' is not allowed because this Mechanism "
                                     f"is not stateful; it does not have an integrator to reset.")
        else:
            raise MechanismError(f"Resetting '{self.name}' is not allowed because this Mechanism is not stateful; "
                                 f"it does not have an accumulator to reset.")

    def _get_current_mechanism_param(self, param_name, context=None):
        if param_name == "variable":
            raise MechanismError(f"The method '_get_current_mechanism_param' is intended for retrieving the current "
                                 f"value of a mechanism parameter; 'variable' is not a mechanism parameter. If looking "
                                 f"for {self.name}'s default variable, try '{self.name}.defaults.variable'.")
        try:
            return self._parameter_ports[param_name].parameters.value._get(context)
        except (AttributeError, TypeError):
            return getattr(self.parameters, param_name)._get(context)

    # when called externally, ContextFlags.PROCESSING is not set. Maintain this behavior here
    # even though it will not update input ports for example
    @handle_external_context(execution_phase=ContextFlags.IDLE)
    def execute(self,
                input=None,
                context=None,
                runtime_params=None,
                ):
        """Carry out a single `execution <Mechanism_Execution>` of the Mechanism.

        .. technical_note::
            Execution sequence:

            * Handle initialization if `initialization_status <Compoonent.initialization_status> is
              *ContextFlags.INITIALIZING*
            * Assign any `Port-specific runtime params <_Mechanism_Runtime_Port_and_Projection_Param_Specification>`
              to corresponding `runtime_params <Mechanism_Base.runtime_params>` dict.
            * While `is_finished <Component_Is_Finished> is not True:
              - validate `variable <Mechanism_Base.variable>` from `InputPorts <Mechanism_Base.input_ports>` and
                `runtime_params <Mechanism_Base.runtime_params>`.
              - update `input_ports <Mechanism_Base.input_ports>`
              - update `parameter_ports <Mechanism_Base.parameter_ports>`
              - execute Mechanism (calling _execute method) and set `value <Mechanism_Base.value>` parameter
              - update `output_ports <Mechanism_Base.output_ports>`
                Note:
                  > if execution is occurring as part of initialization, each output_port is reset to 0
                  > otherwise, their values are left as is until the next update
              - update `num_executions <Component_Num_Executions>` and check `max_executions <Component_Max_Executions>`
            * Report execution (if reportOutputPref is set)

        Arguments
        ---------

        input : List[value] or ndarray : default self.defaults.variable
            input to use for execution of the Mechanism.
            This must be consistent with the format of the Mechanism's `InputPort(s) <Mechanism_InputPorts>`:
            the number of items in the  outermost level of the list, or axis 0 of the ndarray, must equal the number
            of the Mechanism's `input_ports  <Mechanism_Base.input_ports>`, and each item must be compatible with the
            format (number and type of elements) of the `variable <InputPort.variable>` of the corresponding
            InputPort (see `Run Inputs <Composition_Execution_Inputs>` for details of input specification formats).

        runtime_params : [Dict[str, Dict[str, Dict[str, value]]]] : None
            a dictionary specifying values for `Parameters <Parameter>` of the Mechanism or those of any of its
            `Components <Component>` (`function <Mechanism_Base.function>`, `Ports <Mechanism_Ports>` and/or
            their `afferent Projections <Port_Projections>`), that temporarily override their values for the current
            execution, and are then restored to their previous values following execution (see
            `Mechanism_Runtime_Param_Specification` for details of specification).

        context : Context or str : None
            the context in which the Mechanism is executed, usually specified by its `execution_id
            <Context.execution_id>` (see `Composition_Execution_Context` for additional information about execution
            contexts.

        Returns
        -------

        Mechanism's output_values : List[value]
            list with the `value <OutputPort.value>` of each of the Mechanism's `OutputPorts
            <Mechanism_OutputPorts>` after either one `TIME_STEP` or a `TRIAL <TimeScale.TRIAL>`.

        """

        if self.initialization_status == ContextFlags.INITIALIZED:
            context.string = "{} EXECUTING {}: {}".format(context.source.name,self.name,
                                                               ContextFlags._get_context_string(
                                                                       context.flags, EXECUTION_PHASE))
        else:
            context.string = "{} INITIALIZING {}".format(context.source.name, self.name)

        if context.source is ContextFlags.COMMAND_LINE:
            self._initialize_from_context(context, override=False)

        # IMPLEMENTATION NOTE: Re-write by calling execute methods according to their order in functionDict:
        #         for func in self.functionDict:
        #             self.functionsDict[func]()

        # INITIALIZE MECHANISM if needed

        # Limit init to scope specified by context
        if self.initialization_status == ContextFlags.INITIALIZING:
            if context.composition:
                # Run full execute method for init of Composition
                pass
            # Only call subclass' _execute method and then return (do not complete the rest of this method)
            elif self.initMethod == INIT_EXECUTE_METHOD_ONLY:
                return_value = self._execute(variable=self.defaults.variable,
                                             context=context,
                                             runtime_params=runtime_params)

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
                    converted_to_2d = convert_to_np_array(return_value, dimension=2)
                # If return_value is a list of heterogenous elements, return as is
                #     (satisfies requirement that return_value be an array of possibly multidimensional values)
                if converted_to_2d.dtype == object:
                    return return_value
                # Otherwise, return value converted to 2d np.array
                else:
                    return converted_to_2d

            # Call only subclass' function during initialization (not its full _execute method nor rest of this method)
            elif self.initMethod == INIT_FUNCTION_METHOD_ONLY:
                return_value = super()._execute(variable=self.defaults.variable,
                                                context=context,
                                                runtime_params=runtime_params)
                return convert_to_np_array(return_value, dimension=2)

        # SET UP RUNTIME PARAMS if any

        # Extract all param specifications not related to the Mechanism itself or its function and place in subdicts;
        #    when Mechanism executes, _validate_and_assign_runtime_params will throw an error for any others found.
        runtime_port_params = self._parse_runtime_params(runtime_params, context)

        # EXECUTE MECHANISM

        if self.parameters.is_finished_flag._get(context) is True:
            self.parameters.num_executions_before_finished._set(0, override=True, context=context)

        while True:

            # Don't bother executing Mechanism if variable and/or value has been specified for all of its OutputPorts
            # Mechanism value is set to None, so its previous value will be retained (in accord with Lazy Evaluation)
            # However, num_executions and execution_count will be incremented (since the Mechanism was in fact executed)
            if (any(var_or_val in runtime_port_params[OUTPUT_PORT_PARAMS] for var_or_val in {VARIABLE, VALUE})
                    or
                    (PORT_SPECIFIC_PARAMS in runtime_port_params[OUTPUT_PORT_PARAMS]
                     and (all((var_or_val in p for var_or_val in {VARIABLE, VALUE})
                              for p in runtime_port_params[OUTPUT_PORT_PARAMS][PORT_SPECIFIC_PARAMS]
                              if p in {self, self.name})))
            ):
                self.parameters.is_finished_flag._set(True, context)
                value = None

            else:
                # VALIDATE InputPort(S) AND RUNTIME PARAMS
                self._check_args(params=runtime_params,
                                 target_set=runtime_params,
                                 context=context)

                # UPDATE VARIABLE and InputPort(s)
                # Executing or simulating Composition, so get input by updating input_ports
                if (input is None
                    and (context.execution_phase is not ContextFlags.IDLE)
                    and (self.input_port.path_afferents != [])):
                    variable = self._update_input_ports(runtime_port_params[INPUT_PORT_PARAMS], context)

                # Direct call to execute Mechanism with specified input, so assign input to Mechanism's input_ports
                else:
                    if context.source & ContextFlags.COMMAND_LINE:
                        context.execution_phase = ContextFlags.PROCESSING

                        if input is not None:
                            input = convert_all_elements_to_np_array(input)

                    if input is None:
                        input = self.defaults.variable
                    #     FIX:  this input value is sent to input CIMs when compositions are nested
                    #           variable should be based on afferent projections
                    variable = self._get_variable_from_input(input, context)

                self.parameters.variable._set(variable, context=context)

                # UPDATE PARAMETERPORT(S)
                self._update_parameter_ports(runtime_port_params[PARAMETER_PORT_PARAMS], context)

                # EXECUTE MECHANISM BY CALLING SUBCLASS _execute method AND ASSIGN RESULT TO self.value

                # IMPLEMENTATION NOTE: use value as buffer variable until it has been fully processed
                #                      to avoid multiple calls to (and potential log entries for) self.value property

                value = self._execute(variable=variable,
                                      runtime_params=runtime_params,
                                      context=context)

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
                    converted_to_2d = convert_to_np_array(value, dimension=2)
                    # If return_value is a list of heterogenous elements, return as is
                    #     (satisfies requirement that return_value be an array of possibly multidimensional values)
                    if converted_to_2d.dtype == object:
                        pass
                    # Otherwise, return value converted to 2d np.array
                    else:
                        # return converted_to_2d
                        value = converted_to_2d

                self.parameters.value._set(value, context=context)

            # UPDATE OUTPUTPORT(S)
            self._update_output_ports(runtime_port_params[OUTPUT_PORT_PARAMS], context)

            # MANAGE MAX_EXECUTIONS_BEFORE_FINISHED AND DETERMINE WHETHER TO BREAK
            max_executions = self.parameters.max_executions_before_finished._get(context)
            num_executions = self.parameters.num_executions_before_finished._get(context) + 1

            self.parameters.num_executions_before_finished._set(num_executions, override=True, context=context)

            if num_executions >= max_executions:
                self.parameters.is_finished_flag._set(True, context)
                warnings.warn(f"Maximum number of executions ({max_executions}) reached for {self.name}.")
                break

            if self.is_finished(context):
                self.parameters.is_finished_flag._set(True, context)
                break

            self.parameters.is_finished_flag._set(False, context)
            if not self.parameters.execute_until_finished._get(context):
                break

        # REPORT EXECUTION
        if self.prefs.reportOutputPref and (context.execution_phase & ContextFlags.PROCESSING | ContextFlags.LEARNING):
            self._report_mechanism_execution(
                self.get_input_values(context),
                self.parameters.values(),
                self.output_port.parameters.value._get(context),
                context=context
            )

        return value

    def _get_variable_from_input(self, input, context=None):
        input = convert_to_np_array(input, dimension=2)
        num_inputs = np.size(input, 0)
        num_input_ports = len(self.input_ports)
        if num_inputs != num_input_ports:
            # Check if inputs are of different lengths (indicated by dtype == np.dtype('O'))
            num_inputs = np.size(input)
            if isinstance(input, np.ndarray) and input.dtype is np.dtype('O') and num_inputs == num_input_ports:
                # Reduce input back down to sequence of arrays (to remove extra dim added by atleast_2d above)
                input = np.squeeze(input)
            else:
                num_inputs = np.size(input, 0)  # revert num_inputs to its previous value, when printing the error
                raise MechanismError("Number of inputs ({0}) to {1} does not match "
                                  "its number of input_ports ({2})".
                                  format(num_inputs, self.name,  num_input_ports ))
        for input_item, input_port in zip(input, self.input_ports):
            if len(input_port.defaults.value) == len(input_item):
                input_port.parameters.value._set(input_item, context)
            else:
                raise MechanismError(f"Length ({len(input_item)}) of input ({input_item}) does not match "
                                     f"required length ({len(input_port.defaults.variable)}) for input "
                                     f"to {InputPort.__name__} {repr(input_port.name)} of {self.name}.")

        return convert_to_np_array(self.get_input_values(context))

    def _update_input_ports(self, runtime_input_port_params=None, context=None):
        """Update value for each InputPort in self.input_ports:

        Call execute method for all (MappingProjection) Projections in Port.path_afferents
        Aggregate results (using InputPort execute method)
        Update InputPort.value

        """

        for i in range(len(self.input_ports)):
            port= self.input_ports[i]
            port._update(params=runtime_input_port_params,
                         context=context)
        return convert_to_np_array(self.get_input_values(context))

    def _update_parameter_ports(self, runtime_parameter_port_params=None, context=None):

        for port in self._parameter_ports:
            port._update(params=runtime_parameter_port_params,
                         context=context)

    def _get_parameter_port_deferred_init_control_specs(self):
        # FIX: 9/14/19 - THIS ASSUMES THAT ONLY CONTROLPROJECTIONS RELEVANT TO COMPOSITION ARE in DEFERRED INIT;
        #                BUT WHAT IF NODE SPECIFIED CONTROL BY AN EXISTING CONTROLMECHANISM NOT IN A COMPOSITION
        #                THAT WAS THEN ADDED;  COMPOSITION WOULD STILL NEED TO KNOW ABOUT IT TO ACTIVATE THE CTLPROJ
        ctl_specs = []
        for parameter_port in self._parameter_ports:
            for proj in parameter_port.mod_afferents:
                if proj.initialization_status == ContextFlags.DEFERRED_INIT:
                    try:
                        proj_control_signal_specs = proj._init_args['control_signal_params'] or {}
                    except (KeyError, TypeError):
                        proj_control_signal_specs = {}
                    proj_control_signal_specs.update({PROJECTIONS: [proj]})
                    ctl_specs.append(proj_control_signal_specs)
        return ctl_specs

    def _update_output_ports(self, runtime_output_port_params=None, context=None):
        """Execute function for each OutputPort and assign result of each to corresponding item of self.output_values

        owner_value arg can be used to override existing (or absent) value of owner as variable for OutputPorts
        and assign a specified (set of) value(s).

        """
        for i in range(len(self.output_ports)):
            port = self.output_ports[i]
            port._update(params=runtime_output_port_params,
                         context=context)

    def initialize(self, value, context=None):
        """Assign an initial value to the Mechanism's `value <Mechanism_Base.value>` attribute and update its
        `OutputPorts <Mechanism_OutputPorts>`.

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
        self._update_output_ports(context=context)


    def _parse_runtime_params(self, runtime_params, context):
        """Move Port param specifications and nested Project-specific specifications into sub-dicts.

        Move any specifications for Port types into type-specific sub-dicts
        For each type-specific sub-dict,
          - move any specifications for individual Ports into PORT_SPECIFIC sub-dict
          - move any specifications for Projection types into type-specific subdicts
          - move any specifications for individual Projections into PORT_SPECIFIC sub-dict

        Returns
        -------

        dict : {port_param_keyword : Port type-specific dict}
            dict containing three sub-dicts, one for each of the three Port types, of the following form:

            {'<Port type>_PARAMS': parameter of Port type or its function : value,
                                   'PORT_SPECIFIC_PARAMS' : {<Port or Port name> : {parameter : value}}
                                   '<Projection type>_PARAMS' : {parameter of Projection type or its function : value},
                                   'PROJECTION_SPECIFIC_PARAMS' : {<Projection or name> : {parameter : value}}}

        """

        from psyneulink.core.components.projections.projection import projection_param_keywords

        def move_item_specific_params_to_specific_sub_dict(outer_dict,
                                                           dest_dict,
                                                           sub_dict_names,
                                                           item_list,
                                                           specific_dict_name):
            """Move any specifications for individual Ports or Projections into a consolidated SPECIFIC sub-dict

            Arguments
            ---------
            outer_dict : dict
                outer-most dict to be searched;
                runtime_params for port params; a port_params_dict for projection params
            dest_dict : dict
                dict where <COMPONENT>_SPECIFIC_PARAMS will be created; (always a Port type-specific dict)
            sub_dict_names : list(str)
                port_param_keywords or projection_param_keywords()
            item_list : ContentAddressableList
                attribute with list of items to search for specific item being specified;
                self.<port_type> for port param, self.afferents for for projection params
            specific_dict_name : str
                <COMPONENT>_SPECIFIC_PARAMS:  PORT_SPECIFIC_PARAMS or PROJECTION_SPECIFIC_PARAMS

            """
            for key in outer_dict.copy():
                # Recursively check Port or Projection type-specific sub-dicts for entries to be moved; even though
                #    the search is recursive, the move is always to the <COMPONENT>_SPECIFIC_PARAMS dict in dest_dict
                if key in sub_dict_names:
                    move_item_specific_params_to_specific_sub_dict(outer_dict[key],
                                                                   dest_dict,
                                                                   sub_dict_names,
                                                                   item_list,
                                                                   specific_dict_name)
                    continue

                # Skip if entry is not a paramater specification dict;
                #  this is so that a key being used as the name of a parameter itself to be specified, is not treated
                #  below as a specification for the corresponding ParameterPort (which has the key as its name)
                if not isinstance(outer_dict[key], dict):
                    continue

                # Reference can be the Port or Projection itself...
                elif key in item_list:
                    item = key
                # or the Port or Projection's name
                elif key in item_list.names:
                    item = item_list[item]
                else:
                    continue
                # Move param specification dict for item to entry with same key in <COMPONENT>_SPECIFIC_PARAMS dict
                item_specific_dict = {key : outer_dict.pop(key)}
                if specific_dict_name in dest_dict:
                    dest_dict[specific_dict_name].update(item_specific_dict)
                else:
                    dest_dict[specific_dict_name] = defaultdict(lambda:{}, item_specific_dict)

        port_param_dicts = {INPUT_PORT_PARAMS: {},
                            PARAMETER_PORT_PARAMS: {},
                            OUTPUT_PORT_PARAMS: {}}

        if runtime_params:

            for port_type in self.portListAttr:
                port_param_dict_name = port_type.paramsType
                ports_attr = getattr(self, self.portListAttr[port_type])

                # Create port_param_dict if it doesn't yet exist
                if port_param_dict_name not in runtime_params:
                    runtime_params[port_param_dict_name] = defaultdict(lambda:{})

                # Move any specifications of individual Ports of this type to PORT_SPECIFIC_PARAMS dict
                move_item_specific_params_to_specific_sub_dict(outer_dict = runtime_params,
                                                               dest_dict = runtime_params[port_param_dict_name],
                                                               sub_dict_names = [port_param_dict_name],
                                                               item_list = ports_attr,
                                                               specific_dict_name = PORT_SPECIFIC_PARAMS)

                # Move any specifications of individual Projections for this Port type to PROJECTION_SPECIFIC_PARAMS
                move_item_specific_params_to_specific_sub_dict(outer_dict = runtime_params[port_param_dict_name],
                                                               dest_dict = runtime_params[port_param_dict_name],
                                                               sub_dict_names = projection_param_keywords(),
                                                               item_list = self.afferents,
                                                               specific_dict_name = PROJECTION_SPECIFIC_PARAMS)

                # Move the port_specific_param dicts to port_param_dicts for return
                port_param_dicts[port_param_dict_name] = defaultdict(lambda:{},
                                                                     runtime_params.pop(port_param_dict_name, {}))

        return port_param_dicts

    def _get_param_ids(self):
        # FIXME: parameter ports should be part of generated params
        return ["_parameter_ports"] + super()._get_param_ids()

    def _get_param_struct_type(self, ctx):
        ports_params = (ctx.get_param_struct_type(s) for s in self._parameter_ports)
        ports_param_struct = pnlvm.ir.LiteralStructType(ports_params)
        mech_param_struct = ctx.get_param_struct_type(super())

        return pnlvm.ir.LiteralStructType((ports_param_struct,
                                           *mech_param_struct))

    def _get_state_ids(self):
        # FIXME: parameter ports should be part of generated state
        return ["_parameter_ports"] + super()._get_state_ids()

    def _get_state_struct_type(self, ctx):
        ports_state = (ctx.get_state_struct_type(s) for s in self._parameter_ports)
        ports_state_struct = pnlvm.ir.LiteralStructType(ports_state)
        mech_state_struct = ctx.get_state_struct_type(super())

        return pnlvm.ir.LiteralStructType((ports_state_struct,
                                           *mech_state_struct))

    def _get_output_struct_type(self, ctx):
        output_type_list = (ctx.get_output_struct_type(port) for port in self.output_ports)
        return pnlvm.ir.LiteralStructType(output_type_list)

    def _get_input_struct_type(self, ctx):
        # Extract the non-modulation portion of InputPort input struct
        input_type_list = [ctx.get_input_struct_type(port).elements[0] for port in self.input_ports]
        # Get modulatory inputs
        mod_input_type_list = [ctx.get_output_struct_type(proj) for proj in self.mod_afferents]
        if len(mod_input_type_list) > 0:
            input_type_list.append(pnlvm.ir.LiteralStructType(mod_input_type_list))
        return pnlvm.ir.LiteralStructType(input_type_list)

    def _get_param_initializer(self, context):
        port_param_init = tuple(s._get_param_initializer(context) for s in self._parameter_ports)
        mech_param_init = super()._get_param_initializer(context)

        return (port_param_init, *mech_param_init)

    def _get_state_initializer(self, context):
        port_state_init = tuple(s._get_state_initializer(context) for s in self._parameter_ports)
        mech_state_init = super()._get_state_initializer(context)

        return (port_state_init, *mech_state_init)

    def _gen_llvm_ports(self, ctx, builder, ports, group,
                        get_output_ptr, fill_input_data,
                        mech_params, mech_state, mech_input):
        group_ports = getattr(self, group)
        ports_param = pnlvm.helpers.get_param_ptr(builder, self, mech_params, group)
        ports_state = pnlvm.helpers.get_state_ptr(builder, self, mech_state, group, None)

        mod_afferents = self.mod_afferents
        for i, port in enumerate(ports):
            p_function = ctx.import_llvm_function(port)

            # Find output location
            builder, p_output = get_output_ptr(builder, i)

            # Allocate the input structure (data + modulation)
            p_input = builder.alloca(p_function.args[2].type.pointee)

            # Copy input data to input structure
            builder = fill_input_data(builder, p_input, i)

            # Copy mod_afferent inputs
            for idx, p_mod in enumerate(port.mod_afferents):
                mech_mod_afferent_idx = mod_afferents.index(p_mod)
                mod_in_ptr = builder.gep(mech_input, [ctx.int32_ty(0),
                                                      ctx.int32_ty(len(self.input_ports)),
                                                      ctx.int32_ty(mech_mod_afferent_idx)])
                mod_out_ptr = builder.gep(p_input, [ctx.int32_ty(0), ctx.int32_ty(1 + idx)])
                afferent_val = builder.load(mod_in_ptr)
                builder.store(afferent_val, mod_out_ptr)

            port_idx = group_ports.index(port)
            p_params = builder.gep(ports_param, [ctx.int32_ty(0),
                                                 ctx.int32_ty(port_idx)])
            p_state = builder.gep(ports_state, [ctx.int32_ty(0),
                                                ctx.int32_ty(port_idx)])

            builder.call(p_function, [p_params, p_state, p_input, p_output])

        return builder

    def _gen_llvm_input_ports(self, ctx, builder,
                              mech_params, mech_state, mech_input):
        # Allocate temporary storage. We rely on the fact that series
        # of InputPort results should match the main function input.
        ip_output_list = []
        for port in self.input_ports:
            ip_function = ctx.import_llvm_function(port)
            ip_output_list.append(ip_function.args[3].type.pointee)

        # Check if all elements are the same. Function input will be array type if yes.
        if len(set(ip_output_list)) == 1:
            ip_output_type = pnlvm.ir.ArrayType(ip_output_list[0], len(ip_output_list))
        else:
            ip_output_type = pnlvm.ir.LiteralStructType(ip_output_list)

        ip_output = builder.alloca(ip_output_type)

        def _get_output_ptr(b, i):
            ptr = b.gep(ip_output, [ctx.int32_ty(0), ctx.int32_ty(i)])
            return b, ptr

        def _fill_input(b, p_input, i):
            ip_in = builder.gep(mech_input, [ctx.int32_ty(0), ctx.int32_ty(i)])
            # Input port inputs are {original parameter, [modulations]},
            # fill in the first one.
            data_ptr = builder.gep(p_input, [ctx.int32_ty(0), ctx.int32_ty(0)])
            b.store(b.load(ip_in), data_ptr)
            return b

        builder = self._gen_llvm_ports(ctx, builder, self.input_ports, "input_ports",
                                       _get_output_ptr, _fill_input,
                                       mech_params, mech_state, mech_input)

        return ip_output, builder

    def _gen_llvm_param_ports_for_obj(self, obj, params_in, ctx, builder,
                                      mech_params, mech_state, mech_input):
        # Allocate a shadow structure to overload user supplied parameters
        params_out = builder.alloca(params_in.type.pointee)
        # Copy original values. This handles params without param ports.
        # Few extra copies will be eliminated by the compiler.
        builder.store(builder.load(params_in), params_out)

        # Filter out param ports without corresponding params for this function
        param_ports = [p for p in self._parameter_ports if p.name in obj.llvm_param_ids]

        def _get_output_ptr(b, i):
            ptr = pnlvm.helpers.get_param_ptr(b, obj, params_out,
                                              param_ports[i].name)
            return b, ptr

        def _fill_input(b, p_input, i):
            param_in_ptr = pnlvm.helpers.get_param_ptr(b, obj, params_in,
                                                       param_ports[i].name)
            # Parameter port inputs are {original parameter, [modulations]},
            # fill in the first one.
            data_ptr = builder.gep(p_input, [ctx.int32_ty(0), ctx.int32_ty(0)])
            b.store(b.load(param_in_ptr), data_ptr)
            return b

        builder = self._gen_llvm_ports(ctx, builder, param_ports, "_parameter_ports",
                                       _get_output_ptr, _fill_input,
                                       mech_params, mech_state, mech_input)
        return params_out, builder

    def _gen_llvm_output_port_parse_variable(self, ctx, builder,
                                             mech_params, mech_state, value, port):
            port_spec = port._variable_spec
            if port_spec == OWNER_VALUE:
                return value
            elif isinstance(port_spec, tuple) and port_spec[0] == OWNER_VALUE:
                index = port_spec[1]() if callable(port_spec[1]) else port_spec[1]

                assert index < len(value.type.pointee)
                return builder.gep(value, [ctx.int32_ty(0), ctx.int32_ty(index)])
            elif port_spec == OWNER_EXECUTION_COUNT:
                execution_count = pnlvm.helpers.get_state_ptr(builder, self, mech_state, "execution_count")
                return execution_count
            else:
                #TODO: support more spec options
                assert False, "Unsupported OutputPort spec: {} ({})".format(port_spec, value.type)

    def _gen_llvm_output_ports(self, ctx, builder, value,
                               mech_params, mech_state, mech_in, mech_out):
        def _get_output_ptr(b, i):
            ptr = b.gep(mech_out, [ctx.int32_ty(0), ctx.int32_ty(i)])
            return b, ptr

        def _fill_input(b, s_input, i):
            data_ptr = self._gen_llvm_output_port_parse_variable(ctx, b,
               mech_params, mech_state, value, self.output_ports[i])
            # Output port inputs are {original parameter, [modulations]},
            # fill in the first one.
            input_ptr = builder.gep(s_input, [ctx.int32_ty(0), ctx.int32_ty(0)])
            if input_ptr.type != data_ptr.type:
                port = self.output_ports[i]
                warnings.warn("Shape mismatch: {} parsed value does not match "
                              "output port: mech value: {} spec: {} parsed {}.".format(
                              port, self.defaults.value, port._variable_spec,
                              port.defaults.variable))
                input_ptr = builder.gep(input_ptr, [ctx.int32_ty(0), ctx.int32_ty(0)])
            b.store(b.load(data_ptr), input_ptr)
            return b

        builder = self._gen_llvm_ports(ctx, builder, self.output_ports, "output_ports",
                                       _get_output_ptr, _fill_input,
                                       mech_params, mech_state, mech_in)
        return builder

    def _gen_llvm_invoke_function(self, ctx, builder, function, params, state, variable, *, tags:frozenset):
        fun = ctx.import_llvm_function(function, tags=tags)
        fun_out = builder.alloca(fun.args[3].type.pointee)

        builder.call(fun, [params, state, variable, fun_out])

        return fun_out, builder

    def _gen_llvm_is_finished_cond(self, ctx, builder, params, state):
        return pnlvm.ir.IntType(1)(1)

    def _gen_llvm_mechanism_functions(self, ctx, builder, params, state, arg_in,
                                      ip_output, *, tags:frozenset):

        # Default mechanism runs only the main function
        f_params_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, "function")
        f_params, builder = self._gen_llvm_param_ports_for_obj(
                self.function, f_params_ptr, ctx, builder, params, state, arg_in)
        f_state = pnlvm.helpers.get_state_ptr(builder, self, state, "function")

        return self._gen_llvm_invoke_function(ctx, builder, self.function,
                                              f_params, f_state, ip_output,
                                              tags=tags)

    def _gen_llvm_function_internal(self, ctx, builder, params, state, arg_in,
                                    arg_out, *, tags:frozenset):

        ip_output, builder = self._gen_llvm_input_ports(ctx, builder,
                                                        params, state, arg_in)

        value, builder = self._gen_llvm_mechanism_functions(ctx, builder, params,
                                                            state, arg_in,
                                                            ip_output,
                                                            tags=tags)

        # Update execution counter
        exec_count_ptr = pnlvm.helpers.get_state_ptr(builder, self, state, "execution_count")
        exec_count = builder.load(exec_count_ptr)
        exec_count = builder.fadd(exec_count, exec_count.type(1))
        builder.store(exec_count, exec_count_ptr)

        # Update internal clock (i.e. num_executions parameter)
        num_executions_ptr = pnlvm.helpers.get_state_ptr(builder, self, state, "num_executions")
        for scale in [TimeScale.TIME_STEP, TimeScale.PASS, TimeScale.TRIAL, TimeScale.RUN]:
            num_exec_time_ptr = builder.gep(num_executions_ptr, [ctx.int32_ty(0), ctx.int32_ty(scale.value)])
            new_val = builder.load(num_exec_time_ptr)
            new_val = builder.add(new_val, ctx.int32_ty(1))
            builder.store(new_val, num_exec_time_ptr)

        builder = self._gen_llvm_output_ports(ctx, builder, value, params, state, arg_in, arg_out)

        val_ptr = pnlvm.helpers.get_state_ptr(builder, self, state, "value")
        if val_ptr.type.pointee == value.type.pointee:
            pnlvm.helpers.push_state_val(builder, self, state, "value", value)
        else:
            # FIXME: Does this need some sort of parsing?
            warnings.warn("Shape mismatch: function result does not match mechanism value param: {} vs. {}".format(value.type.pointee, val_ptr.type.pointee))

        # is_finished should be checked after output ports ran
        is_finished_f = ctx.import_llvm_function(self, tags=tags.union({"is_finished"}))
        is_finished_cond = builder.call(is_finished_f, [params, state, arg_in,
                                                        arg_out])
        return builder, is_finished_cond

    def _gen_llvm_function_reset(self, ctx, builder, params, state, arg_in, arg_out, *, tags:frozenset):
        assert "reset" in tags
        reinit_func = ctx.import_llvm_function(self.function, tags=tags)
        reinit_params = pnlvm.helpers.get_param_ptr(builder, self, params, "function")
        reinit_state = pnlvm.helpers.get_state_ptr(builder, self, state, "function")
        reinit_in = builder.alloca(reinit_func.args[2].type.pointee)
        reinit_out = builder.alloca(reinit_func.args[3].type.pointee)
        builder.call(reinit_func, [reinit_params, reinit_state, reinit_in,
                                   reinit_out])

        return builder

    def _gen_llvm_function(self, *, extra_args=[], ctx:pnlvm.LLVMBuilderContext, tags:frozenset):
        if "is_finished" not in tags:
            return super()._gen_llvm_function(extra_args=extra_args, ctx=ctx,
                                              tags=tags)

        # Keep all 4 standard arguments to ease invocation
        args = [ctx.get_param_struct_type(self).as_pointer(),
                ctx.get_state_struct_type(self).as_pointer(),
                ctx.get_input_struct_type(self).as_pointer(),
                ctx.get_output_struct_type(self).as_pointer()]

        builder = ctx.create_llvm_function(args, self,
                                           return_type=pnlvm.ir.IntType(1),
                                           tags=tags)
        params, state = builder.function.args[:2]
        finished = self._gen_llvm_is_finished_cond(ctx, builder, params, state)
        builder.ret(finished)
        return builder.function

    def _gen_llvm_function_body(self, ctx, builder, params, state, arg_in, arg_out, *, tags:frozenset):
        assert "reset" not in tags

        params, builder = self._gen_llvm_param_ports_for_obj(
                self, params, ctx, builder, params, state, arg_in)

        is_finished_flag_ptr = pnlvm.helpers.get_state_ptr(builder, self, state,
                                                           "is_finished_flag")
        is_finished_count_ptr = pnlvm.helpers.get_state_ptr(builder, self, state,
                                                            "num_executions_before_finished")
        is_finished_max_ptr = pnlvm.helpers.get_param_ptr(builder, self, params,
                                                          "max_executions_before_finished")

        # Reset the flag and counter
        # FIXME: Use int for flag
        # FIXME: continue previous computation if not finished
        current_flag = builder.load(is_finished_flag_ptr)
        was_finished = builder.fcmp_ordered("==", current_flag, current_flag.type(1))
        with builder.if_then(was_finished):
            builder.store(is_finished_count_ptr.type.pointee(0),
                          is_finished_count_ptr)

        # Enter the loop
        loop_block = builder.append_basic_block(builder.basic_block.name + "_loop")
        end_block = builder.append_basic_block(builder.basic_block.name + "_end")
        builder.branch(loop_block)
        builder.position_at_end(loop_block)

        # Get internal function
        args_t = [a.type for a in builder.function.args]
        internal_builder = ctx.create_llvm_function(args_t, self,
                                                    name=builder.function.name + "_internal",
                                                    return_type=pnlvm.ir.IntType(1))
        iparams, istate, iin, iout = internal_builder.function.args[:4]
        internal_builder, is_finished = self._gen_llvm_function_internal(ctx, internal_builder,
                                                                         iparams, istate, iin, iout, tags=tags)
        internal_builder.ret(is_finished)

        # Call Internal Function
        internal_f = internal_builder.function
        is_finished_cond = builder.call(internal_f, [params, state, arg_in, arg_out, *builder.function.args[4:]])

        #FIXME: Flag and count should be int instead of float
        # Check if we reached maximum iteration count
        is_finished_count = builder.load(is_finished_count_ptr)
        is_finished_count = builder.fadd(is_finished_count,
                                         is_finished_count.type(1))
        builder.store(is_finished_count, is_finished_count_ptr)
        is_finished_max = builder.load(is_finished_max_ptr)
        max_reached = builder.fcmp_ordered(">=", is_finished_count,
                                           is_finished_max)

        # Check if execute until finished mode is enabled
        exec_until_fin_ptr = pnlvm.helpers.get_param_ptr(builder, self, params,
                                                         "execute_until_finished")
        exec_until_fin = builder.load(exec_until_fin_ptr)
        exec_until_off = builder.fcmp_ordered("==", exec_until_fin, exec_until_fin.type(0))

        # Combine conditions
        is_finished = builder.or_(is_finished_cond, max_reached)
        iter_end = builder.or_(is_finished, exec_until_off)

        # Check if in integrator mode
        if hasattr(self, "integrator_mode"):
            int_mode_ptr = pnlvm.helpers.get_param_ptr(builder, self, params,
                                                       "integrator_mode")
            int_mode = builder.load(int_mode_ptr)
            int_mode_off = builder.fcmp_ordered("==", int_mode, int_mode.type(0))
            iter_end = builder.or_(iter_end, int_mode_off)

        with builder.if_then(iter_end):
            new_flag = builder.uitofp(is_finished, current_flag.type)
            builder.store(new_flag, is_finished_flag_ptr)
            builder.branch(end_block)

        builder.branch(loop_block)
        builder.position_at_end(end_block)

        return builder

    def _report_mechanism_execution(self, input_val=None, params=None, output=None, context=None):

        if input_val is None:
            input_val = self.get_input_values(context)
        if output is None:
            output = self.output_port.parameters.value._get(context)
        params = params or self.parameters.values()

        if 'mechanism' in self.name or 'Mechanism' in self.name:
            mechanism_string = ' '
        else:
            mechanism_string = ' mechanism'

        # FIX: kmantel: previous version would fail on anything but iterables of things that can be cast to floats
        #      if you want more specific output, you can add conditional tests here
        try:
            input_string = [float("{:0.3}".format(float(i))) for i in input_val].__str__().strip("[]")
        except TypeError:
            input_string = input_val

        print("\n\'{}\'{} executed:\n- input:  {}".format(self.name, mechanism_string, input_string))

        if params:
            print("- params:")
            # Sort for consistency of output
            params_keys_sorted = sorted(params.keys())
            for param_name in params_keys_sorted:
                # No need to report:
                #    function_params here, as they will be reported for the function itself below;
                #    input_ports or output_ports, as these are inherent in the structure
                if param_name in {FUNCTION_PARAMS, INPUT_PORTS, OUTPUT_PORTS}:
                    continue
                param_is_function = False
                param_value = params[param_name]
                if isinstance(param_value, Function):
                    param = param_value.name
                    param_is_function = True
                elif isinstance(param_value, type(Function)):
                    param = param_value.__name__
                    param_is_function = True
                elif isinstance(param_value, (types.FunctionType, types.MethodType)):
                    param = param_value.__self__.__class__.__name__
                    param_is_function = True
                else:
                    param = param_value
                print("\t{}: {}".format(param_name, str(param).__str__().strip("[]")))
                if param_is_function:
                    # Sort for consistency of output
                    func_params_keys_sorted = sorted(self.function.parameters.names())
                    for fct_param_name in func_params_keys_sorted:
                        print("\t\t{}: {}".
                              format(fct_param_name,
                                     str(getattr(self.function.parameters, fct_param_name)).__str__().strip("[]")))

        # FIX: kmantel: previous version would fail on anything but iterables of things that can be cast to floats
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
                        show_port_function_params:bool=False,
                        show_values:bool=False,
                        use_labels:bool=False,
                        show_headers:bool=False,
                        show_roles:bool=False,
                        show_conditions:bool=False,
                        composition=None,
                        compact_cim:bool=False,
                        condition:tc.optional(Condition)=None,
                        node_border:str="1",
                        output_fmt:tc.enum('pdf','struct')='pdf',
                        context=None
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
            show the `function <Component.function>` of the Mechanism and each of its Ports.

        show_mech_function_params : bool : default False
            show the parameters of the Mechanism's `function <Component.function>` if **show_functions** is True.

        show_port_function_params : bool : default False
            show parameters for the `function <Component.function>` of the Mechanism's Ports if **show_functions** is
            True).

        show_values : bool : default False
            show the `value <Component.value>` of the Mechanism and each of its Ports (prefixed by "=").

        use_labels : bool : default False
            use labels for values if **show_values** is `True`; labels must be specified in the `input_labels_dict
            <Mechanism.input_labels_dict>` (for InputPort values) and `output_labels_dict
            <Mechanism.output_labels_dict>` (for OutputPort values), otherwise the value is used.

        show_headers : bool : default False
            show the Mechanism, InputPort, ParameterPort and OutputPort headers.

            **composition** argument (if **composition** is not specified, show_roles is ignored).

        show_conditions : bool : default False
            show the `conditions <Condition>` used by `Composition` to determine whether/when to execute each Mechanism
            (if **composition** is not specified, show_conditions is ignored).

        composition : Composition : default None
            specifies the `Composition` (to which the Mechanism must belong) for which to show its role (see **roles**);
            if this is not specified, the **show_roles** argument is ignored.

        compact_cim : bool : default False
            specifies whether to suppress InputPort fields for input_CIM and OutputPort fields for output_CIM

        output_fmt : keyword : default 'pdf'
            'pdf': generate and open a pdf with the visualization;\n
            'jupyter': return the object (ideal for working in jupyter/ipython notebooks)\n
            'struct': return a string that specifies the structure of the Mechanism using html table format
            for use in a GraphViz node specification.

        Example HTML for structure:
            .. parsed-literal::

                <<table border="1" cellborder="0" cellspacing="0" bgcolor="tan">          <- MAIN TABLE

                <tr>                                                                      <- BEGIN OutputPorts
                    <td colspan="2"><table border="0" cellborder="0" BGCOLOR="bisque">    <- OutputPorts OUTER TABLE
                        <tr>
                            <td colspan="1"><b>OutputPorts</b></td>                      <- OutputPorts HEADER
                        </tr>
                        <tr>
                            <td><table border="0" cellborder="1">                         <- OutputPort CELLS TABLE
                                <tr>
                                    <td port="OutputPortPort1">OutputPort 1<br/><i>function 1</i><br/><i>=value</i></td>
                                    <td port="OutputPortPort2">OutputPort 2<br/><i>function 2</i><br/><i>=value</i></td>
                                </tr>
                            </table></td>
                        </tr>
                    </table></td>
                </tr>

                <tr>                                                                      <- BEGIN MECHANISM & ParameterPorts
                    <td port="Mech name"><b>Mech name</b><br/><i>Roles</i></td>           <- MECHANISM CELL (OUTERMOST TABLE)
                    <td><table border="0" cellborder="0" BGCOLOR="bisque">                <- ParameterPorts OUTER TABLE
                        <tr>
                            <td><b>ParameterPorts</b></td>                               <- ParameterPorts HEADER
                        </tr>
                        <tr>
                            <td><table border="0" cellborder="1">                         <- ParameterPort CELLS TABLE
                                <tr><td port="ParamPort1">Param 1<br/><i>function 1</i><br/><i>= value</i></td></tr>
                                <tr><td port="ParamPort1">Param 2<br/><i>function 2</i><br/><i>= value</i></td></tr>
                            </table></td>
                        </tr>
                    </table></td>
                </tr>

                <tr>                                                                      <- BEGIN InputPorts
                    <td colspan="2"><table border="0" cellborder="0" BGCOLOR="bisque">    <- InputPortS OUTER TABLE
                        <tr>
                            <td colspan="1"><b>InputPorts</b></td>                       <- InputPorts HEADER
                        </tr>
                        <tr>
                            <td><table border="0" cellborder="1">                         <- InputPort CELLS TABLE
                                <tr>
                                    <td port="InputPortPort1">InputPort 1<br/><i>function 1</i><br/><i>= value</i></td>
                                    <td port="InputPortPort2">InputPort 2<br/><i>function 2</i><br/><i>= value</i></td>
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

        # Outer Port table:
        outer_table_spec = '<table border="0" cellborder="0" bgcolor="#FAFAD0">' # NEAR LIGHTGOLDENRODYELLOW

        # Header cell of outer Port table:
        input_ports_header     = f'<tr><td colspan="1" valign="middle"><b><i>{InputPort.__name__}s</i></b></td></tr>'
        parameter_ports_header = f'<tr><td rowspan="1" valign="middle"><b><i>{ParameterPort.__name__}s</i></b></td>'
        output_ports_header    = f'<tr><td colspan="1" valign="middle"><b><i>{OutputPort.__name__}s</i></b></td></tr>'

        # Inner Port table (i.e., that contains individual ports in each cell):
        inner_table_spec = \
            '<table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD">'

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
                from psyneulink.core.compositions.composition import CompositionInterfaceMechanism, NodeRole
                if self is composition.controller:
                    mech_roles = f'<br/><i>CONTROLLER</i>'
                elif not isinstance(self, CompositionInterfaceMechanism):
                    roles = [role.name for role in list(composition.nodes_to_roles[self])]
                    mech_roles = f'<br/><i>{",".join(roles)}</i>'

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
                        fct_params.append(f'{param.name}={param._get(context)}')
                    fct_params = ", ".join(fct_params)
                mech_function = f'<br/><i>{self.function.__class__.__name__}({fct_params})</i>'
            mech_value = ''
            if show_values:
                mech_value = f'<br/>={self.value}'
            # Mech cell should span full width if there are no ParameterPorts
            cols = 1
            if not len(self.parameter_ports):
                cols = 2
            return f'<td port="{self.name}" colspan="{cols}">' + \
                   mech_name + mech_roles + mech_condition + mech_function + mech_value + '</td>'

        @tc.typecheck
        def port_table(port_list:ContentAddressableList,
                        port_type:tc.enum(InputPort, ParameterPort, OutputPort)):
            """Return html with table for each port in port_list, including functions and/or values as specified

            Each table has a header cell and and inner table with cells for each port in the list
            InputPort and OutputPort cells are aligned horizontally;  ParameterPort cells are aligned vertically.
            Use show_functions, show_values and include_labels arguments from call to _show_structure()
            See _show_structure docstring for full template.
            """

            def port_cell(port, include_function:bool=False, include_value:bool=False, use_label:bool=False):
                """Return html for cell in port inner table
                Format:  <td port="PortPort">PortName<br/><i>function 1</i><br/><i>=value</i></td>
                """

                function = ''
                fct_params = ''
                if include_function:
                    if show_port_function_params:
                        fct_params = []
                        for param in [param for param in port.function_parameters
                                      if param.modulable and param.name not in {ADDITIVE_PARAM, MULTIPLICATIVE_PARAM}]:
                            fct_params.append(f'{param.name}={param._get(context)}')
                        fct_params = ", ".join(fct_params)
                    function = f'<br/><i>{port.function.__class__.__name__}({fct_params})</i>'
                value=''
                if include_value:
                    if use_label and not isinstance(port, ParameterPort):
                        value = f'<br/>={port.label}'
                    else:
                        value = f'<br/>={port.value}'
                return f'<td port="{self._get_port_name(port)}"><b>{port.name}</b>{function}{value}</td>'


            # InputPorts
            if port_type is InputPort:
                if show_headers:
                    ports_header = input_ports_header
                else:
                    ports_header = ''
                table = f'<td colspan="2"> {outer_table_spec} {ports_header}<tr><td>{inner_table_spec}<tr>'
                for port in port_list:
                    table += port_cell(port, show_functions, show_values, use_labels)
                table += '</tr></table></td></tr></table></td>'

            # ParameterPorts
            elif port_type is ParameterPort:
                if show_headers:
                    ports_header = parameter_ports_header
                else:
                    ports_header = '<tr>'
                table = f'<td> {outer_table_spec} {ports_header} <td> {inner_table_spec}'
                for port in port_list:
                    table += '<tr>' + port_cell(port, show_functions, show_values, use_labels) + '</tr>'
                table += '</table></td></tr></table></td>'

            # OutputPorts
            elif port_type is OutputPort:
                if show_headers:
                    ports_header = output_ports_header
                else:
                    ports_header = ''
                table = f'<td colspan="2"> {outer_table_spec} <tr><td>{inner_table_spec}<tr>'
                for port in port_list:
                    table += port_cell(port, show_functions, show_values, use_labels)
                table += f'</tr></table></td></tr> {ports_header} </table></td>'

            return table


        # Construct InputPorts table
        if (len(self.input_ports)
                and (not compact_cim or (self is not composition.input_CIM and self is not composition.parameter_CIM))):
            input_ports_table = f'<tr>{port_table(self.input_ports, InputPort)}</tr>'
        else:
            input_ports_table = ''

        # Construct ParameterPorts table
        if len(self.parameter_ports):
        # if len(self.parameter_ports) and (not compact_cim or self is not composition.parameter_CIM):
            parameter_ports_table = port_table(self.parameter_ports, ParameterPort)
        else:
            parameter_ports_table = ''

        # Construct OutputPorts table
        if len(self.output_ports) and (not compact_cim or self is not composition.output_CIM):
            output_ports_table = f'<tr>{port_table(self.output_ports, OutputPort)}</tr>'

        else:
            output_ports_table = ''

        # Construct full table
        m_node_struct = '<' + node_table_spec + \
                        output_ports_table + \
                        '<tr>' + mech_cell() + parameter_ports_table + '</tr>' + \
                        input_ports_table + \
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
    def _get_port_name(self, port:Port):
        if isinstance(port, InputPort):
            port_type = InputPort.__name__
        elif isinstance(port, ParameterPort):
            port_type = ParameterPort.__name__
        elif isinstance(port, OutputPort):
            port_type = OutputPort.__name__
        else:
            assert False, f'Mechanism._get_port_name() must be called with an ' \
                f'{InputPort.__name__}, {ParameterPort.__name__} or {OutputPort.__name__}'
        return port_type + '-' + port.name

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
    @handle_external_context()
    def add_ports(self, ports, update_variable=True, context=None):
        """
        add_ports(ports)

        Add one or more `Ports <Port>` to the Mechanism.  Only `InputPorts <InputPort>` and `OutputPorts
        <OutputPort>` can be added; `ParameterPorts <ParameterPort>` cannot be added to a Mechanism after it has
        been constructed.

        If the `owner <Port_Base.owner>` of a Port specified in the **ports** argument is not the same as the
        Mechanism to which it is being added an error is generated.    If the name of a specified Port is the same
        as an existing one with the same name, an index is appended to its name, and incremented for each Port
        subsequently added with the same name (see `naming conventions <Registry_Naming>`).  If a specified Port
        already belongs to the Mechanism, the request is ignored.

        .. note::
            Adding InputPorts to a Mechanism changes the size of its `variable <Mechanism_Base.variable>` attribute,
            which may produce an incompatibility with its `function <Mechanism_Base.function>` (see
            `Mechanism InputPorts <Mechanism_InputPorts>` for a more detailed explanation).

        Arguments
        ---------

        ports : Port or List[Port]
            one more `InputPorts <InputPort>` or `OutputPorts <OutputPort>` to be added to the Mechanism.
            Port specification(s) can be an InputPort or OutputPort object, class reference, class keyword, or
            `Port specification dictionary <Port_Specification>` (the latter must have a *PORT_TYPE* entry
            specifying the class or keyword for InputPort or OutputPort).

        Returns a dictionary with two entries, containing the list of InputPorts and OutputPorts added.
        -------

        Dictionary with entries containing InputPorts and/or OutputPorts added

        """
        from psyneulink.core.components.ports.port import _parse_port_type
        from psyneulink.core.components.ports.inputport import InputPort, _instantiate_input_ports
        from psyneulink.core.components.ports.outputport import OutputPort, _instantiate_output_ports

        context = Context(source=ContextFlags.METHOD)

        # Put in list to standardize treatment below
        if not isinstance(ports, list):
            ports = [ports]

        input_ports = []
        output_ports = []
        instantiated_input_ports = None
        instantiated_output_ports = None

        for port in ports:
            # FIX: 11/9/17: REFACTOR USING _parse_port_spec
            port_type = _parse_port_type(self, port)
            if (isinstance(port_type, InputPort) or
                    (inspect.isclass(port_type) and issubclass(port_type, InputPort))):
                input_ports.append(port)

            elif (isinstance(port_type, OutputPort) or
                  (inspect.isclass(port_type) and issubclass(port_type, OutputPort))):
                output_ports.append(port)

        if input_ports:
            added_variable, added_input_port = self._handle_arg_input_ports(input_ports)
            if added_input_port:
                if not isinstance(self.defaults.variable, list):
                    old_variable = self.defaults.variable.tolist()
                else:
                    old_variable = self.defaults.variable
                old_variable.extend(added_variable)
                self.defaults.variable = convert_to_np_array(old_variable)
            instantiated_input_ports = _instantiate_input_ports(self,
                                                                  input_ports,
                                                                  added_variable,
                                                                  context=context)
            for port in instantiated_input_ports:
                if port.name is port.componentName or port.componentName + '-' in port.name:
                        port._assign_default_port_Name(context=context)
            # self._instantiate_function(function=self.function)
        if output_ports:
            instantiated_output_ports = _instantiate_output_ports(self, output_ports, context=context)

        if update_variable:
            self._update_default_variable(self.input_values, context)

        return {INPUT_PORTS: instantiated_input_ports,
                OUTPUT_PORTS: instantiated_output_ports}

    @tc.typecheck
    def remove_ports(self, ports, context=REMOVE_PORTS):
        """
        remove_ports(ports)

        Remove one or more `Ports <Port>` from the Mechanism.  Only `InputPorts <InputPort> and `OutputPorts
        <OutputPort>` can be removed; `ParameterPorts <ParameterPort>` cannot be removed from a Mechanism.

        Each Specified port must be owned by the Mechanism, otherwise the request is ignored.

        .. note::
            Removing InputPorts from a Mechanism changes the size of its `variable <Mechanism_Base.variable>`
            attribute, which may produce an incompatibility with its `function <Mechanism_Base.function>` (see
            `Mechanism InputPorts <Mechanism_InputPorts>` for more detailed information).

        Arguments
        ---------

        ports : Port or List[Port]
            one more ports to be removed from the Mechanism.
            Port specification(s) can be an Port object or the name of one.

        """
        # from psyneulink.core.components.ports.inputPort import INPUT_PORT
        from psyneulink.core.components.ports.outputport import OutputPort

        # Put in list to standardize treatment below
        if not isinstance(ports, (list, ContentAddressableList)):
            ports = [ports]

        def delete_port_Projections(proj_list, port):
            for proj in proj_list:
                try:
                    type(proj)._delete_projection(proj)
                except:
                    raise MechanismError(f"PROGRAM ERROR: {proj} not found when removing {port} from {self.name}.")

        for port in ports:

            delete_port_Projections(port.mod_afferents.copy(), port)

            if port in self.input_ports:
                if isinstance(port, str):
                    port = self.input_ports[port]
                index = self.input_ports.index(port)
                delete_port_Projections(port.path_afferents.copy(), port)
                del self.input_ports[index]
                # If port is subclass of OutputPort:
                #    check if regsistry has category for that class, and if so, use that
                category = INPUT_PORT
                class_name = port.__class__.__name__
                if class_name != INPUT_PORT and class_name in self._portRegistry:
                    category = class_name
                remove_instance_from_registry(registry=self._portRegistry,
                                              category=category,
                                              component=port)
                old_variable = self.defaults.variable
                old_variable = np.delete(old_variable,index,0)
                self.defaults.variable = old_variable

            elif port in self.parameter_ports:
                if isinstance(port, ParameterPort):
                    index = self.parameter_ports.index(port)
                else:
                    index = self.parameter_ports.index(self.parameter_ports[port])
                del self.parameter_ports[index]
                remove_instance_from_registry(registry=self._portRegistry,
                                              category=PARAMETER_PORT,
                                              component=port)

            elif port in self.output_ports:
                if isinstance(port, OutputPort):
                    index = self.output_ports.index(port)
                else:
                    index = self.output_ports.index(self.output_ports[port])
                delete_port_Projections(port.efferents.copy(), port)
                del self.output_values[index]
                del self.output_ports[port]
                # If port is subclass of OutputPort:
                #    check if regsistry has category for that class, and if so, use that
                category = OUTPUT_PORT
                class_name = port.__class__.__name__
                if class_name != OUTPUT_PORT and class_name in self._portRegistry:
                    category = class_name
                remove_instance_from_registry(registry=self._portRegistry,
                                              category=category,
                                              component=port)

        self.defaults.variable = self.input_values

    def _delete_mechanism(mechanism):
        mechanism.remove_ports(mechanism.input_ports)
        mechanism.remove_ports(mechanism.parameter_ports)
        mechanism.remove_ports(mechanism.output_ports)
        # del mechanism.function
        remove_instance_from_registry(MechanismRegistry, mechanism.__class__.__name__,
                                      component=mechanism)

    def _get_standardized_label_dicts(self):
        """
        Gets dict of Mechanism's input and output port labels in a standardized form

        Returns
        -------
            dict
                .. parsed-literal::
                    {
                        INPUT_PORTS:
                                {(int) port_index:
                                        {{label_1: value_1},
                                         {label_2: value_2}}
                                },
                        OUTPUT_PORTS:
                                {(int) port_index:
                                        {{label_1: value_1},
                                         {label_2: value_2}}
                                }
                    }

        """
        input_labels = self._get_standardized_label_dict(INPUT)
        output_labels = self._get_standardized_label_dict(INPUT)
        port_labels = {
            INPUT_PORTS: input_labels,
            OUTPUT_PORTS: output_labels
        }
        return port_labels

    def _get_standardized_label_dict(self, label_type):
        """
        Parses input or output label dicts into a standardized form

        Parameters
        ----------
        (str) port_type: INPUT or OUTPUT keyword, specifying the type of labels to parse and return

        port_type

        Returns
        -------
        dict
            .. parsed-literal::
                {INPUT_PORTS/OUTPUT_PORTS:
                        {(int) port_index:
                                {{label_1: value_1},
                                 {label_2: value_2}}
                        }
                }
        """
        if label_type == INPUT:
            label_dict = self.input_labels_dict
            ports = self.input_ports
        elif label_type == OUTPUT:
            label_dict = self.output_labels_dict
            ports = self.output_ports
        _label_dict = {}
        if label_dict:
            for k, v in label_dict.items():
                if isinstance(k, Number):
                    _label_dict[k] = v
                elif type(v) == dict:
                    i = ports[k].position_in_mechanism
                    _label_dict[i] = v
                else:
                    if 0 not in _label_dict:
                        _label_dict[0] = {}
                    _label_dict[0].update({k:v})
        return _label_dict

    def get_input_port_position(self, port):
        if port in self.input_ports:
            return self.input_ports.index(port)
        raise MechanismError("{} is not an InputPort of {}.".format(port.name, self.name))

    # @tc.typecheck
    # def _get_port_value_labels(self, port_type:tc.any(InputPort, OutputPort)):
    def _get_port_value_labels(self, port_type, context=None):
        """Return list of labels for the value of each Port of specified port_type.
        If the labels_dict has subdicts (one for each Port), get label for the value of each Port from its subdict.
        If the labels dict does not have subdicts, then use the same dict for the only (or all) Port(s)
        """

        if port_type is InputPort:
            ports = self.input_ports

        elif port_type is OutputPort:
            ports = self.output_ports

        labels = []
        for port in ports:
            labels.append(port.get_label(context))
        return labels

    @property
    def input_port(self):
        return self.input_ports[0]

    @property
    def input_values(self):
        try:
            return self.input_ports.values
        except (TypeError, AttributeError):
            return None

    def get_input_values(self, context=None):
        input_values = []
        for input_port in self.input_ports:
            if "LearningSignal" in input_port.name:
                input_values.append(input_port.parameters.value.get(context).flatten())
            else:
                input_values.append(input_port.parameters.value.get(context))
        return input_values

    @property
    def external_input_ports(self):
        try:
            return [input_port for input_port in self.input_ports if not input_port.internal_only]
        except (TypeError, AttributeError):
            return None

    @property
    def external_input_values(self):
        try:
            return [input_port.value for input_port in self.input_ports if not input_port.internal_only]
        except (TypeError, AttributeError):
            return None

    @property
    def default_external_input_values(self):
        try:
            return [input_port.defaults.value for input_port in self.input_ports if not input_port.internal_only]
        except (TypeError, AttributeError):
            return None

    @property
    def input_labels(self):
        """
        Returns a list with as many items as there are InputPorts of the Mechanism. Each list item represents the value
        of the corresponding InputPort, and is populated by a string label (from the input_labels_dict) when one
        exists, and the numeric value otherwise.
        """
        return self.get_input_labels()

    def get_input_labels(self, context=None):
        if self.input_labels_dict:
            return self._get_port_value_labels(InputPort, context)
        else:
            return self.get_input_values(context)

    @property
    def parameter_ports(self):
        return self._parameter_ports

    @property
    def output_port(self):
        return self.output_ports[0]

    @property
    def output_values(self):
        return self.output_ports.values

    def get_output_values(self, context=None):
        return [output_port.parameters.value.get(context) for output_port in self.output_ports]

    @property
    def output_labels(self):
        """
        Returns a list with as many items as there are OutputPorts of the Mechanism. Each list item represents the
        value of the corresponding OutputPort, and is populated by a string label (from the output_labels_dict) when
        one exists, and the numeric value otherwise.
        """
        return self.get_output_labels()

    def get_output_labels(self, context=None):
        if self.output_labels_dict:
            return self._get_port_value_labels(OutputPort, context)
        else:
            return self.get_output_values(context)

    @property
    def ports(self):
        """Return list of all of the Mechanism's Ports"""
        return ContentAddressableList(
                component_type=Port,
                list=list(self.input_ports) +
                     list(self.parameter_ports) +
                     list(self.output_ports))

    @property
    def path_afferents(self):
        """Return list of the `path_afferents <Port_Base.path_afferents>` for all of the Mechanism's input_ports"""
        projs = []
        for input_port in self.input_ports:
            projs.extend(input_port.path_afferents)
        return ContentAddressableList(component_type=Projection, list=projs)

    @property
    def mod_afferents(self):
        """Return all of the Mechanism's afferent modulatory Projections"""
        projs = []
        for input_port in self.input_ports:
            projs.extend(input_port.mod_afferents)
        for parameter_port in self.parameter_ports:
            projs.extend(parameter_port.mod_afferents)
        for output_port in self.output_ports:
            projs.extend(output_port.mod_afferents)
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
            for output_port in self.output_ports:
                projs.extend(output_port.efferents)
        except TypeError:
            # self.output_ports might be None
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
    def _dependent_components(self):
        return list(itertools.chain(
            super()._dependent_components,
            self.input_ports,
            self.output_ports,
            self.parameter_ports,
        ))

    @property
    def _dict_summary(self):
        inputs_dict = {
            MODEL_SPEC_ID_INPUT_PORTS: [
                s._dict_summary for s in self.input_ports
            ]
        }
        inputs_dict[MODEL_SPEC_ID_INPUT_PORTS].extend(
            [s._dict_summary for s in self.parameter_ports]
        )

        outputs_dict = {
            MODEL_SPEC_ID_OUTPUT_PORTS: [
                s._dict_summary for s in self.output_ports
            ]
        }

        return {
            **super()._dict_summary,
            **inputs_dict,
            **outputs_dict
        }


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

class MechanismList(UserList):
    """Provides access to Mechanisms and their attributes in a list Mechanisms of an owner.

    Properties return dicts with item : attribute pairs.
    Recursively process any item that itself is a MechanismList (e.g., a `Nested Composition <Composition_Nested>`.

    Attributes
    ----------
    mechanisms : List[item]

    names : List[str | Dict[str:List[str]]
        each item is an item name or a dict with one item as its key and a list of subitem names as its value.

    values : Dict[str:value]
        each entry is an item name : value pair.

    input_port_names : Dict[str:List[str]]
        each entry is either an item name with a list of its `InputPort` `names <InputPort.name>` or, if the item is
        a nested MechanismList, then a dict with the name of the nested item and a dict with item names and a list of
        their InputPort names.

    input_port_values : Dict[str:Dict[str:value]]
        each entry is either an item name with a dict of `InputPort` `name <InputPort.name>`:`value <InputPort.value>`
        pairs or, if the item is a nested MechanismList, then a dict with the name of the nested item and a dict
        with its InputPort name:value pairs.

    parameter_port_names : Dict[str:List[str]]
        each entry is either an item name with a list of its `ParameterPort` `names <ParameterPort.name>` or, if the
        item is a nested MechanismList, then a dict with the name of the nested item and a dict with item names and a
        list of their ParameterPort names.

    parameter_port_values : Dict[str:Dict[str:value]]
        each entry is either an item name with a dict of `ParameterPort` `name <ParameterPort.name>`:`value
        <ParameterPort.value>` pairs or, if the item is a nested MechanismList, then a dict with the name of the
        nested item and a dict with its ParameterPort name:value pairs.

    output_port_names : Dict[str:List[str]]
        each entry is either an item name with a list of its `OutputPort` `names <OutputPort.name>` or, if the item is
        a nested MechanismList, then a dict with the name of the nested item and a dict with item names and a list of
        their OutputPort names.

    output_port_values : Dict[str:Dict[str:value]]
        each entry is either an item name with a dict of `OutputPort` `name <OutputPort.name>`:`value
        <OutputPort.value>` pairs or, if the item is a nested MechanismList, then a dict with the name of the nested
        item and a dict with its OutputPort name:value pairs.

    """

    def __init__(self, owner, components_list:list):
        super().__init__()
        self.mechs = components_list
        self.data = self.mechs
        self.owner = owner

    def __getitem__(self, item):
        """Return specified Mechanism in MechanismList"""
        return self.mechs[item]

    def __setitem__(self, key, value):
        raise ("MechanismList is read only ")

    def __len__(self):
        return (len(self.mechs))

    def __call__(self):
        return self.data

    def _get_attributes_dict(self, mech_list_attr_name, item_attr_name, sub_attr_name=None, values_only=False):
        """Generate dict of {item.name:item attribute value} pairs in "human readable" form.
        Call recursively if item is itself a MechanismList.
        """
        ret_dict = {}
        for item in self.mechanisms:
            if isinstance(item, Mechanism):
                attr_val = getattr(item, item_attr_name)
                if isinstance(attr_val, (list, ContentAddressableList)):
                    assert sub_attr_name, f"Need to specify sub_attr for attributs that are a list"
                    if sub_attr_name == 'name':
                        sub_items = []
                        for sub_item in attr_val:
                            sub_items.append(getattr(sub_item, sub_attr_name))
                    else:
                        sub_items = {}
                        for sub_item in attr_val:
                            sub_items[sub_item.name] = getattr(sub_item, sub_attr_name)
                    ret_dict[item.name] = sub_items
                elif not sub_attr_name:
                    ret_dict[item.name] = attr_val
                else:
                    ret_dict[item.name] = getattr(attr_val, sub_attr_name)
            else:
                ret_dict[item.owner.name] = getattr(item, mech_list_attr_name)
        if values_only:
            # return list(ret_dict.values())
            return [k if isinstance(v, str) else {k:v} for k,v in ret_dict.items()]
        else:
            return ret_dict

    @property
    def mechs_sorted(self):
        """Return list of Mechanisms sorted by Mechanisms' names"""
        return sorted(self.mechs, key=lambda object_item: object_item.name)

    @property
    def mechanisms(self):
        """Return list of all Mechanisms in MechanismList"""
        return list(self)

    @property
    def names(self):
        """Return dictwith names of all Mechanisms in MechanismList"""
        return self._get_attributes_dict('names', 'name', values_only=True)

    @property
    def values(self):
        """Return dict with values of all Mechanisms in MechanismList"""
        return self._get_attributes_dict('values', 'value')

    @property
    def input_port_names(self):
        """Return dict with names of all OutputPorts for all Mechanisms in MechanismList"""
        return self._get_attributes_dict('input_port_names', 'input_ports', 'name')

    @property
    def input_port_values(self):
        """Return dict with values of OutputPorts for all Mechanisms in MechanismList"""
        return self._get_attributes_dict('input_port_values', 'input_ports', 'value')

    @property
    def input_values(self):
        """Return dict with input_values for all Mechanisms in MechanismList"""
        return self._get_attributes_dict('values', 'value')

    @property
    def parameter_port_names(self):
        """Return dict with names of all OutputPorts for all Mechanisms in MechanismList"""
        return self._get_attributes_dict('parameter_port_names', 'parameter_ports', 'name')

    @property
    def parameter_port_values(self):
        """Return dict with values of OutputPorts for all Mechanisms in MechanismList"""
        return self._get_attributes_dict('parameter_port_values', 'parameter_ports', 'value')

    @property
    def output_port_names(self):
        """Return dict with names of all OutputPorts for all Mechanisms in MechanismList"""
        return self._get_attributes_dict('output_port_names', 'output_ports', 'name')

    @property
    def output_port_values(self):
        """Return dict with values of OutputPorts for all Mechanisms in MechanismList"""
        return self._get_attributes_dict('output_port_values', 'output_ports', 'value')

    @property
    def output_values(self):
        """Return dict with output_values for all Mechanisms in MechanismList"""
        return self._get_attributes_dict('values', 'value')
