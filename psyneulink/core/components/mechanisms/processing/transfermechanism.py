# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# FIX NOTES:
#  * NOW THAT NOISE AND INTEGRATION_RATE ARE Parameters THAT DIRECTLY REFERENCE integrator_function,
#      SHOULD THEY NOW BE VALIDATED ONLY THERE (AND NOT IN TransferMechanism)??
#  * ARE THOSE THE ONLY TWO integrator PARAMS THAT SHOULD BE Parameters??

# ********************************************  TransferMechanism ******************************************************

"""

Contents
--------

  * `TransferMechanism_Overview`
  * `TransferMechanism_Creation`
  * `TransferMechanism_Structure`
        - `TransferMechanism_InputPorts`
        - `TransferMechanism_Function`
        - `TransferMechanism_OutputPorts`
  * `TransferMechanism_Execution`
        - `Without Integration <TransferMechanism_Execution_Without_Integration>`
        - `With Integration <TransferMechanism_Execution_With_Integration>`
             • `TransferMechanism_Execution_Integration_Initialization`
             • `TransferMechanism_Execution_Integration`
             • `TransferMechanism_Execution_Integration_Termination`
  * `TransferMechanism_Examples`
        - `Creation <TransferMechanism_Examples_Creation>`
        - `Execution <TransferMechanism_Examples_Execution>`
            • `Without Integration <TransferMechanism_Examples_Execution_Without_Integration>`
            • `With Integration <TransferMechanism_Examples_Execution_With_Integration>`
                - `Initializing, Resetting and Resuming Integration
                  <TransferMechanism_Examples_Initialization_and_Resetting>`
                - `Terminating Integration <TransferMechanism_Examples_Termination>`
  * `TransferMechanism_Class_Reference`

.. _TransferMechanism_Overview:

Overview
--------

A TransferMechanism is a subclass of `ProcessingMechanism` that adds the ability to integrate its input.

As a ProcessingMechanism, it transforms its input using a simple mathematical function that maintains the shape of its
input.  The input can be a single scalar value, a simple list or array, or a multidimensional one (regular or ragged).
The function used to carry out the transformation can be a `TransferFunction` or a `custom one <UserDefinedFunction>`
that can accept any of these forms of input and generate one of similar form.  A TransferMechanism can also add `noise
<TransferMechanism.noise>` to and/or `clip <TransferMechanism.clip>` the result of its function.

A TransferMechanism has two modes of operation: `without integration
<TransferMechanism_Execution_Without_Integration>` and `with integration enabled
<TransferMechanism_Execution_With_Integration>`.
Integration is disabled by default, so that the Mechanism's `function <Mechanism_Base.function>` executes a full
("instantaneous") transformation of its input on each execution (akin to the standard practice in feedforward neural
networks). However, if integration is enabled, then it uses its `integrator_function
<TransferMechanism.integrator_function>` to integrate its input on each execution, before passing the result on to
its `function <Mechanism_Base.function>` for transformation (akin to time-averaging the net input to a unit in a
neural network before passing that to its activation function). When integration is enabled, using the `integrator_mode
<TransferMechanism.integrator_mode>` Parameter, additional parameters can be used to configure the integration process,
including how it is `initialized <TransferMechanism_Execution_Integration_Initialization>` and when it `terminates
<TransferMechanism_Execution_Integration_Termination>`.

.. _TransferMechanism_Creation:

Creating a TransferMechanism
-----------------------------

The primary arguments that determine the operation of a TransferMechanism are its **function** argument,
that specifies the `function <Mechanism_Base.function>` used to transform its input; and, if **integrator**
mode is set to True, then its *integrator_function** argument and associated ones that specify how `integration
<TransferMechanism_Execution_With_Integration>` occurs (see `TransferMechanism_Examples`).

*Primary Function*
~~~~~~~~~~~~~~~~~~

By default, the primary `function <Mechanism_Base.function>` of a TransferMechanism is `Linear`, however the
**function** argument can be used to specify any subclass or instance of `TransferFunction`. It can also be any
python function or method, so long as it can take a scalar, or a list or array of numerical values as input and
produce a result that is of the same shape;  the function or method is "wrapped" as `UserDefinedFunction`,
assigned as the TransferMechanism's `function <Mechanism_Base.function>` attribute.

*Integator Function*
~~~~~~~~~~~~~~~~~~~~

By default, the `integrator_function <TransferMechanism.integrator_function>` of a
TransferMechanism is `AdaptiveIntegrator`,  however the **integrator_function** argument of the Mechanism's constructor
can be used to specify any subclass of `IntegratorFunction`, so long as it can accept as input the TransferMechanism's
`variable <Mechanism_Base.variable>`, and genereate a result of the same shape that is passed to the Mechanism's
`function <Mechanism_Base.function>`.  In addition to specifying parameters in the constructor for an
`IntegratorFunction` assigined to the **integrator_function** argument, the constructor for the TransferMechanism
itself has arguments that can be used to confifure its `integrator_function <TransferMechanism.integrator_function>`:
**initial_value**, **integration_rate**, and **noise**.  If any of these are specified in the TransferMechanism's
constructor, their value is used to specify the corresponding parameter of its `integrator_function
<TransferMechanism.integrator_function>`. Additonal parameters that govern how integration occurs are described under
`TransferMechanism_Execution_With_Integration`.

.. _TransferMechanism_Structure:

Structure
---------

.. _TransferMechanism_InputPorts:

*InputPorts*
~~~~~~~~~~~~~

By default, a TransferMechanism has a single `InputPort`;  however, more than one can be specified
using the **default_variable** or **size** arguments of its constructor (see `Mechanism`).  The `value
<InputPort.value>` of each InputPort is used as a separate item of the Mechanism's `variable
<Mechanism_Base.variable>`, and transformed independently by its `function <Mechanism_Base.function>`.

.. _TransferMechanism_Function:

*Functions*
~~~~~~~~~~~

A TransferMechanism has two functions:  its primary `function <Mechanism_Base.function>` that transforms its
input, and an `integrator_function <TransferMechanism.integrator_function>` that is used to integrate the input
before passing it to the primary `function <Mechanism_Base.function>` when `integrator_mode
<TransferMechanism.integrator_mode>` is set to True. The default function for a TransferMechanism is `Linear`,
and the defult for its `integrator_function <TransferMechanism.integrator_function>` is `AdaptiveIntegrator`,
how custom functions can be assigned, as described under `TransferMechanism_Creation`.

.. _TransferMechanism_OutputPorts:

*OutputPorts*
~~~~~~~~~~~~~

By default, or if the **output_ports** argument is specified using the keyword *RESULTS*, a TransferMechanism generates
one `OutputPort` for each item in the outer dimension (axis 0) of its `value <Mechanism_Base.value>` (each of which is
the result of the Mechanism's `function <Mechanism_Base.function>` (and possibly its `integrator_function
<TransferMechanism.integrator_function>`) applied to the `value <InputPort.value>` of the corresponding `InputPort`).
If there is only one OutputPort (i.e., the case in which there is only one InputPort and
therefore only one item in Mechanism's `value <Mechanism_Base.value>`), the OutputPort is named *RESULT*.  If there is
more than one item in `value <Mechanism_Base.value>`, then an OuputPort is assigned for each;  the name of the first
is *RESULT-0*, and the names of the subsequent ones are suffixed with an integer that is incremented for each successive
one (e.g., *RESULT-1*, *RESULT-2*, etc.).  Additional OutputPorts can be assigned using the TransferMechanism's
`standard_output_ports <TransferMechanism.standard_output_ports>` (see `OutputPort_Standard`) or by creating `custom
OutputPorts <OutputPort_Customization>` (but see note below).

    .. _TransferMechanism_OutputPorts_Note:

    .. note::
       If any OutputPorts are specified in the **output_ports** argument of the TransferMechanism's constructor,
       then, `as with any Mechanism <Mechanism_Default_Port_Suppression_Note>`, its default OutputPorts are not
       automatically generated.  Therefore, an OutputPort with the appropriate `index <OutputPort.index>` must be
       explicitly specified for each and every item of the Mechanism's `value <Mechanism_Base.value>` (corresponding
       to each InputPort) for which an OutputPort is needed.

.. _TransferMechanism_Execution:

Execution
---------

A TransferMechanism has two modes of execution, determined by its `integrator_mode
<TransferMechanism.integrator_mode>` parameter.  By default (`integrator_mode
<TransferMechanism.integrator_mode>` = False) it `executes without integration
<TransferMechanism_Execution_Without_Integration>`, directly transforming its input using its `function
<Mechanism_Base.function>`, and possibly adding `noise <TransferMechanism.noise>` to and/or `clipping
<TransferMechanism.clip>` the result.  If `integrator_mode <TransferMechanism.integrator_mode>` = True,
it `executes with integration <TransferMechanism_Execution_With_Integration>`, by integrating its input before
transforming it.  Each of these is described in more detail below.

.. _TransferMechanism_Execution_Without_Integration:

*Execution Without Integration*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If `integrator_mode <TransferMechanism.integrator_mode>` is False (the default), the input received over
`input_ports <Mechanism_Base.input_ports>` (assigned as `variable <Mechanism_Base.variable>`) is passed
directly to `function <Mechanism_Base.function>`.  If either the `noise <TransferMechanism.noise>` or
`clip <TransferMechanism.clip>` `Parameters` have been specified, they are applied to the result of `function
<Mechanism_Base.function>`. That is then assigned as the Mechanism's `value <Mechanism_Base.value>`, and well as the
`values <OutputPort.value>` of its `output_ports <Mechanism_Base.output_ports>`, each of which represents the
transformed value of the corresponding `input_ports <Mechanism_Base.input_ports>` (see `examples
<TransferMechanism_Examples_Execution_Without_Integration>`).

.. _TransferMechanism_Execution_With_Integration:

*Execution With Integration*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If `integrator_mode <TransferMechanism.integrator_mode>` is True, the TransferMechanism's input (`variable
<Mechanism_Base.variable>`) is first passed to its `integrator_function <TransferMechanism.integrator_function>`,
the result of which is then passed to its primary `function <Mechanism_Base.function>`.  The TransferMechanis has
several `Parameters` that, in addition to those of its `integrator_function <TransferMechanism.integrator_function>`,
can be used to configure the integration process, as described in the following subsections (also see `examples
<TransferMechanism_Examples_Execution_With_Integration>`).

.. _TransferMechanism_Execution_Integration_Initialization:

**Initialization, Resetting and Resuming Integration**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The staring point for integration can be initialized and reset, and also configured to resume in various ways, as
described below (also `examples <TransferMechanism_Examples_Initialization_and_Resetting>`).

*Initializing integration* -- by default, the the starting point for integration is the Mechanism's `default_variable
<Component_Variable>`, and is usually an appropriately shaped array of 0's.  However, the starting point can be
specified using the **initializer** argument of a TransferMechanism's constructor.

  .. note::
    The value of **initializer** is passed to the `integrator_function <TransferMechanism.integrator_function>` as its
    `initializer <IntegratorFunction.initializer>` `Parameter <Parameters>`.  It can also be specified directly in the
    **initializer** argument of the constructor for an  `IntegratorFunction` assigned to the **integrator_function**
    argument of a TransferMechanism's constructor.  If there is a disagreements between these (i.e., between the
    specifiation of **initial_value** for the TransferMechanism and **initializer** for its `integrator_function
    <TransferMechanism.integrator_function>`, the value specified for the latter takes precedence, and that value is
    assigned as the one for the `initial_value <TransferMechanism.initial_value>` of the TransferMechanism.

*Resetting integration* -- in some cases, it may be useful to reset the integration to the original starting point,
or to a new one. This can be done using the Mechanism's `reset <TransferMechanism.reset>` method. This first sets the
`integrator_function <TransferMechanism.integrator_function>`'s `previous_value <IntegratorFunction.previous_value>`
and `value <Component.value>` attributes to the specified value. That is then passed to the Mechanism's `function
<Mechanism_Base.function>` which is executed, and the result is assigned as the Mechanism current `value
<Mechanism_Base.value>` and to its `output_ports <Mechanism_Base.output_ports>`.

  .. note::
     The TransferMechanism's `reset <TransferMechanism.reset>` method calls the reset method on its
     `integrator_function <TransferMechanism.integrator_function>`, which can also be called directly. The key
     difference is that calling the Mechanism's `reset <TransferMechanism.reset>` method also executes the Mechanism's
     `function <Mechanism_Base.function>` and updates its `output_ports <Mechanism_Base.output_ports>`. This is
     useful if the Mechanism's `value <Mechanism_Base.value>` or that of any of its `output_ports
     <Mechanism_Base.output_ports>` will be used or checked *before* the Mechanism is next executed. This may be
     true if, for example, the Mechanism is a `RecurrentTransferMechanism`, or if a `Scheduler` `Condition` depends on
     it.

.. _TransferMechanism_Execution_Integration_Resumption:

*Resuming integration* -- integration can be enabled and disabled between executions by setting `integrator_mode
<TransferMechanism.integrator_mode>` to True and False, respectively.  When re-enabling  integration, the value used
by the `integrator_function <TransferMechanism.integrator_function>` for resuming integration can be configured using
the TransferMechanism's `on_resume_integrator_mode <TransferMechanism.on_resume_integrator_mode>` Parameter; there are
three options for this:

    * *CURRENT_VALUE* - use the current `value <Mechanism_Base.value>` of the Mechanism as the starting value for
      resuming integration;

    * *LAST_INTEGRATED_VALUE* - resume integration with whatever the `integrator_function
      <TransferMechanism.integrator_function>`' `previous_value <IntegratorFunction.previous_value>` was when
      `integrator_mode <TransferMechanism.integrator_mode>` was last True;

    * *RESET* - call the `integrator_function <TransferMechanism.integrator_function>`\\s `reset` method,
      so that integration resumes using `initial_value <TransferMechanism.initial_value>` as its starting value.

.. _TransferMechanism_Execution_Integration:

**Integration**
^^^^^^^^^^^^^^^

On each execution of the Mechanism, its `variable <Mechanism_Base.variable>` is passed to the `integrator_function
<TransferMechanism.integrator_function>`, which integrates this with the function's `previous_value
<IntegratorFunction.previous_value>`, using the Mechanism's `noise <TransferMechanism.noise>` and
`integration_rate <TransferMechanism.integration_rate>` parameters.

    .. note::
       Like the TransferMechanism's `initial_value <TransferMechanism.initial_value>`, its `noise
       <TransferMechanism.noise>` and `integration_rate <TransferMechanism.integration_rate>` `Parameters` are used to
       specify the `noise <IntegratorFunction.noise>` and `initializer <IntegratorFunction.initializer>` Parameters
       of  its `integrator_function <TransferMechanism.integrator_function>`, respectively. If there are any
       disagreements between these (e.g., any of these parameters is specified with conflicting values for the
       TransferMechanism and its `integrator_function <TransferMechanism.integrator_function>`), the values specified
       for the `integrator_function <TransferMechanism.integrator_function>` take precedence, and those value(s) are
       assigned as the ones for the corresponding Parameters of the TransferMechanism.

After the `integrator_function <TransferMechanism.integrator_function>` executes, its result is passed to the
Mechanism's primary `function <Mechanism_Base.function>`, and its `clip <TransferMechanism.clip>` parameter is applied
if specified, after which it is assigned to as the TransferMechanism's `value <Mechanism_Base.value>` and that of its
`output_ports <Mechanism_Base.output_ports>`.

.. _TransferMechanism_Execution_Integration_Termination:

**Termination**
^^^^^^^^^^^^^^^

If `integrator_mode <TransferMechanism.integrator_mode>` is True then, for each execution of the TransferMechanism, it
can be configured to conduct a single step of integration, or to continue to integrate during that execution until its
termination condition is met.  The latter is specified by the TransferMechanism's `execute_until_finished
<Component.execute_until_finished>` as well as its `termination_threshold
<TransferMechanism.termination_threshold>`, `termination_measure <TransferMechanism.termination_measure>`, and
`termination_comparison_op <TransferMechanism.termination_comparison_op>` `Parameters`.  These configurations are
described below (also see `examples <TransferMechanism_Examples_Termination>`).

*Single step execution* -- If either `execute_until_finished <Component.execute_until_finished>` is set to False,
or no `termination_threshold <TransferMechanism.termination_threshold>` is specified (i.e., it is None, the default),
then only a signle step of integration is carried out each time the TransferMechanism is executed.  In this case,
the `num_executions_before_finished <Component.num_executions_before_finished>` attribute remains equal to 1,
since the `integrator_function <TransferMechanism.integrator_function>` is executed exactly once per call to the
`execute method <Component_Execution>` (and the termination condition does not apply or has not been specified).

*Execute to termination* -- if `execute_until_finished <Component.execute_until_finished>` is True and a value is
specified for the `termination_threshold <TransferMechanism.termination_threshold>` then, during each execution of
the TransferMechanism, it repeatedly calls its `integrator_function <TransferMechanism.integrator_function>`
and primary `function <Mechanism_Base.function>`, using the same input (`variable <Mechanism_Base.variable>`) until
its `termination condition <Transfer_Mechanism_Termination_Condition>`, or the number of executions reaches
`max_executions_before_finished <Component.max_executions_before_finished>`.  The numer of executions that have
taken place since the last time the termination condition was met is contained in `num_executions_before_finished
<Component.num_executions_before_finished>`, and is reset to 0 each time the termination condition is met.

   .. _TransferMechanism_Continued_Execution:

   .. note::
     Even after its termination condition is met, a TransferMechanism will continue to execute if it is called again,
     carrying out one step of integration each time it is called. This can be useful in cases where the initial
     execution of the Mechanism is meant to bring it to some state (e.g., as an initial "settling process"), after
     which subsequent executions are meant to occur in step with the execution of other Mechanisms in a Composition
     (see `example <TransferMechanism_Examples_Termination_By_Time>` below).

.. _Transfer_Mechanism_Termination_Condition:

By default, `execute_until_finished <Component.execute_until_finished>` is True, so that when `integrator_mode
<TranserMechanism.integrator_mode>` is set to True a TransferMechanism will execute until it terminates, using a
`convergence criterion <TransferMechanism_Convergence_Termination>`.  However, the Mechanism's method of termination
can be configured using its `termination_measure <TransferMechanism.termination_measure>` and `termination_comparison_op
<TransferMechanism.termination_comparison_op>` `Parameters` can be used to congifure other termination conditions.
There are two broad types of termination condition: convergence and boundary termination.

.. _TransferMechanism_Convergence_Termination:

*Convergence termination* -- execution terminates based on the difference between the TransferMechanism's current
`value <Mechanism_Base.value>` and its previous_value. This is implemented by specifying `termination_measure
<TransferMechanism.termination_measure>` with a function that accepts a 2d array with *two items* (1d arrays) as its
argument, and returns a scalar (the default for a TransferMechanism is the `Distance` Function with `MAX_ABS_DIFF`
as its metric).  After each execution, the function is passed the Mechanism's current
`value <Mechanism_Base.value>` as well as its previous_value, and the scalar returned is compared to
`termination_threshold <TransferMechanism.termination_threshold>` using the comparison
operator specified by  `termination_comparison_op <TransferMechanism.termination_comparison_op>` (which is
*LESS_THAN_OR_EQUAL* by default).  Execution continues until this returns True. A `Distance` Function with other
metrics (e.g., *ENERGY* or *ENTROPY*) can be specified as the **termination_measure**, as can any other function
that accepts a single argument that is a 2d array with two entries.

.. _TransferMechanism_Boundary_Termination:

*Boundary termination* -- Two types of boundaries can be specified:  value or time.

    .. _TransferMechanism_Termination_By_Value:

    *Termination by value*.  This terminates execution when the Mechanism's `value <Mechanism_Base.value>` reaches the
    the value specified by the `termination_threshold <TransferMechanism.termination_threshold>` Parameter.  This is
    implemented by specifying `termination_measure <TransferMechanism.termination_measure>` with a function that
    accepts a 2d array with a *single entry* as its argument and returns a scalar.  The single entry is the
    TransferMechanism's current `value <Mechanism_Base.value>` (that is, `previous_value
    <Mechanism_Base.previous_value>` is ignored). After each execution, the function is passed the Mechanism's
    current `value <Mechanism_Base.value>`, and the scalar returned is compared to `termination_threshold
    <TransferMechanism.termination_threshold>` using the comparison operator specified by `termination_comparison_op
    <TransferMechanism.termination_comparison_op>`. Execution continues until this returns True.

    .. _TransferMechanism_Termination_By_Time:

    *Termination by time*.  This terminates execution when the Mechanism has executed at least a number of times equal
    to `termination_threshold <TransferMechanism.termination_threshold>` at a particular TimeScale (e.g., within a
    `RUN` or a `TRIAL <TimeScale.TRIAL>`). This is specified by assigning a `TimeScale` to `termination_measure
    <TransferMechanism.termination_measure>`;  execution terminates when the number of
    executions at that TimeScale equals the `termination_threshold <TransferMechanism.termination_threshold>`.
    Note that, in this case, `termination_comparison_op <TransferMechanism.termination_comparison_op>` is automatically
    set to *GREATER_THAN_OR_EQUAL*.

.. _TransferMechanism_Examples:

Examples
--------

    - `Creation <TransferMechanism_Examples_Creation>`
    - `Execution <TransferMechanism_Examples_Execution>`
        • `Without Integration <TransferMechanism_Examples_Execution_Without_Integration>`
        • `With Integration <TransferMechanism_Examples_Execution_With_Integration>`
            - `TransferMechanism_Examples_Initialization_and_Resetting`
            - `TransferMechanism_Examples_Termination`

.. _TransferMechanism_Examples_Creation:

*Examples of Creating a TransferMechanism*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function Specification**

The **function** of a TransferMechanism can be specified as the name of a `Function <Function>` class::

    >>> import psyneulink as pnl
    >>> my_linear_transfer_mechanism = pnl.TransferMechanism(function=pnl.Linear)

or using the constructor for a `TransferFunction`, in which case its `Parameters` can also be specified::

    >>> my_logistic_tm = pnl.TransferMechanism(function=pnl.Logistic(gain=1.0, bias=-4))

**Integrator Mode**

The **integrator_mode** argument allows the TransferMechanism to operate in either an "instantaneous" or
"time averaged" manner. By default, `integrator_mode <TransferMechanism.integrator_mode>` is set to False, meaning
execution is instantaneous. In order to switch to time averaging, the **integrator_mode** argument of the constructor
must be set to True.

    >>> my_logistic_tm = pnl.TransferMechanism(function=pnl.Logistic(gain=1.0, bias=-4),
    ...                                                        integrator_mode=True)

When `integrator_mode <TransferMechanism.integrator_mode>` is True, the TransferMechanism uses its `integrator_function
<TransferMechanism.integrator_function>` to integrate its variable on each execution. The output of the
`integrator_function  <TransferMechanism.integrator_function>` is then used as the input to `function
<Mechanism_Base.function>`.

.. _TransferMechanism_Examples_Execution:

*Examples of Execution*
~~~~~~~~~~~~~~~~~~~~~~~

.. _TransferMechanism_Examples_Execution_Without_Integration:

**Without Integration**
^^^^^^^^^^^^^^^^^^^^^^^

If `integrator_mode <TransferMechanism.integrator_mode>` is False (the default), then the TransferMechanism updates its
`value <Mechanism_Base.value>` and the `value <OutputPort.value>` of its `output_ports <Mechanism_Base.output_ports>`
without using its `integrator_function <TransferMechanism.integrator_function>`, as in the following example::

    # >>> my_mech = pnl.TransferMechanism(size=2)
    # >>> my_mech.execute([0.5, 1])
    # array([[0.5, 1. ]])

    >>> my_logistic_tm = pnl.TransferMechanism(function=pnl.Logistic,
    ...                                        size=3)
    >>> my_logistic_tm.execute([-2.0, 0, 2.0])
    array([[0.11920292, 0.5       , 0.88079708]])

Notice that the result is the full logistic transform of the input (i.e., no integration occured). Noise can also be
added to the result. It can be specified as a float, and array, or function. If it is a float or list of floats,
the value is simply added to the result, as shown in the example below, that uses the TransferMechanism's default
`function <Mechanism_Base>`, `Linear`::

    >>> my_linear_tm = pnl.TransferMechanism(size=3,
    ...                                      noise=2.0)
    >>> my_linear_tm.execute([1.0, 1.0, 1.0])
    array([[3., 3., 3.]])
    >>> my_linear_tm.execute([1.0, 1.0, 1.0])
    array([[3., 3., 3.]])

Since by default `Linear` uses a `slope <Linear.slope>` of ``1`` and `intercept <Linear.intercept` of ``0``,
the result is the same as the input, plus the value specified for **noise**.  A list can also be used to specify
**noise** (it must be the same length as the Mechanism's `variable <Mechanism_Base.variable>`), in which case each
element is applied Hadamard (elementwise) to the result, as shown here::

    >>> my_linear_tm.noise = [1.0,1.2,.9]
    >>> my_linear_tm.execute([1.0, 1.0, 1.0])
    array([[2. , 2.2, 1.9]])

While specifying noise as a constant (or a list of constantss) is not particularly useful, it can be replaced by any
function that specifies a float, for example a `DistributionFunction`.  As with numerical values, if a single function
is specified, it is applied to all elements; however, on each execution, the function is executed indpendently for
each element.  This is shown below using the `NormalDist` function::

    >>> my_linear_tm = pnl.TransferMechanism(size=3,
    ...                                      noise=pnl.NormalDist)
    >>> my_linear_tm.execute([1.0, 1.0, 1.0])
    array([[2.1576537 , 1.60782117, 0.75840058]])
    >>> my_linear_tm.execute([1.0, 1.0, 1.0])
    array([[2.20656132, 2.71995896, 0.57600537]])
    >>> my_linear_tm.execute([1.0, 1.0, 1.0])
    array([[1.03826716, 0.56148871, 0.8394907 ]])

Notice that each element was assigned a different random value for its noise, and that these also varied across
executions.  Notice that since only a single function was specified, it could be the name of a class.  Functions
can also be used in a list to specify **noise**, together with other functions or with numeric values;  however,
when used in a list, functions must be instances, as shown below::

    >>> my_linear_tm = pnl.TransferMechanism(size=3,
    ...                                      noise=[pnl.NormalDist(), pnl.UniformDist(), 3.0])
    >>> my_linear_tm.execute([1.0, 1.0, 1.0])
    array([[-0.22503678,  1.36995517,  4.        ]])
    >>> my_linear_tm.execute([1.0, 1.0, 1.0])
    array([[2.08371805, 1.60392004, 4.        ]])

Notice that since noise is a `modulable Parameter <ParameterPort_Modulable_Parameters>`, assigning it a value
after the TransferMechanism has been constructed must be done to its base value (see `ModulatorySignal_Modulation`
for additional information).

Finally, `clipping <TransferMechanism.clip>` can also be used to cap the result to within specified bounds::

    >>> my_linear_tm.clip = (.5, 1.2)
    >>> my_linear_tm.execute([1.0, 1.0, 1.0])
    array([[1.2, 1.2, 1.2]])
    >>> my_linear_tm.execute([1.0, 1.0, 1.0])
    array([[1.2       , 1.06552886, 1.2       ]])
    >>> my_linear_tm.execute([1.0, 1.0, 1.0])
    array([[0.5       , 1.01316799, 1.2       ]])

Note that the bounds specified in **clip** apply to all elements of the result if it is an array.

.. _TransferMechanism_Examples_Execution_With_Integration:

**With Integration**
^^^^^^^^^^^^^^^^^^^^

The following examples illustate the execution of a TransferMechanism with `integrator_mode
<TransferMechanism.integrator_mode>` set to True. For convenience, a TransferMechanism has three `Parameters` that
are used by most IntegratorFunctions, and that can be used to configure integration:`initial_value
<TransferMechanism.initial_value>`, `integration_rate <TransferMechanism.integration_rate>`, and `noise
<TransferMechanism.noise>`.  If any of these are specified in the TransferMechanism's constructor, their value is
used to specify the corresponding parameter of its `integrator_function <TransferMechanism.integrator_function>`.
In the following example, ``my_linear_tm`` is assigned `Linear` as its primary `function <Mechanism_Base.function>`,
congifured to transform arrays of ``size`` 3, with an **initial_value** of [0.1, 0.5, 0.9] and an **integration_rate**
of 0.5, that are passed as the values for the `initializer <AdaptiveIntegrator.initializer>` and `rate
<AdaptiveIntegrator.rate>` `Parameters` of its `integrator_function <TransferMechanism.integrator_function>`
`Parameters`, respectively.  Since, its `integrator_function <TransferMechanism.integrator_function>` is not specified,
the default for a TransferMechanism is used, which is `AdaptiveIntegrator`.  This integrates its input, returning
results that begin close to its `initializer <AdaptiveIntegrator.initializer>` and asymptotically approach the value
of the current input, which in this example is [1.0, 1.0, 1,0] for each execution::

    >>> my_linear_tm = pnl.TransferMechanism(size=3,
    ...                                      function=pnl.Linear,
    ...                                      integrator_mode=True,
    ...                                      initial_value=np.array([[0.1, 0.5, 0.9]]),
    ...                                      integration_rate=0.5)
    >>> my_linear_tm.integrator_function.initializer
    array([[0.1, 0.5, 0.9]])
    >>> my_linear_tm.integrator_function.previous_value
    array([[0.1, 0.5, 0.9]])
    >>> my_linear_tm.execute([1.0, 1.0, 1.0])
    array([[0.55, 0.75, 0.95]])
    >>> my_linear_tm.execute([1.0, 1.0, 1.0])
    array([[0.775, 0.875, 0.975]])
    >>> my_linear_tm.execute([1.0, 1.0, 1.0])
    array([[0.8875, 0.9375, 0.9875]])

Notice that specifying ``[[0.1, 0.5, 0.9]]`` as the **initial_value** for ``my_linear_tm`` assigns it both as the value
of the `integrator_function <TransferMechanism.integrator_function>`'s `initializer <AdaptiveIntegrator.initializer>`
Parameter, and also as its `previous_value <Mechanism_Base.previous_value>` which is used in the first step of
integration when ``my_linear_tm`` is executed.  For an `AdaptiveIntegrator`, each step of integration returns a
result that is its `previous_value <AdaptiveIntegrator.previous_value>` +  (`rate <AdaptiveIntegrator>` *
`previous_value <AdaptiveIntegrator.previous_value>` - input), asymtotically approaching the input.

In the following example, both the TransferMechanism's **integration_rate** and its `integrator_function
<TransferMechanism.integrator_function>`'s **rate** are specified::

    >>> my_linear_tm = pnl.TransferMechanism(integrator_function=AdaptiveIntegrator(rate=0.3),
    ...                                      integration_rate=0.1)
    >>> my_linear_tm.integration_rate # doctest: +NORMALIZE_WHITESPACE
    (TransferMechanism TransferMechanism-8):
        integration_rate.base: 0.3
        integration_rate.modulated: [0.3]

Notice that the value specified for the TransferMechanism integrator `integrator_function
<TransferMechanism.integrator_function>` (``0.3``) takes precendence, and is assigned as the value of the
TransferMechanism's `integration_rate <TransferMechanism.integration_rate>`, overriding the specified value (``0.1``).
The same applies for the specification of the TransferMechanism's **initial_value** argument and the **initializer**
for its `integration_function <TransferMechanism.integrator_function>`. Notice also that two values are reported for
the Mechanism's `integration_rate <TransferMechanism.integration_rate>`. This is because this is a `modulable Parameter
<ParameterPort_Modulable_Parameters>`.  The ``integration_rate.base`` is the one that is assigned;
``integration_rate.modulated`` reports the value that was actually used when the Mechanism was last executed;
this is the same as the base value if the Parameter is not subject to modulation;  if the Parameter is subject to
modulation <ModulatorySignal_Modulation>`, then the modulated value will be the base value modified by any
`modulatory signals <ModulatorySignal>` that project to the Mechanism for that Parameter.

.. _TransferMechanism_Examples_Initialization_and_Resetting:

*Initializing, Resetting and Resuming Integration*
**************************************************

When `integrator_mode <TransferMechanism.integrator_mode>` is True, the state of integration can be initialized
by specifying its `initial_value <TransferMechanism.initial_value>` using the **initial_value** argument in the
constructor, as shown in the following example:

    >>> my_linear_tm = pnl.TransferMechanism(function=pnl.Linear,
    ...                                      integrator_mode=True,
    ...                                      integration_rate=0.1,
    ...                                      initial_value=np.array([[0.2]]))
    >>> my_linear_tm.integrator_function.previous_value
    array([[0.2]])

It will then begin integration at that point.  The result after each execution is the integrated value
of the input and its `integrator_function <TransferMechanism.integrator_function>`'s `previous_value
<IntegratorFunction.previous_value>`::

    >>> my_linear_tm.execute(0.5)
    array([[0.23]])
    >>> my_linear_tm.execute(0.5)
    array([[0.257]])
    >>> my_linear_tm.execute(0.5)
    array([[0.2813]])

The TransferMechanism's `reset <TransferMechanism.reset>` method can be used to restart integration from its
`initial_value <TransferMechanism.initial_value>` or some other one.  For example, calling `reset
<TransferMechanism.reset>` without an argument resets the starting point of integration for
``my_linear_tm`` back to ``0.2``, and if it is executed `trials <TimeScale.TRIAL>` it produes the same results as
the first 3 executions:

    >>> my_linear_tm.integrator_function.reset()
    [array([[0.2]])]
    >>> my_linear_tm.execute(0.5)
    array([[0.23]])
    >>> my_linear_tm.execute(0.5)
    array([[0.257]])
    >>> my_linear_tm.execute(0.5)
    array([[0.2813]])

The `reset <TransferMechanism.reset>` method can also be used to start integration at a specified value, by providing
it as an argument to the method::

    >>> my_linear_tm.integrator_function.reset([0.4])
    [array([0.4])]
    >>> my_linear_tm.execute(0.5)
    array([[0.41]])
    >>> my_linear_tm.execute(0.5)
    array([[0.419]])

If integration is suspended (by changing `integrator_mode <TransferMechanism.integrator_mode>` from True to False),
the value it uses to resume integration (if `integrator_mode <TransferMechanism.integrator_mode>` is reassigned as
True) can be specified using the `on_resume_integrator_mode <TransferMechanism.on_resume_integrator_mode>` option.
If it is set to *RESET*, it will use `initial_value <TransferMechanism.initial_value>` to resume integration, just as
if `reset() <TransferMechanism.reset>` had been called.  If it is set to *CURRENT_VALUE* (the default), it will resume
integration using the current `value <Mechanism_Base.value>` of the Mechanism, irrespective of the `integrator_function
<TransferMechanism.integrator_function>`'s `previous_value <IntegratorFunction.previous_value>` at the point at which
integration was last suspended, as shown below::

    >>> my_linear_tm.integrator_mode = False
    >>> my_linear_tm.execute(0.2)
    array([[0.2]])
    >>> my_linear_tm.execute(0.2)
    array([[0.2]])
    >>> my_linear_tm.on_resume_integrator_mode = pnl.CURRENT_VALUE
    >>> my_linear_tm.integrator_mode = True
    >>> my_linear_tm.execute(0.5)
    array([[0.23]])
    >>> my_linear_tm.execute(0.5)
    array([[0.257]])

Notice that, with `on_resume_integrator_mode <TransferMechanism.on_resume_integrator_mode>` set to *CURRENT_VALUE*,
when `integrator_mode <TransferMechanism.integrator_mode>` is set back to True, integration proceeds from the most
recent value of ``my_linear_tem``.  In contrast, if `on_resume_integrator_mode
<TransferMechanism.on_resume_integrator_mode>` is set to *LAST_INTEGRATED_VALUE*, integration resumes using the
`integrator_function <TransferMechanism.integrator_function>`'s `previous_value IntegratorFunction.previous_value` at
the point at which integration was last suspended, irrespective of interverning executions::

    >>> my_linear_tm.on_resume_integrator_mode = pnl.LAST_INTEGRATED_VALUE
    >>> my_linear_tm.integrator_mode = False
    >>> my_linear_tm.execute(1.0)
    array([[1.]])
    >>> my_linear_tm.integrator_mode = True
    >>> my_linear_tm.execute(0.5)
    array([[0.2813]])
    >>> my_linear_tm.execute(0.5)
    array([[0.30317]])

Notice in this case that, even though the most recent value of ``my_linear_tm`` is ``1.0``, when `integrator_mode
<TransferMechanism.integrator_mode>` is set back to True, integration resumes from the most recent value when it was
last True (in this case, where it left off in the preceding example, ``0.257``).

.. _TransferMechanism_Examples_Termination:

*Terminating Integration*
*************************


*Termination by value*.  This terminates execution when the Mechanism's `value <Mechanism_Base.value>` reaches the
the value specified by the **threshold** argument.  This is implemented by specifying **termination_measure** with
a function that accepts a 2d array with a *single entry* as its argument and returns a scalar.  The single
entry is the TransferMechanism's current `value <Mechanism_Base.value>` (that is, its previous_value
is ignored). After each execution, the function is passed the Mechanism's current `value <Mechanism_Base.value>`,
and the scalar returned is compared to **termination_threshold** using the comparison operator specified by
**termination_comparison_op**. Execution continues until this returns True, as in the following example::

    >>> my_mech = pnl.TransferMechanism(size=2,
    ...                                 integrator_mode=True,
    ...                                 termination_measure=max,
    ...                                 termination_threshold=0.9,
    ...                                 termination_comparison_op=pnl.GREATER_THAN_OR_EQUAL)
    >>> my_mech.execute([0.5, 1])
    array([[0.46875, 0.9375 ]])
    >>> my_mech.num_executions_before_finished
    4

Here, ``my_mech`` continued to execute for ``5`` times, until the element of the Mechanism's `value
<Mechanism_Base.value>` with the greatest value exceeded ``0.9``.  Note that GREATER_THAN_EQUAL is a keyword for
the string ">=", which is a key in the `comparison_operators` dict for the Python ``operator.ge``; any of these
can be used to specify **termination_comparison_op**).

.. _TransferMechanism_Examples_Termination_By_Time:

*Termination by time*.  This terminates execution when the Mechanism has executed at least a number of times equal
to the **threshold** at a particular TimeScale (e.g., within a `RUN` or a `TRIAL <TimeScale.TRIAL>`). This is
specified by assigning a `TimeScale` to **termination_measure**;  execution terminates when the number of
executions at that TimeScale equals the **termination_threshold**.  Note that, in this case,
the **termination_comparison_op** argument is ignored (the `termination_comparison_op
<TransferMechanism.termination_comparison_op>` is automatically set to *GREATER_THAN_OR_EQUAL*).  For example,
``my_mech`` is configured below to execute at least twice per trial::

    >>> my_mech = pnl.TransferMechanism(size=2,
    ...                                 integrator_mode=True,
    ...                                 termination_measure=TimeScale.TRIAL,
    ...                                 termination_threshold=2)
    >>> my_mech.execute([0.5, 1])
    array([[0.375, 0.75 ]])
    >>> my_mech.num_executions_before_finished
    2

As noted `above <TransferMechanism_Continued_Execution>`, it will continue to execute if it is called again,
but only once per call::

    >>> my_mech.execute([0.5, 1])
    array([[0.4375, 0.875 ]])
    >>> my_mech.num_executions_before_finished
    1
    >>> my_mech.execute([0.5, 1])
    array([[0.46875, 0.9375 ]])
    >>> my_mech.num_executions_before_finished
    1

In the following example, this behavior is exploited to allow a recurrent form of TransferMechanism (``attention``)
to integrate for a fixed number of steps (e.g., to simulate the time taken to encode an instruction regarding the
which feature of the stimulus should be attended) before a stimulus is presented, and then allowing that
Mechanism to continue to integrate the instruction and impact stimulus processing once the stimulus is presented::

    >>> stim_input = pnl.ProcessingMechanism(size=2)
    >>> stim_percept = pnl.TransferMechanism(size=2, function=pnl.Logistic)
    >>> decision = pnl.TransferMechanism(name='Decision', size=2,
    ...                                  integrator_mode=True,
    ...                                  execute_until_finished=False,
    ...                                  termination_threshold=0.65,
    ...                                  termination_measure=max,
    ...                                  termination_comparison_op=pnl.GREATER_THAN)
    >>> instruction_input = pnl.ProcessingMechanism(size=2, function=pnl.Linear(slope=10))
    >>> attention = pnl.LCAMechanism(name='Attention', size=2, function=pnl.Logistic,
    ...                              leak=8, competition=8, self_excitation=0, time_step_size=.1,
    ...                              termination_threshold=3,
    ...                              termination_measure = pnl.TimeScale.TRIAL)
    >>> response = pnl.ProcessingMechanism(name='Response', size=2)
    ...
    >>> comp = pnl.Composition()
    >>> comp.add_linear_processing_pathway([stim_input, [[1,-1],[-1,1]], stim_percept, decision, response]) #doctest: +SKIP
    >>> comp.add_linear_processing_pathway([instruction_input, attention, stim_percept]) #doctest: +SKIP
    >>> comp.scheduler.add_condition(response, pnl.WhenFinished(decision)) #doctest: +SKIP
    ...
    >>> stim_percept.set_log_conditions([pnl.RESULT])
    >>> attention.set_log_conditions([pnl.RESULT])
    >>> decision.set_log_conditions([pnl.RESULT])
    >>> response.set_log_conditions(['OutputPort-0'])
    ...
    >>> inputs = {stim_input:        [[1, 1], [1, 1]],
    ...           instruction_input: [[1, -1], [-1, 1]]}
    >>> comp.run(inputs=inputs) # doctest: +SKIP

This example implements a simple model of attentional selection in perceptual decision making. In the model,
``stim_input`` represents the stimulus input, which is passed to ``stim_percept``, which also receives input
from the ``attention`` Mechanism.  ``stim_percpt passes its output to ``decision``, which integrates its input
until one of the state_features of the input (the first or second) reaches the threshold of 0.65, at which point
``response`` executes (specified by the condition ``(reponse, WhenFinished(decision)``).  In addition to the
``stim_input``, the model an instruction on each trial in ``instruction_input`` that specifies which feature of
the stimulus (i.e., the first or second element) should be "attended".  This is passed to the ``attention``
Mechanism, which uses it to select which feature of ``stim_percept`` should be passed to ``decision``, and thereby
determine the response.  Like the ``decision`` Mechanism, the ``attention`` Mechanism integrates its input.
However, its **threshold_measure** is specified as ``TimeScale.TRIAL`` and its **threshold** as ``3``, so it
carries out 3 steps of integration the first time it is executed in each trial.  Thus, when the input is presented
at the beginning of each trial, first ``stim_input`` and ``instruction_input`` execute.  Then ``attention``
executes, but ``stim_percept`` does not yet do so, since it receives input from ``attention`` and thus must wait
for that to execute first. When ``attention`` executes, it carries out its three steps of integration,
(giving it a chance to "encode" the instruction before the stimulus is processed by ``stim_percept``).  Then
``stim_percept`` executes, followed by ``decision``.  However, the latter carries out only one step of integration,
since its **execute_until_finished** is set to False.  If its output does not meet its termination condition after
that one step of integration, then ``response`` does not execute, since it has been assigned a condition that
requires ``decision`` to terminate before it does so. As a result, since ``response`` has not executed, the trial
continues. On the next pass, ``attention`` carries out only one step of integration, since its termination
condition has already been met, as does ``decision`` since its termination condition has *not* yet been met.  If
it is met, then ``response`` executes and the trial ends (since all Mechanisms have now had an opportunity to
execute). The value of the ``attention`` and ``decision`` Mechanisms after each execution are shown below::

    >>> attention.log.print_entries(display=[pnl.TIME, pnl.VALUE]) #doctest: +SKIP
    Log for Attention:
    Logged Item:   Time          Value
    'RESULT'       0:0:0:1      [0.64565631 0.19781611]  # Trial 0
    'RESULT'       0:0:0:1      [0.72347147 0.1422746 ]
    'RESULT'       0:0:0:1      [0.74621565 0.1258587 ]
    'RESULT'       0:0:1:1      [0.75306362 0.1208305 ]
    'RESULT'       0:0:2:1      [0.75516272 0.11926922]
    'RESULT'       0:0:3:1      [0.75581168 0.11878318]
    'RESULT'       0:0:4:1      [0.75601306 0.11863188]
    'RESULT'       0:1:0:1      [0.2955214  0.49852489]  # Trial 1
    'RESULT'       0:1:0:1      [0.17185129 0.68187518]
    'RESULT'       0:1:0:1      [0.13470156 0.73399742]
    'RESULT'       0:1:1:1      [0.1235536  0.74936691]
    'RESULT'       0:1:2:1      [0.12011584 0.75402671]

    >>> decision.log.print_entries(display=[pnl.TIME, pnl.VALUE]) #doctest: +SKIP
    Log for Decision:
    Logged Item:   Time          Value
    'RESULT'       0:0:0:3      [0.33917677 0.2657116 ]  # Trial 0
    'RESULT'       0:0:1:3      [0.50951133 0.39794126]
    'RESULT'       0:0:2:3      [0.59490696 0.46386164]
    'RESULT'       0:0:3:3      [0.63767534 0.49676128]
    'RESULT'       0:0:4:3      [0.65908142 0.51319226]
    'RESULT'       0:1:0:3      [0.59635299 0.59443706]  # Trial 1
    'RESULT'       0:1:1:3      [0.56360108 0.6367389 ]
    'RESULT'       0:1:2:3      [0.54679699 0.65839718]

    >>> response.log.print_entries(display=[pnl.TIME, pnl.VALUE]) #doctest: +SKIP
    Log for Response:
    Logged Item:   Time          Value
    'OutputPort-0' 0:0:4:4      [0.65908142 0.51319226]  # Trial 0
    'OutputPort-0' 0:1:2:4      [0.54679699 0.65839718]  # Trial 1

The `Time` signatures are ``run:trial:pass:time_step``.  Note that ``attention`` always executes in `time_step` 1
(after ``stim_input`` and ``instruction_input`` which execute in time_step 0).  In trial 0, ``attention``
executes three times in pass 0 (to reach its specified threshold), and then again in passes 1, 2 and 3 and 4
along with ``decision`` (which executes in time_step 3, after ``stim_percept`` in time_step 2),
as the trial continues and ``decision`` executes until reaching its threshold.  Note that ``response`` executed
only executed in pass 4, since it depends on the termination of ``decision``.  Note also that in trial 1
``attention`` executes 3 times in pass 0 as it did in trial 0;  however, ``decision`` executes only 3 times
since it begins closer to its threshold in that trial.

.. _TransferMechanism_Class_Reference:

Class Reference
---------------

"""
import copy
import inspect
import logging
import numbers
import types
import warnings
from collections.abc import Iterable

import numpy as np
import typecheck as tc

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.functions.nonstateful.combinationfunctions import LinearCombination, SUM
from psyneulink.core.components.functions.nonstateful.distributionfunctions import DistributionFunction
from psyneulink.core.components.functions.function import Function, is_function_type
from psyneulink.core.components.functions.nonstateful.objectivefunctions import Distance
from psyneulink.core.components.functions.nonstateful.selectionfunctions import SelectionFunction
from psyneulink.core.components.functions.stateful.integratorfunctions import AdaptiveIntegrator
from psyneulink.core.components.functions.stateful.integratorfunctions import IntegratorFunction
from psyneulink.core.components.functions.nonstateful.transferfunctions import Linear, Logistic, TransferFunction
from psyneulink.core.components.functions.userdefinedfunction import UserDefinedFunction
from psyneulink.core.components.mechanisms.mechanism import Mechanism, MechanismError
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import _is_control_spec
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.mdf import _get_variable_parameter_name
from psyneulink.core.globals.keywords import \
    COMBINE, comparison_operators, EXECUTION_COUNT, FUNCTION, GREATER_THAN_OR_EQUAL, \
    CURRENT_VALUE, LESS_THAN_OR_EQUAL, MAX_ABS_DIFF, \
    NAME, NOISE, NUM_EXECUTIONS_BEFORE_FINISHED, OWNER_VALUE, RESET, RESULT, RESULTS, \
    SELECTION_FUNCTION_TYPE, TRANSFER_FUNCTION_TYPE, TRANSFER_MECHANISM, VARIABLE
from psyneulink.core.globals.parameters import Parameter, FunctionParameter, check_user_specified
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import \
    all_within_range, append_type_to_name, iscompatible, is_comparison_operator, convert_to_np_array, safe_equals, parse_valid_identifier
from psyneulink.core.scheduling.time import TimeScale

__all__ = [
    'INITIAL_VALUE', 'CLIP',  'INTEGRATOR_FUNCTION', 'INTEGRATION_RATE',
    'TERMINATION_THRESHOLD', 'TERMINATION_MEASURE', 'TERMINATION_MEASURE_VALUE',
    'Transfer_DEFAULT_BIAS', 'Transfer_DEFAULT_GAIN', 'Transfer_DEFAULT_LENGTH', 'Transfer_DEFAULT_OFFSET',
    'TransferError', 'TransferMechanism',
]

# TransferMechanism parameter keywords:
CLIP = "clip"
INTEGRATOR_FUNCTION = 'integrator_function'
INTEGRATION_RATE = "integration_rate"
INITIAL_VALUE = 'initial_value'
TERMINATION_THRESHOLD = 'termination_threshold'
TERMINATION_MEASURE = 'termination_measure'
TERMINATION_MEASURE_VALUE = 'termination_measure_value'
termination_keywords = [EXECUTION_COUNT, NUM_EXECUTIONS_BEFORE_FINISHED]


# TransferMechanism default parameter values:
Transfer_DEFAULT_LENGTH = 1
Transfer_DEFAULT_GAIN = 1
Transfer_DEFAULT_BIAS = 0
Transfer_DEFAULT_OFFSET = 0
# Transfer_DEFAULT_RANGE = np.array([])

logger = logging.getLogger(__name__)


class TransferError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


def _integrator_mode_setter(value, owning_component=None, context=None):
    if value:
        if not owning_component.parameters.integrator_mode._get(context):
            # when first creating parameters, integrator_function is not
            # instantiated yet
            if (
                not owning_component.is_initializing
                and owning_component.integrator_function.parameters.execution_count._get(context) > 0
            ):
                # force, because integrator_mode is currently False
                # (will be set after exiting this method)
                if owning_component.on_resume_integrator_mode == CURRENT_VALUE:
                    owning_component.reset(
                        owning_component.parameters.value._get(context),
                        force=True,
                        context=context
                    )
                elif owning_component.on_resume_integrator_mode == RESET:
                    owning_component.reset(force=True, context=context)

    owning_component.parameters.has_initializers._set(value, context)

    return value


# IMPLEMENTATION NOTE:  IMPLEMENTS OFFSET PARAM BUT IT IS NOT CURRENTLY BEING USED
class TransferMechanism(ProcessingMechanism_Base):
    """
    TransferMechanism(                                       \
        noise=0.0,                                           \
        clip=(float:min, float:max),                         \
        integrator_mode=False,                               \
        integrator_function=AdaptiveIntegrator,              \
        initial_value=None,                                  \
        integration_rate=0.5,                                \
        on_resume_integrator_mode=CURRENT_VALUE,             \
        termination_measure=Distance(metric=MAX_ABS_DIFF),   \
        termination_threshold=None,                          \
        termination_comparison_op=LESS_THAN_OR_EQUAL,        \
        output_ports=RESULTS                                 \
        )

    Subclass of `ProcessingMechanism <ProcessingMechanism>` that performs a simple transform of its input.
    See `Mechanism <Mechanism_Class_Reference>` for additional arguments and attributes.

    Arguments
    ---------

    noise : float, function, or a list or array containing either or both : default 0.0
        specifies a value to be added to the result of the TransferMechanism's `function <Mechanism_Base.function>`
        or its `integrator_function <TransferMechanism.integrator_function>`, depending on whether `integrator_mode
        <TransferMechanism.integrator_mode>` is `True` or `False` (see `noise <TransferMechanism.noise>` for details).
        If **noise** is specified as a single function, it can be a reference to a Function class or an instance of one;
        if a function is used in a list, it *must* be an instance.

    clip : tuple(float, float) or list [float, float] : default None
        specifies the allowable range for the result of `function <Mechanism_Base.function>` (see
        `clip <TransferMechanism.clip>` for details).

    integrator_mode : bool : False
        specifies whether or not the TransferMechanism is executed with (True) or without (False) integrating
        its `variable <Mechanism_Base.variable>` using its `integrator_function
        <TransferMechanism.integrator_function>` before executing its primary `function <Mechanism_Base.function>`
        (see `TransferMechanism_Execution` for additional details).

    integrator_function : IntegratorFunction : default AdaptiveIntegrator
        specifies `IntegratorFunction` to use when `integrator_mode <TransferMechanism.integrator_mode>` is True (see
        `Execution with Integration <TransferMechanism_Examples_Execution_With_Integration>` for additional details).

    initial_value :  value, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the starting value for integration when `integrator_mode <TransferMechanism.integrator_mode>` is
        True; must be the same length `variable <Mechanism_Base.variable>` (see
        `TransferMechanism_Execution_Integration_Initialization` for additional details).

    integration_rate : float : default 0.5
        specifies the rate of integration of when the TransferMechanism when `integrator_mode
        <TransferMechanism.integrator_mode>` is True (see `TransferMechanism_Execution_Integration` for additional
        details).

    on_resume_integrator_mode : CURRENT_VALUE, LAST_INTEGRATED_VALUE, or RESET : default CURRENT_VALUE
        specifies value used by the `integrator_function <TransferMechanism.integrator_function>` when integration
        is resumed (see `resuming integration <TransferMechanism_Execution_Integration_Resumption>` for additional
        details).

    termination_measure : function or TimesScale : default Distance(metric=MAX_ABS_DIFF)
        specifies measure used to determine when execution of TransferMechanism is complete if `execute_until_finished
        <Component.execute_until_finished>` is True.  If it is a function, it must take at least one argument, and
        optionally a second, both of which must be arrays, and must return either another array or a scalar;  see
        `termination_measure <TransferMechanism.termination_measure>` for additional details.

    termination_threshold : None or float : default None
        specifies value against which `termination_measure_value <TransferMechanism.termination_measure_value>` is
        compared to determine when execution of TransferMechanism is complete; see `termination_measure
        <TransferMechanism.termination_measure>` for additional details.

    termination_comparison_op : comparator keyword : default LESS_THAN_OR_EQUAL
        specifies how `termination_measure_value <TransferMechanism.termination_measure_value>` is compared with
        `termination_threshold <TransferMechanism.termination_threshold>` to determine when execution of
        TransferMechanism is complete; see `termination_measure <TransferMechanism.termination_measure>`
        for additional details.

    output_ports : str, list or np.ndarray : default RESULTS
        specifies the OutputPorts for the TransferMechanism; the keyword **RESULTS** (the default) specifies that
        one OutputPort be generated for each InputPort specified in the **input_ports** argument (see
        `TransferMechanism_OutputPorts` for additional details, and note <TransferMechanism_OutputPorts_Note>` in
        particular).


    Attributes
    ----------

    noise : float, function or an array containing either or both
        value is applied to the result of `integrator_function <TransferMechanism.integrator_function>` if
        `integrator_mode <TransferMechanism.integrator_mode>` is False; otherwise it is passed as the `noise
        <IntegratorFunction.noise>` Parameter to `integrator_function <TransferMechanism.integrator_function>`. If
        noise is a float or function, it is added to all elements of the array being transformed; if it is a function,
        it is executed independently for each element each time the TransferMechanism is executed.  If noise is an
        array, it is applied Hadamard (elementwise) to the array being transformed;  again, each function is executed
        independently for each corresponding element of the array each time the Mechanism is executed.

        .. note::
            If **noise** is specified as a float or function in the constructor for the TransferMechanism, the noise
            Parameter cannot later be specified as a list or array, and vice versa.

        .. hint::
            To generate random noise that varies for every execution and across all elements of an array, a
            `DistributionFunction` should be used, that generates a new value on each execution. If noise is
            specified as a float, a function with a fixed output, or an array of either of these, then noise
            is simply an offset that is the same across all elements and executions.

    clip : tuple(float, float)
        determines the allowable range for all elements of the result of `function <Mechanism_Base.function>`.
        The 1st item (index 0) determines the minimum allowable value of the result, and the 2nd item (index 1)
        determines the maximum allowable value; any element of the result that exceeds the specified minimum or
        maximum value is set to the value of clip that it exceeds.

    integrator_mode : bool
        determines whether the TransferMechanism uses its `integrator_function <TransferMechanism.integrator_function>`
        to integrate its `variable <Mechanism_Base.variable>` when it executes (see TransferMechanism_Execution for
        additional details).

    integrator_function :  IntegratorFunction
        the `IntegratorFunction` used when `integrator_mode <TransferMechanism.integrator_mode>` is set to
        `True` (see `TransferMechanism_Execution_Integration` for details).

    initial_value :  value, list or np.ndarray
        determines the starting value for the `integrator_function <TransferMechanism.integrator_function>`
        when `integrator_mode <TransferMechanism.integrator_mode>` is `True`
        (see `TransferMechanism_Execution_Integration_Initialization` for additional details).

    integration_rate : float
        determines the rate at which the TransferMechanism's `variable <TransferMechanism>` is integrated when it is
        executed with `integrator_mode <TransferMechanism.integrator_mode>` set to `True`; a higher value specifies
        a faster rate (see `TransferMechanism_Execution_Integration` for additional details).

    on_resume_integrator_mode : CURRENT_VALUE, LAST_INTEGRATED_VALUE, or RESET
        determines value used by the `integrator_function <TransferMechanism.integrator_function>` when integration is
        resumed, and must be one of the following keywords: *CURRENT_VALUE*, *LAST_INTEGRATED_VALUE*, or
        *RESET* (see `resuming integration <TransferMechanism_Execution_Integration_Resumption>` for additional
        details).

    termination_measure : function or TimeScale
        used to determine when execution of the TransferMechanism is complete (i.e., `is_finished` is True), if
        `execute_until_finished <Component.execute_until_finished>` is True.  If it is a `TimeScale`, then execution
        terminates when the value of the Mechanism's `num_executions <Component_Num_Executions>` at that TimeScale is
        is equal to `termination_threshold <TransferMechanism.termination_threshold>`.  If it is a function, it is
        passed the `value <Mechanism_Base.value>` and `previous_value <Mechanism_Base.previous_value>` of the
        TransferMechanism; its result (`termination_measure_value <TransferMechanism.termination_measure_value>`) is
        compared with `termination_threshold <TransferMechanism.termination_threshold>` using
        `TransferMechanism.termination_comparison_op`, the result of which is used as the value of `is_finished`.

        .. note::
           A Mechanism's previous value is distinct from the `previous_value
           <IntegratorFunction.previous_value>` attribute of its `integrator_function
           <Mechanism_Base.integrator_function>`.

    termination_measure_value : array or scalar
        value returned by `termination_measure <TransferMechanism.termination_measure>`;  used to determine when
        `is_finished` is True.

    termination_threshold : None or float
        value with which `termination_measure_value <TransferMechanism.termination_measure_value>` is compared to
        determine when execution of TransferMechanism is complete if `execute_until_finished
        <Component.execute_until_finished>` is True.

    termination_comparison_op : Comparator
        used to compare `termination_measure_value <TransferMechanism.termination_measure_value>` with
        `termination_threshold <TransferMechanism.termination_threshold>` to determine when execution of
        TransferMechanism is complete if `execute_until_finished <Component.execute_until_finished>` is True.

    standard_output_ports : list[dict]
        list of `Standard OutputPort <OutputPort_Standard>` that includes the following in addition to the
        `standard_output_ports <ProcessingMechanism.standard_output_ports>` of a
        `ProcessingMechanism <ProcessingMechanism>`:

        .. _COMBINE:

        *COMBINE* : 1d array
          Element-wise (Hadamard) sum of all items of the TransferMechanism's `value <Mechanism_Base.value>`
          (requires that they all have the same dimensionality).

    Returns
    -------
    instance of TransferMechanism : TransferMechanism

    """

    componentType = TRANSFER_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TYPE_DEFAULT_PREFERENCES
    # classPreferences = {
    #     PREFERENCE_SET_NAME: 'TransferCustomClassPreferences',
    #     # REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    #     }

    # TransferMechanism parameter and control signal assignments):

    standard_output_ports = ProcessingMechanism_Base.standard_output_ports.copy()
    standard_output_ports.extend([{NAME: COMBINE,
                                   VARIABLE: OWNER_VALUE,
                                   FUNCTION: LinearCombination(operation=SUM)}])
    standard_output_port_names = ProcessingMechanism_Base.standard_output_port_names.copy()
    standard_output_port_names.extend([COMBINE])

    class Parameters(ProcessingMechanism_Base.Parameters):
        """
            Attributes
            ----------

                clip
                    see `clip <TransferMechanism.clip>`

                    :default value: None
                    :type:

                initial_value
                    see `initial_value <TransferMechanism.initial_value>`

                    :default value: None
                    :type:

                integration_rate
                    see `integration_rate <TransferMechanism.integration_rate>`

                    :default value: 0.5
                    :type: ``float``

                integrator_function
                    see `integrator_function <TransferMechanism.integrator_function>`

                    :default value: `AdaptiveIntegrator`
                    :type: `Function`

                integrator_function_value
                    see `integrator_function_value <TransferMechanism.integrator_function_value>`

                    :default value: [[0]]
                    :type: ``list``
                    :read only: True

                integrator_mode
                    see `integrator_mode <TransferMechanism_Integrator_Mode>`

                    :default value: False
                    :type: ``bool``

                noise
                    see `noise <TransferMechanism.noise>`

                    :default value: 0.0
                    :type: ``float``

                on_resume_integrator_mode
                    see `on_resume_integrator_mode <TransferMechanism.on_resume_integrator_mode>`

                    :default value: `CURRENT_VALUE`
                    :type: ``str``

                output_ports
                    see `output_ports <Mechanism_Base.output_ports>`

                    :default value: [`RESULTS`]
                    :type: ``list``
                    :read only: True

                termination_comparison_op
                    see `termination_comparison_op <TransferMechanism.termination_comparison_op>`

                    :default value: ``operator.le``
                    :type: ``types.FunctionType``

                termination_measure
                    see `termination_measure <TransferMechanism.termination_measure>`

                    :default value: `Distance`(metric=max_abs_diff)
                    :type: `Function`

                termination_measure_value
                    see `termination_measure_value <TransferMechanism.termination_measure_value>`

                    :default value: 0.0
                    :type: ``float``
                    :read only: True

                termination_threshold
                    see `termination_threshold <TransferMechanism.termination_threshold>`

                    :default value: None
                    :type:
        """
        integrator_mode = Parameter(False, setter=_integrator_mode_setter, valid_types=bool)
        integration_rate = FunctionParameter(
            0.5,
            function_name='integrator_function',
            function_parameter_name='rate',
            primary=True,
        )
        initial_value = FunctionParameter(
            None,
            function_name='integrator_function',
            function_parameter_name='initializer'
        )
        integrator_function = Parameter(AdaptiveIntegrator, stateful=False, loggable=False)
        function = Parameter(Linear, stateful=False, loggable=False, dependencies='integrator_function')
        integrator_function_value = Parameter([[0]], read_only=True)
        on_resume_integrator_mode = Parameter(CURRENT_VALUE, stateful=False, loggable=False)
        clip = None
        noise = FunctionParameter(0.0, function_name='integrator_function')
        termination_measure = Parameter(
            Distance(metric=MAX_ABS_DIFF),
            modulable=False,
            stateful=False,
            loggable=False
        )
        termination_threshold = Parameter(None, modulable=True)
        termination_comparison_op = Parameter(LESS_THAN_OR_EQUAL, modulable=False, loggable=False)
        termination_measure_value = Parameter(0.0, modulable=False, read_only=True, pnl_internal=True)

        output_ports = Parameter(
            [RESULTS],
            stateful=False,
            loggable=False,
            read_only=True,
            structural=True,
        )

        def _validate_variable(self, variable):
            if 'U' in str(variable.dtype):
                return 'may not contain non-numeric entries'

        def _validate_clip(self, clip):
            if clip:
                if (not (isinstance(clip, (list,tuple)) and len(clip)==2
                         and all(isinstance(i, numbers.Number)) for i in clip)):
                    return 'must be a tuple with two numbers.'
                if not clip[0] < clip[1]:
                    return 'first item must be less than the second.'

        def _parse_clip(self, clip):
            if clip:
                return tuple(clip)

        def _validate_integrator_mode(self, integrator_mode):
            if not isinstance(integrator_mode, bool):
                return 'may only be True or False.'

        def _validate_integration_rate(self, integration_rate):
            integration_rate = convert_to_np_array(integration_rate)
            if not all_within_range(integration_rate, 0, 1):
                return 'must be an int or float in the interval [0,1]'

        def _validate_termination_measure(self, termination_measure):
            if not isinstance(termination_measure, TimeScale) and not is_function_type(termination_measure):
                return f"must be a function or a TimeScale."

        def _parse_termination_measure(self, termination_measure):
            if isinstance(termination_measure, type):
                return termination_measure()
            return termination_measure

        def _validate_termination_threshold(self, termination_threshold):
            if (termination_threshold is not None
                    and not isinstance(termination_threshold, (int, float))):
                return 'must be a float or int.'

        def _validate_termination_comparison_op(self, termination_comparison_op):
            if (termination_comparison_op not in comparison_operators.keys()
                    and termination_comparison_op not in comparison_operators.values()):
                return f"must be boolean comparison operator or one of the following strings:" \
                       f" {','.join(comparison_operators.keys())}."

    @check_user_specified
    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_ports:tc.optional(tc.any(Iterable, Mechanism, OutputPort, InputPort))=None,
                 function=None,
                 noise=None,
                 clip=None,
                 integrator_mode=None,
                 integrator_function=None,
                 initial_value=None,
                 integration_rate=None,
                 on_resume_integrator_mode=None,
                 termination_measure=None,
                 termination_threshold:tc.optional(tc.any(int, float))=None,
                 termination_comparison_op: tc.optional(tc.any(str, is_comparison_operator)) = None,
                 output_ports:tc.optional(tc.any(str, Iterable))=None,
                 params=None,
                 name=None,
                 prefs: tc.optional(is_pref_set) = None,
                 **kwargs):
        """Assign type-level preferences and call super.__init__
        """

        # Default output_ports is specified in constructor as a string rather than a list
        # to avoid "gotcha" associated with mutable default arguments
        # (see: bit.ly/2uID3s3 and http://docs.python-guide.org/en/latest/writing/gotchas/)
        if output_ports is None or output_ports == RESULTS:
            output_ports = [RESULTS]

        initial_value = self._parse_arg_initial_value(initial_value)

        self._current_variable_index = 0

        super(TransferMechanism, self).__init__(
            default_variable=default_variable,
            size=size,
            input_ports=input_ports,
            output_ports=output_ports,
            initial_value=initial_value,
            noise=noise,
            integration_rate=integration_rate,
            integrator_mode=integrator_mode,
            clip=clip,
            termination_measure=termination_measure,
            termination_threshold=termination_threshold,
            termination_comparison_op=termination_comparison_op,
            integrator_function=integrator_function,
            on_resume_integrator_mode=on_resume_integrator_mode,
            function=function,
            params=params,
            name=name,
            prefs=prefs,
            **kwargs
        )

    def _parse_arg_initial_value(self, initial_value):
        return self._parse_arg_variable(initial_value)

    def _parse_termination_measure_variable(self, variable):
        # compares to previous value
        # NOTE: this method is for shaping, not for computation, and
        # a previous value should not be passed through here
        return np.array([variable, variable])

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate FUNCTION and Mechanism params

        """

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        # Validate FUNCTION
        if self.parameters.function._user_specified:
            transfer_function = self.defaults.function
            transfer_function_class = None

            # FUNCTION is a Function
            if isinstance(transfer_function, Function):
                transfer_function_class = transfer_function.__class__
            # FUNCTION is a class
            elif inspect.isclass(transfer_function):
                transfer_function_class = transfer_function

            if issubclass(transfer_function_class, Function):
                if not issubclass(transfer_function_class, (TransferFunction, SelectionFunction, UserDefinedFunction)):
                    raise TransferError(f"Function specified as {repr(FUNCTION)} param of {self.name} "
                                        f"({transfer_function_class.__name__}) must be a "
                                        f"{' or '.join([TRANSFER_FUNCTION_TYPE, SELECTION_FUNCTION_TYPE])}.")
            elif not isinstance(transfer_function, (types.FunctionType, types.MethodType)):
                raise TransferError(f"Unrecognized specification for {repr(FUNCTION)} param "
                                    f"of {self.name} ({transfer_function}).")

            # FUNCTION is a function or method, so test that shape of output = shape of input
            if isinstance(transfer_function, (types.FunctionType, types.MethodType, UserDefinedFunction)):
                var_shape = self.defaults.variable.shape
                if isinstance(transfer_function, UserDefinedFunction):
                    val_shape = transfer_function._execute(self.defaults.variable, context=context).shape
                else:
                    val_shape = np.array(transfer_function(self.defaults.variable, context=context)).shape

                if val_shape != var_shape:
                    raise TransferError(f"The shape ({val_shape}) of the value returned by the Python function, "
                                        f"method, or UDF specified as the {repr(FUNCTION)} param of {self.name} "
                                        f"must be the same shape ({var_shape}) as its {repr(VARIABLE)}.")

        # IMPLEMENTATION NOTE:
        #  Need to validate initial_value and integration_rate here (vs. in Parameters._validate_XXX)
        #  as they must be compared against default_variable if it was user-specified
        #  which is not available in Parameters _validation.

        # Validate INITIAL_VALUE
        if INITIAL_VALUE in target_set and target_set[INITIAL_VALUE] is not None:
            initial_value = np.array(target_set[INITIAL_VALUE])
            if (
                not iscompatible(initial_value, self.defaults.variable)
                # extra conditions temporary until universal initializer
                # validation is developed
                and initial_value.shape != self.integrator_function.defaults.variable.shape
                and self._get_parsed_variable(self.parameters.integrator_function,
                                              initial_value).shape != self.integrator_function.defaults.variable.shape
            ):
                raise TransferError(f"The format of the initial_value parameter for {append_type_to_name(self)} "
                                    f"({initial_value}) must match its variable ({self.defaults.variable}).")

        # Validate INTEGRATION_RATE:
        if INTEGRATION_RATE in target_set and target_set[INTEGRATION_RATE] is not None:
            integration_rate = np.array(target_set[INTEGRATION_RATE])
            if (
                not np.isscalar(integration_rate.tolist())
                and integration_rate.shape != self.defaults.variable.shape
                and integration_rate.shape != self.defaults.variable.squeeze().shape
            ):
                raise TransferError(f"{repr(INTEGRATION_RATE)} arg for {self.name} ({integration_rate}) "
                                    f"must be either an int or float, or have the same shape "
                                    f"as its {VARIABLE} ({self.defaults.variable}).")

        # Validate NOISE:
        if NOISE in target_set:
            self._validate_noise(target_set[NOISE])

        # Validate INTEGRATOR_FUNCTION:
        if INTEGRATOR_FUNCTION in target_set and target_set[INTEGRATOR_FUNCTION] is not None:
            integtr_fct = target_set[INTEGRATOR_FUNCTION]
            if not (isinstance(integtr_fct, IntegratorFunction)
                    or (isinstance(integtr_fct, type) and issubclass(integtr_fct, IntegratorFunction))):
                raise TransferError(f"The function specified for the {repr(INTEGRATOR_FUNCTION)} arg of {self.name} "
                                    f"({integtr_fct}) must be an {IntegratorFunction.__class__.__name__}.")

    # FIX: CONSOLIDATE THIS WITH StatefulFunction._validate_noise
    def _validate_noise(self, noise):
        # Noise is a scalar, list, array or DistributionFunction

        if isinstance(noise, DistributionFunction):
            noise = noise.execute

        if isinstance(noise, (np.ndarray, list)):
            if len(noise) == 1:
                pass
            # Variable is a list/array
            elif (not iscompatible(np.atleast_2d(noise), self.defaults.variable)
                  and not iscompatible(np.atleast_1d(noise), self.defaults.variable) and len(noise) > 1):
                raise MechanismError(f"Noise parameter ({noise}) for '{self.name}' does not match default variable "
                                     f"({self.defaults.variable}); it must be specified as a float, a function, "
                                     f"or an array of the appropriate shape "
                                     f"({np.shape(np.array(self.defaults.variable))}).")
            else:
                for i in range(len(noise)):
                    if isinstance(noise[i], DistributionFunction):
                        noise[i] = noise[i].execute
                    if (not np.isscalar(noise[i]) and not callable(noise[i])
                            and not iscompatible(np.atleast_2d(noise[i]), self.defaults.variable[i])
                            and not iscompatible(np.atleast_1d(noise[i]), self.defaults.variable[i])):
                        raise MechanismError(f"The element '{noise[i]}' specified in 'noise' for {self.name} "
                                             f"is not valid; noise must be list or array must be floats or functions.")

        elif _is_control_spec(noise):
            pass

        # Otherwise, must be a float, int or function
        elif noise is not None and not isinstance(noise, (float, int)) and not callable(noise):
            raise MechanismError(f"Noise parameter ({noise}) for {self.name} must be a float, "
                                 f"function, or array/list of these.")

    def _instantiate_parameter_ports(self, function=None, context=None):

        # If function is a logistic, and clip has not been specified, bound it between 0 and 1
        if (
            (
                isinstance(function, Logistic)
                or (
                    inspect.isclass(function)
                    and issubclass(function, Logistic)
                )
            )
            and self.clip is None
        ):
            self.clip = (0,1)

        super()._instantiate_parameter_ports(function=function, context=context)

    def _instantiate_attributes_before_function(self, function=None, context=None):
        super()._instantiate_attributes_before_function(function=function, context=context)

        if self.parameters.initial_value._get(context) is None:
            self.defaults.initial_value = copy.deepcopy(self.defaults.variable)
            self.parameters.initial_value._set(copy.deepcopy(self.defaults.variable), context)

    def _instantiate_output_ports(self, context=None):
        # If user specified more than one item for variable, but did not specify any custom OutputPorts,
        # then assign one OutputPort (with the default name, indexed by the number of the item) per item of variable
        if len(self.output_ports) == 1 and self.output_ports[0] == RESULTS:
            if len(self.defaults.variable) == 1:
                output_ports = [RESULT]
            else:
                output_ports = []
                for i, item in enumerate(self.defaults.variable):
                    output_ports.append({NAME: f'{RESULT}-{i}', VARIABLE: (OWNER_VALUE, i)})
            self.parameters.output_ports._set(output_ports, context)
        super()._instantiate_output_ports(context=context)

        # # Relabel first output_port:
        # #    default (assigned by Mechanism's OutputPort registry) is to name it "RESULT";
        # #    but in this context, explicitly adding -0 index helps put first one on par with others
        # #    (i.e., make clear the alignment of each OutputPort with the items of the TransferMechanmism's value).
        # remove_instance_from_registry(registry=self._portRegistry,
        #                               category=OUTPUT_PORT,
        #                               component=self.output_ports['RESULT'])
        # register_instance(self.output_ports['RESULT'], 'RESULT-0', OutputPort, self._portRegistry, OUTPUT_PORT)

    def _get_instantaneous_function_input(self, function_variable, noise, context=None):
        noise = self._try_execute_param(noise, function_variable, context=context)
        if noise is not None and not safe_equals(noise, 0):
            current_input = function_variable + noise
        else:
            current_input = function_variable

        return current_input

    def _clip_result(self, clip, current_input):
        if clip is not None:
            minCapIndices = np.where(current_input < clip[0])
            maxCapIndices = np.where(current_input > clip[1])
            current_input[minCapIndices] = np.min(clip)
            current_input[maxCapIndices] = np.max(clip)
        return current_input

    def _gen_llvm_is_finished_cond(self, ctx, builder, params, state):
        current = pnlvm.helpers.get_state_ptr(builder, self, state, "value")
        threshold_ptr = pnlvm.helpers.get_param_ptr(builder, self, params,
                                                    "termination_threshold")
        if isinstance(threshold_ptr.type.pointee, pnlvm.ir.LiteralStructType):
            # Threshold is not defined, return the old value of finished flag
            assert len(threshold_ptr.type.pointee) == 0
            is_finished_ptr = pnlvm.helpers.get_state_ptr(builder, self, state,
                                                          "is_finished_flag")
            is_finished_flag = builder.load(is_finished_ptr)
            return builder.fcmp_ordered("!=", is_finished_flag,
                                              is_finished_flag.type(0))

        # If modulated, termination threshold is single element array.
        # Otherwise, it is scalar
        threshold = pnlvm.helpers.load_extract_scalar_array_one(builder,
                                                                threshold_ptr)

        cmp_val_ptr = builder.alloca(threshold.type, name="is_finished_value")
        if self.termination_measure is max:
            assert self._termination_measure_num_items_expected == 1
            # Get inside of the structure
            val = builder.gep(current, [ctx.int32_ty(0), ctx.int32_ty(0)])
            first_val = builder.load(builder.gep(val, [ctx.int32_ty(0), ctx.int32_ty(0)]))
            builder.store(first_val, cmp_val_ptr)
            with pnlvm.helpers.array_ptr_loop(builder, val, "max_loop") as (b, idx):
                test_val = b.load(b.gep(val, [ctx.int32_ty(0), idx]))
                max_val = b.load(cmp_val_ptr)
                cond = b.fcmp_ordered(">=", test_val, max_val)
                max_val = b.select(cond, test_val, max_val)
                b.store(max_val, cmp_val_ptr)

        elif isinstance(self.termination_measure, Function):
            prev_val_ptr = pnlvm.helpers.get_state_ptr(builder, self, state, "value", 1)
            prev_val = builder.load(prev_val_ptr)

            expected = np.empty_like([self.defaults.value[0], self.defaults.value[0]])
            got = np.empty_like(self.termination_measure.defaults.variable)
            if expected.shape != got.shape:
                warnings.warn("Shape mismatch: Termination measure input: "
                              "{self.termination_measure.defaults.variable} should be {expected.shape}.")
                # FIXME: HACK the distance function is not initialized
                self.termination_measure.defaults.variable = expected

            func = ctx.import_llvm_function(self.termination_measure)
            func_params = pnlvm.helpers.get_param_ptr(builder, self, params, "termination_measure")
            func_state = pnlvm.helpers.get_state_ptr(builder, self, state, "termination_measure")
            func_in = builder.alloca(func.args[2].type.pointee, name="is_finished_func_in")
            # Populate input
            func_in_current_ptr = builder.gep(func_in, [ctx.int32_ty(0),
                                                        ctx.int32_ty(0)])
            current_ptr = builder.gep(current, [ctx.int32_ty(0), ctx.int32_ty(0)])
            builder.store(builder.load(current_ptr), func_in_current_ptr)

            func_in_prev_ptr = builder.gep(func_in, [ctx.int32_ty(0),
                                                     ctx.int32_ty(1)])
            builder.store(builder.extract_value(prev_val, 0), func_in_prev_ptr)

            builder.call(func, [func_params, func_state, func_in, cmp_val_ptr])
        elif isinstance(self.termination_measure, TimeScale):
            ptr = builder.gep(pnlvm.helpers.get_state_ptr(builder, self, state, "num_executions"),
                              [ctx.int32_ty(0), ctx.int32_ty(self.termination_measure.value)])
            ptr_val = builder.sitofp(builder.load(ptr), threshold.type)
            pnlvm.helpers.printf(builder, f"TERM MEASURE {self.termination_measure} %d %d\n",ptr_val, threshold)
            builder.store(ptr_val, cmp_val_ptr)
        else:
            assert False, f"Not Supported: {self.termination_measure}."

        cmp_val = builder.load(cmp_val_ptr)
        cmp_str = self.parameters.termination_comparison_op.get(None)
        return builder.fcmp_ordered(cmp_str, cmp_val, threshold)

    def _gen_llvm_mechanism_functions(self, ctx, builder, m_base_params, m_params,
                                      m_state, m_in, m_val, ip_out, *, tags:frozenset):

        if self.integrator_mode:
            if_state = pnlvm.helpers.get_state_ptr(builder, self, m_state,
                                                   "integrator_function")
            if_base_params = pnlvm.helpers.get_param_ptr(builder, self, m_base_params,
                                                         "integrator_function")
            if_params, builder = self._gen_llvm_param_ports_for_obj(
                    self.integrator_function, if_base_params, ctx, builder,
                    m_base_params, m_state, m_in)

            mf_in, builder = self._gen_llvm_invoke_function(
                    ctx, builder, self.integrator_function, if_params,
                    if_state, ip_out, None, tags=tags)
        else:
            mf_in = ip_out

        mf_state = pnlvm.helpers.get_state_ptr(builder, self, m_state, "function")
        mf_base_params = pnlvm.helpers.get_param_ptr(builder, self, m_base_params, "function")
        mf_params, builder = self._gen_llvm_param_ports_for_obj(
                self.function, mf_base_params, ctx, builder, m_base_params, m_state, m_in)

        mf_out, builder = self._gen_llvm_invoke_function(ctx, builder,
                                                         self.function, mf_params,
                                                         mf_state, mf_in, m_val,
                                                         tags=tags)

        clip_ptr = pnlvm.helpers.get_param_ptr(builder, self, m_params, "clip")
        if len(clip_ptr.type.pointee) != 0:
            assert len(clip_ptr.type.pointee) == 2
            clip_lo = builder.load(builder.gep(clip_ptr, [ctx.int32_ty(0),
                                                          ctx.int32_ty(0)]))
            clip_hi = builder.load(builder.gep(clip_ptr, [ctx.int32_ty(0),
                                                          ctx.int32_ty(1)]))
            for i in range(mf_out.type.pointee.count):
                mf_out_local = builder.gep(mf_out, [ctx.int32_ty(0), ctx.int32_ty(i)])
                with pnlvm.helpers.array_ptr_loop(builder, mf_out_local, "clip") as (b1, index):
                    ptri = b1.gep(mf_out_local, [ctx.int32_ty(0), index])
                    ptro = b1.gep(mf_out_local, [ctx.int32_ty(0), index])

                    val = b1.load(ptri)
                    val = pnlvm.helpers.fclamp(b1, val, clip_lo, clip_hi)
                    b1.store(val, ptro)

        return mf_out, builder

    def _execute(self, variable=None, context=None, runtime_params=None):
        """Execute TransferMechanism function and return transform of input"""

        clip = self.parameters.clip._get(context)
        value = super(Mechanism, self)._execute(variable=variable, context=context, runtime_params=runtime_params)
        value = self._clip_result(clip, value)

        return value

    @handle_external_context(fallback_most_recent=True)
    def reset(self, *args, force=False, context=None, **kwargs):

        # # FIX: UNCOMMENT ONCE REMOVED FROM Mechanism_Base.reset()
        # # (1) reset it, (2) run the primary function with the new "previous_value" as input
        # # (3) update value, (4) update output ports
        # if not isinstance(self.integrator_function, IntegratorFunction):
        #     raise TransferError(
        #         f"Resetting '{self.name}' is not allowed because its integrator_function "
        #         f"is not an IntegratorFunction type function, therefore the Mechanism "
        #         f"does not have an integrator to reset."
        #     )
        #
        # if self.parameters.integrator_mode._get(context) or force:
        #     new_input = self.integrator_function.reset(*args, **kwargs, context=context)[0]
        #     self.parameters.value._set(
        #         self.function.execute(variable=new_input, context=context),
        #         context=context,
        #         override=True
        #     )
        #     self._update_output_ports(context=context)
        #
        # else:
        #     raise TransferError(f"Resetting '{self.name}' is not allowed because its `integrator_mode` parameter "
        #                         f"is currently set to 'False'; try setting it to 'True'.")

        super().reset(*args, force=force, context=context, **kwargs)
        self.parameters.value.clear_history(context)

    def _parse_function_variable(self, variable, context=None):
        if self.is_initializing:
            return super(TransferMechanism, self)._parse_function_variable(variable=variable, context=context)

        integrator_mode = self.parameters.integrator_mode._get(context)
        noise = self._get_current_parameter_value(self.parameters.noise, context)

        # Update according to time-scale of integration
        if integrator_mode:
            value = self.integrator_function.execute(variable, context=context)
            self.parameters.integrator_function_value._set(value, context)
            return value

        else:
            return self._get_instantaneous_function_input(variable, noise, context)

    def _instantiate_attributes_after_function(self, context=None):
        """Determine numberr of items expected by termination_measure"""
        super()._instantiate_attributes_after_function(context)

        measure = self.termination_measure

        if isinstance(measure, TimeScale):
            self._termination_measure_num_items_expected = 0
            self.parameters.termination_comparison_op._set(GREATER_THAN_OR_EQUAL, context)
            return

        try:
            # If measure is a Function, use its default_variable to determine expected number of items
            self._termination_measure_num_items_expected = len(measure.parameters.variable.default_value)
        except:
            # Otherwise, use "duck typing"
            try:
                # Try a single item first (only uses value, and not previous_value)
                measure(np.array([0,0]))
                self._termination_measure_num_items_expected = 1
            except:
                try:
                    # termination_measure takes two arguments -- value and previous_value -- (e.g., Distance)
                    measure(np.array([[0,0],[0,0]]))
                    self._termination_measure_num_items_expected = 2
                except:
                    assert False, f"PROGRAM ERROR: Unable to determine length of input for" \
                                  f" {repr(TERMINATION_MEASURE)} arg of {self.name}"

        self.parameters.value.history_min_length = self._termination_measure_num_items_expected - 1

    def _report_mechanism_execution(self, input, params=None, output=None, context=None):
        """Override super to report previous_input rather than input, and selected params
        """
        # KAM Changed 8/29/17 print_input = self.previous_input --> print_input = input
        # because self.previous_input is not a valid attrib of TransferMechanism

        print_input = input
        try:
            params = params.copy()
            # Suppress reporting of range (not currently used)
            del params[CLIP]
        except (AttributeError, KeyError):
            pass

        super()._report_mechanism_execution(input_val=print_input, params=params, context=context)

    @handle_external_context()
    def is_finished(self, context=None):
        """Returns True when value of Mechanism reaches threhsold or if threshold is None.

        Note:  if threshold is None or Mechanism not in integartor_mode,
                  implements single update (cycle) per call to _execute method
                  (equivalent to setting Component.execute_until_finished = False)
        """

        try:
            threshold = self.get_mod_termination_threshold(context)
        except:
            threshold = self.parameters.termination_threshold._get(context)
        integrator_mode = self.parameters.integrator_mode._get(context)

        if (not integrator_mode
                or threshold is None
                or self.initialization_status == ContextFlags.INITIALIZING):
            # return True
            return self.parameters.is_finished_flag._get(context)

        assert self.parameters.value.history_min_length + 1 >= self._termination_measure_num_items_expected,\
            "History of 'value' is not guaranteed enough entries for termination_mesasure"
        measure = self.termination_measure
        value = self.parameters.value._get(context)

        if self._termination_measure_num_items_expected==0:
            status = self.parameters.num_executions._get(context)._get_by_time_scale(self.termination_measure)

        elif self._termination_measure_num_items_expected==1:
            # Squeeze to collapse 2d array with single item
            status = measure(np.squeeze(value))
        else:
            previous_value = self.parameters.value.get_previous(context)
            status = measure([value, previous_value])

        self.parameters.termination_measure_value._set(status, context=context, override=True)

        # comparator = self.parameters.termination_comparison_op._get(context)
        comparator = comparison_operators[self.parameters.termination_comparison_op._get(context)]
        # if any(comparison_operators[comparator](np.atleast_1d(status), threshold)):
        if comparator(np.atleast_1d(status), threshold).any():
            logger.info(f'{type(self).__name__} {self.name} has reached threshold ({threshold})')
            return True
        return False

    @handle_external_context()
    def _update_default_variable(self, new_default_variable, context=None):
        if not self.parameters.initial_value._user_specified:
            integrator_function_variable = self._get_parsed_variable(
                self.parameters.integrator_function,
                new_default_variable,
                context=context
            )
            self.defaults.initial_value = copy.deepcopy(integrator_function_variable)
            self.parameters.initial_value._set(copy.deepcopy(integrator_function_variable), context)

        super()._update_default_variable(new_default_variable, context=context)

    def as_mdf_model(self):
        import modeci_mdf.mdf as mdf

        model = super().as_mdf_model()
        function_model = [
            f for f in model.functions
            if f.id == parse_valid_identifier(self.function.name)
        ][0]
        assert function_model.id == parse_valid_identifier(self.function.name), (function_model.id, parse_valid_identifier(self.function.name))

        if self.defaults.integrator_mode:
            integrator_function_model = self.integrator_function.as_mdf_model()

            primary_input = function_model.args[_get_variable_parameter_name(self.function)]
            self.integrator_function._set_mdf_arg(
                integrator_function_model,
                _get_variable_parameter_name(self.integrator_function),
                primary_input
            )
            self.function._set_mdf_arg(
                function_model,
                _get_variable_parameter_name(self.function),
                integrator_function_model.id
            )

            for _, func_param in integrator_function_model.metadata['function_stateful_params'].items():
                model.parameters.append(mdf.Parameter(**func_param))

            model.functions.append(integrator_function_model)

            res = self.integrator_function._get_mdf_noise_function()
            try:
                main_noise_function, extra_noise_functions = res
            except TypeError:
                pass
            else:
                main_noise_function.id = f'{model.id}_{main_noise_function.id}'
                model.functions.append(main_noise_function)
                model.functions.extend(extra_noise_functions)

                self.integrator_function._set_mdf_arg(
                    integrator_function_model, 'noise', main_noise_function.id
                )

        return model
