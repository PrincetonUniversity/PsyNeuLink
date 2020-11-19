# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# NOTES:
#  * NOW THAT NOISE AND INTEGRATION_RATE ARE PROPRETIES THAT DIRECTLY REFERERNCE integrator_function,
#      SHOULD THEY NOW BE VALIDATED ONLY THERE (AND NOT IN TransferMechanism)??
#  * ARE THOSE THE ONLY TWO integrator PARAMS THAT SHOULD BE PROPERTIES??

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
        - `TransferMechanism_Integration`
        - `TransferMechanism_Termination`
        - `TransferMechanism_Reinitialization`
  * `TransferMechanism_Class_Reference`


.. _TransferMechanism_Overview:

Overview
--------

A TransferMechanism is a subclass of `ProcessingMechanism` that adds the ability to integrate its input.

Like a ProcessingMechanism, it transforms its input using a simple mathematical function, that maintains the form
(dimensionality) of its input.  The input can be a single scalar value, a multidimensional array (list or numpy
array), or several independent ones. The function used to carry out the transformation can be selected from the
following PsyNeuLink `Functions <Function>`: `Linear`, `Exponential`, `Logistic`, or `SoftMax`.

Its **integrator_mode** argument can switch the transformation from an "instantaneous"  to a "time averaged"
(integrated) manner of execution. When `integrator_mode <TransferMechanism.integrator_mode>` is set to True, the
mechanism's input is first transformed by its `integrator_function <TransferMechanism.integrator_function>` (
`AdaptiveIntegrator`). That result is then transformed by the mechanism's `function <Mechanism_Base.function>`.

.. _TransferMechanism_Creation:

Creating a TransferMechanism
-----------------------------

A TransferMechanism is created by calling its constructor.

Its `function <Mechanism_Base.function>` is specified in the **function** argument, which can be the name of a
`Function <Function>` class:

    >>> import psyneulink as pnl
    >>> my_linear_transfer_mechanism = pnl.TransferMechanism(function=pnl.Linear)

in which case all of the function's parameters will be set to their default values. Alternatively, the **function**
argument can be a call to a Function constructor, in which case values may be specified for the Function's parameters:

    >>> my_logistic_transfer_mechanism = pnl.TransferMechanism(function=pnl.Logistic(gain=1.0, bias=-4))

.. _TransferMechanism_Integrator_Mode:

Next, the **integrator_mode** argument allows the TransferMechanism to operate in either an "instantaneous" or
"time averaged" manner. By default, `integrator_mode <TransferMechanism.integrator_mode>` is set to False, meaning
execution is instantaneous. In order to switch to time averaging, the **integrator_mode** argument of the constructor
must be set to True.

    >>> my_logistic_transfer_mechanism = pnl.TransferMechanism(function=pnl.Logistic(gain=1.0, bias=-4),
    ...                                                        integrator_mode=True)

When `integrator_mode <TransferMechanism.integrator_mode>` is True, the TransferMechanism uses its `integrator_function
<TransferMechanism.integrator_function>` to integrate its variable on each execution. The output of the
`integrator_function  <TransferMechanism.integrator_function>` is then used as the input to `function
<Mechanism_Base.function>`.

By default, the `integrator_function <TransferMechanism.integrator_function>` of a TransferMechanism is
`AdaptiveIntegrator`.  However, any `IntegratorFunction` can be assigned. A TransferMechanism has three
parameters that
are used by most IntegratorFunctions:  `initial_value <TransferMechanism.initial_value>`, `integration_rate
<TransferMechanism.integration_rate>`, and `noise <TransferMechanism.noise>`.  If any of these are specified in the
TransferMechanism's constructor, their value is used to specify the corresponding parameter of its `integrator_function
<TransferMechanism.integrator_function>`.  In the following example::

    >>> my_logistic_transfer_mechanism = pnl.TransferMechanism(function=pnl.Logistic(gain=1.0, bias=-4),
    ...                                                        integrator_mode=True,
    ...                                                        initial_value=np.array([[0.2]]),
    ...                                                        integration_rate=0.1)

``my_logistic_transfer_mechanism`` will be assigned an `AdaptiveIntegrator` (the default) as its
`integrator_function
<TrasnferMechanism.integrator_function>`, with ``0.2`` as its `initializer <AdaptiveIntegrator.initializer>`
parameter,
and ``0.`` as its `rate <AdaptiveIntegrator.rate>` parameter.  However, in this example::

    >>> my_logistic_transfer_mechanism = pnl.TransferMechanism(function=pnl.Logistic(gain=1.0, bias=-4),
    ...                                                        integrator_mode=True,
    ...                                                        integrator_function=AdaptiveIntegrator(rate=0.3),
    ...                                                        initial_value=np.array([[0.2]]),
    ...                                                        integration_rate=0.1)

the AdaptiveIntegrator's `rate <AdaptiveIntegrator.rate>` parameter will be assigned ``0.3``,
and this will also
be assigned to the TransferMechanism's `integration_rate <TransferMechanism.integration_rate>` parameter, overriding
the specified value of ``0.1``.

.. note::
    If `integrator_mode <TransferMechanism.integrator_mode>` is False, then the arguments **integration_rate** and
    **initial_value** are ignored, as its `integrator_function <TransferMechanism.integrator_function>` is not executed.

When switching between `integrator_mode <TransferMechanism.integrator_mode>` = True and `integrator_mode
<TransferMechanism.integrator_mode>` = False, the behavior of the `integrator_function
<TransferMechanism.integrator_function>` is determined by `on_resume_integrator_mode
<TransferMechanism.on_resume_integrator_mode>`. There are three options for how the `integrator_function
<TransferMechanism.integrator_function>` may resume accumulating when the Mechanism returns to `integrator_mode
<TransferMechanism.integrator_mode>` = True.

    * *INSTANTANEOUS_MODE_VALUE* - reset the Mechanism with its own current value,
      so that the value computed by the Mechanism during "Instantaneous Mode" is where the
      `integrator_function <TransferMechanism.integrator_function>` begins accumulating.

    * *INTEGRATOR_MODE_VALUE* - resume accumulation wherever the `integrator_function
      <TransferMechanism.integrator_function>` left off the last time `integrator_mode
      <TransferMechanism.integrator_mode>` was True.

    * *RESET* - call the `integrator_function <TransferMechanism.integrator_function>`\\s
      `reset <AdaptiveIntegrator.reset>` method, so that accumulation begins at
      `initial_value <TransferMechanism.initial_value>`

Finally, the TransferMechanism has two arguments that can adjust the final result of the mechanism: **clip** and
**noise**. If `integrator_mode <TransferMechanism.integrator_mode>` is False, `clip <TransferMechanism.clip>` and
`noise <TransferMechanism.noise>` modify the value returned by the mechanism's `function <Mechanism_Base.function>`
before setting it as the mechanism's value. If `integrator_mode <TransferMechanism.integrator_mode>` is True,
**noise** is assigned to the TransferMechanism's `integrator_function <TransferMechanism.integrator_function>`
(as its `noise <IntegratorFunction.noise>` parameter -- in the same manner as `integration_rate
<TransferMechanism.integration_rate>` and `initial_value <TransferMechanism.intial_value>`), whereas `clip
<TransferMechanism.clip>` modifies the value returned by the mechanism's `function <Mechanism_Base.function>`
before setting it as the TransferMechanism's `value <Mechanism_Base.value>`.

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
Like any InputPorts, the `value <OutputPort.value>` of any or all of the TransferMechanism's InputPorts can be
modulated by one or more `GatingSignals <GatingSignal_Modulation>` prior to transformation by its `function
<Mechanism_Base.function>`.

.. _TransferMechanism_Function:

*Function*
~~~~~~~~~~

*Function*.  The default function for a TransferMechanism is `Linear`.  A custom function can be specified in the
**function** argument of the constructor.  This can be any PsyNeuLink `Function <Function>` that is a subtype of
either `TransferFunction` or `NormalizationFunction.` It can also be any python function or method, with the constraint
that it returns an output that is identical in shape to its input;  the function or method is "wrapped" as
`UserDefinedFunction`, and assigned as the TransferMechanism's `function <Mechanism_Base.function>` attribute.

The result of the `function <Mechanism_Base.function>` applied to the `value <InputPort.value>` of each InputPort
is:
    - appended to an array that represents the TransferMechanism's `value <Mechanism_Base.value>`
    - assigned as the `value <OutputPort.value>` of the TransferMechanism's corresponding `OutputPort <OutputPort>`

.. _TransferMechanism_OutputPorts:

*OutputPorts*
~~~~~~~~~~~~~

By default, or if the **output_ports** argument is specified using the keyword *RESULTS*, a TransferMechanism generates
one `OutputPort` for each item in the outer dimension (axis 0) of its `value <Mechanism_Base.value>` (each of which is
the result of the Mechanism's `function <Mechanism_Base.function>` applied to the `value <InputPort.value>` of the
corresponding `InputPort`).  If there is only one OutputPort (i.e., the case in which there is only one InputPort and
therefore only one item in Mechanism's `value <Mechanism_Base.value>`), the OutputPort is named *RESULT*.  If there is
more than one item in `value <Mechanism_Base.value>`, then an OuputPort is assigned for each;  the name of the first
is *RESULT-0*, and the names of the subsequent ones are suffixed with an integer that is incremented for each successive
one (e.g., *RESULT-1*, *RESULT-2*, etc.).  Additional OutputPorts can be assigned using the TransferMechanism's
`standard_output_ports <TransferMechanism.standard_output_ports>` (see `OutputPort_Standard`) or by creating `custom
OutputPorts <OutputPort_Customization>` (but see note below).   Like any OutputPorts, the `value <OutputPort.value>` of
any or all of these can be modulated by one or more `ControlSignals <ControlSignal_Modulation>` or `GatingSignals
<GatingSignal_Modulation>`.

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

COMMENT:
DESCRIBE AS TWO MODES (AKIN TO DDM):  INSTANTANEOUS AND TIME-AVERAGED
INSTANTANEOUS:
input transformed in a single `execution <TransferMechanism_Execution>` of the Mechanism)
TIME-AVERAGED:
input transformed using `step-wise` integration, in which each execution returns the result of a subsequent step of the
integration process).
COMMENT

When a TransferMechanism is executed, it transforms its input using its `function <Mechanism_Base.function>` and
the following parameters (in addition to any specified for the `function <Mechanism_Base.function>`):

    * `integrator_mode <TransferMechanism.integrator_mode>`: determines whether the input is time-averaged before
      passing through the function of the mechanism. When `integrator_mode <TransferMechanism.integrator_mode>` is set
      to True, the TransferMechanism integrates its input, by executing its `integrator_function
      <TransferMechanism.integrator_function>`, before executing its `function <Mechanism_Base.function>`. When
      `integrator_mode <TransferMechanism.integrator_mode>` is False, the `integrator_function
      <TransferMechanism.integrator_function>` is ignored, and time-averaging does not occur.

    * `integration_rate <TransferMechanism.integration_rate>`: if the `integrator_mode
    <TransferMechanism.integrator_mode>`
      attribute is set to True, the `integration_rate <TransferMechanism.integration_rate>` attribute is the rate of
      integration (a higher value specifies a faster rate); if `integrator_mode <TransferMechanism.integrator_mode>`
      is False,
      `integration_rate <TransferMechanism.integration_rate>` is ignored and time-averaging does not occur.

    * `noise <TransferMechanism.noise>`: applied element-wise to the output of its `integrator_function
      <TransferMechanism.integrator_function>` or its `function <Mechanism_Base.function>`, depending on whether
      `integrator_mode <TransferMechanism.integrator_mode>` is True or False.

    * `clip <TransferMechanism.clip>`: caps all elements of the `function <Mechanism_Base.function>` result by the
      lower and upper values specified by clip.

After each execution, the TransferMechanism's `function <Mechanism_Base.function>` -- applied to the `value
<InputPort.value>` of each of its `input_ports <Mechanism_Base.input_ports>` -- generates a corresponding set of
values, each of which is assigned as an item of the Mechanism's `value <Mechanism_Base.value>` attribute, and the
`value <OutputPort.value>` of the corresponding `OutputPort` in its `ouput_ports <Mechanism_Base.output_ports>`.

.. _TransferMechanism_Integration:

*Integration*
~~~~~~~~~~~~~

If `integrator_mode <TransferMechanism.integrator_mode>` is False (the default), then the TransferMechanism updates its
`value <Mechanism_Base.value>` and the `value <OutputPort.value>` of its `output_ports <Mechanism_Base.output_ports>`
without using its `integrator_function <TransferMechanism.integrator_function>`, as in the following example::

    >>> my_mech = pnl.TransferMechanism(size=2)
    >>> my_mech.execute([0.5, 1])
    array([[0.5, 1. ]])

Notice that the result is the full linear transfer of the input (i.e., no integration occured).

If `integrator_mode <TransferMechanism.integrator_mode>` is True, then it can be configured to conduct a single
step of integration per execution, or to continue to integrate until its termination condition is met, as specified
by the **termination_threshold**, **termination_measure**, and **termination_comparison_op** arguments, which are
assigned to the TransferMechanism's `termination_threshold <TransferMechanism.termination_threshold>`,
`termination_measure <TransferMechanism.termination_measure>`, and `termination_comparison_op
<TransferMechanism.termination_comparison_op>` attributes, respectively.

A single step of integration is executed if no **termination_threshold** is specified (i.e., it is None, the default),
as in the following example::

    >>> my_mech = pnl.TransferMechanism(size=2,
    ...                                 integrator_mode=True)
    >>> my_mech.execute([0.5, 1])
    array([[0.25, 0.5 ]])
    >>> my_mech.execute([0.5, 1])
    array([[0.375, 0.75 ]])
    >>> my_mech.execute([0.5, 1])
    array([[0.4375, 0.875 ]])

Notice that every call to the ``my_execute`` produces a single step of integration (at the default `rate
<TransferMechanism.rate>` of 0.5), by executing its `integrator_function <TransferMechanism.integrator_function>`
once.  A single step is also executed if the Mechanism's `execute_until_finished <Component.execute_until_finished>`
attribute is set to False, even if **termination_threshold** is specified. In both cases, the
`num_executions_before_finished <Component.num_executions_before_finished>` attribute remains equal to 1,
since the `integrator_function <TransferMechanism.integrator_function>` is executed exactly once per call to the
`execute method <Component_Execution>` (and no termination condition has been specified).

.. _TransferMechanism_Termination:

*Termination*
~~~~~~~~~~~~~

If `integrator_mode <TransferMechanism.integrator_mode>` is True, and a **termination_threshold** is specified, then the
TransferMechanism continues to execute, integrating its current input until its termination condition is met, or the
number of executions reaches `max_executions_before_finished <Component.max_executions_before_finished>`.  The numer of
executions that have taken place since the last time the termination condition was met is contained in
`num_executions_before_finished <Component.num_executions_before_finished>`; this is set to 0 each time the termination
condition is met.

   .. _TransferMechanism_Continued_Execution:

   .. note::
     Even after its termination condition is met, a TransferMechanism will continue to execute if it is called again,
     carrying out one step of integration each time it is called. This can be useful in cases where the initial
     execution of the Mechanism is meant to bring it to some state (e.g., as an initial "settling process"), after
     which subsequent executions are meant to occur in step with the execution of other Mechanisms in a Composition
     (see `example <TransferMechanism_Termination_By_Time>` below).

By default, `execute_until_finished <Component.execute_until_finished>` is True, and a convergence criterion is used to
terminate integration, as in the following example::

    >>> my_mech = pnl.TransferMechanism(size=2,
    ...                                 integrator_mode=True,
    ...                                 termination_threshold=0.1)
    >>> my_mech.execute([0.5, 1])
    array([[0.46875, 0.9375 ]])
    >>> my_mech.num_executions_before_finished
    4

In this case, the single call to ``my_mech.execute`` caused the Mechanism to integrate for 4 steps, until the
difference between its current `value <Mechanism_Base.value>` and its `previous value
<Mechanism_Base.previous_value>` is less than the specified **termination_threshold**.  However,
the **termination_measure** and **termination_comparison_op** arguments can be used to congifure other termination
conditions.  There are two broad types of termination condition:  convergence and boundary terination.

*Convergence termination* -- execution terminates based on the difference between the TransferMechanism's current
`value <Mechanism_Base.value>` and its `previous_value <Mechanism_Base.previous_value>` (as in the example above).
This is implemented by specifying **termination_measure** with a function that accepts a 2d array with *two items*
(1d arrays) as its argument, and returns a scalar (the default for a TransferMechanism is the `Distance` Function with
`MAX_ABS_DIFF` as its metric).  After each execution, the function is passed the Mechanism's current `value
<Mechanism_Base.value>` as well as its `previous_value <Mechanism_Base.previous_value>`, and the scalar returned is
compared to **termination_threshold** using the comparison operator specified by **termination_comparison_op** (which
is *LESS_THAN_OR_EQUAL* by default).  Execution continues until this returns True.  Thus, in the example above,
execution continued until the difference between the Mechanism's current `value <Mechanism_Base.value>` and
`previous_value <Mechanism_Base.previous_value>` was less than or equal to 0.1.  A `Distance` Function with other
metrics (e.g., *ENERGY* or *ENTROPY*) can be specified as the **termination_measure**, as can any other function that
accepts a single argument that is a 2d array with two entries.

*Boundary termination* -- Two types of boundaries can be specified:  value or time.

    *Termination by value*.  This terminates execution when the Mechanism's `value <Mechanism_Base.value>` reaches the
    the value specified by the **threshold** argument.  This implemented by specifying **termination_measure** with
    a function that accepts a 2d array with a *single entry* as its argument and returns a scalar.  The single
    entry is the TransferMechanism's current `value <Mechanism_Base.value>` (that is, `previous_value
    <Mechanism_Base.previous_value>` is ignored). After each execution, the function is passed the Mechanism's
    current `value <Mechanism_Base.value>`, and the scalar returned is compared to **termination_threshold** using
    the comparison operator specified by **termination_comparison_op**. Execution continues until this returns True,
    as in the following example::

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

    .. _TransferMechanism_Termination_By_Time:

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
    until one of the features of the input (the first or second) reaches the threshold of 0.65, at which point
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
    ``stim_percept``executes, followed by ``decision``.  However, the latter carries out only one step of integration,
    since its **execute_until_finished** is set to False.  If its output does not meet its termination condition after
    that one step of integration, then ``response`` does not execute, since it has been assigned a condition that
    requires ``deciions`` to terminate before it does so. As a result, since ``response`` has not executed, the trial
    continues (see XXX for a full description of XXX). On the next pass, ``attention`` carries out only one step of
    integration, since its termination condition has already been met, as does ``decision`` since its termination
    condition has *not* yet been met.  If it is met, then ``response`` executes and the trial ends (since all
    Mechanisms have now had an opportunity to execute). The value of the ``attention`` and ``decision`` Mechanisms
    after each execution are shown below::

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


.. _TransferMechanism_Reinitialization:

*Reinitialization*
~~~~~~~~~~~~~~~~~~

In some cases, it may be useful to reset the accumulation of a Mechanism back to its original starting point, or a new
starting point. This is done using the `reset <AdaptiveIntegrator.reset>` method on the
mechanism's `integrator_function <TransferMechanism.integrator_function>`, or the mechanisms's own `reset
<Mechanism_Base.reset>` method.

The `reset <AdaptiveIntegrator.reset>` method of the `integrator_function
<TransferMechanism.integrator_function>` sets:

    - the integrator_function's `previous_value <AdaptiveIntegrator.previous_value>` attribute and
    - the integrator_function's `value <AdaptiveIntegrator.value>` attribute

    to the specified value.

The `reset <Mechanism_Base.reset>` method of the `TransferMechanism` first sets:

    - the Mechanismn's `previous_value <Mechanism_Base.previous_value>` attribute,
    - the integrator_function's `previous_value <AdaptiveIntegrator.previous_value>` attribute, and
    - the integrator_function's `value <AdaptiveIntegrator.value>` attribute

    to the specified value. Then:

    - the specified value is passed into the mechanism's `function <Mechanism_Base.function>` and the function is
    executed
    - the TransferMechanism's `value <Mechanism_Base.value>` attribute is set to the output of the function
    - the TransferMechanism updates its `output_ports <Mechanism_Base.output_ports>`

A use case for `reset <AdaptiveIntegrator.reset>` is demonstrated in the following example:

Create a `System` with a TransferMechanism in integrator_mode:

    >>> my_time_averaged_transfer_mechanism = pnl.TransferMechanism(function=pnl.Linear,        #doctest: +SKIP
    ...                                                        integrator_mode=True,            #doctest: +SKIP
    ...                                                        integration_rate=0.1,            #doctest: +SKIP
    ...                                                        initial_value=np.array([[0.2]])) #doctest: +SKIP
    >>> my_process = pnl.Process(pathway=[my_time_averaged_transfer_mechanism]) #doctest: +SKIP
    >>> my_system = pnl.System(processes=[my_process])  #doctest: +SKIP

Then run the system for 5 trials:

    >>> # RUN 1:
    >>> my_system.run(inputs={my_time_averaged_transfer_mechanism: [1.0]},        #doctest: +SKIP
    ...               num_trials=5)                                               #doctest: +SKIP
    >>> assert np.allclose(my_time_averaged_transfer_mechanism.value,  0.527608)  #doctest: +SKIP

After RUN 1, my_time_averaged_transfer_mechanism's integrator_function will preserve its state (its position along its
path of integration).

Run the system again to observe that my_time_averaged_transfer_mechanism's integrator_function continues accumulating
where it left off:

    >>> # RUN 2:
    >>> my_system.run(inputs={my_time_averaged_transfer_mechanism: [1.0]},          #doctest: +SKIP
    ...               num_trials=5)                                                 #doctest: +SKIP
    >>> assert np.allclose(my_time_averaged_transfer_mechanism.value,  0.72105725)  #doctest: +SKIP

The integrator_function's `reset <AdaptiveIntegrator.reset>` method and the TransferMechanism's
`reset <TransferMechanism.reset>` method are useful in cases when the integration should instead start
over at the original initial value, or a new one.

Use `reset <AdaptiveIntegrator.reset>` to re-start the integrator_function's accumulation at 0.2:

    >>> my_time_averaged_transfer_mechanism.integrator_function.reset(np.array([[0.2]]))  #doctest: +SKIP

Run the system again to observe that my_time_averaged_transfer_mechanism's integrator_function will begin accumulating
at 0.2, following the exact same trajectory as in RUN 1:

    >>> # RUN 3
    >>> my_system.run(inputs={my_time_averaged_transfer_mechanism: [1.0]},        #doctest: +SKIP
    ...               num_trials=5)                                               #doctest: +SKIP
    >>> assert np.allclose(my_time_averaged_transfer_mechanism.value,  0.527608)  #doctest: +SKIP

Because `reset <AdaptiveIntegrator.reset>` was set to 0.2 (its original initial_value),
my_time_averaged_transfer_mechanism's integrator_function effectively started RUN 3 in the same state as it began RUN 1.
As a result, it arrived at the exact same value after 5 trials (with identical inputs).

In the examples above, `reset <AdaptiveIntegrator.reset>` was applied directly to the
integrator function. The key difference between the `integrator_function's reset
<AdaptiveIntegrator.reset>` and the `TransferMechanism's reset <TransferMechanism.reset>` is
that the latter will also execute the mechanism's function and update its output ports. This is useful if the
mechanism's value or any of its OutputPort values will be used or checked *before* the mechanism's next execution. (
This may be true if, for example, the mechanism is `recurrent <RecurrentTransferMechanism>`, the mechanism is
responsible for `modulating <ModulatorySignal_Modulation` other components, or if a `Scheduler` condition depends on
the mechanism's activity.)

COMMENT:
.. _TransferMechanism_Examples:

Examples
--------

EXAMPLES HERE
COMMENT

.. _TransferMechanism_Class_Reference:

Class Reference
---------------

"""
import copy
import inspect
import numbers
import warnings
import logging
import types
from collections.abc import Iterable

import numpy as np
import typecheck as tc

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.functions.distributionfunctions import DistributionFunction
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import AdaptiveIntegrator
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import IntegratorFunction
from psyneulink.core.components.functions.function import Function, is_function_type
from psyneulink.core.components.functions.objectivefunctions import Distance
from psyneulink.core.components.functions.selectionfunctions import SelectionFunction
from psyneulink.core.components.functions.transferfunctions import Linear, Logistic, TransferFunction
from psyneulink.core.components.functions.combinationfunctions import LinearCombination, SUM
from psyneulink.core.components.functions.userdefinedfunction import UserDefinedFunction
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import _is_control_spec
from psyneulink.core.components.mechanisms.mechanism import Mechanism, MechanismError
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import \
    COMBINE, comparison_operators, EXECUTION_COUNT, FUNCTION, GREATER_THAN_OR_EQUAL, \
    INSTANTANEOUS_MODE_VALUE, LESS_THAN_OR_EQUAL, MAX_ABS_DIFF, \
    NAME, NOISE, NUM_EXECUTIONS_BEFORE_FINISHED, OWNER_VALUE, RESET, RESULT, RESULTS, \
    SELECTION_FUNCTION_TYPE, TRANSFER_FUNCTION_TYPE, TRANSFER_MECHANISM, VARIABLE
from psyneulink.core.globals.parameters import Parameter, FunctionParameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import \
    all_within_range, append_type_to_name, iscompatible, is_comparison_operator, convert_to_np_array
from psyneulink.core.scheduling.condition import TimeScale
from psyneulink.core.globals.registry import remove_instance_from_registry, register_instance

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
                if owning_component.on_resume_integrator_mode == INSTANTANEOUS_MODE_VALUE:
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
        integrator_mode=False,                               \
        integrator_function=AdaptiveIntegrator,              \
        on_resume_integrator_mode=INSTANTANEOUS_MODE_VALUE,  \
        initial_value=None,                                  \
        integration_rate=0.5,                                \
        noise=0.0,                                           \
        clip=[float:min, float:max],                         \
        termination_measure=Distance(metric=MAX_ABS_DIFF),   \
        termination_threshold=None,                          \
        termination_comparison_op=LESS_THAN_OR_EQUAL,        \
        output_ports=RESULTS                                 \
        )

    Subclass of `ProcessingMechanism <ProcessingMechanism>` that performs a simple transform of its input.
    See `Mechanism <Mechanism_Class_Reference>` for additional arguments and attributes.

    Arguments
    ---------

    integrator_mode : bool : False
        specifies whether or not the TransferMechanism should be executed using its `integrator_function
        <TransferMechanism>` to integrate its `variable <Mechanism_Base.variable>` (
        when set to `True`), or simply report the asymptotic value of the output of its `function
        <Mechanism_Base.function>` (when set to `False`).

    integrator_function : IntegratorFunction : default AdaptiveIntegrator
        specifies `IntegratorFunction` to use in `integration_mode <TransferMechanism.integration_mode>`.

    initial_value :  value, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the starting value for time-averaged input (only relevant if `integrator_mode
        <TransferMechanism.integrator_mode>` is True).
        COMMENT:
            Transfer_DEFAULT_BIAS SHOULD RESOLVE TO A VALUE
        COMMENT

    integration_rate : float : default 0.5
        specifies the rate of integration of `variable <Mechanism_Base.variable>` when the TransferMechanism is
        executed with `integrator_mode` set to `True`.

    on_resume_integrator_mode : keyword : default INSTANTANEOUS_MODE_VALUE
        specifies how the `integrator_function <TransferMechanism.integrator_function>` should resume its accumulation
        when the Mechanism was most recently in "Instantaneous Mode" (`integrator_mode
        <TransferMechanism.intergrator_mode>` = False) and has just switched to "IntegratorFunction Mode"
        (`integrator_mode <TransferMechanism.intergrator_mode>` = True);  can be one of the following keywords:

        * *INSTANTANEOUS_MODE_VALUE* - reset the Mechanism with its own current value,
          so that the value computed by the Mechanism during "Instantaneous Mode" is where the
          `integrator_function <TransferMechanism.integrator_function>` begins accumulating.

        * *INTEGRATOR_MODE_VALUE* - resume accumulation wherever the `integrator_function
          <TransferMechanism.integrator_function>` left off the last time `integrator_mode
          <TransferMechanism.integrator_mode>` was True.

        * *RESET* - call the `integrator_function <TransferMechanism.integrator_function>`\\s
          `reset <AdaptiveIntegrator.reset>` method, so that accumulation begins at
          `initial_value <TransferMechanism.initial_value>`

    noise : float or function : default 0.0
        specifies a value to be added to the result of the TransferMechanism's `function <Mechanism_Base.function>`
        or its `integrator_function <TransferMechanism.integrator_function>`, depending on whether `integrator_mode
        <TransferMechanism.integrator_mode>` is `True` or `False`. See `noise <TransferMechanism.noise>` for details.

    clip : list [float, float] : default None (Optional)
        specifies the allowable range for the result of `function <Mechanism_Base.function>`. The item in index 0
        specifies the minimum allowable value of the result, and the item in index 1 specifies the maximum allowable
        value; any element of the result that exceeds the specified minimum or maximum value is set to the value of
        `clip <TransferMechanism.clip>` that it exceeds.

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

    integrator_mode : bool
        determines whether the TransferMechanism uses its `integrator_function <TransferMechanism.integrator_function>`
        to integrate its `variable <Mechanism_Base.variable>` when it executes.

        **If integrator_mode is set to** `True`:

            the TransferMechanism's `variable <TransferMechanism>` is first passed to its `integrator_function
            <TransferMechanism.integrator_function>`, and then the result is passed to the TransferMechanism's
            `function <Mechanism_Base.function>` which computes the TransferMechanism's `value
            <Mechanism_Base.value>`.

            .. note::
                The TransferMechanism's `integration_rate <TransferMechanism.integration_rate>`, `noise
                <TransferMechanism.noise>`, and `initial_value <TransferMechanism.initial_value>` parameters
                specify the respective parameters of its `integrator_function <TransferMechanism.integrator_function>`
                (with `initial_value <TransferMechanism.initial_value>` corresponding to `initializer
                <IntegratorFunction.initializer>` and `integration_rate <TransferMechanism.integration_rate>`
                corresponding to `rate <IntegratorFunction.rate>` of `integrator_function
                <TransferMechanism.integrator_function>`). However, if there are any disagreements between these
                (e.g., any of these parameters is specified in the constructor for an `IntegratorFunction` assigned
                as the **integrator_function** arg of the TransferMechanism), the values specified for the
                `integrator_function <TransferMechanism.integrator_function>` take precedence, and their value(s) are
                assigned as those of the corresponding parameters on the TransferMechanism.

        **If integrator_mode is set to** `False`:

            if `noise <TransferMechanism.noise>` is non-zero, it is applied to the TransferMechanism's `variable
            <TransferMechanism>` which is htne passed directly to its `function <Mechanism_Base.function>`
             -- that is, its `integrator_function <TransferMechanism.integrator_function>` is bypassed,
             and its related attributes (`initial_value <TransferMechanism.initial_value>` and
            `integration_rate <TransferMechanism.integration_rate>`) are ignored.

    integrator_function :  IntegratorFunction
        the `IntegratorFunction` used when `integrator_mode <TransferMechanism.integrator_mode>` is set to
        `True` (see `integrator_mode <TransferMechanism.integrator_mode>` for details).

    initial_value :  value, list or np.ndarray
        determines the starting value for the `integrator_function <TransferMechanism.integrator_function>`;  only
        relevant if `integrator_mode <TransferMechanism.integrator_mode>` is `True` and `integration_rate
        <TransferMechanism.integration_rate>` is not 1.0 (see `integrator_mode <TransferMechanism.integrator_mode>`
        for additional details).

    integration_rate : float
        the rate at which the TransferMechanism's `variable <TransferMechanism>` is integrated when it is executed with
        `integrator_mode <TransferMechanism.integrator_mode>` set to `True` (see `integrator_mode
        <TransferMechanism.integrator_mode>` for additional details).

    on_resume_integrator_mode : keyword
        specifies how the `integrator_function <TransferMechanism.integrator_function>` should resume its accumulation
        when the Mechanism was most recently in "Instantaneous Mode" (integrator_mode = False) and has just switched to
        "IntegratorFunction Mode" (integrator_mode = True). There are three options:

        (1)     INSTANTANEOUS_MODE_VALUE - reset the Mechanism with its own current value, so that the value
                cmoputed by the Mechanism during "Instantaneous Mode" is where the `integrator_function
                <TransferMechanism.integrator_function>` begins accumulating.

        (2)     INTEGRATOR_MODE_VALUE - resume accumulation wherever the `integrator_function
                <TransferMechanism.integrator_function>` left off the last time `integrator_mode
                <TransferMechanism.integrator_mode>` was True.

        (3)     RESET - call the `integrator_function's <TransferMechanism.integrator_function>` `reset
                method <AdaptiveIntegrator.reset>` so that accumulation Mechanism begins at `initial_value
                <TransferMechanism.initial_value>`

    noise : float or function
        When `integrator_mode <TransferMechanism.integrator_mode>` is set to `True`, `noise <TransferMechanism.noise>`
        is passed into the `integrator_function <TransferMechanism.integrator_function>` (see `integrator_mode
        <TransferMechanism.integrator_mode>` for additional details). Otherwise, noise is added to the output of the
        `function <Mechanism_Base.function>`. If `noise <TransferMechanism.noise>` is a list or array,
        it must be the same length as `variable <TransferMechanism.default_variable>`. If `noise
        <TransferMechanism.noise>` is specified as a single float or function, while `variable
        <Mechanism_Base.variable>` is a list or array, `noise <TransferMechanism.noise>` will be applied to each
        element of `variable <Mechanism_Base.variable>`. In the case that `noise <TransferMechanism.noise>` is
        specified as  a function, the function will be executed separately for each element of `variable
        <Mechanism_Base.variable>`.

        .. note::
            In order to generate random noise, a probability distribution function should be used (see `Distribution
            Functions <DistributionFunction>` for details), that will generate a new noise value from its
            distribution on each execution. If `noise <TransferMechanism.noise>` is specified as a float or as a
            function with a fixed output, then the noise will simply be an offset that remains the same across all
            executions.

    clip : list [float, float]
        specifies the allowable range for the result of `function <Mechanism_Base.function>`.  The 1st item (index
        0) specifies the minimum allowable value of the result, and the 2nd item (index 1) specifies the maximum
        allowable value; any element of the result that exceeds the specified minimum or maximum value is set to
        the value of `clip <TransferMechanism.clip>` that it exceeds.

    termination_measure : function or TimeScale
        used to determine when execution of the TransferMechanism is complete (i.e., `is_finished` is True), if
        `execute_until_finished <Component.execute_until_finished>` is True.  If it is a `TimeScale`, then execution
        terminates when the value of the Mechanism's `num_executions <Compnent_Num_Executions>` at that TimeScale is
        is equal to `termination_threshold <TransferMechanism.termination_threshold>`.  If it is a function, it is
        passed the `value <Mechanism_Base.value>` and `previous_value <Mechanism_Base.previous_value>` of the
        TransferMechanism; its result (`termination_measure_value <TransferMechanism.termination_measure_value>`) is
        compared with `termination_threshold <TransferMechanism.termination_threshold>` using
        `TransferMechanism.termination_comparison_op`, the result of which is used as the value of `is_finished`.

        .. note::
           A Mechanism's `previous_value` attribute is distinct from the `previous_value
           <AdaptiveIntegrator.previous_value>` attribute of its `integrator_function
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

                    :default value: `INSTANTANEOUS_MODE_VALUE`
                    :type: ``str``

                output_ports
                    see `output_ports <TransferMechanism.output_ports>`

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
        on_resume_integrator_mode = Parameter(INSTANTANEOUS_MODE_VALUE, stateful=False, loggable=False)
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
        termination_measure_value = Parameter(0.0, modulable=False, read_only=True)

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

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_ports:tc.optional(tc.any(Iterable, Mechanism, OutputPort, InputPort))=None,
                 function=None,
                 integrator_mode=None,
                 integrator_function=None,
                 initial_value=None,
                 integration_rate=None,
                 on_resume_integrator_mode=None,
                 noise=None,
                 clip=None,
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
                    raise TransferError("Function specified as {} param of {} ({}) must be a {}".
                                        format(repr(FUNCTION), self.name, transfer_function_class.__name__,
                                               " or ".join([TRANSFER_FUNCTION_TYPE, SELECTION_FUNCTION_TYPE])))
            elif not isinstance(transfer_function, (types.FunctionType, types.MethodType)):
                raise TransferError("Unrecognized specification for {} param of {} ({})".
                                    format(repr(FUNCTION), self.name, transfer_function))

            # FUNCTION is a function or method, so test that shape of output = shape of input
            if isinstance(transfer_function, (types.FunctionType, types.MethodType, UserDefinedFunction)):
                var_shape = self.defaults.variable.shape
                if isinstance(transfer_function, UserDefinedFunction):
                    val_shape = transfer_function._execute(self.defaults.variable, context=context).shape
                else:
                    val_shape = np.array(transfer_function(self.defaults.variable, context=context)).shape

                if val_shape != var_shape:
                    raise TransferError("The shape ({}) of the value returned by the Python function, method, or UDF "
                                        "specified as the {} param of {} must be the same shape ({}) as its {}".
                                        format(val_shape, repr(FUNCTION), self.name, var_shape, repr(VARIABLE)))

        # Validate INITIAL_VALUE
        if INITIAL_VALUE in target_set and target_set[INITIAL_VALUE] is not None:
            initial_value = np.array(target_set[INITIAL_VALUE])
            if (
                not iscompatible(initial_value, self.defaults.variable)
                # extra conditions temporary until universal initializer
                # validation is developed
                and initial_value.shape != self.integrator_function.defaults.variable.shape
                and self._get_parsed_variable(self.parameters.integrator_function, initial_value).shape != self.integrator_function.defaults.variable.shape
            ):
                raise TransferError(
                        "The format of the initial_value parameter for {} ({}) must match its variable ({})".
                        format(append_type_to_name(self), initial_value, self.defaults.variable,
                    )
                )

        # FIX: SHOULD THIS (AND INTEGRATION_RATE) JUST BE VALIDATED BY INTEGRATOR FUNCTION NOW THAT THEY ARE PROPERTIES??
        # Validate NOISE:
        if NOISE in target_set:
            noise = target_set[NOISE]
            # If assigned as a Function, set TransferMechanism as its owner, and assign its actual function to noise
            if isinstance(noise, DistributionFunction):
                target_set[NOISE] = noise.execute
            self._validate_noise(target_set[NOISE])

        # Validate INTEGRATOR_FUNCTION:
        if INTEGRATOR_FUNCTION in target_set and target_set[INTEGRATOR_FUNCTION] is not None:
            integtr_fct = target_set[INTEGRATOR_FUNCTION]
            if not (isinstance(integtr_fct, IntegratorFunction)
                    or (isinstance(integtr_fct, type) and issubclass(integtr_fct, IntegratorFunction))):
                raise TransferError("The function specified for the {} arg of {} ({}) must be an {}".
                                    format(repr(INTEGRATOR_FUNCTION), self.name, integtr_fct),
                                    IntegratorFunction.__class__.__name__)

        # Validate INTEGRATION_RATE:
        if INTEGRATION_RATE in target_set and target_set[INTEGRATION_RATE] is not None:
            integration_rate = np.array(target_set[INTEGRATION_RATE])
            if (not np.isscalar(integration_rate.tolist())
                    and integration_rate.shape != self.defaults.variable.squeeze().shape):
                raise TransferError("{} arg for {} ({}) must be either an int or float, "
                                    "or have the same shape as its {} ({})".
                                    format(repr(INTEGRATION_RATE), self.name, integration_rate,
                                           VARIABLE, self.defaults.variable))

        # Validate CLIP:
        if CLIP in target_set and target_set[CLIP] is not None:
            clip = target_set[CLIP]
            if clip:
                if not (isinstance(clip, (list,tuple)) and len(clip)==2 and all(isinstance(i, numbers.Number)
                                                                                for i in clip)):
                    raise TransferError("clip parameter ({}) for {} must be a tuple with two numbers".
                                        format(clip, self.name))
                if not clip[0] < clip[1]:
                    raise TransferError("The first item of the clip parameter ({}) must be less than the second".
                                        format(clip, self.name))
            target_set[CLIP] = list(clip)

    def _validate_noise(self, noise):
        # Noise is a list or array

        if isinstance(noise, (np.ndarray, list)):
            if len(noise) == 1:
                pass
            # Variable is a list/array
            elif not iscompatible(np.atleast_2d(noise), self.defaults.variable) and len(noise) > 1:
                raise MechanismError(
                    "Noise parameter ({}) does not match default variable ({}). Noise parameter of {} must be specified"
                    " as a float, a function, or an array of the appropriate shape ({})."
                    .format(noise, self.defaults.variable, self.name, np.shape(np.array(self.defaults.variable))))
            else:
                for i in range(len(noise)):
                    if isinstance(noise[i], DistributionFunction):
                        noise[i] = noise[i].execute
                    if not isinstance(noise[i], (float, int)) and not callable(noise[i]):
                        raise MechanismError("The elements of a noise list or array must be floats or functions. "
                            "{} is not a valid noise element for {}".format(noise[i], self.name))

        elif _is_control_spec(noise):
            pass

        # Otherwise, must be a float, int or function
        elif noise is not None and not isinstance(noise, (float, int)) and not callable(noise):
            raise MechanismError("Noise parameter ({}) for {} must be a float, "
                                 "function, or array/list of these.".format(noise,
                                                                            self.name))

    def _try_execute_param(self, param, var):

        # param is a list; if any element is callable, execute it
        if isinstance(param, (np.ndarray, list)):
            # NOTE: np.atleast_2d will cause problems if the param has "rows" of different lengths
            param = np.atleast_2d(param)
            for i in range(len(param)):
                for j in range(len(param[i])):
                    if callable(param[i][j]):
                        param[i][j] = param[i][j]()

        # param is one function
        elif callable(param):
            # NOTE: np.atleast_2d will cause problems if the param has "rows" of different lengths
            new_param = []
            for row in np.atleast_2d(var):
                new_row = []
                for item in row:
                    new_row.append(param())
                new_param.append(new_row)
            param = new_param

        return param

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
                self.output_ports = [RESULT]
            else:
                self.output_ports = []
                for i, item in enumerate(self.defaults.variable):
                    self.output_ports.append({NAME: f'{RESULT}-{i}', VARIABLE: (OWNER_VALUE, i)})
        super()._instantiate_output_ports(context=context)

        # # Relabel first output_port:
        # #    default (assigned by Mechanism's OutputPort registry) is to name it "RESULT";
        # #    but in this context, explicitly adding -0 index helps put first one on par with others
        # #    (i.e., make clear the alignment of each OutputPort with the items of the TransferMechanmism's value).
        # remove_instance_from_registry(registry=self._portRegistry,
        #                               category=OUTPUT_PORT,
        #                               component=self.output_ports['RESULT'])
        # register_instance(self.output_ports['RESULT'], 'RESULT-0', OutputPort, self._portRegistry, OUTPUT_PORT)

    def _get_instantaneous_function_input(self, function_variable, noise):
        noise = self._try_execute_param(noise, function_variable)
        if (np.array(noise) != 0).any():
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

        # If modulated, termination threshold is single element array
        if isinstance(threshold_ptr.type.pointee, pnlvm.ir.ArrayType):
            assert len(threshold_ptr.type.pointee) == 1
            threshold_ptr = builder.gep(threshold_ptr, [ctx.int32_ty(0),
                                                        ctx.int32_ty(0)])

        threshold = builder.load(threshold_ptr)
        cmp_val_ptr = builder.alloca(threshold.type)
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
                warnings.warn("Shape mismatch: Termination measure input: {} should be {}".format(self.termination_measure.defaults.variable, expected.shape))
                # FIXME: HACK the distance function is not initialized
                self.termination_measure.defaults.variable = expected

            func = ctx.import_llvm_function(self.termination_measure)
            func_params = pnlvm.helpers.get_param_ptr(builder, self, params, "termination_measure")
            func_state = pnlvm.helpers.get_state_ptr(builder, self, state, "termination_measure")
            func_in = builder.alloca(func.args[2].type.pointee)
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
            ptr = builder.gep(pnlvm.helpers.get_state_ptr(builder, self, state, "num_executions"), [ctx.int32_ty(0), ctx.int32_ty(self.termination_measure.value)])
            ptr_val = builder.sitofp(builder.load(ptr), threshold.type)
            pnlvm.helpers.printf(builder, f"TERM MEASURE {self.termination_measure} %d %d\n",ptr_val, threshold)
            builder.store(ptr_val, cmp_val_ptr)
        else:
            assert False, "Not Supported: {}".format(self.termination_measure)

        cmp_val = builder.load(cmp_val_ptr)
        cmp_str = self.parameters.termination_comparison_op.get(None)
        return builder.fcmp_ordered(cmp_str, cmp_val, threshold)

    def _gen_llvm_mechanism_functions(self, ctx, builder, params, state, arg_in,
                                      ip_out, *, tags:frozenset):

        if self.integrator_mode:
            if_state = pnlvm.helpers.get_state_ptr(builder, self, state,
                                                   "integrator_function")
            if_param_ptr = pnlvm.helpers.get_param_ptr(builder, self, params,
                                                       "integrator_function")
            if_params, builder = self._gen_llvm_param_ports_for_obj(
                    self.integrator_function, if_param_ptr, ctx, builder,
                    params, state, arg_in)

            mf_in, builder = self._gen_llvm_invoke_function(
                    ctx, builder, self.integrator_function, if_params, if_state, ip_out, tags=tags)
        else:
            mf_in = ip_out

        mf_state = pnlvm.helpers.get_state_ptr(builder, self, state, "function")
        mf_param_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, "function")
        mf_params, builder = self._gen_llvm_param_ports_for_obj(
                self.function, mf_param_ptr, ctx, builder, params, state, arg_in)

        mf_out, builder = self._gen_llvm_invoke_function(ctx, builder, self.function, mf_params, mf_state, mf_in, tags=tags)

        clip_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, "clip")
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

    def _execute(self,
        variable=None,
        context=None,
        runtime_params=None,

    ):
        """Execute TransferMechanism function and return transform of input

        Execute TransferMechanism function on input, and assign to output_values:
            - Activation value for all units
            - Mean of the activation values across units
            - Variance of the activation values across units
        Return:
            value of input transformed by TransferMechanism function in outputPort[TransferOuput.RESULT].value
            mean of items in RESULT outputPort[TransferOuput.MEAN].value
            variance of items in RESULT outputPort[TransferOuput.VARIANCE].value

        Arguments:

        # CONFIRM:
        variable (float): set to self.value (= self.input_value)
        - params (dict):  runtime_params passed from Mechanism, used as one-time value for current execution:
            + NOISE (float)
            + INTEGRATION_RATE (float)
            + RANGE ([float, float])
        - context (str)

        Returns the following values in self.value (2D np.array) and in
            the value of the corresponding outputPort in the self.output_ports list:
            - activation value (float)
            - mean activation value (float)
            - standard deviation of activation values (float)

        :param self:
        :param variable (float)
        :param params: (dict)
        :param context: (str)
        :rtype self.outputPort.value: (number)
        """

        # FIX: ??CALL check_args()??

        # FIX: IS THIS CORRECT?  SHOULD THIS BE SET TO INITIAL_VALUE
        # FIX:     WHICH SHOULD BE DEFAULTED TO 0.0??
        # Use self.defaults.variable to initialize state of input

        # EXECUTE TransferMechanism FUNCTION ---------------------------------------------------------------------

        # FIX: JDC 7/2/18 - THIS SHOULD BE MOVED TO A STANDARD OUTPUT_PORT
        # Clip outputs
        clip = self.parameters.clip._get(context)

        value = super(Mechanism, self)._execute(variable=variable,
                                                context=context,
                                                runtime_params=runtime_params,

                                                )
        value = self._clip_result(clip, value)

        return value

    @handle_external_context(fallback_most_recent=True)
    def reset(self, *args, force=False, context=None, **kwargs):
        super().reset(*args, force=force, context=context, **kwargs)
        self.parameters.value.clear_history(context)

    def _parse_function_variable(self, variable, context=None):
        if context.source is ContextFlags.INSTANTIATE:

            return super(TransferMechanism, self)._parse_function_variable(variable=variable, context=context)

        # FIX: NEED TO GET THIS TO WORK WITH CALL TO METHOD:
        integrator_mode = self.parameters.integrator_mode._get(context)
        noise = self._get_current_parameter_value(self.parameters.noise, context)

        # Update according to time-scale of integration
        if integrator_mode:
            value = self.integrator_function.execute(variable, context=context)
            self.parameters.integrator_function_value._set(value, context)
            return value

        else:
            return self._get_instantaneous_function_input(variable, noise)

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

    def _report_mechanism_execution(self, input, params, output, context=None):
        """Override super to report previous_input rather than input, and selected params
        """
        # KAM Changed 8/29/17 print_input = self.previous_input --> print_input = input
        # because self.previous_input is not a valid attrib of TransferMechanism

        print_input = input
        print_params = params.copy()
        # Suppress reporting of range (not currently used)
        del print_params[CLIP]

        super()._report_mechanism_execution(input_val=print_input, params=print_params, context=context)

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

        assert self.parameters.value.history_min_length + 1 >= self._termination_measure_num_items_expected, "History of 'value' is not guaranteed enough entries for termination_mesasure"
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
