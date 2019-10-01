# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  ModulatoryMechanism ************************************************


"""
Sections
--------

  * `ModulatoryMechanism_Overview`
  * `ModulatoryMechanism_Creation`
      - `ModulatoryMechanism_Monitor_for_Modulation`
      - `ModulatoryMechanism_ObjectiveMechanism`
      - `ModulatoryMechanism_Modulatory_Signals`
  * `ModulatoryMechanism_Structure`
      - `ModulatoryMechanism_Input`
      - `ModulatoryMechanism_Function`
      - `ModulatoryMechanism_Output`
  * `ModulatoryMechanism_Execution`
  * `ModulatoryMechanism_Examples`
  * `ModulatoryMechanism_Class_Reference`

.. _ModulatoryMechanism_Overview:

Overview
--------

A ModulatoryMechanism is an `AdaptiveMechanism <AdaptiveMechanism>` that `modulates the value(s)
<ModulatorySignal_Modulation>` of one or more `States <State>` of other Mechanisms in the `Composition` to which it
belongs.  It's `function <ModulatoryMechanism.function>` calculates a `modulatory_allocation
<ModulatoryMechanism.modulatory_allocation>`: a list of values provided to each of its `modulatory_signals
<ModulatoryMechanism.modulatory_signals>`.  These can be `ControlSignals <ControlSignal>` and/or `GatingSignals
<GatingSignal>`.  In general, `ControlSignals <ControlSignal>` are used to modulate the value of a `ParameterState
<ParameterState>` of another Mechanism, but can also be used to modulate the value of an `InputState` or
`OutputState`.  `GatingSignals <GatingSignal>` are restricting to modulating only an `InputState` or `OutputState`.
A ModulatoryMechanism can be configured to monitor the outputs of other Mechanisms in order to determine its
`modulatory_allocation <ModulatoryMechanism.modulatory_allocation>`, by specifying these in the
**monitor_for_modulation** `argument <ModulatoryMechanism_Monitor_for_Modulation_Argument>` of its constructor, or in
the **monitor** `argument <ObjectiveMechanism_Monitor>` of an ObjectiveMechanism` assigned to its
**objective_mechanism** `argument <ModulatoryMechanism_Objective_Mechanism_Argument>` (see
`ModulatoryMechanism_Monitor_for_Modulation` below).


.. _ModulatoryMechanism_Creation:

Creating a ModulatoryMechanism
------------------------------

A ModulatoryMechanism is created by calling its constructor.  When a ModulatoryMechanism is created, the OutputStates it
monitors and the States it modulates can be specified can be specified in the **montior_for_modulation** and
**objective_mechanism** arguments of its constructor, respectively.  Each can be specified in several ways,
as described below. If neither of those arguments is specified, then only the ModulatoryMechanism is constructed,
and its inputs and the parameters it modulates must be specified in some other way.


.. _ModulatoryMechanism_Monitor_for_Modulation:

*Specifying OutputStates to be monitored*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ModulatoryMechanism can be configured to monitor the output of other Mechanisms directly (by receiving direct
Projections from their OutputStates), or by way of an `ObjectiveMechanism` that evaluates those outputs and passes the
result to the ModulatoryMechanism (see `below <ModulatoryMechanism_ObjectiveMechanism>` for more detailed description).

Which configuration is used is determined by how the following arguments of the ModulatoryMechanism's constructor are
specified:

  .. _ModulatoryMechanism_Monitor_for_Modulation_Argument:

  * **monitor_for_modulation** -- a list of `OutputState specifications <OutputState_Specification>`.  If the
    **objective_mechanism** argument is not specified (or is *False* or *None*) then, when the ModulatoryMechanism is
    added to a `Composition`, a `MappingProjection` is created for each OutputState specified to the
    ModulatoryMechanism's *OUTCOME* `input_state <ModulatoryMechanism_Input>`.  If the **objective_mechanism** `argument
    <ModulatoryMechanism_Objective_Mechanism_Argument>` is specified, then the OutputStates specified in
    **monitor_for_modulation** are assigned to the `ObjectiveMechanism` rather than the ModulatoryMechanism itself (see
    `ModulatoryMechanism_ObjectiveMechanism` for details).

  .. _ModulatoryMechanism_Objective_Mechanism_Argument:

  * **objective_mechanism** -- if this is specfied in any way other than **False** or **None** (the default),
    then an ObjectiveMechanism is created that projects to the ModulatoryMechanism and, when added to a `Composition`,
    is assigned Projections from all of the OutputStates specified either in the  **monitor_for_modulation** argument of
    the ModulatoryMechanism's constructor, or the **monitor** `argument <ObjectiveMechanism_Monitor>` of the
    ObjectiveMechanism's constructor (see `ModulatoryMechanism_ObjectiveMechanism` for details).  The
    **objective_mechanism** argument can be specified in any of the following ways:

    - *False or None* -- no ObjectiveMechanism is created and, when the ModulatoryMechanism is added to a
      `Composition`, Projections from the OutputStates specified in the ModulatoryMechanism's **monitor_for_modulation**
      argument are sent directly to ModulatoryMechanism (see specification of **monitor_for_modulation** `argument
      <ModulatoryMechanism_Monitor_for_Modulation_Argument>`).

    - *True* -- an `ObjectiveMechanism` is created that projects to the ModulatoryMechanism, and any OutputStates
      specified in the ModulatoryMechanism's **monitor_for_modulation** argument are assigned to ObjectiveMechanism's
      **monitor** `argument <ObjectiveMechanism_Monitor>` instead (see `ModulatoryMechanism_ObjectiveMechanism` for
      additional details).

    - *a list of* `OutputState specifications <ObjectiveMechanism_Monitor>`; an ObjectiveMechanism is created that
      projects to the ModulatoryMechanism, and the list of OutputStates specified, together with any specified in the
      ModulatoryMechanism's **monitor_for_modulation** `argument <ModulatoryMechanism_Monitor_for_Modulation_Argument>`,
      are assigned to the ObjectiveMechanism's **monitor** `argument <ObjectiveMechanism_Monitor>` (see
      `ModulatoryMechanism_ObjectiveMechanism` for additional details).

    - *a constructor for an* `ObjectiveMechanism` -- the specified ObjectiveMechanism is created, adding any
      OutputStates specified in the ModulatoryMechanism's **monitor_for_modulation** `argument
      <ModulatoryMechanism_Monitor_for_Modulation_Argument>` to any specified in the ObjectiveMechanism's **monitor**
      `argument <ObjectiveMechanism_Monitor>` .  This can be used to specify the `function
      <ObjectiveMechanism.function>` used by the ObjectiveMechanism to evaluate the OutputStates monitored as well as
      how it weights those OutputStates when they are evaluated  (see `below
      <ModulatoryMechanism_ObjectiveMechanism_Function>` for additional details).

    - *an existing* `ObjectiveMechanism` -- for any OutputStates specified in the ModulatoryMechanism's
      **monitor_for_modulation** `argument <ModulatoryMechanism_Monitor_for_Modulation_Argument>`, an InputState is
      added to the ObjectiveMechanism, along with `MappingProjection` to it from the specified OutputState.    This
      can be used to specify an ObjectiveMechanism with a custom `function <ObjectiveMechanism.function>` and
      weighting of the OutputStates monitored (see `below <ModulatoryMechanism_ObjectiveMechanism_Function>` for
      additional details).

The OutputStates monitored by a ModulatoryMechanism or its `objective_mechanism
<ModulatoryMechanism.objective_mechanism>` are listed in the ModulatoryMechanism's `monitor_for_modulation
<ModulatoryMechanism.monitor_for_modulation>` attribute (and are the same as those listed in the `monitor
<ObjectiveMechanism.monitor>` attribute of the `objective_mechanism <ModulatoryMechanism.objective_mechanism>`,
if specified).

.. _ModulatoryMechanism_Add_Linear_Processing_Pathway:

Note that the MappingProjections created by specification of a ModulatoryMechanism's **monitor_for_modulation**
`argument <ModulatoryMechanism_Monitor_for_Modulation_Argument>` or the **monitor** argument in the constructor for an
ObjectiveMechanism in the ModulatoryMechanism's **objective_mechanism** `argument
<ModulatoryMechanism_Objective_Mechanism_Argument>` supercede any MappingProjections that would otherwise be created for
them when included in the **pathway** argument of a Composition's `add_linear_processing_pathway
<Composition.add_linear_processing_pathway>` method.


.. _ModulatoryMechanism_ObjectiveMechanism:


ObjectiveMechanism
^^^^^^^^^^^^^^^^^^

If an `ObjectiveMechanism` is specified for a ModulatoryMechanism (in the **objective_mechanism** `argument
<ModulatoryMechanism_Objective_Mechanism_Argument>` of its constructor; also see `ModulatoryMechanism_Examples`),
it is assigned to the ModulatoryMechanism's `objective_mechanism <ModulatoryMechanism.objective_mechanism>` attribute,
and a `MappingProjection` is created automatically that projects from the ObjectiveMechanism's *OUTCOME*
`output_state <ObjectiveMechanism_Output>` to the *OUTCOME* `input_state <ModulatoryMechanism_Input>` of the
ModulatoryMechanism.

The `objective_mechanism <ModulatoryMechanism.objective_mechanism>` is used to monitor the OutputStates specified in
the **monitor_for_modulation** `argument <ModulatoryMechanism_Monitor_for_Modulation_Argument>` of the
ModulatoryMechanism's constructor, as well as any specified in the **monitor** `argument <ObjectiveMechanism_Monitor>`
of the ObjectiveMechanism's constructor.  Specifically, for each OutputState specified in either place, an `input_state
<ObjectiveMechanism.input_states>` is added to the ObjectiveMechanism.  OutputStates to be monitored (and
corresponding `input_states <ObjectiveMechanism.input_states>`) can be added to the `objective_mechanism
<ModulatoryMechanism.objective_mechanism>` later, by using its `add_to_monitor <ObjectiveMechanism.add_to_monitor>`
method. The set of OutputStates monitored by the `objective_mechanism <ModulatoryMechanism.objective_mechanism>` are
listed in its `monitor <ObjectiveMechanism>` attribute, as well as in the ModulatoryMechanism's `monitor_for_modulation
<ModulatoryMechanism.monitor_for_modulation>` attribute.

When the ModulatoryMechanism is added to a `Composition`, the `objective_mechanism
<ModulatoryMechanism.objective_mechanism>` is also automatically added, and MappingProjectons are created from each
of the OutputStates that it monitors to its corresponding `input_states <ObjectiveMechanism.input_states>`.  When the
Composition is run, the `value <OutputState.value>`\\(s) of the OutputState(s) monitored are evaluated using the
`objective_mechanism`\\'s `function <ObjectiveMechanism.function>`, and the result is assigned to its *OUTCOME*
`output_state <ObjectiveMechanism_Output>`.  That `value <ObjectiveMechanism.value>` is then passed to the
ModulatoryMechanism's *OUTCOME* `input_state <ModulatoryMechanism_Input>`, which is used by the ModulatoryMechanism's
`function <ModulatoryMechanism.function>` to determine its `modulatory_allocation
<ModulatoryMechanism.modulatory_allocation>`.

.. _ModulatoryMechanism_ObjectiveMechanism_Function:

If a default ObjectiveMechanism is created by the ModulatoryMechanism (i.e., when *True* or a list of OutputStates is
specified for the **objective_mechanism** `argument <ModulatoryMechanism_Objective_Mechanism_Argument>` of the
constructor), then the ObjectiveMechanism is created with its standard default `function <ObjectiveMechanism.function>`
(`LinearCombination`), but using *PRODUCT* (rather than the default, *SUM*) as the value of the function's `operation
<LinearCombination.operation>` parameter. The result is that the `objective_mechanism
<ModulatoryMechanism.objective_mechanism>` multiplies the `value <OutputState.value>`\\s of the OutputStates that it
monitors, which it passes to the ModulatoryMechanism.  However, if the **objective_mechanism** is specified using either
a constructor for, or an existing ObjectiveMechanism, then the defaults for the `ObjectiveMechanism` class -- and any
attributes explicitly specified in its construction -- are used.  In that case, if the `LinearCombination` with
*PRODUCT* as its `operation <LinearCombination.operation>` parameter are still desired, this must be explicitly
specified.  This is illustrated in the following examples.

The following example specifies a `ModulatoryMechanism` that automatically constructs its `objective_mechanism
<ModulatoryMechanism.objective_mechanism>`::

    >>> from psyneulink import *
    >>> my_ctl_mech = ModulatoryMechanism(objective_mechanism=True)
    >>> assert isinstance(my_ctl_mech.objective_mechanism.function, LinearCombination)
    >>> assert my_ctl_mech.objective_mechanism.function.operation == PRODUCT

Notice that `LinearCombination` was assigned as the `function <ObjectiveMechanism.function>` of the `objective_mechanism
<ModulatoryMechanism.objective_mechanism>`, and *PRODUCT* as its `operation <LinearCombination.operation>` parameter.

By contrast, the following example explicitly specifies the **objective_mechanism** argument using a constructor for
an ObjectiveMechanism::

    >>> my_ctl_mech = ModulatoryMechanism(objective_mechanism=ObjectiveMechanism())
    >>> assert isinstance(my_ctl_mech.objective_mechanism.function, LinearCombination)
    >>> assert my_ctl_mech.objective_mechanism.function.operation == SUM

In this case, the defaults for the ObjectiveMechanism's class are used for its `function <ObjectiveMechanism.function>`,
which is a `LinearCombination` function with *SUM* as its `operation <LinearCombination.operation>` parameter.

Specifying the ModulatoryMechanism's `objective_mechanism <ModulatoryMechanism.objective_mechanism>` with a constructor
also provides greater control over how ObjectiveMechanism evaluates the OutputStates it monitors.  In addition to
specifying its `function <ObjectiveMechanism.function>`, the **monitor_weights_and_exponents** `argument
<ObjectiveMechanism_Monitor_Weights_and_Exponents>` can be used to parameterize the relative contribution made by the
monitored OutputStates when they are evaluated by that `function <ObjectiveMechanism.function>` (see
`ModulatoryMechanism_Examples`).


.. _ModulatoryMechanism_Modulatory_Signals:

*Specifying States to Modulate*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ModulatoryMechanism modulates the values of `States <State>` using the `ControlSignals <ControlSignal>` and/or
`GatingSignals <GatingSignal>` assigned to its `modulatory_signals <ModulatoryMechanism.modulatory_signals>` attribute,
each of which is assigned, respectively, a `ControlProjection` or `GatingProjection` to the State it modulates (see
`Modulation <ModulatorySignal_Modulation>` for a description of how modulation operates). Modulation of a State can
be specified either where the Component to which the State belongs is created (see specifying
`ControlMechanism_Control_Signals` and `ControlSignal_Specification`, or `GatingMechanism_Specifying_Gating` and
`GatingSignal_Specification` respectively), or in the **modulatory_signals** argument of the ModulatoryMechanism's
constructor.  For the latter, the argument must be a specification for one or more `ControlSignals
<ControlSignal_Specification>` and/or `GatingSignals <GatingSignal_Specification>`.  States to be modulated can also
be added to an existing ModulatoryMechanism by using its `assign_params` method to add a `ControlSignal` and/or
`GatingSignal` for each additional State. All of a ModulatoryMechanism's `ModulatorySignals <ModulatorySignal>` are
listed in its `modulatory_signals <ModulatoryMechanism>` attribute.  Any ControlSignals are also listed in the
`control_signals <ModulatoryMechanism.control_signals>` attribute, and GatingSignals in the `gating_signals
<GatingMechanism.gating_signals>` attribute.  The projections from these to the States they modulate are listed in
the `modulatory_projections <ModulatoryMechanism.modulatory_projections>`, `control_projections
<ModulatoryMechanism.control_projections>`, and `gating_projections <ModulatoryMechanism.gating_projections>`
attributes, respectively.


.. _ModulatoryMechanism_Structure:

Structure
---------

.. _ModulatoryMechanism_Input:

*Input*
~~~~~~~

By default, a ModulatoryMechanism has a single (`primary <InputState_Primary>`) `input_state
<ModulatoryMechanism.input_state>` that is named *OUTCOME*.  If the ModulatoryMechanism has an `objective_mechanism
<ModulatoryMechanism.objective_mechanism>`, then the *OUTCOME* `input_state <ModulatoryMechanism.input_state>` receives
a single `MappingProjection` from the `objective_mechanism <ModulatoryMechanism.objective_mechanism>`\\'s *OUTCOME*
OutputState (see `ModulatoryMechanism_ObjectiveMechanism` for additional details). Otherwise, when the
ModulatoryMechanism is added to a `Composition`, MappingProjections are created that project to the
ModulatoryMechanism's *OUTCOME* `input_state <ModulatoryMechanism.input_state>` from each of the OutputStates specified
in the **monitor_for_modulation** `argument <ModulatoryMechanism_Monitor_for_Modulatory_Argument>` of its constructor.
The `value <InputState.value>` of the ModulatoryMechanism's *OUTCOME* InputState is assigned to its `outcome
<ModulatoryMechanism.outcome>` attribute), and is used as the input to the ModulatoryMechanism's `function
<ModulatoryMechanism.function>` to determine its `modulatory_allocation <ModulatoryMechanism.modulatory_allocation>`.

.. _ModulatoryMechanism_Function:

*Function*
~~~~~~~~~~

A ModulatoryMechanism's `function <ModulatoryMechanism.function>` uses its `outcome <ModulatoryMechanism.outcome>`
attribute (the `value <InputState.value>` of its *OUTCOME* `InputState`) to generate a `modulatory_allocation
<ModulatoryMechanism.modulatory_allocation>`.  By default, its `function <ModulatoryMechanism.function>` is assigned
the `DefaultAllocationFunction`, which takes a single value as its input, and assigns this as the value of
each item of `modulatory_allocation <ModulatoryMechanism.modulatory_allocation>`.  Each of these items is assigned as
the allocation for the corresponding `ControlSignal` or `GatingSignal` in `modulatory_signals
<ModulatoryMechanism.modulatory_signals>`. This distributes the ModulatoryMechanism's input as the allocation to each of
`modulatory_signals  <ModulatoryMechanism.modulatory_signals>.  This same behavior also applies to any custom function
assigned to a ModulatoryMechanism that returns a 2d array with a single item in its outer dimension (axis 0).  If a
function is assigned that returns a 2d array with more than one item, and it has the same number of `modulatory_signals
<ModulatoryMechanism.modulatory_signals>`, then each ModulatorySignal is assigned to the corresponding item of the
function's value.  However, these default behaviors can be modified by specifying that individual ModulatorySignals
reference different items in `modulatory_allocation` as their `variable <ModulatorySignal.variable>`
(see `OutputState_Variable`).

.. _ModulatoryMechanism_Output:

*Output*
~~~~~~~~


The OutputStates of a ModulatoryMechanism are `ControlSignals <ControlSignal>` and/or `GatingSignals <GatingSignal>`
(listed in its `modulatory_signals <ModulatoryMechanism.modulatory_signals>` attribute). It has a `ControlSignal` or
`GatingSignal` for each parameter specified in the **modulatory_signals** argument of its constructor, that sends a
`ControlProjection <ControlProjections>` or `GatingProjection <GatingProjection>` to the corresponding State.
The `States <State>` modulated by a ModulatoryMechanism's `modulatory_signals <ModulatoryMechanism.modulatory_signals>`
can be displayed using its `show <ModulatoryMechanism.show>` method. By default, each item of the ModulatoryMechanism's
`modulatory_allocation <ModulatoryMechanism.modulatory_allocation>` attribute is assigned to the `variable` of the
corresponding `ControlSignal` or `GatingSignal` in its `modulatory_signals <ModulatoryMechanism.modulatory_signals>`
attribute;  however, subclasses of ModulatoryMechanism may assign values differently;  the `default_allocation
<ModulatoryMechanism.default_allocation>` attribute can be used to specify a default allocation for ModulatorySignals
that have not been assigned their own `default_allocation <ModulatorySignal.default_allocation>`. The  current
allocations to ControlSignals are also listed in the `control_allocation <ModulatoryMechanism.control_allocation>`
attribute; and the  allocations to GatingSignals are listed in the  `gating_allocation
<ModulatoryMechanism.gating_allocation>` attribute.

.. _ModulatoryMechanism_Costs:

*Costs and Net Outcome*
~~~~~~~~~~~~~~~~~~~~~~~

If a ModulatoryMechanism has any `control_signals <ModulatoryMechanism.control_signals>`, then it also computes
the combined `costs <ModulatoryMechanism.costs>` of those, and a `net_outcome <ModulatoryMechanism.net_outcome>`
based on them (see `below <ModulatoryMechanism_Costs_Computation>`). This is used by some subclasses of
ModulatoryMechanism (e.g., `OptimizationControlMechanism`) to compute the `modulatory_allocation
<ModulatoryMechanism.modulatory_allocation>`. These are computed using the ModulatoryMechanism's default
`compute_reconfiguration_cost <ModulatoryMechanism.compute_reconfiguration_cost>`, `combine_costs
<ModulatoryMechanism.combine_costs>`, and `compute_net_outcome <ModulatoryMechanism.compute_net_outcome>` functions,
but these can also be assigned custom functions (see links to attributes for details).

.. _ModulatoryMechanism_Reconfiguration_Cost:

*Reconfiguration Cost*

A ModulatoryMechanism's ``reconfiguration_cost  <ModulatoryMechanism.reconfiguration_cost>` is distinct from the
costs of the ModulatoryMechanism's ControlSignals (if it has any), and in particular it is not the same as their
`adjustment_cost <ControlSignal.adjustment_cost>`.  The latter, if specified by a ControlSignal, is computed
individually by that ControlSignal using its `adjustment_cost_function <ControlSignal.adjustment_cost_function>` based
on the change in its `intensity <ControlSignal.intensity>` from its last execution. In contrast, a ModulatoryMechanism's
`reconfiguration_cost  <ModulatoryMechanism.reconfiguration_cost>` is computed by its `compute_reconfiguration_cost
<ModulatoryMechanism.compute_reconfiguration_cost>` function, based on the change in its `modulatory_allocation
ModulatoryMechanism.modulatory_allocation>` from the last execution, that will be applied to *all* of its
`modulatory_signals <ModulatoryMechanism.modulatory_signals>` (including any `gating_signals
<ModulatoryMechanism.gating_signals>` it has). By default, it uses the `Distance` function with the `EUCLIDEAN` metric).

.. _ModulatoryMechanism_Execution:

Execution
---------

The ModulatoryMechanism's `function <ModulatoryMechanism.function>` takes as its input the `value <InputState.value>`
of its *OUTCOME* `input_state <Mechanism_Base.input_state>` (also contained in `outcome <ControlSignal.outcome>`). It
uses that to determine its `modulatory_allocation <ModulatoryMechanism.modulatory_allocation>` that specifies the
value assigned to its `modulatory_signals <ModulatoryMechanism.modulatory_signals>`.  Each of those uses the
allocation it is assigned to calculate its `ControlSignal` `intensity <ControlSignal.intensity>` or `GatingSignal`
`intensity <GatingSignal.intensity>`. These are used by their `ControlProjection(s) <ControlProjection>` or
`GatingProjection(s) <GatingProjection>`, respectively, to modulate the value of the States to which they project.
Those values are then used in the subsequent `TRIAL` of execution. If a ModulatoryMechanism is a Composition's
`controller <Composition.controller>`, it is generally either the first or the last `Mechanism <Mechanism>` to be
executed in a `TRIAL`, although this can be customized (see `Composition Controller <Composition_Controller_Execution>`).

.. note::
   `States <State>` that receive `ControlProjections <ControlProjection>` or `GatingProjections <GatingProjection>`
   do not update their values until their owner Mechanisms execute (see `Lazy Evaluation <LINK>` for an explanation of
   "lazy" updating).  This means that even if a ModulatoryMechanism has executed, the States that it modulates will
   not assume their new values until the Mechanisms to which they belong have executed.

.. _ModulatoryMechanism_Costs_Computation:

*Computation of Costs and Net_Outcome*

Once the ModulatoryMechanism's `function <ModulatoryMechanism.function>` has executed,
if `compute_reconfiguration_cost <ModulatoryMechanism.compute_reconfiguration_cost>` has been specified, then it is
used to compute the `reconfiguration_cost <ModulatoryMechanism.reconfiguration_cost>` for its `modulatory_allocation
<ModulatoryMechanism.modulatory_allocation>` (see `above <ModulatoryMechanism_Reconfiguration_Cost>`.  Then, if the
ModulatoryMechanism has any `control_signals <ModulatoryMechanism.contro._signals>`, they calculate their costs
which are combined with the ModulatoryMechanism's `reconfiguration_cost <ModulatoryMechanism.reconfiguration_cost>`
using its `combine_costs <ModulatoryMechanism.combine_costs>` function, and the result is assigned to its `costs
<ModulatoryMechanism.costs>` attribute.  The ModulatoryMechanism uses this, together with its `outcome
<ModulatoryMechanism.outcome>` attribute, to compute a  `net_outcome <ModulatoryMechanism.net_outcome>` using its
`compute_net_outcome <ModulatoryMechanism.compute_net_outcome>` function.  This is used by some subclasses of
ModulatoryMechanism (e.g., `OptimizationControlMechanism`) to  compute its `modulatory_allocation
<ModulatoryMechanism.modulatory_allocation>`.

.. _ModulatoryMechanism_Examples:

Examples
--------

COMMENT:
FIX: ADD EXAMPLES WITH GATING SIGNALS
COMMENT

The following example creates a ModulatoryMechanism by specifying its **objective_mechanism** using a constructor
that specifies the OutputStates to be monitored by its `objective_mechanism <ModulatoryMechanism.objective_mechanism>`
and the function used to evaluated these::

    >>> import psyneulink as pnl
    >>> my_transfer_mech_A = pnl.TransferMechanism(name="Transfer Mech A")
    >>> my_DDM = pnl.DDM(name="My DDM")
    >>> my_transfer_mech_B = pnl.TransferMechanism(function=pnl.Logistic,
    ...                                            name="Transfer Mech B")

    >>> my_modulatory_mech = pnl.ModulatoryMechanism(
    ...                          objective_mechanism=pnl.ObjectiveMechanism(monitor=[(my_transfer_mech_A, 2, 1),
    ...                                                                                               my_DDM.output_states[pnl.RESPONSE_TIME]],
    ...                                                                     name="Objective Mechanism"),
    ...                          function=pnl.LinearCombination(operation=pnl.PRODUCT),
    ...                          modulatory_signals=[(pnl.THRESHOLD, my_DDM),
    ...                                              (pnl.GAIN, my_transfer_mech_B)],
    ...                          name="My Modulatory Mech")


This creates an ObjectiveMechanism for the ModulatoryMechanism that monitors the `primary OutputState
<OutputState_Primary>` of ``my_Transfer_mech_A`` and the *RESPONSE_TIME* OutputState of ``my_DDM``;  its function
first multiplies the former by 2 before, then takes product of their values and passes the result as the input to the
ModulatoryMechanism.  The ModulatoryMechanism's `function <ModulatoryMechanism.function>` uses this value to determine
the allocation for its ControlSignals, that control the value of the `threshold <DDM.threshold>` parameter of
``my_DDM`` and the  `gain <Logistic.gain>` parameter of the `Logistic` Function for ``my_transfer_mech_B``.

The following example specifies the same set of OutputStates for the ObjectiveMechanism, by assigning them directly
to the **objective_mechanism** argument::

    >>> my_modulatory_mech = pnl.ModulatoryMechanism(
    ...                             objective_mechanism=[(my_transfer_mech_A, 2, 1),
    ...                                                  my_DDM.output_states[pnl.RESPONSE_TIME]],
    ...                             modulatory_signals=[(pnl.THRESHOLD, my_DDM),
    ...                                                 (pnl.GAIN, my_transfer_mech_B)])
    ...

Note that, while this form is more succinct, it precludes specifying the ObjectiveMechanism's function.  Therefore,
the values of the monitored OutputStates will be added (the default) rather than multiplied.

The ObjectiveMechanism can also be created on its own, and then referenced in the constructor for the ModulatoryMechanism::

    >>> my_obj_mech = pnl.ObjectiveMechanism(monitor=[(my_transfer_mech_A, 2, 1),
    ...                                                               my_DDM.output_states[pnl.RESPONSE_TIME]],
    ...                                      function=pnl.LinearCombination(operation=pnl.PRODUCT))

    >>> my_modulatory_mech = pnl.ModulatoryMechanism(
    ...                        objective_mechanism=my_obj_mech,
    ...                        modulatory_signals=[(pnl.THRESHOLD, my_DDM),
    ...                                            (pnl.GAIN, my_transfer_mech_B)])

Here, as in the first example, the constructor for the ObjectiveMechanism can be used to specify its function, as well
as the OutputState that it monitors.

COMMENT:
TBI FOR COMPOSITION
See `System_Control_Examples` for examples of how a ModulatoryMechanism, the OutputStates its
`objective_mechanism <ControlSignal.objective_mechanism>`, and its `control_signals <ModulatoryMechanism.control_signals>`
can be specified for a System.
COMMENT

.. _ModulatoryMechanism_Class_Reference:

Class Reference
---------------

"""

import copy
import itertools
import numpy as np
import threading
import typecheck as tc
import warnings

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.functions.function import Function_Base, is_function_type
from psyneulink.core.components.functions.combinationfunctions import LinearCombination
from psyneulink.core.components.mechanisms.adaptive.adaptivemechanism import AdaptiveMechanism_Base
from psyneulink.core.components.mechanisms.mechanism import Mechanism, Mechanism_Base
from psyneulink.core.components.shellclasses import Composition_Base, System_Base
from psyneulink.core.components.states.state import State
from psyneulink.core.components.states.modulatorysignals.modulatorysignal import ModulatorySignal
from psyneulink.core.components.states.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.components.states.modulatorysignals.gatingsignal import GatingSignal
from psyneulink.core.components.states.inputstate import InputState
from psyneulink.core.components.states.outputstate import OutputState
from psyneulink.core.components.states.parameterstate import ParameterState
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.defaults import defaultControlAllocation, defaultGatingAllocation
from psyneulink.core.globals.keywords import \
    AUTO_ASSIGN_MATRIX, CONTEXT, CONTROL, CONTROL_PROJECTIONS, CONTROL_SIGNAL, CONTROL_SIGNALS, \
    EID_SIMULATION, GATING_SIGNAL, GATING_SIGNALS, INIT_EXECUTE_METHOD_ONLY, \
    MODULATORY_PROJECTION, MODULATORY_SIGNAL, MODULATORY_SIGNALS, MONITOR_FOR_MODULATION, MULTIPLICATIVE, \
    OBJECTIVE_MECHANISM, OUTCOME, OWNER_VALUE, PRODUCT, PROJECTIONS, SYSTEM
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import ContentAddressableList, is_iterable, convert_to_list

__all__ = [
    'CONTROL_ALLOCATION', 'GATING_ALLOCATION', 'MODULATORY_ALLOCATION',
    'ModulatoryMechanism', 'ModulatoryMechanismError', 'ModulatoryMechanismRegistry'
]

MODULATORY_ALLOCATION = 'modulatory_allocation'
CONTROL_ALLOCATION = 'control_allocation'
GATING_ALLOCATION = 'gating_allocation'

ModulatoryMechanismRegistry = {}

class ModulatoryMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

def _control_allocation_getter(owning_component=None, context=None):
    try:
        control_signal_indices = [owning_component.modulatory_signals.index(c)
                                  for c in owning_component.control_signals]
        return np.array([owning_component.modulatory_allocation[i] for i in control_signal_indices])
    except TypeError:
        return owning_component.parameters.default_allocation or \
               [owning_component.parameters.control_allocation.default_value]

def _control_allocation_setter(value, owning_component=None, context=None):
    control_signal_indices = [owning_component.modulatory_signals.index(c)
                              for c in owning_component.control_signals]
    if len(value)!=len(control_signal_indices):
        raise ModulatoryMechanismError(f"Attempt to set '{CONTROL_ALLOCATION}' parameter of {owning_component.name} "
                                       f"with value ({value} that has a different length ({len(value)}) "
                                       f"than the number of its {CONTROL_SIGNALS} ({len(control_signal_indices)})")
    mod_alloc = owning_component.parameters.modulatory_allocation._get(context)
    for j, i in enumerate(control_signal_indices):
        mod_alloc[i] = value[j]
    owning_component.parameters.modulatory_allocation._set(np.array(mod_alloc), context)
    return value

def _gating_allocation_getter(owning_component=None, context=None):
    try:
        gating_signal_indices = [owning_component.modulatory_signals.index(g)
                                  for g in owning_component.gating_signals]
        return np.array([owning_component.modulatory_allocation[i] for i in gating_signal_indices])
    except TypeError:
        return owning_component.parameters.default_allocation or \
               [owning_component.parameters.gating_allocation.default_value]

def _gating_allocation_setter(value, owning_component=None, context=None):
    gating_signal_indices = [owning_component.modulatory_signals.index(c)
                              for c in owning_component.gating_signals]
    if len(value)!=len(gating_signal_indices):
        raise ModulatoryMechanismError(f"Attempt to set {GATING_ALLOCATION} parameter of {owning_component.name} "
                                       f"with value ({value} that has a different length than the number of its"
                                       f"{GATING_SIGNALS} ({len(gating_signal_indices)})")
    mod_alloc = owning_component.parameters.modulatory_allocation._get(context)
    for j, i in enumerate(gating_signal_indices):
        mod_alloc[i] = value[j]
    owning_component.parameters.modulatory_allocation._set(np.array(mod_alloc), context)
    return value

def _modulatory_mechanism_costs_getter(owning_component=None, context=None):
    # NOTE: In cases where there is a reconfiguration_cost, that cost is not returned by this method
    try:
        # # MODIFIED 8/30/19 OLD:
        # costs = [c.compute_costs(c.parameters.variable._get(context), context=context)
        #          for c in owning_component.control_signals]
        # MODIFIED 8/30/19 NEW: [JDC]
        # FIX 8/30/19: SHOULDN'T THIS JUST GET ControlSignal.cost FOR EACH ONE?
        costs = [c.compute_costs(c.parameters.value._get(context), context=context)
                 for c in owning_component.control_signals]
        # MODIFIED 8/30/19 END
        return costs

    except TypeError:
        return None

def _outcome_getter(owning_component=None, context=None):
    try:
        return owning_component.parameters.variable._get(context)[0]
    except TypeError:
        return None

def _net_outcome_getter(owning_component=None, context=None):
    # NOTE: In cases where there is a reconfiguration_cost, that cost is not included in the net_outcome
    try:
        c = owning_component
        return c.compute_net_outcome(c.parameters.outcome._get(context),
                                     c.combine_costs(c.parameters.costs._get(context)))
    except TypeError:
        return [0]

class DefaultAllocationFunction(Function_Base):
    """Take a single 1d item and return a 2d array with n identical items
    Takes the default input (a single value in the *OUTCOME* InputState of the ModulatoryMechanism),
    and returns the same allocation for each of its `modulatory_signals <ModulatoryMechanism.modulatory_signals>`.
    """
    componentName = 'Default Modulatory Function'
    class Parameters(Function_Base.Parameters):
        """
            Attributes
            ----------

                num_modulatory_signals
                    see `num_modulatory_signals <DefaultAllocationFunction.num_modulatory_signals>`

                    :default value: 1
                    :type: int

        """
        num_modulatory_signals = Parameter(1, stateful=False)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 params=None,
                 owner=None
                 ):
        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(params=params)
        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         )

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        num_mod_sigs = self.get_current_function_param('num_modulatory_signals')
        result = np.array([variable[0]] * num_mod_sigs)
        return self.convert_output_type(result)

    def _gen_llvm_function_body(self, ctx, builder, _1, _2, arg_in, arg_out):
        val_ptr = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(0)])
        val = builder.load(val_ptr)
        with pnlvm.helpers.array_ptr_loop(builder, arg_out, "alloc_loop") as (b, idx):
            out_ptr = builder.gep(arg_out, [ctx.int32_ty(0), idx])
            builder.store(val, out_ptr)
        return builder


class ModulatoryMechanism(AdaptiveMechanism_Base):
    """
    ModulatoryMechanism(                                         \
        system=None                                              \
        monitor_for_modulation=None,                             \
        objective_mechanism=None,                                \
        function=DefaultAllocationFunction,                      \
        default_allocation=None,                                 \
        modulatory_signals=None,                                 \
        modulation=MULTIPLICATIVE                                \
        combine_costs=np.sum,                                    \
        compute_reconfiguration_cost=None,                       \
        compute_net_outcome=lambda x,y:x-y,                      \
        params=None,                                             \
        name=None,                                               \
        prefs=None)

    Subclass of `AdaptiveMechanism <AdaptiveMechanism>` that modulates the parameter(s)
    of one or more `Component(s) <Component>`.


    COMMENT:
    .. note::
       ModulatoryMechanism is an abstract class and should NEVER be instantiated by a direct call to its constructor.
       It should be instantiated using the constructor for a `subclass <ModulatoryMechanism_Subtypes>`.

        Description:
            Protocol for instantiating unassigned ControlProjections (i.e., w/o a sender specified):
               If sender is not specified for a ControlProjection (e.g., in a parameter specification tuple)
                   it is flagged for deferred_init() in its __init__ method
               If ModulatoryMechanism is instantiated or assigned as the controller for a System:
                   the System calls its _get_monitored_output_states() method which returns all of the OutputStates
                       within the System that have been specified to be MONITORED_FOR_CONTROL, and then assigns
                       them (along with any specified in the **monitored_for_control** arg of the System's constructor)
                       to the `objective_mechanism` argument of the ModulatoryMechanism's constructor;
                   the System calls its _get_control_signals_for_system() method which returns all of the parameters
                       that have been specified for control within the System, assigns them a ControlSignal
                       (with a ControlProjection to the ParameterState for the parameter), and assigns the
                       ControlSignals (alogn with any specified in the **control_signals** argument of the System's
                       constructor) to the **control_signals** argument of the ModulatoryMechanism's constructor

            OBJECTIVE_MECHANISM param determines which States will be monitored.
                specifies the OutputStates of the terminal Mechanisms in the System to be monitored by ModulatoryMechanism
                this specification overrides any in System.params[], but can be overridden by Mechanism.params[]
                ?? if MonitoredOutputStates appears alone, it will be used to determine how States are assigned from
                    System.execution_graph by default
                if MonitoredOutputStatesOption is used, it applies to any Mechanisms specified in the list for which
                    no OutputStates are listed; it is overridden for any Mechanism for which OutputStates are
                    explicitly listed
                TBI: if it appears in a tuple with a Mechanism, or in the Mechamism's params list, it applies to
                    just that Mechanism

        Class attributes:
            + componentType (str): System Default Mechanism
            + paramClassDefaults (dict):
                + FUNCTION: Linear
                + FUNCTION_PARAMS:{SLOPE:1, INTERCEPT:0}
                + OBJECTIVE_MECHANISM: List[]
    COMMENT

    Arguments
    ---------

    system : System or bool : default None
        specifies the `System` to which the ModulatoryMechanism should be assigned as its `controller
        <System.controller>`.

    monitor_for_modulation : List[OutputState or Mechanism] : default None
        specifies the `OutputStates <OutputState>` to be monitored by the `ObjectiveMechanism` specified in the
        **objective_mechanism** argument; for any Mechanisms specified, their `primary OutputState
        <OutputState_Primary>` are used.

    objective_mechanism : ObjectiveMechanism | List[OutputState specification] | bool : default None
        specifies either an `ObjectiveMechanism` to use for the ModulatoryMechanism, the list of OutputStates that
        one constructed automatically should use, or the construction of a default ObjectiveMechanism (for True);
        if a list of `OutputState specifications <ObjectiveMechanism_Monitor>` is specified,
        the list is passed to as the **monitor** argument in the constructor for a default ObjectiveMechanism.

    function : TransferFunction : default Linear(slope=1, intercept=0)
        specifies function used to combine values of monitored OutputStates.

    modulatory_signals : ControlSignal specification or List[ControlSignal specification, ...]
        specifies the parameters to be controlled by the ModulatoryMechanism; a `ControlSignal` is created for each
        (see `ControlSignal_Specification` for details of specification).

    default_allocation : number, list or 1d array : None
        specifies the default_allocation of any `modulatory_signals <ModulatoryMechanism.modulatory.signals>` for
        which the **default_allocation** was not specified in its constructor (see default_allocation
        <ModulatoryMechanism.default_allocation>` for additional details).

    modulation : ModulationParam : MULTIPLICATIVE
        specifies the default form of modulation used by the ModulatoryMechanism's `ControlSignals <ControlSignal>`,
        unless they are `individually specified <ControlSignal_Specification>`.

    combine_costs : Function, function or method : default np.sum
        specifies function used to combine the `cost <ControlSignal.cost>` of the ModulatoryMechanism's `control_signals
        <ModulatoryMechanism.control_signals>`;  must take a list or 1d array of scalar values as its argument and
        return a list or array with a single scalar value.

    compute_reconfiguration_cost : Function, function or method : default None
        specifies function used to compute the ModulatoryMechanism's `reconfiguration_cost
        <ModulatoryMechanism.reconfiguration_cost>`; must take a list or 2d array containing two lists or 1d arrays,
        both with the same shape as the ModulatoryMechanism's control_allocation attribute, and return a scalar value.

    compute_net_outcome : Function, function or method : default lambda outcome, cost: outcome-cost
        function used to combine the values of its `outcome <ModulatoryMechanism.outcome>` and `costs
        <ModulatoryMechanism.costs>` attributes;  must take two 1d arrays (outcome and cost) with scalar values as its
        arguments and return an array with a single scalar value.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters
        for the Mechanism, parameters for its function, and/or a custom function and its parameters. Values
        specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default see `name <ModulatoryMechanism.name>`
        specifies the name of the ModulatoryMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the ModulatoryMechanism; see `prefs <ModulatoryMechanism.prefs>` for details.

    Attributes
    ----------

    COMMENT:
    system : System_Base
        The `System` for which the ModulatoryMechanism is a `controller <System>`.  Note that this is distinct from
        a Mechanism's `systems <Mechanism_Base.systems>` attribute, which lists all of the Systems to which a
        `Mechanism` belongs -- a ModulatoryMechanism can belong to but not be the `controller of a Composition
        <ModulatoryMechanism_Composition_Controller>`.
    COMMENT

    composition : Composition
        The `Composition` for which the ModulatoryMechanism is a `controller <Composition.controller>`
        (see `<`ModulatoryMechanism_Composition_Controller` for additional details).  Note that a ModulatoryMechanism
        can belong to but not be the `controller <Composition.controller>` of a Composition;  if the ModulatoryMechanism
        is *not* a `controller <Composition.controller>` of a Composition, this attribute does not exist.

    objective_mechanism : ObjectiveMechanism
        `ObjectiveMechanism` that monitors and evaluates the values specified in the ModulatoryMechanism's
        **objective_mechanism** argument, and transmits the result to the ModulatoryMechanism's *OUTCOME*
        `input_state <Mechanism_Base.input_state>`.

    monitor_for_modulation : List[OutputState]
        each item is an `OutputState` monitored by the ObjectiveMechanism listed in the ModulatoryMechanism's
        `objective_mechanism <ModulatoryMechanism.objective_mechanism>` attribute;  it is the same as that
        ObjectiveMechanism's `monitor <ObjectiveMechanism.monitor>` attribute
        (see `ObjectiveMechanism_Monitor` for specification).  The `value <OutputState.value>`
        of the OutputStates in the list are used by the ObjectiveMechanism to generate the ModulatoryMechanism's `input
        <ModulatoryMechanism_Input>`.

    monitored_output_states_weights_and_exponents : List[Tuple(float, float)]
        each tuple in the list contains the weight and exponent associated with a corresponding OutputState specified
        in `monitor_for_modulation <ModulatoryMechanism.monitor_for_modulation>`;  these are the same as those in the
        `monitored_output_states_weights_and_exponents
        <ObjectiveMechanism.monitored_output_states_weights_and_exponents>` attribute of the `objective_mechanism
        <ModulatoryMechanism.objective_mechanism>`, and are used by the ObjectiveMechanism's `function
        <ObjectiveMechanism.function>` to parametrize the contribution made to its output by each of the values that
        it monitors (see `ObjectiveMechanism Function <ObjectiveMechanism_Function>`).

    outcome : 1d array
        the `value <InputState.value>` of the ModulatoryMechanism's `primary InputState <InputState_Primary>`;
        this receives its `Projection <Projection>` from the *OUTCOME* `OutputState` of its `objective_mechanism
        <ModulatoryMechanism.objective_mechanism>` if that is specified

    function : TransferFunction : default Linear(slope=1, intercept=0)
        determines how the `value <OuputState.value>` \\s of the `OutputStates <OutputState>` specified in the
        **monitor_for_modulation** argument of the ModulatoryMechanism's constructor are used to generate its
        `modulatory_allocation <ModulatoryMechanism.modulatory_allocation>`.

    default_allocation : number, list or 1d array
        determines the default_allocation of any `modulatory_signals <ModulatoryMechanism.modulatory.signals>` for
        which the **default_allocation** was not specified in its constructor;  if it is None (not specified)
        then the ModulatorySignal's parameters.allocation.default_value, specified by its class, is used.
        See documentation for **default_allocation** argument of ModulatorySignal constructor for additional details.

    modulatory_signals : ContentAddressableList[ModulatorySignal]
        list of the ModulatoryMechanism's `ControlSignals <ControlSignals>` and `GatingSignals <GatingSignals>`.
        COMMENT:
        TBI FOR COMPOSITION
        including any inherited from a `system <ModulatoryMechanism.system>` for which it is a `controller
        <System.controller>`.
        COMMENT
        This is the same as the ModulatoryMechanism's `output_states <Mechanism_Base.output_states>` attribute).

    modulatory_allocation : 2d array
        contains allocations for all the ModulatoryMechanism's `modulatory_signals
        <ModulatoryMechanism.modulatory_signals>`;  each item is the value assigned as the `allocation` for a
        `ControlSignal` (listed in the `control_signals  <ModulatoryMechanism.control_signals>` attribute,
        or a `GatingSignal` (listed in the `gating_signals <ModulatoryMechanism.gating_signals>` attribute (these
        are also listed in the `control_allocation <ModulatoryMechanism.control_allocation>` and `gating_allocation
        <ModulatoryMechanism.gating_allocation>` attributes, respectively.  The modulatory_allocation is the same as
        the ModulatoryMechanism's `value <Mechanism_Base.value>` attribute).

    control_signals : ContentAddressableList[ControlSignal]
        list of the `ControlSignals <ControlSignals>` for the ModulatoryMechanism.
        COMMENT:
        TBI FOR COMPOSITION
        , including any inherited from a
        `system <ModulatoryMechanism.system>` for which it is a `controller <System.controller>`.
        COMMENT

    control_allocation : 2d array
        each item is the value assigned as the `allocation <ControlSignal.allocation>` for the corresponding
        ControlSignal listed in the `control_signals <ModulatoryMechanism.control_signals>` attribute.

    gating_signals : ContentAddressableList[GatingSignal]
        list of the `GatingSignals <ControlSignals>` for the ModulatoryMechanism.
        COMMENT:
        TBI FOR COMPOSITION
        , including any inherited from a
        `system <ModulatoryMechanism.system>` for which it is a `controller <System.controller>`.
        COMMENT

    gating_allocation : 2d array
        each item is the value assigned as the `allocation <GatingSignal.allocation>` for the corresponding
        GatingSignal listed in the `gating_signals` attribute <ModulatoryMechanism.gating_signals>`.

    reconfiguration_cost : scalar
        result of `compute_reconfiguration_cost <ModulatoryMechanism.compute_reconfiguration_cost>` function, that
        computes the difference between the values of the ModulatoryMechanism's current and last `modulatory_alloction
        <ModulatoryMechanism.modulatory_allocation>`; its value is None and is ignored if `compute_reconfiguration_cost
        <ModulatoryMechanism.compute_reconfiguration_cost>` has not been specified.

        .. note::
        A ModulatoryMechanism's reconfiguration_cost is not the same as the `adjustment_cost
        <ControlSignal.adjustment_cost>` of its ControlSignals (see `ModulatoryMechanism Reconfiguration Cost
        <ModulatoryMechanism_Reconfiguration_Cost>` for additional detals).

    compute_reconfiguration_cost : Function, function or method
        function used to compute the ModulatoryMechanism's `reconfiguration_cost
        <ModulatoryMechanism.reconfiguration_cost>`.

    costs : list
        current costs for the ModulatoryMechanism's `control_signals <ModulatoryMechanism.control_signals>`, computed
        for each using its `compute_costs <ControlSignals.compute_costs>` method.

    combine_costs : Function, function or method
        function used to combine the `cost <ControlSignal.cost>` of its `control_signals
        <ModulatoryMechanism.control_signals>`; result is an array with a scalar value that can be accessed by
        `combined_costs <ModulatoryMechanism.combined_costs>`.

        .. note::
          This function is distinct from the `combine_costs_function <ControlSignal.combine_costs_function>` of a
          `ControlSignal`.  The latter combines the different `costs <ControlSignal_Costs>` for an individual
          ControlSignal to yield its overall `cost <ControlSignal.cost>`; the ModulatoryMechanism's
          `combine_costs <ModulatoryMechanism.combine_costs>` function combines those `cost <ControlSignal.cost>`\\s
          for its `control_signals <ModulatoryMechanism.control_signals>`.

    combined_costs : 1d array
        result of the ModulatoryMechanism's `combine_costs <ModulatoryMechanism.combine_costs>` function.

    compute_net_outcome : Function, function or method
        function used to combine the values of its `outcome <ModulatoryMechanism.outcome>` and `costs
        <ModulatoryMechanism.costs>` attributes;  result is an array with a scalar value that can be accessed
        by the the `net_outcome <ModulatoryMechanism.net_outcome>` attribute.

    net_outcome : 1d array
        result of the ModulatoryMechanism's `compute_net_outcome <ModulatoryMechanism.compute_net_outcome>` function.

    modulatory_projections : List[ControlProjection | GatingProjection]
        list of `ControlProjections <ControlProjection>` and/or `GatingProjections <GatingProjection>`, one for each
        of the corresponding `ControlSignals <ControlSignal>` and/or `GatingSignals <GatingSignal>` in
        `modulatory_signals <ModulatoryMechanism.modulatory_signals>`.

    control_projections : List[ControlProjection]
        list of `ControlProjections <ControlProjection>`, one for each `ControlSignal` in `control_signals
        <ModulatoryMechanism.control_signals>`.

    gating_projections : List[GatingProjection]
        list of `GatingProjections <GatingProjection>`, one for each `GatingSignal` in `gating_signals
        <ModulatoryMechanism.gating_signals>`.

    modulation : ModulationParam
        the default form of modulation used by the ModulatoryMechanism's `ControlSignals <GatingSignal>`,
        unless they are `individually specified <ControlSignal_Specification>`.

    name : str
        the name of the ModulatoryMechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the ModulatoryMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentType = "ModulatoryMechanism"

    initMethod = INIT_EXECUTE_METHOD_ONLY

    outputStateTypes = [ControlSignal, GatingSignal]
    stateListAttr = Mechanism_Base.stateListAttr.copy()
    stateListAttr.update({ControlSignal:CONTROL_SIGNALS,
                          GatingSignal:GATING_SIGNALS})

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'ModulatoryMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    class Parameters(AdaptiveMechanism_Base.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <ModulatoryMechanism.variable>`

                    :default value: numpy.array([[1.]])
                    :type: numpy.ndarray

                value
                    see `value <ModulatoryMechanism.value>`

                    :default value: numpy.array([[1.]])
                    :type: numpy.ndarray

                combine_costs
                    see `combine_costs <ModulatoryMechanism.combine_costs>`

                    :default value: numpy.core.fromnumeric.sum
                    :type: <class 'function'>

                compute_net_outcome
                    see `compute_net_outcome <ModulatoryMechanism.compute_net_outcome>`

                    :default value: lambda outcome, cost: outcome - cost
                    :type: <class 'function'>

                compute_reconfiguration_cost
                    see `compute_reconfiguration_cost <ModulatoryMechanism.compute_reconfiguration_cost>`

                    :default value: None
                    :type:

                control_allocation
                    see `control_allocation <ModulatoryMechanism.control_allocation>`

                    :default value: numpy.array([1.])
                    :type: numpy.ndarray
                    :read only: True

                control_signal_costs
                    see `control_signal_costs <ModulatoryMechanism.control_signal_costs>`

                    :default value: None
                    :type:
                    :read only: True

                costs
                    see `costs <ModulatoryMechanism.costs>`

                    :default value: None
                    :type:
                    :read only: True

                default_allocation
                    see `default_allocation <ModulatoryMechanism.default_allocation>`

                    :default value: (None,)
                    :type: <class 'tuple'>

                gating_allocation
                    see `gating_allocation <ModulatoryMechanism.gating_allocation>`

                    :default value: numpy.array([0.5])
                    :type: numpy.ndarray
                    :read only: True

                modulation
                    see `modulation <ModulatoryMechanism.modulation>`

                    :default value: MULTIPLICATIVE
                    :type: `ModulationParam`

                net_outcome
                    see `net_outcome <ModulatoryMechanism.net_outcome>`

                    :default value: None
                    :type:
                    :read only: True

                outcome
                    see `outcome <ModulatoryMechanism.outcome>`

                    :default value: None
                    :type:
                    :read only: True

                reconfiguration_cost
                    see `reconfiguration_cost <ModulatoryMechanism.reconfiguration_cost>`

                    :default value: None
                    :type:
                    :read only: True

        """
        # This must be a list, as there may be more than one (e.g., one per control_signal)
        variable = np.array([[defaultControlAllocation]])
        value = Parameter(np.array([[defaultControlAllocation]]), aliases='modulatory_allocation')
        default_allocation = None,
        control_allocation = Parameter(np.array([defaultControlAllocation]),
                                       getter=_control_allocation_getter,
                                       setter=_control_allocation_setter,
                                       read_only=True)
        gating_allocation = Parameter(np.array([defaultGatingAllocation]),
                                      getter=_gating_allocation_getter,
                                      setter=_gating_allocation_setter,
                                      read_only=True)
        outcome = Parameter(None, read_only=True, getter=_outcome_getter)

        reconfiguration_cost = Parameter(None, read_only=True)
        compute_reconfiguration_cost = Parameter(None, stateful=False, loggable=False)

        combine_costs = Parameter(np.sum, stateful=False, loggable=False)
        costs = Parameter(None, read_only=True, getter=_modulatory_mechanism_costs_getter)
        control_signal_costs = Parameter(None, read_only=True)

        compute_net_outcome = Parameter(lambda outcome, cost: outcome - cost, stateful=False, loggable=False)
        net_outcome = Parameter(None, read_only=True,
                                getter=_net_outcome_getter)

        simulation_ids = Parameter([], user=False)

        modulation = MULTIPLICATIVE

        objective_mechanism = Parameter(None, stateful=False, loggable=False)

    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        OBJECTIVE_MECHANISM: None,
        CONTROL_PROJECTIONS: None})

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 system:tc.optional(tc.any(System_Base, Composition_Base))=None,
                 monitor_for_modulation:tc.optional(tc.any(is_iterable, Mechanism, OutputState))=None,
                 # objective_mechanism:tc.optional(ObjectiveMechanism, list, bool)=None,
                 objective_mechanism=None,
                 function=None,
                 default_allocation:tc.optional(tc.any(int, float, list, np.ndarray))=None,
                 modulatory_signals:tc.optional(tc.any(is_iterable,
                                                       ParameterState,
                                                       InputState,
                                                       OutputState,
                                                       ControlSignal,
                                                       GatingSignal))=None,
                 modulation:tc.optional(str)=MULTIPLICATIVE,
                 combine_costs:is_function_type=np.sum,
                 compute_reconfiguration_cost:tc.optional(is_function_type)=None,
                 compute_net_outcome:is_function_type=lambda outcome, cost : outcome - cost,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs
                 ):
        function = function or DefaultAllocationFunction
        modulatory_signals = modulatory_signals or []
        if not isinstance(modulatory_signals, list):
            modulatory_signals = [modulatory_signals]
        self.combine_costs = combine_costs
        self.compute_net_outcome = compute_net_outcome
        self.compute_reconfiguration_cost = compute_reconfiguration_cost

        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(system=system,
                                                  monitor_for_modulation=monitor_for_modulation,
                                                  objective_mechanism=objective_mechanism,
                                                  function=function,
                                                  default_allocation=default_allocation,
                                                  modulatory_signals=modulatory_signals,
                                                  modulation=modulation,
                                                  params=params)

        self._sim_counts = {}

        super(ModulatoryMechanism, self).__init__(default_variable=default_variable,
                                                  size=size,
                                                  modulation=modulation,
                                                  params=params,
                                                  name=name,
                                                  function=function,
                                                  prefs=prefs,
                                                  **kwargs)

        if system is not None:
            self._activate_projections_for_compositions(system)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate SYSTEM, monitor_for_modulation, CONTROL_SIGNALS and GATING_SIGNALS

        If System is specified, validate it
        Check that all items in monitor_for_modulation are Mechanisms or OutputStates for Mechanisms in self.system
        Check that all items in CONTROL_SIGNALS are parameters or ParameterStates for Mechanisms in self.system
        Check that all items in GATING_SIGNALS are States for Mechanisms in self.system
        """
        from psyneulink.core.components.system import MonitoredOutputStateTuple
        from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
        from psyneulink.core.components.states.inputstate import InputState
        from psyneulink.core.components.states.state import _parse_state_spec

        super(ModulatoryMechanism, self)._validate_params(request_set=request_set,
                                                       target_set=target_set,
                                                       context=context)

        def validate_monitored_state_spec(spec_list):
            for spec in spec_list:
                if isinstance(spec, MonitoredOutputStateTuple):
                    spec = spec.output_state
                elif isinstance(spec, tuple):
                    spec = spec[0]
                elif isinstance(spec, dict):
                    # If it is a dict, parse to validate that it is an InputState specification dict
                    #    (for InputState of ObjectiveMechanism to be assigned to the monitored_output_state)
                    spec = _parse_state_spec(owner=self,
                                             state_type=InputState,
                                             state_spec=spec,
                                             context=context)
                    # Get the OutputState, to validate that it is in the ModulatoryMechanism's System (below);
                    #    presumes that the monitored_output_state is the first in the list of projection_specs
                    #    in the InputState state specification dictionary returned from the parse,
                    #    and that it is specified as a projection_spec (parsed into that in the call
                    #    to _parse_connection_specs by _parse_state_spec)

                    spec = spec[PROJECTIONS][0][0]

                if not isinstance(spec, (OutputState, Mechanism)):
                    if isinstance(spec, type) and issubclass(spec, Mechanism):
                        raise ModulatoryMechanismError(
                                f"Mechanism class ({spec.__name__}) specified in '{MONITOR_FOR_MODULATION}' arg "
                                f"of {self.name}; it must be an instantiated {Mechanism.__name__} or "
                                f"{OutputState.__name__} of one.")
                    elif isinstance(spec, State):
                        raise ModulatoryMechanismError(
                                f"{spec.__class__.__name__} specified in '{MONITOR_FOR_MODULATION}' arg of {self.name} "
                                f"({spec.name} of {spec.owner.name}); "
                                f"it must be an {OutputState.__name__} or {Mechanism.__name__}.")
                    else:
                        raise ModulatoryMechanismError(
                                f"Erroneous specification of '{MONITOR_FOR_MODULATION}' arg for {self.name} ({spec}); "
                                f"it must be an {OutputState.__name__} or a {Mechanism.__name__}.")
                # If ModulatoryMechanism has been assigned to a System, check that
                #    all the items in the list used to specify objective_mechanism are in the same System
                # FIX: TBI FOR COMPOSITION
                if self.system:
                    if not isinstance(spec, (list, ContentAddressableList)):
                        spec = [spec]
                    self.system._validate_monitored_states_in_system(spec, context=context)


        if SYSTEM in target_set:
            if not isinstance(target_set[SYSTEM], System_Base):
                raise KeyError
            else:
                self.paramClassDefaults[SYSTEM] = request_set[SYSTEM]

        if MONITOR_FOR_MODULATION in target_set and target_set[MONITOR_FOR_MODULATION] is not None:
            spec = target_set[MONITOR_FOR_MODULATION]
            if not isinstance(spec, (list, ContentAddressableList)):
                spec = [spec]
            validate_monitored_state_spec(spec)

        if OBJECTIVE_MECHANISM in target_set and \
                target_set[OBJECTIVE_MECHANISM] is not None and\
                target_set[OBJECTIVE_MECHANISM] is not False:

            if isinstance(target_set[OBJECTIVE_MECHANISM], list):

                obj_mech_spec_list = target_set[OBJECTIVE_MECHANISM]

                # Check if there is any ObjectiveMechanism is in the list;
                #    incorrect but possibly forgivable mis-specification --
                #    if an ObjectiveMechanism is specified, it should be "exposed" (i.e., not in a list)
                if any(isinstance(spec, ObjectiveMechanism) for spec in obj_mech_spec_list):
                    # If an ObjectiveMechanism is the *only* item in the list, forgive the mis-spsecification and use it
                    if len(obj_mech_spec_list)==1 and isinstance(obj_mech_spec_list[0], ObjectiveMechanism):
                        if self.verbosePref:
                            warnings.warn("Specification of {} arg for {} is an {} in a list; it will be used, "
                                                        "but, for future reference, it should not be in a list".
                                                        format(OBJECTIVE_MECHANISM,
                                                               ObjectiveMechanism.__name__,
                                                               self.name))
                        target_set[OBJECTIVE_MECHANISM] = target_set[OBJECTIVE_MECHANISM][0]
                    else:
                        raise ModulatoryMechanismError("Ambigusous specification of {} arg for {}; "
                                                    " it is in a list with other items ({})".
                                                    format(OBJECTIVE_MECHANISM, self.name, obj_mech_spec_list))
                else:
                    validate_monitored_state_spec(obj_mech_spec_list)

            if not isinstance(target_set[OBJECTIVE_MECHANISM], (ObjectiveMechanism, list, bool)):
                raise ModulatoryMechanismError("Specification of {} arg for {} ({}) must be an {}"
                                            "or a list of Mechanisms and/or OutputStates to be monitored for control".
                                            format(OBJECTIVE_MECHANISM,
                                                   self.name, target_set[OBJECTIVE_MECHANISM],
                                                   ObjectiveMechanism.componentName))

        if MODULATORY_SIGNALS in target_set and target_set[MODULATORY_SIGNALS]:
            modulatory_signals = target_set[MODULATORY_SIGNALS]
            if not isinstance(modulatory_signals, list):
                modulatory_signals = [modulatory_signals]
            from psyneulink.core.components.projections.projection import ProjectionError
            for modulatory_signal in modulatory_signals:
                # _parse_state_spec(state_type=ControlSignal, owner=self, state_spec=control_signal)
                try:
                    _parse_state_spec(state_type=ControlSignal, owner=self, state_spec=modulatory_signal)
                except ProjectionError:
                    _parse_state_spec(state_type=GatingSignal, owner=self, state_spec=modulatory_signal)

    # IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
    def _instantiate_objective_mechanism(self, context=None):
        """
        # FIX: ??THIS SHOULD BE IN OR MOVED TO ObjectiveMechanism
        Assign InputState to ObjectiveMechanism for each OutputState to be monitored;
            uses _instantiate_monitoring_input_state and _instantiate_control_mechanism_input_state to do so.
            For each item in self.monitored_output_states:
            - if it is a OutputState, call _instantiate_monitoring_input_state()
            - if it is a Mechanism, call _instantiate_monitoring_input_state for relevant Mechanism.output_states
                (determined by whether it is a `TERMINAL` Mechanism and/or MonitoredOutputStatesOption specification)
            - each InputState is assigned a name with the following format:
                '<name of Mechanism that owns the monitoredOutputState>_<name of monitoredOutputState>_Monitor'

        Notes:
        * self.monitored_output_states is a list, each item of which is a Mechanism.output_state from which a
          Projection will be instantiated to a corresponding InputState of the ModulatoryMechanism
        * self.input_states is the usual ordered dict of states,
            each of which receives a Projection from a corresponding OutputState in self.monitored_output_states
        """
        from psyneulink.core.components.system import MonitoredOutputStateTuple
        from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
        from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism, ObjectiveMechanismError
        from psyneulink.core.components.states.inputstate import EXPONENT_INDEX, WEIGHT_INDEX
        from psyneulink.core.components.functions.function import FunctionError

        # GET OutputStates to Monitor (to specify as or add to ObjectiveMechanism's monitored_output_states attribute

        monitored_output_states = []

        # If the ModulatoryMechanism has already been assigned to a System
        #    get OutputStates in System specified as monitor_for_control or already being monitored:
        #        do this by calling _get_monitored_output_states_for_system(),
        #        which also gets any OutputStates already being monitored by the ModulatoryMechanism
        if self.system:
            monitored_output_states.extend(self.system._get_monitored_output_states_for_system(self,context=context))

        self.monitor_for_modulation = self.monitor_for_modulation or []
        if not isinstance(self.monitor_for_modulation, list):
            self.monitor_for_modulation = [self.monitor_for_modulation]

        # If objective_mechanism is used to specify OutputStates to be monitored (legacy feature)
        #    move them to monitor_for_modulation
        if isinstance(self.objective_mechanism, list):
            self.monitor_for_modulation.extend(self.objective_mechanism)

        # Add items in monitor_for_modulation to monitored_output_states
        for i, item in enumerate(self.monitor_for_modulation):
            # If it is already in the list received from System, ignore
            if item in monitored_output_states:
                # NOTE: this can happen if ModulatoryMechanisms is being constructed by System
                #       which passed its monitor_for_control specification
                continue
            monitored_output_states.extend([item])

        # INSTANTIATE ObjectiveMechanism

        # If *objective_mechanism* argument is an ObjectiveMechanism, add monitored_output_states to it
        if isinstance(self.objective_mechanism, ObjectiveMechanism):
            if monitored_output_states:
                self.objective_mechanism.add_to_monitor(monitor_specs=monitored_output_states,
                                                        context=context)
        # Otherwise, instantiate ObjectiveMechanism with list of states in monitored_output_states
        else:
            try:
                self.objective_mechanism = ObjectiveMechanism(monitor=monitored_output_states,
                                                               function=LinearCombination(operation=PRODUCT),
                                                               name=self.name + '_ObjectiveMechanism')
            except (ObjectiveMechanismError, FunctionError) as e:
                raise ObjectiveMechanismError(f"Error creating {OBJECTIVE_MECHANISM} for {self.name}: {e}")

        # Print monitored_output_states
        if self.prefs.verbosePref:
            print("{0} monitoring:".format(self.name))
            for state in self.monitored_output_states:
                weight = self.monitored_output_states_weights_and_exponents[
                                                         self.monitored_output_states.index(state)][WEIGHT_INDEX]
                exponent = self.monitored_output_states_weights_and_exponents[
                                                         self.monitored_output_states.index(state)][EXPONENT_INDEX]
                print(f"\t{weight} (exp: {weight}; wt: {exponent})")

        # Assign ObjectiveMechanism's role as CONTROL
        self.objective_mechanism._role = CONTROL

        # Instantiate MappingProjection from ObjectiveMechanism to ModulatoryMechanism
        projection_from_objective = MappingProjection(sender=self.objective_mechanism,
                                                      receiver=self,
                                                      matrix=AUTO_ASSIGN_MATRIX,
                                                      context=context)

        # CONFIGURE FOR ASSIGNMENT TO COMPOSITION

        # Insure that ObjectiveMechanism's input_states are not assigned projections from a Composition's input_CIM
        for input_state in self.objective_mechanism.input_states:
            input_state.internal_only = True
        # Flag ObjectiveMechanism and its Projection to ModulatoryMechanism for inclusion in Composition
        self.aux_components.append(self.objective_mechanism)
        self.aux_components.append(projection_from_objective)

        # ASSIGN ATTRIBUTES

        self._objective_projection = projection_from_objective
        self.monitor_for_modulation = self.monitored_output_states

    def _register_modulatory_signal_type(self, modulatory_signal_type:ModulatorySignal, context=None):
        from psyneulink.core.globals.registry import register_category
        from psyneulink.core.components.states.state import State_Base

        # Create registry for ControlSignals (to manage names)
        register_category(entry=modulatory_signal_type,
                          base_class=State_Base,
                          registry=self._stateRegistry,
                          context=context)

    def _instantiate_input_states(self, context=None):

        super()._instantiate_input_states(context=context)
        self.input_state.name = OUTCOME
        self.input_state.name = OUTCOME

        # If objective_mechanism is specified, instantiate it,
        #     including Projections to it from monitor_for_control
        if self.objective_mechanism:
            self._instantiate_objective_mechanism(context=context)

        # Otherwise, instantiate Projections from monitor_for_modulation to ModulatoryMechanism
        elif self.monitor_for_modulation:
            from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
            for sender in convert_to_list(self.monitor_for_modulation):
                self.aux_components.append(MappingProjection(sender=sender, receiver=self.input_states[OUTCOME]))

    def _instantiate_output_states(self, context=None):

    # ---------------------------------------------------
    # FIX 5/23/17: PROJECTIONS AND PARAMS SHOULD BE PASSED BY ASSIGNING TO STATE SPECIFICATION DICT
    # FIX          UPDATE parse_state_spec TO ACCOMODATE (param, ControlSignal) TUPLE
    # FIX          TRACK DOWN WHERE PARAMS ARE BEING HANDED OFF TO ControlProjection
    # FIX                   AND MAKE SURE THEY ARE NOW ADDED TO ControlSignal SPECIFICATION DICT
    # ---------------------------------------------------

        if self.modulatory_signals:
            self._instantiate_modulatory_signals(context=context)

        super()._instantiate_output_states(context=context)

        # Reassign modulatory_signals, control_signals and gating_signals to backing fields of corresponding params
        # to capture any user_defined ControlSignals and/or GatingSignals instantiated in call to super
        # and assign to ContentAddressableLists
        self._modulatory_signals = ContentAddressableList(component_type=ModulatorySignal,
                                                       list=[state for state in self.output_states
                                                             if isinstance(state, (ControlSignal, GatingSignal))])

    def _instantiate_modulatory_signals(self, context):
        """Subclassess can override for class-specific implementation (see OptimiziationControlMechanism for example)"""
        for i, modulatory_signal in enumerate(self.modulatory_signals):
            self.modulatory_signals[i] = self._instantiate_modulatory_signal(modulatory_signal, context=context)
        num_modulatory_signals = i+1

        # For DefaultAllocationFunction, set defaults.value to have number of items equal to num modulatory_signals
        if isinstance(self.function, DefaultAllocationFunction):
            self.defaults.value = np.tile(self.function.value, (num_modulatory_signals, 1))
            self.parameters.modulatory_allocation._set(copy.deepcopy(self.defaults.value), context)
            self.function.num_modulatory_signals = num_modulatory_signals

        # For other functions, assume that if its value has:
        # - one item, all modulatory_signals should get it (i.e., the default: (OWNER_VALUE, 0));
        # - same number of items as the number of modulatory_signals;
        #     assign each modulatory_signal to the corresponding item of the function's value
        # - a different number of items than number of modulatory_signals,
        #     leave things alone, and allow any errant indices for modulatory_signals to be caught later.
        else:
            self.defaults.value = np.array(self.function.value)
            self.parameters.value._set(copy.deepcopy(self.defaults.value), context)

            len_fct_value = len(self.function.value)

            # Assign each ModulatorySignal's variable_spec to index of ModulatoryMechanism's value
            for i, modulatory_signal in enumerate(self.modulatory_signals):

                # If number of modulatory_signals is same as number of items in function's value,
                #    assign each ModulatorySignal to the corresponding item of the function's value
                if len_fct_value == num_modulatory_signals:
                    modulatory_signal._variable_spec = [(OWNER_VALUE, i)]

                if not isinstance(modulatory_signal.owner_value_index, int):
                    assert False, \
                        f"PROGRAM ERROR: The \'owner_value_index\' attribute for {modulatory_signal.name} " \
                            f"of {self.name} ({modulatory_signal.owner_value_index})is not an int."


    def _instantiate_modulatory_signal(self,  modulatory_signal, context=None):
        """Parse and instantiate modulatory_signal specifications (in call to State._parse_state_spec)
           and any embedded Projection specifications (in call to <State>._instantiate_projections)

        Temporarily assign variable to default allocation value to avoid chicken-and-egg problem:
           value, output_states and modulatory_signals haven't been expanded yet to accomodate the new
           ModulatorySignal; reassign modulatory_signal.variable to actual OWNER_VALUE below,
           once value has been expanded
        """
        from psyneulink.core.components.states.state import _instantiate_state, StateError
        from psyneulink.core.components.projections.projection import ProjectionError
        from psyneulink.core.components.states.state import StateError

        if self._output_states is None:
            self._output_states = []
        mod_spec = modulatory_signal

        # Try to instantiate as ControlSignal;  if that fails, try GatingSignal
        try:
            modulatory_signal = _instantiate_state(state_type=ControlSignal,
                                                   owner=self,
                                                   variable=self.default_allocation or
                                                            self.parameters.control_allocation.default_value,
                                                   reference_value=self.parameters.control_allocation.default_value,
                                                   modulation=self.modulation,
                                                   state_spec=mod_spec,
                                                   context=context)
            if not type(modulatory_signal) in convert_to_list(self.outputStateTypes):
                raise ProjectionError(f'{type(modulatory_signal)} inappropriate for {self.name}')

        except (ProjectionError, StateError):
            try:
                modulatory_signal = _instantiate_state(state_type=GatingSignal,
                                                       owner=self,
                                                       variable=self.parameters.gating_allocation.default_value,
                                                       # reference_value=GatingSignal.defaults.allocation,
                                                       reference_value=self.parameters.gating_allocation.default_value,
                                                       modulation=self.modulation,
                                                       state_spec=mod_spec,
                                                       context=context)
            except StateError as e:
                raise ModulatoryMechanismError(f"\nPROGRAM ERROR: Unrecognized {repr(MODULATORY_SIGNAL)} "
                                               f"specification for {self.name} ({modulatory_signal}); \n"
                                               f"ERROR MESSAGE: {e.args[0]}")

        modulatory_signal.owner = self

        # Check that modulatory_signal is not a duplicate of one already instantiated for the ModulatoryMechanism
        # (viz., if control of parameter was specified both in constructor for Mechanism and in ModulatoryMechanism)
        for existing_mod_sig in [ms for ms in self._modulatory_signals if isinstance(ms, ModulatorySignal)]:

            # OK if modulatory_signal is one already assigned to ModulatoryMechanism (i.e., let it get processed below);
            # this can happen if it was in deferred_init status and initalized in call to _instantiate_state above.
            if modulatory_signal == existing_mod_sig:
                continue

            # Return if *all* projections from modulatory_signal are identical to ones in an existing modulatory_signal
            for proj in modulatory_signal.efferents:
                if proj not in existing_mod_sig.efferents:
                    # A Projection in modulatory_signal is not in this existing one: it is different,
                    #    so break and move on to next existing_mod_sig
                    break
                return

            # Warn if *any* projections from modulatory_signal are identical to ones in an existing modulatory_signal
            if any(
                    any(new_p.receiver == existing_p.receiver
                        for existing_p in existing_mod_sig.efferents) for new_p in modulatory_signal.efferents):
                # warnings.warn(f"{modulatory_signal.__class__.__name__} ({modulatory_signal.name}) has ")
                warnings.warn(f"Specification of {modulatory_signal.name} for {self.name} "
                              f"has one or more {MODULATORY_PROJECTION}s redundant with ones already on "
                              f"an existing {ModulatorySignal.__name__} ({existing_mod_sig.name}).")

        if isinstance(modulatory_signal, ControlSignal):
            # Update control_signal_costs to accommodate instantiated Projection
            control_signal_costs = self.parameters.control_signal_costs._get(context)
            try:
                control_signal_costs = np.append(control_signal_costs, np.zeros((1, 1)), axis=0)
            except (AttributeError, ValueError):
                control_signal_costs = np.zeros((1, 1))
            self.parameters.control_signal_costs._set(control_signal_costs, context)

        # UPDATE output_states AND modulatory_projections -------------------------------------------------------------

        # FIX: 9/14/19 - THIS SHOULD BE IMPLEMENTED
        # TBI: For modulatory mechanisms that accumulate, starting output must be equal to the initial "previous value"
        # so that modulation that occurs BEFORE the control mechanism executes is computed appropriately
        # if (isinstance(self.function, IntegratorFunction)):
        #     control_signal._intensity = function.initializer

        # Add ModulatorySignal to output_states list
        self._output_states.append(modulatory_signal)

        return modulatory_signal

    def show(self):
        """Display the OutputStates monitored by ModulatoryMechanism's `objective_mechanism
        <ModulatoryMechanism.objective_mechanism>` and the parameters modulated by its `control_signals
        <ModulatoryMechanism.control_signals>`.
        """

        print("\n---------------------------------------------------------")

        print("\n{0}".format(self.name))
        print("\n\tMonitoring the following Mechanism OutputStates:")
        for state in self.objective_mechanism.input_states:
            for projection in state.path_afferents:
                monitored_state = projection.sender
                monitored_state_mech = projection.sender.owner
                # ContentAddressableList
                monitored_state_index = self.monitored_output_states.index(monitored_state)

                weight = self.monitored_output_states_weights_and_exponents[monitored_state_index][0]
                exponent = self.monitored_output_states_weights_and_exponents[monitored_state_index][1]

                print ("\t\t{0}: {1} (exp: {2}; wt: {3})".
                       format(monitored_state_mech.name, monitored_state.name, weight, exponent))

        try:
            if self.control_signals:
                print ("\n\tControlling the following Mechanism parameters:".format(self.name))
                # Sort for consistency of output:
                state_names_sorted = sorted(self.control_signals.names)
                for state_name in state_names_sorted:
                    for projection in self.control_signals[state_name].efferents:
                        print ("\t\t{0}: {1}".format(projection.receiver.owner.name, projection.receiver.name))
        except:
            pass

        try:
            if self.gating_signals:
                print ("\n\tGating the following States:".format(self.name))
                # Sort for consistency of output:
                state_names_sorted = sorted(self.gating_signals.names)
                for state_name in state_names_sorted:
                    for projection in self.gating_signals[state_name].efferents:
                        print ("\t\t{0}: {1}".format(projection.receiver.owner.name, projection.receiver.name))
        except:
            pass

        print ("\n---------------------------------------------------------")

    def add_to_monitor(self, monitor_specs, context=None):
        """Instantiate OutputStates to be monitored by ModulatoryMechanism's `objective_mechanism
        <ModulatoryMechanism.objective_mechanism>`.

        **monitored_output_states** can be any of the following:
            - `Mechanism`;
            - `OutputState`;
            - `tuple specification <InputState_Tuple_Specification>`;
            - `State specification dictionary <InputState_Specification_Dictionary>`;
            - list with any of the above.
        If any item is a Mechanism, its `primary OutputState <OutputState_Primary>` is used.
        OutputStates must belong to Mechanisms in the same `System` as the ModulatoryMechanism.
        """
        output_states = self.objective_mechanism.add_to_monitor(monitor_specs=monitor_specs, context=context)
        if self.system:
            self.system._validate_monitored_states_in_system(output_states, context=context)

    def _add_process(self, process, role:str):
        super()._add_process(process, role)
        if self.objective_mechanism:
            self.objective_mechanism._add_process(process, role)

    # FIX: TBI FOR COMPOSITION
    @tc.typecheck
    @handle_external_context()
    def assign_as_controller(self, system:System_Base, context=None):
        """Assign ModulatoryMechanism as `controller <System.controller>` for a `System`.

        **system** must be a System for which the ModulatoryMechanism should be assigned as the `controller
        <System.controller>`.
        If the specified System already has a `controller <System.controller>`, it will be replaced by the current
        one, and the current one will inherit any ControlSignals previously specified for the old controller or the
        System itself.
        If the current one is already the `controller <System.controller>` for another System, it will be disabled
        for that System.
        COMMENT:
            [TBI:
            The ModulatoryMechanism's `objective_mechanism <ModulatoryMechanism.objective_mechanism>`,
            `monitored_output_states` and `control_signal <ModulatoryMechanism.control_signals>` attributes will also be
            updated to remove any assignments that are not part of the new System, and add any that are specified for
            the new System.]
        COMMENT

        COMMENT:
            IMPLEMENTATION NOTE:  This is handled as a method on ModulatoryMechanism (rather than System) so that:

                                  - [TBI: if necessary, it can detach itself from a System for which it is already the
                                    `controller <System.controller>`;]

                                  - any class-specific actions that must be taken to instantiate the ModulatoryMechanism
                                    can be handled by subclasses of ModulatoryMechanism (e.g., an EVCModulatoryMechanism must
                                    instantiate its Prediction Mechanisms). However, the actual assignment of the
                                    ModulatoryMechanism the System's `controller <System.controller>` attribute must
                                    be left to the System to avoid recursion, since it is a property, the setter of
                                    which calls the current method.
        COMMENT
        """

        if context.source == ContextFlags.COMMAND_LINE:
            system.controller = self
            return

        if self.objective_mechanism is None:
            self._instantiate_objective_mechanism(context=context)

        # NEED TO BUFFER OBJECTIVE_MECHANISM AND CONTROL_SIGNAL ARGUMENTS FOR USE IN REINSTANTIATION HERE
        # DETACH AS CONTROLLER FOR ANY EXISTING SYSTEM (AND SET THAT ONE'S CONTROLLER ATTRIBUTE TO None)
        # DELETE ALL EXISTING OBJECTIVE_MECHANISM AND CONTROL_SIGNAL ASSIGNMENTS
        # REINSTANTIATE ITS OWN OBJECTIVE_MECHANISM and CONTROL_SIGNAL ARGUMENT AND THOSE OF THE SYSTEM
        # SUBCLASSES SHOULD ADD OVERRIDE FOR ANY CLASS-SPECIFIC ACTIONS (E.G., INSTANTIATING PREDICTION MECHANISMS)
        # DO *NOT* ASSIGN AS CONTROLLER FOR SYSTEM... LET THE SYSTEM HANDLE THAT
        # Assign the current System to the ModulatoryMechanism

        # Validate that all of the ModulatoryMechanism's monitored_output_states and controlled parameters
        #    are in the new System
        system._validate_monitored_states_in_system(self.monitored_output_states)
        system._validate_control_signals(self.control_signals)

        # Get any and all OutputStates specified in:
        # - **monitored_output_states** argument of the System's constructor
        # - in a monitor_for_control specification for individual OutputStates and/or Mechanisms
        # - already being montiored by the ModulatoryMechanism being assigned
        monitored_output_states = list(system._get_monitored_output_states_for_system(controller=self, context=context))

        # Don't add any OutputStates that are already being monitored by the ModulatoryMechanism's ObjectiveMechanism
        for monitored_output_state in monitored_output_states.copy():
            if monitored_output_state.output_state in self.monitored_output_states:
                monitored_output_states.remove(monitored_output_state)

        # Add all other monitored_output_states to the ModulatoryMechanism's monitored_output_states attribute
        #    and to its ObjectiveMechanisms monitored_output_states attribute
        if monitored_output_states:
            self.add_to_monitor(monitored_output_states)

        # The system does NOT already have a controller,
        #    so assign it ControlSignals for any parameters in the System specified for control
        if system.controller is None:
            system_control_signals = system._get_control_signals_for_system(system.control_signals_arg, context=context)
        # The system DOES already have a controller,
        #    so assign it the old controller's ControlSignals
        else:
            system_control_signals = system.control_signals
            for control_signal in system_control_signals:
                control_signal.owner = None

        # Get rid of default ControlSignal if it has no ControlProjections
        if (len(self.control_signals)==1
                and self.control_signals[0].name=='ControlSignal-0'
                and not self.control_signals[0].efferents):
            del self._output_states[0]
            del self.control_signals[0]

        # Add any ControlSignals specified for System
        for control_signal_spec in system_control_signals:
            control_signal = self._instantiate_control_signal(control_signal=control_signal_spec, context=context)
            # FIX: 1/18/18 - CHECK FOR SAME NAME IN _instantiate_control_signal
            # # Don't add any that are already on the ModulatoryMechanism
            if control_signal.name in self.control_signals.names and (self.verbosePref or system.verbosePref):
                warnings.warn("{} specified for {} has same name (\'{}\') "
                              "as one in controller ({}) being assigned to the {}."
                              "".format(ControlSignal.__name__, system.name,
                                        control_signal.name, self.name, system.__class__.__name__))
            self.control_signals.append(control_signal)

        # If it HAS been assigned a System, make sure it is the current one
        if self.system and not self.system is system:
            raise SystemError("The controller being assigned to {} ({}) already belongs to another System ({})".
                              format(system.name, self.name, self.system.name))

        # Assign assign the current System to the ModulatoryMechanism's system attribute
        #    (needed for it to validate and instantiate monitored_output_states and control_signals)
        self.system = system

        # Flag ObjectiveMechanism as associated with a ModulatoryMechanism that is a controller for the System
        self.objective_mechanism.for_controller = True

        if context.source != ContextFlags.PROPERTY:
            system._controller = self

        self._activate_projections_for_compositions(system)

    def _remove_default_modulatory_signal(self, type:tc.enum(MODULATORY_SIGNAL, CONTROL_SIGNAL, GATING_SIGNAL)):
        if type == MODULATORY_SIGNAL:
            mod_sig_attribute = self.modulatory_signals
        elif type == CONTROL_SIGNAL:
            mod_sig_attribute = self.control_signals
        elif type == GATING_SIGNAL:
            mod_sig_attribute = self.gating_signals
        else:
            assert False, \
                f"PROGRAM ERROR:  bad 'type' arg ({type})passed to " \
                    f"{ModulatoryMechanism.__name__}._remove_default_modulatory_signal" \
                    f"(should have been caught by typecheck"

        if (len(mod_sig_attribute)==1
                and mod_sig_attribute[0].name==type+'-0'
                and not mod_sig_attribute[0].efferents):
            self.remove_states(mod_sig_attribute[0])

    def _activate_projections_for_compositions(self, composition=None):
        """Activate eligible Projections to or from nodes in composition.
        If Projection is to or from a node NOT (yet) in the Composition,
        assign it the node's aux_components attribute but do not activate it.
        """
        dependent_projections = set()

        if self.objective_mechanism:
            # Safe to add this, as it is already in the ModulatoryMechanism's aux_components
            #    and will therefore be added to the Composition along with the ModulatoryMechanism
            assert self.objective_mechanism in self.aux_components, \
                f"PROGRAM ERROR:  {OBJECTIVE_MECHANISM} for {self.name} not listed in its 'aux_components' attribute."
            dependent_projections.add(self._objective_projection)

            for aff in self.objective_mechanism.afferents:
                dependent_projections.add(aff)

        for ms in self.modulatory_signals:
            for eff in ms.efferents:
                dependent_projections.add(eff)

        # FIX: 9/15/19 - HOW IS THIS DIFFERENT THAN objective_mechanism's AFFERENTS ABOVE?
        # assign any deferred init objective mech monitored output state projections to this system
        if self.objective_mechanism:
            for output_state in self.objective_mechanism.monitored_output_states:
                for eff in output_state.efferents:
                    dependent_projections.add(eff)

        # FIX: 9/15/19 - HOW IS THIS DIFFERENT THAN modulatory_signal's EFFERENTS ABOVE?
        for eff in self.efferents:
            dependent_projections.add(eff)

        for proj in dependent_projections:
            proj._activate_for_compositions(composition)

    def _apply_modulatory_allocation(self, modulatory_allocation, runtime_params, context):
        """Update values to `modulatory_signals <ModulatoryMechanism.modulatory_signals>`
        based on specified `modulatory_allocation <ModulatoryMechanism.modulatory_allocation>`
        (used by controller of a Composition in simulations)
        """
        value = [np.atleast_1d(a) for a in modulatory_allocation]
        self.parameters.value._set(value, context)
        self._update_output_states(context=context,
                                   runtime_params=runtime_params,
                                   )

    @property
    def monitored_output_states(self):
        try:
            return self.objective_mechanism.monitored_output_states
        except AttributeError:
            return None

    @monitored_output_states.setter
    def monitored_output_states(self, value):
        try:
            self.objective_mechanism._monitored_output_states = value
        except AttributeError:
            return None

    @property
    def monitored_output_states_weights_and_exponents(self):
        try:
            return self.objective_mechanism.monitored_output_states_weights_and_exponents
        except:
            return None

    @property
    def modulatory_projections(self):
        try:
            return [projection for modulatory_signal in self.modulatory_signals
                    for projection in modulatory_signal.efferents]
        except:
            return None

    @property
    def control_signals(self):
        try:
            return ContentAddressableList(component_type=ControlSignal,
                                          list=[state for state in self.output_states
                                                if isinstance(state, ControlSignal)])
        except:
            return None

    @property
    def control_projections(self):
        try:
            return [projection for control_signal in self.control_signals for projection in control_signal.efferents]
        except:
            return None

    @property
    def gating_signals(self):
        try:
            return ContentAddressableList(component_type=GatingSignal,
                                          list=[state for state in self.output_states
                                                if isinstance(state, GatingSignal)])
        except:
            return None

    @property
    def gating_projections(self):
        try:
            return [projection for gating_signal in self.gating_signals for projection in gating_signal.efferents]
        except:
            return None

    @property
    def _sim_count_lock(self):
        try:
            return self.__sim_count_lock
        except AttributeError:
            self.__sim_count_lock = threading.Lock()
            return self.__sim_count_lock

    def get_next_sim_id(self, context):
        with self._sim_count_lock:
            try:
                sim_num = self._sim_counts[context.execution_id]
                self._sim_counts[context.execution_id] += 1
            except KeyError:
                sim_num = 0
                self._sim_counts[context.execution_id] = 1

        return '{0}{1}-{2}'.format(context.execution_id, EID_SIMULATION, sim_num)

    @property
    def _dependent_components(self):
        return list(itertools.chain(
            super()._dependent_components,
            # [self.objective_mechanism],
            [self.objective_mechanism] if self.objective_mechanism else [],
        ))
