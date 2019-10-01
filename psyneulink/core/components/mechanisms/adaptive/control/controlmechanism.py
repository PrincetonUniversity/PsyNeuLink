# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# **************************************  ControlMechanism ************************************************

"""
Sections
--------

  * `ControlMechanism_Overview`
  * `ControlMechanism_Composition_Controller`
  * `ControlMechanism_Creation`
      - `ControlMechanism_Monitor_for_Control`
      - `ControlMechanism_ObjectiveMechanism`
      - `ControlMechanism_Control_Signals`
  * `ControlMechanism_Structure`
      - `ControlMechanism_Input`
      - `ControlMechanism_Function`
      - 'ControlMechanism_Output`
      - `ControlMechanism_Costs_NetOutcome`
  * `ControlMechanism_Execution`
  * `ControlMechanism_Examples`
  * `ControlMechanism_Class_Reference`

.. _ControlMechanism_Overview:

Overview
--------

A ControlMechanism is a `ModulatoryMechanism` that `modulates the value(s) <ModulatorySignal_Modulation>` of one or
more `States <State>` of other Mechanisms in the `Composition` to which it belongs. In general, a ControlMechanism is
used to modulate the `ParameterState(s) <ParameterState>` of one or more Mechanisms, that determine the value(s) of
the parameter(s) of the `function(s) <Mechanism_Base.function>` of those Mechanism(s). However, a ControlMechanism
can also be used to modulate the function of `InputStates <InputState>` and/or `OutputState <OutputStates>`,
much like a `GatingMechanism`.  A ControlMechanism's `function <ControlMechanism.function>` calculates a
`control_allocation <ControlMechanism.control_allocation>`: a list of values provided to each of its `control_signals
<ControlMechanism.control_signals>`.  Its control_signals are `ControlSignal` OutputStates that are used to modulate
the parameters of other Mechanisms' `function <Mechanism_Base.function>` (see `ControlSignal_Modulation` for a more
detailed description of how modulation operates).  A ControlMechanism can be configured to monitor the outputs of
other Mechanisms in order to determine its `control_allocation <ControlMechanism.control_allocation>`, by specifying
these in the **monitor_for_control** `argument <ControlMechanism_Monitor_for_Control_Argument>` of its constructor,
or in the **monitor** `argument <ObjectiveMechanism_Monitor>` of an ObjectiveMechanism` assigned to its
**objective_mechanism** `argument <ControlMechanism_Objective_Mechanism_Argument>` (see `ControlMechanism_Creation`
below).  A ControlMechanism can also be assigned as the `controller <Composition.controller>` of a `Composition`,
which has a special relation to that Composition: it generally executes either before or after all of the other
Mechanisms in that Composition (see `Composition_Controller_Execution`).  The OutputStates monitored by the
ControlMechanism or its `objective_mechanism <ControlMechanism.objective_mechanism>`, and the parameters it modulates
can be listed using its `show <ControlMechanism.show>` method.

Note that a ControlMechanism is a subclass of `ModulatoryMechanism` that is restricted to using only `ControlSignals
<ControlSignal>`.  Accordingly, its constructor has a **control_signals** argument in place of a **modulatory_signals**
argument, and it lacks any attributes related to gating.  In all other respects it is identical to its parent class,
ModulatoryMechanism.

.. _ControlMechanism_Composition_Controller:

*ControlMechanisms and a Composition*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ControlMechanism can be assigned to a `Composition` and executed just like any other Mechanism. It can also be
assigned as the `controller <Composition.controller>` of a `Composition`, that has a special relation
to the Composition: it is used to control all of the parameters that have been `specified for control
<ControlMechanism_Control_Signals>` in that Composition.  A ControlMechanism can be the `controller
<Composition.controller>` for only one Composition, and a Composition can have only one `controller
<Composition.controller>`.  When a ControlMechanism is assigned as the `controller <Composition.controller>` of a
Composition (either in the Composition's constructor, or using its `add_controller <Composition.add_controller>`
method, the ControlMechanism assumes control over all of the parameters that have been `specified for control
<ControlMechanism_Control_Signals>` for Components in the Composition.  The Composition's `controller
<Composition.controller>` is executed either before or after all of the other Components in the Composition are
executed, including any other ControlMechanisms that belong to it (see `Composition_Controller_Execution`).  A
ControlMechanism can be assigned as the `controller <Composition.controller>` for a Composition by specifying it in
the **controller** argument of the Composition's constructor, or by using the Composition's `add_controller
<Composition.add_controller>` method.  A Composition's `controller <Composition.controller>` and its associated
Components can be displayed using the Composition's `show_graph <Composition.show_graph>` method with its
**show_control** argument assigned as `True`.


.. _ControlMechanism_Creation:

Creating a ControlMechanism
---------------------------

A ControlMechanism is created by calling its constructor.  When a ControlMechanism is created, the OutputStates it
monitors and the States it modulates can be specified in the **montior_for_control** and **objective_mechanism**
arguments of its constructor, respectively.  Each can be specified in several ways, as described below. If neither of
those arguments is specified, then only the ControlMechanism is constructed, and its inputs and the parameters it
modulates must be specified in some other way.

.. _ControlMechanism_Monitor_for_Control:

*Specifying OutputStates to be monitored*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ControlMechanism can be configured to monitor the output of other Mechanisms directly (by receiving direct
Projections from their OutputStates), or by way of an `ObjectiveMechanism` that evaluates those outputs and passes the
result to the ControlMechanism (see `below <ControlMechanism_ObjectiveMechanism>` for more detailed description).
The following figures show an example of each:

+-------------------------------------------------------------------------+----------------------------------------------------------------------+
| .. figure:: _static/ControlMechanism_without_ObjectiveMechanism_fig.svg | .. figure:: _static/ControlMechanism_with_ObjectiveMechanism_fig.svg |
+-------------------------------------------------------------------------+----------------------------------------------------------------------+

COMMENT:
FIX: USE THIS IF MOVED TO SECTION AT END THAT CONSOLIDATES EXAMPLES
**ControlMechanism with and without ObjectiveMechanism**

+-------------------------------------------------------------------------+-------------------------------------------------------------------------+
| >>> mech_A = ProcessingMechanism(name='ProcessingMechanism A')          | .. figure:: _static/ControlMechanism_without_ObjectiveMechanism_fig.svg |
| >>> mech_B = ProcessingMechanism(name='ProcessingMechanism B')          |                                                                         |
| >>> ctl_mech = ControlMechanism(name='ControlMechanism',                |                                                                         |
| ...                             monitor_for_control=[mech_A,            |                                                                         |
| ...                                                  mech_B],           |                                                                         |
| ...                             control_signals=[(SLOPE,mech_A),        |                                                                         |
| ...                                              (SLOPE,mech_B)])       |                                                                         |
| >>> comp = Composition()                                                |                                                                         |
| >>> comp.add_linear_processing_pathway([mech_A,mech_B, ctl_mech])       |                                                                         |
| >>> comp.show_graph()                                                   |                                                                         |
+-------------------------------------------------------------------------+-------------------------------------------------------------------------+
| >>> mech_A = ProcessingMechanism(name='ProcessingMechanism A')          | .. figure:: _static/ControlMechanism_with_ObjectiveMechanism_fig.svg    |
| >>> mech_B = ProcessingMechanism(name='ProcessingMechanism B')          |                                                                         |
| >>> ctl_mech = ControlMechanism(name='ControlMechanism',                |                                                                         |
| ...                             monitor_for_control=[mech_A,            |                                                                         |
| ...                                                  mech_B],           |                                                                         |
| ...                             objective_mechanism=True,               |                                                                         |
| ...                             control_signals=[(SLOPE,mech_A),        |                                                                         |
| ...                                              (SLOPE,mech_B)])       |                                                                         |
| >>> comp = Composition()                                                |                                                                         |
| >>> comp.add_linear_processing_pathway([mech_A,mech_B, ctl_mech])       |                                                                         |
| >>> comp.show_graph()                                                   |                                                                         |
+-------------------------------------------------------------------------+-------------------------------------------------------------------------+
COMMENT

Note that, in the figures above, the `ControlProjections <ControlProjection>` are designated with square "arrowheads",
and the ControlMechanisms are shown as septagons to indicate that their ControlProjections create a feedback loop
(see `Composition_Initial_Values_and_Feedback`;  also, see `below <ControlMechanism_Add_Linear_Processing_Pathway>`
regarding specification of a ControlMechanism and associated ObjectiveMechanism in a Composition's
`add_linear_processing_pathway <Composition.add_linear_processing_pathway>` method).

Which configuration is used is determined by how the following arguments of the ControlMechanism's constructor are
specified (also see `ControlMechanism_Examples`):

  .. _ControlMechanism_Monitor_for_Control_Argument:

  * **monitor_for_control** -- a list of `OutputState specifications <OutputState_Specification>`.  If the
    **objective_mechanism** argument is not specified (or is *False* or *None*) then, when the ControlMechanism is
    added to a `Composition`, a `MappingProjection` is created for each OutputState specified to the ControlMechanism's
    *OUTCOME* `input_state <ControlMechanism_Input>`.  If the **objective_mechanism** `argument
    <ControlMechanism_Objective_Mechanism_Argument>` is specified, then the OutputStates specified in
    **monitor_for_control** are assigned to the `ObjectiveMechanism` rather than the ControlMechanism itself (see
    `ControlMechanism_ObjectiveMechanism` for details).

  .. _ControlMechanism_Objective_Mechanism_Argument:

  * **objective_mechanism** -- if this is specfied in any way other than **False** or **None** (the default),
    then an ObjectiveMechanism is created that projects to the ControlMechanism and, when added to a `Composition`,
    is assigned Projections from all of the OutputStates specified either in the  **monitor_for_control** argument of
    the ControlMechanism's constructor, or the **monitor** `argument <ObjectiveMechanism_Monitor>` of the
    ObjectiveMechanism's constructor (see `ControlMechanism_ObjectiveMechanism` for details).  The
    **objective_mechanism** argument can be specified in any of the following ways:

    - *False or None* -- no ObjectiveMechanism is created and, when the ControlMechanism is added to a
      `Composition`, Projections from the OutputStates specified in the ControlMechanism's **monitor_for_control**
      argument are sent directly to ControlMechanism (see specification of **monitor_for_control** `argument
      <ControlMechanism_Monitor_for_Control_Argument>`).

    - *True* -- an `ObjectiveMechanism` is created that projects to the ControlMechanism, and any OutputStates
      specified in the ControlMechanism's **monitor_for_control** argument are assigned to ObjectiveMechanism's
      **monitor** `argument <ObjectiveMechanism_Monitor>` instead (see `ControlMechanism_ObjectiveMechanism` for
      additional details).

    - *a list of* `OutputState specifications <ObjectiveMechanism_Monitor>`; an ObjectiveMechanism is created that
      projects to the ControlMechanism, and the list of OutputStates specified, together with any specified in the
      ControlMechanism's **monitor_for_control** `argument <ControlMechanism_Monitor_for_Control_Argument>`, are
      assigned to the ObjectiveMechanism's **monitor** `argument <ObjectiveMechanism_Monitor>` (see
      `ControlMechanism_ObjectiveMechanism` for additional details).

    - *a constructor for an* `ObjectiveMechanism` -- the specified ObjectiveMechanism is created, adding any
      OutputStates specified in the ControlMechanism's **monitor_for_control** `argument
      <ControlMechanism_Monitor_for_Control_Argument>` to any specified in the ObjectiveMechanism's **monitor**
      `argument <ObjectiveMechanism_Monitor>` .  This can be used to specify the `function
      <ObjectiveMechanism.function>` used by the ObjectiveMechanism to evaluate the OutputStates monitored as well as
      how it weights those OutputStates when they are evaluated  (see `below
      <ControlMechanism_ObjectiveMechanism_Function>` for additional details).

    - *an existing* `ObjectiveMechanism` -- for any OutputStates specified in the ControlMechanism's
      **monitor_for_control** `argument <ControlMechanism_Monitor_for_Control_Argument>`, an InputState is added to the
      ObjectiveMechanism, along with `MappingProjection` to it from the specified OutputState.    This can be used to
      specify an ObjectiveMechanism with a custom `function <ObjectiveMechanism.function>` and weighting of the
      OutputStates monitored (see `below <ControlMechanism_ObjectiveMechanism_Function>` for additional details).

The OutputStates monitored by a ControlMechanism or its `objective_mechanism <ControlMechanism.objective_mechanism>`
are listed in the ControlMechanism's `monitor_for_control <ControlMechanism.monitor_for_control>` attribute
(and are the same as those listed in the `monitor <ObjectiveMechanism.monitor>` attribute of the
`objective_mechanism <ControlMechanism.objective_mechanism>`, if specified).

.. _ControlMechanism_Add_Linear_Processing_Pathway:

Note that the MappingProjections created by specification of a ControlMechanism's **monitor_for_control** `argument
<ControlMechanism_Monitor_for_Control_Argument>` or the **monitor** argument in the constructor for an
ObjectiveMechanism in the ControlMechanism's **objective_mechanism** `argument
<ControlMechanism_Objective_Mechanism_Argument>` supercede any MappingProjections that would otherwise be created for
them when included in the **pathway** argument of a Composition's `add_linear_processing_pathway
<Composition.add_linear_processing_pathway>` method.

.. _ControlMechanism_ObjectiveMechanism:

Objective Mechanism
^^^^^^^^^^^^^^^^^^^

COMMENT:
TBI FOR COMPOSITION
If the ControlMechanism is created automatically by a System (as its `controller <System.controller>`), then the
specification of OutputStates to be monitored and parameters to be controlled are made on the System and/or the
Components themselves (see `System_Control_Specification`).  In either case, the Components needed to monitor the
specified OutputStates (an `ObjectiveMechanism` and `Projections <Projection>` to it) and to control the specified
parameters (`ControlSignals <ControlSignal>` and corresponding `ControlProjections <ControlProjection>`) are created
automatically, as described below.
COMMENT

If an `ObjectiveMechanism` is specified for a ControlMechanism (in the **objective_mechanism** `argument
<ControlMechanism_Objective_Mechanism_Argument>` of its constructor; also see `ControlMechanism_Examples`),
it is assigned to the ControlMechanism's `objective_mechanism <ControlMechanism.objective_mechanism>` attribute,
and a `MappingProjection` is created automatically that projects from the ObjectiveMechanism's *OUTCOME*
`output_state <ObjectiveMechanism_Output>` to the *OUTCOME* `input_state <ControlMechanism_Input>` of the
ControlMechanism.

The `objective_mechanism <ControlMechanism.objective_mechanism>` is used to monitor the OutputStates
specified in the **monitor_for_control** `argument <ControlMechanism_Monitor_for_Control_Argument>` of the
ControlMechanism's constructor, as well as any specified in the **monitor** `argument <ObjectiveMechanism_Monitor>` of
the ObjectiveMechanism's constructor.  Specifically, for each OutputState specified in either place, an `input_state
<ObjectiveMechanism.input_states>` is added to the ObjectiveMechanism.  OutputStates to be monitored (and
corresponding `input_states <ObjectiveMechanism.input_states>`) can be added to the `objective_mechanism
<ControlMechanism.objective_mechanism>` later, by using its `add_to_monitor <ObjectiveMechanism.add_to_monitor>` method.
The set of OutputStates monitored by the `objective_mechanism <ControlMechanism.objective_mechanism>` are listed in
its `monitor <ObjectiveMechanism>` attribute, as well as in the ControlMechanism's `monitor_for_control
<ControlMechanism.monitor_for_control>` attribute.

When the ControlMechanism is added to a `Composition`, the `objective_mechanism <ControlMechanism.objective_mechanism>`
is also automatically added, and MappingProjectons are created from each of the OutputStates that it monitors to
its corresponding `input_states <ObjectiveMechanism.input_states>`.  When the Composition is run, the `value
<OutputState.value>`\\(s) of the OutputState(s) monitored are evaluated using the `objective_mechanism`\\'s `function
<ObjectiveMechanism.function>`, and the result is assigned to its *OUTCOME* `output_state
<ObjectiveMechanism_Output>`.  That `value <ObjectiveMechanism.value>` is then passed to the ControlMechanism's
*OUTCOME* `input_state <ControlMechanism_Input>`, which is used by the ControlMechanism's `function
<ControlMechanism.function>` to determine its `control_allocation <ControlMechanism.control_allocation>`.

.. _ControlMechanism_ObjectiveMechanism_Function:

If a default ObjectiveMechanism is created by the ControlMechanism (i.e., when *True* or a list of OutputStates is
specified for the **objective_mechanism** `argument <ControlMechanism_Objective_Mechanism_Argument>` of the
constructor), then the ObjectiveMechanism is created with its standard default `function <ObjectiveMechanism.function>`
(`LinearCombination`), but using *PRODUCT* (rather than the default, *SUM*) as the value of the function's `operation
<LinearCombination.operation>` parameter. The result is that the `objective_mechanism
<ControlMechanism.objective_mechanism>` multiplies the `value <OutputState.value>`\\s of the OutputStates that it
monitors, which it passes to the ControlMechanism.  However, if the **objective_mechanism** is specified using either
a constructor for, or an existing ObjectiveMechanism, then the defaults for the `ObjectiveMechanism` class -- and any
attributes explicitly specified in its construction -- are used.  In that case, if the `LinearCombination` with
*PRODUCT* as its `operation <LinearCombination.operation>` parameter are still desired, this must be explicitly
specified.  This is illustrated in the following examples.

The following example specifies a `ControlMechanism` that automatically constructs its `objective_mechanism
<ControlMechanism.objective_mechanism>`::

    >>> from psyneulink import *
    >>> my_ctl_mech = ControlMechanism(objective_mechanism=True)
    >>> assert isinstance(my_ctl_mech.objective_mechanism.function, LinearCombination)
    >>> assert my_ctl_mech.objective_mechanism.function.operation == PRODUCT

Notice that `LinearCombination` was assigned as the `function <ObjectiveMechanism.function>` of the `objective_mechanism
<ControlMechanism.objective_mechanism>`, and *PRODUCT* as its `operation <LinearCombination.operation>` parameter.

By contrast, the following example explicitly specifies the **objective_mechanism** argument using a constructor for
an ObjectiveMechanism::

    >>> my_ctl_mech = ControlMechanism(objective_mechanism=ObjectiveMechanism())
    >>> assert isinstance(my_ctl_mech.objective_mechanism.function, LinearCombination)
    >>> assert my_ctl_mech.objective_mechanism.function.operation == SUM

In this case, the defaults for the ObjectiveMechanism's class are used for its `function <ObjectiveMechanism.function>`,
which is a `LinearCombination` function with *SUM* as its `operation <LinearCombination.operation>` parameter.

Specifying the ControlMechanism's `objective_mechanism <ControlMechanism.objective_mechanism>` with a constructor
also provides greater control over how ObjectiveMechanism evaluates the OutputStates it monitors.  In addition to
specifying its `function <ObjectiveMechanism.function>`, the **monitor_weights_and_exponents** `argument
<ObjectiveMechanism_Monitor_Weights_and_Exponents>` can be used to parameterize the relative contribution made by the
monitored OutputStates when they are evaluated by that `function <ObjectiveMechanism.function>` (see
`ControlMechanism_Examples`).

COMMENT:
TBI FOR COMPOSITION
When a ControlMechanism is created for or assigned as the `controller <Composition.controller>` of a `Composition` (see
`ControlMechanism_Composition_Controller`), any OutputStates specified to be monitored by the System are assigned as
inputs to the ObjectiveMechanism.  This includes any specified in the **monitor_for_control** argument of the
System's constructor, as well as any specified in a MONITOR_FOR_CONTROL entry of a Mechanism `parameter specification
dictionary <ParameterState_Specification>` (see `Mechanism_Constructor_Arguments` and `System_Control_Specification`).

FOR DEVELOPERS:
    If the ObjectiveMechanism has not yet been created, these are added to the **monitored_output_states** of its
    constructor called by ControlMechanism._instantiate_objective_mechanmism;  otherwise, they are created using the
    ObjectiveMechanism.add_to_monitor method.
COMMENT

.. _ControlMechanism_Control_Signals:

*Specifying Parameters to Control*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This can be specified in either of two ways:

*On a ControlMechanism itself*

The parameters controlled by a ControlMechanism can be specified in the **control_signals** argument of its constructor;
the argument must be a `specification for one more ControlSignals <ControlSignal_Specification>`.  The parameter to
be controlled must belong to a Component in the same `Composition` as the ControlMechanism when it is added to the
Composition, or an error will occur.

*On a Parameter to be controlled by the `controller <Composition.controller>` of a `Composition`*

Control can also be specified for a parameter where the `parameter itself is specified <ParameterState_Specification>`,
in the constructor for the Component to which it belongs, by including a `ControlProjection`, `ControlSignal` or
the keyword `CONTROL` in a `tuple specification <ParameterState_Tuple_Specification>` for the parameter.  In this
case, the specified parameter will be assigned for control by the `controller <controller.Composition>` of any
`Composition` to which its Component belongs, when the Component is executed in that Composition (see
`ControlMechanism_Composition_Controller`).  Conversely, when a ControlMechanism is assigned as the `controller
<Composition.controller>` of a Composition, a `ControlSignal` is created and assigned to the ControlMechanism
for every parameter of any `Component <Component>` in the Composition that has been `specified for control
<ParameterState_Modulatory_Specification>`.

In general, a `ControlSignal` is created for each parameter specified to be controlled by a ControlMechanism.  These
are a type of `OutputState` that send a `ControlProjection` to the `ParameterState` of the parameter to be
controlled. All of the ControlSignals for a ControlMechanism are listed in its `control_signals
<ControlMechanism.control_signals>` attribute, and all of its ControlProjections are listed in
its`control_projections <ControlMechanism.control_projections>` attribute. Additional parameters to be controlled can
be added to a ControlMechanism by using its `assign_params` method to add a `ControlSignal` for each additional
parameter.  See `ControlMechanism_Examples`.

.. _ControlMechanism_Structure:

Structure
---------

.. _ControlMechanism_Input:

*Input*
~~~~~~~

By default, a ControlMechanism has a single (`primary <InputState_Primary>`) `input_state
<ControlMechanism.input_state>` that is named *OUTCOME*.  If the ControlMechanism has an `objective_mechanism
<ControlMechanism.objective_mechanism>`, then the *OUTCOME* `input_state <ControlMechanism.input_state>` receives a
single `MappingProjection` from the `objective_mechanism <ControlMechanism.objective_mechanism>`\\'s *OUTCOME*
OutputState (see `ControlMechanism_ObjectiveMechanism` for additional details). Otherwise, when the ControlMechanism is
added to a `Composition`, MappingProjections are created that project to the ControlMechanism's *OUTCOME* `input_state
<ControlMechanism.input_state>` from each of the OutputStates specified in the **monitor_for_control** `argument
<ControlMechanism_Monitor_for_Control_Argument>` of its constructor.  The `value <InputState.value>` of the
ControlMechanism's *OUTCOME* InputState is assigned to its `outcome <ControlMechanism.outcome>` attribute),
and is used as the input to the ControlMechanism's `function <ControlMechanism.function>` to determine its
`control_allocation <ControlMechanism.control_allocation>`.

.. _ControlMechanism_Function:

*Function*
~~~~~~~~~~

A ControlMechanism's `function <ControlMechanism.function>` uses its `outcome <ControlMechanism.outcome>`
attribute (the `value <InputState.value>` of its *OUTCOME* `InputState`) to generate a `control_allocation
<ControlMechanism.control_allocation>`.  By default, its `function <ControlMechanism.function>` is assigned
the `DefaultAllocationFunction`, which takes a single value as its input, and assigns that as the value of
each item of `control_allocation <ControlMechanism.control_allocation>`.  Each of these items is assigned as
the allocation for the corresponding  `ControlSignal` in `control_signals <ControlMechanism.control_signals>`. This
distributes the ControlMechanism's input as the allocation to each of its `control_signals
<ControlMechanism.control_signals>`.  This same behavior also applies to any custom function assigned to a
ControlMechanism that returns a 2d array with a single item in its outer dimension (axis 0).  If a function is
assigned that returns a 2d array with more than one item, and it has the same number of `control_signals
<ControlMechanism.control_signals>`, then each ControlSignal is assigned to the corresponding item of the function's
value.  However, these default behaviors can be modified by specifying that individual ControlSignals reference
different items in `control_allocation` as their `variable <ControlSignal.variable>`
(see `OutputState_Variable`).

.. _ControlMechanism_Output:

*Output*
~~~~~~~~

The OutputStates of a ControlMechanism are `ControlSignals <ControlSignal>` (listed in its `control_signals
<ControlMechanism.control_signals>` attribute). It has a `ControlSignal` for each parameter specified in the
**control_signals** argument of its constructor, that sends a `ControlProjection` to the `ParameterState` for the
corresponding parameter.  The ControlSignals are listed in the `control_signals <ControlMechanism.control_signals>`
attribute;  since they are a type of `OutputState`, they are also listed in the ControlMechanism's `output_states
<ControlMechanism.output_states>` attribute. The parameters modulated by a ControlMechanism's ControlSignals can be
displayed using its `show <ControlMechanism.show>` method. By default, each `ControlSignal` is assigned as its
`allocation <ControlSignal.allocation>` the value of the  corresponding item of the ControlMechanism's
`control_allocation <ControlMechanism.control_allocation>`;  however, subtypes of ControlMechanism may assign
allocations differently. The `default_allocation  <ControlMechanism.default_allocation>` attribute can be used to
specify a  default allocation for ControlSignals that have not been assigned their own `default_allocation
<ControlSignal.default_allocation>`. The `allocation <ControlSignal.allocation>` is used by each ControlSignal to
determine its `intensity <ControlSignal.intensity>`, which is then assigned to the `value <ControlProjection.value>`
of the ControlSignal's `ControlProjection`.   The `value <ControlProjection.value>` of the ControlProjection is used
by the `ParameterState` to which it projects to modify the value of the parameter it controls (see
`ControlSignal_Modulation` for description of how a ControlSignal modulates the value of a parameter).

.. _ControlMechanism_Costs_NetOutcome:

*Costs and Net Outcome*
~~~~~~~~~~~~~~~~~~~~~~~

A ControlMechanism's `control_signals <ControlMechanism.control_signals>` are each associated with a set of `costs
<ControlSignal_Costs>`, that are computed individually by each `ControlSignal` when they are `executed
<ControlSignal_Execution>` by the ControlMechanism.  The costs last computed by the `control_signals
<ControlMechanism>` are assigned to the ControlMechanism's `costs <ControlSignal.costs>` attribute.  A ControlMechanism
also has a set of methods -- `combine_costs <ControlMechanism.combine_costs>`, `compute_reconfiguration_cost
<ControlMechanism.compute_reconfiguration_cost>`, and `compute_net_outcome <ControlMechanism.compute_net_outcome>` --
that can be used to compute the `combined costs <ControlMechanism.combined_costs>` of its `control_signals
<ControlMechanism.control_signals>`, a `reconfiguration_cost <ControlSignal.reconfiguration_cost>` based on their change
in value, and a `net_outcome <ControlMechanism.net_outcome>` (the `value <InputState.value>` of the ControlMechanism's
*OUTCOME* `input_state <ControlMechanism_Input>` minus its `combined costs <ControlMechanism.combined_costs>`),
respectively (see `ControlMechanism_Costs_Computation` below for additional details). These methods are used by some
subclasses of ControlMechanism (e.g., `OptimizationControlMechanism`) to compute their `control_allocation
<ControlMechanism.control_allocation>`.  Each method is assigned a default function, but can be assigned a custom
functions in a corrsponding argument of the ControlMechanism's constructor (see links to attributes for details).

.. _ControlMechanism_Reconfiguration_Cost:

*Reconfiguration Cost*

A ControlMechanism's `reconfiguration_cost <ControlMechanism.reconfiguration_cost>` is distinct from the
costs of the ControlMechanism's `ControlSignals <ControlSignal>`, and in particular it is not the same as their
`adjustment_cost <ControlSignal.adjustment_cost>`.  The latter, if specified by a ControlSignal, is computed
individually by that ControlSignal using its `adjustment_cost_function <ControlSignal.adjustment_cost_function>`, based
on the change in its `intensity <ControlSignal.intensity>` from its last execution. In contrast, a ControlMechanism's
`reconfiguration_cost  <ControlMechanism.reconfiguration_cost>` is computed by its `compute_reconfiguration_cost
<ControlMechanism.compute_reconfiguration_cost>` function, based on the change in its `control_allocation
ControlMechanism.control_allocation>` from the last execution, that will be applied to *all* of its
`control_signals <ControlMechanism.control_signals>`. By default, `compute_reconfiguration_cost
<ControlMechanism.compute_reconfiguration_cost>` is assigned as the `Distance` function with the `EUCLIDEAN` metric).

.. _ControlMechanism_Execution:

Execution
---------

If a ControlMechanism is assigned as the `controller` of a `Composition`, then it is executed either before or after
all of the other  `Mechanisms <Mechanism_Base>` executed in a `TRIAL` for that Composition, depending on the
value assigned to the Composition's `controller_mode <Composition.controller_mode>` attribute (see
`Composition_Controller_Execution`).  If a ControlMechanism is added to a Composition for which it is not a
`controller <Composition.controller>`, then it executes in the same way as a `ProcessingMechanism
<ProcessingMechanism>`, based on its place in the Composition's `graph <Composition.graph>`.  Because
`ControlProjections <ControlProjection>` are likely to introduce cycles (recurrent connection loops) in the graph,
the effects of a ControlMechanism and its projections will generally not be applied in the first `TRIAL` (see
COMMENT:
FIX 8/27/19 [JDC]:
`Composition_Initial_Values_and_Feedback` and
COMMENT
**feedback** argument for the `add_projection <Composition.add_projection>` method of `Composition` for a
description of how to configure the initialization of feedback loops in a Composition; also see `Scheduler` for a
description of detailed ways in which a GatingMechanism and its dependents can be scheduled to execute).

The ControlMechanism's `function <ControlMechanism.function>` takes as its input the `value <InputState.value>` of
its *OUTCOME* `input_state <ControlMechanism.input_state>` (also contained in `outcome <ControlSignal.outcome>`).
It uses that to determine the `control_allocation <ControlMechanism.control_allocation>`, which specifies the value
assigned to the `allocation <ControlSignal.allocation>` of each of its `ControlSignals <ControlSignal>`.  Each
ControlSignal uses that value to calculate its `intensity <ControlSignal.intensity>`, as well as its `cost
<ControlSignal.cost>.  The `intensity <ControlSignal.intensity>`is used by its `ControlProjection(s)
<ControlProjection>` to modulate the value of the ParameterState(s) for the parameter(s) it controls, which are then
used in the subsequent `TRIAL` of execution.

.. note::
   `States <State>` that receive a `ControlProjection` does not update its value until its owner Mechanism
   executes (see `Lazy Evaluation <LINK>` for an explanation of "lazy" updating).  This means that even if a
   ControlMechanism has executed, a parameter that it controls will not assume its new value until the Mechanism
   to which it belongs has executed.

.. _ControlMechanism_Costs_Computation:

*Computation of Costs and Net_Outcome*

Once the ControlMechanism's `function <ControlMechanism.function>` has executed, if `compute_reconfiguration_cost
<ControlMechanism.compute_reconfiguration_cost>` has been specified, then it is used to compute the
`reconfiguration_cost <ControlMechanism.reconfiguration_cost>` for its `control_allocation
<ControlMechanism.control_allocation>` (see `above <ControlMechanism_Reconfiguration_Cost>`. After that, each
of the ControlMechanism's `control_signals <ControlMechanism.control_signals>` calculates its `cost
<ControlSignal.cost>`, based on its `intensity  <ControlSignal/intensity>`.  The ControlMechanism then combines these
with the `reconfiguration_cost <ControlMechanism.reconfiguration_cost>` using its `combine_costs
<ControlMechanism.combine_costs>` function, and the result is assigned to the `costs <ControlMechanism.costs>`
attribute.  Finally, the ControlMechanism uses this, together with its `outcome <ControlMechanism.outcome>` attribute,
to compute a `net_outcome <ControlMechanism.net_outcome>` using its `compute_net_outcome
<ControlMechanism.compute_net_outcome>` function.  This is used by some subclasses of ControlMechanism
(e.g., `OptimizationControlMechanism`) to  compute its `control_allocation <ControlMechanism.control_allocation>`
for the next `TRIAL` of execution.

.. _ControlMechanism_Examples:

Examples
--------

The following example creates a ControlMechanism by specifying its **objective_mechanism** using a constructor
that specifies the OutputStates to be monitored by its `objective_mechanism <ControlMechanism.objective_mechanism>`
and the function used to evaluated these::

    >>> my_mech_A = ProcessingMechanism(name="Mech A")
    >>> my_DDM = DDM(name="My DDM")
    >>> my_mech_B = ProcessingMechanism(function=Logistic,
    ...                                            name="Mech B")

    >>> my_control_mech = ControlMechanism(
    ...                          objective_mechanism=ObjectiveMechanism(monitor=[(my_mech_A, 2, 1),
    ...                                                                           my_DDM.output_states[RESPONSE_TIME]],
    ...                                                                     name="Objective Mechanism"),
    ...                          function=LinearCombination(operation=PRODUCT),
    ...                          control_signals=[(THRESHOLD, my_DDM),
    ...                                           (GAIN, my_mech_B)],
    ...                          name="My Control Mech")

This creates an ObjectiveMechanism for the ControlMechanism that monitors the `primary OutputState
<OutputState_Primary>` of ``my_mech_A`` and the *RESPONSE_TIME* OutputState of ``my_DDM``;  its function
first multiplies the former by 2 before, then takes product of their values and passes the result as the input to the
ControlMechanism.  The ControlMechanism's `function <ControlMechanism.function>` uses this value to determine
the allocation for its ControlSignals, that control the value of the `threshold <DDM.threshold>` parameter of
``my_DDM`` and the  `gain <Logistic.gain>` parameter of the `Logistic` Function for ``my_transfer_mech_B``.

The following example specifies the same set of OutputStates for the ObjectiveMechanism, by assigning them directly
to the **objective_mechanism** argument::

    >>> my_control_mech = ControlMechanism(
    ...                             objective_mechanism=[(my_mech_A, 2, 1),
    ...                                                  my_DDM.output_states[RESPONSE_TIME]],
    ...                             control_signals=[(THRESHOLD, my_DDM),
    ...                                              (GAIN, my_mech_B)])

Note that, while this form is more succinct, it precludes specifying the ObjectiveMechanism's function.  Therefore,
the values of the monitored OutputStates will be added (the default) rather than multiplied.

The ObjectiveMechanism can also be created on its own, and then referenced in the constructor for the ControlMechanism::

    >>> my_obj_mech = ObjectiveMechanism(monitored_output_states=[(my_mech_A, 2, 1),
    ...                                                            my_DDM.output_states[RESPONSE_TIME]],
    ...                                      function=LinearCombination(operation=PRODUCT))

    >>> my_control_mech = ControlMechanism(
    ...                        objective_mechanism=my_obj_mech,
    ...                        control_signals=[(THRESHOLD, my_DDM),
    ...                                         (GAIN, my_mech_B)])

Here, as in the first example, the constructor for the ObjectiveMechanism can be used to specify its function, as well
as the OutputState that it monitors.

COMMENT:
FIX 8/27/19 [JDC]:  ADD TO COMPOSITION
See `System_Control_Examples` for examples of how a ControlMechanism, the OutputStates its
`objective_mechanism <ControlSignal.objective_mechanism>`, and its `control_signals <ControlMechanism.control_signals>`
can be specified for a System.
COMMENT

.. _ControlMechanism_Class_Reference:

Class Reference
---------------

"""
import numpy as np
import typecheck as tc
import warnings

from psyneulink.core.components.functions.function import is_function_type
from psyneulink.core.components.mechanisms.adaptive.modulatorymechanism import ModulatoryMechanism
from psyneulink.core.components.mechanisms.mechanism import Mechanism, Mechanism_Base
from psyneulink.core.components.shellclasses import Composition_Base, System_Base
from psyneulink.core.components.states.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.components.states.outputstate import OutputState
from psyneulink.core.components.states.parameterstate import ParameterState
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.defaults import defaultControlAllocation
from psyneulink.core.globals.keywords import \
    CONTROL, CONTROL_PROJECTION, CONTROL_SIGNAL, CONTROL_SIGNALS, \
    INIT_EXECUTE_METHOD_ONLY, MULTIPLICATIVE, PROJECTION_TYPE
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import ContentAddressableList, is_iterable

__all__ = [
    'ControlMechanism', 'ControlMechanismError', 'ControlMechanismRegistry'
]


ControlMechanismRegistry = {}

def _is_control_spec(spec):
    from psyneulink.core.components.projections.modulatory.controlprojection import ControlProjection
    from psyneulink.core.components.mechanisms.adaptive.modulatorymechanism import ModulatoryMechanism
    if isinstance(spec, tuple):
        return any(_is_control_spec(item) for item in spec)
    if isinstance(spec, dict) and PROJECTION_TYPE in spec:
        return _is_control_spec(spec[PROJECTION_TYPE])
    elif isinstance(spec, (ControlMechanism,
                           ControlSignal,
                           ControlProjection,
                           ModulatoryMechanism)):
        return True
    elif isinstance(spec, type) and issubclass(spec, (ControlMechanism,
                                                      ControlSignal,
                                                      ControlProjection,
                                                      ModulatoryMechanism)):
        return True
    elif isinstance(spec, str) and spec in {CONTROL, CONTROL_PROJECTION, CONTROL_SIGNAL}:
        return True
    else:
        return False


class ControlMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


def _control_allocation_getter(owning_component=None, context=None):
    return owning_component.parameters.modulatory_allocation._get(context)

def _control_allocation_setter(value, owning_component=None, context=None):
    owning_component.parameters.modulatory_allocation._set(np.array(value), context)
    return value

def _gating_allocation_getter(owning_component=None, context=None):
    from psyneulink.core.components.mechanisms.adaptive.gating import GatingMechanism
    from psyneulink.core.components.states.modulatorysignals.gatingsignal import GatingSignal
    raise ControlMechanismError(f"'gating_allocation' attribute is not implemented on {owning_component.name};  "
                                f"consider using a {GatingMechanism.__name__} instead, "
                                f"or a {ModulatoryMechanism.__name__} if both {ControlSignal.__name__}s and "
                                f"{GatingSignal.__name__}s are needed.")


def _gating_allocation_setter(value, owning_component=None, context=None, **kwargs):
    from psyneulink.core.components.mechanisms.adaptive.gating import GatingMechanism
    from psyneulink.core.components.states.modulatorysignals.gatingsignal import GatingSignal
    raise ControlMechanismError(f"'gating_allocation' attribute is not implemented on {owning_component.name};  "
                                f"consider using a {GatingMechanism.__name__} instead, "
                                f"or a {ModulatoryMechanism.__name__} if both {ControlSignal.__name__}s and "
                                f"{GatingSignal.__name__}s are needed.")


class ControlMechanism(ModulatoryMechanism):
    """
    ControlMechanism(                               \
        system=None,                                \
        monitor_for_control=None,                   \
        objective_mechanism=None,                   \
        function=Linear,                            \
        default_allocation=None,                    \
        control_signals=None,                       \
        modulation=MULTIPLICATIVE,                  \
        combine_costs=np.sum,                       \
        compute_reconfiguration_cost=None,          \
        compute_net_outcome=lambda x,y:x-y,         \
        params=None,                                \
        name=None,                                  \
        prefs=None)

    Subclass of `AdaptiveMechanism <AdaptiveMechanism>` that modulates the parameter(s)
    of one or more `Component(s) <Component>`.


    COMMENT:
    .. note::
       ControlMechanism is an abstract class and should NEVER be instantiated by a direct call to its constructor.
       It should be instantiated using the constructor for a `subclass <ControlMechanism_Subtypes>`.

        Description:
            Protocol for instantiating unassigned ControlProjections (i.e., w/o a sender specified):
               If sender is not specified for a ControlProjection (e.g., in a parameter specification tuple)
                   it is flagged for deferred_init() in its __init__ method
               If ControlMechanism is instantiated or assigned as the controller for a System:
                   the System calls its _get_monitored_output_states() method which returns all of the OutputStates
                       within the System that have been specified to be MONITORED_FOR_CONTROL, and then assigns
                       them (along with any specified in the **monitored_for_control** arg of the System's constructor)
                       to the `objective_mechanism` argument of the ControlMechanism's constructor;
                   the System calls its _get_control_signals_for_system() method which returns all of the parameters
                       that have been specified for control within the System, assigns them a ControlSignal
                       (with a ControlProjection to the ParameterState for the parameter), and assigns the
                       ControlSignals (alogn with any specified in the **control_signals** argument of the System's
                       constructor) to the **control_signals** argument of the ControlMechanism's constructor

            OBJECTIVE_MECHANISM param determines which States will be monitored.
                specifies the OutputStates of the terminal Mechanisms in the System to be monitored by ControlMechanism
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
        specifies the `System` to which the ControlMechanism should be assigned as its `controller
        <System.controller>`.

    monitor_for_control : List[OutputState or Mechanism] : default None
        specifies the `OutputStates <OutputState>` to be monitored by the `ObjectiveMechanism` specified in the
        **objective_mechanism** argument; if any specification is a Mechanism (rather than its OutputState),
        its `primary OutputState <OutputState_Primary>` is used (see `ControlMechanism_Monitor_for_Control` for
        additional details).

    objective_mechanism : ObjectiveMechanism or List[OutputState specification] : default None
        specifies either an `ObjectiveMechanism` to use for the ControlMechanism, or a list of the OutputStates it
        should monitor; if a list of `OutputState specifications <ObjectiveMechanism_Monitor>` is used,
        a default ObjectiveMechanism is created and the list is passed to its **monitor** argument, along with any
        OutputStates specified in the ControlMechanism's **monitor_for_control** `argument
        <ControlMechanism_Monitor_for_Control_Argument>`.

    function : TransferFunction : default Linear(slope=1, intercept=0)
        specifies function used to combine values of monitored OutputStates.

    default_allocation : number, list or 1d array : None
        specifies the default_allocation of any `control_signals <ControlMechanism.control.signals>` for
        which the **default_allocation** was not specified in its constructor (see default_allocation
        <ControlMechanism.default_allocation>` for additional details).

    control_signals : ControlSignal specification or List[ControlSignal specification, ...]
        specifies the parameters to be controlled by the ControlMechanism; a `ControlSignal` is created for each
        (see `ControlSignal_Specification` for details of specification).

    modulation : ModulationParam : MULTIPLICATIVE
        specifies the default form of modulation used by the ControlMechanism's `ControlSignals <ControlSignal>`,
        unless they are `individually specified <ControlSignal_Specification>`.

    combine_costs : Function, function or method : default np.sum
        specifies function used to combine the `cost <ControlSignal.cost>` of the ControlMechanism's `control_signals
        <ControlMechanism.control_signals>`;  must take a list or 1d array of scalar values as its argument and
        return a list or array with a single scalar value.

    compute_reconfiguration_cost : Function, function or method : default None
        specifies function used to compute the ControlMechanism's `reconfiguration_cost
        <ControlMechanism.reconfiguration_cost>`; must take a list or 2d array containing two lists or 1d arrays,
        both with the same shape as the ControlMechanism's control_allocation attribute, and return a scalar value.

    compute_net_outcome : Function, function or method : default lambda outcome, cost: outcome-cost
        function used to combine the values of its `outcome <ControlMechanism.outcome>` and `costs
        <ControlMechanism.costs>` attributes;  must take two 1d arrays (outcome and cost) with scalar values as its
        arguments and return an array with a single scalar value.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters
        for the Mechanism, parameters for its function, and/or a custom function and its parameters. Values
        specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default see `name <ControlMechanism.name>`
        specifies the name of the ControlMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the ControlMechanism; see `prefs <ControlMechanism.prefs>` for details.

    Attributes
    ----------

    system : System_Base
        The `System` for which the ControlMechanism is a `controller <System>`.  Note that this is distinct from
        a Mechanism's `systems <Mechanism_Base.systems>` attribute, which lists all of the Systems to which a
        `Mechanism` belongs -- a ControlMechanism can belong to but not be the `controller of a System
        <ControlMechanism_Composition_Controller>`.

    objective_mechanism : ObjectiveMechanism
        `ObjectiveMechanism` that monitors and evaluates the values specified in the ControlMechanism's
        **objective_mechanism** argument, and transmits the result to the ControlMechanism's *OUTCOME*
        `input_state <Mechanism_Base.input_state>`.

    monitor_for_control : List[OutputState]
        each item is an `OutputState` monitored by the ControlMechanism or its `objective_mechanism
        <ControlMechanism.objective_mechanism>` if that is specified (see `ControlMechanism_Monitor_for_Control`);
        in the latter case, the list returned is ObjectiveMechanism's `monitor <ObjectiveMechanism.monitor>` attribute.

    monitored_output_states_weights_and_exponents : List[Tuple(float, float)]
        each tuple in the list contains the weight and exponent associated with a corresponding OutputState specified
        in `monitor_for_control <ControlMechanism.monitor_for_control>`; if `objective_mechanism
        <ControlMechanism.objective_mechanism>` is specified, these are the same as those in the ObjectiveMechanism's
        `monitored_output_states_weights_and_exponents
        <ObjectiveMechanism.monitored_output_states_weights_and_exponents>` attribute, and are used by the
        ObjectiveMechanism's `function <ObjectiveMechanism.function>` to parametrize the contribution made to its
        output by each of the values that it monitors (see `ObjectiveMechanism Function <ObjectiveMechanism_Function>`).

    input_state : InputState
        the ControlMechanism's `primary InputState <InputState_Primary>`, named *OUTCOME*;  this receives a
        `MappingProjection` from the *OUTCOME* `OutputState <ObjectiveMechanism_Output>` of `objective_mechanism
        <ControlMechanism.objective_mechanism>` if that is specified; otherwise, it receives MappingProjections
        from each of the OutputStates specifed in `monitor_for_control <ControlMechanism.monitor_for_control>`
        (see `_ControlMechanism_Input` for additional details).

    outcome : 1d array
        the `value <InputState.value>` of the ControlMechanism's *OUTCOME* `input_state <ControlMechanism.input_state>`.

    function : TransferFunction : default Linear(slope=1, intercept=0)
        determines how the `value <OuputState.value>` \\s of the `OutputStates <OutputState>` specified in the
        **monitor_for_control** `argument <ControlMechanism_Monitor_for_Control_Argument>` of the ControlMechanism's
        constructor are used to generate its `control_allocation <ControlMechanism.control_allocation>`.

    default_allocation : number, list or 1d array
        determines the default_allocation of any `control_signals <ControlMechanism.control.signals>` for
        which the **default_allocation** was not specified in its constructor;  if it is None (not specified)
        then the ControlSignal's parameters.allocation.default_value is used. See documentation for
        **default_allocation** argument of ControlSignal constructor for additional details.

    control_allocation : 2d array
        each item is the value assigned as the `allocation <ControlSignal.allocation>` for the corresponding
        ControlSignal listed in the `control_signals` attribute;  the control_allocation is the same as the
        ControlMechanism's `value <Mechanism_Base.value>` attribute).

    control_signals : ContentAddressableList[ControlSignal]
        list of the `ControlSignals <ControlSignals>` for the ControlMechanism, including any inherited from a
        `Composition` for which it is a `controller <Composition.controller>` (same as ControlMechanism's
        `output_states <Mechanism_Base.output_states>` attribute); each sends a `ControlProjection`
        to the `ParameterState` for the parameter it controls

    compute_reconfiguration_cost : Function, function or method
        function used to compute the ControlMechanism's `reconfiguration_cost  <ControlMechanism.reconfiguration_cost>`;
        result is a scalar value representing the difference — defined by the function — between the values of the
        ControlMechanism's current and last `control_alloction <ControlMechanism.control_allocation>`, that can be
        accessed by `reconfiguration_cost <ControlMechanism.reconfiguration_cost>` attribute.

    reconfiguration_cost : scalar
        result of `compute_reconfiguration_cost <ControlMechanism.compute_reconfiguration_cost>` function, that
        computes the difference between the values of the ControlMechanism's current and last `control_alloction
        <ControlMechanism.control_allocation>`; value is None and is ignored if `compute_reconfiguration_cost
        <ControlMechanism.compute_reconfiguration_cost>` has not been specified.

        .. note::
        A ControlMechanism's reconfiguration_cost is not the same as the `adjustment_cost
        <ControlSignal.adjustment_cost>` of its ControlSignals (see `Reconfiguration Cost
        <ControlMechanism_Reconfiguration_Cost>` for additional details).

    costs : list
        current costs for the ControlMechanism's `control_signals <ControlMechanism.control_signals>`, computed
        for each using its `compute_costs <ControlSignals.compute_costs>` method.

    combine_costs : Function, function or method
        function used to combine the `cost <ControlSignal.cost>` of its `control_signals
        <ControlMechanism.control_signals>`; result is an array with a scalar value that can be accessed by
        `combined_costs <ControlMechanism.combined_costs>`.

        .. note::
          This function is distinct from the `combine_costs_function <ControlSignal.combine_costs_function>` of a
          `ControlSignal`.  The latter combines the different `costs <ControlSignal_Costs>` for an individual
          ControlSignal to yield its overall `cost <ControlSignal.cost>`; the ControlMechanism's
          `combine_costs <ControlMechanism.combine_costs>` function combines those `cost <ControlSignal.cost>`\\s
          for its `control_signals <ControlMechanism.control_signals>`.

    combined_costs : 1d array
        result of the ControlMechanism's `combine_costs <ControlMechanism.combine_costs>` function.

    compute_net_outcome : Function, function or method
        function used to combine the values of its `outcome <ControlMechanism.outcome>` and `costs
        <ControlMechanism.costs>` attributes;  result is an array with a scalar value that can be accessed
        by the the `net_outcome <ControlMechanism.net_outcome>` attribute.

    net_outcome : 1d array
        result of the ControlMechanism's `compute_net_outcome <ControlMechanism.compute_net_outcome>` function.

    control_projections : List[ControlProjection]
        list of `ControlProjections <ControlProjection>`, one for each `ControlSignal` in `control_signals`.

    modulation : ModulationParam
        the default form of modulation used by the ControlMechanism's `ControlSignals <GatingSignal>`,
        unless they are `individually specified <ControlSignal_Specification>`.

    name : str
        the name of the ControlMechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the ControlMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentType = "ControlMechanism"

    initMethod = INIT_EXECUTE_METHOD_ONLY

    outputStateTypes = ControlSignal
    stateListAttr = ModulatoryMechanism.stateListAttr.copy()
    stateListAttr.update({ControlSignal:CONTROL_SIGNALS})

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'ControlMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    # Override control_allocation and suppress gating_allocation
    class Parameters(ModulatoryMechanism.Parameters):
        """
            Attributes
            ----------

                value
                    see `value <ControlMechanism.value>`

                    :default value: numpy.array([1.])
                    :type: numpy.ndarray

                control_allocation
                    see `control_allocation <ControlMechanism.control_allocation>`

                    :default value: numpy.array([1.])
                    :type: numpy.ndarray
                    :read only: True

                gating_allocation
                    see `gating_allocation <ControlMechanism.gating_allocation>`

                    :default value: NotImplemented
                    :type: <class 'NotImplementedType'>
                    :read only: True

        """
        # This must be a list, as there may be more than one (e.g., one per control_signal)
        value = Parameter(np.array([defaultControlAllocation]), aliases='modulatory_allocation')
        control_allocation = Parameter(np.array([defaultControlAllocation]),
                                      getter=_control_allocation_getter,
                                      setter=_control_allocation_setter,
                                      read_only=True)

        gating_allocation = Parameter(NotImplemented,
                                      getter=_gating_allocation_getter,
                                      setter=_gating_allocation_setter,
                                      read_only=True)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 system:tc.optional(tc.any(System_Base, Composition_Base))=None,
                 monitor_for_control:tc.optional(tc.any(is_iterable, Mechanism, OutputState))=None,
                 objective_mechanism=None,
                 function=None,
                 default_allocation:tc.optional(tc.any(int, float, list, np.ndarray))=None,
                 control_signals:tc.optional(tc.any(is_iterable, ParameterState, ControlSignal))=None,
                 modulation:tc.optional(str)=MULTIPLICATIVE,
                 combine_costs:is_function_type=np.sum,
                 compute_reconfiguration_cost:tc.optional(is_function_type)=None,
                 compute_net_outcome:is_function_type=lambda outcome, cost : outcome - cost,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs
                 ):

        # if kwargs:
        #         for i in kwargs.keys():
        #             raise ControlMechanismError("Unrecognized arg in constructor for {}: {}".
        #                                         format(self.__class__.__name__, repr(i)))

        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(system=system,
                                                  params=params)

        # FIX: REMOVE system ARGUMENT
        super(ControlMechanism, self).__init__(system=system,
                                               default_variable=default_variable,
                                               size=size,
                                               monitor_for_modulation=monitor_for_control,
                                               objective_mechanism=objective_mechanism,
                                               function=function,
                                               default_allocation=default_allocation,
                                               combine_costs=combine_costs,
                                               compute_reconfiguration_cost=compute_reconfiguration_cost,
                                               compute_net_outcome=compute_net_outcome,
                                               modulatory_signals=control_signals,
                                               modulation=modulation,
                                               params=params,
                                               name=name,
                                               prefs=prefs,

                                               **kwargs)

    def _instantiate_output_states(self, context=None):
        self._register_modulatory_signal_type(ControlSignal,context)
        super()._instantiate_output_states(context)

    def _instantiate_control_signal(self, control_signal, context):
        return super()._instantiate_modulatory_signal(modulatory_signal=control_signal, context=context)

    # FIX: TBI FOR COMPOSITION
    @handle_external_context()
    @tc.typecheck
    def assign_as_controller(self, system:System_Base, context=None):
        """Assign ControlMechanism as `controller <System.controller>` for a `System`.

        **system** must be a System for which the ControlMechanism should be assigned as the `controller
        <System.controller>`.
        If the specified System already has a `controller <System.controller>`, it will be replaced by the current
        one, and the current one will inherit any ControlSignals previously specified for the old controller or the
        System itself.
        If the current one is already the `controller <System.controller>` for another System, it will be disabled
        for that System.
        COMMENT:
            [TBI:
            The ControlMechanism's `objective_mechanism <ControlMechanism.objective_mechanism>`,
            `monitored_output_states` and `control_signal <ControlMechanism.control_signals>` attributes will also be
            updated to remove any assignments that are not part of the new System, and add any that are specified for
            the new System.]
        COMMENT

        COMMENT:
            IMPLEMENTATION NOTE:  This is handled as a method on ControlMechanism (rather than System) so that:

                                  - [TBI: if necessary, it can detach itself from a System for which it is already the
                                    `controller <System.controller>`;]

                                  - any class-specific actions that must be taken to instantiate the ControlMechanism
                                    can be handled by subclasses of ControlMechanism (e.g., an EVCControlMechanism must
                                    instantiate its Prediction Mechanisms). However, the actual assignment of the
                                    ControlMechanism the System's `controller <System.controller>` attribute must
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
        # Assign the current System to the ControlMechanism

        # Validate that all of the ControlMechanism's monitored_output_states and controlled parameters
        #    are in the new System
        system._validate_monitored_states_in_system(self.monitored_output_states)
        system._validate_control_signals(self.control_signals)

        # Get any and all OutputStates specified in:
        # - **monitored_output_states** argument of the System's constructor
        # - in a MONITOR_FOR_CONTROL specification for individual OutputStates and/or Mechanisms
        # - already being montiored by the ControlMechanism being assigned
        monitored_output_states = list(system._get_monitored_output_states_for_system(controller=self, context=context))

        # Don't add any OutputStates that are already being monitored by the ControlMechanism's ObjectiveMechanism
        for monitored_output_state in monitored_output_states.copy():
            if monitored_output_state.output_state in self.monitored_output_states:
                monitored_output_states.remove(monitored_output_state)

        # Add all other monitored_output_states to the ControlMechanism's monitored_output_states attribute
        #    and to its ObjectiveMechanisms monitored_output_states attribute
        if monitored_output_states:
            self.add_to_monitor(monitored_output_states, context=context)

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

        # Add any ControlSignals specified for System
        for control_signal_spec in system_control_signals:
            control_signal = self._instantiate_control_signal(control_signal=control_signal_spec, context=context)
            if not control_signal:
                continue
            # FIX: 1/18/18 - CHECK FOR SAME NAME IN _instantiate_control_signal
            # # Don't add any that are already on the ControlMechanism
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

        # Assign assign the current System to the ControlMechanism's system attribute
        #    (needed for it to validate and instantiate monitored_output_states and control_signals)
        self.system = system

        # Flag ObjectiveMechanism as associated with a ControlMechanism that is a controller for the System
        self.objective_mechanism.for_controller = True

        if context.source != ContextFlags.PROPERTY:
            system._controller = self

        self._activate_projections_for_compositions(system)

    def _apply_control_allocation(self, control_allocation, runtime_params, context):
        self._apply_modulatory_allocation(modulatory_allocation=control_allocation,
                                          runtime_params=runtime_params,
                                          context=context)

    # Override control_signals
    @property
    def control_signals(self):
        try:
            return ContentAddressableList(component_type=ControlSignal,
                                          list=[state for state in self.output_states
                                                if isinstance(state, ControlSignal)])
        except:
            return None

    @control_signals.setter
    def control_signals(self, value):
        self._modulatory_signals = value

    # Suppress gating_signals
    @property
    def gating_signals(self):
        from psyneulink.core.components.mechanisms.adaptive.gating import GatingMechanism
        from psyneulink.core.components.states.modulatorysignals.gatingsignal import GatingSignal
        raise ControlMechanismError(f"'gating_signals' attribute is not implemented on {self.name} (a "
                                    f"{self.__class__.__name__}); consider using a {GatingMechanism.__name__} instead, "
                                    f"or a {ModulatoryMechanism.__name__} if both {ControlSignal.__name__}s and "
                                    f"{GatingSignal.__name__}s are needed.")

    @gating_signals.setter
    def gating_signals(self, value):
        from psyneulink.core.components.mechanisms.adaptive.gating import GatingMechanism
        from psyneulink.core.components.states.modulatorysignals.gatingsignal import GatingSignal
        raise ControlMechanismError(f"'gating_signals' attribute is not implemented on {self.name} (a "
                                    f"{self.__class__.__name__}); consider using a {GatingMechanism.__name__} instead, "
                                    f"or a {ModulatoryMechanism.__name__} if both {ControlSignal.__name__}s and "
                                    f"{GatingSignal.__name__}s are needed.")
