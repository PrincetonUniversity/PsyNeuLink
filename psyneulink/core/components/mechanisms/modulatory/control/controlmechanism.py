# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# **************************************  ControlMechanism ************************************************

"""

Contents
--------

  * `ControlMechanism_Overview`
  * `ControlMechanism_Composition_Controller`
  * `ControlMechanism_Creation`
      - `ControlMechanism_Monitor_for_Control`
      - `ControlMechanism_ObjectiveMechanism`
      - `ControlMechanism_ControlSignals`
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
more `Ports <Port>` of other Mechanisms in the `Composition` to which it belongs. In general, a ControlMechanism is
used to modulate the `ParameterPort(s) <ParameterPort>` of one or more Mechanisms, that determine the value(s) of
the parameter(s) of the `function(s) <Mechanism_Base.function>` of those Mechanism(s). However, a ControlMechanism
can also be used to modulate the function of `InputPorts <InputPort>` and/or `OutputPort <OutputPorts>`,
much like a `GatingMechanism`.  A ControlMechanism's `function <ControlMechanism.function>` calculates a
`control_allocation <ControlMechanism.control_allocation>`: a list of values provided to each of its `control_signals
<ControlMechanism.control_signals>`.  Its control_signals are `ControlSignal` OutputPorts that are used to modulate
the parameters of other Mechanisms' `function <Mechanism_Base.function>` (see `ControlSignal_Modulation` for a more
detailed description of how modulation operates).  A ControlMechanism can be configured to monitor the outputs of
other Mechanisms in order to determine its `control_allocation <ControlMechanism.control_allocation>`, by specifying
these in the **monitor_for_control** `argument <ControlMechanism_Monitor_for_Control_Argument>` of its constructor,
or in the **monitor** `argument <ObjectiveMechanism_Monitor>` of an ObjectiveMechanism` assigned to its
**objective_mechanism** `argument <ControlMechanism_Objective_Mechanism_Argument>` (see `ControlMechanism_Creation`
below).  A ControlMechanism can also be assigned as the `controller <Composition.controller>` of a `Composition`,
which has a special relation to that Composition: it generally executes either before or after all of the other
Mechanisms in that Composition (see `Composition_Controller_Execution`).  The OutputPorts monitored by the
ControlMechanism or its `objective_mechanism <ControlMechanism.objective_mechanism>`, and the parameters it modulates
can be listed using its `show <ControlMechanism.show>` method.

.. _ControlMechanism_Composition_Controller:

*ControlMechanisms and a Composition*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ControlMechanism can be assigned to a `Composition` and executed just like any other Mechanism. It can also be
assigned as the `controller <Composition.controller>` of a `Composition`, that has a special relation
to the Composition: it is used to control all of the parameters that have been `specified for control
<ControlMechanism_ControlSignals>` in that Composition.  A ControlMechanism can be the `controller
<Composition.controller>` for only one Composition, and a Composition can have only one `controller
<Composition.controller>`.  When a ControlMechanism is assigned as the `controller <Composition.controller>` of a
Composition (either in the Composition's constructor, or using its `add_controller <Composition.add_controller>`
method, the ControlMechanism assumes control over all of the parameters that have been `specified for control
<ControlMechanism_ControlSignals>` for Components in the Composition.  The Composition's `controller
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

A ControlMechanism is created by calling its constructor.  When a ControlMechanism is created, the OutputPorts it
monitors and the Ports it modulates can be specified in the **montior_for_control** and **objective_mechanism**
arguments of its constructor, respectively.  Each can be specified in several ways, as described below. If neither of
those arguments is specified, then only the ControlMechanism is constructed, and its inputs and the parameters it
modulates must be specified in some other way.

.. _ControlMechanism_Monitor_for_Control:

*Specifying OutputPorts to be monitored*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ControlMechanism can be configured to monitor the output of other Mechanisms directly (by receiving direct
Projections from their OutputPorts), or by way of an `ObjectiveMechanism` that evaluates those outputs and passes the
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
(see `Composition_Cycles_and_Feedback`;  also, see `below <ControlMechanism_Add_Linear_Processing_Pathway>`
regarding specification of a ControlMechanism and associated ObjectiveMechanism in a Composition's
`add_linear_processing_pathway <Composition.add_linear_processing_pathway>` method).

Which configuration is used is determined by how the following arguments of the ControlMechanism's constructor are
specified (also see `ControlMechanism_Examples`):

  .. _ControlMechanism_Monitor_for_Control_Argument:

  * **monitor_for_control** -- a list of `OutputPort specifications <OutputPort_Specification>`.  If the
    **objective_mechanism** argument is not specified (or is *False* or *None*) then, when the ControlMechanism is
    added to a `Composition`, a `MappingProjection` is created for each OutputPort specified to the ControlMechanism's
    *OUTCOME* `input_port <ControlMechanism_Input>`.  If the **objective_mechanism** `argument
    <ControlMechanism_Objective_Mechanism_Argument>` is specified, then the OutputPorts specified in
    **monitor_for_control** are assigned to the `ObjectiveMechanism` rather than the ControlMechanism itself (see
    `ControlMechanism_ObjectiveMechanism` for details).

  .. _ControlMechanism_Objective_Mechanism_Argument:

  * **objective_mechanism** -- if this is specfied in any way other than **False** or **None** (the default),
    then an ObjectiveMechanism is created that projects to the ControlMechanism and, when added to a `Composition`,
    is assigned Projections from all of the OutputPorts specified either in the  **monitor_for_control** argument of
    the ControlMechanism's constructor, or the **monitor** `argument <ObjectiveMechanism_Monitor>` of the
    ObjectiveMechanism's constructor (see `ControlMechanism_ObjectiveMechanism` for details).  The
    **objective_mechanism** argument can be specified in any of the following ways:

    - *False or None* -- no ObjectiveMechanism is created and, when the ControlMechanism is added to a
      `Composition`, Projections from the OutputPorts specified in the ControlMechanism's **monitor_for_control**
      argument are sent directly to ControlMechanism (see specification of **monitor_for_control** `argument
      <ControlMechanism_Monitor_for_Control_Argument>`).

    - *True* -- an `ObjectiveMechanism` is created that projects to the ControlMechanism, and any OutputPorts
      specified in the ControlMechanism's **monitor_for_control** argument are assigned to ObjectiveMechanism's
      **monitor** `argument <ObjectiveMechanism_Monitor>` instead (see `ControlMechanism_ObjectiveMechanism` for
      additional details).

    - *a list of* `OutputPort specifications <ObjectiveMechanism_Monitor>`; an ObjectiveMechanism is created that
      projects to the ControlMechanism, and the list of OutputPorts specified, together with any specified in the
      ControlMechanism's **monitor_for_control** `argument <ControlMechanism_Monitor_for_Control_Argument>`, are
      assigned to the ObjectiveMechanism's **monitor** `argument <ObjectiveMechanism_Monitor>` (see
      `ControlMechanism_ObjectiveMechanism` for additional details).

    - *a constructor for an* `ObjectiveMechanism` -- the specified ObjectiveMechanism is created, adding any
      OutputPorts specified in the ControlMechanism's **monitor_for_control** `argument
      <ControlMechanism_Monitor_for_Control_Argument>` to any specified in the ObjectiveMechanism's **monitor**
      `argument <ObjectiveMechanism_Monitor>` .  This can be used to specify the `function
      <ObjectiveMechanism.function>` used by the ObjectiveMechanism to evaluate the OutputPorts monitored as well as
      how it weights those OutputPorts when they are evaluated  (see `below
      <ControlMechanism_ObjectiveMechanism_Function>` for additional details).

    - *an existing* `ObjectiveMechanism` -- for any OutputPorts specified in the ControlMechanism's
      **monitor_for_control** `argument <ControlMechanism_Monitor_for_Control_Argument>`, an InputPort is added to the
      ObjectiveMechanism, along with `MappingProjection` to it from the specified OutputPort.    This can be used to
      specify an ObjectiveMechanism with a custom `function <ObjectiveMechanism.function>` and weighting of the
      OutputPorts monitored (see `below <ControlMechanism_ObjectiveMechanism_Function>` for additional details).

The OutputPorts monitored by a ControlMechanism or its `objective_mechanism <ControlMechanism.objective_mechanism>`
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
specification of OutputPorts to be monitored and parameters to be controlled are made on the System and/or the
Components themselves (see `System_Control_Specification`).  In either case, the Components needed to monitor the
specified OutputPorts (an `ObjectiveMechanism` and `Projections <Projection>` to it) and to control the specified
parameters (`ControlSignals <ControlSignal>` and corresponding `ControlProjections <ControlProjection>`) are created
automatically, as described below.
COMMENT

If an `ObjectiveMechanism` is specified for a ControlMechanism (in the **objective_mechanism** `argument
<ControlMechanism_Objective_Mechanism_Argument>` of its constructor; also see `ControlMechanism_Examples`),
it is assigned to the ControlMechanism's `objective_mechanism <ControlMechanism.objective_mechanism>` attribute,
and a `MappingProjection` is created automatically that projects from the ObjectiveMechanism's *OUTCOME*
`output_port <ObjectiveMechanism_Output>` to the *OUTCOME* `input_port <ControlMechanism_Input>` of the
ControlMechanism.

The `objective_mechanism <ControlMechanism.objective_mechanism>` is used to monitor the OutputPorts
specified in the **monitor_for_control** `argument <ControlMechanism_Monitor_for_Control_Argument>` of the
ControlMechanism's constructor, as well as any specified in the **monitor** `argument <ObjectiveMechanism_Monitor>` of
the ObjectiveMechanism's constructor.  Specifically, for each OutputPort specified in either place, an `input_port
<ObjectiveMechanism.input_ports>` is added to the ObjectiveMechanism.  OutputPorts to be monitored (and
corresponding `input_ports <ObjectiveMechanism.input_ports>`) can be added to the `objective_mechanism
<ControlMechanism.objective_mechanism>` later, by using its `add_to_monitor <ObjectiveMechanism.add_to_monitor>` method.
The set of OutputPorts monitored by the `objective_mechanism <ControlMechanism.objective_mechanism>` are listed in
its `monitor <ObjectiveMechanism>` attribute, as well as in the ControlMechanism's `monitor_for_control
<ControlMechanism.monitor_for_control>` attribute.

When the ControlMechanism is added to a `Composition`, the `objective_mechanism <ControlMechanism.objective_mechanism>`
is also automatically added, and MappingProjectons are created from each of the OutputPorts that it monitors to
its corresponding `input_ports <ObjectiveMechanism.input_ports>`.  When the Composition is run, the `value
<OutputPort.value>`\\(s) of the OutputPort(s) monitored are evaluated using the `objective_mechanism`\\'s `function
<ObjectiveMechanism.function>`, and the result is assigned to its *OUTCOME* `output_port
<ObjectiveMechanism_Output>`.  That `value <ObjectiveMechanism.value>` is then passed to the ControlMechanism's
*OUTCOME* `input_port <ControlMechanism_Input>`, which is used by the ControlMechanism's `function
<ControlMechanism.function>` to determine its `control_allocation <ControlMechanism.control_allocation>`.

.. _ControlMechanism_ObjectiveMechanism_Function:

If a default ObjectiveMechanism is created by the ControlMechanism (i.e., when *True* or a list of OutputPorts is
specified for the **objective_mechanism** `argument <ControlMechanism_Objective_Mechanism_Argument>` of the
constructor), then the ObjectiveMechanism is created with its standard default `function <ObjectiveMechanism.function>`
(`LinearCombination`), but using *PRODUCT* (rather than the default, *SUM*) as the value of the function's `operation
<LinearCombination.operation>` parameter. The result is that the `objective_mechanism
<ControlMechanism.objective_mechanism>` multiplies the `value <OutputPort.value>`\\s of the OutputPorts that it
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
also provides greater control over how ObjectiveMechanism evaluates the OutputPorts it monitors.  In addition to
specifying its `function <ObjectiveMechanism.function>`, the **monitor_weights_and_exponents** `argument
<ObjectiveMechanism_Monitor_Weights_and_Exponents>` can be used to parameterize the relative contribution made by the
monitored OutputPorts when they are evaluated by that `function <ObjectiveMechanism.function>` (see
`ControlMechanism_Examples`).

COMMENT:
TBI FOR COMPOSITION
When a ControlMechanism is created for or assigned as the `controller <Composition.controller>` of a `Composition` (see
`ControlMechanism_Composition_Controller`), any OutputPorts specified to be monitored by the System are assigned as
inputs to the ObjectiveMechanism.  This includes any specified in the **monitor_for_control** argument of the
System's constructor, as well as any specified in a MONITOR_FOR_CONTROL entry of a Mechanism `parameter specification
dictionary <ParameterPort_Specification>` (see `Mechanism_Constructor_Arguments` and `System_Control_Specification`).

FOR DEVELOPERS:
    If the ObjectiveMechanism has not yet been created, these are added to the **monitored_output_ports** of its
    constructor called by ControlMechanism._instantiate_objective_mechanism;  otherwise, they are created using the
    ObjectiveMechanism.add_to_monitor method.
COMMENT

.. _ControlMechanism_ControlSignals:

*Specifying Parameters to Control*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This can be specified in either of two ways (see `ControlSignal_Examples` in `ControlSignal`):

*With a ControlMechanism itself*

The parameters controlled by a ControlMechanism can be specified in the **control** argument of its constructor;
the argument must be a `specification for one more ControlSignals <ControlSignal_Specification>`.  The parameter to
be controlled must belong to a Component in the same `Composition` as the ControlMechanism when it is added to the
Composition, or an error will occur.

*With a Parameter to be controlled by the* `controller <Composition.controller>` *of a* `Composition`

Control can also be specified for a parameter where the `parameter itself is specified <ParameterPort_Specification>`,
by including the specification of a `ControlSignal`, `ControlProjection`, or the keyword `CONTROL` in a `tuple
specification <ParameterPort_Tuple_Specification>` for the parameter.  In this case, the specified parameter will be
assigned for control by the `controller <Composition.controller>` of any `Composition` to which its Component belongs,
when the Component is added to the Composition (see `ControlMechanism_Composition_Controller`).  Conversely, when
a ControlMechanism is assigned as the `controller <Composition.controller>` of a Composition, a `ControlSignal` is
created and assigned to the ControlMechanism for every parameter of any `Component <Component>` in the Composition
that has been `specified for control <ParameterPort_Modulatory_Specification>`.

In general, a `ControlSignal` is created for each parameter specified to be controlled by a ControlMechanism.  These
are a type of `OutputPort` that send a `ControlProjection` to the `ParameterPort` of the parameter to be
controlled. All of the ControlSignals for a ControlMechanism are listed in its `control_signals
<ControlMechanism.control_signals>` attribute, and all of its ControlProjections are listed in
its `control_projections <ControlMechanism.control_projections>` attribute (see `ControlMechanism_Examples`).

.. _ControlMechanism_Structure:

Structure
---------

.. _ControlMechanism_Input:

*Input*
~~~~~~~

By default, a ControlMechanism has a single (`primary <InputPort_Primary>`) `input_port
<ControlMechanism.input_port>` that is named *OUTCOME*.  If the ControlMechanism has an `objective_mechanism
<ControlMechanism.objective_mechanism>`, then the *OUTCOME* `input_port <ControlMechanism.input_port>` receives a
single `MappingProjection` from the `objective_mechanism <ControlMechanism.objective_mechanism>`\\'s *OUTCOME*
OutputPort (see `ControlMechanism_ObjectiveMechanism` for additional details). Otherwise, when the ControlMechanism is
added to a `Composition`, MappingProjections are created that project to the ControlMechanism's *OUTCOME* `input_port
<ControlMechanism.input_port>` from each of the OutputPorts specified in the **monitor_for_control** `argument
<ControlMechanism_Monitor_for_Control_Argument>` of its constructor.  The `value <InputPort.value>` of the
ControlMechanism's *OUTCOME* InputPort is assigned to its `outcome <ControlMechanism.outcome>` attribute),
and is used as the input to the ControlMechanism's `function <ControlMechanism.function>` to determine its
`control_allocation <ControlMechanism.control_allocation>`.

.. _ControlMechanism_Function:

*Function*
~~~~~~~~~~

A ControlMechanism's `function <ControlMechanism.function>` uses its `outcome <ControlMechanism.outcome>`
attribute (the `value <InputPort.value>` of its *OUTCOME* `InputPort`) to generate a `control_allocation
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
different items in `control_allocation` as their `variable <Projection_Base.variable>`
(see `OutputPort_Custom_Variable`).

.. _ControlMechanism_Output:

*Output*
~~~~~~~~

The OutputPorts of a ControlMechanism are `ControlSignals <ControlSignal>` (listed in its `control_signals
<ControlMechanism.control_signals>` attribute). It has a `ControlSignal` for each parameter specified in the
**control** argument of its constructor, that sends a `ControlProjection` to the `ParameterPort` for the
corresponding parameter.  The ControlSignals are listed in the `control_signals <ControlMechanism.control_signals>`
attribute;  since they are a type of `OutputPort`, they are also listed in the ControlMechanism's `output_ports
<Mechanism_Base.output_ports>` attribute. The parameters modulated by a ControlMechanism's ControlSignals can be
displayed using its `show <ControlMechanism.show>` method. By default, each `ControlSignal` is assigned as its
`allocation <ControlSignal.allocation>` the value of the  corresponding item of the ControlMechanism's
`control_allocation <ControlMechanism.control_allocation>`;  however, subtypes of ControlMechanism may assign
allocations differently. The `default_allocation  <ControlMechanism.default_allocation>` attribute can be used to
specify a  default allocation for ControlSignals that have not been assigned their own `default_allocation
<ControlSignal.default_allocation>`. The `allocation <ControlSignal.allocation>` is used by each ControlSignal to
determine its `intensity <ControlSignal.intensity>`, which is then assigned to the `value <ControlProjection.value>`
of the ControlSignal's `ControlProjection`.   The `value <ControlProjection.value>` of the ControlProjection is used
by the `ParameterPort` to which it projects to modify the value of the parameter it controls (see
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
in value, and a `net_outcome <ControlMechanism.net_outcome>` (the `value <InputPort.value>` of the ControlMechanism's
*OUTCOME* `input_port <ControlMechanism_Input>` minus its `combined costs <ControlMechanism.combined_costs>`),
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

A ControlMechanism is executed using the same sequence of actions as any `Mechanism <Mechanism_Execution>`, with the
following additions.

The ControlMechanism's `function <ControlMechanism.function>` takes as its input the `value <InputPort.value>` of
its *OUTCOME* `input_port <ControlMechanism.input_port>` (also contained in `outcome <ControlSignal.outcome>`).
It uses that to determine the `control_allocation <ControlMechanism.control_allocation>`, which specifies the value
assigned to the `allocation <ControlSignal.allocation>` of each of its `ControlSignals <ControlSignal>`.  Each
ControlSignal uses that value to calculate its `intensity <ControlSignal.intensity>`, as well as its `cost
<ControlSignal.cost>.  The `intensity <ControlSignal.intensity>`is used by its `ControlProjection(s)
<ControlProjection>` to modulate the value of the ParameterPort(s) for the parameter(s) it controls.  Note that
the modulated value of the parameter may not be used until the subsequent `TRIAL <TimeScale.TRIAL>` of execution,
if the ControlMechansim is not executed until after the Component to which the paramter belongs is executed
(see `note <ModulatoryMechanism_Lazy_Evaluation_Note>`).

.. _ControlMechanism_Costs_Computation:

*Computation of Costs and Net_Outcome*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the ControlMechanism's `function <ControlMechanism.function>` has executed, if `compute_reconfiguration_cost
<ControlMechanism.compute_reconfiguration_cost>` has been specified, then it is used to compute the
`reconfiguration_cost <ControlMechanism.reconfiguration_cost>` for its `control_allocation
<ControlMechanism.control_allocation>` (see `above <ControlMechanism_Reconfiguration_Cost>`. After that, each
of the ControlMechanism's `control_signals <ControlMechanism.control_signals>` calculates its `cost
<ControlSignal.cost>`, based on its `intensity  <ControlSignal.intensity>`.  The ControlMechanism then combines these
with the `reconfiguration_cost <ControlMechanism.reconfiguration_cost>` using its `combine_costs
<ControlMechanism.combine_costs>` function, and the result is assigned to the `costs <ControlMechanism.costs>`
attribute.  Finally, the ControlMechanism uses this, together with its `outcome <ControlMechanism.outcome>` attribute,
to compute a `net_outcome <ControlMechanism.net_outcome>` using its `compute_net_outcome
<ControlMechanism.compute_net_outcome>` function.  This is used by some subclasses of ControlMechanism
(e.g., `OptimizationControlMechanism`) to  compute its `control_allocation <ControlMechanism.control_allocation>`
for the next `TRIAL <TimeScale.TRIAL>` of execution.

.. _ControlMechanism_Controller_Execution:

*Execution as Controller of a Composition*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if a ControlMechanism is assigned as the `controller of a `Composition <ControlMechanism_Composition_Controller>`,
then it is executed either before or after all of the other  `Mechanisms <Mechanism>` executed in a `TRIAL
<TimeScale.TRIAL>` for that Composition, depending on the value assigned to the Composition's `controller_mode
<Composition.controller_mode>` attribute (see `Composition_Controller_Execution`).  If a ControlMechanism is added to
a Composition for which it is not a `controller <Composition.controller>`, then it executes in the same way as any
`Mechanism <Mechanism>`, based on its place in the Composition's `graph <Composition.graph>`.  Because
`ControlProjections <ControlProjection>` are likely to introduce cycles (recurrent connection loops) in the
graph, the effects of a ControlMechanism and its projections will generally not be applied in the first `TRIAL
<TimeScale.TRIAL>` (see `Composition_Cycles_and_Feedback` for configuring the initialization of feedback
loops in a Composition; also see `Scheduler` for a description of additional ways in which a ControlMechanism and its
dependents can be scheduled to execute).

.. _ControlMechanism_Examples:

Examples
--------

The examples below focus on the specificaiton of the `objective_mechanism <ControlMechanism.objective_mechanism>`
for a ControlMechanism.  See `Control Signal Examples <ControlSignal_Examples>` for examples of how to specify the
ControlSignals for a ControlMechanism.

The following example creates a ControlMechanism by specifying its **objective_mechanism** using a constructor
that specifies the OutputPorts to be monitored by its `objective_mechanism <ControlMechanism.objective_mechanism>`
and the function used to evaluate these::

    >>> my_mech_A = ProcessingMechanism(name="Mech A")
    >>> my_DDM = DDM(name="My DDM")
    >>> my_mech_B = ProcessingMechanism(function=Logistic,
    ...                                 name="Mech B")

    >>> my_control_mech = ControlMechanism(
    ...                          objective_mechanism=ObjectiveMechanism(monitor=[(my_mech_A, 2, 1),
    ...                                                                           my_DDM.output_ports[RESPONSE_TIME]],
    ...                                                                 name="Objective Mechanism"),
    ...                          function=LinearCombination(operation=PRODUCT),
    ...                          control_signals=[(THRESHOLD, my_DDM),
    ...                                           (GAIN, my_mech_B)],
    ...                          name="My Control Mech")

This creates an ObjectiveMechanism for the ControlMechanism that monitors the `primary OutputPort <OutputPort_Primary>`
of ``my_mech_A`` and the *RESPONSE_TIME* OutputPort of ``my_DDM``;  its function first multiplies the former by ``2``,
then takes product of their values and passes the result as the input to the ControlMechanism.  The ControlMechanism's
`function <ControlMechanism.function>` uses this value to determine the allocation for its ControlSignals, that control
the value of the `threshold <DriftDiffusionAnalytical.threshold>` parameter of the `DriftDiffusionAnalytical` Function
for ``my_DDM`` and the  `gain <Logistic.gain>` parameter of the `Logistic` Function for ``my_transfer_mech_B``.

The following example specifies the same set of OutputPorts for the ObjectiveMechanism, by assigning them directly
to the **objective_mechanism** argument::

    >>> my_control_mech = ControlMechanism(
    ...                             objective_mechanism=[(my_mech_A, 2, 1),
    ...                                                  my_DDM.output_ports[RESPONSE_TIME]],
    ...                             control_signals=[(THRESHOLD, my_DDM),
    ...                                              (GAIN, my_mech_B)])

Note that, while this form is more succinct, it precludes specifying the ObjectiveMechanism's function.  Therefore,
the values of the monitored OutputPorts will be added (the default) rather than multiplied.

The ObjectiveMechanism can also be created on its own, and then referenced in the constructor for the ControlMechanism::

    >>> my_obj_mech = ObjectiveMechanism(monitored_output_ports=[(my_mech_A, 2, 1),
    ...                                                            my_DDM.output_ports[RESPONSE_TIME]],
    ...                                      function=LinearCombination(operation=PRODUCT))

    >>> my_control_mech = ControlMechanism(
    ...                        objective_mechanism=my_obj_mech,
    ...                        control_signals=[(THRESHOLD, my_DDM),
    ...                                         (GAIN, my_mech_B)])

Here, as in the first example, the constructor for the ObjectiveMechanism can be used to specify its function, as well
as the OutputPort that it monitors.

COMMENT:
FIX 8/27/19 [JDC]:  ADD TO COMPOSITION
See `System_Control_Examples` for examples of how a ControlMechanism, the OutputPorts its
`objective_mechanism <ControlSignal.objective_mechanism>`, and its `control_signals <ControlMechanism.control_signals>`
can be specified for a System.
COMMENT

.. _ControlMechanism_Class_Reference:

Class Reference
---------------

"""

import copy
import collections
import itertools
import numpy as np
import threading
import typecheck as tc
import warnings

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.functions.function import Function_Base, is_function_type
from psyneulink.core.components.functions.combinationfunctions import LinearCombination
from psyneulink.core.components.mechanisms.modulatory.modulatorymechanism import ModulatoryMechanism_Base
from psyneulink.core.components.mechanisms.mechanism import Mechanism, Mechanism_Base
from psyneulink.core.components.ports.port import Port, _parse_port_spec
from psyneulink.core.components.ports.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.components.ports.parameterport import ParameterPort
from psyneulink.core.globals.defaults import defaultControlAllocation
from psyneulink.core.globals.keywords import \
    AUTO_ASSIGN_MATRIX, CONTROL, CONTROL_PROJECTION, CONTROL_SIGNAL, CONTROL_SIGNALS, \
    EID_SIMULATION, GATING_SIGNAL, INIT_EXECUTE_METHOD_ONLY, NAME, \
    MECHANISM, MULTIPLICATIVE, MODULATORY_SIGNALS, MONITOR_FOR_CONTROL, MONITOR_FOR_MODULATION, \
    OBJECTIVE_MECHANISM, OUTCOME, OWNER_VALUE, PRODUCT, PROJECTION_TYPE, PROJECTIONS, PORT_TYPE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import ContentAddressableList, convert_to_list, convert_to_np_array, copy_iterable_with_shared, is_iterable

__all__ = [
    'CONTROL_ALLOCATION', 'GATING_ALLOCATION', 'ControlMechanism', 'ControlMechanismError', 'ControlMechanismRegistry',
    'DefaultAllocationFunction'
]

CONTROL_ALLOCATION = 'control_allocation'
GATING_ALLOCATION = 'gating_allocation'

MonitoredOutputPortTuple = collections.namedtuple("MonitoredOutputPortTuple", "output_port weight exponent matrix")

ControlMechanismRegistry = {}

def _is_control_spec(spec):
    from psyneulink.core.components.projections.modulatory.controlprojection import ControlProjection
    if isinstance(spec, tuple):
        return any(_is_control_spec(item) for item in spec)
    if isinstance(spec, dict) and PROJECTION_TYPE in spec:
        return _is_control_spec(spec[PROJECTION_TYPE])
    elif isinstance(spec, (ControlMechanism,
                           ControlSignal,
                           ControlProjection)):
        return True
    elif isinstance(spec, type) and issubclass(spec, (ControlMechanism,
                                                      ControlSignal,
                                                      ControlProjection)):
        return True
    elif isinstance(spec, str) and spec in {CONTROL, CONTROL_PROJECTION, CONTROL_SIGNAL}:
        return True
    else:
        return False

class ControlMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


def validate_monitored_port_spec(owner, spec_list):
    for spec in spec_list:
        if isinstance(spec, MonitoredOutputPortTuple):
            spec = spec.output_port
        elif isinstance(spec, tuple):
            spec = _parse_port_spec(
                owner=owner,
                port_type=InputPort,
                port_spec=spec,
            )
            spec = spec['params'][PROJECTIONS][0][0]
        elif isinstance(spec, dict):
            # If it is a dict, parse to validate that it is an InputPort specification dict
            #    (for InputPort of ObjectiveMechanism to be assigned to the monitored_output_port)
            spec = _parse_port_spec(
                owner=owner,
                port_type=InputPort,
                port_spec=spec,
            )
            # Get the OutputPort, to validate that it is in the ControlMechanism's Composition (below);
            #    presumes that the monitored_output_port is the first in the list of projection_specs
            #    in the InputPort port specification dictionary returned from the parse,
            #    and that it is specified as a projection_spec (parsed into that in the call
            #    to _parse_connection_specs by _parse_port_spec)
            spec = spec[PROJECTIONS][0][0]

        if not isinstance(spec, (OutputPort, Mechanism)):
            if isinstance(spec, type) and issubclass(spec, Mechanism):
                raise ControlMechanismError(
                    f"Mechanism class ({spec.__name__}) specified in '{MONITOR_FOR_CONTROL}' arg "
                    f"of {self.name}; it must be an instantiated {Mechanism.__name__} or "
                    f"{OutputPort.__name__} of one."
                )
            elif isinstance(spec, Port):
                raise ControlMechanismError(
                    f"{spec.__class__.__name__} specified in '{MONITOR_FOR_CONTROL}' arg of {owner.name} "
                    f"({spec.name} of {spec.owner.name}); "
                    f"it must be an {OutputPort.__name__} or {Mechanism.__name__}."
                )
            else:
                raise ControlMechanismError(
                    f"Erroneous specification of '{MONITOR_FOR_CONTROL}' arg for {owner.name} ({spec}); "
                    f"it must be an {OutputPort.__name__} or a {Mechanism.__name__}."
                )


def _control_mechanism_costs_getter(owning_component=None, context=None):
    # NOTE: In cases where there is a reconfiguration_cost, that cost is not returned by this method
    try:
        costs = [
            convert_to_np_array(
                c.compute_costs(c.parameters.value._get(context), context=context)
            )
            for c in owning_component.control_signals
            if hasattr(c, 'compute_costs')
        ] # GatingSignals don't have cost fcts
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
        return c.compute_net_outcome(
            c.parameters.outcome._get(context),
            c.combine_costs()
        )
    except TypeError:
        return [0]

class DefaultAllocationFunction(Function_Base):
    """Take a single 1d item and return a 2d array with n identical items
    Takes the default input (a single value in the *OUTCOME* InputPort of the ControlMechanism),
    and returns the same allocation for each of its `control_signals <ControlMechanism.control_signals>`.
    """
    componentName = 'Default Control Function'
    class Parameters(Function_Base.Parameters):
        """
            Attributes
            ----------

                num_control_signals
                    see `num_control_signals <DefaultAllocationFunction.num_control_signals>`

                    :default value: 1
                    :type: ``int``
        """
        num_control_signals = Parameter(1, stateful=False)

    def __init__(self,
                 default_variable=None,
                 params=None,
                 owner=None
                 ):

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         )

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        num_ctl_sigs = self._get_current_parameter_value('num_control_signals')
        result = np.array([variable[0]] * num_ctl_sigs)
        return self.convert_output_type(result)

    def reset(self, *args, force=False, context=None, **kwargs):
        # Override Component.reset which requires that the Component is stateful
        pass

    def _gen_llvm_function_body(self, ctx, builder, _1, _2, arg_in, arg_out, *, tags:frozenset):
        val_ptr = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(0)])
        val = builder.load(val_ptr)
        with pnlvm.helpers.array_ptr_loop(builder, arg_out, "alloc_loop") as (b, idx):
            out_ptr = builder.gep(arg_out, [ctx.int32_ty(0), idx])
            builder.store(val, out_ptr)
        return builder


class ControlMechanism(ModulatoryMechanism_Base):
    """
    ControlMechanism(                       \
        monitor_for_control=None,           \
        objective_mechanism=None,           \
        function=Linear,                    \
        default_allocation=None,            \
        control=None,                       \
        modulation=MULTIPLICATIVE,          \
        combine_costs=np.sum,               \
        compute_reconfiguration_cost=None,  \
        compute_net_outcome=lambda x,y:x-y)

    Subclass of `ModulatoryMechanism <ModulatoryMechanism>` that modulates the parameter(s) of one or more
    `Component(s) <Component>`.  See `Mechanism <Mechanism_Class_Reference>` for additional arguments and attributes.

    COMMENT: FIX 5/8/20
        Description:
            Protocol for instantiating unassigned ControlProjections (i.e., w/o a sender specified):
               If sender is not specified for a ControlProjection (e.g., in a parameter specification tuple)
                   it is flagged for deferred_init() in its __init__ method
               If ControlMechanism is instantiated or assigned as the controller for a System:
                   the System calls its _get_monitored_output_ports() method which returns all of the OutputPorts
                       within the System that have been specified to be MONITORED_FOR_CONTROL, and then assigns
                       them (along with any specified in the **monitored_for_control** arg of the System's constructor)
                       to the `objective_mechanism` argument of the ControlMechanism's constructor;
                   the System calls its _get_control_signals_for_system() method which returns all of the parameters
                       that have been specified for control within the System, assigns them a ControlSignal
                       (with a ControlProjection to the ParameterPort for the parameter), and assigns the
                       ControlSignals (alogn with any specified in the **control** argument of the System's
                       constructor) to the **control** argument of the ControlMechanism's constructor

            OBJECTIVE_MECHANISM param determines which Ports will be monitored.
                specifies the OutputPorts of the terminal Mechanisms in the System to be monitored by ControlMechanism
                this specification overrides any in System.], but can be overridden by mechanism.params[
                ?? if MonitoredOutputPorts appears alone, it will be used to determine how Ports are assigned from
                    System.execution_graph by default
                if MonitoredOutputPortsOption is used, it applies to any Mechanisms specified in the list for which
                    no OutputPorts are listed; it is overridden for any Mechanism for which OutputPorts are
                    explicitly listed
                TBI: if it appears in a tuple with a Mechanism, or in the Mechamism's params list, it applies to
                    just that Mechanism
    COMMENT

    Arguments
    ---------

    monitor_for_control : List[OutputPort or Mechanism] : default None
        specifies the `OutputPorts <OutputPort>` to be monitored by the `ObjectiveMechanism`, if specified in the
        **objective_mechanism** argument (see `ControlMechanism_ObjectiveMechanism`), or directly by the
        ControlMechanism itself if an **objective_mechanism** is not specified.  If any specification is a Mechanism
        (rather than its OutputPort), its `primary OutputPort <OutputPort_Primary>` is used (see
        `ControlMechanism_Monitor_for_Control` for additional details).

    objective_mechanism : ObjectiveMechanism or List[OutputPort specification] : default None
        specifies either an `ObjectiveMechanism` to use for the ControlMechanism, or a list of the OutputPorts it
        should monitor; if a list of `OutputPort specifications <ObjectiveMechanism_Monitor>` is used,
        a default ObjectiveMechanism is created and the list is passed to its **monitor** argument, along with any
        OutputPorts specified in the ControlMechanism's **monitor_for_control** `argument
        <ControlMechanism_Monitor_for_Control_Argument>`.

    function : TransferFunction : default Linear(slope=1, intercept=0)
        specifies function used to combine values of monitored OutputPorts.

    default_allocation : number, list or 1d array : None
        specifies the default_allocation of any `control_signals <ControlMechanism.control_signals>` for
        which the **default_allocation** was not specified in its constructor (see `default_allocation
        <ControlMechanism.default_allocation>` for additional details).

    control : ControlSignal specification or list[ControlSignal specification, ...]
        specifies the parameters to be controlled by the ControlMechanism; a `ControlSignal` is created for each
        (see `ControlSignal_Specification` for details of specification).

    modulation : str : MULTIPLICATIVE
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

    Attributes
    ----------

    objective_mechanism : ObjectiveMechanism
        `ObjectiveMechanism` that monitors and evaluates the values specified in the ControlMechanism's
        **objective_mechanism** argument, and transmits the result to the ControlMechanism's *OUTCOME*
        `input_port <Mechanism_Base.input_port>`.

    monitor_for_control : List[OutputPort]
        each item is an `OutputPort` monitored by the ControlMechanism or its `objective_mechanism
        <ControlMechanism.objective_mechanism>` if that is specified (see `ControlMechanism_Monitor_for_Control`);
        in the latter case, the list returned is ObjectiveMechanism's `monitor <ObjectiveMechanism.monitor>` attribute.

    monitored_output_ports_weights_and_exponents : List[Tuple(float, float)]
        each tuple in the list contains the weight and exponent associated with a corresponding OutputPort specified
        in `monitor_for_control <ControlMechanism.monitor_for_control>`; if `objective_mechanism
        <ControlMechanism.objective_mechanism>` is specified, these are the same as those in the ObjectiveMechanism's
        `monitor_weights_and_exponents <ObjectiveMechanism.monitor_weights_and_exponents>` attribute,
        and are used by the ObjectiveMechanism's `function <ObjectiveMechanism.function>` to parametrize the
        contribution made to its output by each of the values that it monitors (see `ObjectiveMechanism Function
        <ObjectiveMechanism_Function>`).

    input_port : InputPort
        the ControlMechanism's `primary InputPort <InputPort_Primary>`, named *OUTCOME*;  this receives a
        `MappingProjection` from the *OUTCOME* `OutputPort <ObjectiveMechanism_Output>` of `objective_mechanism
        <ControlMechanism.objective_mechanism>` if that is specified; otherwise, it receives MappingProjections
        from each of the OutputPorts specifed in `monitor_for_control <ControlMechanism.monitor_for_control>`
        (see `ControlMechanism_Input` for additional details).

    outcome : 1d array
        the `value <InputPort.value>` of the ControlMechanism's *OUTCOME* `input_port <ControlMechanism.input_port>`.

    function : TransferFunction : default Linear(slope=1, intercept=0)
        determines how the `value <OuputPort.value>`\\s of the `OutputPorts <OutputPort>` specified in the
        **monitor_for_control** `argument <ControlMechanism_Monitor_for_Control_Argument>` of the ControlMechanism's
        constructor are used to generate its `control_allocation <ControlMechanism.control_allocation>`.

    default_allocation : number, list or 1d array
        determines the default_allocation of any `control_signals <ControlMechanism.control_signals>` for
        which the **default_allocation** was not specified in its constructor;  if it is None (not specified)
        then the ControlSignal's parameters.allocation.default_value is used. See documentation for
        **default_allocation** argument of ControlSignal constructor for additional details.

    control_allocation : 2d array
        each item is the value assigned as the `allocation <ControlSignal.allocation>` for the corresponding
        ControlSignal listed in the `control_signals` attribute;  the control_allocation is the same as the
        ControlMechanism's `value <Mechanism_Base.value>` attribute).

    control_signals : ContentAddressableList[ControlSignal]
        list of the `ControlSignals <ControlSignal>` for the ControlMechanism, including any inherited from a
        `Composition` for which it is a `controller <Composition.controller>` (same as ControlMechanism's
        `output_ports <Mechanism_Base.output_ports>` attribute); each sends a `ControlProjection`
        to the `ParameterPort` for the parameter it controls

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
        for each using its `compute_costs <ControlSignal.compute_costs>` method.

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
        list of `ControlProjections <ControlProjection>` that project from the ControlMechanism's `control_signals
        <ControlMechanism.control_signals>`.

    modulation : str
        the default form of modulation used by the ControlMechanism's `ControlSignals <GatingSignal>`,
        unless they are `individually specified <ControlSignal_Specification>`.

    """

    componentType = "ControlMechanism"

    initMethod = INIT_EXECUTE_METHOD_ONLY

    outputPortTypes = ControlSignal
    portListAttr = Mechanism_Base.portListAttr.copy()
    portListAttr.update({ControlSignal:CONTROL_SIGNALS})

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TYPE_DEFAULT_PREFERENCES
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     PREFERENCE_SET_NAME: 'ControlMechanismClassPreferences',
    #     PREFERENCE_KEYWORD<pref>: <setting>...}

    class Parameters(ModulatoryMechanism_Base.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <ControlMechanism.variable>`

                    :default value: numpy.array([[1.]])
                    :type: ``numpy.ndarray``

                value
                    see `value <ControlMechanism.value>`

                    :default value: numpy.array([1.])
                    :type: ``numpy.ndarray``

                combine_costs
                    see `combine_costs <ControlMechanism.combine_costs>`

                    :default value: sum
                    :type: ``types.FunctionType``

                compute_net_outcome
                    see `compute_net_outcome <ControlMechanism.compute_net_outcome>`

                    :default value: lambda outcome, cost: outcome - cost
                    :type: ``types.FunctionType``

                compute_reconfiguration_cost
                    see `compute_reconfiguration_cost <ControlMechanism.compute_reconfiguration_cost>`

                    :default value: None
                    :type:

                control_signal_costs
                    see `control_signal_costs <ControlMechanism.control_signal_costs>`

                    :default value: None
                    :type:
                    :read only: True

                costs
                    see `costs <ControlMechanism.costs>`

                    :default value: None
                    :type:
                    :read only: True

                default_allocation
                    see `default_allocation <ControlMechanism.default_allocation>`

                    :default value: None
                    :type:

                input_ports
                    see `input_ports <ControlMechanism.input_ports>`

                    :default value: [`OUTCOME`]
                    :type: ``list``
                    :read only: True

                modulation
                    see `modulation <ControlMechanism.modulation>`

                    :default value: `MULTIPLICATIVE_PARAM`
                    :type: ``str``

                monitor_for_control
                    see `monitor_for_control <ControlMechanism.monitor_for_control>`

                    :default value: [`OUTCOME`]
                    :type: ``list``
                    :read only: True

                net_outcome
                    see `net_outcome <ControlMechanism.net_outcome>`

                    :default value: None
                    :type:
                    :read only: True

                objective_mechanism
                    see `objective_mechanism <ControlMechanism.objective_mechanism>`

                    :default value: None
                    :type:

                outcome
                    see `outcome <ControlMechanism.outcome>`

                    :default value: None
                    :type:
                    :read only: True

                output_ports
                    see `output_ports <ControlMechanism.output_ports>`

                    :default value: None
                    :type:
                    :read only: True

                reconfiguration_cost
                    see `reconfiguration_cost <ControlMechanism_Reconfiguration_Cost>`

                    :default value: None
                    :type:
                    :read only: True

        """
        # This must be a list, as there may be more than one (e.g., one per control_signal)
        variable = Parameter(np.array([[defaultControlAllocation]]), pnl_internal=True, constructor_argument='default_variable')
        value = Parameter(np.array([defaultControlAllocation]), aliases='control_allocation', pnl_internal=True)
        default_allocation = None
        combine_costs = Parameter(np.sum, stateful=False, loggable=False)
        costs = Parameter(None, read_only=True, getter=_control_mechanism_costs_getter)
        control_signal_costs = Parameter(None, read_only=True, pnl_internal=True)
        compute_reconfiguration_cost = Parameter(None, stateful=False, loggable=False)
        reconfiguration_cost = Parameter(None, read_only=True)
        outcome = Parameter(None, read_only=True, getter=_outcome_getter, pnl_internal=True)
        compute_net_outcome = Parameter(lambda outcome, cost: outcome - cost, stateful=False, loggable=False)
        net_outcome = Parameter(
            None,
            read_only=True,
            getter=_net_outcome_getter,
            pnl_internal=True
        )
        simulation_ids = Parameter([], user=False, pnl_internal=True)
        modulation = Parameter(MULTIPLICATIVE, pnl_internal=True)

        objective_mechanism = Parameter(None, stateful=False, loggable=False, structural=True)

        input_ports = Parameter(
            [OUTCOME],
            stateful=False,
            loggable=False,
            read_only=True,
            structural=True,
            parse_spec=True,
        )

        monitor_for_control = Parameter(
            [OUTCOME],
            stateful=False,
            loggable=False,
            read_only=True,
        )

        output_ports = Parameter(
            None,
            stateful=False,
            loggable=False,
            read_only=True,
            structural=True,
            parse_spec=True,
            aliases=['control', 'control_signals'],
            constructor_argument='control'
        )

        def _parse_output_ports(self, output_ports):
            def is_2tuple(o):
                return isinstance(o, tuple) and len(o) == 2

            if not isinstance(output_ports, list):
                output_ports = [output_ports]

            for i in range(len(output_ports)):
                # handle 2-item tuple
                if is_2tuple(output_ports[i]):

                    # this is an odd case that uses two names in the name entry
                    # unsure what it means
                    if isinstance(output_ports[i][0], list):
                        continue

                    output_ports[i] = {
                        NAME: output_ports[i][0],
                        MECHANISM: output_ports[i][1]
                    }
                # handle dict of form {PROJECTIONS: <2 item tuple>, <param1>: <value1>, ...}
                elif (
                    isinstance(output_ports[i], dict)
                    and PROJECTIONS in output_ports[i]
                    and is_2tuple(output_ports[i][PROJECTIONS])
                ):
                    full_spec_dict = {
                        NAME: output_ports[i][PROJECTIONS][0],
                        MECHANISM: output_ports[i][PROJECTIONS][1],
                        **{k: v for k, v in output_ports[i].items() if k != PROJECTIONS}
                    }
                    output_ports[i] = full_spec_dict

            return output_ports

        def _validate_input_ports(self, input_ports):
            if input_ports is None:
                return

            # FIX 5/28/20:
            # TODO: uncomment this method or remove this block entirely.
            # This validation check was never being run due to an
            # unintentionally suppressed exception. Why is the default
            # specification ([OUTCOME]) invalid according to this
            # method?
            # validate_monitored_port_spec(self._owner, input_ports)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 monitor_for_control:tc.optional(tc.any(is_iterable, Mechanism, OutputPort))=None,
                 objective_mechanism=None,
                 function=None,
                 default_allocation:tc.optional(tc.any(int, float, list, np.ndarray))=None,
                 control:tc.optional(tc.any(is_iterable,
                                            ParameterPort,
                                            InputPort,
                                            OutputPort,
                                            ControlSignal))=None,
                 modulation:tc.optional(str)=None,
                 combine_costs:tc.optional(is_function_type)=None,
                 compute_reconfiguration_cost:tc.optional(is_function_type)=None,
                 compute_net_outcome=None,
                 params=None,
                 name=None,
                 prefs:tc.optional(is_pref_set)=None,
                 **kwargs
                 ):

        monitor_for_control = convert_to_list(monitor_for_control) or []
        control = convert_to_list(control) or []

        # For backward compatibility:
        if kwargs:
            if MONITOR_FOR_MODULATION in kwargs:
                args = kwargs.pop(MONITOR_FOR_MODULATION)
                if args:
                    monitor_for_control.extend(convert_to_list(args))
            if MODULATORY_SIGNALS in kwargs:
                args = kwargs.pop(MODULATORY_SIGNALS)
                if args:
                    control.extend(convert_to_list(args))
            if CONTROL_SIGNALS in kwargs:
                args = kwargs.pop(CONTROL_SIGNALS)
                if args:
                    control.extend(convert_to_list(args))

        function = function or DefaultAllocationFunction

        self._sim_counts = {}

        super(ControlMechanism, self).__init__(
            default_variable=default_variable,
            size=size,
            modulation=modulation,
            params=params,
            name=name,
            function=function,
            monitor_for_control=monitor_for_control,
            control=control,
            output_ports=control,
            objective_mechanism=objective_mechanism,
            default_allocation=default_allocation,
            combine_costs=combine_costs,
            compute_net_outcome=compute_net_outcome,
            compute_reconfiguration_cost=compute_reconfiguration_cost,
            prefs=prefs,
            **kwargs
        )

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate SYSTEM, monitor_for_control, CONTROL_SIGNALS and GATING_SIGNALS

        """
        from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism

        super(ControlMechanism, self)._validate_params(request_set=request_set,
                                                       target_set=target_set,
                                                       context=context)

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
                        raise ControlMechanismError("Ambigusous specification of {} arg for {}; "
                                                    " it is in a list with other items ({})".
                                                    format(OBJECTIVE_MECHANISM, self.name, obj_mech_spec_list))
                else:
                    validate_monitored_port_spec(self, obj_mech_spec_list)

            if not isinstance(target_set[OBJECTIVE_MECHANISM], (ObjectiveMechanism, list, bool)):
                raise ControlMechanismError("Specification of {} arg for {} ({}) must be an {}"
                                            "or a list of Mechanisms and/or OutputPorts to be monitored for control".
                                            format(OBJECTIVE_MECHANISM,
                                                   self.name, target_set[OBJECTIVE_MECHANISM],
                                                   ObjectiveMechanism.componentName))

        if CONTROL in target_set and target_set[CONTROL]:
            control = target_set[CONTROL]
            assert isinstance(control, list), \
                f"PROGRAM ERROR: control arg {control} of {self.name} should have been converted to a list."
            for ctl_spec in control:
                ctl_spec = _parse_port_spec(port_type=ControlSignal, owner=self, port_spec=ctl_spec)
                if not (isinstance(ctl_spec, ControlSignal)
                        or (isinstance(ctl_spec, dict) and ctl_spec[PORT_TYPE]==ControlSignal.__name__)):
                    raise ControlMechanismError(f"Invalid specification for '{CONTROL}' argument of {self.name}:"
                                                f"({ctl_spec})")

    # IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
    def _instantiate_objective_mechanism(self, context=None):
        """
        # FIX: ??THIS SHOULD BE IN OR MOVED TO ObjectiveMechanism
        Assign InputPort to ObjectiveMechanism for each OutputPort to be monitored;
            uses _instantiate_monitoring_input_port and _instantiate_control_mechanism_input_port to do so.
            For each item in self.monitored_output_ports:
            - if it is a OutputPort, call _instantiate_monitoring_input_port()
            - if it is a Mechanism, call _instantiate_monitoring_input_port for relevant Mechanism_Base.output_ports
                (determined by whether it is a `TERMINAL` Mechanism and/or MonitoredOutputPortsOption specification)
            - each InputPort is assigned a name with the following format:
                '<name of Mechanism that owns the monitoredOutputPort>_<name of monitoredOutputPort>_Monitor'

        Notes:
        * self.monitored_output_ports is a list, each item of which is a Mechanism_Base.output_port from which a
          Projection will be instantiated to a corresponding InputPort of the ControlMechanism
        * self.input_ports is the usual ordered dict of ports,
            each of which receives a Projection from a corresponding OutputPort in self.monitored_output_ports
        """
        from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
        from psyneulink.core.components.mechanisms.processing.objectivemechanism import \
            ObjectiveMechanism, ObjectiveMechanismError
        from psyneulink.core.components.ports.inputport import EXPONENT_INDEX, WEIGHT_INDEX
        from psyneulink.core.components.functions.function import FunctionError

        # GET OutputPorts to Monitor (to specify as or add to ObjectiveMechanism's monitored_output_ports attribute

        monitored_output_ports = []

        monitor_for_control = self.monitor_for_control or []
        if not isinstance(monitor_for_control, list):
            monitor_for_control = [monitor_for_control]

        # If objective_mechanism is used to specify OutputPorts to be monitored (legacy feature)
        #    move them to monitor_for_control
        if isinstance(self.objective_mechanism, list):
            monitor_for_control.extend(self.objective_mechanism)

        # Add items in monitor_for_control to monitored_output_ports
        for i, item in enumerate(monitor_for_control):
            # If it is already in the list received from System, ignore
            if item in monitored_output_ports:
                # NOTE: this can happen if ControlMechanisms is being constructed by System
                #       which passed its monitor_for_control specification
                continue
            monitored_output_ports.extend([item])

        # INSTANTIATE ObjectiveMechanism

        # If *objective_mechanism* argument is an ObjectiveMechanism, add monitored_output_ports to it
        if isinstance(self.objective_mechanism, ObjectiveMechanism):
            if monitored_output_ports:
                self.objective_mechanism.add_to_monitor(monitor_specs=monitored_output_ports,
                                                        context=context)
        # Otherwise, instantiate ObjectiveMechanism with list of ports in monitored_output_ports
        else:
            try:
                self.objective_mechanism = ObjectiveMechanism(monitor=monitored_output_ports,
                                                               function=LinearCombination(operation=PRODUCT),
                                                               name=self.name + '_ObjectiveMechanism')
            except (ObjectiveMechanismError, FunctionError) as e:
                raise ObjectiveMechanismError(f"Error creating {OBJECTIVE_MECHANISM} for {self.name}: {e}")

        # Print monitored_output_ports
        if self.prefs.verbosePref:
            print("{0} monitoring:".format(self.name))
            for port in self.monitored_output_ports:
                weight = self.monitored_output_ports_weights_and_exponents[
                                                         self.monitored_output_ports.index(port)][WEIGHT_INDEX]
                exponent = self.monitored_output_ports_weights_and_exponents[
                                                         self.monitored_output_ports.index(port)][EXPONENT_INDEX]
                print(f"\t{weight} (exp: {weight}; wt: {exponent})")

        # Instantiate MappingProjection from ObjectiveMechanism to ControlMechanism
        projection_from_objective = MappingProjection(sender=self.objective_mechanism,
                                                      receiver=self,
                                                      matrix=AUTO_ASSIGN_MATRIX,
                                                      context=context)

        # CONFIGURE FOR ASSIGNMENT TO COMPOSITION

        # Insure that ObjectiveMechanism's input_ports are not assigned projections from a Composition's input_CIM
        for input_port in self.objective_mechanism.input_ports:
            input_port.internal_only = True

        # Flag ObjectiveMechanism and its Projection to ControlMechanism for inclusion in Composition
        from psyneulink.core.compositions.composition import NodeRole
        self.aux_components.append((self.objective_mechanism, NodeRole.CONTROL_OBJECTIVE))
        self.aux_components.append(projection_from_objective)

        # ASSIGN ATTRIBUTES

        self._objective_projection = projection_from_objective
        self.parameters.monitor_for_control._set(self.monitored_output_ports, context)

    def _instantiate_input_ports(self, context=None):

        super()._instantiate_input_ports(context=context)
        self.input_port.name = OUTCOME
        self.input_port.name = OUTCOME

        # If objective_mechanism is specified, instantiate it,
        #     including Projections to it from monitor_for_control
        if self.objective_mechanism:
            self._instantiate_objective_mechanism(context=context)

        # Otherwise, instantiate Projections from monitor_for_control to ControlMechanism
        elif self.monitor_for_control:
            from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
            for sender in convert_to_list(self.monitor_for_control):
                self.aux_components.append(MappingProjection(sender=sender, receiver=self.input_ports[OUTCOME]))

    def _instantiate_output_ports(self, context=None):

    # ---------------------------------------------------
    # FIX 5/23/17: PROJECTIONS AND PARAMS SHOULD BE PASSED BY ASSIGNING TO PORT SPECIFICATION DICT
    # FIX          UPDATE parse_port_spec TO ACCOMODATE (param, ControlSignal) TUPLE
    # FIX          TRACK DOWN WHERE PARAMS ARE BEING HANDED OFF TO ControlProjection
    # FIX                   AND MAKE SURE THEY ARE NOW ADDED TO ControlSignal SPECIFICATION DICT
    # ---------------------------------------------------

        self._register_control_signal_type(context=None)

        if self.control:
            self._instantiate_control_signals(context=context)

        super()._instantiate_output_ports(context=context)

    def _register_control_signal_type(self, context=None):
        from psyneulink.core.globals.registry import register_category
        from psyneulink.core.components.ports.port import Port_Base

        # Create registry for ControlSignals (to manage names)
        register_category(entry=ControlSignal,
                          base_class=Port_Base,
                          registry=self._portRegistry,
                          )

    def _instantiate_control_signals(self, context):
        """Subclassess can override for class-specific implementation (see OptimiziationControlMechanism for example)"""
        output_port_specs = list(enumerate(self.output_ports))

        for i, control_signal in output_port_specs:
            self.control[i] = self._instantiate_control_signal(control_signal, context=context)
        num_control_signals = i + 1

        # For DefaultAllocationFunction, set defaults.value to have number of items equal to num control_signals
        if isinstance(self.function, DefaultAllocationFunction):
            self.defaults.value = np.tile(self.function.value, (num_control_signals, 1))
            self.parameters.control_allocation._set(copy.deepcopy(self.defaults.value), context)
            self.function.num_control_signals = num_control_signals

        # For other functions, assume that if its value has:
        # - one item, all control_signals should get it (i.e., the default: (OWNER_VALUE, 0));
        # - same number of items as the number of control_signals;
        #     assign each control_signal to the corresponding item of the function's value
        # - a different number of items than number of control_signals,
        #     leave things alone, and allow any errant indices for control_signals to be caught later.
        else:
            self.defaults.value = np.array(self.function.value)
            self.parameters.value._set(copy.deepcopy(self.defaults.value), context)

            len_fct_value = len(self.function.value)

            # Assign each ControlSignal's variable_spec to index of ControlMechanism's value
            for i, control_signal in enumerate(self.control):

                # If number of control_signals is same as number of items in function's value,
                #    assign each ControlSignal to the corresponding item of the function's value
                if len_fct_value == num_control_signals:
                    control_signal._variable_spec = [(OWNER_VALUE, i)]

                if not isinstance(control_signal.owner_value_index, int):
                    assert False, \
                        f"PROGRAM ERROR: The \'owner_value_index\' attribute for {control_signal.name} " \
                            f"of {self.name} ({control_signal.owner_value_index})is not an int."

    def _instantiate_control_signal(self,  control_signal, context=None):
        """Parse and instantiate ControlSignal (or subclass relevant to ControlMechanism subclass)

        Temporarily assign variable to default allocation value to avoid chicken-and-egg problem:
           value, output_ports and control_signals haven't been expanded yet to accomodate the new
           ControlSignal; reassign control_signal.variable to actual OWNER_VALUE below,
           once value has been expanded
        """

        if self.output_ports is None:
            self.parameters.output_ports._set([], context)

        control_signal = self._instantiate_control_signal_type(control_signal, context)
        control_signal.owner = self

        self._check_for_duplicates(control_signal, self.control_signals, context)

        # Update control_signal_costs to accommodate instantiated Projection
        control_signal_costs = self.parameters.control_signal_costs._get(context)
        try:
            control_signal_costs = np.append(control_signal_costs, np.zeros((1, 1)), axis=0)
        except (AttributeError, ValueError):
            control_signal_costs = np.zeros((1, 1))
        self.parameters.control_signal_costs._set(control_signal_costs, context)

        # UPDATE output_ports AND control_projections -------------------------------------------------------------

        # FIX: 9/14/19 - THIS SHOULD BE IMPLEMENTED
        # TBI: For ControlMechanisms that accumulate, starting output must be equal to the initial "previous value"
        # so that modulation that occurs BEFORE the control mechanism executes is computed appropriately
        # if (isinstance(self.function, IntegratorFunction)):
        #     control_signal._intensity = function.initializer

        return control_signal

    def _instantiate_control_signal_type(self, control_signal_spec, context):
        """Instantiate actual ControlSignal, or subclass if overridden"""
        from psyneulink.core.components.ports.port import _instantiate_port
        from psyneulink.core.components.projections.projection import ProjectionError

        try:
            # set the default by implicit shape defined by one of the
            # allocation_samples if possible
            try:
                allocation_parameter_default = control_signal_spec._init_args['allocation_samples'][0]
            except AttributeError:
                allocation_parameter_default = control_signal_spec['allocation_samples'][0]

            # several tests depend on the default value being 1
            # tests/composition/test_control.py::TestControlSpecification::test_deferred_init
            # tests/composition/test_control.py::TestModelBasedOptimizationControlMechanisms::test_evc
            # tests/composition/test_control.py::TestModelBasedOptimizationControlMechanisms::test_laming_validation_specify_control_signals
            # tests/composition/test_control.py::TestModelBasedOptimizationControlMechanisms::test_stateful_mechanism_in_simulation
            allocation_parameter_default = np.ones(np.asarray(allocation_parameter_default).shape)
        except (KeyError, IndexError, TypeError):
            allocation_parameter_default = self.parameters.control_allocation.default_value

        control_signal = _instantiate_port(port_type=ControlSignal,
                                               owner=self,
                                               variable=self.defaults.default_allocation           # User specified value
                                                        or allocation_parameter_default,  # Parameter default
                                               reference_value=allocation_parameter_default,
                                               modulation=self.defaults.modulation,
                                               port_spec=control_signal_spec,
                                               context=context)
        if not type(control_signal) in convert_to_list(self.outputPortTypes):
            raise ProjectionError(f'{type(control_signal)} inappropriate for {self.name}')
        return control_signal

    def _check_for_duplicates(self, control_signal, control_signals, context):
        """
        Check that control_signal is not a duplicate of one already instantiated for the ControlMechanism

        Can happen if control of parameter is specified in constructor for a Mechanism
            and also in the ControlMechanism's **control** arg

        control_signals arg passed in to allow override by subclasses
        """

        for existing_ctl_sig in control_signals:
            # OK if control_signal is one already assigned to ControlMechanism (i.e., let it get processed below);
            # this can happen if it was in deferred_init status and initalized in call to _instantiate_port above.
            if control_signal == existing_ctl_sig:
                continue

            # Return if *all* projections from control_signal are identical to ones in an existing control_signal
            for proj in control_signal.efferents:
                if proj not in existing_ctl_sig.efferents:
                    # A Projection in control_signal is not in this existing one: it is different,
                    #    so break and move on to next existing_mod_sig
                    break
                return

            # Warn if *any* projections from control_signal are identical to ones in an existing control_signal
            projection_type = existing_ctl_sig.projection_type
            if any(
                    any(new_p.receiver == existing_p.receiver
                        for existing_p in existing_ctl_sig.efferents) for new_p in control_signal.efferents):
                warnings.warn(f"Specification of {control_signal.name} for {self.name} "
                              f"has one or more {projection_type}s redundant with ones already on "
                              f"an existing {ControlSignal.__name__} ({existing_ctl_sig.name}).")

    def show(self):
        """Display the OutputPorts monitored by ControlMechanism's `objective_mechanism
        <ControlMechanism.objective_mechanism>` and the parameters modulated by its `control_signals
        <ControlMechanism.control_signals>`.
        """

        print("\n---------------------------------------------------------")

        print("\n{0}".format(self.name))
        print("\n\tMonitoring the following Mechanism OutputPorts:")
        for port in self.objective_mechanism.input_ports:
            for projection in port.path_afferents:
                monitored_port = projection.sender
                monitored_port_Mech = projection.sender.owner
                # ContentAddressableList
                monitored_port_index = self.monitored_output_ports.index(monitored_port)

                weight = self.monitored_output_ports_weights_and_exponents[monitored_port_index][0]
                exponent = self.monitored_output_ports_weights_and_exponents[monitored_port_index][1]

                print("\t\t{0}: {1} (exp: {2}; wt: {3})".
                      format(monitored_port_Mech.name, monitored_port.name, weight, exponent))

        try:
            if self.control_signals:
                print("\n\tControlling the following Mechanism parameters:".format(self.name))
                # Sort for consistency of output:
                port_Names_sorted = sorted(self.control_signals.names)
                for port_Name in port_Names_sorted:
                    for projection in self.control_signals[port_Name].efferents:
                        print("\t\t{0}: {1}".format(projection.receiver.owner.name, projection.receiver.name))
        except:
            pass

        try:
            if self.gating_signals:
                print("\n\tGating the following Ports:".format(self.name))
                # Sort for consistency of output:
                port_Names_sorted = sorted(self.gating_signals.names)
                for port_Name in port_Names_sorted:
                    for projection in self.gating_signals[port_Name].efferents:
                        print("\t\t{0}: {1}".format(projection.receiver.owner.name, projection.receiver.name))
        except:
            pass

        print("\n---------------------------------------------------------")

    def add_to_monitor(self, monitor_specs, context=None):
        """Instantiate OutputPorts to be monitored by ControlMechanism's `objective_mechanism
        <ControlMechanism.objective_mechanism>`.

        **monitored_output_ports** can be any of the following:
            - `Mechanism <Mechanism>`;
            - `OutputPort`;
            - `tuple specification <InputPort_Tuple_Specification>`;
            - `Port specification dictionary <InputPort_Specification_Dictionary>`;
            - list with any of the above.
        If any item is a Mechanism, its `primary OutputPort <OutputPort_Primary>` is used.
        OutputPorts must belong to Mechanisms in the same `System` as the ControlMechanism.
        """
        output_ports = self.objective_mechanism.add_to_monitor(monitor_specs=monitor_specs, context=context)

    def _add_process(self, process, role:str):
        super()._add_process(process, role)
        if self.objective_mechanism:
            self.objective_mechanism._add_process(process, role)

    def _remove_default_control_signal(self, type:tc.enum(CONTROL_SIGNAL, GATING_SIGNAL)):
        if type == CONTROL_SIGNAL:
            ctl_sig_attribute = self.control_signals
        elif type == GATING_SIGNAL:
            ctl_sig_attribute = self.gating_signals
        else:
            assert False, \
                f"PROGRAM ERROR:  bad 'type' arg ({type})passed to " \
                    f"{ControlMechanism.__name__}._remove_default_control_signal" \
                    f"(should have been caught by typecheck"

        if (len(ctl_sig_attribute)==1
                and ctl_sig_attribute[0].name==type + '-0'
                and not ctl_sig_attribute[0].efferents):
            self.remove_ports(ctl_sig_attribute[0])

    def _activate_projections_for_compositions(self, composition=None):
        """Activate eligible Projections to or from Nodes in Composition.
        If Projection is to or from a node NOT (yet) in the Composition,
        assign it the node's aux_components attribute but do not activate it.
        """
        dependent_projections = set()

        if self.objective_mechanism and composition and self.objective_mechanism in composition.nodes:
            # Safe to add this, as it is already in the ControlMechanism's aux_components
            #    and will therefore be added to the Composition along with the ControlMechanism
            from psyneulink.core.compositions.composition import NodeRole
            assert (self.objective_mechanism, NodeRole.CONTROL_OBJECTIVE) in self.aux_components, \
                f"PROGRAM ERROR:  {OBJECTIVE_MECHANISM} for {self.name} not listed in its 'aux_components' attribute."
            dependent_projections.add(self._objective_projection)

            for aff in self.objective_mechanism.afferents:
                dependent_projections.add(aff)

        for ms in self.control_signals:
            for eff in ms.efferents:
                dependent_projections.add(eff)

        # ??ELIMINATE SYSTEM
        # FIX: 9/15/19 - HOW IS THIS DIFFERENT THAN objective_mechanism's AFFERENTS ABOVE?
        # assign any deferred init objective mech monitored OutputPort projections to this system
        if self.objective_mechanism and composition and self.objective_mechanism in composition.nodes:
            for output_port in self.objective_mechanism.monitored_output_ports:
                for eff in output_port.efferents:
                    dependent_projections.add(eff)

        # ??ELIMINATE SYSTEM
        # FIX: 9/15/19 - HOW IS THIS DIFFERENT THAN control_signal's EFFERENTS ABOVE?
        for eff in self.efferents:
            dependent_projections.add(eff)

        if composition:
            deeply_nested_aux_components = composition._get_deeply_nested_aux_projections(self)
            dependent_projections -= set(deeply_nested_aux_components.values())

        for proj in dependent_projections:
            proj._activate_for_compositions(composition)

        for proj in deeply_nested_aux_components.values():
            composition.add_projection(proj, sender=proj.sender, receiver=proj.receiver)

    def _apply_control_allocation(self, control_allocation, runtime_params, context):
        """Update values to `control_signals <ControlMechanism.control_signals>`
        based on specified `control_allocation <ControlMechanism.control_allocation>`
        (used by controller of a Composition in simulations)
        """
        value = [a for a in control_allocation]
        self.parameters.value._set(value, context)
        self._update_output_ports(runtime_params, context)

    @property
    def monitored_output_ports(self):
        try:
            return self.objective_mechanism.monitored_output_ports
        except AttributeError:
            return None

    @monitored_output_ports.setter
    def monitored_output_ports(self, value):
        try:
            self.objective_mechanism._monitored_output_ports = value
        except AttributeError:
            return None

    @property
    def monitored_output_ports_weights_and_exponents(self):
        try:
            return self.objective_mechanism.monitored_output_ports_weights_and_exponents
        except:
            return None

    @property
    def control_signals(self):
        """Get ControlSignals from OutputPorts"""
        try:
            return ContentAddressableList(component_type=ControlSignal,
                                          list=[port for port in self.output_ports
                                                if isinstance(port, (ControlSignal))])
        except:
            return []

    @property
    def control_projections(self):
        try:
            return [projection for control_signal in self.control_signals for projection in control_signal.efferents]
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
