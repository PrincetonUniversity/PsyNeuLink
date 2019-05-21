# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  ModulatoryMechanism ************************************************


"""
Overview
--------

A ModulatoryMechanism is an `AdaptiveMechanism <AdaptiveMechanism>` that `modulates the value(s)
<ModulatorySignal_Modulation>` of one or more `States <State>` of other Mechanisms in the `Composition` to which it
belongs.  It's `function <ModulatoryMechanism.function>` calculates a `modulatory_allocation
<ModulatoryMechanism.modulatory_allocation>`: a list of values provided to each of its `modulatory_signals
<ModulatoryMechanism.modulatory_signals>`.  These can be `ControlSignals <ControlSignal>`, that modulate the value of a
`ParameterState <ParameterState>` of another Mechanism, and/or `GatingSignals <GatingSignal>` that modulate the value of
an `InputState` or `OutputState` of another Mechanism.  A ModulatoryMechanism can be configured to monitor the outputs
of other Mechanisms in order to determine its `modulatory_allocation <ModulatoryMechanism.modulatory_allocation>`, by
assigning it an `ObjectiveMechanism` and/or specifying a list of `OutputStates <OutputState_Specification>` to monitor
(see `ModulatoryMechanism_ObjectiveMechanism` below).  A ModulatoryMechanism can also be assigned as the `controller
<Composition.controller>` of a `Composition`, which has a special relation to that Composition: it generally executes
either before or after all of the other Mechanisms in that Composition (see `Composition_Controller_Execution`).


.. _ModulatoryMechanism_Creation:

Creating a ModulatoryMechanism
------------------------------

A ModulatoryMechanism is created by calling its constructor.  If neither its **montior_for_control** nor
**objective_mechanism** arguments are specified, then only the ModulatoryMechanism is constructed, and its inputs
must be specified in some other way.  However, either of those arguments can be used to configure its inputs, as
described below.

.. _ModulatoryMechanism_ObjectiveMechanism:


*ObjectiveMechanism*
~~~~~~~~~~~~~~~~~~~~

If an `ObjectiveMechanism` is specified in the **objective_mechanism** of a ModulatoryMechanism's constructor, or
any `OutputStates <OutputState>` are specified in its **monitor_for_modulation** argument, then an ObjectiveMechanism
is automatically created and assigned as the ModulatoryMechanism's `objective_mechanism
<ModulatoryMechanism.objective_mechanism>` attribute.  This is used to monitor the OutputStates specified in either
the **monitor** argument of the ObjectiveMechanism's constructor and/or the  **monitor_for_modulation** argument of
the ModulatoryMechanism's constructor.  The values of these OutputStates are  evaluted by the ObjectiveMechanism's
`function <ObjectiveMechanism.function>`, and the result is conveyed to the  ModulatoryMechanism by way of a
`MappingProjection` created from the *OUTCOME* Outputstate of the ObjectiveMechanism to the *OUTCOME* InputState of
the ModulatoryMechanism, and used by it to determine its `modulatory_allocation
<ModulatoryMechanism.modulatory_allocation>`.  The OutputStates monitored by the ModulatoryMechahism's
`objective_mechanism <ModulatoryMechanism.objective_mechanism>` are listed in its `monitor_for_modulation
<ModulatoryMechanism.monitor_for_modulation>` attribute, and in the ObjectiveMechanism's `monitor
<ObjectiveMechanism.monitor>` attribute.  These, along with the States modulated by the ModulatoryMechanism,
can also be listed using its `show <ModulatoryMechanism.show>` method.

The ObjectiveMechanism and/or OutputStates it monitors can be specified using any of the following in the constructor
for the ModualtoryMechanism:

  * in the **objective_mechanism** argument:

    - An existing `ObjectiveMechanism`
    ..
    - A constructor for an ObjectiveMechanism; its **monitor** argument can be used to specify `the OutputStates to
      be monitored <ObjectiveMechanism_Monitor>`, and its **function** argument can be used to
      specify how those OutputStates are evaluated (see `ModulatoryMechanism_Examples`).
    ..
    - A list of `OutputState specifications <ObjectiveMechanism_Monitor>`; a default ObjectiveMechanism
      is created, and the list of OutputState specifications are assigned to its **monitor** argument.
  ..
  * in the **monitor_for_modulation** argument, a list of `OutputState specifications
    <ObjectiveMechanism_Monitor>`;  a default ObjectiveMechanism is created, and the list of OutputState specifications
    are assigned to its **monitor** argument.

  If OutputStates to be monitored are specified in both the **objective_mechanism** argument (on their own, or within
  the constructor for an ObjectiveMechanism) *and* the **monitor_for_modulation** argument, both sets are used in
  creating the ObjectiveMechanism.

  If an ObjectiveMechanism is specified using its constructor, any specifications in that constructor
  override any attributes specified for the default `objective_mechanism
  <ModulatoryMechanism.objective_mechanism>` of a ModulatoryMechanism, including those of its `function
  <ObjectiveMechanism.function>` (see `note <EVCControlMechanism_Objective_Mechanism_Function_Note>` in
  EVCModulatoryMechanism for an example);

OutputStates to be monitored can also be added using either the ModulatoryMechanism's `add_to_monitor
<ModulatoryMechanism.add_to_monitor>` method, or the `corresponding method <ObjectiveMechanism.add_to_monitor>` of
the ObjectiveMechanism.

COMMENT:
TBI [Functionality for System that has yet to be ported to Composition]
If a ModulatoryMechanism is specified as the `controller <Composition.controller>` of a Composition (see `below
<ModulatoryMechanism_Composition_Controller>`), any OutputStates specified to be monitored by the System are
assigned as inputs to the ObjectiveMechanism.  This includes any specified in the **monitor_for_modulation** argument
of the System's constructor, as well as any specified in a monitor_for_modulation entry of a Mechanism `parameter
specification dictionary <ParameterState_Specification>` (see `Mechanism_Constructor_Arguments` and
`System_Control_Specification`).

FOR DEVELOPERS:
    If the ObjectiveMechanism has not yet been created, these are added to the **monitor** argument of its
    constructor called by ModulatoryMechanism._instantiate_objective_mechanism;  otherwise, they are created using the
    ObjectiveMechanism.add_to_monitor method.

INTEGRATE WTH ABOVE:
If the
ModulatoryMechanism is created automatically by a System (as its `controller <System.controller>`), then the specification
of OutputStates to be monitored and parameters to be controlled are made on the System and/or the Components
themselves (see `System_Control_Specification`).  In either case, the Components needed to monitor the specified
OutputStates (an `ObjectiveMechanism` and `Projections <Projection>` to it) and to control the specified parameters
(`ControlSignals <ControlSignal>` and corresponding `ControlProjections <ControlProjection>`) are created
automatically, as described below.
COMMENT


.. _ModulatoryMechanism_Modulatory_Signals:

*Specifying States to Modulate*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ModulatoryMechanism modulates the values of `States <State>` using the `ControlSignals <ControlSignal>` and/or
`GatingSignals <GatingSignal>` assigned to its `modulatory_signals <ModulatoryMechanism.modulatory_signals>` attribute,
each of which is assigned, respectively, a `ControlProjection` or `GatingProjection` to the State it modulates (see
`Modulation <ModulatorySignal_Modulation>` for a description of how modulation operates). Modulation of a State can
be specified either where the Component to which the State belongs is created (see specifying
`ControlSignal_Specification` or `GatingSignal_Specification`), or in the **modulatory_signals** argument of the
ModulatoryMechanism's constructor.  For the latter, the argument must be a specification for one or more
`ControlSignals <ControlSignal_Specification>` and/or `GatingSignals <GatingSignal_Specification>`.  States to be
modulate can also be added to an existing ModulatoryMechanism by using its `assign_params` method to add a
`ControlSignal` and/or `GatingSignal` for each additional State. All of a ModulatoryMechanism's `ModulatorySignals
<ModulatorySignal>` are listed in its `modulatory_signals <ModulatoryMechanism>` attribute.  Any ControlSignals are
also listed in the `control_signals <ModulatoryMechanism.control_signals>` attribute, and GatingSignals in the
`gating_signals <GatingMechanism.gating_signals>` attribute.  The projections from these to the States they modulate
are listed in the `modulatory_projections <ModulatoryMechanism.modulatory_projections>`, `control_projections
<ModulatoryMechanism.control_projections>`, and `gating_projections <ModulatoryMechanism.gating_projections>`
attributes, respectively.

COMMENT:
TBI:
If the ModulatoryMechanism is created as part of a `System`, the States to be modulated by it can be specified in
one of two ways:

  * in the **modulatory_signals** argument of the Composition's constructor, using one or more `ControlSignal`
    specifications <ControlSignal_Specification>` or `GatingSignal` specifications <GatingSignal_Specification>`

  * where the `parameter is specified <ParameterState_Specification>` to be controlled by a ControlSignal,
    where the `InputState <InputState_Specifiation>` or `OutputState  <OutputState_Specifiation>` to be gated is
    specified, or by including a `ControlProjection` or `GatingProjection` or
    `ControlSignal` in a `tuple specification <ParameterState_Tuple_Specification>` for the parameter,
    or `GatingSignal` in an `InputState tuple specification <InputState_Tuple_Specification>` or
    `OutputState tuple specification <OutputState_Tuple_Specification>` for the State to be gated.

When a ModulatoryMechanism is created as part of a Composition, a `ControlSignal` or `GatingSignal` is created and
assigned to the ModulatoryMechanism for every State of any `Component <Component>` in the Composition that has been
specified for control or gating using either of the methods above.
COMMENT


.. _ModulatoryMechanism_Composition_Controller:

*ModulatoryMechanisms and a Composition*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ModulatoryMechanism can be assigned to a `Composition` and executed just as any other Mechanism. It can also be
assigned as the `controller <Composition.controller>` of a `Composition`, that has a special relation
to the Composition: it is generally run either before or after all of the other Mechanisms in that Composition,
including any other ModulatoryMechanisms that belong to it (see `Composition_Controller`). A ModulatoryMechanism can
be the `controller <Composition.controller>` for only one Composition, and a Composition can have only one `controller
<Composition.controller>`.  A ModulatoryMechanism can be assigned as the `controller <Composition.controller>` for a
Composition in any of the following ways, by specifying:

    * the Composition in the **composition** argument of the ModulatoryMechanism's constructor;
    COMMENT:
    * the Composition in the **(composition** argument of the ModulatoryMechanism's `assign_as_controller
      <ModulatoryMechanism.assign_as_controller>` method;
    COMMENT
    * the ModulatoryMechanism in the **controller** argument of the Composition's constructor;
    * the ModulatoryMechanism in the **controller** argument of the Composition's `add_controller` method.


.. _ModulatoryMechanism_Structure:

Structure
---------

.. _ModulatoryMechanism_Input:

*Input*
~~~~~~~

A ModulatoryMechanism has a single `InputState`, named *OUTCOME*.  If its `objective_mechanism
<ModulatoryMechanism.objective_mechanism>` is implemented, then it receives a MappingProjection from that
ObjectiveMechanism's *OUTCOME* `OutputState <ObjectiveMechanism_Output>`, the value of which is also stored in the
ModulatoryMechanism's `outcome  <ModulatoryMechanism.outcome>` attribute.  This is used as the input to the
ModulatoryMechanism's `function <ModulatoryMechanism.function>`, that determines its `modulatory_allocation
<ModulatoryMechanism.modulatory_allocation>`.

.. _ModulatoryMechanism_Function:

*Function*
~~~~~~~~~~

A ModulatoryMechanism's `function <ModulatoryMechanism.function>` uses `outcome <ModulatoryMechanism.outcome>`
(the `value <InputState.value>` of its *OUTCOME* `InputState`) to generate a `modulatory_allocation
<ModulatoryMechanism.modulatory_allocation>`.  By default, `function <ModulatoryMechanism.function>` is assigned
the `DefaultAllocationFunction`, which takes a single value as its input, and assigns this as the value of
each item of `modulatory_allocation <ModulatoryMechanism.modulatory_allocation>`.  Each of these items is assigned as
the allocation for the corresponding  `ControlSignal` or `GatingSignal` in `modulatory_signals
<ModulatoryMechanism.modulatory_signals>`. Thus, by default, ModulatoryMechanism distributes its input as the
allocation to each of its `modulatory_signals  <ModulatoryMechanism.modulatory_signals>. However, this behavior can
be modified either by specifying a different `function <ModulatoryMechanism.function>`, and/or by specifying that
individual ControlSignals and/or GatingSignals reference different items in `modulatory_allocation` as their
allocation (i.e., the value of their `variable <ModulatorySignal.variable>`.

.. _ModulatoryMechanism_Output:

*Output*
~~~~~~~~

The OutputStates of a ModulatoryMechanism are `ControlSignals <ControlSignal>` and/or `GatingSignals <GatingSignal>`
(listed in its `modulatory_signals <ModulatoryMechanism.modulatory_signals>` attribute) that send `ControlProjection
<ControlProjections>` and/or `GatingProjections <GatingProjection>` to the corresponding States. The `States <State>`
modulated by a ModulatoryMechanism's `modulatory_signals <ModulatoryMechanism.modulatory_signals>` can be displayed
using its `show <ModulatoryMechanism.show>` method. By default, each item of the ModulatoryMechanism's
`modulatory_allocation <ModulatoryMechanism.modulatory_allocation>` attribute is assigned to the `variable` of the
corresponding `ControlSignal` or `GatingSignal` in its `modulatory_signals <ModulatoryMechanism.modulatory_signals>`
attribute;  however, subtypes of ModulatoryMechanism may assign values differently.  The allocations to any
ControlSignals are also listed in the `control_signals <ModulatoryMechanism.control_signals>` attribute; and the
allocations to any GatingSignals are listed in the `gating_signals <ModulatoryMechanism.gating_signals>` attribute.

.. _ModulatoryMechanism_Costs:

*Costs and Net Outcome*

If a ModulatoryMechanism has any `ControlSignals <ControlSignal>` in its `modulatory_signals
<ModulatoryMechanism.modulatory_signals>`, then it also computes the combined `costs <ModulatoryMechanism.costs>` of
those, and a `net_outcome <ModulatoryMechanism.net_outcome>` based on them (see `below
<ModulatoryMechanism_Costs_Computation>`). This is used by some subclasses of ModulatoryMechanism (e.g.,
`OptimizationControlMechanism`) to compute the `modulatory_allocation <ModulatoryMechanism.modulatory_allocation>`.
These are computed using the ModulatoryMechanism's default `compute_reconfiguration_cost
<ModulatoryMechanism.compute_reconfiguration_cost>`, `combine_costs <ModulatoryMechanism.combine_costs>`,
and `compute_net_outcome <ModulatoryMechanism.compute_net_outcome>` functions, but these can also be assigned custom
functions (see links to attributes for details).


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
   `ParameterStates <ParameterState>` that receive `ControlProjections <ControlProjection>`, and `InputStates
   <InputState>` and/or `OutputStates <OutputState>` that receive `GatingProjections <GatingProjection>`, do not
   update their values until their owner Mechanisms execute (see `Lazy Evaluation <LINK>` for an explanation of
   "lazy" updating).  This means that even if a ModulatoryMechanism has executed, the States that it modulates will
   not assume their new values until the Mechanisms to which they belong have executed.

.. _ModulatoryMechanism_Costs_Computation:

*Computation of Costs and Net_Outcome*

When a ModulatoryMechanism updates the `intensity <ControlSignal.intensity>` of any ControlSignals in its
`modulatory_signals <ModulatoryMechanism.modulatory_signals>`, each of those ControlSignals calculates a `cost
<ControlSignal.cost>`, based on its `intensity  <ControlSignal/intensity>`.  The ModulatoryMechanism computes a
`reconfiguration_cost  <ModulatoryMechanism.reconfiguration_cost>` based on these,  using its
`compute_reconfiguration_cost <ModulatoryMechanism.compute_reconfiguration_cost>` function.  It then  combines this
with the `cost  <ControlSignal.cost>` of its individual ControlSignals, using its `combine_costs
<ModulatoryMechanism.combine_costs>` function, and assigns the result to its `costs <ModulatoryMechanism.costs>`
attribute.  The ModulatoryMechanism uses this, together with its `outcome <ModulatoryMechanism.outcome>` attribute,
to compute a  `net_outcome <ModulatoryMechanism.net_outcome>` using its `compute_net_outcome
<ModulatoryMechanism.compute_net_outcome>` function.  This is used by some subclasses of ModulatoryMechanism
(e.g., `OptimizationControlMechanism`) to  compute its `modulatory_allocation
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
    ...                          name="My Control Mech")


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

See `System_Control_Examples` for examples of how a ModulatoryMechanism, the OutputStates its
`objective_mechanism <ControlSignal.objective_mechanism>`, and its `control_signals <ModulatoryMechanism.control_signals>`
can be specified for a System.


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

from psyneulink.core.components.functions.function import \
    Function_Base, ModulationParam, _is_modulation_param, is_function_type
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
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.defaults import defaultControlAllocation, defaultGatingAllocation
from psyneulink.core.globals.keywords import AUTO_ASSIGN_MATRIX, CONTEXT, \
    CONTROL, CONTROL_PROJECTIONS, CONTROL_SIGNALS, \
    EID_SIMULATION, GATING_SIGNALS, INIT_EXECUTE_METHOD_ONLY, MODULATORY_SIGNALS, MONITOR_FOR_MODULATION, \
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

def _control_allocation_getter(owning_component=None, execution_id=None):
    try:
        control_signal_indices = [owning_component.modulatory_signals.index(c)
                                  for c in owning_component.control_signals]
        return np.array([owning_component.modulatory_allocation[i] for i in control_signal_indices])
    except TypeError:
        return defaultControlAllocation

def _control_allocation_setter(value, owning_component=None, execution_id=None):
    control_signal_indices = [owning_component.modulatory_signals.index(c)
                              for c in owning_component.control_signals]
    if len(value)!=len(control_signal_indices):
        raise ModulatoryMechanismError(f"Attempt to set '{CONTROL_ALLOCATION}' parameter of {owning_component.name} "
                                       f"with value ({value} that has a different length ({len(value)}) "
                                       f"than the number of its {CONTROL_SIGNALS} ({len(control_signal_indices)})")
    mod_alloc = owning_component.parameters.modulatory_allocation.get(execution_id)
    for j, i in enumerate(control_signal_indices):
        mod_alloc[i] = value[j]
    owning_component.parameters.modulatory_allocation.set(np.array(mod_alloc), execution_id)
    return value

def _gating_allocation_getter(owning_component=None, execution_id=None):
    # try:
    #     return np.array([c.parameters.variable.get(execution_id) for c in owning_component.gating_signals])
    # except TypeError:
    #     return defaultGatingAllocation
    try:
        gating_signal_indices = [owning_component.modulatory_signals.index(g)
                                  for g in owning_component.gating_signals]
        return np.array([owning_component.modulatory_allocation[i] for i in gating_signal_indices])
    # MODIFIED 5/18/19 OLD:
    except (TypeError):
    # # MODIFIED 5/18/19 NEW: [JDC]
    # except (TypeError, ValueError):
    # MODIFIED 5/18/19 END
        return defaultGatingAllocation

def _gating_allocation_setter(value, owning_component=None, execution_id=None):
    # for c in owning_component.gating_signals:
    #     c.parameters.variable.set(value, execution_id)
    # return value
    gating_signal_indices = [owning_component.modulatory_signals.index(c)
                              for c in owning_component.gating_signals]
    if len(value)!=len(gating_signal_indices):
        raise ModulatoryMechanismError(f"Attempt to set {GATING_ALLOCATION} parameter of {owning_component.name} "
                                       f"with value ({value} that has a different length than the number of its"
                                       f"{GATING_SIGNALS} ({len(gating_signal_indices)})")
    mod_alloc = owning_component.parameters.modulatory_allocation.get(execution_id)
    for j, i in enumerate(gating_signal_indices):
        mod_alloc[i] = value[j]
    owning_component.parameters.modulatory_allocation.set(np.array(mod_alloc), execution_id)
    return value

def _modulatory_mechanism_costs_getter(owning_component=None, execution_id=None):
    # NOTE: In cases where there is a reconfiguration_cost, that cost is not returned by this method
    try:
        costs = [c.compute_costs(c.parameters.variable.get(execution_id), execution_id=execution_id)
                 for c in owning_component.control_signals]
        return costs

    except TypeError:
        return None

def _outcome_getter(owning_component=None, execution_id=None):
    try:
        return owning_component.parameters.variable.get(execution_id)[0]
    except TypeError:
        return None

def _net_outcome_getter(owning_component=None, execution_id=None):
    # NOTE: In cases where there is a reconfiguration_cost,
    # that cost is not included in the net_outcome

    try:
        c = owning_component
        return c.compute_net_outcome(c.parameters.outcome.get(execution_id),
                                     c.combine_costs(c.parameters.costs.get(execution_id)))
    except TypeError:
        return [0]

class DefaultAllocationFunction(Function_Base):
    '''Take a single 1d item and return a 2d array with n identical items
    Takes the default input (a single value in the *OUTCOME* InputState of the ModulatoryMechanism),
    and returns the same allocation for each of its `modulatory_signals <ModulatoryMechanism.modulatory_signals>`.
    '''
    componentName = 'Default Modulatory Function'
    class Parameters(Function_Base.Parameters):
        num_modulatory_signals = Parameter(1, stateful=False)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 params=None
                 ):
        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(params=params)
        super().__init__(default_variable=default_variable,
                         params=params,
                         context=ContextFlags.CONSTRUCTOR)

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)
        num_mod_sigs = self.get_current_function_param('num_modulatory_signals')
        result = np.array([variable[0]] * num_mod_sigs)
        return self.convert_output_type(result)


class ModulatoryMechanism(AdaptiveMechanism_Base):
    """
    ModulatoryMechanism(                                         \
        system=None                                              \
        monitor_for_modulation=None,                             \
        objective_mechanism=None,                                \
        function=DefaultAllocationFunction,                      \
        modulatory_signals=None,                                 \
        modulation=ModulationParam.MULTIPLICATIVE                \
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

    objective_mechanism : ObjectiveMechanism or List[OutputState specification] : default None
        specifies either an `ObjectiveMechanism` to use for the ModulatoryMechanism, or a list of the OutputStates it
        should monitor; if a list of `OutputState specifications <ObjectiveMechanism_Monitor>` is used,
        a default ObjectiveMechanism is created and the list is passed to its **monitor** argument.

    function : TransferFunction : default Linear(slope=1, intercept=0)
        specifies function used to combine values of monitored OutputStates.

    modulatory_signals : ControlSignal specification or List[ControlSignal specification, ...]
        specifies the parameters to be controlled by the ModulatoryMechanism; a `ControlSignal` is created for each
        (see `ControlSignal_Specification` for details of specification).

    modulation : ModulationParam : ModulationParam.MULTIPLICATIVE
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

    system : System_Base
        The `System` for which the ModulatoryMechanism is a `controller <System>`.  Note that this is distinct from
        a Mechanism's `systems <Mechanism_Base.systems>` attribute, which lists all of the Systems to which a
        `Mechanism` belongs -- a ModulatoryMechanism can belong to but not be the `controller of a Composition
        <ModulatoryMechanism_Composition_Controller>`.

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
        the `value <InputState.value>` of the ModulatoryMechanism's `primary InputState <InputState_Primary>`,
        which receives its `Projection <Projection>` from the *OUTCOME* `OutputState` of its `objective_mechanism
        <ModulatoryMechanism.objective_mechanism>`.

    function : TransferFunction : default Linear(slope=1, intercept=0)
        determines how the `value <OuputState.value>` \\s of the `OutputStates <OutputState>` specified in the
        **monitor_for_modulation** argument of the ModulatoryMechanism's constructor are used to generate its
        `modulatory_allocation <ModulatoryMechanism.modulatory_allocation>`.

    modulatory_signals : ContentAddressableList[ModulatorySignal]
        list of the ModulatoryMechanisms `ControlSignals <ControlSignals>` and `GatingSignals <GatingSignals>`,
        including any inherited from a `system <ModulatoryMechanism.system>` for which it is a `controller
        <System.controller>`.  This is the same as the ModulatoryMechanism's `output_states
        <Mechanism_Base.output_states>` attribute).

    modulatory_allocation : 2d array
        contains allocations for all the ModulatoryMechanism's `modulatory_signals
        <ModulatoryMechanism.modulatory_signals>`;  each item is the value assigned as the `allocation` for a
        `ControlSignal` (listed in the `control_signals  <ModulatoryMechanism.control_signals>` attribute,
        or a `GatingSignal` (listed in the `gating_signals <ModulatoryMechanism.gating_signals>` attribute (these
        are also listed in the `control_allocation <ModulatoryMechanism.control_allocation>` and `gating_allocation
        <ModulatoryMechanism.gating_allocation>` attributes, respectively.  The modulatory_allocation is the same as
        the ModulatoryMechanism's `value <Mechanism_Base.value>` attribute).

    control_signals : ContentAddressableList[ControlSignal]
        list of the `ControlSignals <ControlSignals>` for the ModulatoryMechanism, including any inherited from a
        `system <ModulatoryMechanism.system>` for which it is a `controller <System.controller>`.

    control_allocation : 2d array
        each item is the value assigned as the `allocation <ControlSignal.allocation>` for the corresponding
        ControlSignal listed in the `control_signals <ModulatoryMechanism.control_signals>` attribute.

    gating_signals : ContentAddressableList[GatingSignal]
        list of the `GatingSignals <ControlSignals>` for the ModulatoryMechanism, including any inherited from a
        `system <ModulatoryMechanism.system>` for which it is a `controller <System.controller>`.

    gating_allocation : 2d array
        each item is the value assigned as the `allocation <GatingSignal.allocation>` for the corresponding
        GatingSignal listed in the `gating_signals` attribute <ModulatoryMechanism.gating_signals>`.

    compute_reconfiguration_cost : Function, function or method
        function used to compute the ModulatoryMechanism's `reconfiguration_cost  <ModulatoryMechanism.reconfiguration_cost>`;
        result is a scalar value representing the difference — defined by the function — between the values of the
        ModulatoryMechanism's current and last `control_alloction <ModulatoryMechanism.control_allocation>`, that can be
        accessed by `reconfiguration_cost <ModulatoryMechanism.reconfiguration_cost>` attribute.

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

                    :default value: numpy.array([1.])
                    :type: numpy.ndarray

                outcome
                    see `outcome <ModulatoryMechanism.outcome>

                    :default value: None
                    :type:
                    :read only: True

                control_allocation
                    see `control_allocation <ModulatoryMechanism.control_allocation>

                    :default value: defaultControlAllocation
                    :type:
                    :read only: True

                gating_allocation
                    see `gating_allocation <ModulatoryMechanism.gating_allocation>

                    :default value: defaultGatingAllocation
                    :type:
                    :read only: True

                costs
                    see `costs <ModulatoryMechanism.costs>`

                    :default value: None
                    :type:
                    :read only: True

                control_signal_costs
                    see `control_signal_costs <ModulatoryMechanism.control_signal_costs>`

                    :default value: None
                    :type:
                    :read only: True

                compute_reconfiguration_cost
                     see 'compute_reconfiguration_cost <ModulatoryMechanism.compute_reconfiguration_cost>`

                reconfiguration_cost
                     see 'reconfiguration_cost <ModulatoryMechanism.reconfiguration_cost>`

                combine_costs
                    see `combine_costs <ModulatoryMechanism.combine_costs>`

                    :default value: numpy.core.fromnumeric.sum
                    :type: <class 'function'>

                compute_net_outcome
                    see `compute_net_outcome <ModulatoryMechanism.compute_net_outcome>`

                    :default value: lambda outcome, cost: outcome - cost
                    :type: <class 'function'>

                modulation
                    see `modulation <ModulatoryMechanism.modulation>`

                    :default value: ModulationParam.MULTIPLICATIVE
                    :type: `ModulationParam`

                net_outcome
                    see `net_outcome <ModulatoryMechanism.net_outcome>`

                    :default value: None
                    :type:
                    :read only: True

        """
        # This must be a list, as there may be more than one (e.g., one per control_signal)
        variable = np.array([defaultControlAllocation])
        value = Parameter(np.array(defaultControlAllocation), aliases='modulatory_allocation')
        control_allocation = Parameter(np.array(defaultControlAllocation),
                                       getter=_control_allocation_getter,
                                       setter=_control_allocation_setter,
                                       read_only=True)
        gating_allocation = Parameter(np.array(defaultGatingAllocation),
                                      getter=_gating_allocation_getter,
                                      setter=_gating_allocation_setter,
                                      read_only=True)
        outcome = Parameter(None, read_only=True, getter=_outcome_getter)

        compute_reconfiguration_cost = Parameter(None, stateful=False, loggable=False)
        # reconfiguration_cost = Parameter(None, read_only=True, getter=_reconfiguration_cost_getter)

        combine_costs = Parameter(np.sum, stateful=False, loggable=False)
        costs = Parameter(None, read_only=True, getter=_modulatory_mechanism_costs_getter)
        control_signal_costs = Parameter(None, read_only=True)

        compute_net_outcome = Parameter(lambda outcome, cost: outcome - cost, stateful=False, loggable=False)
        net_outcome = Parameter(None, read_only=True,
                                getter=_net_outcome_getter)

        simulation_ids = Parameter([], user=False)

        modulation = ModulationParam.MULTIPLICATIVE

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
                 objective_mechanism=None,
                 function=None,
                 modulatory_signals:tc.optional(tc.any(is_iterable,
                                                       ParameterState,
                                                       InputState,
                                                       OutputState,
                                                       ControlSignal,
                                                       GatingSignal))=None,
                 modulation:tc.optional(_is_modulation_param)=ModulationParam.MULTIPLICATIVE,
                 combine_costs:is_function_type=np.sum,
                 compute_reconfiguration_cost:tc.optional(is_function_type)=None,
                 compute_net_outcome:is_function_type=lambda outcome, cost : outcome - cost,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs
                 ):

        if kwargs:
                for k in kwargs.keys():
                    if CONTEXT in k:
                        context=kwargs[CONTEXT]
                        continue
                    raise ModulatoryMechanismError("Unrecognized arg in constructor for {}: {}".
                                                format(self.__class__.__name__, repr(i)))

        function = function or DefaultAllocationFunction
        modulatory_signals = modulatory_signals or []
        if not isinstance(modulatory_signals, list):
            modulatory_signals = [modulatory_signals]
        self.combine_costs = combine_costs
        self.compute_net_outcome = compute_net_outcome
        self.compute_reconfiguration_cost = compute_reconfiguration_cost

        # If the user passed in True for objective_mechanism, means one needs to be created automatically.
        # Set it to None to signal this downstream (vs. False which means *don't* create one),
        #    while still causing tests to indicate that one does NOT yet exist
        #    (since actual assignment of one registers as True).
        if objective_mechanism is True:
            objective_mechanism = None

        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(system=system,
                                                  monitor_for_modulation=monitor_for_modulation,
                                                  objective_mechanism=objective_mechanism,
                                                  function=function,
                                                  modulatory_signals=modulatory_signals,
                                                  # control_signals=None,
                                                  # gating_signals=None,
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
                                               context=ContextFlags.CONSTRUCTOR)

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

            if not isinstance(target_set[OBJECTIVE_MECHANISM], (ObjectiveMechanism, list)):
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
                self._objective_mechanism = ObjectiveMechanism(monitor=monitored_output_states,
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

        # If ModulatoryMechanism is a System controller, name Projection from
        # ObjectiveMechanism based on the System
        if self.system is not None:
            name = self.system.name + ' outcome signal'
        # Otherwise, name it based on the ObjectiveMechanism
        else:
            name = self.objective_mechanism.name + ' outcome signal'

        projection_from_objective = MappingProjection(sender=self.objective_mechanism,
                                                      receiver=self,
                                                      matrix=AUTO_ASSIGN_MATRIX,
                                                      name=name)
        for input_state in self.objective_mechanism.input_states:
            input_state.internal_only = True

        self.aux_components.append(self.objective_mechanism)
        self.aux_components.append((projection_from_objective, True))
        self._objective_projection = projection_from_objective
        self.monitor_for_modulation = self.monitored_output_states

    def _instantiate_input_states(self, context=None):
        super()._instantiate_input_states(context=context)
        self.input_state.name = OUTCOME

        # IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
        # if self.monitor_for_control or self._objective_mechanism:
        if self._objective_mechanism:
            self._instantiate_objective_mechanism(context=context)

    def _instantiate_output_states(self, context=None):
        from psyneulink.core.globals.registry import register_category
        from psyneulink.core.components.states.state import State_Base

        # Create registry for ControlSignals (to manage names)
        register_category(entry=ControlSignal,
                          base_class=State_Base,
                          registry=self._stateRegistry,
                          context=context)

    # ---------------------------------------------------
    # FIX 5/23/17: PROJECTIONS AND PARAMS SHOULD BE PASSED BY ASSIGNING TO STATE SPECIFICATION DICT
    # FIX          UPDATE parse_state_spec TO ACCOMODATE (param, ControlSignal) TUPLE
    # FIX          TRACK DOWN WHERE PARAMS ARE BEING HANDED OFF TO ControlProjection
    # FIX                   AND MAKE SURE THEY ARE NOW ADDED TO ControlSignal SPECIFICATION DICT
    # ---------------------------------------------------

        if self.modulatory_signals:
            self._output_states = []
            self.defaults.value = None

            for modulatory_signal in self.modulatory_signals:
                self._instantiate_modulatory_signal(modulatory_signal, context=context)

        super()._instantiate_output_states(context=context)

        # Reassign modulatory_signals, control_signals and gating_signals to backing fields of corresponding params
        # to capture any user_defined ControlSignals and/or GatingSignals instantiated in call to super
        # and assign to ContentAddressableLists
        self._modulatory_signals = ContentAddressableList(component_type=ModulatorySignal,
                                                       list=[state for state in self.output_states
                                                             if isinstance(state, (ControlSignal, GatingSignal))])

        # If the ModulatoryMechanism's control_allocation has more than one item,
        #    warn if the number of items does not equal the number of its ControlSignals
        #    (note:  there must be fewer ControlSignals than items in control_allocation,
        #            as the reverse is an error that is checked for in _instantiate_control_signal)
        if len(self.defaults.value) > 1 and len(self.modulatory_signals) != len(self.defaults.value):
            if self.verbosePref:
                warnings.warning("The number of {}s for {} ({}) does not equal the number of items in its {} ({})".
                                 format(ControlSignal.__name__, self.name, len(self.modulatory_signals),
                                        MODULATORY_ALLOCATION, len(self.defaults.value)))

    def _instantiate_modulatory_signal(self,  modulatory_signal, context=None):
        '''Parse and instantiate modulatory_signal specifications (in call to State._parse_state_spec)
           and any embedded Projection specifications (in call to <State>._instantiate_projections)

        Temporarily assign variable to default allocation value to avoid chicken-and-egg problem:
           value, output_states and modulatory_signals haven't been expanded yet to accomodate the new
           ModulatorySignal; reassign modulatory_signal.variable to actual OWNER_VALUE below,
           once value has been expanded
        '''
        from psyneulink.core.components.states.state import _instantiate_state
        from psyneulink.core.components.projections.projection import ProjectionError

        # Try to instantiate as ControlSignal;  if that fails, try GatingSignal
        mod_spec = modulatory_signal
        try:
            modulatory_signal = _instantiate_state(state_type=ControlSignal,
                                                   owner=self,
                                                   variable=defaultControlAllocation,
                                                   reference_value=ControlSignal.defaults.allocation,
                                                   modulation=self.modulation,
                                                   state_spec=mod_spec,
                                                   context=context)
            if not type(modulatory_signal) in convert_to_list(self.outputStateTypes):
                raise ProjectionError(f'{type(modulatory_signal)} inappropriate for {self.name}')

        except ProjectionError:
            modulatory_signal = _instantiate_state(state_type=GatingSignal,
                                                   owner=self,
                                                   variable=defaultGatingAllocation,
                                                   reference_value=GatingSignal.defaults.allocation,
                                                   modulation=self.modulation,
                                                   state_spec=mod_spec,
                                                   context=context)
        modulatory_signal.owner = self

        if isinstance(modulatory_signal, ControlSignal):
            # Update control_signal_costs to accommodate instantiated Projection
            control_signal_costs = self.parameters.control_signal_costs.get()
            try:
                control_signal_costs = np.append(control_signal_costs, np.zeros((1, 1)), axis=0)
            except (AttributeError, ValueError):
                control_signal_costs = np.zeros((1, 1))
            self.parameters.control_signal_costs.set(control_signal_costs, override=True)

        # UPDATE output_states AND modulatory_projections -------------------------------------------------------------

        # TBI: For modulatory mechanisms that accumulate, starting output must be equal to the initial "previous value"
        # so that modulation that occurs BEFORE the control mechanism executes is computed appropriately
        # if (isinstance(self.function, IntegratorFunction)):
        #     control_signal._intensity = function.initializer

        # Add ModulatorySignal to output_states list
        self._output_states.append(modulatory_signal)

        # FIX: 5/18/19 - NEEDS TO GET APPROPRIATE DEFAULT FOR TYPE BEING IMPLEMENTED
        # FIX: 5/18/19 - NEEDS TO ACCOMODATE owner_value_index WHICH MIGHT REMAIN 0 (THE WAY OLD GATINGMECHANISM DID)
        # # MODIFIED 5/18/19 OLD:
        # # since output_states is exactly control_signals is exactly the shape of value, we can just construct it here
        # self.defaults.value = np.array([[ControlSignal.defaults.allocation]
        #                                          for i in range(len(self._output_states))])
        # self.parameters.value.set(copy.deepcopy(self.defaults.value))
        #
        # # Assign ModulatorySignal's variable to index of owner's value
        # modulatory_signal._variable_spec = [(OWNER_VALUE, len(self.defaults.value) - 1)]
        # if not isinstance(modulatory_signal.owner_value_index, int):
        #     raise ModulatoryMechanismError(
        #             "PROGRAM ERROR: The \'owner_value_index\' attribute for {} of {} ({})is not an int."
        #                 .format(modulatory_signal.name, self.name, modulatory_signal.owner_value_index))
        # # Validate index
        # try:
        #     self.defaults.value[modulatory_signal.owner_value_index]
        # except IndexError:
        #     raise ModulatoryMechanismError("Index specified for {} of {} ({}) "
        #                                    "exceeds the number of items of its {} ({})".
        #                                    format(modulatory_signal.__class__.__name__, self.name,
        #                                           modulatory_signal.owner_value_index,
        #                                           MODULATORY_ALLOCATION, len(self.defaults.value)
        #         )
        #     )
        # # MODIFIED 5/18/19 NEW: [JDC]
        # # Adjust shape of ModulatoryMechanism's value parameter to accomodate modulatory_signal
        # # - check modulatory_signal's owner_value_index (the value of the ModulatoryMechanism to which it refers)
        # # - if it's index is within the range of number of items already on value, then leave as is
        # # - if it's index is outside the range, then add a new item and reassign value accordingly
        # self.defaults.value = np.array([[type(modulatory_signal).defaults.allocation]
        #                                          for i in range(len(self._output_states))])
        # self.parameters.value.set(copy.deepcopy(self.defaults.value))
        #
        # # Assign ModulatorySignal's variable to index of owner's value
        # modulatory_signal._variable_spec = [(OWNER_VALUE, len(self.defaults.value) - 1)]
        # if not isinstance(modulatory_signal.owner_value_index, int):
        #     raise ModulatoryMechanismError(
        #             "PROGRAM ERROR: The \'owner_value_index\' attribute for {} of {} ({})is not an int."
        #                 .format(modulatory_signal.name, self.name, modulatory_signal.owner_value_index))
        # # Validate index
        # try:
        #     self.defaults.value[modulatory_signal.owner_value_index]
        # except IndexError:
        #     raise ModulatoryMechanismError("Index specified for {} of {} ({}) "
        #                                    "exceeds the number of items of its {} ({})".
        #                                    format(modulatory_signal.__class__.__name__, self.name,
        #                                           modulatory_signal.owner_value_index,
        #                                           MODULATORY_ALLOCATION, len(self.defaults.value)
        #         )
        #     )
        # MODIFIED 5/18/19 NEWER:
        # # Adjust shape of ModulatoryMechanism's value parameter to accomodate modulatory_signal
        self.defaults.value = np.array([[type(modulatory_signal).defaults.allocation]
                                                 for i in range(len(self._output_states))])
        self.parameters.value.set(copy.deepcopy(self.defaults.value))
        if isinstance(self.function, DefaultAllocationFunction):
            self.function.num_modulatory_signals = len(self._output_states)

        # Assign ModulatorySignal's variable to index of owner's value
        modulatory_signal._variable_spec = [(OWNER_VALUE, len(self.defaults.value) - 1)]
        if not isinstance(modulatory_signal.owner_value_index, int):
            raise ModulatoryMechanismError(
                    "PROGRAM ERROR: The \'owner_value_index\' attribute for {} of {} ({})is not an int."
                        .format(modulatory_signal.name, self.name, modulatory_signal.owner_value_index))
        # Validate index
        try:
            self.defaults.value[modulatory_signal.owner_value_index]
        except IndexError:
            raise ModulatoryMechanismError("Index specified for {} of {} ({}) "
                                           "exceeds the number of items of its {} ({})".
                                           format(modulatory_signal.__class__.__name__, self.name,
                                                  modulatory_signal.owner_value_index,
                                                  MODULATORY_ALLOCATION, len(self.defaults.value)
                )
            )
        # MODIFIED 5/18/19 END

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

        if self.control_signals:
            print ("\n\tControlling the following Mechanism parameters:".format(self.name))
            # Sort for consistency of output:
            state_names_sorted = sorted(self.control_signals.names)
            for state_name in state_names_sorted:
                for projection in self.control_signals[state_name].efferents:
                    print ("\t\t{0}: {1}".format(projection.receiver.owner.name, projection.receiver.name))

        if self.gating_signals:
            print ("\n\tGating the following States:".format(self.name))
            # Sort for consistency of output:
            state_names_sorted = sorted(self.gating_signals.names)
            for state_name in state_names_sorted:
                for projection in self.gating_signals[state_name].efferents:
                    print ("\t\t{0}: {1}".format(projection.receiver.owner.name, projection.receiver.name))

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
        if self._objective_mechanism:
            self.objective_mechanism._add_process(process, role)

    @tc.typecheck
    def assign_as_controller(self, system:System_Base, context=ContextFlags.COMMAND_LINE):
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

        if context == ContextFlags.COMMAND_LINE:
            system.controller = self
            return

        if self._objective_mechanism is None:
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
        self._objective_mechanism.for_controller = True

        if context != ContextFlags.PROPERTY:
            system._controller = self

        self._activate_projections_for_compositions(system)

    def _activate_projections_for_compositions(self, compositions=None):

        if self.objective_mechanism:
            self._objective_projection._activate_for_compositions(compositions)

        for ms in self.modulatory_signals:
            for eff in ms.efferents:
                eff._activate_for_compositions(compositions)

        # assign any deferred init objective mech monitored output state projections to this system
        if self.objective_mechanism:
            for output_state in self.objective_mechanism.monitored_output_states:
                for eff in output_state.efferents:
                    eff._activate_for_compositions(compositions)

        for eff in self.efferents:
            eff._activate_for_compositions(compositions)

        if self.objective_mechanism:
            for aff in self._objective_mechanism.afferents:
                aff._activate_for_compositions(compositions)

    def _apply_control_allocation(self, control_allocation, runtime_params, context, execution_id=None):
        self._apply_modulatory_allocation(self, control_allocation, runtime_params, context, execution_id=None)

    def _apply_modulatory_allocation(self, modulatory_allocation, runtime_params, context, execution_id=None):
        '''Update values to `modulatory_signals <ModulatoryMechanism.modulatory_signals>`
        based on specified `modulatory_allocation <ModulatoryMechanism.modulatory_allocation>`
        (used by controller of a Composition in simulations)
        '''
        value = [np.atleast_1d(a) for a in modulatory_allocation]
        self.parameters.value.set(value, execution_id)
        self._update_output_states(execution_id=execution_id,
                                   runtime_params=runtime_params,
                                   context=ContextFlags.COMPOSITION)

    @property
    def monitored_output_states(self):
        try:
            return self._objective_mechanism.monitored_output_states
        except AttributeError:
            return None

    @monitored_output_states.setter
    def monitored_output_states(self, value):
        try:
            self._objective_mechanism._monitored_output_states = value
        except AttributeError:
            return None

    @property
    def monitored_output_states_weights_and_exponents(self):
        try:
            return self._objective_mechanism.monitored_output_states_weights_and_exponents
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

    def get_next_sim_id(self, execution_id):
        with self._sim_count_lock:
            try:
                sim_num = self._sim_counts[execution_id]
                self._sim_counts[execution_id] += 1
            except KeyError:
                sim_num = 0
                self._sim_counts[execution_id] = 1

        return '{0}{1}-{2}'.format(execution_id, EID_SIMULATION, sim_num)

    @property
    def _dependent_components(self):
        return list(itertools.chain(
            super()._dependent_components,
            # [self.objective_mechanism],
            [self._objective_mechanism] if self.objective_mechanism else [],
        ))
