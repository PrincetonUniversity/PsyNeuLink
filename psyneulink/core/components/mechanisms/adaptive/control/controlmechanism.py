# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  ControlMechanism ************************************************

"""
Overview
--------

A ControlMechanism is an `AdaptiveMechanism <AdaptiveMechanism>` that modifies the parameter(s) of one or more
`Components <Component>`, in response to an evaluative signal received from an `ObjectiveMechanism`.  The
ObjectiveMechanism monitors a specified set of OutputStates, and from these generates the evaluative signal that is
used by the ControlMechanism's `function <ControlMechanism.function>` to calculate an `control_allocation
<ControlMechanism.control_allocation>`: a list of `allocation <ControlSignal.allocation>` values for each of its
`ControlSignals <ControlSignal>`.  Each ControlSignal uses its `allocation <ControlSignal.allocation>` to calculate its
`intensity`, which is then transmitted by the ControlSignal's `ControlProjection(s) <ControlProjection>` to the
`ParameterState(s) <ParameterState>` to which they project.  Each ParameterState uses the value received by a
ControlProjection to modify the value of the parameter for which it is responsible (see `ModulatorySignal_Modulation`
for a more detailed description of how modulation operates).  A ControlMechanism can regulate only the parameters of
Components in the `System` to which it belongs. The OutputStates used to determine the ControlMechanism's
`control_allocation <ControlMechanism.control_allocation>`, the `ObjectiveMechanism` used to evalute these, and the
parameters controlled by the ControlMechanism can be listed using its `show <ControlMechanism.show>` method.

COMMENT:
    ALTERNATE VERSION
    and has a `ControlSignal` for each parameter of the Components in the `system <EVCControlMechanism.system>` that it
    controls.  Each ControlSignal is associated with a `ControlProjection` that regulates the value of the parameter it
    controls, with the magnitude of that regulation determined by the ControlSignal's `intensity`.  A particular
    combination of ControlSignal `intensity` values is called an `control_allocation`. When a `System` is executed that
    uses an EVCControlMechanism as its `controller <System.controller>`, it concludes by executing the EVCControlMechanism, which
    determines its `control_allocation` for the next `TRIAL`.  That, in turn, determines the `intensity` for each of the
    ControlSignals, and therefore the values of the parameters they control on the next `TRIAL`. The OutputStates used
    to determine an EVCControlMechanism's `control_allocation <EVCControlMechanism.control_allocation>` and the parameters it
    controls can be listed using its `show <EVCControlMechanism.show>` method.
COMMENT

.. _ControlMechanism_System_Controller:

*ControlMechanisms and a System*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ControlMechanism can be assigned to a `Process` and executed within one or more Systems, just like any other
Mechanism. It can also be assigned as the `controller <System.controller>` of a `System`, that has a special relation
to the System: it is used to control all of the parameters that have been `specified for control
<ControlMechanism_Control_Signals>` in that System.  A ControlMechanism can be the `controller <System.controller>`
for only one System, and a System can have only one `controller <System.controller>`.  The System's `controller
<System.controller>` is executed after all of the other Components in the System have been executed, including any
other ControlMechanisms that belong to it (see `System Execution <System_Execution>`).  A ControlMechanism can be
assigned as the `controller <System.controller>` for a System by specifying it in the **controller** argument of the
System's constructor, or by specifying the System as the **system** argument of either the ControlMechanism's
constructor or its `assign_as_controller <ControlMechanism.assign_as_controller>` method. A System's `controller
<System.controller>` and its associated Components can be displayed using the System's `show_graph
<System.show_graph>` method with its **show_control** argument assigned as `True`.


.. _ControlMechanism_Creation:

Creating a ControlMechanism
---------------------------

A ControlMechanism can be created using the standard Python method of calling the constructor for the desired type.
A ControlMechanism is also created automatically whenever a `System is created <System_Creation>`, and the
ControlMechanism class or one of its subtypes is specified in the **controller** argument of the System's constructor
(see `System_Creation`).  If the ControlMechanism is created explicitly (using its constructor), it must be included
in a `Process` assigned to the System.  The `OutputStates <OutputState>` monitored by its `ObjectiveMechanism` are
specified in the **monitor_for_control** argument of its constructor, and the parameters it controls are specified in
the **control_signals** argument; an ObjectiveMechanism is automatically created that monitors and evaluates the
specified OutputStates.  The ObjectiveMechanism can also be explicitly specified in the **objective_mechanism**
argument of the ControlMechanism's constructor (see `below <ControlMechanism_ObjectiveMechanism>`). If the
ControlMechanism is created automatically by a System (as its `controller <System.controller>`), then the specification
of OutputStates to be monitored and parameters to be controlled are made on the System and/or the Components
themselves (see `System_Control_Specification`).  In either case, the Components needed to monitor the specified
OutputStates (an `ObjectiveMechanism` and `Projections <Projection>` to it) and to control the specified parameters
(`ControlSignals <ControlSignal>` and corresponding `ControlProjections <ControlProjection>`) are created
automatically, as described below.

.. _ControlMechanism_ObjectiveMechanism:

*ObjectiveMechanism*
~~~~~~~~~~~~~~~~~~~~

Whenever a ControlMechanism is created, it automatically creates an `ObjectiveMechanism` that monitors and evaluates
the `value <OutputState.value>`\\(s) of a set of `OutputState(s) <OutputState>`; this evaluation is used to determine
the ControlMechanism's `control_allocation <ControlMechanism.control_allocation>`. The ObjectiveMechanism, the
OutputStates that it monitors, and how it evaluates them can be specified in a variety of ways, that depend on the
context in which the ControlMechanism is created, as described in the subsections below. In all cases,
the ObjectiveMechanism is assigned to the ControlMechanism's `objective_mechanism
<ControlMechanism.objective_mechanism>` attribute, and a `MappingProjection` is created that projects from the
ObjectiveMechanism's *OUTCOME* `OutputState <ObjectiveMechanism_Output>` to the ControlMechanism's *OUTCOME*
`InputState` (which is its  `primary InputState <InputState_Primary>`.  All of the OutputStates monitored by the
ObjectiveMechanism are listed in its `monitored_output_States <ObjectiveMechanism.monitored_output_states>`
attribute, and in the ControlMechanism's `monitor_for_control <ControlMechanism.montior_for_control>` attribute.

*When the ControlMechanism is created explicitly*

When a ControlMechanism is created explicitly -- either on its own, or in the **controller** argument of the
`constructor for a System <System_Control_Specification>`) -- the following arguments of the ControlMechanism's
constructor can be used to specify its ObjectiveMechanism and/or the OutputStates it monitors:

  * **objective_mechanism** -- this can be specified using any of the following:

    - an existing `ObjectiveMechanism`;
    |
    - a constructor for an ObjectiveMechanism; its **monitored_output_states** argument can be used to specify
      `the OutputStates to be monitored <ObjectiveMechanism_Monitored_Output_States>`, and its **function**
      argument can be used to specify how those OutputStates are evaluated (see `ControlMechanism_Examples`).
    |
    - a list of `OutputState specifications <ObjectiveMechanism_Monitored_Output_States>`; a default ObjectiveMechanism
      is created, using the list of OutputState specifications for its **monitored_output_states** argument.
    |
    Note that if the ObjectiveMechanism is explicitly (using either of the first two methods above), its
    attributes override any attributes specified by the ControlMechanism for its default `objective_mechanism
    <ControlMechanism.objective_mechanism>`, including those of its `function <ObjectiveMechanism.function>` (see
    `note <EVCControlMechanism_Objective_Mechanism_Function_Note>` in EVCControlMechanism for an example);
  ..
  * **monitor_for_control** -- a list a list of `OutputState specifications
    <ObjectiveMechanism_Monitored_Output_States>`;  a default ObjectiveMechanism is created, using the list of
    OutputState specifications for its **monitored_output_states** argument.

  If OutputStates to be monitored are specified in both the **objective_mechanism** argument (on their own, or within
  the constructor for an ObjectiveMechanism) and the **monitor_for_control** argument of the ControlMechanism,
  both sets are used in creating the ObjectiveMechanism.

*When the ControlMechanism is created for or assigned as the controller a System*

If a ControlMechanism is specified as the `controller <System.controller>` of a System (see
`ControlMechanism_System_Controller`), any OutputStates specified to be monitored by the System are assigned as
inputs to the ObjectiveMechanism.  This includes any specified in the **monitor_for_control** argument of the
System's constructor, as well as any specified in a MONITOR_FOR_CONTROL entry of a Mechanism `parameter specification
dictionary <ParameterState_Specification>` (see `Mechanism_Constructor_Arguments` and `System_Control_Specification`).

COMMENT:
FOR DEVELOPERS:
    If the ObjectiveMechanism has not yet been created, these are added to the **monitored_output_states** of its
    constructor called by ControlMechanism._instantiate_objective_mechanmism;  otherwise, they are created using the
    ObjectiveMechanism.add_monitored_output_states method.
COMMENT

* Adding OutputStates to be monitored to a ControlMechanism*

OutputStates to be monitored can also be added to an existing ControlMechanism by using the `add_monitored_output_states
<ObjectiveMechanism.add_monitored_output_states>` method of the ControlMechanism's `objective_mechanism
<ControlMechanism.objective_mechanism>`.


.. _ControlMechanism_Control_Signals:

*Specifying Parameters to Control*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ControlMechanism is used to control the parameter values of other `Components <Component>`.  A `ControlSignal` is
assigned for each parameter controlled by a ControlMechanism, and a `ControlProjection` is assigned from each
ControlSignal to the `ParameterState` for the corresponding parameter to be controlled.

The parameters to be controlled by a ControlMechanism can be specified where it is created.

If it is created explicitly, the parameters to be  controlled can be specified in the **control_signals** argument of
its constructor.  The argument must be a `specification for one more ControlSignals <ControlSignal_Specification>`.

If the ControlMechanism is created as part of a `System`, the parameters to be controlled by it can be specified in
one of two ways:

  * in the **control_signals** argument of the System's constructor, using one or more `ControlSignal specifications
    <ControlSignal_Specification>`;

  * where the `parameter is specified <ParameterState_Specification>`, by including a `ControlProjection` or
    `ControlSignal` in a `tuple specification <ParameterState_Tuple_Specification>` for the parameter.

When a ControlMechanism is created as part of a System, a `ControlSignal` is created and assigned to the
ControlMechanism for every parameter of any `Component <Component>` in the System that has been specified for control
using either of the methods above.

Parameters to be controlled can be added to an existing ControlMechanism by using its `assign_params` method to
add a `ControlSignal` for each additional parameter.

All of the ControlSignals for a ControlMechanism are listed in its `control_signals
<ControlMechanism.control_signals>` attribute, and all of its ControlProjections are listed in its
`control_projections <ControlMechanism.control_projections>` attribute.

.. _ControlMechanism_Structure:

Structure
---------

.. _ControlMechanism_Input:

*Input*
~~~~~~~

A ControlMechanism has a single *OUTCOME* `InputState`. Its `value <InputState.value>` (that can be referenced
by its `outcome <ControlMechanism.outcome>` attribute) is used as the input to the ControlMechanism's `function
<ControlMechanism.function>`, that determines the ControlMechanism's `control_allocation
<ControlMechanism.control_allocation>`. The *OUTCOME* InputState receives its input via a `MappingProjection` from the
*OUTCOME* `OutputState <ObjectiveMechanism_Output>` of an `ObjectiveMechanism`. The Objective Mechanism is specified
in the **objective_mechanism** argument of its constructor, and listed in its `objective_mechanism
<EVCControlMechanism.objective_mechanism>` attribute.  The OutputStates monitored by the ObjectiveMechanism (listed
in its `monitored_output_states <ObjectiveMechanism.monitored_output_states>` attribute) are also listed in the
`monitor_for_control <ControlMechanism.monitor_for_control>` of the ControlMechanism (see
`ControlMechanism_ObjectiveMechanism` for how the ObjectiveMechanism and the OutputStates it monitors are specified).
The OutputStates monitored by the ControlMechanism's `objective_mechanism <ControlMechanism.objective_mechanism>` can
be displayed using its `show <ControlMechanism.show>` method. The ObjectiveMechanism's `function <ObjectiveMechanism>`
evaluates the specified OutputStates, and the result is conveyed as the input to the ControlMechanism.


.. _ControlMechanism_Function:

*Function*
~~~~~~~~~~

A ControlMechanism's `function <ControlMechanism.function>` uses the `value <InputState.value>` of its
*OUTCOME* `InputState` (`outcome <ControlMechanism.outcome>` to generate an `control_allocation
<ControlMechanism.control_allocation>`.  By default, each item of the `control_allocation
<ControlMechanism.control_allocation>` is assigned as the `allocation <ControlSignal.allocation>` of the corresponding
`ControlSignal` in `control_signals <ControlMechanism.control_signals>`;  however, subtypes of ControlMechanism may
assign values differently (for example, an `LCControlMechanism` assigns a single value to all of its ControlSignals).


.. _ControlMechanism_Output:

*Output*
~~~~~~~~

A ControlMechanism has a `ControlSignal` for each parameter specified in its `control_signals
<ControlMechanism.control_signals>` attribute, that sends a `ControlProjection` to the `ParameterState` for the
corresponding parameter. ControlSignals are a type of `OutputState`, and so they are also listed in the
ControlMechanism's `output_states <ControlMechanism.output_states>` attribute. The parameters modulated by a
ControlMechanism's ControlSignals can be displayed using its `show <ControlMechanism.show>` method. By default,
each value of each `ControlSignal` is assigned the value of the corresponding item from the ControlMechanism's
`control_allocation <ControlMechanism.control_allocation>`;  however, subtypes of ControlMechanism may assign values
differently.  The `allocation <ControlSignal.allocation>` is used by each ControlSignal to determine
its `intensity <ControlSignal.intensity>`, which is then assigned as the `value <ControlProjection.value>` of the
ControlSignal's `ControlProjection`.   The `value <ControlProjection.value>` of the ControlProjection is used by the
`ParameterState` to which it projects to modify the value of the parameter it controls (see
`ControlSignal_Modulation` for description of how a ControlSignal modulates the value of a parameter).

.. _ControlMechanism_Output:

*Costs and Net Outcome*
~~~~~~~~~~~~~~~~~~~~~~~

When a ControlMechanism executes, each of its `control_signals <ControlMechanmism>` can incur a `cost
<ControlSignal.cost>`.  The costs


.. _ControlMechanism_Execution:

Execution
---------

If a ControlMechanism is a System's `controller`, it is always the last `Mechanism <Mechanism>` to be executed in a
`TRIAL` for that System (see `System Control <System_Execution_Control>` and `Execution <System_Execution>`).  The
ControlMechanism's `function <ControlMechanism.function>` takes as its input the `value <InputState.value>` of
its *OUTCOME* `input_state <Mechanism_Base.input_state>` (also contained in `outcome <ControlSignal.outcome>`
and uses that to determine its `control_allocation <ControlMechanism.control_allocation>` which specifies the value
assigned to the `allocation <ControlSignal.allocation>` of each of its `ControlSignals <ControlSignal>`.  Each
ControlSignal uses that value to calculate its `intensity <ControlSignal.intensity>`, which is used by its
`ControlProjection(s) <ControlProjection>` to modulate the value of the ParameterState(s) for the parameter(s) it
controls, which are then used in the subsequent `TRIAL` of execution.

.. note::
   A `ParameterState` that receives a `ControlProjection` does not update its value until its owner Mechanism
   executes (see `Lazy Evaluation <LINK>` for an explanation of "lazy" updating).  This means that even if a
   ControlMechanism has executed, a parameter that it controls will not assume its new value until the Mechanism
   to which it belongs has executed.


.. _ControlMechanism_Examples:

Examples
--------

The following example creates a ControlMechanism by specifying its **objective_mechanism** using a constructor
that specifies the OutputStates to be monitored by its `objective_mechanism <ControlMechanism.objective_mechanism>`
and the function used to evaluated these::

    >>> import psyneulink as pnl
    >>> my_transfer_mech_A = pnl.TransferMechanism(name="Transfer Mech A")
    >>> my_DDM = pnl.DDM(name="My DDM")
    >>> my_transfer_mech_B = pnl.TransferMechanism(function=pnl.Logistic,
    ...                                            name="Transfer Mech B")

    >>> my_control_mech = pnl.ControlMechanism(
    ...                          objective_mechanism=pnl.ObjectiveMechanism(monitored_output_states=[(my_transfer_mech_A, 2, 1),
    ...                                                                                               my_DDM.output_states[pnl.RESPONSE_TIME]],
    ...                                                                     name="Objective Mechanism"),
    ...                          function=pnl.LinearCombination(operation=pnl.PRODUCT),
    ...                          control_signals=[(pnl.THRESHOLD, my_DDM),
    ...                                           (pnl.GAIN, my_transfer_mech_B)],
    ...                          name="My Control Mech")


This creates an ObjectiveMechanism for the ControlMechanism that monitors the `primary OutputState
<OutputState_Primary>` of ``my_Transfer_mech_A`` and the *RESPONSE_TIME* OutputState of ``my_DDM``;  its function
first multiplies the former by 2 before, then takes product of their values and passes the result as the input to the
ControlMechanism.  The ControlMechanism's `function <ControlMechanism.function>` uses this value to determine
the allocation for its ControlSignals, that control the value of the `threshold <DDM.threshold>` parameter of
``my_DDM`` and the  `gain <Logistic.gain>` parameter of the `Logistic` Function for ``my_transfer_mech_B``.

The following example specifies the same set of OutputStates for the ObjectiveMechanism, by assigning them directly
to the **objective_mechanism** argument::

    >>> my_control_mech = pnl.ControlMechanism(
    ...                             objective_mechanism=[(my_transfer_mech_A, 2, 1),
    ...                                                  my_DDM.output_states[pnl.RESPONSE_TIME]],
    ...                             control_signals=[(pnl.THRESHOLD, my_DDM),
    ...                                              (pnl.GAIN, my_transfer_mech_B)])
    ...

Note that, while this form is more succinct, it precludes specifying the ObjectiveMechanism's function.  Therefore,
the values of the monitored OutputStates will be added (the default) rather than multiplied.

The ObjectiveMechanism can also be created on its own, and then referenced in the constructor for the ControlMechanism::

    >>> my_obj_mech = pnl.ObjectiveMechanism(monitored_output_states=[(my_transfer_mech_A, 2, 1),
    ...                                                               my_DDM.output_states[pnl.RESPONSE_TIME]],
    ...                                      function=pnl.LinearCombination(operation=pnl.PRODUCT))

    >>> my_control_mech = pnl.ControlMechanism(
    ...                        objective_mechanism=my_obj_mech,
    ...                        control_signals=[(pnl.THRESHOLD, my_DDM),
    ...                                         (pnl.GAIN, my_transfer_mech_B)])

Here, as in the first example, the constructor for the ObjectiveMechanism can be used to specify its function, as well
as the OutputState that it monitors.

See `System_Control_Examples` for examples of how a ControlMechanism, the OutputStates its
`objective_mechanism <ControlSignal.objective_mechanism>`, and its `control_signals <ControlMechanism.control_signals>`
can be specified for a System.


.. _ControlMechanism_Class_Reference:

Class Reference
---------------

"""

import copy
import itertools
import numpy as np
import threading
import typecheck as tc
import warnings

from psyneulink.core.components.functions.combinationfunctions import LinearCombination
from psyneulink.core.components.functions.function import ModulationParam, _is_modulation_param, is_function_type
from psyneulink.core.components.mechanisms.adaptive.adaptivemechanism import AdaptiveMechanism_Base
from psyneulink.core.components.mechanisms.mechanism import Mechanism, Mechanism_Base
from psyneulink.core.components.shellclasses import Composition_Base, Composition_Base, System_Base
from psyneulink.core.components.states.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.components.states.outputstate import OutputState
from psyneulink.core.components.states.parameterstate import ParameterState
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.defaults import defaultControlAllocation
from psyneulink.core.globals.keywords import AUTO_ASSIGN_MATRIX, CONTROL, CONTROL_PROJECTION, CONTROL_PROJECTIONS, CONTROL_SIGNAL, CONTROL_SIGNALS, INIT_EXECUTE_METHOD_ONLY, MONITOR_FOR_CONTROL, OBJECTIVE_MECHANISM, OUTCOME, OWNER_VALUE, PRODUCT, PROJECTIONS, PROJECTION_TYPE, SYSTEM
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import CNodeRole, ContentAddressableList, is_iterable

__all__ = [
    'CONTROL_ALLOCATION', 'ControlMechanism', 'ControlMechanismError', 'ControlMechanismRegistry'
]

CONTROL_ALLOCATION = 'control_allocation'

ControlMechanismRegistry = {}

def _is_control_spec(spec):
    from psyneulink.core.components.projections.modulatory.controlprojection import ControlProjection
    if isinstance(spec, tuple):
        return any(_is_control_spec(item) for item in spec)
    if isinstance(spec, dict) and PROJECTION_TYPE in spec:
        return _is_control_spec(spec[PROJECTION_TYPE])
    elif isinstance(spec, (ControlMechanism, ControlSignal, ControlProjection)):
        return True
    elif isinstance(spec, type) and issubclass(spec, (ControlMechanism, ControlSignal, ControlProjection)):
        return True
    elif isinstance(spec, str) and spec in {CONTROL, CONTROL_PROJECTION, CONTROL_SIGNAL}:
        return True
    else:
        return False


class ControlMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


def _control_mechanism_costs_getter(owning_component=None, execution_id=None):
    try:
        return [c.compute_costs(c.parameters.variable.get(execution_id), execution_id=execution_id) for c in owning_component.control_signals]
    except TypeError:
        return None


# class ControlMechanism(Mechanism_Base):
class ControlMechanism(AdaptiveMechanism_Base):
    """
    ControlMechanism(                              \
        system=None                                \
        objective_mechanism=None,                  \
        origin_objective_mechanism=False           \
        terminal_objective_mechanism=False         \
        function=Linear,                           \
        combine_costs=np.sum,             \
        compute_net_outcome=lambda x,y:x-y,        \
        control_signals=None,                      \
        modulation=ModulationParam.MULTIPLICATIVE  \
        params=None,                               \
        name=None,                                 \
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

    objective_mechanism : ObjectiveMechanism or List[OutputState specification] : default None
        specifies either an `ObjectiveMechanism` to use for the ControlMechanism, or a list of the OutputStates it
        should monitor; if a list of `OutputState specifications <ObjectiveMechanism_Monitored_Output_States>` is used,
        a default ObjectiveMechanism is created and the list is passed to its **monitored_output_states** argument.

    origin_objective_mechanism : Boolean : default False
        specifies whether the `objective_mechanism <LVOCControlMechanism.objective_mechanism>` may be an `ORIGIN`
        node of `composition <LVOCControlMechanism.composition>`.

        When False, even if the `ObjectiveMechanism` is an `ORIGIN` node according to the structure of the
        Composition's `graph <Composition.graph>`, the ObjectiveMechanism is not marked as `ORIGIN`. If the
        ObjectiveMechanism would be the only `ORIGIN` node, then the user must use `required_roles
        <Composition.required_roles>` to assign another node as `ORIGIN`.

        When True, if the ObjectiveMechanism is an `ORIGIN` node according to the structure of the Composition's `graph
        <Composition.graph>`, it is treated normally. If the ObjectiveMechanism is not an `ORIGIN` node according to
        the structure of the graph, then it takes on `ORIGIN` as a required role.

    terminal_objective_mechanism : Boolean : default False
        specifies whether the `objective_mechanism <LVOCControlMechanism.objective_mechanism>` may be an `TERMINAL`
        node of `composition <LVOCControlMechanism.composition>`.

        When False, even if the ObjectiveMechanism is a `TERMINAL` node according to the structure of the Composition's
        `graph <Composition.graph>`, the ObjectiveMechanism is not marked as `TERMINAL`. If the ObjectiveMechanism
        was the only `TERMINAL` node, then the user must use `required_roles <Composition.required_roles>` to assign
        another node as `TERMINAL` for the Composition.

        When True, if the ObjectiveMechanism is a `TERMINAL` node according to the structure of the Composition's
        `graph <Composition.graph>`, it is treated normally. If the ObjectiveMechanism is not a `TERMINAL` node
        according to the structure of the graph, then it takes on `TERMINAL` as a required role.

    function : TransferFunction : default Linear(slope=1, intercept=0)
        specifies function used to combine values of monitored OutputStates.

    combine_costs : Function, function or method : default np.sum
        specifies function used to combine the `cost <ControlSignal.cost>` of the ControlMechanism's `control_signals
        <ControlMechanism.control_signals>`;  must take a list or 1d array of scalar values as its argument and
        return a list or array with a single scalar value.

    compute_net_outcome : Function, function or method : default lambda outcome, cost: outcome-cost
        function used to combine the values of its `outcome <ControlMechanism.outcome>` and `costs
        <ControlMechanism.costs>` attributes;  must take two 1d arrays with scalar values as its arguments
        and return an array with a single scalar value.

    control_signals : ControlSignal specification or List[ControlSignal specification, ...]
        specifies the parameters to be controlled by the ControlMechanism; a `ControlSignal` is created for each
        (see `ControlSignal_Specification` for details of specification).

    modulation : ModulationParam : ModulationParam.MULTIPLICATIVE
        specifies the default form of modulation used by the ControlMechanism's `ControlSignals <ControlSignal>`,
        unless they are `individually specified <ControlSignal_Specification>`.

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
        <ControlMechanism_System_Controller>`.

    objective_mechanism : ObjectiveMechanism
        `ObjectiveMechanism` that monitors and evaluates the values specified in the ControlMechanism's
        **objective_mechanism** argument, and transmits the result to the ControlMechanism's *OUTCOME*
        `input_state <Mechanism_Base.input_state>`.

    origin_objective_mechanism : Boolean
        specifies whether the ObjectiveMechanism of a ControlMechanism may be an "origin" node of the Composition.

        When False, even if the ObjectiveMechanism is an origin node according to the structure of the graph, the
        ObjectiveMechanism is not marked as origin. If the ObjectiveMechanism was the only origin node, then the
        user must use required_roles to assign the origin role to another node.

        When True, if the ObjectiveMechanism is an origin node according to the structure of the graph, it is treated
        normally. If the ObjectiveMechanism is not an origin node according to the structure of the graph, then it
        takes on origin as a required role.

    terminal_objective_mechanism : Boolean
        specifies whether the ObjectiveMechanism of a ControlMechanism may be a "terminal" node of the Composition.

        When False, even if the ObjectiveMechanism is a terminal node according to the structure of the graph, the
        ObjectiveMechanism is not marked as terminal. If the ObjectiveMechanism was the only terminal node, then the
        user must use required_roles to assign the terminal role to another node.

        When True, if the ObjectiveMechanism is a terminal node according to the structure of the graph, it is treated
        normally. If the ObjectiveMechanism is not a terminal node according to the structure of the graph, then it
        takes on terminal as a required role.

    monitor_for_control : List[OutputState]
        each item is an `OutputState` monitored by the ObjectiveMechanism listed in the ControlMechanism's
        `objective_mechanism <ControlMechanism.objective_mechanism>` attribute;  it is the same as that
        ObjectiveMechanism's `monitored_output_states <ObjectiveMechanism.monitored_output_states>` attribute
        (see `ObjectiveMechanism_Monitored_Output_States` for specification).  The `value <OutputState.value>`
        of the OutputStates in the list are used by the ObjectiveMechanism to generate the ControlMechanism's `input
        <ControlMechanism_Input>`.

    monitored_output_states_weights_and_exponents : List[Tuple(float, float)]
        each tuple in the list contains the weight and exponent associated with a corresponding OutputState specified
        in `monitor_for_control <ControlMechanism.monitor_for_control>`;  these are the same as those in the
        `monitored_output_states_weights_and_exponents
        <ObjectiveMechanism.monitored_output_states_weights_and_exponents>` attribute of the `objective_mechanism
        <ControlMechanism.objective_mechanism>`, and are used by the ObjectiveMechanism's `function
        <ObjectiveMechanism.function>` to parametrize the contribution made to its output by each of the values that
        it monitors (see `ObjectiveMechanism Function <ObjectiveMechanism_Function>`).

    outcome : 1d array
        the `value <InputState.value>` of the ControlMechanism's `primary InputState <InputState_Primary>`,
        which receives its `Projection <Projection>` from the *OUTCOME* `OutputState` of its `objective_mechanism
        <ControlMechanism.objective_mechanism>`.

    function : TransferFunction : default Linear(slope=1, intercept=0)
        determines how the `value <OuputState.value>` \\s of the `OutputStates <OutputState>` specified in the
        **monitor_for_control** argument of the ControlMechanism's constructor are used to generate its
        `control_allocation <ControlMechanism.control_allocation>`.

    control_allocation : 2d array
        each item is the value assigned as the `allocation <ControlSignal.allocation>` for the corresponding
        ControlSignal listed in the `control_signals` attribute;  the control_allocation is the same as the
        ControlMechanism's `value <Mechanism_Base.value>` attribute).

    control_signals : ContentAddressableList[ControlSignal]
        list of the `ControlSignals <ControlSignals>` for the ControlMechanism, including any inherited from a
        `system <ControlMechanism.system>` for which it is a `controller <System.controller>` (same as
        ControlMechanism's `output_states <Mechanism_Base.output_states>` attribute); each sends a `ControlProjection`
        to the `ParameterState` for the parameter it controls

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
        result of the ControlMechanism's `combine_costs <ControlMechanism.combine_costs>` function;

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

    outputStateType = ControlSignal
    stateListAttr = Mechanism_Base.stateListAttr.copy()
    stateListAttr.update({ControlSignal:CONTROL_SIGNALS})

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'ControlMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    class Parameters(AdaptiveMechanism_Base.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <ControlMechanism.variable>`

                    :default value: numpy.array([[1.]])
                    :type: numpy.ndarray

                value
                    see `value <ControlMechanism.value>`

                    :default value: numpy.array([1.])
                    :type: numpy.ndarray

                combine_costs
                    see `combine_costs <ControlMechanism.combine_costs>`

                    :default value: numpy.core.fromnumeric.sum
                    :type: <class 'function'>

                compute_net_outcome
                    see `compute_net_outcome <ControlMechanism.compute_net_outcome>`

                    :default value: lambda outcome, cost: outcome - cost
                    :type: <class 'function'>

                costs
                    see `costs <ControlMechanism.costs>`

                    :default value: None
                    :type:
                    :read only: True

                modulation
                    see `modulation <ControlMechanism.modulation>`

                    :default value: ModulationParam.MULTIPLICATIVE
                    :type: `ModulationParam`

        """
        # This must be a list, as there may be more than one (e.g., one per control_signal)
        variable = np.array([defaultControlAllocation])
        value = Parameter(np.array(defaultControlAllocation), aliases='control_allocation')

        combine_costs = Parameter(np.sum, stateful=False, loggable=False)
        compute_net_outcome = Parameter(lambda outcome, cost: outcome - cost, stateful=False, loggable=False)

        costs = Parameter(None, read_only=True, getter=_control_mechanism_costs_getter)
        control_signal_costs = Parameter(None, read_only=True)

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
                 monitor_for_control:tc.optional(tc.any(is_iterable, Mechanism, OutputState))=None,
                 objective_mechanism=None,
                 origin_objective_mechanism=False,
                 terminal_objective_mechanism=False,
                 function=None,
                 combine_costs:is_function_type=np.sum,
                 compute_net_outcome:is_function_type=lambda outcome, cost : outcome - cost,
                 control_signals:tc.optional(tc.any(is_iterable, ParameterState, ControlSignal))=None,
                 modulation:tc.optional(_is_modulation_param)=ModulationParam.MULTIPLICATIVE,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None):

        control_signals = control_signals or []
        if not isinstance(control_signals, list):
            control_signals = [control_signals]
        self.combine_costs = combine_costs
        self.compute_net_outcome = compute_net_outcome

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(system=system,
                                                  monitor_for_control=monitor_for_control,
                                                  objective_mechanism=objective_mechanism,
                                                  origin_objective_mechanism=origin_objective_mechanism,
                                                  terminal_objective_mechanism=terminal_objective_mechanism,
                                                  function=function,
                                                  control_signals=control_signals,
                                                  modulation=modulation,
                                                  params=params)

        self._sim_counts = {}

        super(ControlMechanism, self).__init__(default_variable=default_variable,
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
        """Validate SYSTEM, MONITOR_FOR_CONTROL and CONTROL_SIGNALS

        If System is specified, validate it
        Check that all items in MONITOR_FOR_CONTROL are Mechanisms or OutputStates for Mechanisms in self.system
        Check that all items in CONTROL_SIGNALS are parameters or ParameterStates for Mechanisms in self.system
        """
        from psyneulink.core.components.system import MonitoredOutputStateTuple
        from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
        from psyneulink.core.components.states.inputstate import InputState
        from psyneulink.core.components.states.state import _parse_state_spec

        super(ControlMechanism, self)._validate_params(request_set=request_set,
                                                       target_set=target_set,
                                                       context=context)

        def validate_monitored_state_spec(spec_list):
            for spec in spec_list:
                if isinstance(spec, MonitoredOutputStateTuple):
                    spec = spec.output_state
                if isinstance(spec, dict):
                    # If it is a dict, parse to validate that it is an InputState specification dict
                    #    (for InputState of ObjectiveMechanism to be assigned to the monitored_output_state)
                    spec = _parse_state_spec(owner=self,
                                             state_type=InputState,
                                             state_spec=spec,
                                             context=context)
                    # Get the OutputState, to validate that it is in the ControlMechanism's System (below);
                    #    presumes that the monitored_output_state is the first in the list of projection_specs
                    #    in the InputState state specification dictionary returned from the parse,
                    #    and that it is specified as a projection_spec (parsed into that in the call
                    #    to _parse_connection_specs by _parse_state_spec)

                    spec = spec[PROJECTIONS][0][0]

                # If ControlMechanism has been assigned to a System, check that
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

        if MONITOR_FOR_CONTROL in target_set and target_set[MONITOR_FOR_CONTROL] is not None:
            spec = target_set[MONITOR_FOR_CONTROL]
            if not isinstance(spec, list):
                spec = [spec]
            validate_monitored_state_spec(spec)

        if OBJECTIVE_MECHANISM in target_set and target_set[OBJECTIVE_MECHANISM] is not None:

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
                    validate_monitored_state_spec(obj_mech_spec_list)

            if not isinstance(target_set[OBJECTIVE_MECHANISM], (ObjectiveMechanism, list)):
                raise ControlMechanismError("Specification of {} arg for {} ({}) must be an {}"
                                            "or a list of Mechanisms and/or OutputStates to be monitored for control".
                                            format(OBJECTIVE_MECHANISM,
                                                   self.name, target_set[OBJECTIVE_MECHANISM],
                                                   ObjectiveMechanism.componentName))

        if CONTROL_SIGNALS in target_set and target_set[CONTROL_SIGNALS]:
            if not isinstance(target_set[CONTROL_SIGNALS], list):
                target_set[CONTROL_SIGNALS] = [target_set[CONTROL_SIGNALS]]
            for control_signal in target_set[CONTROL_SIGNALS]:
                _parse_state_spec(state_type=ControlSignal, owner=self, state_spec=control_signal)

    # IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION
    # ONCE THAT IS IMPLEMENTED
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
          Projection will be instantiated to a corresponding InputState of the ControlMechanism
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

        # If the ControlMechanism has already been assigned to a System
        #    get OutputStates in System specified as monitor_for_control or already being monitored:
        #        do this by calling _get_monitored_output_states_for_system(),
        #        which also gets any OutputStates already being monitored by the ControlMechanism
        if self.system:
            monitored_output_states.extend(self.system._get_monitored_output_states_for_system(self,context=context))

        self.monitor_for_control = self.monitor_for_control or []
        if not isinstance(self.monitor_for_control, list):
            self.monitor_for_control = [self.monitor_for_control]

        # If objective_mechanism is used to specify OutputStates to be monitored (legacy feature)
        #    move them to monitor_for_control
        if isinstance(self.objective_mechanism, list):
            self.monitor_for_control.extend(self.objective_mechanism)

        # Add items in monitor_for_control to monitored_output_states
        for i, item in enumerate(self.monitor_for_control):
            # If it is already in the list received from System, ignore
            if item in monitored_output_states:
                # NOTE: this can happen if ControlMechanisms is being constructed by System
                #       which passed its monitor_for_control specification
                continue
            monitored_output_states.extend([item])

        # INSTANTIATE ObjectiveMechanism

        # If *objective_mechanism* argument is an ObjectiveMechanism, add monitored_output_states to it
        if isinstance(self.objective_mechanism, ObjectiveMechanism):
            if monitored_output_states:
                self.objective_mechanism.add_monitored_output_states(
                                                              monitored_output_states_specs=monitored_output_states,
                                                              context=context)
        # Otherwise, instantiate ObjectiveMechanism with list of states in monitored_output_states
        else:
            try:
                self._objective_mechanism = ObjectiveMechanism(monitored_output_states=monitored_output_states,
                                                               function=LinearCombination(operation=PRODUCT),
                                                               name=self.name + '_ObjectiveMechanism')
            except (ObjectiveMechanismError, FunctionError) as e:
                raise ObjectiveMechanismError("Error creating {} for {}: {}".format(OBJECTIVE_MECHANISM, self.name, e))

        # Print monitored_output_states
        if self.prefs.verbosePref:
            print("{0} monitoring:".format(self.name))
            for state in self.monitored_output_states:
                weight = self.monitored_output_states_weights_and_exponents[
                                                         self.monitored_output_states.index(state)][WEIGHT_INDEX]
                exponent = self.monitored_output_states_weights_and_exponents[
                                                         self.monitored_output_states.index(state)][EXPONENT_INDEX]
                print("\t{0} (exp: {1}; wt: {2})".format(state.name, weight, exponent))

        # Assign ObjectiveMechanism's role as CONTROL
        self.objective_mechanism._role = CONTROL

        # If ControlMechanism is a System controller, name Projection from
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

        objective_roles = [CNodeRole.OBJECTIVE]
        if self.origin_objective_mechanism:
            objective_roles.append(CNodeRole.ORIGIN)
        if self.terminal_objective_mechanism:
            objective_roles.append(CNodeRole.TERMINAL)
        self.aux_components.append((self.objective_mechanism, objective_roles))
        self.aux_components.append((projection_from_objective, True))
        self._objective_projection = projection_from_objective
        self.monitor_for_control = self.monitored_output_states

    def _instantiate_input_states(self, context=None):
        super()._instantiate_input_states(context=context)
        self.input_state.name = OUTCOME

        # IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
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

        if self.control_signals:
            self._output_states = []
            self.defaults.value = None

            for control_signal in self.control_signals:
                self._instantiate_control_signal(control_signal, context=context)

        super()._instantiate_output_states(context=context)

        # Reassign control_signals to capture any user_defined ControlSignals instantiated in call to super
        #    and assign to ContentAddressableList
        self._control_signals = ContentAddressableList(component_type=ControlSignal,
                                                       list=[state for state in self.output_states
                                                             if isinstance(state, ControlSignal)])

        # If the ControlMechanism's control_allocation has more than one item,
        #    warn if the number of items does not equal the number of its ControlSignals
        #    (note:  there must be fewer ControlSignals than items in control_allocation,
        #            as the reverse is an error that is checked for in _instantiate_control_signal)
        if len(self.defaults.value) > 1 and len(self.control_signals) != len(self.defaults.value):
            if self.verbosePref:
                warnings.warning("The number of {}s for {} ({}) does not equal the number of items in its {} ({})".
                                 format(ControlSignal.__name__, self.name, len(self.control_signals),
                                        CONTROL_ALLOCATION, len(self.defaults.value)))

    def _instantiate_control_signal(self, control_signal, context=None):
        from psyneulink.core.components.states.state import _instantiate_state
        # Parses and instantiates control_signal specifications (in call to State._parse_state_spec)
        #    and any embedded Projection specifications (in call to <State>._instantiate_projections)
        # Temporarily assign variable to default allocation value to avoid chicken-and-egg problem:
        #    value, output_states and control_signals haven't been expanded yet to accomodate the new ControlSignal;
        #    reassign ControlSignal.variable to actual OWNER_VALUE below, once value has been expanded

        control_signal = _instantiate_state(state_type=ControlSignal,
                                            owner=self,
                                            variable=defaultControlAllocation,
                                            reference_value=ControlSignal.defaults.allocation,
                                            modulation=self.modulation,
                                            state_spec=control_signal,
                                            context=context)
        control_signal.owner = self

        # Update control_signal_costs to accommodate instantiated Projection
        # MODIFIED 11/2/18 OLD:
        control_signal_costs = self.parameters.control_signal_costs.get()
        try:
            control_signal_costs = np.append(control_signal_costs, np.zeros((1, 1)), axis=0)
        except (AttributeError, ValueError):
            control_signal_costs = np.zeros((1, 1))
        self.parameters.control_signal_costs.set(control_signal_costs, override=True)

        # MODIFIED 11/2/18 END

        # UPDATE output_states AND control_projections -------------------------------------------------------------

        # TBI: For control mechanisms that accumulate, starting output must be equal to the initial "previous value"
        # so that modulation that occurs BEFORE the control mechanism executes is computed appropriately
        # if (isinstance(self.function, IntegratorFunction)):
        #     control_signal._intensity = function.initializer

        # Add ControlSignal to output_states list
        self._output_states.append(control_signal)

        # since output_states is exactly control_signals is exactly the shape of value, we can just construct it here
        self.defaults.value = np.array([[ControlSignal.defaults.allocation]
                                                 for i in range(len(self._output_states))])
        self.parameters.value.set(copy.deepcopy(self.defaults.value))

        # Assign ControlSignal's variable to index of owner's value
        control_signal._variable_spec = [(OWNER_VALUE, len(self.defaults.value) - 1)]
        if not isinstance(control_signal.owner_value_index, int):
            raise ControlMechanismError(
                    "PROGRAM ERROR: The \'owner_value_index\' attribute for {} of {} ({})is not an int."
                        .format(control_signal.name, self.name, control_signal.owner_value_index))
        # Validate index
        try:
            self.defaults.value[control_signal.owner_value_index]
        except IndexError:
            raise ControlMechanismError(
                "Index specified for {} of {} ({}) exceeds the number of items of its {} ({})".
                    format(ControlSignal.__name__, self.name, control_signal.owner_value_index,
                           CONTROL_ALLOCATION, len(self.defaults.value)
                )
            )

        return control_signal

    def show(self):
        """Display the OutputStates monitored by ControlMechanism's `objective_mechanism
        <ControlMechanism.objective_mechanism>` and the parameters modulated by its `control_signals
        <ControlMechanism.control_signals>`.
        """

        print("\n---------------------------------------------------------")

        print("\n{0}".format(self.name))
        print("\n\tMonitoring the following Mechanism OutputStates:")
        for state in self.objective_mechanism.input_states:
            for projection in state.path_afferents:
                monitored_state = projection.sender
                monitored_state_mech = projection.sender.owner
                # FIX: 10/3/17 - self.monitored_output_states IS A LIST OF INPUT_STATES,
                # FIX:            BUT monitored_state IS AN INPUT_STATE
                # FIX:            * ??USE monitored_state.name,
                # FIX:              BUT THEN NEED TO UPDATE index METHOD OF
                # ContentAddressableList
                monitored_state_index = self.monitored_output_states.index(monitored_state)

                weight = self.monitored_output_states_weights_and_exponents[monitored_state_index][0]
                exponent = self.monitored_output_states_weights_and_exponents[monitored_state_index][1]

                print ("\t\t{0}: {1} (exp: {2}; wt: {3})".
                       format(monitored_state_mech.name, monitored_state.name, weight, exponent))

        print ("\n\tControlling the following Mechanism parameters:".format(self.name))
        # Sort for consistency of output:
        state_names_sorted = sorted(self.output_states.names)
        for state_name in state_names_sorted:
            for projection in self.output_states[state_name].efferents:
                print ("\t\t{0}: {1}".format(projection.receiver.owner.name, projection.receiver.name))

        print ("\n---------------------------------------------------------")

    def add_monitored_output_states(self, monitored_output_states, context=None):
        """Instantiate OutputStates to be monitored by ControlMechanism's `objective_mechanism
        <ControlMechanism.objective_mechanism>`.

        **monitored_output_states** can be any of the following:
            - `Mechanism`;
            - `OutputState`;
            - `tuple specification <InputState_Tuple_Specification>`;
            - `State specification dictionary <InputState_Specification_Dictionary>`;
            - list with any of the above.
        If any item is a Mechanism, its `primary OutputState <OutputState_Primary>` is used.
        OutputStates must belong to Mechanisms in the same `System` as the ControlMechanism.
        """
        output_states = self.objective_mechanism.add_monitored_output_states(
                                                                 monitored_output_states_specs=monitored_output_states,
                                                                 context=context)
        if self.system:
            self.system._validate_monitored_states_in_system(output_states, context=context)

    def _add_process(self, process, role:str):
        super()._add_process(process, role)
        self.objective_mechanism._add_process(process, role)

    @tc.typecheck
    def assign_as_controller(self, system:System_Base, context=ContextFlags.COMMAND_LINE):
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

        if context == ContextFlags.COMMAND_LINE:
            system.controller = self
            return

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
            self.add_monitored_output_states(monitored_output_states)

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
        self._objective_mechanism.for_controller = True

        if context != ContextFlags.PROPERTY:
            system._controller = self

        self._activate_projections_for_compositions(system)

    def _activate_projections_for_compositions(self, compositions=None):
        self._objective_projection._activate_for_compositions(compositions)

        for cs in self.control_signals:
            for eff in cs.efferents:
                eff._activate_for_compositions(compositions)

        # assign any deferred init objective mech monitored output state projections to this system
        for output_state in self.objective_mechanism.monitored_output_states:
            for eff in output_state.efferents:
                eff._activate_for_compositions(compositions)

        for eff in self.efferents:
            eff._activate_for_compositions(compositions)

        for aff in self._objective_mechanism.afferents:
            aff._activate_for_compositions(compositions)

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
        return self._objective_mechanism.monitored_output_states_weights_and_exponents

    # @property
    # def outcome(self):
    #     return self.variable[0]

    # @property
    # def costs(self):
    #     # FIX: 11/9/19 LOCALLY MANAGE STATEF     ULNESS OF ControlSignals AND costs [JDC]
    #     # MODIFIED 11/9/18 OLD:
    #     return [c.compute_costs(c.variable) for c in self.control_signals]
    #     # # MODIFIED 11/9/18 NEW:
    #     # return [c.compute_costs(c.variable, c.last_intensity) for c in self.control_signals]
    #     # MODIFIED 11/9/18 END

    # @property
    # def combined_costs(self):
    #     return self.combine_costs(self.costs)
    #
    # @property
    # def net_outcome(self):
    #     # return self.compute_net_outcome(self.outcome, self.costs)
    #     return self.outcome - self.combined_costs

    @property
    def control_projections(self):
        return [projection for control_signal in self.control_signals for projection in control_signal.efferents]

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

        return '{0}-sim-{1}'.format(execution_id, sim_num)

    @property
    def _dependent_components(self):
        return list(itertools.chain(
            super()._dependent_components,
            [self.objective_mechanism],
        ))
