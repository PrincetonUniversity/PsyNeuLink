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

A ControlMechanism is a subclass of `ModulatoryMechanism` that is restricted to using only `ControlSignals
<ControlSignal>` and not GatingSignals.  Accordingly, its constructor has a **control_signals** argument in place of
a **modulatory_signals** argument.  It also lacks any attributes related to gating.  In all other respects it is
identical to its parent class, ModulatoryMechanism.

COMMENT:
using only `ControlSignals
<ControlSignal>` and not `Gating
an `AdaptiveMechanism <AdaptiveMechanism>` that modifies the parameter(s) of one or more
`Components <Component>` in response to an evaluative signal received from its `objective_mechanism
<ControlMechanism.objective_mechanism>`.  The `objective_mechanism
<ControlMechanism.objective_mechanism>` monitors a specified set of OutputStates, and from these generates the
evaluative signal that is used by the ControlMechanism's `function <ControlMechanism.function>` to calculate a
`control_allocation <ControlMechanism.control_allocation>`: a list of values provided to each of its `control_signals
<ControlMechanism.control_signals>`.  Its control_signals are `ControlSignal` OutputStates that are used to
that modulate the parameters of other Mechanisms' `function <Mechanism.function>` (see `ControlSignal_Modulation` for a
more detailed description of how modulation operates). A ControlMechanism can modulate only Components in the
`Composition` to which it belongs. The OutputStates monitored by the ControlMechanism's `objective_mechanism
<ControlMechanism.objective_mechanism>` and the parameters it modulates can be listed using its `show
<ControlMechanism.show>` method.

.. _ControlMechanism_System_Controller:

*ControlMechanisms and a Composition*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ControlMechanism can be assigned to a `Composition` and executed just like any other Mechanism. It can also be
assigned as the `controller <Composition.controller>` of a `Composition`, that has a special relation
to the Composition: it is used to control all of the parameters that have been `specified for control
<ControlMechanism_Control_Signals>` in that Composition.  A ControlMechanism can be the `controller
<Composition.controller>` for only one Composition, and a Composition can have only one `controller
<Composition.controller>`.  The Composition's `controller <Composition.controller>` is executed either before or after
all of the other Components in the Composition are executed, including any other ControlMechanisms that belong to it
(see `Composition Execution <Composition_Execution>`).  A ControlMechanism can be assigned as the `controller
<Composition.controller>` for a Composition by specifying it in the **controller** argument of the Composition's
constructor, or by specifying the Composition as the **composition       ** argument of either the ControlMechanism's
constructor or its `assign_as_controller <ControlMechanism.assign_as_controller>` method. A Composition's `controller
<Composition.controller>` and its associated Components can be displayed using the Composition's `show_graph
<Composition.show_graph>` method with its **show_control** argument assigned as `True`.


.. _ControlMechanism_Creation:

Creating a ControlMechanism
---------------------------

A ControlMechanism can be created by calling its constructor. Whenever a ControlMechanism is created,
if no `ObjectiveMechanism` is specified in the **objective_mechanism** of its
constructor, then  one is automatically created and assigned as its `objective_mechanism
<ControlMechanism.objective_mechanism>` attribute (see `ControlMechanism_ObjectiveMechanism` below).  This is used to
monitor and evaluate the OutputStates that are are used to determine the ControlMechanism's `control_allocation
<ControlMechanism.control_allocation>`.  The `OutputStates <OutputState>` monitored by the `objective_mechanism
<ControlMechanism.objective_mechanism>` can be specified in the **monitor_for_control** argument of the
ControlMechanism's constructor, or in the **monitor** argument of the constructor for the `ObjectiveMechanism`
itself.  The parameters to be controlled by the  ControlMechanism are specified in the **control_signals** argument
(see `ControlMechanism_Control_Signals` below).

VERIFY FOR Composition:
If the
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
      `the OutputStates to be monitored <ObjectiveMechanism_Monitor>`, and its **function**
      argument can be used to specify how those OutputStates are evaluated (see `ControlMechanism_Examples`).
    |
    - a list of `OutputState specifications <ObjectiveMechanism_Monitor>`; a default ObjectiveMechanism
      is created, using the list of OutputState specifications for its **monitored_output_states** argument.
    |
    Note that if the ObjectiveMechanism is explicitly (using either of the first two methods above), its
    attributes override any attributes specified by the ControlMechanism for its default `objective_mechanism
    <ControlMechanism.objective_mechanism>`, including those of its `function <ObjectiveMechanism.function>` (see
    `note <EVCControlMechanism_Objective_Mechanism_Function_Note>` in EVCControlMechanism for an example);
  ..
  * **monitor_for_control** -- a list a list of `OutputState specifications
    <ObjectiveMechanism_Monitor>`;  a default ObjectiveMechanism is created, using the list of
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

FOR DEVELOPERS:
    If the ObjectiveMechanism has not yet been created, these are added to the **monitored_output_states** of its
    constructor called by ControlMechanism._instantiate_objective_mechanmism;  otherwise, they are created using the
    ObjectiveMechanism.add_to_monitor method.

* Adding OutputStates to be monitored to a ControlMechanism*

OutputStates to be monitored can also be added to an existing ControlMechanism by using the `add_to_monitor
<ObjectiveMechanism.add_to_monitor>` method of the ControlMechanism's `objective_mechanism
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

A ControlMechanism's `function <ControlMechanism.function>` uses `outcome <ControlMechanism.outcome>`
(the `value <InputState.value>` of its *OUTCOME* `InputState`) to generate a `control_allocation
<ControlMechanism.control_allocation>`.  By default, `function <ControlMechanism.function>` is assigned
the `DefaultAllocationFunction`, which takes a single value as its input, and assigns that as the value of
each item of `modulatory_allocation <ControlMechanism.control_allocation>`.  Each of these items is assigned as
the allocation for the corresponding  `ControlSignal` in `control_signals <ControlMechanism.control_signals>`. Thus,
by default, the ControlMechanism distributes its input as the allocation to each of its `control_signals
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
COMMENT


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
    ...                          objective_mechanism=pnl.ObjectiveMechanism(monitor=[(my_transfer_mech_A, 2, 1),
    ...                                                                               my_DDM.output_states[pnl.RESPONSE_TIME]],
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

import numpy as np
import typecheck as tc
import warnings

from psyneulink.core.components.functions.function import ModulationParam, _is_modulation_param, is_function_type
from psyneulink.core.components.mechanisms.adaptive.modulatorymechanism import ModulatoryMechanism
from psyneulink.core.components.mechanisms.mechanism import Mechanism, Mechanism_Base
from psyneulink.core.components.shellclasses import Composition_Base, System_Base
from psyneulink.core.components.states.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.components.states.outputstate import OutputState
from psyneulink.core.components.states.parameterstate import ParameterState
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.defaults import defaultControlAllocation
from psyneulink.core.globals.keywords import CONTROL, CONTROL_PROJECTION, CONTROL_SIGNAL, CONTROL_SIGNALS, \
    GATING_SIGNALS, INIT_EXECUTE_METHOD_ONLY, PROJECTION_TYPE
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


# MODIFIED 5/18/19 NEW: [JDC]
def _control_allocation_getter(owning_component=None, execution_id=None):
    return owning_component.modulatory_allocation

def _control_allocation_setter(value, owning_component=None, execution_id=None):
    owning_component.parameters.modulatory_allocation.set(np.array(value), execution_id)
    return value

def _gating_allocation_getter(owning_component=None, execution_id=None):
    from psyneulink.core.components.mechanisms.adaptive.gating import GatingMechanism
    from psyneulink.core.components.states.modulatorysignals.gatingsignal import GatingSignal
    raise ControlMechanismError(f"'gating_allocation' attribute is not implemented on {owning_component.__name__};  "
                                f"consider using a {GatingMechanism.__name__} instead, "
                                f"or a {ModulatoryMechanism.__name__} if both {ControlSignal.__name__}s and "
                                f"{GatingSignal.__name__}s are needed.")


def _gating_allocation_setter(value, owning_component=None, execution_id=None, **kwargs):
    from psyneulink.core.components.mechanisms.adaptive.gating import GatingMechanism
    from psyneulink.core.components.states.modulatorysignals.gatingsignal import GatingSignal
    raise ControlMechanismError(f"'gating_allocation' attribute is not implemented on {owning_component.__name__};  "
                                f"consider using a {GatingMechanism.__name__} instead, "
                                f"or a {ModulatoryMechanism.__name__} if both {ControlSignal.__name__}s and "
                                f"{GatingSignal.__name__}s are needed.")
# MODIFIED 5/18/19 END


class ControlMechanism(ModulatoryMechanism):
    """
    ControlMechanism(                                            \
        system=None,                                             \
        monitor_for_control=None,                                \
        objective_mechanism=None,                                \
        function=Linear,                                         \
        control_signals=None,                                    \
        modulation=ModulationParam.MULTIPLICATIVE,               \
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
        **objective_mechanism** argument; for any Mechanisms specified, their `primary OutputState
        <OutputState_Primary>` are used.

    objective_mechanism : ObjectiveMechanism or List[OutputState specification] : default None
        specifies either an `ObjectiveMechanism` to use for the ControlMechanism, or a list of the OutputStates it
        should monitor; if a list of `OutputState specifications <ObjectiveMechanism_Monitor>` is used,
        a default ObjectiveMechanism is created and the list is passed to its **monitored_output_states** argument.

    function : TransferFunction : default Linear(slope=1, intercept=0)
        specifies function used to combine values of monitored OutputStates.

    control_signals : ControlSignal specification or List[ControlSignal specification, ...]
        specifies the parameters to be controlled by the ControlMechanism; a `ControlSignal` is created for each
        (see `ControlSignal_Specification` for details of specification).

    modulation : ModulationParam : ModulationParam.MULTIPLICATIVE
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
        <ControlMechanism_System_Controller>`.

    objective_mechanism : ObjectiveMechanism
        `ObjectiveMechanism` that monitors and evaluates the values specified in the ControlMechanism's
        **objective_mechanism** argument, and transmits the result to the ControlMechanism's *OUTCOME*
        `input_state <Mechanism_Base.input_state>`.

    monitor_for_control : List[OutputState]
        each item is an `OutputState` monitored by the ObjectiveMechanism listed in the ControlMechanism's
        `objective_mechanism <ControlMechanism.objective_mechanism>` attribute;  it is the same as that
        ObjectiveMechanism's `monitored_output_states <ObjectiveMechanism.monitored_output_states>` attribute
        (see `ObjectiveMechanism_Monitor` for specification).  The `value <OutputState.value>`
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

    compute_reconfiguration_cost : Function, function or method
        function used to compute the ControlMechanism's `reconfiguration_cost  <ControlMechanism.reconfiguration_cost>`;
        result is a scalar value representing the difference — defined by the function — between the values of the
        ControlMechanism's current and last `control_alloction <ControlMechanism.control_allocation>`, that can be
        accessed by `reconfiguration_cost <ControlMechanism.reconfiguration_cost>` attribute.

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

    # # MODIFIED 5/18/19 NEW: [JDC]
    # Override control_allocation and suppress gating_allocation
    class Parameters(ModulatoryMechanism.Parameters):
        """
            Attributes
            ----------

                control_allocation
                    see `control_allocation <ControlMechanism.control_allocation>

                    :default value: defaultControlAllocation
                    :type:
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
    # MODIFIED 5/18/19 END

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 system:tc.optional(tc.any(System_Base, Composition_Base))=None,
                 monitor_for_control:tc.optional(tc.any(is_iterable, Mechanism, OutputState))=None,
                 objective_mechanism=None,
                 function=None,
                 default_allocation=None,
                 control_signals:tc.optional(tc.any(is_iterable, ParameterState, ControlSignal))=None,
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
                for i in kwargs.keys():
                    raise ControlMechanismError("Unrecognized arg in constructor for {}: {}".
                                                format(self.__class__.__name__, repr(i)))

        if default_allocation is not None:
            self.parameters.control_allocation.default_value = np.atleast_1d(default_allocation)

        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(system=system,
                                                  params=params)

        super(ControlMechanism, self).__init__(system=system,
                                               default_variable=default_variable,
                                               size=size,
                                               monitor_for_modulation=monitor_for_control,
                                               objective_mechanism=objective_mechanism,
                                               function=function,
                                               combine_costs=combine_costs,
                                               compute_reconfiguration_cost=compute_reconfiguration_cost,
                                               compute_net_outcome=compute_net_outcome,
                                               modulatory_signals=control_signals,
                                               modulation=modulation,
                                               params=params,
                                               name=name,
                                               prefs=prefs,
                                               context=ContextFlags.CONSTRUCTOR)

    def _instantiate_output_states(self, context=None):
        self._register_modulatory_signal_type(ControlSignal,context)
        super()._instantiate_output_states(context)

    def _instantiate_control_signal(self, control_signal, context):
        return super()._instantiate_modulatory_signal(modulatory_signal=control_signal, context=context)

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

        if self._objective_mechanism is None:
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

    def _apply_control_allocation(self, control_allocation, runtime_params, context, execution_id=None):
        self._apply_modulatory_allocation(modulatory_allocation=control_allocation,
                                          runtime_params=runtime_params,
                                          context=context,
                                          execution_id=execution_id)

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
