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
used by the ControlMechanism's `function <ControlMechanism.function>` to calculate an `allocation_policy
<ControlMechanism.allocation_policy>`: a list of `allocation <ControlSignal.allocation>` values for each of its
`ControlSignals <ControlSignal>`.  Each ControlSignal uses its `allocation <ControlSignal.allocation>` to calculate its
`intensity`, which is then conveyed by the ControlSignal's `ControlProjection(s) <ControlProjection>` to the
`ParameterState(s) <ParameterState>` to which they project.  Each ParameterState then uses the value received by a
ControlProjection to modify the value of the parameter for which it is responsible (see `ModulatorySignal_Modulation`
for a more detailed description of how modulation operates).  A ControlMechanism can regulate only the parameters of
Components in the `System` to which it belongs. The OutputStates used to determine the ControlMechanism's
`allocation_policy <ControlMechanism.allocation_policy>`, the `ObjectiveMechanism` used to evalute these, and the
parameters controlled by the ControlMechanism can be listed using its `show <ControlMechanism.show>` method.

COMMENT:
    ALTERNATE VERSION
    and has a `ControlSignal` for each parameter of the Components in the `system <EVCControlMechanism.system>` that it
    controls.  Each ControlSignal is associated with a `ControlProjection` that regulates the value of the parameter it
    controls, with the magnitude of that regulation determined by the ControlSignal's `intensity`.  A particular
    combination of ControlSignal `intensity` values is called an `allocation_policy`. When a `System` is executed that
    uses an EVCControlMechanism as its `controller <System.controller>`, it concludes by executing the EVCControlMechanism, which
    determines its `allocation_policy` for the next `TRIAL`.  That, in turn, determines the `intensity` for each of the
    ControlSignals, and therefore the values of the parameters they control on the next `TRIAL`. The OutputStates used
    to determine an EVCControlMechanism's `allocation_policy <EVCControlMechanism.allocation_policy>` and the parameters it
    controls can be listed using its `show <EVCControlMechanism.show>` method.
COMMENT

.. _ControlMechanism_System_Controller:

ControlMechanisms and a System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ControlMechanism can be assigned to and executed within one or more Systems (listed in its `systems
<Mechanism_Base.systems>` attribute), just like any other Mechanism.  It also be assigned as the `controller
<System.controller>` of a `System`, that has a special relation to the System: it is used to control any and all
parameters that have been `specified for control <ControlMechanism_Control_Signals>` in that System.  A
ControlMechanism can be the `controller <System.controller>` for only one System, and a System can have only one
one `controller <System.controller>`.  The System's `controller <System.controller>` is executed after all
of the other Components in the System have been executed, including any other ControlMechanisms that belong to it (see
`System Execution <System_Execution>`).  A ControlMechanism can be assigned as the `controller <System.controller>`
for a System by specifying it in the **controller** argument of the System's constructor, or by specifying the System
as the **system** argument of either the ControlMechanism's constructor or its `assign_as_controller
<ControlMechanism.assign_as_controller>` method. A System's `controller  <System.controller>` and its
associated Components can be displayed using the System's `show_graph <System.show_graph>` method with its
**show_control** argument assigned as `True`.


.. _ControlMechanism_Creation:

Creating a ControlMechanism
---------------------------

A ControlMechanism can be created using the standard Python method of calling the constructor for the desired type.
A ControlMechanism is also created automatically whenever a `System is created <System_Creation>`, and the
ControlMechanism class or one of its subtypes is specified in the **controller** argument of the System's constructor
(see `System_Creation`).  If the ControlMechanism is created explicitly (using its constructor), the
`ObjectiveMechanism` it uses to monitor and evaluate `OutputStates <OutputState>` is specified in the
**objective_mechanism** argument of its constructor, and the parameters it controls are specified in the
**control_signals** argument.  If the ControlMechanism is created automatically by a System, then the specification of
OutputStates to be monitored and parameters to be controlled are made on the System and/or the Components themselves
(see `System_Control_Specification`).  In either case the Components needed to monitor the specified OutputStates (an
`ObjectiveMechanism` and `Projections <Projection>` to it) and to control the specified parameters (`ControlSignals
<ControlSignal>` and corresponding `ControlProjections <ControlProjection>`) are created automatically, as described
below.

.. _ControlMechanism_ObjectiveMechanism:

ObjectiveMechanism and Monitored OutputStates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a ControlMechanism is created, it is associated with an `ObjectiveMechanism` that is used to monitor and
evaluate a set of `OutputStates <OutputState>` upon which it bases it `allocation_policy
<ControlMechanism.allocation_policy>`.  If the ControlMechanism is created explicitly, its ObjectiveMechanism
can be specified in the **objective_mechanism** argument of its constructor, using either of the following:

  * an existing `ObjectiveMechanism`, or a constructor for one;  in this case the **monitored_output_states** argument
    of the ObjectiveMechanism's constructor is used to specify the OutputStates to be `monitored and evaluated
    <ObjectiveMechanism_Monitored_Output_States>` (see `ControlMechanism_Examples`); note that, in this case, the
    default values for the attributes of the ObjectiveMechanism override any that ControlMechanism uses for its
    default `objective_mechanism <ControlMechanism.objective_mechanism>`, including those of its `function
    <ObjectiveMechanism.function>` (see `note <EVCControlMechanism_Objective_Mechanism_Function_Note>` in EVCControlMechanism for
    an example);
  ..
  * a list of `OutputState specifications <ObjectiveMechanism_Monitored_Output_States>`;  in this case, a default
    ObjectiveMechanism is created, using the list of OutputState specifications as the **monitored_output_states**
    argument of the ObjectiveMechanism's constructor.

If the **objective_mechanism** argument is not specified, a default ObjectiveMechanism is created that is not assigned
any OutputStates to monitor; this must then be done explicitly after the ControlMechanism is created.

When a ControlMechanism is created automatically as part of a `System <System_Creation>`:

  * a default ObjectiveMechanism is created for the ControlMechanism, using the list of `OutputStates <OutputState>`
    specified in the **monitor_for_control** argument of the System's contructor, and any others within the System that
    have been specified to be monitored (using the MONITOR_FOR_CONTROL keyword), as the **monitored_output_states**
    argument for the ObjectiveMechanism's constructor (see `System_Control_Specification`).

In all cases, the ObjectiveMechanism is assigned to the ControlMechanism's `objective_mechanism
<ControlMechanism.objective_mechanism>` attribute, and a `MappingProjection` is created that projects from the
ObjectiveMechanism's *OUTCOME* `OutputState <ObjectiveMechanism_Output>` to the ControlMechanism's `primary
InputState <InputState_Primary>`.

OutputStates to be monitored can be added to an existing ControlMechanism by using the `add_monitored_output_states
<ObjectiveMechanism.add_monitored_output_states>` method of the ControlMechanism's `objective_mechanism
<ControlMechanism.objective_mechanism>`.

.. _ControlMechanism_Control_Signals:

Specifying Parameters to Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Input
~~~~~

A ControlMechanism has a single *ERROR_SIGNAL* `InputState`, the `value <InputState.value>` of which is used as the
input to the ControlMechanism's `function <ControlMechanism.function>`, that determines the ControlMechanism's
`allocation_policy <ControlMechanism.allocation_policy>`. The *ERROR_SIGNAL* InputState receives its input
via a `MappingProjection` from the *OUTCOME* `OutputState <ObjectiveMechanism_Output>` of an `ObjectiveMechanism`.
The Objective Mechanism is specified in the **objective_mechanism** argument of its constructor, and listed in its
`objective_mechanism <EVCControlMechanism.objective_mechanism>` attribute.  The OutputStates monitored by the
ObjectiveMechanism (listed in its `monitored_output_states <ObjectiveMechanism.monitored_output_states>`
attribute) are also listed in the `monitored_output_states <ControlMechanism.monitored_output_states>`
of the ControlMechanism (see `ControlMechanism_ObjectiveMechanism` for how the ObjectiveMechanism and the
OutputStates it monitors are specified).  The OutputStates monitored by the ControlMechanism's `objective_mechanism
<ControlMechanism.objective_mechanism>` can be displayed using its `show <ControlMechanism.show>` method.
The ObjectiveMechanism's `function <ObjectiveMechanism>` evaluates the specified OutputStates, and the result is
conveyed as the input to the ControlMechanism.


.. _ControlMechanism_Function:

Function
~~~~~~~~

A ControlMechanism's `function <ControlMechanism.function>` uses the `value <InputState.value>` of its
*ERROR_SIGNAL* `InputState` to generate an `allocation_policy <ControlMechanism.allocation_policy>`.  By
default, each item of the `allocation_policy <ControlMechanism.allocation_policy>` is assigned as the
`allocation <ControlSignal.allocation>` of the corresponding `ControlSignal` in `control_signals
<ControlMechanism.control_signals>`;  however, subtypes of ControlMechanism may assign values differently
(for example, an `LCControlMechanism` assigns a single value to all of its ControlSignals).


.. _ControlMechanism_Output:

Output
~~~~~~

A ControlMechanism has a `ControlSignal` for each parameter specified in its `control_signals
<ControlMechanism.control_signals>` attribute, that sends a `ControlProjection` to the `ParameterState` for the
corresponding parameter. ControlSignals are a type of `OutputState`, and so they are also listed in the
ControlMechanism's `output_states <ControlMechanism.output_states>` attribute. The parameters modulated by a
ControlMechanism's ControlSignals can be displayed using its `show <ControlMechanism.show>` method. By default,
each value of each `ControlSignal` is assigned the value of the corresponding item from the ControlMechanism's
`allocation_policy <ControlMechanism.allocation_policy>`;  however, subtypes of ControlMechanism may assign values
differently.  The `allocation <ControlSignal.allocation>` is used by each ControlSignal to determine
its `intensity <ControlSignal.intensity>`, which is then assigned as the `value <ControlProjection.value>` of the
ControlSignal's `ControlProjection`.   The `value <ControlProjection.value>` of the ControlProjection is used by the
`ParameterState` to which it projects to modify the value of the parameter it controls (see
`ControlSignal_Modulation` for description of how a ControlSignal modulates the value of a parameter).


.. _ControlMechanism_Execution:

Execution
---------

A ControlMechanism that is a System's `controller` is always the last `Mechanism <Mechanism>` to be executed in a
`TRIAL` for that System (see `System Control <System_Execution_Control>` and `Execution <System_Execution>`).  The
ControlMechanism's `function <ControlMechanism.function>` takes as its input the `value <InputState.value>` of
its *ERROR_SIGNAL* `input_state <Mechanism_Base.input_state>`, and uses that to determine its `allocation_policy
<ControlMechanism.allocation_policy>` which specifies the value assigned to the `allocation
<ControlSignal.allocation>` of each of its `ControlSignals <ControlSignal>`.  Each ControlSignal uses that value to
calculate its `intensity <ControlSignal.intensity>`, which is used by its `ControlProjection(s) <ControlProjection>`
to modulate the value of the ParameterState(s) for the parameter(s) it controls, which are then used in the
subsequent `TRIAL` of execution.

.. note::
   A `ParameterState` that receives a `ControlProjection` does not update its value until its owner Mechanism
   executes (see `Lazy Evaluation <LINK>` for an explanation of "lazy" updating).  This means that even if a
   ControlMechanism has executed, a parameter that it controls will not assume its new value until the Mechanism
   to which it belongs has executed.


.. _ControlMechanism_Examples:

Examples
~~~~~~~~

The following example creates a ControlMechanism by specifying its **objective_mechanism** using a constructor
that specifies the OutputStates to be monitored by its `objective_mechanism <ControlMechanism.objective_mechanism>`::

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

import warnings

import numpy as np
import typecheck as tc

from psyneulink.components.functions.function import LinearCombination, ModulationParam, _is_modulation_param
from psyneulink.components.mechanisms.adaptive.adaptivemechanism import AdaptiveMechanism_Base
from psyneulink.components.mechanisms.mechanism import Mechanism_Base
from psyneulink.components.states.outputstate import SEQUENTIAL, INDEX
from psyneulink.components.states.modulatorysignals.controlsignal import ControlSignal
from psyneulink.components.shellclasses import System_Base
from psyneulink.globals.defaults import defaultControlAllocation
from psyneulink.globals.keywords import \
    AUTO_ASSIGN_MATRIX,  INIT__EXECUTE__METHOD_ONLY, \
    PROJECTION_TYPE, CONTROL, CONTROL_PROJECTION, CONTROL_PROJECTIONS, CONTROL_SIGNAL, CONTROL_SIGNALS, \
    NAME, OBJECTIVE_MECHANISM, PRODUCT, PROJECTIONS, SYSTEM, VARIABLE, WEIGHT, EXPONENT
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.globals.utilities import ContentAddressableList
from psyneulink.scheduling.timescale import CentralClock, TimeScale

__all__ = [
    'ALLOCATION_POLICY', 'ControlMechanism', 'ControlMechanismError', 'ControlMechanismRegistry'
]

ALLOCATION_POLICY = 'allocation_policy'

ControlMechanismRegistry = {}

# MODIFIED 11/28/17 OLD:
# def _is_control_spec(spec):
#     from psyneulink.components.projections.modulatory.controlprojection import ControlProjection
#     if isinstance(spec, tuple):
#         return _is_control_spec(spec[1])
#     elif isinstance(spec, (ControlMechanism, ControlSignal, ControlProjection)):
#         return True
#     elif isinstance(spec, type) and issubclass(spec, ControlSignal):
#         return True
#     elif isinstance(spec, str) and spec in {CONTROL, CONTROL_PROJECTION, CONTROL_SIGNAL}:
#         return True
#     else:
#         return False

# MODIFIED 11/28/17 NEW:
def _is_control_spec(spec):
    from psyneulink.components.projections.modulatory.controlprojection import ControlProjection
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
# MODIFIED 11/28/17 END


class ControlMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


# class ControlMechanism(Mechanism_Base):
class ControlMechanism(AdaptiveMechanism_Base):
    """
    ControlMechanism(                         \
        system=None                                \
        objective_mechanism=None,                  \
        function=Linear,                           \
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

    function : TransferFunction : default Linear(slope=1, intercept=0)
        specifies function used to combine values of monitored OutputStates.

    control_signals : ControlSignal specification or List[ControlSignal specification, ...]
        specifies the parameters to be controlled by the ControlMechanism; a `ControlSignal` is created for each
        (see `ControlSignal_Specification` for details of specification).

    modulation : ModulationParam : ModulationParam.MULTIPLICATIVE
        specifies the default form of modulation used by the ControlMechanism's `ControlSignals <ControlSignal>`,
        unless they are `individually specified <ControlSignal_Specification>`.

    params : Dict[param keyword, param value] : default None
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
        **objective_mechanism** argument, and transmits the result to the ControlMechanism's *ERROR_SIGNAL*
        `input_state <Mechanism_Base.input_state>`.

    monitored_output_states : List[OutputState]
        each item is an `OutputState` monitored by the ObjectiveMechanism listed in the ControlMechanism's
        `objective_mechanism <ControlMechanism.objective_mechanism>` attribute;  it is the same as that
        ObjectiveMechanism's `monitored_output_states <ObjectiveMechanism.monitored_output_states>` attribute
        (see `ObjectiveMechanism_Monitored_Output_States` for specification).  The `value <OutputState.value>`
        of the OutputStates listed are used by the ObjectiveMechanism to generate the ControlMechanism's `input
        <ControlMechanism_Input>`.

    monitored_output_states_weights_and_exponents : List[Tuple(float, float)]
        each tuple in the list contains the weight and exponent associated with a corresponding item of
        `monitored_output_states <ControlMechanism.monitored_output_states>`;  these are the same as those in
        the `monitored_output_states_weights_and_exponents
        <ObjectiveMechanism.monitored_output_states_weights_and_exponents>` attribute of the `objective_mechanism
        <ControlMechanism.objective_mechanism>`, and are used by the ObjectiveMechanism's `function
        <ObjectiveMechanism.function>` to parametrize the contribution made to its output by each of the values that
        it monitors (see `ObjectiveMechanism Function <ObjectiveMechanism_Function>`).

    function : TransferFunction : default Linear(slope=1, intercept=0)
        determines how the `value <OuputState.value>` \\s of the `OutputStates <OutputState>` specified in the
        **monitor_for_control** argument of the ControlMechanism's constructor are used to generate its
        `allocation_policy <ControlMechanism.allocation_policy>`.

    allocation_policy : 2d np.array
        each item is the value assigned as the `allocation <ControlSignal.allocation>` for the corresponding
        ControlSignal listed in the `control_signals` attribute;  the allocation_policy is the same as the
        ControlMechanism's `value <Mechanism_Base.value>` attribute).

    control_signals : ContentAddressableList[ControlSignal]
        list of the `ControlSignals <ControlSignals>` for the ControlMechanism, including any inherited from a
        `system <ControlMechanism.system>` for which it is a `controller <System.controller>` (same as
        ControlMechanism's `output_states <Mechanism_Base.output_states>` attribute); each sends a `ControlProjection`
        to the `ParameterState` for the parameter it controls

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

    initMethod = INIT__EXECUTE__METHOD_ONLY

    outputStateType = ControlSignal
    stateListAttr = Mechanism_Base.stateListAttr.copy()
    stateListAttr.update({ControlSignal:CONTROL_SIGNALS})

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'ControlMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    class ClassDefaults(AdaptiveMechanism_Base.ClassDefaults):
        # This must be a list, as there may be more than one (e.g., one per control_signal)
        variable = defaultControlAllocation

    from psyneulink.components.functions.function import Linear
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        OBJECTIVE_MECHANISM: None,
        ALLOCATION_POLICY: None,
        CONTROL_PROJECTIONS: None})

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 system:tc.optional(System_Base)=None,
                 objective_mechanism=None,
                 function = Linear(slope=1, intercept=0),
                 control_signals=None,
                 modulation:tc.optional(_is_modulation_param)=ModulationParam.MULTIPLICATIVE,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(system=system,
                                                  objective_mechanism=objective_mechanism,
                                                  function=function,
                                                  control_signals=control_signals,
                                                  modulation=modulation,
                                                  params=params)

        super(ControlMechanism, self).__init__(variable=default_variable,
                                                    size=size,
                                                    modulation=modulation,
                                                    params=params,
                                                    name=name,
                                                    prefs=prefs,
                                                    context=self)

        try:
            self.monitored_output_states
        except AttributeError:
            raise ControlMechanismError("{} (subclass of {}) must implement a \'monitored_output_states\' attribute".
                                              format(self.__class__.__name__,
                                                     self.__class__.__bases__[0].__name__))

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate SYSTEM, MONITOR_FOR_CONTROL and CONTROL_SIGNALS

        If System is specified, validate it
        Check that all items in MONITOR_FOR_CONTROL are Mechanisms or OutputStates for Mechanisms in self.system
        Check that all items in CONTROL_SIGNALS are parameters or ParameterStates for Mechanisms in self.system
        """
        from psyneulink.components.system import MonitoredOutputStateTuple
        from psyneulink.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
        from psyneulink.components.states.inputstate import InputState
        from psyneulink.components.states.state import _parse_state_spec

        super(ControlMechanism, self)._validate_params(request_set=request_set,
                                                       target_set=target_set,
                                                       context=context)
        if SYSTEM in target_set:
            if not isinstance(target_set[SYSTEM], System_Base):
                raise KeyError
            else:
                self.paramClassDefaults[SYSTEM] = request_set[SYSTEM]

        if OBJECTIVE_MECHANISM in target_set and target_set[OBJECTIVE_MECHANISM] is not None:

            if isinstance(target_set[OBJECTIVE_MECHANISM], list):

                obj_mech_spec_list = target_set[OBJECTIVE_MECHANISM]

                # Check if there is any ObjectiveMechanism in the list
                #    (incorrect but possibly forgivable mis-specification
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
                    for spec in obj_mech_spec_list:
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
                            self.system._validate_monitored_state_in_system([spec], context=context)

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
        Assign InputState to ControlMechanism for each OutputState to be monitored;
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
        from psyneulink.components.system import MonitoredOutputStateTuple
        from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
        from psyneulink.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism, ObjectiveMechanismError
        from psyneulink.components.states.inputstate import EXPONENT_INDEX, WEIGHT_INDEX

        monitored_output_states = None

        # GET OutputStates to Monitor (to specify as or add to ObjectiveMechanism's monitored_output_states attribute

        # If the ControlMechanism has already been assigned to a System
        #    get OutputStates in System specified as MONITOR_FOR_CONTROL
        #        do this by calling _get_monitored_output_states_for_system(),
        #        which also gets any OutputStates already being monitored by the ControlMechanism
        if self.system:
            monitored_output_states = self.system._get_monitored_output_states_for_system(self, context=context)

        # Otherwise, if objective_mechanism argument was specified as a list, get the OutputStates specified in it
        # - IF ControlMechanism HAS NOT ALREADY BEEN ASSIGNED TO A SYSTEM:
        #      IF objective_mechanism IS SPECIFIED AS A LIST:
        #          CALL _parse_monitored_output_states_list() TO GET LIST OF OutputStates
        #          CALL CONSTRUCTOR WITH monitored_output_states AND monitoring_input_states
        #      IF objective_mechanism IS ALREADY AN INSTANTIATED ObjectiveMechanism:
        #          JUST ASSIGN TO objective_mechanism ATTRIBUTE
        if isinstance(self.objective_mechanism, list):
            monitored_output_states = [None] * len(self.objective_mechanism)
            for i, item in enumerate(self.objective_mechanism):
                # If it is a MonitoredOutputStateTuple, create InputState specification dictionary
                # Otherwise, assume it is a valid form of InputSate specification, and pass to ObjectiveMechanism
                if isinstance(item, MonitoredOutputStateTuple):
                    # Create InputState specification dictionary:
                    monitored_output_states[i] = {NAME: item.output_state.name,
                                                  VARIABLE: item.output_state.value,
                                                  WEIGHT:item.weight,
                                                  EXPONENT:item.exponent,
                                                  PROJECTIONS:[(item.output_state, item.matrix)]}

        # INSTANTIATE ObjectiveMechanism

        # If *objective_mechanism* argument si an ObjectiveMechanism, add monitored_output_states to it
        if isinstance(self.objective_mechanism, ObjectiveMechanism):
            if monitored_output_states:
                self.objective_mechanism.add_monitored_output_states(
                                                              monitored_output_states_specs=monitored_output_states,
                                                              context=context)
        # Otherwise, instantiate ObjectiveMechanism with list of states in *objective_mechanism* arg
        else:
            # Create specification for ObjectiveMechanism InputStates corresponding to
            #    monitored_output_states and their exponents and weights
            try:
                self._objective_mechanism = ObjectiveMechanism(monitored_output_states=monitored_output_states,
                                                               function=LinearCombination(operation=PRODUCT),
                                                               name=self.name + '_ObjectiveMechanism')

            except ObjectiveMechanismError as e:
                raise ObjectiveMechanismError(e)

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

        MappingProjection(sender=self.objective_mechanism,
                          receiver=self,
                          matrix=AUTO_ASSIGN_MATRIX,
                          name=name)

    def _instantiate_input_states(self, context=None):
        super()._instantiate_input_states(context=context)

        # IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION
        # ONCE THAT IS IMPLEMENTED
        self._instantiate_objective_mechanism(context=context)

    def _instantiate_output_states(self, context=None):

        from psyneulink.globals.registry import register_category
        from psyneulink.components.states.state import State_Base

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

            # for i, control_signal in enumerate(self.control_signals):
            #     self._instantiate_control_signal(control_signal, index=i, context=context)
            for control_signal in self.control_signals:
                self._instantiate_control_signal(control_signal, context=context)

        super()._instantiate_output_states(context=context)

        # Reassign control_signals to capture any user_defined ControlSignals instantiated by in call to super
        #    and assign to ContentAddressableList
        self._control_signals = ContentAddressableList(component_type=ControlSignal,
                                                       list=[state for state in self.output_states
                                                             if isinstance(state, ControlSignal)])

        if self.allocation_policy is None:
            self.allocation_policy = self.default_value

        # If the ControlMechanism's allocation_policy has more than one item,
        #    warn if the number of items does not equal the number of its ControlSignals
        #    (note:  there must be fewer ControlSignals than items in allocation_policy,
        #            as the reverse is an error that is checked for in _instantiate_control_signal)
        if len(self.allocation_policy)>1 and len(self.control_signals) != len(self.allocation_policy):
            if self.verbosePref:
                warnings.warning("The number of {}s for {} ({}) does not equal the number of items in its {} ({})".
                                 format(ControlSignal.__name__, self.name, len(self.control_signals),
                                        ALLOCATION_POLICY, len(self.allocation_policy)))


    # def _instantiate_control_signal(self, control_signal, index=0, context=None):
    def _instantiate_control_signal(self, control_signal, context=None):

        # EXTEND allocation_policy TO ACCOMMODATE NEW ControlSignal -------------------------------------------------
        #        also used to determine constraint on ControlSignal value
        if self.allocation_policy is None:
            self.allocation_policy = np.atleast_2d(defaultControlAllocation)
        else:
            self.allocation_policy = np.append(self.allocation_policy, [defaultControlAllocation], axis=0)

        # Update self.value to reflect change in allocation_policy (and the new number of  control_signals).
        #    This is necessary, since function is not fully executed during init (in _instantiate_function);
        #    it returns the default_allocation policy which has only a single item,
        #    however validation of indices for OutputStates requires proper number of items be in self.value
        self.value = self.allocation_policy
        self._default_value = self.value

        from psyneulink.components.states.state import _instantiate_state
        # Parses control_signal specifications (in call to State._parse_state_spec)
        #    and any embedded Projection specifications (in call to <State>._instantiate_projections)
        control_signal = _instantiate_state(state_type=ControlSignal,
                                            owner=self,
                                            reference_value=defaultControlAllocation,
                                            modulation=self.modulation,
                                            state_spec=control_signal)

        if control_signal.index is SEQUENTIAL:
            control_signal.index = len(self.allocation_policy)-1
        elif not isinstance(control_signal.index, int):
            raise ControlMechanismError("PROGRAM ERROR: {} attribute of {} for {} is not {} or an int".
                                        format(INDEX, ControlSignal.__name__, SEQUENTIAL, self.name))

        # Validate index
        try:
            self.allocation_policy[control_signal.index]
        except IndexError:
            raise ControlMechanismError("Index specified for {} of {} ({}) "
                                       "exceeds the number of items of its {} ({})".
                                       format(ControlSignal.__name__, self.name, control_signal.index,
                                              ALLOCATION_POLICY, len(self.allocation_policy)))

        # Add ControlProjection(s) to ControlMechanism's list of ControlProjections
        try:
            self.control_projections.extend(control_signal.efferents)
        except AttributeError:
            self.control_projections = control_signal.efferents.copy()

        # Update control_signal_costs to accommodate instantiated Projection
        try:
            self.control_signal_costs = np.append(self.control_signal_costs, np.empty((1,1)),axis=0)
        except AttributeError:
            self.control_signal_costs = np.empty((1,1))

        # UPDATE output_states AND control_projections -------------------------------------------------------------

        # TBI: For control mechanisms that accumulate, starting output must be equal to the initial "previous value"
        # so that modulation that occurs BEFORE the control mechanism executes is computed appropriately
        # if (isinstance(self.function_object, Integrator)):
        #     control_signal._intensity = function_object.initializer

        # Add ControlSignal to output_states list
        self._output_states.append(control_signal)

        return control_signal



    def _execute(self,
                 variable=None,
                 runtime_params=None,
                 clock=CentralClock,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """Updates ControlSignals based on inputs

        Must be overriden by subclass
        """
        # raise ControlMechanismError("{0} must implement execute() method".format(self.__class__.__name__))
        return self.input_values or [defaultControlAllocation]

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

        **monitored_output_states** can be a `Mechanism`, `OutputState`, `tuple specification
        <InputState_Tuple_Specification>`, a `State specification dicionary <InputState_Specification_Dictionary>`,
        or list with any of these. If item is a Mechanism, its `primary OutputState <OutputState_Primary>` is used.
        OutputStates must belong to Mechanisms in the same `System` as the ControlMechanism.
        """
        output_states = self.objective_mechanism.add_monitored_output_states(
                                                                 monitored_output_states_specs=monitored_output_states,
                                                                 context=context)
        if self.system:
            self.system._validate_monitored_state_in_system(output_states, context=context)

    @tc.typecheck
    def assign_as_controller(self, system:System_Base, context=None):
        """Assign ControlMechanism as `controller <System.controller>` for a `System`.

        **system** must be a System for which the ControlMechanism should be assigned as the `controller
        <System.controller>`;  if the specified System already has a `controller <System.controller>`,
        it will be replaced by the current one;  if the current one is already the `controller <System.controller>`
        for another System, it will be disabled for that System.
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

        # NEED TO BUFFER OBJECTIVE_MECHANISM AND CONTROL_SIGNAL ARGUMENTS FOR USE IN REINSTANTIATION HERE
        # DETACH AS CONTROLLER FOR ANY EXISTING SYSTEM (AND SET THAT ONE'S CONTROLLER ATTRIBUTE TO None)
        # DELETE ALL EXISTING OBJECTIVE_MECHANISM AND CONTROL_SIGNAL ASSIGNMENTS
        # REINSTANTIATE ITS OWN OBJECTIVE_MECHANISM and CONTROL_SIGNAL ARGUMENT AND THOSE OF THE SYSTEM
        # SUBCLASSES SHOULD ADD OVERRIDE FOR ANY CLASS-SPECIFIC ACTIONS (E.G., INSTANTIATING PREDICTION MECHANISMS)
        # DO *NOT* ASSIGN AS CONTROLLER FOR SYSTEM... LET THE SYSTEM HANDLE THAT
        # Assign the current System to the ControlMechanism

        # First, validate that all of the ControlMechanism's monitored_output_states and controlled parameters
        #    are in the new System
        system._validate_monitored_state_in_system(self.monitored_output_states)
        system._validate_control_signals(self.control_signals)

        # Next, get any OutputStates specified in the **monitored_output_states** argument of the System's
        #    constructor and/or in a MONITOR_FOR_CONTROL specification for individual OutputStates and/or Mechanisms,
        #    and add them to the ControlMechanism's monitored_output_states attribute and to its
        #    ObjectiveMechanisms monitored_output_states attribute
        monitored_output_states = list(system._get_monitored_output_states_for_system(controller=self, context=context))
        self.add_monitored_output_states(monitored_output_states)

        # Then, assign it ControlSignals for any parameters in the current System specified for control
        system_control_signals = system._get_control_signals_for_system(system.control_signals, context=context)
        for control_signal_spec in system_control_signals:
            self._instantiate_control_signal(control_signal=control_signal_spec, context=context)

        # If it HAS been assigned a System, make sure it is the current one
        if self.system and not self.system is system:
            raise SystemError("The controller being assigned to {} ({}) already belongs to another System ({})".
                              format(system.name, self.name, self.system.name))

        # Assign assign the current System to the ControlMechanism's system attribute
        #    (needed for it to validate and instantiate monitored_output_states and control_signals)
        self.system = system

        # Flag ObjectiveMechanism as associated with a ControlMechanism that is a controller for the System
        self._objective_mechanism.controller = True

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
            # # MODIFIED 11/25/17 OLD:
            # raise ControlMechanismError("Control Mechanism {}'s Objective "
            #                             "Mechanism has not been "
            #                             "instantiated.".format(self.name))
            # MODIFIED 11/25/17 NEW:
            return None

            # # MODIFIED 11/25/17 NEWER:
            # self._instantiate_objective_mechanism(context='INSTANTIATE_OBJECTIVE_MECHANISM')
            # return self.monitored_output_states
            # MODIFIED 11/25/17 END:

    @property
    def monitored_output_states_weights_and_exponents(self):
        return self._objective_mechanism.monitored_output_states_weights_and_exponents



