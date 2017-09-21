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
    uses an EVCControlMechanism as its `controller <System_Base.controller>`, it concludes by executing the EVCControlMechanism, which
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
<System_Base.controller>` of a `System`, that has a special relation to the System: it is used to control any and all
parameters that have been `specified for control <ControlMechanism_Control_Signals>` in that System.  A
ControlMechanism can be the `controller <System_Base.controller>` for only one System, and a System can have only one
one `controller <System_Base.controller>`.  The System's `controller <System_Base.controller>` is executed after all
of the other Components in the System have been executed, including any other ControlMechanisms that belong to it (see
`System Execution <System_Execution>`).  A ControlMechanism can be assigned as the `controller <System_Base.controller>`
for a System by specifying it in the **controller** argument of the System's constructor, or by specifying the System
as the **system** argument of either the ControlMechanism's constructor or its `assign_as_controller
<ControlMechanism.assign_as_controller>` method. A System's `controller  <System_Base.controller>` and its
associated Components can be displayed using the System's `show_graph <System_Base.show_graph>` method with its
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
    `ControlSignal` in a `tuple specification for the parameter.

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

    my_transfer_mech_A = TransferMechanism()
    my_DDM = DDM()
    my_transfer_mech_B = TransferMechanism(function=Logistic)

    my_control_mech = ControlMechanism(
                         objective_mechanism=ObjectiveMechanism(monitored_output_states=[(my_transfer_mech_A, 2, 1),
                                                                                  my_DDM.output_states[RESPONSE_TIME]],
                                                                function=LinearCombination(operation=PRODUCT)),
                         control_signals=[(THRESHOLD, my_DDM),
                                          (GAIN, my_transfer_mech_B)])

This creates an ObjectiveMechanism for the ControlMechanism that monitors the `primary OutputState
<OutputState_Primary>` of ``my_Transfer_mech_A`` and the *RESPONSE_TIME* OutputState of ``my_DDM``;  its function
first multiplies the former by 2 before, then takes product of ther values and passes the result as the input to the
ControlMechanism.  The ControlMechanism's `function <ControlMechanism.function>` uses this value to determine
the allocation for its ControlSignals, that control the value of the `threshold <DDM.threshold>` parameter of
``my_DDM`` and the  `gain <Logistic.gain>` parameter of the `Logistic` Function for ``my_transfer_mech_B``.

The following example specifies the same set of OutputStates for the ObjectiveMechanism, by assigning them directly
to the **objective_mechanism** argument::

    my_control_mech = ControlMechanism(
                            objective_mechanism=[(my_transfer_mech_A, 2, 1),
                                                 my_DDM.output_states[RESPONSE_TIME]],
                            control_signals=[(THRESHOLD, my_DDM),
                                             (GAIN, my_transfer_mech_B)])

Note that, while this form is more succinct, it precludes specifying the ObjectiveMechanism's function.  Therefore,
the values of the monitored OutputStates will be added (the default) rather than multiplied.

The ObjectiveMechanism can also be created on its own, and then referenced in the constructor for the ControlMechanism::

    my_obj_mech=ObjectiveMechanism(monitored_output_states=[(my_transfer_mech_A, 2, 1),
                                                     my_DDM.output_states[RESPONSE_TIME]],
                                   function=LinearCombination(operation=PRODUCT))

    my_control_mech = ControlMechanism(
                            objective_mechanism=my_obj_mech,
                            control_signals=[(THRESHOLD, my_DDM),
                                             (GAIN, my_transfer_mech_B)])

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
from collections import UserList

from PsyNeuLink.Components.Component import InitStatus
from PsyNeuLink.Components.Functions.Function import ModulationParam, _is_modulation_param, LinearCombination
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism_Base, MonitoredOutputStatesOption
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.AdaptiveMechanism import AdaptiveMechanism_Base
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism \
                                                           import ObjectiveMechanism, _parse_monitored_output_states
from PsyNeuLink.Components.Projections.Projection import _validate_receiver
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.ShellClasses import Mechanism, System
from PsyNeuLink.Components.States.ModulatorySignals.ControlSignal import _parse_control_signal_spec
from PsyNeuLink.Components.States.OutputState import OutputState
from PsyNeuLink.Globals.Defaults import defaultControlAllocation
from PsyNeuLink.Globals.Keywords import NAME, PARAMS, OWNER, INIT__EXECUTE__METHOD_ONLY, SYSTEM, MECHANISM, \
                                        PARAMETER_STATE, OBJECTIVE_MECHANISM, \
                                        PRODUCT, AUTO_ASSIGN_MATRIX, REFERENCE_VALUE, \
                                        CONTROLLED_PARAM, CONTROL_PROJECTION, CONTROL_PROJECTIONS, CONTROL_SIGNAL, \
                                        CONTROL_SIGNALS, CONTROL
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceLevel
from PsyNeuLink.Globals.Utilities import ContentAddressableList
from PsyNeuLink.Scheduling.TimeScale import CentralClock, TimeScale

OBJECTIVE_MECHANISM = 'objective_mechanism'
ALLOCATION_POLICY = 'allocation_policy'

ControlMechanismRegistry = {}

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
                this specification overrides any in System_Base.params[], but can be overridden by Mechanism.params[]
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
        <System_Base.controller>`.

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

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters
        for the Mechanism, parameters for its function, and/or a custom function and its parameters. Values
        specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default ControlMechanism-<index>
        a string used for the name of the Mechanism.
        If not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the `PreferenceSet` for the Mechanism.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    Attributes
    ----------

    system : System
        The `System` for which the ControlMechanism is a `controller <System_Base>`.  Note that this is distinct from
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
        ObjectiveMechanism's `monitored_output_states <ObjectiveMechanism.monitored_output_states>` attribute. The
        `value <OutputState.value>` of the OutputStates listed are used by the ObjectiveMechanism to generate the
        ControlMechanism's `input <ControlMechanism_Input>`.

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

    control_signals : List[ControlSignal]
        list of the `ControlSignals <ControlSignals>` for the ControlMechanism, including any inherited from a
        `system <ControlMechanism.system>` for which it is a `controller <System_Base.controller>` (same as
        ControlMechanism's `output_states <Mechanism_Base.output_states>` attribute); each sends a `ControlProjection`
        to the `ParameterState` for the parameter it controls

    control_projections : List[ControlProjection]
        list of `ControlProjections <ControlProjection>`, one for each `ControlSignal` in `control_signals`.

    modulation : ModulationParam
        the default form of modulation used by the ControlMechanism's `ControlSignals <GatingSignal>`,
        unless they are `individually specified <ControlSignal_Specification>`.
    """

    componentType = "ControlMechanism"

    initMethod = INIT__EXECUTE__METHOD_ONLY

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'ControlMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    class ClassDefaults(AdaptiveMechanism_Base.ClassDefaults):
        # This must be a list, as there may be more than one (e.g., one per control_signal)
        variable = defaultControlAllocation

    from PsyNeuLink.Components.Functions.Function import Linear
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        OBJECTIVE_MECHANISM: None,
        ALLOCATION_POLICY: None,
        CONTROL_PROJECTIONS: None})

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 system:tc.optional(System)=None,
                 objective_mechanism:tc.optional(tc.any(ObjectiveMechanism, list))=None,
                 function = Linear(slope=1, intercept=0),
                 # control_signals:tc.optional(list) = None,
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

        super(ControlMechanism, self)._validate_params(request_set=request_set,
                                                                 target_set=target_set,
                                                                 context=context)
        if SYSTEM in target_set:
            if not isinstance(target_set[SYSTEM], System):
                raise KeyError
            else:
                self.paramClassDefaults[SYSTEM] = request_set[SYSTEM]

        if OBJECTIVE_MECHANISM in target_set and target_set[OBJECTIVE_MECHANISM] is not None:

            if isinstance(target_set[OBJECTIVE_MECHANISM], list):
                output_state_list = target_set[OBJECTIVE_MECHANISM]
                for spec in output_state_list:
                    # MODIFIED 9/16/17 OLD:
                    if isinstance(spec, MonitoredOutputStatesOption):
                        continue
                    if isinstance(spec, tuple):
                        spec = spec[0]
                    if isinstance(spec, dict):
                        spec = spec[MECHANISM]
                    if isinstance(spec, (OutputState, Mechanism_Base)):
                        spec = spec.name
                    if not isinstance(spec, str):
                        raise ControlMechanismError("Specification of {} arg for {} appears to be a list of "
                                                    "Mechanisms and/or OutputStates to be monitored, but one"
                                                    "of the items ({}) is invalid".
                                                    format(OBJECTIVE_MECHANISM, self.name, spec))
                    # # MODIFIED 9/16/17 NEW:
                    # _parse_monitored_output_states(source=self, output_state_list=spec, context=context)
                    # MODIFIED 9/16/17 END

                    # If ControlMechanism has been assigned to a System,
                    #    check that all the items in the list used to specify objective_mechanism are in the same System
                    if self.system:
                        self.system._validate_monitored_state_in_system([spec], context=context)

            elif not isinstance(target_set[OBJECTIVE_MECHANISM], ObjectiveMechanism):
                raise ControlMechanismError("Specification of {} arg for {} ({}) must be an {}"
                                            "or a list of Mechanisms and/or OutputStates to be monitored for control".
                                            format(OBJECTIVE_MECHANISM,
                                                   self.name, target_set[OBJECTIVE_MECHANISM],
                                                   ObjectiveMechanism.componentName))

        if CONTROL_SIGNALS in target_set and target_set[CONTROL_SIGNALS]:
            from PsyNeuLink.Components.States.ModulatorySignals.ControlSignal import ControlSignal
            # # MODIFIED 9/17/17 OLD:
            # if not isinstance(target_set[CONTROL_SIGNALS], (list, UserList)):
            #     raise ControlMechanismError("{} arg of {} must be list or ContentAddressableList".
            #                                 format(CONTROL_SIGNAL, self.name))
            # MODIFIED 9/17/17 NEW:
            if not isinstance(target_set[CONTROL_SIGNALS], list):
                target_set[CONTROL_SIGNALS] = [target_set[CONTROL_SIGNALS]]
            # _parse_control_signal_spec(self, target_set[CONTROL_SIGNALS], context=context)
            for control_signal in target_set[CONTROL_SIGNALS]:
                _parse_control_signal_spec(self, control_signal, context=context)
            # MODIFIED 9/17/17 END


    # IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
    def _instantiate_objective_mechanism(self, context=None):
        """
        Assign InputState to ControlMechanism for each OutputState to be monitored;
            uses _instantiate_monitoring_input_state and _instantiate_control_mechanism_input_state to do so.
            For each item in self.monitored_output_states:
            - if it is a OutputState, call _instantiate_monitoring_input_state()
            - if it is a Mechanism, call _instantiate_monitoring_input_state for relevant Mechanism.outputStates
                (determined by whether it is a `TERMINAL` Mechanism and/or MonitoredOutputStatesOption specification)
            - each InputState is assigned a name with the following format:
                '<name of Mechanism that owns the monitoredOutputState>_<name of monitoredOutputState>_Monitor'

        Notes:
        * self.monitored_output_states is a list, each item of which is a Mechanism.output_state from which a
          Projection will be instantiated to a corresponding InputState of the ControlMechanism
        * self.input_states is the usual ordered dict of states,
            each of which receives a Projection from a corresponding OutputState in self.monitored_output_states
        """
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
        elif isinstance(self.objective_mechanism, list):
            monitored_output_states = _parse_monitored_output_states(source=self,
                                                              output_state_list=self.objective_mechanism,
                                                              context=context)

        if isinstance(self.objective_mechanism, ObjectiveMechanism):
            if monitored_output_states:
                self.objective_mechanism.add_monitored_output_states(
                                                              monitored_output_states_specs=monitored_output_states,
                                                              context=context)
        else:
            # Create specification for ObjectiveMechanism InputStates corresponding to
            #    monitored_output_states and their exponents and weights
            self._objective_mechanism = ObjectiveMechanism(monitored_output_states=monitored_output_states,
                                                           function=LinearCombination(operation=PRODUCT),
                                                           name=self.name + '_ObjectiveMechanism')
        # Print monitored_output_states
        if self.prefs.verbosePref:
            print ("{0} monitoring:".format(self.name))
            for state in self.monitored_output_states:
                weight = self.monitored_output_states_weights_and_exponents[
                                                                self.monitored_output_states.index(state)][0]
                exponent = self.monitored_output_states_weights_and_exponents[
                                                                self.monitored_output_states.index(state)][1]
                print ("\t{0} (exp: {1}; wt: {2})".format(state.name, weight, exponent))

        # Assign ObjetiveMechanism's role as CONTROL
        self.objective_mechanism._role = CONTROL

        # If ControlMechanism is a System controller, name Projection from ObjectiveMechanism based on the System
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

        # IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
        self._instantiate_objective_mechanism(context=context)

    def _instantiate_output_states(self, context=None):

        # Create registry for GatingSignals (to manage names)
        from PsyNeuLink.Globals.Registry import register_category
        from PsyNeuLink.Components.States.ModulatorySignals.ControlSignal import ControlSignal
        from PsyNeuLink.Components.States.State import State_Base
        register_category(entry=ControlSignal,
                          base_class=State_Base,
                          registry=self._stateRegistry,
                          context=context)

        if self.control_signals:
            for control_signal in self.control_signals:
                self._instantiate_control_signal(control_signal=control_signal, context=context)

        # IMPLEMENTATION NOTE:  Don't want to call this because it instantiates undesired default OutputState
        # super()._instantiate_output_states(context=context)

    # ---------------------------------------------------
    # IMPLEMENTATION NOTE:  IMPLEMENT _instantiate_output_states THAT CALLS THIS FOR EACH ITEM
    #                       DESIGN PATTERN SHOULD COMPLEMENT THAT FOR _instantiate_input_states of ObjectiveMechanism
    #                           (with control_signals taking the place of monitored_output_states)
    # FIX 5/23/17: PROJECTIONS AND PARAMS SHOULD BE PASSED BY ASSIGNING TO STATE SPECIFICATION DICT
    # FIX          UPDATE parse_state_spec TO ACCOMODATE (param, ControlSignal) TUPLE
    # FIX          TRACK DOWN WHERE PARAMS ARE BEING HANDED OFF TO ControlProjection
    # FIX                   AND MAKE SURE THEY ARE NOW ADDED TO ControlSignal SPECIFICATION DICT
    #
    def _instantiate_control_signal(self, control_signal=None, context=None):
        """Instantiate ControlSignal OutputState and assign (if specified) or instantiate ControlProjection

        # Extends allocation_policy and control_signal_costs attributes to accommodate instantiated Projection

        Notes:
        * control_signal arg can be a:
            - ControlSignal object;
            - ControlProjection;
            - ParameterState;
            - params dict containing a ControlProjection;
            - tuple (param_name, Mechanism), from control_signals arg of constructor;
                    [NOTE: this is a convenience format;
                           it precludes specification of ControlSignal params (e.g., ALLOCATION_SAMPLES)]
            - ControlSignal specification dictionary, from control_signals arg of constructor
                    [NOTE: this must have at least NAME:str (param name) and MECHANISM:Mechanism entries;
                           it can also include a PARAMS entry with a params dict containing ControlSignal params]
        * State._parse_state_spec() is used to parse control_signal arg
        * params are expected to be for (i.e., to be passed to) ControlSignal;
        * wait to instantiate deferred_init() Projections until after ControlSignal is instantiated,
             so that correct OutputState can be assigned as its sender;
        * index of OutputState is incremented based on number of ControlSignals already instantiated;
            this means that the ControlMechanism's function must return as many items as it has ControlSignals,
            with each item of the function's value used by a corresponding ControlSignal.
            Note: multiple ControlProjections can be assigned to the same ControlSignal to achieve "divergent control"
                  (that is, control of many parameters with a single value)

        Returns ControlSignal (OutputState)
        """
        from PsyNeuLink.Components.States.ModulatorySignals.ControlSignal import ControlSignal
        # from PsyNeuLink.Components.States.ParameterState import _get_parameter_state
        # from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection

        # EXTEND allocation_policy TO ACCOMMODATE NEW ControlSignal -------------------------------------------------
        #        also used to determine constraint on ControlSignal value

        if self.allocation_policy is None:
            self.allocation_policy = np.array(defaultControlAllocation)
        else:
            self.allocation_policy = np.append(self.allocation_policy, defaultControlAllocation)

        # Update self.value to reflect change in allocation_policy (and the new number of  control_signals);
        #    this is necessary, since function is not fully executed during initialization (in _instantiate_function)
        #    it returns default_allocation policy which has only a single item,
        #    however validation of indices for OutputStates requires that proper number of items be in self.value
        self.value = self.allocation_policy
        self._default_value = self.value

        # PARSE control_signal SPECIFICATION -----------------------------------------------------------------------
        control_signal_spec = _parse_control_signal_spec(owner=self, control_signal_spec=control_signal)
        param_name = control_signal_spec[NAME]
        control_signal_params = control_signal_spec[PARAMS]
        control_projection = control_signal_spec[CONTROL_PROJECTION]
        parameter_state = control_signal_spec[PARAMETER_STATE]

        default_name = param_name + '_' + ControlSignal.__name__

        # MODIFIED 9/11/17 OLD:
        # # Get constraint for ControlSignal value
        # #    - get ControlMechanism's value
        # self._update_value(context=context)
        # MODIFIED 9/11/17 END

        # - get OutputState's index
        try:
            output_state_index = len(self.output_states)
        except (AttributeError, TypeError):
            output_state_index = 0
        # - get constraint for OutputState's value
        output_state_constraint_value = self.allocation_policy[output_state_index]

        # Specification is a ControlSignal (either passed in directly, or parsed from tuple above)
        if isinstance(control_signal_spec, ControlSignal):
            # Deferred Initialization, so assign owner, name, and initialize
            if control_signal_spec.init_status is InitStatus.DEFERRED_INITIALIZATION:
                control_signal_spec.init_args[OWNER] = self
                control_signal_spec.init_args[NAME] = control_signal_spec.init_args[NAME] or default_name
                # control_signal_spec.init_args[REFERENCE_VALUE] = output_state_constraint_value
                control_signal_spec.init_args[REFERENCE_VALUE] = defaultControlAllocation
                control_signal_spec._deferred_init(context=context)
                control_signal = control_signal_spec
            elif not control_signal_spec.owner is self:
                raise ControlMechanismError("Attempt to assign ControlSignal to {} ({}) that is already owned by {}".
                                            format(self.name, control_signal_spec.name, control_signal_spec.owner.name))
            else:
                control_signal = control_signal_spec
                control_signal_name = control_signal_spec.name
                control_projections = control_signal_spec.efferents

                # IMPLEMENTATION NOTE:
                #    THIS IS TO HANDLE FUTURE POSSIBILITY OF MULTIPLE ControlProjections FROM A SINGLE ControlSignal;
                #    FOR NOW, HOWEVER, ONLY A SINGLE ONE IS SUPPORTED
                # parameter_states = [proj.recvr for proj in control_projections]
                if len(control_projections) > 1:
                    raise ControlMechanismError("PROGRAM ERROR: list of ControlProjections is not currently supported "
                                                "as specification in a ControlSignal")
                else:
                    control_projection = control_projections[0]
                    parameter_state = control_projection.receiver

        # Instantiate OutputState for ControlSignal
        else:
            control_signal_name = default_name

            from PsyNeuLink.Components.States.ModulatorySignals.ControlSignal import ControlSignal
            from PsyNeuLink.Components.States.State import _instantiate_state

            control_signal_params.update({CONTROLLED_PARAM:param_name})

            # FIX 5/23/17: CALL super()_instantiate_output_states ??
            # FIX:         OR AGGREGATE ALL ControlSignals AND SEND AS LIST (AS FOR input_states IN ObjectiveMechanism)
            control_signal = _instantiate_state(owner=self,
                                                state_type=ControlSignal,
                                                state_name=control_signal_name,
                                                state_spec=defaultControlAllocation,
                                                state_params=control_signal_params,
                                                constraint_value=output_state_constraint_value,
                                                constraint_value_name='Default control allocation',
                                                context=context)

        # VALIDATE OR INSTANTIATE ControlProjection(s) TO ControlSignal  -------------------------------------------

        # Validate control_projection (if specified) and get receiver's name
        control_projection_name = parameter_state.name + ' ' + 'control signal'
        if control_projection:
            _validate_receiver(self, control_projection, Mechanism, CONTROL_SIGNAL, context=context)

            from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
            if not isinstance(control_projection, ControlProjection):
                raise ControlMechanismError("PROGRAM ERROR: Attempt to assign {}, "
                                                  "that is not a ControlProjection, to ControlSignal of {}".
                                                  format(control_projection, self.name))
            if control_projection.init_status is InitStatus.DEFERRED_INITIALIZATION:
                control_projection.init_args['sender']=control_signal
                if control_projection.init_args['name'] is None:
                    # FIX 5/23/17: CLEAN UP NAME STUFF BELOW:
                    control_projection.init_args['name'] = control_projection_name
                        # CONTROL_PROJECTION + ' for ' + parameter_state.owner.name + ' ' + parameter_state.name
                control_projection._deferred_init()
            else:
                control_projection.sender = control_signal

        # Instantiate ControlProjection
        else:
            # IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
            from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
            control_projection = ControlProjection(sender=control_signal,
                                                   receiver=parameter_state,
                                                   # name=CONTROL_PROJECTION + control_signal_name)
                                                   name=control_projection_name)

        # Add ControlProjection to list of OutputState's outgoing Projections
        # (note: if it was deferred, it just added itself, skip)
        if not control_projection in control_signal.efferents:
            control_signal.efferents.append(control_projection)

        # Add ControlProjection to ControlMechanism's list of ControlProjections
        try:
            self.control_projections.append(control_projection)
        except AttributeError:
            self.control_projections = [control_projection]

        # Update control_signal_costs to accommodate instantiated Projection
        try:
            self.control_signal_costs = np.append(self.control_signal_costs, np.empty((1,1)),axis=0)
        except AttributeError:
            self.control_signal_costs = np.empty((1,1))

        # UPDATE output_states AND control_projections -------------------------------------------------------------

        try:
            self._output_states[control_signal.name] = control_signal
        except (AttributeError, TypeError):
            from PsyNeuLink.Components.States.State import State_Base
            self._output_states = ContentAddressableList(component_type=State_Base,
                                                        list=[control_signal],
                                                        name = self.name+'.output_states')

        # Add index assignment to OutputState
        control_signal.index = output_state_index

        # (Re-)assign control_signals attribute to output_states
        self._control_signals = self.output_states

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

        print ("\n---------------------------------------------------------")

        print ("\n{0}".format(self.name))
        print("\n\tMonitoring the following Mechanism OutputStates:")
        for state in self.objective_mechanism.input_states:
            for projection in state.path_afferents:
                monitored_state = projection.sender
                monitored_state_mech = projection.sender.owner
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

        **monitored_output_states** can be a `Mechanism`, `OutputState`, `monitored_output_states tuple
        <ObjectiveMechanism_OutputState_Tuple>`, or list with any of these. If item is a Mechanism, its `primary
        OutputState <OutputState_Primary>` is used. OutputStates must belong to Mechanisms in the same `System` as
        the ControlMechanism.
        """
        output_states = self.objective_mechanism.add_monitored_output_states(
                                                                 monitored_output_states_specs=monitored_output_states,
                                                                 context=context)
        if self.system:
            self.system._validate_monitored_state_in_system(output_states, context=context)

    @tc.typecheck
    def assign_as_controller(self, system:System, context=None):
        """Assign ControlMechanism as `controller <System_Base.controller>` for a `System`.

        **system** must be a System for which the ControlMechanism should be assigned as the `controller
        <System_Base.controller>`;  if the specified System already has a `controller <System_Base.controller>`,
        it will be replaced by the current one;  if the current one is already the `controller <System_Base.controller>`
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
                                    `controller <System_Base.controller>`;]

                                  - any class-specific actions that must be taken to instantiate the ControlMechanism
                                    can be handled by subclasses of ControlMechanism (e.g., an EVCControlMechanism must
                                    instantiate its Prediction Mechanisms). However, the actual assignment of the
                                    ControlMechanism the System's `controller <System_Base.controller>` attribute must
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
        return self.objective_mechanism.monitored_output_states

    @property
    def monitored_output_states_weights_and_exponents(self):
        return self.objective_mechanism.monitored_output_states_weights_and_exponents



