# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  AGTControlMechanism ************************************************

"""

Overview
--------

An AGTControlMechanism is a `ControlMechanism <ControlMechanism>` that uses an ObjectiveMechanism with a `AGTUtilityIntegrator`
Function to regulate its `allocation_policy <ControlMechanism.allocation_policy>`.  When used with an `LCControlMechanism`
to regulate the `mode <FHNIntegrator.mode>` parameter of its `FHNIntegrator` Function, it implements a form of the 
`Adaptive Gain Theory <http://www.annualreviews.org/doi/abs/10.1146/annurev.neuro.28.061604.135709>`_ of the locus 
coeruleus-norepinephrine (LC-NE) system.

.. _AGTControlMechanism_Creation:

Creating an AGTControlMechanism
-----------------------

An AGTControlMechanism can be created in any of the ways used to `create a ControlMechanism <ControlMechanism_Creation>`.

Like all ControlMechanisms, an AGTControlMechanism it receives its `input <AGTControlMechanism_Input>` from an `ObjectiveMechanism`.
However, unlike standard ControlMechanism, an AGTControlMechanism does not have an **objective_mechanism** argument in its
constructor.  When an AGTControlMechanism is created, it automatically creates an ObjectiveMechanism and assigns a
`AGTUtilityIntegrator` Function as its `function <ObjectiveMechanism.function>`.

The OutputStates to be monitored by the AGTControlMechanism's `objective_mechanism <AGTControlMechanism.objective_mechanism>` are
specified using the **monitored_output_states** argument of the AGTControlMechanism's constructor, using any of the ways to
`specify the OutputStates monitored by ObjectiveMechanism <ObjectiveMechanism_Monitored_Output_States>`.  The
monitored OutputStates are listed in the LCControlMechanism's `monitored_output_states <AGTControlMechanism.monitored_output_states>`
attribute,  as well as that of its `objective_mechanism <AGTControlMechanism.objective_mechanism>`.

The parameter(s) controlled by an AGTControlMechanism are specified in the **control_signals** argument of its constructor,
in the `standard way for a ControlMechanism <ControlMechanism_Control_Signals>`.

.. _AGTControlMechanism_Structure:

Structure
---------

.. _AGTControlMechanism_Input:

Input: ObjectiveMechanism and Monitored OutputStates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An AGTControlMechanism has a single (primary) `InputState <InputState_Primary>` that receives its input via a
`MappingProjection` from the *OUTCOME* `OutputState <ObjectiveMechanism_Output>` of an `ObjectiveMechanism`.
The ObjectiveMechanism is created automatically when the AGTControlMechanism is created, using a `AGTUtilityIntegrator` as its
`function <ObjectiveMechanism.function>`, and is listed in the AGTControlMechanism's `objective_mechanism
<AGTControlMechanism.objective_mechanism>` attribute.  The ObjectiveMechanism aggregates the `value <OutputState.value>`\\s
of the OutputStates that it monitors, integrates their aggregated value at two different rates, and combines those to
generate the its output, which is used by the AGTControlMechanism as its input. The OutputStates monitored by the
ObjectiveMechanism, listed in its `monitored_output_states <ObjectiveMechanism.monitored_output_states>`
attribute, are also listed in the AGTControlMechanism's `monitored_output_states <AGTControlMechanism_Base.monitored_output_states>`
attribute.  They can be displayed using the AGTControlMechanism's `show <AGTControlMechanism.show>` method.

.. _AGTControlMechanism_Function:

Function
~~~~~~~~

An AGTControlMechanism uses the default function for a `ControlMechanism` (a default `Linear` Function), that simply passes
its input to its output.  Thus, it is the output of the AGTControlMechanism's `objective_mechanism
<AGTControlMechanism.objective_mechanism>` that determines its `allocation_policy <ControlMechanism.allocation_policy>`
and the `allocation <ControlSignal.allocation>` of its `ControlSignal(s) <ControlSignal>`.

.. _AGTControlMechanism_Output:

Output
~~~~~~

An AGTControlMechanism has a `ControlSignal` for each parameter specified in its `control_signals
<ControlMechanism.control_signals>` attribute, that sends a `ControlProjection` to the `ParameterState` for the
corresponding parameter. ControlSignals are a type of `OutputState`, and so they are also listed in the
AGTControlMechanism's `output_states <AGTControlMechanism_Base.output_states>` attribute. The parameters modulated by an
AGTControlMechanism's ControlSignals can be displayed using its `show <AGTControlMechanism_Base.show>` method. By default,
all of its ControlSignals are assigned the result of the AGTControlMechanism's `function <AGTControlMechanism.function>`, which is
the `input <AGTControlMechanism_Input>` it receives from its `objective_mechanism <AGTControlMechanism.objective_mechanism>`.
above).  The `allocation <ControlSignal.allocation>` is used by the ControlSignal(s) to determine
their `intensity <ControlSignal.intensity>`, which is then assigned as the `value <ControlProjection.value>` of the
ControlSignal's `ControlProjection`.   The `value <ControlProjection.value>` of the ControlProjection is used by the
`ParameterState` to which it projects to modify the value of the parameter it controls (see
`ControlSignal_Modulation` for description of how a ControlSignal modulates the value of a parameter).


.. _AGTControlMechanism_Execution:

Execution
---------

An AGTControlMechanism's `function <AGTControlMechanism_Base.function>` takes as its input the `value <InputState.value>` of
its *ERROR_SIGNAL* `input_state <Mechanism_Base.input_state>`, and uses that to determine its `allocation_policy
<ITC.allocation_policy>` which specifies the value assigned to the `allocation <ControlSignal.allocation>` of each of
its `ControlSignals <ControlSignal>`.  An AGTControlMechanism assigns the same value (the `input <AGTControlMechanism_Input>` it
receives from its `objective_mechanism <AGTControlMechanism.objective_mechanism>` to all of its ControlSignals.  Each
ControlSignal uses that value to calculate its `intensity <ControlSignal.intensity>`, which is used by its
`ControlProjection(s) <ControlProjection>` to modulate the value of the ParameterState(s) for the parameter(s) it
controls, which are then used in the subsequent `TRIAL` of execution.

.. note::
   A `ParameterState` that receives a `ControlProjection` does not update its value until its owner Mechanism
   executes (see `Lazy Evaluation <LINK>` for an explanation of "lazy" updating).  This means that even if a
   ControlMechanism has executed, a parameter that it controls will not assume its new value until the Mechanism
   to which it belongs has executed.


.. _AGTControlMechanism_Class_Reference:

Class Reference
---------------

"""
import typecheck as tc
import warnings

from PsyNeuLink.Components.Functions.Function \
    import ModulationParam, _is_modulation_param, AGTUtilityIntegrator
from PsyNeuLink.Components.System import System
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism \
    import ObjectiveMechanism, _parse_monitored_output_states, MonitoredOutputStatesOption
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.AdaptiveMechanism import AdaptiveMechanism_Base
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanism.ControlMechanism import ControlMechanism
from PsyNeuLink.Components.States.OutputState import OutputState
from PsyNeuLink.Components.ShellClasses import Mechanism
from PsyNeuLink.Globals.Defaults import defaultControlAllocation
from PsyNeuLink.Globals.Keywords import INIT__EXECUTE__METHOD_ONLY, MECHANISM, \
                                        OBJECTIVE_MECHANISM, CONTROL_PROJECTIONS, CONTROL_SIGNALS, CONTROL

from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceLevel
from PsyNeuLink.Scheduling.TimeScale import CentralClock, TimeScale

MONITORED_OUTPUT_STATES = 'monitored_output_states'
MONITORED_OUTPUT_STATE_NAME_SUFFIX = '_Monitor'

ControlMechanismRegistry = {}

class AGTControlMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class AGTControlMechanism(ControlMechanism):
    """
    AGTControlMechanism(                       \
        system=None,                    \
        monitored_output_states=None,   \
        function=Linear,                \
        control_signals=None,           \
        params=None,                    \
        name=None,                      \
        prefs=None)

    Subclass of `ControlMechanism <AdaptiveMechanism>` that modulates the `multiplicative_param
    <Function_Modulatory_Params>` of the `function <Mechanism_Base.function>` of one or more `Mechanisms <Mechanism>`.

    Arguments
    ---------

    system : System : default None
        specifies the `System` for which the AGTControlMechanism should serve as a `controller <System_Base.controller>`;
        the AGTControlMechanism will inherit any `OutputStates <OutputState>` specified in the **monitor_for_control**
        argument of the `system <EVCControlMechanism.system>`'s constructor, and any `ControlSignals <ControlSignal>`
        specified in its **control_signals** argument.

    monitored_output_states : List[`OutputState`, `Mechanism`, str, value, dict, `MonitoredOutputStatesOption`] or Dict
        specifies the OutputStates to be monitored by the `objective_mechanism <AGTControlMechanism.objective_mechanism>`
        (see `monitored_output_states <ObjectiveMechanism.monitored_output_states>` for details of specification).

    function : TransferFunction :  default Linear(slope=1, intercept=0)
        specifies the Function used to convert the AGTControlMechanism's `input <AGTControlMechanism_Input>` into its
        `allocation_policy <AGTControlMechanism.allocation_policy>`, that is used to assign the `allocation
        <ControlSignal.allocation>` of its `ControlSignal(s) <ControlSignal>`.

    control_signals : List[ParameterState, tuple[str, Mechanism] or dict]
        specifies the parameters to be controlled by the AGTControlMechanism; a `ControlSignal` is created for each
        (see `ControlSignal_Specification` for details of specification).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters
        for the Mechanism, parameters for its function, and/or a custom function and its parameters. Values
        specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default AGTControlMechanism-<index>
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
        the `System` for which AGTControlMechanism is the `controller <System_Base.controller>`;
        the AGTControlMechanism inherits any `OutputStates <OutputState>` specified in the **monitor_for_control**
        argument of the `system <EVCControlMechanism.system>`'s constructor, and any `ControlSignals <ControlSignal>`
        specified in its **control_signals** argument.

    objective_mechanism : ObjectiveMechanism
        `ObjectiveMechanism` that monitors and evaluates the values specified in the ControlMechanism's
        **objective_mechanism** argument, the output of which is used as `input <AGTControlMechanism_Input>` to the
        AGTControlMechanism. It is created automatically when AGTControlMechanism is created, and uses as a `AGTUtilityIntegrator` as
        is `function <ObjectiveMechanism.function>`.

    monitored_output_states : List[OutputState]
        each item is an `OutputState` monitored by the `objective_mechanism <AGTControlMechanism.objective_mechanism>`; it is
        the same as the ObjectiveMechanism's `monitored_output_states <ObjectiveMechanism.monitored_output_states>`
        attribute. The `value <OutputState.value>` of the OutputStates listed are used by the ObjectiveMechanism to
        generate the AGTControlMechanism's `input <AGTControlMechanism_Input>`.

    monitored_output_states_weights_and_exponents : List[Tuple(float, float)]
        each tuple in the list contains the weight and exponent associated with a corresponding item of
        `monitored_output_states <AGTControlMechanism.monitored_output_states>`;  these are the same as those in
        the `monitored_output_states_weights_and_exponents
        <ObjectiveMechanism.monitored_output_states_weights_and_exponents>` attribute of the `objective_mechanism
        <AGTControlMechanism.objective_mechanism>`, and are used by the ObjectiveMechanism's `function
        <ObjectiveMechanism.function>` to parametrize the contribution made to its output by each of the values that
        it monitors (see `ObjectiveMechanism Function <ObjectiveMechanism_Function>`).

    function : TransferFunction :  default Linear(slope=1, intercept=0)
        determines the Function used to convert the AGTControlMechanism's `input <AGTControlMechanism_Input>` into its
        `allocation_policy <AGTControlMechanism.allocation_policy>`, that is used to assign the
        `allocation <ControlSignal.allocation>` for its `ControlSignal(s) <ControlSignal>`.

    allocation_policy : 2d np.array
        contains the value(s) assigned as the `allocation <ControlSignal.allocation>` for the `ControlSignal(s)
        <ControlSignal>` listed in the `control_signals` attribute;  if the default `function <AGTControlMechanism.function>`
        is used, it contains a single value that is assigned as the `allocation <ControlSignal.allocation>` for
        all of the AGTControlMechanism's `control_signals <AGTControlMechanism.control_signals>`. The AGTControlMechanism's allocation_policy
        is the same as its `value <Mechanism_Base.value>` attribute).

    control_signals : List[ControlSignal]
        list of the AGTControlMechanism's `ControlSignals <ControlSignals>` , including any inherited from a `system
        <ControlMechanism.system>` for which it is a `controller <System_Base.controller>` (same as
        ControlMechanism's `output_states <Mechanism_Base.output_states>` attribute); each sends a `ControlProjection`
        to the `ParameterState` for the parameter it controls

    control_projections : List[ControlProjection]
        list of `ControlProjections <ControlProjection>`, one for each `ControlSignal` in `control_signals`.

    modulation : ModulationParam
        the default form of modulation used by the ControlMechanism's `ControlSignals <GatingSignal>`,
        unless they are `individually specified <ControlSignal_Specification>`.
   """

    componentName = "AGTControlMechanism"

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
    paramClassDefaults = ControlMechanism.paramClassDefaults.copy()
    paramClassDefaults.update({CONTROL_SIGNALS: None,
                               CONTROL_PROJECTIONS: None
                               })

    @tc.typecheck
    def __init__(self,
                 system:tc.optional(System)=None,
                 monitored_output_states=None,
                 function = Linear(slope=1, intercept=0),
                 # control_signals:tc.optional(list) = None,
                 control_signals= None,
                 modulation:tc.optional(_is_modulation_param)=ModulationParam.MULTIPLICATIVE,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function,
                                                  control_signals=control_signals,
                                                  params=params)

        super().__init__(system=system,
                         objective_mechanism=ObjectiveMechanism(monitored_output_states=monitored_output_states,
                                                                function=AGTUtilityIntegrator),
                         control_signals=control_signals,
                         modulation=modulation,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

        self.objective_mechanism.name = self.name+'_ObjectiveMechanism'
        self.objective_mechanism._role = CONTROL

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate SYSTEM, MONITOR_FOR_CONTROL and CONTROL_SIGNALS

        Check that all items in MONITOR_FOR_CONTROL are Mechanisms or OutputStates for Mechanisms in self.system
        Check that every item in `modulated_mechanisms <AGTControlMechanism.modulated_mechanisms>` is a Mechanism
            and that its function has a multiplicative_param
        """

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if MONITORED_OUTPUT_STATES in target_set and target_set[MONITORED_OUTPUT_STATES] is not None:
            # It is a MonitoredOutputStatesOption specification
            if isinstance(target_set[MONITORED_OUTPUT_STATES], MonitoredOutputStatesOption):
                # Put in a list (standard format for processing by _parse_monitored_output_states_list)
                target_set[MONITORED_OUTPUT_STATES] = [target_set[MONITORED_OUTPUT_STATES]]
            # It is NOT a MonitoredOutputStatesOption specification, so assume it is a list of Mechanisms or States
            else:
                # Validate each item of MONITORED_OUTPUT_STATES
                for item in target_set[MONITORED_OUTPUT_STATES]:
                    if isinstance(item, MonitoredOutputStatesOption):
                        continue
                    if isinstance(item, tuple):
                        item = item[0]
                    if isinstance(item, dict):
                        item = item[MECHANISM]
                    if isinstance(item, (OutputState, Mechanism)):
                        item = item.name
                    if not isinstance(item, str):
                        raise AGTControlMechanismError("Specification of {} arg for {} appears to be a list of "
                                                    "Mechanisms and/or OutputStates to be monitored, but one"
                                                    "of the items ({}) is invalid".
                                                    format(OBJECTIVE_MECHANISM, self.name, item))
                    _parse_monitored_output_states(source=self, output_state_list=item, context=context)


    def _execute(self,
                    variable=None,
                    runtime_params=None,
                    clock=CentralClock,
                    time_scale=TimeScale.TRIAL,
                    context=None):
        """Updates AGTControlMechanism's ControlSignal based on input and mode parameter value
        """
        return self.function(variable=variable,
                             params=runtime_params,
                             time_scale=time_scale,
                             context=context)

    @property
    def initial_short_term_utility(self):
        return self._objective_mechanism.function_object._initial_short_term_utility

    @initial_short_term_utility.setter
    def initial_short_term_utility(self, value):
        self._objective_mechanism.function_object.initial_short_term_utility = value

    @property
    def initial_long_term_utility(self):
        return self._objective_mechanism.function_object._initial_long_term_utility

    @initial_long_term_utility.setter
    def initial_long_term_utility(self, value):
        self._objective_mechanism.function_object.initial_long_term_utility = value

    @property
    def short_term_gain(self):
        return self._objective_mechanism.function_object._short_term_gain

    @short_term_gain.setter
    def short_term_gain(self, value):
        self._objective_mechanism.function_object.short_term_gain = value

    @property
    def long_term_gain(self):
        return self._objective_mechanism.function_object._long_term_gain

    @long_term_gain.setter
    def long_term_gain(self, value):
        self._objective_mechanism.function_object.long_term_gain = value

    @property
    def short_term_bias(self):
        return self._objective_mechanism.function_object._short_term_bias

    @short_term_bias.setter
    def short_term_bias(self, value):
        self._objective_mechanism.function_object.short_term_bias = value

    @property
    def    long_term_bias(self):
        return self._objective_mechanism.function_object._long_term_bias

    @long_term_bias.setter
    def long_term_bias(self, value):
        self._objective_mechanism.function_object.long_term_bias = value

    @property
    def    short_term_rate(self):
        return self._objective_mechanism.function_object._short_term_rate

    @short_term_rate.setter
    def short_term_rate(self, value):
        self._objective_mechanism.function_object.short_term_rate = value

    @property
    def    long_term_rate(self):
        return self._objective_mechanism.function_object._long_term_rate

    @long_term_rate.setter
    def long_term_rate(self, value):
        self._objective_mechanism.function_object.long_term_rate = value

    @property
    def operation(self):
        return self._objective_mechanism.function_object._operation

    @operation.setter
    def operation(self, value):
        self._objective_mechanism.function_object.operation = value



    def show(self):
        """Display the `OutputStates <OutputState>` monitored by the AGTControlMechanism's `objective_mechanism`
        and the `multiplicative_params <Function_Modulatory_Params>` modulated by the AGTControlMechanism.
        """

        print ("\n---------------------------------------------------------")

        print ("\n{0}".format(self.name))
        print("\n\tMonitoring the following Mechanism OutputStates:")
        if self.objective_mechanism is None:
            print ("\t\tNone")
        else:
            for state in self.objective_mechanism.input_states:
                for projection in state.path_afferents:
                    monitored_state = projection.sender
                    monitored_state_mech = projection.sender.owner
                    monitored_state_index = self.monitored_output_states.index(monitored_state)

                    weight = self.monitored_output_states_weights_and_exponents[monitored_state_index][0]
                    exponent = self.monitored_output_states_weights_and_exponents[monitored_state_index][1]

                    print ("\t\t{0}: {1} (exp: {2}; wt: {3})".
                           format(monitored_state_mech.name, monitored_state.name, weight, exponent))

        print ("\n\tModulating the following parameters:".format(self.name))
        # Sort for consistency of output:
        state_names_sorted = sorted(self.output_states.names)
        for state_name in state_names_sorted:
            for projection in self.output_states[state_name].efferents:
                print ("\t\t{0}: {1}".format(projection.receiver.owner.name, projection.receiver.name))

        print ("\n---------------------------------------------------------")
