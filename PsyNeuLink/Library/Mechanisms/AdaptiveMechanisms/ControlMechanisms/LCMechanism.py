# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  LCMechanism ************************************************

"""
Overview
--------

An LCMechanism is a `ControlMechanism <ControlMechanism>` that multiplicatively modulates the `function
<Mechanism.function>` of one or more `Mechanisms <Mechanism>` (usually `TransferMechanisms <TransferMechanism>`).
It implements an abstract model of the `locus coeruleus (LC)  <https://www.ncbi.nlm.nih.gov/pubmed/12371518>`_ that,
together with a `UtilityIntegrator` Mechanism, implement a form of the `Adaptive Gain Theory
<http://www.annualreviews.org/doi/abs/10.1146/annurev.neuro.28.061604.135709>`_ of the locus coeruleus-norepinephrine
(LC-NE) system.  The LCMechanism uses a `FitzHughNagumoIntegration` Function to generate its output, under the
influence of a `mode <LCMechanisms.mode>` parameter that regulates its operation between "tonic" to "phasic" modes of
responding -- see `Gilzenrat et al., <2002https://www.ncbi.nlm.nih.gov/pubmed/12371518>`_).

.. _LCMechanism_Creation:

Creating an LCMechanism
---------------------------

An LCMechanism can be created in any of the ways used to `create Mechanisms <Mechanism_Creation>`.  Like any Mechanism,
its **input_states** argument can be used to `specify Mechanisms (and/or their OutputState(s)
<Mechanism_State_Specification>` to project to the LCMechanism (i.e., to drive its response).  The `Mechanisms
<Mechanism>` it controls are specified in the **modulate** argument of its constructor (see `LCMechanism_modulate`).
COMMENT:
In addition, one or more Mechanisms can be specified to govern the LCMechanism's
`mode <LCMechanism.mode>` parameter, by specifying them in the **monitor_for_control** argument of its constructor
(see `LCMechanism_Monitored_OutputStates`).
COMMENT

.. _LCMechanism_Modulate:

Specifying Mechanisms to Modulate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Mechanisms to be modulated by a LCMechanism are specified in its **modulate** argument. An LCMechanism
controls a `Mechanism` by modifying the `multiplicative parameter <Function_Modulatory_Params>` of the
Mechanism's `function <TransferMechanism.function>`.  Therefore, any Mechanism specified for control by
an LCMechanism must be either a `TransferMechanism`, or a Mechanism that uses a `TransferFunction` or a class of
`Function <Function>` that implements a `multiplicative parameter <Function_Modulatory_Params>`.  The
**controls_signals** argument must be a list of such Mechanisms.  The keyword *ALL* can also be used to specify all
of the eligible `ProcessMechanisms <ProcessingMechanism> in all of the `Compositions <Composition>` to which the
LCMechanism belongs.  If a Mechanism specified in the **modulate** argument does not implement a multiplicative
parameter, it is ignored. A `ControlProjection` is automatically created that projects from the LCMechanism to the
`ParameterState` for the `multiplicative parameter <Function_Modulatory_Params>` of every Mechanism specified in the
**modulate** argument (and listed in its `modulate <LCMechanism.modulate>` attribute).


COMMENT:
.. _LCMechanism_Monitored_OutputStates:

Specifying Values to Monitor for Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the **monitor_for_control** argument is specified in the LCMechanism's constructor, it automatically creates a
`UtilityIntegratorMechanism` that is used to monitor and evaluate the `value <OutputState.value>` of the `OutputStates
<OutputState>` specified.  The **monitor_for_control** argument must be a list, each item of which must refer to a
`Mechanism <Mechanism>` or the `OutputState` of one.  These are assigned to the UtilityIntegratorMechanism's
`monitored_values <UtilityIntegratorMechanism>` attribute (and the LCMechanism's `monitored_output_states
<LCMechanism.monitored_output_states>` attribute). The UtilityIntegratorMechanism itself is assigned to the
LCMechanism's `monitoring_mechanism <LCMechanism.monitoring_mechanism>` attribute).
COMMENT

.. _LCMechanism_Structure:

Structure
---------

.. _LCMechanism_Input:

Input
~~~~~

An LCMechanism has a single (primary) `InputState <InputState_Primary>` that receives projections from any Mechanisms
specified in the **input_states** argument of the LCMechanism's constructor;  its `value <InputState.value>` is used as
the input to the LCMechanism's `function <LCMechanism.function>`.

.. _LCMechanism_Function:

Function
~~~~~~~~

An LCMechanism uses the `FitzHughNagumoIntegrator` as its Function.  This takes the input the LCMechanism as its
`variable <FitzHughNagumoIntegrator.variable>`, and uses the LCMechanism's `mode <LCMechanism.mode>` attribute as its
XXX parameter.  Its result is assigned as the `value <ControlSignal.value>` of the LCMechanism's `ControlSignal`, which
is used to modulate the Mechanisms specified in its `modulate <LCMechanism.modulate>` attribute.

COMMENT:
If the **monitor_for_control** argument of the LCMechanism's constructor is specified, the following
Components are also automatically created and assigned to the LCMechanism when it is created:

XXX ASSIGN CONTROLLER:  USES THE monitored_values ATTRIBUTE OF ITS CONTROLLER, AS WELL AS ANY SPECIFIED IN monitor_for_control
XXX ASSIGN monitor_for_control:  THESE ARE ADDED TO ITS CONTROLLER'S monitored_values LIST;
                                 IF NO CONTROLLER IS SPECIFIED, ONE IS CREATED

* a `UtilityIntegratorMechanism` -- this monitors the `value <OutputState.value>` of each of the `OutputStates
  <OutputState>` specified in the **monitor_for_control** argument of the LCMechanism's constructor;  these are
  listed in the LCMechanism's `monitored_output_states <LCMechanism.monitored_output_states>` attribute, and the
  `monitored_values <UtilityIntegratorMechanism>` attribute of the UtilityIntegratorMechanism.  They are evaluated by
  the UtilityIntegratorMechanism's `function <UtilityIntegratorMechanism>`;  the result is assigned as the `value
  <OutputState.value>` of the UtilityIntegratorMechanism's , and (by way of a
  `MappingProjection` -- see below) to the LCMechanism's *MODE* `InputState`. This information is used by the
  LCMechanism to set the `value <ControlSignal.value>` for its `ControlSignal`.
..
* a `MappingProjection` that projects from the UtilityIntegratorMechanism's *UTILITY_SIGNAL* `OutputState
  <UtilityIntegratorMechanism_Structure>` to the LCMechanism's *MODE* <InputState_Primary>`.
..
* `MappingProjections <MappingProjection>` from Mechanisms or OutputStates specified in **monitor_for_control** to
  the UtilityIntegratorMechanism's `primary InputState <InputState_Primary>`.
COMMENT

.. _LCMechanism_Output:

Output
~~~~~~

An LCMechanism has a single `ControlSignal` used to modulate the function of the Mechanism(s) listed in its `modulate
<LCMechanism.modulate>` attribute.  The ControlSignal is assigned a `ControlProjection` to the `ParameterState` for
the `multiplicative parameter <Function_Modulatory_Params>` of the `function <Mechanism.function>` for each of those
Mechanisms.

COMMENT:

.. _LCMechanism_Examples:

Examples
~~~~~~~~

EXAMPLES HERE

EXAMPLES HERE OF THE DIFFERENT FORMS OF SPECIFICATION FOR **monitor_for_control** and **modulate**

STRUCTURE:
MODE INPUT_STATE <- NAMED ONE, LAST?
SIGNAL INPUT_STATE(S) <- PRIMARY;  MUST BE FROM PROCESSING MECHANISMS
CONTROL SIGNALS

COMMENT

.. _LCMechanism_Execution:

Execution
---------

Like other `ControlMechanisms <ControlMechanism>`, an LCMechanism executes after all of the `ProcessingMechanisms
<ProcessingMechanism>` in the `Composition` to which it belongs have `executed <Composition_Execution>` in a `TRIAL`.
It's `function <LCMechanism.function>` takes the `value <InputState.value>` of the LCMechanism's `primary InputState
<InputState_Primary>` as its input, and generates a response -- under the influence of its `mode <LCMechanism.mode>`
parameter -- that is assigned as the `value <ControlSignal.value>` of its `ControlSignal`.  The latter is used by its
`ControlProjections <ControlProjection>` to modulate the response -- in the next `TRIAL` of execution --  of the
Mechanisms to which the LCMechanism projects

.. note::
   The `ParameterState` that receives a `ControlProjection` does not update its value until its owner Mechanism
   executes (see `Lazy Evaluation <LINK>` for an explanation of "lazy" updating).  This means that even if a
   LCMechanism has executed, the `multiplicative parameter <Function_Modulatory_Params>` parameter of the `function
   <Mechanism.function>` of a Mechanism that it controls will not assume its new value until that Mechanism has
   executed.

.. _LCMechanism_Class_Reference:

Class Reference
---------------

"""
import numpy as np
import typecheck as tc

from PsyNeuLink.Components.Component import InitStatus
from PsyNeuLink.Components.Functions.Function import ModulationParam, _is_modulation_param, MULTIPLICATIVE_PARAM
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.AdaptiveMechanism import AdaptiveMechanism_Base
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanism.ControlMechanism import ControlMechanism_Base
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism_Base, MonitoredOutputStatesOption
from PsyNeuLink.Components.Projections.Projection import _validate_receiver
from PsyNeuLink.Components.ShellClasses import Mechanism, System
from PsyNeuLink.Components.States.ModulatorySignals.ModulatorySignal import modulatory_signal_keywords
from PsyNeuLink.Components.States.OutputState import OutputState
from PsyNeuLink.Components.States.ParameterState import ParameterState
from PsyNeuLink.Components.States.State import _parse_state_spec
from PsyNeuLink.Globals.Defaults import defaultControlAllocation
from PsyNeuLink.Globals.Keywords import ALL, CONTROLLED_PARAM, CONTROL_PROJECTION, CONTROL_PROJECTIONS, \
                                        CONTROL_SIGNAL, CONTROL_SIGNALS, CONTROL_SIGNAL_SPECS, \
                                        INIT__EXECUTE__METHOD_ONLY, MAKE_DEFAULT_CONTROLLER, MECHANISM, \
                                        MONITOR_FOR_CONTROL, NAME, OWNER, \
                                        PARAMETER_STATE, PARAMS, PROJECTIONS, RECEIVER, REFERENCE_VALUE, SENDER, SYSTEM
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceLevel
from PsyNeuLink.Globals.Utilities import ContentAddressableList
from PsyNeuLink.Scheduling.TimeScale import CentralClock, TimeScale

MODULATE = 'modulate'

ControlMechanismRegistry = {}

class LCMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class LCMechanism(ControlMechanism_Base):
    """
    LCMechanism(                                   \
        monitor_for_control=None,                  \
        mode=0.0,                                  \
        modulate=None,                             \
        params=None,                               \
        name=None,                                 \
        prefs=None)

    Subclass of `ControlMechanism <AdaptiveMechanism>` that modulates the `multiplicative parameter
    <Function_Modulatory_Params>` of the `function <Mechanism.function>` of one or more `Mechanisms <Mechanism>`.

    Arguments
    ---------

COMMENT:
    monitor_for_control : List[OutputState specification] : default None
        specifies set of OutputStates to monitor (see :ref:`LCMechanism_Monitored_OutputStates` for
        specification options).
COMMENT

    mode : float : default 0.0
        specifies the default value for the mode parameter of the LCMechanism's `function <LCMechanism.function>`.

    modulate : List[Mechanism] or *ALL*
        specifies the Mechanisms to be modulated by the LCMechanism.
        If it is a list, every item must be a Mechanism with a `function <Mechanism.function>` that implements a
        `multiplicative parameter <Function_Modulatory_Params>`;  alternatively the keyword *ALL* can be used to
        specify all of the `ProcessingMechanisms <ProcessingMechanism>` in the Composition(s) to which the LCMechanism
        belongs.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters
        for the Mechanism, parameters for its function, and/or a custom function and its parameters. Values
        specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default LCMechanism-<index>
        a string used for the name of the Mechanism.
        If not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the `PreferenceSet` for the Mechanism.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    Attributes
    ----------

COMMENT
    monitoring_mechanism : ObjectiveMechanism
        Mechanism that monitors and evaluates the values specified in the LCMechanism's **monitor_for_control**
        argument, and transmits the result to the LCMechanism's *ERROR_SIGNAL*
        `input_state <Mechanism_Base.input_state>`.

    monitored_output_states : List[OutputState]
        each item is an `OutputState` of a `Mechanism <Mechanism>` specified in the **monitor_for_control** argument of
        the LCMechanism's constructor, the `value <OutputState.value>` \\s of which serve as the items of the
        LCMechanism's `variable <Mechanism_Base.variable>`.
COMMENT:

    mode : float : default 0.0
        determines the value for the mode parameter of the LCMechanism's `FitzHughNagumoIntegrator` function.

    function : `FitzHughNagumoIntegrator`
        takes the LCMechanism's `input <LCMechanism_Input>` and generates its response <LCMechanism_Output>` under
        the influence of its `mode <LCMechanism.mode>` parameter.

    control_signals : List[ControlSignal]
        contains the LCMechanism's single `ControlSignal`, which sends `ControlProjections` to the
        `multiplicative parameter <Function_Modulatory_Params>` of each of the Mechanisms the LCMechanism
        controls (listed in its `modulate <LCMechanism.modulate>` attribute).

    control_projections : List[ControlProjection]
        list of `ControlProjections <ControlProjection>` sent by the LCMechanism's `ControlSignal`, each of which
        projects to the `ParameterState` for the `multiplicative parameter <Function_Modulatory_Params>` of the
        `function <Mechanism.function>` of one of the Mechanisms listed in `modulate <LCMechanism.modulate>` attribute.

    modulate : List[Mechanism]
        list of Mechanisms modulated by the LCMechanism.

    modulation : ModulationParam : default ModulationParam.MULTIPLICATIVE
        the default form of modulation used by the LCMechanism's `ControlProjections`,
        unless they are `individually specified <ControlSignal_Specification>`.
        XXX CORRECT??

    """

    componentType = "LCMechanism"

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
    paramClassDefaults.update({CONTROL_PROJECTIONS: None})

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 monitor_for_control:tc.optional(list)=None,
                 mode:tc.optional(float)=0.0,
                 modulate:tc.optional(tc.any(list,str)) = None,
                 modulation:tc.optional(_is_modulation_param)=ModulationParam.MULTIPLICATIVE,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(monitor_for_control=monitor_for_control,
                                                  mode=mode,
                                                  modulate=modulate,
                                                  # modulation=modulation,
                                                  params=params)

        super().__init__(variable=default_variable,
                         size=size,
                         modulation=modulation,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate SYSTEM, MONITOR_FOR_CONTROL and CONTROL_SIGNALS

        Check that all items in MONITOR_FOR_CONTROL are Mechanisms or OutputStates for Mechanisms in self.system
        Check that every item in `modulate <LCMechanism.modulate>` is a Mechanism and that its function has a
            multiplicative_param
        """

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        # if MONITOR_FOR_CONTROL in target_set:
        #     for spec in target_set[MONITOR_FOR_CONTROL]:
        #         if isinstance(spec, MonitoredOutputStatesOption):
        #             continue
        #         if isinstance(spec, tuple):
        #             spec = spec[0]
        #         if isinstance(spec, (OutputState, Mechanism_Base)):
        #             spec = spec.name
        #         if not isinstance(spec, str):
        #             raise LCMechanismError("Invalid specification in {} arg for {} ({})".
        #                                         format(MONITOR_FOR_CONTROL, self.name, spec))
        #         # If controller has been assigned to a System,
        #         #    check that all the items in monitor_for_control are in the same System
        #         # IMPLEMENTATION NOTE:  If self.system is None, onus is on doing the validation
        #         #                       when the controller is assigned to a System [TBI]
        #         if self.system:
        #             if not any((spec is mech.name or spec in mech.output_states.names)
        #                        for mech in self.system.mechanisms):
        #                 raise LCMechanismError("Specification in {} arg for {} ({}) must be a "
        #                                             "Mechanism or an OutputState of one in {}".
        #                                             format(MONITOR_FOR_CONTROL, self.name, spec, self.system.name))


        if MODULATE in target_set and target_set[MODULATE]:

            from PsyNeuLink.Components.States.ModulatorySignals.ControlSignal import ControlSignal
            from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms import ProcessingMechanism

            spec = target_set[MODULATE]

            if isinstance (spec, str):
                if not spec == ALL:
                    raise LCMechanismError("A string other than the keyword \'ALL\' was specified for the {} argument "
                                           "the constructor for {}".format(MODULATE, self.name))

            if not isinstance(spec, list):
                spec = [spec]

            for mech in spec:
                if not isinstance(mech, Mechanism):
                    raise LCMechanismError("The specification of the {} argument for {} contained an item ({})"
                                           "that is not a Mechanism.".format(MODULATE, self.name, mech))

                if not hasattr(mech.function_object, MULTIPLICATIVE_PARAM):
                    raise LCMechanismError("The specification of the {} argument for {} contained a Mechanism ({})"
                                           "that does not have a {}.".
                                           format(MODULATE, self.name, mech, MULTIPLICATIVE_PARAM))

    def _instantiate_monitored_output_states(self, context=None):
        raise LCMechanismError("{0} (subclass of {1}) must implement _instantiate_monitored_output_states".
                                          format(self.__class__.__name__,
                                                 self.__class__.__bases__[0].__name__))

    def _instantiate_output_states(self, context=None):

        self._instantiate_control_signal(context=context)

        # IMPLEMENTATION NOTE:  Don't want to call this because it instantiates undesired default OutputState
        # super()._instantiate_output_states(context=context)

    def _instantiate_control_signal(self, control_signal=None, context=None):
        """Instantiate ControlSignal OutputState and assign ControlProjections to Mechanisms in self.modulate

        If **modulate** argument of constructor was specified as *ALL*,
            assign all ProcessingMechanisms in Compositions to which LCMechanism belongs to self.modulate
        Instantiate ControlSignal with Projections to the ParameterState for the multiplicative parameter of every
           Mechanism listed in self.modulate

        Returns ControlSignal (OutputState)
        """
        from PsyNeuLink.Components.States.ModulatorySignals.ControlSignal import ControlSignal
        from PsyNeuLink.Components.States.ParameterState import _get_parameter_state
        from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection

       # @@@@@@@@@@@@@@@@@

        # Assign all Processing Mechanisms in the LCMechanism's Composition(s) to its modulate attribute
        if isinstance(spec, str) and spec is ALL:
            for system in self.systems:
                for mech in system.mechanisms:
                    if isinstance(mech, ProcessingMechanism):
                        if hasattr(mech.function, MULTIPLICATIVE_PARAM):
                            ASSIGN
            DO SAME FOR PROCESS, BUT AVOID DUPS
            xxx

        ASSIGN CONTROLPROJECTIONS FOR ALL MECHS LISTED IN self.modulate

       # @@@@@@@@@@@@@@@@@





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

        control_projection = None
        control_signal_params = None

        control_signal_spec = _parse_state_spec(owner=self, state_type=ControlSignal, state_spec=control_signal)

        # Specification is a ParameterState
        if isinstance(control_signal_spec, ParameterState):
            mech = control_signal_spec.owner
            param_name = control_signal_spec.name
            parameter_state = _get_parameter_state(self, CONTROL_SIGNAL, param_name, mech)

        # Specification was tuple or dict, now parsed into a dict
        elif isinstance(control_signal_spec, dict):
            param_name = control_signal_spec[NAME]
            control_signal_params = control_signal_spec[PARAMS]

            # control_signal was a specification dict, with MECHANISM as an entry (and parameter as NAME)
            if control_signal_params and MECHANISM in control_signal_params:
                mech = control_signal_params[MECHANISM]
                # Delete MECHANISM entry as it is not a parameter of ControlSignal
                #     (which will balk at it in ControlSignal._validate_params)
                del control_signal_params[MECHANISM]
                parameter_state = _get_parameter_state(self, CONTROL_SIGNAL, param_name, mech)

            # Specification was originally a tuple, either in parameter specification or control_signal arg;
            #    1st item was either assigned to the NAME entry of the control_signal_spec dict
            #        (if tuple was a (param_name, Mechanism tuple) for control_signal arg;
            #        or used as param value, if it was a parameter specification tuple
            #    2nd item was placed in CONTROL_SIGNAL_PARAMS entry of params dict in control_signal_spec dict,
            #        so parse:
            # IMPLEMENTATION NOTE:
            #    CONTROL_SIGNAL_SPECS is used by _assign_as_controller,
            #                         to pass specification from a parameter specification tuple
            #    PROJECTIONS is used by _parse_state_spec to place the 2nd item of any tuple in params dict;
            #                      here, the tuple comes from a (param, Mechanism) specification in control_signal arg
            #    Delete whichever one it was, as neither is a recognized ControlSignal param
            #        (which will balk at it in ControlSignal._validate_params)
            elif (control_signal_params and
                    any(kw in control_signal_spec[PARAMS] for kw in {CONTROL_SIGNAL_SPECS, PROJECTIONS})):
                if CONTROL_SIGNAL_SPECS in control_signal_spec[PARAMS]:
                    spec = control_signal_params[CONTROL_SIGNAL_SPECS]
                    del control_signal_params[CONTROL_SIGNAL_SPECS]
                elif PROJECTIONS in control_signal_spec[PARAMS]:
                    spec = control_signal_params[PROJECTIONS]
                    del control_signal_params[PROJECTIONS]

                # ControlSignal
                if isinstance(spec, ControlSignal):
                    control_signal_spec = spec

                else:
                    # Mechanism
                    # IMPLEMENTATION NOTE: Mechanism was placed in list in PROJECTIONS entry by _parse_state_spec
                    if isinstance(spec, list) and isinstance(spec[0], Mechanism):
                        mech = spec[0]
                        parameter_state = _get_parameter_state(self, CONTROL_SIGNAL, param_name, mech)

                    # Projection (in a list)
                    elif isinstance(spec, list):
                        control_projection = spec[0]
                        if not isinstance(control_projection, ControlProjection):
                            raise LCMechanismError("PROGRAM ERROR: list in {} entry of params dict for {} of {} "
                                                        "must contain a single ControlProjection".
                                                        format(CONTROL_SIGNAL_SPECS, CONTROL_SIGNAL, self.name))
                        if len(spec)>1:
                            raise LCMechanismError("PROGRAM ERROR: Multiple ControlProjections are not "
                                                        "currently supported in specification of a ControlSignal")
                        # Get receiver mech
                        if control_projection.init_status is InitStatus.DEFERRED_INITIALIZATION:
                            parameter_state = control_projection.init_args[RECEIVER]
                            # ControlProjection was created in response to specification of ControlSignal
                            #     (in a 2-item tuple where the parameter was specified),
                            #     so get ControlSignal spec
                            if SENDER in control_projection.init_args:
                                control_signal_spec = control_projection.init_args[SENDER]
                                if control_signal_spec and not isinstance(control_signal_spec, ControlSignal):
                                    raise LCMechanismError("PROGRAM ERROR: "
                                                                "Sender of {} for {} {} of {} is not a {}".
                                                                format(CONTROL_PROJECTION,
                                                                       parameter_state.name,
                                                                       PARAMETER_STATE,
                                                                       parameter_state.owner.name,
                                                                       CONTROL_SIGNAL))
                        else:
                            parameter_state = control_projection.receiver
                        param_name = parameter_state.name

                    else:
                        raise LCMechanismError("PROGRAM ERROR: failure to parse specification of {} for {}".
                                                    format(CONTROL_SIGNAL, self.name))
            else:
                raise LCMechanismError("PROGRAM ERROR: No entry found in params dict with specification of "
                                            "parameter Mechanism or ControlProjection for {} of {}".
                                            format(CONTROL_SIGNAL, self.name))


        default_name = param_name + '_' + ControlSignal.__name__

        # Get constraint for ControlSignal value
        #    - get LCMechanism's value
        self._update_value(context=context)
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
                raise LCMechanismError("Attempt to assign ControlSignal to {} ({}) that is already owned by {}".
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
                    raise LCMechanismError("PROGRAM ERROR: list of ControlProjections is not currently supported "
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
        if control_projection:
            _validate_receiver(self, control_projection, Mechanism, CONTROL_SIGNAL, context=context)

            from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
            if not isinstance(control_projection, ControlProjection):
                raise LCMechanismError("PROGRAM ERROR: Attempt to assign {}, "
                                                  "that is not a ControlProjection, to ControlSignal of {}".
                                                  format(control_projection, self.name))
            if control_projection.init_status is InitStatus.DEFERRED_INITIALIZATION:
                control_projection.init_args['sender']=control_signal
                if control_projection.init_args['name'] is None:
                    # FIX 5/23/17: CLEAN UP NAME STUFF BELOW:
                    control_projection.init_args['name'] = CONTROL_PROJECTION + \
                                                   ' for ' + parameter_state.owner.name + ' ' + parameter_state.name
                control_projection._deferred_init()
            else:
                control_projection.sender = control_signal

        # Instantiate ControlProjection
        else:
            # IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
            from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
            control_projection = ControlProjection(sender=control_signal,
                                                   receiver=parameter_state,
                                                   name=CONTROL_PROJECTION + control_signal_name)

        # Add ControlProjection to list of OutputState's outgoing Projections
        # (note: if it was deferred, it just added itself, skip)
        if not control_projection in control_signal.efferents:
            control_signal.efferents.append(control_projection)

        # Add ControlProjection to LCMechanism's list of ControlProjections
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
            self.output_states[control_signal.name] = control_signal
        except (AttributeError, TypeError):
            from PsyNeuLink.Components.States.State import State_Base
            self.output_states = ContentAddressableList(component_type=State_Base,
                                                        list=[control_signal],
                                                        name = self.name+'.output_states')

        # Add index assignment to OutputState
        control_signal.index = output_state_index

        # (Re-)assign control_signals attribute to output_states
        self.control_signals = self.output_states

        return control_signal

    def _instantiate_attributes_after_function(self, context=None):
        """Implment ControlSignals specified in control_signals arg or "locally" in parameter specification(s)

        Calls super's instantiate_attributes_after_function, which calls _instantiate_output_states;
            that insures that any ControlSignals specified in control_signals arg are instantiated first
        Then calls _assign_as_controller to instantiate any ControlProjections/ControlSignals specified
            along with parameter specification(s) (i.e., as part of a (<param value>, ControlProjection) tuple
        """

        super()._instantiate_attributes_after_function(context=context)

    def _execute(self,
                    variable=None,
                    runtime_params=None,
                    clock=CentralClock,
                    time_scale=TimeScale.TRIAL,
                    context=None):
        """Updates ControlSignals based on inputs

        Must be overriden by subclass
        """
        raise LCMechanismError("{0} must implement execute() method".format(self.__class__.__name__))

    def show(self):

        print ("\n---------------------------------------------------------")

        print ("\n{0}".format(self.name))
        print("\n\tMonitoring the following Mechanism OutputStates:")
        for state in self.monitoring_mechanism.input_states:
            for projection in state.path_afferents:
                monitored_state = projection.sender
                monitored_state_mech = projection.sender.owner
                monitored_state_index = self.monitored_output_states.index(monitored_state)

                weight = self.monitor_for_control_weights_and_exponents[monitored_state_index][0]
                exponent = self.monitor_for_control_weights_and_exponents[monitored_state_index][1]

                print ("\t\t{0}: {1} (exp: {2}; wt: {3})".
                       format(monitored_state_mech.name, monitored_state.name, weight, exponent))

        print ("\n\tControlling the following Mechanism parameters:".format(self.name))
        # Sort for consistency of output:
        state_names_sorted = sorted(self.output_states.names)
        for state_name in state_names_sorted:
            for projection in self.output_states[state_name].efferents:
                print ("\t\t{0}: {1}".format(projection.receiver.owner.name, projection.receiver.name))

        print ("\n---------------------------------------------------------")
