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

A ControlMechanism is an `AdaptiveMechanism` that modifies the parameter(s) of one or more
`Components <Component>`. Its `function <ControlMechanism_Base.function>` takes an evaluative signal (usually the
output of an `ObjectiveMechanism`) and uses that to calculate an
`allocation_policy <ControlMechanism_Base.allocation_policy>`:  a list of `allocation <ControlSignal.allocation>`
values for each of its `ControlSignals <ControlSignal>`.  This is used by each ControlSignal to calculate its
`intensity`, which is then conveyed by the ControlSignal's `ControlProjection(s) <ControlProjection>` to the
`ParameterState(s) <ParameterState>` to which they project.  Each ParameterState then uses the value received by a
ControlProjection to modify the value of the parameter for which it is responsible (see `ModulatorySignal_Modulation`
for a more detailed description of how modulation operates).  A ControlMechanism can regulate only the parameters of
Components in the System for which it is the `controller <System_Execution_Control>`.  The control Components of a
system can be displayed using the System's `show_graph` method with its **show_control** argument assigned as `True`.
COMMENT: TBI
The control components of a system can be displayed using the system's 
`show_graph` method with its **show_control** argument assigned as :keyword:``True`.  
COMMENT

The control components of a System are executed after all `ProcessingMechanisms <ProcessingMechanism>` and
`learning components <LearningMechanism>` in that system have been executed (see `System Execution <System_Execution>`).

.. _ControlMechanism_Creation:

Creating A ControlMechanism
---------------------------

A ControlMechanism can be created using the standard Python method of calling the constructor for the desired type.
A ControlMechanism is also created automatically whenever a `System is created <System_Creation>`, and it is assigned as
the `controller <System_Execution_Control>` for that System. The values to be monitored by the ControlMechanism are
specified in the **monitor_for_control** argument of its constructor, and the parameters it controls are specified in
the **control_signals** argument.  When the ControlMechanism is created, it automatically creates
an ObjectiveMechanism (used to monitor and evaluate the values specified in **monitor_for_control**) 
as well as `ControlSignals <ControlSignal>` and `ControlProjections <ControlProjection>` used to control the parameters
specified in **control_signals**, as described below. The kind of ObjectiveMechanism created by a ControlMechanism,
and how it evaluates the values it monitors, depends on the :ref:`subclass <LINK>` of ControlMechanism.

.. _ControlMechanism_Control_Signals:

Specifying Parameters to Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ControlMechanisms are used to control the parameter values of other `Components <Component>` (including
`Functions <Function>`).  A parameter can be specified for control by assigning it a `ControlProjection`
or `ControlSignal` (along with the parameter's value) wherever a parameter can be specified (see
`ParameterState_Specification`).  The parameters to be controlled by a ControlMechanism can also be specified in the
**control_signals**  argument of the constructor for a ControlMechanism.  The **control_signals** argument must be a
list, each item of which can use any of the forms used for `specifying a ControlSignal <ControlSignal_Specification>`.

A `ControlSignal` is created for each item listed in the **control_signals** argument of its constructor, and all of
the ControlSignals for a ControlMechanism are listed in its `control_signals <ControlMechanism_Base.control_signals>`
attribute.  Each ControlSignal is assigned a `ControlProjection` to the `ParameterState` associated with each parameter
it controls. ControlSignals are a type of `OutputState`, and so they are also listed in the ControlMechanism's
`output_states <Mechanism_Base.output_states>` attribute.

.. _ControlMechanism_Monitored_Values:

Specifying Values to Monitor for Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a ControlMechanism is created, it automatically creates an `ObjectiveMechanism` that is used to monitor and
evaluate the values specified in the **monitor_for_control** argument of the ControlMechanism's constructor. 
The **monitor_for_control** argument must be a list, each item of which must refer to a `Mechanism` or the `OutputState`
of one.  These are assigned to the ObjectiveMechanism's `monitored_values <ObjectiveMechanism>` attribute, and the
ObjectiveMechanism is referenced by the ControlMechanism's
`monitoring_mechanism <ControlMechanism_Base.monitoring_mechanism>` attribute. The ObjectiveMechanism monitors each
Mechanism and/or OutputState listed in the ControlMechanism's
`monitor_for_control <ControlMechanism_Base.monitor_for_control>` attribute, and evaluates them using the its
`function <ControlMechanism_Base.function>`. This information is used to set the `allocation <ControlSignal.allocation>`
 for each of the ControlMechanism's ControlSignals.

COMMENT:

.. _ControlMechanism_Examples:

Examples
~~~~~~~~

EXAMPLES HERE

EXAMPLES HERE OF THE DIFFERENT FORMS OF SPECIFICATION FOR **monitor_for_control** and **control_signals**
COMMENT


.. _ControlMechanism_Execution:

Execution
---------

A ControlMechanism that is a System's `controller` is always the last Mechanism to be executed in a `TRIAL` for that
System (see `System Control <System_Execution_Control>` and `Execution <System_Execution>`).  The ControlMechanism's
`function <ControlMechanism_Base.function>` takes as its input the `value <InputState.value>` of its *ERROR_SIGNAL*
`input_state <Mechanism.input_state>`, and uses that to determine its
`allocation_policy <ControlMechanism_Base.allocation_policy>` which specifies the value assigned to the
`allocation <ControlSignal.allocation>` of each of its `ControlSignals <ControlSignal>`.  Each
ControlSignal uses that value to calculate its `intensity <ControlSignal.intensity>`, which is used by its
`ControlProjection(s) <ControlProjection>` to modulate the value of the ParameterState(s) for the parameter(s)
it controls, which are then used in the subsequent `TRIAL` of execution.

.. note::
   A `ParameterState` that receives a `ControlProjection` does not update its value until its owner Mechanism
   executes (see `Lazy Evaluation <LINK>` for an explanation of "lazy" updating).  This means that even if a
   ControlMechanism has executed, a parameter that it controls will not assume its new value until the Mechanism
   to which it belongs has executed.

.. _ControlMechanism_Class_Reference:

Class Reference
---------------

"""

from PsyNeuLink.Components.Functions.Function import ModulationParam, _is_modulation_param
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism_Base, MonitoredOutputStatesOption
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.AdaptiveMechanism import AdaptiveMechanism_Base
from PsyNeuLink.Components.Projections.Projection import _validate_receiver
from PsyNeuLink.Components.ShellClasses import *
from PsyNeuLink.Components.States.OutputState import OutputState
from PsyNeuLink.Components.States.ParameterState import ParameterState
from PsyNeuLink.Components.States.State import _parse_state_spec
from PsyNeuLink.Components.States.ModulatorySignals.ModulatorySignal import *

ControlMechanismRegistry = {}

class ControlMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


# class ControlMechanism_Base(Mechanism_Base):
class ControlMechanism_Base(AdaptiveMechanism_Base):
    """
    ControlMechanism_Base(                         \
        monitor_for_control=None,                  \
        control_signals=None,                      \
        modulation=ModulationParam.MULTIPLICATIVE  \
        function=Linear,                           \
        params=None,                               \
        name=None,                                 \
        prefs=None)

    Abstract class for ControlMechanism.

    .. note::
       ControlMechanisms should NEVER be instantiated by a direct call to the base class.
       They should be instantiated using the constructor for a :doc:`subclass <ControlMechanism>`.

    COMMENT:
        Description:
            Protocol for instantiating unassigned ControlProjections (i.e., w/o a sender specified):
               If sender is not specified for a ControlProjection (e.g., in a parameter specification tuple) 
                   it is flagged for deferred_init() in its __init__ method
               When the next ControlMechanism is instantiated, if its params[MAKE_DEFAULT_CONTROLLER] == True
                   its _assign_as_controller method is called in _instantiate_attributes_after_function;
                   it then iterates through all of the ParameterStates of all of the Mechanisms in its System,
                   identifies ones without a sender specified, calls its deferred_init() method,
                   instantiates a ControlSignal for it, and assigns it as the ControlProjection's sender.

            MONITOR_FOR_CONTROL param determines which States will be monitored.
                specifies the OutputStates of the terminal Mechanisms in the System to be monitored by ControlMechanism
                this specification overrides any in System.params[], but can be overridden by Mechanism.params[]
                ?? if MonitoredOutputStates appears alone, it will be used to determine how States are assigned from
                    System.executionGraph by default
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
                + MONITOR_FOR_CONTROL: List[]
    COMMENT

    Arguments
    ---------

    monitor_for_control : List[OutputState specification] : default None
        specifies set of OutputStates to monitor (see :ref:`ControlMechanism_Monitored_OutputStates` for
        specification options).

    control_signals : List[parameter of Mechanism or its function, \
                      ParameterState, Mechanism tuple[str, Mechanism] or dict]
        specifies the parameters to be controlled by the ControlMechanism 
        (see `control_signals <ControlMechanism_Base.control_signals>` for details).

    modulation : ModulationParam : ModulationParam.MULTIPLICATIVE
        specifies the default form of modulation used by the ControlMechanism's `ControlSignals <ControlSignal>`,
        unless they are `individually specified <ControlSignal_Specification>`.

    function : TransferFunction : default Linear(slope=1, intercept=0)
        specifies function used to combine values of monitored OutputStates.
        
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

    monitoring_mechanism : ObjectiveMechanism
        Mechanism that monitors and evaluates the values specified in the ControlMechanism's **monitor_for_control**
        argument, and transmits the result to the ControlMechanism's *ERROR_SIGNAL* 
        `input_state <Mechanism.input_state>`.

    monitored_output_states : List[OutputState]
        each item is an `OutputState` of a `Mechanism` specified in the **monitor_for_control** argument of the
        ControlMechanism's constructor, the `value <OutputState.value>` \s of which serve as the items of the
        ControlMechanism's `variable <Mechanism.variable>`.

    control_signals : List[ControlSignal]
        list of `ControlSignals <ControlSignals>` for the ControlMechanism, each of which sends a `ControlProjection`
        to the `ParameterState` for the parameter it controls (same as ControlMechanism's
        `output_states <Mechanism.output_states>` attribute).

    control_projections : List[ControlProjection]
        list of `ControlProjections <ControlProjection>`, one for each `ControlSignal` in `control_signals`.

    function : TransferFunction : default Linear(slope=1, intercept=0)
        determines how the `value <OuputState.value>` \s of the `OutputStates <OutputState>` specified in the
        **monitor_for_control** argument of the ControlMechanism's constructor are used to generate its
        `allocation_policy <ControlMechanism_Base.allocation_policy>`.

    allocation_policy : 2d np.array
        each item is the value assigned as the `allocation <ControlSignal.allocation>` for the corresponding
        ControlSignal listed in the `control_signals` attribute;  the allocation_policy is the same as the
        ControlMechanism's `value <Mechanism.value>` attribute).
        
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

    # This must be a list, as there may be more than one (e.g., one per control_signal)
    variableClassDefault = defaultControlAllocation

    from PsyNeuLink.Components.Functions.Function import Linear
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({CONTROL_PROJECTIONS: None})

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 system=None,
                 monitor_for_control:tc.optional(list)=None,
                 function = Linear(slope=1, intercept=0),
                 control_signals:tc.optional(list) = None,
                 modulation:tc.optional(_is_modulation_param)=ModulationParam.MULTIPLICATIVE,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(monitor_for_control=monitor_for_control,
                                                  function=function,
                                                  control_signals=control_signals,
                                                  # modulation=modulation,
                                                  params=params)

        super(ControlMechanism_Base, self).__init__(variable=default_variable,
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

        super(ControlMechanism_Base, self)._validate_params(request_set=request_set,
                                                                 target_set=target_set,
                                                                 context=context)
        if SYSTEM in target_set:
            if not isinstance(target_set[SYSTEM], System):
                raise KeyError
            else:
                self.paramClassDefaults[SYSTEM] = request_set[SYSTEM]

        if MONITOR_FOR_CONTROL in target_set:
            for spec in target_set[MONITOR_FOR_CONTROL]:
                if isinstance(spec, MonitoredOutputStatesOption):
                    continue
                if isinstance(spec, tuple):
                    spec = spec[0]
                if isinstance(spec, (OutputState, Mechanism_Base)):
                    spec = spec.name
                if not isinstance(spec, str):
                    raise ControlMechanismError("Invalid specification in {} arg for {} ({})".
                                                format(MONITOR_FOR_CONTROL, self.name, spec))
                # If controller has been assigned to a System,
                #    check that all the items in monitor_for_control are in the same System
                # IMPLEMENTATION NOTE:  If self.system is None, onus is on doing the validation
                #                       when the controller is assigned to a System [TBI]
                if self.system:
                    if not any((spec is mech.name or spec in mech.output_states.names)
                               for mech in self.system.mechanisms):
                        raise ControlMechanismError("Specification in {} arg for {} ({}) must be a "
                                                    "Mechanism or an OutputState of one in {}".
                                                    format(MONITOR_FOR_CONTROL, self.name, spec, self.system.name))

        # FIX: REPLACE WITH CALL TO _parse_state_spec WITH APPROPRIATE PARAMETERS
        if CONTROL_SIGNALS in target_set and target_set[CONTROL_SIGNALS]:

            from PsyNeuLink.Components.States.ModulatorySignals.ControlSignal import ControlSignal

            for spec in target_set[CONTROL_SIGNALS]:

                # Specification is for a ControlSignal
                if isinstance(spec, ControlSignal):
                    # If controller has been assigned to a System,
                    #    check that any ControlProjections the ControlSignal has
                    #    are to Mechanisms in the controller's System
                    # IMPLEMENTATION NOTE:  If self.system is None, onus is on doing the validation
                    #                       when the controller is assigned to a System [TBI]
                    if self.system:
                        if not all(control_proj.receiver.owner in self.system.mechanisms
                                   for control_proj in spec.efferents):
                            raise ControlMechanismError("The {} specified in the {} arg for {} ({}) "
                                                        "has one or more ControlProjections to a Mechanism "
                                                        "that is not in {}".
                                                        format(CONTROL_SIGNAL,
                                                               CONTROL_SIGNALS,
                                                               self.name,
                                                               spec.name,
                                                               self.system.name))
                    continue

                # Specification is for a ParameterState
                elif isinstance(spec, ParameterState):
                    param_name = spec.name
                    mech = spec.owner
                    #  Check that owner is in controller's System
                    if not self.system in mech.systems:
                        raise ControlMechanismError("The {} specified in the {} arg for {} ({}) "
                                                    "belongs to a Mechanism ({}) that is not in "
                                                    "the System for which {} is a controller ({})".
                                                    format(PARAMETER_STATE,
                                                           CONTROL_SIGNALS,
                                                           self.name,
                                                           spec.name,
                                                           mech.name,
                                                           self.name,
                                                           self.system.name))

                # Specification is for a tuple (str, Mechanism):
                elif isinstance(spec, tuple):
                    param_name = spec[0]
                    mech = spec[1]
                    # Check that 1st item is a str (presumably the name of the Mechanism's attribute for the param)
                    if not isinstance(param_name, str):
                        raise ControlMechanismError("1st item of tuple in specification of {} for {} ({}) "
                                                    "must be a string".format(CONTROL_SIGNAL, self.name, param_name))
                    # Check that 2nd item is a Mechanism
                    if not isinstance(mech, Mechanism):
                        raise ControlMechanismError("2nd item of tuple in specification of {} for {} ({}) "
                                                    "must be a Mechanism".format(CONTROL_SIGNAL, self.name, mech))


                # ControlSignal specification dictionary, must have the following entries:
                #    NAME:str - must be the name of an attribute of MECHANISM
                #    MECHANISM:Mechanism - must have an attribute and corresponding ParameterState with PARAMETER
                #    PARAMS:dict - entries must be valid ControlSignal parameters (e.g., ALLOCATION_SAMPLES)
                elif isinstance(spec, dict):
                    if not NAME in spec:
                        raise ControlMechanismError("Specification dict for {} of {} must have a NAME entry".
                                                    format(CONTROL_SIGNAL, self.name))
                    param_name = spec[NAME]
                    if not MECHANISM in spec:
                        raise ControlMechanismError("Specification dict for {} of {} must have a MECHANISM entry".
                                                    format(CONTROL_SIGNAL, self.name))
                    mech = spec[MECHANISM]
                    # Check that all of the other entries in the specification dictionary are valid ControlSignal params
                    for param in spec:
                        if param in {NAME, MECHANISM} | modulatory_signal_keywords:
                            continue
                        if not hasattr(mech, param):
                            raise ControlMechanismError("\'{}\' entry in specification dictionary for {} arg of {} "
                                                       "is not a valid {} specification".
                                                       format(CONTROL_SIGNAL, param, self.name,
                                                              ControlSignal.__name__))
                else:
                    raise ControlMechanismError("PROGRAM ERROR: unrecognized specification of the {} arg for {} ({})".
                                                format(CONTROL_SIGNALS, self.name, spec))
                    # raise ControlMechanismError("Specification of {} for {} ({}) must be a ParameterState, Mechanism, "
                    #                             "a tuple specifying a parameter and Mechanism, "
                    #                             "a ControlSignal specification dictionary, "
                    #                             "or an existing ControlSignal".
                    #                             format(CONTROL_SIGNAL, self.name, spec))

                # Check that param_name is the name of an attribute of the Mechanism
                if not hasattr(mech, param_name) and not hasattr(mech.function_object, param_name):
                    raise ControlMechanismError("{} (in specification of {} for {}) is not an "
                                                "attribute of {} or its function"
                                                .format(param_name, CONTROL_SIGNAL, self.name, mech))
                # Check that the Mechanism has a ParameterState for the param
                if not param_name in mech._parameter_states.names:
                    raise ControlMechanismError("There is no ParameterState for the parameter ({}) of {} "
                                                "specified in {} for {}".
                                                format(param_name, mech.name, CONTROL_SIGNAL, self.name))
                # If self has been assigned to a System,
                #    check that the Mechanism to which the parameter belongs is in the controller's System
                # IMPLEMENTATION NOTE:  If self.system is None, onus is on doing the validation
                #                       when the controller is assigned to a System [TBI]
                if self.system and not mech in self.system.mechanisms:
                    raise ControlMechanismError("Specification in {} arg for {} ({} param of {}) "
                                                "must be for a Mechanism in {}".
                                                format(CONTROL_SIGNALS,
                                                       self.name,
                                                       param_name,
                                                       mech.name,
                                                       self.system.name))

    def _instantiate_monitored_output_states(self, context=None):
        raise ControlMechanismError("{0} (subclass of {1}) must implement _instantiate_monitored_output_states".
                                          format(self.__class__.__name__,
                                                 self.__class__.__bases__[0].__name__))

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
    #                           (with control_signals taking the place of monitored_values)
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
            - params dict, from _assign_as_controller(), containing a ControlProjection;
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
        from PsyNeuLink.Components.States.ParameterState import _get_parameter_state
        from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection

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
                            raise ControlMechanismError("PROGRAM ERROR: list in {} entry of params dict for {} of {} "
                                                        "must contain a single ControlProjection".
                                                        format(CONTROL_SIGNAL_SPECS, CONTROL_SIGNAL, self.name))
                        if len(spec)>1:
                            raise ControlMechanismError("PROGRAM ERROR: Multiple ControlProjections are not "
                                                        "currently supported in specification of a ControlSignal")
                        # Get receiver mech
                        if control_projection.value is DEFERRED_INITIALIZATION:
                            parameter_state = control_projection.init_args[RECEIVER]
                            # ControlProjection was created in response to specification of ControlSignal
                            #     (in a 2-item tuple where the parameter was specified),
                            #     so get ControlSignal spec
                            if SENDER in control_projection.init_args:
                                control_signal_spec = control_projection.init_args[SENDER]
                                if control_signal_spec and not isinstance(control_signal_spec, ControlSignal):
                                    raise ControlMechanismError("PROGRAM ERROR: "
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
                        raise ControlMechanismError("PROGRAM ERROR: failure to parse specification of {} for {}".
                                                    format(CONTROL_SIGNAL, self.name))
            else:
                raise ControlMechanismError("PROGRAM ERROR: No entry found in params dict with specification of "
                                            "parameter Mechanism or ControlProjection for {} of {}".
                                            format(CONTROL_SIGNAL, self.name))


        default_name = param_name + '_' + ControlSignal.__name__

        # Get constraint for ControlSignal value
        #    - get ControlMechanism's value
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
            if control_signal_spec.value is DEFERRED_INITIALIZATION:
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
        if control_projection:
            _validate_receiver(self, control_projection, Mechanism, CONTROL_SIGNAL, context=context)

            from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
            if not isinstance(control_projection, ControlProjection):
                raise ControlMechanismError("PROGRAM ERROR: Attempt to assign {}, "
                                                  "that is not a ControlProjection, to ControlSignal of {}".
                                                  format(control_projection, self.name))
            if control_projection.value is DEFERRED_INITIALIZATION:
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

        if MAKE_DEFAULT_CONTROLLER in self.paramsCurrent:
            if self.paramsCurrent[MAKE_DEFAULT_CONTROLLER]:
                self._assign_as_controller(context=context)
            if not self.system.enable_controller:
                return

    def _assign_as_controller(self, context=None):

        # Check the ParameterStates of the System's Mechanisms for any ControlProjections with deferred_init()
        # Note: this includes any ControlProjections created where a ControlSignal rather than a ControlProjection
        #       was used to specify control for a parameter (e.g., in a 2-item tuple specification for the parameter);
        #       the initialization of the ControlProjection and, if specified, the ControlSignal
        #       are completed in the call to _instantiate_control_signal() below
        for mech in self.system.mechanisms:
            for parameter_state in mech._parameter_states:
                for projection in parameter_state.mod_afferents:
                    # If Projection was deferred for init, instantiate its ControlSignal and then initialize it
                    if projection.value is DEFERRED_INITIALIZATION:
                        control_signal_specs = projection.control_signal_params or {}
                        control_signal_specs.update({CONTROL_SIGNAL_SPECS: [projection]})
                        self._instantiate_control_signal(control_signal_specs, context=context)

    def _execute(self,
                    variable=None,
                    runtime_params=None,
                    clock=CentralClock,
                    time_scale=TimeScale.TRIAL,
                    context=None):
        """Updates ControlSignals based on inputs

        Must be overriden by subclass
        """
        raise ControlMechanismError("{0} must implement execute() method".format(self.__class__.__name__))

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