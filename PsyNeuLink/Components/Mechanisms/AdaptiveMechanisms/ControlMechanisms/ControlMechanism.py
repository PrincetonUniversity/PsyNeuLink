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

A ControlMechanism is an `AdaptiveMechanism` that modifies the parameter(s) of one or more `ProcessingMechanisms`.
It's function takes a value (usually the output of an `ObjectiveMechanism`) and uses that to calculate an
`allocation_policy`:  a list of `allocation` values for each of its ControlSignals that specify the value to assign
to each parameter of a ProcessingMechanism (or its function) that it controls.  Each of these values is assigned as
the value of a corresponding `ControlSignal` (a subclass of `OutputState` used by ControlMechanisms), and conveyed by
the associated `ControlProjection` to the `parameterState <ParameterState>` of the relevant ProcessingMechanism.
A ControlMechanism can regulate only the parameters of mechanisms in the system for which it is the
`controller <System_Execution_Control>`.  The control components of a system can be displayed using the system's 
`show_graph` method with its **show_control** argument assigned :keyword:``True`.  
COMMENT: TBI
The control components of a system can be displayed using the system's 
`show_graph` method with its **show_control** argument assigned as :keyword:``True`.  
COMMENT

The control components of a 
system are executed after all ProcessingMechanisms and `learning components <LearningMechanism>` in that system have 
been executed.

.. _ControlMechanism_Creation:

Creating A ControlMechanism
---------------------------

ControlMechanisms can be created using the standard Python method of calling the constructor for the desired type.
A ControlMechanism is also created automatically whenever a `system is created <System_Creation>`, and it is assigned as
the `controller <System_Execution_Control>` for that system. The values to be monitored by the ControlMechanism are  
specified in the **monitor_for_control** argument of its constructor, and the parameters it controls are specified in
the **control_signals** argument.  When the ControlMechanism is created, it automatically creates
an ObjectiveMechanism (used to monitor and evaluate the values specified in **monitor_for_control**) 
as well as `ControlSignals <ControlSignal>` and `ControlProjections <ControlProjection>` used to control the parameters
specified in **control_signals**, as described below. The kind of ObjectiveMechanism created by a ControlMechanism,
and how it evalutes the values it monitors, depends on the `subclass <ControlMechanism>` of ControlMechanism.

.. _ControlMechanism_Control_Signals:

Specifying Parameters to Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ControlMechanisms are used to control the parameter values of mechanisms and/or their functions.  A parameter can be 
specified for control by assigning it a `ControlProjection` (along with the parameter's value) when creating the 
mechanism or function to which the parameter belongs (see `Mechanism_Parameters`).  The parameters to be controlled by
a ControlMechanism can also be specified in the **control_signals**  argument of the constructor for a ControlMechanism.  
The **control_signals** argument must be a list, each item of which must refer to a parameter to be controlled specified 
in any of the following ways:

  * *ParameterState* of the Mechanism to which the parameter belongs;
  |
  * *tuple*, with the *name* of the parameter as its 1st item. and the *mechanism* to which it belongs as the 2nd;
    note that this is a convenience format, which is simpler to use than a specification dictionary (see below), 
    but precludes specification of any `parameters <ControlSignal_Structure>` for the ControlSignal.
  |
  * *specification dictionary*, that must contain at least the following two entries:
    * *NAME*:str - a string that is the name of the parameter to be controlled;
    * *MECHANISM*:Mechanism - the Mechanism to which the parameter belongs; 
      (note: the Mechanism itself should be specified even if the parameter belongs to its function).
    The dictionary can also contain entries for any other ControlSignal parameters to be specified
    (e.g., *MODULATION_OPERATION*:ModulationOperation to specify how the parameter will be modulated;
    see `below <ControlSignal_Structure>` for a list of parameters).

A `ControlSignal` is created for each item listed in **control_signals**, and all of the ControlSignals for a 
ControlMechanism are listed in its `control_signals <ControlMechanism.control_signals>` attribute.  Each ControlSignal
is assigned a `ControlProjection` to the parameterState of the mechanisms associated with the specified parameter of the
mechanism or its function, that is used to control the parameter's value. ControlSignals are a type of `OutputState`, 
and so they are also listed in the ControlMechanism's `output_states <ControlMechanism.outut_states>` attribute.

.. _ControlMechanism_Monitored_Values:

Specifying Values to Monitor for Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When an ControlMechanism is created, it automatically creates an `ObjectiveMechanism` that is used to monitor and 
evaluate the values specified in the **monitor_for_control** argument of the ControlMechanism's constructor. 
The **monitor_for_control** argument must be a list, each item of which must refer to a mechanism or the outputState
of one.  These are assigned to the ObjectiveMechanism's `monitored_values <ObjectiveMechanism>` attribute, and the
ObjectiveMechanism is referenced by the ControlMechanism's 
`monitoring_mechanism <ControlMechanism.monitoring_mechanism>` attribute.

ObjectiveMechanism monitors each mechanism and/or outputState listed in the ControlMechanism's
'monitor_for_control <ControlMechanism.monitor_for_control>` attribute, and evaluates them using the its `function`.
This information is used to set the value of the ControlMechanism's ControlSignals.

.. _ControlMechanism_Examples:

Examples
~~~~~~~~

EXAMPLES HERE

COMMENT:
EXAMPLES HERE OF THE DIFFERENT FORMS OF SPECIFICATION FOR **monitor_for_control** and **control_signals**
COMMENT


.. _ControlMechanism_Execution:

Execution
---------

A ControlMechanism that is a system's `controller` is always the last mechanism to be executed (see `System Control
<System_Execution_Control>`).  Its `function <ControlMechanism.function>` takes as its input the value in its 
*ERROR_SIGNAL* `input_state <ControlMechanism.input_state>`, and use that to determine its 
`allocation_policy <ControlMechanism.allocation_policy>` that specifies the value assigned to each of its   
`ControlSignals <ControlSignal>`.  Each of those is used by its associated `ControlProjection` to set the
value of the ParameterState for the parameter it controls.  In the subsequent round of execution, 
those parameter values are used by the mechanism when it executes.

.. note::
   A `ParameterState` that receives a `ControlProjection` does not update its value until its owner mechanism
   executes (see `Lazy Evaluation <LINK>` for an explanation of "lazy" updating).  This means that even if a
   ControlMechanism has executed, a parameter that it controls will not assume its new value until the mechanism
   to which it belongs has executed.

.. _ControlMechanism_Class_Reference:

Class Reference
---------------

"""


from PsyNeuLink.Components.ShellClasses import *
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism_Base, MonitoredOutputStatesOption
from PsyNeuLink.Components.States.ParameterState import ParameterState
from PsyNeuLink.Components.States.State import _parse_state_spec
from PsyNeuLink.Components.States.OutputState import OutputState

ControlMechanismRegistry = {}

class ControlMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class ControlMechanism_Base(Mechanism_Base):
    """
    ControlMechanism_Base(     \
    monitor_for_control=None,  \
    control_signals=None,      \
    function=Linear,           \
    params=None,               \
    name=None,                 \
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
                   its _take_over_as_default_controller method is called in _instantiate_attributes_after_function;
                   it then iterates through all of the parameterStates of all of the mechanisms in its system, 
                   identifies ones without a sender specified, calls its deferred_init() method,
                   instantiates a ControlSignal for it, and assigns it as the ControlProjection's sender.

            MONITOR_FOR_CONTROL param determines which states will be monitored.
                specifies the outputStates of the terminal mechanisms in the System to be monitored by ControlMechanism
                this specification overrides any in System.params[], but can be overridden by Mechanism.params[]
                ?? if MonitoredOutputStates appears alone, it will be used to determine how states are assigned from
                    system.executionGraph by default
                if MonitoredOutputStatesOption is used, it applies to any mechanisms specified in the list for which
                    no outputStates are listed; it is overridden for any mechanism for which outputStates are
                    explicitly listed
                TBI: if it appears in a tuple with a Mechanism, or in the Mechamism's params list, it applies to
                    just that mechanism

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
        specifies set of outputStates to monitor (see :ref:`ControlMechanism_Monitored_OutputStates` for
        specification options).

    control_signals : List[parameter of Mechanism or its function, ParameterState, tuple[str, Mechanism] or dict]
        specifies the parameters to be controlled by the ControlMechanism 
        (see `control_signals <ControlMechanism.control_signals>` for details).

    function : TransferFunction : default Linear(slope=1, intercept=0)
        specifies function used to combine values of monitored output states.
        
    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that can be used to specify the parameters
        for the mechanism, parameters for its function, and/or a custom function and its parameters. Values
        specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default ControlMechanism-<index>
        a string used for the name of the mechanism.
        If not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the `PreferenceSet` for the mechanism.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    Attributes
    ----------

    monitoring_mechanism : ObjectiveMechanism
        mechanism that monitors and evalutes the values specified in the ControlMechanism's **monitor_for_control** 
        argument, and transmits the result to the ControlMechanism's *ERROR_SIGNAL* 
        `input_state <ControlMechanism.input_state>`.    

    control_signals : List[ControlSignal]
        list of `ControlSignals <ControlSignals>` for the ControlMechanism, each of which sends a `ControlProjection`
        to the `parameterState <ParameterState>` for the parameter it controls (same as ControlMechanism's 
        `output_states <Mechanism.output_states>` attribute).

    control_projections : List[ControlProjection]
        list of `ControlProjections <ControlProjection>`, one for each `ControlSignal` in `control_signals`.

    allocation_policy : 2d np.array
        each item is the value assigned to the corresponding ControlSignal in `control_signals`
        (same as the ControlMechanism's `value <Mechanism.value>` attribute).
        
    """

    componentType = "ControlMechanism"

    initMethod = INIT__EXECUTE__METHOD_ONLY

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'ControlMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    # variableClassDefault = defaultControlAllocation
    # This must be a list, as there may be more than one (e.g., one per control_signal)
    variableClassDefault = defaultControlAllocation

    from PsyNeuLink.Components.Functions.Function import Linear
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({CONTROL_PROJECTIONS: None})

    @tc.typecheck
    def __init__(self,
                 default_input_value=None,
                 system=None,
                 monitor_for_control:tc.optional(list)=None,
                 function = Linear(slope=1, intercept=0),
                 control_signals:tc.optional(list) = None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(monitor_for_control=monitor_for_control,
                                                  control_signals=control_signals,
                                                  function=function,
                                                  params=params)

        super(ControlMechanism_Base, self).__init__(variable=default_input_value,
                                                    params=params,
                                                    name=name,
                                                    prefs=prefs,
                                                    context=self)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate SYSTEM, MONITOR_FOR_CONTROL and CONTROL_SIGNALS

        If system is specified, validate it
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
                if not any((spec is mech.name or spec in mech.output_states.names) for mech in self.system.mechanisms):
                    raise ControlMechanismError("Specification in {} arg for {} ({}) must be a "
                                                "Mechanism or an OutputState of one in {}".
                                                format(MONITOR_FOR_CONTROL, self.name, spec, self.system.name))

        if CONTROL_SIGNALS in target_set and target_set[CONTROL_SIGNALS]:

            from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.ControlSignal \
                import ControlSignal

            for spec in target_set[CONTROL_SIGNALS]:

                # Specification is for a tuple (str, Mechanism):
                #    string must be the name of an attribute of the Mechanism,
                #    the Mechanism must have a ParameterState with the name of that attribute,
                #    and the Mechanism must be in the current system.
                if isinstance(spec, tuple):
                    param_name = spec[0]
                    mech = spec[1]

                    # Check that 1st item is a str (presumably the name of mechanism attribute for the param)
                    if not isinstance(param_name, str):
                        raise ControlMechanismError("1st item of tuple in specification of {} for {} ({}) "
                                                    "must be a string".format(CONTROL_SIGNAL, owner.name, param_name))
                    # Check that 2nd item is a mechanism
                    if not isinstance(mech, Mechanism):
                        raise ControlMechanismError("2nd item of tuple in specification of {} for {} ({}) "
                                                    "must be a Mechanism".format(CONTROL_SIGNAL, owner.name, mech))
                    # Check that param (named by str) is an attribute of the mechanism
                    if not hasattr(mech, param_name) and not hasattr(mech.function_object, param_name):
                        raise ControlMechanismError("{} (in specification of {}  {}) is not an attribute of {} or its function"
                                                    .format(param_name, CONTROL_SIGNAL, owner.name, mech))
                    # Check that the mechanism has a parameterState for the param
                    if not param_name in mech._parameter_states.names:
                        raise ControlMechanismError("There is no ParameterState for the parameter ({}) of {} "
                                                    "specified in {} for {}".
                                                    format(param_name, mech.name, CONTROL_SIGNAL, owner.name))
                    # Check that the mechanism to which the parameter belongs is in the controller's system
                    if not mech in self.system.mechanisms:
                        raise ControlMechanismError("Specification in {} arg for {} ({} param of {}) "
                                                    "must be for a Mechanism in {}".
                                                    format(CONTROL_SIGNALS,
                                                           self.name,
                                                           param_name,
                                                           mech.name,
                                                           self.system.name))

                # Specification is for a ControlSignal
                elif isinstance(spec, ControlSignal):
                    #  Check that any ControlProjections it has are to mechanisms in the controller's system
                    if not all(control_proj.receiver.owner in self.system.mechanisms
                               for control_proj in spec.efferents):
                        raise ControlMechanismError("The ControlSignal specified in the {} arg for {} ({}) "
                                                    "has one more more ControlProjections to a mechanism "
                                                    "that is not in {}".
                                                    format(CONTROL_SIGNALS, self.name, spec.name, self.system.name))

                # ControlSignal specification dictionary, must have the following entries:
                #    NAME:str - must be the name of an attribute of MECHANISM
                #    MECHANISM:Mechanism - must have an attribute and corresponding ParameterState with PARAMETER
                #    PARAMS:dict - entries must be valid ControlSignal parameters (e.g,. ALLOCATION_SAMPLES)
                elif isinstance(spec, dict):
                    if not NAME in spec:
                        raise ControlMechanismError("Specification dict for {} of {} must have a NAME entry".
                                                    format(CONTROL_SIGNAL, self.name))
                    if not MECHANISM in spec:
                        raise ControlMechanismError("Specification dict for {} of {} must have a MECHANISM entry".
                                                    format(CONTROL_SIGNAL, self.name))
                else:
                    # raise ControlMechanismError("PROGRAM ERROR: unrecognized ControlSignal specification for {} ({})".
                    #                             format(self.name, spec))
                    #
                    raise ControlMechanismError("Specification of {} for {} ({}) must be a "
                                                "ParameterState, a tuple specifying a parameter and mechanism, a "
                                                "ControlSignal specification dictionary, or an existing ControlSignal".
                                                format(CONTROL_SIGNAL, self.name, spec))

    def _validate_projection(self, projection, context=None):
        """Insure that projection is to mechanism within the same system as self
        """

        if projection.value is DEFERRED_INITIALIZATION:
            receiver_mech = projection.init_args['receiver'].owner
        else:
            receiver_mech = projection.receiver.owner
        if not receiver_mech in self.system.mechanisms:
            raise ControlMechanismError("Attempt to assign ControlProjection {} to a mechanism ({}) that is not in {}".
                                              format(projection.name, receiver_mech.name, self.system.name))

    def _instantiate_monitored_output_states(self, context=None):
        raise ControlMechanismError("{0} (subclass of {1}) must implement _instantiate_monitored_output_states".
                                          format(self.__class__.__name__,
                                                 self.__class__.__bases__[0].__name__))

    def _instantiate_output_states(self, context=None):

        if self.control_signals:
            for control_signal in self.control_signals:
                self._instantiate_control_signal(control_signal=control_signal, context=context)

        # IMPLEMENTATION NOTE:  Don't want to call this because it instantiates undesired default outputState
        # super()._instantiate_output_states(context=context)

    def _instantiate_attributes_after_function(self, context=None):
        """Implment ControlSignals specified in control_signals arg or "locally" in parameter specification(s)

        Calls super's instantiate_attributes_after_function, which calls _instantiate_output_states;
            that insures that any ControlSignals specified in control_signals arg are instantiated first
        Then calls _take_over_as_default_controller to instantiate any ControlProjections/ControlSignals specified 
            along with parameter specification(s) (i.e., as part of a (<param value>, ControlProjection) tuple
        """

        super()._instantiate_attributes_after_function(context=context)

        if MAKE_DEFAULT_CONTROLLER in self.paramsCurrent:
            if self.paramsCurrent[MAKE_DEFAULT_CONTROLLER]:
                self._take_over_as_default_controller(context=context)
            if not self.system.enable_controller:
                return

        # FIX: 5/23/17 CONSOLIDATE/SIMPLIFY THIS RE: control_signal ARG??  USE OF STATE_PROJECTIONS, ETC.
        # FIX:         ?? WHERE WOULD CONTROL_PROJECTIONS HAVE BEEN SPECIFIED IN paramsCURRENT??
        # FIX:         DOCUMENT THAT VALUE OF CONTROL ENTRY CAN BE A PROJECTION
        # FIX:         RE-WRITE parse_state_spec TO TAKE TUPLE THAT SPECIFIES (PARAM VALUE, CONTROL SIGNAL)
        #                       RATHER THAN (PARAM VALUE, CONTROL PROJECTION)
        # FIX: NOT CLEAR THIS IS GETTING USED AT ALL; ALSO, ??REDUNDANT WITH CALL IN _instantiate_output_states
        # If ControlProjections were specified, implement them
        if CONTROL_PROJECTIONS in self.paramsCurrent:
            if self.paramsCurrent[CONTROL_PROJECTIONS]:
                for key, projection in self.paramsCurrent[CONTROL_PROJECTIONS].items():
                    control_signal_spec = {CONTROL:[projection]}
                    self._instantiate_control_signal(control_signal_spec, context=self.name)

    def _take_over_as_default_controller(self, context=None):

        # FIX 5/23/17: INTEGRATE THIS WITH ASSIGNMENT OF control_signals
        # FIX:         (E.G., CHECK IF SPECIFIED ControlSignal ALREADY EXISTS)
        # Check the parameterStates of the system's mechanisms for any ControlProjections with deferred_init()
        for mech in self.system.mechanisms:
            for parameter_state in mech._parameter_states:
                for projection in parameter_state.afferents:
                    # If projection was deferred for init, initialize it now and instantiate for self
                    if projection.value is DEFERRED_INITIALIZATION and projection.init_args['sender'] is None:
                        # FIX 5/23/17: MODIFY THIS WHEN (param, ControlProjection) tuple
                        # FIX:         IS REPLACED WITH (param, ControlSignal) tuple
                        # Add projection itself to any params specified in the ControlProjection for the ControlSignal
                        #    (cached in the ControlProjection's control_signal attrib)
                        control_signal_specs = projection.control_signal or {}
                        control_signal_specs.update({CONTROL_SIGNAL_SPECS: [projection]})
                        self._instantiate_control_signal(control_signal_specs, context=context)

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
        """Instantiate OutputState for ControlSignal and assign (if specified) or instantiate ControlProjection

        # Extends allocation_policy and control_signal_costs attributes to accommodate instantiated projection

        Notes:
        * control_signal arg can be a:
            - ControlSignal object;
            - ControlProjection;
            - ParameterState;
            - params dict, from _take_over_as_default_controller(), containing a ControlProjection;
            - tuple (param_name, Mechanism), from control_signals arg of constructor;
                    [NOTE: this is a convenience format;
                           it precludes specification of ControlSignal params (e.g., ALLOCATION_SAMPLES)]
            - ControlSignal specification dictionary, from control_signals arg of constructor
                    [NOTE: this must have at least NAME:str (param name) and MECHANISM:Mechanism entries; 
                           it can also include a PARAMS entry with a params dict containing ControlSignal params] 
        * State._parse_state_spec() is used to parse control_signal arg
        * params are expected to be for (i.e., to be passed to) ControlSignal;
        * wait to instantiate deferred_init() projections until after ControlSignal is instantiated,
             so that correct outputState can be assigned as its sender;
        * index of outputState is incremented based on number of ControlSignals already instantiated;

        Returns ControlSignal (OutputState)
        """
        from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.ControlSignal import ControlSignal
        from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection


        # EXTEND allocation_policy TO ACCOMODATE NEW ControlSignal -------------------------------------------------
        #        also used to determine contraint on ControlSignal value

        if self.allocation_policy is None:
            self.allocation_policy = np.array(defaultControlAllocation)
        else:
            self.allocation_policy = np.append(self.allocation_policy, defaultControlAllocation)

        # PARSE control_signal SPECIFICATION -----------------------------------------------------------------------

        control_projection = None
        control_signal_params = None

        control_signal_spec = _parse_state_spec(owner=self, state_type=ControlSignal, state_spec=control_signal)

        def _get_parameter_state(param_name, mech):
            try:
                return mech._parameter_states[param_name]
            except KeyError:
                # Check that param (named by str) is an attribute of the mechanism
                if not (hasattr(mech, param_name) or hasattr(mech.function_object, param_name)):
                    raise ControlMechanismError("{} (in specification of {}  {}) is not an attribute "
                                                "of {} or its function"
                                                .format(param_name, CONTROL_SIGNAL, owner.name, mech))
                # Check that the mechanism has a parameterState for the param
                if not param_name in mech._parameter_states.names:
                    raise ControlMechanismError("There is no ParameterState for the parameter ({}) of {} "
                                                "specified in {} for {}".
                                                format(param_name, mech.name, CONTROL_SIGNAL, owner.name))

        # Specification is a ParameterState
        if isinstance(control_signal_spec, ParameterState):
            mech = control_signal_spec.owner
            param_name = control_signal_spec.name
            parameter_state = _get_parameter_state(param_name, mech)

        # Specification was tuple or dict, and parsed into a dict
        elif isinstance(control_signal_spec, dict):
            param_name = control_signal_spec[NAME]
            control_signal_params = control_signal_spec[PARAMS]

            # control_signal was a specification dict, with MECHANISM as an entry (and parameter as NAME)
            if control_signal_params and MECHANISM in control_signal_params:
                mech = control_signal_params[MECHANISM]
                del control_signal_params[MECHANISM]
                parameter_state = _get_parameter_state(param_name, mech)

            # Specification was originally a tuple, either in parameter specification or control_signal arg;
            #    1st item was either assigned to the NAME entry of the control_signal_spec dict
            #        (if tuple was a (param_name, Mechanism tuple) for control_signal arg;
            #        or used as param value, if it was a parameter specification tuple
            #    2nd item was placed CONTROL_SIGNAL_PARAMS entry of params dict in control_signal_spec dict,
            #        so parse:
            # FIX 5/23/17: NEED TO GET THE KEYWORDS STRAIGHT FOR PASSING ControlSignal SPECIFICATIONS
            # IMPLEMENTATION NOTE:
            #     CONTROL_SIGNAL_SPECS is used by _take_over_as_default_controller,
            #                          to pass specification from a parameter specification tuple
            #     STATE_PROJECTIONS is used by _parse_state_spec to place the 2nd item of any tuple in params dict;
            #                       here, the tuple comes from a (param, mechanism) specification in control_signal arg
            elif (control_signal_params and
                    any(kw in control_signal_spec[PARAMS] for kw in {CONTROL_SIGNAL_SPECS, STATE_PROJECTIONS})):
                if CONTROL_SIGNAL_SPECS in control_signal_spec[PARAMS]:
                    spec = control_signal_params[CONTROL_SIGNAL_SPECS]
                    del control_signal_params[CONTROL_SIGNAL_SPECS]
                elif STATE_PROJECTIONS in control_signal_spec[PARAMS]:
                    spec = control_signal_params[STATE_PROJECTIONS]
                    del control_signal_params[STATE_PROJECTIONS]

                # ControlSignal
                if isinstance(spec, ControlSignal):
                    control_signal_spec = spec

                else:
                    # Mechanism
                    # IMPLEMENTATION NOTE: Mechanism was placed in list in STATE_PROJECTIONS entry by _parse_state_spec
                    if isinstance(spec, list) and isinstance(spec[0], Mechanism):
                        mech = spec[0]
                        parameter_state = _get_parameter_state(param_name, mech)

                    # Projection (in a list)
                    elif isinstance(spec, list):
                        control_projection = spec[0]
                        if not isinstance(control_projection, ControlProjection):
                            raise ControlMechanismError("PROGRAM ERROR: list in {} entry of params dict for {} of {} "
                                                        "must contain a single ControlProjection".
                                                        format(CONTROL_SIGNAL_SPECS, CONTROL_SIGNAL, self.name))
                        if len(spec)>1:
                            raise ControlMechanismError("PROGRAM ERROR: list of ControlProjections is not "
                                                        "currently supported in specification of a ControlSignal")
                        # Get receiver mech
                        if control_projection.value is DEFERRED_INITIALIZATION:
                            parameter_state = control_projection.init_args['receiver']
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


        # Specification is a ControlSignal (either passed in directly, or parsed from tuple above)
        if isinstance(control_signal_spec, ControlSignal):
            # Deferred Initialization, so assign owner, name, and initialize
            if control_signal_spec.value is DEFERRED_INITIALIZATION:
                # FIX 5/23/17:  IMPLEMENT DEFERRED_INITIALIZATION FOR ControlSignal
                # CALL DEFERRED INIT WITH SELF AS OWNER ??AND NAME FROM control_signal_dict?? (OR WAS IT SPECIFIED)
                # OR ASSIGN NAME IF IT IS DEFAULT, USING CONTROL_SIGNAL_DICT??
                pass
            elif not control_signal_spec.owner is self:
                raise ControlMechanismError("Attempt to assign ControlSignal to {} ({}) that is already owned by {}".
                                            format(self.name, control_signal_spec.name, control_signal_spec.owner.name))
            control_signal = control_signal_spec
            control_signal_name = control_signal_spec.name
            control_projections = control_signal_spec.efferents

            # IMPLEMENTATION NOTE:
            #    THIS IS TO HANDLE FUTURE POSSIBILITY OF MULTIPLE ControlProjections FROM A SIGN ControlSignal;
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
            control_signal_name = param_name + '_' + ControlSignal.__name__

            from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.ControlSignal \
                import ControlSignal
            from PsyNeuLink.Components.States.State import _instantiate_state

            # Get constraint for OutputState's value
            #    - get ControlMechanism's value
            self._update_value(context=context)
            # - get OutputState's index
            try:
                output_state_index = len(self.output_states)
            except (AttributeError, TypeError):
                output_state_index = 0
            # - get constraint for OutputState's value
            output_state_constraint_value = self.allocation_policy[output_state_index]

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
            self._validate_projection(control_projection)

            from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
            if not isinstance(control_projection, ControlProjection):
                raise ControlMechanismError("PROGRAM ERROR: Attempt to assign {0}, "
                                                  "that is not a ControlProjection, to outputState of {1}".
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

        # Add ControlProjection to list of outputState's outgoing projections
        # (note: if it was deferred, it just added itself, skip)
        if not control_projection in control_signal.efferents:
            control_signal.efferents.append(control_projection)

        # Add ControlProjection to ControlMechanism's list of ControlProjections
        try:
            self.control_projections.append(control_projection)
        except AttributeError:
            self.control_projections = [control_projection]

        # Update control_signal_costs to accommodate instantiated projection
        try:
            self.control_signal_costs = np.append(self.control_signal_costs, np.empty((1,1)),axis=0)
        except AttributeError:
            self.control_signal_costs = np.empty((1,1))

        # UPDATE output_states AND control_projections -------------------------------------------------------------

        try:
            self.output_states[control_signal.name] = control_signal
        except (AttributeError, TypeError):
            from PsyNeuLink.Components.States.State import State_Base
            self.output_states = ContentAddressableList(component_type=State_Base, list=[control_signal])

        # Add index assignment to outputState
        control_signal.index = output_state_index

        # (Re-)assign control_signals attribute to output_states
        self.control_signals = self.output_states

        return control_signal

    def _execute(self,
                    variable=None,
                    runtime_params=None,
                    clock=CentralClock,
                    time_scale=TimeScale.TRIAL,
                    context=None):
        """Updates ControlProjections based on inputs

        Must be overriden by subclass
        """
        raise ControlMechanismError("{0} must implement execute() method".format(self.__class__.__name__))

    def show(self):

        print ("\n---------------------------------------------------------")

        print ("\n{0}".format(self.name))
        print("\n\tMonitoring the following mechanism outputStates:")
        for state in self.monitoring_mechanism.input_states:
            for projection in state.afferents:
                monitored_state = projection.sender
                monitored_state_mech = projection.sender.owner
                monitored_state_index = self.monitored_output_states.index(monitored_state)

                weight = self.monitor_for_control_weights_and_exponents[monitored_state_index][0]
                exponent = self.monitor_for_control_weights_and_exponents[monitored_state_index][1]

                print ("\t\t{0}: {1} (exp: {2}; wt: {3})".
                       format(monitored_state_mech.name, monitored_state.name, weight, exponent))

        print ("\n\tControlling the following mechanism parameters:".format(self.name))
        # Sort for consistency of output:
        state_names_sorted = sorted(self.output_states.names)
        for state_name in state_names_sorted:
            for projection in self.output_states[state_name].efferents:
                print ("\t\t{0}: {1}".format(projection.receiver.owner.name, projection.receiver.name))

        print ("\n---------------------------------------------------------")