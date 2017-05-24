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
A ControlMechanism can regulate only the parameters of mechanism in the system for which it is the
`controller <System_Execution_Control>`.  The control components of a system can be displayed using the system's 
`show_graph` method with its **show_control** argument assigned :keyword:``True`.  The control components of a 
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
mechanism or function to which the parameter belongs (see `Mechanism_Parameters`), or by specifying the parameter in 
the **control_signals**  argument of the constructor for the ControlMechanism.  The **control_signals** argument must 
be a list, each item of which must refer to a parameter to be controlled specified in any of the following forms:

  * a **ParameterState** (of a Mechanism) for the parameter;
  |
  * a **tuple**, with the *name* of the parameter as its 1st item. and the *mechanism* to which it belongs as the 2nd.

A `ControlSignal` is created for each item listed in **control_signals**, and all of a ControlMechanism's ControlSignals 
are listed in ControlMechanism's `control_signals <ControlMechanism.control_signals>` attribute.  Each ControlSignal is 
assigned a `ControlProjection` to the parameterState of the mechanisms associated with the specified parameter of the
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

# IMPLEMENTATION NOTE: COPIED FROM DefaultProcessingMechanism;
#                      ADD IN GENERIC CONTROL STUFF FROM DefaultControlMechanism

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
            # DOCUMENTATION NEEDED:
              ._instantiate_control_signal INSTANTIATES OUTPUT STATE FOR EACH CONTROL SIGNAL ASSIGNED TO THE
             INSTANCE
            .EXECUTE MUST BE OVERRIDDEN BY SUBCLASS
            WHETHER AND HOW MONITORING INPUT STATES ARE INSTANTIATED IS UP TO THE SUBCLASS

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

    COMMENT:
        Arguments
        ---------

        monitor_for_control : List[OutputState specification] : default None
            specifies set of outputStates to monitor (see :ref:`ControlMechanism_Monitored_OutputStates` for
            specification options).

        control_signals : List[Attribute of Mechanism or its function, ParameterState, or tuple[str, Mechanism]
            specifies the parameters to be controlled by the ControlMechanism 
            (see `control_signals <ControMechanism.control_signals>` for details).

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
    COMMENT


    Attributes
    ----------

    monitoring_mechanism : ObjectiveMechanism
        mechanism that monitors and evalutes the values specified in the ControlMechanism's **monitor_for_control** 
        argument, and transmits the result to the ControlMechanism's *ERROR_SIGNAL* 
        `input_state <ControlMechanism.input_state>`.    

    control_signals : List[ControlSignal]
        list of `ControlSignals <ControlSignals>` for the ControlMechanism, each of which sends a `ControlProjection`
        to the parameterState for the parameter it controls (this is the same as ControlMechanism's 
        `output_states <ControlMechanism.output_states>` attribute).


    COMMENT:  [REPLACE THIS WITH control_signals]    
    control_projections : List[ControlProjection]
        list of `ControlProjections <ControlProjection>` managed by the ControlMechanism.
        There is one for each ouputState in the `outputStates` dictionary.
    COMMENT

    allocation_policy : 2d np.array
        array of values assigned to each ControlSignal listed in the 
        `control_signals <ControlMechanism.control_signals>` attribute
        (this is the same as the ControlMechanism's `value <ControlMechanism.value>` attribute).
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

            for spec in target_set[CONTROL_SIGNALS]:

                # FIX: 5/23/17 MODIFY TO USE SPECIFICATION DICTIONARY RATHER THAN TUPLE (AND MAYBE _parse_state_spec)
                # ControlSignal specification
                # Check that all of its ControlProjections are to mechanisms in the controller's system
                from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.ControlSignal \
                    import ControlSignal
                if isinstance(spec, ControlSignal):
                    if not all(control_proj.receiver.owner in self.system.mechanisms
                               for control_proj in spec.efferents):
                        raise ControlMechanismError("The ControlSignal specified in the {} arg for {} ({}) "
                                                    "has one more more ControlProjections to a mechanism "
                                                    "that is not in {}".
                                                    format(CONTROL_SIGNALS, self.name, spec.name, self.system.name))
                    continue

                # ParameterState specification
                if isinstance(spec, ParameterState):
                    mech_spec = ParameterState.owner
                    param_name = ParameterState.name

                # Tuple (parameter name, mechanism) specification
                elif isinstance(spec, tuple):
                    param_name = spec[0]
                    mech_spec = spec[1]
                    # Check that 1st item is a str (presumably the name of mechanism attribute for the param)
                    if not isinstance(param_name, str):
                        raise ControlMechanismError("1st item of tuple in specification in {} arg for {} ({}) "
                                                    "must be a string".format(CONTROL_SIGNALS, self.name, param_name))
                    # Check that 2nd item is a mechanism
                    if not isinstance(mech_spec, Mechanism):
                        raise ControlMechanismError("2nd item of tuple in specification in {} arg for {} ({}) "
                                                    "must be a Mechanism".format(CONTROL_SIGNALS, self.name, mech_spec))
                    # Check that param (named by str) is an attribute of the mechanism
                    if not hasattr(mech_spec, param_name) and not hasattr(mech_spec.function_object, param_name):
                        raise ControlMechanismError("{} is not an attribute of {} (in {} arg for {})"
                                                    .format(param_name, mech_spec, CONTROL_SIGNALS, self.name))
                    # Check that the mechanism has a parameterState for the param
                    if not param_name in mech_spec._parameter_states.names:
                        raise ControlMechanismError("There is no ParameterState for the parameter ({}) of {} "
                                                    "specified in the {} argument for {}".
                                                    format(param_name, mech_spec.name, CONTROL_SIGNALS, self.name))

                else:
                    raise ControlMechanismError("Specification in {} arg for {} ({}) must be a "
                                                "ParameterState, a tuple specifying a parameter and mechanism, "
                                                "or an existing ControlSignal".
                                                format(MONITOR_FOR_CONTROL, self.name, spec))

                # Check that the mechanism (to which the param belongs) is in the controller's system
                if not mech_spec in self.system.mechanisms:
                    raise ControlMechanismError("Specification in {} arg for {} ({} param of {}) "
                                                "must be for a Mechanism in {}".
                                                format(param_name,
                                                       CONTROL_SIGNALS,
                                                       self.name,
                                                       mech_spec.name,
                                                       self.system.name))

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

        for control_signal in self.control_signals:
            self._instantiate_control_signal(control_signal=control_signal, context=context)

        super()._instantiate_output_states(context=context)

    def _instantiate_attributes_after_function(self, context=None):
        """Take over as default controller (if specified) and implement any specified ControlProjections
        """
        
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
                        projection.control_signal.update({MODULATORY_PROJECTIONS: [projection]})
                        control_signal_spec = {PARAMS: projection.control_signal}
                        self._instantiate_control_signal(control_signal_spec, context=context)

    # ---------------------------------------------------
    # IMPLEMENTATION NOTE:  IMPLEMENT _instantiate_output_states THAT CALLS THIS FOR EACH ITEM
    #                       DESIGN PATTERN SHOULD COMPLEMENT THAT FOR _instantiate_input_states of ObjectiveMechanism
    #                           (with control_signals taking the place of monitored_values)
    # FIX 5/23/17: PROJECTIONS AND PARAMS SHOULD BE PASSED BY ASSIGNING TO STATE SPECIFICATION DICT
    # FIX          UPDATE parse_state_spec TO ACCOMODATE (param, ControlSignal) TUPLE
    # FIX          TRACK DOWN WHERE PARAMS ARE BEING HANDED OFF TO ControlProjection
    # FIX                   AND MAKE SURE THEY ARE NOW ADDED TO ControlSignal SPECIFICATION DICT
    #
    # def _instantiate_control_signal(self, projection=None, params=None, context=None):
    def _instantiate_control_signal(self, control_signal=None, context=None):
        """Add outputState (as ControlSignal) and assign as sender to requesting ControlProjection

        # Updates allocation_policy and control_signal_costs attributes to accommodate instantiated projection

        Notes:  
        * params are expected to be for (i.e., to be passed to) ControlSignal;
        * wait to instantiate deferred_init() projections until after ControlSignal is instantiated,
             so that correct outputState can be assigned as its sender;
        * index of outputState is incremented based on number of ControlSignals already instantiated;
        * assume that self.allocation_policy has already been extended 
            to include the particular (indexed) allocation to be used for the outputState being created here.

        Returns ControlSignal (OutputState)
        """

        if self.allocation_policy is None:
            self.allocation_policy = np.array(defaultControlAllocation)
        else:
            self.allocation_policy = np.append(self.allocation_policy, defaultControlAllocation)

        # Parse control_signal to get projection and params
        from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.ControlSignal import ControlSignal
        control_signal_dict = _parse_state_spec(owner=self, state_type=ControlSignal, state_spec=control_signal)
        # FIX: 5/23/17 ??IS [CONTROL] THE RIGHT KEY TO USE HERE (VS. [PARAMS] OR [PARAMS][MODULATORY_PROJECTIONS]
        # FIX: 5/23/17 **SHOULD HANDLE MULTIPLE PROJECTIONS
        # projection = control_signal_dict[PARAMS][MODULATORY_PROJECTIONS]
        projection = control_signal_dict[PARAMS][MODULATORY_PROJECTIONS][0]
        params = control_signal_dict[PARAMS]

        # Validate projection (if specified) and get receiver's name
        if projection:
            self._validate_projection(projection)

            # get name of projection receiver (for use in naming the ControlSignal)
            if projection.value is DEFERRED_INITIALIZATION:
                receiver = projection.init_args['receiver']
            else:
                receiver = projection.receiver

            from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
            if not isinstance(projection, ControlProjection):
                raise ControlMechanismError("PROGRAM ERROR: Attempt to assign {0}, "
                                                  "that is not a ControlProjection, to outputState of {1}".
                                                  format(projection, self.name))
        # Otherwise, use name specified in control_signals

        #  Update self.value by evaluating function
        self._update_value(context=context)

        # Instantiate new outputState and assign as sender of ControlProjection
        try:
            output_state_index = len(self.output_states)
        except (AttributeError, TypeError):
            output_state_index = 0
        from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.ControlSignal import ControlSignal
        output_state_name = receiver.name + '_' + ControlSignal.__name__
        output_state_value = self.allocation_policy[output_state_index]

        # FIX: CALL super()_instantiate_output_states ??
        # FIX:     OR AGGREGATE ALL ControlSignals AND SEND AS LIST (AS FOR input_states IN ObjectiveMechanism)
        from PsyNeuLink.Components.States.State import _instantiate_state
        from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.ControlSignal import ControlSignal
        state = _instantiate_state(owner=self,
                                   state_type=ControlSignal,
                                   state_name=output_state_name,
                                   state_spec=defaultControlAllocation,
                                   state_params=params,
                                   constraint_value=output_state_value,
                                   constraint_value_name='Default control allocation',
                                   # constraint_output_state_index=output_item_output_state_index,
                                   context=context)

        # Assign outputState as ControlProjection's sender
        if projection.value is DEFERRED_INITIALIZATION:
            projection.init_args['sender']=state
            if projection.init_args['name'] is None:
                projection.init_args['name'] = CONTROL_PROJECTION + ' for ' + receiver.owner.name + ' ' + receiver.name
            projection._deferred_init()
        else:
            projection.sender = state

        # Update self.output_state and self.output_states
        try:
            self.output_states[state.name] = state
        except (AttributeError, TypeError):
            # self.output_states = OrderedDict({output_state_name:state})
            from PsyNeuLink.Components.States.State import State_Base
            self.output_states = ContentAddressableList(component_type=State_Base, list=[state])

        # Add index assignment to outputState
        state.index = output_state_index

        # Add ControlProjection to list of outputState's outgoing projections
        # (note: if it was deferred, it just added itself, skip)
        if not projection in state.efferents:
            state.efferents.append(projection)

        # Add ControlProjection to ControlMechanism's list of ControlProjections
        try:
            self.control_projections.append(projection)
        except AttributeError:
            self.control_projections = [projection]

        # Update control_signal_costs to accommodate instantiated projection
        try:
            self.control_signal_costs = np.append(self.control_signal_costs, np.empty((1,1)),axis=0)
        except AttributeError:
            self.control_signal_costs = np.empty((1,1))

        # Assign ControlSignals in the order they are stored of output_states
        self.control_signals = self.output_states

        return state

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

                # # MODIFIED 1/9/16 OLD:
                # exponent = \
                #     np.ndarray.item(self.paramsCurrent[OUTCOME_FUNCTION].__self__.exponents[
                #     monitored_state_index])
                # weight = \
                #     np.ndarray.item(self.paramsCurrent[OUTCOME_FUNCTION].__self__.weights[monitored_state_index])
                # MODIFIED 1/9/16 NEW:
                weight = self.monitor_for_control_weights_and_exponents[monitored_state_index][0]
                exponent = self.monitor_for_control_weights_and_exponents[monitored_state_index][1]
                # MODIFIED 1/9/16 END

                print ("\t\t{0}: {1} (exp: {2}; wt: {3})".
                       format(monitored_state_mech.name, monitored_state.name, weight, exponent))

        print ("\n\tControlling the following mechanism parameters:".format(self.name))
        # Sort for consistency of output:
        state_names_sorted = sorted(self.output_states.names)
        for state_name in state_names_sorted:
            for projection in self.output_states[state_name].efferents:
                print ("\t\t{0}: {1}".format(projection.receiver.owner.name, projection.receiver.name))

        print ("\n---------------------------------------------------------")