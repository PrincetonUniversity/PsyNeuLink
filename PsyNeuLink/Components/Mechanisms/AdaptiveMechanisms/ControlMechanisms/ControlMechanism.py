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
A ControlMechanism is also created automatically whenever a `system is created <System_Creation>`, and assigned as
the `controller <System_Execution_Control>` for that system. The `outputStates <OutputState>` to be monitored by a
ControlMechanism are specified in its `monitored_output_states` argument, which can take  a number of
`forms <ObjectiveMechanism_Monitored_Values>`.  When the ControlMechanism is created, it automatically creates
an ObjectiveMechanism that is used to monitor and evaluate the mechanisms and/or outputStates specified in its
`monitor_for_control <ControlMechanism.monitor_for_control>` attribute.  The result of the evaluation is used to
specify the value of the ControlMechanism's `ControlProjections <ControlProjection>`. How a ControlMechanism creates its
ControlProjections and determines their value based on the outcome of its evaluation  depends on the
`subclass <ControlMechanism>`.

.. _ControlMechanism_Specifying_Control:

Specifying control for a parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ControlMechanisms are used to control the parameter values of mechanisms and/or their functions.  A parameter can be
specified for control by assigning a `ControlProjection` as part of its value when creating the mechanism or function
to which the parameter belongs (see `Mechanism_Parameters`).

.. _ControlMechanism_Monitored_OutputStates:

Monitored OutputStates
~~~~~~~~~~~~~~~~~~~~~~

When an ControlMechanism is constructed automatically, it creates an `ObjectiveMechanism` (specified in its
`montioring_mechanism` attribute) that is used to monitor and evaluate the system's performance.  The
ObjectiveMechanism monitors each mechanism and/or outputState listed in the ControlMechanism's
'monitor_for_control <ControlMechanism.monitor_for_control>` attribute, and evaluates them using the its `function`.
This information is used to set the value of the ControlMechanism's ControlProjections.

.. _ControlMechanism_Execution:

Execution
---------

A ControlMechanism that is a system's `controller` is always the last mechanism to be executed (see `System Control
<System_Execution_Control>`).  Its `function <ControlMechanism.function>` takes as its input the values of the
outputStates in its `monitored_output_states` attribute, and uses those to determine the value of its
`ControlProjections <ControlProjection>`. In the subsequent round of execution, each ControlProjection's value is
used by the `ParameterState` to which it projects to update the parameter being controlled.

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
from PsyNeuLink.Components.States.OutputState import OutputState

ControlMechanismRegistry = {}


class ControlMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class ControlMechanism_Base(Mechanism_Base):
    """
    ControlMechanism_Base(     \
    default_input_value=None,  \
    monitor_for_control=None,  \
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
              ._instantiate_control_projection INSTANTIATES OUTPUT STATE FOR EACH CONTROL SIGNAL ASSIGNED TO THE
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

            NOT CURRENTLY IN USE:
            default_input_value : value, list or np.ndarray : :py:data:`defaultControlAllocation <LINK]>`
                the default allocation for the ControlMechanism;
                its length should equal the number of ``controlSignals``.

        monitor_for_control : List[OutputState specification] : default None
            specifies set of outputStates to monitor (see :ref:`ControlMechanism_Monitored_OutputStates` for
            specification options).

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

    controlProjections : List[ControlProjection]
        list of `ControlProjections <ControlProjection>` managed by the ControlMechanism.
        There is one for each ouputState in the `outputStates` dictionary.

    controlProjectionCosts : 2d np.array
        array of costs associated with each of the control signals in the `controlProjections` attribute.

    allocation_policy : 2d np.array
        array of values assigned to each control signal in the `controlProjections` attribute.
        This is the same as the ControlMechanism's `value <ControlMechanism.value>` attribute.


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
    # This must be a list, as there may be more than one (e.g., one per controlSignal)
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
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(monitor_for_control=monitor_for_control,
                                                  function=function,
                                                  params=params)

        super(ControlMechanism_Base, self).__init__(variable=default_input_value,
                                                    params=params,
                                                    name=name,
                                                    prefs=prefs,
                                                    context=self)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate SYSTEM, MONITOR_FOR_CONTROL and FUNCTION_PARAMS

        If system is specified, validate it
        Check that all items in MONITOR_FOR_CONTROL are Mechanisms or OutputStates for Mechanisms in self.system
        Check that len(WEIGHTS) = len(MONITOR_FOR_CONTROL)
        """

        if SYSTEM in request_set:
            if not isinstance(request_set[SYSTEM], System):
                raise KeyError
            else:
                self.paramClassDefaults[SYSTEM] = request_set[SYSTEM]

        if MONITOR_FOR_CONTROL in request_set:
            for spec in request_set[MONITOR_FOR_CONTROL]:
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

        super(ControlMechanism_Base, self)._validate_params(request_set=request_set,
                                                                 target_set=target_set,
                                                                 context=context)

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

    def _instantiate_attributes_after_function(self, context=None):
        """Take over as default controller (if specified) and implement any specified ControlProjections

        """

        if MAKE_DEFAULT_CONTROLLER in self.paramsCurrent:
            if self.paramsCurrent[MAKE_DEFAULT_CONTROLLER]:
                self._take_over_as_default_controller(context=context)
            if not self.system.enable_controller:
                return

        # If ControlProjections were specified, implement them
        if CONTROL_PROJECTIONS in self.paramsCurrent:
            if self.paramsCurrent[CONTROL_PROJECTIONS]:
                for key, projection in self.paramsCurrent[CONTROL_PROJECTIONS].items():
                    self._instantiate_control_projection(projection, context=self.name)

    def _take_over_as_default_controller(self, context=None):

        # Check the parameterStates of the system's mechanisms for any ControlProjections with deferred_init()
        for mech in self.system.mechanisms:
            for parameter_state in mech._parameter_states.values():
                for projection in parameter_state.receivesFromProjections:
                    # If projection was deferred for init, initialize it now and instantiate for self
                    if projection.value is DEFERRED_INITIALIZATION and projection.init_args['sender'] is None:
                        # Get params specified with projection for its ControlSignal (cached in control_signal attrib)
                        params = projection.control_signal
                        self._instantiate_control_projection(projection, params=params, context=context)

    def _instantiate_control_projection(self, projection, params=None, context=None):
        """Add outputState (as ControlSignal) and assign as sender to requesting ControlProjection

        # Updates allocation_policy and controlSignalCosts attributes to accommodate instantiated projection

        Notes:  
        * params are expected to be for (i.e., to be passed to) ControlSignal;
        * wait to instantiate deferred_init() projections until after ControlSignal is instantiated,
             so that correct outputState can be assigned as its sender;
        * index of outputState is incremented based on number of ControlSignals already instantiated;
        * assume that self.allocation_policy has already been extended 
            to include the particular (indexed) allocation to be used for the outputState being created here.


        Returns state: (OutputState)
        """

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

        # Update self.outputState and self.outputStates
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
        if not projection in state.sendsToProjections:
            state.sendsToProjections.append(projection)

        # Add ControlProjection to ControlMechanism's list of ControlProjections
        try:
            self.controlProjections.append(projection)
        except AttributeError:
            self.controlProjections = [projection]

        # Update controlSignalCosts to accommodate instantiated projection
        try:
            self.controlSignalCosts = np.append(self.controlSignalCosts, np.empty((1,1)),axis=0)
        except AttributeError:
            self.controlSignalCosts = np.empty((1,1))

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
            for projection in state.receivesFromProjections:
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
            for projection in self.output_states[state_name].sendsToProjections:
                print ("\t\t{0}: {1}".format(projection.receiver.owner.name, projection.receiver.name))

        print ("\n---------------------------------------------------------")