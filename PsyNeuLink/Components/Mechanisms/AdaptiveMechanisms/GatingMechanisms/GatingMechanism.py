# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  GatingMechanism ************************************************

"""
Overview
--------

A GatingMechanism is an `AdaptiveMechanism` that modifies the inputState(s) and/or outputState(s) of one or more 
`ProcessingMechanisms`.   It's function takes a value and uses that to calculate a `gating_policy`:  a list of 
`gating` values, one for each of states that it gates.  Each of these values is assigned as the value of a 
corresponding `GatingSignal` (a subclass of `OutputState` used by ControlMechanisms), and used by the
associated `GatingProjection` to gate the state to which it projects.  
COMMENT: TBI
The gating components of a system can be displayed using the system's 
`show_graph` method with its **show_gating** argument assigned :keyword:``True`.  
COMMENT
?????
The gating components of a 
system are executed after all ProcessingMechanisms and `learning components <LearningMechanism>` in that system have 
been executed.

.. _GatingMechanism_Creation:

Creating A GatingMechanism
---------------------------

GatingMechanisms can be created using the standard Python method of calling the constructor for the desired type.
COMMENT: ??TBI
A GatingMechanism is also created automatically if a `gating is specified <GatingMechanism_Specifying_Gating>` for a 
state but its `sender <GatingProjection.sender>` is not assigned.  In that case, a Gating
COMMENT
When gating is specified for a state, a `GatingProjection` is automatically instantiated that projects from the
designated GatingMechanism to the state. How a GatingMechanism creates its `GatingProjections <GatingProjection>` and 
determines their value depends on the `subclass <GatingMechanism>`.

.. _GatingMechanism_Specifying_Gating:

Specifying gating
~~~~~~~~~~~~~~~~~

Gating can be specified for an `InputState` or `OutputState` in any of the following way:

XXX DOCUMENTATION NEEDED HERE

.. _GatingMechanism_Execution:

Execution
---------

A GatingMechanism executes in the same way as a ProcessingMechanism, based on its place in the system's 
`graph <System.graph>`.  Because GatingProjections are likely to introduce cycles (loops) in the graph,
the effects of a GatingMechanism and its projections will generally not be applied in the first
`round of execution <LINK>` (see `initialization <LINK>` for a description of how to configure the initialization
of feedback loops in a System).  When executd, a GatingMechanism uses its input to determine the value of its
`GatingSignals <GatingSignal>` and their corresponding `GatingProjections <GatingProjection>`.  In the subsequent 
round of execution , each GatingProjection's value is used by the state to which it projects to modulate the 
`value <State.value>` of that state.

.. note::
   A state that receives a `GatingProjection` does not update its value until its owner mechanism executes 
   (see `Lazy Evaluation <LINK>` for an explanation of "lazy" updating).  This means that even if a GatingMechanism 
   has executed, a state that it gates will not assume its new value until the state's owner has executed.

.. _GatingMechanism_Class_Reference:

Class Reference
---------------

"""

# IMPLEMENTATION NOTE: COPIED FROM DefaultProcessingMechanism;
#                      ADD IN GENERIC CONTROL STUFF FROM DefaultGatingMechanism

from collections import OrderedDict

from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.AdaptiveMechanism import AdaptiveMechanism_Base
from PsyNeuLink.Components.ShellClasses import *

GatingMechanismRegistry = {}


class GatingMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class GatingMechanism(AdaptiveMechanism_Base):
    """
    GatingMechanism_Base(     \
    default_input_value=None,  \
    monitor_for_control=None,  \
    function=Linear,           \
    params=None,               \
    name=None,                 \
    prefs=None)

    Abstract class for GatingMechanism.

    .. note::
       GatingMechanisms should NEVER be instantiated by a direct call to the base class.
       They should be instantiated using the constructor for a :doc:`subclass <GatingMechanism>`.

    COMMENT:
        Description:
            # DOCUMENTATION NEEDED:
              ._instantiate_gating_projection INSTANTIATES OUTPUT STATE FOR EACH CONTROL SIGNAL ASSIGNED TO THE
             INSTANCE
            .EXECUTE MUST BE OVERRIDDEN BY SUBCLASS
            WHETHER AND HOW MONITORING INPUT STATES ARE INSTANTIATED IS UP TO THE SUBCLASS

            Protocol for instantiating unassigned GatingProjections (i.e., w/o a sender specified):
               If sender is not specified for a GatingProjection (e.g., in a parameter specification tuple) 
                   it is flagged for deferred_init() in its __init__ method
               When the next ControlMechanism is instantiated, if its params[MAKE_DEFAULT_CONTROLLER] == True
                   its _take_over_as_default_controller method is called in _instantiate_attributes_after_function;
                   it then iterates through all of the parameterStates of all of the mechanisms in its system, 
                   identifies ones without a sender specified, calls its deferred_init() method,
                   instantiates a ControlSignal for it, and assigns it as the GatingProjection's sender.

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

        name : str : default GatingMechanism-<index>
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

    gatingProjections : List[GatingProjection]
        list of `GatingProjections <GatingProjection>` managed by the GatingMechanism.
        There is one for each ouputState in the `outputStates` dictionary.

    controlProjectionCosts : 2d np.array
        array of costs associated with each of the control signals in the `controlProjections` attribute.

    allocation_policy : 2d np.array
        array of values assigned to each control signal in the `controlProjections` attribute.
        This is the same as the ControlMechanism's `value <ControlMechanism.value>` attribute.


    """

    componentType = "GatingMechanism"

    initMethod = INIT__EXECUTE__METHOD_ONLY

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'GatingMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    # variableClassDefault = defaultControlAllocation
    # This must be a list, as there may be more than one (e.g., one per controlSignal)
    variableClassDefault = defaultControlAllocation

    from PsyNeuLink.Components.Functions.Function import Linear
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({GATING_PROJECTIONS: None})

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

        # self.system = None

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(monitor_for_control=monitor_for_control,
                                                  function=function,
                                                  params=params)

        super(GatingMechanism_Base, self).__init__(variable=default_input_value,
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

        super(GatingMechanism, self)._validate_params(request_set=request_set,
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
            raise GatingMechanismError("Attempt to assign GatingProjection {} to a mechanism ({}) that is not in {}".
                                              format(projection.name, receiver_mech.name, self.system.name))

    def _instantiate_monitored_output_states(self, context=None):
        raise GatingMechanismError("{0} (subclass of {1}) must implement _instantiate_monitored_output_states".
                                          format(self.__class__.__name__,
                                                 self.__class__.__bases__[0].__name__))

    def _instantiate_attributes_after_function(self, context=None):
        """Take over as default controller (if specified) and implement any specified GatingProjections

        """

        if MAKE_DEFAULT_CONTROLLER in self.paramsCurrent:
            if self.paramsCurrent[MAKE_DEFAULT_CONTROLLER]:
                self._take_over_as_default_controller(context=context)
            if not self.system.enable_controller:
                return

        # If GatingProjections were specified, implement them
        if GATING_PROJECTIONS in self.paramsCurrent:
            if self.paramsCurrent[GATING_PROJECTIONS]:
                for key, projection in self.paramsCurrent[GATING_PROJECTIONS].items():
                    self._instantiate_gating_projection(projection, context=self.name)

    def _take_over_as_default_controller(self, context=None):

        # Check the parameterStates of the system's mechanisms for any GatingProjections with deferred_init()
        for mech in self.system.mechanisms:
            for parameter_state in mech._parameter_states.values():
                for projection in parameter_state.receivesFromProjections:
                    # If projection was deferred for init, initialize it now and instantiate for self
                    if projection.value is DEFERRED_INITIALIZATION and projection.init_args['sender'] is None:
                        # Get params specified with projection for its ControlSignal (cached in control_signal attrib)
                        params = projection.control_signal
                        self._instantiate_gating_projection(projection, params=params, context=context)

    def _instantiate_gating_projection(self, projection, params=None, context=None):
        """Add outputState (as ControlSignal) and assign as sender to requesting GatingProjection

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

        from PsyNeuLink.Components.Projections.ModulatoryProjections.GatingProjection import GatingProjection
        if not isinstance(projection, GatingProjection):
            raise GatingMechanismError("PROGRAM ERROR: Attempt to assign {0}, "
                                              "that is not a GatingProjection, to outputState of {1}".
                                              format(projection, self.name))

        #  Update self.value by evaluating function
        self._update_value(context=context)

        # Instantiate new outputState and assign as sender of GatingProjection
        try:
            output_state_index = len(self.output_states)
        except AttributeError:
            output_state_index = 0
        from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.GatingMechanisms.ControlSignal import ControlSignal
        output_state_name = receiver.name + '_' + ControlSignal.__name__
        output_state_value = self.allocation_policy[output_state_index]
        from PsyNeuLink.Components.States.State import _instantiate_state
        from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.GatingMechanisms.ControlSignal import ControlSignal
        state = _instantiate_state(owner=self,
                                            state_type=ControlSignal,
                                            state_name=output_state_name,
                                            state_spec=defaultControlAllocation,
                                            state_params=params,
                                            constraint_value=output_state_value,
                                            constraint_value_name='Default control allocation',
                                            # constraint_output_state_index=output_item_output_state_index,
                                            context=context)

        # Assign outputState as GatingProjection's sender
        if projection.value is DEFERRED_INITIALIZATION:
            projection.init_args['sender']=state
            if projection.init_args['name'] is None:
                projection.init_args['name'] = GATING_PROJECTION + ' for ' + receiver.owner.name + ' ' + receiver.name
            projection._deferred_init()
        else:
            projection.sender = state

        # Update self.outputState and self.outputStates
        try:
            self.output_states[state.name] = state
        except AttributeError:
            self.output_states = OrderedDict({output_state_name:state})

        # Add index assignment to outputState
        state.index = output_state_index

        # Add GatingProjection to list of outputState's outgoing projections
        # (note: if it was deferred, it just added itself, skip)
        if not projection in state.sendsToProjections:
            state.sendsToProjections.append(projection)

        # Add GatingProjection to GatingMechanism's list of GatingProjections
        try:
            self.gatingProjections.append(projection)
        except AttributeError:
            self.gatingProjections = [projection]

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
        """Updates GatingProjections based on inputs

        Must be overriden by subclass
        """
        raise GatingMechanismError("{0} must implement execute() method".format(self.__class__.__name__))

    def show(self):

        print ("\n---------------------------------------------------------")

        print ("\n{0}".format(self.name))
        print("\n\tMonitoring the following mechanism outputStates:")
        for state_name, state in list(self.monitoring_mechanism.input_states.items()):
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
        state_names_sorted = sorted(self.output_states.keys())
        for state_name in state_names_sorted:
            for projection in self.output_states[state_name].sendsToProjections:
                print ("\t\t{0}: {1}".format(projection.receiver.owner.name, projection.receiver.name))

        print ("\n---------------------------------------------------------")
