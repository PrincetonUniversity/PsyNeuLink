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

ControlMechanisms monitor the `outputState(s) <OutputState>` of `ProcessingMechanisms <ProcessingMechanism>` in a
`System`, to assess the outcome of processing of those mechanisms.  They use this information to regulate the value of
parameters of those or other mechanisms (or their functions) in the system.  This is done by way of
`ControlProjections <ControlProjection>` from the ControlMechanism to the `ParameterStates <ParameterState>` for the
parameter(s) to be controlled.  A ControlMechanism can regulate only the parameters of mechanism in the system for
which it is the `controller <System_Execution_Control>`.

.. _ControlMechanism_Creation:

Creating A ControlMechanism
---------------------------

ControlMechanisms can be created by using the standard Python method of calling the constructor for the desired type.
A ControlMechanism is also created automatically whenever a `system is created <System_Creation>`, and assigned as
the `controller <System_Execution_Control>` for that system. The `outputStates <OutputState>` to be monitored by a
ControlMechanism are specified in its `monitored_output_states` argument, which can take  a number of
`forms <ObjectiveMechanism_Monitored_OutputStates>`.  When the ControlMechanism is created, it automatically creates
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
   ControlMechanism has executed, a parameter that it controls will not assume its new value until the corresponding
   receiver mechanism has executed.

.. _ControlMechanism_Class_Reference:

Class Reference
---------------

"""

# IMPLEMENTATION NOTE: COPIED FROM DefaultProcessingMechanism;
#                      ADD IN GENERIC CONTROL STUFF FROM DefaultControlMechanism

from collections import OrderedDict

from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism_Base
from PsyNeuLink.Components.ShellClasses import *

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

            Protocol for assigning DefaultController:
               Initial assignment is to SystemDefaultController (instantiated and assigned in Components.__init__.py)
               When any other ControlMechanism is instantiated, if its params[MAKE_DEFAULT_CONTROLLER] == True
                   then its _take_over_as_default_controller method is called in _instantiate_attributes_after_function()
                   which moves all ControlProjections from DefaultController to itself, and deletes them there

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
                 monitor_for_control:tc.optional(list)=None,
                 function = Linear(slope=1, intercept=0),
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        self.system = None

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

        If SYSTEM is not specified:
        - OK if controller is DefaultControlMechanism
        - otherwise, raise an exception
        Check that all items in MONITOR_FOR_CONTROL are Mechanisms or OutputStates for Mechanisms in self.system
        Check that len(WEIGHTS) = len(MONITOR_FOR_CONTROL)
        """

        # DefaultController does not require a system specification
        #    (it simply passes the defaultControlAllocation for default ConrolSignal Projections)
        from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.DefaultControlMechanism import DefaultControlMechanism
        if isinstance(self,DefaultControlMechanism):
            pass

        # For all other ControlMechanisms, validate System specification
        else:
            try:
                if not isinstance(request_set[SYSTEM], System):
                    raise KeyError
            except KeyError:
                # Validation called by assign_params() for user-specified param set, so SYSTEM need not be included
                if COMMAND_LINE in context:
                    pass
                else:
                    raise ControlMechanismError("A system must be specified in the SYSTEM param to instantiate {0}".
                                                format(self.name))
            else:
                self.paramClassDefaults[SYSTEM] = request_set[SYSTEM]

        super(ControlMechanism_Base, self)._validate_params(request_set=request_set,
                                                                 target_set=target_set,
                                                                 context=context)

    def _validate_projection(self, projection, context=None):
        """Insure that projection is to mechanism within the same system as self
        """

        receiver_mech = projection.receiver.owner
        if not receiver_mech in self.system.mechanisms:
            raise ControlMechanismError("Attempt to assign ControlProjection {} to a mechanism ({}) that is not in {}".
                                              format(projection.name, receiver_mech.name, self.system.name))

    def _instantiate_attributes_before_function(self, context=None):
        """Instantiate self.system attribute

        Assign self.system
        """
        self.system = self.paramsCurrent[SYSTEM]
        super()._instantiate_attributes_before_function(context=context)

    def _instantiate_monitored_output_states(self, context=None):
        raise ControlMechanismError("{0} (subclass of {1}) must implement _instantiate_monitored_output_states".
                                          format(self.__class__.__name__,
                                                 self.__class__.__bases__[0].__name__))

    def _instantiate_attributes_after_function(self, context=None):
        """Take over as default controller (if specified) and implement any specified ControlProjections

        """

        try:
            # If specified as DefaultController, reassign ControlProjections from DefaultController
            if self.paramsCurrent[MAKE_DEFAULT_CONTROLLER]:
                self._take_over_as_default_controller(context=context)
        except KeyError:
            pass

        # If ControlProjections were specified, implement them
        try:
            if self.paramsCurrent[CONTROL_PROJECTIONS]:
                for key, projection in self.paramsCurrent[CONTROL_PROJECTIONS].items():
                    self._instantiate_control_projection(projection, context=self.name)
        except:
            pass

    def _take_over_as_default_controller(self, context=None):

        from PsyNeuLink.Components import DefaultController

        # Iterate through old controller's outputStates
        to_be_deleted_outputStates = []
        for outputState in DefaultController.outputStates:

            # Iterate through projections sent for outputState
            for projection in DefaultController.outputStates[outputState].sendsToProjections:

                # Move ControlProjection to self (by creating new outputState)
                # IMPLEMENTATION NOTE: Method 1 -- Move old ControlProjection to self
                #    Easier to implement
                #    - call _instantiate_control_projection directly here (which takes projection as arg)
                #        instead of instantiating a new ControlProjection (more efficient, keeps any settings);
                #    - however, this bypasses call to Projection._instantiate_sender()
                #        which calls Mechanism.sendsToProjections.append(),
                #        so need to do that in _instantiate_control_projection
                #    - this is OK, as it is case of a Mechanism managing its *own* projections list (vs. "outsider")
                params = projection.control_signal
                self._instantiate_control_projection(projection, params=params, context=context)

                # # IMPLEMENTATION NOTE: Method 2 - Instantiate new ControlProjection
                # #    Cleaner, but less efficient and ?? may lose original params/settings for ControlProjection
                # # TBI: Implement and then use Mechanism.add_project_from_mechanism()
                # self._add_projection_from_mechanism(projection, new_output_state, context=context)

                # Remove corresponding projection from old controller
                DefaultController.outputStates[outputState].sendsToProjections.remove(projection)

            # Current controller's outputState has no projections left (after removal(s) above)
            if not DefaultController.outputStates[outputState].sendsToProjections:
                # If this is the old controller's primary outputState, set it to None
                if DefaultController.outputState is DefaultController.outputStates[outputState]:
                    DefaultController.outputState = None
                # Delete outputState from old controller's outputState dict
                to_be_deleted_outputStates.append(DefaultController.outputStates[outputState])
        for item in to_be_deleted_outputStates:
            del DefaultController.outputStates[item.name]

    def _instantiate_control_projection(self, projection, params=None, context=None):
        """Add outputState and assign as sender to requesting ControlProjection

        # Updates allocation_policy and controlSignalCosts attributes to accommodate instantiated projection

        Note:  params are expected to be params for controlSignal (outputState of ControlMechanism)

        Assume that:
            # - self.value is populated (in _update_value) with an array of allocations from self.allocation_policy;
            - self.allocation_policy has already been extended to include the particular (indexed) allocation
                to be used for the outputState being created here.

        INCREMENT BASED ON TOTAL NUMBER OF OUTPUTSTATES SO FAR

        Returns state: (OutputState)
        """

        self._validate_projection(projection)

        from PsyNeuLink.Components.Projections.ControlProjection import ControlProjection
        if not isinstance(projection, ControlProjection):
            raise ControlMechanismError("PROGRAM ERROR: Attempt to assign {0}, "
                                              "that is not a ControlProjection, to outputState of {1}".
                                              format(projection, self.name))


        #  Update self.value by evaluating function
        self._update_value(context=context)

        # Instantiate new outputState and assign as sender of ControlProjection
        try:
            output_state_index = len(self.outputStates)
        except AttributeError:
            output_state_index = 0
        output_state_name = projection.receiver.name + '_ControlSignal'
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

        # Add index assignment to outputState
        state.index = output_state_index

        # Assign outputState as ControlProjection's sender
        projection.sender = state

        # Update self.outputState and self.outputStates
        try:
            self.outputStates[state.name] = state
        except AttributeError:
            self.outputStates = OrderedDict({output_state_name:state})
            self.outputState = self.outputStates[output_state_name]

        # Add ControlProjection to list of outputState's outgoing projections
        state.sendsToProjections.append(projection)

        # Add ControlProjection to ControlMechanism's list of ControlProjections
        try:
            self.controlProjections.append(projection)
        except AttributeError:
            self.controlProjections = []

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
        for state_name, state in list(self.monitoring_mechanism.inputStates.items()):
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
        state_names_sorted = sorted(self.outputStates.keys())
        for state_name in state_names_sorted:
            for projection in self.outputStates[state_name].sendsToProjections:
                print ("\t\t{0}: {1}".format(projection.receiver.owner.name, projection.receiver.name))

        print ("\n---------------------------------------------------------")
