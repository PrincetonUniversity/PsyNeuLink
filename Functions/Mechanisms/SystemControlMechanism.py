#
# **************************************  SystemControlMechanism ************************************************
#

# IMPLEMENTATION NOTE: COPIED FROM SystemDefaultMechanism;
#                      ADD IN GENERIC CONTROL STUFF FROM SystemDefaultControlMechanism

from collections import OrderedDict
from inspect import isclass

from Functions.ShellClasses import *
from Functions.Mechanisms.Mechanism import Mechanism_Base

class MonitoredStatesOption(AutoNumber):
    DEFAULT_ALLOCATION_POLICY = ()
    PRIMARY_OUTPUT_STATES = ()
    ALL_OUTPUT_STATES = ()
    NUM_MONITOR_STATES_OPTIONS = ()

class SystemControlMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class SystemControlMechanism_Base(Mechanism_Base):
    """Abstract class for control mechanism subclasses

    Description:
# DOCUMENTATION NEEDED:
    .instantiate_control_signal_projection INSTANTIATES OUTPUT STATE FOR EACH CONTROL SIGNAL ASSIGNED TO THE INSTANCE
    .update MUST BE OVERRIDDEN BY SUBCLASS
    WHETHER AND HOW MONITORING INPUT STATES ARE INSTANTIATED IS UP TO THE SUBCLASS

# PROTOCOL FOR ASSIGNING DefaultController (defined in Functions.__init__.py)
#    Initial assignment is to SystemDefaultCcontroller (instantiated and assigned in Functions.__init__.py)
#    When any other SystemControlMechanism is instantiated, if its params[kwMakeDefaultController] == True
#        then its take_over_as_default_controller method is called in instantiate_attributes_after_execute_method()
#        which moves all ControlSignal Projections from SystemDefaultController to itself, and deletes them there
# params[kwMontioredStates]: Determines which states will be monitored.
#        can be a list of Mechanisms, MechanismOutputStates, a MonitoredStatesOption, or a combination
#        if MonitoredStates appears alone, it will be used to determine how states are assigned from system.graph by default
#        TBI: if it appears in a tuple with a Mechanism, or in the Mechamism's params list, it applied to just that mechanism


    Class attributes:
        + functionType (str): System Default Mechanism
        + paramClassDefaults (dict):
            # + kwMechanismInputStateValue: [0]
            # + kwMechanismOutputStateValue: [1]
            + kwExecuteMethod: Linear
            + kwExecuteMethodParams:{kwSlope:1, kwIntercept:0}
    """

    functionType = "SystemControlMechanism"

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'SystemDefaultMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    # variableClassDefault = defaultControlAllocation
    # This must be a list, as there may be more than one (e.g., one per controlSignal)
    variableClassDefault = [defaultControlAllocation]

    from Functions.Utility import Linear
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        kwExecuteMethod:Linear,
        kwExecuteMethodParams:{Linear.kwSlope:1, Linear.kwIntercept:0},
        kwControlSignalProjections: None
    })

    def __init__(self,
                 default_input_value=NotImplemented,
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented,
                 context=NotImplemented):
        """Abstract class for system control mechanisms

        :param default_input_value: (value)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        """

        # Assign functionType to self.name as default;
        #  will be overridden with instance-indexed name in call to super
        if name is NotImplemented:
            self.name = self.functionType

        self.functionName = self.functionType
        # self.controlSignalChannels = OrderedDict()
        self.system = None

        super(SystemControlMechanism_Base, self).__init__(variable=default_input_value,
                                                          params=params,
                                                          name=name,
                                                          prefs=prefs,
                                                          context=self)

    def instantiate_attributes_after_execute_method(self, context=NotImplemented):

        try:
            # If specified as defaultController, reassign ControlSignal projections from SystemDefaultController
            if self.paramsCurrent[kwMakeDefaultController]:
                self.take_over_as_default_controller()
        except KeyError:
            pass

        # If controlSignal projections were specified, implement them
        try:
            if self.paramsCurrent[kwControlSignalProjections]:
                for key, projection in self.paramsCurrent[kwControlSignalProjections].items():
                    self.instantiate_control_signal_projection(projection, context=self.name)
        except:
            pass


    def take_over_as_default_controller(self, context=NotImplemented):

        from Functions import SystemDefaultController
        # Iterate through old controller's outputStates
        for outputState in SystemDefaultController.outputStates:

            # Iterate through projections sent for outputState
            for projection in SystemDefaultController.outputStates[outputState].sendsToProjections:

                # Move ControlSignal projection to self (by creating new outputState)
                # IMPLEMENTATION NOTE: Method 1 — Move old ControlSignal Projection to self
                #    Easier to implement
                #    - call instantiate_control_signal_projection directly here (which takes projection as arg)
                #        instead of instantiating a new ControlSignal Projection (more efficient, keeps any settings);
                #    - however, this bypasses call to Projection.instantiate_sender()
                #        which calls Mechanism.sendsToProjections.append(), so need to do that here
                #    - this is OK, as it is case of a Mechanism managing its *own* projections list (vs. "outsider")
                new_output_state = self.instantiate_control_signal_projection(projection, context=context)
                new_output_state.sendsToProjections.append(projection)

                # # IMPLMENTATION NOTE: Method 2 (Cleaner) Instantiate new ControlSignal Projection
                # #    Cleaner, but less efficient and ?? may lose original params/settings for ControlSignal
                # # TBI: Implement and then use Mechanism.add_project_from_mechanism()
                # self.add_projection_from_mechanism(projection, new_output_state, context=context)

                # Remove corresponding projection from old controller
                SystemDefaultController.outputStates[outputState].sendsToProjections.remove(projection)

            # Current controller's outputState has no projections left (after removal(s) above)
            if not SystemDefaultController.outputStates[outputState].sendsToProjections:
                # If this is the old controller's primary outputState, set it to None
                if SystemDefaultController.outputState is SystemDefaultController.outputStates[outputState]:
                    SystemDefaultController.outputState = None
                # Delete outputState from old controller's outputState dict
                del SystemDefaultController.outputStates[outputState]

    def instantiate_control_signal_projection(self, projection, context=NotImplemented):
        """Add outputState and assign as sender to requesting controlSignal projection

        Args:
            projection:
            context:

        Returns state: (MechanismOutputState)

        """

        output_name = projection.receiver.name + '_ControlSignal' + '_Output'

        #  Update value by evaluating executeMethod
        self.update_value()
        output_item_index = len(self.value)-1

        # Instantiate outputState for self as sender of ControlSignal
        from Functions.MechanismStates.MechanismOutputState import MechanismOutputState
        state = self.instantiate_mechanism_state(
                                    state_type=MechanismOutputState,
                                    state_name=output_name,
                                    state_spec=defaultControlAllocation,
                                    constraint_values=self.value[output_item_index],
                                    constraint_values_name='Default control allocation',
                                    # constraint_index=output_item_index,
                                    context=context)

        projection.sender = state

        # Update self.outputState and self.outputStates
        try:
            self.outputStates[output_name] = state
# FIX:  ASSIGN outputState to ouptustates[0]
        except AttributeError:
            self.outputStates = OrderedDict({output_name:state})
            # self.outputState = list(self.outputStates)[0]
            # self.outputState = list(self.outputStates.items())[0]
            self.outputState = self.outputStates[output_name]

        return state

    def update(self, time_scale=TimeScale.TRIAL, runtime_params=NotImplemented, context=NotImplemented):
        """Updates controlSignals based on inputs

        Must be overriden by subclass
        """
        raise SystemControlMechanismError("{0} must implement update() method".format(self.__class__.__name__))