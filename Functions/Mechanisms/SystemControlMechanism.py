#
# **************************************  SystemControlMechanism ************************************************
#

# IMPLEMENTATION NOTE: COPIED FROM SystemDefaultMechanism;
#                      ADD IN GENERIC CONTROL STUFF FROM SystemDefaultControlMechanism

from collections import OrderedDict
from inspect import isclass

from Functions.ShellClasses import *
from Functions.Mechanisms.Mechanism import Mechanism_Base


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

        super(SystemControlMechanism_Base, self).__init__(variable=default_input_value,
                                                          params=params,
                                                          name=name,
                                                          prefs=prefs,
                                                          context=self)

    def instantiate_control_signal_projection(self, projection, context=NotImplemented):
        """Add outputState and assign as sender to requesting controlSignal projection

        Args:
            projection:
            context:

        """

        output_name = projection.receiver.name + '_ControlSignal' + '_Output'

        #  Update value by evaluating executeMethod
        self.update_value()
        output_item_index = len(self.value)-1

        # Instantiate outputState as sender of ControlSignal
        from Functions.MechanismStates.MechanismOutputState import MechanismOutputState
        projection.sender = self.instantiate_mechanism_state(
                                    state_type=MechanismOutputState,
                                    state_name=output_name,
                                    state_spec=defaultControlAllocation,
                                    constraint_values=self.value[output_item_index],
                                    constraint_values_name='Default control allocation',
                                    # constraint_index=output_item_index,
                                    context=context)

        # Update outputState and outputStates
        try:
            self.outputStates[output_name] = projection.sender
        except AttributeError:
            self.outputStates = OrderedDict({output_name:projection.sender})
            self.outputState = list(self.outputStates)[0]

    def update(self, time_scale=TimeScale.TRIAL, runtime_params=NotImplemented, context=NotImplemented):
        """Updates controlSignals based on inputs

        Must be overriden by subclass
        """
        raise SystemControlMechanismError("{0} must implement update() method".format(self.__class__.__name__))