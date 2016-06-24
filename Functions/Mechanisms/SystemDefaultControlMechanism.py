#
# **************************************  SystemDefaultControlMechanism ************************************************
#

from collections import OrderedDict
from inspect import isclass

from Functions.ShellClasses import *
from Functions.Mechanisms.Mechanism import SystemDefaultMechanism_Base


ControlSignalChannel = namedtuple('ControlSignalChannel',
                                  'inputState, variableIndex, variableValue, outputState, outputIndex, outputValue')


class SystemDefaultControlMechanism(SystemDefaultMechanism_Base):
    """Implements default control mechanism (AKA EVC)

    Description:
        Implements default source of control signals, with one inputState and outputState for each.

# IMPLEMENTATION NOTE:
    - EVERY DEFAULT CONTROL PROJECTION SHOULD ASSIGN THIS MECHANISM AS ITS SENDER
    - AN OUTPUT STATE SHOULD BE CREATED FOR EACH OF THOSE SENDERS
    - AN INPUT STATE SHOULD BE CREATED FOR EACH OUTPUTSTATE
    - THE EXECUTE METHOD SHOULD SIMPLY MAP THE INPUT STATE TO THE OUTPUT STATE
    - EVC CAN THEN BE A SUBCLASS THAT OVERRIDES EXECUTE METHOD AND DOES SOMETHING MORE SOPHISTICATED
        (E.G,. KEEPS TRACK OF IT'S SENDER PROJECTIONS AND THEIR COSTS, ETC.)
    * MAY NEED TO AUGMENT OUTPUT STATES TO KNOW ABOUT THEIR SENDERS
    * MAY NEED TO ADD NEW CONSTRAINT ON ASSIGNING A STATE AS A SENDER:  IT HAS TO BE AN OUTPUTSTATE


    Class attributes:
        + functionType (str): System Default Mechanism
        + paramClassDefaults (dict):
            # + kwMechanismInputStateValue: [0]
            # + kwMechanismOutputStateValue: [1]
            + kwExecuteMethod: Linear
    """

    functionType = "SystemDefaultControlMechanism"

    classPreferenceLevel = PreferenceLevel.TYPE

    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to Type automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'SystemDefaultControlMechanismCustomClassPreferences',
    #     kp<pref>: <setting>...}


    # variableClassDefault = defaultControlAllocation
    # This must be a list, as there may be more than one (e.g., one per controlSignal)
    variableClassDefault = [defaultControlAllocation]

    # paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    # paramClassDefaults.update({
    #     kwExecuteMethod:LinearMatrix,
    #     kwExecuteMethodParams:{LinearMatrix.kwMatrix: LinearMatrix.kwIdentityMatrix}
    # })

    def __init__(self,
                 default_input_value=NotImplemented,
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented):
                 # context=NotImplemented):

        # Assign functionType to self.name as default;
        #  will be overridden with instance-indexed name in call to super
        if name is NotImplemented:
            self.name = self.functionType

        self.functionName = self.functionType
        self.controlSignalChannels = OrderedDict()

        super(SystemDefaultMechanism_Base, self).__init__(variable=default_input_value,
                                                          params=params,
                                                          name=name,
                                                          prefs=prefs,
                                                          context=self)

    def update(self, time_scale=TimeScale.TRIAL, runtime_params=NotImplemented, context=NotImplemented):
        """
# DOCUMENTATION NEEDED HERE
        :return:
        """

        for channel_name, channel in self.controlSignalChannels.items():

            channel.inputState.value = defaultControlAllocation

            # IMPLEMENTATION NOTE:  ADD EVC HERE
            # Currently, just maps input to output for each controlChannel

            # Note: self.execute is not implemented as a method;  it defaults to Lineaer
            #       from paramClassDefaults[kwExecuteMethod] from SystemDefaultMechanism
            channel.outputState.value = self.execute(channel.inputState.value)

    def instantiate_control_signal_channels(self, projection, context=NotImplemented):
        """Add control channels for each control signal assigned to SystemDefaultControlMechanism

        Extend self.variable by one item to accommodate new "channel"
        Assign dedicated inputState to each controlSignal with value that matches defaultControlAllocation
        Assign corresponding outputState

        Args:
            projection:
            context:

        """

        channel_name = projection.receiver.name + '_ControlSignal'
        input_name = channel_name + '_Input'
        output_name = channel_name + '_Output'

        # ----------------------------------------
        # Extend self.variable to accommodate new ControlSignalChannel
        self.variable = np.append(self.variable, defaultControlAllocation)
        variable_item_index = self.variable.size-1

        # ----------------------------------------
        # Instantiate inputState for ControlSignalChannel:
        from Functions.MechanismStates.MechanismInputState import MechanismInputState
        input_state = self.instantiate_mechanism_state(
                                        state_type=MechanismInputState,
                                        state_name=input_name,
                                        state_spec=defaultControlAllocation,
                                        constraint_values=np.array(self.variable[variable_item_index]),
                                        constraint_values_name='Default control allocation',
                                        # constraint_index=variable_item_index,
                                        context=context)
        #  Update inputState and inputStates
        try:
            self.inputStates[input_name] = input_state
        # No inputState(s) yet, so create them
        except AttributeError:
            self.inputStates = OrderedDict({input_name:input_state})
            self.inputState = list(self.inputStates)[0]

        # ----------------------------------------
        #  Update value by evaluating executeMethod
        self.update_value()
        output_item_index = len(self.value)-1

        # ----------------------------------------
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
        # ----------------------------------------
        # Put inputState, outputState and variable item in ControlSignalChannels dict
        try:
            # Check that it has one
            self.controlSignalChannels[channel_name] = \
                                    ControlSignalChannel(
                                        inputState=input_state,
                                        variableIndex=variable_item_index,
                                        variableValue=self.variable[variable_item_index],
                                        outputState=projection.sender,
                                        outputIndex=output_item_index,
                                        outputValue=self.value[output_item_index])
        # If it does not exisit, initialize it
        except AttributeError:
            self.controlSignalChannels = OrderedDict({
                        output_name:ControlSignalChannel(
                                        inputState=input_state,
                                        variableIndex=variable_item_index,
                                        variableValue=self.variable[variable_item_index],
                                        outputState=projection.sender,
                                        outputIndex=output_item_index,
                                        outputValue=self.value[output_item_index])})

