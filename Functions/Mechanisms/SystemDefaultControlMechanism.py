#
# **************************************  SystemDefaultControlMechanism ************************************************
#

from collections import OrderedDict
from inspect import isclass

from Functions.ShellClasses import *
from Functions.Mechanisms.Mechanism import SystemDefaultMechanism_Base

# IMPLEMENT: EVC SPEC:
# INSTANTIATION:
# - inputStates: one for each performance/environment variable monitiored
# - evaluation function (as execute method) with one variable item (1D array) for each inputState
#      (??how should they be named/referenced:
#         maybe reverse instantation of variable and executeMethod, so that
#         execute method is parsed, and the necessary inputStates are created for it)
# - mapping projections from monitored states to inputStates
# - control signal projections established automatically by system implementation (using kwConrolSignal)
# - poll control signal projections for ranges to create matrix of search space

# EXECUTION:
# - call system.execute for each point in search space
# - compute evaluation function, and keep track of performance outcomes


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
    # This should be a list, as there may be more than one (e.g., one per controlSignal)
    variableClassDefault = [defaultControlAllocation]

    # paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    # paramClassDefaults.update({
    #     kwExecuteMethod:LinearMatrix,
    #     kwExecuteMethodParams:{LinearMatrix.kwMatrix: LinearMatrix.kwIdentityMatrix}
    # })

    def __init__(self,
                 # default_input_value=NotImplemented,
                 default_input_value=NotImplemented,
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented):
                 # context=NotImplemented):

        # Assign functionType to self.name as default;
        #  will be overridden with instance-indexed name in call to super
        if name is NotImplemented:
            self.name = self.functionType
        # Not needed:  handled by subclass
        # else:
        #     self.name = name

        self.functionName = self.functionType
        self.controlSignalChannels = OrderedDict()

# FIX: 5/31/16
        # self.inputState = NotImplemented
        # self.outputState = NotImplemented

        # # No prefs arg, so create prefs for SystemDefaultControlMechanism
        # if prefs is NotImplemented:
        #     prefs = SystemDefaultControlMechanismPreferenceSet

        # super(SystemDefaultMechanism_Base, self).__init__(variable=self.variableClassDefault,
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
        channel_name = projection.receiver.name + '_ControlSignal'
        input_name = channel_name + '_Input'
        output_name = channel_name + '_Output'

        # ----------------------------------------
        # Extend self's variable attribute to accommodate new ControlSignalChannel
        self.variable = np.append(self.variable, defaultControlAllocation)
        variable_item_index = self.variable.size-1
# FIX: ??COMMENT OUT THE FOLLOWING LINE:
        x = np.array(self.variable[variable_item_index])

        # ----------------------------------------
# FIX:  RESTRICT constraint_values TO CURRENT ITEM IN INPUT: 5/26/16
# FIX: constraint_values = self.variable[variable_item_index]
        # Instantiate inputState in self for ControlSignalChannel:
        from Functions.MechanismStates.MechanismInputState import MechanismInputState
        input_state = self.instantiate_mechanism_state(
                                        state_type=MechanismInputState,
                                        state_name=input_name,
                                        state_spec=defaultControlAllocation,
                                        # constraint_values=self.variable,
                                        constraint_values=np.array(self.variable[variable_item_index]),
                                        constraint_values_name='Default control allocation',
                                        constraint_index=variable_item_index,
                                        context=context)
        #  Update self's inputState and inputStates
        try:
            self.inputStates[input_name] = input_state
        except AttributeError:
            from collections import OrderedDict
            self.inputStates = OrderedDict({input_name:input_state})
            self.inputState = list(self.inputStates)[0]

        # ----------------------------------------
        #  Update self's value by evaluating executeMethod
        self.update_value()
        output_item_index = len(self.value)-1

        # ----------------------------------------
# FIX:  RESTRICT constraint_values TO CURRENT ITEM IN OUTPUT: 5/26/16
# FIX: constraint_values = self.executeMethodOutputDefault[output_item_index]
        # Instantiate outputState in self as sender of ControlSignal
        from Functions.MechanismStates.MechanismOutputState import MechanismOutputState
        self.sender = self.instantiate_mechanism_state(
                                    state_type=MechanismOutputState,
                                    state_name=output_name,
                                    # state_spec=[defaultControlAllocation],
                                    # constraint_values=[defaultControlAllocation],
                                    state_spec=defaultControlAllocation,
                                    constraint_values=self.value[output_item_index],
                                    constraint_values_name='Default control allocation',
                                    constraint_index=output_item_index,
                                    context=context)
        # Update sender's outputState and outputStates
        try:
            self.outputStates[output_name] = self.sender
        except AttributeError:
            from collections import OrderedDict
            self.outputStates = OrderedDict({output_name:self.sender})
            self.outputState = list(self.outputStates)[0]
        # ----------------------------------------
        # Put inputState, outputState and variable item in sender mechanism's ControlSignalChannels dict
        try:
            # Check that it has one
            self.controlSignalChannels[channel_name] = \
                                    ControlSignalChannel(
                                        inputState=input_state,
                                        variableIndex=variable_item_index,
                                        variableValue=self.variable[variable_item_index],
                                        outputState=self.sender,
                                        outputIndex=output_item_index,
                                        outputValue=self.value[output_item_index])
        # If it does not have one, initialize it
        except AttributeError:
            self.controlSignalChannels = OrderedDict({
                        output_name:ControlSignalChannel(
                                        inputState=input_state,
                                        variableIndex=variable_item_index,
                                        variableValue=self.variable[variable_item_index],
                                        outputState=self.sender,
                                        outputIndex=output_item_index,
                                        outputValue=self.value[output_item_index])})


