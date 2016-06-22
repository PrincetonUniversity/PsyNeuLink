#
# **************************************  SystemDefaultControlMechanism ************************************************
#

from collections import OrderedDict
from inspect import isclass

from Functions.ShellClasses import *
from Functions.Mechanisms.Mechanism import SystemDefaultMechanism_Base


ControlSignalChannel = namedtuple('ControlSignalChannel',
                                  'inputState, variableIndex, variableValue, outputState, outputIndex, outputValue')


class EVCError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class EVCMechanism(SystemDefaultMechanism_Base):
    """Implements default control mechanism (AKA EVC)

    Description:
        Implements default source of control signals, with one inputState and outputState for each.


NOTES:
    - MAKE SURE ALL PROJECTIONS TO EVC ARE FROM MECHANISMS IN THE SAME SYSTEM
    - USE THAT SYSTEM AS THE ONE TO EXECUTE
    - SYSTEM EXECUTE NEEDS TO BE SURE THAT IT DOESN'T CALL EVC FOR EXECUTION (WHICH WILL RESULT IN INFINITE RECURSION)

NEED TO:
   - MAKE receivesFromProjections @PROPERTY, THAT REFUSES ANY DIRECT ASSIGNMENTS
   - ADD METHOD TO MECHANISM FOR ADDING PROJECTIONS
   - DEFAULT: ADD PROJECTION TO (primary) inputState
   - TAKE ARG THAT SPECIFIES WHICH inputState TO USE, OR CREATE A NEW ONE
   - EVC SHOULD OVERRIDE THIS, AND *ALWAYS* CREATE A NEW ONE
   - USE CONTEXT TO DETERMINE WHETHER SETTER IS PERMITTED TO MAKE ASSIGNMENT (ONLY OWNER MECH IS ALLOWED)

INSTANTIATION:
- inputStates: one for each performance/environment variable monitored
- specification of system:  required param: kwSystem
- specification of inputStates:  required param: kwInputStates
    names must match names in list in kwMonitor in executeMethodParam

- specification of executeMethod: default is default allocation policy (BADGER/GUMBY)
    constraint:  if specified, number of items in variable must match number of inputStates in kwInputStates
                 and names in list in kwMonitor must match those in kwInputStates

EVALUATION:
- evaluation function (as execute method) with one variable item (1D array) for each inputState
     (??how should they be named/referenced:
        maybe reverse instantation of variable and executeMethod, so that
        execute method is parsed, and the necessary inputStates are created for it)
- mapping projections from monitored states to inputStates
- control signal projections established automatically by system implementation (using kwConrolSignal)
- poll control signal projections for ranges to create matrix of search space

EXECUTION:
- call system.execute for each point in search space
- compute evaluation function, and keep track of performance outcomes


    Class attributes:
        + functionType (str): System Default Mechanism
        + paramClassDefaults (dict):
            # + kwMechanismInputStateValue: [0]
            # + kwMechanismOutputStateValue: [1]
            + kwExecuteMethod: Linear
    """

    functionType = "EVCMechanism"

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

    def instantiate_monitor_input_state(self):

        input_name = channel_name + '_Input'

        # Extend self.variable to accommodate number of inputStates
        # Assign dedicated inputState to each controlSignal with value that matches defaultControlAllocation

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
                                        constraint_index=variable_item_index,
                                        context=context)
        #  Update inputState and inputStates
        try:
            self.inputStates[input_name] = input_state
        except AttributeError:
            self.inputStates = OrderedDict({input_name:input_state})
            self.inputState = list(self.inputStates)[0]

        # ----------------------------------------
        #  Update value by evaluating executeMethod
        self.update_value()
        output_item_index = len(self.value)-1


    def instantiate_control_signal_projection(self, projection, context=NotImplemented):
        """Add outputState and assign as sender to requesting controlSignal projection

        Assign corresponding outputState
        ?? Register controlSignal range and cost attributes in local attributes

        Args:
            projection:
            context:

        """

        from Functions.Projections.ControlSignal import ControlSignal
        if not isinstance(projection, ControlSignal):
            raise EVCError("Request to instantiate outputState from a non-ControlSignal projection ({0})".
                           format(projection.name))

        channel_name = projection.receiver.name + '_ControlSignal'
        output_name = channel_name + '_Output'

        # ----------------------------------------
        # Instantiate outputState as sender of ControlSignal
        from Functions.MechanismStates.MechanismOutputState import MechanismOutputState
        projection.sender = self.instantiate_mechanism_state(
                                    state_type=MechanismOutputState,
                                    state_name=output_name,
                                    state_spec=defaultControlAllocation,
                                    constraint_values=self.value[output_item_index],
                                    constraint_values_name='Default control allocation',
                                    constraint_index=output_item_index,
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

    def update(self, time_scale=TimeScale.TRIAL, runtime_params=NotImplemented, context=NotImplemented):
        """Search space of control signals for maximum EVC and set value of outputStates accordingly

         Determine search space for EVC by convolving ranges of controlSignal projections associated with outputStates
         Call System.execute for all combinations of controlSignal allocations:
         Call self.execute to compute EVC based on value of each inputState, for each combination of control signals
         Set outputStates[i].value to controlSignal allocation for max EVC
         Return max EVC

        Args:
            time_scale:
            runtime_params:
            context:

        Returns: (value) - maximum EVC for space of possible controlSignals
        """

        # Get controlSignal projections associated with each outputState


        # Construct search space for EVC by convolving ranges

        # Call System.execute for all combinations of controlSignal allocations:

        # Call self.execute to compute EVC based on value of each inputState, for each combination of control signals

        # Set outputStates[i].value to controlSignal allocation for max EVC

        for channel_name, channel in self.controlSignalChannels.items():
            channel.inputState.value = defaultControlAllocation


            # Note: self.execute is not implemented as a method;  it defaults to Lineaer
            #       from paramClassDefaults[kwExecuteMethod] from SystemDefaultMechanism
            channel.outputState.value = self.execute(channel.inputState.value)




    def execute(self, params, time_scale, context):
        """


        Args:
            params:
            time_scale:
            context:

        Returns:

        """
