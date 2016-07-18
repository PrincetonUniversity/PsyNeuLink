# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# **************************************  SystemDefaultControlMechanism ************************************************
#

from collections import OrderedDict
from inspect import isclass

from Functions.ShellClasses import *
from Functions.Mechanisms.SystemControlMechanism import SystemControlMechanism_Base


ControlSignalChannel = namedtuple('ControlSignalChannel',
                                  'inputState, variableIndex, variableValue, outputState, outputIndex, outputValue')


class SystemDefaultControlMechanism(SystemControlMechanism_Base):
    """Implement default control mechanism

    Description:
        Implement default source of control signals, with one inputState and outputState for each
        Use defaultControlAllocation as input(s) and pass value(s) unchanged to ouputState(s) and controlSignal(s)


# DOCUMENTATION NEEDED
    - EXPLAIN WHAT ControlSignalChannel IS
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

    paramClassDefaults = SystemControlMechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({kwSystem: None
                               # kwExecuteMethod:LinearMatrix,
                               # kwExecuteMethodParams:{LinearMatrix.kwMatrix: LinearMatrix.kwIdentityMatrix}
    })

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

        super(SystemDefaultControlMechanism, self).__init__(default_input_value =default_input_value,
                                                         params=params,
                                                         name=name,
                                                         prefs=prefs,
                                                         context=self)

    def update(self, time_scale=TimeScale.TRIAL, runtime_params=NotImplemented, context=NotImplemented):

        # super(SystemDefaultControlMechanism, self).update(time_scale=time_scale,
        #                                                   runtime_params=runtime_params,
        #                                                   context=context)
        for channel_name, channel in self.controlSignalChannels.items():

            channel.inputState.value = defaultControlAllocation

            # Note: self.execute is not implemented as a method;  it defaults to Lineaer
            #       from paramClassDefaults[kwExecuteMethod] (see above)
            channel.outputState.value = self.execute(channel.inputState.value, context=context)

    def instantiate_monitored_output_states(self, context=NotImplemented):
        """Suppress instantiation of default inputState

        """
# FIX: NEED TO SUPPRESS ASSIGNEMENT RATHER THAN RETURN NONE
        return None


    def instantiate_control_signal_projection(self, projection, context=NotImplemented):
        # DOCUMENTATION NEEDED:  EXPLAIN WHAT CONTROL SIGNAL CHANNELS ARE
        """

        Args:
            projection:
            context:

        Returns:

        """

        # Instantiate inputStates and "channels" for controlSignal allocations
        self.instantiate_control_signal_channel(projection=projection, context=context)

        # Call super to instantiate outputStates
        super(SystemDefaultControlMechanism, self).instantiate_control_signal_projection(projection=projection,
                                                                                         context=context)

    def instantiate_control_signal_channel(self, projection, context=NotImplemented):
        """
        DOCUMENTATION:
            As SystemDefaultController, also add corresponding inputState and ControlSignalChannel:
            Extend self.variable by one item to accommodate new "channel"
            Assign dedicated inputState to controlSignal with value set to defaultControlAllocation
            Assign corresponding outputState

        Args:
            projection:
            context:

        Returns:

        """
        channel_name = projection.receiver.name + '_ControlSignal'
        input_name = channel_name + '_Input'

        # Extend self.variable to accommodate new ControlSignalChannel
        self.variable = np.append(self.variable, defaultControlAllocation)
# FIX: GET RID OF THIS IF contraint_values IS CORRECTED BELOW
        variable_item_index = self.variable.size-1

        # Instantiate inputState for ControlSignalChannel:
        from Functions.MechanismStates.MechanismInputState import MechanismInputState
        input_state = self.instantiate_mechanism_state(
                                        state_type=MechanismInputState,
                                        state_name=input_name,
                                        state_spec=defaultControlAllocation,
                                        constraint_values=np.array(self.variable[variable_item_index]),
                                        constraint_values_name='Default control allocation',
                                        context=context)
        #  Update inputState and inputStates
        try:
            self.inputStates[input_name] = input_state
        # No inputState(s) yet, so create them
        except AttributeError:
            self.inputStates = OrderedDict({input_name:input_state})
            self.inputState = list(self.inputStates)[0]

