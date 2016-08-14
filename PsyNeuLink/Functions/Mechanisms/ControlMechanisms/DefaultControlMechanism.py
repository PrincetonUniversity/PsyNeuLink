# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# **************************************  DefaultControlMechanism ************************************************
#

from collections import OrderedDict

from PsyNeuLink.Functions.Mechanisms.ControlMechanisms.ControlMechanism import ControlMechanism_Base
from PsyNeuLink.Functions.ShellClasses import *


ControlSignalChannel = namedtuple('ControlSignalChannel',
                                  'inputState, variableIndex, variableValue, outputState, outputIndex, outputValue')


class DefaultControlMechanism(ControlMechanism_Base):
    """Implement default control mechanism

    Description:
        Implement default source of control signals, with one inputState and outputState for each
        Use defaultControlAllocation as input(s) and pass value(s) unchanged to ouputState(s) and controlSignal(s)


# DOCUMENTATION NEEDED
    - EXPLAIN WHAT ControlSignalChannel IS:
            A ControlSignalChannel is instantiated for each ControlSignal projection assigned to DefaultController
        It simply passes the defaultControlAllocation value to the ControlSignal projection


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
            # + kwInputStateValue: [0]
            # + kwOutputStateValue: [1]
            + kwExecuteMethod: Linear
    """

    functionType = "DefaultControlMechanism"

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # classPreferenceLevel = PreferenceLevel.TYPE

    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to Type automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'DefaultControlMechanismCustomClassPreferences',
    #     kp<pref>: <setting>...}


    # variableClassDefault = defaultControlAllocation
    # This must be a list, as there may be more than one (e.g., one per controlSignal)
    variableClassDefault = [defaultControlAllocation]

    paramClassDefaults = ControlMechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({kwSystem: None,
                               # # Assigns DefaultControlMechanism, when instantiated, as the DefaultController
                               # kwMakeDefaultController:True
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

        super(DefaultControlMechanism, self).__init__(default_input_value =default_input_value,
                                                         params=params,
                                                         name=name,
                                                         prefs=prefs,
                                                         context=self)

    def update(self, time_scale=TimeScale.TRIAL, runtime_params=NotImplemented, context=NotImplemented):

        # super(DefaultControlMechanism, self).update(time_scale=time_scale,
        #                                                   runtime_params=runtime_params,
        #                                                   context=context)
        for channel_name, channel in self.controlSignalChannels.items():

            channel.inputState.value = defaultControlAllocation

            # Note: self.execute is not implemented as a method;  it defaults to Lineaer
            #       from paramClassDefaults[kwExecuteMethod] (see above)
            channel.outputState.value = self.execute(channel.inputState.value, context=context)

    def instantiate_input_states(self, context=NotImplemented):
        """Suppress assignement of inputState(s) - this is done by instantiate_control_signal_channel
        """
        # IMPLEMENTATION NOTE:  Assigning to None currently causes problems, so just pass
        # self.inputState = None
        # self.inputStates = None
        pass

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
        super().instantiate_control_signal_projection(projection=projection,
                                                      context=context)

    def instantiate_control_signal_channel(self, projection, context=NotImplemented):
        """Instantiate inputState that passes defaultControlAllocation to ControlSignal projection

        Instantiate an inputState with defaultControlAllocation as its value

        Args:
            projection:
            context:

        Returns:

        """
        input_name = 'DefaultControlAllocation for ' + projection.receiver.name + '_ControlSignal'

        self.instantiate_control_mechanism_input_state(input_name, defaultControlAllocation, context=context)
