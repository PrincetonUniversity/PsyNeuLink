# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  DefaultControlMechanism ************************************************

"""
**[DOCUMENTATION STILL UNDER CONSTRUCTION]**

"""

from collections import OrderedDict

from PsyNeuLink.Components.Mechanisms.ControlMechanisms.ControlMechanism import ControlMechanism_Base
from PsyNeuLink.Components.ShellClasses import *


# ControlSignalChannel = namedtuple('ControlSignalChannel',
#                                   'inputState, variableIndex, variableValue, outputState, outputIndex, outputValue')


class DefaultControlMechanism(ControlMechanism_Base):
    """Implement default control mechanism

    Description:
        Implement default source of control signals, with one inputState and outputState for each
        Use defaultControlAllocation as input(s) and pass value(s) unchanged to ouputState(s) and ControlProjection(s)


# DOCUMENTATION NEEDED
    - EXPLAIN WHAT ControlSignalChannel IS:
            A ControlSignalChannel is instantiated for each ControlProjection assigned to DefaultController
        It simply passes the defaultControlAllocation value to the ControlProjection


    - EVERY DEFAULT CONTROL PROJECTION SHOULD ASSIGN THIS MECHANISM AS ITS SENDER
    - AN OUTPUT STATE SHOULD BE CREATED FOR EACH OF THOSE SENDERS
    - AN INPUT STATE SHOULD BE CREATED FOR EACH OUTPUTSTATE
    - THE EXECUTE METHOD SHOULD SIMPLY MAP THE INPUT STATE TO THE OUTPUT STATE
    - EVC CAN THEN BE A SUBCLASS THAT OVERRIDES EXECUTE METHOD AND DOES SOMETHING MORE SOPHISTICATED
        (E.G,. KEEPS TRACK OF IT'S SENDER PROJECTIONS AND THEIR COSTS, ETC.)
    * MAY NEED TO AUGMENT OUTPUT STATES TO KNOW ABOUT THEIR SENDERS
    * MAY NEED TO ADD NEW CONSTRAINT ON ASSIGNING A STATE AS A SENDER:  IT HAS TO BE AN OUTPUTSTATE


    Class attributes:
        + componentType (str): System Default Mechanism
        + paramClassDefaults (dict):
            # + kwInputStateValue: [0]
            # + kwOutputStateValue: [1]
            + FUNCTION: Linear
    """

    componentType = "DefaultControlMechanism"

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # classPreferenceLevel = PreferenceLevel.TYPE

    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to Type automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'DefaultControlMechanismCustomClassPreferences',
    #     kp<pref>: <setting>...}


    # variableClassDefault = defaultControlAllocation
    # This must be a list, as there may be more than one (e.g., one per controlSignal)
    variableClassDefault = defaultControlAllocation

    from PsyNeuLink.Components.Functions.Function import Linear
    paramClassDefaults = ControlMechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({SYSTEM: None,
                               # MAKE_DEFAULT_CONTROLLER:True  <- No need, it is the default by default
                               FUNCTION:Linear,
                               FUNCTION_PARAMS:{SLOPE:1, INTERCEPT:0},
                               MONITOR_FOR_CONTROL:None
                               })

    from PsyNeuLink.Components.Functions.Function import Linear
    @tc.typecheck
    def __init__(self,
                 default_input_value=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None):


        super(DefaultControlMechanism, self).__init__(default_input_value =default_input_value,
                                                         params=params,
                                                         name=name,
                                                         prefs=prefs,
                                                         context=self)



    def __execute__(self, variable=None, runtime_params=None, time_scale=TimeScale.TRIAL, context=None):

        # FIX: REVISE TO NOT USE CHANNELS;  JUST RETURN INPUTSTATE VALUES, AS PASS-THROUGH THAT WILL BE ASSIGNED TO
        # MODIFIED 12/9/16 OLD:
        # # OUTPUT STATES
        # for channel_name, channel in self.controlSignalChannels.items():
        #
        #     channel.inputState.value = defaultControlAllocation
        #
        #     # Note: self.execute is not implemented as a method;  it defaults to Linear
        #     #       from paramClassDefaults[FUNCTION] (see above)
        #     channel.outputState.value = self.function(channel.inputState.value, context=context)
        #
        # # # FIX: CONSTRUCT np.array OF outputState.values
        # # return output
        # MODIFIED 12/9/16 NEW:
        return self.inputValue or [defaultControlAllocation]
        # MODIFIED 12/9/16 END

    def _instantiate_input_states(self, context=None):
        """Instantiate inputValue attribute

        Otherwise, no need to do anything, as DefaultControllerMechanism only adds inputStates
        when a ControlProjection is instantiated, and uses _instantiate_control_mechanism_input_state
        """

        try:
            self.inputStates
        except AttributeError:
            self.inputValue = None
        else:
            pass


    def _instantiate_control_projection(self, projection, context=None):
        """Instantiate requested controlProjection and associated inputState
        """

        # Instantiate inputStates and allocationPolicy attribute for controlSignal allocations
        input_name = 'DefaultControlAllocation for ' + projection.receiver.name + '_ControlSignal'
        self._instantiate_control_mechanism_input_state(input_name, defaultControlAllocation, context=context)
        self.allocationPolicy = self.inputValue


        # Call super to instantiate outputStates
        super()._instantiate_control_projection(projection=projection,
                                                      context=context)
