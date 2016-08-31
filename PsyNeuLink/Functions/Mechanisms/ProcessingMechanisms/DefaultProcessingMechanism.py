# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# **********************************************  Mechanism ***********************************************************
#

from PsyNeuLink.Functions.ShellClasses import *
from PsyNeuLink.Functions.Mechanisms.Mechanism import *


# **************************************** DefaultProcessingMechanism ******************************************************


class DefaultProcessingMechanism_Base(Mechanism_Base):
    """Use to implement SystemDefaultInputMechanism, DefaultControlMechanism, and SystemDefaultOutputMechanism

    Description:
        Implements "dummy" mechanism used to implement default input, control signals, and outputs to other mechanisms

    Class attributes:
        + functionType (str): System Default Mechanism
        + paramClassDefaults (dict):
            # + kwInputStateValue: [0]
            # + kwOutputStateValue: [1]
            + kwFunction: Linear
            + kwFunctionParams:{kwSlope:1, kwIntercept:0}
    """

    functionType = "DefaultProcessingMechanism"

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # Any preferences specified below will override those specified in SubtypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to SUBTYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'DefaultProcessingMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    variableClassDefault = SystemDefaultInputValue

    from PsyNeuLink.Functions.Utility import Linear
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        kwFunction:Linear,
        kwFunctionParams:{Linear.kwSlope:1, Linear.kwIntercept:0}
    })

    def __init__(self,
                 default_input_value=NotImplemented,
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented):
        """Add Linear as default function, assign default name, and call super.__init__

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

        super(DefaultProcessingMechanism_Base, self).__init__(variable=default_input_value,
                                                              params=params,
                                                              name=name,
                                                              prefs=prefs,
                                                              context=self)
