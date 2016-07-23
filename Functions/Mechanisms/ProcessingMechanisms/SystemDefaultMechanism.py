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

from Functions.ShellClasses import *
from Functions.Mechanisms.Mechanism import *


# **************************************** SystemDefaultMechanism ******************************************************


class SystemDefaultMechanism_Base(Mechanism_Base):
    """Use to implement SystemDefaultInputMechanism, SystemDefaultControlMechanism, and SystemDefaultOutputMechanism

    Description:
        Implements "dummy" mechanism used to implement default input, control signals, and outputs to other mechanisms

    Class attributes:
        + functionType (str): System Default Mechanism
        + paramClassDefaults (dict):
            # + kwMechanismInputStateValue: [0]
            # + kwMechanismOutputStateValue: [1]
            + kwExecuteMethod: Linear
            + kwExecuteMethodParams:{kwSlope:1, kwIntercept:0}
    """

    functionType = "SystemDefaultMechanism"

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # Any preferences specified below will override those specified in SubtypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to SUBTYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'SystemDefaultMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    variableClassDefault = SystemDefaultInputValue

    from Functions.Utility import Linear
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        kwExecuteMethod:Linear,
        kwExecuteMethodParams:{Linear.kwSlope:1, Linear.kwIntercept:0}
    })

    def __init__(self,
                 default_input_value=NotImplemented,
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented):
        """Add Linear as default executeMethod, assign default name, and call super.__init__

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

        super(SystemDefaultMechanism_Base, self).__init__(variable=default_input_value,
                                                       params=params,
                                                       name=name,
                                                       prefs=prefs,
                                                       context=self)

