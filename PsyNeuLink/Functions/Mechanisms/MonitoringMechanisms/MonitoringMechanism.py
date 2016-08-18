# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# **************************************  ControlMechanism ************************************************
#

from PsyNeuLink.Functions.Mechanisms.Mechanism import *
from PsyNeuLink.Functions.ShellClasses import *


class MonitoringMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class MonitoringMechanism_Base(Mechanism_Base):
    # DOCUMENTATION: this is a TYPE and subclasses are SUBTYPES
    #                primary purpose is to implement TYPE level preferences for all monitoring mechanisms
    #                inherits all attributes and methods of Mechanism -- see Mechanism for documentation
    #                all subclasses must call update_monitored_state_flag
    # Instance Attributes:
    #     monitoredStateChanged
    # Instance Methods:
    #     update_monitored_state_changed_attribute(current_monitored_state)

    """Abstract class for processing mechanism subclasses
   """

    functionType = "ProcessingMechanism"

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'ProcessingMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    # variableClassDefault = defaultControlAllocation
    # This must be a list, as there may be more than one (e.g., one per controlSignal)
    variableClassDefault = [defaultControlAllocation]

    def __init__(self,
                 variable=NotImplemented,
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented,
                 context=NotImplemented):
        """Abstract class for MonitoringMechanisms

        :param variable: (value)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        :param context: (str)
        """

        # Assign functionType to self.name as default;
        #  will be overridden with instance-indexed name in call to super
        if name is NotImplemented:
            self.name = self.functionType

        self.functionName = self.functionType
        self.system = None

        self.monitoredStateChanged = False
        self._last_monitored_state = None

        super().__init__(variable=variable,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)


    def update_monitored_state_changed_attribute(self, current_monitored_state):
        """Test whether monitored state has changed and set monitoredStateChanged attribute accordingly

        Args:
            current_monitored_state:

        Returns:
            value of self.monitoredStateChanged

        """

        # if current_monitored_state != self._last_monitored_state:
        if not np.array_equal(current_monitored_state,self._last_monitored_state):
            self.monitoredStateChanged = True
            self._last_monitored_state = current_monitored_state
        else:
            self.monitoredStateChanged = False

        return self.monitoredStateChanged
