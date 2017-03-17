# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************  ProcessingMechanism ****************************************************

"""
**[DOCUMENTATION STILL UNDER CONSTRUCTION]**

COMMENT:
  MOVE TO ProcessingMechanisms overview:
  Different ProcessingMechanisms transform their input in different ways, and some allow this to be customized
  by modifying their ``function`` parameter.  For example, a :doc:`TransferMechanism` can be configured to produce a
  linear, logistic, or exponential transform of its input.
COMMENT

"""

from PsyNeuLink.Components.Mechanisms.Mechanism import *
from PsyNeuLink.Components.ShellClasses import *

# ControlMechanismRegistry = {}


class ProcessingMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class ProcessingMechanism_Base(Mechanism_Base):
    # DOCUMENTATION: this is a TYPE and subclasses are SUBTYPES
    #                primary purpose is to implement TYPE level preferences for all processing mechanisms
    #                inherits all attributes and methods of Mechanism -- see Mechanism for documentation
    # IMPLEMENT: consider moving any properties of processing mechanisms not used by control mechanisms to here
    """Abstract class for processing mechanism subclasses
   """

    componentType = "ProcessingMechanism"

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'ProcessingMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    # variableClassDefault = defaultControlAllocation
    # This must be a list, as there may be more than one (e.g., one per controlSignal)
    variableClassDefault = defaultControlAllocation

    def __init__(self,
                 variable=None,
                 params=None,
                 name=None,
                 prefs=None,
                 context=None):
        """Abstract class for processing mechanisms

        :param variable: (value)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        :param context: (str)
        """

        self.system = None

        super().__init__(variable=variable,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)

    def _validate_inputs(self, inputs=None):
        # Let mechanism itself do validation of the input
        pass
