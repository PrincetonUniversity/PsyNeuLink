# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ******************************************  MonitoringMechanism ******************************************************

"""
Overview
--------

A MonitoringMechanism monitors the `outputState <OutputState>` of a `ProcessingMechanism <ProcessingMechanism>` in a
`process <Process>` or `system <System>`.  This can generate an error signal (e.g., for use in
`learning <LearningProjection>`) or some other value (e.g., a conflict signal).

.. _MonitoringMechanism_Creation:

Creating A MonitoringMechanism
---------------------------

Generally, one or more `MonitoringMechanisms  <MonitoringMechanism>` are `created automatically
<LearningProjection_Automatic_Creation>` when learning is specified for a `process  <Process_Learning> or `system
<System_Execution_Learning>`.  However, MonitoringMechanisms can also be created using the standard Python method of
calling the constructor for the desired type.  Different types of MonitoringMechanisms monitor  different types of
information, and therefore have varying input requirements.

Execution
---------

MonitoringMechanisms always execute after all of the mechanism being monitored in the system have execute.
The `value <OutputState.OutputState.value>` of the `outputState <OutputState>` of the mechanism being monitored is
assigned as the primary input to the MonitoringMechanism. Other items may be also be assigned (for example,
a `ComparatorMechanism` takes an additional  input `target <ComparatorMechanism.ComparatorMechanism.target>` against
which it compares the monitored value).

.. _MonitoringMechanism_Class_Reference:

Class Reference
---------------

"""

from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanism import defaultControlAllocation
from PsyNeuLink.Components.Mechanisms.Mechanism import *
from PsyNeuLink.Components.ShellClasses import *


class MonitoringMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class MonitoringMechanism_Base(Mechanism_Base):
    # Instance Attributes:
    #     monitoredStateChanged
    # Instance Methods:
    #     _update_status(current_monitored_state)

    """
    MonitoringMechanism( \
    variable=None,       \
    params=None,         \
    name=None,           \
    prefs=None,          \
    context=None):       \

    Abstract class for MonitoringMechanism

    .. note::
       MonitoringMechanisms should NEVER be instantiated by a direct call to the base class.
       They should be instantiated using the constructor for a `subclass <MonitoringMechanism>`.

    COMMENT:
        Description:
            This is a TYPE and subclasses are SUBTYPES.
            The primary purpose is to implement TYPE level preferences for all monitoring mechanisms.
    COMMENT

    Attributes
    ----------

    monitoredStateChanged : bool
        identifies whether the outputState being monitored has changed since the last execution.

    """

    componentType = MONITORING_MECHANISM

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'ProcessingMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    # variableClassDefault = defaultControlAllocation
    # This must be a list, as there may be more than one (e.g., one per controlSignal)
    variableClassDefault = defaultControlAllocation

    @tc.typecheck
    def __init__(self,
                 variable=None,
                 params=None,
                 name=None,
                 prefs=None,
                 context=None):
        """Abstract class for MonitoringMechanisms

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
