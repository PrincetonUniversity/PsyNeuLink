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

A MonitoringMechanism monitors the outputState of a ProcessingMechanisms in a :doc:`process <Process>` or
:doc:`system <System>`, which is evaluated by its ``function``.  This can generate an error signal (e.g.,
for learning[LINK]) or some other value (e.g., conflict).

.. _MonitoringMechanism_Creation:

Creating A MonitoringMechanism
---------------------------

a MonitoringMechanism can be created by using the standard Python method of calling the constructor for the desired
type. One or more MonitoringMechanisms are also created automatically when a process or system is created for
which learning is specified (see :ref:`learning in a process <Process_Learning>`,
and :ref:`automatic creation of LearningSignals <LearningSignal_Automatic_Creation> for details).
Different types of MonitoringMechanisms monitor  different types of information, and therefore have varying input
reqirements. See :doc:`subclasses  <MonitoringMechanisms>` for the specific requirements of each type.

Execution
---------

A MonitoringMechanism always executes after the mechanism it is monitoring.  The ``value`` of the outputState of the
mechanism being monitored is assigned as an item in the ``variable`` for the MonitoringMechanism's ``function``.
Other items may be also be assigned (for example, a :doc:`Comparator` takes an additional input against which it
compares the monitored value).

.. _MonitoringMechanism_Class_Reference:

Class Reference
---------------

"""


from PsyNeuLink.Components.Mechanisms.Mechanism import *
from PsyNeuLink.Components.ShellClasses import *
from PsyNeuLink.Components.Mechanisms.ControlMechanisms.ControlMechanism import defaultControlAllocation

COMPARATOR = 'Comparator'

# Comparator parameter keywords:
SAMPLE = "comparatorSampleSource"
TARGET = "comparatorTargetSource"
COMPARISON_OPERATION = "comparison_operation"

# Comparator outputs (used to create and name outputStates):
COMPARISON_RESULT = 'ComparisonArray'
COMPARISON_MEAN = 'ComparisonMean'
COMPARISON_SUM = 'ComparisonSum'
COMPARISON_SSE = 'ComparisonSumSquares'
COMPARISON_MSE = 'ComparisonMSE'

# Comparator output indices (used to index output values):
class ComparatorOutput(AutoNumber):
    COMPARISON_RESULT = ()
    COMPARISON_MEAN = ()
    COMPARISON_SUM = ()
    COMPARISON_SSE = ()
    COMPARISON_MSE = ()


class MonitoringMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class MonitoringMechanism_Base(Mechanism_Base):
    # Instance Attributes:
    #     monitoredStateChanged
    # Instance Methods:
    #     _update_monitored_state_changed_attribute(current_monitored_state)

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
       They should be instantiated using the constructor for a :doc:`subclass <MonitoringMechanism>`.

    COMMENT:
        Description:
            This is a TYPE and subclasses are SUBTYPES.
            The primary purpose is to implement TYPE level preferences for all monitoring mechanisms.
            All subclasses must call update_monitored_state_flag
    COMMENT

    Attributes
    ----------

    monitoredStateChanged : bool
        identifies whether the outputState being monitored has changed since the last execution.

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
    variableClassDefault = [defaultControlAllocation]

    @tc.typecheck
    def __init__(self,
                 variable=NotImplemented,
                 params=NotImplemented,
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

        self.monitoredStateChanged = False
        self._last_monitored_state = None

        super().__init__(variable=variable,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)


    def _update_monitored_state_changed_attribute(self, current_monitored_state):
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

