# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************  ProcessingMechanism ****************************************************

"""

Overview
--------

A ProcessingMechanism is a type of `Mechanism <Mechanism>` that transforms its input in some way.  A
ProcessingMechanism always receives its input either from another ProcessingMechanism, or from the input to a `process
<Process>` or `system <System>` when it is executed.  Similarly, its output is generally conveyed to another
ProcessingMechanism or used as the ouput for a process or system.  However, the output of a ProcessingMechanism may
also be used by an `AdaptiveMechanism` to modify the parameters of other components (or its own).
ProcessingMechanisms are always executed before all AdpativeMechanisms in the process and/or system to which they
belong, so that any modificatons made by the AdpativeMechanism are available to all ProcessingMechanisms in the next
round of execution.

.. _ProcessingMechanism_Creation:

Creating a ProcessingMechanism
------------------------------

A ProcessingMechanism can be created by using the standard Python method of calling the constructor for the desired
type. Some types of ProcessingMechanism (for example, `ObjectiveMechanisms <ObjectiveMechanism>`) are also created
when a system or process is created, if `learning <LINK>` and/or `control <LINK>` have been specified for it.

.. _AdaptiveMechanism_Structure:

Structure
---------

A ProcessingMechanism has the same basic structure as a `Mechanism <Mechanisms>`.  See the documentation for
individual subtypes of ProcessingMechanism for more specific information about their structure.

.. _Comparator_Execution:

Execution
---------

A ProcessingMechanism always executes before any `AdaptiveMechanisms <AdaptiveMechanism>` in the process or
system to which it belongs.

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
    # This must be a list, as there may be more than one (e.g., one per control_signal)
    variableClassDefault = defaultControlAllocation

    def __init__(self,
                 variable=None,
                 input_states=None,
                 output_states=None,
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
                         input_states=input_states,
                         output_states=output_states,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)

    def _validate_inputs(self, inputs=None):
        # Let mechanism itself do validation of the input
        pass
