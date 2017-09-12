# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************  ProcessingMechanism ****************************************************

"""
.. _ProcessingMechanism_Overview:

Overview
--------

A ProcessingMechanism is a type of `Mechanism <>` that transforms its input in some way.  A ProcessingMechanism always
receives its input either from another ProcessingMechanism, or from the input to a `Process` or `System` when it is
executed.  Similarly, its output is generally conveyed to another ProcessingMechanism or used as the ouput for a Process
or System.  However, the output of a ProcessingMechanism may also be used by an `AdaptiveMechanism <AdaptiveMechanism>`
to modify the parameters of other components (or its own). ProcessingMechanisms are always executed before all
AdaptiveMechanisms in the Process and/or System to which they belong, so that any modifications made by the
AdaptiveMechanism are available to all ProcessingMechanisms in the next `TRIAL`.

.. _ProcessingMechanism_Creation:

Creating a ProcessingMechanism
------------------------------

A ProcessingMechanism can be created by using the standard Python method of calling the constructor for the desired
type. Some types of ProcessingMechanism (for example, `ObjectiveMechanisms <ObjectiveMechanism>`) are also created
when a System or Process is created, if `learning <LINK>` and/or `control <LINK>` have been specified for it.

.. _AdaptiveMechanism_Structure:

Structure
---------

A ProcessingMechanism has the same basic structure as a `Mechanism <Mechanisms>`.  See the documentation for
individual subtypes of ProcessingMechanism for more specific information about their structure.

.. _Comparator_Execution:

Execution
---------

A ProcessingMechanism always executes before any `AdaptiveMechanisms <AdaptiveMechanism>` in the `Process
<Process_Execution>` or `System <System_Execution>` to which it belongs.

"""

from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism_Base
from PsyNeuLink.Globals.Defaults import defaultControlAllocation
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceLevel

# ControlMechanismRegistry = {}


class ProcessingMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class ProcessingMechanism_Base(Mechanism_Base):
    # DOCUMENTATION: this is a TYPE and subclasses are SUBTYPES
    #                primary purpose is to implement TYPE level preferences for all processing mechanisms
    #                inherits all attributes and methods of Mechanism -- see Mechanism for documentation
    # IMPLEMENT: consider moving any properties of processing mechanisms not used by control mechanisms to here
    """Subclass of `Mechanism <Mechanism>` that implements processing in a :ref:`Pathway`.

    .. note::
       ProcessingMechanism is an abstract class and should NEVER be instantiated by a call to its constructor.
       It should be instantiated using the constructor for a `subclass <ProcessingMechanism_Subtypes>`.
   """

    componentType = "ProcessingMechanism"

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'ProcessingMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    class ClassDefaults(Mechanism_Base.ClassDefaults):
        # This must be a list, as there may be more than one (e.g., one per control_signal)
        variable = defaultControlAllocation

    def __init__(self,
                 variable=None,
                 size=None,
                 input_states=None,
                 output_states=None,
                 params=None,
                 name=None,
                 prefs=None,
                 context=None):
        """Abstract class for processing mechanisms

        :param variable: (value)
        :param size: (int or list/array of ints)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        :param context: (str)
        """

        self.system = None

        super().__init__(variable=variable,
                         size=size,
                         input_states=input_states,
                         output_states=output_states,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)

    def _validate_inputs(self, inputs=None):
        # Let mechanism itself do validation of the input
        pass

    def _instantiate_attributes_before_function(self, context=None):

        super()._instantiate_attributes_before_function(context=context)
