# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *********************************************  AdaptiveMechanism *****************************************************

"""

Overview
--------

An AdaptiveMechanism is a type of `Mechanism <Mechanisms>` that uses its input to modify the parameters of one or more
other PsyNeuLink components.  In general, an AdaptiveMechanism receives its input from an `ObjectiveMechanism`, however
this need not be the case. There are two types of AdaptiveMechanism: `LearningMechanisms <LearningMechanism>`, that
modify the parameters of `MappingProjections <MappingProjection>`; and `ControlMechanisms <ControlMechanism>` that
modify the  parameters of other ProcessingMechanisms.  AdaptiveMechanisms are always executed after all
ProcessingMechanisms in the `process <Process>` or `system <System>` to which they belong have been
:ref:`executed <LINK>`, with all LearningMechanisms then executed before all ControlMechanisms. Both types of
AdaptiveMechanisms are executed before the next :ref:`round of execution <LINK>`, so that the modifications they make
are available during the next round of execution of the process or system.

.. _AdaptiveMechanism_Creation:

Creating an AdaptiveMechanism
------------------------------

An AdaptiveMechanism can be created by using the standard Python method of calling the constructor for the desired type.
AdaptiveMechanisms of the appropriate subtype are also created automatically when a :ref:`system
<System.System_Creation>` is created,  and/or learning is  specified for a :ref:`system <System.System_Learning>`,
a `process <Process_Learning>`, or any `projection <LearningProjection_Automatic_Creation>` within one.  See the
documentation for the individual subtypes of AdaptiveMechanisms for more specific information about how to create them.

.. _AdaptiveMechanism_Structure:

Structure
---------

An AdaptiveMechanism has the same basic structure as a `Mechanism <Mechanisms>`.  In addition, every AdaptiveMechanism
has a `modulation <AdpativeMechanism.modulation>` attribute, that determines the default method by which its
ModulatorySignals modify the value of objects that they modulate (see the `modulation <ModulatorySignal_Modulation>`
for a description of how modulation operates, and the documentation for individual subtypes of AdaptiveMechanism for
more specific information about their structure and modulatory operation).

.. _Comparator_Execution:

Execution
---------

An AdaptiveMechanism always executes after execution of all of the ProcessingMechanisms in the process or system to
which it belongs.  All of the `LearningMechanisms <LearningMechanism>` are then executed, followed by all of the
`ControlMechanisms <ControlMechanism>`.

"""

from PsyNeuLink.Components.Mechanisms.Mechanism import *
from PsyNeuLink.Components.ShellClasses import *

class AdpativeMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class AdaptiveMechanism_Base(Mechanism_Base):
    # IMPLEMENT: consider moving any properties of adaptive mechanisms not used by control mechanisms to here
    """An AdaptiveMechanism is a Type of the `Mechanism <Mechanism>` Category of Component

    Attributes
    ----------

    modulation : ModulationParam
        determines how the output of the AdaptiveMechanism's `ModulatorySignal(s) <ModulatorySignal>` are used to
        modulate the value of the State(s) to which their `ModulatoryProjection(s) <ModulatoryProjection>` project.
   """

    componentType = ADAPTIVE_MECHANISM

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'AdaptiveMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    # variableClassDefault = defaultControlAllocation
    # This must be a list, as there may be more than one (e.g., one per control_signal)
    variableClassDefault = defaultControlAllocation

    def __init__(self,
                 variable,
                 modulation,
                 params,
                 name,
                 prefs,
                 context):
        """Abstract class for AdaptiveMechanism
        """

        if not hasattr(self, 'system'):
            self.system = None

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(params=params,
                                                  modulation=modulation)


        super().__init__(variable=variable,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)

#     def _instantiate_output_states(self, context=None):
#         super()._instantiate_output_states(context=context)
#
#
# def _instantiate_adaptive_projections()