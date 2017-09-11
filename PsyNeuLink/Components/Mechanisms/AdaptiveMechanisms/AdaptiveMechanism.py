# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *********************************************  AdaptiveMechanism *****************************************************

"""

.. _AdaptiveMechanism_Overview:

Overview
--------

An AdaptiveMechanism is a type of `Mechanism <Mechanism>` that modifies the parameters of one or more other `Components
<Component>`.  In general, an AdaptiveMechanism receives its input from an `ObjectiveMechanism`, however
this need not be the case. There are three types of AdaptiveMechanism:

* `LearningMechanism`
    takes an error signal (generally received from an `ObjectiveMechanism`) and generates a `learning_signal
    <LearningMechanism.learning_signal>` that is provided to its `LearningSignal(s) <LearningSignal>`, and used
    by their `LearningProjections <LearningProjection>` to modulate the `matrix <MappingProjection.matrix>` parameter
    of a `MappingProjection`.
..
* `ControlMechanism <ControlMechanism>`
    takes an evaluative signal (generally received from an `ObjectiveMechanism`) and generates an
    `allocation_policy <ControlMechanism_Base.allocation_policy>`, each item of which is assigned to one of its
    `ControlSignals <ControlSignal>`;  each of those generates a `control_signal <ControlSignal.control_signal>`
    that is used by its `ControlProjection(s) <ControlProjection>` to modulate the parameter of a Component.
..
* `GatingMechanism`
    takes an evaluative signal (generally received from an `ObjectiveMechanism`) and generates a
    `gating_policy <GatingMechanism.gating_policy>`, each item of which is assigned to one of its
    `GatingSignals <ControlSignal>`;  each of those generates a `gating_signal <ControlSignal.control_signal>`
    that is used by its `GatingProjection(s) <ControlProjection>` to modulate the value of the `InputState` or
    `OutputState` of a `Mechanism <Mechanism>`.


See `ModulatorySignal <ModulatorySignal_Naming>` for conventions used for the names of Modulatory components.

COMMENT:
AdaptiveMechanisms are always executed after all `ProcessingMechanisms <ProcessingMechanism>` in the `Process` or
`System` to which they belong have been executed, with all LearningMechanism executed first, then GatingMechanism,
ControlMechanism. All three types of AdaptiveMechanisms are executed before the next `TRIAL`, so that the
modifications they make are available during the next `TRIAL` run for the Process or System.
COMMENT

.. _AdaptiveMechanism_Creation:

Creating an AdaptiveMechanism
------------------------------

An AdaptiveMechanism can be created by using the standard Python method of calling the constructor for the desired type.
AdaptiveMechanisms of the appropriate subtype are also created automatically when other Components are created that
require them, or a form of modulation is specified for them. For example, a `ControlMechanism <ControlMechanism>` is
automatically created as part of a `System <System_Creation>` (for use as its `controller
<System_Base.controller>`), or when `control is specified <ControlMechanism_Control_Signals>` for the parameter of a
`Mechanism <Mechanism>`; and one or more `LearningMechanism <LearningMechanism>` are created when learning is
specified for a `Process <Process_Learning_Sequence>` or a `System <System_Learning>` (see the documentation for
`subtypes <AdaptiveMechanism_Subtypes>` of AdaptiveMechanisms for more specific information about how to create them).

.. _AdaptiveMechanism_Structure:

Structure
---------

An AdaptiveMechanism has the same basic structure as a `Mechanism <Mechanisms>`.  In addition, every AdaptiveMechanism
has a `modulation <AdpativeMechanism.modulation>` attribute, that determines the default method by which its
`ModulatorySignals <ModulatorySignal>` modify the value of the Components that they modulate (see the `modulation
<ModulatorySignal_Modulation>` for a description of how modulation operates, and the documentation for individual
subtypes of AdaptiveMechanism for more specific information about their structure and modulatory operation).

.. _AdaptiveMechanism_Execution:

Execution
---------

LearningMechanism and ControlMechanism are always executed at the end of a `TRIAL`, after all `ProcessingMechanisms
<ProcessingMechanism>` in the `Process` or `System` to which they belong have been executed; all LearningMechanism
executed first, and then ControlMechanism.  All modifications made are available during the next `TRIAL`.
GatingMechanism are executed in the same manner as ProcessingMechanisms;  however, because they almost invariably
introduce recurrent connections, care must be given to their `initialization and/or scheduling
<GatingMechanism_Execution>`).


.. _AdaptiveMechanism_Class_Reference:

Class Reference
---------------

"""

from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism_Base
from PsyNeuLink.Globals.Defaults import defaultControlAllocation
from PsyNeuLink.Globals.Keywords import ADAPTIVE_MECHANISM
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceLevel


class AdpativeMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class AdaptiveMechanism_Base(Mechanism_Base):
    """Subclass of `Mechanism <Mechanism>` that modulates the value(s) of one or more other `Component(s) <Component>`.

    .. note::
       AdaptiveMechanism is an abstract class and should NEVER be instantiated by a call to its constructor.
       They should be instantiated using the constructor for a `subclass <AdaptiveMechanism_Subtypes>`.

    COMMENT:

    Description:
        An AdaptiveMechanism is a Type of the `Mechanism <Mechanism>` Category of Component

    COMMENT


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

    class ClassDefaults(Mechanism_Base.ClassDefaults):
        # This must be a list, as there may be more than one (e.g., one per control_signal)
        variable = defaultControlAllocation

    def __init__(self,
                 variable,
                 size,
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
                         size=size,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)

#     def _instantiate_output_states(self, context=None):
#         super()._instantiate_output_states(context=context)
#
#
# def _instantiate_adaptive_projections()
