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

An AdaptiveMechanism is a type of `Mechanism <Mechanism>` that modifies the operation of one or more other `Components
<Component>`.  In general, an AdaptiveMechanism receives its input from an `ObjectiveMechanism`, however
this need not be the case.

.. _AdaptiveMechanism_Types:


There are four types of AdaptiveMechanism:

* `ModulatoryMechanism`
    takes an evaluative signal (generally received from an `ObjectiveMechanism`) and generates an
    `modulatory_allocation <ModulatoryMechanism.modulatory_allocation>`, each item of which is assigned to one of its
    `ModulatorySignals <ModulatorySignal>`;  each of those generates a `modulatory_signal
    <ModulatorySignal.modulatory_signal>` that is used by its `ModulatoryProjection(s) <ModulatoryProjection>` to
    modulate the parameter of a `function <State_Base.function>` (and thereby the `value <State_Base.value>`) of a
    `State`.  A ModulatoryMechanism can be assigned any combination of `ControlSignals <ControlSignal>` and
    `GatingSignals <GatingSignal>`.
..
* `ControlMechanism`
    a subclsass of `ModulatoryMechanism` that adds support for `costs <ControlMechanism.costs>`;  it takes an
    evaluative signal (generally received from an `ObjectiveMechanism`) and generates a `control_allocation
    <ControlMechanism.control_allocation>`, each item of which is assigned to one of its `ControlSignals
    <ControlSignal>`;  each of those generates a `control_signal <ControlSignal.control_signal>` that is used by its
    `ControlProjection(s) <ControlProjection>` to modulate the parameter of a `function <State_Base.function>` (and
    thereby the `value <State_Base.value>`) of a `State`.  A ControlMechanism can be assigned only the `ControlSignal`
    class of `ModulatorySignal`, but can be also be assigned other generic `OutputStates <OutputState>`.
..
* `GatingMechanism`
    a subclsass of `ModulatoryMechanism` that is specialized for modulating the input to or ouput from a `Mechanism`;
    it takes an evaluative signal (generally received from an `ObjectiveMechanism`) and generates a
    `gating_allocation <GatingMechanism.gating_allocation>`, each item of which is assigned to one of its
    `GatingSignals <ControlSignal>`;  each of those generates a `gating_signal <ControlSignal.control_signal>`
    that is used by its `GatingProjection(s) <ControlProjection>` to modulate the parameter of a `function
    <State_Base.function>` (and thereby the `value <State_Base.value>`) of an `InputState` or `OutputState`.
    A GatingMechanism can be assigned only the `GatingSignal` class of `ModulatorySignal`, but can be also be assigned
    other generic `OutputStates <OutputState>`.
.
..
* `LearningMechanism`
    takes an error signal (received from an `ObjectiveMechanism` or another `LearningMechanism`) and generates a
    `learning_signal <LearningMechanism.learning_signal>` that is provided to its `LearningSignal(s)
    <LearningSignal>`, and used by their `LearningProjections <LearningProjection>` to modulate the `matrix
    <MappingProjection.matrix>` parameter of a `MappingProjection`. A LearningMechanism can be assigned only
    `LearningSignals <LearningSignal>` as its `OuputStates <OutputState>`.

See `ModulatorySignal <ModulatorySignal_Naming>` for conventions used for the names of Modulatory components.

A single `AdaptiveMechanism` can be assigned more than one ModulatorySignal of the appropriate

which, each of which can
be assigned
different `allocations <ModulatorySignal.allocation>` (for ControlSignals and GatingSignals) or `learning_signals
<LearningMechanism.learning_signal>` (for LearningSignals).  A single ModulatorySignal can also be assigned multiple
ModulatoryProjections; however, as described  under `_ModulatorySignal_Projections`, they will all be assigned the
same `variable <ModulatoryProjection_Base.variable>`.


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
<System.controller>`), or when `control is specified <ControlMechanism_Control_Signals>` for the parameter of a
`Mechanism <Mechanism>`; and one or more `LearningMechanism <LearningMechanism>` are created when learning is
specified for a `Process <Process_Learning_Sequence>` or a `System <System_Learning>` (see the documentation for
`subtypes <AdaptiveMechanism_Subtypes>` of AdaptiveMechanisms for more specific information about how to create them).

.. _AdaptiveMechanism_Structure:

Structure
---------

An AdaptiveMechanism has the same basic structure as a `Mechanism <Mechanisms>`.  In addition, every AdaptiveMechanism
has a `modulation <AdaptiveMechanism.modulation>` attribute, that determines the default method by which its
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

from psyneulink.core.components.mechanisms.mechanism import Mechanism_Base
from psyneulink.core.globals.keywords import ADAPTIVE_MECHANISM
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel

__all__ = [
    'AdaptiveMechanism_Base', 'AdaptiveMechanismError'
]


class AdaptiveMechanismError(Exception):
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

    class Parameters(Mechanism_Base.Parameters):
        """
            Attributes
            ----------

                modulation
                    see `modulation <AdaptiveMechanism_Base.modulation>`

                    :default value: None
                    :type:

        """
        modulation = None

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'AdaptiveMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    def __init__(self,
                 default_variable,
                 size,
                 modulation,
                 params,
                 name,
                 prefs,
                 context=None,
                 function=None,
                 **kwargs
                 ):
        """Abstract class for AdaptiveMechanism
        """

        if not hasattr(self, 'system'):
            self.system = None

        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(params=params,
                                                  modulation=modulation)

        super().__init__(default_variable=default_variable,
                         size=size,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context,
                         function=function,
                         **kwargs
                         )
