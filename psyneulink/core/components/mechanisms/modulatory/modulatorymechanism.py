# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *********************************************  ModulatoryMechanism *****************************************************

"""

Contents
--------

  * `ModulatoryMechanism_Overview`
  * `ModulatoryMechanism_Creation`
  * `ModulatoryMechanism_Structure`
  * `ModulatoryMechanism_Execution`
  * `ModulatoryMechanism_Class_Reference`


.. _ModulatoryMechanism_Overview:

Overview
--------

A ModulatoryMechanism is a type of `Mechanism <Mechanism>` that modifies the operation of one or more other `Components
<Component>`.  In general, a ModulatoryMechanism receives its input from an `ObjectiveMechanism`, however
this need not be the case.

.. _ModulatoryMechanism_Types:

There are two primary types of ModulatoryMechanism:

* `ControlMechanism`
    modulates the `value <Port_Base.value>` of a `Port` of a `Mechanism <Mechanism>`.  Takes an evaluative signal
    (generally received from an `ObjectiveMechanism`) and generates a `control_allocation
    <ControlMechanism.control_allocation>`, each item of which is assigned to one of its `ControlSignals
    <ControlSignal>`;  each of those generates a `control_signal <ControlSignal.control_signal>` that is used by its
    `ControlProjection(s) <ControlProjection>` to modulate the parameter of a `function <Port_Base.function>` (and
    thereby the `value <Port_Base.value>`) of a `Port`.  ControlSignals have `costs <ControlSignal_Costs>`,
    and a ControlMechanism has a `costs <ControlMechanism.costs>` and a `net_outcome <ControlMechanism.net_outcome>`
    that is computed based on the `costs <ControlSignal.costs>` of its ControlSignals. A ControlMechanism can be
    assigned only the `ControlSignal` class of `ModulatorySignal`, but can be also be assigned other generic
    `OutputPorts <OutputPort>` that appear after its ControlSignals in its `output_ports
    <ControlMechanism.output_ports>` attribute.

    `GatingMechanism` is a specialized subclass of ControlMechanism,
    that is used to modulate the `value <Port_Base.value>` of an `InputPort` or `OutputPort`, and that uses
    `GatingSignals <GatingSignal>` which do not have any cost attributes.

* `LearningMechanism`
    modulates the `matrix <MappingProjection.matrix>` parameter of a `MappingProjection`.  Takes an error signal
    (received from an `ObjectiveMechanism` or another `LearningMechanism`) and generates a `learning_signal
    <LearningMechanism.learning_signal>` that is provided to its `LearningSignal(s) <LearningSignal>`, and used by
    their `LearningProjections <LearningProjection>` to modulate the `matrix <MappingProjection.matrix>` parameter
    of a `MappingProjection`. A LearningMechanism can be assigned only the `LearningSignal` class of `ModulatorySignal`
    as its `OuputStates <OutputPort>`, but can be also be assigned other generic `OutputPorts <OutputPort>`,
    that appear after its LearningSignals in its `output_ports <LearningMechanism.output_ports>` attribute.

A single `ModulatoryMechanism` can be assigned more than one ModulatorySignal of the appropriate type, each of which
can be assigned different `control_allocations <ControlSignal.control_allocation>` (for ControlSignals) or
`learning_signals <LearningMechanism.learning_signal>` (for LearningSignals).  A single ModulatorySignal can also be
assigned multiple ModulatoryProjections; however, as described  in `ModulatorySignal_Projections`, they will all
be assigned the same `variable <ModulatoryProjection_Base.variable>`.

.. _ModulatoryMechanism_Naming:

*Naming Conventions for Modulatory Components*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modulatory Components and their attributes are named according to the type of modulation using the following templates:

  * ModulatoryMechanism name
      <*Type*>Mechanism (e.g., ControlMechanism)
  * ModulatorySignal name
      <*Type*>Signal (e.g., ControlSignal)
  * ModulatoryProjection name
      <*Type*>Projection (e.g., ControlProjection)
  * List of a ModulatoryMechanism's ModulatorySignals
      <*Type*>Mechanism.<type>_signals (e.g., ControlMechanism.control_signals)
  * Value of a ModulatorySignal
      <*Type*>Signal.<type>_signal (e.g., ControlSignal.control_signal)


.. _ModulatoryMechanism_Creation:

Creating a ModulatoryMechanism
------------------------------

A ModulatoryMechanism is created by calling the constructor for the desired type. ModulatoryMechanisms of the
appropriate subtype can also be created automatically when other Components are created that require them (for
example, one or more `LearningMechanism <LearningMechanism>` are created when a `learning pathway
<Component_Learning_Pathway>` is specified for a `Composition` (see `subtypes <ModulatoryMechanism_Subtypes>` of
ModulatoryMechanisms for more specific information about how to create them).

.. _ModulatoryMechanism_Structure:

Structure
---------

A ModulatoryMechanism has the same basic structure as a `Mechanism <Mechanisms>`.  In addition, every ModulatoryMechanism
has a `modulation <ModulatoryMechanism.modulation>` attribute, that determines the default method by which its
`ModulatorySignals <ModulatorySignal>` modify the value of the Components that they modulate (see the `modulation
<ModulatorySignal_Modulation>` for a description of how modulation operates, and the documentation for individual
subtypes of ModulatoryMechanism for more specific information about their structure and modulatory operation).

.. _ModulatoryMechanism_Execution:

Execution
---------

The execution of a ModulatoryMechanism follows the same sequence of actions as a standard `Mechanism <Mechanism>`
(`Mechanism_Execution`).  However, in the context of a Composition, its `ModulatoryProjection`\\s are often feedback
connections to other Mechanisms in the Composition, forming a `cycle <Composition_Acyclic_Cyclic>` in the Composition,
and thus the ModulatoryMechanism may need to be properly initialized (see `Composition_Cycles_and_Feedback`).
Also, a `ControlMechanism` can be the `controller <Composition.controller>` of a `Composition` in which case its
execution is handled separately from other Mechanisms in the Composition (see `Composition_Controller_Execution`).
As noted below, the modifications made by a ModulatoryMechanism to a parameter that it modulates are not available to
that parameter's Component until *after* the Modulatory executes, which may not be until the next `TRIAL
<TimeScale.TRIAL>` (e.g., the `matrix <MappingProjection.matrix>` parameter of a `MappingProjection` that is subject
to `learning <Composition_Learning>` in a Composition; or a parameter modulated by a `ControlMechanism` that is the
`controller <Composition_Controller>` of a Composition that executes last (see `Composition_Controller_Execution`).

.. ModulatoryMechanism_Lazy_Evaluation_Note:

.. note::
   A `Port` that receives a `ModulatoryProjection` from a ModulatoryMechanism does not update its value until its
   owner Mechanism executes (see `Lazy Evaluation <Component_Lazy_Updating>` for an explanation of "lazy" updating).
   This means that even if a ModulatoryMechanism has executed, a parameter that it modulates will not assume its new
   value until the Mechanism to which it belongs has executed.

.. _ModulatoryMechanism_Class_Reference:

Class Reference
---------------

"""

from psyneulink.core.components.mechanisms.mechanism import Mechanism_Base
from psyneulink.core.globals.keywords import ADAPTIVE_MECHANISM
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel

__all__ = [
    'ModulatoryMechanism_Base', 'ModulatoryMechanismError'
]


class ModulatoryMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class ModulatoryMechanism_Base(Mechanism_Base):
    """Subclass of `Mechanism <Mechanism>` that modulates the value(s) of one or more other `Component(s) <Component>`.
    See `Mechanism <Mechanism_Class_Reference>` and subclasses for arguments and additional attributes.

    .. note::
       ModulatoryMechanism is an abstract class and should *never* be instantiated by a call to its constructor.
       They should be instantiated using the constructor for a `subclass <ModulatoryMechanism_Subtypes>`.


    Attributes
    ----------

    modulation : str
        determines how the output of the ModulatoryMechanism's `ModulatorySignal(s) <ModulatorySignal>` are used to
        modulate the value of the Port(s) to which their `ModulatoryProjection(s) <ModulatoryProjection>` project.
   """

    componentType = ADAPTIVE_MECHANISM

    class Parameters(Mechanism_Base.Parameters):
        """
            Attributes
            ----------

                modulation
                    see `modulation <ModulatoryMechanism_Base.modulation>`

                    :default value: None
                    :type:
        """
        modulation = None

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TYPE_DEFAULT_PREFERENCES
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     PREFERENCE_SET_NAME: 'ModulatoryMechanismClassPreferences',
    #     PREFERENCE_KEYWORD<pref>: <setting>...}

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
        """Abstract class for ModulatoryMechanism
        """

        super().__init__(
            default_variable=default_variable,
            size=size,
            modulation=modulation,
            params=params,
            name=name,
            prefs=prefs,
            context=context,
            function=function,
            **kwargs
        )
