# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ******************************************  ModulatoryProjection *****************************************************

"""

Contents
--------

  * `ModulatoryProjection_Overview`
  * `ModulatoryProjection_Creation`
  * `ModulatoryProjection_Structure`
  * `ModulatoryProjection_Execution`
  * `ModulatoryProjection_Class_Reference`


.. _ModulatoryProjection_Overview:

Overview
--------

A ModulatoryProjection is a subclass of `Projection <Projection>` that takes the value of a
`ModulatorySignal <ModulatorySignal>` belonging to a `ModulatoryMechanism <ModulatoryMechanism>`, and uses that to
modulate the function of the `Port <Port>` to which it projects.  There are three types of ModulatoryProjections,
that modulate different types of Components and their Ports:

* `LearningProjection`
    takes the `value <LearningSignal.value>` of a `LearningSignal` belonging to a `LearningMechanism`,
    and conveys it to the *MATRIX* `ParameterPort` of a `MappingProjection`, for use by its
    `function <ParameterPort.function>` in modulating the value of the MappingProjection's
    `matrix <MappingProjection.matrix>` parameter.
..
* `ControlProjection`
    takes the `value <ControlSignal.value>` of a `ControlSignal` belonging to a `ControlMechanism`,
    and conveys it to the `ParameterPort` for the parameter of a `Mechanism <Mechanism>` or its
    `function <Mechanism_Base.function>`, for use in modulating the value of the parameter.
..
* `GatingProjection`
    takes the `value <GatingSignal.value>` of a `GatingSignal` belonging to a `GatingMechanism`, and conveys it
    to the `InputPort` or `OutputPort` of a `ProcessingMechanism <ProcessingMechanism>` for use by the Port's
    `function <Port_Base.function>` in modulating its `value <Port_Base.value>`.

See `ModulatoryMechanism <ModulatoryMechanism_Naming>` for conventions used for the names of Modulatory components.

.. _ModulatoryProjection_Creation:

Creating a ModulatoryProjection
-------------------------------

A ModulatoryProjection is a base class, and cannot be instantiated directly.  However, the three types of
ModulatoryProjections listed above can be created directly, by calling the constructor for the desired type.
More commonly, however, ModulatoryProjections are either specified in the context of the Ports to or from
which they project (`Port_Projections` in Port, and `Projection_Specification`), or are `created automatically
<Projection_Automatic_Creation>`, the details of which are described in the documentation for each type of
ModulatoryProjection.

.. _ModulatoryProjection_Structure:

Structure
---------

A ModulatoryProjection has the same basic structure as a `Projection <Projection>`, augmented by type-specific
attributes and methods described under each type of ModulatoryProjection.  The ModulatoryProjections received by a
`Port <Port>` are listed in the Port's `mod_afferents` attribute.

.. _ModulatoryProjection_Execution:

Execution
---------

A ModulatoryProjection, like any Projection, cannot be executed directly.  It is executed when the `Port <Port>` to
which it projects — its `receiver <Projection_Base.receiver>` — is updated;  that occurs when the Port's owner
Mechanism is executed.  When a ModulatoryProjection executes, it conveys both the `value <ModulatorySignal.value>` of
the `ModulatorySignal <ModulatorySignal>` from which it projects, and the ModulatorySignal's `modulation
<ModulatorySignal.modulation>` attribute, to the Port that receives the Projection.  The Port assigns the value to
the parameter of the Port's `function <Port_Base.function>` specified by the `modulation` attribute, and then calls
the `function <Port_Base.function>` to determine the `value <Port_Base.value>` of the Port.

.. note::
   The change made to the parameter of the Port's Function in response to the execution of a ModulatoryProjection
   are not applied until the Port is updated which, in turn, does not occur until the Mechanism to which the Port
   belongs is next executed; see `Lazy Evaluation <Component_Lazy_Updating>` for an explanation of "lazy" updating).

.. _ModulatoryProjection_Class_Reference:

Class Reference
---------------

"""

from psyneulink.core.components.projections.projection import Projection_Base, ProjectionRegistry
from psyneulink.core.globals.keywords import MODULATORY_PROJECTION, NAME
from psyneulink.core.globals.log import ContextFlags, LogEntry
from psyneulink.core.globals.registry import remove_instance_from_registry


__all__ = [
    'MODULATORY_SIGNAL_PARAMS'
]

MODULATORY_SIGNAL_PARAMS = 'modulatory_signal_params'


class ModulatoryProjectionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class ModulatoryProjection_Base(Projection_Base):
    """Subclass of `Projection <Projection>` that modulates the value of a `Port <Port>`.
    See `Projection <Projection_Class_Reference>` and subclasses for arguments and additonal attributes.

    .. note::
       ModulatoryProjection is an abstract class and should **never** be instantiated by a call to its constructor.
       It should be instantiated using the constructor for a `subclass <ModulatoryProjection_Subtypes>`.

    Attributes
    ----------

    name : str
        the name of the ModulatoryProjection. If the ModulatoryProjection's `initialization has been deferred
        <Projection_Deferred_Initialization>`, it is assigned a temporary name (indicating its deferred initialization
        status) until initialization is completed, at which time it is assigned its designated name.  If that is the
        name of an existing ModulatoryProjection, it is appended with an indexed suffix, incremented for each
        ModulatoryProjection with the same base name (see `Registry_Naming`). If the name is not specified in the
        **name** argument of its constructor, a default name is assigned using the following format:
        '<ModualatorySignal type> for <receiver owner Mechanism's name>[<receiver's name>]'
        (for example, ``'GatingSignal for my_mech[InputPort-0]'``).

    """
    componentCategory = MODULATORY_PROJECTION

    def _assign_default_projection_name(self, port=None, sender_name=None, receiver_name=None):

        template = "{} for {}[{}]"

        if self.initialization_status & (ContextFlags.INITIALIZED | ContextFlags.INITIALIZING):
            # If the name is not a default name for the class, return
            if not self.className + '-' in self.name:
                return self.name
            self.name = template.format(self.className, self.receiver.owner.name, self.receiver.name)

        elif self.initialization_status == ContextFlags.DEFERRED_INIT:
            projection_name = template.format(self.className, port.owner.name, port.name)
            # self._init_args[NAME] = self._init_args[NAME] or projection_name
            self.name = self._init_args[NAME] or projection_name

        else:
            raise ModulatoryProjectionError("PROGRAM ERROR: {} has unrecognized initialization_status ({})".
                                            format(self, self.initialization_status))
