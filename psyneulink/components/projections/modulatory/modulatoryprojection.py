# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ******************************************  ModulatoryProjection *****************************************************

"""
.. _ModulatoryProjection_Overview:

Overview
--------

A ModulatoryProjection is a subclass of `Projection <Projection>` that takes the value of a
`ModulatorySignal <ModulatorySignal>` belonging to an `AdaptiveMechanism <AdaptiveMechanism>`, and uses that to
modulate the function of the `State <State>` to which it projects.  There are three types of ModulatoryProjections,
that modulate different types of Components and their States:

* `LearningProjection`
    takes the `value <LearningSignal.value>` of a `LearningSignal` belonging to a `LearningMechanism`,
    and conveys it to the *MATRIX* `ParameterState` of a `MappingProjection`, for use by its
    `function <ParameterState.function>` in modulating the value of the MappingProjection's
    `matrix <MappingProjection.matrix>` parameter.
..
* `GatingProjection`
    takes the `value <GatingSignal.value>` of a `GatingSignal` belonging to a `GatingMechanism`, and conveys it
    to the `InputState` or `OutputState` of a `ProcessingMechanism <ProcessingMechanism>` for use by the State's
    `function <State_Base.function>` in modulating its `value <State_Base.value>`.
..
* `ControlProjection`
    takes the `value of a <ControlSignal.value>` of a `ControlSignal` belonging to a `ControlMechanism`,
    and conveys it to the `ParameterState` for the parameter of a `Mechanism <Mechanism>` or its
    `function <Mechanism_Base.function>`, for use in modulating the value of the parameter.

.. _Projection_Creation:

Creating a ModulatoryProjection
-------------------------------

A ModulatoryProjection is a base class, and cannot be instantiated directly.  However, the three types of
ModulatoryProjections listed above can be created directly, by calling the constructor for the desired type.
More commonly, however, ModulatoryProjections are either specified in the context of the States to or from
which they project (`State_Projections` in State, and `Projection_Specification>`), or are `created automatically
<Projection_Automatic_Creation>`, the details of which are described in the documentation for each type of
ModulatoryProjection.

.. _ModulatoryProjection_Structure:

Structure
---------

A ModulatoryProjection has the same basic structure as a `Projection <Projection>`, augmented by type-specific
attributes and methods described under each type of ModulatoryProjection.  The ModulatoryProjections received by a
`State <State>` are listed in the State's `mod_afferents` attribute.

.. _ModulatoryProjection_Execution:

Execution
---------

A ModulatoryProjection, like any Projection, cannot be executed directly.  It is executed when the `State <State>` to
which it projects — its `receiver <Projection_Base.receiver>` — is updated;  that occurs when the State's owner
Mechanism is executed.  When a ModulatoryProjection executes, it conveys both the `value <ModulatorySignal.value>` of
the `ModulatorySignal <ModulatorySignal>` from which it projects, and the ModulatorySignal's `modulation
<ModulatorySignal.modulation>` attribute, to the State that receives the Projection.  The State assigns the value to
the parameter of the State's `function <State_Base.function>` specified by the `modulation` attribute, and then calls
the `function <State_Base.function>` to determine the `value <State_Base.value>` of the State.

.. note::
   The change made to the parameter of the State's Function in response to the execution of a ModulatoryProjection
   are not applied until the State is updated which, in turn, does not occur until the Mechanism to which the State
   belongs is next executed; see :ref:`Lazy Evaluation` for an explanation of "lazy" updating).

.. _ModulatoryProjection_Class_Reference:

Class Reference
---------------

"""

from psyneulink.components.projections.projection import Projection_Base
from psyneulink.globals.keywords import MODULATORY_PROJECTION, NAME
from psyneulink.components.component import InitStatus

__all__ = [
    'MODULATORY_SIGNAL_PARAMS'
]

MODULATORY_SIGNAL_PARAMS = 'modulatory_signal_params'


class ModulatoryProjectionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class ModulatoryProjection_Base(Projection_Base):
    """
    ModulatoryProjection_Base(     \
        receiver,                  \
        sender=None,               \
        weight=None,               \
        exponent=None,             \
        params=None,               \
        name=None,                 \
        prefs=None,                \
        context=None)

    Subclass of `Projection <Projection>` that modulates the value of a `State <State>`.

    .. note::
       ModulatoryProjection is an abstract class and should NEVER be instantiated by a call to its constructor.
       It should be instantiated using the constructor for a `subclass <ModulatoryProjection_Subtypes>`.

    Arguments
    ---------

    receiver : Optional[State or Mechanism]
        specifies the State to which the ModulatoryProjection projects.

    sender : Optional[OutputState or Mechanism] : default None
        specifies the Component from which the ModulatoryProjection projects.

    weight : number : default None
       specifies the value by which to multiply the ModulatoryProjection's `value <ModulatoryProjection.value>`
       before combining it with others (see `weight <ModulatoryProjection.weight>` for additional details).

    exponent : number : default None
       specifies the value by which to exponentiate the ModulatoryProjection's `value <ModulatoryProjection.value>`
       before combining it with others (see `exponent <ModulatoryProjection.exponent>` for additional details).

    params : Dict[param keyword, param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        ModulatoryProjection, its `function <ModulatoryProject.function>`, and/or a custom function and its parameters.
        By default, it contains an entry for the ModulatoryProjection's default `function <ModulatoryProject.function>`
        and parameter assignments.  Values specified for parameters in the dictionary override any assigned to those
        parameters in arguments of the constructor.

    name : str : default see ModulatoryProjection `name <ModulatoryProjection.name>`
        specifies the name of the ModulatoryProjection; see ModulatoryProjection `name <ModulatoryProjection.name>` 
        for details.

    prefs : PreferenceSet or specification dict : default Projection.classPreferences
        specifies the `PreferenceSet` for the ModulatoryProjection; see `prefs <ModulatoryProjection.prefs>` for 
        details.

    context : str : default None
        optional reference to a subclass

    Attributes
    ----------

    receiver : MATRIX ParameterState of a MappingProjection
        the State to which the ModulatoryProjection projects, the `function <State_Base.function>` of which is
        modulated by it.

    sender : LEARNING_SIGNAL OutputState of a LearningMechanism
        the `ModulatorySignal <ModulatorySignal>` from which the ModulatoryProjection projects.

    variable : 2d np.array
        value received from the `ModulatorySignal <ModulatorySignal>` that is the ModulatoryProjection's
        `sender <ModulatoryProjection.sender`.

    function : Function : default Linear
        assigns the value received from the ModulatoryProjection's `sender <ModualatoryProjection.sender>` to
        its `value <ModulatoryProjection.value>`.

    value : 2d np.array
        value used to modulate the `function <State_Base.function>` of the State that is its `receiver
        <ModulatoryProjection.receiver>`.

    weight : number
       multiplies the `value <ModulatoryProjection.value>` of the ModulatoryProjection after applying `exponent
       <ModulatoryProjection.exponent>`, and before combining it with any others that project to the same `State` to
       determine that State's `variable <State.variable>` is modified (see description in `Projection
       <Projection_Weight_and_Exponent>` for details).

    exponent : number
        exponentiates the `value <ModulatoryProjection.value>` of the ModulatoryProjection, before applying `weight
        <ModulatoryProjection.weight>`, and before combining it with any others that project to the same `State` to
        determine that State's `variable <State.variable>` is modified (see description in `Projection
        <Projection_Weight_and_Exponent>` for details).

    name : str
        the name of the ModulatoryProjection. If the ModulatoryProjection's `initialization has been deferred
        <Projection_Deferred_Initialization>`, it is assigned a temporary name (indicating its deferred initialization
        status) until initialization is completed, at which time it is assigned its designated name.  If that is the
        name of an existing ModulatoryProjection, it is appended with an indexed suffix, incremented for each
        ModulatoryProjection with the same base name (see `Naming`). If the name is not specified in the **name**
        argument of its constructor, a default name is assigned using the following format:
        '<ModualatorySignal type> for <receiver owner Mechanism's name>[<receiver's name>]'
        (for example, ``'GatingSignal for my_mech[InputState-0]'``).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the ModulatoryProjection; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    """
    componentCategory = MODULATORY_PROJECTION

    def __init__(self,
                 receiver,
                 sender=None,
                 weight=None,
                 exponent=None,
                 params=None,
                 name=None,
                 prefs=None,
                 context=None):

        super().__init__(receiver=receiver,
                         sender=sender,
                         params=params,
                         weight=weight,
                         exponent=exponent,
                         name=name,
                         prefs=prefs,
                         context=context)

    def _assign_default_projection_name(self, state=None, sender_name=None, receiver_name=None):

        template = "{} for {}[{}]"

        if self.init_status in {InitStatus.INITIALIZED, InitStatus.UNSET}:
            # If the name is not a default name for the class, return
            if not self.className + '-' in self.name:
                return self.name
            self.name = template.format(self.className, self.receiver.owner.name, self.receiver.name)

        elif self.init_status is InitStatus.DEFERRED_INITIALIZATION:
            projection_name = template.format(self.className, state.owner.name, state.name)
            # self.init_args[NAME] = self.init_args[NAME] or projection_name
            self.name = self.init_args[NAME] or projection_name

        else:
            raise ModulatoryProjectionError("PROGRAM ERROR: {} has unrecognized InitStatus ({})".
                                            format(self, self.init_status))
