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

A ModulatoryProjection is a subclass of `Projection` that takes the value of a `ModulatoryOutputState` of an
`AdaptiveMechanism`, and uses that to modulate the function of a `state <State>`.  There are three types of
ModulatoryProjections, that modulate different types of components and their states:

* `LearningProjection`
    This takes the `value <LearningSignal.value>` of a `LearningSignal` belonging to a LearningMechanism,
    and conveys it to the *MATRIX* `ParameterState` of a `MappingProjection`, for use by its
    `function <ParameterState.function>` in modulating the value of the MappingProjection's
    `matrix <MappingProjection.matrix>` parameter.
..
* `GatingProjection`
    This takes the `value <GatingSignal.value>` of a `GatingSignal` belonging to a GatingMechanism,
    and conveys it to the `inputState <InputState>` or `outputState <OutputState>` of a `ProcessingMechanism`.
    for use by the state's :keyword:`funtion` in modulating its :keyword:`value`.
..
* `ControlProjection`
    This takes the `value of a <ControlSignal.value> of a `ControlSignal` belonging to a ControlMechanism,
    and conveys it to the `ParameterState` for the parameter of a `Mechanism` or its
    `function <Mechanism.function>`, for use in modulating the value of the parameter.

.. _Projection_Creation:

Creating a ModulatoryProjection
-------------------------------

A ModulatoryProjection is a base class, and cannot be instantiated directly.  A ModulatoryProjection can be created
directly by calling the constructor for one of the types listed above.  More commonly, however, ModulatoryProjections
are either specified `in context <Projection_In_Context_Specification>`, or are `created automatically
<Projection_Automatic_Creation>`, the details of which are described in the documentation for each type.

.. _ModulatoryProjection_Structure:

Structure
---------

A ModulatoryProjection has the same basic structure as a `Projection`, augmented by type-specific attributes
and methods described under each type of ModulatoryProjection.

.. _ModulatoryProjection_Execution:

Execution
---------

.. _ModulatoryProjection_Class_Reference:

A ModulatoryProjection, like any projection, cannot be executed directly.  It is executed when the `state <State>` to 
which it projects — its `receiver <Projection.receiver>` — is updated;  that occurs when the state's owner mechanism 
is executed.  When a ModulatoryProjection executes, it conveys both the value of its `sender <Projection.sender>`
and the value of its sender's :keyword:`modulation` attribute to the state that receives the ModulatoryProjection.  The
state assigns the value to the parameter of the state's function specified by the modulation attribute,
and then calls the function to determine the `value <State.value>` of the state.

.. note::
   The change made to the parameter of the state's function in response to the execution of a ModulatoryProjection
   are not applied until the state is updated which, in turn, does not occur until the mechanism to which the state 
   belongs is next executed; see :ref:`Lazy Evaluation` for an explanation of "lazy" updating).

.. _ModulatoryProjection_Class_Reference:

Class Reference
---------------

"""

from PsyNeuLink.Components.Projections.Projection import Projection_Base
from PsyNeuLink.Globals.Keywords import MODULATORY_PROJECTION

MODULATORY_SIGNAL_PARAMS = 'modulatory_signal_params'

class ModulatoryProjection_Base(Projection_Base):
    """
    ModulatoryProjection(                   \
                 sender=None,               \
                 receiver=None,             \
                 modulation=None, \
                 params=None,               \
                 name=None,                 \
                 prefs=None)

    Implements a projection that modulates the value of a `state <State>`.

    Arguments
    ---------
    sender : Optional[OutputState or Mechanism]
        specifies the component from which the ModulatoryProjection projects.

    receiver : Optional[State or Mechanism]
        specifies the state to which the ModulatoryProjection projects.

    modulation : Optional[Modulation] : default determined by State
        specifies the manner by which the ModulatoryProjection modulates the `function <State.function>`
        (and thereby `value <State.value>` of the `State` to which it projects.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
        projection, its function, and/or a custom function and its parameters. By default, it contains an entry for
        the projection's default `function <LearningProjection.function>` and parameter assignments.  Values specified
        for parameters in the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default LearningProjection-<index>
        a string used for the name of the LearningProjection.
        If not is specified, a default is assigned by ProjectionRegistry
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Projection.classPreferences]
        the `PreferenceSet` for the LearningProjection.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    sender : LEARNING_SIGNAL OutputState of a LearningMechanism
        the component from which the ModulatoryProjection projects (usually the outputState of an
        `AdaptiveMechanism`).

    receiver : MATRIX ParameterState of a MappingProjection
        the state to which the ModulatoryProjection projects, the `function <State.function>`
        of which is modulated by it.

    variable : 2d np.array
        value received from the `sender <ModulatoryProjection.sender`.

    modulation : Optional[Function parameter name]
        the manner by which the ModulataoryProjection modulates the `function <State.function>` of the state to which
        it projects - see `ModulatoryProjection_Structure` for a description of specifications.

    function : Function : default Linear
        assigns the value received from the ModualatoryProjection's `sender <ModualatoryProjection.sender>` to
        its `value <ModulatoryProjection.value>`.

    value : 2d np.array
        value used to modulate the function of the `receiver <ModulatoryProjection.receiver>`.

    name : str : default LearningProjection-<index>
        the name of the LearningProjection.
        Specified in the **name** argument of the constructor for the projection;
        if not is specified, a default is assigned by ProjectionRegistry
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for projection.
        Specified in the **prefs** argument of the constructor for the projection;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """
    componentCategory = MODULATORY_PROJECTION

    def __init__(self,
                 receiver,
                 sender=None,
                 params=None,
                 name=None,
                 prefs=None,
                 context=None):

        super().__init__(receiver=receiver,
                         sender=sender,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)
