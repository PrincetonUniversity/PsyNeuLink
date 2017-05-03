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

A ModulatoryProjection is a subclass of `Projection` that takes the output of an `AdaptiveMechanism`, and uses that
to modulate the function of a `state <State>`.  There are three types of ModulatoryProjections, that
modulate different types of components and their states: 

* `LearningProjection`
    These take an `learning_signal <LearningMechanism.learning_signal>` (the output of a
    `LearningMechanism`), and use that to modulate the function of the MATRIX `parameterState <ParameterState>` of a
    `MappingProjection`, which in turn uses it to modify its `matrix <MappingProjection.MappingProjection.matrix>`
    parameter. LearningProjections are used when learning has been specified for a `process <Process_Learning>`
    or `system <System_Execution_Learning>`.
..
* `GatingProjection`
    These take a `gating_signal <GatingMechanism.gating_signal>` (usually the ouptput of a `GatingMechanism`), 
    and use that to modulate the function of an `inputState <InputState>` or `outputState <OutputState>` of a 
    ProcessingMechanism, which in turn modulates that state's value.
..
* `ControlProjection`
    These take a `control_signal <ControlProjection.ControlProjection.allocation>` (usually the ouptput
    of a `ControlMechanism <ControlMechanism>`) and use that to modulate the function of a 
    `parameterState <ParameterState>` of a ProcessingMechanism, which in turn uses it to modulate the value of
    a parameter of the mechanism or its function.  ControlProjections are typically used in the context of a `System`.
..

COMMENT:
* Gating: takes an input signal and uses it to modulate the inputState and/or outputState of the receiver
COMMENT

.. _Projection_Creation:

Creating a ModulatoryProjection
-------------------------------

The types of ModulatoryProjections listed above are base classes, and cannot be instantiated directly.  A  
ModulatoryProjection can be created directly by calling the constructor for the desired type of projection.  More
commonly, however, projections are either specified `in context <Projection_In_Context_Specification>`, or
are `created automatically <Projection_Automatic_Creation>`, the details of which are described in the documentation
for each type.

.. _ModulatoryProjection_Structure:

Structure
---------

A ModulatoryProjection has the same basic structure as a `Projection`.  See the documentation for
individual subtypes of ModulatoryProjection for more specific information about their structure.  In addition
to the standard attributes of a projection, it also has a `modulate <ModulatoryProjection.modulation>` attribute
that determines how it influences the `State` to which it projects.  Specifically, it determines which parameter of 
the state's `function <State.function>` is modulated by the ModulatryProjection. 

.. _ModulatoryProjection_Execution:

Execution
---------

.. _ModulatoryProjection_Class_Reference:

A ModulatoryProjection, like any projection, cannot be executed directly.  It is executed when the `state <State>` to 
which it projects — its `receiver <Projection.receiver>` — is updated;  that occurs when the state's owner mechanism 
is executed.  When a ModulatoryProjection executes, it gets the value of its `sender <Projection.sender>`, 
and uses that to adjust the parameter of the state's function (determined by its 
`modulate <ModulatoryProjection.modulate>` attribute. 

.. note::
   The change made to the parameter of the state's function in response to the execution of a ModulatoryProjection
   are not applied until the state is updated which, in turn, does not occur until the mechanism to which the state 
   belongs is next executed; see :ref:`Lazy Evaluation` for an explanation of "lazy" updating).

.. _ModulatoryProjection_Class_Reference:

Class Reference
---------------


"""

from PsyNeuLink.Components.Projections.Projection import Projection_Base

class ModulatoryProjection_Base(Projection_Base):
    """
    LearningProjection(               \
                 sender=None,         \
                 receiver=None,       \
                 learning_function,   \
                 learning_rate=None,  \
                 params=None,         \
                 name=None,           \
                 prefs=None)

    Implements a projection that modifies the matrix parameter of a MappingProjection.


    Arguments
    ---------
    sender : Optional[LearningMechanism or LEARNING_SIGNAL OutputState of one]
        the source of the `error_signal` for the LearningProjection. If it is not specified, one will be
        `automatically created <LearningProjection_Automatic_Creation>` that is appropriate for the
        LearningProjection's `errorSource <LearningProjection.errorSource>`.

    receiver : Optional[MappingProjection or ParameterState for ``matrix`` parameter of one]
        the `parameterState <ParameterState>` (or the `MappingProjection` that owns it) for the
        `matrix <MappingProjection.MappingProjection.matrix>` to be modified by the LearningProjection.

    learning_function : Optional[LearningFunction or function] : default BackPropagation
        specifies a function to be used for learning by the `sender <LearningMechanism.sender>` (i.e., its
        `function <LearningMechanism.function>` attribute).

    learning_rate : Optional[float or int]
        if specified, it is applied mulitiplicatively to `learning_signal` received from the `LearningMechanism`
        from which it projects (see `learning_rate <LearningProjection.learning_rate>` for additional details).

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

    componentType : LEARNING_PROJECTION

    sender : LEARNING_SIGNAL OutputState of a LearningMechanism
        source of `learning_signal <LearningProjection.learning_signal>`.

    receiver : MATRIX ParameterState of a MappingProjection
        parameterState for the `matrix <MappingProjection.MappingProjection.matrix>` parameter of
        the `learned_projection` to be modified by the LearningProjection.

    variable : 2d np.array
        same as `learning_signal <LearningProjection.learning_signal>`.

    function : Function : default Linear
        assigns the learning_signal received from `LearningMechanism` as the value of the projection.

    value : 2d np.array
        same as `weight_change_matrix`.

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

    def __init__(self,
                 receiver,
                 sender=None,
                 modulate=None,  # <- this is not used; here just to force inclusion of attribute in docstring below
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
