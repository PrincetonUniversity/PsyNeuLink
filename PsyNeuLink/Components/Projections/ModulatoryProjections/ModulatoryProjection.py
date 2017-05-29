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
    This takes a `learning_signal <LearningMechanism.learning_signal>` (the output of a `LearningMechanism`), 
    and uses it to modulate the value of the `matrix <MappingProjection.MappingProjection.matrix>`
    parameter of a MappingProjection.
..
* `GatingProjection`
    This takes a `gating_signal <GatingMechanism.gating_signal>` (usually the output of a `GatingMechanism`), 
    and uses it to modulate the value of an `inputState <InputState>` or `outputState <OutputState>` of a 
    ProcessingMechanism.
..
* `ControlProjection`
    This takes a `control_signal <ControlProjection.ControlProjection.allocation>` (usually the ouptput
    of a `ControlMechanism <ControlMechanism>`) and uses it to modulate the value of a parameter of a 
    ProcessingMechanism or its function.
..

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

A ModulatoryProjection has the same basic structure as a `Projection`, augmented by type-specific attributes
and methods described under each type of projection.  In addition, all ModulatoryProjections have a 
`modulation_operation <ModulatoryProjection.modulation_operation>` attribute that determines how the projection 
modifies the value of the state to which it projects.  The modulation_operation is specified using a value of 
`ModulationOperation`, which designates either a parameter of the state's function to modulate, or one of two other 
actions to take, as follows:

    * `ModulatonOperation.ADDITIVE` - this specifies use of the parameter designated by the  
      `TransferFunction's <TransferFunction_ModulationOperation>` :keyword:`additive` attribute
      (for example, it is the `intercept <Linear.slope>` parameter of the `Linear` Function;    
      
    * `ModulatonOperation.MULTIPLICATIVE` - this specifies use of the parameter designated by the  
      `TransferFunction's <TransferFunction_ModulationOperation>` :keyword:`multiplicative` attribute
      (for example, it is the `slope <Linear.slope>` parameter of the `Linear` Function;    
    
    * `ModulatonOperation.OVERRIDE` - this specifies direct use of the ModulatoryProjection's value, bypassing
      the state's `function <State.function>` or its `base_value <State.base_value>`. 
  
    * `ModulatonOperation.DISABLED` - this speifies disabling of any modulation of the state's value, and use only
      of its `base_value <State.base_value>`.

These values can be assigned when a state is created, or at `runtime <>` 

The `modulation_operation <ModulatoryProjection>` parameter can also be specified using a custom function or method, 
so long as it receives and returns values that are compatible with the `value <State.value>` of the state. 
If the `modulation_operation <ModulatoryProjection>` parameter is not specified for a ModulatoryProjection,
a default specified by the state itself is used.

.. _ModulatoryProjection_Execution:

Execution
---------

.. _ModulatoryProjection_Class_Reference:

A ModulatoryProjection, like any projection, cannot be executed directly.  It is executed when the `state <State>` to 
which it projects — its `receiver <Projection.receiver>` — is updated;  that occurs when the state's owner mechanism 
is executed.  When a ModulatoryProjection executes, it gets the value of its `sender <Projection.sender>`, 
and uses that to adjust the parameter of the state's function (determined by its 
`modulation_operation <ModulatoryProjection.modulate_operation>` attribute. 

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
                 modulation_operation=None, \
                 params=None,               \
                 name=None,                 \
                 prefs=None)

    Implements a projection that modulates the value of a `state <State>`.

    Arguments
    ---------
    sender : Optional[OutputState or Mechanism]
        specifies the component from which the ModulatoryProjection projects. 

    receiver : Optional[State or Mechanism]
        specifies the state to which the ModulatoryProjection projects, and the value of which is modulated by it.

    modulation_operation : Optional[ModulationOperation, function or method] : default determined by State
        specifies the manner by which the ModulatoryProjection modulates the `value <State.value>` of the 
        `state <State>` to which it projects.    

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
        the component from which the ModulatoryProjection projects. 

    receiver : MATRIX ParameterState of a MappingProjection
        the state to which the ModulatoryProjection projects, and the value of which is modulated by it.

    variable : 2d np.array
        same as `learning_signal <LearningProjection.learning_signal>`.

    modulation_operation : Optional[Function parameter name]
        the manner by which the ModulataoryProjection modulates the `value <State.value>` of the state to which it
        projects - see `ModulatoryProjection_Structure` for a description of specifications.

    function : Function : default Linear
        assigns the value received from the ModualatoryProjection's `sender <ModualatoryProjection.sender>` to
        its `value <ModulatoryProjection.value>`.

    value : 2d np.array
        value used to modulate the value of the `receiver <ModulatoryProjection.receiver>`. 

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
