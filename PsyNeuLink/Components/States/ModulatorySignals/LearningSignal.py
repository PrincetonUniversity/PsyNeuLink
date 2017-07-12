# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ******************************************  LearningSignal *****************************************************

"""
Overview
--------

A LearningSignal is a type of `ModulatorySignal`, that is specialized for use with a `LearningMechanism` and a
`LearningProjection`, to modify the parameter of a `MappingProjection`.  A LearningSignal receives an error signal
from the `LearningMechanism` to which it belongs, and uses that to compute a `learning_signal` that is assigned as the
`value <LearningProjection.value>` of its `LearningProjection`. The LearningProjection conveys its value to a
`ParameterState` of the projection being learned, which in turns uses that to modify the corresponding parameter.
By default, the projection is a `MappingProjection`, the parameter is its `matrix <MappingProjection.matrix>`
parameter, and the `learning_signal` is a matrix of weight changes that are added to MappingProjection's
`matrix <MappingProjection.matrix>`.

.. _LearningSignal_Creation:

Creating a LearningSignal
------------------------

A LearningSignal is created automatically whenever a `MappingProjection` is 
`specified for learning <LearningMechanism_Creation>` and the projection belongs to the same System as the
`LearningMechanism`.  LearningSignals can also be specified in the **learning_signals**
argument of the constructor for a `LearningMechanism`.  Although a LearningSignal can be created directly using its 
constructor (or any of the other ways for `creating an OutputState <OutputStates_Creation>`), this is neither
necessary nor advisable, as a LearningSignal has dedicated components and requirements for configuration that must be 
met for it to function properly.

.. _LearningSignal_Specification:

Specifying LearningSignals
~~~~~~~~~~~~~~~~~~~~~~~~

When a LearningSignal is specified in context (e.g., the **learning_signals** argument of the constructor for a
`LearningMechanism`, the specification can take any of the following forms:

  * a *ParameterState* of the Projection to which the parameter belongs;
  |
  * a *Projection*, which must be either a `LearningProjection`, or a `MappingProjection` to which the 
    LearningSignal should send a `LearningProjection`.  In both cases, it is assumed that the LearningProjection
    projects to the *MATRIX* ParameterState of a `MappingProjection`. 
  |
  * a *tuple*, with the *name* of the parameter as its 1st item. and the *projection* to which it belongs as the 2nd;
    note that this is a convenience format, which is simpler to use than a specification dictionary (see below), 
    but precludes specification of any `parameters <LearningSignal_Structure>` for the LearningSignal.
  |
  * a *specification dictionary*, that must contain at least the following two entries:

    * *NAME*:str - a string that is the name of the parameter to be controlled;
    * *PROJECTION*:Projection - the Projection to which the parameter belongs; 
      (note: the Projection itself should be specified even if the parameter belongs to its function).
    The dictionary can also contain entries for any other LearningSignal attributes to be specified
    (e.g., a LEARNING_RATE entry; see `below <ControlSignal_Structure>` for a description of LearningSignal attributes).

.. _LearningSignal_Structure:

Structure
---------

A LearningSignal is owned by an `LearningMechanism`, and associated with one or more 
`LearningProjections <LearningProjection>` that project(s) to the `ParameterStates <ParameterState>` associated with
the parameter(s) to be learned.  A LearningSignal has the following primary attributes:

.. _LearningSignal_Modulation:

* `modulation <LearningSignal.modulation>` : determines how the LearningSignal's `value <LearningSignal.value>` is
  used by the ParameterState(s) to which it projects to modify their value (see `ModulatorySignal_Modulation` for an
  explanation of how the modulation is specified and used to modulate the value of a parameter). The default value
  is set to the value of the `modulation <LearningMechanism.modulation>` attribute of the LearningMechanism to which 
  the LearningSignal belongs;  this is the same for all of the LearningSignals belonging to that LearningMechanism.
  However, the `modulation <LearningSignal.modulation>` can be specified individually for a LearningSignal using a
  specification dictionary where the LearningSignal is specified, as described `above <LearningSignal_Specification>`.
  The `modulation <LearningSignal.modulation>` value of a LearningSignal is used by all of the 
  `LearningProjections <LearningProjection>` that project from that LearningSignal.
    
.. _LearningSignal_Learning_Rate:

* `learning_rate <LearningSignal.learning_rate>`: the learning_rate for a LearningSignal is used to specify the
  `learning_rate <LearningProjection.learning_rate>` parameter for its `LearningProjection(s) <LearningProjection>`
  (i.e., those listed in its `efferents <LearningSignal.efferents>` attribute).  If specified, it is applied
  multiplicatively to the LearningProjection`s `learning_signal <LearningProjection.learning_signal>` and thus can be
  used to modulate the learning_rate in addition to (and on top of) one specified for the `LearningMechanism` or its
  `function <LearningMechanism.function>`.  Specification of the `learning_rate <LearningSignal.learning_rate> for a
  LearningSignal supercedes any specification(s) of the `learning_rate <LearningProjection.learning_rate>` for its
  LearningProjections, as well as for any `Process <Process.Process_Base.learning_rate>` and/or
  `System <System.System_Base.learning_rate>` to which the LearningSignal's owner belongs
  (see `learning_rate <LearningMechanism_Learning_Rate>` of LearningMechanism for additional details).

.. _LearningSignal_Function:

* `function <LearningSignal.function>`: converts the `learning_signal <LearningMechanism.learning_signal>` it receives
  from the LearningMechanism to which it belongs to its `value <LearningSignal.value>`. By default this is an identity 
  function (:keyword:`Linear(slope=1, intercept=0))`), that simply uses the `learning_signal` as its 
  `value <LearningSignal.value>`.  However, :keyword:`function` can be assigned another `TransferFunction`, or any 
  other function that takes a 2d array or matrix and returns a similar value.

  .. note:: The `index <OutputState.OutputState.index>` and `calculate <OutputState.OutputState.calculate>`
            attributes of a LearningSignal are automatically assigned and should not be modified.


.. _LearningSignal_Execution:

Execution
---------

A LearningSignal cannot be executed directly.  It is executed whenever the `LearningMechanism` to which it belongs is
executed.  When this occurs, the LearningMechanism provides the LearningSignal with a `learning_signal`, that is used
by its `function <LearningSignal.function>` to compute its `value <LearningSignal.value>` for that `TRIAL`.
That is used by its associated `LearningProjection` to modify the :keyword:`value` of the `matrix
<MappingProjection.matrix>` parameter of the `MappingProjection` being learned.

.. note::
   The changes in a MappingProjection's matrix parameter in response to the execution of a LearningMechanism are not
   applied until the MappingProjection is next executed; see :ref:`Lazy Evaluation <LINK>` for an explanation of
   "lazy" updating).

Class Reference
---------------

"""

# import Components
from PsyNeuLink.Components.Functions.Function import _is_modulation_param
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanisms.LearningMechanism import *
from PsyNeuLink.Components.States.OutputState import OutputState, PRIMARY_OUTPUT_STATE
from PsyNeuLink.Components.States.ModulatorySignals.ModulatorySignal import *


class LearningSignalError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


    def __str__(self):
        return repr(self.error_value)


class LearningSignal(ModulatorySignal):
    """
    LearningSignal(                                      \
        owner,                                           \
        function=LinearCombination(operation=SUM),       \
        modulation=ModulationParam.MULTIPLICATIVE        \
        learning_rate=None                               \
        params=None,                                     \
        name=None,                                       \
        prefs=None)

    A subclass of OutputState that represents the ControlSignal of a `ControlMechanism` provided to a 
    `ControlProjection`.

    COMMENT:

        Description
        -----------
            The ControlSignal class is a subtype of the OutputState type in the State category of Component,
            It is used as the sender for ControlProjections
            Its FUNCTION updates its value:
                note:  currently, this is the identity function, that simply maps variable to self.value

        Class attributes:
            + componentType (str) = CONTROL_SIGNAL
            + paramClassDefaults (dict)
                + FUNCTION (LinearCombination)
                + FUNCTION_PARAMS   (Operation.PRODUCT)
            + paramNames (dict)

        Class methods:
            function (executes function specified in params[FUNCTION];  default: Linear)

        StateRegistry
        -------------
            All OutputStates are registered in StateRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances
    COMMENT


    Arguments
    ---------

    owner : ControlMechanism
        specifies the `ControlMechanism` to which to assign the ControlSignal.

    function : Function or method : default Linear
        specifies the function used to determine the `intensity` of the ControlSignal from its `allocation`.
    
    learning_rate : float or None : default None
        specifies the learning_rate for the LearningSignal's `LearningProjections <LearningProjection>`
        (see `learning_rate <LearningSignal.learning_rate>` for details).

    modulation : ModulationParam : default ModulationParam.MULTIPLICATIVE
        specifies the way in which the `value <LearningSignal.value>` of the LearningSignal is used to modify the value          of the `matrix <MappingProjection.matrix>` parameter for the `MappingProjection` to which the LearningSignal's
        `LearningProjection(s) <LearningProjection>` project.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that can be used to specify the parameters for
        the ControlSignal and/or a custom function and its parameters. Values specified for parameters in the dictionary
        override any assigned to those parameters in arguments of the constructor.

    name : str : default OutputState-<index>
        a string used for the name of the OutputState.
        If not is specified, a default is assigned by the StateRegistry of the Mechanism to which the OutputState
        belongs (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : State.classPreferences]
        the `PreferenceSet` for the OutputState.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    owner : ControlMechanism
        the `ControlMechanism` to which the ControlSignal belongs.

    variable : number, list or np.ndarray
        same as `allocation`;  used by `function <ControlSignal.function>` to compute the ControlSignal's `intensity`.

    function : TransferFunction :  default Linear(slope=1, intercept=0)
        converts `allocation` into the ControlSignal's `intensity`.  The default is the identity function, which
        assigns the ControlSignal's `allocation` as its `intensity`.

    value : number, list or np.ndarray
        result of `function <ControlSignal.function>`; same as `intensity`.

    learning_rate : float : None
        determines the learning rate for the LearningSignal.  It is used to specify the 
        `learning_rate <LearningProjection.learning_rate>` parameter for the LearningProjection(s) listed in the
        `efferents <LearningSignal.efferents>` attribute (i.e., that project from the LearningSignal). See
        `LearningSignal learning_rate <LearningSignal_Learning_Rate>` for additional details.

    modulation : ModulationParam
        determines the way in which the `value <LearningSignal.value>` of the LearningSignal is used to modify the
        value of the `matrix <MappingProjection.matrix>` parameter for the `MappingProjection` to which the
        LearningSignal's `LearningProjection(s) <LearningProjection>` project.

    efferents : [List[ControlProjection]]
        a list of the `LearningProjections <LearningProjection>` assigned to (i.e., that project from) the
        LearningSignal.

    name : str : default <State subclass>-<index>
        name of the OutputState.
        Specified in the **name** argument of the constructor for the OutputState.  If not is specified, a default is
        assigned by the StateRegistry of the Mechanism to which the OutputState belongs
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

        .. note::
            Unlike other PsyNeuLink components, state names are "scoped" within a Mechanism, meaning that states with
            the same name are permitted in different Mechanisms.  However, they are *not* permitted in the same
            Mechanism: states within a Mechanism with the same base name are appended an index in the order of their
            creation.

    prefs : PreferenceSet or specification dict : State.classPreferences
        the `PreferenceSet` for the OutputState.
        Specified in the **prefs** argument of the constructor for the projection;  if it is not specified, a default is
        assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    #region CLASS ATTRIBUTES

    componentType = OUTPUT_STATES
    paramsType = OUTPUT_STATE_PARAMS

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'OutputStateCustomClassPreferences',
    #     kp<pref>: <setting>...}

    paramClassDefaults = State_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        PROJECTION_TYPE: LEARNING_PROJECTION,
        LEARNED_PARAM:None
    })
    #endregion


    @tc.typecheck
    def __init__(self,
                 owner=None,
                 reference_value=None,
                 variable=None,
                 index=PRIMARY_OUTPUT_STATE,
                 calculate=Linear,
                 function=LinearCombination(operation=SUM),
                 learning_rate: tc.optional(parameter_spec) = None,
                 modulation:tc.optional(_is_modulation_param)=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function,
                                                  learning_rate=learning_rate,
                                                  params=params)

        # FIX: 5/26/16
        # IMPLEMENTATION NOTE:
        # Consider adding self to owner.outputStates here (and removing from ControlProjection._instantiate_sender)
        #  (test for it, and create if necessary, as per OutputStates in ControlProjection._instantiate_sender),

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        super().__init__(owner=owner,
                         reference_value=reference_value,
                         variable=variable,
                         modulation=modulation,
                         index=index,
                         calculate=calculate,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)
