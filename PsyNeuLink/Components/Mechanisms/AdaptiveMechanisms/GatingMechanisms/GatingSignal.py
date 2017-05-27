# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ******************************************  GatingSignal *****************************************************

"""
Overview
--------

A GatingSignal is an `OutputState` specialized for use with an `GatingMechanism`. It is used to modify the value of 
InputState(s) and/or OutputState(s) of one or more Mechanisms that have been specified for gating. A GatingSignal is 
associated with one or more `GatingProjections <GatingProjection>`, each of which projects to an InputStates and/or 
OutputState to be gated, and that is used to modulate that state's `value <State.value>`. 

.. _GatingSignal_Creation:

Creating a GatingSignal
-----------------------

A GatingSignal is created automatically whenever an InputState or OutputState of a mechanism 
`specified for gating <GatingMechanism_Gating_Signals>`.  GatingSignals can also be specified in the **gating_signals**
argument of the constructor for a `GatingMechanism`.  Although a GatingSignal can be created directly using its 
constructor (or any of the other ways for `creating an outputState <OutputStates_Creation>`),  this is neither 
necessary nor advisable, as a GatingSignal has dedicated components and requirements for configuration that must be 
met for it to function properly.

.. _GatingSignal_Structure:

Structure
---------

A GatingSignal is owned by a `GatingMechanism`, and associated with one or more `GatingProjections <GatingProjection>`, 
each of which projects to the InputState or OutputState that it gates.  Each GatingSignal has a 
`modulation_operation <GatingSignal.modulation_operation>` attribute that determines how the GatingProjection 
is used by the state to which it projects to modify its value (see 
`Modulatory Projections <ModulatoryProjection.modulation_operation>` for an explanation of how the 
modulation_operaton is specified and used to modulate a function).  

.. _ControlSignal_Execution:

Execution
---------

A GatingSignal cannot be executed directly.  It is executed whenever the `GatingMechanism` to which it belongs is
executed.  When this occurs, the GatingMechanism provides the GatingSignal with a value that is used by its 
`GatingProjection(s) <GatingProjection>` to modulate the :keyword:`value` of the states to which they project. Those 
states use the value of the GatingProjection they receive to modify a parameter of their function.  How the modulation
is executed is determined by the GatingSignal's `modulation_operation` attribute
(see `ModulatoryProjections_Modulation_Operation).

.. note::
   The change in the value of InputStates and OutputStates in response to the execution of a GatingMechanism are not 
   applied until the mechanism(s) to which those states belong are next executed; see :ref:`Lazy Evaluation <LINK>` 
   for an explanation of "lazy" updating).

Class Reference
---------------

"""

from PsyNeuLink.Components.Functions.Function import *
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCMechanism import *
from PsyNeuLink.Components.States.OutputState import OutputState, PRIMARY_OUTPUT_STATE
from PsyNeuLink.Components.States.State import *

class GatingSignalError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class GatingSignal(OutputState):
    """
    GatingSignal(                                                \
        owner,                                                   \
        function=LinearCombination(operation=SUM),               \
        modulation_operation=ModulationOperation.MULTIPLICATIVE  \
        params=None,                                             \
        name=None,                                               \
        prefs=None)

    A subclass of OutputState that represents the value of a GatingSignal provided to a `GatingProjection`.

    COMMENT:

        Description
        -----------
            The GatingSignal class is a subtype of the OutputState class in the State category of Component,
            It is used primarily as the sender for GatingProjections
            Its FUNCTION updates its value:
                note:  currently, this is the identity function, that simply maps variable to self.value

        Class attributes:
            + componentType (str) = GATING_SIGNAL
            + paramClassDefaults (dict)
                + FUNCTION (LinearCombination)
                + FUNCTION_PARAMS (ModulationOperation.MULTIPLY)
            + paramNames (dict)

        Class methods:
            function (executes function specified in params[FUNCTION];  default: Linear

        StateRegistry
        -------------
            All OutputStates are registered in StateRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances
    COMMENT


    Arguments
    ---------

    owner : GatingMechanism
        specifies the `GatingMechanism` to which to assign the GatingSignal.

    function : Function or method : default Linear
        specifies the function used to determine the value of the GatingSignal from the value of its 
        `owner <GatingMechanism.owner>`.
    
    modulation_operation : ModulationParam : default ModulationParam.MULTIPLICATIVE 

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that can be used to specify the parameters for
        the ControlSignal and/or a custom function and its parameters. Values specified for parameters in the dictionary
        override any assigned to those parameters in arguments of the constructor.

    name : str : default OutputState-<index>
        a string used for the name of the outputState.
        If not is specified, a default is assigned by the StateRegistry of the mechanism to which the outputState
        belongs (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : State.classPreferences]
        the `PreferenceSet` for the outputState.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    owner : GatingMechanism
        the `GatingMechanism` to which the GatingSignal belongs.

    variable : number, list or np.ndarray
        used by `function <GatingSignal.function>` to compute the GatingSignal's `value <GatingSignal.value>`.

    function : TransferFunction :  default Linear(slope=1, intercept=0)
        provides the GatingSignal's `value <GatingMechanism.value>`; the default is an identity function that
        passes the input to the GatingMechanism as value for the GatingSignal. 

    value : number, list or np.ndarray
        result of `function <GatingSignal.function>`.
    
    modulation_operation : ModulationParam
        specifies the way in which the output of the GatingSignal is used to modulate the value of the states
        to which its GatingProjections project.

    efferents : [List[GatingProjection]]
        a list of the `GatingProjections <GatingProjection>` assigned to the GatingSignal.

    name : str : default <State subclass>-<index>
        name of the outputState.
        Specified in the **name** argument of the constructor for the outputState.  If not is specified, a default is
        assigned by the StateRegistry of the mechanism to which the outputState belongs
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

        .. note::
            Unlike other PsyNeuLink components, state names are "scoped" within a mechanism, meaning that states with
            the same name are permitted in different mechanisms.  However, they are *not* permitted in the same
            mechanism: states within a mechanism with the same base name are appended an index in the order of their
            creation.

    prefs : PreferenceSet or specification dict : State.classPreferences
        the `PreferenceSet` for the outputState.
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
        PROJECTION_TYPE: GATING_PROJECTION,
        GATED_STATE:None,
    })
    #endregion

    @tc.typecheck
    def __init__(self,
                 owner,
                 reference_value,
                 variable=None,
                 index=PRIMARY_OUTPUT_STATE,
                 calculate=Linear,
                 function=LinearCombination(operation=SUM),
                 modulation_operation=ModulationParam.MULTIPLICATIVE,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Note index and calculate are not used by ControlSignal, but included here for consistency with OutputState

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function,
                                                  modulation_operation=modulation_operation,
                                                  params=params)
        self.reference_value = reference_value

        # FIX: 5/26/16
        # IMPLEMENTATION NOTE:
        # Consider adding self to owner.outputStates here (and removing from GatingProjection._instantiate_sender)
        #  (test for it, and create if necessary, as per outputStates in GatingProjection._instantiate_sender),

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        super().__init__(owner,
                         reference_value,
                         variable=variable,
                         index=index,
                         calculate=calculate,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)