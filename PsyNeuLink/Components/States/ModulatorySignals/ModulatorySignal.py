# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************  ModulatorySignal *******************************************************

# ***ADD mod_afferents TO STATE doscstring (IF NOT ALREADY THERE)
# ***REFERENCE THIS DOCSTRING IN AdaptiveMechanism DOSCTRING, AND ITS SUBCLASSSES

"""
Overview
--------
A ModulatorySignal is a subclass of `OutputState` that belongs to an `AdaptiveMechanism`, and is used to `modulate
<ModulatorySignal_Modulation>` the `value <State.value>` of another `State` by way of one or more
`ModulatoryProjections <ModulatoryProjection>`.  There are three types of ModulatySignals, each of which is
associated with a particular type of `AdaptiveMechanism` and `ModulatoryProjection`, as described below:

* `LearningSignal`
    This takes the `value <LearningSignal.value>` assigned to it by the `LearningMechanism` to which it belongs,
    and uses it to modulate the parameter of a `PathwayProjection` -- usually the `matrix <MappingProjection.matrix>`
    parameter of a `MappingProjection`.
..
* `ControlSignal`
    This takes the `value <ControlSignal.value>` assigned to it by the `ControlMechanism` to which it belongs,
    and uses it to modulate the parameter of a `Mechanism` or its `function <Mechanism.function>`.
..
* `GatingSignal`
    This takes the `value <GatingSignal.value>` assigned to it by the `GatingMechanism` to which it belongs,
    and uses it to modulate the value of the `InputState` or `OutputState` of a `Mechanism`.

.. _ModulatorySignal_Creation:

Creating a ModulatorySignal
---------------------------

A ModulatorySignal is a base class, and cannot be instantiated directly.  However, the three types of ModulatorySignals
listed above can be created directly, by calling the constructor for the desired type.  More commonly, however,
ModulatorySignals are created automatically by the `AdaptiveMechanism` to which they belong, or by specifying them
in the constructor for the `AdaptiveMechanism` to which it belongs (the details of which are described in the
documentation for each type of ModulatorySignal).

.. _ModulatorySignal_Structure:

Structure
---------

A ModulatorySignal is associated with one or more `ModulatoryProjections <ModulatoryProjection>` of the
corresponding type, that project to the State(s), the value(s) of which it modulates.  THe ModulatoryProjections
received by a `State` are listed in its `mod_afferents` attribute. The method by which a ModulatorySignal
modulates a State's `value <State.value>` is determined by the ModulatorySignal's
`modulation <ModulatorySignal.modulation>` attribute, as described below.

.. _ModulatorySignal_Modulation:

Modulation
~~~~~~~~~~

A ModulatorySignal modulates the value of a `State` by modifying a parameter of the state's `function <State.function>`,
which determines the state's `value <State.value>`.  The `function <State.function>` of every state assigns one of its
parameters as its *MULTIPLICATIVE_PARAM* and another as its *MULTIPLICATIVE_PARAM*.  The
`modulation <ModulatorySigal.modulation>` attribute of a ModulatorySignal determines which of these should be modified
by the ModulatorySignal, or one of two other actions that can be taken in determining the State's `value <State.value>`.
The `modulation <ModulatorySigal.modulation>` attribute is specified using one of the following 'Modulation` values
(also see `examples <ModulatorySignal_Examples>`:

  * `ModulationParam.MULTIPLICATIVE` - modify the *MULTIPLICATIVE* parameter of the state's `function <State.function>`;

  * `ModulationParam.ADDITIVE` - modify the *ADDITIVE* parameter of the state's `function <State.function>`;

  * `ModulationParam.OVERRIDE` - assign the ModulatorySignal's value as the state's `value <State.value>`;

  * `ModulationParam.DISABLE` - assign the State's `value <State.value>` ignoring the ModulatorySignal.

The value of the `modulation <ModulatorySignal.modulation>` attribute can be specified in the **modulation** arg of the
ModulatorySignal's constructor, or in a *MODULATION* entry of a `state specification dictionary <LINK>` used to
create the ModulatorySignal. If the value of the `modulation <ModulatorySignal.modulation>` attribute is not specified
when a ModulatorySignal is created, it is assigned the value of the `modulation <AdaptiveMechanism.modulation>`
attribute for the `AdaptiveMechanism` to which it belongs.

.. _ModulatorySignal_Examples:

Examples
~~~~~~~~

Example 1:  Modulation by a GatingMechanism (default is gain of logistic)

Example 2: Modulation by a ControlSignal specified in a parameter tuple

Example 2: Modulation by a ControlSignal specified in a parameter tuple using OVERRIDE


EXMAMPLES HERE
      (for example, it is the `slope <Linear.slope>` parameter of the `Linear` Function);
      (for example, it is the `bias <Logistic.bias>` parameter of the `Logistic` Function);

.. _ModulatorySignal_Execution:



Execution
---------

XXXXXX FROM STATE:
State cannot be executed.  They are updated when the component to which they belong is executed.  InputStates and
parameterStates belonging to a mechanism are updated before the mechanism's function is called.  OutputStates
are updated after the mechanism's function is called.  When a state is updated, it executes any projections that
project to it (listed in its `afferents <State.afferents>` attribute.  It uses the values it receives from any
`PathWayProjections` (listed in its `path_afferents` attribute) as the variable for its `function <State.function>`,
and the values it receives from any `ModulatoryProjections` (listed in its `mod_afferents` attribute) to determine
the parameters of its `function <State.function>`.  It then calls its `function <State.function>` to determine its
`value <State.value>`. This conforms to a "lazy evaluation" protocol (see :ref:`Lazy Evaluation <LINK>` for a more
detailed discussion).
XXXXXXX

.. note::
   The change in the value of a `State` in response to a ModulatorySignal is not applied until the mechanism(s) to
   which the state belongs is next executed; see :ref:`Lazy Evaluation <LINK>` for an explanation of "lazy" updating).

Class Reference
---------------

"""

from PsyNeuLink.Components.States.State import *
from PsyNeuLink.Components.States.OutputState import OutputState


class ModulatorySignalError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

modulatory_signal_keywords = {MECHANISM, MODULATION}
modulatory_signal_keywords.update(component_keywords)


class ModulatorySignal(OutputState):
    """
    ModulatorySignal(                                   \
        owner,                                      \
        function=LinearCombination(operation=SUM),  \
        modulation=ModulationParam.MULTIPLICATIVE   \
        params=None,                                \
        name=None,                                  \
        prefs=None)

    A subclass of OutputState that represents the value of a ModulatorySignal provided to a `GatingProjection`.

    COMMENT:

        Description
        -----------
            The ModulatorySignal class is a subtype of the OutputState class in the State category of Component,
            It is used primarily as the sender for GatingProjections
            Its FUNCTION updates its value:
                note:  currently, this is the identity function, that simply maps variable to self.value

        Class attributes:
            + componentType (str) = GATING_SIGNAL
            + paramClassDefaults (dict)
                + FUNCTION (LinearCombination)
                + FUNCTION_PARAMS (Modulation.MULTIPLY)
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
        specifies the `GatingMechanism` to which to assign the ModulatorySignal.

    function : Function or method : default Linear
        specifies the function used to determine the value of the ModulatorySignal from the value of its
        `owner <GatingMechanism.owner>`.

    COMMENT: [NEEDS DOCUMENTATION]
    COMMENT
    modulation : ModulationParam : default ModulationParam.MULTIPLICATIVE 

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
        the `GatingMechanism` to which the ModulatorySignal belongs.

    variable : number, list or np.ndarray
        used by `function <ModulatorySignal.function>` to compute the ModulatorySignal's `value <ModulatorySignal.value>`.

    function : TransferFunction :  default Linear(slope=1, intercept=0)
        provides the ModulatorySignal's `value <GatingMechanism.value>`; the default is an identity function that
        passes the input to the GatingMechanism as value for the ModulatorySignal.

    value : number, list or np.ndarray
        result of `function <ModulatorySignal.function>`.
    
    modulation : ModulationParam
        determines how the output of the ModulatorySignal is used to modulate the value of the state(s)
        to which its GatingProjection(s) project(s).

    efferents : [List[GatingProjection]]
        a list of the `GatingProjections <GatingProjection>` assigned to the ModulatorySignal.

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

    componentType = MODULATORY_SIGNAL
    # paramsType = OUTPUT_STATE_PARAMS

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'OutputStateCustomClassPreferences',
    #     kp<pref>: <setting>...}

    paramClassDefaults = State_Base.paramClassDefaults.copy()

    def __init__(self,
                 owner,
                 reference_value,
                 variable,
                 modulation,
                 index,
                 calculate,
                 params,
                 name,
                 prefs,
                 context):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(params=params,
                                                  modulation=modulation)

        super().__init__(owner,
                         reference_value,
                         variable=variable,
                         index=index,
                         calculate=calculate,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)

        self._modulation = self.modulation or owner.modulation