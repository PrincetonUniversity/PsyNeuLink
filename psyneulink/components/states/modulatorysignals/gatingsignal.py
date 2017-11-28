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

A GatingSignal is a type of `ModulatorySignal <ModulatorySignal>` that is specialized for use with a `GatingMechanism`
and one or more `GatingProjections <GatingProjection>`, to modify the `value <State_Base.value>`\\(s) of the
`InputState(s) <InputState>` and/or `OutputState(s) <OutputState>` to which they project. A GatingSignal receives the
value from the `gating_policy <GatingMechanism.gating_policy>` of the GatingMechanism to which it belongs,
and assigns that as the value of its `gating_signal <GatingSignal.gating_signal>` to its `GatingProjection(s)
<GatingProjection>`, each of which projects to an InputState or OutputState and is used to modulate the `value
<State_Base.value>` of that State.


.. _GatingSignal_Creation:

Creating a GatingSignal
-----------------------

A GatingSignal is created automatically whenever an `InputState` or `OutputState` of a `Mechanism <Mechanism>` is
specified for gating.  This can be done either in the **gating_signals** argument of the constructor for a
`GatingMechanism <GatingMechanism_GatingSignals>`, or in the `specification of projections <State_Projections>` to
the InputState or OutputState.  Although a GatingSignal can be created directly using its constructor (or any of the
other ways for `creating an OutputState <OutputStates_Creation>`), this is usually not necessary nor is it advisable,
as a GatingSignal has dedicated components and requirements for configuration that must be met for it to function
properly.

.. _GatingSignal_Specification:

Specifying GatingSignals
~~~~~~~~~~~~~~~~~~~~~~~~

When a GatingSignal is specified in the **gating_signals** argument of the constructor for a `GatingMechanism`, the
InputState(s) and/or OutputState(s) it gates must be specified. This can take any of the following forms:

  * **InputState** or **OutputState** of a Mechanism;
  ..
  * **Mechanism** -- the `primary `InputState <InputState_Primary>` or `OutputState <OutputState_Primary>` is used;
  ..
  * **specification dictionary** -- can take either of the following two forms:

    * for gating a single state, the dictionary can have the following two entries:

        * *NAME*: str
            the string must be the name of the State to be gated; the GatingSignal will named by appending
            "_GatingSignal" to the name of the State.

        * *MECHANISM*: Mechanism
            the Mechanism must be the one to the which the State to be gated belongs.

    * for gating multiple states, the dictionary can have the following entry:

        * <str>:list
            the string used as the key specifies the name to be used for the GatingSignal,
            and each item of the list must be a `specification of a State <State_Creation>` to be
            gated by the GatingSignal (and that will receive a `GatingProjection` from it).

    The dictionary can also contain entries for any other GatingSignal attributes to be specified
    (e.g., a *MODULATION* entry, the value of which determines how the GatingSignal modulates the
    `value <State_Base.value>` of the State(s) that it gates; or an *INDEX* entry specifying which item
    of the GatingMechanism's `gating_policy <GatingMechanism.gating_policy>` it should use as its `value
    <GatingSignal,value>`).
  ..
  * **2-item tuple** -- the 1st item must be the name of the State (or list of State names), and the 2nd item the
    Mechanism to which it (they) belong(s); this is a convenience format, which is simpler to use than a specification
    dictionary (see below), but precludes specification of `parameters <GatingSignal_Structure>` for the GatingSignal.

.. _GatingSignal_Structure:

Structure
---------

A GatingSignal is owned by a `GatingMechanism`, and associated with one or more `GatingProjections <GatingProjection>`,
each of which projects to the InputState or OutputState that it gates.

.. _GatingSignal_Projections:

Projections
~~~~~~~~~~~

When a GatingSignal is created, it can be assigned one or more `GatingProjections <GatingProjection>`, using either
the **projections** argument of its constructor, or in an entry of a dictionary assigned to the **params** argument
with the key *PROJECTIONS*.  These will be assigned to its `efferents  <GatingSignal.efferents>` attribute.  See
`State Projections <State_Projections>` for additional details concerning the specification of Projections when
creating a State.

.. note::
   Although a GatingSignal can be assigned more than one `GatingProjection`, all of those Projections will convey
   the same `gating_signal <LearningMechanism>` (received from the GatingMechanism to which the GatingSignal
   belongs), and use the same form of `modulation <GatingSignal_Modulation>`.  This can be useful for implementing
   broadly projecting modulatory effects.

.. _GatingSignal_Modulation:

Modulation
~~~~~~~~~~

Each GatingSignal has a `modulation <GatingSignal.modulation>` attribute that determines how the GatingSignal's
`value <GatingSignal.value>` (i.e., its `gating_signal <GatingSignal.gating_signal>`) is used by the States to which it
projects to modify their `value <State_Base.value>` \\s (see `ModulatorySignal_Modulation` for an explanation of how the
`modulation <GatingSignal.modulation>` attribute is specified and used to modulate the `value <State_Base.value>` of a
State). The `modulation <GatingSignal.modulation>` attribute can be specified in the **modulation** argument of the
constructor for a GatingSignal, or in a specification dictionary as described `above <GatingSignal_Specification>`.
The value must be a value of `ModulationParam`;  if it is not specified, its default is the value of the `modulation
<GatingMechanism.modulation>` attribute of the GatingMechanism to which the GatingSignal belongs (which is the same
for all of the GatingSignals belonging to that GatingMechanism).  The value of the
`modulation <GatingSignal.modulation>` attribute of a GatingSignal is used by all of the
`GatingProjections <GatingProjection>` that project from that GatingSignal.


.. _GatingSignal_Execution:

Execution
---------

A GatingSignal cannot be executed directly.  It is executed whenever the `GatingMechanism` to which it belongs is
executed.  When this occurs, the GatingMechanism provides the GatingSignal with one of the values from its
`gating_policy <GatingMechanism.gating_signal>`, that is used by its `function <GatingSignal.function>` to generate its
the value of its `gating_signal <GatingSignal.gating_signal>`.  That, in turn, is used by its `GatingProjection(s)
<GatingProjection>` to modulate the `value <State_Base.value>` of the States to which they project. How the modulation
is executed is determined by the GatingSignal's `modulation <GatingSignal.modulation>` attribute
(see `above <GatingSigna_Modulation>`, and `ModulatorySignal_Modulation` for a more detailed explanation of how
modulation operates).

.. note::
   The change in the `value <State_Base.value>` of InputStates and OutputStates in response to the execution of a
   GatingSignal are not applied until the Mechanism(s) to which those states belong are next executed;
   see :ref:`Lazy Evaluation <LINK>` for an explanation of "lazy" updating).

.. _GatingSignal_Examples:

Examples
~~~~~~~~

**Gate an InputState and OutputState**.  In the following example, `GatingMechanism` is configured to gate the
`primary InputState <InputState_Primary>` of one Mechanism, and the `primary OutputState <OutputState_Primary>`
of another::

    >>> import psyneulink as pnl
    >>> my_mechanism_A = pnl.TransferMechanism(name="Mechanism A")
    >>> my_mechanism_B = pnl.TransferMechanism(name="Mechanism B")
    >>> my_gating_mechanism = pnl.GatingMechanism(gating_signals=[my_mechanism_A.input_state,
    ...                                                           my_mechanism_B.output_state])

Note that, in the **gating_signals** argument, the first item references a Mechanism (``my_mechanism_A``) rather than
one of its states -- this is all that is necessary, since the default for a `GatingSignal` is to modulate the
`primary InputState <InputState_Primary>` of a Mechanism.  The second item explicitly specifies the State to be gated,
since it is an OutputState.  This will generate two GatingSignals, each of which will multiplicatively modulate the
value of the State to which it projects.  This is because, by default, the `modulation <GatingSignal.modulation>`
attribute of a GatingSignal is the *MULTIPLICATIVE_PARAM* for the `function <State_Base.function>` of the State to which
it projects.  For an InputState, the default `function <InputState.function>` is `Linear` and its *MULTIPLICATIVE_PARAM*
is its `slope <Linear.slope>` parameter.  Thus, the value of the GatingSignal is assigned to the slope, which multiplies
the State`s `variable <State_Base.variable>` (i.e., its input(s)) to determine its `value <State_Base.value>`.

**Modulate the InputStates of several Mechanisms**.  In next example, a `GatingMechanism` is created that modulates
the `InputState` of all the layers in a 3-layered feedforward neural network.  Ordinarily, gating modulates the
*MULTIPLICATIVE_PARAM* of an InputState's `function <InputState.function>`.  In the example, this is changed so that
it *adds* the `value <GatingSignal.value>` of the `GatingSignal` to the `value <InputState.value>` of each InputState::

    >>> my_input_layer = pnl.TransferMechanism(size=3)
    >>> my_hidden_layer = pnl.TransferMechanism(size=5)
    >>> my_output_layer = pnl.TransferMechanism(size=2)
    >>> my_gating_mechanism = pnl.GatingMechanism(gating_signals=[{pnl.NAME: 'GATE_ALL',
    ...                                                            pnl.PROJECTIONS: [my_input_layer,
    ...                                                                              my_hidden_layer,
    ...                                                                              my_output_layer]}],
    ...                                           modulation=pnl.ModulationParam.ADDITIVE)

Note that, again, the **gating_signals** are listed as Mechanisms, since in this case it is their primary InputStates
that are to be gated. Since they are all listed in a single entry of a
`specification dictionary <GatingSignal_Specification>`, they will all be gated by a single GatingSignal named
``GATE_ALL``, that will send a `GatingProjection` to the InputState of each of the Mechanisms listed (the next example
shows how different InputStates can be differentially gated by a `GatingMechanism`). Finally, note that the
`ModulationParam` specified for the `GatingMechanism` (and therefore the default for its GatingSignals) pertains to
the `function <InputState.function>` of each `InputState`. By default that is a `Linear` function, the *ADDITIVE_PARAM*
of which is its `intercept <Linear.intercept>` parameter. Therefore, in the example above, each time the InputStates
are updated, the value of the GatingSignal will be assigned as the `intercept` of each InputState's
`function <InputState.function>`, thus adding that amount to the input to the State before determining its
`value <InputState.value>`.

**Gate InputStates differentially**.  In the example above, the InputStates for all of the Mechanisms were gated
using a single GatingSignal.  In the example below, a different GatingSignal is assigned to the InputState of each
Mechanism::

    >>> my_gating_mechanism = pnl.GatingMechanism(gating_signals=[{pnl.NAME: 'GATING_SIGNAL_A',
    ...                                                            pnl.MODULATION: pnl.ModulationParam.ADDITIVE,
    ...                                                            pnl.PROJECTIONS: my_input_layer},
    ...                                                           {pnl.NAME: 'GATING_SIGNAL_B',
    ...                                                            pnl.PROJECTIONS: [my_hidden_layer,
    ...                                                                              my_output_layer]}])

Here, two GatingSignals are specified as `specification dictionaries <GatingSignal_Specification>`, each of which
contains an entry for the name of the GatingSignal, and a *PROJECTIONS* entry that specifies the States to which the
GatingSignal should project (i.e., the ones to be gated).  Once again, the specifications exploit the fact that the
default is to gate the `primary InputState <InputState_Primary>` of a Mechanism, so those are what are referenced. The
first dict also contains a  *MODULATION* entry that specifies the value of the `modulation <GatingSignal.modulation>`
attribute for the GatingSignal.  The second one does not, so the default will be used (which, for a GatingSignal, is
`ModulationParam.MULTIPLICATIVE`).  Thus, the InputState of ``my_input_layer`` will be additively modulated by
``GATING_SIGNAL_A``, while the InputStates of ``my_hidden_layer`` and ``my_output_layer`` will be multiplicatively
modulated by ``GATING_SIGNAL_B``.

**Creating and assigning stand-alone GatingSignals**.  GatingSignals can also be created on their own, and then later
assigned to a GatingMechanism.  In the example below, the same GatingSignals specified in the previous example are
created directly and then assigned to ``my_gating_mechanism``::

    >>> my_gating_signal_A = pnl.GatingSignal(name='GATING_SIGNAL_A',
    ...                                       modulation=pnl.ModulationParam.ADDITIVE,
    ...                                       projections=my_input_layer)
    >>> my_gating_signal_B = pnl.GatingSignal(name='GATING_SIGNAL_B',
    ...                                       projections=[my_hidden_layer,
    ...                                                    my_output_layer])
    >>> my_gating_mechanism = pnl.GatingMechanism(gating_signals=[my_gating_signal_A,
    ...                                                           my_gating_signal_B])


Class Reference
---------------

"""

import typecheck as tc

from psyneulink.components.functions.function import Linear, LinearCombination, _is_modulation_param
from psyneulink.components.mechanisms.mechanism import Mechanism
from psyneulink.components.states.state import State_Base, _parse_state_type, _get_state_for_socket
from psyneulink.components.states.inputstate import InputState
from psyneulink.components.states.outputstate import OutputState, PRIMARY, SEQUENTIAL
from psyneulink.components.states.modulatorysignals.modulatorysignal import ModulatorySignal, modulatory_signal_keywords
from psyneulink.globals.keywords import \
    COMMAND_LINE, GATING_PROJECTION, GATING_SIGNAL, GATE, RECEIVER, SUM, PROJECTION_TYPE, \
    INPUT_STATE, INPUT_STATES, OUTPUT_STATE, OUTPUT_STATES, OUTPUT_STATE_PARAMS, PROJECTIONS
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.globals.preferences.preferenceset import PreferenceLevel

__all__ = [
    'gating_signal_keywords', 'GatingSignal', 'GatingSignalError',
]


class GatingSignalError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

gating_signal_keywords = {GATE}
gating_signal_keywords.update(modulatory_signal_keywords)


class GatingSignal(ModulatorySignal):
    """
    GatingSignal(                                   \
        owner,                                      \
        index=PRIMARY                  \
        function=LinearCombination(operation=SUM),  \
        modulation=ModulationParam.MULTIPLICATIVE,  \
        projections=None,                           \
        params=None,                                \
        name=None,                                  \
        prefs=None)

    A subclass of `ModulatorySignal <ModulatorySignal>` used by a `GatingMechanism` to modulate the value(s)
    of one more `InputState(s) <InputState>` and/or `OutputState(s) <OutputState>`.

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
                + FUNCTION_PARAMS (Modulation.MULTIPLY)

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

    index : int : default PRIMARY
        specifies the item of the owner GatingMechanism's `gating_policy <GatingMechanism.gating_policy>` used as the
        GatingSignal's `value <GatingSignal.value>`.

    function : Function or method : default Linear
        specifies the function used to determine the value of the GatingSignal from the value of its
        `owner <GatingMechanism.owner>`.

    modulation : ModulationParam : default ModulationParam.MULTIPLICATIVE
        specifies the way in which the `value <GatingSignal.value>` the GatingSignal is used to modify the
        `value <State_Base.value>` of the State(s) to which the GatingSignal's
        `GatingProjection(s) <GatingProjection>` project.

    projections : list of Projection specifications
        specifies the `GatingProjection(s) <GatingProjection>` to be assigned to the GatingSignal, and that will be
        listed in its `efferents <GatingSignal.efferents>` attribute (see `GatingSignal_Projections` for additional
        details).

    params : Dict[param keyword, param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the ControlSignal and/or a custom function and its parameters. Values specified for parameters in the dictionary
        override any assigned to those parameters in arguments of the constructor.

    name : str : default see ModulatorySignal `name <ModulatorySignal.name>`
        specifies the name of the GatingSignal;  see GatingSignal `name <ModulatorySignal.name>` for additional
        details.

    prefs : PreferenceSet or specification dict : default State.classPreferences
        specifies the `PreferenceSet` for the GatingSignal; see `prefs <GatingSignal.prefs>` for details.


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
        result of the GatingSignal's `function <GatingSignal.function>`
        (same as its `gating_signal <GatingSignal.gating_signal>`).

    index : int
        the item of the owner GatingMechanism's `gating_policy <GatingMechanism.gating_policy>` used as the
        GatingSignal's `value <GatingSignal.value>`.

    gating_signal : number, list or np.ndarray
        result of the GatingSignal's `function <GatingSignal.function>` (same as its `value <GatingSignal.value>`).


    modulation : ModulationParam
        determines the way in the which `value <GatingSignal.value>` of the GatingSignal is used to modify the
        `value <State_Base.value>` of the State(s) to which the GatingSignal's
        `GatingProjection(s) <GatingProjection>` project.

    efferents : [List[GatingProjection]]
        a list of the `GatingProjections <GatingProjection>` assigned to (i.e., that project from) the GatingSignal.

    name : str
        name of the GatingSignal; if not is specified in the **name** argument of its constructor, a default name
        is assigned (see `name <ModulatorySignal.name>`).

        .. note::
            Unlike other PsyNeuLink components, State names are "scoped" within a Mechanism, meaning that States with
            the same name are permitted in different Mechanisms.  However, they are *not* permitted in the same
            Mechanism: States within a Mechanism with the same base name are appended an index in the order of their
            creation.

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the GatingSignal; if it is not specified in the **prefs** argument of the constructor,
        a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet <LINK>` for
        details).

    """

    #region CLASS ATTRIBUTES

    componentType = GATING_SIGNAL
    componentName = 'GatingSignal'
    paramsType = OUTPUT_STATE_PARAMS

    stateAttributes = ModulatorySignal.stateAttributes | {GATE}

    connectsWith = [INPUT_STATE, OUTPUT_STATE]
    connectsWithAttribute = [INPUT_STATES, OUTPUT_STATES]
    projectionSocket = RECEIVER
    modulators = []

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'OutputStateCustomClassPreferences',
    #     kp<pref>: <setting>...}

    paramClassDefaults = State_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        PROJECTION_TYPE: GATING_PROJECTION,
        GATE:None,
    })
    #endregion

    @tc.typecheck
    def __init__(self,
                 owner=None,
                 reference_value=None,
                 variable=None,
                 size=None,
                 index=None,
                 calculate=Linear,
                 function=LinearCombination(operation=SUM),
                 modulation:tc.optional(_is_modulation_param)=None,
                 projections=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        if context is None:
            context = COMMAND_LINE
        else:
            context = self

        # Note: calculate is not currently used by GatingSignal;
        #       it is included here for consistency with OutputState and possible use by subclasses.
        if index is None and owner is not None:
            if len(owner.gating_policy)==1:
                index = PRIMARY
            else:
                index = SEQUENTIAL

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function,
                                                  params=params)

        # FIX: 5/26/16
        # IMPLEMENTATION NOTE:
        # Consider adding self to owner.output_states here (and removing from GatingProjection._instantiate_sender)
        #  (test for it, and create if necessary, as per OutputStates in GatingProjection._instantiate_sender),

        # Validate sender (as variable) and params, and assign to variable and paramInstanceDefaults
        super().__init__(owner=owner,
                         reference_value=reference_value,
                         variable=variable,
                         size=size,
                         modulation=modulation,
                         index=index,
                         calculate=calculate,
                         projections=projections,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)

    def _parse_state_specific_specs(self, owner, state_dict, state_specific_spec):
            """Get connections specified in a ParameterState specification tuple
    
            Tuple specification can be:
                (State name, Mechanism)
            [TBI:] (Mechanism, State name, weight, exponent, projection_specs)

            Returns params dict with CONNECTIONS entries if any of these was specified.
    
            """
            from psyneulink.components.projections.projection import _parse_connection_specs
    
            params_dict = {}
            state_spec = state_specific_spec
    
            if isinstance(state_specific_spec, dict):
                return None, state_specific_spec

            elif isinstance(state_specific_spec, tuple):
                state_spec = None
                params_dict[PROJECTIONS] = _parse_connection_specs(connectee_state_type=self,
                                                                   owner=owner,
                                                                   connections=state_specific_spec)
            elif state_specific_spec is not None:
                raise GatingSignalError("PROGRAM ERROR: Expected tuple or dict for {}-specific params but, got: {}".
                                      format(self.__class__.__name__, state_specific_spec))
    
            if params_dict[PROJECTIONS] is None:
                raise GatingSignalError("PROGRAM ERROR: No entry found in {} params dict for {} "
                                         "with specification of {}, {} or GatingProjection(s) to it".
                                            format(GATING_SIGNAL, INPUT_STATE, OUTPUT_STATE, owner.name))
            return state_spec, params_dict

    def _execute(self, function_params, context):
        return float(super()._execute(function_params=function_params, context=context))

    def _get_primary_state(self, mechanism):
        return mechanism.input_state

    @property
    def gating_signal(self):
        return self.value
