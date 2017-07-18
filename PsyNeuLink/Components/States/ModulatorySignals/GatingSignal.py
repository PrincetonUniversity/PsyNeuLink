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

A GatingSignal is a type of `ModulatorySignal` that is specialized for use with a `GatingMechanism` and one or more
`GatingProjections <GatingProjection>`, to modify the `value <State.value>` \(s) of the InputState(s) and/or
OutputState(s) to which they project. A GatingSignal receives a value from the GatingMechanism to which it belongs,
and assigns that to its `GatingProjection(s) <GatingProjection>`, each of which projects to an InputState or
OutputState and is used to modulate the `value <State.value>` of that State.


.. _GatingSignal_Creation:

Creating a GatingSignal
-----------------------

A GatingSignal is created automatically whenever an `InputState` or `OutputState` of a `Mechanism` is specified for
gating.  This can be done either in the **gating_signals** argument of the constructor for a
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

  * an **InputState** or **OutputState** of a Mechanism;
  ..
  * a **Mechanism**, in which case its `primary `InputState <InputState_Primary>` or `OutputState <OutputState_Primary>`
    is used;
  ..
  * a **tuple**, with the name of the state as the 1st item, and the Mechanism to which it belongs as the 2nd;
    note that this is a convenience format, which is simpler to use than a specification dictionary (see below), 
    but precludes specification of any `parameters <GatingSignal_Structure>` for the GatingSignal.
  ..
  * a **specification dictionary**, that can take either of the following two forms:

    * for a single state, the dictionary must have the following two entries:

        * *NAME*: str
            the string must be the name of the State to be gated; the GatingSignal will named by appending
            "_GatingSignal" to the name of the State.

        * *MECHANISM*: Mechanism
            the Mechanism must be the one to the which the State to be gated belongs.

    * for multiple states, the dictionary must have the following entry:

        * <str>:list
            the string used as the key specifies the name to be used for the GatingSignal,
            and each item of the list must be a `specification of a State <State_Creation>` to be
            gated by the GatingSignal (and that will receive a `GatingProjection` from it).

    The dictionary can also contain entries for any other GatingSignal attributes to be specified
    (e.g., a *MODULATION* entry, the value of which determines how the GatingSignal modulates the
    `value <State.value>` of the State(s) that it gates); see `below <GatingSignal_Structure>`
    for a description of GatingSignal attributes.

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

.. _GatingSignal_Modulation:

Modulation
~~~~~~~~~~

Each GatingSignal has a `modulation <GatingSignal.modulation>` attribute that determines how the GatingSignal's
`value <GatingSignal.value>` is used by the States to which it projects to modify their `value <State.value>` \s
(see `ModulatorySignal_Modulation` for an explanation of how the `modulation <GatingSignal.modulation>` attribute is
specified and used to modulate the `value <State.value>` of a State). The `modulation <GatingSignal.modulation>`
attribute can be specified in the **modulation** argument of the constructor for a GatingSignal, or in a specification
dictionary as described `above <GatingSignal_Specification>`.  The value must be a value of `ModulationParam`;  if it
is not specified, its default is the value of the `modulation <GatingMechanism.modulation>` attribute of the
GatingMechanism to which the GatingSignal belongs (which is the same for all of the GatingSignals belonging to that
GatingMechanism).  The value of the `modulation <GatingSignal.modulation>` attribute of a GatingSignal is used by all
of the `GatingProjections <GatingProjection>` that project from that GatingSignal.


.. _GatingSignal_Execution:

Execution
---------

A GatingSignal cannot be executed directly.  It is executed whenever the `GatingMechanism` to which it belongs is
executed.  When this occurs, the GatingMechanism provides the GatingSignal with a value that is used by its 
`GatingProjection(s) <GatingProjection>` to modulate the `value <State.value>` of the states to which they project.
Those States use the `value <GatingProjection.valu>` of the `GatingProjection` they receive to modify a parameter of
their function.  How the modulation is executed is determined by the GatingSignal's
`modulation <GatingSignal.modulation>` attribute (see `above <GatingSigna_Modulation>`, and
`ModulatorySignal_Modulation` for a more detailed explanation of how modulation operates).

.. note::
   The change in the `value <State.value>` of InputStates and OutputStates in response to the execution of a
   GatingMechanism are not applied until the Mechanism(s) to which those states belong are next executed;
   see :ref:`Lazy Evaluation <LINK>` for an explanation of "lazy" updating).

.. _GatingSignal_Examples:

Examples
~~~~~~~~

**Gate an InputState and OutputState**.  In the following example, `GatingMechanism` is configured to gate the
`primary InputState <InputState_Primary>` of one Mechanism, and the `primary OutputState <OutputState_Primary>`
of another::

    my_mechanism_A = TransferMechanism()
    my_mechanism_B = TransferMechanism()
    my_gating_mechanism = GatingMechanism(gating_signals=[my_mechanism_A, my_mechanism_B.output_state)

Note that, in the **gating_signals** argument, the first item references a Mechanism (``my_mechanism_A``) rather than
one of its states -- this is all that is necessary, since the default for a `GatingSignal` is to modulate the
`primary InputState <InputState_Primary>` of a Mechanism.  The second item explicitly specifies the State to be gated,
since it is an OutputState.  This will generate two GatingSignals, each of which will multiplicatively modulate the
value of the State to which it projects.  This is because, by default, the `modulation <GatingSignal.modulation>`
attribute of a GatingSignal is the *MULTIPLICATIVE_PARAM* for the `function <State.function>` of the State to which it
projects.  For an InputState, the default `function <InputState.function>` is `Linear` and its *MULTIPLICATIVE_PARAM*
is its `slope <Linear.slope>` parameter.  Thus, the value of the GatingSignal is assigned to the slope,
which multiplies the State`s `variable <State.variable>` (i.e., its input(s)) to determine its `value <State.value>`.

**Modulate the InputStates of several Mechanisms**.  In next example, a `GatingMechanism` is created that modulates
the `InputState` of all the layers in a 3-layered feedforward neural network.  Ordinarily, gating modulates the
*MULTIPLICATIVE_PARAM* of an InputState's `function <InputState.function>`.  In the example, this is changed so that
it adds the `value <GatingSignal.value>` of the `GatingSignal` to the `value <InputState.value>` of each InputState::

    my_input_layer = TransferMechanism(size=3)
    my_hidden_layer = TransferMechanism(size=5)
    my_output_layer = TransferMechanism(size=2)
    my_gating_mechanism = GatingMechanism(gating_signals=[
                                            {'GATE_ALL': [my_input_layer,
                                                          my_hidden_layer,
                                                          my_output_layer]},
                                             modulation=ModulationParam.ADDITIVE)

Note that, again, the **gating_signals** are listed as Mechanisms, since in this case it is their primary InputStates
that are to be gated. Since they are all listed in a single entry of a
`specification dictionary <_GatingSignal_Specification>`, they will all be gated by a single GatingSignal named
``GATE_ALL``, that will send a `GatingProjection` to the InputState of each of the Mechanisms listed (the next example
shows how different InputStates can be differentially gated by a `GatingMechanism`). Finally, note that the
`ModulationParam` specified for the `GatingMechanism` (and therefore the default for its GatingSignals, pertains to
the `function <InputState.function>` of each `InputState`. By default that is a `Linear` function, the *ADDITIVE_PARAM*
of which is its `intercept <Linear.intercept>` parameter. Therefore, in the example above, each time the InputStates
are updated, the value of the GatingSignal will be assigned as the `intercept` of each InputState's
`function <InputState.function>`, thus adding that amount to the input to the State before determining its
`value <InputStat.value>`.

**Gate InputStates differentially**.  In the example above, the InputStates for all of the Mechanisms were gated
using a single GatingSignal.  In the example below, a different GatingSignal is assigned to the InputState of each
Mechanism::

    my_gating_mechanism = GatingMechanism(gating_signals=[{NAME: 'GATING_SIGNAL_A',
                                                           MODULATION: ModulationParam.ADDITIVE,
                                                           PROJECTIONS: my_input_layer},
                                                          {NAME: 'GATING_SIGNAL_B',
                                                           PROJECTIONS: [my_hidden_layer,
                                                                         my_output_layer]}])

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

    my_gating_signal_A = GatingSignal(name='GATING_SIGNAL_A',
                                      modulation=ModulationParams.ADDITIVE,
                                      projections=my_input_layer)
    my_gating_signal_B = GatingSignal(name='GATING_SIGNAL_B',
                                      projections=my_hidden_layer,
                                                  my_output_layer)
    my_gating_mechanism = GatingMechanism(gating_signals=[my_gating_signal_A,
                                                          my_gating_signal_B])


Class Reference
---------------

"""

from PsyNeuLink.Components.Functions.Function import _is_modulation_param
from PsyNeuLink.Components.States.InputState import InputState
from PsyNeuLink.Components.States.OutputState import OutputState, PRIMARY_OUTPUT_STATE
from PsyNeuLink.Components.States.ModulatorySignals.ModulatorySignal import *


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
        function=LinearCombination(operation=SUM),  \
        modulation=ModulationParam.MULTIPLICATIVE,  \
        projections=None,                           \
        params=None,                                \
        name=None,                                  \
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

    function : Function or method : default Linear
        specifies the function used to determine the value of the GatingSignal from the value of its 
        `owner <GatingMechanism.owner>`.

    modulation : ModulationParam : default ModulationParam.MULTIPLICATIVE
        specifies the way in which the `value <GatingSignal.value>` the GatingSignal is used to modify the `value
        <State.value>` of the State(s) to which the GatingSignal's `GatingProjection(s) <GatingProjection>` project.

    projections : list of Projection specifications
        specifies the `GatingProjection(s) <GatingProjection>` to be assigned to the GatingSignal, and that will be
        listed in its `efferents <GatingSignal.efferents>` attribute (see `GatingSignal_Projections` for additional
        details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the ControlSignal and/or a custom function and its parameters. Values specified for parameters in the dictionary
        override any assigned to those parameters in arguments of the constructor.

    name : str : default OutputState-<index>
        a string used for the name of the OutputState.
        If not is specified, a default is assigned by the StateRegistry of the mechanism to which the OutputState
        belongs (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : State.classPreferences]
        the `PreferenceSet` for the OutputState.
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
    
    modulation : ModulationParam
        determines the way in the which `value <GatingSignal.value>` the GatingSignal is used to modify the `value
        <State.value>` of the State(s) to which the GatingSignal's `GatingProjection(s) <GatingProjection>` project.

    efferents : [List[GatingProjection]]
        a list of the `GatingProjections <GatingProjection>` assigned to (i.e., that project from) the GatingSignal.

    name : str : default <State subclass>-<index>
        name of the OutputState.
        Specified in the **name** argument of the constructor for the OutputState.  If not is specified, a default is
        assigned by the StateRegistry of the mechanism to which the OutputState belongs
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

        .. note::
            Unlike other PsyNeuLink components, state names are "scoped" within a mechanism, meaning that states with
            the same name are permitted in different mechanisms.  However, they are *not* permitted in the same
            mechanism: states within a mechanism with the same base name are appended an index in the order of their
            creation.

    prefs : PreferenceSet or specification dict : State.classPreferences
        the `PreferenceSet` for the OutputState.
        Specified in the **prefs** argument of the constructor for the projection;  if it is not specified, a default is
        assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    #region CLASS ATTRIBUTES

    componentType = OUTPUT_STATES
    componentName = 'GatingSignal'
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
        GATE:None,
    })
    #endregion

    @tc.typecheck
    def __init__(self,
                 owner=None,
                 reference_value=None,
                 variable=None,
                 size=None,
                 index=PRIMARY_OUTPUT_STATE,
                 calculate=Linear,
                 function=LinearCombination(operation=SUM),
                 modulation:tc.optional(_is_modulation_param)=None,
                 projections=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Note: index and calculate are not used by GatingSignal;
        #       they are included here for consistency with OutputState and possible use by subclasses.

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function,
                                                  params=params)

        # FIX: 5/26/16
        # IMPLEMENTATION NOTE:
        # Consider adding self to owner.outputStates here (and removing from GatingProjection._instantiate_sender)
        #  (test for it, and create if necessary, as per OutputStates in GatingProjection._instantiate_sender),

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
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
                         context=self)

    def _execute(self, function_params, context):
        return float(super()._execute(function_params=function_params, context=context))

def _parse_gating_signal_spec(owner, state_spec):
    """Take specifications for one or more states to be gated, and return GatingSignal specification dictionary

    state_spec can take any of the following forms:
        - an existing GatingSignal
        - an existing InputState or OutputState for a Mechanisms in self.system
        - a list of state specifications (see below)
        - a dictionary that contains either a:
            - single state specification:
                NAME:str - contains the name of an InputState or OutputState belonging to MECHANISM
                MECHANISM:Mechanism - contains a reference to a Mechanism in self.system that owns NAME'd state
                <PARAM_KEYWORD>:<GatingSignal param value>
            - multiple state specification:
                NAME:str - used as name of GatingSignal
                STATES:List[tuple, dict] - each item must be state specification tuple or dict
                <PARAM_KEYWORD>:<GatingSignal param value>
    
    Each state specification must be a:
        - (str, Mechanism) tuple
        - {NAME:str, MECHANISM:Mechanism} dict
        where:
            str is the name of an InputState or OutputState of the Mechanism,
            Mechanism is a reference to an existing that belongs to self.system 
    
    Checks for duplicate state specifications within state_spec or with any existing GatingSignal of the owner
        (i.e., states that will receive more than one GatingProjection from the owner)
        
    If state_spec is already a GatingSignal, it is returned (in the GATING_SIGNAL entry) along with its parsed elements 
    
    Returns dictionary with the following entries:
        NAME:str - name of either the gated state (if there is only one) or the GatingSignal
        STATES:list - list of states to be gated
        GATING_SIGNAL:GatingSignal or None
        PARAMS:dict - params dict if any were included in the state_spec
    """
    
    from PsyNeuLink.Components.States.ModulatorySignals.GatingSignal import GatingSignal
    from PsyNeuLink.Components.Projections.Projection import _validate_receiver
    from PsyNeuLink.Components.Projections.ModulatoryProjections.GatingProjection import GatingProjection

    GATING_SIGNAL_SUFFIX = '_' + GatingSignal.__name__
    DEFAULT_GATING_SIGNAL_NAME = 'Default'+ GATING_SIGNAL_SUFFIX

    gating_signal = None
    states = []
    params = {}
    state_name = None
    mech = None

    # def _get_default_gating_signal_name():
    #     """Return default name for gating signal
    #     Default_GatingSignal if it is the first,
    #     Default_GatingSignal-2 for second
    #     Default_GatingSignal-3 for third, etc.
    #     """
    #
    #     # FIX: USE REGISTRY HERE (AS FOR OUTPUTSTATE REGISTRY ON MECHANISM)
    #     if owner.gating_signals:
    #         # # Get the number of existing gating_signals with the default name
    #         # index = len([gs for gs in owner.gating_signals if ('Default'+ GATING_SIGNAL_SUFFIX) in gs.name])
    #     else:
    #         index = ''
    #     if index:
    #         index = repr(index+1)
    #     return 'Default'+GATING_SIGNAL_SUFFIX+index

    # Specification is for a GatingSignal
    if isinstance(state_spec, GatingSignal):
        gating_signal = state_spec
        # GatingSignal initialization has been deferred, so just get name and return
        if gating_signal.value is DEFERRED_INITIALIZATION:
            gating_signal_name = gating_signal.init_args[NAME]
            return {NAME: gating_signal_name,
                    STATES: [],
                    PARAMS: [],
                    GATING_SIGNAL: gating_signal}
        else:
            gating_signal_name = gating_signal.name
            states = []
            for proj in gating_signal.efferents:
                _validate_receiver(owner, proj, Mechanism, GATING_SIGNAL)
                states.append(proj.receiver.owner)
            if not states:
                raise GatingSignalError("Attempt to assign an existing {} to {} that has no GatingProjections".
                                           format(GATING_SIGNAL, owner.name))

    # For all other specs:
    #    - if it is a single spec (state name and mech):
    #        validate that the mech is in self.system, has a state of that name, and then return the state
    #    - if it is a list:
    #        iterate through list, calling _parse_gating_signal_spec recursively, to build up the list of states

    # Specification is for an existing GatingProjection
    #    so check if it is to a state of a mechanism in self.system
    elif isinstance(state_spec, GatingProjection):
        _validate_receiver(owner, state_spec, Mechanism, GATING_SIGNAL)
        state_name = state_spec.receiver.name
        gating_signal_name = state_name + GATING_SIGNAL_SUFFIX
        mech = state_spec.reciever.owner

    # Specification is for an existing InputState or OutputState,
    #    so check that it's owner belongs to self.system
    elif isinstance(state_spec, (InputState, OutputState)):
        # if not state_spec.owner.system in owner.system.mechanisms:
        # # IMPLEMENTATION NOTE: REINSTATE WHEN ASSIGNMENT OF GatingMechanism TO SYSTEM IS RESOLVED (IN COMPOSITION??)
        # if not (set(state_spec.owner.systems) & set(owner.systems)):
        #     raise GatingSignalError("The State specified in the {} arg for {} ({}) "
        #                                 "belongs to a mechanism that is not in the same system ({})".
        #                                 format(GATING_SIGNALS, owner.name,
        #                                        state_spec.name,
        #                                        state_spec.owner.systems))
        state_name = state_spec.name
        gating_signal_name = state_name + GATING_SIGNAL_SUFFIX
        mech = state_spec.owner

    # Specification is for a Mechanism,
    #    so check that it belongs to the same system as self
    #    and use primary InputState as the default
    elif isinstance(state_spec, Mechanism):
        # if state_spec.system and not state_spec.system in owner.system.mechanisms:
        # # IMPLEMENTATION NOTE: REINSTATE WHEN ASSIGNMENT OF GatingMechanism TO SYSTEM IS RESOLVED (IN COMPOSITION??)
        # if state_spec.systems and not (set(state_spec.systems) & set(owner.systems)):
        #     raise GatingSignalError("The Mechanism specified in the {} arg for {} ({}) "
        #                                 "does not belong to the same system ({})".
        #                                 format(GATING_SIGNALS, owner.name,
        #                                        state_spec.name,
        #                                        state_spec.owner.systems))
        mech = state_spec
        state_spec = state_spec.input_states[0]
        state_name = state_spec.name
        gating_signal_name = state_name + GATING_SIGNAL_SUFFIX

    elif isinstance(state_spec, tuple):
        state_name = state_spec[0]
        mech = state_spec[1]
        gating_signal_name = state_name + GATING_SIGNAL_SUFFIX
        # Check that 1st item is a str (presumably the name of one of the mechanism's states)
        if not isinstance(state_name, str):
            raise GatingSignalError("1st item of specification tuple for the state to be gated by {} of {} ({})"
                                       "must be a string that is the name of the state".
                                       format(GATING_SIGNAL, owner.name, state_name))
        # Check that 2nd item is a mechanism
        if not isinstance(mech, Mechanism):
            raise GatingSignalError("2nd item of specification tuple for the state to be gated by {} of {} ({})"
                                       "must be a Mechanism that is the mech to which the state {} belongs".
                                       format(GATING_SIGNAL, owner.name, mech, state_name))

    # Specification is a list, presumably of one or more states specs
    elif isinstance(state_spec, list):
        # Validate each item in the list (which should be a state state_spec), and
        #    - add the state(s) returned to state list
        #    - assign state_name as None,
        #        since there is no single name that can be used as the name for the GatingSignal
        gating_signal_name = DEFAULT_GATING_SIGNAL_NAME
        for spec in state_spec:
            spec_dict = _parse_gating_signal_spec(owner, spec)
            states.extend(spec_dict[STATES])

    # Specification is a dict that could be for a single state state_spec or a list of ones
    elif isinstance(state_spec, dict):

        # FIX: IS THIS NECESSARY? (GIVEN THE FUNCTIONALITY UNDER 'ELSE':  USE KEY AS NAME AND VALUE AS LIST OF STATES)
        # If it has a STATES entry, it must be for a list
        if STATES in state_spec:
            # Validate that the STATES entry has a list
            state_specs = state_spec[STATES]
            if not isinstance(state_specs, list):
                raise GatingSignalError("The {} entry of the dict in the {} arg for {} must be "
                                           "a list of state specifications".
                                            format(STATES, GATING_SIGNALS, owner.name))
            # Validate each item in the list (which should be a state state_spec), and
            #    - add the state(s) returned to state list
            #    - assign state_name to the NAME entry
            #        (which will be used as the name for the GatingSignal in _instantiate_gating_signal)
            for spec in state_specs:
                spec_dict = _parse_gating_signal_spec(owner, spec)
                states.extend(spec_dict[STATES])
            if NAME in state_spec:
                state_name = state_spec[NAME]
                gating_signal_name = state_name
            else:
                gating_signal_name = DEFAULT_GATING_SIGNAL_NAME

        # If it doesn't have a STATES entry
        else:
            # If there is a non-keyword key, treat as the name to be used for the GatingSignal,
            #    and the value a state spec or list of ones
            state_name = next((key for key in state_spec if not key in gating_signal_keywords), None)
            key_as_name = explicit_name = None
            if state_name:
                key_as_name = state_name
                spec_dict = _parse_gating_signal_spec(owner, state_spec[key_as_name])
                states = spec_dict[STATES]
                # If there *IS* a NAME entry, then use that (i.e., override key as the name)
                if NAME in state_spec:
                    explicit_name = state_spec[NAME]
                gating_signal_name = explicit_name or key_as_name
            # Otherwise, it must be for a single state state_spec,
            #    which means it must have a NAME and a MECHANISM entry:
            else:
                if not NAME in state_spec:
                    raise GatingSignalError("Specification dict for the state to be gated by {} of {} must have a "
                                               "NAME entry that is the name of the state".
                                               format(GATING_SIGNAL, owner.name))
                state_name = state_spec[NAME]
                gating_signal_name = state_name + GATING_SIGNAL_SUFFIX

                # GatingSignal projects to a single state (named in NAME entry)
                if not MECHANISM in state_spec:
                    # raise GatingSignalError("Specification dict for State to be gated by {} of {} ({}) must have a "
                    #                            "MECHANISM entry specifying the mechanism to which the state belongs".
                    #                            format(GATING_SIGNAL, owner.name, state_name))
                    raise GatingSignalError("Use of \'NAME\' entry ({0}) in specification dict for "
                                            "{1} of {2} is ambiguous: "
                                            "it must be accompanied by either a \'MECHANISM\' entry "
                                            "(to be interpreted as the name of a State, "
                                            "or a \'PROJECTIONS\' entry (to interpreted as the name of the {1})".
                                            format(state_name, GatingSignal.componentName, owner.name))
                mech = state_spec[MECHANISM]

        # Check that all of the other entries in the dict are for valid GatingSignal params
        #    - skip any entries specifying gating signal (i.e., non-keyword keys being used as the GatingSignal name
        #    - place others in params
        for param_entry in [entry for entry in state_spec if not entry in {gating_signal_name, key_as_name, MECHANISM}]:
            if not param_entry in gating_signal_keywords:
                raise GatingSignalError("Entry in specification dictionary for {} arg of {} ({}) "
                                           "is not a valid {} parameter".
                                           format(GATING_SIGNAL, owner.name, param_entry,
                                                  GatingSignal.__name__))
            params[param_entry] = state_spec[param_entry]

    else:
        # raise GatingSignalError("PROGRAM ERROR: unrecognized GatingSignal specification for {} ({})".
        #                             format(self.name, state_spec))
        raise GatingSignalError("Specification of {} for {} ({}) is not a valid {} specification".
                                    format(GATING_SIGNAL, owner.name, state_spec, GATING_SIGNAL))
        # raise GatingSignalError("Specification of {} for {} ({}) must be an InputState or OutputState, "
        #                            "a tuple specifying a name for one and a mechanism to which it belongs ,"
        #                            "a list of state specifications, "
        #                            "a {} specification dict with one or more state specifications and "
        #                            "entries for {} parameters, or an existing GatingSignal".
        #                             format(GATING_SIGNAL, owner.name, state_spec, GATING_SIGNAL, GATING_SIGNAL))

    # If a states list has not already been constructed, do so here
    if not states:
        # Check that specified state is an InputState or OutputState of the Mechanism
        if state_name in mech.input_states:
            state_type = INPUT_STATE
            state = mech.input_states[state_name]
        elif state_name in mech.output_states:
            state_type = OUTPUT_STATE
            state = mech.output_states[state_name]
        else:
            raise GatingSignalError("{} (in specification of {} for {}) is not the name of an "
                                       "InputState or OutputState of {}".
                                        format(state_name, GatingSignal.componentName, owner.name, mech))
        # Check that the Mechanism is in GatingMechanism's system
        # if not owner.system and not mech in owner.system.mechanisms:
        # # IMPLEMENTATION NOTE: REINSTATE WHEN ASSIGNMENT OF GatingMechanism TO SYSTEM IS RESOLVED (IN COMPOSITION??)
        # if not (set(owner.systems) & set(mech.systems)):
        #     raise GatingSignalError("Specification in {} arg for {} ({} {} of {}) "
        #                                 "must be for a Mechanism in {}".
        #                                 format(GATING_SIGNALS,
        #                                        owner.name,
        #                                        state_name,
        #                                        state_type,
        #                                        mech.name,
        #                                        owner.systems))
        states = [state]

    # Check for any duplicate states in specification for this GatingSignal or existing ones for the owner
    all_gated_states = []
    # Get gated states from any already instantiated GatingSignals in gating_signals arg
    if owner.gating_signals:
        #                                   _gating_signal_arg
        for owner_gs in [gs for gs in owner.gating_signals #   is already an instantiated GatingSignal
                              if (isinstance(gs, GatingSignal) and not gs.value is DEFERRED_INITIALIZATION)]:
            all_gated_states.extend([proj.receiver for proj in owner_gs.efferents])
    # Add states for current GatingSignal
    all_gated_states.extend(states)
    # Check for duplicates
    if len(all_gated_states) != len(set(all_gated_states)):
        for test_state in all_gated_states:
            if next((test_state == state  for state in all_gated_states), None):
                raise GatingSignalError("{} of {} receives more than one GatingProjection from the {}s in {}".
                                        format(test_state.name, test_state.owner.name,
                                               GatingSignal.__name__, owner.name))
        raise GatingSignalError("PROGRAM ERROR: duplicate state detected in {} specifications for {} ({})"
                                   "but could not find the offending state".
                                   format(GATING_SIGNAL, owner.name, gating_signal_name))

    return {NAME: gating_signal_name,
            STATES: states,
            PARAMS: params,
            GATING_SIGNAL: gating_signal}