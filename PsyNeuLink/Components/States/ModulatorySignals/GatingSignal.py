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


A GatingSignal is a type of `ModulatorySignal`.  A ModulatorySignal is a type of `OutputState` that belongs to an
`AdaptiveMechanism`, and is used to `modulate <ModulatorySignal_Modulation>` the `value <State.value>` of another
`State`.  A GatingSignal is specialized for use with a `GatingMechanism` and a `GatingProjection`, to modify the
`value <State.value> of the InputState(s) and/or OutputState(s) of one or more `Mechanisms <Mechanism>`.
A GatingSignal receives a value specified by the the `function <ControlMechanism.function>` of the
GatingMechanism to which it belongs, and assigns that to one or more `GatingProjections <GatingProjection>`,
each of which projects to an InputStates and/or OutputState to be gated, and is used to modulate that state's
`value <State.value>`.


.. _GatingSignal_Creation:

Creating a GatingSignal
-----------------------

A GatingSignal is created automatically whenever an `InputState` or `OutputState` of a `Mechanism` are
`specified for gating <GatingMechanism_Gating_Signals>`.  GatingSignals can also be specified in the 
**gating_signals** argument of the constructor for a `GatingMechanism`, using any of the formats described 
`below <GatingSignal_Specification>`.  Although a GatingSignal can be created directly using its constructor 
(or any of the other ways for `creating an outputState <OutputStates_Creation>`),  this is neither necessary 
nor advisable, as a GatingSignal has dedicated components and requirements for configuration that must be met 
for it to function properly.

.. _GatingSignal_Specification:

Specifying GatingSignals
~~~~~~~~~~~~~~~~~~~~~~~~

When a GatingSignal is specified in context (e.g., the **gating_signals** argument of the constructor for a
`GatingMechanism`, the specification can take any of the following forms:

  * an *InputState* or *OutputState* of a Mechanism;
  |
  * a *Mechanism*, in which case its `primary inputState <Mechanism_InputStates>` will be used; 
  |
  * a *tuple*, with the *name* of the state as the 1st item, and the mechanism to which it belongs as the 2nd;  
    note that this is a convenience format, which is simpler to use than a specification dictionary (see below), 
    but precludes specification of any parameters <GatingSignal_Structure>` for the GatingSignal.
  |
  * a *specification dictionary*, that can take either of the following two forms:
    * for a single state, the dictionary must have the following two entries: 
        * *NAME*:str - a string that is the name of the state to be gated;
        * *MECHANISM*:Mechanism - the Mechanism to which the state belongs;
      in this case, the GatingSignal will named by appending "_GatingSignal" to the name of the state.
    * for multiple states, the dictionary must have the following entry:
        * <str>:list - where the string used for the key specifies the name to be used for the GatingSignal,
          and the value is a list, each item of which is a `specification of a state <LINK>` to be gated by 
          the GatingSignal (and that will receive a `GatingProjection` from it).
    In either case, the dictionary can also contain a *MODULATION* entry, the value of which must be a `Modulation` 
    value that determines how that particular GatingSignal will modulate the value of all of the states that it gates.

.. _GatingSignal_Structure:

Structure
---------

A GatingSignal is owned by a `GatingMechanism`, and associated with one or more `GatingProjections <GatingProjection>`, 
each of which projects to the InputState or OutputState that it gates.  

.. _GatingSignal_Modulation:

Modulation
~~~~~~~~~~

Each GatingSignal has a `modulation <GatingSignal.modulation>` attribute that determines how the GatingProjection 
is used by the state to which it projects to modify its value (see `modulation <ModulatoryProjection.modulation>` 
for an explanation of how this attribute is specified and used to modulate the value of a state).  The default value 
is set to the value of the `modulation <GatingMechanism.modulation>` attribute of the GatingMechanism to which the 
GatingSignal belongs;  this the is same for all of the GatingSignals belonging to that GatingMechanism.  However, the
`modulation <GatingSignal.modulation>` can be specified individually for a GatingSignal using a specification 
dictionary where the GatingSignal is specified, as described `above <GatingSignal_Specification>`. The 
`modulation <GatingSignal.modulation>` value of a GatingSignal is used by all of the 
`GatingProjections <GatingProjection>` that project from that GatingSignal.


.. _ControlSignal_Execution:

Execution
---------

A GatingSignal cannot be executed directly.  It is executed whenever the `GatingMechanism` to which it belongs is
executed.  When this occurs, the GatingMechanism provides the GatingSignal with a value that is used by its 
`GatingProjection(s) <GatingProjection>` to modulate the :keyword:`value` of the states to which they project. Those 
states use the value of the GatingProjection they receive to modify a parameter of their function.  How the modulation
is executed is determined by the GatingSignal's `modulation` attribute
(see `ModulatoryProjections_Modulation_Operation).

.. note::
   The change in the value of InputStates and OutputStates in response to the execution of a GatingMechanism are not 
   applied until the mechanism(s) to which those states belong are next executed; see :ref:`Lazy Evaluation <LINK>` 
   for an explanation of "lazy" updating).

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

gating_signal_keywords = {GATED_STATE}
gating_signal_keywords.update(modulatory_signal_keywords)


class GatingSignal(ModulatorySignal):
    """
    GatingSignal(                                   \
        owner,                                      \
        function=LinearCombination(operation=SUM),  \
        modulation=ModulationParam.MULTIPLICATIVE   \
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
        the `GatingMechanism` to which the GatingSignal belongs.

    variable : number, list or np.ndarray
        used by `function <GatingSignal.function>` to compute the GatingSignal's `value <GatingSignal.value>`.

    function : TransferFunction :  default Linear(slope=1, intercept=0)
        provides the GatingSignal's `value <GatingMechanism.value>`; the default is an identity function that
        passes the input to the GatingMechanism as value for the GatingSignal. 

    value : number, list or np.ndarray
        result of `function <GatingSignal.function>`.
    
    modulation : ModulationParam
        determines how the output of the GatingSignal is used to modulate the value of the state(s)
        to which its GatingProjection(s) project(s).

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
                 modulation:tc.optional(_is_modulation_param)=None,
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
        #  (test for it, and create if necessary, as per outputStates in GatingProjection._instantiate_sender),

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        super().__init__(owner,
                         reference_value,
                         variable=variable,
                         modulation=modulation,
                         index=index,
                         calculate=calculate,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

    # def _instantiate_function(self, context=None):
    #     super()._instantiate_function(context=context)
    #     self.function_object.FunctionOutputTypeConversion = True
    #     self.function_object.functionOutputType = FunctionOutputType.RAW_NUMBER
    #     TEST = True

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

    # Specification is for a GatingSignal - return as is
    if isinstance(state_spec, GatingSignal):
        gating_signal = state_spec
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
                    raise GatingSignalError("Specification dict for state to be gated by {} of {} ({}) must have a "
                                               "MECHANISM entry specifying the mechanism to which the state belongs".
                                               format(GATING_SIGNAL, owner.name, state_name))
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
            raise GatingSignalError("{} (in specification of {}  {}) is not an "
                                       "InputState or OutputState of {}".
                                        format(state_name, GATING_SIGNAL, owner.name, mech))
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
        #                                   _gating_signal_arg     already instantiated GatingSignal
        for gating_signal in [gs for gs in owner.gating_signals if isinstance(gs, GatingSignal)]:
            #                  gated state
            all_gated_states.extend([proj.receiver for proj in gating_signal.efferents])
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