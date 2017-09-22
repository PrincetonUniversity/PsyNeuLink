# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *************************************  AutoAssociativeLearningMechanism **********************************************

"""
.. _AutoAssociativeLearningMechanism_Overview:

Overview
--------

An AutoAssociativeLearningMechanism is a subclass `LearningMechanism <LearningMechanism>`, streamlined for use with a
`RecurrentTransferMechanism` to train its `recurrent_projection <RecurrentTransferMechanism.recurrent_projection>`.
It is identical in all respects to a LearningMechanism, with the following exceptions:

  * it has only a single *ACTIVATION_INPUT* `InputState`, that receives a `MappingProjection` from an `OutputState` of
    another `Mechanism` (identified by the `activity_source <AutoAssociativeLearningMechanism.activity_source>`,
    typically, the `primary OutputState <OutputState_Primary>` of a RecurrentTransferMechanism);
    
  * it has a single *LEARNING_SIGNAL* `OutputState`, that sends a `LearningProjection` to the `matrix
    <AutoAssociativeProjection>` parameter of an 'AutoAssociativeProjection` (typically, the `recurrent_projection
    <RecurrentTransferMechanism.recurrent_projection>` of a RecurrentTransferMechanism).

  * it has no :keyword:`input_source`, :keyword:`output_source`, or :keyword:`error_source` attributes;  for an
    AutoAssociativeLearningProjection;  instead, it has a single `activity_source` attribute that identifies the
    source of the activity vector used by the Mechanism's `function <AutoAssociativeLearningProjection.function>`.

It is created automatically when a RecurrentTransferMechanism is `specified for learning <Recurrent_Transfer_Learning>`.

Function:  XXXX
LearningRate:  XXX

.. _AutoAssociativeLearningMechanism_Class_Reference:

Class Reference
---------------

"""

import numpy as np
import typecheck as tc

from PsyNeuLink.Components.Component import InitStatus, parameter_keywords
from PsyNeuLink.Components.Functions.Function \
    import Hebbian, ModulationParam, _is_modulation_param, is_function_type
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism_Base
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.AdaptiveMechanism import AdaptiveMechanism_Base
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanism.LearningMechanism \
    import LearningMechanism, ACTIVATION_INPUT
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism import ObjectiveMechanism
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.Projections.Projection \
    import Projection_Base, _is_projection_spec, _validate_receiver, projection_keywords
from PsyNeuLink.Components.ShellClasses import Mechanism, Projection
from PsyNeuLink.Globals.Keywords import CONTROL_PROJECTIONS, FUNCTION_PARAMS, IDENTITY_MATRIX, INDEX, INITIALIZING, \
    INPUT_STATES, LEARNED_PARAM, LEARNING, LEARNING_MECHANISM, LEARNING_PROJECTION, LEARNING_SIGNAL, LEARNING_SIGNALS, \
    LEARNING_SIGNAL_SPECS, MAPPING_PROJECTION, MATRIX, NAME, OUTPUT_STATES, PARAMETER_STATE, PARAMS, PROJECTION, \
    PROJECTIONS, AUTOASSOCIATIVE_LEARNING_MECHANISM
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceLevel
from PsyNeuLink.Globals.Utilities import is_numeric, parameter_spec
from PsyNeuLink.Scheduling.TimeScale import CentralClock, TimeScale

# Params:

parameter_keywords.update({LEARNING_PROJECTION, LEARNING})
projection_keywords.update({LEARNING_PROJECTION, LEARNING})

def _is_learning_spec(spec):
    """Evaluate whether spec is a valid learning specification

    Return `True` if spec is LEARNING or a valid projection_spec (see Projection._is_projection_spec
    Otherwise, return :keyword:`False`

    """
    if spec is LEARNING:
        return True
    else:
        return _is_projection_spec(spec)


input_state_names =  [ACTIVATION_INPUT]
output_state_names = [LEARNING_SIGNAL]

DefaultTrainingMechanism = ObjectiveMechanism

class AutoAssociativeLearningMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class AutoAssociativeLearningMechanism(LearningMechanism):
    """
    AutoAssociativeLearningMechanism(              \
        variable,                                  \
        function=Hebbian,                          \
        learning_rate=None,                        \
        learning_signals=LEARNING_SIGNAL,          \
        modulation=ModulationParam.ADDITIVE,       \
        params=None,                               \
        name=None,                                 \
        prefs=None)

    Implements a `LearningMechanism` that modifies the `matrix <MappingProjection.matrix>` parameter of an
    `AutoAssociativeProjection (typically the `recurrent_projection <RecurrentTransferMechanism.recurrent_projection>`
    of a `RecurrentTransferMechanism`).


    Arguments
    ---------

    variable : List or 2d np.array
        it must have a single item that corresponds to the value required by the AutoAssociativeLearningMechanism's
        `function <AutoAssociativeLearningMechanism..function>`;  it must each be compatible (in number and type)
        with the `value <InputState.value>` of the Mechanism's `InputState <LearningMechanism_InputStates>` (see
        `variable <AutoAssociativeLearningMechanism..variable>` for additional details).

    learning_signals : List[parameter of Projection, ParameterState, Projection, tuple[str, Projection] or dict]
        specifies the `matrix <AutoAssociativeProjection.matrix>` to be learned (see `learning_signals
        <LearningMechanism.learning_signals>` for details of specification).

    modulation : ModulationParam : ModulationParam.ADDITIVE
        specifies the default form of modulation used by the AutoAssociativeLearningMechanism's LearningSignals,
        unless they are `individually specified <LearningSignal_Specification>`.

    function : LearningFunction or function
        specifies the function used to calculate the AutoAssociativeLearningMechanism's `learning_signal
        <AutoAssociativeLearningMechanism.learning_signal>` attribute.  It must take as its **variable** argument a
        list or 1d array of numeric values (the "activity vector") and return a list, 2d np.array or np.matrix
        representing a square matrix with dimensions that equal the length of its variable (the "weight change
        matrix").

    learning_rate : float
        specifies the learning rate for the AutoAssociativeLearningMechanism. (see `learning_rate
        <AutoAssociativeLearningMechanism.learning_rate>` for details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        Projection, its function, and/or a custom function and its parameters. By default, it contains an entry for
        the Projection's default `function <LearningProjection.function>` and parameter assignments.  Values specified
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

    COMMENT:
        componentType : LEARNING_MECHANISM
    COMMENT

    variable : 2d np.array
        has a single item, that serves as the template for the input required by the AutoAssociativeLearningMechanism's
        `function <AutoAssociativeLearningMechanism.function>`, corresponding to the `value
        <OutputState.value>` of the `activity_source <AutoAssociativeLearningMechanism.activity_source>`.

    activity_source : OutputState
        the `OutputState` that is the `sender <AutoAssociativeProjection.sender>` of the `AutoAssociativeProjection`
        that the Mechanism trains.

    input_states : ContentAddressableList[OutputState]
        has a single item, that contains the AutoAssociativeLearningMechanism's single *ACTIVATION_INPUT* `InputState`.

    primary_learned_projection : AutoAssociativeProjection
        the `Projection` with the `matrix <AutoAssociativeProjection.matrix>` parameter being trained by the
        AutoAssociativeLearningMechanism.  It is always the first Projection listed in the
        AutoAssociativeLearningMechanism's `learned_projections <AutoAssociativeLearningMechanism.learned_projections>`
        attribute.

    learned_projections : List[MappingProjection]
        all of the `AutoAssociativeProjections <AutoAssociativeProjection>` modified by the
        AutoAssociativeLearningMechanism;  the first item in the list is always the `primary_learned_projection
        <AutoAssociativeLearningMechanism.primary_learned_projection>`.

    function : LearningFunction or function : default Hebbian
        specifies the function used to calculate the `learning_signal
        <AutoAssociativeLearningMechanism.learning_signal>` (assigned to the AutoAssociativeLearningMechanism's
        `LearningSignal(s) <LearningMechanism_LearningSignal>`). It's `variable <Function_Base.variable>` must be
        a list or 1d np.array of numeric entries, corresponding in length to the AutoAssociativeLearningMechanism's
        *ACTIVATION_INPUT* (`primary <InputState_Primary>`) InputState.

    learning_rate : float : None
        determines the learning rate for the AutoAssociativeLearningMechanism.  It is used to specify the :keyword:`learning_rate`
        parameter for the AutoAssociativeLearningMechanism.'s `learning function <AutoAssociativeLearningMechanism.function>`
        (see description of `learning_rate <LearningMechanism_Learning_Rate>` for additional details).

    learning_rate : float, 1d or 2d np.array, or np.matrix of numeric values : default None
        used by `function <AutoAsociativeLearningMechanism.funtion>` to scale the weight change matrix it returns.
        If specified, it supersedes the learning_rate assigned to any `Process` or `System` to which the
        RecurrentMechanism is assigned.  If it is a scalar, it is multiplied by the weight change matrix generated by
        the `function <AutoAssociativeLearningMechanism.function>`;  if it is a 1d np.array, it is
        multiplied Hadamard (elementwise) by the input to the `function <AutoAssociativeLearningMechanism.function>`
        ("activity vector") before calculating the weight change matrix;  if it is a 2d np.array, it is multiplied
        Hadamard (elementwise) by the weight change matrix; if it is `None`, then the `learning_rate
        <Process_Base.learning_rate>` specified for the Process to which the AutoAssociativeLearningMechanism belongs
        belongs is used;  and, if that is `None`, then the `learning_rate <System_Base.learning_rate>`
        for the System to which it belongs is used. If all are `None`, then the `default_learning_rate
        <LearningMechanism.default_learning_rate>` for the function <AutoAssociativeLearningMechanism.function>` is
        used (see `learning_rate <LearningMechanism_Learning_Rate>` for additional details).

    learning_signal : 2d ndarray or matrix of numeric values
        the value returned by `function <AutoAssociativeLearningMechanism.function>`, that specifies
        the changes to the weights of the `matrix <AutoAssociativeProjection.matrix>` parameter for the
        AutoAssociativeLearningMechanism's`learned_projections <AutoAssociativeLearningMechanism.learned_projections>`;
        It is assigned as the value of the AutoAssociativeLearningMechanism's `LearningSignal(s)
        <LearningMechanism_LearningSignal>` and, in turn, its `LearningProjection(s) <LearningProjection>`.

    learning_signals : List[LearningSignal]
        list of all of the `LearningSignals <LearningSignal>` for the AutoAssociativeLearningMechanism, each of which
        sends one or more `LearningProjections <LearningProjection>` to the `ParameterState(s) <ParameterState>` for
        the `matrix <AutoAssociativeProjection.matrix>` parameter of the `AutoAssociativeProjection(s)
        <AutoAssociativeProjection>` trained by the AutoAssociativeLearningMechanism.  Although in most instances an
        AutoAssociativeLearningMechanism is used to train a single AutoAssociativeProjection, like a standard
        `LearningMechanism`, it can be assigned additional LearningSignals and/or LearningProjections to train
        additional ones;  in such cases, the `value <LearningSignal>` for all of the LearningSignals is the
        the same:  the AutoAssociativeLearningMechanism's `learning_signal
        <AutoAssociativeLearningMechanism.learning_signal>` attribute, based on its `activity_source
        <AutoAssociativeLearningMechanism>.activity_source>`.  Since LearningSignals are `OutputStates
        <OutputState>`, they are also listed in the AutoAssociativeLearningMechanism's `output_states
        <AutoAssociativeLearningMechanism.output_states>` attribute.

    learning_projections : List[LearningProjection]
        list of all of the LearningProjections <LearningProject>` from the AutoAssociativeLearningMechanism, listed in
        the order of the `LearningSignals <LearningSignal>` to which they belong (that is, in the order they are
        listed in the `learning_signals <AutoAssociativeLearningMechanism.learning_signals>` attribute).

    output_states : ContentAddressableList[OutputState]
        list of the AutoAssociativeLearningMechanism's `OutputStates <OutputState>`, beginning with its
        `LearningSignal(s) <AutoAssociativeLearningMechanism_LearningSignal>`, and followed by any additional
        (user-specified) `OutputStates <OutputState>`.

    output_values : 2d np.array
        the first item is the `value <OutputState.value>` of the LearningMechanism's `learning_signal
        <AutoAssociativeLearningMechanism.learning_signal>`, followed by the `value <OutputState.value>`\\s
        of any additional (user-specified) `OutputStates <OutputState>`.

    modulation : ModulationParam
        the default form of modulation used by the AutoAssociativeLearningMechanism's `LearningSignal(s)
        <LearningMechanism_LearningSignal>`, unless they are `individually specified <LearningSignal_Specification>`.

    name : str : default LearningProjection-<index>
        the name of the AutoAssociativeLearningMechanism.
        Specified in the **name** argument of the constructor for the Projection;
        if not is specified, a default is assigned by ProjectionRegistry
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for Projection.
        Specified in the **prefs** argument of the constructor for the Projection;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentType = AUTOASSOCIATIVE_LEARNING_MECHANISM
    className = componentType
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    # ClassDefaults.variable = None

    paramClassDefaults = Projection_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        CONTROL_PROJECTIONS: None,
        INPUT_STATES:input_state_names,
        OUTPUT_STATES:[{NAME:LEARNING_SIGNAL,  # NOTE: This is the default, but is overridden by any LearningSignal arg
                        INDEX:0}
                       ]})

    @tc.typecheck
    def __init__(self,
                 variable:tc.any(list, np.ndarray),
                 size=None,
                 function:is_function_type=Hebbian,
                 learning_signals:tc.optional(list) = None,
                 modulation:tc.optional(_is_modulation_param)=ModulationParam.ADDITIVE,
                 learning_rate:tc.optional(parameter_spec)=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function,
                                                  learning_signals=learning_signals,
                                                  params=params)

        # # USE FOR IMPLEMENTATION OF deferred_init()
        # # Store args for deferred initialization
        # self.init_args = locals().copy()
        # self.init_args['context'] = self
        # self.init_args['name'] = name

        # # Flag for deferred initialization
        # self.init_status = InitStatus.DEFERRED_INITIALIZATION

        self._learning_rate = learning_rate

        super().__init__(variable=variable,
                         size=size,
                         function=function,
                         modulation=modulation,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

    def _validate_variable(self, variable, context=None):
        """Validate that variable has exactly three items: activation_input, activation_output and error_signal
        """

        # Skip LearningMechanism._validate_variable in call to super(), as it requires variable to have 3 items
        variable = self._update_variable(super(LearningMechanism, self)._validate_variable(variable, context))

        # MODIFIED 9/22/17 NEW: [HACK]
        if np.array(np.squeeze(variable)).ndim != 1 or not is_numeric(variable):
        # MODIFIED 9/22/17 END
            raise AutoAssociativeLearningMechanismError("Variable for {} ({}) must be "
                                                        "a list or 1d np.array containing only numbers".
                                                        format(self.name, variable))
        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        # Skip LearningMechanism._validate_params, as it has different requirements
        super(LearningMechanism, self)._validate_params(request_set=request_set,target_set=target_set,context=context)

        # FIX: REPLACE WITH CALL TO _parse_state_spec WITH APPROPRIATE PARAMETERS (AKIN TO CONTROL_SIGNAL
        if LEARNING_SIGNALS in target_set and target_set[LEARNING_SIGNALS]:

            from PsyNeuLink.Components.States.ModulatorySignals.LearningSignal \
                import LearningSignal
            from PsyNeuLink.Components.States.ParameterState import ParameterState
            from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection

            for spec in target_set[LEARNING_SIGNALS]:

                learning_proj = None  # Projection from LearningSignal to MappingProjection

                # Specification is for a LearningSignal
                if isinstance(spec, LearningSignal):
                    #  Check that any LearningProjections it has
                    #    are to MappingProjections to Mechanisms in the same System
                    for learning_proj in spec.efferents:
                        _validate_receiver(self,learning_proj, MappingProjection, LEARNING_SIGNAL, context)
                    continue

                # Specification is for a ParameterState
                elif isinstance(spec, ParameterState):
                    param_name = spec.name
                    mapping_proj = spec.owner

                # Specification is for a Projection
                elif isinstance(spec, Projection):
                    if isinstance(spec, LearningProjection):
                        param_name = spec.receiver.name
                        learning_proj = spec
                        mapping_proj = learning_proj.receiver.owner
                    elif isinstance(spec, MappingProjection):
                        param_name = MATRIX
                        mapping_proj = spec
                    else:
                        raise AutoAssociativeLearningMechanismError("The {} specified in the {} arg for {} ({}) must be a {}".
                                                     format(PROJECTION,
                                                            LEARNING_SIGNALS,
                                                            self.name,
                                                            spec.name,
                                                            MAPPING_PROJECTION))

                # Specification is for a tuple (str, MappingProjection):
                elif isinstance(spec, tuple):
                    param_name = spec[0]
                    mapping_proj = spec[1]
                    # Check that 1st item is a str (presumably the name of the learned Projection's attribute
                    #    for the param to be learned; e.g., 'MATRIX' for MappingProjection)
                    if not isinstance(param_name, str):
                        raise AutoAssociativeLearningMechanismError("1st item of tuple in specification of {} for {} ({}) "
                                                     "must be a string".format(LEARNING_SIGNAL, self.name, param_name))
                    # Check that 2nd item is a MappingProjection
                    if not isinstance(mapping_proj, MappingProjection):
                        raise AutoAssociativeLearningMechanismError("2nd item of tuple in specification of {} for {} ({}) "
                                                     "must be a {}".
                                                     format(LEARNING_SIGNAL,
                                                            self.name,
                                                            mapping_proj,
                                                            MAPPING_PROJECTION))

                # LearningSignal specification dictionary, must have the following entries:
                #    NAME:str - must be the name of an attribute of PROJECTION
                #    PROJECTION:Projection - must be a MappingProjection
                #                            and have an attribute and corresponding ParameterState named NAME
                #    PARAMS:dict - entries must be valid LearningSignal parameters (e.g., LEARNING_RATE)
                elif isinstance(spec, dict):
                    if not NAME in spec:
                        raise AutoAssociativeLearningMechanismError("Specification dict for {} of {} must have a {} entry".
                                                    format(LEARNING_SIGNAL, self.name, NAME))
                    param_name = spec[NAME]
                    if not PROJECTION in spec:
                        raise AutoAssociativeLearningMechanismError("Specification dict for {} of {} must have a {} entry".
                                                    format(LEARNING_SIGNAL, self.name, PROJECTION))
                    mapping_proj = spec[PROJECTION]
                    if not isinstance(mapping_proj, MappingProjection):
                        raise AutoAssociativeLearningMechanismError("{} entry of specification dict for {} of {} must be a {}".
                                                    format(PROJECTION, LEARNING_SIGNAL, self.name, MAPPING_PROJECTION))
                    # Check that all of the other entries in the specification dictionary
                    #    are valid LearningSignal params
                    for param in spec:
                        if param in {NAME, PROJECTION}:
                            continue
                        if not hasattr(LearningSignal, param):
                            raise AutoAssociativeLearningMechanismError("Entry in specification dictionary for {} arg of {} ({}) "
                                                       "is not a valid {} parameter".
                                                       format(LEARNING_SIGNAL, self.name, param,
                                                              LearningSignal.__class__.__name__))
                else:
                    raise AutoAssociativeLearningMechanismError("PROGRAM ERROR: unrecognized specification for {} arg of {} ({})".
                                                format(LEARNING_SIGNALS, self.name, spec))
                    # raise LearningMechanismError("Specification of {} for {} ({}) must be a "
                    #                             "ParameterState, Projection, a tuple specifying a parameter and "
                    #                              "Projection, a LearningSignal specification dictionary, "
                    #                              "or an existing LearningSignal".
                    #                             format(CONTROL_SIGNAL, self.name, spec))

                # Validate that the receiver of the LearningProjection (if specified)
                #     is a MappingProjection and in the same System as self (if specified)
                if learning_proj:
                    _validate_receiver(sender_mech=self,
                                       projection=learning_proj,
                                       expected_owner_type=MappingProjection,
                                       spec_type=LEARNING_SIGNAL,
                                       context=context)

                # IMPLEMENTATION NOTE: the tests below allow for the possibility that the MappingProjection
                #                      may not yet be fully implemented (e.g., this can occur if the
                #                      LearningMechanism being implemented here is as part of a LearningProjection
                #                      specification for the MappingProjection's matrix param)
                # Check that param_name is the name of a parameter of the MappingProjection to be learned
                if not param_name in (set(mapping_proj.user_params) | set(mapping_proj.user_params[FUNCTION_PARAMS])):
                    raise AutoAssociativeLearningMechanismError("{} (in specification of {} for {}) is not an "
                                                "attribute of {} or its function"
                                                .format(param_name, LEARNING_SIGNAL, self.name, mapping_proj))
                # Check that the MappingProjection to be learned has a ParameterState for the param
                if mapping_proj._parameter_states and not param_name in mapping_proj._parameter_states.names:
                    raise AutoAssociativeLearningMechanismError("There is no ParameterState for the parameter ({}) of {} "
                                                "specified in {} for {}".
                                                format(param_name, mapping_proj.name, LEARNING_SIGNAL, self.name))

    def _instantiate_attributes_before_function(self, context=None):
        # Skip LearningMechanism, as it checks for error_source, which AutoAssociativeLearningMechanism doesn't have
        super(LearningMechanism, self)._instantiate_attributes_before_function(context=context)

    def _instantiate_output_states(self, context=None):

        # Create registry for LearningSignals (to manage names)
        from PsyNeuLink.Globals.Registry import register_category
        from PsyNeuLink.Components.States.ModulatorySignals.LearningSignal import LearningSignal
        from PsyNeuLink.Components.States.State import State_Base
        register_category(entry=LearningSignal,
                          base_class=State_Base,
                          registry=self._stateRegistry,
                          context=context)

        # Instantiate LearningSignals if they are specified, and assign to self._output_states
        # Note: if any LearningSignals are specified they will replace the default LEARNING_SIGNAL OutputState
        #          in the OUTPUT_STATES entry of paramClassDefaults;
        #       the LearningSignals are appended to _output_states, leaving ERROR_SIGNAL as the first entry.
        if self.learning_signals:
            # Delete default LEARNING_SIGNAL item in output_states
            del self._output_states[0]
            for i, learning_signal in enumerate(self.learning_signals):
                # Instantiate LearningSignal
                ls = self._instantiate_learning_signal(learning_signal=learning_signal, context=context)
                # Add LearningSignal to ouput_states list
                self._output_states.append(ls)
                # Replace spec in learning_signals list with actual LearningSignal
                self.learning_signals[i] = ls

        super(LearningMechanism, self)._instantiate_output_states(context=context)

    def _execute(self,
                variable=None,
                runtime_params=None,
                clock=CentralClock,
                time_scale = TimeScale.TRIAL,
                context=None):
        """Execute AutoAssociativeLearningMechanism. function and return learning_signal

        :return: (2D np.array) self.learning_signal
        """

        # COMPUTE LEARNING SIGNAL (dE/dW):
        self.learning_signal = self.function(variable=variable,
                                             params=runtime_params,
                                             context=context)

        if not INITIALIZING in context and self.reportOutputPref:
            print("\n{} weight change matrix: \n{}\n".format(self.name, self.learning_signal))

        self.value = [self.learning_signal]
        return self.value

    @property
    def learning_rate(self):
        return self.function_object.learning_rate

    @learning_rate.setter
    def learning_rate(self, assignment):
        self.function_object.learning_rate = assignment

    @property
    def primary_learned_projection(self):
        return self.learned_projection[0]

    @property
    def learned_projections(self):
        return [lp.receiver.owner for lp in self.learning_projections]

    @property
    def activity_source(self):
        return self.input_state.path_afferents[0].sender.owner