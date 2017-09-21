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

An AutoAssociativeLearningMechanism is a subclass `LearningMechanism <LearningMechanism>` designed for use with
a `RecurrentTransferMechanism`, to train its `recurrent_projection <RecurrentTransferMechanism.recurrent_projection>`.
It is identical in all respects to a LearningMechanism, with the following exceptions:

  * it has only a single `InputState`, that receives a `MappingProjection` from an `OutputState` of another
  `Mechanism` (typically, the `primary OutputState <OutputState_Primary>` of a RecurrentTransferMechanism;
    
  * it has a single `OutputState`, that sends a `LearningProjection` to the `matrix <AutoAssociativeProjection>`
    parameter of an 'AutoAssociativeProjection` (typically, the `recurrent_projection
    <RecurrentTransferMechanism.recurrent_projection>` of a RecurrentTransferMechanism).

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
    import BackPropagation, ModulationParam, _is_modulation_param, is_function_type
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanism import LearningMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism \
    import OUTCOME, ObjectiveMechanism
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.Projections.Projection \
    import Projection_Base, _is_projection_spec, _validate_receiver, projection_keywords
from PsyNeuLink.Components.ShellClasses import Mechanism, Projection
from PsyNeuLink.Globals.Keywords import CONTROL_PROJECTIONS, FUNCTION_PARAMS, IDENTITY_MATRIX, INDEX, INITIALIZING, \
    INPUT_STATES, LEARNED_PARAM, LEARNING, LEARNING_MECHANISM, LEARNING_PROJECTION, LEARNING_SIGNAL, LEARNING_SIGNALS, \
    LEARNING_SIGNAL_SPECS, MAPPING_PROJECTION, MATRIX, NAME, OUTPUT_STATES, PARAMETER_STATE, PARAMS, PROJECTION, \
    PROJECTIONS
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


# Used to index variable:
ACTIVATION_INPUT_INDEX = 0
ACTIVATION_OUTPUT_INDEX = 1
ERROR_OUTPUT_INDEX = 2
ERROR_SIGNAL_INDEX = 3

# Used to name input_states and output_states:
ACTIVATION_INPUT = 'activation_input'     # InputState
ACTIVATION_OUTPUT = 'activation_output'   # InputState
ERROR_SIGNAL = 'error_signal'
input_state_names =  [ACTIVATION_INPUT, ACTIVATION_OUTPUT, ERROR_SIGNAL]
output_state_names = [ERROR_SIGNAL, LEARNING_SIGNAL]

ERROR_SOURCE = 'error_source'

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
        error_source,                              \
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

    error_source : OutputState
        specifies the source of the error signal used by the AutoAssociativeLearningMechanism's `function
        <AutoAssociativeLearningMechanism.function>`.  It must be an `OutputState` of a `Mechanism`, the value of which
        is a list or 1d array of numeric values; otherwise it must be a `LearningProjection`.
        COMMENT:
            A LearningProjection can be specified (e.g., by LearningAuxiliary._instantiate_learning_components()
            from which it will identify the error_source
        COMMENT

    learning_signals : List[parameter of Projection, ParameterState, Projection, tuple[str, Projection] or dict]
        specifies the parameter(s) to be learned (see `learning_signals <LearningMechanism.learning_signals>` for
        details).

    modulation : ModulationParam : ModulationParam.ADDITIVE
        specifies the default form of modulation used by the AutoAssociativeLearningMechanism's LearningSignals,
        unless they are `individually specified <LearningSignal_Specification>`.

    function : LearningFunction or function
        specifies the function used to calculate the AutoAssociativeLearningMechanism's `learning_signal
        <AutoAssociativeLearningMechanism.learning_signal>` attribute.  It must take as its **variable** argument a
        list or 1d array of numeric values (the "activity vector") and return a list, 2d np.array or np.matrix
        representing a square matrix with dimensions that equal the length of its variable.

    learning_rate : float
        specifies the learning rate for the AutoAssociativeLearningMechanism. (see `learning_rate
        <LearningMechanism.learning_rate>` for details).

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
        has one item that serves as the template for the input required by the AutoAssociativeLearningMechanism's
        `function <AutoAssociativeLearningMechanism.function>`, corresponding to the `value <OutputState.value>`
        of the `OutputState` that is the `sender <AutoAssociativeProjection.sender>` of the AutoAssociativeProjection
        that the AutoAssociativeLearningMechanism trains (i.e., the `error_source
        <AutoAssociativeLearningMechanism.error_source>`.

XXXX
    input_states : ContentAddressableList[OutputState]
        list containing the AutoAssociativeLearningMechanism.'s three `InputStates <LearningMechanism_InputStates>`:
        *ACTIVATION_INPUT*,  *ACTIVATION_OUTPUT*, and *ERROR_SIGNAL*.

    input_source : ProcessingMechanism
        the Mechanism that sends the `primary_learned_projection`, and projects to the
        AutoAssociativeLearningMechanism.'s *ACTIVATION_INPUT* `InputState <LearningMechanism_Activation_Input>`.

    output_source : ProcessingMechanism
        the Mechanism that receives the `primary_learned_projection`, and  projects to the
        LearningMechanism's *ACTIVATION_OUTPUT* `InputState <LearningMechanism_Activation_Output>`.

    error_source : ComparatorMechanism or LearningMechanism
        the Mechanism that calculates the error signal provided to the
        LearningMechanism's *ERROR_SIGNAL* `InputState <LearningMechanism_Input_Error_Signal>`.

    primary_learned_projection : MappingProjection
        the Projection with the `matrix <MappingProjection.matrix>` parameter used to generate the
        AutoAssociativeLearningMechanism.'s `error_signal <LearningMechanism.error_signal>` and `learning_signal
        <AutoAssociativeLearningMechanism.learning_signal>` attributes.  It is always the first Projection listed in the
        AutoAssociativeLearningMechanism.'s `learned_projections` attribute.

    learned_projections : List[MappingProjection]
        all of the MappingProjections modified by the AutoAssociativeLearningMechanism.;  the first item in the list is always the
        `primary_learned_projection`.

    function : LearningFunction or function : default BackPropagation
        specifies the function used to calculate the `learning_signal <AutoAssociativeLearningMechanism.learning_signal>` (assigned
        to the AutoAssociativeLearningMechanism.'s `LearningSignal(s) <LearningMechanism_LearningSignal>`), and the `error_signal
        <LearningMechanism.error_signal>` (passed to the next LearningMechanism in a learning sequence for
        `multilayer learning <LearningMechanism_Multilayer_Learning>`).  It takes the following three arguments,
        each of which must be a list or 1d array: **input**,  **output**, and **error** (see
        `LearningMechanism_Function` for additional details).

    learning_rate : float : None
        determines the learning rate for the AutoAssociativeLearningMechanism.  It is used to specify the :keyword:`learning_rate`
        parameter for the AutoAssociativeLearningMechanism.'s `learning function <AutoAssociativeLearningMechanism.function>`
        (see description of `learning_rate <LearningMechanism_Learning_Rate>` for additional details).

    error_signal : 1d np.array
        one of two values returned by the LearningMechanism's `function <AutoAssociativeLearningMechanism.function>`.  For
        `single layer learning <LearningMechanism_Single_Layer_Learning>`, this is the same as the value received in
        the LearningMechanism's *ERROR_SIGNAL* `InputState <LearningMechanism_Input_Error_Signal>`;  for `multilayer
        learning <LearningMechanism_Multilayer_Learning>`, it is a modified version of the value received, that takes
        account of the contribution made by the learned_projection and its input to the error signal received. This
        is assigned as the `value <OutputState.value>` of the LearningMechanism's *ERROR_SIGNAL* `OutputState
        <LearningMechanism_Output_Error_Signal>`.

    learning_signal : number, ndarray or matrix
        one of two values returned by the AutoAssociativeLearningMechanism.'s `function <AutoAssociativeLearningMechanism.function>`, that specifies
        the changes to the weights of the `matrix <MappingProjection.matrix>` parameter for the AutoAssociativeLearningMechanism.'s
        `learned_projections <AutoAssociativeLearningMechanism..learned_projections>`;  it is calculated to reduce the error signal
        associated with the `primary_learned_projection` and received from the AutoAssociativeLearningMechanism.'s `error_source`.
        It is assigned as the value of the AutoAssociativeLearningMechanism.'s `LearningSignal(s) <LearningMechanism_LearningSignal>`
        and, in turn, its LearningProjection(s).

    learning_signals : List[LearningSignal]
        list of all of the `LearningSignals <LearningSignal>` for the AutoAssociativeLearningMechanism., each of which sends one or
        more `LearningProjections <LearningProjection>` to the `ParameterState(s) <ParameterState>` for the `matrix
        <MappingProjection.matrix>` parameter of the `MappingProjection(s) <MappingProjection>` trained by the
        AutoAssociativeLearningMechanism.  The `value <LearningSignal>` of each LearningSignal is the AutoAssociativeLearningMechanism.'s
        `learning_signal <AutoAssociativeLearningMechanism.learning_signal>` attribute. Since LearningSignals are `OutputStates
        <OutputState>`, they are also listed in the AutoAssociativeLearningMechanism.'s `output_states
        <AutoAssociativeLearningMechanism.output_states>` attribute, after it *ERROR_SIGNAL* `OutputState
        <LearningMechanism_Output_Error_Signal>`.

    learning_projections : List[LearningProjection]
        list of all of the LearningProjections <LearningProject>` from the AutoAssociativeLearningMechanism., listed in the order of
        the `LearningSignals <LearningSignal>` to which they belong (that is, in the order they are listed in
        the `learning_signals <AutoAssociativeLearningMechanism.>` attribute).

    output_states : ContentAddressableList[OutputState]
        list of the AutoAssociativeLearningMechanism.'s `OutputStates <OutputState>`, including its *ERROR_SIGNAL* `OutputState
        <LearningMechanism_Output_Error_Signal>`, followed by its `LearningSignal(s)
        <LearningMechanism_LearningSignal>`, and then any additional (e.g., user-specified)
        `OutputStates <OutputState>`.

    COMMENT:
       #  FIX: THIS MAY NEED TO BE A 3d array (TO ACCOMDOATE 2d array (MATRICES) AS ENTRIES)\
    COMMENT

    output_values : 2d np.array
        the first item is the `value <OutputState.value>` of the LearningMechanism's *ERROR_SIGNAL* `OutputState
        <LearningMechanism_Output_Error_Signal>`, followed by the `value <LearningSignal.value>` \\(s) of its
        `LearningSignal(s) <LearningMechanism_LearningSignal>`, and then those of any additional (e.g., user-specified)
        `OutputStates <OutputState>`.

    modulation : ModulationParam
        the default form of modulation used by the AutoAssociativeLearningMechanism.'s `LearningSignal(s)
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

    componentType = LEARNING_MECHANISM
    className = componentType
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    # ClassDefaults.variable = None

    paramClassDefaults = Projection_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        CONTROL_PROJECTIONS: None,
        INPUT_STATES:input_state_names,
        OUTPUT_STATES:[{NAME:ERROR_SIGNAL,
                        INDEX:1},
                       {NAME:LEARNING_SIGNAL,  # NOTE: This is the default, but is overridden by any LearningSignal arg
                        INDEX:0}
                       ]})

    @tc.typecheck
    def __init__(self,
                 variable:tc.any(list, np.ndarray),
                 size=None,
                 error_source:tc.optional(Mechanism)=None,
                 function:is_function_type=BackPropagation,
                 learning_signals:tc.optional(list) = None,
                 modulation:tc.optional(_is_modulation_param)=ModulationParam.ADDITIVE,
                 learning_rate:tc.optional(parameter_spec)=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(error_source=error_source,
                                                  function=function,
                                                  learning_signals=learning_signals,
                                                  params=params)

        # # USE FOR IMPLEMENTATION OF deferred_init()
        # # Store args for deferred initialization
        # self.init_args = locals().copy()
        # self.init_args['context'] = self
        # self.init_args['name'] = name
        # delete self.init_args[ERROR_SOURCE]

        # # Flag for deferred initialization
        # self.init_status = InitStatus.DEFERRED_INITIALIZATION

        self._learning_rate = learning_rate

        super().__init__(variable=variable,
                         size=size,
                         modulation=modulation,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

    def _validate_variable(self, variable, context=None):
        """Validate that variable has exactly three items: activation_input, activation_output and error_signal
        """

        variable = self._update_variable(super()._validate_variable(variable, context))

        if len(variable) != 3:
            raise AutoAssociativeLearningMechanismError("Variable for {} ({}) must have three items ({}, {}, and {})".
                                format(self.name, variable,
                                       ACTIVATION_INPUT,
                                       ACTIVATION_OUTPUT,
                                       ERROR_SIGNAL))

        # Validate that activation_input, activation_output, and error_signal are numeric and lists or 1d np.ndarrays
        for i in range(len(variable)):
            item_num_string = ['first', 'second', 'third'][i]
            item_name = input_state_names[i]
            if not np.array(variable[i]).ndim == 1:
                raise AutoAssociativeLearningMechanismError("The {} item of variable for {} ({}:{}) is not a list or 1d np.array".
                                              format(item_num_string, self.name, item_name, variable[i]))
            if not (is_numeric(variable[i])):
                raise AutoAssociativeLearningMechanismError("The {} item of variable for {} ({}:{}) is not numeric".
                                              format(item_num_string, self.name, item_name, variable[i]))
        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate error_source as an Objective Mechanism or another LearningMechanism
        """

        super()._validate_params(request_set=request_set, target_set=target_set,context=context)

        if ERROR_SOURCE in target_set:
            if not isinstance(target_set[ERROR_SOURCE], (ObjectiveMechanism, LearningMechanism)):
                raise AutoAssociativeLearningMechanismError("{} arg for {} ({}) must be an ObjectiveMechanism or another "
                                             "LearningMechanism".
                                             format(ERROR_SOURCE, self.name, target_set[ERROR_SOURCE]))


        # FIX: REPLACE WITH CALL TO _parse_state_spec WITH APPROPRIATE PARAMETERS
        if LEARNING_SIGNALS in target_set and target_set[LEARNING_SIGNALS]:

            from PsyNeuLink.Components.States.ModulatorySignals.LearningSignal \
                import LearningSignal
            from PsyNeuLink.Components.States.ParameterState import ParameterState
            from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection

            for spec in target_set[LEARNING_SIGNALS]:

                learning_proj = None  # Projection from LearningSignal to MappingProjection
                mapping_proj = None   # MappingProjection that receives Projection from LearningSignal

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
        """Instantiates MappingProjection from error_source (if specified) to the AutoAssociativeLearningMechanism.

        Also assigns learned_projection attribute (to MappingProjection being learned)
        """

        super()._instantiate_attributes_before_function(context=context)

        if self.error_source:
            _instantiate_error_signal_projection(sender=self.error_source, receiver=self)

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
            del self._output_states[1]
            for i, learning_signal in enumerate(self.learning_signals):
                # Instantiate LearningSignal
                ls = self._instantiate_learning_signal(learning_signal=learning_signal, context=context)
                # Add LearningSignal to ouput_states list
                self._output_states.append(ls)
                # Replace spec in learning_signals list with actual LearningSignal
                self.learning_signals[i] = ls

        super()._instantiate_output_states(context=context)

    def _instantiate_learning_signal(self, learning_signal=None, context=None):
        """Instantiate LearningSignal OutputState and assign (if specified) or instantiate LearningProjection

        Notes:
        * learning_signal arg can be a:
            - LearningSignal object;
            - LearningProjection;
            - ParameterState;
            - Projection (in which case, MATRIX parameter is used as receiver of LearningProjection)
            - params dict containing a LearningProjection;
            - tuple (param_name, PROJECTION), from learning_signals arg of constructor;
                    [NOTE: this is a convenience format;
                           it precludes specification of LearningSignal params (??e.g., LEARNING_RATE)]
            - LearningSignal specification dictionary, from learning_signals arg of constructor
                    [NOTE: this must have at least NAME:str (param name) and MECHANISM:Mechanism entries;
                           it can also include a PARAMS entry with a params dict containing LearningSignal params]
            * NOTE: ParameterState must be for a Projection, and generally for MATRIX parameter of a MappingProjection;
                    however, LearningSignal is implemented to be applicable for any ParameterState of any Projection.
        * State._parse_state_spec() is used to parse learning_signal arg
        * params are expected to be for (i.e., to be passed to) LearningSignal;
        * wait to instantiate deferred_init() Projections until after LearningSignal is instantiated,
             so that correct OutputState can be assigned as its sender;
        * index of OutputState is incremented based on number of LearningSignals already instantiated;
            this means that a AutoAssociativeLearningMechanism.'s function must return as many items as it has LearningSignals,
            with each item of the function's value used by a corresponding LearningSignal.
            NOTE: multiple LearningProjections can be assigned to the same LearningSignal to implement "ganged" learning
                  (that is, learning of many Projections with a single value)

        Returns LearningSignal (OutputState)
        """

# FIX: THESE NEEDS TO BE DEALT WITH
# FIX: learning_projection -> learning_projections
# FIX: trained_projection -> learned_projection
# FIX: error_signal ??OR?? error_signals??
# FIX: LearningMechanism: learned_projection attribute -> learned_projections list
#                         learning_signal -> learning_signals (WITH SINGULAR ONE INDEXING INTO learning_signals.values)
#  FIX: THIS MAY NEED TO BE A 3d array (TO ACCOMDOATE 2d array (MATRICES) AS ENTRIES)

        from PsyNeuLink.Components.States.ModulatorySignals.LearningSignal import LearningSignal
        from PsyNeuLink.Components.States.State import _parse_state_spec
        from PsyNeuLink.Components.States.ParameterState import ParameterState, _get_parameter_state
        from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection

        # FIX: NEED TO CHARACTERIZE error_signal FOR BELOW
        # # EXTEND error_signals TO ACCOMMODATE NEW LearningSignal -------------------------------------------------
        # #        also used to determine constraint on LearningSignal output value
        #
        # if not hasattr(self, ERROR_SIGNALS) or self.error_signals is None:
        #     self.error_signals = np.array(defaultErrorSignal)
        # else:
        #     self.error_signals = np.append(self.error_signals, defaultErrorSignal)

        # GET index FOR LearningSignal OutputState
        try:
            output_state_index = len(self.output_states)
        except (AttributeError, TypeError):
            output_state_index = 0


        # PARSE learning_signal SPECIFICATION -----------------------------------------------------------------------

        learning_projection = None
        mapping_projection = None
        learning_signal_params = None

        learning_signal_spec = _parse_state_spec(owner=self, state_type=LearningSignal, state_spec=learning_signal)

        # Specification is a ParameterState
        if isinstance(learning_signal_spec, ParameterState):
            mapping_projection = learning_signal_spec.owner
            if not isinstance(mapping_projection, MappingProjection):
                raise AutoAssociativeLearningMechanismError("{} specified for {} of {} ({}) must be a {}".
                                             format(PARAMETER_STATE,
                                                    LEARNING_SIGNAL,
                                                    self.name,
                                                    mapping_projection,
                                                    PROJECTION))
            param_name = learning_signal_spec.name
            parameter_state = _get_parameter_state(self, LEARNING_SIGNAL, param_name, mapping_projection)

        # Specification was projection, tuple or dict, and parsed into a dict
        elif isinstance(learning_signal_spec, dict):
            param_name = learning_signal_spec[NAME]
            learning_signal_params = learning_signal_spec[PARAMS]

            # learning_signal was a specification dict, with PROJECTION as an entry (and parameter as NAME)
            if learning_signal_params and PROJECTION in learning_signal_params:
                mapping_projection = learning_signal_params[PROJECTION]
                # Delete PROJECTION entry as it is not a parameter of LearningSignal
                #     (which will balk at it in LearningSignal._validate_params)
                del learning_signal_params[PROJECTION]
                parameter_state = _get_parameter_state(self, LEARNING_SIGNAL, param_name, mapping_projection)

            # Specification either a projection
            # or originally a tuple (either in parameter specification or learning_signal arg):
            #    1st item was either assigned to the NAME entry of the learning_signal_spec dict
            #        (if tuple was a (param_name, Projection tuple) for learning_signal arg;
            #        or used as param value, if it was a parameter specification tuple
            #    2nd item was placed in learning_signal_params entry of params dict in learning_signal_spec dict,
            #        so parse:
            # FIX 5/23/17: NEED TO GET THE KEYWORDS STRAIGHT FOR PASSING LearningSignal SPECIFICATIONS
            # IMPLEMENTATION NOTE:
            #    PROJECTIONS is used by _parse_state_spec to place the 2nd item of any tuple in params dict;
            #                      here, the tuple comes from a (param, Projection) specification in learning_signal arg
            #    Delete whichever one it was, as neither is a recognized LearningSignal param
            #        (which will balk at it in LearningSignal._validate_params)
            elif (learning_signal_params and
                    any(kw in learning_signal_spec[PARAMS] for kw in {LEARNING_SIGNAL_SPECS, PROJECTIONS})):
                if LEARNING_SIGNAL_SPECS in learning_signal_spec[PARAMS]:
                    spec = learning_signal_params[LEARNING_SIGNAL_SPECS]
                    del learning_signal_params[LEARNING_SIGNAL_SPECS]
                elif PROJECTIONS in learning_signal_spec[PARAMS]:
                    spec = learning_signal_params[PROJECTIONS]
                    del learning_signal_params[PROJECTIONS]

                # LearningSignal
                if isinstance(spec, LearningSignal):
                    learning_signal_spec = spec

                # Projection
                else:
                    # IMPLEMENTATION NOTE: Projection was placed in list in PROJECTIONS entry by _parse_state_spec
                    if isinstance(spec, list) and isinstance(spec[0], Projection):
                        if isinstance(spec[0], MappingProjection):
                            mapping_projection = spec[0]
                            param_name = MATRIX
                            parameter_state = _get_parameter_state(self,
                                                                   LEARNING_SIGNAL,
                                                                   param_name,
                                                                   mapping_projection)
                            # MODIFIED 7/21/17 NEW:
                            learning_projection = LearningProjection(receiver=parameter_state)
                            # MODIFIED 7/21/17 END
                        elif isinstance(spec[0], LearningProjection):
                            learning_projection = spec[0]
                            if learning_projection.init_status is InitStatus.DEFERRED_INITIALIZATION:
                                parameter_state = learning_projection.init_args['receiver']
                            else:
                                parameter_state = learning_projection.receiver
                            param_name = parameter_state.name
                        else:
                            raise AutoAssociativeLearningMechanismError("PROGRAM ERROR: list in {} entry of params dict for {} of {} "
                                                        "must contain a single MappingProjection or LearningProjection".
                                                        format(LEARNING_SIGNAL_SPECS, learning_signal, self.name))

                        if len(spec)>1:
                            raise AutoAssociativeLearningMechanismError("PROGRAM ERROR: Multiple LearningProjections is not "
                                                        "currently supported in specification of a LearningSignal")
                    else:
                        raise AutoAssociativeLearningMechanismError("PROGRAM ERROR: failure to parse specification of {} for {}".
                                                    format(learning_signal, self.name))
            else:
                raise AutoAssociativeLearningMechanismError("PROGRAM ERROR: No entry found in params dict with specification of "
                                            "parameter Projection or LearningProjection for {} of {}".
                                            format(learning_signal, self.name))


        # INSTANTIATE LearningSignal -----------------------------------------------------------------------------------

        # Specification is a LearningSignal (either passed in directly, or parsed from tuple above)
        if isinstance(learning_signal_spec, LearningSignal):
            # Deferred Initialization, so assign owner, name, and initialize
            if learning_signal_spec.init_status is InitStatus.DEFERRED_INITIALIZATION:
                # FIX 5/23/17:  IMPLEMENT DEFERRED_INITIALIZATION FOR LearningSignal
                # CALL DEFERRED INIT WITH SELF AS OWNER ??AND NAME FROM learning_signal_dict?? (OR WAS IT SPECIFIED)
                # OR ASSIGN NAME IF IT IS DEFAULT, USING learning_signal_DICT??
                pass
            elif not learning_signal_spec.owner is self:
                raise AutoAssociativeLearningMechanismError("Attempt to assign LearningSignal to {} ({}) that is already owned by {}".
                                            format(self.name,
                                                   learning_signal_spec.name,
                                                   learning_signal_spec.owner.name))
            learning_signal = learning_signal_spec
            learning_signal_name = learning_signal_spec.name
            learning_projections = learning_signal_spec.efferents

            # IMPLEMENTATION NOTE:
            #    THIS IS TO HANDLE FUTURE POSSIBILITY OF MULTIPLE ControlProjections FROM A SINGLE LearningSignal;
            #    FOR NOW, HOWEVER, ONLY A SINGLE ONE IS SUPPORTED
            # parameter_states = [proj.recvr for proj in learning_projections]
            if len(learning_projections) > 1:
                raise AutoAssociativeLearningMechanismError("PROGRAM ERROR: list of ControlProjections is not currently supported "
                                            "as specification in a LearningSignal")
            else:
                learning_projection = learning_projections[0]
                parameter_state = learning_projection.receiver

        # Specification is not a LearningSignal, so create OutputState for it
        else:
            learning_signal_name = param_name + '_' + LearningSignal.__name__

            from PsyNeuLink.Components.States.ModulatorySignals.LearningSignal \
                import LearningSignal
            from PsyNeuLink.Components.States.State import _instantiate_state

            # Get constraint for OutputState's value
            # - assume that AutoAssociativeLearningMechanism.value has only two items (learning_signal and error_signal)
            # - use learning_signal (stored in self.learning_signal) as value for all LearningSignals
            self._update_value(context=context)
            constraint_value = self.learning_signal
            learning_signal_params.update({LEARNED_PARAM:param_name})

            learning_signal = _instantiate_state(owner=self,
                                                state_type=LearningSignal,
                                                state_name=learning_signal_name,
                                                state_spec=constraint_value,
                                                state_params=learning_signal_params,
                                                constraint_value=constraint_value,
                                                constraint_value_name='Default control allocation',
                                                context=context)

        # VALIDATE OR INSTANTIATE LearningProjection(s) FROM LearningSignal  -------------------------------------------

        # Validate learning_projection (if specified) and get receiver's name
        if learning_projection:
            _validate_receiver(self, learning_projection, MappingProjection, LEARNING_SIGNAL, context=context)

            from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection
            if not isinstance(learning_projection, LearningProjection):
                raise AutoAssociativeLearningMechanismError("PROGRAM ERROR: Attempt to assign {}, "
                                                  "that is not a LearningProjection, to LearningSignal of {}".
                                                  format(learning_projection, self.name))
            if learning_projection.init_status is InitStatus.DEFERRED_INITIALIZATION:
                learning_projection.init_args['sender']=learning_signal
                if learning_projection.init_args['name'] is None:
                    # FIX 5/23/17: CLEAN UP NAME STUFF BELOW:
                    learning_projection.init_args['name'] = LEARNING_PROJECTION + \
                                                   ' for ' + parameter_state.owner.name + ' ' + parameter_state.name
                learning_projection._deferred_init()
            else:
                learning_projection.sender = learning_signal

            # Add LearningProjection to list of LearningSignal's outgoing Projections
            # (note: if it was deferred, it just added itself, skip)
            if not learning_projection in learning_signal.efferents:
                learning_signal.efferents.append(learning_projection)

            # Add LearningProjection to AutoAssociativeLearningMechanism.'s list of LearningProjections
            try:
                self.learning_projections.append(learning_projection)
            except AttributeError:
                self.learning_projections = [learning_projection]

        return learning_signal

    def _instantiate_attributes_after_function(self, context=None):

        if self._learning_rate is not None:
            self.learning_rate = self._learning_rate

        super()._instantiate_attributes_after_function(context=context)

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
        self.learning_signal, self.error_signal = self.function(variable=variable,
                                                                params=runtime_params,
                                                                context=context)

        if not INITIALIZING in context and self.reportOutputPref:
            print("\n{} weight change matrix: \n{}\n".format(self.name, self.learning_signal))

        self.value = [self.learning_signal, self.error_signal]
        return self.value

    @property
    def learning_rate(self):
        return self.function_object.learning_rate

    @learning_rate.setter
    def learning_rate(self, assignment):
        self.function_object.learning_rate = assignment

    @property
    def input_source(self):
        try:
            return self.input_states[ACTIVATION_INPUT].path_afferents[0].sender.owner
        except IndexError:
            return None

    @property
    def output_source(self):
        try:
            return self.input_states[ACTIVATION_OUTPUT].path_afferents[0].sender.owner
        except IndexError:
            return None

    @property
    def primary_learned_projection(self):
        return self.learned_projection[0]

    @property
    def learned_projections(self):
        return [lp.receiver.owner for lp in self.learning_projections]
