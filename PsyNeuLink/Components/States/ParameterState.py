# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  ParameterState ******************************************************

"""
**[DOCUMENTATION STILL UNDER CONSTRUCTION]**

Overview
--------

A parameterState accepts one or more ControlProjections and/or LearningProjections that modify the
parameters of its owner's ``function``.   A list of the projections received by an parameterState is
kept in its ``receivesFromProjections`` attribute.  It's ``function`` combines the values of these inputs,
and uses the result to modify the value of the ``function`` parameter for which it is responsible.


.. _ParameterState_Creation:

Creating a ParameterState
-------------------------

ParameterStates cannot be created directly.
are created automatically by the object to which they belong when that object is created;  they
cannot be constructed directly  one
ParameterState is created for each parameter of the object's ``function``.
for each
parameter of the mechanis'ms

  For example,
if the mechanism is
being created within the :ref:`pathway of a process <Process_Pathway>`, its inputState will be created and assigned as
the ``receiver`` of a MappingProjection from the  preceding mechanism in the pathway. If one or more custom inputStates
need to be specified when a mechanism is created, or added to an existing mechanism, this can be done in an entry of
the mechanism's parameter dictionary, using the key :keyword:`INPUT_STATES` [LINK] and a value that specifies one or
more inputStates. For a single inputState, the value can be any of the specifications in the the list below.  To
create multiple inputStates, the value of the :keyword:`INPUT_STATES` entry can be either a list, each item of
which is any of the specifications below;  or, it can be an OrderedDict, in which the key for each entry is a string
specifying the name for the inputState to be created, and its value is one of the specifications below:


                + ParameterState class ref: default will be instantiated using param with same name in EMP
                + ParameterState object: its value must be compatible with param of same name in EMP
                - projection: Projection object, Projection specifications dict, or list of either)
                    + Projection subclass ref:
                        default ParameterState will be instantiated using EMP
                        default projection (for ParameterState) will be instantiated using EMP
                            and assigned to ParameterState
                    + Projection object:
                        ParameterState will be instantiated using output of projection as its value;
                        this must be compatible with EMP
                    + Projection specification dict
                    + List[any of the above]
                + State specification dict:  ParameterState will be instantiated using EMP as its value;
                    must contain the following entries: (see Instantiation arguments for ParameterState):
                        + FUNCTION (method)
                        + FUNCTION_PARAMS (dict)
                        + STATE_PROJECTIONS (Projection, specifications dict, or list of either of these)
                + ParamValueProjection tuple:
                    value will be used as variable to instantiate a default ParameterState
                    projection will be assigned as projection to ParameterState
                + 2-item tuple : (value, projectionType)
                    [convenience notation;  should use ParamValueProjection for clarity]:
                    first item will be used as variable to instantiate a default ParameterState
                    second item will be assigned as projection to ParameterState
                + value : list of numbers (no projections will be assigned)
                    will be used as variable to instantiate a default ParameterState


parameter_modulation_operation:  ModulationOperation - list values and their meaning

"""

from PsyNeuLink.Components.States.State import *
from PsyNeuLink.Components.States.State import _instantiate_state
from PsyNeuLink.Components.Functions.Function import *

# class ParameterStateLog(IntEnum):
#     NONE            = 0
#     TIME_STAMP      = 1 << 0
#     ALL = TIME_STAMP
#     DEFAULTS = NONE


class ParameterStateError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


# class ParameterState_Base(State_Base):
class ParameterState(State_Base):
    """Implement subclass type of State that represents parameter value for function of a Mechanism

    Definition for ParameterState componentType in State category of Function class

    Description:
        The ParameterState class is a componentType in the State category of Function,
        Its FUNCTION executes the projections that it receives and updates the ParameterState's value

    Instantiation:
        - ParameterStates can be instantiated in one of two ways:
            - directly: requires explicit specification of its value and owner;
                - specification of value can be any of the forms allowed for specifying a State
                    (default value will be inferred from anything other than a value or ParamValueProjection tuple)
                - owner must be a reference to a Mechanism object, or DefaultProcessingMechanism_Base will be used
            - as part of the instantiation of a mechanism:
                - the mechanism for which it is being instantiated will automatically be used as the owner
                - the value of the owner's param for which the ParameterState is being instantiated
                    will be used as its variable (that must also be compatible with its self.value)
        - self.variable must be compatible with self.value (enforced in _validate_variable)
            note: although it may receive multiple projections, the output of each must conform to self.variable,
                  as they will be combined to produce a single value that must be compatible with self.variable
        - self.function (= params[FUNCTION]) must be Function.LinearCombination (enforced in _validate_params)

    Execution:
        - get ParameterStateParams
        - pass params to super, which aggregates inputs from projections
        - combine input from projections (processed in super) with baseValue using paramModulationOperation
        - combine result with value specified at runtime in PARAMETER_STATE_PARAMS
        - assign result to self.value

    StateRegistry:
        All ParameterStates are registered in StateRegistry, which maintains an entry for the subclass,
          a count for all instances of it, and a dictionary of those instances

    Naming:
        ParameterStates can be named explicitly (using the name argument). If this argument is omitted,
         it will be assigned "ParameterState" with a hyphenated, indexed suffix ('ParameterState-n')

    Parameters:
        The default for FUNCTION is LinearCombination using kwAritmentic.Operation.PRODUCT:
           self.value is multiplied by  the output of each of the  projections it receives
               (generally ControlProjections)
# IMPLEMENTATION NOTE:  *** CONFIRM THAT THIS IS TRUE:
        FUNCTION can be set to another function, so long as it has type kwLinearCombinationFunction
        The parameters of FUNCTION can be set:
            - by including them at initialization (param[FUNCTION] = <function>(sender, params)
            - calling the adjust method, which changes their default values (param[FUNCTION].adjust(params)
            - at run time, which changes their values for just for that call (self.execute(sender, params)
    Class attributes:
        + componentType (str) = kwMechanisParameterState
        + classPreferences
        + classPreferenceLevel (PreferenceLevel.Type)
        + paramClassDefaults (dict)
            + FUNCTION (LinearCombination)
            + FUNCTION_PARAMS  (Operation.PRODUCT)
            + PROJECTION_TYPE (CONTROL_PROJECTION)
            + PARAMETER_MODULATION_OPERATION   (ModulationOperation.MULTIPLY)
        + paramNames (dict)
    Class methods:
        _instantiate_function: insures that function is ARITHMETIC) (default: Operation.PRODUCT)
        update_state: updates self.value from projections, baseValue and runtime in PARAMETER_STATE_PARAMS

    Instance attributes:
        + paramInstanceDefaults (dict) - defaults for instance (created and validated in Components init)
        + params (dict) - set currently in effect
        + paramNames (list) - list of keys for the params dictionary
        + owner (Mechanism)
        + value (value)
        + params (dict)
        + baseValue (value)
        + projections (list)
        + modulationOperation (ModulationOperation)
        + name (str)
        + prefs (dict)

    Instance methods:
        none
    """

    #region CLASS ATTRIBUTES

    componentType = kwParameterState
    paramsType = PARAMETER_STATE_PARAMS

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'ParameterStateCustomClassPreferences',
    #     kp<pref>: <setting>...}


    paramClassDefaults = State_Base.paramClassDefaults.copy()
    paramClassDefaults.update({PROJECTION_TYPE: CONTROL_PROJECTION})
    #endregion

    tc.typecheck
    def __init__(self,
                 owner,
                 reference_value=None,
                 value=None,
                 function=LinearCombination(operation=PRODUCT),
                 parameter_modulation_operation=ModulationOperation.MULTIPLY,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function,
                                                 parameter_modulation_operation=parameter_modulation_operation,
                                                 params=params)

        self.reference_value = reference_value

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        # Note: pass name of mechanism (to override assignment of componentName in super.__init__)
        super(ParameterState, self).__init__(owner,
                                             value=value,
                                             params=params,
                                             name=name,
                                             prefs=prefs,
                                             context=self)

        self.modulationOperation = self.paramsCurrent[PARAMETER_MODULATION_OPERATION]

    def _instantiate_function(self, context=None):
        """Insure function is LinearCombination and that its output is compatible with param with which it is associated

        Notes:
        * Relevant param should have been provided as reference_value arg in the call to InputState__init__()
        * Insures that self.value has been assigned (by call to super()._validate_function)
        * This method is called only if the parameterValidationPref is True

        :param context:
        :return:
        """

        # If parameterState is for a matrix of a MappingProjection,
        #     its parameter_modulation_operation should be SUM (rather than PRODUCT)
        #         so that weight changes (e.g., from a learningSignals) are added rather than multiplied
        if self.name == MATRIX:
            # IMPLEMENT / TEST: ZZZ 10/20/16 THIS SHOULD BE ABLE TO REPLACE SPECIFICATION IN LEARNING PROJECTION
            self.params[PARAMETER_MODULATION_OPERATION] = ModulationOperation.ADD

        super()._instantiate_function(context=context)

        # Insure that function is LinearCombination
        if not isinstance(self.function.__self__, (LinearCombination)):
            raise StateError("Function {0} for {1} of {2} must be of LinearCombination type".
                                 format(self.function.__self__.componentName, FUNCTION, self.name))

        # # Insure that output of function (self.value) is compatible with relevant parameter value
        if not iscompatible(self.value, self.reference_value):
            raise ParameterStateError("Value ({0}) of {1} for {2} mechanism is not compatible with "
                                           "the variable ({3}) of its function".
                                           format(self.value,
                                                  self.name,
                                                  self.owner.name,
                                                  self.owner.variable))


    def update(self, params=NotImplemented, time_scale=TimeScale.TRIAL, context=None):
        """Parse params for parameterState params and XXX ***

# DOCUMENTATION:  MORE HERE:
        - get ParameterStateParams
        - pass params to super, which aggregates inputs from projections
        - combine input from projections (processed in super) with baseValue using paramModulationOperation
        - combine result with value specified at runtime in PARAMETER_STATE_PARAMS
        - assign result to self.value

        :param params:
        :param time_scale:
        :param context:
        :return:
        """

        super().update(params=params,
                       time_scale=time_scale,
                       context=context)

        #region COMBINE PROJECTIONS INPUT WITH BASE PARAM VALUE
        try:
            # Check whether modulationOperation has been specified at runtime
            self.modulationOperation = self.stateParams[PARAMETER_MODULATION_OPERATION]
        except (KeyError, TypeError):
            # If not, try to get from params (possibly passed from projection to ParameterState)
            try:
                self.modulationOperation = params[PARAMETER_MODULATION_OPERATION]
            except (KeyError, TypeError):
                pass
            # If not, ignore (leave self.modulationOperation assigned to previous value)
            pass

        # If self.value has not been set, assign to baseValue
        if self.value is None:
            if not context:
                context = kwAssign + ' Base Value'
            else:
                context = context + kwAssign + ' Base Value'
            self.value = self.baseValue

        # Otherwise, combine param's value with baseValue using modulatonOperation
        else:
            if not context:
                context = kwAssign + ' Modulated Value'
            else:
                context = context + kwAssign + ' Modulated Value'
            self.value = self.modulationOperation(self.baseValue, self.value)
        #endregion

        #region APPLY RUNTIME PARAM VALUES
        # If there are not any runtime params, or functionRuntimeParamsPref is disabled, return
        if (self.stateParams is NotImplemented or
                    self.prefs.functionRuntimeParamsPref is ModulationOperation.DISABLED):
            return

        # Assign class-level pref as default operation
        default_operation = self.prefs.functionRuntimeParamsPref

        try:
            value, operation = self.stateParams[self.name]

        except KeyError:
            # No runtime param for this param state
            return

        except TypeError:
            # If single ("exposed") value, use default_operation (class-level functionRuntimeParamsPref)
            self.value = default_operation(self.stateParams[self.name], self.value)
        else:
            # If tuple, use param-specific ModulationOperation as operation
            self.value = operation(value, self.value)

            # Assign class-level pref as default operation
            default_operation = self.prefs.functionRuntimeParamsPref
        #endregion

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, assignment):
        self._value = assignment

def _instantiate_parameter_states(owner, context=None):
    """Call _instantiate_state_list() to instantiate ParameterStates for subclass' function

    Instantiate parameter states for params specified in FUNCTION_PARAMS unless PARAMETER_STATES == False
    Use constraints (for compatibility checking) from paramsCurrent (inherited from paramClassDefaults)

    :param context:
    :return:
    """

    owner.parameterStates = {}

    try:
        function_param_specs = owner.paramsCurrent[FUNCTION_PARAMS]
    except KeyError:
        # No need to warn, as that already occurred in _validate_params (above)
        return
    else:
        try:
            no_parameter_states = not owner.params[PARAMETER_STATES]
        except KeyError:
            # PARAMETER_STATES not specified, so continue
            pass
        else:
            # PARAMETER_STATES was set to False, so do not instantiate any parameterStates
            if no_parameter_states:
                return
            # TBI / IMPLEMENT: use specs to implement paramterStates below
            # Notes:
            # * functionParams are still available in paramsCurrent;
            # # just no parameterStates instantiated for them.

        # Instantiate parameterState for each param in functionParams, using its value as the state_spec
        for param_name, param_value in function_param_specs.items():

            state = _instantiate_state(owner=owner,
                                      state_type=ParameterState,
                                      state_name=param_name,
                                      state_spec=param_value,
                                      state_params=None,
                                      constraint_value=param_value,
                                      constraint_value_name=param_name,
                                      context=context)
            if state:
                owner.parameterStates[param_name] = state
