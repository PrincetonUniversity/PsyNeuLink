# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *******************************************  InputState *****************************************************
#

from PsyNeuLink.Functions.States.State import *
from PsyNeuLink.Functions.Utility import *


# InputStatePreferenceSet = FunctionPreferenceSet(log_pref=logPrefTypeDefault,
#                                                          reportOutput_pref=reportOutputPrefTypeDefault,
#                                                          verbose_pref=verbosePrefTypeDefault,
#                                                          param_validation_pref=paramValidationTypeDefault,
#                                                          level=PreferenceLevel.TYPE,
#                                                          name='InputStateClassPreferenceSet')

# class InputStateLog(IntEnum):
#     NONE            = 0
#     TIME_STAMP      = 1 << 0
#     ALL = TIME_STAMP
#     DEFAULTS = NONE


class InputStateError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class InputState(State_Base):
    """Implement subclass type of State that calculates and represents the input of a Function object

    Description:
        The InputState class is a functionType in the State category of Function,
        Its kwFunction executes the projections that it receives and updates the InputState's value

    Instantiation:
        - kwInputState can be instantiated in one of two ways:
            - directly: requires explicit specification of its value and owner
            - as part of the instantiation of a mechanism:
                - the mechanism for which it is being instantiated will automatically be used as the owner
                - the value of the owner's variable will be used as its value
        - self.value is set to self.variable (enforced in State_Base.validate_variable)
        - self.value must be compatible with self.owner.variable (enforced in validate_variable)
            note: although it may receive multiple projections, the output of each must conform to self.variable,
                  as they will be combined to produce a single value that must be compatible with self.variable
        - self.function (= params[kwFunction]) must be Utility.LinearCombination (enforced in validate_params)
        - output of self.function must be compatible with self.value (enforced in validate_params)
        - if owner is being instantiated within a configuration:
            - InputState will be assigned as the receiver of a Mapping projection from the preceding mechanism
            - if it is the first mechanism in the list, it will receive a Mapping projection from process.input

    Parameters:
        The default for kwFunction is LinearCombination using kwAritmentic.Operation.SUM:
            the output of all projections it receives are summed
# IMPLEMENTATION NOTE:  *** CONFIRM THAT THIS IS TRUE:
        kwFunction can be set to another function, so long as it has type kwLinearCombinationFunction
        The parameters of kwFunction can be set:
            - by including them at initialization (param[kwFunction] = <function>(sender, params)
            - calling the adjust method, which changes their default values (param[kwFunction].adjust(params)
            - at run time, which changes their values for just for that call (self.execute(sender, params)

    StateRegistry:
        All kwInputState are registered in StateRegistry, which maintains an entry for the subclass,
          a count for all instances of it, and a dictionary of those instances

    Naming:
        kwInputState can be named explicitly (using the name='<name>' argument). If this argument is omitted,
         it will be assigned "InputState" with a hyphenated, indexed suffix ('InputState-n')


    Class attributes:
        + functionType (str) = kwInputState
        + paramClassDefaults (dict)
            + kwFunction (LinearCombination, Operation.SUM)
            + kwFunctionParams (dict)
            # + kwStateProjectionAggregationFunction (LinearCombination, Operation.SUM)
            # + kwStateProjectionAggregationMode (LinearCombination, Operation.SUM)
        + paramNames (dict)

    Class methods:
        instantiate_execute_method: insures that execute method is ARITHMETIC)
        update_state: gets InputStateParams and passes to super (default: LinearCombination with Operation.SUM)



    Instance attributes:
        + paramInstanceDefaults (dict) - defaults for instance (created and validated in Functions init)
        + params (dict) - set currently in effect
        + paramNames (list) - list of keys for the params dictionary
        + owner (Function object)
        + value (value)
        + projections (list)
        + params (dict)
        + name (str)
        + prefs (dict)

    Instance methods:
        none
    """

    #region CLASS ATTRIBUTES

    functionType = kwInputState

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'InputStateCustomClassPreferences',
    #     kp<pref>: <setting>...}

    # Note: the following enforce encoding as 1D np.ndarrays (one variable/value array per state)
    variableEncodingDim = 1
    valueEncodingDim = 1

    paramClassDefaults = State_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
                               # kwFunction: LinearCombination,
                               # kwFunctionParams: {kwOperation: LinearCombination.Operation.SUM},
                               kwProjectionType: kwMapping})

    #endregion

    def __init__(self,
                 owner,
                 reference_value=NotImplemented,
                 value=NotImplemented,
                 function=LinearCombination(operation=LinearCombination.Operation.SUM),
                 params=None,
                 name=NotImplemented,
                 prefs=NotImplemented,
                 context=NotImplemented):
        """
IMPLEMENTATION NOTE:  *** DOCUMENTATION NEEDED (SEE CONTROL SIGNAL??)
reference_value is component of owner.variable that corresponds to the current State

        :param owner: (Function object)
        :param reference_value: (value)
        :param value: (value)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        :param context: (str)
        :return:


        """

        params = self.assign_args_to_param_dicts(function=function,
                                                 params=params)

        # Assign functionType to self.name as default;
        #  will be overridden with instance-indexed name in call to super
        if name is NotImplemented:
            self.name = self.functionType
        else:
            self.name = name

        self.functionName = self.functionType

        self.reference_value = reference_value

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        # Note: pass name of owner (to override assignment of functionName in super.__init__)
        super(InputState, self).__init__(owner,
                                                  value=value,
                                                  params=params,
                                                  name=name,
                                                  prefs=prefs,
                                                  context=self)

    def instantiate_execute_method(self, context=NotImplemented):
        """Insure that execute method is LinearCombination and that output is compatible with owner.variable

        Insures that execute method:
            - is LinearCombination (to aggregate projection inputs)
            - generates an output (assigned to self.value) that is compatible with the component of
                owner.function's variable that corresponds to this inputState,
                since the latter will be called with the value of this InputState;

        Notes:
        * Relevant component of owner.function's variable should have been provided
            as reference_value arg in the call to InputState__init__()
        * Insures that self.value has been assigned (by call to super().validate_execute_method)
        * This method is called only if the parameterValidationPref is True

        :param context:
        :return:
        """

        super(InputState, self).instantiate_execute_method(context=context)

        # Insure that execute method is Utility.LinearCombination
        if not isinstance(self.execute.__self__, (LinearCombination, Linear)):
            raise StateError("{0} of {1} for {2} is {3}; it must be of LinearCombination or Linear type".
                                      format(kwFunction,
                                             self.name,
                                             self.owner.name,
                                             self.execute.__self__.functionName, ))

        # Insure that self.value is compatible with (relevant item of ) self.owner.variable
        if not iscompatible(self.value, self.reference_value):
            raise InputStateError("Value ({0}) of {1} for {2} owner is not compatible with "
                                           "the variable ({2}) of its execute method".
                                           format(self.value,
                                                  self.name,
                                                  self.owner.name,
                                                  self.owner.variable))

    def update(self, params=NotImplemented, time_scale=TimeScale.TRIAL, context=NotImplemented):
        """Process inputState params, and pass params for inputState projections to super for processing

        :param params:
        :param time_scale:
        :param context:
        :return:
        """

        try:
            # Get inputState params
            input_state_params = params[kwInputStateParams]

        except (KeyError, TypeError):
            input_state_params = NotImplemented

        # Process any inputState params here
        pass

        super(InputState, self).update(params=input_state_params,
                                                      time_scale=time_scale,
                                                      context=context)

def instantiate_input_states(owner, context=NotImplemented):
    """Call State.instantiate_state_list() to instantiate orderedDict of inputState(s)

    Create OrderedDict of inputState(s) specified in paramsCurrent[kwInputStates]
    If kwInputStates is not specified, use self.variable to create a default input state
    When completed:
        - self.inputStates contains an OrderedDict of one or more inputStates
        - self.inputState contains first or only inputState in OrderedDict
        - paramsCurrent[kwOutputStates] contains the same OrderedDict (of one or more inputStates)
        - each inputState corresponds to an item in the variable of the owner's execute method (EMV)
        - if there is only one inputState, it is assigned the full value

    Note: State.instantiate_state_list()
              parses self.variable (2D np.array, passed in constraint_value)
              into individual 1D arrays, one for each input state

    (See State.instantiate_state_list() for additional details)

    :param context:
    :return:
    """
    owner.inputStates = instantiate_state_list(owner=owner,
                                                        state_list=owner.paramsCurrent[kwInputStates],
                                                        state_type=InputState,
                                                        state_param_identifier=kwInputStates,
                                                        constraint_value=owner.variable,
                                                        constraint_value_name="execute method variable",
                                                        context=context)

    # Initialize self.inputValue to correspond to format of owner's variable, and zero it
# FIX: INSURE THAT ELEMENTS CAN BE FLOATS HERE:  GET AND ASSIGN SHAPE RATHER THAN COPY? XXX
    owner.inputValue = owner.variable.copy() * 0.0

    # Assign self.inputState to first inputState in dict
    try:
        owner.inputState = list(owner.inputStates.values())[0]
    except AttributeError:
        owner.inputState = None


