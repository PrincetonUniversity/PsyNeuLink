# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ******************************************  OutputState *****************************************************
#

# import Functions
from PsyNeuLink.Functions.States.State import *
from PsyNeuLink.Functions.Utilities.Utility import *

# class OutputStateLog(IntEnum):
#     NONE            = 0
#     TIME_STAMP      = 1 << 0
#     ALL = TIME_STAMP
#     DEFAULTS = NONE


class OutputStateError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class OutputState(State_Base):
    """Implement subclass type of State, that represents output of its owner

    Description:
        The OutputState class is a functionType in the State category of Function,
        It is used primarily as the sender for Mapping projections
        Its FUNCTION updates its value:
            note:  currently, this is the identity function, that simply maps variable to self.value

    Instantiation:
        - OutputStates can be instantiated in one of two ways:
            - directly: requires explicit specification of its value and owner
            - as part of the instantiation of a mechanism:
                - the mechanism for which it is being instantiated will automatically be used as the owner
                - the owner's self.value will be used as its value
        - self.value is set to self.variable (enforced in State_Base.validate_variable)
        - self.function (= params[FUNCTION]) should be an identity function (enforced in validate_params)

        - if owner is being instantiated within a configuration:
            - OutputState will be assigned as the sender of a projection to the subsequent mechanism
            - if it is the last mechanism in the list, it will send a projection to process.output

    StateRegistry:
        All OutputStates are registered in StateRegistry, which maintains an entry for the subclass,
          a count for all instances of it, and a dictionary of those instances

    Naming:
        kwInputState can be named explicitly (using the name='<name>' argument). If this argument is omitted,
         it will be assigned "OutputState" with a hyphenated, indexed suffix ('OutputState-n')

    Parameters:
        The default for FUNCTION is LinearMatrix using MATRIX: IDENTITY_MATRIX:
        The parameters of FUNCTION can be set:
            - by including them at initialization (param[FUNCTION] = <function>(sender, params)
            - calling the adjust method, which changes their default values (param[FUNCTION].adjust(params)
            - at run time, which changes their values for just for that call (self.execute(sender, params)

    Class attributes:
        + functionType (str) = kwOutputStates
        + paramClassDefaults (dict)
            + FUNCTION (LinearCombination)
            + FUNCTION_PARAMS   (Operation.PRODUCT)
        + paramNames (dict)

    Class methods:
        function (executes function specified in params[FUNCTION];  default: LinearCombination with Operation.SUM)

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

    functionType = kwOutputStates
    paramsType = kwOutputStateParams

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'OutputStateCustomClassPreferences',
    #     kp<pref>: <setting>...}

    paramClassDefaults = State_Base.paramClassDefaults.copy()
    paramClassDefaults.update({PROJECTION_TYPE: MAPPING})
    #endregion

    def __init__(self,
                 owner,
                 reference_value,
                 value=NotImplemented,
                 function=LinearCombination(operation=LinearCombination.Operation.SUM),
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented,
                 context=NotImplemented):
        """
IMPLEMENTATION NOTE:  *** DOCUMENTATION NEEDED (SEE CONTROL SIGNAL??)
                      *** EXPLAIN owner_output_value:
reference_value is component of owner.variable that corresponds to the current State

        # Potential problem:
        #    - a OutputState may correspond to a particular item of owner.value
        #        in which case there will be a mismatch here
        #    - if OutputState is being instantiated from Mechanism (in instantiate_output_states)
        #        then the item of owner.value is known and has already been checked
        #        (in the call to instantiate_state)
        #    - otherwise, should ignore

        :param owner: (Function object)
        :param reference_value: (value)
        :param value: (value)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        :param context: (str)
        :return:
        """

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self.assign_args_to_param_dicts(function=function, params=params)

        self.reference_value = reference_value

        # FIX: 5/26/16
        # IMPLEMENTATION NOTE:
        # Consider adding self to owner.outputStates here (and removing from ControlSignal.instantiate_sender)
        #  (test for it, and create if necessary, as per outputStates in ControlSignal.instantiate_sender),

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        super().__init__(owner,
                         value=value,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)


    def validate_variable(self, variable, context=NotImplemented):
        """Insure variable is compatible with output component of owner.function relevant to this state

        Validate self.variable against component of owner's value (output of owner's function)
             that corresponds to this outputState (since that is what is used as the input to OutputState);
             this should have been provided as reference_value in the call to OutputState__init__()

        Note:
        * This method is called only if the parameterValidationPref is True

        :param variable: (anything but a dict) - variable to be validated:
        :param context: (str)
        :return none:
        """

        super(OutputState,self).validate_variable(variable, context)

        self.variableClassDefault = self.reference_value

        # Insure that self.variable is compatible with (relevant item of) output value of owner's function
        if not iscompatible(self.variable, self.reference_value):
            raise OutputStateError("Value ({0}) of outputState for {1} is not compatible with "
                                           "the output ({2}) of its function".
                                           format(self.value,
                                                  self.owner.name,
                                                  self.reference_value))

def instantiate_output_states(owner, context=NotImplemented):
    """Call State.instantiate_state_list() to instantiate orderedDict of outputState(s)

    Create OrderedDict of outputState(s) specified in paramsCurrent[kwInputStates]
    If kwInputStates is not specified, use self.variable to create a default output state
    When completed:
        - self.outputStates contains an OrderedDict of one or more outputStates
        - self.outputState contains first or only outputState in OrderedDict
        - paramsCurrent[kwOutputStates] contains the same OrderedDict (of one or more outputStates)
        - each outputState corresponds to an item in the output of the owner's function
        - if there is only one outputState, it is assigned the full value

    (See State.instantiate_state_list() for additional details)

    IMPLEMENTATION NOTE:
        default(s) for self.paramsCurrent[kwOutputStates] (self.value) is assigned here
        rather than in validate_params, as it requires function to have been instantiated first

    :param context:
    :return:
    """
    owner.outputStates = instantiate_state_list(owner=owner,
                                                state_list=owner.paramsCurrent[kwOutputStates],
                                                state_type=OutputState,
                                                state_param_identifier=kwOutputStates,
                                                constraint_value=owner.value,
                                                constraint_value_name="output",
                                                context=context)
    # Assign self.outputState to first outputState in dict
    owner.outputState = list(owner.outputStates.values())[0]
