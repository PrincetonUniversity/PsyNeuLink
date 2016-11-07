# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ******************************************  OutputState *****************************************************

"""

.. _OutputStates_Creating_An_OutputState:

Creating an OutputState
-----------------------

An outputState can be created by calling the class directly, but more commonly it is done by specifying one (or more)
outputStates in the `OUTPUT_STATES` entry of a params dictionary when creating a :class:`mechanism`. An outputState must
be owned by a mechanism.  If the outputState is created directly, the mechanism to which it belongs must be specified
in ``owner`` argument when calling the class;  if the outputState is specified in the OUTPUT_STATES entry of parameter
dictionary for a mechanism, then the owner is inferred from the context.

+ OUTPUT_STATES (value, list, dict):
    supports the ability of a subclass to define specialized outputStates;
    only used if OUTPUT_STATES is an argument in the subclass' __init__ or
    is specified as a parameter in the subclass' paramClassDefaults.
    In those cases:
        if param is absent or is a str:
            a default OutputState will be instantiated using output of mechanism's execute method (EMO)
            (and the str, if provided used as its name);
            it will be placed as the single entry in an OrderedDict
        if param is a single value:
            it will (if necessary) be instantiated and placed as the single entry in an OrderedDict
        if param is a list:
            each item will (if necessary) be instantiated and placed in an OrderedDict
        if param is an OrderedDict:
            each entry will (if necessary) be instantiated as a OutputState
        in each case, the result will be an OrderedDict of one or more entries:
            the key for the entry will be the name of the outputState if provided, otherwise
                OUTPUT_STATES-n will used (with n incremented for each entry)
            the value of the outputState in each entry will be assigned to the corresponding item of the EMO
            the dict will be assigned to both self.outputStates and paramsCurrent[OUTPUT_STATES]
            self.outputState will be pointed to self.outputStates[0] (the first entry of the dict)
        notes:
            * if there is only one outputState, but the EMV has more than one item, it is assigned to the
                the sole outputState, which is assumed to have a multi-item value
            * if there is more than one outputState, the number must match length of EMO,
              or an exception is raised
        specification of the param value, list item, or dict entry value can be any of the following, a
        long as it is compatible with the relevant item of the output of the mechanism's function (EMO):
            + OutputState class: default outputState will be instantiated using EMO as its value
            + OutputState object: its value must be compatible with EMO
            + specification dict:  OutputState will be instantiated using EMO as its value;
                must contain the following entries: (see Initialization arguments for State):
                    + FUNCTION (method)
                    + FUNCTION_PARAMS (dict)
            + str:
                will be used as name of a default outputState (and key for its entry in self.outputStates)
                value must match value of the corresponding item of the mechanism's EMO
            + value:
                will be used a variable to instantiate a OutputState; value must be compatible with EMO
        * note: inputStates can also be added using State.instantiate_state()

"""

# import Components
from PsyNeuLink.Components.States.State import *
from PsyNeuLink.Components.Functions.Function import *

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
        The OutputState class is a type in the State category of Component,
        It is used primarily as the sender for Mapping projections
        Its FUNCTION updates its value:
            note:  currently, this is the identity function, that simply maps variable to self.value

    Instantiation:
        - OutputStates can be instantiated in one of two ways:
            - directly: requires explicit specification of its value and owner
            - as part of the instantiation of a mechanism:
                - the mechanism for which it is being instantiated will automatically be used as the owner
                - the owner's self.value will be used as its value
        - self.value is set to self.variable (enforced in State_Base._validate_variable)
        - self.function (= params[FUNCTION]) should be an identity function (enforced in _validate_params)

        - if owner is being instantiated within a pathway:
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
        + componentType (str) = OUTPUT_STATES
        + paramClassDefaults (dict)
            + FUNCTION (LinearCombination)
            + FUNCTION_PARAMS   (Operation.PRODUCT)
        + paramNames (dict)

    Class methods:
        function (executes function specified in params[FUNCTION];  default: LinearCombination with Operation.SUM)

    Instance attributes:
        + paramInstanceDefaults (dict) - defaults for instance (created and validated in Components init)
        + params (dict) - set currently in effect
        + paramNames (list) - list of keys for the params dictionary
        + owner (Mechanism)
        + value (value)
        + projections (list)
        + params (dict)
        + name (str)
        + prefs (dict)

    Instance methods:
        none
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
    paramClassDefaults.update({PROJECTION_TYPE: MAPPING})
    #endregion

    tc.typecheck
    def __init__(self,
                 owner,
                 reference_value,
                 value=NotImplemented,
                 function=LinearCombination(operation=SUM),
                 params=NotImplemented,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):
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

        :param owner: (Mechanism object)
        :param reference_value: (value)
        :param value: (value)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        :param context: (str)
        :return:
        """

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function, params=params)

        self.reference_value = reference_value

        # FIX: 5/26/16
        # IMPLEMENTATION NOTE:
        # Consider adding self to owner.outputStates here (and removing from ControlSignal._instantiate_sender)
        #  (test for it, and create if necessary, as per outputStates in ControlSignal._instantiate_sender),

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        super().__init__(owner,
                         value=value,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)


    def _validate_variable(self, variable, context=None):
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

        super(OutputState,self)._validate_variable(variable, context)

        self.variableClassDefault = self.reference_value

        # Insure that self.variable is compatible with (relevant item of) output value of owner's function
        if not iscompatible(self.variable, self.reference_value):
            raise OutputStateError("Value ({0}) of outputState for {1} is not compatible with "
                                           "the output ({2}) of its function".
                                           format(self.value,
                                                  self.owner.name,
                                                  self.reference_value))

def instantiate_output_states(owner, context=None):
    """Call State.instantiate_state_list() to instantiate orderedDict of outputState(s)

    Create OrderedDict of outputState(s) specified in paramsCurrent[INPUT_STATES]
    If INPUT_STATES is not specified, use self.variable to create a default output state
    When completed:
        - self.outputStates contains an OrderedDict of one or more outputStates
        - self.outputState contains first or only outputState in OrderedDict
        - paramsCurrent[OUTPUT_STATES] contains the same OrderedDict (of one or more outputStates)
        - each outputState corresponds to an item in the output of the owner's function
        - if there is only one outputState, it is assigned the full value

    (See State.instantiate_state_list() for additional details)

    IMPLEMENTATION NOTE:
        default(s) for self.paramsCurrent[OUTPUT_STATES] (self.value) is assigned here
        rather than in _validate_params, as it requires function to have been instantiated first

    :param context:
    :return:
    """
    owner.outputStates = instantiate_state_list(owner=owner,
                                                state_list=owner.paramsCurrent[OUTPUT_STATES],
                                                state_type=OutputState,
                                                state_param_identifier=OUTPUT_STATES,
                                                constraint_value=owner.value,
                                                constraint_value_name="output",
                                                context=context)
    # Assign self.outputState to first outputState in dict
    owner.outputState = list(owner.outputStates.values())[0]
