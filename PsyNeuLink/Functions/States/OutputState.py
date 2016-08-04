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
from PsyNeuLink.Functions.Utility import *


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
    """Implement subclass type of State, that represents output of a Mechanism

    Description:
        The OutputState class is a functionType in the State category of Function,
        It is used primarily as the sender for Mapping projections
        Its kwExecuteMethod updates its value:
            note:  currently, this is the identity function, that simply maps variable to self.value

    Instantiation:
        - OutputStates can be instantiated in one of two ways:
            - directly: requires explicit specification of its value and owner
            - as part of the instantiation of a mechanism:
                - the mechanism for which it is being instantiated will automatically be used as the owner
                - the owner's self.value will be used as its value
        - self.value is set to self.variable (enforced in State_Base.validate_variable)
        - self.executeMethod (= params[kwExecuteMethod]) should be an identity function (enforced in validate_params)

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
        The default for kwExecuteMethod is LinearMatrix using kwMatrix: kwIdentityMatrix:
        The parameters of kwExecuteMethod can be set:
            - by including them at initialization (param[kwExecuteMethod] = <function>(sender, params)
            - calling the adjust method, which changes their default values (param[kwExecuteMethod].adjust(params)
            - at run time, which changes their values for just for that call (self.execute(sender, params)

    Class attributes:
        + functionType (str) = kwOutputStates
        + paramClassDefaults (dict)
            + kwExecuteMethod (LinearCombination)
            + kwExecuteMethodParams   (Operation.PRODUCT)
        + paramNames (dict)

    Class methods:
        function (executes function specified in params[kwExecuteMethod];  default: LinearCombination with Operation.SUM)

    Instance attributes:
        + paramInstanceDefaults (dict) - defaults for instance (created and validated in Functions init)
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

    functionType = kwOutputStates

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'OutputStateCustomClassPreferences',
    #     kp<pref>: <setting>...}

    paramClassDefaults = State_Base.paramClassDefaults.copy()
    paramClassDefaults.update({kwExecuteMethod: LinearCombination,
                               kwExecuteMethodParams : {kwOperation: LinearCombination.Operation.SUM},
                               kwProjectionType: kwMapping})
    #endregion

    def __init__(self,
                 owner_mechanism,
                 reference_value,
                 value=NotImplemented,
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented,
                 context=NotImplemented):
        """
IMPLEMENTATION NOTE:  *** DOCUMENTATION NEEDED (SEE CONTROL SIGNAL??)
                      *** EXPLAIN owner_mechanism_output_value:
reference_value is component of Mechanism.variable that corresponds to the current State

        # Potential problem:
        #    - a OutputState may correspond to a particular item of owner.value
        #        in which case there will be a mismatch here
        #    - if OutputState is being instantiated from Mechanism (in instantiate_output_states)
        #        then the item of owner.value is known and has already been checked
        #        (in the call to instantiate_mechanism_state)
        #    - otherwise, should ignore

        :param owner_mechanism: (Mechanism)
        :param reference_value: (value)
        :param value: (value)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        :param context: (str)
        :return:
        """

        # Assign functionType to self.name as default;
        #  will be overridden with instance-indexed name in call to super
        if name is NotImplemented:
            self.name = self.functionType
        else:
            self.name = name

        self.functionName = self.functionType

        self.reference_value = reference_value

# FIX: 5/26/16
        # IMPLEMENTATION NOTE:
        # Consider adding self to owner.outputStates here (and removing from ControlSignal.instantiate_sender)
        #  (test for it, and create if necessary, as per outputStates in ControlSignal.instantiate_sender),

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        super(OutputState, self).__init__(owner_mechanism,
                                                  value=value,
                                                  params=params,
                                                  name=name,
                                                  prefs=prefs,
                                                  context=self)


    def validate_variable(self, variable, context=NotImplemented):
        """Insure variable is compatible with output component of owner.executeMethod relevant to this state

        Validate self.variable against component of owner's value (output of Mechanism's execute method)
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

        # Insure that self.variable is compatible with (relevant item of) output value of owner's execute method
        if not iscompatible(self.variable, self.reference_value):
            raise OutputStateError("Value ({0}) of outputState for {1} is not compatible with "
                                           "the output ({2}) of its execute method".
                                           format(self.value,
                                                  self.owner.name,
                                                  self.reference_value))

    def update(self, params=NotImplemented, time_scale=TimeScale.TRIAL, context=NotImplemented):
        """Process outputState params and pass params for inputState projections to super for processing

        :param params:
        :param time_scale:
        :param context:
        :return:
        """

        try:
            # Get outputState params
            output_state_params = params[kwOutputStateParams]

        except (KeyError, TypeError):
            output_state_params = NotImplemented

        # Process any outputState params here
        pass

        super(OutputState, self).update(params=output_state_params,
                                                      time_scale=time_scale,
                                                      context=context)
