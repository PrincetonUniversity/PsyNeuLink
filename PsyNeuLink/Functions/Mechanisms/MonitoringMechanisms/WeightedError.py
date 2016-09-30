# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *********************************************  WeightedError *******************************************************
#

# from numpy import sqrt, random, abs, tanh, exp
from PsyNeuLink.Functions.Mechanisms.MonitoringMechanisms.MonitoringMechanism import *
# from PsyNeuLink.Functions.States.InputState import InputState

# WeightedError output (used to create and name outputStates):
kwWeightedErrors = 'WeightedErrors'
NEXT_LEVEL_PROJECTION = 'next_level_projection'

# WeightedError output indices (used to index output values):
class WeightedErrorOutput(AutoNumber):
    ERROR_SIGNAL = ()

class WeightedErrorError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class WeightedError(MonitoringMechanism_Base):
    """Implement WeightedError subclass

    Description:
        WeightedError is a Subtype of the MonitoringMechanism Type of the Mechanism Category of the Function class
        It's function computes the contribution of each sender element (rows of the NEXT_LEVEL_PROJECTION param)
            to the error values of the receivers
             (elements of the error_signal array, columns of the matrix of the NEXT_LEVEL_PROJECTION param),
            weighted by the association of each sender with each receiver (specified in NEXT_LEVEL_PROJECTION.matrix)
        The function returns an array with the weighted errors for each sender element

    Instantiation:
        - A WeightedError can be instantiated in several ways:
            - directly, by calling WeightedError()
            - by assigning a LearningSignal Projection to a ProcessingMechanism that has at least one other
                ProcessingMechanism to which it projects

    Initialization arguments:
        In addition to standard arguments params (see Mechanism), WeightedError also implements the following params:
        - error_signal (1D np.array)
        - params (dict):
            + NEXT_LEVEL_PROJECTION (Mapping Projection):
                projection, the matrix of which is used to calculate error_array
                width (number of columns) must match error_signal
        Notes:
        *  params can be set in the standard way for any Function subclass:
            - params provided in param_defaults at initialization will be assigned as paramInstanceDefaults
                 and used for paramsCurrent unless and until the latter are changed in a function call
            - paramInstanceDefaults can be later modified using assign_defaults
            - params provided in a function call (to execute or adjust) will be assigned to paramsCurrent

    MechanismRegistry:
        All instances of WeightedError are registered in MechanismRegistry, which maintains an entry for the subclass,
          a count for all instances of it, and a dictionary of those instances

    Naming:
        Instances of WeightedError can be named explicitly (using the name='<name>' argument).
        If this argument is omitted, it will be assigned "WeightedError" with a hyphenated, indexed suffix ('WeightedError-n')

    Execution:
        - Computes comparison of two inputStates of equal length and generates array of same length,
            as well as summary statistics (sum, sum of squares, and variance of comparison array values) 
        - self.execute returns self.value
        Notes:
        * WeightedError handles "runtime" parameters (specified in call to execute method) differently than std Functions:
            any specified params are kept separate from paramsCurrent (Which are not overridden)
            if the FUNCTION_RUN_TIME_PARMS option is set, they are added to the current value of the
                corresponding ParameterState;  that is, they are combined additively with controlSignal output

    Class attributes:
        + functionType (str): WeightedError
        + classPreference (PreferenceSet): WeightedError_PreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.SUBTYPE
        + variableClassDefault (1D np.array):
        + paramClassDefaults (dict): {NEXT_LEVEL_PROJECTION: Mapping}
        + paramNames (dict): names as above

    Class methods:
        None

    Instance attributes: none
        + variable (1D np.array): error_signal used by execute method
        + value (value): output of execute method
        + name (str): if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet): if not specified as arg, default set is created by copying WeightedError_PreferenceSet

    Instance methods:
        - validate_params(self, request_set, target_set, context):
            validates that width of matrix for projection in NEXT_LEVEL_PROJECTION param equals length of error_signal
        - execute(error_signal, params, time_scale, context)
            calculates and returns weighted error array (in self.value and values of self.outputStates)
    """

    functionType = "WeightedError"

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'WeightedErrorCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    variableClassDefault = [0]  # error_signal

    # WeightedError parameter assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        NEXT_LEVEL_PROJECTION: None,
        kwOutputStates:[kwWeightedErrors],
    })

    paramNames = paramClassDefaults.keys()

    @tc.typecheck
    def __init__(self,
                 error_signal=NotImplemented,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):
        """Assign type-level preferences and call super.__init__
        """

        self.function = self.execute

# # FIX: MODIFY get_param_value_for_keyword TO TAKE PARAMS DICT

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self.assign_args_to_param_dicts(params=params)

        super().__init__(variable=error_signal,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

    def validate_params(self, request_set, target_set=NotImplemented, context=None):
        """Insure that width (number of columns) of NEXT_LEVEL_PROJECTION equals length of error_signal
        """

        super().validate_params(request_set=request_set, target_set=target_set, context=context)
        cols = target_set[NEXT_LEVEL_PROJECTION].matrix.shape[1]
        error_signal_len = len(self.variable[0])
        if  cols != error_signal_len:
            raise WeightedErrorError("Number of columns ({}) of weight matrix for {}"
                                     " must equal length of error_signal ({})".
                                     format(cols,self.name,error_signal_len))

    def instantiate_attributes_before_function(self, context=None):

        # Map indices of output to outputState(s)
        self.outputStateValueMapping = {}
        self.outputStateValueMapping[kwWeightedErrors] = WeightedErrorOutput.ERROR_SIGNAL.value

        super().instantiate_attributes_before_function(context=context)


    def __execute__(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale = TimeScale.TRIAL,
                context=None):

        """Compute error_signal for current layer from derivative of error_signal at next layer
        """

        if not context:
            context = kwExecuting + self.name

        self.check_args(variable=variable, params=params, context=context)

        # Get error signal from monitoring mechanism in next layer
        error = self.variable[0]

        # Get weight matrix for projection at next layer
        next_level_matrix = self.paramsCurrent[NEXT_LEVEL_PROJECTION].matrix

        # Get output of next layer
        next_level_output = self.paramsCurrent[NEXT_LEVEL_PROJECTION].receiver.owner.outputState.value

        # Get derivative for projection's receiver's function
        derivative_fct = self.paramsCurrent[NEXT_LEVEL_PROJECTION].receiver.owner.function_object.derivative

        # Compute derivative of error with respect to current output
        output_derivative = derivative_fct(output=next_level_output)
        error_derivative = error * output_derivative

        # Compute error terms for each unit of current layer weighted by contribution to error at next level
        error_array = np.dot(next_level_matrix, error_derivative)

        # Compute summed error for use by callers to decide whether to update
        self.summedErrorSignal = np.sum(error_array)

        # Assign output values
        self.outputValue[WeightedErrorOutput.ERROR_SIGNAL.value] = error_array

        return self.outputValue