# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *********************************************  Transfer *******************************************************

"""
..
    Sections:
      * :ref:`Transfer_Overview`
      * :ref:`Transfer_Creating_A_Transfer_Mechanism`
      * :ref:`Transfer_Execution`
      * :ref:`Transfer_Class_Reference`

.. _Transfer_Overview:

Overview
--------

A Transfer mechanism transforms its input using a simple mathematical function.  The input can be a single scalar
value or an an array (list or 1d np.array).  The function used can be selected from a standard set, or specified
using a custom function.


.. _Transfer_Creating_A_Transfer_Mechanism:

Creating a Transfer Mechanism
-----------------------------

A Transfer mechanism can be created either directly, by calling its constructor, or using the :class:`mechanism`
function and specifying "Transfer" as its ``mech_spec`` argument.  Its function is specified in the ``function``
argument, which can be simply the name of the class (first example below), or a call to its constructor which can
include arguments specifying the function's parameters (second example)::

    my_linear_transfer_mechanism = Transfer(function=Linear)
    my_logistic_transfer_mechanism = Transfer(function=Logistic(gain=1.0, bias=-4)

In addition to function-specific parameters, ``noise`` and ``rate`` parameters can be specified (see Execution below).


.. _Transfer_Structure:

Structure
---------

A Transfer mechanism has a single inputState, the ``value`` of which is used as the ``variable`` for its ``function``.
The function can be selected from one of three standard PsyNeuLink :doc:`Functions`: :any:`Linear`, :any:`Logistic` or
:any:`Exponential`; or a custom function can be specified, so long as it returns a numeric value or list or
np.ndarray of numeric values.  A Transfer mechanism has three outputStates, described under Execution  below.


.. _Transfer_Execution:

Execution
---------

When a Transfer mechanism is executed, it transforms its input using the specified function and the following
parameters (in addition to those specified for the function):

If the ``noise`` parameter is specified, it is applied element-wise to the input before transforming it.
If the ``rate`` parameter is specified and ``time_scale`` is :keyword:`TimeScale.TIME_STEP`, the input is
exponentially time-averaged before transforming it (higher value specifies faster rate); if ``time_scale`` is
:keyword:`TimeScale.TIME_STEP` the ``rate`` parameter is ignored.
If the ``range`` parameter is specified, all elements of the output are capped by the lower and upper values of
the range.  After each execution of the mechanism:

.. _Transfer_Results:

    * **result** is assigned to the mechanism's ``value`` attribute, the value of its :keyword:`TRANSFER_RESULT`
      outputState, and to the 1st item of the mechanism's ``outputValue`` attribute;
    ..
    * **mean** of the result is assigned to the value of the mechanism's :keyword:`TRANSFER_MEAN` outputState,
      and to the 2nd item of the mechanism's ``outputValue`` attribute;
    ..
    * **variance** of the result is assigned to the value of the mechanism's :keyword:`TRANSFER_VARIANCE` outputState,
      and to the 3rd item of the mechanism's ``outputValue`` attribute.


COMMENT:
    ?? IS THIS TRUE, OR JUST A CARRYOVER FROM DDM??
    Notes:
    * Transfer handles "runtime" parameters (specified in call to function) differently than standard functions:
        any specified params are kept separate from paramsCurrent (Which are not overridden)
        if the FUNCTION_RUN_TIME_PARMS option is set, they are added to the current value of the
            corresponding ParameterState;  that is, they are combined additively with controlSignal output
COMMENT

.. _Transfer_Class_Reference:

Class Reference
---------------

"""

# from numpy import sqrt, random, abs, tanh, exp
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import *
from PsyNeuLink.Components.Functions.Function import Linear

# Transfer parameter keywords:
RANGE = "range"
INITIAL_VALUE = 'initial_value'

# Transfer outputs (used to create and name outputStates):
TRANSFER_RESULT = "transfer_result"
TRANSFER_MEAN = "transfer_mean "
TRANSFER_VARIANCE = "transfer_variance"

# Transfer output indices (used to index output values):
class Transfer_Output(AutoNumber):
    RESULT = ()
    MEAN = ()
    VARIANCE = ()

# Transfer default parameter values:
Transfer_DEFAULT_LENGTH= 1
Transfer_DEFAULT_GAIN = 1
Transfer_DEFAULT_BIAS = 0
Transfer_DEFAULT_OFFSET = 0
Transfer_DEFAULT_RANGE = np.array([])


class TransferError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

# IMPLEMENTATION NOTE:  IMPLEMENTS OFFSET PARAM BUT IT IS NOT CURRENTLY BEING USED
class Transfer(ProcessingMechanism_Base):
    """
    Transfer(                    \
    default_input_value=None,    \
    function=Linear,             \
    initial_value=None,          \
    noise=0.0,                   \
    rate=1.0,                    \
    range=(float:min, float:max),\
    time_scale=TimeScale.TRIAL,  \
    params=None,                 \
    name=None,                   \
    prefs=None)

    Implements Transfer subclass of Mechanism

    COMMENT:
        Description
        -----------
            Transfer is a Subtype of the ProcessingMechanism Type of the Mechanism Category of the Function class
            It implements a Mechanism that transforms its input variable based on FUNCTION (default: Linear)

        Class attributes
        ----------------
            + componentType (str): Transfer
            + classPreference (PreferenceSet): Transfer_PreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.SUBTYPE
            + variableClassDefault (value):  Transfer_DEFAULT_BIAS
            + paramClassDefaults (dict): {TIME_SCALE: TimeScale.TRIAL}
            + paramNames (dict): names as above

        Class methods
        -------------
            None

        MechanismRegistry
        -----------------
            All instances of Transfer are registered in MechanismRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances
    COMMENT

    Arguments
    ---------

    default_input_value : value, list or np.ndarray : Transfer_DEFAULT_BIAS [LINK] -> SHOULD RESOLVE TO VALUE
        the input to the mechanism to use if none is provided in a call to its ``execute`` or ``run`` methods;
        also serves as a template to specify the length of ``variable`` for ``function``, and the primary  outputState
        of the mechanism.

    function : TransferFunction : default Linear
        specifies function used to transform input;  can be :class:`Linear`, :class:`Logistic`, :class:`Exponential`,
        or a custom function.

    function_params : Dict[str, value]
        contains one entry for each parameter of the mechanism's function.
        The key of each entry is the name of (keyword for) a function parameter, and its value is the parameter's value.

    initial_value :  value, list or np.ndarray : Transfer_DEFAULT_BIAS [LINK] -> SHOULD RESOLVE TO VALUE
        specifies the starting value for time-averaged input (only relevant if ``rate`` parameter is not 1.0).

    noise : float or function : default 0.0
        a stochastically-sampled value added to the output of the ``function``.
        If it is a float, it must be in the interval [0,1] and is used to scale the variance of a zero-mean Gaussian;
        If it is a function, it must return a scalar value.

    rate : float : default 1.0
        the time constant for exponential time averaging of input
        when the mechanism is executed at the time_step time scale:
        input on current time_step = (rate * specified input) + (1-rate * input on previous time_step).

    range : Optional[Tuple[float, float]]
        specifies the allowable range of the result: the first value specifies the minimum allowable value
        and the second the maximum allowable value;  any element of the result that execeeds minimum or maximum
        is set to the corresponding value.

    params : Optional[Dict[param keyword, param value]]
        a dictionary that can be used to specify the parameters for the mechanism, parameters for its function,
        and/or a custom function and its parameters (see :doc:`Mechanism` for specification of a parms dict).

    time_scale :  TimeScale : TimeScale.TRIAL
        specifies whether the mechanism is executed on the :keyword:`TIME_STEP` or :keyword:`TRIAL` time scale.
        This must be set to :keyword:`TimeScale.TIME_STEP` for the ``rate`` parameter to have an effect.

    name : str : default Transfer-<index>
        a string used for the name of the mechanism.
        If not is specified, a default is assigned by MechanismRegistry
        (see :doc:`Registry` for conventions used in naming, including for default and duplicate names).[LINK]

    prefs : Optional[PreferenceSet or specification dict : Process.classPreferences]
        the PreferenceSet for mechanism.
        If it is not specified, a default is assigned using ``classPreferences`` defined in __init__.py
        (see Description under PreferenceSet for details) [LINK].

    .. context=componentType+INITIALIZING):
            context : str : default ''None''
                   string used for contextualization of instantiation, hierarchical calls, executions, etc.

    Returns
    -------
    instance of Transfer mechanism : Transfer


    Attributes
    ----------

    variable : value: default Transfer_DEFAULT_BIAS [LINK] -> SHOULD RESOLVE TO VALUE
        the input to mechanism's function.

    function : Function :  default Linear
        the function used to transform the input.

    COMMENT:
       THE FOLLOWING IS THE CURRENT ASSIGNMENT
    COMMENT

    value : List[1d np.array, float, float]
        same as ``outputValue``.

    COMMENT:
        CORRECTED:
        value : 1d np.array
            the output of ``function``;  also assigned to ``value`` of the :keyword:`TRANSFER_RESULT` outputState
            and the first item of ``outputValue``.
    COMMENT

    outputValue : List[1d np.array, float, float]
        a list with the following items:
        * **result** of the ``function`` calculation (value of :keyword:`TRANSFER_RESULT` outputState);
        * **mean** of the result (``value`` of :keyword:`TRANSFER_MEAN` outputState)
        * **variance** of the result (``value`` of :keyword:`TRANSFER_VARIANCE` outputState)

    name : str : default Transfer-<index>
        the name of the mechanism.
        Specified in the name argument of the call to create the projection;
        if not is specified, a default is assigned by MechanismRegistry
        (see :doc:`Registry` for conventions used in naming, including for default and duplicate names).[LINK]

    prefs : PreferenceSet or specification dict : Mechanism.classPreferences
        the PreferenceSet for mechanism.
        Specified in the prefs argument of the call to create the mechanism;
        if it is not specified, a default is assigned using ``classPreferences`` defined in __init__.py
        (see Description under PreferenceSet for details) [LINK].

    """

    componentType = "Transfer"

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'TransferCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    variableClassDefault = Transfer_DEFAULT_BIAS # Sets template for variable (input)
                                                 #  to be compatible with Transfer_DEFAULT_BIAS

    # Transfer parameter and control signal assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        # TIME_SCALE: TimeScale.TRIAL,
        INPUT_STATES: None,
        OUTPUT_STATES:[TRANSFER_RESULT,
                       TRANSFER_MEAN,
                       TRANSFER_VARIANCE]
    })

    paramNames = paramClassDefaults.keys()

    @tc.typecheck
    def __init__(self,
                 default_input_value=None,
                 function=Linear(),
                 initial_value=None,
                 noise=0.0,
                 rate=1.0,
                 range=np.array([]),
                 time_scale=TimeScale.TRIAL,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=componentType+INITIALIZING):
        """Assign type-level preferences, default input value (Transfer_DEFAULT_BIAS) and call super.__init__

        :param default_input_value: (value)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        """

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function,
                                                 initial_value=initial_value,
                                                 noise=noise,
                                                 rate=rate,
                                                 time_scale=time_scale,
                                                 range=range,
                                                 params=params)

        if default_input_value is None:
            default_input_value = Transfer_DEFAULT_BIAS

        super(Transfer, self).__init__(variable=default_input_value,
                                       params=params,
                                       name=name,
                                       prefs=prefs,
                                       # context=context,
                                       context=self)

    def _validate_params(self, request_set, target_set=NotImplemented, context=None):
        """Get (and validate) self.function from FUNCTION if specified

        Intercept definition of FUNCTION and assign to self.combinationFunction;
            leave defintion of self.execute below intact;  it will call combinationFunction

        Args:
            request_set:
            target_set:
            context:
        """
        transfer_function = request_set[FUNCTION]
        if isinstance(transfer_function, Component):
            transfer_function_class = transfer_function.__class__
            transfer_function_name = transfer_function.__class__.__name__
        elif isclass(transfer_function):
            transfer_function_class = transfer_function
            transfer_function_name = transfer_function.__name__

        # Validate FUNCTION
        if not transfer_function_class.componentType is kwTransferFunction:
            raise TransferError("Function {} specified as FUNCTION param of {} must be a {}".
                                format(transfer_function_name, self.name, kwTransferFunction))

        # Validate INITIAL_VALUE
        initial_value = request_set[INITIAL_VALUE]
        if initial_value:
            if not iscompatible(initial_value, self.variable[0]):
                raise TransferError("The format of the initial_value parameter for {} ({}) must match its input ({})".
                                    format(append_type_to_name(self), initial_value, self.variable[0]))

        # Validate NOISE:
        noise = request_set[NOISE]
        if isinstance(noise, float) and noise>=0 and noise<=1:
            self.noise_function = False
        elif isinstance(noise, function_type):
            self.noise_function = True
        else:
            raise TransferError("noise parameter ({}) for {} must be a float between 0 and 1 or a function".
                                format(noise, self.name))

        # Validate RATE:
        rate = request_set[RATE]
        if not (isinstance(rate, float) and rate>=0 and rate<=1):
            raise TransferError("rate parameter ({}) for {} must be a float between 0 and 1".
                                format(rate, self.name))

        # Validate RANGE:
        range = request_set[RANGE]
        if range:
            if not (isinstance(range, tuple) and len(range)==2 and all(isinstance(i, numbers.Number) for i in range)):
                raise TransferError("range parameter ({}) for {} must be a tuple with two numbers".
                                    format(range, self.name))
            if not range[0] < range[1]:
                raise TransferError("The first item of the range parameter ({}) must be less than the second".
                                    format(range, self.name))


        super()._validate_params(request_set=request_set, target_set=target_set, context=context)


    def _instantiate_attributes_before_function(self, context=None):

        self.initial_value = self.initial_value or self.variableInstanceDefault

        # Map indices of output to outputState(s)
        self._outputStateValueMapping = {}
        self._outputStateValueMapping[TRANSFER_RESULT] = Transfer_Output.RESULT.value
        self._outputStateValueMapping[TRANSFER_MEAN] = Transfer_Output.MEAN.value
        self._outputStateValueMapping[TRANSFER_VARIANCE] = Transfer_Output.VARIANCE.value

        super()._instantiate_attributes_before_function(context=context)

    def __execute__(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale = TimeScale.TRIAL,
                context=None):
        """Execute Transfer function and return transform of input

        Execute Transfer function on input, and assign to outputValue:
            - Activation value for all units
            - Mean of the activation values across units
            - Variance of the activation values across units
        Return:
            value of input transformed by transfer function in outputState[TransferOuput.RESULT].value
            mean of items in RESULT outputState[TransferOuput.MEAN].value
            variance of items in RESULT outputState[TransferOuput.VARIANCE].value

        Arguments:

        # CONFIRM:
        variable (float): set to self.value (= self.inputValue)
        - params (dict):  runtime_params passed from Mechanism, used as one-time value for current execution:
            + NOISE (float)
            + RATE (float)
            + RANGE ([float, float])
        - time_scale (TimeScale): specifies "temporal granularity" with which mechanism is executed
        - context (str)

        Returns the following values in self.value (2D np.array) and in
            the value of the corresponding outputState in the self.outputStates dict:
            - activation value (float)
            - mean activation value (float)
            - standard deviation of activation values (float)

        :param self:
        :param variable (float)
        :param params: (dict)
        :param time_scale: (TimeScale)
        :param context: (str)
        :rtype self.outputState.value: (number)
        """

        # FIX: IS THIS CORRECT?  SHOULD THIS BE SET TO INITIAL_VALUE
        # FIX:     WHICH SHOULD BE DEFAULTED TO 0.0??
        # Use self.variable to initialize state of input
        if INITIALIZING in context:
            self.previous_input = self.variable

        # FIX: NEED TO GET THIS TO WORK WITH CALL TO METHOD:
        time_scale = self.time_scale

        #region ASSIGN PARAMETER VALUES
        # - convolve inputState.value (signal) w/ driftRate param value (attentional contribution to the process)

        # Scale noise to be between +noise and -noise
        if self.noise_function:
            noise = self.noise()
        else:
            noise = self.noise * ((2 * np.random.normal()) - 1)
        rate = self.rate
        range = self.range
        #endregion


        #region EXECUTE TRANSFER FUNCTION ------------------------------------------------------------------------------

        # FIX: NOT UPDATING self.previous_input CORRECTLY

        # Update according to time-scale of integration
        if time_scale is TimeScale.TIME_STEP:
            current_input = (rate * self.inputState.value) + ((1-rate) * self.previous_input) + noise
        elif time_scale is TimeScale.TRIAL:
            current_input = self.inputState.value + noise
        else:
            raise MechanismError("time_scale not specified for Transfer")

        self.previous_input = current_input

        # Apply transfer function
        output_vector = self.function(variable=current_input, params=params)

        if range:
            minCapIndices = np.where(output_vector < range[0])
            maxCapIndices = np.where(output_vector > range[1])
            output_vector[minCapIndices] = np.min(range)
            output_vector[maxCapIndices] = np.max(range)
        mean = np.mean(output_vector)
        variance = np.var(output_vector)

        self.outputValue[Transfer_Output.RESULT.value] = output_vector
        self.outputValue[Transfer_Output.MEAN.value] = mean
        self.outputValue[Transfer_Output.VARIANCE.value] = variance

        return self.outputValue
        #endregion


    def _report_mechanism_execution(self, input, params, output):
        """Override super to report previous_input rather than input, and selected params
        """
        print_input = self.previous_input
        print_params = params.copy()
        # Only report rate if in TIME_STEP mode
        if params['time_scale'] is TimeScale.TRIAL:
            del print_params[RATE]
        # Suppress reporting of range (not currently used)
        del print_params[RANGE]

        super()._report_mechanism_execution(input=print_input, params=print_params)


    # def terminate_function(self, context=None):
    #     """Terminate the process
    #
    #     called by process.terminate() - MUST BE OVERRIDDEN BY SUBCLASS IMPLEMENTATION
    #     returns output
    #
    #     :rtype CurrentStateTuple(state, confidence, duration, controlModulatedParamValues)
    #     """
    #     # IMPLEMENTATION NOTE:  TBI when time_step is implemented for Transfer


