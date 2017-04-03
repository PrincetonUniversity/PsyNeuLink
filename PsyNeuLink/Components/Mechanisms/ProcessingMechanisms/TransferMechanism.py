# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************  TransferMechanism ******************************************************

"""
..
    Sections:
      * :ref:`Transfer_Overview`
      * :ref:`Transfer_Creation`
      * :ref:`Transfer_Execution`
      * :ref:`Transfer_Class_Reference`

.. _Transfer_Overview:

Overview
--------

A TransferMechanism transforms its input using a simple mathematical function.  The input can be a single scalar
value or an an array of scalars (list or 1d np.array).  The function used can be selected from a standard set
of PsyNeuLink Functions (`Linear`, `Exponential` or `Logistic`), or specified using a user-defined custom function.


.. _Transfer_Creation:

Creating a TransferMechanism
-----------------------------

A TransferMechanism can be created directly by calling its constructor, or using the :py:func:`mechanism`
function and specifying `TRANSFER_MECHANISM` as its `mech_spec` argument.  Its function is specified in the
:keyword:`function` argument, which can be simply the name of the class (first example below), or a call to its
constructor which can include arguments specifying the function's parameters (second example)::

    my_linear_transfer_mechanism = TransferMechanism(function=Linear)
    my_logistic_transfer_mechanism = TransferMechanism(function=Logistic(gain=1.0, bias=-4)

In addition to function-specific parameters, `noise <TransferMechanism.noise>` and `time_constant <TransferMechanism.time_constant>`
parameters can be specified (see `Execution` below).


.. _Transfer_Structure:

Structure
---------

A TransferMechanism has a single `inputState <InputState>`, the `value <InputState.InputState.value>` of which is
used as the `variable <TransferMechanism.variable>` for its `function <TransferMechanism.function>`. The
:keyword:`function` can be selected from one of three standard PsyNeuLink `Function <Functions>`: `Linear`,
`Logistic` or `Exponential`; or a custom function can be specified, so long as it returns a numeric value or
list or np.ndarray of numeric values.  A TransferMechanism has three `outputStates <OutputStates>, described under
`Execution` below.


.. _Transfer_Execution:

Execution
---------

When a TransferMechanism is executed, it transforms its input using the specified function and the following
parameters (in addition to those specified for the function):

    * `noise <TransferMechanism.noise>`: applied element-wise to the input before transforming it.
    ..
    * `time_constant <TransferMechanism.time_constant>`: if `time_scale` is :keyword:`TimeScale.TIME_STEP`,
      the input is exponentially time-averaged before transforming it (higher value specifies faster rate);
      if `time_scale` is :keyword:`TimeScale.TRIAL`, `time_constant <TransferMechanism.time_constant>` is ignored.
    ..
    * `range <TransferMechanism.range>`: caps all elements of the `function <TransferMechanism.function>` result by
      the lower and upper values specified by range.

After each execution of the mechanism:

.. _Transfer_Results:

    * **result** of `function <TransferMechanism.function>` is assigned to the mechanism's
      `value <TransferMechanism.value>` attribute, the :keyword:`value` of its `TRANSFER_RESULT` outputState,
      and to the 1st item of the mechanism's `outputValue <TransferMechanism.outputValue>` attribute;
    ..
    * **mean** of the result is assigned to the the :keyword:`value` of the mechanism's `TRANSFER_MEAN` outputState,
      and to the 2nd item of its `outputValue <TransferMechanism.outputValue>` attribute;
    ..
    * **variance** of the result is assigned to the :keyword:`value` of the mechanism's `TRANSFER_VARIANCE` outputState,
      and to the 3rd item of its `outputValue <TransferMechanism.outputValue>` attribute.

COMMENT

.. _Transfer_Class_Reference:

Class Reference
---------------

"""

# from numpy import sqrt, random, abs, tanh, exp
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import *
from PsyNeuLink.Components.Functions.Function import Linear, TransferFunction, Integrator, NormalDist

# TransferMechanism parameter keywords:
RANGE = "range"
TIME_CONSTANT = "time_constant"
INITIAL_VALUE = 'initial_value'

# TransferMechanism outputs (used to create and name outputStates):
TRANSFER_RESULT = "transfer_result"
TRANSFER_MEAN = "transfer_mean "
TRANSFER_VARIANCE = "transfer_variance"
TRANSFER_DIFFERENTIAL = "transfer_differential"

# TransferMechanism output indices (used to index output values):
class Transfer_Output(AutoNumber):
    """Indices of the `outputValue <TransferMechanism.outputValue>` attribute of the TransferMechanism containing the
    values described below."""
    RESULT = ()
    """Result of the TransferMechanism's `function <TransferMechanism.function>`."""
    MEAN = ()
    """Mean of the elements in the :keyword`value` of the RESULT outputState."""
    VARIANCE = ()
    """Variance of the elements in the :keyword`value` of the RESULT outputState."""

# TransferMechanism default parameter values:
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
class TransferMechanism(ProcessingMechanism_Base):
    """
    TransferMechanism(                    \
    default_input_value=None,    \
    function=Linear,             \
    initial_value=None,          \
    noise=0.0,                   \
    time_constant=1.0,                    \
    range=(float:min, float:max),\
    time_scale=TimeScale.TRIAL,  \
    params=None,                 \
    name=None,                   \
    prefs=None)

    Implements TransferMechanism subclass of `Mechanism`.

    COMMENT:
        Description
        -----------
            TransferMechanism is a Subtype of the ProcessingMechanism Type of the Mechanism Category of the
                Component class
            It implements a Mechanism that transforms its input variable based on FUNCTION (default: Linear)

        Class attributes
        ----------------
            + componentType (str): TransferMechanism
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
            All instances of TransferMechanism are registered in MechanismRegistry, which maintains an
              entry for the subclass, a count for all instances of it, and a dictionary of those instances
    COMMENT

    Arguments
    ---------

    default_input_value : number, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the input to the mechanism to use if none is provided in a call to its
        `execute <Mechanism.Mechanism_Base.execute>` or `run <Mechanism.Mechanism_Base.run>` method;
        also serves as a template to specify the length of `variable <TransferMechanism.variable>` for
        `function <TransferMechanism.function>`, and the `primary outputState <OutputState_Primary>`
        of the mechanism.

    function : TransferFunction : default Linear
        specifies the function used to transform the input;  can be `Linear`, `Logistic`, `Exponential`,
        or a custom function.

    initial_value :  value, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the starting value for time-averaged input (only relevant if
        `time_constant <TransferMechanism.time_constant>` is not 1.0).
        :py:data:`Transfer_DEFAULT_BIAS <LINK->SHOULD RESOLVE TO VALUE>`

    noise : float or function : default 0.0
        a stochastically-sampled value added to the result of the `function <TransferMechanism.function>`:
        if it is a float, it must be in the interval [0,1] and is used to scale the variance of a zero-mean Gaussian;
        if it is a function, it must return a scalar value.

    time_constant : float : default 1.0
        the time constant for exponential time averaging of input when the mechanism is executed with `time_scale`
        set to `TimeScale.TIME_STEP`::

         result = (time_constant * current input) + (1-time_constant * result on previous time_step)

    range : Optional[Tuple[float, float]]
        specifies the allowable range for the result of `function <TransferMechanism.function>`:
        the first item specifies the minimum allowable value of the result, and the second its maximum allowable value;
        any element of the result that exceeds the specified minimum or maximum value is set to the value of
        `range <TransferMechanism.range>` that it exceeds.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that can be used to specify the parameters for
        the mechanism, its function, and/or a custom function and its parameters.  Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the constructor.

    time_scale :  TimeScale : TimeScale.TRIAL
        specifies whether the mechanism is executed using the `TIME_STEP` or `TRIAL` `TimeScale`.
        This must be set to `TimeScale.TIME_STEP` for the `time_constant <TransferMechanism.time_constant>`
        parameter to have an effect.

    name : str : default TransferMechanism-<index>
        a string used for the name of the mechanism.
        If not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the `PreferenceSet` for mechanism.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    .. context=componentType+INITIALIZING):
            context : str : default ''None''
                   string used for contextualization of instantiation, hierarchical calls, executions, etc.

    Returns
    -------
    instance of TransferMechanism : TransferMechanism


    Attributes
    ----------

    variable : value: default Transfer_DEFAULT_BIAS
        the input to mechanism's ``function``.  :py:data:`Transfer_DEFAULT_BIAS <LINK->SHOULD RESOLVE TO VALUE>`

    function : Function :  default Linear
        the function used to transform the input.

    COMMENT:
       THE FOLLOWING IS THE CURRENT ASSIGNMENT
    COMMENT
    initial_value :  value, list or np.ndarray : Transfer_DEFAULT_BIAS
        determines the starting value for time-averaged input
        (only relevant if `time_constant <TransferMechanism.time_constant>` parameter is not 1.0).
        :py:data:`Transfer_DEFAULT_BIAS <LINK->SHOULD RESOLVE TO VALUE>`

    noise : float or function : default 0.0
        a stochastically-sampled value added to the output of the `function <TransferMechahnism.function>`:
        if it is a float, it must be in the interval [0,1] and is used to scale the variance of a zero-mean Gaussian;
        if it is a function, it must return a scalar value.

    time_constant : float : default 1.0
        the time constant for exponential time averaging of input
        when the mechanism is executed using the `TIME_STEP` `TimeScale`::

          result = (time_constant * current input) + (1-time_constant * result on previous time_step)

    range : Optional[Tuple[float, float]]
        determines the allowable range of the result: the first value specifies the minimum allowable value
        and the second the maximum allowable value;  any element of the result that exceeds minimum or maximum
        is set to the value of `range <TransferMechanism.range>` it exceeds.  If `function <TransferMechanism.function>`
        is `Logistic`, `range <TransferMechanism.range>` is set by default to (0,1).

    value : 2d np.array [array(float64)]
        result of executing `function <TransferMechanism.function>`; same value as fist item of
        `outputValue <TransferMechanism.outputValue>`.

    COMMENT:
        CORRECTED:
        value : 1d np.array
            the output of ``function``;  also assigned to ``value`` of the :keyword:`TRANSFER_RESULT` outputState
            and the first item of ``outputValue``.
    COMMENT

    outputStates : Dict[str, OutputState]
        an OrderedDict with three `outputStates <OutputState>`:
        * `TRANSFER_RESULT`, the :keyword:`value` of which is the **result** of `function <TransferMechanism.function>`;
        * `TRANSFER_MEAN`, the :keyword:`value` of which is the mean of the result;
        * `TRANSFER_VARIANCE`, the :keyword:`value` of which is the variance of the result;

    outputValue : List[array(float64), float, float]
        a list with the following items:
        * **result** of the ``function`` calculation (value of `TRANSFER_RESULT` outputState);
        * **mean** of the result (``value`` of `TRANSFER_MEAN` outputState)
        * **variance** of the result (``value`` of `TRANSFER_VARIANCE` outputState)

    time_scale :  TimeScale : defaul tTimeScale.TRIAL
        specifies whether the mechanism is executed using the `TIME_STEP` or `TRIAL` `TimeScale`.

    name : str : default TransferMechanism-<index>
        the name of the mechanism.
        Specified in the `name` argument of the constructor for the projection;
        if not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Mechanism.classPreferences
        the `PreferenceSet` for mechanism.
        Specified in the `prefs` argument of the constructor for the mechanism;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentType = TRANSFER_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'TransferCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
        kpRuntimeParamStickyAssignmentPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
    }

    variableClassDefault = Transfer_DEFAULT_BIAS # Sets template for variable (input)
                                                 #  to be compatible with Transfer_DEFAULT_BIAS


    # TransferMechanism parameter and control signal assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        # TIME_SCALE: TimeScale.TRIAL,
        INPUT_STATES: None,
        OUTPUT_STATES:[
            {NAME:TRANSFER_RESULT},

            {NAME:TRANSFER_MEAN,
             CALCULATE:lambda x: np.mean(x)},

            {NAME:TRANSFER_VARIANCE,
             CALCULATE:lambda x: np.var(x)}
        ]})

    paramNames = paramClassDefaults.keys()

    @tc.typecheck
    def __init__(self,
                 default_input_value=None,
                 function=Linear,
                 initial_value=None,
                 noise=0.0,
                 time_constant=1.0,
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
                                                  time_constant=time_constant,
                                                  time_scale=time_scale,
                                                  range=range,
                                                  params=params)
        if default_input_value is None:
            default_input_value = Transfer_DEFAULT_BIAS

        super(TransferMechanism, self).__init__(variable=default_input_value,
                                       params=params,
                                       name=name,
                                       prefs=prefs,
                                       context=self)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate FUNCTION and mechanism params

        """

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        transfer_function = target_set[FUNCTION]
        if isinstance(transfer_function, Component):
            transfer_function_class = transfer_function.__class__
            transfer_function_name = transfer_function.__class__.__name__
        elif isclass(transfer_function):
            transfer_function_class = transfer_function
            transfer_function_name = transfer_function.__name__

        # Validate FUNCTION
        if not transfer_function_class.componentType is TRANFER_FUNCTION_TYPE:
            raise TransferError("Function {} specified as FUNCTION param of {} must be a {}".
                                format(transfer_function_name, self.name, TRANFER_FUNCTION_TYPE))

        # Validate INITIAL_VALUE
        initial_value = target_set[INITIAL_VALUE]
        if initial_value:
            if not iscompatible(initial_value, self.variable[0]):
                raise TransferError("The format of the initial_value parameter for {} ({}) must match its input ({})".
                                    format(append_type_to_name(self), initial_value, self.variable[0]))

        # # Validate NOISE:
        # noise = target_set[NOISE]
        # if (isinstance(noise, float) == False) and (callable(noise) == False):
        #     raise TransferError("noise parameter ({}) for {} must be a float or a function".
        #                         format(noise, self.name))

        # Validate TIME_CONSTANT:
        time_constant = target_set[TIME_CONSTANT]
        if not (isinstance(time_constant, float) and time_constant>=0 and time_constant<=1):
            raise TransferError("time_constant parameter ({}) for {} must be a float between 0 and 1".
                                format(time_constant, self.name))

        # Validate RANGE:
        range = target_set[RANGE]
        if range:
            if not (isinstance(range, tuple) and len(range)==2 and all(isinstance(i, numbers.Number) for i in range)):
                raise TransferError("range parameter ({}) for {} must be a tuple with two numbers".
                                    format(range, self.name))
            if not range[0] < range[1]:
                raise TransferError("The first item of the range parameter ({}) must be less than the second".
                                    format(range, self.name))
        self.integrator_function = Integrator(variable_default = self.variable, integration_type=ADAPTIVE, rate=self.time_constant, noise=self.noise)

    def _instantiate_parameter_states(self, context=None):

        from PsyNeuLink.Components.Functions.Function import Logistic
        # If function is a logistic, and range has not been specified, bound it between 0 and 1
        if ((isinstance(self.function, Logistic) or
                 (inspect.isclass(self.function) and issubclass(self.function,Logistic))) and
                not list(self.range)):
            self.user_params[RANGE] = np.array([0,1])

        super()._instantiate_parameter_states(context=context)

    def _instantiate_attributes_before_function(self, context=None):

        self.initial_value = self.initial_value or self.variableInstanceDefault

        super()._instantiate_attributes_before_function(context=context)

    def _execute(self,
                variable=None,
                runtime_params=None,
                clock=CentralClock,
                time_scale = TimeScale.TRIAL,
                context=None):
        """Execute TransferMechanism function and return transform of input

        Execute TransferMechanism function on input, and assign to outputValue:
            - Activation value for all units
            - Mean of the activation values across units
            - Variance of the activation values across units
        Return:
            value of input transformed by TransferMechanism function in outputState[TransferOuput.RESULT].value
            mean of items in RESULT outputState[TransferOuput.MEAN].value
            variance of items in RESULT outputState[TransferOuput.VARIANCE].value

        Arguments:

        # CONFIRM:
        variable (float): set to self.value (= self.inputValue)
        - params (dict):  runtime_params passed from Mechanism, used as one-time value for current execution:
            + NOISE (float)
            + TIME_CONSTANT (float)
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


        time_constant = self.time_constant
        range = self.range
        noise = self.noise

        #endregion


        #region EXECUTE TransferMechanism FUNCTION ---------------------------------------------------------------------

        # FIX: NOT UPDATING self.previous_input CORRECTLY

        # Update according to time-scale of integration
        if time_scale is TimeScale.TIME_STEP:
            current_input = self.integrator_function.function(self.inputState.value,
                                                              params = {NOISE: noise, RATE: time_constant},
                                                              context=context)

        elif time_scale is TimeScale.TRIAL:
            current_input = self.inputState.value + noise
        else:
            raise MechanismError("time_scale not specified for TransferMechanism")

        self.previous_input = current_input

        # Apply TransferMechanism function
        output_vector = self.function(variable=current_input, params=runtime_params)

        if list(range):
            minCapIndices = np.where(output_vector < range[0])
            maxCapIndices = np.where(output_vector > range[1])
            output_vector[minCapIndices] = np.min(range)
            output_vector[maxCapIndices] = np.max(range)

        return output_vector

        #endregion


    def _report_mechanism_execution(self, input, params, output):
        """Override super to report previous_input rather than input, and selected params
        """
        print_input = self.previous_input
        print_params = params.copy()
        # Only report time_constant if in TIME_STEP mode
        if params['time_scale'] is TimeScale.TRIAL:
            del print_params[TIME_CONSTANT]
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
    #     # IMPLEMENTATION NOTE:  TBI when time_step is implemented for TransferMechanism

    @property
    def range(self):
        return self._range


    @ range.setter
    def range(self, value):
        self._range = value
