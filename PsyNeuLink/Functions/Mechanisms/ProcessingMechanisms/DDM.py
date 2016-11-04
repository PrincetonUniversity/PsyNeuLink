# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ***************************************************  DDM *************************************************************
#
"""
Sections:
  * :ref:`DDM_Overview`
  * :ref:`DDM_Creating_A_DDM_Mechanism`
  * :ref:`DDM_Execution`
  * :ref:`DDM_Class_Reference`

.. _DDM_Overview:

Overview
--------

The DDM mechanism implements the "Drift Diffusion Model" (also know as the Diffusion Decision, Accumulation to Bound,
Linear Integrator, and Wiener Process First Passage Time Model [REFS]). This corresponds to a continuous version of
the sequential probability ratio test (SPRT [REF]), which is the statistically optimal procedure for two alternative
forced choice (TAFC) decision making ([REF]).  The DDM mechanism implements three versions of the process: two
analytic solutions (that operate at the :keyword:`TRIAL` time scale) and return an expected mean response time and
accuracy (:keyword:`BogaczEtAl` and :keyword:`NavarroAndFuss`); and one that numerically integrates the path
of the decision variable (at the in :keyword:`TIME_STEP` time scale).

.. _DDM_Creating_A_DDM_Mechanism:

Creating a DDM Mechanism
-----------------------------

A DDM Mechanism can be instantiated either directly, by calling the class, or using the :class:`mechanism`
function and specifying DDM as its ``mech_spec`` argument.  The analytic solution to use in :keyword:`TRIAL` mode
can be selected using the ``function`` argument, and the parameters of the process can be specified using the
argument corresponding to each (see Execution below).  The process used is specified with the ``function`` argument,
which can be simply the name of a DDM function (first example below), or a call to the function that can
include arguments specifying the function's parameters as well as a :doc:`ControlSignal` (second example)::

    my_DDM = DDM(function=BogaczEtAL)
    my_DDM = DDM(function=BogaczEtAl(drift_rate=0.2, threshold=(1, ControlSignal))

The input to a DDM mechanism can be a single scalar value or an an array (list or 1d np.array). If it is a single value,
a single DDM process is implemented, and the input is used as the ``drift_rate`` parameter (see below).  If the input is
an array, multiple parallel DDM processes are implemented all of which use the same parameters but each of which
receives its own input (from the corresponding element of the input array) and is executed independently of the others.


Parameters
----------

The parameters of the DDM process can be specified as arguments for either of the functions used for :keyword:`TRIAL`
mode (i.e., an analytic solution), as described below.  Those parameters will also be used for :keyword:`TIME_STEP`
mode  (step-wise integration), or they can be specified  in the ``params`` dict argument  for the mechanism when it
is created, using the keywords in the list below, as in the following example::

    my_DDM = DDM(function=BogaczEtAl(drift_rate=0.1,
                 params={DRIFT_RATE:(0.2, ControlSignal),
                         STARTING_POINT:-0.5),
                 time_scale=TimeScale.TIME_STEP}

Note that parameters specified in the params argument will be used for both :keyword:`TRIAL` and
:keyword:`TIME_STEP` mode (since parameters specified in the  ``params`` argument when creating a mechanism override
those specified directly as arguments to its ``function``).[LINK]  In the example above, this means that even
if the ``time_scale`` parameter is set to :keyword:`TimeScale.TRIAL``, the ``drift_rate`` of 0.2 (rather than 0.1)
will still be used.

The parameters for the DDM are:

* :keyword:`DRIFT_RATE` (default 0.0).
  - multiplies the input to the mechanism before it is assigned to the ``variable`` on every call of ``function``.
  The product is further multiplied by the value received from any ControlSignal projections to the
  ``DRIFT_RATE` parameterState.  The ``drift_rate`` parameter can be thought of as the "automatic" component
  (baseline strength) of the decision process, and the value received from a ControlSignal projection as the
  "attentional" component, both of which multiplicatively scale the input which constitutes the "stimulus" component.

* :keyword:`STARTING_POINT` (default 0.0).
  - specifies the initial value of the decision variable.  If ``time_scale`` is :keyword:`TimeScale.TIME_STEP`
  the ``starting_point`` is added to the decision variable on the first call to ``function`` but not subsequently.

* :keyword:`THRESHOLD` (default 1.0)
  - specifies the stopping value for the decision process.  When ``time_scale`` is :keyword:`TIME_STEP`, the
   integration process is terminated when the absolute value of the decision variable equals the absolute value
   of threshold.  The sign used to specify the threshold parameter determines the sign of the "correct response"
   (used for calculating error rate): if the value of the decision variable equals the value of the threshold,
   the result is coded as a correct response;  if the value of the decision  variable equals the negative of the
   threshold, the result is coded as an error.

* :keyword:`NOISE` (default 0.5)
   - specifies the variance of the stochastic ("diffusion") component of the decision process.  If ``time_scale``
   is :keyword:`TIME_STEP`, this value is multiplied by a random sample drawn from a zero-mean normal (Gaussian)
   distribution on every call of ``function``, and added to the decision variable.

* :keyword:`NON_DECISION_TIME` (default 200)
  specifies the ``t0`` parameter of the process;  when ``time_scale`` is :keyword:`TIME_STEP`, it is added to
  the number of time steps taken to complete the decision process (i.e., the response time).

.. _DDM_Execution:

Execution
---------

When a DDM mechanism is executed it computes the decision process, either analytically (in :keyword:`TRIAL` mode)
or by step-wise integration (in :keyword:`TIME_STEP` mode).  As noted above, if the input is a single scalar value,
it computes a single DDM process.  If the input is an array, then multiple parallel DDM processes are executed,
with each element of the input used for the corresponding process (all use the same set of parameters; to implement
processes that use their own parameters, a separate DDM mechanism should explicitly be created for each).

.. note::
   DDM handles "runtime" parameters (specified in call to execute method) differently than standard Functions:
   runtime parameters are added to the mechanism's current value of the corresponding parameterState (rather
   than overriding it);  that is, they are combined additively with controlSignal output to determine the parameter's
   value for that execution.  The parameterState's value is then restored for the next execution.

After each execution of the mechanism, the following output assignments are made:

    * value of the **decision variable** is assigned to the mechanism's ``value`` attribute, the value of its
      :keyword:`DECISION_VARIABLE` outputState, and to the 1st item of the mechanism's ``outputValue`` attribute;
    ..
    * the **response time** is assigned as the value of the mechanism's ``RT_MEAN`` outputState and to the
      2nd item of the mechanism's ``outputValue`` attribute.  If ``time_scale`` is :keyword:`TimeScale.TRIAL`,
      the value is the mean response time estimated by the analytic solution used in ``function``.
      [TBI:]
      If ``times_scale`` is :keyword:`TimeScale.TIME_STEP`, the value is the number of time_steps that have transpired
      since the start of the current execution in the current phase [LINK].  If execution completes, this is the number
      of time_steps it took for the decision variable to reach the (positive or negative) value of the ``threshold``
      parameter;  if execution was interrupted (using ``terminate_function``), then it corresponds to the time_step at
      which the interruption occurred.
    ..
    The following assignments are made only if time_scale is :keyword:`TimeScale.TRIAL`;  otherwise the value of the
    corresponding attributes is :keyword:`None`.

    * the **error** is assigned as the value of the mechanism's ``RT_MEAN`` outputState and to the
      2nd item of the mechanism's ``outputValue`` attribute.  If ``time_scale`` is :keyword:`TimeScale.TRIAL`,
      the value is the mean response time estimated by the analytic solution used in ``function``.
      [TBI:]
      If ``times_scale`` is :keyword:`TimeScale.TIME_STEP`, the value is the number of time_steps that have transpired
      since the start of the current execution in the current phase [LINK].  If execution completes, this is the number
      of time_steps it took for the decision variable to reach the (positive or negative) value of the ``threshold``
      parameter;  if execution was interrupted (using ``terminate_function``), then it corresponds to the time_step at
      which the interruption occurred.
      ONLY IF TRIAL mode



    * **probability of an incorrect response** is assigned to the value of the mechanism's ``ERROR_RATE`` outputState
      and to the 2nd item of the mechanism's ``outputValue`` attribute.  This is only assigned if ``time_scale``
      is :keyword:`TimeScale.TRIAL`;  if it is :keyword:`TimeScale.TIME_STEP` these values are set to ``None``.
      ONLY IF TRIAL mode
    ..
    * **probability of reaching the ** is assigned to the value of the mechanism's ``ERROR_RATE`` outputState
      and to the 2nd item of the mechanism's ``outputValue`` attribute.  This is only assigned if ``time_scale``
      is :keyword:`TimeScale.TRIAL`;  if it is :keyword:`TimeScale.TIME_STEP` these values are set to ``None``.
      ONLY IF TRIAL mode

                                                           ERROR_RATE,
                                                           PROBABILITY_UPPER_BOUND,
                                                           PROBABILITY_LOWER_BOUND,
                                                           RT_CORRECT_MEAN,
                                                           RT_CORRECT_VARIANCE,

DECISION_VARIABLE = "DecisionVariable"
RT_CORRECT_MEAN = "RT_Correct_Mean"
ERROR_RATE = "Error_Rate"
PROBABILITY_UPPER_BOUND = "Probability_upperBound"
PROBABILITY_LOWER_BOUND = "Probability_lowerBound"
RT_CORRECT_VARIANCE = "RT_Correct_Variance"



In :keyword:`TIME_STEP` mode, this corrsponds to the instantaneous value of the decision variable

In :keyword:`TIME_STEP` mode, this corrsponds to the number of time_steps since execution of the process was begun
within the current :ref:phase [LINK] of the trial

    - self.value (and values of outputStates) contain each outcome value (e.g., ER, DT, etc.)

    NOTE: RT value is "time-steps" assumed to represent ms (re: t0??)

    TRIAL MODE:
       **mean RT (all solutions)
       probability of crossing positive threshold (all solutions)
       variance of RT (NavarroAndFuss only)
       variance of ER (NavarroAndFuss only)

        * **result** to the mechanism's ``value`` attribute, the value of its ``RESULT`` outputState,
          and to the 1st item of the mechanism's ``outputValue`` attribute;
        ..
        * **mean** of the result to the value of the mechanism's ``RESULT_MEAN`` outputState and
          and to the 2nd item of the mechanism's ``outputValue`` attribute;
        ..
        * **variance** of the result to the value of the mechanism's ``RESULT_VARIANCE`` outputState and
          and to the 3rd item of the mechanism's ``outputValue`` attribute.

    TIME_STEP MODE:
        * at each time_step, value of decision variable
        * on completion: number of time_steps (RT)
        * ??variance of the path? (as confirmation of /deviation from noise param??)




COMMENT:
    ?? IS THIS TRUE, OR JUST A CARRYOVER FROM DDM??
    Notes:
    * DDM handles "runtime" parameters (specified in call to function) differently than standard functions:
        any specified params are kept separate from paramsCurrent (Which are not overridden)
        if the FUNCTION_RUN_TIME_PARMS option is set, they are added to the current value of the
            corresponding ParameterState;  that is, they are combined additively with controlSignal output
COMMENT

.. _DDM_Class_Reference:

Class Reference
---------------

"""

# from numpy import sqrt, random, abs, tanh, exp
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.ProcessingMechanism import *
from PsyNeuLink.Functions.Utilities.Utility import *

# DDM outputs (used to create and name outputStates):
DECISION_VARIABLE = "DecisionVariable"
RT_MEAN = "RT_Mean"
ERROR_RATE = "Error_Rate"
PROBABILITY_UPPER_BOUND = "Probability_upperBound"
PROBABILITY_LOWER_BOUND = "Probability_lowerBound"
RT_CORRECT_MEAN = "RT_Correct_Mean"
RT_CORRECT_VARIANCE = "RT_Correct_Variance"


# Indices for results used in return value tuple; auto-numbered to insure sequentiality
class DDM_Output(AutoNumber):
    DECISION_VARIABLE = ()
    RT_MEAN = ()
    ER_MEAN = ()
    P_UPPER_MEAN = ()
    P_LOWER_MEAN = ()
    RT_CORRECT_MEAN = ()
    RT_CORRECT_VARIANCE = ()
    NUM_OUTPUT_VALUES = ()


class DDMError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class DDM(ProcessingMechanism_Base):
# DOCUMENT:   COMBINE WITH INITIALIZATION WITH PARAMETERS
#                    ADD INFO ABOUT B VS. N&F
#                    ADD instantiate_output_states TO INSTANCE METHODS, AND EXPLAIN RE: NUM OUTPUT VALUES FOR B VS. N&F
    """Implements a Drift Diffusion Process

    Computes an analytic solution (for :keyword:`TimeScale.TRIAL`) or numerically integrates
    (for :keyword:`TimeScale.TIME_STEP) a drift diffusion decision process.

    COMMENT:
        Description
        -----------
            DDM is a subclass Type of the Mechanism Category of the Function class
            It implements a Mechanism for several forms of the Drift Diffusion Model (DDM) for
                two alternative forced choice (2AFC) decision making:
                - Bogacz et al. (2006) analytic solution (TimeScale.TRIAL mode -- see kwBogaczEtAl option below):
                    generates error rate (ER) and decision time (DT);
                    ER is used to stochastically generate a decision outcome (+ or - valued) on every run
                - Navarro and Fuss (2009) analytic solution (TImeScale.TRIAL mode -- see kwNavarrosAndFuss:
                    generates error rate (ER), decision time (DT) and their distributions;
                    ER is used to stochastically generate a decision outcome (+ or - valued) on every run
                - stepwise integrator that simulates each step of the integration process (TimeScale.TIME_STEP mode)

        Class attributes
        ----------------
            + functionType (str): DDM
            + classPreference (PreferenceSet): DDM_PreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE
            + variableClassDefault (value):  DDM_Defaults.starting_point
            + paramClassDefaults (dict): {kwTimeScale: TimeScale.TRIAL,
                                          kwDDM_AnalyticSolution: kwBogaczEtAl,
                                          FUNCTION_PARAMS: {DRIFT_RATE:<>
                                                                  STARTING_POINT:<>
                                                                  THRESHOLD:<>
                                                                  NOISE:<>
                                                                  NON_DECISION_TIME:<>},
                                          OUTPUT_STATES: [DECISION_VARIABLE,
                                                           RT_MEAN,
                                                           ERROR_RATE,
                                                           PROBABILITY_UPPER_BOUND,
                                                           PROBABILITY_LOWER_BOUND,
                                                           RT_CORRECT_MEAN,
                                                           RT_CORRECT_VARIANCE,
            + paramNames (dict): names as above

        Class methods
        -------------
            None

        MechanismRegistry
        -----------------
            All instances of DDM are registered in MechanismRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances
    COMMENT


    Arguments
    ---------

    default_input_value : value, list or np.ndarray : Transfer_DEFAULT_BIAS [LINK] -> SHOULD RESOLVE TO VALUE

    function : IntegratorFunction : default BogaczEtAl
        specifies the analytic solution to use for the decision process if ``time_scale`` is set to TimeScale.TRIAL;
        can be :class:`BogaczEtAl` or :class:`NavarroAndFuss (note:  the latter requires that the MatLab engine is
        installed). If ``time_scale`` is set to TimeScale.TIME_STEP, ``function`` is automatically assigned to
        :class:`Integrator`.

    params : Optional[Dict[param keyword, param value]]
        dictionary that can be used to specify parameters of the mechanism, parameters of its function,
        and/or  a custom function and its parameters (see :doc:`Mechanism` for specification of a parms dict).

    time_scale :  TimeScale : TimeScale.TRIAL
        determines whether the mechanism is executed on the :keyword:`TIME_STEP` or :keyword:`TRIAL` time scale.
        This must be set to :keyword:`TimeScale.TRIAL` to use one of the analytic solutions specified by ``function``.
        This must be set to :keyword:`TimeScale.TIME_STEP` to numerically integrate the decision variable.

    name : str : default Transfer-[index]
        string used for the name of the mechanism.
        If not is specified, a default is assigned by MechanismRegistry
        (see :doc:`Registry` for conventions used in naming, including for default and duplicate names).[LINK]

    prefs : Optional[PreferenceSet or specification dict : Process.classPreferences]
        preference set for process.
        if it is not specified, a default is assigned using ``classPreferences`` defined in __init__.py
        (see Description under PreferenceSet for details) [LINK].

    .. context=functionType+kwInit):
            context : str : default ''None''
                   string used for contextualization of instantiation, hierarchical calls, executions, etc.

    Attributes
    ----------

    variable : value : default  DDM_Defaults.starting_point
        Input to mechanism's execute method.  Serves as the "stimulus" component of the drift rate.

    function :  IntegratorFunction : default BogaczEtAl
        Determines method used to compute the outcome of the decision process when ``time_scale`` is
        :keyword:`TimeScale.TRIAL`.  If ``times_scale`` is :keyword:`TimeScale.TIME_STEP`, ``function``
        is automatically assigned to :class`Integrator`, and used to compute the decision process by
        stepwise integration of the decision variable (one step per ``CentralClock.time_step``).

    function_params : Dict[str, value]
        Contains one entry for each parameter of the mechanism's function.
        The key of each entry is the name of (keyword for) a function parameter, and its value is the parameter's value.

    value : value
        output of execute method.

    name : str : default DDM-[index]
        Name of the mechanism.
        Specified in the name argument of the call to create the projection;
        if not is specified, a default is assigned by MechanismRegistry
        (see :doc:`Registry` for conventions used in naming, including for default and duplicate names).[LINK]

    prefs : PreferenceSet or specification dict : Mechanism.classPreferences
        Preference set for mechanism.
        Specified in the prefs argument of the call to create the mechanism;
        if it is not specified, a default is assigned using ``classPreferences`` defined in __init__.py
        (see Description under PreferenceSet for details) [LINK].

    MOVE TO METHOD DEFINITIONS:
    Instance methods:
        - _instantiate_function(context)
            deletes params not in use, in order to restrict outputStates to those that are computed for specified params
        - execute(variable, time_scale, params, context)
            executes specified version of DDM and returns outcome values (in self.value and values of self.outputStates)
        - ou_update(particle, drift, noise, time_step_size, decay)
            single update for OU (special case l=0 is DDM) -- from Michael Shvartsman
        - ddm_update(particle, a, s, dt)
            DOCUMENTATION NEEDED
            from Michael Shvartsman
        - ddm_rt(x0, t0, a, s, z, dt)
            DOCUMENTATION NEEDED
            from Michael Shvartsman
        - ddm_distr(n, x0, t0, a, s, z, dt)
            DOCUMENTATION NEEDED
            from Michael Shvartsman
        - ddm_analytic(bais, t0, drift_rate, noise, threshold)
            DOCUMENTATION NEEDED
            from Michael Shvartsman

    """

    functionType = "DDM"

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in SubtypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'DDMCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    # Assigned in __init__ to match default staring_point
    variableClassDefault = None

    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        kwTimeScale: TimeScale.TRIAL,
        # Assign internal params here (not accessible to user)
        # User accessible params are assigned in assign_defaults_to_paramClassDefaults (in __init__)
        OUTPUT_STATES:[DECISION_VARIABLE,        # Full set specified to include Navarro and Fuss outputs
                        RT_MEAN,
                        ERROR_RATE,
                        PROBABILITY_UPPER_BOUND, # Probability of hitting upper bound
                        PROBABILITY_LOWER_BOUND, # Probability of hitting lower bound
                        RT_CORRECT_MEAN,         # NavarroAnd Fuss only
                        RT_CORRECT_VARIANCE]     # NavarroAnd Fuss only
        # MONITORED_OUTPUT_STATES:[ERROR_RATE,(RT_MEAN, -1, 1)]
    })

    # Set default input_value to default bias for DDM
    paramNames = paramClassDefaults.keys()


    @tc.typecheck
    def __init__(self,
                 default_input_value=NotImplemented,
                 # function:tc.enum(type(BogaczEtAl), type(NavarroAndFuss))=BogaczEtAl(drift_rate=1.0,
                 function=BogaczEtAl(drift_rate=1.0,
                                     starting_point=0.0,
                                     threshold=1.0,
                                     noise=0.5,
                                     t0=.200),
                 params=None,
                 name=None,
                 # prefs:tc.optional(FunctionPreferenceSet)=None,
                 prefs:is_pref_set=None,
                 # context=None):
                 context=functionType+kwInit):
        """Assign type-level preferences, default input value (DDM_Defaults.starting_point) and call super.__init__

        """

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function,
                                                 params=params)

        self.variableClassDefault = self.paramClassDefaults[FUNCTION_PARAMS][STARTING_POINT]

        if default_input_value is NotImplemented:
            default_input_value = params[FUNCTION_PARAMS][STARTING_POINT]

        super(DDM, self).__init__(variable=default_input_value,
                                  params=params,
                                  name=name,
                                  prefs=prefs,
                                  # context=context,
                                  context=self)

    def _validate_params(self, request_set, target_set=NotImplemented, context=None):

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        functions = {BogaczEtAl, NavarroAndFuss}
        if not target_set[FUNCTION] in functions:
            function_names = list(function.functionName for function in functions)
            raise DDMError("{} param of {} must be one of the following functions: {}".
                           format(self.name, function_names))

    # def _instantiate_function(self, context=NotImplemented):
    def _instantiate_attributes_before_function(self, context=None):
        """Delete params not in use, call super.instantiate_execute_method
        :param context:
        :return:
        """

        # Assign output mappings:
        self._outputStateValueMapping = {}
        self._outputStateValueMapping[DECISION_VARIABLE] = DDM_Output.DECISION_VARIABLE.value
        self._outputStateValueMapping[RT_MEAN] = DDM_Output.RT_MEAN.value
        self._outputStateValueMapping[ERROR_RATE] = DDM_Output.ER_MEAN.value
        self._outputStateValueMapping[PROBABILITY_UPPER_BOUND] = DDM_Output.P_UPPER_MEAN.value
        self._outputStateValueMapping[PROBABILITY_LOWER_BOUND] = DDM_Output.P_LOWER_MEAN.value

        # If not using Navarro and Fuss, get rid of extra params:
        if self.function is BogaczEtAl:
            outputStates = self.params[OUTPUT_STATES]
            try:
                del outputStates[outputStates.index(RT_CORRECT_MEAN)]
                del outputStates[outputStates.index(RT_CORRECT_VARIANCE)]
                # del outputStates[outputStates.index(TOTAL_ALLOCATION)]
                # del outputStates[outputStates.index(TOTAL_COST)]
            except ValueError:
                pass
        else:
            self._outputStateValueMapping[RT_CORRECT_MEAN] = DDM_Output.RT_CORRECT_MEAN.value
            self._outputStateValueMapping[RT_CORRECT_VARIANCE] = DDM_Output.RT_CORRECT_VARIANCE.value
            # self._outputStateValueMapping[TOTAL_ALLOCATION] = DDM_Output.TOTAL_ALLOCATION.value
            # self._outputStateValueMapping[TOTAL_COST] = DDM_Output.TOTAL_COST.value

        super()._instantiate_attributes_before_function(context=context)

    def __execute__(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale = TimeScale.TRIAL,
                context=None):
        """Execute DDM function (currently only trial-level, analytic solution)

        Execute DDM and estimate outcome or calculate trajectory of decision variable
        Currently implements only trial-level DDM (analytic solution) and returns:
            - stochastically estimated decion outcome (convert mean ER into value between 1 and -1)
            - mean ER
            - mean DT
            - mean ER and DT variabilty (kwNavarroAndFuss ony)
        Return current decision variable (self.outputState.value) and other output values (self.outputStates[].value

        Arguments:

        # CONFIRM:
        variable (float): set to self.value (= self.inputValue)
        - params (dict):  runtime_params passed from Mechanism, used as one-time value for current execution:
            + DRIFT_RATE (float)
            + THRESHOLD (float)
            + kwDDM_Bias (float)
            + NON_DECISION_TIME (float)
            + NOISE (float)
        - time_scale (TimeScale): determines "temporal granularity" with which mechanism is executed
        - context (str)

        Returns the following values in self.value (2D np.array) and in
            the value of the corresponding outputState in the self.outputStates dict:
            - decision variable (float)
            - mean error rate (float)
            - mean RT (float)
            - correct mean RT (float) - Navarro and Fuss only
            - correct mean ER (float) - Navarro and Fuss only

        :param self:
        :param variable (float)
        :param params: (dict)
        :param time_scale: (TimeScale)
        :param context: (str)
        :rtype self.outputState.value: (number)
        """

        # EXECUTE INTEGRATOR SOLUTION (TIME_STEP TIME SCALE) -----------------------------------------------------
        if time_scale == TimeScale.TIME_STEP:
            raise MechanismError("TIME_STEP mode not yet implemented for DDM")
            # IMPLEMENTATION NOTES:
            # Implement with calls to a step_function, that does not reset self.outputValue
            # Should be sure that initial value of self.outputState.value = self.parameterStates[BIAS]
            # Assign "self.decision_variable"
            # Implement terminate() below

        # EXECUTE ANALYTIC SOLUTION (TRIAL TIME SCALE) -----------------------------------------------------------
        elif time_scale == TimeScale.TRIAL:

            # # Get length of self.outputValue from OUTPUT_STATES
            # # Note: use paramsCurrent here (instead of outputStates), as during initialization the execute method
            # #       is run (to evaluate self.outputValue) before outputStates have been instantiated
            # self.outputValue = [None] * len(self.paramsCurrent[OUTPUT_STATES])

            # # TEST PRINT:
            # print ("\nDDM RUN")
            # print ("stimulus: {}".format(self.inputState.value))
            # print ("control signal: {}\n".format(self.parameterStates[DRIFT_RATE].value))

            # - convolve inputState.value (signal) w/ driftRate param value (attentional contribution to the process)
            drift_rate = float((self.inputState.value * self.parameterStates[DRIFT_RATE].value))
            starting_point = float(self.parameterStates[STARTING_POINT].value)
            threshold = float(self.parameterStates[THRESHOLD].value)
            noise = float(self.parameterStates[NOISE].value)
            t0 = float(self.parameterStates[NON_DECISION_TIME].value)

            result = self.function(params={DRIFT_RATE:drift_rate,
                                           STARTING_POINT:starting_point,
                                           THRESHOLD:threshold,
                                           NOISE:noise,
                                           NON_DECISION_TIME:t0})

            # Assign outputValue

            if isinstance(self.function.__self__, BogaczEtAl):
                self.outputValue[DDM_Output.RT_MEAN.value], self.outputValue[DDM_Output.ER_MEAN.value] = result
                self.outputValue[DDM_Output.P_UPPER_MEAN.value] = 1 - self.outputValue[DDM_Output.ER_MEAN.value]
                self.outputValue[DDM_Output.P_LOWER_MEAN.value] = self.outputValue[DDM_Output.ER_MEAN.value]

            elif isinstance(self.function.__self__, NavarroAndFuss):
                self.outputValue[DDM_Output.RT_MEAN.value] = result[NF_Results.MEAN_DT.value]
                self.outputValue[DDM_Output.ER_MEAN.value] = 1-result[NF_Results.MEAN_ER.value]
                self.outputValue[DDM_Output.P_UPPER_MEAN.value] = result[NF_Results.MEAN_ER.value]
                self.outputValue[DDM_Output.P_LOWER_MEAN.value] = 1 - result[NF_Results.MEAN_ER.value]
                self.outputValue[DDM_Output.RT_CORRECT_MEAN.value] = result[NF_Results.MEAN_CORRECT_RT.value]
                self.outputValue[DDM_Output.RT_CORRECT_VARIANCE.value] = result[NF_Results.MEAN_CORRECT_VARIANCE.value]
                # CORRECT_RT_SKEW = results[DDMResults.MEAN_CORRECT_SKEW_RT.value]

            # Convert ER to decision variable:
            if random() < self.outputValue[DDM_Output.ER_MEAN.value]:
                self.outputValue[DDM_Output.DECISION_VARIABLE.value] = np.atleast_1d(-1 * threshold)
            else:
                # # MODIFIED 10/5/16 OLD:
                # self.outputValue[DDM_Output.DECISION_VARIABLE.value] = np.atleast_1d(threshold)
                # MODIFIED 10/5/16 NEW:
                self.outputValue[DDM_Output.DECISION_VARIABLE.value] = threshold
                # MODIFIED 10/5/16 END

            return self.outputValue

        else:
            raise MechanismError("time_scale not specified for DDM")

    def ou_update(self, particle, drift, noise, time_step_size, decay):
        ''' Single update for OU (special case l=0 is DDM)'''
        return particle + time_step_size * (decay * particle + drift) + random.normal(0, noise) * sqrt(time_step_size)


    def ddm_update(self, particle, a, s, dt):
        return self.ou_update(particle, a, s, dt, decay=0)


    def ddm_rt(self, x0, t0, a, s, z, dt):
        samps = 0
        particle = x0
        while abs(particle) < z:
            samps = samps + 1
            particle = self.ou_update(particle, a, s, dt, decay=0)
        # return -rt for errors as per convention
        return (samps * dt + t0) if particle > 0 else -(samps * dt + t0)

    def ddm_distr(self, n, x0, t0, a, s, z, dt):
        return np.fromiter((self.ddm_rt(x0, t0, a, s, z, dt) for i in range(n)), dtype='float64')


    def terminate_function(self, context=None):
        """Terminate the process

        called by process.terminate() - MUST BE OVERRIDDEN BY SUBCLASS IMPLEMENTATION
        returns output

        :rtype CurrentStateTuple(state, confidence, duration, controlModulatedParamValues)
        """
        # IMPLEMENTATION NOTE:  TBI when time_step is implemented for DDM