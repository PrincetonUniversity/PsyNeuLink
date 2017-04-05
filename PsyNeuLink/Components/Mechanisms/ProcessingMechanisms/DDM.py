# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ***************************************************  DDM *************************************************************
#
"""
..
    Sections:
      * :ref:`DDM_Overview`
      * :ref:`DDM_Creation`
      * :ref:`DDM_Execution`
      * :ref:`DDM_Class_Reference`

.. _DDM_Overview:

Overview
--------
The DDM mechanism implements the "Drift Diffusion Model" (also know as the Diffusion Decision, Accumulation to Bound,
Linear Integrator, and Wiener Process First Passage Time Model [REFS]). This corresponds to a continuous version of
the sequential probability ratio test (SPRT [REF]), which is the statistically optimal procedure for two alternative
forced choice (TAFC) decision making ([REF]).  It can be executed analytically using one of two solutions (in `TRIAL`
mode), or integrated numerically (in `TIME_STEP` mode).

.. _DDM_Creation:

Creating a DDM Mechanism
-----------------------------
A DDM Mechanism can be instantiated directly by calling its constructor, or by using the `mechanism` function
and specifying DDM as its `mech_spec` argument.  The analytic solution used in `TRIAL` mode is selected
using the `function <DDM.function>` argument, which can be simply the name of a DDM function (first example below),
or a call to the function with arguments specifying its parameters (see `DDM_Execution` below for a description of
DDM function parameters) and, optionally, a `ControlProjection` (second example)::
    my_DDM = DDM(function=BogaczEtAl)
    my_DDM = DDM(function=BogaczEtAl(drift_rate=0.2, threshold=(1, ControlProjection))
    COMMENT:
    my_DDM = DDM(default_input_value=[0, 0, 0]
                 function=BogaczEtAl(drift_rate=0.2, threshold=(1, ControlProjection))
    COMMENT
.. _DDM_Input:
**Input**.  The `default_input_value` argument specifies the default value to use as the stimulus component of the
:ref:`drift rate <DDM_Drift_Rate>` for the decision process.  It must be a single scalar value.
COMMENT:
[TBI - MULTIPROCESS DDM - REPLACE ABOVE]
**Input**.  The ``default_input_value`` argument specifies the default value to use as the stimulus component of the
:ref:`drift rate <DDM_Drift_Rate>` for each decision process, as well as the number of decision processes implemented
and the corresponding format of the ``input`` required by calls to its ``execute`` and ``run`` methods.  This can be a
single scalar value or an an array (list or 1d np.array). If it is a single value (as in the first two examples above),
a single DDM process is implemented, and the input is used as the stimulus component of the
:ref:`drift rate <DDM_Drift_Rate>` for the process. If the input is an array (as in the third example above),
multiple parallel DDM processes are implemented all of which use the same parameters but each of which receives its
own input (from the corresponding element of the input array) and is executed independently of the others.
COMMENT

.. _DDM_Structure:

Structure
---------
The DDM mechanism implements a general form of the decision process.  A DDM mechanism has a single `inputState
<InputState>` the `value <DDM.value>` of which is used is assigned to the `input <DDM.input>` specified by its
:py:meth:`execute <Mechanism.Mechanism_Base.execute>` or :py:meth:`run <Mechanism.Mechanism_Base.run>` methods,
and that is used as the stimulus component of the :ref:`drift rate <DDM_Drift_Rate>` for the decision process.  The
decision process can be configured to execute in different modes.  The `function <DDM.function>` and
`time_scale <DDM.time_scale>` parameters are the primary determinants of how the decision process is executed,
and what information is returned. The `function <DDM.function>` parameter specifies the analytical solution to use
when `time_scale <DDM.time_scale>` is  set to `TimeScale.TRIAL` (see :ref:`Functions <DDM_Functions>` below); when
`time_scale <DDM.time_scale>` set to  `TimeScale.TIME_STEP`, executing the DDM mechanism numerically integrates the
path of the decision variable (see :ref:`Execution <DDM_Execution>` below). The number of **outputStates** is
determined by the `function <DDM.function>` in use (see :ref:`list of output values <DDM_Results>` below).
COMMENT:
[TBI - MULTIPROCESS DDM - REPLACE ABOVE]
The DDM mechanism implements a general form of the decision process.  A DDM mechanism assigns one **inputState** to
each item in the `default_input_value` argument, corresponding to each of the decision processes implemented
(see :ref:`Input <DDM_Input>` above). The decision process can be configured to execute in different modes.  The
`function <DDM.function>` and `time_scale <DDM.time_scale>` parameters are the primary determinants of how the
decision process is executed, and what information is returned. The `function <DDM.function>` parameter specifies
the analytical solution to use when `time_scale <DDM.time_scale>` is  set to :keyword:`TimeScale.TRIAL` (see
:ref:`Functions <DDM_Functions>` below); when `time_scale <DDM.time_scale>` set to `TimeScale.TIME_STEP`,
executing the DDM mechanism numerically integrates the path of the decision variable (see `Execution <DDM_Execution>`
below).  The number of `outputStates <OutputState>` is determined by the `function <DDM.function>` in use (see
:ref:`list of output values <DDM_Results>` below).
COMMENT
COMMENT:
[TBI - average_output_states ARGUMENT/OPTION AFTER IMPLEMENTING MULTIPROCESS DDM]
OUTPUT MEASURE?? OUTCOME MEASURE?? RESULT?? TYPE OF RESULT??
If only a single decision process was run, then the value of each outputState is the corresponding output of
the decision process.  If there is more than one decision process (i.e., the input has more than one item), then
the content of the outputStates is determined by the ``average_output_states`` argument.  If it is :keyword:`True`,
then each outputState (and item of ``outputValue``) contains a single value, which is the average of the output
values of that type over all of the processes run.  If ``average_output_states`` is :keyword:`False` (the default),
then the value of each ouputState is a 1d array, each element of which is the outcome of that type for the
corresponding decision process.
COMMENT

.. _DDM_Functions:

DDM Functions
~~~~~~~~~~~~~

The `function <DDM.function>` parameter can be used to select one of two analytic solutions (`BogaczEtAl` and
`NavarroAndFuss`) that are used when `time_scale <DDM.time_scale>` is set to `TimeScale.TRIAL`.  These both return an
expected mean response time and accuracy, while `NavarroAndFuss` also returns an expected mean correct response time
and accuracy.

.. _DDM_Parameters:

DDM Parameters
~~~~~~~~~~~~~~

The DDM process uses the same set of parameters for all modes of execution.  These can be specified as arguments
for the functions used in :keyword:`TRIAL` mode, or in a params dictionary assigned to the `params` argument,
using the keywords in the list below, as in the following example::
    my_DDM = DDM(function=BogaczEtAl(drift_rate=0.1),
                 params={DRIFT_RATE:(0.2, ControlProjection),
                         STARTING_POINT:-0.5},
                 time_scale=TimeScale.TIME_STEP)

.. note::  Parameters specified in the `params` argument (as in the example above) will be used for both
   `TRIAL` and `TIME_STEP` mode, since parameters specified in the `params` argument of a mechanism's constructor
   override corresponding ones specified as arguments of its `function <Mechanism.Mechanism_Base.function>`
   (see :doc:`COMPONENT`).  In the example above, this means that even if the `time_scale <DDM.time_scale>` parameter is
   set to `TimeScale.TRIAL`, the `drift_rate` of 0.2 will be used (rather than 0.1).  For parameters NOT specified
   as entries in the `params` dictionary, the value specified for those in the function will be used in both `TRIAL`
   and `TIME_STEP` mode.

The parameters for the DDM are:

.. _DDM_Drift_Rate:

* `DRIFT_RATE <drift_rate>` (default 0.0)
  - multiplies the input to the mechanism before assigning it to the `variable <DDM.variable>` on each call of
  `function <DDM.function>`.  The resulting value is further multiplied by the value of any ControlProjections to the
  `DRIFT_RATE` parameterState. The `drift_rate` parameter can be thought of as the "automatic" component (baseline
  strength) of the decision process, the value received from a ControlProjection as the "attentional" component,
  and the input its "stimulus" component.  The product of all three determines the drift rate in effect for each
  time_step of the decision process.
..
* `STARTING_POINT <starting_point>` (default 0.0)
  - specifies the starting value of the decision variable.  If `time_scale <DDM.time_scale>` is `TimeScale.TIME_STEP`,
  the `starting_point` is added to the decision variable on the first call to `function <DDM.function>` but not
  subsequently.
..
* `THRESHOLD` (default 1.0)
  - specifies the stopping value for the decision process.  When `time_scale <DDM.time_scale>` is `TIME_STEP`, the
  integration process is terminated when the absolute value of the decision variable equals the absolute value
  of threshold.  The `threshold` parameter must be greater than or equal to zero.
..
* `NOISE` (default 0.5)
  - specifies the variance of the stochastic ("diffusion") component of the decision process.  If
  `time_scale <DDM.time_scale>` is `TIME_STEP`, this value is multiplied by a random sample drawn from a zero-mean
  normal (Gaussian) distribution on every call of function <DDM.function>`, and added to the decision variable.
..
* `NON_DECISION_TIME` (default 0.2)
  specifies the `t0` parameter of the decision process (in units of seconds).
  COMMENT:
  when ``time_scale <DDM.time_scale>`` is  :keyword:`TIME_STEP`, it is added to the number of time steps
  taken to complete the decision process when reporting the response time.
  COMMENT

.. _DDM_Execution:

Execution
---------

When a DDM mechanism is executed it computes the decision process, either analytically (in `TRIAL` mode) or by
step-wise integration (in `TIME_STEP` mode).
COMMENT:
[TBI - MULTIPROCESS DDM - REPLACE ABOVE]
When a DDM mechanism is executed it computes the decision process, either analytically (in :keyword:`TRIAL` mode)
or by step-wise integration (in :keyword:`TIME_STEP` mode).  As noted above, if the input is a single value,
it computes a single DDM process.  If the input is a list or array, then multiple parallel DDM processes are executed,
with each element of the input used for the corresponding process.  All use the same set of parameters,
so the analytic solutions (used in :keyword:`TRIAL` mode) for a given input will be the same; to implement processes in
this mode that use different parameters, a separate DDM mechanism should explicitly be created for each. In
:keyword:`TIME_STEP` mode, the noise term will resolve to different values in each time step, so the integration
paths and outcomes for the same input value will vary. This can be used to generate distributions of the process for a
single set of parameters that are not subject to the analytic solution (e.g., for time-varying drift rates).
COMMENT

.. note::
   DDM handles "runtime" parameters (specified in a call to its
   :py:meth:`execute <Mechanism.Mechanism_Base.exeucte>` or :py:meth:`run <Mechanism.Mechanism_Base.run>` methods)
   differently than standard Components: runtime parameters are added to the mechanism's current value of the
   corresponding parameterState (rather than overriding it);  that is, they are combined additively with the value of
   any `ControlProjection` it receives to determine the parameter's value for that execution.  The parameterState's
   value is then restored to its original value (i.e., either its default value or the one assigned when it was
   created) for the next execution.
COMMENT:
  ADD NOTE ABOUT INTERROGATION PROTOCOL, USING ``terminate_function``
  ADD NOTE ABOUT RELATIONSHIP OF RT TO time_steps TO t0 TO ms
COMMENT

.. _DDM_Results:

After each execution of the mechanism:

    * the value of the **decision variable** is assigned to the mechanism's `value <DDM.value>` attribute, the value of
      the 1st item of its `outputValue <DDM.outputState>` attribute, and as the value of its `DDM_DECISION_VARIABLE`
      outputState.
    ..
    * **response time** is assigned as the value of the 2nd item of the mechanism's `outputValue <DDM.outputValue>`
      attribute and as the value of its `RESPONSE_TIME` outputState.  If `time_scale <DDM.time_scale>` is
      `TimeScale.TRIAL`, the value is the mean response time (in seconds) estimated by the analytic solution used in
      `function <DDM.function>`.
      COMMENT:
        [TBI:]
        If ``time_scale <DDM.time_scale>`` is :py:data:`TimeScale.TIME_STEP <TimeScale.TimeScale.TIME_STEP>`,
        the value is the number of time_steps that have transpired since the start of the current execution in the
        current :ref:`phase <System_Execution_Phase>`.  If execution completes, this is the number of time_steps it
        took for the decision variable to reach the (positive or negative) value of the `threshold` parameter;  if
        execution was interrupted (using :py:meth:`terminate_function <DDM.terminate_function>`), then it corresponds
        to the time_step at which the interruption occurred.
      COMMENT
    ..
    The following assignments are made only if `time_scale <DDM.time_scale>` is
    :py:data:`TimeScale.TIME_STEP <TimeScale.TimeScale.TIME_STEP>`;  otherwise the value of the corresponding
    attributes is `None`.
    * **probability of reaching the upper threshold** is assigned to the 3rd item of the mechanism's
      `outputValue <DDM.outputValue>` attribute, and as the value of its `PROBABILITY_UPPER_THRESHOLD` outputState.
      If `time_scale <DDM.time_scale>` is `TimeScale.TRIAL`, the value is the probability (calculated by the analytic
      solution used in `function <DDM.function>`) that the value of the decision variable reached the upper (
      positive) threshold. Often, by convention, the upper threshold is associated with the ccorrect response,
      in which case `PROBABILITY_LOWER_THRESHOLD` corresponds to the accuracy of the decision process.
      COMMENT:
      [TBI:]
          If ``time_scale <DDM.time_scale>`` is :keyword:`TimeScale.TIME_STEP` and the execution has completed, this is a binary value
          indicating whether the decision process reached the upper (positive) threshold. If execution was interrupted
          (using :py:meth:`terminate_function <DDM.terminate_function>`, sometimes referred to as the
          :ref:`interrogation protocol <LINK>`, then the value corresponds to the current likelihood that the upper
          threshold would have been reached.
      COMMENT
    ..
    * **probability of reaching the lower threshold** is assigned to the 4th item of the mechanism's
      `outputValue <DDM.outputValue>` attribute and as the value of its `PROBABILITY_LOWER_THRESHOLD` outputState.
      If `time_scale <DDM.time_scale>` is `TimeScale.TRIAL`, the value is the probability (calculated by the analytic
      solution used in `function <DDM.function>`) that the value of the decision variable reached the lower (negative)
      threshold.  Often, by convention, the lower threshold is associated with the incorrect response, in which case
      `PROBABILITY_LOWER_THRESHOLD` corresponds to the error rate of the decision process.
      COMMENT:
          [TBI:]
          If ``time_scale <DDM.time_scale>`` is :keyword:`TimeScale.TIME_STEP` and the execution has completed, this is a binary value
          indicating whether the decision process reached the lower (negative) threshold. If execution was interrupted
          (using :py:method:`terminate_method <DDM.terminate_function>`, sometimes referred to as the
          :ref:`interrogation protocol <LINK>`), then the value corresponds to the current likelihood that the lower
          threshold would have been reached.
      COMMENT:
    ..
    The following assignments are made only assigned if the `NavarroAndFuss` function is used, and
    `time_scale <DDM.time_scale>` is `TimeScale.TRIAL`.  Otherwise, neither the `outputValue <DDM.outputValue>`
    nor the `outputState` attribute have a 6th item (if another function is assigned) or they are asssigned
    `None` if `time_scale <DDM.time_scale>` is `TimeScale.TIME_STEP`.
    * **mean correct response time** (in seconds) is assigned to the 5th item of the mechanism's
    `outputValue <DDM.outputValue>` attribute and as  the value of its `RT_CORRECT_MEAN` outputState.
    ..
    * **variance of correct response time** is assigned to the 6th item of the mechanism's
      `outputValue <DDM.outputValue>` attribute and as the value of its `RT_CORRECT_VARIANCE` outputState.
    COMMENT:
        In time_step mode, compute and report variance of the path
        (e.g., as confirmation of /deviation from noise param??)??
    COMMENT
    ..

.. _DDM_Class_Reference:

Class Reference
---------------
"""

# from numpy import sqrt, random, abs, tanh, exp
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import *
from PsyNeuLink.Components.Functions.Function import *

# DDM outputs (used to create and name outputStates):
DDM_DECISION_VARIABLE = "DDM_decision_variable"
DDM_RESPONSE_TIME = "DDM_response_time"
DDM_PROBABILITY_UPPER_THRESHOLD = "DDM_probability_upperBound"
DDM_PROBABILITY_LOWER_THRESHOLD = "DDM_probability_lowerBound"
DDM_RT_CORRECT_MEAN = "DDM_RT_Correct_Mean"
DDM_RT_CORRECT_VARIANCE = "DDM_RT_correct_variance"


# Indices for results used in return value tuple; auto-numbered to insure sequentiality
class DDM_Output(AutoNumber):
    DECISION_VARIABLE = ()
    RESPONSE_TIME = ()
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
#                    ADD _instantiate_output_states TO INSTANCE METHODS, AND EXPLAIN RE: NUM OUTPUT VALUES FOR B VS. N&F
    """
    DDM(                       \
    default_input_value=None,  \
    function=BogaczEtAl,       \
    params=None,               \
    name=None,                 \
    prefs=None)
    Implements a Drift Diffusion Process
    Computes an analytic solution when `time_scale <DDM.time_scale>` is `TimeScale.TRIAL`, or numerically integrates it
    when `time_scale <DDM.time_scale>` is `TimeScale.TIME_STEP`.
    COMMENT:
        Description
        -----------
            DDM is a subclass Type of the Mechanism Category of the Component class
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
            + componentType (str): DDM
            + classPreference (PreferenceSet): DDM_PreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE
            + variableClassDefault (value):  STARTING_POINT
            + paramClassDefaults (dict): {TIME_SCALE: TimeScale.TRIAL,
                                          kwDDM_AnalyticSolution: kwBogaczEtAl,
                                          FUNCTION_PARAMS: {DRIFT_RATE:<>
                                                                  STARTING_POINT:<>
                                                                  THRESHOLD:<>
                                                                  NOISE:<>
                                                                  NON_DECISION_TIME:<>},
                                          OUTPUT_STATES: [DDM_DECISION_VARIABLE,
                                                          DDM_RESPONSE_TIME,
                                                          DDM_PROBABILITY_UPPER_THRESHOLD,
                                                          DDM_PROBABILITY_LOWER_THRESHOLD,
                                                          DDM_RT_CORRECT_MEAN,
                                                          DDM_RT_CORRECT_VARIANCE,
            + paramNames (dict): names as above
        Class methods
        -------------
            - plot() : generates a dynamic plot of the DDM
        MechanismRegistry
        -----------------
            All instances of DDM are registered in MechanismRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances
    COMMENT
    Arguments
    ---------
    default_input_value : value, list or np.ndarray : default FUNCTION_PARAMS[STARTING_POINT]
        the input to the mechanism to use if none is provided in a call to its
        :py:data:`execute <Mechanism.Mechanism_Base.execute>` or :py:data:`run <Mechanism.Mechanism_Base.run>` methods;
        also serves as a template to specify the length of `variable <DDM.variable>` for `function <DDM.function>`,
        and the primary outputState of the mechanism (see :ref:`Input` <DDM_Creation>` for how an input with a length
        of greater than 1 is handled).
    function : IntegratorFunction : default BogaczEtAl
        specifies the analytic solution to use for the decision process if `time_scale <DDM.time_scale>` is set to
        `TimeScale.TRIAL`; can be `BogaczEtAl` or `NavarroAndFuss` (note:  the latter requires that the MatLab engine
        is installed). If `time_scale <DDM.time_scale>` is set to `TimeScale.TIME_STEP`, `function <DDM.function>` must
        be `Integrator` with an 'integration_type <Integrator.integration_type>' of DIFFUSION, and the mechanism
        will return the result of one time step.
    time_scale :  TimeScale : default TimeScale.TRIAL
        specifies whether the mechanism is executed on the time_step or trial time scale.
        This must be set to `TimeScale.TRIAL` to use one of the analytic solutions specified by
        `function <DDM.function>`. This  must be set to `TimeScale.TIME_STEP` to numerically (path) integrate the
        decision variable.
    params : Optional[Dict[param keyword, param value]]
        a dictionary that can be used to specify parameters of the mechanism, parameters of its function,
        and/or  a custom function and its parameters (see `Mechanism` for specification of a params dict).
    name : str : default DDM-<index>
        a string used for the name of the mechanism.
        If not is specified, a default is assigned by `MechanismRegistry`
        (see `Registry <LINK>` for conventions used in naming, including for default and duplicate names).
    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the PreferenceSet for the process.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see `PreferenceSet <LINK>` for details).
    COMMENT:
    context=componentType+INITIALIZING):
        context : str : default ''None''
               string used for contextualization of instantiation, hierarchical calls, executions, etc.
    COMMENT
    Attributes
    ----------
    variable : value : default  FUNCTION_PARAMS[STARTING_POINT]
        the input to mechanism's execute method.  Serves as the "stimulus" component of the drift rate.
    function :  IntegratorFunction : default BogaczEtAl
        the function used to compute the outcome of the decision process when `time_scale <DDM.time_scale>` is
        `TimeScale.TRIAL`.  If `time_scale <DDM.time_scale>` is set to `TimeScale.TIME_STEP`, `function <DDM.function>`
        must be `Integrator` with an 'integration_type <Integrator.integration_type>' of DIFFUSION, and the mechanism
        will return the result of one time step.
    function_params : Dict[str, value]
        contains one entry for each parameter of the mechanism's function.
        The key of each entry is the name of (keyword for) a function parameter, and its value is the parameter's value.
    value : 2d np.array[array(float64),array(float64),array(float64),array(float64)]
        result of executing DDM `function <DDM.function>`; same items as `outputValue <DDM.outputValue>`.
    plot :
    COMMENT:
        CORRECTED:
        value : 1d np.array
            the output of `function <DDM.function>`;  also assigned to `value <DDM.value>` of the
            `DDM_DECISION_VARIABLE` outputState and the first item of `outputValue <DDM.outputValue>`.
    COMMENT
    outputValue : List[array(float64),array(float64),array(float64),array(float64)]
        a list with the following items:
        * **decision variable** (value of `DDM_DECISION_VARIABLE` outputState);
        * **response time** (value of `DDM_RESPONSE_TIME` outputState);
        * **probability of reaching upper threshold** (value of `DDM_PROBABILITY_UPPER_THRESHOLD` outputState);
          if `time_scale <DDM.time_scale>` is `TimeScale.TIME_STEP, this is `None`;
        * **probability of reaching lower threshold** (value of `DDM_PROBABILITY_LOWER_THRESHOLD` outputState);
          if `time_scale <DDM.time_scale>` is `TimeScale.TIME_STEP, this is `None`;
        * **mean of correct response times** (value of :keyword:`DDM_RT_CORRECT_MEAN` outputState);
          only assigned if `function <DDM.function>` is `NavarroAndFuss` and `time_scale <DDM.time_scale>` is
          `TimeScale.TRIAL, otherwise it is `None`;
        * **variance of correct response times** (value of `DDM_RT_CORRECT_VARIANCE` outputState);
          only assigned if `function <DDM.function>` is `NavarroAndFuss` and `time_scale <DDM.time_scale>` is
          `TimeScale.TRIAL, otherwise it is `None`;
    time_scale : TimeScale : default TimeScale.TRIAL
        determines the TimeScale at which the decision process is executed.
    name : str : default DDM-<index>
        the name of the mechanism.
        Specified in the name argument of the call to create the projection;
        if not is specified, a default is assigned by MechanismRegistry
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).
    prefs : PreferenceSet or specification dict : Mechanism.classPreferences
        a PreferenceSet for the mechanism.
        Specified in the prefs argument of the call to create the mechanism;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :py:class:`PreferenceSet <LINK>` for details).
    COMMENT:
        MOVE TO METHOD DEFINITIONS:
        Instance methods:
            - _instantiate_function(context)
                deletes params not in use, in order to restrict outputStates to those that are computed for
                specified params
            - execute(variable, time_scale, params, context)
                executes specified version of DDM and returns outcome values (in self.value and values of
                self.outputStates)
            - _out_update(particle, drift, noise, time_step_size, decay)
                single update for OU (special case l=0 is DDM) -- from Michael Shvartsman
            - _ddm_update(particle, a, s, dt)
                DOCUMENTATION NEEDED
                from Michael Shvartsman
            - _ddm_rt(x0, t0, a, s, z, dt)
                DOCUMENTATION NEEDED
                from Michael Shvartsman
            - _ddm_distr(n, x0, t0, a, s, z, dt)
                DOCUMENTATION NEEDED
                from Michael Shvartsman
            - _ddm_analytic(bais, t0, drift_rate, noise, threshold)
                DOCUMENTATION NEEDED
                from Michael Shvartsman
    COMMENT
    """

    componentType = "DDM"

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in SubtypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'DDMCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    # Assigned in __init__ to match default staring_point
    variableClassDefault = None

    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        TIME_SCALE: TimeScale.TRIAL,
        # Assign internal params here (not accessible to user)
        # User accessible params are assigned in _assign_defaults_to_paramClassDefaults (in __init__)
        # # MONITOR_FOR_CONTROL:[PROBABILITY_LOWER_THRESHOLD,(RESPONSE_TIME, -1, 1)]
        OUTPUT_STATES:[                                        # Full set specified to include Navarro and Fuss outputs
            {NAME:DDM_DECISION_VARIABLE,
             INDEX:DDM_Output.DECISION_VARIABLE.value},

            {NAME:DDM_RESPONSE_TIME,
             INDEX:DDM_Output.RESPONSE_TIME.value},

            {NAME:DDM_PROBABILITY_UPPER_THRESHOLD,          # Probability of hitting upper bound
             INDEX:DDM_Output.P_UPPER_MEAN.value},

            {NAME:DDM_PROBABILITY_LOWER_THRESHOLD,          # Probability of hitting lower bound
             INDEX:DDM_Output.P_LOWER_MEAN.value},

            {NAME:DDM_RT_CORRECT_MEAN,                      # NavarroAnd Fuss only
             INDEX:DDM_Output.RT_CORRECT_MEAN.value},

            {NAME:DDM_RT_CORRECT_VARIANCE,                  # NavarroAnd Fuss only
             INDEX:DDM_Output.RT_CORRECT_VARIANCE.value}
        ]
    })

    # Set default input_value to default bias for DDM
    paramNames = paramClassDefaults.keys()


    @tc.typecheck
    def __init__(self,
                 default_input_value=None,
                 # function:tc.enum(type(BogaczEtAl), type(NavarroAndFuss))=BogaczEtAl(drift_rate=1.0,
                 function=BogaczEtAl(drift_rate=1.0,
                                     starting_point=0.0,
                                     threshold=1.0,
                                     noise=0.5, 
                                     t0=.200),
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 name=None,
                 plot_threshold=None,
                 # prefs:tc.optional(ComponentPreferenceSet)=None,
                 prefs:is_pref_set=None,
                 # context=None):
                 context=componentType+INITIALIZING):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function,
                                                  time_scale=time_scale,
                                                  plot_threshold = plot_threshold,
                                                  params=params,
                                                  )

        self.variableClassDefault = self.paramClassDefaults[FUNCTION_PARAMS][STARTING_POINT]

        if default_input_value is None:
          try: 
            default_input_value = params[FUNCTION_PARAMS][STARTING_POINT]
          except: 

            default_input_value = 0.0
            

        super(DDM, self).__init__(variable=default_input_value,
                                  params=params,
                                  name=name,
                                  prefs=prefs,
                                  # context=context)
                                  context=self)
    def plot(self, threshold = 10.0):
        import matplotlib.pyplot as plt
        plt.ion()

        # # Select a random seed to ensure that the test run will be the same as the real run
        seed_value = np.random.randint(0, 100)
        np.random.seed(seed_value)

        result_check = 0
        time_check = 0

        while abs(result_check) < threshold:
            time_check += 1
            result_check = self.get_axes_function(context='plot')

        # Re-set random seed for the real run
        np.random.seed(seed_value)
        axes = plt.gca()
        axes.set_xlim([0, time_check])
        axes.set_xlabel("Time Step", weight ="heavy", size="large")
        axes.set_ylim([-1.25 * threshold, 1.25 * threshold])
        axes.set_ylabel("Position", weight ="heavy", size="large")
        plt.axhline(y=threshold, linewidth=1, color='k', linestyle='dashed')
        plt.axhline(y=-threshold, linewidth=1, color='k', linestyle='dashed')
        plt.plot()

        result = 0
        time = 0
        while abs(result) < threshold:
            time += 1
            result = self.plot_function(context='plot')
            plt.plot(time, float(result), '-o', color='r', ms=2.5)
            plt.pause(0.01)

        plt.pause(10000)

    # MODIFIED 11/21/16 NEW:
    def _validate_variable(self, variable, context=None):
        """Insures that input to DDM is a single value.
        Remove when MULTIPROCESS DDM is implemented.
        """
        if not isinstance(variable, numbers.Number) and len(variable) > 1:
            raise DDMError("Input to DDM ({}) must have only a single numeric item".format(variable))
        super()._validate_variable(variable=variable, context=context)
    # MODIFIED 11/21/16 END


    def _validate_params(self, request_set, target_set=None, context=None):

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)
        functions = {BogaczEtAl, NavarroAndFuss, Integrator}
        if not target_set[FUNCTION] in functions:
            function_names = list (function.componentName for function in functions)
            raise DDMError("{} param of {} must be one of the following functions: {}".
                           format(FUNCTION, self.name, function_names))
        if self.timeScale == TimeScale.TRIAL:
            if target_set[FUNCTION] == Integrator:
                raise DDMError("In TRIAL mode, the {} param of {} cannot be Integrator. Please choose an analytic "
                               "solution for the function param: BogaczEtAl or NavarroAndFuss.".
                               format(FUNCTION, self.name))
        else:
            if target_set[FUNCTION] != Integrator:
                raise DDMError("In TIME_STEP mode, the {} param of {} must be Integrator with DIFFUSION integration.".
                               format(FUNCTION, self.name))
            else:
                self.get_axes_function = Integrator(integration_type=DIFFUSION, rate=self.function_params['rate'],
                                                    noise=self.function_params['noise'], context='plot').function
                self.plot_function = Integrator(integration_type=DIFFUSION, rate=self.function_params['rate'],
                                                noise=self.function_params['noise'], context='plot').function

        if not isinstance(target_set[FUNCTION], NavarroAndFuss):
            # OUTPUT_STATES is a list, so need to delete the first, so that the index doesn't go out of range
            del target_set[OUTPUT_STATES][DDM_Output.RT_CORRECT_VARIANCE.value]
            del target_set[OUTPUT_STATES][DDM_Output.RT_CORRECT_MEAN.value]

        try:
            threshold = target_set[FUNCTION_PARAMS][THRESHOLD]
        except KeyError:
            pass
        else:
            if isinstance(threshold, tuple):
                threshold = threshold[0]
            if not threshold >= 0:
                raise DDMError("{} param of {} ({}) must be >= zero".
                               format(THRESHOLD, self.name, threshold))



    def _instantiate_attributes_before_function(self, context=None):
        """Delete params not in use, call super.instantiate_execute_method
        """

        super()._instantiate_attributes_before_function(context=context)

    # def _instantiate_attributes_after_function(self, context=None):
    #
    #     super()._instantiate_attributes_after_function(context=context)
    #
    #     if not isinstance(self.paramsCurrent[FUNCTION], NavarroAndFuss):
    #         # OUTPUT_STATES is a list, so need to delete the first, so that the index doesn't go out of range
    #         del self.outputStates[DDM_RT_CORRECT_MEAN]
    #         del self.outputStates[DDM_RT_CORRECT_VARIANCE]

    def _execute(self,
                variable=None,
                runtime_params=None,
                clock=CentralClock,
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
        - time_scale (TimeScale): specifies "temporal granularity" with which mechanism is executed
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

        # PLACEHOLDER for a time_step_size parameter when time_step_mode/scheduling is implemented: 
        time_step_size = 1.0

        if variable is None or np.isnan(variable):
            # IMPLEMENT: MULTIPROCESS DDM:  ??NEED TO DEAL WITH PARTIAL NANS
            variable = self.variableInstanceDefault


        # EXECUTE INTEGRATOR SOLUTION (TIME_STEP TIME SCALE) -----------------------------------------------------
        if self.timeScale == TimeScale.TIME_STEP:
            if self.function_params['integration_type'] == 'diffusion':
                if (self.plot_threshold != None) and (INITIALIZING not in context):
                    import matplotlib.pyplot as plt
                    plt.ion()
                    axes = plt.gca()
                    axes.set_ylim([-1.25 * self.plot_threshold, 1.25 * self.plot_threshold])
                    plt.axhline(y=self.plot_threshold, linewidth=1, color='k', linestyle='dashed')
                    plt.axhline(y=-self.plot_threshold, linewidth=1, color='k', linestyle='dashed')
                    plt.plot()

                    result = 0
                    time = 0
                    while abs(result) < self.plot_threshold:
                        time += 1
                        result = self.function(context=context)
                        plt.plot(time, float(result), '-o', color='r', ms=5)
                        plt.pause(0.05)

                    plt.pause(5)

                else:
                    result = self.function(context=context)

                return np.array([result,[0.0],[0.0],[0.0]])
            else:
                raise MechanismError("Invalid integration_type: '{}'. For the DDM mechanism, integration_type must be set"
                                     " to 'DIFFUSION'".format(self.function_params['integration_type']))

        # EXECUTE ANALYTIC SOLUTION (TRIAL TIME SCALE) -----------------------------------------------------------
        elif self.timeScale == TimeScale.TRIAL:

            # # Get length of self.outputValue from OUTPUT_STATES
            # # Note: use paramsCurrent here (instead of outputStates), as during initialization the execute method
            # #       is run (to evaluate self.outputValue) before outputStates have been instantiated
            # self.outputValue = [None] * len(self.paramsCurrent[OUTPUT_STATES])

            # # TEST PRINT:
            # print ("\nDDM RUN")
            # print ("stimulus: {}".format(self.inputState.value))
            # print ("control signal: {}\n".format(self.parameterStates[DRIFT_RATE].value))

            # - convolve inputState.value (signal) w/ driftRate param value (attentional contribution to the process)
            # MODIFIED 11/21/16 OLD:
            # drift_rate = float((self.inputState.value * self.parameterStates[DRIFT_RATE].value))
            # # MODIFIED 11/21/16 NEW:
            drift_rate = float((self.variable * self.parameterStates[DRIFT_RATE].value))

            # MODIFIED 11/21/16 END

            starting_point = float(self.parameterStates[STARTING_POINT].value)
            threshold = float(self.parameterStates[THRESHOLD].value)
            noise = float(self.parameterStates[NOISE].value)
            t0 = float(self.parameterStates[NON_DECISION_TIME].value)

            result = self.function(params={DRIFT_RATE:drift_rate,
                                           STARTING_POINT:starting_point,
                                           THRESHOLD:threshold,
                                           NOISE:noise,
                                           NON_DECISION_TIME:t0})

            # # MODIFIED 12/7/16 OLD:
            # # Assign outputValue
            # if isinstance(self.function.__self__, BogaczEtAl):
            #     self.outputValue[DDM_Output.RESPONSE_TIME.value], self.outputValue[DDM_Output.P_LOWER_MEAN.value] = result
            #     self.outputValue[DDM_Output.P_UPPER_MEAN.value] = 1 - self.outputValue[DDM_Output.P_LOWER_MEAN.value]
            #
            # elif isinstance(self.function.__self__, NavarroAndFuss):
            #     self.outputValue[DDM_Output.RESPONSE_TIME.value] = result[NF_Results.MEAN_DT.value]
            #     self.outputValue[DDM_Output.P_LOWER_MEAN.value] = result[NF_Results.MEAN_ER.value]
            #     self.outputValue[DDM_Output.P_UPPER_MEAN.value] = 1 - result[NF_Results.MEAN_ER.value]
            #     self.outputValue[DDM_Output.RT_CORRECT_MEAN.value] = result[NF_Results.MEAN_CORRECT_RT.value]
            #     self.outputValue[DDM_Output.RT_CORRECT_VARIANCE.value] = result[NF_Results.MEAN_CORRECT_VARIANCE.value]
            #     # CORRECT_RT_SKEW = results[DDMResults.MEAN_CORRECT_SKEW_RT.value]
            #
            # # Convert ER to decision variable:
            # if random() < self.outputValue[DDM_Output.P_LOWER_MEAN.value]:
            #     self.outputValue[DDM_Output.DECISION_VARIABLE.value] = np.atleast_1d(-1 * threshold)
            # else:
            #     self.outputValue[DDM_Output.DECISION_VARIABLE.value] = threshold
            #
            # return self.outputValue
            # MODIFIED 12/7/16 NEW:
            if isinstance(self.function.__self__, BogaczEtAl):
                return_value = np.array([[0.0],[0.0],[0.0],[0.0]])
                return_value[DDM_Output.RESPONSE_TIME.value], return_value[DDM_Output.P_LOWER_MEAN.value] = result
                return_value[DDM_Output.P_UPPER_MEAN.value] = 1 - return_value[DDM_Output.P_LOWER_MEAN.value]

            elif isinstance(self.function.__self__, NavarroAndFuss):
                return_value = np.array([[0],[0],[0],[0],[0],[0]])
                return_value[DDM_Output.RESPONSE_TIME.value] = result[NF_Results.MEAN_DT.value]
                return_value[DDM_Output.P_LOWER_MEAN.value] = result[NF_Results.MEAN_ER.value]
                return_value[DDM_Output.P_UPPER_MEAN.value] = 1 - result[NF_Results.MEAN_ER.value]
                return_value[DDM_Output.RT_CORRECT_MEAN.value] = result[NF_Results.MEAN_CORRECT_RT.value]
                return_value[DDM_Output.RT_CORRECT_VARIANCE.value] = result[NF_Results.MEAN_CORRECT_VARIANCE.value]
                # CORRECT_RT_SKEW = results[DDMResults.MEAN_CORRECT_SKEW_RT.value]

            # Convert ER to decision variable:
            if random() < return_value[DDM_Output.P_LOWER_MEAN.value]:
                return_value[DDM_Output.DECISION_VARIABLE.value] = np.atleast_1d(-1 * threshold)
            else:
                return_value[DDM_Output.DECISION_VARIABLE.value] = threshold

            return return_value
            # MODIFIED 12/7/16 END

        else:
            raise MechanismError("time_scale not specified for DDM")

    # def _out_update(self, particle, drift, noise, time_step_size, decay):
    #     ''' Single update for OU (special case l=0 is DDM)'''
    #     return particle + time_step_size * (decay * particle + drift) + random.normal(0, noise) * sqrt(time_step_size)


    # def _ddm_update(self, particle, a, s, dt):
    #     return self._out_update(particle, a, s, dt, decay=0)


    # def _ddm_rt(self, x0, t0, a, s, z, dt):
    #     samps = 0
    #     particle = x0
    #     while abs(particle) < z:
    #         samps = samps + 1
    #         particle = self._out_update(particle, a, s, dt, decay=0)
    #     # return -rt for errors as per convention
    #     return (samps * dt + t0) if particle > 0 else -(samps * dt + t0)

    # def _ddm_distr(self, n, x0, t0, a, s, z, dt):
    #     return np.fromiter((self._ddm_rt(x0, t0, a, s, z, dt) for i in range(n)), dtype='float64')


    # def terminate_function(self, context=None):
    #     """Terminate the process
    #     called by process.terminate() - MUST BE OVERRIDDEN BY SUBCLASS IMPLEMENTATION
    #     returns output
    #     Returns: value
    #     """
    #     # IMPLEMENTATION NOTE:  TBI when time_step is implemented for DDM