# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ***************************************************  DDM *************************************************************

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
The DDM Mechanism implements the "Drift Diffusion Model" (also know as the Diffusion Decision, Accumulation to Bound,
Linear Integrator, and Wiener Process First Passage Time Model [REFS]). This corresponds to a continuous version of
the sequential probability ratio test (SPRT [REF]), that is the statistically optimal procedure for two alternative
forced choice (TAFC) decision making ([REF]).  It can be executed analytically using one of two solutions (in `TRIAL`
mode), or integrated numerically (in `TIME_STEP` mode).

.. _DDM_Creation:

Creating a DDM Mechanism
-----------------------------
A DDM Mechanism can be instantiated directly by calling its constructor, or by using the `mechanism` function
and specifying DDM as its **mech_spec** argument.  The analytic solution used in `TRIAL` mode is selected
using the `function <DDM.function>` argument, which can be simply the name of a DDM function (first example below),
or a call to the function with arguments specifying its parameters (see `DDM_Execution` below for a description of
DDM function parameters)::
    my_DDM = DDM(function=BogaczEtAl)
    my_DDM = DDM(function=BogaczEtAl(drift_rate=0.2, threshold=1.0))

COMMENT:
.. _DDM_Input:
**Input**.  The `default_variable` argument specifies the default value to use as the stimulus component of the
:ref:`drift rate <DDM_Drift_Rate>` for the decision process.  It must be a single scalar value.
[TBI - MULTIPROCESS DDM - REPLACE ABOVE]
**Input**.  The ``default_variable`` argument specifies the default value to use as the stimulus component of the
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

The DDM Mechanism implements a general form of the decision process.  A DDM Mechanism has a single `InputState` the
`value <DDM.value>` of which is assigned to the `input <DDM.input>` specified by its
`execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` methods,
and that is used as the stimulus component of the `drift rate <DDM_Drift_Rate>` for the decision process.  The
decision process can be configured to execute in different modes.

The `function <DDM.function>` and `time_scale <DDM.time_scale>` parameters are the primary determinants of how the
decision process is executed, and what information is returned. The `function <DDM.function>` parameter specifies the
analytic solution to use when `time_scale <DDM.time_scale>` is  set to `TimeScale.TRIAL`
(see `Functions <DDM_Functions>` below); when `time_scale <DDM.time_scale>` set to  `TimeScale.TIME_STEP`, the
**function** argument must be assigned an `Integrator` Function with an
`integration_type <Integrator.integration_type>` of *DIFFUSION*, so that executing the DDM mechanism numerically
integrates the path of the decision variable (see `Execution <DDM_Execution>` below). The number of `output_states
<DDM.output_states>` is determined by the `function <DDM.function>` in use (see `DDM_Execution` and
`Standard OutputStates <DDM_Standard_OutputStates>`).

COMMENT:
[TBI - MULTIPROCESS DDM - REPLACE ABOVE]
The DDM mechanism implements a general form of the decision process.  A DDM mechanism assigns one **inputState** to
each item in the `default_variable` argument, corresponding to each of the decision processes implemented
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
the content of the outputStates is determined by the ``average_output_states`` argument.  If it is `True`,
then each outputState (and item of ``output_values``) contains a single value, which is the average of the output
values of that type over all of the processes run.  If ``average_output_states`` is :keyword:`False` (the default),
then the value of each ouputState is a 1d array, each element of which is the outcome of that type for the
corresponding decision process.
COMMENT

.. _DDM_Functions:

DDM Functions
~~~~~~~~~~~~~

The `function <DDM.function>` parameter can be used to select one of two analytic solutions (`BogaczEtAl` and
`NavarroAndFuss`) that are used when `time_scale <DDM.time_scale>` is set to `TimeScale.TRIAL`.  These both return
the `decision variable <DECISION_VARIABLE>` for a given trial, as well as an expected
`mean response time <RESPONSE_TIME>`, `accuracy <PROBABILITY_UPPER_THRESHOLD>` and
 `error rate <PROBABILITY_LOWER_THRESHOLD>`, while `NavarroAndFuss` also returns expected values for
`mean correct response time <RT_CORRECT_MEAN>` and `variance of correct response times <RT_CORRECT_VARIANCE>`.
When `time_scale <DDM.time_scale>` is set to `TimeScale.TIME_STEP`, the `function <DDM.function>` parameter must be set
to 'Integrator' with an `integration_type <Integrator.integration_type>` of *DIFFUSION*.

.. _DDM_Parameters:

DDM Parameters
~~~~~~~~~~~~~~
COMMENT:
The DDM process uses the same set of parameters for all modes of execution.  These can be specified as arguments
for the functions used in TRIAL mode, or in a params dictionary assigned to the `params` argument,
using the keywords in the list below, as in the following example::
    my_DDM = DDM(function=BogaczEtAl(drift_rate=0.1),
                 params={DRIFT_RATE:(0.2, ControlProjection),
                         STARTING_POINT:-0.5},
                 time_scale=TimeScale.TRIAL)

.. note::  Parameters specified in the `params` argument (as in the example above) will be used for both
   `TRIAL` and `TIME_STEP` mode, since parameters specified in the `params` argument of a mechanism's constructor
   override corresponding ones specified as arguments of its `function <Mechanism_Base.function>`
   (see :doc:`COMPONENT`).  In the example above, this means that even if the `time_scale <DDM.time_scale>` parameter is
   set to `TimeScale.TRIAL`, the `drift_rate` of 0.2 will be used (rather than 0.1).  For parameters NOT specified
   as entries in the `params` dictionary, the value specified for those in the function will be used in both `TRIAL`
   and `TIME_STEP` mode.

The parameters for the DDM when `time_scale <DDM.time_scale>` is set to `TimeScale.TRIAL` and `function <DDM.function>`
is set to `BogaczEtAl` or `NavarroAndFuss` are:

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
  when ``time_scale <DDM.time_scale>`` is  TIME_STEP, it is added to the number of time steps
  taken to complete the decision process when reporting the response time.
COMMENT

All DDM parameter should be specified within the `function <DDM.function>` of the Mechanism. The examples below
demonstrate all of the possible parameters (see individual `Functions <Funtion>` for additional details).

`BogaczEtAl <BogaczEtAl>` ::

    my_DDM_BogaczEtAl = DDM(function=BogaczEtAl( drift_rate=3.0,
                                       starting_point=1.0,
                                       threshold=30.0,
                                       noise=1.5,
                                       t0 = 2.0),
                  time_scale= TimeScale.TRIAL,
                  name='MY_DDM_BogaczEtAl'
                  )

`NavarroAndFuss <NavarroAndFuss>` ::

    my_DDM_NavarroAndFuss = DDM(function=NavarroAndFuss( drift_rate=3.0,
                                       starting_point=1.0,
                                       threshold=30.0,
                                       noise=1.5,
                                       t0 = 2.0),
                  time_scale= TimeScale.TRIAL,
                  name='MY_DDM_NavarroAndFuss'
                  )

`Integrator <Integrator>` ::

    my_DDM_TimeStep = DDM(function=DriftDiffusionIntegrator(
                                      noise=0.5,
                                      initializer = 0.0,
                                      ),
                time_scale=TimeScale.TIME_STEP,
                name='My_DDM_TimeStep',
                )


.. _DDM_Results:

DDM Results
~~~~~~~~~~~

COMMENT:
[TBI - MULTIPROCESS DDM - REPLACE BELOW]
When a DDM mechanism is executed it computes the decision process, either analytically (in TRIAL mode)
or by step-wise integration (in TIME_STEP mode).  As noted above, if the input is a single value,
it computes a single DDM process.  If the input is a list or array, then multiple parallel DDM processes are executed,
with each element of the input used for the corresponding process.  All use the same set of parameters,
so the analytic solutions (used in TRIAL mode) for a given input will be the same; to implement processes in
this mode that use different parameters, a separate DDM mechanism should explicitly be created for each. In
TIME_STEP mode, the noise term will resolve to different values in each time step, so the integration
paths and outcomes for the same input value will vary. This can be used to generate distributions of the process for a
single set of parameters that are not subject to the analytic solution (e.g., for time-varying drift rates).

.. note::
   DDM handles "runtime" parameters (specified in a call to its
   :py:meth:`execute <Mechanism_Base.exeucte>` or :py:meth:`run <Mechanism_Base.run>` methods)
   differently than standard Components: runtime parameters are added to the mechanism's current value of the
   corresponding parameterState (rather than overriding it);  that is, they are combined additively with the value of
   any `ControlProjection` it receives to determine the parameter's value for that execution.  The parameterState's
   value is then restored to its original value (i.e., either its default value or the one assigned when it was
   created) for the next execution.

  ADD NOTE ABOUT INTERROGATION PROTOCOL, USING ``terminate_function``
  ADD NOTE ABOUT RELATIONSHIP OF RT TO time_steps TO t0 TO ms
COMMENT

****
MOVE MOST OF THIS TO STANDARD OUTPUT STATES, AND THEN SAY THAT ITS FUNCTION RETURENS THE VALUES DESCTIBED THERE,
UNDER THE FOLLOWING CONDITIONS, AND WITH THE FOLLOWING ASSIGNMENTS:
****

The `value <DDM.value>` of a DDM Mechanism has six items. The first two of these are always assigned, corresponding
to two `output_states <DDM.output_states>` that are assigned by default;  these represent the two results always
returned by the DDM's `function <DDM.function>`: `DECISION_VARIABLE <DDM_DECISION_VARIABLE>` and
`RESPONSE_TIME <DDM_RESPONSE_TIME>`. Other items of its `value <DDM.value>`, and entries in its
`output_states <DDM.output_states>` attribute, may also be assigned,  depending on whether it is executed in
`TRIAL` or `TIME_STEP` mode, and its `function <DDM.function>`, as described under `DDM_Execution` and
`standard OutputStates <DDM_Standard_OutputStates>`.  Unassigned items of `value <DDM.value>` have the value `None`.

.. _DDM_Execution:

Execution
---------

When a DDM mechanism is executed, it computes the decision process either analytically (in `TRIAL` mode) or by
step-wise integration using the `Euler method <https://en.wikipedia.org/wiki/Euler_method>`_ (in `TIME_STEP` mode).
It `function <DDM>function>` always returns its *DECISION_VARIABLE* and *RESPONSE_TIME* results
(see `DDM_Standard_OutputStates`), and assigns these as the first two items of its `value <DDM.value>` attribute.
If it is executed in `TRIAL` mode, it also returns its *PROBABILITY_UPPER_THRESHOLD* and *PROBABILITY_LOWER_THRESHOLD*
results and assigns these as the third and fourth items of its `value <DDM.value>` attribute.  If it is executed
in `TRIAL` mode using `NavarroAndFuss` as its `function <DDM.function>`, it also returns its *RT_CORRECT_MEAN* and
*RT_CORRECT_VARIANCE* results and assigns these as the fifth and sixth items of its `value <DDM.value>` attribute.

.. _DDM_Class_Reference:

Class Reference
---------------
"""
import logging
import numbers
import random

import numpy as np
import typecheck as tc

from PsyNeuLink.Components.Component import method_type
from PsyNeuLink.Components.Functions.Function import BogaczEtAl, DriftDiffusionIntegrator, Integrator, NF_Results, NavarroAndFuss, STARTING_POINT, THRESHOLD
from PsyNeuLink.Components.Mechanisms.Mechanism import MechanismError, Mechanism_Base
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import ProcessingMechanism_Base
from PsyNeuLink.Components.States.OutputState import SEQUENTIAL
from PsyNeuLink.Globals.Keywords import FUNCTION, FUNCTION_PARAMS, INITIALIZING, NAME, OUTPUT_STATES, TIME_SCALE, kwPreferenceSetName
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set, kpReportOutputPref
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceEntry, PreferenceLevel
from PsyNeuLink.Scheduling.TimeScale import CentralClock, TimeScale

logger = logging.getLogger(__name__)

DECISION_VARIABLE='DECISION_VARIABLE'
RESPONSE_TIME = 'RESPONSE_TIME'
PROBABILITY_UPPER_THRESHOLD = 'PROBABILITY_UPPER_THRESHOLD'
PROBABILITY_LOWER_THRESHOLD = 'PROBABILITY_LOWER_THRESHOLD'
RT_CORRECT_MEAN = 'RT_CORRECT_MEAN'  # NavarroAnd Fuss only
RT_CORRECT_VARIANCE = 'RT_CORRECT_VARIANCE'  # NavarroAnd Fuss only

DDM_standard_output_states = [{NAME: DECISION_VARIABLE,},           # Upper or lower threshold in TRIAL mode
                              {NAME: RESPONSE_TIME},                # TIME_STEP within TRIAL in TIME_STEP mode
                              {NAME: PROBABILITY_UPPER_THRESHOLD},  # Accuracy (TRIAL mode only)
                              {NAME: PROBABILITY_LOWER_THRESHOLD},  # Error rate (TRIAL mode only)
                              {NAME: RT_CORRECT_MEAN},              # (NavarroAndFuss only)
                              {NAME: RT_CORRECT_VARIANCE}]          # (NavarroAndFuss only)

# This is a convenience class that provides list of standard_output_state names in IDE
class DDM_OUTPUT():
    """
    .. _DDM_Standard_OutputStates:

    `Standard OutputStates <OutputState_Standard>` for `DDM`:

    .. _DDM_DECISION_VARIABLE:

    *DECISION_VARIABLE* : float
      • in `TRIAL` mode: the value of the threshold crossed by the decision variable on the current TRIAL (which is
        either the value of the Mechanism's `threshold <DDM.threshold>` attribute or its negative); \n
      • in `TIME_STEP` mode: the value of the decision variable at the current TIME_STEP of execution. \n
      Corresponds to the first item of the Mechanism's `value <DDM.value>`.

    .. _DDM_RESPONSE_TIME:

    *RESPONSE_TIME* : float
      • in `TRIAL` mode: mean time (in seconds) for the decision variable to reach the positive or negative value of
        `threshold <DDM.threshold>` as estimated by the analytic solution specified in `function <DDM.function>`); \n
      • in `TIME_STEP` mode: the number of `TIME_STEP` that have occurred since the Mechanism began to execute in
        the current `TRIAL` or, if it has reached the positive or negative value of `threshold <DDM.threshold>`,
        the `TIME_STEP` at which that occurred. \n
      Corresponds to the second item of the Mechanism's `value <DDM.value>`.

    .. _DDM_PROBABILITY_UPPER_THRESHOLD:

    *PROBABILITY_UPPER_THRESHOLD* : float
      • in `TRIAL` mode: the probability of the decision variable reaching the positive value of
        `threshold <DDM.threshold>`  as estimated by the analytic solution specified in `function <DDM.function>`);
        often, by convention, the positive (upper) threshold is associated with the correct response, in which case
        *PROBABILITY_UPPER_THRESHOLD* corresponds to the accuracy of the decision process. \n
      • in `TIME_STEP` mode, `None`.
      Corresponds to the third item of the Mechanism's `value <DDM.value>`.

        COMMENT:
          [TBI:]
              In `TIME_STEP` mode, if execution has completed, this is a binary value indicating whether the decision
              process reached the upper (positive) threshold. If execution was interrupted
              (using :py:meth:`terminate_function  <DDM.terminate_function>`, sometimes referred to as the
              :ref:`interrogation protocol <LINK>`, then the value corresponds to the current likelihood that the upper
              threshold would have been reached.
        COMMENT

    .. _DDM_PROBABILITY_LOWER_THRESHOLD:

    *PROBABILITY_LOWER_THRESHOLD* : float
      • in `TRIAL` mode: the probability of the decision variable reaching the negative value of
        `threshold <DDM.threshold>`  as estimated by the analytic solution specified in `function <DDM.function>`);
        often, by convention, the negative (lower) threshold is associated with an error response, in which case
        *PROBABILITY_LOWER_THRESHOLD* corresponds to the error rate of the decision process; \n
      • in `TIME_STEP` mode: `None`.
      Corresponds to the fourth item of the Mechanism's `value <DDM.value>`.

        COMMENT:
          [TBI:]
              In `TIME_STEP` mode, if execution has completed, this is a binary value indicating whether the decision
              process reached the lower (negative) threshold. If execution was interrupted
              (using :py:method:`terminate_method <DDM.terminate_function>`, sometimes referred to as the
              :ref:`interrogation protocol <LINK>`), then the value corresponds to the current likelihood that the lower
              threshold would have been reached.
        COMMENT

    .. _DDM_RT_CORRECT_MEAN:

    *RT_CORRECT_MEAN* : float
      (only applicable if `function <DDM.function>` is `NavarroAndFuss`) \n
      •  in `TRIAL` mode:  the mean decision time (in seconds) for responses in which the decision variable reached
        the positive value of `threshold <DDM.threshold>` as estimated by the `NavarroAndFuss` analytic solution; \n
      • in `TIME_STEP` mode: `None`.
      Corresponds to the fifth item of the Mechanism's `value <DDM.value>`.

    .. _DDM_RT_CORRECT_VARIANCE:

    *RT_CORRECT_MEAN* : float
      (only applicable if `function <DDM.function>` is `NavarroAndFuss`) \n
      • in `TRIAL` mode:  the variance of the decision time for responses in which the decision variable reached
        the positive value of `threshold <DDM.threshold>` as estimated by the `NavarroAndFuss` analytic solution; \n
      • in `TIME_STEP` mode: `None`.
      Corresponds to the sixth item of the Mechanism's `value <DDM.value>`.

    """
    DECISION_VARIABLE=DECISION_VARIABLE
    RESPONSE_TIME=RESPONSE_TIME
    PROBABILITY_UPPER_THRESHOLD=PROBABILITY_UPPER_THRESHOLD
    PROBABILITY_LOWER_THRESHOLD=PROBABILITY_LOWER_THRESHOLD
    RT_CORRECT_MEAN=RT_CORRECT_MEAN
    RT_CORRECT_VARIANCE=RT_CORRECT_VARIANCE
# THE FOLLOWING WOULD HAVE BEEN NICE, BUT IDE DOESN'T EXECUTE IT, SO NAMES DON'T SHOW UP
# for item in [item[NAME] for item in DDM_standard_output_states]:
#     setattr(DDM_OUTPUT.__class__, item, item)


class DDMError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class DDM(ProcessingMechanism_Base):
    # DOCUMENT:   COMBINE WITH INITIALIZATION WITH PARAMETERS
    #             ADD INFO ABOUT B VS. N&F
    #             ADD _instantiate_output_states TO INSTANCE METHODS, AND EXPLAIN RE: NUM OUTPUT VALUES FOR B VS. N&F
    """
    DDM(                       \
    default_variable=None,  \
    size=None,                 \
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
    default_variable : value, list or np.ndarray : default FUNCTION_PARAMS[STARTING_POINT]
        the input to the mechanism to use if none is provided in a call to its
        :py:data:`execute <Mechanism_Base.execute>` or :py:data:`run <Mechanism_Base.run>` methods;
        also serves as a template to specify the length of `variable <DDM.variable>` for `function <DDM.function>`,
        and the primary outputState of the mechanism (see :ref:`Input` <DDM_Creation>` for how an input with a length
        of greater than 1 is handled).
    size : int, list or np.ndarray of ints
        specifies default_variable as array(s) of zeros if **default_variable** is not passed as an argument;
        if **default_variable** is specified, it takes precedence over the specification of **size**.
    function : IntegratorFunction : default BogaczEtAl
        specifies the analytic solution to use for the decision process if `time_scale <DDM.time_scale>` is set to
        `TimeScale.TRIAL`; can be `BogaczEtAl` or `NavarroAndFuss` (note:  the latter requires that the MatLab engine
        is installed). If `time_scale <DDM.time_scale>` is set to `TimeScale.TIME_STEP`, `function <DDM.function>` must
        be `DriftDiffusionIntegrator`, and the mechanism
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
        must be `DriftDiffusionIntegrator`, and the mechanism
        will return the result of one time step.
    function_params : Dict[str, value]
        contains one entry for each parameter of the mechanism's function.
        The key of each entry is the name of (keyword for) a function parameter, and its value is the parameter's value.
    value : 2d np.array[array(float64),array(float64),array(float64),array(float64)]
        result of executing DDM `function <DDM.function>`;  see `DDM_Execution` for a description of item assignments,
        and `DDM Standard OutputStates <DDM_Standard_OutputStates>` for a description of their values.
    output_states : *ContentAddressableList[OutputState]* : default [`DECISION_VARIABLE <DDM_DECISION_VARIABLE>`, \
    `RESPONSE_TIME <DDM_RESPONSE_TIME>`]
        the OutputStates assigned to the Mechanism.
    output_values : List[array(float64),array(float64),array(float64),array(float64)]
        each item is the value of the corresponding OutputState in `output_states <DDM.output_states>`.
    time_scale : TimeScale : default TimeScale.TRIAL
        determines the `TimeScale` at which the decision process is executed.
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
        OUTPUT_STATES: None})

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 # function:tc.enum(type(BogaczEtAl), type(NavarroAndFuss))=BogaczEtAl(drift_rate=1.0,
                 function=BogaczEtAl(drift_rate=1.0,
                                     starting_point=0.0,
                                     threshold=1.0,
                                     noise=0.5,
                                     t0=.200),
                 output_states:tc.optional(tc.any(list, dict))=[DECISION_VARIABLE, RESPONSE_TIME],
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 name=None,
                 # prefs:tc.optional(ComponentPreferenceSet)=None,
                 prefs: is_pref_set = None,
                 thresh=0,
                 context=componentType + INITIALIZING
    ):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function,
                                                  output_states=output_states,
                                                  time_scale=time_scale,
                                                  params=params)

        self.variableClassDefault = self.paramClassDefaults[FUNCTION_PARAMS][STARTING_POINT]

        # IMPLEMENTATION NOTE: this manner of setting default_variable works but is idiosyncratic
        # compared to other mechanisms: see TransferMechanism.py __init__ function for a more normal example.
        if default_variable is None and size is None:
            try:
                default_variable = params[FUNCTION_PARAMS][STARTING_POINT]
            except:
                default_variable = 0.0

        # # Conflict with above
        # self.size = size
        self.threshold = thresh

        from PsyNeuLink.Components.States.OutputState import StandardOutputStates
        self.standard_output_states = StandardOutputStates(self, DDM_standard_output_states, SEQUENTIAL)

        super(DDM, self).__init__(variable=default_variable,
                                  output_states=output_states,
                                  params=params,
                                  name=name,
                                  prefs=prefs,
                                  size=size,
                                  # context=context)
                                  context=self)

        # # TEST PRINT
        # print("\n{} user_params:".format(self.name))
        # for param in self.user_params.keys():
        #     print("\t{}: {}".format(param, self.user_params[param]))


    def plot(self, stimulus=1.0, threshold=10.0):
        """
        Generate a dynamic plot of the DDM integrating over time towards a threshold.

        NOTE: plot is only available in `TIME_STEP` mode (with the Integrator function).

        Arguments
        ---------

        threshold: float: default 10.0
             specify the threshold at which the DDM will stop integrating

        Returns
        -------
        mechanism's function plot : Matplotlib window
            Matplotlib window of the mechanism's function plotting dynamically over time with specified parameters
            towards a specified threshold

        """
        import matplotlib.pyplot as plt
        import time
        plt.ion()

        # set initial values and threshold
        time_step = [0]
        position = [float(self.variable)]
        self.variable = stimulus

        # execute the mechanism once to begin the loop
        result_check = self.plot_function(self.variable, context="plot")[0][0]

        # continue executing the ddm until its value exceeds the threshold
        while abs(result_check) < threshold:
            time_step.append(time_step[-1] + 1)
            position.append(result_check)
            result_check = self.plot_function(self.variable, context="plot")[0][0]

        # add the ddm's final position to the list of positions
        time_step.append(time_step[-1] + 1)
        position.append(result_check)

        figure, ax = plt.subplots(1, 1)
        lines, = ax.plot([], [], 'o')
        ax.set_xlim(0, time_step[-1])
        ax.set_ylim(-threshold, threshold)
        ax.grid()
        xdata = []
        ydata = []

        # add each of the position values to the plot one at a time
        for t in range(time_step[-1]):
            xdata.append(t)
            ydata.append(position[t])
            lines.set_xdata(xdata)
            lines.set_ydata(ydata)
            figure.canvas.draw()
            # number of seconds to wait before next point is plotted
            time.sleep(.1)


        #
        # import matplotlib.pyplot as plt
        # plt.ion()
        #
        # # # Select a random seed to ensure that the test run will be the same as the real run
        # seed_value = np.random.randint(0, 100)
        # np.random.seed(seed_value)
        # self.variable = stimulus
        #
        # result_check = 0
        # time_check = 0
        #
        # while abs(result_check) < threshold:
        #     time_check += 1
        #     result_check = self.get_axes_function(self.variable, context='plot')
        #
        # # Re-set random seed for the real run
        # np.random.seed(seed_value)
        # axes = plt.gca()
        # axes.set_xlim([0, time_check])
        # axes.set_xlabel("Time Step", weight="heavy", size="large")
        # axes.set_ylim([-1.25 * threshold, 1.25 * threshold])
        # axes.set_ylabel("Position", weight="heavy", size="large")
        # plt.axhline(y=threshold, linewidth=1, color='k', linestyle='dashed')
        # plt.axhline(y=-threshold, linewidth=1, color='k', linestyle='dashed')
        # plt.plot()
        #
        # result = 0
        # time = 0
        # while abs(result) < threshold:
        #     time += 1
        #     result = self.plot_function(self.variable, context='plot')
        #     plt.plot(time, float(result), '-o', color='r', ms=2.5)
        #     plt.pause(0.01)
        #
        # plt.pause(10000)

    # MODIFIED 11/21/16 NEW:
    def _validate_variable(self, variable, context=None):
        """Ensures that input to DDM is a single value.
        Remove when MULTIPROCESS DDM is implemented.
        """
        # this test may become obsolete when size is moved to Component.py
        if len(variable) > 1:
            raise DDMError("Length of input to DDM ({}) is greater than 1, implying there are multiple "
                           "input states, which is currently not supported in DDM, but may be supported"
                           " in the future under a multi-process DDM. Please use a single numeric "
                           "item as the default_variable, or use size = 1.".format(variable))
        # MODIFIED 6/28/17 (CW): changed len(variable) > 1 to len(variable[0]) > 1
        if not isinstance(variable, numbers.Number) and len(variable[0]) > 1:
            raise DDMError("Input to DDM ({}) must have only a single numeric item".format(variable))
        super()._validate_variable(variable=variable, context=context)

    # MODIFIED 11/21/16 END


    def _validate_params(self, request_set, target_set=None, context=None):

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)
        functions = {BogaczEtAl, NavarroAndFuss, DriftDiffusionIntegrator}

        if FUNCTION in target_set:
            # If target_set[FUNCTION] is a method of a Function (e.g., being assigned in _instantiate_function),
            #   get the Function to which it belongs
            function = target_set[FUNCTION]
            if isinstance(function, method_type):
                function = function.__self__.__class__

            if not function in functions:
                function_names = [function.componentName for function in functions]
                raise DDMError("{} param of {} must be one of the following functions: {}".
                               format(FUNCTION, self.name, function_names))

            if self.timeScale == TimeScale.TRIAL:
                if function == Integrator:
                    raise DDMError("In TRIAL mode, the {} param of {} cannot be Integrator. Please choose an analytic "
                                   "solution for the function param: BogaczEtAl or NavarroAndFuss.".
                                   format(FUNCTION, self.name))
            else:
                if function != DriftDiffusionIntegrator:
                    raise DDMError("In TIME_STEP mode, the {} param of {} "
                                   "must be DriftDiffusionIntegrator.".
                                   format(FUNCTION, self.name))
                else:
                    self.get_axes_function = DriftDiffusionIntegrator(rate=self.function_params['rate'],
                                                        noise=self.function_params['noise'], context='plot').function
                    self.plot_function = DriftDiffusionIntegrator(rate=self.function_params['rate'],
                                                    noise=self.function_params['noise'], context='plot').function

            if not isinstance(function, NavarroAndFuss) and OUTPUT_STATES in target_set:
                # OUTPUT_STATES is a list, so need to delete the first, so that the index doesn't go out of range
                # if DDM_OUTPUT_INDEX.RT_CORRECT_VARIANCE.value in target_set[OUTPUT_STATES]:
                #     del target_set[OUTPUT_STATES][DDM_OUTPUT_INDEX.RT_CORRECT_VARIANCE.value]
                # if DDM_OUTPUT_INDEX.RT_CORRECT_VARIANCE.value in target_set[OUTPUT_STATES]:
                #     del target_set[OUTPUT_STATES][DDM_OUTPUT_INDEX.RT_CORRECT_MEAN.value]
                if self.RT_CORRECT_MEAN in target_set[OUTPUT_STATES]:
                    del target_set[OUTPUT_STATES][self.RT_CORRECT_MEAN_INDEX]
                if self.RT_CORRECT_VARIANCE in target_set[OUTPUT_STATES]:
                    del target_set[OUTPUT_STATES][self.RT_CORRECT_MEAN_INDEX]

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

    def _execute(self,
                 variable=None,
                 runtime_params=None,
                 clock=CentralClock,
                 time_scale=TimeScale.TRIAL,
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
        variable (float): set to self.value (= self.input_value)
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

        # PLACEHOLDER for a time_step_size parameter when time_step_mode/Scheduling is implemented:
        time_step_size = 1.0

        if variable is None or np.isnan(variable):
            # IMPLEMENT: MULTIPROCESS DDM:  ??NEED TO DEAL WITH PARTIAL NANS
            variable = self.variableInstanceDefault

        # EXECUTE INTEGRATOR SOLUTION (TIME_STEP TIME SCALE) -----------------------------------------------------
        if self.timeScale == TimeScale.TIME_STEP:

            result = self.function(self.variable, context=context)
            if INITIALIZING not in context:
                logger.info('{0} {1} is at {2}'.format(type(self).__name__, self.name, result))
            if abs(result) >= self.threshold:
                logger.info('{0} {1} has reached threshold {2}'.format(type(self).__name__, self.name, self.threshold))
                self.is_finished = True

            return np.array([result, [0.0], [0.0], [0.0]])


        # EXECUTE ANALYTIC SOLUTION (TRIAL TIME SCALE) -----------------------------------------------------------
        elif self.timeScale == TimeScale.TRIAL:

            result = self.function(variable=self.variable,
                                   params=runtime_params,
                                   context=context)

            if isinstance(self.function.__self__, BogaczEtAl):
                return_value = np.array([[0.0], [0.0], [0.0], [0.0]])
                return_value[self.RESPONSE_TIME_INDEX], return_value[self.PROBABILITY_LOWER_THRESHOLD_INDEX] = result
                return_value[self.PROBABILITY_UPPER_THRESHOLD_INDEX] = \
                                                               1 - return_value[self.PROBABILITY_LOWER_THRESHOLD_INDEX]

            elif isinstance(self.function.__self__, NavarroAndFuss):
                return_value = np.array([[0], [0], [0], [0], [0], [0]])
                return_value[self.RESPONSE_TIME_INDEX] = result[NF_Results.MEAN_DT.value]
                return_value[self.PROBABILITY_LOWER_THRESHOLD_INDEX] = result[NF_Results.MEAN_ER.value]
                return_value[self.PROBABILITY_UPPER_THRESHOLD_INDEX] = 1 - result[NF_Results.MEAN_ER.value]
                return_value[self.RT_CORRECT_MEAN_INDEX] = result[NF_Results.MEAN_CORRECT_RT.value]
                return_value[self.RT_CORRECT_VARIANCE_INDEX]= result[NF_Results.MEAN_CORRECT_VARIANCE.value]
                # CORRECT_RT_SKEW = results[DDMResults.MEAN_CORRECT_SKEW_RT.value]

            else:
                raise DDMError("PROGRAM ERROR: Unrecognized analytic fuction ({}) for DDM".
                               format(self.function.__self__))

            # Convert ER to decision variable:
            threshold = float(self.function_object.threshold)
            if random.random() < return_value[self.PROBABILITY_LOWER_THRESHOLD_INDEX]:
                return_value[self.DECISION_VARIABLE_INDEX] = np.atleast_1d(-1 * threshold)
            else:
                return_value[self.DECISION_VARIABLE_INDEX] = threshold

            return return_value

        else:
            raise MechanismError("PROGRAM ERROR: unrecognized or unspecified time_scale ({}) for DDM".
                                 format(self.timeScale))

            # def _out_update(self, particle, drift, noise, time_step_size, decay):
            #     ''' Single update for OU (special case l=0 is DDM)'''
            #     return particle + time_step_size * (decay * particle + drift)
            #                     + random.normal(0, noise) * sqrt(time_step_size)


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
