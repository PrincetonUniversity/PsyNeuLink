# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ***************************************************  DDM *************************************************************

"""
..
Sections
--------
  * `DDM_Overview`
  * `DDM_Creation`
  * `DDM_Execution`
  * `DDM_Class_Reference`

.. _DDM_Overview:

Overview
--------
The DDM Mechanism implements the "Drift Diffusion Model" (also know as the Diffusion Decision, Accumulation to Bound,
Linear IntegratorFunction, and Wiener Process First Passage Time Model [REFS]). This corresponds to a continuous version of
the sequential probability ratio test (SPRT [REF]), that is the statistically optimal procedure for two alternative
forced choice (TAFC) decision making ([REF]).

The DDM Mechanism may be constructed with a choice of several functions that fall into to general categories: analytic
solutions and path integration (see `DDM_Modes` below for more about these options.)

.. _DDM_Creation:

Creating a DDM Mechanism
-----------------------------
A DDM Mechanism can be instantiated directly by calling its constructor, or by using the `mechanism` command and
specifying DDM as its **mech_spec** argument.  The model implementation is selected using the `function <DDM.function>`
argument. The function selection can be simply the name of a DDM function::

    >>> import psyneulink as pnl
    >>> my_DDM = pnl.DDM(function=pnl.DriftDiffusionAnalytical)

or a call to the function with arguments specifying its parameters::

    >>> my_DDM = pnl.DDM(function=pnl.DriftDiffusionAnalytical(drift_rate=0.2, threshold=1.0))


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

The DDM Mechanism implements a general form of the decision process.

.. _DDM_Input:

*Input*
~~~~~~~

The input to the `function <DDM_Function>` of a DDM Mechanism is always a scalar, irrespective of `type of function
<DDM_Modes>` that is used.  Accordingly, the default `InputState` for a DDM takes a single scalar value as its input,
that represents the stimulus for the decision process.  However, this can be configured using the **input_format**
argument of the DDM's consructor, to accomodate use of the DDM with other Mechanisms that generate a stimulus array
(e.g., representing the stimuli associated with each of the two choices). By default, the **input_format** is
*SCALAR*.  However, if it is specified as *ARRAY*, the DDM's InputState is configured to accept a 1d 2-item vector,
and to use `Reduce` as its Function, which subtracts the 2nd element of the vector from the 1st, and provides this as
the input to the DDM's `function <DDM.function>`.  If *ARRAY* is specified, two  `Standard OutputStates
<DDM_Standard_OutputStates>` are added to the DDM, that allow the result of the decision process to be represented
as an array corresponding to the input array (see `below <DDM_Custom_OutputStates>`).

COMMENT:
ADD EXAMPLE HERE
COMMENT

COMMENT
NOTE SURE WHAT THIS MEANS:
That parameter, along with all of the others for the DDM, must be assigned as
parameters of the DDM's `function <DDM.function>` (see examples under `DDM_Modes` below, and individual `Functions
<Function>` for additional details).
COMMENT

.. _DDM_Output:

*Output*
~~~~~~~~

The DDM Mechanism can generate two different types of results depending on which function is selected. When a
function representing an analytic solution is selected, the mechanism generates a single estimation for the process.
When the path integration function is selected, the mechanism carries out step-wise integration of the process; each
execution of the mechanism computes one step. (see `DDM_Modes` and `DDM_Execution` for additional details).

The `value <DDM.value>` of the DDM Mechanism may have up to ten items. The first two of these are always assigned, and
are represented by the DDM Mechanism's two default `output_states <DDM.output_states>`: `DECISION_VARIABLE
<DDM_DECISION_VARIABLE>` and `RESPONSE_TIME <DDM_RESPONSE_TIME>`.  Other `output_states <DDM.output_states>` may be
automatically assigned, depending on the `function <DDM.function>` that has been assigned to the DDM, as shown in the
table below:

+------------------------------------+---------------------------------------------------------+
|                                    |                     **Function**                        |
|                                    |                      *(type)*                           |
+                                    +----------------------------+----------------------------+
|                                    | `DriftDiffusionAnalytical` | `DriftDiffusionIntegrator` |
|                                    |   (`analytic               |   (`path integration)      |
| **OutputStates:**                  |   <DDM_Analytic_Mode>`)    |   <DDM_Integration_Mode>`) |
+------------------------------------+----------------------------+----------------------------+
| `DECISION_VARIABLE                 |                            |                            |
| <DDM_DECISION_VARIABLE>`           |       X                    |          X                 |
+------------------------------------+----------------------------+----------------------------+
| `RESPONSE_TIME                     |                            |                            |
| <DDM_RESPONSE_TIME>`               |       X                    |          X                 |
+------------------------------------+----------------------------+----------------------------+
| `PROBABILITY_UPPER_THRESHOLD       |                            |                            |
| <DDM_PROBABILITY_UPPER_THRESHOLD>` |       X                    |                            |
+------------------------------------+----------------------------+----------------------------+
| `PROBABILITY_LOWER_THRESHOLD       |                            |                            |
| <DDM_PROBABILITY_LOWER_THRESHOLD>` |       X                    |                            |
+------------------------------------+----------------------------+----------------------------+
| `RT_CORRECT_MEAN                   |                            |                            |
| <DDM_RT_CORRECT_MEAN>`             |       X                    |                            |
+------------------------------------+----------------------------+----------------------------+
| `RT_CORRECT_VARIANCE               |                            |                            |
| <DDM_RT_CORRECT_VARIANCE>`         |       X                    |                            |
+------------------------------------+----------------------------+----------------------------+
| `RT_CORRECT_SKEW                   |                            |                            |
| <DDM_RT_CORRECT_SKEW>`             |       X                    |                            |
+------------------------------------+----------------------------+----------------------------+
| `RT_INCORRECT_MEAN                 |                            |                            |
| <DDM_RT_INCORRECT_MEAN>`           |       X                    |                            |
+------------------------------------+----------------------------+----------------------------+
| `RT_INCORRECT_VARIANCE             |                            |                            |
| <DDM_RT_INCORRECT_VARIANCE>`       |       X                    |                            |
+------------------------------------+----------------------------+----------------------------+
| `RT_INCORRECT_SKEW                 |                            |                            |
| <DDM_RT_INCORRECT_SKEW>`           |       X                    |                            |
+------------------------------------+----------------------------+----------------------------+


.. _DDM_Custom_OutputStates:

The `output_states <DDM.output_states>` assigned to a DDM can be customized by specifying a list of the desired DDM
`Standard OutputStates <DDM_Standard_OutputStates>` in the **output_states** argument of its constructor, or the
*OUTPUT_STATES* entry of an `OutputState specification dictionary <OutputState_Specification_Dictionary>`.  This can
include two additional `Standard OutputStates <DDM_Standard_OutputStates>` for the DDM - `DECISION_VARIABLE_ARRAY
<DDM_OUTPUT.DDM_DECISION_VARIABLE_ARRAY>` and `SELECTED_INPUT_ARRAY <DDM_OUTPUT.DDM_SELECTED_INPUT_ARRAY>`,
that are available  if the *ARRAY* option is specified in its **input_format** argument (see `DDM_Input`).  As with
any Mechanism, `customized OutputStates <OutputState_Customization>` can also be created and assigned.

.. _DDM_Modes:

*DDM Function Types*
~~~~~~~~~~~~~~~~~~~~

.. _DDM_Analytic_Mode:

Analytic Solutions
^^^^^^^^^^^^^^^^^^

The Drift Diffusion Model `Functions <Function>` that calculate analytic solutions
[Bogacz et al (2006), Srivastava et al. (2016)] is `DriftDiffusionAnalytical <DriftDiffusionAnalytical>`
`function <DDM.function>`, the mechanism generates a single estimate of the outcome for the decision process (see
`DDM_Execution` for details). In addition to `DECISION_VARIABLE <DDM_DECISION_VARIABLE>` and
`RESPONSE_TIME <DDM_RESPONSE_TIME>`, the Function returns an accuracy value (represented in the
`PROBABILITY_UPPER_THRESHOLD <DDM_PROBABILITY_UPPER_THRESHOLD>` OutputState), and an error rate value (in the `PROBABILITY_LOWER_THRESHOLD <DDM_PROBABILITY_LOWER_THRESHOLD>`
OutputState, and moments (mean, variance, and skew) for conditional (correct\\positive or incorrect\\negative) response time distributions.
These are; the mean RT for correct responses  (`RT_CORRECT_MEAN <DDM_RT_CORRECT_MEAN>`, the RT variance for correct responses
(`RT_CORRECT_VARIANCE <DDM_RT_CORRECT_VARIANCE>`, the RT skew for correct responses (`RT_CORRECT_SKEW <DDM_RT_CORRECT_SKEW>`,
the mean RT for incorrect responses  (`RT_INCORRECT_MEAN <DDM_RT_INCORRECT_MEAN>`, the RT variance for incorrect
responses (`RT_INCORRECT_VARIANCE <DDM_RT_INCORRECT_VARIANCE>`, the RT skew for incorrect responses
(`RT_INCORRECT_SKEW <DDM_RT_INCORRECT_SKEW>`.

An example that illustrate all of the parameters is shown below:

`DriftDiffusionAnalytical <DriftDiffusionAnalytical>` Function::

    >>> my_DDM_DriftDiffusionAnalytical = pnl.DDM(
    ...     function=pnl.DriftDiffusionAnalytical(
    ...         drift_rate=0.08928,
    ...         starting_point=0.5,
    ...         threshold=0.2645,
    ...         noise=0.5,
    ...         t0=0.15
    ...     ),
    ...     name='my_DDM_DriftDiffusionAnalytical'
    ... )

.. _DDM_Integration_Mode:

Path Integration
^^^^^^^^^^^^^^^^

The Drift Diffusion Model `Function <Function>` that calculates a path integration is `DriftDiffusionIntegrator
<DriftDiffusionIntegrator>`. The DDM Mechanism uses the `Euler method <https://en.wikipedia.org/wiki/Euler_method>`_ to
carry out numerical step-wise integration of the decision process (see `Execution <DDM_Execution>` below).  In this
mode, only the `DECISION_VARIABLE <DDM_DECISION_VARIABLE>` and `RESPONSE_TIME <DDM_RESPONSE_TIME>` are available.

`IntegratorFunction <IntegratorFunction>` Function::

    >>> my_DDM_path_integrator = pnl.DDM(
    ...     function=pnl.DriftDiffusionIntegrator(
    ...         noise=0.5,
    ...         initializer=1.0,
    ...         starting_point=2.0,
    ...         rate=3.0
    ...     ),
    ...     name='my_DDM_path_integrator'
    ... )

COMMENT:
[TBI - MULTIPROCESS DDM - REPLACE ABOVE]
The DDM Mechanism implements a general form of the decision process.  A DDM Mechanism assigns one **inputState** to
each item in the `default_variable` argument, corresponding to each of the decision processes implemented
(see :ref:`Input <DDM_Input>` above). The decision process can be configured to execute in different modes.  The
`function <DDM.function>` parameters is the primary determinants of how the
decision process is executed, and what information is returned. The `function <DDM.function>` parameter specifies
the analytical solution to use. The number of `OutputStates <OutputState>` is determined by the `function <DDM.function>` in use (see
:ref:`list of output values <DDM_Results>` below).

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


COMMENT:  [OLD;  PUT SOMEHWERE ELSE??]

    The DDM process uses the same set of parameters for all modes of execution.  These can be specified as arguments
    for the functions used in TRIAL mode, or in a params dictionary assigned to the `params` argument,
    using the keywords in the list below, as in the following example::
        my_DDM = DDM(
            function=DriftDiffusionAnalytical(drift_rate=0.1),
            params={
                DRIFT_RATE:(0.2, ControlProjection),
                STARTING_POINT:-0.5
            },
        )
    The parameters for the DDM when `function <DDM.function>` is set to `DriftDiffusionAnalytical` are:

    .. _DDM_Drift_Rate:

    * `DRIFT_RATE <drift_rate>` (default 0.0)
      - multiplies the input to the Mechanism before assigning it to the `variable <DDM.variable>` on each call of
      `function <DDM.function>`.  The resulting value is further multiplied by the value of any ControlProjections to
      the `DRIFT_RATE` parameterState. The `drift_rate` parameter can be thought of as the "automatic" component
      (baseline strength) of the decision process, the value received from a ControlProjection as the "attentional"
      component, and the input its "stimulus" component.  The product of all three determines the drift rate in
      effect for each time_step of the decision process.
    ..
    * `STARTING_POINT <starting_point>` (default 0.0)
      - specifies the starting value of the decision variable.
    ..
    * `THRESHOLD` (default 1.0)
      - specifies the stopping value for the decision process. The `threshold` parameter must be greater than or
      equal to zero.
    ..
    * `NOISE` (default 0.5)
      - specifies the variance of the stochastic ("diffusion") component of the decision process.
    ..
    * `NON_DECISION_TIME` (default 0.2)
      specifies the `t0` parameter of the decision process (in units of seconds).

[TBI - MULTIPROCESS DDM - REPLACE BELOW]
When a DDM Mechanism is executed it computes the decision process, either analytically (in TRIAL mode)
or by step-wise integration (in TIME_STEP mode).  As noted above, if the input is a single value,
it computes a single DDM process.  If the input is a list or array, then multiple parallel DDM processes are executed,
with each element of the input used for the corresponding process.  All use the same set of parameters,
so the analytic solutions (used in TRIAL mode) for a given input will be the same; to implement processes in
this mode that use different parameters, a separate DDM Mechanism should explicitly be created for each. In
TIME_STEP mode, the noise term will resolve to different values in each time step, so the integration
paths and outcomes for the same input value will vary. This can be used to generate distributions of the process for a
single set of parameters that are not subject to the analytic solution (e.g., for time-varying drift rates).

.. note::
   DDM handles "runtime" parameters (specified in a call to its
   :py:meth:`execute <Mechanism_Base.exeucte>` or :py:meth:`run <Mechanism_Base.run>` methods)
   differently than standard Components: runtime parameters are added to the Mechanism's current value of the
   corresponding ParameterState (rather than overriding it);  that is, they are combined additively with the value of
   any `ControlProjection` it receives to determine the parameter's value for that execution.  The ParameterState's
   value is then restored to its original value (i.e., either its default value or the one assigned when it was
   created) for the next execution.

  ADD NOTE ABOUT INTERROGATION PROTOCOL, USING ``terminate_function``
  ADD NOTE ABOUT RELATIONSHIP OF RT TO time_steps TO t0 TO ms
COMMENT

.. _DDM_Execution:

Execution
---------

When a DDM Mechanism is executed, it computes the decision process either `analytically <DDM_Analytic_Mode>`  or by
`numerical step-wise integration <DDM_Integration_Mode>` of its path.  The method used is determined by its `function
<DDM.function>` (see `DDM_Modes`). The DDM's `function <DDM.function>` always returns values for the `DECISION_VARIABLE
<DDM_DECISION_VARIABLE>` and `RESPONSE_TIME <DDM_RESPONSE_TIME>`, and assigns these as the first two items of its `value
<DDM.value>` attribute, irrespective of its function.

When an `analytic <DDM_Analytic_Mode>` function is selected, the same set of values is returned for every execution.
The returned values are determined entirely by the set of parameters passed to its `function <DDM.function>`.

When the `path integration <DDM_Integration_Mode>`, function is selected, a single step of integration is conducted each
time the Mechanism is executed. The returned values accumulate on every execution.

The analytic functions return a final positon and time of the model, along with other statistics, where as the path
integration function returns intermediate position and time values. The two types of functions can be thought of as
happening on different time scales: trial (analytic) and time step (path integration).

References
----------

*   Bogacz, R., Brown, E., Moehlis, J., Holmes, P., & Cohen, J. D. (2006).
    The physics of optimal decision making: A formal analysis of models of performance in two-alternative forced-choice tasks.
    Psychological Review, 113(4), 700-765.
    http://dx.doi.org/10.1037/0033-295X.113.4.700

*   Srivastava, V., Holmes, P., Simen., P. (2016).
    Explicit moments of decision times for single- and double-threshold drift-diffusion processes,
    Journal of Mathematical Psychology, 75, 96-109,
    ISSN 0022-2496,
    https://doi.org/10.1016/j.jmp.2016.03.005.
    (http://www.sciencedirect.com/science/article/pii/S0022249616000341)

.. _DDM_Class_Reference:

Class Reference
---------------
"""
import logging
import random

from collections.abc import Iterable

import numpy as np
import typecheck as tc

from psyneulink.core.components.component import method_type
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import \
    DriftDiffusionIntegrator, IntegratorFunction
from psyneulink.core.components.functions.distributionfunctions import THRESHOLD, STARTING_POINT, \
    DriftDiffusionAnalytical
from psyneulink.core.components.functions.combinationfunctions import Reduce
from psyneulink.core.components.mechanisms.adaptive.control.controlmechanism import _is_control_spec
from psyneulink.core.components.mechanisms.mechanism import Mechanism_Base
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.states.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.components.states.outputstate import SEQUENTIAL, StandardOutputStates
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import ALLOCATION_SAMPLES, FUNCTION, FUNCTION_PARAMS, INPUT_STATE_VARIABLES, NAME, OUTPUT_STATES, OWNER_VALUE, VARIABLE, kwPreferenceSetName
from psyneulink.core.globals.parameters import Parameter, parse_context
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set, kpReportOutputPref
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel
from psyneulink.core.globals.utilities import is_numeric, is_same_function_spec, object_has_single_value

__all__ = [
    'DDM', 'DDM_OUTPUT', 'DDM_standard_output_states', 'DDMError',
    'DECISION_VARIABLE', 'DECISION_VARIABLE_ARRAY', 'PROBABILITY_LOWER_THRESHOLD', 'PROBABILITY_UPPER_THRESHOLD',
    'RESPONSE_TIME', 'RT_CORRECT_MEAN', 'RT_CORRECT_VARIANCE',
    'SCALAR', 'SELECTED_INPUT_ARRAY', 'ARRAY', 'VECTOR'
]

logger = logging.getLogger(__name__)

DEFAULT_VARIABLE = 0.0

DECISION_VARIABLE = 'DECISION_VARIABLE'
DECISION_VARIABLE_ARRAY = 'DECISION_VARIABLE_ARRAY'
SELECTED_INPUT_ARRAY = 'SELECTED_INPUT_ARRAY'
RESPONSE_TIME = 'RESPONSE_TIME'
PROBABILITY_UPPER_THRESHOLD = 'PROBABILITY_UPPER_THRESHOLD'
PROBABILITY_LOWER_THRESHOLD = 'PROBABILITY_LOWER_THRESHOLD'
RT_CORRECT_MEAN = 'RT_CORRECT_MEAN'
RT_CORRECT_VARIANCE = 'RT_CORRECT_VARIANCE'
RT_CORRECT_SKEW = 'RT_CORRECT_SKEW'
RT_INCORRECT_MEAN = 'RT_INCORRECT_MEAN'
RT_INCORRECT_VARIANCE = 'RT_INCORRECT_VARIANCE'
RT_INCORRECT_SKEW = 'RT_INCORRECT_SKEW'


# input_format Keywords:
SCALAR='SCALAR'
ARRAY='ARRAY'
VECTOR='VECTOR'

def decision_variable_to_array(x):
    """Generate "one-hot" 1d array designating selected action from DDM's scalar decision variable
    (used to generate value of OutputState for action_selection Mechanism
    """
    if x >= 0:
        return [x,0]
    else:
        return [0,x]

DDM_standard_output_states = [{NAME: DECISION_VARIABLE,},           # Upper or lower threshold for Analtyic function
                              {NAME: RESPONSE_TIME},                # TIME_STEP within TRIAL for Integrator function
                              {NAME: PROBABILITY_UPPER_THRESHOLD},  # Accuracy (TRIAL mode only)
                              {NAME: PROBABILITY_LOWER_THRESHOLD},  # Error rate (TRIAL mode only)
                              {NAME: RT_CORRECT_MEAN},              # (DriftDiffusionAnalytical only)
                              {NAME: RT_CORRECT_VARIANCE},          # (DriftDiffusionAnalytical only)
                              {NAME: RT_CORRECT_SKEW},              # (DriftDiffusionAnalytical only)
                              {NAME: RT_INCORRECT_MEAN},            # (DriftDiffusionAnalytical only)
                              {NAME: RT_INCORRECT_VARIANCE},        # (DriftDiffusionAnalytical only)
                              {NAME: RT_INCORRECT_SKEW},            # (DriftDiffusionAnalytical only)
                              ]

# This is a convenience class that provides list of standard_output_state names in IDE
class DDM_OUTPUT():
    """
    .. _DDM_Standard_OutputStates:

    `Standard OutputStates <OutputState_Standard>` for `DDM`:

    .. _DDM_DECISION_VARIABLE:

    *DECISION_VARIABLE* : float
      • `analytic mode <DDM_Analytic_Mode>`: the value of the threshold crossed by the decision variable on the
        current TRIAL (which is either the value of the DDM `function <DDM.function>`'s threshold attribute or its
        negative); \n
      • `integration mode <DDM_Integration_Mode>`: the value of the decision variable at the current TIME_STEP of
        execution. \n
      Corresponds to the 1st item of the DDM's `value <DDM.value>`.

    .. _DDM_DECISION_VARIABLE_ARRAY:

    *DECISION_VARIABLE_ARRAY* : 1d nparray
      .. note::
         This is available only if **input_format** is specified as *ARRAY* in the DDM Mechanism's constructor
         (see `DDM_Input`).
      • `analytic mode <DDM_Analytic_Mode>`: two element array, with the decision variable (1st item of the DDM's
        `value <DDM.value>`) as the 1st element if the decision process crossed the upper threshold, and the 2nd element
        if it is closer to the lower threshold; the other element is set to 0. \n
      • `integration mode <DDM_Integration_Mode>`: the value of the decision variable at the current TIME_STEP of
        execution, assigned to the 1st element if the decision variable is closer to the upper threshold, and to the
        2nd element if it is closer to the lower threshold; the other element is set to 0. \n

    .. _DDM_DECISION_VARIABLE_ARRAY:

    *SELECTED_INPUT_ARRAY* : 1d nparray
      .. note::
         This is available only if **input_format** is specified as *ARRAY* in the DDM Mechanism's constructor
         (see `DDM_Input`).
      • `analytic mode <DDM_Analytic_Mode>`: two element array, with one ("selected") element -- determined by the
        outcome of the decision process -- set to the value of the corresponding element in the stimulus array (i.e.,
        the DDM's input_state `variable <InputState.variable>`).  The "selected" element is the 1st one if the decision
        process resulted in crossing the upper threshold, and the 2nd if it crossed the lower threshold; the other
        element is set to 0. \n
      • `integration mode <DDM_Integration_Mode>`: the value of the element in the stimulus array based on the
        decision variable (1st item of the DDM's `value <DDM.value>`) at the current TIME_STEP of execution:
        it is assigned to the 1st element if the decision variable is closer to the upper threshold, and to the  2nd
        element if the decision variable is closer to the lower threshold; the other element is set to 0. \n

    .. _DDM_RESPONSE_TIME:

    *RESPONSE_TIME* : float
      • `analytic mode <DDM_Analytic_Mode>`: mean time (in seconds) for the decision variable to reach the positive
        or negative value of the DDM `function <DDM.function>`'s threshold attribute as estimated by the analytic
        solution calculated by the `function <DDM.function>`); \n
      • `integration mode <DDM_Integration_Mode>`: the number of `TIME_STEP` that have occurred since the DDM began
        to execute in the current `TRIAL` or, if it has reached the positive or negative value of the DDM `function
        <DDM.function>`'s threshold attribute, the `TIME_STEP` at which that occurred. \n
      Corresponds to the 2nd item of the DDM's `value <DDM.value>`.

    .. _DDM_PROBABILITY_UPPER_THRESHOLD:

    *PROBABILITY_UPPER_THRESHOLD* : float
      • `analytic mode <DDM_Analytic_Mode>`: the probability of the decision variable reaching the positive value of
        the DDM `function <DDM.function>`'s threshold attribute as estimated by the analytic solution calculated by the
        `function <DDM.function>`; often, by convention, the positive (upper) threshold is associated with the
        correct response, in which case *PROBABILITY_UPPER_THRESHOLD* corresponds to the accuracy of the decision
        process. \n
      • `integration mode <DDM_Integration_Mode>`: `None`.
      Corresponds to the 3rd item of the DDM's `value <DDM.value>`.

    COMMENT:
      [TBI:]
          `integration mode <DDM_Integration_Mode>`, if execution has completed, this is a binary value
          indicating whether the decision process reached the upper (positive) threshold. If execution was
          interrupted (using :py:meth:`terminate_function  <DDM.terminate_function>`, sometimes referred to as the
          :ref:`interrogation protocol <LINK>`, then the value corresponds to the current likelihood that the upper
          threshold would have been reached.
    COMMENT

    .. _DDM_PROBABILITY_LOWER_THRESHOLD:

    *PROBABILITY_LOWER_THRESHOLD* : float
      • `analytic mode <DDM_Analytic_Mode>`: the probability of the decision variable reaching the negative value of
        the DDM `function <DDM.function>`'s threshold attribute as estimated by the analytic solution calculate by the
        `function <DDM.function>`); often, by convention, the negative (lower) threshold is associated with an error
        response, in which case *PROBABILITY_LOWER_THRESHOLD* corresponds to the error rate of the decision process; \n
      • `integration mode <DDM_Integration_Mode>`: `None`.
      Corresponds to the 4th item of the DDM's `value <DDM.value>`.

        COMMENT:
          [TBI:]
              `integration mode <DDM_Integration_Mode>`, if execution has completed, this is a binary value
              indicating whether the decision process reached the lower (negative) threshold. If execution was
              interrupted (using :py:method:`terminate_method <DDM.terminate_function>`, sometimes referred to as the
              :ref:`interrogation protocol <LINK>`), then the value corresponds to the current likelihood that the lower
              threshold would have been reached.
        COMMENT

    .. _DDM_RT_CORRECT_MEAN:

    *RT_CORRECT_MEAN* : floa
      (only applicable if `function <DDM.function>` is `DriftDiffusionAnalytical`) \n
      • `analytic mode <DDM_Analytic_Mode>`:  the mean reaction time (in seconds) for responses in which the decision
        variable reached the positive value of the DDM `function <DDM.function>`'s threshold attribute as estimated by
        closed form analytic solutions from Srivastava et al. (https://arxiv.org/abs/1601.06420) \n
      • `integration mode <DDM_Integration_Mode>`: `None`.
      Corresponds to the 5th item of the DDM's `value <DDM.value>`.

    .. _DDM_RT_CORRECT_VARIANCE:

    *RT_CORRECT_VARIANCE* : float
      (only applicable if `function <DDM.function>` is `DriftDiffusionAnalytical`) \n
      • `analytic mode <DDM_Analytic_Mode>`:  the variance of reaction time (in seconds) for responses in which the decision
        variable reached the positive value of the DDM `function <DDM.function>`'s threshold attribute as estimated by
        closed form analytic solutions from Srivastava et al. (https://arxiv.org/abs/1601.06420) \n
      • `integration mode <DDM_Integration_Mode>`: `None`.
      Corresponds to the 6th item of the DDM's `value <DDM.value>`.

    .. _DDM_RT_CORRECT_SKEW:

    *RT_CORRECT_SKEW* : float
      (only applicable if `function <DDM.function>` is `DriftDiffusionAnalytical`) \n
      • `analytic mode <DDM_Analytic_Mode>`:  the skew of decision time (in seconds) for responses in which the decision
        variable reached the positive value of the DDM `function <DDM.function>`'s threshold attribute as estimated by
        closed form analytic solutions from Srivastava et al. (https://arxiv.org/abs/1601.06420) \n
      • `integration mode <DDM_Integration_Mode>`: `None`.
      Corresponds to the 7th item of the DDM's `value <DDM.value>`.

    .. _DDM_RT_INCORRECT_MEAN:

    *RT_INCORRECT_MEAN* : float
      (only applicable if `function <DDM.function>` is `DriftDiffusionAnalytical`) \n
      • `analytic mode <DDM_Analytic_Mode>`:  the mean reaction time (in seconds) for responses in which the decision
        variable reached the negative value of the DDM `function <DDM.function>`'s threshold attribute as estimated by
        closed form analytic solutions from Srivastava et al. (https://arxiv.org/abs/1601.06420) \n
      • `integration mode <DDM_Integration_Mode>`: `None`.
      Corresponds to the 5th item of the DDM's `value <DDM.value>`.

    .. _DDM_RT_INCORRECT_VARIANCE:

    *RT_INCORRECT_VARIANCE* : float
      (only applicable if `function <DDM.function>` is `DriftDiffusionAnalytical`) \n
      • `analytic mode <DDM_Analytic_Mode>`:  the variance of reaction time (in seconds) for responses in which the decision
        variable reached the negative value of the DDM `function <DDM.function>`'s threshold attribute as estimated by
        closed form analytic solutions from Srivastava et al. (https://arxiv.org/abs/1601.06420) \n
      • `integration mode <DDM_Integration_Mode>`: `None`.
      Corresponds to the 6th item of the DDM's `value <DDM.value>`.

    .. _DDM_RT_INCORRECT_SKEW:

    *RT_INCORRECT_SKEW* : float
      (only applicable if `function <DDM.function>` is `DriftDiffusionAnalytical`) \n
      • `analytic mode <DDM_Analytic_Mode>`:  the skew of decision time (in seconds) for responses in which the decision
        variable reached the negative value of the DDM `function <DDM.function>`'s threshold attribute as estimated by
        closed form analytic solutions from Srivastava et al. (https://arxiv.org/abs/1601.06420) \n
      • `integration mode <DDM_Integration_Mode>`: `None`.
      Corresponds to the 7th item of the DDM's `value <DDM.value>`.

    """
    DECISION_VARIABLE=DECISION_VARIABLE
    RESPONSE_TIME=RESPONSE_TIME
    PROBABILITY_UPPER_THRESHOLD=PROBABILITY_UPPER_THRESHOLD
    PROBABILITY_LOWER_THRESHOLD=PROBABILITY_LOWER_THRESHOLD
    RT_CORRECT_MEAN=RT_CORRECT_MEAN
    RT_CORRECT_VARIANCE=RT_CORRECT_VARIANCE
    RT_CORRECT_SKEW = RT_CORRECT_SKEW
    RT_INCORRECT_MEAN=RT_INCORRECT_MEAN
    RT_INCORRECT_VARIANCE=RT_INCORRECT_VARIANCE
    RT_INCORRECT_SKEW=RT_INCORRECT_SKEW
    DECISION_VARIABLE_ARRAY=DECISION_VARIABLE_ARRAY
    SELECTED_INPUT_ARRAY=SELECTED_INPUT_ARRAY
# THE FOLLOWING WOULD HAVE BEEN NICE, BUT IDE DOESN'T EXECUTE IT, SO NAMES DON'T SHOW UP
# for item in [item[NAME] for item in DDM_standard_output_states]:
#     setattr(DDM_OUTPUT.__class__, item, item)


class DDMError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class DDM(ProcessingMechanism):
    # DOCUMENT:   COMBINE WITH INITIALIZATION WITH PARAMETERS
    #             ADD INFO ABOUT B VS. N&F
    #             ADD _instantiate_output_states TO INSTANCE METHODS, AND EXPLAIN RE: NUM OUTPUT VALUES FOR B VS. N&F
    """
    DDM(                               \
    default_variable=None,             \
    size=None,                         \
    function=DriftDiffusionAnalytical, \
    params=None,                       \
    name=None,                         \
    prefs=None)

    Implement a Drift Diffusion Process, either by calculating an `analytic solution <DDM_Analytic_Mode>` or carrying
    out `step-wise numerical integration <DDM_Integration_Mode>`.

    COMMENT:
        Description
        -----------
            DDM is a subclass Type of the Mechanism Category of the Component class
            It implements a Mechanism for several forms of the Drift Diffusion Model (DDM) for
                two alternative forced choice (2AFC) decision making:
                - analytical solution [Bogacz et al. (2006), Srivastava et al. (2016)]
                     - stochastically estimated decion outcome (convert mean ER into value between 1 and -1)
                     - mean ER
                     - mean RT
                     - mean, variance, and skew of RT for correct (posititive threshold) responses
                     - mean, variance, and skew of RT for incorrect (negative threshold) responses
                - stepwise integrator that simulates each step of the integration process
        Class attributes
        ----------------
            + componentType (str): DDM
            + classPreference (PreferenceSet): DDM_PreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE
            + paramClassDefaults (dict): {
                                          kwDDM_AnalyticSolution: kwDriftDiffusionAnalytical,
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
                                                          DDM_RT_CORRECT_SKEW,
                                                          DDM_RT_INCORRECT_MEAN,
                                                          DDM_RT_INCORRECT_VARIANCE,
                                                          DDM_RT_INCORRECT_SKEW,
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
        the input to the Mechanism used if none is provided in a call to its `execute <Mechanism_Base.execute>` or
        `run <Mechanism_Base.run>` methods; also serves as a template to specify the length of the `variable
        <DDM.variable>` for its `function <DDM.function>`, and the `primary OutputState <OuputState_Primary>` of the
        DDM (see `Input` <DDM_Creation>` for how an input with a length of greater than 1 is handled).

    size : int, list or np.ndarray of ints
        specifies the `default_variable <DDM.default_variable>` as array(s) of zeros if **default_variable** is not
        passed as an argument; if **default_variable** is specified, it takes precedence over the specification of
        **size**. As an example, the following mechanisms are equivalent::
            T1 = TransferMechanism(size = [3, 2])
            T2 = TransferMechanism(default_variable = [[0, 0, 0], [0, 0]])

    function : IntegratorFunction : default DriftDiffusionAnalytical
        specifies the function to use to `execute <DDM_Execution>` the decision process; determines the mode of
        execution (see `function <DDM.function>` and `DDM_Modes` for additional information).

    params : Dict[param keyword: param value] : default None
        a dictionary that can be used to specify parameters of the Mechanism, parameters of its `function
        <DDM.function>`, and/or  a custom function and its parameters (see `Mechanism <Mechanism>` for specification of
        a params dict).

    name : str : default see `name <DDM.name>`
        specifies the name of the DDM.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the DDM; see `prefs <DDM.prefs>` for details.

    COMMENT:
    context=componentType+INITIALIZING):
        context : str : default ''None''
               string used for contextualization of instantiation, hierarchical calls, executions, etc.
    COMMENT

    Attributes
    ----------
    variable : value : default  FUNCTION_PARAMS[STARTING_POINT]
        the input to Mechanism's execute method.  Serves as the "stimulus" component of the `function <DDM.function>`'s
        **drift_rate** parameter.

    function :  IntegratorFunction : default DriftDiffusionAnalytical
        the function used to `execute <DDM_Execution>` the decision process; determines the mode of execution.
        If it is `DriftDiffusionAnalytical <DriftDiffusionAnalytical>`, an `analytic solution
        <DDM_Analytic_Mode>` is calculated (note:  the latter requires that the MatLab engine is installed); if it is
        an `IntegratorFunction` Function with an `integration_type <IntegratorFunction.integration_type>` of *DIFFUSION*,
        then `numerical step-wise integration <DDM_Integration_Mode>` is carried out.  See `DDM_Modes` and
        `DDM_Execution` for additional information.
        COMMENT:
           IS THIS MORE CORRECT FOR ABOVE:
               if it is `DriftDiffusionIntegrator`, then `numerical step-wise integration <DDM_Integration_Mode>`
               is carried out.
        COMMENT

    value : 2d np.array[array(float64),array(float64),array(float64),array(float64)]
        result of executing DDM `function <DDM.function>`;  has six items, that are assigned based on the `function
        <DDM.function>` attribute.  The first two items are always assigned the values of `DECISION_VARIABLE
        <DDM_DECISION_VARIABLE>` and `RESPONSE_TIME <DDM_RESPONSE_TIME>` (though their interpretation depends on the
        `function <DDM.function>` and corresponding `mode of <DDM_Modes>` of operation).  See `DDM_Modes`,
        `DDM_Execution`, and `DDM Standard OutputStates <DDM_Standard_OutputStates>` for additional information about
        other values that can be reported and their interpretation.

    output_states : ContentAddressableList[OutputState]
        list of the DDM's `OutputStates <OutputState>`.  There are always two OutputStates, `DECISION_VARIABLE
        <DDM_DECISION_VARIABLE>` and `RESPONSE_TIME <DDM_RESPONSE_TIME>`; additional ones may be included
        based on the `function <DDM.function>` and/or any specifications made in the **output_states** argument of the
        DDM's constructor (see `DDM Standard OutputStates <DDM_Standard_OutputStates>`).

    output_values : List[array(float64),array(float64),array(float64),array(float64)]
        each item is the `value <OutputState.value> of the corresponding OutputState in `output_states
        <DDM.output_states>`.  The first two items are always the `value <OutputState.value>`\\s of the
        `DECISION_VARIABLE <DDM_DECISION_VARIABLE>` and `RESPONSE_TIME <DDM_RESPONSE_TIME>` OutputStates;  additional
        ones may be included, based on the `function <DDM.function>` and any specifications made in the
        **output_states** argument of the DDM's constructor (see `DDM Standard OutputStates
        <DDM_Standard_OutputStates>`).

    name : str
        the name of the DDM; if it is not specified in the **name** argument of the constructor, a default is
        assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the DDM; if it is not specified in the **prefs** argument of the constructor, a default
        is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).

    COMMENT:
        MOVE TO METHOD DEFINITIONS:
        Instance methods:
            - _instantiate_function(context)
                deletes params not in use, in order to restrict outputStates to those that are computed for
                specified params
            - execute(variable, params, context)
                executes specified version of DDM and returns outcome values (in self.value and values of
                self.output_states)
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

    class Parameters(ProcessingMechanism.Parameters):
        """
            Attributes
            ----------

                function
                    see `function <DDM.function>`

                    :default value: `DriftDiffusionAnalytical`
                    :type: `Function`

                initializer
                    see `initializer <DDM.initializer>`

                    :default value: numpy.array([[0]])
                    :type: numpy.ndarray

                input_format
                    see `input_format <DDM.input_format>`

                    :default value: `SCALAR`
                    :type: str

        """
        function = Parameter(
            DriftDiffusionAnalytical(
                drift_rate=1.0,
                starting_point=0.0,
                threshold=1.0,
                noise=0.5,
                t0=.200,
            ),
            stateful=False,
            loggable=False
        )
        input_format = Parameter(SCALAR, stateful=False, loggable=False)

        initializer = np.array([[0]])

    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        OUTPUT_STATES: None})

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_format:tc.optional(tc.enum(SCALAR, ARRAY, VECTOR))=SCALAR,
                 function=DriftDiffusionAnalytical(drift_rate=1.0,
                                                   starting_point=0.0,
                                                   threshold=1.0,
                                                   noise=0.5,
                                                   t0=.200),
                 output_states:tc.optional(tc.any(str, Iterable))=(DECISION_VARIABLE, RESPONSE_TIME),
                 params=None,
                 name=None,
                 prefs: is_pref_set = None,
                 **kwargs):

        self.standard_output_states = StandardOutputStates(self,
                                                           DDM_standard_output_states,
                                                           indices=SEQUENTIAL)

        # If input_format is specified to be ARRAY or VECTOR, instantiate:
        #    InputState with:
        #        2-item array as its variable
        #        Reduce as its function, which will generate an array of len 1
        #        and therefore specify size of Mechanism's variable as 1
        #    OutputStates that report the decision variable and selected input in array format
        #        IMPLEMENTATION NOTE:
        #            These are created here rather than as StandardOutputStates
        #            since they require input_format==ARRAY to be meaningful
        if input_format in {ARRAY, VECTOR}:
            size=1 # size of variable for DDM Mechanism
            input_states = [
                {NAME:'ARRAY',
                 VARIABLE: np.array([[0.0, 0.0]]),
                 FUNCTION: Reduce(weights=[1,-1])}
            ]
            self.standard_output_states.add_state_dicts([
                # Provides a 1d 2-item array with:
                #    decision variable in position corresponding to threshold crossed, and 0 in the other position
                {NAME: DECISION_VARIABLE_ARRAY, # 1d len 2, DECISION_VARIABLE as element 0 or 1
                 VARIABLE:[(OWNER_VALUE, self.DECISION_VARIABLE_INDEX), THRESHOLD],
                           # per VARIABLE assignment above, items of v of lambda function below are:
                           #    v[0]=self.value[self.DECISION_VARIABLE_INDEX]
                           #    v[1]=self.parameter_states[THRESHOLD]
                 FUNCTION: lambda v: [float(v[0]), 0] if (v[1]-v[0]) < (v[1]+v[0]) else [0, float(v[0])]},
                # Provides a 1d 2-item array with:
                #    input value in position corresponding to threshold crossed by decision variable, and 0 in the other
                {NAME: SELECTED_INPUT_ARRAY, # 1d len 2, DECISION_VARIABLE as element 0 or 1
                 VARIABLE:[(OWNER_VALUE, self.DECISION_VARIABLE_INDEX), THRESHOLD, (INPUT_STATE_VARIABLES, 0)],
                 # per VARIABLE assignment above, items of v of lambda function below are:
                 #    v[0]=self.value[self.DECISION_VARIABLE_INDEX]
                 #    v[1]=self.parameter_states[THRESHOLD]
                 #    v[2]=self.input_states[0].variable
                 FUNCTION: lambda v: [float(v[2][0][0]), 0] \
                                      if (v[1]-v[0]) < (v[1]+v[0]) \
                                      else [0,float(v[2][0][1])]

                 }

            ])

        else:
            input_states = None

        # Default output_states is specified in constructor as a tuple rather than a list
        # to avoid "gotcha" associated with mutable default arguments
        # (see: bit.ly/2uID3s3 and http://docs.python-guide.org/en/latest/writing/gotchas/)
        if isinstance(output_states, (str, tuple)):
            output_states = list(output_states)

        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(function=function,
                                                  # input_format=input_format,
                                                  input_states=input_states,
                                                  output_states=output_states,
                                                  params=params)

        # IMPLEMENTATION NOTE: this manner of setting default_variable works but is idiosyncratic
        # compared to other mechanisms: see TransferMechanism.py __init__ function for a more normal example.
        if default_variable is None and size is None:
            try:
                default_variable = params[FUNCTION_PARAMS][STARTING_POINT]
                if not is_numeric(default_variable):
                    # set normally by default
                    default_variable = None
            except KeyError:
                # set normally by default
                pass

        # # Conflict with above
        # self.size = size

        super(DDM, self).__init__(default_variable=default_variable,
                                  input_states=input_states,
                                  output_states=output_states,
                                  function=function,
                                  params=params,
                                  name=name,
                                  prefs=prefs,
                                  size=size,
                                  **kwargs),

        self._instantiate_plotting_functions()


    def plot(self, stimulus=1.0, threshold=10.0):
        """
        Generate a dynamic plot of the DDM integrating over time towards a threshold.

        .. note::
            The plot method is only available when the DriftDiffusionIntegrator function is in use. The plot method does
            not represent the results of this DDM mechanism in particular, and does not affect the current state of this
            mechanism's DriftDiffusionIntegrator. The plot method is only meant to visualize a possible path of a DDM
            mechanism with these function parameters.

        Arguments
        ---------
        stimulus: float: default 1.0
            specify a stimulus value for the AdaptiveIntegrator function

        threshold: float: default 10.0
            specify the threshold at which the DDM will stop integrating

        Returns
        -------
        Mechanism's function plot : Matplotlib window
            Matplotlib window of the Mechanism's function plotting dynamically over time with specified parameters
            towards a specified threshold


        """
        import matplotlib.pyplot as plt
        import time
        plt.ion()

        # set initial values and threshold
        time_step = [0]
        position = [float(self.defaults.variable)]
        variable = stimulus

        # execute the mechanism once to begin the loop
        result_check = self.plot_function(variable, context="plot")[0][0]

        # continue executing the ddm until its value exceeds the threshold
        while abs(result_check) < threshold:
            time_step.append(time_step[-1] + 1)
            position.append(result_check)
            result_check = self.plot_function(variable, context="plot")[0][0]

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

    def _validate_variable(self, variable, context=None):
        """Ensures that input to DDM is a single value.
        Remove when MULTIPROCESS DDM is implemented.
        """

        # this test may become obsolete when size is moved to Component.py
        # if len(variable) > 1 and not self.input_format in {ARRAY, VECTOR}:
        if not object_has_single_value(variable) and not object_has_single_value(np.array(variable)):
            raise DDMError("Length of input to DDM ({}) is greater than 1, implying there are multiple "
                           "input states, which is currently not supported in DDM, but may be supported"
                           " in the future under a multi-process DDM. Please use a single numeric "
                           "item as the default_variable, or use size = 1.".format(variable))
        # # MODIFIED 6/28/17 (CW): changed len(variable) > 1 to len(variable[0]) > 1
        # # if not isinstance(variable, numbers.Number) and len(variable[0]) > 1:
        # if not is_numeric(variable) and len(variable[0]) > 1:
        #     raise DDMError("Input to DDM ({}) must have only a single numeric item".format(variable))
        return super()._validate_variable(variable=variable, context=context)

    def _validate_params(self, request_set, target_set=None, context=None):

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)
        functions = {DriftDiffusionAnalytical,
                     DriftDiffusionIntegrator}

        if FUNCTION in target_set:
            # If target_set[FUNCTION] is a method of a Function (e.g., being assigned in _instantiate_function),
            #   get the Function to which it belongs
            fun = target_set[FUNCTION]
            if isinstance(fun, method_type):
                fun = fun.__self__.__class__

            for function_type in functions:
                if is_same_function_spec(fun, function_type):
                    break
            else:
                function_names = [fun.componentName for fun in functions]
                raise DDMError("{} param of {} must be one of the following functions: {}".
                               format(FUNCTION, self.name, function_names))

        try:
            threshold = target_set[FUNCTION_PARAMS][THRESHOLD]
        except KeyError:
            pass
        else:
            if isinstance(threshold, tuple):
                threshold = threshold[0]
            if is_numeric(threshold):
                if not threshold >= 0:
                    raise DDMError("{} param of {} ({}) must be >= zero".
                                   format(THRESHOLD, self.name, threshold))
            elif isinstance(threshold, ControlSignal):
                threshold = threshold.allocation_samples
                if not np.amin(threshold) >= 0:
                    raise DDMError("The lowest value of {} for the {} "
                                   "assigned to the {} param of {} must be >= zero".
                                   format(ALLOCATION_SAMPLES, ControlSignal.__name__, THRESHOLD, self.name, threshold))
            elif _is_control_spec(threshold):
                pass
            else:
                raise DDMError("PROGRAM ERROR: unrecognized specification for {} of {} ({})".
                               format(THRESHOLD, self.name, threshold))

    def _instantiate_attributes_before_function(self, function=None, context=None):
        """Delete params not in use, call super.instantiate_execute_method
        """

        super()._instantiate_attributes_before_function(function=function, context=context)

    def _instantiate_plotting_functions(self, context=None):
        if "DriftDiffusionIntegrator" in str(self.function):
            self.get_axes_function = DriftDiffusionIntegrator(rate=self.function_params['rate'],
                                                              noise=self.function_params['noise']).function
            self.plot_function = DriftDiffusionIntegrator(rate=self.function_params['rate'],
                                                          noise=self.function_params['noise']).function

    def _execute(
        self,
        variable=None,
        context=None,
        runtime_params=None,

    ):
        """Execute DDM function (currently only trial-level, analytic solution)
        Execute DDM and estimate outcome or calculate trajectory of decision variable
        Currently implements only trial-level DDM (analytic solution) and returns:
            - stochastically estimated decion outcome (convert mean ER into value between 1 and -1)
            - mean ER
            - mean RT
            - mean, variance, and skew of RT for correct (posititive threshold) responses
            - mean, variance, and skew of RT for incorrect (negative threshold) responses
        Return current decision variable (self.outputState.value) and other output values (self.output_states[].value
        Arguments:
        # CONFIRM:
        variable (float): set to self.value (= self.input_value)
        - params (dict):  runtime_params passed from Mechanism, used as one-time value for current execution:
            + DRIFT_RATE (float)
            + THRESHOLD (float)
            + kwDDM_Bias (float)
            + NON_DECISION_TIME (float)
            + NOISE (float)
        - context (str)
        Returns the following values in self.value (2D np.array) and in
            the value of the corresponding outputState in the self.output_states dict:
            - decision variable (float)
            - mean error rate (float)
            - mean RT (float)
            - mean, variance, and skew of RT for correct (posititive threshold) responses
            - mean, variance, and skew of RT for incorrect (negative threshold) responses

        :param self:
        :param variable (float)
        :param params: (dict)
        :param context: (str)
        :rtype self.outputState.value: (number)
        """

        if variable is None or np.isnan(variable):
            # IMPLEMENT: MULTIPROCESS DDM:  ??NEED TO DEAL WITH PARTIAL NANS
            variable = self.defaults.variable

        variable = self._validate_variable(variable)

        # EXECUTE INTEGRATOR SOLUTION (TIME_STEP TIME SCALE) -----------------------------------------------------
        if isinstance(self.function, IntegratorFunction):

            result = super()._execute(variable, context=context)

            if self.initialization_status != ContextFlags.INITIALIZING:
                logger.info('{0} {1} is at {2}'.format(type(self).__name__, self.name, result))

            return np.array([result[0], [result[1]]])

        # EXECUTE ANALYTIC SOLUTION (TRIAL TIME SCALE) -----------------------------------------------------------
        else:

            result = super()._execute(
                variable=variable,
                context=context,
                runtime_params=runtime_params,

            )

            if isinstance(self.function, DriftDiffusionAnalytical):
                return_value = np.zeros(shape=(10,1))
                return_value[self.RESPONSE_TIME_INDEX] = result[0]
                return_value[self.PROBABILITY_LOWER_THRESHOLD_INDEX] = result[1]
                return_value[self.PROBABILITY_UPPER_THRESHOLD_INDEX] = \
                                                               1 - return_value[self.PROBABILITY_LOWER_THRESHOLD_INDEX]
                return_value[self.RT_CORRECT_MEAN_INDEX] = result[2]
                return_value[self.RT_CORRECT_VARIANCE_INDEX] = result[3]
                return_value[self.RT_CORRECT_SKEW_INDEX] = result[4]
                return_value[self.RT_INCORRECT_MEAN_INDEX] = result[5]
                return_value[self.RT_INCORRECT_VARIANCE_INDEX] = result[6]
                return_value[self.RT_INCORRECT_SKEW_INDEX] = result[7]

            else:
                raise DDMError("The function specified ({}) for {} is not a valid function selection for the DDM".
                               format(self.function.name, self.name))

            # Convert ER to decision variable:
            threshold = float(self.function.get_current_function_param(THRESHOLD, context))
            if random.random() < return_value[self.PROBABILITY_LOWER_THRESHOLD_INDEX]:
                return_value[self.DECISION_VARIABLE_INDEX] = np.atleast_1d(-1 * threshold)
            else:
                return_value[self.DECISION_VARIABLE_INDEX] = threshold
            return return_value

    @handle_external_context(execution_id=NotImplemented)
    def reinitialize(self, *args, context=None):
        from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import IntegratorFunction

        if context.execution_id is NotImplemented:
            context.execution_id = self.most_recent_context.execution_id

        # (1) reinitialize function, (2) update mechanism value, (3) update output states
        if isinstance(self.function, IntegratorFunction):
            new_values = self.function.reinitialize(*args, context=context)
            self.parameters.value._set(np.array(new_values), context)
            self._update_output_states(context=context)

    @handle_external_context()
    def is_finished(self, context=None):
        # find the single numeric entry in previous_value
        try:
            single_value = self.function.get_previous_value(context)
        except AttributeError:
            # Analytical function so it is always finished after it is called
            return True

        # indexing into a matrix doesn't reduce dimensionality
        if not isinstance(single_value, (np.matrix, str)):
            while True:
                try:
                    single_value = single_value[0]
                except (IndexError, TypeError):
                    break

        if (
            abs(single_value) >= self.function.get_current_function_param(THRESHOLD, context)
            and isinstance(self.function, IntegratorFunction)
        ):
            logger.info(
                '{0} {1} has reached threshold {2}'.format(
                    type(self).__name__,
                    self.name,
                    self.function.get_current_function_param(THRESHOLD, context)
                )
            )
            return True
        return False
