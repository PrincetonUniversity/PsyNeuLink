# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ***************************************************  DDM *************************************************************

"""

Contents
--------
  * `DDM_Overview`
  * `DDM_Creation`
  * `DDM_Structure`
      - `DDM_Input`
      - `DDM_Output`
          • `Default OutputPorts <DDM_Default_OutputPorts>`
          • `Custom OutputPorts <DDM_Custom_OutputPorts>`
      - `DDM_Modes`
          • `DDM_Analytic_Mode`
          • `DDM_Integration_Mode`
  * `DDM_Execution`
  * `DDM_Class_Reference`


.. _DDM_Overview:

Overview
--------
The DDM Mechanism implements the "Drift Diffusion Model" (also know as the Diffusion Decision, Accumulation to Bound,
Linear IntegratorFunction, and `First Passage Time Model <https://en.wikipedia.org/wiki/First-hitting-time_model>`_
for a `Wiener Process <https://en.wikipedia.org/wiki/Wiener_process>`_. This corresponds to a continuous version
of the `sequential probability ratio test (SPRT) <https://en.wikipedia.org/wiki/Sequential_probability_ratio_test>`_,
that is the statistically optimal procedure for `two alternative forced choice (TAFC) decision making
<https://en.wikipedia.org/wiki/Two-alternative_forced_choice>`_ (see `drift-diffusion model
<https://en.wikipedia.org/wiki/Two-alternative_forced_choice#Drift-diffusion_model>`_ in partciular).

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
`drift rate <DDM_Drift_Rate>` for the decision process.  It must be a single scalar value.
[TBI - MULTIPROCESS DDM - REPLACE ABOVE]
**Input**.  The ``default_variable`` argument specifies the default value to use as the stimulus component of the
`drift rate <DDM_Drift_Rate>` for each decision process, as well as the number of decision processes implemented
and the corresponding format of the ``input`` required by calls to its ``execute`` and ``run`` methods.  This can be a
single scalar value or an an array (list or 1d np.array). If it is a single value (as in the first two examples above),
a single DDM process is implemented, and the input is used as the stimulus component of the
`drift rate <DDM_Drift_Rate>` for the process. If the input is an array (as in the third example above),
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
<DDM_Modes>` that is used.  Accordingly, the default `InputPort` for a DDM takes a single scalar value as its input,
that represents the stimulus for the decision process.  However, this can be configured using the **input_format**
argument of the DDM's consructor, to accomodate use of the DDM with other Mechanisms that generate a stimulus array
(e.g., representing the stimuli associated with each of the two choices). By default, the **input_format** is
*SCALAR*.  However, if it is specified as *ARRAY*, the DDM's InputPort is configured to accept a 1d 2-item vector,
and to use `Reduce` as its Function, which subtracts the 2nd element of the vector from the 1st, and provides this as
the input to the DDM's `function <DDM.function>`.  If *ARRAY* is specified, two  `Standard OutputPorts
<OutputPorts_Standard>` are added to the DDM, that allow the result of the decision process to be represented
as an array corresponding to the input array (see `below <DDM_Custom_OutputPorts>`).

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
are represented by the DDM Mechanism's two default `OutputPorts``: `DECISION_VARIABLE <DDM_DECISION_VARIABLE>` and
`RESPONSE_TIME <DDM_RESPONSE_TIME>`.  Other `output_ports <DDM.output_ports>` may be automatically assigned,
depending on the `function <DDM.function>` that has been assigned to the DDM, as shown in the table below:

.. _DDM_Default_OutputPorts:

+------------------------------------+---------------------------------------------------------+
|                                    |                     **Function**                        |
|                                    |                      *(type)*                           |
+                                    +----------------------------+----------------------------+
|                                    | `DriftDiffusionAnalytical` | `DriftDiffusionIntegrator` |
|                                    |   (`analytic               |   (`path integration)      |
| **OutputPorts:**                  |   <DDM_Analytic_Mode>`)    |   <DDM_Integration_Mode>`) |
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

.. _DDM_Custom_OutputPorts:

The `output_ports <DDM.output_ports>` assigned to a DDM can explicilty determined by specifying ones from its list
of `standard_output_ports <DDM.standard_output_ports>` in the **output_ports** argument of its constructor, or the
*OUTPUT_PORTS* entry of an `OutputPort specification dictionary <OutputPort_Specification_Dictionary>`.  This
can include any of the OutputPorts `listed above `DDM_Default_OutputPorts`, as well as two additional ones --
-- `DECISION_VARIABLE_ARRAY <DDM_DECISION_VARIABLE_ARRAY>` and `SELECTED_INPUT_ARRAY <DDM_SELECTED_INPUT_ARRAY>` --
that are available if the *ARRAY* option is specified in its **input_format** argument (see `DDM_Input`).  All of
these `Standard OutputPorts <OutputPort_Standard>` are listed in the DDM's `standard_output_ports
<DDM.standard_output_ports>` attribute. As with any Mechanism, `customized OutputPorts <OutputPort_Customization>`
can also be created and assigned.

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
`PROBABILITY_UPPER_THRESHOLD <DDM_PROBABILITY_UPPER_THRESHOLD>` OutputPort), and an error rate value (in the
`PROBABaILITY_LOWER_THRESHOLD <DDM_PROBABILITY_LOWER_THRESHOLD>` OutputPort, and moments (mean, variance, and skew)
for conditional (correct\\positive or incorrect\\negative) response time distributions. These are; the mean RT for
correct responses  (`RT_CORRECT_MEAN <DDM_RT_CORRECT_MEAN>`, the RT variance for correct responses
(`RT_CORRECT_VARIANCE <DDM_RT_CORRECT_VARIANCE>`, the RT skew for correct responses
(`RT_CORRECT_SKEW <DDM_RT_CORRECT_SKEW>`, the mean RT for incorrect responses  (`RT_INCORRECT_MEAN
<DDM_RT_INCORRECT_MEAN>`, the RT variance for incorrect responses (`RT_INCORRECT_VARIANCE
<DDM_RT_INCORRECT_VARIANCE>`, the RT skew for incorrect responses (`RT_INCORRECT_SKEW <DDM_RT_INCORRECT_SKEW>`.

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
The DDM Mechanism implements a general form of the decision process.  A DDM Mechanism assigns one **inputPort** to
each item in the `default_variable` argument, corresponding to each of the decision processes implemented
(see `Input <DDM_Input>` above). The decision process can be configured to execute in different modes.  The
`function <DDM.function>` parameters is the primary determinants of how the
decision process is executed, and what information is returned. The `function <DDM.function>` parameter specifies
the analytical solution to use. The number of `OutputPorts <OutputPort>` is determined by the `function <DDM.function>`
in use (see `list of output values <DDM_Results>` below).

[TBI - average_output_ports ARGUMENT/OPTION AFTER IMPLEMENTING MULTIPROCESS DDM]
OUTPUT MEASURE?? OUTCOME MEASURE?? RESULT?? TYPE OF RESULT??
If only a single decision process was run, then the value of each outputPort is the corresponding output of
the decision process.  If there is more than one decision process (i.e., the input has more than one item), then
the content of the outputPorts is determined by the ``average_output_ports`` argument.  If it is `True`,
then each outputPort (and item of ``output_values``) contains a single value, which is the average of the output
values of that type over all of the processes run.  If ``average_output_ports`` is :keyword:`False` (the default),
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
      the `DRIFT_RATE` parameterPort. The `drift_rate` parameter can be thought of as the "automatic" component
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
   :py:meth:`execute <Mechanism_Base.execte>` or :py:meth:`run <Mechanism_Base.run>` methods)
   differently than standard Components: runtime parameters are added to the Mechanism's current value of the
   corresponding ParameterPort (rather than overriding it);  that is, they are combined additively with the value of
   any `ControlProjection` it receives to determine the parameter's value for that execution.  The ParameterPort's
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
    The physics of optimal decision making: A formal analysis of models of performance in two-alternative forced-choice
    tasks. Psychological Review, 113(4), 700-765.   http://dx.doi.org/10.1037/0033-295X.113.4.700

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
import types
from collections.abc import Iterable

import numpy as np
import typecheck as tc

from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import \
    DriftDiffusionIntegrator, IntegratorFunction
from psyneulink.core.components.functions.distributionfunctions import STARTING_POINT, \
    DriftDiffusionAnalytical
from psyneulink.core.components.functions.combinationfunctions import Reduce
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import _is_control_spec
from psyneulink.core.components.mechanisms.mechanism import Mechanism_Base
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.ports.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.components.ports.outputport import SEQUENTIAL, StandardOutputPorts
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import \
    ALLOCATION_SAMPLES, FUNCTION, FUNCTION_PARAMS, INPUT_PORT_VARIABLES, NAME, OUTPUT_PORTS, OWNER_VALUE, \
    THRESHOLD, VARIABLE, PREFERENCE_SET_NAME
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set, REPORT_OUTPUT_PREF
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel
from psyneulink.core.globals.utilities import convert_to_np_array, is_numeric, is_same_function_spec, object_has_single_value, get_global_seed

from psyneulink.core import llvm as pnlvm

__all__ = [
    'ARRAY', 'DDM', 'DDMError', 'DECISION_VARIABLE', 'DECISION_VARIABLE_ARRAY',
    'PROBABILITY_LOWER_THRESHOLD', 'PROBABILITY_UPPER_THRESHOLD', 'RESPONSE_TIME',
    'RT_CORRECT_MEAN', 'RT_CORRECT_SKEW', 'RT_CORRECT_VARIANCE',
    'RT_INCORRECT_MEAN', 'RT_INCORRECT_SKEW', 'RT_INCORRECT_VARIANCE',
    'SCALAR', 'SELECTED_INPUT_ARRAY', 'VECTOR'
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
    (used to generate value of OutputPort for action_selection Mechanism
    """
    if x >= 0:
        return [x,0]
    else:
        return [0,x]


class DDMError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class DDM(ProcessingMechanism):
    # DOCUMENT:   COMBINE WITH INITIALIZATION WITH PARAMETERS
    #             ADD INFO ABOUT B VS. N&F
    #             ADD _instantiate_output_ports TO INSTANCE METHODS, AND EXPLAIN RE: NUM OUTPUT VALUES FOR B VS. N&F
    """
    DDM(                                   \
        default_variable=None,             \
        function=DriftDiffusionAnalytical)

    Implements a drift diffusion process (also known as the `Diffusion Decision Model
    <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2474742/>`_, either by calculating an `analytic solution
    <DDM_Analytic_Mode>` or carrying out `step-wise numerical integration <DDM_Integration_Mode>`.
    See `Mechanism <Mechanism_Class_Reference>` for additional arguments and attributes.


    Arguments
    ---------

    default_variable : value, list or np.ndarray : default FUNCTION_PARAMS[STARTING_POINT]
        the input to the Mechanism used if none is provided in a call to its `execute <Mechanism_Base.execute>` or
        `run <Mechanism_Base.run>` methods; also serves as a template to specify the length of the `variable
        <DDM.variable>` for its `function <DDM.function>`, and the `primary OutputPort <OuputState_Primary>` of the
        DDM (see `Input` <DDM_Creation>` for how an input with a length of greater than 1 is handled).

    function : IntegratorFunction : default DriftDiffusionAnalytical
        specifies the function to use to `execute <DDM_Execution>` the decision process; determines the mode of
        execution (see `function <DDM.function>` and `DDM_Modes` for additional information).

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
        an `IntegratorFunction` Function with an `integration_type <IntegratorFunction.integration_type>` of
        *DIFFUSION*, then `numerical step-wise integration <DDM_Integration_Mode>` is carried out.  See `DDM_Modes` and
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
        `DDM_Execution`, and `DDM_Output` for additional information about other values that can be reported and
        their interpretation.

    random_state : numpy.RandomState
        private pseudorandom number generator

    output_ports : ContentAddressableList[OutputPort]
        list of the DDM's `OutputPorts <OutputPort>`.  There are always two OutputPorts, `DECISION_VARIABLE
        <DDM_DECISION_VARIABLE>` and `RESPONSE_TIME <DDM_RESPONSE_TIME>`; additional ones may be included
        based on the `function <DDM.function>` and/or any specifications made in the **output_ports** argument of the
        DDM's constructor (see `DDM_Output` for additional details).

    output_values : List[array(float64)]
        each item is the `value <OutputPort.value>` of the corresponding `OutputPort` in `output_ports
        <DDM.output_ports>`.  The first two items are always the `value <OutputPort.value>`\\s of the
        `DECISION_VARIABLE <DDM_DECISION_VARIABLE>` and `RESPONSE_TIME <DDM_RESPONSE_TIME>` OutputPorts;  additional
        ones may be included, based on the `function <DDM.function>` and any specifications made in the
        **output_ports** argument of the DDM's constructor  (see `DDM_Output` for additional details).

    standard_output_ports : list[str]
        list of `Standard OutputPorts <OutputPort_Standard>` that includes the following addition to the
        `standard_output_ports <Mechanism_Base.standard_output_ports>` of a `Mechanism <Mechanism>`:

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

        .. _DDM_SELECTED_INPUT_ARRAY:

        *SELECTED_INPUT_ARRAY* : 1d nparray
          .. note::
             This is available only if **input_format** is specified as *ARRAY* in the DDM Mechanism's constructor
             (see `DDM_Input`).
          • `analytic mode <DDM_Analytic_Mode>`: two element array, with one ("selected") element -- determined by the
            outcome of the decision process -- set to the value of the corresponding element in the stimulus array (i.e.,
            the DDM's input_port `variable <InputPort.variable>`).  The "selected" element is the 1st one if the decision
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
            to execute in the current `TRIAL <TimeScale.TRIAL>` or, if it has reached the positive or negative value of
            the DDM `function <DDM.function>`'s threshold attribute, the `TIME_STEP` at which that occurred. \n
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
          • `analytic mode <DDM_Analytic_Mode>`:  the variance of reaction time (in seconds) for responses in which the
            decision variable reached the positive value of the DDM `function <DDM.function>`'s threshold attribute as
            estimated by closed form analytic solutions from Srivastava et al. (https://arxiv.org/abs/1601.06420) \n
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
          • `analytic mode <DDM_Analytic_Mode>`:  the variance of reaction time (in seconds) for responses in which the
            decision variable reached the negative value of the DDM `function <DDM.function>`'s threshold attribute as
            estimated by closed form analytic solutions from Srivastava et al. (https://arxiv.org/abs/1601.06420) \n
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

    componentType = "DDM"

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in SUBTYPE_DEFAULT_PREFERENCES
    classPreferences = {
        PREFERENCE_SET_NAME: 'DDMCustomClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

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
                    :type: ``numpy.ndarray``

                input_format
                    see `input_format <DDM.input_format>`

                    :default value: `SCALAR`
                    :type: ``str``

                random_state
                    see `random_state <DDM.random_state>`

                    :default value: None
                    :type: ``numpy.random.RandomState``
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
        random_state = Parameter(None, stateful=True, loggable=False)

        output_ports = Parameter(
            [DECISION_VARIABLE, RESPONSE_TIME],
            stateful=False,
            loggable=False,
            read_only=True,
            structural=True,
        )

    standard_output_ports =[{NAME: DECISION_VARIABLE,},           # Upper or lower threshold for Analtyic function
                            {NAME: RESPONSE_TIME},                # TIME_STEP within TRIAL for Integrator function
                            {NAME: PROBABILITY_UPPER_THRESHOLD},  # Accuracy (TRIAL mode only)
                            {NAME: PROBABILITY_LOWER_THRESHOLD},  # Error rate (TRIAL mode only)
                            {NAME: RT_CORRECT_MEAN},              # (DriftDiffusionAnalytical only)
                            {NAME: RT_CORRECT_VARIANCE},          # (DriftDiffusionAnalytical only)
                            {NAME: RT_CORRECT_SKEW},              # (DriftDiffusionAnalytical only)
                            {NAME: RT_INCORRECT_MEAN},            # (DriftDiffusionAnalytical only)
                            {NAME: RT_INCORRECT_VARIANCE},        # (DriftDiffusionAnalytical only)
                            {NAME: RT_INCORRECT_SKEW}             # (DriftDiffusionAnalytical only)
                            ]
    standard_output_port_names = [i['name'] for i in standard_output_ports]

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_format:tc.optional(tc.enum(SCALAR, ARRAY, VECTOR))=None,
                 function=None,
                 input_ports=None,
                 output_ports: tc.optional(tc.any(str, Iterable)) = None,
                 seed=None,
                 params=None,
                 name=None,
                 prefs: tc.optional(is_pref_set) = None,
                 **kwargs):

        # Override instantiation of StandardOutputPorts usually done in _instantiate_output_ports
        #    in order to use SEQUENTIAL indices
        self.standard_output_ports = StandardOutputPorts(self, self.standard_output_ports, indices=SEQUENTIAL)

        if seed is None:
            seed = get_global_seed()

        if input_format is not None and input_ports is not None:
            raise DDMError(
                'Only one of input_format and input_ports should be specified.'
            )
        elif input_format is None:
            input_format = SCALAR

        # If input_format is specified to be ARRAY or VECTOR, instantiate:
        #    InputPort with:
        #        2-item array as its variable
        #        Reduce as its function, which will generate an array of len 1
        #        and therefore specify size of Mechanism's variable as 1
        #    OutputPorts that report the decision variable and selected input in array format
        #        IMPLEMENTATION NOTE:
        #            These are created here rather than as StandardOutputPorts
        #            since they require input_format==ARRAY to be meaningful
        if input_format in {ARRAY, VECTOR}:
            size=1 # size of variable for DDM Mechanism
            input_ports = [
                {NAME:'ARRAY',
                 VARIABLE: np.array([[0.0, 0.0]]),
                 FUNCTION: Reduce(weights=[1,-1])}
            ]
            self.standard_output_ports.add_port_dicts([
                # Provides a 1d 2-item array with:
                #    decision variable in position corresponding to threshold crossed, and 0 in the other position
                {NAME: DECISION_VARIABLE_ARRAY, # 1d len 2, DECISION_VARIABLE as element 0 or 1
                 VARIABLE:[(OWNER_VALUE, self.DECISION_VARIABLE_INDEX), THRESHOLD],
                           # per VARIABLE assignment above, items of v of lambda function below are:
                           #    v[0]=self.value[self.DECISION_VARIABLE_INDEX]
                           #    v[1]=self.parameter_ports[THRESHOLD]
                 FUNCTION: lambda v: [float(v[0]), 0] if (v[1] - v[0]) < (v[1] + v[0]) else [0, float(v[0])]},
                # Provides a 1d 2-item array with:
                #    input value in position corresponding to threshold crossed by decision variable, and 0 in the other
                {NAME: SELECTED_INPUT_ARRAY, # 1d len 2, DECISION_VARIABLE as element 0 or 1
                 VARIABLE:[(OWNER_VALUE, self.DECISION_VARIABLE_INDEX), THRESHOLD, (INPUT_PORT_VARIABLES, 0)],
                 # per VARIABLE assignment above, items of v of lambda function below are:
                 #    v[0]=self.value[self.DECISION_VARIABLE_INDEX]
                 #    v[1]=self.parameter_ports[THRESHOLD]
                 #    v[2]=self.input_ports[0].variable
                 FUNCTION: lambda v: [float(v[2][0][0]), 0] \
                                      if (v[1] - v[0]) < (v[1] + v[0]) \
                                      else [0, float(v[2][0][1])]

                 }

            ])

        # Add StandardOutputPorts for Mechanism (after ones for DDM, so that their indices are not messed up)
        # FIX 11/9/19:  ADD BACK ONCE Mechanism_Base.standard_output_ports ONLY HAS RESULTS IN ITS
        # self.standard_output_ports.add_port_dicts(Mechanism_Base.standard_output_ports)

        # Default output_ports is specified in constructor as a tuple rather than a list
        # to avoid "gotcha" associated with mutable default arguments
        # (see: bit.ly/2uID3s3 and http://docs.python-guide.org/en/latest/writing/gotchas/)
        if isinstance(output_ports, (str, tuple)):
            output_ports = list(output_ports)

        # Instantiate RandomState
        random_state = np.random.RandomState([seed])

        # IMPLEMENTATION NOTE: this manner of setting default_variable works but is idiosyncratic
        # compared to other mechanisms: see TransferMechanism.py __init__ function for a more normal example.
        if default_variable is None and size is None:
            try:
                default_variable = params[FUNCTION_PARAMS][STARTING_POINT]
                if not is_numeric(default_variable):
                    # set normally by default
                    default_variable = None
            except (KeyError, TypeError):
                # set normally by default
                pass

        # # Conflict with above
        # self.size = size

        self.parameters.execute_until_finished.default_value = False

        super(DDM, self).__init__(default_variable=default_variable,
                                  random_state=random_state,
                                  input_ports=input_ports,
                                  output_ports=output_ports,
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
                           "input ports, which is currently not supported in DDM, but may be supported"
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

        if FUNCTION in target_set and target_set[FUNCTION] is not None:
            # If target_set[FUNCTION] is a method of a Function (e.g., being assigned in _instantiate_function),
            #   get the Function to which it belongs
            fun = target_set[FUNCTION]
            if isinstance(fun, types.MethodType):
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

    def _instantiate_plotting_functions(self, context=None):
        if "DriftDiffusionIntegrator" in str(self.function):
            self.get_axes_function = DriftDiffusionIntegrator(
                rate=self.function.defaults.rate,
                noise=self.function.defaults.noise
            ).function
            self.plot_function = DriftDiffusionIntegrator(
                rate=self.function.defaults.rate,
                noise=self.function.defaults.noise
            ).function

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
        Return current decision variable (self.outputPort.value) and other output values (self.output_ports[].value
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
            the value of the corresponding outputPort in the self.output_ports dict:
            - decision variable (float)
            - mean error rate (float)
            - mean RT (float)
            - mean, variance, and skew of RT for correct (posititive threshold) responses
            - mean, variance, and skew of RT for incorrect (negative threshold) responses

        :param self:
        :param variable (float)
        :param params: (dict)
        :param context: (str)
        :rtype self.outputPort.value: (number)
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

            return convert_to_np_array([result[0], [result[1]]])

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
            threshold = float(self.function._get_current_parameter_value(THRESHOLD, context))
            random_state = self._get_current_parameter_value(self.parameters.random_state, context)
            if random_state.rand() < return_value[self.PROBABILITY_LOWER_THRESHOLD_INDEX]:
                return_value[self.DECISION_VARIABLE_INDEX] = np.atleast_1d(-1 * threshold)
            else:
                return_value[self.DECISION_VARIABLE_INDEX] = threshold
            return return_value

    def _gen_llvm_invoke_function(self, ctx, builder, function, params, state, variable, *, tags:frozenset):
        mf_out, builder = super()._gen_llvm_invoke_function(ctx, builder, function, params, state, variable, tags=tags)

        mech_out_ty = ctx.convert_python_struct_to_llvm_ir(self.defaults.value)
        mech_out = builder.alloca(mech_out_ty)

        if isinstance(self.function, IntegratorFunction):
            # Integrator version of the DDM mechanism converts the
            # second element to a 2d array
            builder.store(builder.load(builder.gep(mf_out, [ctx.int32_ty(0),
                                                            ctx.int32_ty(0)])),
                          builder.gep(mech_out, [ctx.int32_ty(0),
                                                 ctx.int32_ty(0)]))
            builder.store(builder.load(builder.gep(mf_out, [ctx.int32_ty(0),
                                                            ctx.int32_ty(1)])),
                          builder.gep(mech_out, [ctx.int32_ty(0),
                                                 ctx.int32_ty(1),
                                                 ctx.int32_ty(0)]))
        elif isinstance(self.function, DriftDiffusionAnalytical):
            for res_idx, idx in enumerate((self.RESPONSE_TIME_INDEX,
                                           self.PROBABILITY_LOWER_THRESHOLD_INDEX,
                                           self.RT_CORRECT_MEAN_INDEX,
                                           self.RT_CORRECT_VARIANCE_INDEX,
                                           self.RT_CORRECT_SKEW_INDEX,
                                           self.RT_INCORRECT_MEAN_INDEX,
                                           self.RT_INCORRECT_VARIANCE_INDEX,
                                           self.RT_INCORRECT_SKEW_INDEX)):
                src = builder.gep(mf_out, [ctx.int32_ty(0), ctx.int32_ty(res_idx)])
                dst = builder.gep(mech_out, [ctx.int32_ty(0), ctx.int32_ty(idx)])
                builder.store(builder.load(src), dst)

            # Handle upper threshold probability
            src = builder.gep(mf_out, [ctx.int32_ty(0), ctx.int32_ty(1),
                                       ctx.int32_ty(0)])
            dst = builder.gep(mech_out, [ctx.int32_ty(0),
                ctx.int32_ty(self.PROBABILITY_UPPER_THRESHOLD_INDEX),
                ctx.int32_ty(0)])
            prob_lower_thr = builder.load(src)
            prob_upper_thr = builder.fsub(prob_lower_thr.type(1),
                                          prob_lower_thr)
            builder.store(prob_upper_thr, dst)

            # Load function threshold
            threshold_ptr = pnlvm.helpers.get_param_ptr(builder, self.function,
                                                        params, THRESHOLD)
            threshold = pnlvm.helpers.load_extract_scalar_array_one(builder,
                                                                    threshold_ptr)
            # Load mechanism state to generate random numbers
            state = builder.function.args[1]
            random_state = pnlvm.helpers.get_state_ptr(builder, self, state, "random_state")
            random_f = ctx.import_llvm_function("__pnl_builtin_mt_rand_double")
            random_val_ptr = builder.alloca(random_f.args[1].type.pointee)
            builder.call(random_f, [random_state, random_val_ptr])
            random_val = builder.load(random_val_ptr)

            # Convert ER to decision variable:
            dst = builder.gep(mech_out, [ctx.int32_ty(0),
                ctx.int32_ty(self.DECISION_VARIABLE_INDEX),
                ctx.int32_ty(0)])
            thr_cmp = builder.fcmp_ordered("<", random_val, prob_lower_thr)
            neg_threshold = builder.fsub(threshold.type(-0.0), threshold)
            res = builder.select(thr_cmp, neg_threshold, threshold)

            builder.store(res, dst)
        else:
            assert False, "Unknown mode in compiled DDM!"

        return mech_out, builder

    @handle_external_context(fallback_most_recent=True)
    def reset(self, *args, force=False, context=None, **kwargs):
        from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import IntegratorFunction

        # (1) reset function, (2) update mechanism value, (3) update output ports
        if isinstance(self.function, IntegratorFunction):
            new_values = self.function.reset(*args, **kwargs, context=context)
            self.parameters.value._set(convert_to_np_array(new_values), context)
            self._update_output_ports(context=context)

    @handle_external_context()
    def is_finished(self, context=None):
        # find the single numeric entry in previous_value
        try:
            single_value = self.function.parameters.previous_value._get(context)
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
            abs(single_value) >= self.function._get_current_parameter_value(THRESHOLD, context)
            and isinstance(self.function, IntegratorFunction)
        ):
            logger.info(
                '{0} {1} has reached threshold {2}'.format(
                    type(self).__name__,
                    self.name,
                    self.function._get_current_parameter_value(THRESHOLD, context)
                )
            )
            return True
        return False

    def _gen_llvm_is_finished_cond(self, ctx, builder, params, state):
        # Setup pointers to internal function
        func_state_ptr = pnlvm.helpers.get_state_ptr(builder, self, state, "function")
        func_param_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, "function")

        # Find the single numeric entry in previous_value
        try:
            prev_val_ptr = pnlvm.helpers.get_state_ptr(builder, self.function, func_state_ptr, "previous_value")
        except ValueError:
            return pnlvm.ir.IntType(1)(1)

        # Extract scalar value from ptr
        prev_val_ptr = builder.gep(prev_val_ptr, [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(0)])
        prev_val = builder.load(prev_val_ptr)

        # Take abs of previous val
        llvm_fabs = ctx.get_builtin("fabs", [ctx.float_ty])
        prev_val = builder.call(llvm_fabs, [prev_val])


        # obtain threshold value
        threshold_ptr = pnlvm.helpers.get_param_ptr(builder,
                                                    self.function,
                                                    func_param_ptr,
                                                    "threshold")

        threshold_ptr = builder.gep(threshold_ptr, [ctx.int32_ty(0), ctx.int32_ty(0)])
        threshold = builder.load(threshold_ptr)
        is_prev_greater_or_equal = builder.fcmp_ordered('>=', prev_val, threshold)

        return is_prev_greater_or_equal
