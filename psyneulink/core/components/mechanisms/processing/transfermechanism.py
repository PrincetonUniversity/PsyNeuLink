# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# NOTES:
#  * COULD NOT IMPLEMENT integrator_function in paramClassDefaults (see notes below)
#  * NOW THAT NOISE AND INTEGRATION_RATE ARE PROPRETIES THAT DIRECTLY REFERERNCE integrator_function,
#      SHOULD THEY NOW BE VALIDATED ONLY THERE (AND NOT IN TransferMechanism)??
#  * ARE THOSE THE ONLY TWO integrator PARAMS THAT SHOULD BE PROPERTIES??

# ********************************************  TransferMechanism ******************************************************

"""
..
Sections
--------
  * `Transfer_Overview`
  * `Transfer_Creation`
  * `Transfer_Execution`
  * `Transfer_Class_Reference`

.. _Transfer_Overview:

Overview
--------

A TransferMechanism is a subclass of `ProcessingMechanism` that adds the ability to integrate its input.

Like a ProcessingMechanism, it transforms its input using a simple mathematical function, that maintains the form
(dimensionality) of its input.  The input can be a single scalar value, a multidimensional array (list or numpy
array), or several independent ones. The function used to carry out the transformation can be selected from the
following PsyNeuLink `Functions <Function>`: `Linear`, `Exponential`, `Logistic`, or `SoftMax`.

Its **integrator_mode** argument can switch the transformation from an "instantaneous"  to a "time averaged"
(integrated) manner of execution. When `integrator_mode <TransferMechanism.integrator_mode>` is set to True, the
mechanism's input is first transformed by its `integrator_function <TransferMechanism.integrator_function>` (
`AdaptiveIntegrator`). That result is then transformed by the mechanism's `function <TransferMechanism.function>`.

.. _Transfer_Creation:

Creating a TransferMechanism
-----------------------------

A TransferMechanism is created by calling its constructor.

Its `function <TransferMechanism.function>` is specified in the **function** argument, which can be the name of a
`Function <Function>` class:

    >>> import psyneulink as pnl
    >>> my_linear_transfer_mechanism = pnl.TransferMechanism(function=pnl.Linear)

in which case all of the function's parameters will be set to their default values. Alternatively, the **function**
argument can be a call to a Function constructor, in which case values may be specified for the Function's parameters:

    >>> my_logistic_transfer_mechanism = pnl.TransferMechanism(function=pnl.Logistic(gain=1.0, bias=-4))

Next, the **integrator_mode** argument allows the TransferMechanism to operate in either an "instantaneous" or
"time averaged" manner. By default, `integrator_mode <TransferMechanism.integrator_mode>` is set to False, meaning
execution is instantaneous. In order to switch to time averaging, the **integrator_mode** argument of the constructor
must be set to True.

    >>> my_logistic_transfer_mechanism = pnl.TransferMechanism(function=pnl.Logistic(gain=1.0, bias=-4),
    ...                                                        integrator_mode=True)

When `integrator_mode <TransferMechanism.integrator_mode>` is True, the TransferMechanism uses its `integrator_function
<TransferMechanism.integrator_function>` to integrate its variable on each execution. The output of the
`integrator_function  <TransferMechanism.integrator_function>` is then used as the input to `function
<TransferMechanism.function>`.

By default, the `integrator_function <TransferMechanism.integrator_function>` of a TransferMechanism is
`AdaptiveIntegrator`.  However, any `IntegratorFunction` can be assigned. A TransferMechanism has three
parameters that
are used by most IntegratorFunctions:  `initial_value <TransferMechanism.initial_value>`, `integration_rate
<TransferMechanism.integration_rate>`, and `noise <TransferMechanism.noise>`.  If any of these are specified in the
TransferMechanism's constructor, their value is used to specify the corresponding parameter of its `integrator_function
<TransferMechanism.integrator_function>`.  In the following example::

    >>> my_logistic_transfer_mechanism = pnl.TransferMechanism(function=pnl.Logistic(gain=1.0, bias=-4),
    ...                                                        integrator_mode=True,
    ...                                                        initial_value=np.array([[0.2]]),
    ...                                                        integration_rate=0.1)

``my_logistic_transfer_mechanism`` will be assigned an `AdaptiveIntegrator` (the default) as its
`integrator_function
<TrasnferMechanism.integrator_function>`, with ``0.2`` as its `initializer <AdaptiveIntegrator.initializer>`
parameter,
and ``0.`` as its `rate <AdaptiveIntegrator.rate>` parameter.  However, in this example::

    >>> my_logistic_transfer_mechanism = pnl.TransferMechanism(function=pnl.Logistic(gain=1.0, bias=-4),
    ...                                                        integrator_mode=True,
    ...                                                        integrator_function=AdaptiveIntegrator(rate=0.3),
    ...                                                        initial_value=np.array([[0.2]]),
    ...                                                        integration_rate=0.1)

the AdaptiveIntegrator's `rate <AdaptiveIntegrator.rate>` parameter will be assigned ``0.3``,
and this will also
be assigned to the TransferMechanism's `integration_rate <TransferMechanism.integration_rate>` parameter, overriding
the specified value of ``0.1``.

.. note::
    If `integrator_mode <TransferMechanism.integrator_mode>` is False, then the arguments **integration_rate** and
    **initial_value** are ignored, as its `integrator_function <TransferMechanism.integrator_function>` is not executed.

When switching between `integrator_mode <TransferMechanism.integrator_mode>` = True and `integrator_mode
<TransferMechanism.integrator_mode>` = False, the behavior of the `integrator_function
<TransferMechanism.integrator_function>` is determined by `on_resume_integrator_mode
<TransferMechanism.on_resume_integrator_mode>`. There are three options for how the `integrator_function
<TransferMechanism.integrator_function>` may resume accumulating when the Mechanism returns to `integrator_mode
<TransferMechanism.integrator_mode>` = True.

        (1)     INSTANTANEOUS_MODE_VALUE - reinitialize the Mechanism with its own current value, so that the value
        computed by
                the Mechanism during "Instantaneous Mode" is where the `integrator_function
                <TransferMechanism.integrator_function>` begins accumulating.

        (2)     INTEGRATOR_MODE_VALUE - resume accumulation wherever the `integrator_function
                <TransferMechanism.integrator_function>` left off the last time `integrator_mode
                <TransferMechanism.integrator_mode>` was True.

        (3)     REINITIALIZE - call the `integrator_function's <TransferMechanism.integrator_function>` `reinitialize
        method
                <AdaptiveIntegrator.reinitialize>` so that accumulation Mechanism begins at `initial_value
                <TransferMechanism.initial_value>`

Finally, the TransferMechanism has two arguments that can adjust the final result of the mechanism: **clip** and
**noise**. If `integrator_mode <TransferMechanism.integrator_mode>` is False, `clip <TransferMechanism.clip>` and
`noise <TransferMechanism.noise>` modify the value returned by the mechanism's `function <TransferMechanism.function>`
before setting it as the mechanism's value. If `integrator_mode <TransferMechanism.integrator_mode>` is True,
**noise** is assigned to the TransferMechanism's `integrator_function <TransferMechanism.integrator_function>`
(as its `noise <IntegratorFunction.noise>` parameter -- in the same manner as `integration_rate
<TransferMechanism.integration_rate>` and `initial_value <TransferMechanism.intial_value>`), whereas `clip
<TransferMechanism.clip>` modifies the value returned by the mechanism's `function <TransferMechanism.function>`
before setting it as the TransferMechanism's `value <TransferMechanism.value>`.

.. _Transfer_Structure:

Structure
---------

.. _TransferMechanism_InputStates:

*InputStates*
~~~~~~~~~~~~~

By default, a TransferMechanism has a single `InputState`;  however, more than one can be specified
using the **default_variable** or **size** arguments of its constructor (see `Mechanism`).  The `value
<InputState.value>` of each InputState is used as a separate item of the Mechanism's `variable
<TransferMechanism.variable>`, and transformed independently by its `function <TransferMechanism.function>`.
Like any InputStates, the `value <OutputState.value>` of any or all of the TransferMechanism's InputStates can be
modulated by one or more `GatingSignals <GatingSignal_Modulation>` prior to transformation by its `function
<TransferMechanism.function>`.

.. _TransferMechanism_Function:

*Function*
~~~~~~~~~~

*Function*.  The default function for a TransferMechanism is `Linear`.  A custom function can be specified in the
**function** argument of the constructor.  This can be any PsyNeuLink `Function <Function>` that is a subtype of
either `TransferFunction` or `NormalizationFunction.` It can also be any python function or method, with the constraint
that it returns an output that is identical in shape to its input;  the function or method is "wrapped" as
`UserDefinedFunction`, and assigned as the TransferMechanism's `function <TransferMechanism.function>` attribute.

The result of the `function <TransferMechanism.function>` applied to the `value <InputState.value>` of each InputState
is:
    - appended to an array that represents the TransferMechanism's `value <TransferMechanism.value>`
    - assigned as the `value <OutputState.value>` of the TransferMechanism's corresponding `OutputState <OutputState>`

.. _TransferMechanism_OutputStates:

*OutputStates*
~~~~~~~~~~~~~~

By default, a TransferMechanism generates one `OutputState` for each of its `InputStates`.  The first (and `primary
<OutputState_Primary>`) OutputState is named *RESULT*; subsequent ones use that as the base name, suffixed with an
incrementing integer starting at '-1' for each additional OutputState (e.g., *RESULT-1*, *RESULT-2*, etc.; see
`Naming`). The `value <OutputState.value>` of each OutputState is assigned the result of the Mechanism's `function
<TransferMechanism.function>` applied to the `value <InputState.value>` of the corresponding InputState.

Additional OutputStates can be assigned using the TransferMechanism's `Standard OutputStates
<TransferMechanism_Standard_OutputStates>` (see `OutputState_Standard`) or by creating `custom OutputStates
<OutputState_Customization>` (but see note below).  Like any OutputStates, the `value <OutputState.value>` of any or
all of these can be modulated by one or more `GatingSignals <GatingSignal_Modulation>`.

    .. _TransferMechanism_OutputStates_Note:

    .. note::
       If any OutputStates are specified in the **output_states** argument of the TransferMechanism's constructor,
       then, `as with any Mechanism <Mechanism_Default_State_Suppression_Note>`, its default OutputStates are not
       automatically generated.  Therefore, an OutputState with the appropriate `index <OutputState.index>` must be
       explicitly specified for each and every item of the Mechanism's `value <TransferMechanism.value>` (corresponding
       to each InputState) for which an OutputState is needed.

.. _Transfer_Execution:

Execution
---------

COMMENT:
DESCRIBE AS TWO MODES (AKIN TO DDM):  INSTANTANEOUS AND TIME-AVERAGED
INSTANTANEOUS:
input transformed in a single `execution <Transfer_Execution>` of the Mechanism)
TIME-AVERAGED:
input transformed using `step-wise` integration, in which each execution returns the result of a subsequent step of the
integration process).
COMMENT

When a TransferMechanism is executed, it transforms its input using its `function <TransferMechanism.function>` and
the following parameters (in addition to any specified for the `function <TransferMechanism.function>`):


    * `integrator_mode <TransferMechanism.integrator_mode>`: determines whether the input will be time-averaged before
      passing through the function of the mechanism. When `integrator_mode <TransferMechanism.integrator_mode>` is set
      to True, the TransferMechanism integrates its input, by executing its `integrator_function
      <TransferMechanism.integrator_function>`, before executing its `function <TransferMechanism.function>`. When
      `integrator_mode <TransferMechanism.integrator_mode>` is False, the `integrator_function
      <TransferMechanism.integrator_function>` is ignored, and time-averaging does not occur.

    * `integration_rate <TransferMechanism.integration_rate>`: if the `integrator_mode
    <TransferMechanism.integrator_mode>`
      attribute is set to True, the `integration_rate <TransferMechanism.integration_rate>` attribute is the rate of
      integration (a higher value specifies a faster rate); if `integrator_mode <TransferMechanism.integrator_mode>`
      is False,
      `integration_rate <TransferMechanism.integration_rate>` is ignored and time-averaging does not occur.

    * `noise <TransferMechanism.noise>`: applied element-wise to the output of its `integrator_function
      <TransferMechanism.integrator_function>` or its `function <TransferMechanism.function>`, depending on whether
      `integrator_mode <TransferMechanism.integrator_mode>` is True or False.

    * `clip <TransferMechanism.clip>`: caps all elements of the `function <TransferMechanism.function>` result by the
      lower and upper values specified by clip.

After each execution of the Mechanism the result of `function <TransferMechanism.function>` applied to each
`InputState` is assigned as an item of the Mechanism's `value <TransferMechanism.value>`, and the `value
<OutputState.value>` of each of its `OutputStates <OutputState>`, and to the 1st item of the Mechanism's
`output_values <TransferMechanism.output_values>` attribute.


.. _Transfer_Reinitialization:

*Reinitialization*
~~~~~~~~~~~~~~~~~~

In some cases, it may be useful to reset the accumulation of a mechanism back to its original starting point, or a new
starting point. This is done using the `reinitialize <AdaptiveIntegrator.reinitialize>` method on the
mechanism's
`integrator_function <TransferMechanism.integrator_function>`, or the mechanisms's own `reinitialize
<TransferMechanism.reinitialize>` method.

The `reinitialize <AdaptiveIntegrator.reinitialize>` method of the `integrator_function
<TransferMechanism.integrator_function>` sets:

    - the integrator_function's `previous_value <AdaptiveIntegrator.previous_value>` attribute
    - the integrator_function's `value <AdaptiveIntegrator.value>` attribute

    to the specified value.

The `reinitialize <TransferMechanism.reinitialize>` method of the `TransferMechanism` first sets:

    - the integrator_function's `previous_value <AdaptiveIntegrator.previous_value>` attribute
    - the integrator_function's `value <AdaptiveIntegrator.value>` attribute

    to the specified value. Then:

    - the specified value is passed into the mechanism's `function <TransferMechanism.function>` and the function is
    executed
    - the TransferMechanism's `value <TransferMechanism.value>` attribute is set to the output of the function
    - the TransferMechanism updates its `output_states <TransferMechanism.output_states>`

A use case for `reinitialize <AdaptiveIntegrator.reinitialize>` is demonstrated in the following example:

Create a `System` with a TransferMechanism in integrator_mode:

    >>> my_time_averaged_transfer_mechanism = pnl.TransferMechanism(function=pnl.Linear,        #doctest: +SKIP
    ...                                                        integrator_mode=True,            #doctest: +SKIP
    ...                                                        integration_rate=0.1,            #doctest: +SKIP
    ...                                                        initial_value=np.array([[0.2]])) #doctest: +SKIP
    >>> my_process = pnl.Process(pathway=[my_time_averaged_transfer_mechanism]) #doctest: +SKIP
    >>> my_system = pnl.System(processes=[my_process])  #doctest: +SKIP

Then run the system for 5 trials:

    >>> # RUN 1:
    >>> my_system.run(inputs={my_time_averaged_transfer_mechanism: [1.0]},        #doctest: +SKIP
    ...               num_trials=5)                                               #doctest: +SKIP
    >>> assert np.allclose(my_time_averaged_transfer_mechanism.value,  0.527608)  #doctest: +SKIP

After RUN 1, my_time_averaged_transfer_mechanism's integrator_function will preserve its state (its position along its
path of integration).

Run the system again to observe that my_time_averaged_transfer_mechanism's integrator_function continues accumulating
where it left off:

    >>> # RUN 2:
    >>> my_system.run(inputs={my_time_averaged_transfer_mechanism: [1.0]},          #doctest: +SKIP
    ...               num_trials=5)                                                 #doctest: +SKIP
    >>> assert np.allclose(my_time_averaged_transfer_mechanism.value,  0.72105725)  #doctest: +SKIP

The integrator_function's `reinitialize <AdaptiveIntegrator.reinitialize>` method and the TransferMechanism's
`reinitialize <TransferMechanism.reinitialize>` method are useful in cases when the integration should instead start
over at the original initial value, or a new one.

Use `reinitialize <AdaptiveIntegrator.reinitialize>` to re-start the integrator_function's accumulation at 0.2:

    >>> my_time_averaged_transfer_mechanism.integrator_function.reinitialize(np.array([[0.2]]))  #doctest: +SKIP

Run the system again to observe that my_time_averaged_transfer_mechanism's integrator_function will begin accumulating
at 0.2, following the exact same trajectory as in RUN 1:

    >>> # RUN 3
    >>> my_system.run(inputs={my_time_averaged_transfer_mechanism: [1.0]},        #doctest: +SKIP
    ...               num_trials=5)                                               #doctest: +SKIP
    >>> assert np.allclose(my_time_averaged_transfer_mechanism.value,  0.527608)  #doctest: +SKIP

Because `reinitialize <AdaptiveIntegrator.reinitialize>` was set to 0.2 (its original initial_value),
my_time_averaged_transfer_mechanism's integrator_function effectively started RUN 3 in the same state as it began RUN 1.
As a result, it arrived at the exact same value after 5 trials (with identical inputs).

In the examples above, `reinitialize <AdaptiveIntegrator.reinitialize>` was applied directly to the
integrator function.
The key difference between the `integrator_function's reinitialize <AdaptiveIntegrator.reinitialize>` and the
`TransferMechanism's reinitialize <TransferMechanism.reinitialize>` is that the latter will also execute the mechanism's
function and update its output states. This is useful if the mechanism's value or any of its output state values will
be used or checked *before* the mechanism's next execution. (This may be true if, for example, the mechanism is
`recurrent <RecurrentTransferMechanism>`, the mechanism is responsible for `modulating <ModulatorySignal_Modulation`
other components, or if a `Scheduler` condition depends on the mechanism's activity.)

COMMENT:
.. _Transfer_Examples:

Examples
--------

EXAMPLES HERE
COMMENT

.. _Transfer_Class_Reference:

Class Reference
---------------

"""
import inspect
import itertools
import numbers
import warnings

from collections.abc import Iterable

import numpy as np
import typecheck as tc

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.component import function_type, method_type
from psyneulink.core.components.functions.distributionfunctions import DistributionFunction
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import AdaptiveIntegrator
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import IntegratorFunction
from psyneulink.core.components.functions.transferfunctions import TransferFunction, Linear, Logistic
from psyneulink.core.components.functions.function import Function, is_function_type
from psyneulink.core.components.functions.objectivefunctions import Distance
from psyneulink.core.components.functions.selectionfunctions import SelectionFunction
from psyneulink.core.components.functions.transferfunctions import Linear, Logistic, TransferFunction
from psyneulink.core.components.functions.userdefinedfunction import UserDefinedFunction
from psyneulink.core.components.mechanisms.adaptive.control.controlmechanism import _is_control_spec
from psyneulink.core.components.mechanisms.mechanism import Mechanism, MechanismError
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.core.components.states.inputstate import InputState
from psyneulink.core.components.states.outputstate import OutputState, PRIMARY, StandardOutputStates, standard_output_states
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import DIFFERENCE, FUNCTION, INITIALIZER, INSTANTANEOUS_MODE_VALUE, \
    MAX_ABS_INDICATOR, MAX_ABS_VAL, MAX_INDICATOR, MAX_VAL, NAME, NOISE, OUTPUT_MEAN, OUTPUT_MEDIAN, OUTPUT_STD_DEV, OUTPUT_VARIANCE, OWNER_VALUE, PREVIOUS_VALUE, PROB, RATE, REINITIALIZE, RESULT, RESULTS, SELECTION_FUNCTION_TYPE, TRANSFER_FUNCTION_TYPE, TRANSFER_MECHANISM, VARIABLE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import all_within_range, append_type_to_name, iscompatible
from psyneulink.core.scheduling.condition import Never

__all__ = [
    'INITIAL_VALUE', 'CLIP',  'INTEGRATOR_FUNCTION', 'INTEGRATION_RATE', 'Transfer_DEFAULT_BIAS',
    'Transfer_DEFAULT_GAIN', 'Transfer_DEFAULT_LENGTH', 'Transfer_DEFAULT_OFFSET', 'TRANSFER_OUTPUT',
    'TransferError', 'TransferMechanism',
]

# TransferMechanism parameter keywords:
CLIP = "clip"
INTEGRATOR_FUNCTION = 'integrator_function'
INTEGRATION_RATE = "integration_rate"
INITIAL_VALUE = 'initial_value'

# TransferMechanism default parameter values:
Transfer_DEFAULT_LENGTH = 1
Transfer_DEFAULT_GAIN = 1
Transfer_DEFAULT_BIAS = 0
Transfer_DEFAULT_OFFSET = 0
# Transfer_DEFAULT_RANGE = np.array([])

# This is a convenience class that provides list of standard_output_state names in IDE
class TRANSFER_OUTPUT():
    """
    .. _TransferMechanism_Standard_OutputStates:

    `Standard OutputStates <OutputState_Standard>` for `TransferMechanism`: \n

    .. _TRANSFER_MECHANISM_RESULT:

    *RESULT* : 1d np.array
      first item of TransferMechanism's `value <TransferMechanism.value>` (corresponding to input from its
      first InputState)

    *RESULTS* : 2d np.array
      each item of TransferMechanism's `value <TransferMechanism.value>` (corresponding to input from each
      of its `input_states <TransferMechanism.input_states>`) is assigned as the `value <OutputState.value>`
      of a corresponding OutputState of its `output_states <TransferMechanism.output_states>`.

    .. _TRANSFER_MECHANISM_MEAN:

    *OUTPUT_MEAN* : float
      mean of `value <TransferMechanism.value>`.

    .. _TRANSFER_MECHANISM_MEDIAN:

    *OUTPUT_MEDIAN* : float
      median of `value <TransferMechanism.value>`.

    .. _TRANSFER_MECHANISM_STD_DEV:

    *OUTPUT_STD_DEV* : float
      standard deviation of `value <TransferMechanism.value>`.

    .. _TRANSFER_MECHANISM_VARIANCE:

    *OUTPUT_VARIANCE* : float
      variance of `output_state.value`.

    *MECHANISM_VALUE* : list
      TransferMechanism's `value <TransferMechanism.value>` used as OutputState's value.

    COMMENT:
    *COMBINE* : scalar or numpy array
      linear combination of the `value <TransferMechanism.value>` of all items of the TransferMechanism's `value
      <TransferMechanism.value>` (requires that they all have the same dimensionality).
    COMMENT

    """

    RESULTS=RESULTS
    RESULT=RESULT
    MEAN=OUTPUT_MEAN
    MEDIAN=OUTPUT_MEDIAN
    STANDARD_DEVIATION=OUTPUT_STD_DEV
    VARIANCE=OUTPUT_VARIANCE
    MAX_VAL=MAX_VAL
    MAX_ABS_VAL=MAX_ABS_VAL
    MAX_INDICATOR=MAX_INDICATOR
    MAX_ABS_INDICATOR=MAX_ABS_INDICATOR
    PROB=PROB

# THE FOLLOWING WOULD HAVE BEEN NICE, BUT IDE DOESN'T EXECUTE IT, SO NAMES DON'T SHOW UP
# for item in [item[NAME] for item in DDM_standard_output_states]:
#     setattr(DDM_OUTPUT.__class__, item, item)

class TransferError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


def _integrator_mode_setter(value, owning_component=None, context=None):
    if value is True:
        if (
            not owning_component.parameters.integrator_mode._get(context)
            and owning_component.parameters.has_integrated._get(context)
        ):
            if owning_component.integrator_function is not None:
                if owning_component.on_resume_integrator_mode == INSTANTANEOUS_MODE_VALUE:
                    owning_component.reinitialize(owning_component.parameters.value._get(context), context=context)
                elif owning_component.on_resume_integrator_mode == REINITIALIZE:
                    owning_component.reinitialize(context=context)
            owning_component._parameter_components.add(owning_component.integrator_function)
        owning_component.parameters.has_initializers._set(True, context)
    elif value is False:
        owning_component.parameters.has_initializers._set(False, context)
        if not hasattr(owning_component, "reinitialize_when"):
            owning_component.reinitialize_when = Never()

    return value


# IMPLEMENTATION NOTE:  IMPLEMENTS OFFSET PARAM BUT IT IS NOT CURRENTLY BEING USED
class TransferMechanism(ProcessingMechanism_Base):
    """
    TransferMechanism(                                                            \
    default_variable=None,                                                        \
    size=None,                                                                    \
    input_states=None,                                                            \
    function=Linear,                                                              \
    integrator_mode=False,                                                        \
    integrator_function=AdaptiveIntegrator,                                       \
    on_resume_integrator_mode=INSTANTANEOUS_MODE_VALUE,                           \
    initial_value=None,                                                           \
    integration_rate=0.5,                                                         \
    noise=0.0,                                                                    \
    clip=[float:min, float:max],                                                  \
    convergence_function=Distance(metric=DIFFERENCE),                             \
    convergence_criterion=None,                                                   \
    max_passes=None,                                                              \
    output_states=RESULTS                                                         \
    params=None,                                                                  \
    name=None,                                                                    \
    prefs=None)

    Subclass of `ProcessingMechanism <ProcessingMechanism>` that performs a simple transform of its input.

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
            + class_defaults.variable (value):  Transfer_DEFAULT_BIAS

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

    default_variable : number, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the input to the Mechanism to use if none is provided in a call to its
        `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` method;
        also serves as a template to specify the length of `variable <TransferMechanism.variable>` for
        `function <TransferMechanism.function>`, and the `primary outputState <OutputState_Primary>`
        of the Mechanism.

    size : int, list or np.ndarray of ints
        specifies default_variable as array(s) of zeros if **default_variable** is not passed as an argument;
        if **default_variable** is specified, it takes precedence over the specification of **size**.
        As an example, the following mechanisms are equivalent::
            T1 = TransferMechanism(size = [3, 2])
            T2 = TransferMechanism(default_variable = [[0, 0, 0], [0, 0]])

    input_states : str, list, dict, or np.ndarray
        specifies the InputStates for the TransferMechanism; by default, a single InputState is created using the
        value of default_variable as its `variable <InputState.variable>`;  if more than one is specified, the number
        and, if specified, their values must be compatible with any specifications in **default_variable** or
        **size** (see `Mechanism_InputStates`);  see `input_states <TransferMechanism.output_states>` for additional
        details.

    function : TransferFunction : default Linear
        specifies the function used to transform the input;  can be `Linear`, `Logistic`, `Exponential`,
        or a custom function.

    integrator_mode : bool : False
        specifies whether or not the TransferMechanism should be executed using its `integrator_function
        <TransferMechanism>` to integrate its `variable <TransferMechanism.variable>` (
        when set to `True`), or simply report the asymptotic value of the output of its `function
        <TransferMechanism.function>` (when set to `False`).

    integrator_function : IntegratorFunction : default AdaptiveIntegrator
        specifies `IntegratorFunction` to use in `integration_mode <TransferMechanism.integration_mode>`.

    initial_value :  value, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the starting value for time-averaged input (only relevant if `integrator_mode
        <TransferMechanism.integrator_mode>` is True).
        COMMENT:
            Transfer_DEFAULT_BIAS SHOULD RESOLVE TO A VALUE
        COMMENT

    integration_rate : float : default 0.5
        specifies the rate of integration of `variable <TransferMechanism.variable>` when the TransferMechanism is
        executed with `integrator_mode` set to `True`.

    on_resume_integrator_mode : keyword : default INSTANTANEOUS_MODE_VALUE
        specifies how the `integrator_function <TransferMechanism.integrator_function>` should resume its accumulation
        when the Mechanism was most recently in "Instantaneous Mode" (integrator_mode = False) and has just switched to
        "IntegratorFunction Mode" (integrator_mode = True).

        (1)     INSTANTANEOUS_MODE_VALUE - reinitialize the Mechanism with its own current value, so that the value computed by
                the Mechanism during "Instantaneous Mode" is where the `integrator_function
                <TransferMechanism.integrator_function>` begins accumulating.

        (2)     INTEGRATOR_MODE_VALUE - resume accumulation wherever the `integrator_function
                <TransferMechanism.integrator_function>` left off the last time `integrator_mode
                <TransferMechanism.integrator_mode>` was True.

        (3)     REINITIALIZE - call the `integrator_function's <TransferMechanism.integrator_function>` `reinitialize method
                <AdaptiveIntegrator.reinitialize>` so that accumulation Mechanism begins at `initial_value
                <TransferMechanism.initial_value>`

    noise : float or function : default 0.0
        specifies a value to be added to the result of the TransferMechanism's `function <TransferMechanism.function>`
        or its `integrator_function <TransferMechanism.integrator_function>`, depending on whether `integrator_mode
        <TransferMechanism.integrator_mode>` is `True` or `False`. See `noise <TransferMechanism.noise>` for details.

    clip : list [float, float] : default None (Optional)
        specifies the allowable range for the result of `function <TransferMechanism.function>`. The item in index 0
        specifies the minimum allowable value of the result, and the item in index 1 specifies the maximum allowable
        value; any element of the result that exceeds the specified minimum or maximum value is set to the value of
        `clip <TransferMechanism.clip>` that it exceeds.

    convergence_function : function : default Distance(metric=DIFFERENCE)
        specifies the function that calculates `delta <TransferMechanism.delta>`, and determines when `is_converged
        <TransferMechanism.is_converged>` is `True`.

    convergence_criterion : float : default 0.01
        specifies the value of `delta <TransferMechanism.delta>` at which `is_converged
        <TransferMechanism.is_converged>` is `True`.

    max_passes : int : default 1000
        specifies maximum number of executions (`passes <TimeScale.PASS>`) that can occur in a trial before reaching
        the `convergence_criterion <RecurrentTransferMechanism.convergence_criterion>`, after which an error occurs;
        if `None` is specified, execution may continue indefinitely or until an interpreter exception is generated.

    output_states : str, list or np.ndarray : default RESULTS
        specifies the OutputStates for the TransferMechanism; by default, one is created for each InputState
        specified in **input_states**;  see `note <TransferMechanism_OutputStates_Note>`, and `output_states
        <TransferMechanism.output_states>` for additional details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the Mechanism, its `function <Mechanism_Base.function>`, and/or a custom function and its parameters.  Values
        specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default see `name <TransferMechanism.name>`
        specifies the name of the TransferMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the TransferMechanism; see `prefs <TransferMechanism.prefs>` for details.

    context : str : default componentType+INITIALIZING
        string used for contextualization of instantiation, hierarchical calls, executions, etc.

    Attributes
    ----------

    variable : value
        the input to Mechanism's `function <TransferMechanism.function>`.
        COMMENT:
            :py:data:`Transfer_DEFAULT_BIAS <LINK->SHOULD RESOLVE TO VALUE>`
        COMMENT

    input_states : *ContentAddressableList[InputState]*
        list of Mechanism's `InputStates <InputStates>` (see `TransferMechanism_InputStates` for additional details).

    function : Function
        the Function used to transform the input.

    integrator_mode : bool
        determines whether the TransferMechanism uses its `integrator_function <TransferMechanism.integrator_function>`
        to integrate its `variable <TransferMechanism.variable>` when it executes.

        **If integrator_mode is set to** `True`:

            the TransferMechanism's `variable <TransferMechanism>` is first passed to its `integrator_function
            <TransferMechanism.integrator_function>`, and then the result is passed to the TransferMechanism's
            `function <TransferMechanism.function>` which computes the TransferMechanism's `value
            <TransferMechanism.value>`.

            .. note::
                The TransferMechanism's `integration_rate <TransferMechanism.integration_rate>`, `noise
                <TransferMechanism.noise>`, and `initial_value <TransferMechanism.initial_value>` parameters
                specify the respective parameters of its `integrator_function <TransferMechanism.integrator_function>`
                (with `initial_value <TransferMechanism.initial_value>` corresponding to `initializer
                <IntegratorFunction.initializer>` and `integration_rate <TransferMechanism.integration_rate>`
                corresponding to `rate <IntegratorFunction.rate>` of `integrator_function
                <TransferMechanism.integrator_function>`). However, if there are any disagreements between these
                (e.g., any of these parameters is specified in the constructor for an `IntegratorFunction` assigned
                as the **integration_function** arg of the TransferMechanism), the values specified for the
                `integrator_function <TransferMechanism.integrator_function>` take precedence, and their value(s) are
                assigned as those of the corresponding parameters on the TransferMechanism.

        **If integrator_mode is set to** `False`:

            if `noise <TransferMechanism.noise>` is non-zero, it is applied to the TransferMechanism's `variable
            <TransferMechanism>` which is htne passed directly to its `function <TransferMechanism.function>`
             -- that is, its `integrator_function <TransferMechanism.integrator_function>` is bypassed,
             and its related attributes (`initial_value <TransferMechanism.initial_value>` and
            `integration_rate <TransferMechanism.integration_rate>`) are ignored.

    integrator_function :  IntegratorFunction
        the `IntegratorFunction` used when `integrator_mode <TransferMechanism.integrator_mode>` is set to
        `True` (see `integrator_mode <TransferMechanism.integrator_mode>` for details).

    initial_value :  value, list or np.ndarray
        specifies the starting value for the `integration_function <TransferMechanism.integrator_function>`;  only
        relevant if `integrator_mode <TransferMechanism.integrator_mode>` is `True` and `integration_rate
        <TransferMechanism.integration_rate>` is not 1.0 (see `integrator_mode <TransferMechanism.integrator_mode>`
        for additional details).

    integration_rate : float
        the rate at which the TransferMechanism's `variable <TransferMechanism>` is integrated when it is executed with
        `integrator_mode <TransferMechanism.integrator_mode>` set to `True` (see `integrator_mode
        <TransferMechanism.integrator_mode>` for additional details).

    on_resume_integrator_mode : keyword
        specifies how the `integrator_function <TransferMechanism.integrator_function>` should resume its accumulation
        when the Mechanism was most recently in "Instantaneous Mode" (integrator_mode = False) and has just switched to
        "IntegratorFunction Mode" (integrator_mode = True). There are three options:

        (1)     INSTANTANEOUS_MODE_VALUE - reinitialize the Mechanism with its own current value, so that the value computed by
                the Mechanism during "Instantaneous Mode" is where the `integrator_function
                <TransferMechanism.integrator_function>` begins accumulating.

        (2)     INTEGRATOR_MODE_VALUE - resume accumulation wherever the `integrator_function
                <TransferMechanism.integrator_function>` left off the last time `integrator_mode
                <TransferMechanism.integrator_mode>` was True.

        (3)     REINITIALIZE - call the `integrator_function's <TransferMechanism.integrator_function>` `reinitialize method
                <AdaptiveIntegrator.reinitialize>` so that accumulation Mechanism begins at `initial_value
                <TransferMechanism.initial_value>`

    noise : float or function
        When `integrator_mode <TransferMechanism.integrator_mode>` is set to `True`, `noise <TransferMechanism.noise>`
        is passed into the `integrator_function <TransferMechanism.integrator_function>` (see `integrator_mode
        <TransferMechanism.integrator_mode>` for additional details). Otherwise, noise is added to the output of the
        `function <TransferMechanism.function>`. If `noise <TransferMechanism.noise>` is a list or array,
        it must be the same length as `variable <TransferMechanism.default_variable>`. If `noise
        <TransferMechanism.noise>` is specified as a single float or function, while `variable
        <TransferMechanism.variable>` is a list or array, `noise <TransferMechanism.noise>` will be applied to each
        element of `variable <TransferMechanism.variable>`. In the case that `noise <TransferMechanism.noise>` is
        specified as  a function, the function will be executed separately for each element of `variable
        <TransferMechanism.variable>`.

        .. note::
            In order to generate random noise, a probability distribution function should be used (see `Distribution
            Functions <DistributionFunction>` for details), that will generate a new noise value from its
            distribution on each execution. If `noise <TransferMechanism.noise>` is specified as a float or as a
            function with a fixed output, then the noise will simply be an offset that remains the same across all
            executions.

    clip : list [float, float]
        specifies the allowable range for the result of `function <TransferMechanism.function>`.  The 1st item (index
        0) specifies the minimum allowable value of the result, and the 2nd item (index 1) specifies the maximum
        allowable value; any element of the result that exceeds the specified minimum or maximum value is set to
        the value of `clip <TransferMechanism.clip>` that it exceeds.

    value : 2d np.array [array(float64)]
        result of executing `function <TransferMechanism.function>`.

    previous_value : 2d np.array [array(float64)] : default None
        `value <TransferMechanism.value>` after the previous execution of the Mechanism.  It is assigned `None` on
        the first execution, and when the Mechanism's `reinitialize <Mechanism.reinitialize>` method is called.

        .. note::
           The TransferMechanism's `previous_value` attribute is distinct from the `previous_value
           <AdaptiveIntegrator.previous_value>` attribute of its `integrator_function
           <TransferMechanism.integrator_function>`.

    delta : scalar
        value returned by `convergence_function <TransferMechanism.convergence_function>`;  used to determined
        when `is_converged <TransferMechanism.is_converged>` is `True`.

    is_converged : bool
        `True` if `delta <TransferMechanism.delta>` is less than or equal to `convergence_criterion
        <TransferMechanism.convergence_criterion>`.

    convergence_function : function
        compares `value <TransferMechanism.value>` with `previous_value <TransferMechanism.previous_value>`;
        result is used to determine when `is_converged <TransferMechanism.is_converged>` is `True`.

    convergence_criterion : float
        determines the value of `delta <TransferMechanism.delta>` at which `is_converged
        <TransferMechanism.is_converged>` is `True`.

    max_passes : int or None
        determines maximum number of executions (`passes <TimeScale.PASS>`) that can occur in a trial before reaching
        the `convergence_criterion <TransferMechanism.convergence_criterion>`, after which an error occurs;
        if `None` is specified, execution may continue indefinitely or until an interpreter exception is generated.

    output_states : *ContentAddressableList[OutputState]*
        list of Mechanism's `OutputStates <OutputStates>`; by default there is one OutputState for each InputState,
        with the base name `RESULT` (see `TransferMechanism_OutputStates` for additional details).

    output_values : List[array(float64)]
        each item is the `value <OutputState.value>` of the corresponding OutputState in `output_states
        <TransferMechanism.output_states>`.  The default is a single item containing the result of the
        TransferMechanism's `function <TransferMechanism.function>`;  additional
        ones may be included, based on the specifications made in the
        **output_states** argument of the Mechanism's constructor (see `TransferMechanism Standard OutputStates
        <TransferMechanism_Standard_OutputStates>`).

    name : str
        the name of the TransferMechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the TransferMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    Returns
    -------
    instance of TransferMechanism : TransferMechanism

    """

    componentType = TRANSFER_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TypeDefaultPreferences
    # classPreferences = {
    #     kwPreferenceSetName: 'TransferCustomClassPreferences',
    #     # kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    #     }

    # TransferMechanism parameter and control signal assignments):
    paramClassDefaults = ProcessingMechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({NOISE: None})

    standard_output_states = standard_output_states.copy()

    class Parameters(ProcessingMechanism_Base.Parameters):
        """
            Attributes
            ----------

                clip
                    see `clip <TransferMechanism.clip>`

                    :default value: None
                    :type:

                convergence_criterion
                    see `convergence_criterion <TransferMechanism.convergence_criterion>`

                    :default value: 0.01
                    :type: float

                convergence_function
                    see `convergence_function <TransferMechanism.convergence_function>`

                    :default value: `Distance`
                    :type: `Function`

                initial_value
                    see `initial_value <TransferMechanism.initial_value>`

                    :default value: None
                    :type:

                integration_rate
                    see `integration_rate <TransferMechanism.integration_rate>`

                    :default value: 0.5
                    :type: float

                integrator_function
                    see `integrator_function <TransferMechanism.integrator_function>`

                    :default value: `AdaptiveIntegrator`
                    :type: `Function`

                integrator_function_value
                    see `integrator_function_value <TransferMechanism.integrator_function_value>`

                    :default value: [[0]]
                    :type: list
                    :read only: True

                integrator_mode
                    see `integrator_mode <TransferMechanism.integrator_mode>`

                    :default value: False
                    :type: bool

                max_passes
                    see `max_passes <TransferMechanism.max_passes>`

                    :default value: 1000
                    :type: int

                noise
                    see `noise <TransferMechanism.noise>`

                    :default value: 0.0
                    :type: float

                on_resume_integrator_mode
                    see `on_resume_integrator_mode <TransferMechanism.on_resume_integrator_mode>`

                    :default value: `INSTANTAENOUS_MODE_VALUE`
                    :type: str

                previous_value
                    see `previous_value <TransferMechanism.previous_value>`

                    :default value: None
                    :type:
                    :read only: True

        """
        integrator_mode = Parameter(False, setter=_integrator_mode_setter)
        integration_rate = Parameter(0.5, modulable=True)
        initial_value = None
        previous_value = Parameter(None, read_only=True)
        integrator_function = Parameter(AdaptiveIntegrator, stateful=False, loggable=False)
        integrator_function_value = Parameter([[0]], read_only=True)
        has_integrated = Parameter(False, user=False)
        on_resume_integrator_mode = Parameter(INSTANTANEOUS_MODE_VALUE, stateful=False, loggable=False)
        clip = None
        noise = Parameter(0.0, modulable=True)
        convergence_criterion = Parameter(0.01, modulable=True)
        convergence_function = Parameter(Distance, stateful=False, loggable=False)
        max_passes = Parameter(1000, stateful=False)

        def _validate_integrator_mode(self, integrator_mode):
            if not isinstance(integrator_mode, bool):
                return 'may only be True or False.'

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_states:tc.optional(tc.any(Iterable, Mechanism, OutputState, InputState))=None,
                 function=Linear,
                 integrator_mode=False,
                 integrator_function=AdaptiveIntegrator,
                 initial_value=None,
                 integration_rate=0.5,
                 on_resume_integrator_mode=INSTANTANEOUS_MODE_VALUE,
                 noise=0.0,
                 clip=None,
                 convergence_function=None,
                 convergence_criterion:float=0.01,
                 max_passes:tc.optional(int)=1000,
                 output_states:tc.optional(tc.any(str, Iterable))=RESULTS,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs):
        """Assign type-level preferences and call super.__init__
        """

        # Default output_states is specified in constructor as a string rather than a list
        # to avoid "gotcha" associated with mutable default arguments
        # (see: bit.ly/2uID3s3 and http://docs.python-guide.org/en/latest/writing/gotchas/)
        if output_states is None or output_states is RESULTS:
            output_states = [RESULTS]

        initial_value = self._parse_arg_initial_value(initial_value)
        self.integrator_function = integrator_function or AdaptiveIntegrator # In case any subclass set it to None

        params = self._assign_args_to_param_dicts(function=function,
                                                  initial_value=initial_value,
                                                  input_states=input_states,
                                                  output_states=output_states,
                                                  noise=noise,
                                                  integration_rate=integration_rate,
                                                  integrator_mode=integrator_mode,
                                                  clip=clip,
                                                  convergence_function=convergence_function,
                                                  convergence_criterion=convergence_criterion,
                                                  max_passes=max_passes,
                                                  params=params)
        self.on_resume_integrator_mode = on_resume_integrator_mode
        # self.integrator_function = None
        self.has_integrated = False
        self._current_variable_index = 0

        if not isinstance(self.standard_output_states, StandardOutputStates):
            self.standard_output_states = StandardOutputStates(self,
                                                               self.standard_output_states,
                                                               indices=PRIMARY)

        super(TransferMechanism, self).__init__(
                default_variable=default_variable,
                size=size,
                input_states=input_states,
                function=function,
                params=params,
                name=name,
                prefs=prefs,

                **kwargs
        )

    def _parse_arg_initial_value(self, initial_value):
        return self._parse_arg_variable(initial_value)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate FUNCTION and Mechanism params

        """

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        # Validate FUNCTION
        if FUNCTION in target_set:
            transfer_function = target_set[FUNCTION]
            transfer_function_class = None

            # FUNCTION is a Function
            if isinstance(transfer_function, Function):
                transfer_function_class = transfer_function.__class__
            # FUNCTION is a class
            elif inspect.isclass(transfer_function):
                transfer_function_class = transfer_function

            if issubclass(transfer_function_class, Function):
                if not issubclass(transfer_function_class, (TransferFunction, SelectionFunction, UserDefinedFunction)):
                    raise TransferError("Function specified as {} param of {} ({}) must be a {}".
                                        format(repr(FUNCTION), self.name, transfer_function_class.__name__,
                                               " or ".join([TRANSFER_FUNCTION_TYPE, SELECTION_FUNCTION_TYPE])))
            elif not isinstance(transfer_function, (function_type, method_type)):
                raise TransferError("Unrecognized specification for {} param of {} ({})".
                                    format(repr(FUNCTION), self.name, transfer_function))

            # FUNCTION is a function or method, so test that shape of output = shape of input
            if isinstance(transfer_function, (function_type, method_type, UserDefinedFunction)):
                var_shape = self.defaults.variable.shape
                if isinstance(transfer_function, UserDefinedFunction):
                    val_shape = transfer_function._execute(self.defaults.variable, context=context).shape
                else:
                    val_shape = np.array(transfer_function(self.defaults.variable, context=context)).shape

                if val_shape != var_shape:
                    raise TransferError("The shape ({}) of the value returned by the Python function, method, or UDF "
                                        "specified as the {} param of {} must be the same shape ({}) as its {}".
                                        format(val_shape, repr(FUNCTION), self.name, var_shape, repr(VARIABLE)))

        # Validate INITIAL_VALUE
        if INITIAL_VALUE in target_set and target_set[INITIAL_VALUE] is not None:
            initial_value = np.array(target_set[INITIAL_VALUE])
            # Need to compare with variable, since default for initial_value on Class is None
            if initial_value.dtype != object:
                initial_value = np.atleast_2d(initial_value)
            if not iscompatible(initial_value, self.defaults.variable):
                raise TransferError(
                        "The format of the initial_value parameter for {} ({}) must match its variable ({})".
                        format(append_type_to_name(self), initial_value, self.defaults.variable,
                    )
                )

        # FIX: SHOULD THIS (AND INTEGRATION_RATE) JUST BE VALIDATED BY INTEGRATOR FUNCTION NOW THAT THEY ARE PROPERTIES??
        # Validate NOISE:
        if NOISE in target_set:
            noise = target_set[NOISE]
            # If assigned as a Function, set TransferMechanism as its owner, and assign its actual function to noise
            if isinstance(noise, DistributionFunction):
                noise.owner = self
                target_set[NOISE] = noise.execute
            self._validate_noise(target_set[NOISE])

        # Validate INTEGRATOR_FUNCTION:
        if INTEGRATOR_FUNCTION in target_set:
            integtr_fct = target_set[INTEGRATOR_FUNCTION]
            if not (isinstance(integtr_fct, IntegratorFunction)
                    or (isinstance(integtr_fct, type) and issubclass(integtr_fct, IntegratorFunction))):
                raise TransferError("The function specified for the {} arg of {} ({}) must be an {}".
                                    format(repr(INTEGRATOR_FUNCTION), self.name, integtr_fct),
                                    IntegratorFunction.__class__.__name__)

        # Validate INTEGRATION_RATE:
        if INTEGRATION_RATE in target_set and target_set[INTEGRATION_RATE] is not None:
            integration_rate = np.array(target_set[INTEGRATION_RATE])
            if not all_within_range(integration_rate, 0, 1):
                raise TransferError("Value(s) in {} arg for {} ({}) must be an int or float in the interval [0,1]".
                                    format(repr(INTEGRATION_RATE), self.name, integration_rate, ))
            if (not np.isscalar(integration_rate.tolist())
                    and integration_rate.shape != self.defaults.variable.squeeze().shape):
                raise TransferError("{} arg for {} ({}) must be either an int or float, "
                                    "or have the same shape as its {} ({})".
                                    format(repr(INTEGRATION_RATE), self.name, integration_rate,
                                           VARIABLE, self.defaults.variable))

        # Validate CLIP:
        if CLIP in target_set and target_set[CLIP] is not None:
            clip = target_set[CLIP]
            if clip:
                if not (isinstance(clip, (list,tuple)) and len(clip)==2 and all(isinstance(i, numbers.Number)
                                                                                for i in clip)):
                    raise TransferError("clip parameter ({}) for {} must be a tuple with two numbers".
                                        format(clip, self.name))
                if not clip[0] < clip[1]:
                    raise TransferError("The first item of the clip parameter ({}) must be less than the second".
                                        format(clip, self.name))
            target_set[CLIP] = list(clip)

    def _validate_noise(self, noise):
        # Noise is a list or array

        if isinstance(noise, (np.ndarray, list)):
            if len(noise) == 1:
                pass
            # Variable is a list/array
            elif not iscompatible(np.atleast_2d(noise), self.defaults.variable) and len(noise) > 1:
                raise MechanismError(
                    "Noise parameter ({}) does not match default variable ({}). Noise parameter of {} must be specified"
                    " as a float, a function, or an array of the appropriate shape ({})."
                    .format(noise, self.defaults.variable, self.name, np.shape(np.array(self.defaults.variable))))
            else:
                for i in range(len(noise)):
                    if isinstance(noise[i], DistributionFunction):
                        noise[i] = noise[i].execute
                    if not isinstance(noise[i], (float, int)) and not callable(noise[i]):
                        raise MechanismError("The elements of a noise list or array must be floats or functions. "
                            "{} is not a valid noise element for {}".format(noise[i], self.name))

        elif _is_control_spec(noise):
            pass

        # Otherwise, must be a float, int or function
        elif not isinstance(noise, (float, int)) and not callable(noise):
            raise MechanismError("Noise parameter ({}) for {} must be a float, "
                                 "function, or array/list of these.".format(noise,
                                                                            self.name))

    def _try_execute_param(self, param, var):

        # param is a list; if any element is callable, execute it
        if isinstance(param, (np.ndarray, list)):
            # NOTE: np.atleast_2d will cause problems if the param has "rows" of different lengths
            param = np.atleast_2d(param)
            for i in range(len(param)):
                for j in range(len(param[i])):
                    if callable(param[i][j]):
                        param[i][j] = param[i][j]()

        # param is one function
        elif callable(param):
            # NOTE: np.atleast_2d will cause problems if the param has "rows" of different lengths
            new_param = []
            for row in np.atleast_2d(var):
                new_row = []
                for item in row:
                    new_row.append(param())
                new_param.append(new_row)
            param = new_param

        return param

    def _instantiate_parameter_states(self, function=None, context=None):

        # If function is a logistic, and clip has not been specified, bound it between 0 and 1
        if ((isinstance(self.function, Logistic) or
                 (inspect.isclass(self.function) and issubclass(self.function,Logistic))) and
                self.clip is None):
            self.clip = (0,1)

        super()._instantiate_parameter_states(function=function, context=context)

    def _instantiate_attributes_before_function(self, function=None, context=None):
        super()._instantiate_attributes_before_function(function=function, context=context)

        if self.initial_value is None:
            self.initial_value = self.defaults.variable

    def _instantiate_integrator_function(self, variable, noise, initializer,  rate,
                                         context):

        if isinstance(self.integrator_function, type):
            self.integrator_function = self.integrator_function(default_variable=variable,
                                                                initializer=initializer,
                                                                noise=noise,
                                                                rate=rate,
                                                                owner=self)
        # User specified integrator_function in constructor
        # If the values of any of these parameters differ from the default on either the Mechanism or function:
        #     - use the value that differs (on the assumption that that was assigned by user;
        #     - if both differ, warn and give precedence to the value specified for the Function
        else:

            # FIX: 12/9/18 USE CALL TO reinitialize HERE??

            # Relabel to identify parameters passed in as the Mechainsm's values,
            #    and standardize format for comparison against values specified for functiom (by user or defaults)
            # mech_noise = np.array(noise).squeeze()
            # mech_init_val = np.array(initializer).squeeze()
            # mech_rate = np.array(rate).squeeze()
            mech_noise, mech_init_val, mech_rate = map(lambda x: np.array(x).squeeze(), [noise, initializer, rate])

            if self.integrator_function.owner is None:
                self.integrator_function.owner = self

            if hasattr(self.integrator_function, NOISE):
                fct_noise = np.array(self.integrator_function.noise)
                mech_specified = not np.array_equal(mech_noise, np.array(self.class_defaults.noise))
                fct_specified = not np.array_equal(np.array(self.integrator_function.noise),
                                                   np.array(self.integrator_function.class_defaults.noise))

                # Mechanism and function noise are not the same
                if not np.array_equal(mech_noise, fct_noise):
                    # If function's noise was not specified, assign Mechanism's value to it
                    if not fct_specified:
                        self.integrator_function.parameters.noise._set(mech_noise, context)
                    # Otherwise, given precedence to function's value
                    else:
                        if mech_specified:
                            warnings.warn("Specification of the {} argument for {} ({}) conflicts with specification of"
                                          " the {} parameter ({}) for its {} ({});  the Function's value will be used.".
                                          format(repr(NOISE), self.name, mech_noise,
                                                 repr(NOISE), self.integrator_function.noise,
                                                 repr(INTEGRATOR_FUNCTION),
                                                 self.integrator_function.__class__.__name__))
                        # Assign funciton's noise to Mechanism
                        self.parameters.noise._set(self.integrator_function.noise, context)

                        # KDM 12/21/18: validating here until a standard scheme is designed, because it's tested for
                        self._validate_params(
                            request_set={'noise': self.integrator_function.noise},
                            target_set={'noise': self.integrator_function.noise},
                            context=context
                        )

            if hasattr(self.integrator_function, INITIALIZER):
                fct_intlzr = np.array(self.integrator_function.initializer)
                # Check against variable, as class.default is None, but initial_value assigned to variable before here
                mech_specified = not np.array_equal(mech_init_val, np.array(self.defaults.variable))
                fct_specified = not np.array_equal(np.array(self.integrator_function.initializer),
                                                   np.array(self.integrator_function.class_defaults.initializer))

                # Mechanism initial_value and function initializer are not the same
                if not np.array_equal(mech_init_val, fct_intlzr):
                    # If function's initializer was not specified, assign Mechanism's initial_value to it
                    if not fct_specified:
                        self.integrator_function.parameters.initializer._set(initializer, context)
                        self.integrator_function._initialize_previous_value(initializer, context)
                    # Otherwise, give precedence to function's value
                    else:
                        if mech_specified:
                            warnings.warn("Specification of the {} argument for {} ({}) conflicts with specification of"
                                          " the {} parameter ({}) for its {} ({});  the Function's value will be used.".
                                          format(repr(INITIAL_VALUE), self.name, mech_init_val,
                                                 repr(INITIALIZER), self.integrator_function.initializer,
                                                 repr(INTEGRATOR_FUNCTION),
                                                 self.integrator_function.__class__.__name__))
                        # Assign function's initializer to Mechanism
                        self.parameters.initial_value._set(self.integrator_function.initializer, context)

            if hasattr(self.integrator_function, RATE):
                fct_rate = np.array(self.integrator_function.rate)
                mech_specified = not np.array_equal(mech_rate, np.array(self.class_defaults.integration_rate))
                fct_specified = not np.array_equal(np.array(self.integrator_function.rate),
                                                   np.array(self.integrator_function.class_defaults.rate))
                # Mechanism and function rate are not the same
                if not np.array_equal(mech_rate, fct_rate):
                    # If function's rate was not specified, assign Mechanism's value to it
                    if not fct_specified:
                        self.integrator_function.parameters.rate._set(rate, context)
                    # Otherwise, warn and then give precedence to function's value
                    else:
                        if mech_specified:
                            warnings.warn("Specification of the {} argument for {} ({}) conflicts with specification of"
                                          " the {} parameter ({}) for its {} ({});  the Function's value will be used.".
                                          format(repr(INTEGRATION_RATE), self.name, rate,
                                                 repr(RATE), self.integrator_function.rate, repr(INTEGRATOR_FUNCTION),
                                                 self.integrator_function.__class__.__name__))
                        # Assign function's rate to Mechanism
                        self.parameters.integration_rate._set(self.integrator_function.rate, context)

                        # KDM 12/21/18: validating here until a standard scheme is designed, because it's tested for
                        self._validate_params(
                            request_set={'integration_rate': self.integrator_function.rate},
                            target_set={'integration_rate': self.integrator_function.rate},
                            context=context
                        )

        # MODIFIED 6/24/19 NEW:
        # Insure that integrator_function's variable and value have same shape as TransferMechanism's variable
        integrator_fct_variable = self.integrator_function.parameters.variable.default_value
        if integrator_fct_variable.shape != variable.shape:
            fct_var_inner_dim = integrator_fct_variable.shape[-1]
            # If inner dimension of function's variable is not same as Mechanism's and is user_specified, raise error
            if integrator_fct_variable.shape[-1] != variable.shape[-1] and\
                    self.integrator_function.parameters.variable._user_specified:
                raise TransferError(f"The length ({fct_var_inner_dim}) of the {repr(VARIABLE)} or one of the parameters"
                                    f" specified for the {repr(INTEGRATOR_FUNCTION)} of {self.name} doesn't match the "
                                    f"size ({variable.shape[-1]}) of the innermost dimension (axis 0) of its "
                                    f"{repr(VARIABLE)} (i.e., the length of its items .")
            self.integrator_function.parameters.variable.default_value = variable
            function_context_buffer = self.integrator_function.initialization_status
            self.integrator_function.initialization_status = ContextFlags.INITIALIZING
            self.integrator_function.parameters.value.default_value = self.integrator_function(variable, context=context)
            self.integrator_function.initialization_status = function_context_buffer
        # MODIFIED 6/24/19 END

        self.has_integrated = True

    def _instantiate_output_states(self, context=None):
        # If user specified more than one item for variable, but did not specify any custom OutputStates
        # then assign one OutputState (with the default name, indexed by the number of them) per item of variable
        if len(self.defaults.variable) > 1 and len(self.output_states) == 1 and self.output_states[0] == RESULTS:
            self.output_states = []
            for i, item in enumerate(self.defaults.variable):
                self.output_states.append({NAME: RESULT, VARIABLE: (OWNER_VALUE, i)})
        super()._instantiate_output_states(context=context)

    def _get_instantaneous_function_input(self, function_variable, noise):
        noise = self._try_execute_param(noise, function_variable)
        if (np.array(noise) != 0).any():
            current_input = function_variable + noise
        else:
            current_input = function_variable

        return current_input

    def _get_integrated_function_input(self, function_variable, initial_value, noise, context, **kwargs):

        integration_rate = self.get_current_mechanism_param(INTEGRATION_RATE, context)

        if self.initialization_status == ContextFlags.INITIALIZING:
            self._instantiate_integrator_function(variable=function_variable,
                                                  noise=noise,
                                                  initializer=initial_value,
                                                  rate=integration_rate,
                                                  context=context)
            # Update param assignments with ones determined to be relevant (mech vs. fct)
            #    and assigned to integrator_function in _instantiate_integrator_function
            initial_value = self.integrator_function.initializer
            integration_rate = self.integrator_function.rate
            noise = self.integrator_function.noise

        current_input = self.integrator_function.execute(
            function_variable,
            context=context,
            # Should we handle runtime params?
            runtime_params={
                INITIALIZER: initial_value,
                NOISE: noise,
                RATE: integration_rate
            },

        )

        return current_input

    def _clip_result(self, clip, current_input):
        if clip is not None:
            minCapIndices = np.where(current_input < clip[0])
            maxCapIndices = np.where(current_input > clip[1])
            current_input[minCapIndices] = np.min(clip)
            current_input[maxCapIndices] = np.max(clip)
        return current_input

    def _get_function_param_struct_type(self, ctx):
        param_type_list = [ctx.get_param_struct_type(self.function)]
        if self.integrator_mode:
            assert self.integrator_function is not None
            param_type_list.append(ctx.get_param_struct_type(self.integrator_function))
        return pnlvm.ir.LiteralStructType(param_type_list)

    def _get_function_state_struct_type(self, ctx):
        state_struct_type_list = [ctx.get_state_struct_type(self.function)]
        if self.integrator_mode:
           assert self.integrator_function is not None
           state_struct_type_list.append(ctx.get_state_struct_type(self.integrator_function))

        return pnlvm.ir.LiteralStructType(state_struct_type_list)

    def _get_function_param_initializer(self, context):
        function_param_list = [self.function._get_param_initializer(context)]
        if self.integrator_mode:
            assert self.integrator_function is not None
            function_param_list.append(self.integrator_function._get_param_initializer(context))
        return tuple(function_param_list)

    def _get_function_state_initializer(self, context):
        context_list = [self.function._get_state_initializer(context)]
        if self.integrator_mode:
            assert self.integrator_function is not None
            context_list.append(self.integrator_function._get_state_initializer(context))
        return tuple(context_list)

    def _gen_llvm_function_body(self, ctx, builder, params, context, arg_in, arg_out):
        is_out, builder = self._gen_llvm_input_states(ctx, builder, params, context, arg_in)

        # Parameters and context for both integrator and main function
        f_params = builder.gep(params, [ctx.int32_ty(0), ctx.int32_ty(1)])
        f_context = builder.gep(context, [ctx.int32_ty(0), ctx.int32_ty(1)])

        if self.integrator_mode:
            # IntegratorFunction function is the second in the function param aggregate
            if_context = builder.gep(f_context, [ctx.int32_ty(0), ctx.int32_ty(1)])
            if_param_ptr = builder.gep(f_params, [ctx.int32_ty(0), ctx.int32_ty(1)])
            if_params, builder = self._gen_llvm_param_states(self.integrator_function, if_param_ptr, ctx, builder, params, context, arg_in)

            mf_in, builder = self._gen_llvm_invoke_function(ctx, builder, self.integrator_function, if_params, if_context, is_out)
        else:
            mf_in = is_out

        # Main function is the first in the function param aggregate
        mf_context = builder.gep(f_context, [ctx.int32_ty(0), ctx.int32_ty(0)])
        mf_param_ptr = builder.gep(f_params, [ctx.int32_ty(0), ctx.int32_ty(0)])
        mf_params, builder = self._gen_llvm_param_states(self.function, mf_param_ptr, ctx, builder, params, context, arg_in)

        mf_out, builder = self._gen_llvm_invoke_function(ctx, builder, self.function, mf_params, mf_context, mf_in)

        # FIXME: Convert to runtime instead of compile time
        clip = self.parameters.clip.get()
        if clip is not None:
            for i in range(mf_out.type.pointee.count):
                mf_out_local = builder.gep(mf_out, [ctx.int32_ty(0), ctx.int32_ty(i)])
                with pnlvm.helpers.array_ptr_loop(builder, mf_out_local, "clip") as (b1, index):
                    ptri = b1.gep(mf_out_local, [ctx.int32_ty(0), index])
                    ptro = b1.gep(mf_out_local, [ctx.int32_ty(0), index])

                    val = b1.load(ptri)
                    val = pnlvm.helpers.fclamp(b1, val, clip[0], clip[1])
                    b1.store(val, ptro)

        builder = self._gen_llvm_output_states(ctx, builder, mf_out, params, context, arg_in, arg_out)

        return builder

    def _execute(self,
        variable=None,
        context=None,
        runtime_params=None,

    ):
        """Execute TransferMechanism function and return transform of input

        Execute TransferMechanism function on input, and assign to output_values:
            - Activation value for all units
            - Mean of the activation values across units
            - Variance of the activation values across units
        Return:
            value of input transformed by TransferMechanism function in outputState[TransferOuput.RESULT].value
            mean of items in RESULT outputState[TransferOuput.OUTPUT_MEAN].value
            variance of items in RESULT outputState[TransferOuput.OUTPUT_VARIANCE].value

        Arguments:

        # CONFIRM:
        variable (float): set to self.value (= self.input_value)
        - params (dict):  runtime_params passed from Mechanism, used as one-time value for current execution:
            + NOISE (float)
            + INTEGRATION_RATE (float)
            + RANGE ([float, float])
        - context (str)

        Returns the following values in self.value (2D np.array) and in
            the value of the corresponding outputState in the self.output_states list:
            - activation value (float)
            - mean activation value (float)
            - standard deviation of activation values (float)

        :param self:
        :param variable (float)
        :param params: (dict)
        :param context: (str)
        :rtype self.outputState.value: (number)
        """

        # FIX: ??CALL check_args()??

        # FIX: IS THIS CORRECT?  SHOULD THIS BE SET TO INITIAL_VALUE
        # FIX:     WHICH SHOULD BE DEFAULTED TO 0.0??
        # Use self.defaults.variable to initialize state of input

        # EXECUTE TransferMechanism FUNCTION ---------------------------------------------------------------------

        # FIX: JDC 7/2/18 - THIS SHOULD BE MOVED TO AN STANDARD OUTPUT_STATE
        # Clip outputs
        clip = self.get_current_mechanism_param("clip", context)

        value = super(Mechanism, self)._execute(variable=variable,
                                                context=context,
                                                runtime_params=runtime_params,

                                                )
        value = self._clip_result(clip, value)

        # Used by update_previous_value, convergence_function and delta
        # self.parameters.value._set(np.atleast_2d(value), context, skip_history=True, skip_log=True)

        return value

    @handle_external_context(execution_id=NotImplemented)
    def reinitialize(self, *args, context=None):
        super().reinitialize(*args, context=context)
        self.parameters.previous_value.set(None, context, override=True)

    def _update_previous_value(self, context=None):
        if self.parameters.integrator_mode._get(context):
            value = self.parameters.value._get(context)
            if value is None:
                value = self.defaults.value
            self.parameters.previous_value._set(value, context)

    def _parse_function_variable(self, variable, context=None):
        if context.source is ContextFlags.INSTANTIATE:

            return super(TransferMechanism, self)._parse_function_variable(variable=variable, context=context)

        # FIX: NEED TO GET THIS TO WORK WITH CALL TO METHOD:
        integrator_mode = self.parameters.integrator_mode._get(context)
        noise = self.get_current_mechanism_param(NOISE, context)

        # FIX: SHOULD UPDATE PARAMS PASSED TO integrator_function WITH ANY RUNTIME PARAMS THAT ARE RELEVANT TO IT
        # Update according to time-scale of integration
        if integrator_mode:
            initial_value = self.get_current_mechanism_param(INITIAL_VALUE, context)

            value = self._get_integrated_function_input(variable,
                                                        initial_value,
                                                        noise,
                                                        context,
                                                        )

            self.parameters.integrator_function_value._set(value, context)
            return value

        else:
            return self._get_instantaneous_function_input(variable, noise)

    def _report_mechanism_execution(self, input, params, output, context=None):
        """Override super to report previous_input rather than input, and selected params
        """
        # KAM Changed 8/29/17 print_input = self.previous_input --> print_input = input
        # because self.previous_input is not a valid attrib of TransferMechanism

        print_input = input
        print_params = params.copy()
        # Suppress reporting of range (not currently used)
        del print_params[CLIP]

        super()._report_mechanism_execution(input_val=print_input, params=print_params, context=context)

    def delta(self, value=NotImplemented, context=None):
        if value is NotImplemented:
            value = self.parameters.value._get(context)
        return self.convergence_function([value[0], self.parameters.previous_value._get(context)[0]])

    @handle_external_context()
    def is_converged(self, value=NotImplemented, context=None):
        # Check for convergence
        if (
            self.convergence_criterion is not None
            and self.parameters.previous_value._get(context) is not None
            and self.initialization_status != ContextFlags.INITIALIZING
        ):
            if self.delta(value, context) <= self.convergence_criterion:
                return True
            elif self.get_current_execution_time(context).pass_ >= self.max_passes:
                raise TransferError("Maximum number of executions ({}) has occurred before reaching "
                                    "convergence_criterion ({}) for {} in trial {} of run {}".
                                    format(self.max_passes, self.convergence_criterion, self.name,
                                           self.get_current_execution_time(context).trial, self.get_current_execution_time(context).run))
            else:
                return False
        # Otherwise just return True
        else:
            return None
