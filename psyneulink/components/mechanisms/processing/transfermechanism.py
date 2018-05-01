# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# NOTES:
#  * COULD NOT IMPLEMENT integrator_function in paramClassDefaults (see notes below)
#  * NOW THAT NOISE AND SMOOTHING_FACTOR ARE PROPRETIES THAT DIRECTLY REFERERNCE integrator_function,
#      SHOULD THEY NOW BE VALIDATED ONLY THERE (AND NOT IN TransferMechanism)??
#  * ARE THOSE THE ONLY TWO integrator PARAMS THAT SHOULD BE PROPERTIES??

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

A TransferMechanism transforms its input using a simple mathematical function, that maintains the form (dimensionality)
of its input.  The input can be a single scalar value, a multidimensional array (list or numpy array), or several
independent ones.

The function used to carry out the transformation can be selected from the following PsyNeuLink
`Functions <Function>`: `Linear`, `Exponential`, `Logistic`, or `SoftMax`.

The **integrator_mode** argument can switch the transformation from an "instantaneous"  to a "time averaged"
(integrated) manner of execution. When `integrator_mode <TransferMechanism.integrator_mode>` is set to True, the
mechanism's input is first transformed by its `integrator_function <TransferMechanism.integrator_function>` (the
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

When `integrator_mode <TransferMechanism.integrator_mode>` is True, the TransferMechanism has an `integrator_function
<TransferMechanism.integrator_function>` which it applies to its variable on each execution. The output of the
`integrator_function  <TransferMechanism.integrator_function>` is then used as the input to its `function
<TransferMechanism.function>`.

The `integrator_function <TransferMechanism.integrator_function>` of a TransferMechanism is always the
`AdaptiveIntegrator`. Two parameters of the `AdaptiveIntegrator` are exposed on the TransferMechanism. Specifying the
arguments **smoothing_factor** and/or **initial_value** in the mechanism's constructor will actually set the mechanism's
`integrator_function <TransferMechanism.integrator_function>` to an `AdaptiveIntegrator` with those values specified for
`rate <AdaptiveIntegrator.rate>` and `initializer <AdaptiveIntegrator.initializer>`, respectively.

    >>> my_logistic_transfer_mechanism = pnl.TransferMechanism(function=pnl.Logistic(gain=1.0, bias=-4),
    ...                                                        integrator_mode=True,
    ...                                                        smoothing_factor=0.1,
    ...                                                        initial_value=np.array([[0.2]]))

.. note::
    If `integrator_mode <TransferMechanism.integrator_mode>` is False, then the arguments **smoothing_factor** and
    **initial_value** are ignored, because the mechanism does not have an `integrator_function
    <TransferMechanism.integrator_function>` to construct.

Finally, the TransferMechanism has two arguments that can adjust the final result of the mechanism: **clip** and
**noise**. If `integrator_mode <TransferMechanism.integrator_mode>` is False, `clip <TransferMechanism.clip>` and
`noise <TransferMechanism.noise>` modify the value returned by the mechanism's `function <TransferMechanism.function>`
before setting it as the mechanism's value. If `integrator_mode <TransferMechanism.integrator_mode>` is True,
**noise** is simply handed to the mechanism's `integrator_function <TransferMechanism.integrator_function>` (in the same
manner as **smoothing_factor** and **initial_value**), whereas `clip <TransferMechanism.clip>` modifies the value
returned by the mechanism's `function <TransferMechanism.function>` before setting it as the mechanism's value.

.. _Transfer_Structure:

Structure
---------

.. _TransferMechanism_InputStates:

InputStates
~~~~~~~~~~~

By default, a TransferMechanism has a single `InputState`;  however, more than one can be specified
using the **default_variable** or **size** arguments of its constructor (see `Mechanism`).  The `value
<InputState.value>` of each InputState is used as a separate item of the Mechanism's `variable
<TransferMechanism.variable>`, and transformed independently by its `function <TransferMechanism.function>`.
Like any InputStates, the `value <OutputState.value>` of any or all of the TransferMechanism's InputStates can be
modulated by one or more `GatingSignals <GatingSignal_Modulation>` prior to transformation by its `function
<TransferMechanism.function>`.

.. _TransferMechanism_Function:

Function
~~~~~~~~

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

OutputStates
~~~~~~~~~~~~

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
      passing through the function of the mechanism. When `integrator_mode <TransferMechanism.integrator_mode>` is set to
      True, the TransferMechanism exponentially time-averages its input, by executing its `integrator_function
      <TransferMechanism.integrator_function>`, before executing its `function <TransferMechanism.function>`. When
      `integrator_mode <TransferMechanism.integrator_mode>` is False, the `integrator_function
      <TransferMechanism.integrator_function>` is ignored, and time-averaging does not occur.

    * `smoothing_factor <TransferMechanism.smoothing_factor>`: if the `integrator_mode <TransferMechanism.integrator_mode>`
      attribute is set to True, the `smoothing_factor <TransferMechanism.smoothing_factor>` attribute is the rate of
      integration (a higher value specifies a faster rate); if `integrator_mode <TransferMechanism.integrator_mode>` is False,
      `smoothing_factor <TransferMechanism.smoothing_factor>` is ignored and time-averaging does not occur.

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

Reinitialization
~~~~~~~~~~~~

In some cases, it may be useful to reset the accumulation of a mechanism back to its original starting point, or a new
starting point. This is done using the `reinitialize <AdaptiveIntegrator.reinitialize>` method on the mechanism's
`integrator_function <TransferMechanism.integrator_function>`, or the mechanisms's own `reinitialize
<TransferMechanism.reinitialize>` method.

The `reinitialize <AdaptiveIntegrator.reinitialize>` method of the `integrator_function
<TransferMechanism.integrator_function>` sets:

    - the integrator_function's `initializer <AdaptiveIntegrator.initializer>` attribute
    - the integrator_function's `previous_value <AdaptiveIntegrator.previous_value>` attribute
    - the integrator_function's `value <AdaptiveIntegrator.value>` attribute

    to the specified value.

The `reinitialize <TransferMechanism.reinitialize>` method of the `TransferMechanism` first sets:

    - the integrator_function's `initializer <AdaptiveIntegrator.initializer>` attribute
    - the integrator_function's `previous_value <AdaptiveIntegrator.previous_value>` attribute
    - the integrator_function's `value <AdaptiveIntegrator.value>` attribute
    - the TransferMechanism's `initial_value <TransferMechanism.initial_value>` attribute

    to the specified value. Then:

    - the specified value is passed into the mechanism's `function <TransferMechanism.function>` and the function is executed
    - the TransferMechanism's `value <TransferMechanism.value>` attribute is set to the output of the function
    - the TransferMechanism updates is `output_states <TransferMechanism.output_states>`

A use case for `reinitialize <AdaptiveIntegrator.reinitialize>` is demonstrated in the following example:

Create a `System` with a TransferMechanism in integrator_mode:

    >>> my_time_averaged_transfer_mechanism = pnl.TransferMechanism(function=pnl.Linear,        #doctest: +SKIP
    ...                                                        integrator_mode=True,            #doctest: +SKIP
    ...                                                        smoothing_factor=0.1,            #doctest: +SKIP
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

In the examples above, `reinitialize <AdaptiveIntegrator.reinitialize>` was applied directly to the integrator function.
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
import numbers
from collections import Iterable

import numpy as np
import typecheck as tc

from psyneulink.components.component import Component, function_type, method_type
from psyneulink.components.functions.function import AdaptiveIntegrator, Linear, NormalizingFunction, UserDefinedFunction
from psyneulink.components.mechanisms.adaptive.control.controlmechanism import _is_control_spec
from psyneulink.components.mechanisms.mechanism import Mechanism, MechanismError
from psyneulink.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.components.states.inputstate import InputState
from psyneulink.components.states.outputstate import OutputState, PRIMARY, StandardOutputStates, standard_output_states
from psyneulink.globals.context import ContextFlags
from psyneulink.globals.keywords import FUNCTION, INITIALIZER, INITIALIZING, MAX_ABS_INDICATOR, MAX_ABS_VAL, MAX_INDICATOR, MAX_VAL, MEAN, MEDIAN, NAME, NOISE, NORMALIZING_FUNCTION_TYPE, OWNER_VALUE, PROB, RATE, RESULT, RESULTS, STANDARD_DEVIATION, TRANSFER_FUNCTION_TYPE, TRANSFER_MECHANISM, VARIABLE, VARIANCE
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.globals.utilities import append_type_to_name, iscompatible

__all__ = [
    'INITIAL_VALUE', 'CLIP', 'SMOOTHING_FACTOR', 'Transfer_DEFAULT_BIAS', 'Transfer_DEFAULT_GAIN',
    'Transfer_DEFAULT_LENGTH', 'Transfer_DEFAULT_OFFSET', 'TRANSFER_OUTPUT', 'TransferError', 'TransferMechanism',
]

# TransferMechanism parameter keywords:
CLIP = "clip"
SMOOTHING_FACTOR = "smoothing_factor"
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

    *MEAN* : float
      mean of `value <TransferMechanism.value>`.

    .. _TRANSFER_MECHANISM_MEDIAN:

    *MEDIAN* : float
      median of `value <TransferMechanism.value>`.

    .. _TRANSFER_MECHANISM_STD_DEV:

    *STANDARD_DEVIATION* : float
      standard deviation of `value <TransferMechanism.value>`.

    .. _TRANSFER_MECHANISM_VARIANCE:

    *VARIANCE* : float
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
    MEAN=MEAN
    MEDIAN=MEDIAN
    STANDARD_DEVIATION=STANDARD_DEVIATION
    VARIANCE=VARIANCE
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

# IMPLEMENTATION NOTE:  IMPLEMENTS OFFSET PARAM BUT IT IS NOT CURRENTLY BEING USED
class TransferMechanism(ProcessingMechanism_Base):
    """
    TransferMechanism(           \
    default_variable=None,       \
    size=None,                   \
    input_states=None,           \
    function=Linear,             \
    initial_value=None,          \
    noise=0.0,                   \
    smoothing_factor=0.5,        \
    integrator_mode=False,       \
    clip=[float:min, float:max], \
    output_states=RESULTS        \
    params=None,                 \
    name=None,                   \
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
            + ClassDefaults.variable (value):  Transfer_DEFAULT_BIAS

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

    input_states : str, list or np.ndarray
        specifies the InputStates for the TransferMechanism; by default, a single InputState is created using the
        value of default_variable as its `variable <InputState.variable>`;  if more than one is specified, the number
        and, if specified, their values must be compatible with any specifications in **default_variable** or
        **size** (see `Mechanism_InputStates`);  see `input_states <TransferMechanism.output_states>` for additional
        details.

    function : TransferFunction : default Linear
        specifies the function used to transform the input;  can be `Linear`, `Logistic`, `Exponential`,
        or a custom function.

    initial_value :  value, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the starting value for time-averaged input (only relevant if `integrator_mode
        <TransferMechanism.integrator_mode>` is True).
        COMMENT:
            Transfer_DEFAULT_BIAS SHOULD RESOLVE TO A VALUE
        COMMENT

    noise : float or function : default 0.0
        specifies a value to be added to the result of the TransferMechanism's `function <TransferMechanism.function>`
        or its `integrator_function <TransferMechanism.integrator_function>`, depending on whether `integrator_mode
        <TransferMechanism.integrator_mode>` is `True` or `False`. See `noise <TransferMechanism.noise>` for details.

    smoothing_factor : float : default 0.5
        specifies the smoothing factor used for exponential time averaging of input when the TransferMechanism is
        executed with `integrator_mode` set to `True`.

    integrator_mode : bool : False
        specifies whether or not the TransferMechanism should be executed using its `integrator_function
        <TransferMechanism>` to integrate (exponentialy time-average) its `variable <TransferMechanism.variable>` (
        when set to `True`), or simply report the asymptotic value of the output of its `function
        <TransferMechanism.function>` (when set to `False`).

    clip : list [float, float] : default None (Optional)
        specifies the allowable range for the result of `function <TransferMechanism.function>`. The item in index 0
        specifies the minimum allowable value of the result, and the item in index 1 specifies the maximum allowable
        value; any element of the result that exceeds the specified minimum or maximum value is set to the value of
        `clip <TransferMechanism.clip>` that it exceeds.

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

    Returns
    -------
    instance of TransferMechanism : TransferMechanism


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

    COMMENT:
       THE FOLLOWING IS THE CURRENT ASSIGNMENT
    COMMENT

    initial_value :  value, list or np.ndarray
        specifies the starting value for time-averaged input (only relevant if `integrator_mode
        <TransferMechanism.integrator_mode>` is `True` and `smoothing_factor <TransferMechanism.smoothing_factor>` is
        not 1.0).
        COMMENT:
            Transfer_DEFAULT_BIAS SHOULD RESOLVE TO A VALUE
        COMMENT

    noise : float or function
        When `integrator_mode <TransferMechanism.integrator_mode>` is set to `True`, `noise <TransferMechanism.noise>`
        is passed into the `integrator_function <TransferMechanism.integrator_function>`. Otherwise, noise is added
        to the output of the `function <TransferMechanism.function>`.

        If `noise <TransferMechanism.noise>` is a list or array, it must be the same length as `variable
        <TransferMechanism.default_variable>`.

        If `noise <TransferMechanism.noise>` is specified as a single float or function, while `variable
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

    smoothing_factor : float
        the smoothing factor used for exponential time averaging of the TransferMechanism's `variable
        <TransferMechanism>` when it is executed with `integrator_mode <TransferMechanism.integrator_mode>`
        set to True (see `integrator_mode <TransferMechanism.integrator_mode>` for details).

    integrator_mode : bool
        determines whether the TransferMechanism uses its `integrator_function <TransferMechanism.integrator_function>`
        to exponentially time average its `variable <TransferMechanism.variable>` when it executes.

        **If integrator_mode is set to** `True`:

            the TransferMechanism's `variable <TransferMechanism>` is first
            passed into the `AdaptiveIntegrator` Function, that carries out the following calculation:

            .. math::
                result = previous\\_value(1-smoothing\\_factor) + variable \\cdot smoothing\\_factor + noise

            where *previous_value* is set to the value of the TransferMechanism's `initial_value
            <TransferMechanism.initial_value>` attribute on the first execution, and *smoothing_factor* and *noise*
            are determined by the TransferMechanism's `smoothing_factor <TransferMechanism.smoothing_factor>` and
            `noise <TransferMechanism.noise>` attributes, respectively.  The result is then passed to the
            TransferMechanism's `function <TransferMechanism.function>` which computes the TransferMechanism's `value
            <TransferMechanism.value>`.

        **If integrator_mode is set to** `False`:

            the TransferMechanism's `variable <TransferMechanism>` is passed directly to its `function
            <TransferMechanism.function>` -- that is, its `integrator_function <TransferMechanism.integrator_function>`
            is bypassed, and all related attributes (`initial_value <TransferMechanism.initial_value>`, smoothing_factor
            <TransferMechanism.smoothing_factor>`, and `noise <TransferMechanism.noise>`) are ignored.
            COMMENT:
            leak and time_step_size were previoulsy mentioned, but don't appear in the integrator_mode equation above
            COMMENT

    integrator_function :  Function
        the `AdaptiveIntegrator` Function used when `integrator_mode <TransferMechanism.integrator_mode>` is set to
        `True` (see `integrator_mode <TransferMechanism.integrator_mode>` for details).

        .. note::
            The TransferMechanism's `smoothing_factor <TransferMechanism.smoothing_factor>` parameter
            specifies the `rate <AdaptiveIntegrator.rate>` of the `AdaptiveIntegrator` Function.

    clip : list [float, float]
        specifies the allowable range for the result of `function <TransferMechanism.function>`.  The 1st item (index
        0) specifies the minimum allowable value of the result, and the 2nd item (index 1) specifies the maximum
        allowable value; any element of the result that exceeds the specified minimum or maximum value is set to
        the value of `clip <TransferMechanism.clip>` that it exceeds.

    value : 2d np.array [array(float64)]
        result of executing `function <TransferMechanism.function>`.

    previous_value : float
        the `value <TransferMechanism.value>` on the previous execution of the Mechanism.

    delta : float
        the change in `value <TransferMechanism.value>` from the previous execution of the TransferMechanism
        (i.e., `value <TransferMechanism.value>` - `previous_value <TransferMechanism.previous_value>`).

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

    """

    componentType = TRANSFER_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TypeDefaultPreferences
    # classPreferences = {
    #     kwPreferenceSetName: 'TransferCustomClassPreferences',
    #     # kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    #     kpRuntimeParamStickyAssignmentPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
    # }

    # TransferMechanism parameter and control signal assignments):
    paramClassDefaults = ProcessingMechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({NOISE: None})

    standard_output_states = standard_output_states.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_states:tc.optional(tc.any(Iterable, Mechanism, OutputState, InputState))=None,
                 function=Linear,
                 initial_value=None,
                 noise=0.0,
                 smoothing_factor=0.5,
                 integrator_mode=False,
                 clip=None,
                 output_states:tc.optional(tc.any(str, Iterable))=RESULTS,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None):
        """Assign type-level preferences and call super.__init__
        """

        # Default output_states is specified in constructor as a string rather than a list
        # to avoid "gotcha" associated with mutable default arguments
        # (see: bit.ly/2uID3s3 and http://docs.python-guide.org/en/latest/writing/gotchas/)
        if output_states is None or output_states is RESULTS:
            output_states = [RESULTS]

        initial_value = self._parse_arg_initial_value(initial_value)

        params = self._assign_args_to_param_dicts(function=function,
                                                  initial_value=initial_value,
                                                  input_states=input_states,
                                                  output_states=output_states,
                                                  noise=noise,
                                                  smoothing_factor=smoothing_factor,
                                                  integrator_mode=integrator_mode,
                                                  clip=clip,
                                                  params=params)

        self.integrator_function = None
        self.original_integrator_function = None

        if not isinstance(self.standard_output_states, StandardOutputStates):
            self.standard_output_states = StandardOutputStates(self,
                                                               self.standard_output_states,
                                                               indices=PRIMARY)

        super(TransferMechanism, self).__init__(
            default_variable=default_variable,
            size=size,
            params=params,
            name=name,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR,
            input_states=input_states,
            function=function,
        )

    def _parse_arg_initial_value(self, initial_value):
        return self._parse_arg_variable(initial_value)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate FUNCTION and Mechanism params

        """
        from psyneulink.components.functions.function import \
            Function, TransferFunction, NormalizingFunction, DistributionFunction

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
                if not issubclass(transfer_function_class, (TransferFunction, NormalizingFunction, UserDefinedFunction)):
                    raise TransferError("Function type specified as {} param of {} ({}) must be a {}".
                                        format(repr(FUNCTION), self.name, transfer_function_class.__name__,
                                               TRANSFER_FUNCTION_TYPE + ' or ' + NORMALIZING_FUNCTION_TYPE))
            elif not isinstance(transfer_function, (function_type, method_type)):
                raise TransferError("Unrecognized specification for {} param of {} ({})".
                                    format(repr(FUNCTION), self.name, transfer_function))

            # FUNCTION is a function or method, so test that shape of output = shape of input
            if isinstance(transfer_function, (function_type, method_type, UserDefinedFunction)):
                var_shape = self.instance_defaults.variable.shape
                if isinstance(transfer_function, UserDefinedFunction):
                    val_shape = transfer_function._execute(self.instance_defaults.variable).shape
                else:
                    val_shape = np.array(transfer_function(self.instance_defaults.variable)).shape

                if val_shape != var_shape:
                    raise TransferError("The shape ({}) of the value returned by the Python function, method, or UDF "
                                        "specified as the {} param of {} must be the same shape ({}) as its {}".
                                        format(val_shape, repr(FUNCTION), self.name, var_shape, repr(VARIABLE)))


        # Validate INITIAL_VALUE
        if INITIAL_VALUE in target_set:
            initial_value = target_set[INITIAL_VALUE]
            if initial_value is not None:
                if not iscompatible(initial_value, self.instance_defaults.variable):
                    raise TransferError(
                        "The format of the initial_value parameter for {} ({}) must match its variable ({})".format(
                            append_type_to_name(self),
                            initial_value,
                            self.instance_defaults.variable,
                        )
                    )

        # FIX: SHOULD THIS (AND SMOOTHING_FACTOR) JUST BE VALIDATED BY INTEGRATOR FUNCTION NOW THAT THEY ARE PROPERTIES??
        # Validate NOISE:
        if NOISE in target_set:
            noise = target_set[NOISE]
            # If assigned as a Function, set TransferMechanism as its owner, and assign its actual function to noise
            if isinstance(noise, DistributionFunction):
                noise.owner = self
                target_set[NOISE] = noise.function
            self._validate_noise(target_set[NOISE])

        # Validate SMOOTHING_FACTOR:
        if SMOOTHING_FACTOR in target_set:
            smoothing_factor = target_set[SMOOTHING_FACTOR]
            if (not (isinstance(smoothing_factor, (int, float)) and 0 <= smoothing_factor <= 1)) and (smoothing_factor != None):
                raise TransferError("smoothing_factor parameter ({}) for {} must be a float between 0 and 1".
                                    format(smoothing_factor, self.name))

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

        # self.integrator_function = Integrator(
        #     # default_variable=self.default_variable,
        #                                       initializer = self.instance_defaults.variable,
        #                                       noise = self.noise,
        #                                       rate = self.smoothing_factor,
        #                                       integration_type= ADAPTIVE)

    def _validate_noise(self, noise):
        # Noise is a list or array

        if isinstance(noise, (np.ndarray, list)):
            if len(noise) == 1:
                pass
            # Variable is a list/array
            elif not iscompatible(np.atleast_2d(noise), self.instance_defaults.variable) and len(noise) > 1:
                raise MechanismError(
                    "Noise parameter ({}) does not match default variable ({}). Noise parameter of {} must be specified"
                    " as a float, a function, or an array of the appropriate shape ({})."
                    .format(noise, self.instance_defaults.variable, self.name, np.shape(np.array(self.instance_defaults.variable))))
            else:
                for noise_item in noise:
                    if not isinstance(noise_item, (float, int)) and not callable(noise_item):
                        raise MechanismError(
                            "The elements of a noise list or array must be floats or functions. {} is not a valid noise"
                            " element for {}".format(noise_item, self.name))

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

        from psyneulink.components.functions.function import Logistic
        # If function is a logistic, and clip has not been specified, bound it between 0 and 1
        if ((isinstance(self.function, Logistic) or
                 (inspect.isclass(self.function) and issubclass(self.function,Logistic))) and
                self.clip is None):
            self.clip = (0,1)

        super()._instantiate_parameter_states(function=function, context=context)

    def _instantiate_attributes_before_function(self, function=None, context=None):

        super()._instantiate_attributes_before_function(function=function, context=context)

        if self.initial_value is None:
            self.initial_value = self.instance_defaults.variable

    def _instantiate_output_states(self, context=None):
        # If user specified more than one item for variable, but did not specify any custom OutputStates
        # then assign one OutputState (with the default name, indexed by the number of them) per item of variable
        if len(self.instance_defaults.variable) > 1 and len(self.output_states) == 1 and self.output_states[0] == RESULTS:
            self.output_states = []
            for i, item in enumerate(self.instance_defaults.variable):
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

        smoothing_factor = self.get_current_mechanism_param("smoothing_factor")

        if not self.integrator_function:

            self.integrator_function = AdaptiveIntegrator(
                function_variable,
                initializer=initial_value,
                noise=noise,
                rate=smoothing_factor,
                owner=self
            )

            self.original_integrator_function = self.integrator_function

        current_input = self.integrator_function.execute(
            function_variable,
            # Should we handle runtime params?
            runtime_params={
                # FIX: 4/30/18 - SHOULDN'T THESE BE THE PARAMS PASSED IN OR RETRIEVED ABOVE??
                INITIALIZER: self.initial_value,
                NOISE: self.noise,
                RATE: self.smoothing_factor
            },
            context=context
        )

        return current_input

    def _clip_result(self, clip, current_input, runtime_params, context):

        outputs = super(Mechanism, self)._execute(function_variable=current_input,
                                                  runtime_params=runtime_params,
                                                  context=context)
        if clip is not None:
            minCapIndices = np.where(outputs < clip[0])
            maxCapIndices = np.where(outputs > clip[1])
            outputs[minCapIndices] = np.min(clip)
            outputs[maxCapIndices] = np.max(clip)
        return outputs

    def _execute(
        self,
        variable=None,
        function_variable=None,
        runtime_params=None,
        context=None
    ):
        """Execute TransferMechanism function and return transform of input

        Execute TransferMechanism function on input, and assign to output_values:
            - Activation value for all units
            - Mean of the activation values across units
            - Variance of the activation values across units
        Return:
            value of input transformed by TransferMechanism function in outputState[TransferOuput.RESULT].value
            mean of items in RESULT outputState[TransferOuput.MEAN].value
            variance of items in RESULT outputState[TransferOuput.VARIANCE].value

        Arguments:

        # CONFIRM:
        variable (float): set to self.value (= self.input_value)
        - params (dict):  runtime_params passed from Mechanism, used as one-time value for current execution:
            + NOISE (float)
            + SMOOTHING_FACTOR (float)
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
        # Use self.instance_defaults.variable to initialize state of input

        # FIX: NEED TO GET THIS TO WORK WITH CALL TO METHOD:
        integrator_mode = self.integrator_mode
        noise = self.get_current_mechanism_param("noise")
        initial_value = self.get_current_mechanism_param("initial_value")


        # EXECUTE TransferMechanism FUNCTION ---------------------------------------------------------------------

        # FIX: NOT UPDATING self.previous_input CORRECTLY
        # FIX: SHOULD UPDATE PARAMS PASSED TO integrator_function WITH ANY RUNTIME PARAMS THAT ARE RELEVANT TO IT

        # Update according to time-scale of integration
        if integrator_mode:
            current_input = self._get_integrated_function_input(function_variable,
                                                                    initial_value,
                                                                    noise,
                                                                    context)

        else:
            current_input = self._get_instantaneous_function_input(function_variable, noise)

        clip = self.get_current_mechanism_param("clip")

        if isinstance(self.function_object, NormalizingFunction):
            # Apply TransferMechanism's function to each input state separately
            outputs = []
            for elem in current_input:
                output_item = self._clip_result(clip, elem, runtime_params, context)
                outputs.append(output_item)

        else:
            outputs = self._clip_result(clip, current_input, runtime_params, context)

        return outputs

    def _report_mechanism_execution(self, input, params, output):
        """Override super to report previous_input rather than input, and selected params
        """
        # KAM Changed 8/29/17 print_input = self.previous_input --> print_input = input
        # because self.previous_input is not a valid attrib of TransferMechanism

        print_input = input
        print_params = params.copy()
        # Suppress reporting of range (not currently used)
        del print_params[CLIP]

        super()._report_mechanism_execution(input_val=print_input, params=print_params)


    # def terminate_function(self, context=None):
    #     """Terminate the process
    #
    #     called by process.terminate() - MUST BE OVERRIDDEN BY SUBCLASS IMPLEMENTATION
    #     returns output
    #
    #     :rtype CurrentStateTuple(state, confidence, duration, controlModulatedParamValues)
    #     """
    #     # IMPLEMENTATION NOTE:  TBI when time_step is implemented for TransferMechanism
    #
    @property
    def clip(self):
        return self._clip


    @clip.setter
    def clip(self, value):
        self._clip = value

    # # MODIFIED 4/17/17 NEW:
    # @property
    # def noise (self):
    #     return self._noise
    #
    # @noise.setter
    # def noise(self, value):
    #     self._noise = value
    #
    # @property
    # def smoothing_factor(self):
    #     return self._time_constant
    #
    # @smoothing_factor.setter
    # def smoothing_factor(self, value):
    #     self._time_constant = value
    # # # MODIFIED 4/17/17 END

    @property
    def previous_value(self):
        if self.integrator_function:
            return self.integrator_function.previous_value
        return None

    @property
    def delta(self):
        if self.integrator_function:
            return self.value - self.integrator_function.previous_value
        return None

    @property
    def integrator_mode(self):
        return self._integrator_mode

    @integrator_mode.setter
    def integrator_mode(self, val):
        if val is True:
            if self.integrator_function is None:
                self.integrator_function = self.original_integrator_function
                self._integrator_mode = True
        elif val is False:
            if self.integrator_function is not None:
                self.original_integrator_function = self.integrator_function
            self.integrator_function = None
            self._integrator_mode = False
        else:
            raise MechanismError("{}'s integrator_mode attribute may only be True or False.".format(self.name))



