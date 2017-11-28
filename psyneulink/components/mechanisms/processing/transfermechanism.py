# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# NOTES:
#  * COULD NOT IMPLEMENT integrator_function in paramClassDefaults (see notes below)
#  * NOW THAT NOISE AND TIME_CONSTANT ARE PROPRETIES THAT DIRECTLY REFERERNCE integrator_function,
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
independent ones.  The function used to carry out the transformation can be selected from a standard set of `Functions
<Function>` (such as `Linear`, `Exponential`, `Logistic`, and `SoftMax`) or specified using a user-defined custom
function.  The transformation can be carried out instantaneously or in "time averaged" (integrated) manner, as described
in `Transfer_Execution`.

.. _Transfer_Creation:

Creating a TransferMechanism
-----------------------------

A TransferMechanism is created by calling its constructor.  Its `function <TransferMechanism.function>` is specified in
the **function** argument, which can be the name of a `Function <Function>` class (first example below), or a call to
a Function constructor that can include arguments specifying the Function's parameters (second example)::

    >>> import psyneulink as pnl
    >>> my_linear_transfer_mechanism = pnl.TransferMechanism(function=pnl.Linear)
    >>> my_logistic_transfer_mechanism = pnl.TransferMechanism(function=pnl.Logistic(gain=1.0, bias=-4))

In addition to Function-specific parameters, `noise <TransferMechanism.noise>` and `time_constant
<TransferMechanism.time_constant>` parameters can be specified for the Mechanism (see `Transfer_Execution`).


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

*Function*.  The `function <TransferMechanism.function>` can be selected from one of four standard PsyNeuLink
`Functions <Function>`: `Linear`, `Logistic`, `Exponential` or `SoftMax`; or a custom function can be specified,
so long as it returns a numeric value or a list or numpy array of numeric values.  The result of the `function
<TransferMechanism.function>` applied to the `value <InputState.value>` of each InputState is to an item of an
array as the TransferMechanism's `value <TransferMechanism.value>`, and as the `value <OutputState.value>` of each
of its `OutputStates <OutputState>` (one corresponding to each InputState).

.. _TransferMechanism_OutputStates:

OutputStates
~~~~~~~~~~~~

By default, a TransferMechanism generates one `OutputState` for each of its `InputStates`.  The first (and `primary
<OutputState_Primary>`) OutputState is named *RESULT*; subsequent ones use that as the base name, suffixed with an
incrementing integer starting at '-1' for each additional OutputState (e.g., *RESULT-1*, *RESULT-2*, etc.; see
`Naming`).  The `value <OutputState.value>` of each OutputState is assigned the result of the Mechanism's `function
<TransferMechanism.function>` applied to the `value <InputState.value>` of the corresponding InputState. Additional
OutputStates can be assigned using the TransferMechanism's `Standard OutputStates
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

    * `noise <TransferMechanism.noise>`: applied element-wise to the input before transforming it.
    ..
    * `clip <TransferMechanism.clip>`: caps all elements of the `function <TransferMechanism.function>` result by
      the lower and upper values specified by clip.
    ..
    * `integrator_mode <TransferMechanism.integrator_mode>`: determines whether the input will be time-averaged before
      passing through the function of the mechanisms. When `integrator_mode <TransferMechanism.integrator_mode>` is set
      to True, the TransferMechanism exponentially time-averages its input before transforming it.
    ..
    * `time_constant <TransferMechanism.time_constant>`: if the `integrator_mode <TransferMechanism.integrator_mode>`
      attribute is set to True, the `time_constant <TransferMechanism.time_constant>` attribute is the rate of
      integration (a higher value specifies a faster rate); if `integrator_mode <TransferMechanism.integrator_mode>` is
      False, `time_constant <TransferMechanism.time_constant>` is ignored and time-averaging does not occur.

After each execution of the Mechanism the result of `function <TransferMechanism.function>` applied to each
`InputState` is assigned as an item of the Mechanism's `value <TransferMechanism.value>`, and the `value
<OutputState.value>` of each of its `OutputStates <OutputState>`, and to the 1st item of the Mechanism's
`output_values <TransferMechanism.output_values>` attribute.

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
from psyneulink.components.functions.function import AdaptiveIntegrator, Linear, TransferFunction
from psyneulink.components.mechanisms.mechanism import Mechanism, MechanismError
from psyneulink.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.components.mechanisms.adaptive.control.controlmechanism import _is_control_spec
from psyneulink.components.states.inputstate import InputState
from psyneulink.components.states.outputstate import OutputState, PRIMARY, StandardOutputStates, standard_output_states
from psyneulink.globals.keywords import NAME, INDEX, FUNCTION, INITIALIZER, INITIALIZING, MEAN, MEDIAN, NOISE, RATE, \
    RESULT, RESULTS, STANDARD_DEVIATION, TRANSFER_FUNCTION_TYPE, NORMALIZING_FUNCTION_TYPE, TRANSFER_MECHANISM, \
    VARIANCE, kwPreferenceSetName
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set, kpReportOutputPref, kpRuntimeParamStickyAssignmentPref
from psyneulink.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel
from psyneulink.globals.utilities import append_type_to_name, iscompatible
from psyneulink.scheduling.timescale import CentralClock, TimeScale

__all__ = [
    'INITIAL_VALUE', 'CLIP', 'TIME_CONSTANT', 'Transfer_DEFAULT_BIAS', 'Transfer_DEFAULT_GAIN', 'Transfer_DEFAULT_LENGTH',
    'Transfer_DEFAULT_OFFSET', 'TRANSFER_OUTPUT', 'TransferError', 'TransferMechanism',
]

# TransferMechanism parameter keywords:
CLIP = "clip"
TIME_CONSTANT = "time_constant"
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
    time_constant=1.0,           \
    integrator_mode=False,       \
    clip=(float:min, float:max), \
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
        a stochastically-sampled value added to the result of the `function <TransferMechanism.function>`:
        if it is a float, it must be in the interval [0,1] and is used to scale the variance of a zero-mean Gaussian;
        if it is a function, it must return a scalar value.

    time_constant : float : default 1.0
        the time constant for exponential time averaging of input when the Mechanism is executed with `integrator_mode`
        set to True::

         result = (time_constant * current input) + ((1-time_constant) * result on previous time_step)

    clip : Optional[Tuple[float, float]]
        specifies the allowable range for the result of `function <TransferMechanism.function>`:
        the first item specifies the minimum allowable value of the result, and the second its maximum allowable value;
        any element of the result that exceeds the specified minimum or maximum value is set to the value of
        `clip <TransferMechanism.clip>` that it exceeds.

    output_states : str, list or np.ndarray : default RESULTS
        specifies the OutputStates for the TransferMechanism; by default, one is created for each InputState
        specified in **input_states**;  see `note <TransferMechanism_OutputStates_Note>`, and `output_states
        <TransferMechanism.output_states>` for additional details).

    params : Dict[param keyword, param value] : default None
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
    initial_value :  value, list or np.ndarray : Transfer_DEFAULT_BIAS
        specifies the starting value for time-averaged input (only relevant if `integrator_mode
        <TransferMechanism.integrator_mode>` is True and `time_constant <TransferMechanism.time_constant>` is not 1.0).
        COMMENT:
            Transfer_DEFAULT_BIAS SHOULD RESOLVE TO A VALUE
        COMMENT

    noise : float or function
        a stochastically-sampled value added to the output of the `function <TransferMechanism.function>`:
        if it is a float, it must be in the interval [0,1] and is used to scale the variance of a zero-mean Gaussian;
        if it is a function, it must return a scalar value.

    time_constant : float
        the time constant for exponential time averaging of input when the Mechanism is executed with `integrator_mode`
        set to True::

          result = (time_constant * current input) + ( (1-time_constant) * result on previous time_step)

    integrator_mode : booleane
        when set to True, the Mechanism time averages its input according to an exponentially weighted moving average
        (see `time_constant <TransferMechanisms.time_constant>`).

    clip : Optional[Tuple[float, float]]
        determines the allowable range of the result: the first value specifies the minimum allowable value
        and the second the maximum allowable value;  any element of the result that exceeds minimum or maximum
        is set to the value of `clip <TransferMechanism.clip>` it exceeds.  If `function <TransferMechanism.function>`
        is `Logistic`, `clip <TransferMechanism.clip>` is set by default to (0,1).

    value : 2d np.array [array(float64)]
        result of executing `function <TransferMechanism.function>`.

    previous_value : float
        the `value <TransferMechanism.value>` on the previous execution of the Mechanism.

    delta : float
        the change in `value <TransferMechanism.value>` from the previous execution of the Mechanism
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
    classPreferences = {
        kwPreferenceSetName: 'TransferCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
        kpRuntimeParamStickyAssignmentPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
    }

    # TransferMechanism parameter and control signal assignments):
    paramClassDefaults = ProcessingMechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({NOISE: None})

    standard_output_states = standard_output_states.copy()

    class ClassDefaults(ProcessingMechanism_Base.ClassDefaults):
        variable = [[0]]

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_states:tc.optional(tc.any(Iterable, Mechanism, OutputState, InputState))=None,
                 function=Linear,
                 initial_value=None,
                 noise=0.0,
                 time_constant=1.0,
                 integrator_mode=False,
                 clip=None,
                 output_states:tc.optional(tc.any(str, Iterable))=RESULTS,
                 time_scale=TimeScale.TRIAL,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=componentType+INITIALIZING):
        """Assign type-level preferences and call super.__init__
        """

        # Default output_states is specified in constructor as a string rather than a list
        # to avoid "gotcha" associated with mutable default arguments
        # (see: bit.ly/2uID3s3 and http://docs.python-guide.org/en/latest/writing/gotchas/)
        if output_states is None or output_states is RESULTS:
            output_states = [RESULTS]

        params = self._assign_args_to_param_dicts(function=function,
                                                  initial_value=initial_value,
                                                  input_states=input_states,
                                                  output_states=output_states,
                                                  noise=noise,
                                                  time_constant=time_constant,
                                                  integrator_mode=integrator_mode,
                                                  time_scale=time_scale,
                                                  clip=clip,
                                                  params=params)

        self.integrator_function = None

        if not isinstance(self.standard_output_states, StandardOutputStates):
            self.standard_output_states = StandardOutputStates(self,
                                                               self.standard_output_states,
                                                               indices=PRIMARY)

        super(TransferMechanism, self).__init__(
            variable=default_variable,
            size=size,
            params=params,
            name=name,
            prefs=prefs,
            context=self,
            input_states=input_states,
        )

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate FUNCTION and Mechanism params

        """

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        # Validate FUNCTION
        if FUNCTION in target_set:
            transfer_function = target_set[FUNCTION]
            # FUNCTION is a Function
            if isinstance(transfer_function, Component):
                transfer_function_class = transfer_function.__class__
                transfer_function_name = transfer_function.__class__.__name__
            # FUNCTION is a function or method
            elif isinstance(transfer_function, (function_type, method_type)):
                transfer_function_class = transfer_function.__self__.__class__
                transfer_function_name = transfer_function.__self__.__class__.__name__
            # FUNCTION is a class
            elif inspect.isclass(transfer_function):
                transfer_function_class = transfer_function
                transfer_function_name = transfer_function.__name__

            if not transfer_function_class.componentType is TRANSFER_FUNCTION_TYPE and not transfer_function_class.componentType is NORMALIZING_FUNCTION_TYPE:
                raise TransferError("Function {} specified as FUNCTION param of {} must be a {}".
                                    format(transfer_function_name, self.name, TRANSFER_FUNCTION_TYPE))

        # Validate INITIAL_VALUE
        if INITIAL_VALUE in target_set:
            initial_value = target_set[INITIAL_VALUE]
            if initial_value is not None:
                if not iscompatible(initial_value, self.instance_defaults.variable):
                    raise Exception(
                        "initial_value is {}, type {}\nself.instance_defaults.variable is {}, type {}".format(
                            initial_value,
                            type(initial_value).__name__,
                            self.instance_defaults.variable,
                            type(self.instance_defaults.variable).__name__,
                        )
                    )
                    raise TransferError(
                        "The format of the initial_value parameter for {} ({}) must match its input ({})".format(
                            append_type_to_name(self),
                            initial_value,
                            self.instance_defaults.variable[0],
                        )
                    )

        # FIX: SHOULD THIS (AND TIME_CONSTANT) JUST BE VALIDATED BY INTEGRATOR FUNCTION NOW THAT THEY ARE PROPERTIES??
        # Validate NOISE:
        if NOISE in target_set:
            self._validate_noise(target_set[NOISE], self.instance_defaults.variable)
        # Validate TIME_CONSTANT:
        if TIME_CONSTANT in target_set:
            time_constant = target_set[TIME_CONSTANT]
            if (not (isinstance(time_constant, float) and 0 <= time_constant <= 1)) and (time_constant != None):
                raise TransferError("time_constant parameter ({}) for {} must be a float between 0 and 1".
                                    format(time_constant, self.name))

        # Validate RANGE:
        if CLIP in target_set:
            clip = target_set[CLIP]
            if clip:
                if not (isinstance(clip, tuple) and len(clip)==2 and all(isinstance(i, numbers.Number) for i in clip)):
                    raise TransferError("clip parameter ({}) for {} must be a tuple with two numbers".
                                        format(clip, self.name))
                if not clip[0] < clip[1]:
                    raise TransferError("The first item of the clip parameter ({}) must be less than the second".
                                        format(clip, self.name))

        # self.integrator_function = Integrator(
        #     # default_variable=self.default_variable,
        #                                       initializer = self.instance_defaults.variable,
        #                                       noise = self.noise,
        #                                       rate = self.time_constant,
        #                                       integration_type= ADAPTIVE)

    def _validate_noise(self, noise, var):
        # Noise is a list or array
        if isinstance(noise, (np.ndarray, list)):
            if len(noise) == 1:
                pass
            # Variable is a list/array
            elif not iscompatible(np.atleast_2d(noise), var) and len(noise) > 1:
                raise MechanismError(
                    "Noise parameter ({}) does not match default variable ({}). Noise parameter of {} must be specified"
                    " as a float, a function, or an array of the appropriate shape ({})."
                    .format(noise, self.instance_defaults.variable, self.name, np.shape(np.array(var))))
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

    def _instantiate_parameter_states(self, context=None):

        from psyneulink.components.functions.function import Logistic
        # If function is a logistic, and clip has not been specified, bound it between 0 and 1
        if ((isinstance(self.function, Logistic) or
                 (inspect.isclass(self.function) and issubclass(self.function,Logistic))) and
                self.clip is None):
            self.clip = (0,1)

        super()._instantiate_parameter_states(context=context)

    def _instantiate_attributes_before_function(self, context=None):

        super()._instantiate_attributes_before_function(context=context)

        if self.initial_value is None:
            self.initial_value = self.instance_defaults.variable

    def _instantiate_output_states(self, context=None):
        # If user specified more than one item for variable, but did not specify any custom OutputStates
        # then assign one OutputState (with the default name, indexed by the number of them) per item of variable
        if len(self.variable) > 1 and len(self.output_states) == 1 and self.output_states[0] == RESULTS:
            self.output_states = []
            for i, item in enumerate(self.variable):
                self.output_states.append({NAME: RESULT, INDEX: i})
        super()._instantiate_output_states(context=context)

    def _execute(self,
                 variable=None,
                 runtime_params=None,
                 clock=CentralClock,
                 time_scale=TimeScale.TRIAL,
                 context=None):
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
            + TIME_CONSTANT (float)
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

        #region ASSIGN PARAMETER VALUES

        time_constant = self.time_constant
        clip = self.clip
        noise = self.noise
        #endregion

        #region EXECUTE TransferMechanism FUNCTION ---------------------------------------------------------------------

        # FIX: NOT UPDATING self.previous_input CORRECTLY
        # FIX: SHOULD UPDATE PARAMS PASSED TO integrator_function WITH ANY RUNTIME PARAMS THAT ARE RELEVANT TO IT

        # Update according to time-scale of integration
        if integrator_mode:
        # if time_scale is TimeScale.TIME_STEP:

            if not self.integrator_function:

                self.integrator_function = AdaptiveIntegrator(
                                            variable,
                                            initializer = self.initial_value,
                                            noise = self.noise,
                                            rate = self.time_constant,
                                            owner = self)

            current_input = self.integrator_function.execute(variable,
                                                        # Should we handle runtime params?
                                                              params={INITIALIZER: self.initial_value,
                                                                      NOISE: self.noise,
                                                                      RATE: self.time_constant},
                                                              context=context

                                                             )
        else:
        # elif time_scale is TimeScale.TRIAL:
            noise = self._try_execute_param(self.noise, variable)
            # formerly: current_input = self.input_state.value + noise
            # (MODIFIED 7/13/17 CW) this if/else below is hacky: just allows a nicer error message
            # when the input is given as a string.
            if (np.array(noise) != 0).any():
                current_input = variable + noise
            else:
                current_input = variable

        if isinstance(self.function_object, TransferFunction):

            outputs = self.function(variable=current_input, params= runtime_params)
            # if clip is not None:
            #     print(clip)
            #     minCapIndices = np.where(outputs < clip[0])
            #     print(minCapIndices)
            #     maxCapIndices = np.where(outputs > clip[1])
            #     print(maxCapIndices)
            #     outputs[minCapIndices] = np.min(clip)
            #     outputs[maxCapIndices] = np.max(clip)
        else:
            # Apply TransferMechanism's function to each input state separately
            outputs = []
            for elem in current_input:
                output_item = self.function(variable=elem, params=runtime_params)
                # if clip is not None:
                #     minCapIndices = np.where(output_item < clip[0])
                #     maxCapIndices = np.where(output_item > clip[1])
                #     output_item[minCapIndices] = np.min(clip)
                #     output_item[maxCapIndices] = np.max(clip)
                outputs.append(output_item)

        # outputs = []
        # for elem in current_input:
        #     output_item = self.function(variable=elem, params=runtime_params)
        #     if clip is not None:
        #         minCapIndices = np.where(output_item < clip[0])
        #         maxCapIndices = np.where(output_item > clip[1])
        #         output_item[minCapIndices] = np.min(clip)
        #         output_item[maxCapIndices] = np.max(clip)
        #     outputs.append(output_item)
        return outputs
        #endregion

    def _report_mechanism_execution(self, input, params, output):
        """Override super to report previous_input rather than input, and selected params
        """
        # KAM Changed 8/29/17 print_input = self.previous_input --> print_input = input
        # because self.previous_input is not a valid attrib of TransferMechanism

        print_input = input
        print_params = params.copy()
        # Only report time_constant if in TIME_STEP mode
        if params['time_scale'] is TimeScale.TRIAL:
            del print_params[TIME_CONSTANT]
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

    # MODIFIED 4/17/17 NEW:
    @property
    def noise (self):
        return self._noise

    @noise.setter
    def noise(self, value):
        self._noise = value

    @property
    def time_constant(self):
        return self._time_constant

    @time_constant.setter
    def time_constant(self, value):
        self._time_constant = value
    # # MODIFIED 4/17/17 END

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
