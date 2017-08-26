# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ****************************************************** KWTA **********************************************************

"""

Overview
--------

A KWTA is a subclass of `RecurrentTransferMechanism` that implements a k-winners-take-all (kWTA) constraint on the
number of elements of the Mechanism's `value <KWTA.value>` that are above a specified threshold.  The implementation is
based on the one  described in `O'Reilly and Munakata, 2012
<https://grey.colorado.edu/CompCogNeuro/index.php/CCNBook/Networks/kWTA_Equations>`_.

.. _KWTA_Creation:

Creating a KWTA
---------------

A KWTA Mechanism can be created directly by calling its constructor, or by using the `mechanism()
<Mechanism.mechanism>` command and specifying KWTA as its **mech_spec** argument. The **k_value**, **threshold**,
and **ratio** arguments can be used to specify the function of the KWTA Mechanism, and default to a condition
in which half of the elements in the KWTA Mechanism's `value <KWTA.value>` (**k_value** = 0.5) are above 0 and half
are below (**threshold** = 0), achieved using an intermediate degree of value displacement (**ratio** = 0.5).


.. _KWTA_Structure:

Structure
---------

The KWTA calculates an offset to apply to all elements of the Mechanism's `variable <KWTA.variable>` array so that it
has a specified number of the elements that are at or above a specified threshold value.  Typically, this constraint
can be satisfied in a number of ways;  how it is satisfied is determined by three parameters and two options of the
KWTA:

.. _KWTA_k_value:

* `k_value <KWTA.k_value>` parameter -- determines the number of elements of its `variable <KWTA.value>` that should be
  at or above the specified `threshold <KWTA.threshold>`.  A value between 0 and 1 specifies the proportion of elements
  that should be at or above the `threshold <KWTA.threshold>`, while a positive integer specifies the number of values
  that should be at or above the `threshold <KWTA.threshold>`. A negative integer specifies the number of elements that
  should be below the `threshold <KWTA.threshold>`.  Whether or not the exact specificadtion is achieved depends on the
  settings of the `average_based <KWTA.average_based>` and `inhibition_only <KWTA.inhibition_only>` options (see below).

.. _KWTA_threshold:

* `threshold <KWTA.threshold>` parameter -- determines the value at or above which the KTWA seeks to assign `k_value
  <KWTA.k_value>` elements of its `variable <KWTA.variable>`.

.. _KWTA_ratio:

* `ratio <KWTA.ratio>` parameter -- determines how the offset applied to the elements of the KWTA's `variable
  <KWTA.variable>` is selected from the scope of possible values;  the `ratio <KWTA.ratio>`  must be a number between 0
  and 1.  An offset is picked that is above the low end of the scope by a proportion of the scope equal to the `ratio
  <KWTA.ratio>` parameter.  How the scope is calculated is determined by the `average_based <KWTA.average_based>`
  option, as described below.

.. _KWTA_average_based:

* `average_based <KWTA.average_based>` option -- determines the scope of values for the offset applied to the KWTA's
  `variable <KWTA.variable>` is calculated; the offset is then chosen based on the `ratio <KWTA.ratio>` parameter.  If
  `average_based <KWTA.average_based>` is `False`, the low end of the scope is the offset that sets the k-th highest
  element exactly at the threshold (that is, the smallest value that insures that `k_value <KWTA.k_value>` elements are
  at or above the `threshold <KWTA.threshold>`;  the high end of the scope is the offset that sets the k+1-th highest
  element exactly at the threshold (that is, the largest possible value, such that the `k_value <KWTA.k_value>` elements
  but no more are above the `threshold <KWTA.threshold>` (i.e., the next one is exactly at it). In this case, all values
  of offset within the scope generate exactly `k_value <KTWA.k_value>` elements at or above the `threshold
  <KWTA.threshold>`.  If `average_based <KWTA.average_based>` is `True`, the low end of the scope is the offset that
  places the *average* of the elements with the `k_value <KWTA.k_value>` highest values at the `threshold
  <KWTA.threshold>`, and the high end of the scope is the offset that places the average of the remaining elements at
  the `threshold <KWTA.threshold>`.  In this case, the lowest values of offset within the scope may produce fewer than
  `k_value <KWTA.k_value>` elements at or above the `threshold <KWTA.threshold>`, while the highest values within the
  scope may produce more.  An offset is picked from the scope as specified by the `ratio <KWTA.ratio>` parameter
  (see `above <KWTA_ratio>`).

  .. note::
     If the `average_based <KWTA.average_based>` option is `False` (the default), the KWTA's `variable <KWTA.variable>`
     always has exactly `k_value <KTWA.k_value>` elements at or above the `threshold <KWTA.threshold>` (that is, for
     *any* value of the `ratio <KTWA.ratio>`).  However, if `average_based <KWTA.average_based>` is `True`, the
     `variable <KWTA.variable` may have fewer than `k_value <KWTA.k_value>` elements at or above the `threshold
     <KWTA.threshold>` (if the `ratio <KWTA.ratio>` is low), or more than `k_value <KWTA.k_value>` (if the `ratio
     <KWTA.ratio>` is high).

  COMMENT:
  WHY USE THIS OPTION?
  COMMENT

.. _KWTA_inhibition_only:

* `inhibition_only <KWTA.inhibition_only>` option -- determines whether the offset applied to the elements of the
  KWTA's `variable <KWTA.variable>` is allowed to be positive (i.e., whether the KWTA can increase the value of any
  elements of its `variable <KWTA.variable>`).  If set to `False`, the KWTA will use any offset value determined by
  the `ratio <KWTA.ratio>` parameter from the scope determined by the `average_based <KTWA.average_based>` option
  (including positive offsets). If `inhibition_only <KWTA.inhibition_only>` is `True`, then any positive offset
  selected is "clipped" at (i.e re-assigned a value of) 0.  This insures that the values of the elements of the KWTA's
  `variable <KWTA.variable>` are never increased.

  .. note::
     If the `inhibition_only <KWTA.inhibition_only>` option is set to `True`, the number of elements at or above the
     `threshold <KWTA.threshold>` may fall below `k_value <KWTA.k_value>`; and, if the input to the KWTA is sufficiently
     low, the value of all elements may decay to 0 (depending on the value of the `decay <KWTA.decay>` parameter.

In all other respects, a KWTA has the same attributes and is specified in the same way as a standard
`RecurrentTransferMechanism`.


.. _KWTA_Execution:

Execution
---------

When a KTWA is executed, it first determines its `variable <KWTA.variable>` as follows:

* First, like every `RecurrentTransferMechanism`, it combines the input it receives from its recurrent
  `AutoAssociativeProjection` (see `Recurrent_Transfer_Structure <Recurrent_Transfer_Structure>`) with the input
  from any other `MappingProjections <MappingProjection>` it receives, and assigns this to its `variable
  <KWTA.variable>` attribute.

* Then it modifies its `variable <KWTA.variable>`, by calculating and assigning an offset to its elements, so that
  as close to `k_value <KWTA.k_value>` elements as possible are at or above the `threshold <KWTA.threshold>`.  The
  offset is determined by carrying out the following steps in each execution of the KTWA:

  - calculate the scope of offsets that will satisfy the constraint; how this is done is determined by the
    `average_based <KWTA.average_based>` attribute (see `above <KWTA_average_based>`);

  - select an offset from the scope based on the `ratio <KWTA.ratio>` option (see `above <KWTA_ratio>`);

  - constrain the offset to be 0 or negative if the `inhibition_only <KWTA.inhibition_only>` option is set (see `above
    <KWTA_inhibition_only>`;

  - apply the offset to all elements of the `variable <KWTA.variable>`.

The modifed `variable <KWTA.variable>` is then passed to the KWTA's `function <KWTA.function>` to determine its
`value <KWTA.value>`.


.. _KWTA_Reference:

Class Reference
---------------

"""

import builtins
import numbers
import warnings

import numpy as np
import typecheck as tc

import logging

from PsyNeuLink.Components.Functions.Function import Logistic, get_matrix
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism_Base
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.RecurrentTransferMechanism import RecurrentTransferMechanism
from PsyNeuLink.Components.Projections.PathwayProjections.AutoAssociativeProjection import AutoAssociativeProjection, get_auto_matrix, get_hetero_matrix
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Globals.Keywords import AUTO, FULL_CONNECTIVITY_MATRIX, HETERO, INITIALIZING, KWTA, K_VALUE, MATRIX, RATIO, RESULT, THRESHOLD
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set, kpVerbosePref
from PsyNeuLink.Globals.Utilities import is_numeric_or_none
from PsyNeuLink.Scheduling.TimeScale import CentralClock, TimeScale

logger = logging.getLogger(__name__)

class KWTAError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

class KWTA(RecurrentTransferMechanism):
    """
    KWTA(                       \
    default_variable=None,      \
    size=None,                  \
    function=Logistic,          \
    matrix=None,                \
    auto=None,                  \
    hetero=None,                \
    initial_value=None,         \
    decay=1.0,                  \
    noise=0.0,                  \
    time_constant=1.0,          \
    k_value=0.5,                \
    threshold=0,                \
    ratio=0.5,                  \
    average_based=False,        \
    inhibition_only=True,       \
    range=None,                 \
    time_scale=TimeScale.TRIAL, \
    params=None,                \
    name=None,                  \
    prefs=None)

    Subclass of `RecurrentTransferMechanism` that dynamically regulates its input relative to a given threshold.

    Arguments
    ---------

    default_variable : number, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the input to the mechanism to use if none is provided in a call to its
        `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` method;
        also serves as a template to specify the length of `variable <KWTA.variable>` for
        `function <KWTA.function>`, and the `primary outputState <OutputState_Primary>`
        of the mechanism.

    size : int, list or np.ndarray of ints
        specifies variable as array(s) of zeros if **variable** is not passed as an argument;
        if **variable** is specified, it takes precedence over the specification of **size**.

    function : TransferFunction : default Linear
        specifies the function used to transform the input;  can be `Linear`, `Logistic`, `Exponential`,
        or a custom function.

    matrix : list, np.ndarray, np.matrix, matrix keyword, or AutoAssociativeProjection : default FULL_CONNECTIVITY_MATRIX
        specifies the matrix to use for creating a `recurrent AutoAssociativeProjection <Recurrent_Transfer_Structure>`,
        or a AutoAssociativeProjection to use. If **auto** or **hetero** arguments are specified, the **matrix** argument
        will be ignored in favor of those arguments.

    auto : number, 1D array, or None : default None
        specifies matrix as a diagonal matrix with diagonal entries equal to **auto**, if **auto** is not None;
        If **auto** and **hetero** are both specified, then matrix is the sum of the two matrices from **auto** and
        **hetero**. For example, setting **auto** to 1 and **hetero** to -1 would set matrix to have a diagonal of
        1 and all non-diagonal entries -1. if the **matrix** argument is specified, it will be overwritten by
        **auto** and/or **hetero**, if either is specified. **auto** can be specified as a 1D array with length equal
        to the size of the mechanism, if a non-uniform diagonal is desired. Can be modified by control.

    hetero : number, 2D array, or None : default None
        specifies matrix as a hollow matrix with all non-diagonal entries equal to **hetero**, if **hetero** is not None;
        If **auto** and **hetero** are both specified, then matrix is the sum of the two matrices from **auto** and
        **hetero**. For example, setting **auto** to 1 and **hetero** to -1 would set matrix to have a diagonal of
        1 and all non-diagonal entries -1. if the **matrix** argument is specified, it will be overwritten by
        **auto** and/or **hetero**, if either is specified. **hetero** can be specified as a 2D array with dimensions
        equal to the matrix dimensions, if a non-uniform diagonal is desired. Can be modified by control.

    initial_value :  value, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the starting value for time-averaged input (only relevant if
        `time_constant <KWTA.time_constant>` is not 1.0).
        :py:data:`Transfer_DEFAULT_BIAS <LINK->SHOULD RESOLVE TO VALUE>`

    decay : number : default 1.0
        specifies the amount by which to decrement its `previous_input <KWTA.previous_input>` each time it is executed.

    noise : float or function : default 0.0
        specifies a stochastically-sampled value added to the result of the `function <KWTA.function>`:
        if it is a float, it must be in the interval [0,1] and is used to scale the variance of a zero-mean Gaussian;
        if it is a function, it must return a scalar value.

    time_constant : float : default 1.0
        the time constant for exponential time averaging of input when the mechanism is executed with `time_scale`
        set to `TimeScale.TIME_STEP`::

         result = (time_constant * current input) +
         (1-time_constant * result on previous time_step)

    k_value : number : default 0.5
        specifies the proportion or number of the elements of `variable <KWTA.variable>` that should be at or above
        the `threshold <KWTA.threshold>`. A value between 0 and 1 specifies the proportion of elements that should be at
        or above the `threshold <KWTA.threshold>`, while a positive integer specifies the number of values that should
        be at or above the `threshold <KWTA.threshold>`. A negative integer specifies the number of elements that should
        be below the `threshold <KWTA.threshold>`.

    threshold : number : default 0
        specifies the threshold at or above which the KTWA seeks to assign `k_value <KWTA.k_value>` elements of its
        `variable <KWTA.variable>`.

    ratio : number : default 0.5
        specifies the offset used to adjust the elements of `variable <KWTA.variable>` so that there are the number
        specified by `k_value <KWTA.k_value>` at or above the `threshold <KWTA.threshold>`;  it must be a
        number from 0 to 1 (see `ratio <KWTA_ratio>` for additional information).

    average_based : boolean : default False
        specifies whether the average-based scaling is used to determine the scope of offsets (see `average_based
        <KWTA_average_based>` for additional information).

    inhibition_only : boolean : default True
        specifies whether positive offsets can be applied to the `variable <KWTA.variable>` in an effort to achieve
        `k_value <KWTA.k_value>` elements at or above the `threshold <KWTA.threshold>`.  If set to `False`, any offset
        is allowed, including positive offsets;  if set to `True`, a positive offset will be re-assigned the value of 0
        (see `inhibition_only <KWTA_inhibition_only>` for additional information).

    range : Optional[Tuple[float, float]]
        specifies the allowable range for the result of `function <KWTA.function>`:
        the first item specifies the minimum allowable value of the result, and the second its maximum allowable value;
        any element of the result that exceeds the specified minimum or maximum value is set to the value of
        `range <KWTA.range>` that it exceeds.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the mechanism, its function, and/or a custom function and its parameters.  Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the constructor.

    time_scale :  TimeScale : TimeScale.TRIAL
        specifies whether the mechanism is executed using the `TIME_STEP` or `TRIAL` `TimeScale`.
        This must be set to `TimeScale.TIME_STEP` for the `time_constant <KWTA.time_constant>`
        parameter to have an effect.

    name : str : default KWTA-<index>
        a string used for the name of the mechanism.
        If not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Mechanism.classPreferences]
        the `PreferenceSet` for mechanism.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    context : str : default componentType+INITIALIZING
        string used for contextualization of instantiation, hierarchical calls, executions, etc.

    Attributes
    ----------

    variable : value
        the input to Mechanism's `function <KWTA.variable>`.

    function : Function
        the Function used to transform the input.

    matrix : 2d np.array
        the `matrix <AutoAssociativeProjection.matrix>` parameter of the `recurrent_projection` for the Mechanism.

    recurrent_projection : AutoAssociativeProjection
        an `AutoAssociativeProjection` that projects from the Mechanism's `primary outputState <OutputState_Primary>`
        back to its `primary inputState <Mechanism_InputStates>`.

    decay : float : default 1.0
        determines the amount by which to multiply the `previous_input <KWTA.previous_input>` value
        each time it is executed.

    COMMENT:
       THE FOLLOWING IS THE CURRENT ASSIGNMENT
    COMMENT
    initial_value :  value, list or np.ndarray : Transfer_DEFAULT_BIAS
        determines the starting value for time-averaged input
        (only relevant if `time_constant <KWTA.time_constant>` parameter is not 1.0).
        :py:data:`Transfer_DEFAULT_BIAS <LINK->SHOULD RESOLVE TO VALUE>`

    noise : float or function : default 0.0
        a stochastically-sampled value added to the output of the `function <KWTA.function>`:
        if it is a float, it must be in the interval [0,1] and is used to scale the variance of a zero-mean Gaussian;
        if it is a function, it must return a scalar value.

    time_constant : float
        the time constant for exponential time averaging of input
        when the Mechanism is executed using the `TIME_STEP` `TimeScale`::

          result = (time_constant * current input) + (1-time_constant * result on previous time_step)

    k_value : number
        determines the number or proportion of elements of `variable <KWTA.variable>` that should be above the
        `threshold <KWTA.threshold>` of the KWTA (see `k_value <KWTA_k_value>` for additional information).

    threshold : number
        determines the threshold at or above which the KTWA seeks to assign `k_value <KWTA.k_value>` elements of its
        `variable <KWTA.variable>`.

    ratio : number
        determines the offset used to adjust the elements of `variable <KWTA.variable>` so that there are `k_value
        <KWTA.k_value>` elements at or above the `threshold <KWTA.threshold>` (see `ratio <KWTA_ratio>` for additional
        information).

    average_based : boolean : default False
        determines the way in which the scope of offsets is determined, from which the one is selected that is applied
        to the elements of the `variable <KWTA.variable>` (see `average_based <KWTA_average_based>` for additional
        information).

    inhibition_only : boolean : default True
        determines whether a positive offset is allowed;  if it is `True`, then the value of the offset is
        "clipped" at (that is, any positive value is replaced by) 0.  Otherwise, any offset is allowed (see
        `inhibition_only <KWTA_inhibition_only>` for additional information).

    range : Tuple[float, float]
        determines the allowable range of the result: the first value specifies the minimum allowable value
        and the second the maximum allowable value;  any element of the result that exceeds minimum or maximum
        is set to the value of `range <KWTA.range>` it exceeds.  If `function <KWTA.function>`
        is `Logistic`, `range <KWTA.range>` is set by default to (0,1).

    previous_input : 1d np.array of floats
        the value of the input on the previous execution, including the value of `recurrent_projection`.

    value : 2d np.array [array(float64)]
        result of executing `function <KWTA.function>`; same value as first item of
        `output_values <KWTA.output_values>`.

    COMMENT:
        CORRECTED:
        value : 1d np.array
            the output of ``function``;  also assigned to ``value`` of the TRANSFER_RESULT OutputState
            and the first item of ``output_values``.
    COMMENT

    outputStates : Dict[str, OutputState]
        an OrderedDict with the following `outputStates <OutputState>`:

        * `TRANSFER_RESULT`, the :keyword:`value` of which is the **result** of `function <KWTA.function>`;
        * `TRANSFER_MEAN`, the :keyword:`value` of which is the mean of the result;
        * `TRANSFER_VARIANCE`, the :keyword:`value` of which is the variance of the result;
        * `ENERGY`, the :keyword:`value` of which is the energy of the result,
          calculated using the `Stability` Function with the ENERGY metric;
        * `ENTROPY`, the :keyword:`value` of which is the entropy of the result,
          calculated using the `Stability` Function with the ENTROPY metric;
          note:  this is only present if the mechanism's :keyword:`function` is bounded between 0 and 1
          (e.g., the `Logistic` function).

    output_values : List[array(float64), float, float]
        a list with the following items:

        * **result** of the ``function`` calculation (value of TRANSFER_RESULT outputState);
        * **mean** of the result (``value`` of TRANSFER_MEAN outputState)
        * **variance** of the result (``value`` of TRANSFER_VARIANCE outputState);
        * **energy** of the result (``value`` of ENERGY outputState);
        * **entropy** of the result (if the ENTROPY outputState is present).

    time_scale :  TimeScale
        specifies whether the mechanism is executed using the `TIME_STEP` or `TRIAL` `TimeScale`.

    name : str : default KWTA-<index>
        the name of the Mechanism.
        Specified in the **name** argument of the constructor for the Projection;
        if not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Mechanism.classPreferences
        the `PreferenceSet` for Mechanism.
        Specified in the **prefs** argument of the constructor for the Mechanism;
        if it is not specified, a default is assigned using `classPreferences` defined in ``__init__.py``
        (see :doc:`PreferenceSet <LINK>` for details).

    Returns
    -------
    instance of KWTA : KWTA

    """

    componentType = KWTA

    paramClassDefaults = RecurrentTransferMechanism.paramClassDefaults.copy()
    paramClassDefaults.update({'function': Logistic})  # perhaps hacky? not sure (7/10/17 CW)

    standard_output_states = RecurrentTransferMechanism.standard_output_states.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 function=Logistic,
                 matrix=None,
                 auto: is_numeric_or_none=None,
                 hetero: is_numeric_or_none=None,
                 initial_value=None,
                 decay: tc.optional(tc.any(int, float)) = 1.0,
                 noise: is_numeric_or_none = 0.0,
                 time_constant: is_numeric_or_none = 1.0,
                 k_value: is_numeric_or_none = 0.5,
                 threshold: is_numeric_or_none = 0,
                 ratio: is_numeric_or_none = 0.5,
                 average_based=False,
                 inhibition_only=True,
                 range=None,
                 input_states: tc.optional(tc.any(list, dict)) = None,
                 output_states: tc.optional(tc.any(list, dict))=None,
                 time_scale=TimeScale.TRIAL,
                 params=None,
                 name=None,
                 prefs: is_pref_set = None,
                 context=componentType + INITIALIZING,
                 ):
        if output_states is None:
            output_states = [RESULT]

        params = self._assign_args_to_param_dicts(input_states=input_states,
                                                  k_value=k_value,
                                                  threshold=threshold,
                                                  ratio=ratio,
                                                  inhibition_only=inhibition_only,
                                                  average_based=average_based)

        # this defaults the matrix to be an identity matrix (self excitation)
        if matrix is None:
            if auto is None:
                auto = 5 # this value is bad: there should be a better way to estimate this?
            if hetero is None:
                hetero = 0

        super().__init__(default_variable=default_variable,
                         size=size,
                         input_states=input_states,
                         function=function,
                         matrix=matrix,
                         auto=auto,
                         hetero=hetero,
                         initial_value=initial_value,
                         decay=decay,
                         noise=noise,
                         time_constant=time_constant,
                         range=range,
                         output_states=output_states,
                         time_scale=time_scale,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)

    # adds indexOfInhibitionInputState to the attributes of KWTA
    def _instantiate_attributes_before_function(self, context=None):

        super()._instantiate_attributes_before_function(context=context)

        # this index is saved so the KWTA mechanism knows which input state represents inhibition
        # (it will be wrong if the user deletes an input state: currently, deleting input states is not supported,
        # so it shouldn't be a problem)
        self.indexOfInhibitionInputState = len(self.input_states) - 1

        # try:
        #     int_k_value = int(self.k_value[0])
        # except TypeError: # if self.k_value is a single value rather than a list or array
        #     int_k_value = int(self.k_value)
        # # ^ this is hacky but necessary for now, since something is
        # # incorrectly turning self.k_value into an array of floats
        # n = self.size[0]
        # if (self.k_value[0] > 0) and (self.k_value[0] < 1):
        #     k = int(round(self.k_value[0] * n))
        # elif (int_k_value < 0):
        #     k = n - int_k_value
        # else:
        #     k = int_k_value
        #
        # self.int_k = k

    def _kwta_scale(self, current_input, context=None):
        try:
            int_k_value = int(self.k_value[0])
        except TypeError: # if self.k_value is a single value rather than a list or array
            int_k_value = int(self.k_value)
        # ^ this is hacky but necessary for now, since something is
        # incorrectly turning self.k_value into an array of floats
        n = self.size[0]
        if (self.k_value[0] > 0) and (self.k_value[0] < 1):
            k = int(round(self.k_value[0] * n))
        elif (int_k_value < 0):
            k = n - int_k_value
        else:
            k = int_k_value
        # k = self.int_k

        diffs = self.threshold - current_input[0]

        sorted_diffs = sorted(diffs)

        if self.average_based:
            top_k_mean = np.mean(sorted_diffs[0:k])
            other_mean = np.mean(sorted_diffs[k:n])
            final_diff = other_mean * self.ratio + top_k_mean * (1 - self.ratio)
        else:
            if k == 0:
                final_diff = sorted_diffs[k]
            elif k == len(sorted_diffs):
                final_diff = sorted_diffs[k - 1]
            elif k > len(sorted_diffs):
                raise KWTAError("k value ({}) is greater than the length of the first input ({}) for KWTA mechanism {}".
                                format(k, current_input[0], self.name))
            else:
                final_diff = sorted_diffs[k] * self.ratio + sorted_diffs[k-1] * (1 - self.ratio)



        if self.inhibition_only and final_diff > 0:
            final_diff = 0

        new_input = np.array(current_input[0] + final_diff)
        if (sum(new_input > self.threshold) > k) and not self.average_based:
            warnings.warn("KWTA scaling was not successful: the result was too high. The original input was {}, "
                          "and the KWTA-scaled result was {}".format(current_input, new_input))
        new_input = list(new_input)
        for i in range(1, len(current_input)):
            new_input.append(current_input[i])
        print('current_input: ', current_input)
        print('new_input: ', new_input)
        return np.atleast_2d(new_input)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate shape and size of matrix and decay.
        """

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        if RATIO in target_set:
            ratio_param = target_set[RATIO]
            if not isinstance(ratio_param, numbers.Real):
                if not (isinstance(ratio_param, (np.ndarray, list)) and len(ratio_param) == 1):
                    raise KWTAError("ratio parameter ({}) for {} must be a single number".format(ratio_param, self))

            if ratio_param > 1 or ratio_param < 0:
                raise KWTAError("ratio parameter ({}) for {} must be between 0 and 1".format(ratio_param, self))

        if K_VALUE in target_set:
            k_param = target_set[K_VALUE]
            if not isinstance(k_param, numbers.Real):
                if not (isinstance(k_param, (np.ndarray, list)) and len(k_param) == 1):
                    raise KWTAError("k-value parameter ({}) for {} must be a single number".format(k_param, self))
            if (isinstance(ratio_param, (np.ndarray, list)) and len(ratio_param) == 1):
                k_num = k_param[0]
            else:
                k_num = k_param
            if not isinstance(k_num, int):
                try:
                    if not k_num.is_integer() and (k_num > 1 or k_num < 0):
                        raise KWTAError("k-value parameter ({}) for {} must be an integer, or between 0 and 1.".
                                        format(k_param, self))
                except AttributeError:
                    raise KWTAError("k-value parameter ({}) for {} was an unexpected type.".format(k_param, self))
            if abs(k_num) > self.size[0]:
                raise KWTAError("k-value parameter ({}) for {} was larger than the total number of elements.".
                                format(k_param, self))

        if THRESHOLD in target_set:
            threshold_param = target_set[THRESHOLD]
            if not isinstance(threshold_param, numbers.Real):
                if not (isinstance(threshold_param, (np.ndarray, list)) and len(threshold_param) == 1):
                    raise KWTAError("k-value parameter ({}) for {} must be a single number".
                                    format(threshold_param, self))

    def execute(self,
                input=None,
                runtime_params=None,
                clock=CentralClock,
                time_scale=TimeScale.TRIAL,
                ignore_execution_id=False,
                context=None):
        if isinstance(input, str) or (isinstance(input, (list, np.ndarray)) and isinstance(input[0], str)):
            raise KWTAError("input ({}) to {} was a string, which is not supported for {}".
                            format(input, self, self.__class__.__name__))
        return super().execute(input=input, runtime_params=runtime_params, clock=clock, time_scale=time_scale,
                               ignore_execution_id=ignore_execution_id, context=context)

    def _execute(self,
                variable=None,
                runtime_params=None,
                clock=CentralClock,
                time_scale = TimeScale.TRIAL,
                context=None):

        variable = self._update_variable(self._kwta_scale(variable, context=context))

        return super()._execute(variable=variable,
                       runtime_params=runtime_params,
                       clock=clock,
                       time_scale=time_scale,
                       context=context)

        # NOTE 7/10/17 CW: this version of KWTA executes scaling _before_ noise or integration is applied. This can be
        # changed, but I think it requires overriding the whole _execute function (as below),
        # rather than calling super._execute()
        #
        # """Execute TransferMechanism function and return transform of input
        #
        # Execute TransferMechanism function on input, and assign to output_values:
        #     - Activation value for all units
        #     - Mean of the activation values across units
        #     - Variance of the activation values across units
        # Return:
        #     value of input transformed by TransferMechanism function in outputState[TransferOuput.RESULT].value
        #     mean of items in RESULT outputState[TransferOuput.MEAN].value
        #     variance of items in RESULT outputState[TransferOuput.VARIANCE].value
        #
        # Arguments:
        #
        # # CONFIRM:
        # variable (float): set to self.value (= self.input_value)
        # - params (dict):  runtime_params passed from Mechanism, used as one-time value for current execution:
        #     + NOISE (float)
        #     + TIME_CONSTANT (float)
        #     + RANGE ([float, float])
        # - time_scale (TimeScale): specifies "temporal granularity" with which mechanism is executed
        # - context (str)
        #
        # Returns the following values in self.value (2D np.array) and in
        #     the value of the corresponding outputState in the self.outputStates dict:
        #     - activation value (float)
        #     - mean activation value (float)
        #     - standard deviation of activation values (float)
        #
        # :param self:
        # :param variable (float)
        # :param params: (dict)
        # :param time_scale: (TimeScale)
        # :param context: (str)
        # :rtype self.outputState.value: (number)
        # """
        #
        # # NOTE: This was heavily based on 6/20/17 devel branch version of _execute from TransferMechanism.py
        # # Thus, any errors in that version should be fixed in this version as well.
        #
        # # FIX: ??CALL check_args()??
        #
        # # FIX: IS THIS CORRECT?  SHOULD THIS BE SET TO INITIAL_VALUE
        # # FIX:     WHICH SHOULD BE DEFAULTED TO 0.0??
        # # Use self.instance_defaults.variable to initialize state of input
        #
        #
        # if INITIALIZING in context:
        #     self.previous_input = self.instance_defaults.variable
        #
        # if self.decay is not None and self.decay != 1.0:
        #     self.previous_input *= self.decay
        #
        # # FIX: NEED TO GET THIS TO WORK WITH CALL TO METHOD:
        # time_scale = self.time_scale
        #
        # #region ASSIGN PARAMETER VALUES
        #
        # time_constant = self.time_constant
        # range = self.range
        # noise = self.noise
        #
        # #endregion
        #
        # #region EXECUTE TransferMechanism FUNCTION ---------------------------------------------------------------------
        #
        # # FIX: NOT UPDATING self.previous_input CORRECTLY
        # # FIX: SHOULD UPDATE PARAMS PASSED TO integrator_function WITH ANY RUNTIME PARAMS THAT ARE RELEVANT TO IT
        #
        # # Update according to time-scale of integration
        # if time_scale is TimeScale.TIME_STEP:
        #
        #     if not self.integrator_function:
        #
        #         self.integrator_function = AdaptiveIntegrator(
        #                                     self.instance_defaults.variable,
        #                                     initializer = self.initial_value,
        #                                     noise = self.noise,
        #                                     rate = self.time_constant
        #                                     )
        #
        #     current_input = self.integrator_function.execute(variable,
        #                                                 # Should we handle runtime params?
        #                                                      # params={INITIALIZER: self.previous_input,
        #                                                      #         INTEGRATION_TYPE: ADAPTIVE,
        #                                                      #         NOISE: self.noise,
        #                                                      #         RATE: self.time_constant}
        #                                                      # context=context
        #                                                      # name=Integrator.componentName + '_for_' + self.name
        #                                                      )
        #
        # elif time_scale is TimeScale.TRIAL:
        #     if self.noise_function:
        #         if isinstance(noise, (list, np.ndarray)):
        #             new_noise = []
        #             for n in noise:
        #                 new_noise.append(n())
        #             noise = new_noise
        #         elif isinstance(variable, (list, np.ndarray)):
        #             new_noise = []
        #             for v in variable[0]:
        #                 new_noise.append(noise())
        #             noise = new_noise
        #         else:
        #             noise = noise()
        #
        #     current_input = self.input_state.value + noise
        # else:
        #     raise MechanismError("time_scale not specified for KWTA")
        #
        # # this is the primary line that's different in KWTA compared to TransferMechanism
        # # this scales the current_input properly
        # current_input = self._kwta_scale(current_input)
        #
        # self.previous_input = current_input
        #
        # # Apply TransferMechanism function
        # output_vector = self.function(variable=current_input, params=runtime_params)
        #
        # # # MODIFIED  OLD:
        # # if list(range):
        # # MODIFIED  NEW:
        # if range is not None:
        # # MODIFIED  END
        #     minCapIndices = np.where(output_vector < range[0])
        #     maxCapIndices = np.where(output_vector > range[1])
        #     output_vector[minCapIndices] = np.min(range)
        #     output_vector[maxCapIndices] = np.max(range)
        #
        # return output_vector
        # #endregion

    # @tc.typecheck
    # def _instantiate_recurrent_projection(self,
    #                                       mech: Mechanism_Base,
    #                                       matrix=FULL_CONNECTIVITY_MATRIX,
    #                                       context=None):
    #     """Instantiate a MappingProjection from mech to itself
    #
    #     """
    #
    #     if isinstance(matrix, str):
    #         size = len(mech.instance_defaults.variable[0])
    #         matrix = get_matrix(matrix, size, size)
    #
    #     return AutoAssociativeProjection(sender=mech,
    #                                      receiver=mech.input_states[mech.indexOfInhibitionInputState],
    #                                      matrix=matrix,
    #                                      name=mech.name + ' recurrent projection')

    # @property
    # def k_value(self):
    #     return super(KWTA, self.__class__).k_value.fget(self)
    #
    # @k_value.setter
    # def k_value(self, setting):
    #     super(KWTA, self.__class__).k_value.fset(self, setting)
    #     try:
    #         int_k_value = int(setting[0])
    #     except TypeError: # if setting is a single value rather than a list or array
    #         int_k_value = int(setting)
    #     n = self.size[0]
    #     if (setting > 0) and (setting < 1):
    #         k = int(round(setting * n))
    #     elif (int_k_value < 0):
    #         k = n - int_k_value
    #     else:
    #         k = int_k_value
    #     self._int_k = k
    #
    # @property
    # def int_k(self):
    #     return self._int_k
    #
    # @int_k.setter
    # def int_k(self, setting):
    #     self._int_k = setting
    #     self.k_value = setting
