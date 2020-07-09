# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ********************************************* KWTAMechanism *************************************************

"""

Contents
--------

  * `KWTAMechanism_Overview`
  * `KWTAMechanism_Creation`
  * `KWTAMechanism_Structure`
  * `KWTAMechanism_Execution`
  * `KWTAMechanism_Class_Reference`


.. _KWTAMechanism_Overview:

Overview
--------

A KWTAMechanism is a subclass of `RecurrentTransferMechanism` that implements a k-winners-take-all (kWTA)
constraint on the number of elements of the Mechanism's `variable <Mechanism_Base.variable>` that are above a
specified threshold.  The implementation is based on the one  described in `O'Reilly and Munakata, 2012
<https://grey.colorado.edu/CompCogNeuro/index.php/CCNBook/Networks/kWTA_Equations>`_.

.. _KWTAMechanism_Creation:

Creating a KWTAMechanism
---------------

A KWTAMechanism can be created directly by calling its constructor. The **k_value**, **threshold**,
and **ratio** arguments can be used to specify the function of the KWTAMechanism, and default to a condition
in which half of the elements in the KWTAMechanism's `variable <Mechanism_Base.variable>`
(**k_value** = 0.5) are above 0 and half are below (**threshold** = 0), achieved using an intermediate degree of
value displacement (**ratio** = 0.5).

.. _KWTAMechanism_Structure:

Structure
---------

The KWTAMechanism calculates an offset to apply to all elements of the Mechanism's `variable
<Mechanism_Base.variable>` array so that it has a specified number of the elements that are at or above a
specified threshold value.  Typically, this constraint can be satisfied in a number of ways;  how it is satisfied is
determined by three parameters and two options of the KWTAMechanism:

.. _KWTAMechanism_k_value:

* `k_value <KWTAMechanism.k_value>` parameter -- determines the number of elements of its `variable
  <Mechanism_Base.variable>` that should be at or above the specified `threshold
  <KWTAMechanism.threshold>`.  A value between 0 and 1 specifies the *proportion* of elements that should be
  at or above the `threshold <KWTAMechanism.threshold>`, while a positive integer specifies the *number* of
  elements that should be at or above the `threshold <KWTAMechanism.threshold>`.  A negative integer specifies
  the number of elements that should be below the `threshold <KWTAMechanism.threshold>`.  Whether or not the
  exact specification is achieved depends on the settings of the `average_based <KWTAMechanism.average_based>`
  and `inhibition_only <KWTAMechanism.inhibition_only>` options (see below).

.. _KWTAMechanism_threshold:

* `threshold <KWTAMechanism.threshold>` parameter -- determines the value at or above which the KTWA seeks to
  assign `k_value <KWTAMechanism.k_value>` elements of its `variable <Mechanism_Base.variable>`.

.. _KWTAMechanism_ratio:

* `ratio <KWTAMechanism.ratio>` parameter -- determines how the offset applied to the elements of the
  KWTAMechanism's `variable <Mechanism_Base.variable>` is selected from the scope of possible values;  the `ratio
  <KWTAMechanism.ratio>` must be a number between 0 and 1.  An offset is picked that is above the low end of
  the scope by a proportion of the scope equal to the `ratio <KWTAMechanism.ratio>` parameter.  How the scope
  is calculated is determined by the `average_based <KWTAMechanism.average_based>` option, as described below.

.. _KWTAMechanism_average_based:

* `average_based <KWTAMechanism.average_based>` option -- determines how the scope of values is calculated from which
  the offset applied to the elements of the KWTAMechanism's `variable <Mechanism_Base.variable>` is selected;  If
  `average_based <KWTAMechanism.average_based>` is `False`, the low end of the scope is the offset that sets the k-th
  highest element exactly at the threshold (that is, the smallest value that insures that `k_value
  <KWTAMechanism.k_value>` elements are at or above the `threshold <KWTAMechanism.threshold>`;  the high end of the
  scope is the offset that sets the k+1-th highest element exactly at the threshold (that is, the largest possible
  value, such that the `k_value <KWTAMechanism.k_value>` elements but no more are above the `threshold
  <KWTAMechanism.threshold>` (i.e., the next one is exactly at it). With this setting, all values of offset within
  the scope generate exactly `k_value <KTWA.k_value>` elements at or above the `threshold <KWTAMechanism.threshold>`.
  If `average_based <KWTAMechanism.average_based>` is `True`, the low end of the scope is the offset that places the
  *average* of the elements with the `k_value <KWTAMechanism.k_value>` highest values at the `threshold
  <KWTAMechanism.threshold>`, and the high end of the scope is the offset that places the average of the
  remaining elements at the `threshold <KWTAMechanism.threshold>`.  In this case, the lowest values of
  offset within the scope may produce fewer than `k_value <KWTAMechanism.k_value>` elements at or above the
  `threshold <KWTAMechanism.threshold>`, while the highest values within the scope may produce more.  An
  offset is picked from the scope as specified by the `ratio <KWTAMechanism.ratio>` parameter (see `above
  <KWTAMechanism_ratio>`).

  .. note::
     If the `average_based <KWTAMechanism.average_based>` option is `False` (the default), the
     KWTAMechanism's `variable <Mechanism_Base.variable>`
     is guaranteed to have exactly `k_value <KTWA.k_value>` elements at or above the `threshold
     <KWTAMechanism.threshold>` (that is, for *any* value of the `ratio <KTWA.ratio>`).  However, if
     `average_based <KWTAMechanism.average_based>` is `True`, this guarantee does not hold;  `variable
     <Mechanism_Base.variable>` may have fewer than `k_value <KWTAMechanism.k_value>` elements at or
     above the `threshold <KWTAMechanism.threshold>` (if the `ratio <KWTAMechanism.ratio>` is low),
     or more than `k_value <KWTAMechanism.k_value>` (if the `ratio <KWTAMechanism.ratio>` is high).

  Although setting the `average_based <KWTAMechanism.average_based>` option to `True` does not guarantee that
  *exactly* `k_value <KWTAMechanism.k_value>` elements will be above the threshold, the additional
  flexibility it affords in the Mechanism's `variable <Mechanism_Base.variable>` attribute  can be useful in
  some settings -- for example, when training hidden layers in a `multilayered network
  <LearningMechanism_Multilayer_Learning>`, which may require different numbers of elements to be above the
  specified `threshold <KWTAMechanism.threshold>` for different input-target pairings.

.. KWTAMechanism_Inhibition_only:

* `inhibition_only <KWTAMechanism.inhibition_only>` option -- determines whether the offset applied to the
  elements of the KWTAMechanism's `variable <Mechanism_Base.variable>` is allowed to be positive
  (i.e., whether the KWTAMechanism can increase the value of any elements of its `variable
  <Mechanism_Base.variable>`).  If set to `False`, the KWTAMechanism will use any offset value
  determined by the `ratio <KWTAMechanism.ratio>` parameter from the scope determined by the `average_based
  <KTWA.average_based>` option (including positive offsets). If `inhibition_only
  <KWTAMechanism.inhibition_only>` is `True`, then any positive offset selected is "clipped" at (i.e
  re-assigned a value of) 0.  This ensures that the values of the elements of the KWTAMechanism's
  `variable <Mechanism_Base.variable>` are never increased.

COMMENT:
  .. note::
     If the `inhibition_only <KWTAMechanism.inhibition_only>` option is set to `True`, the number of elements
     at or above the `threshold <KWTAMechanism.threshold>` may fall below `k_value
     <KWTAMechanism.k_value>`; and, if the input to the KWTAMechanism is sufficiently low,
     the value of all elements may decay to 0 (depending on the value of the `decay <KWTAMechanism.decay>`
     parameter.
COMMENT

In all other respects, a KWTAMechanism has the same attributes and is specified in the same way as a standard
`RecurrentTransferMechanism`.


.. _KWTAMechanism_Execution:

Execution
---------

When a KTWA is executed, it first determines its `variable <Mechanism_Base.variable>` as follows:

* First, like every `RecurrentTransferMechanism`, it combines the input it receives from its recurrent
  `AutoAssociativeProjection` (see `RecurrentTransferMechanism_Structure`) with the input from any other
  `MappingProjections <MappingProjection>` it receives, and assigns this to its `variable <Mechanism_Base.variable>`
  attribute.
..
* Then it modifies its `variable <Mechanism_Base.variable>`, by calculating and assigning an offset to its
  elements, so that as close to `k_value <KWTAMechanism.k_value>` elements as possible are at or above the
  `threshold <KWTAMechanism.threshold>`.  The offset is determined by carrying out the following steps in
  each execution of the KTWA:

  - calculate the scope of offsets that will satisfy the constraint; how this is done is determined by the
    `average_based <KWTAMechanism.average_based>` attribute (see `above
    <KWTAMechanism_average_based>`);

  - select an offset from the scope based on the `ratio <KWTAMechanism.ratio>` option (see `above
    <KWTAMechanism_ratio>`);

  - constrain the offset to be 0 or negative if the `inhibition_only <KWTAMechanism.inhibition_only>` option
    is set (see `above <KWTAMechanism_inhibition_only>`;

  - apply the offset to all elements of the `variable <Mechanism_Base.variable>`.
..
The modified `variable <Mechanism_Base.variable>` is then passed to the KWTAMechanism's `function
<KWTAMechanism.function>` to determine its `value <KWTAMechanism.value>`.


.. _KWTAMechanism_Reference:

Class Reference
---------------

"""

import logging
import numbers
import warnings

from collections.abc import Iterable

import numpy as np
import typecheck as tc

from psyneulink.core.components.functions.transferfunctions import Logistic
from psyneulink.core.globals.keywords import INITIALIZING, KWTA_MECHANISM, K_VALUE, RATIO, RESULT, THRESHOLD
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.utilities import is_numeric_or_none
from psyneulink.library.components.mechanisms.processing.transfer.recurrenttransfermechanism import RecurrentTransferMechanism

__all__ = [
    'KWTAMechanism', 'KWTAError',
]

logger = logging.getLogger(__name__)

class KWTAError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

class KWTAMechanism(RecurrentTransferMechanism):
    """
    KWTAMechanism(            \
        k_value=0.5,          \
        threshold=0,          \
        ratio=0.5,            \
        average_based=False,  \
        inhibition_only=True)

    Subclass of `RecurrentTransferMechanism` that dynamically regulates its input relative to a given threshold. See
    `RecurrentTransferMechanism <RecurrentTransferMechanism_Class_Reference>` for additional arguments and attributes.

    Arguments
    ---------

    k_value : number : default 0.5
        specifies the proportion or number of the elements of `variable <Mechanism_Base.variable>` that should
        be at or above the `threshold <KWTAMechanism.threshold>`. A value between 0 and 1 specifies the
        proportion of elements that should be at or above the `threshold <KWTAMechanism.threshold>`, while a
        positive integer specifies the number of values that should be at or above the `threshold
        <KWTAMechanism.threshold>`. A negative integer specifies the number of elements that should be below
        the `threshold <KWTAMechanism.threshold>`.

    threshold : number : default 0
        specifies the threshold at or above which the KTWA seeks to assign `k_value <KWTAMechanism.k_value>`
        elements of its `variable <Mechanism_Base.variable>`.

    ratio : number : default 0.5
        specifies the offset used to adjust the elements of `variable <Mechanism_Base.variable>` so that there
        are the number specified by `k_value <KWTAMechanism.k_value>` at or above the `threshold
        <KWTAMechanism.threshold>`;  it must be a number from 0 to 1 (see `ratio
        <KWTAMechanism_ratio>` for additional information).

    average_based : boolean : default False
        specifies whether the average-based scaling is used to determine the scope of offsets (see `average_based
        <KWTAMechanism_average_based>` for additional information).

    inhibition_only : boolean : default True
        specifies whether positive offsets can be applied to the `variable <Mechanism_Base.variable>` in an
        effort to achieve `k_value <KWTAMechanism.k_value>` elements at or above the `threshold
        <KWTAMechanism.threshold>`.  If set to `False`, any offset is allowed, including positive offsets;
        if set to `True`, a positive offset will be re-assigned the value of 0 (see `inhibition_only
        <KWTAMechanism_inhibition_only>` for additional information).


    Attributes
    ----------

    k_value : number
        determines the number or proportion of elements of `variable <Mechanism_Base.variable>` that should be
        above the `threshold <KWTAMechanism.threshold>` of the KWTAMechanism (see `k_value
        <KWTAMechanism_k_value>` for additional information).

    threshold : number
        determines the threshold at or above which the KTWA seeks to assign `k_value <KWTAMechanism.k_value>`
        elements of its `variable <Mechanism_Base.variable>`.

    ratio : number
        determines the offset used to adjust the elements of `variable <Mechanism_Base.variable>` so that there
        are `k_value <KWTAMechanism.k_value>` elements at or above the `threshold
        <KWTAMechanism.threshold>` (see `ratio <KWTAMechanism_ratio>` for additional information).

    average_based : boolean : default False
        determines the way in which the scope of offsets is determined, from which the one is selected that is applied
        to the elements of the `variable <Mechanism_Base.variable>` (see `average_based
        <KWTAMechanism_average_based>` for additional information).

    inhibition_only : boolean : default True
        determines whether a positive offset is allowed;  if it is `True`, then the value of the offset is
        "clipped" at (that is, any positive value is replaced by) 0.  Otherwise, any offset is allowed (see
        `inhibition_only <KWTAMechanism_inhibition_only>` for additional information).

    Returns
    -------
    instance of KWTAMechanism : KWTAMechanism

    """

    componentType = KWTA_MECHANISM

    class Parameters(RecurrentTransferMechanism.Parameters):
        """
            Attributes
            ----------

                average_based
                    see `average_based <KWTAMechanism.average_based>`

                    :default value: False
                    :type: ``bool``

                function
                    see `function <KWTAMechanism.function>`

                    :default value: `Logistic`
                    :type: `Function`

                inhibition_only
                    see `inhibition_only <KWTAMechanism.inhibition_only>`

                    :default value: True
                    :type: ``bool``

                k_value
                    see `k_value <KWTAMechanism.k_value>`

                    :default value: 0.5
                    :type: ``float``

                ratio
                    see `ratio <KWTAMechanism.ratio>`

                    :default value: 0.5
                    :type: ``float``

                threshold
                    see `threshold <KWTAMechanism.threshold>`

                    :default value: 0.0
                    :type: ``float``
        """
        function = Parameter(Logistic, stateful=False, loggable=False)
        k_value = Parameter(0.5, modulable=True)
        threshold = Parameter(0.0, modulable=True)
        ratio = Parameter(0.5, modulable=True)

        output_ports = Parameter(
            [RESULT],
            stateful=False,
            loggable=False,
            read_only=True,
            structural=True,
        )

        average_based = False
        inhibition_only = True

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 function=None,
                 matrix=None,
                 auto: is_numeric_or_none=None,
                 hetero: is_numeric_or_none=None,
                 integrator_function=None,
                 initial_value=None,
                 noise: tc.optional(is_numeric_or_none) = None,
                 integration_rate: tc.optional(is_numeric_or_none) = None,
                 integrator_mode=None,
                 k_value: tc.optional(is_numeric_or_none) = None,
                 threshold: tc.optional(is_numeric_or_none) = None,
                 ratio: tc.optional(is_numeric_or_none) = None,
                 average_based=None,
                 inhibition_only=None,
                 clip=None,
                 input_ports:tc.optional(tc.optional(tc.any(list, dict))) = None,
                 output_ports:tc.optional(tc.any(str, Iterable))=None,
                 params=None,
                 name=None,
                 prefs: tc.optional(is_pref_set) = None,
                 **kwargs
                 ):
        # this defaults the matrix to be an identity matrix (self excitation)
        if matrix is None:
            if auto is None:
                auto = 5 # this value is bad: there should be a better way to estimate this?
            if hetero is None:
                hetero = 0

        super().__init__(
            default_variable=default_variable,
            size=size,
            input_ports=input_ports,
            function=function,
            matrix=matrix,
            auto=auto,
            hetero=hetero,
            integrator_function=integrator_function,
            integrator_mode=integrator_mode,
            k_value=k_value,
            threshold=threshold,
            ratio=ratio,
            inhibition_only=inhibition_only,
            average_based=average_based,
            initial_value=initial_value,
            noise=noise,
            integration_rate=integration_rate,
            clip=clip,
            output_ports=output_ports,
            params=params,
            name=name,
            prefs=prefs,
            **kwargs
        )

    def _parse_function_variable(self, variable, context=None):
        if variable.dtype.char == "U":
            raise KWTAError(
                "input ({0}) to {1} was a string, which is not supported for {2}".format(
                    variable, self, self.__class__.__name__
                )
            )

        return self._kwta_scale(variable, context=context)

    # adds indexOfInhibitionInputPort to the attributes of KWTAMechanism
    def _instantiate_attributes_before_function(self, function=None, context=None):

        super()._instantiate_attributes_before_function(function=function, context=context)

        # this index is saved so the KWTAMechanism mechanism knows which InputPort represents inhibition
        # (it will be wrong if the user deletes an InputPort: currently, deleting input ports is not supported,
        # so it shouldn't be a problem)
        self.indexOfInhibitionInputPort = len(self.input_ports) - 1

    def _kwta_scale(self, current_input, context=None):
        k_value = self._get_current_mechanism_param("k_value", context)
        threshold = self._get_current_mechanism_param("threshold", context)
        average_based = self._get_current_mechanism_param("average_based", context)
        ratio = self._get_current_mechanism_param("ratio", context)
        inhibition_only = self._get_current_mechanism_param("inhibition_only", context)

        try:
            int_k_value = int(k_value[0])
        except TypeError: # if k_value is a single value rather than a list or array
            int_k_value = int(k_value)
        # ^ this is hacky but necessary for now, since something is
        # incorrectly turning k_value into an array of floats
        n = self.size[0]
        if (k_value[0] > 0) and (k_value[0] < 1):
            k = int(round(k_value[0] * n))
        elif (int_k_value < 0):
            k = n - int_k_value
        else:
            k = int_k_value
        # k = self.int_k

        diffs = threshold - current_input[0]

        sorted_diffs = sorted(diffs)

        if average_based:
            top_k_mean = np.mean(sorted_diffs[0:k])
            other_mean = np.mean(sorted_diffs[k:n])
            final_diff = other_mean * ratio + top_k_mean * (1 - ratio)
        else:
            if k == 0:
                final_diff = sorted_diffs[k]
            elif k == len(sorted_diffs):
                final_diff = sorted_diffs[k - 1]
            elif k > len(sorted_diffs):
                raise KWTAError("k value ({}) is greater than the length of the first input ({}) for KWTAMechanism mechanism {}".
                                format(k, current_input[0], self.name))
            else:
                final_diff = sorted_diffs[k] * ratio + sorted_diffs[k - 1] * (1 - ratio)

        if inhibition_only and final_diff > 0:
            final_diff = 0

        new_input = np.array(current_input[0] + final_diff)
        if (sum(new_input > threshold) > k) and not average_based:
            warnings.warn("KWTAMechanism scaling was not successful: the result was too high. The original input was {}, "
                          "and the KWTAMechanism-scaled result was {}".format(current_input, new_input))
        new_input = list(new_input)
        for i in range(1, len(current_input)):
            new_input.append(current_input[i])
        return np.atleast_2d(new_input)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate shape and size of matrix.
        """

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        if RATIO in target_set and target_set[RATIO] is not None:
            ratio_param = target_set[RATIO]
            if not isinstance(ratio_param, numbers.Real):
                if not (isinstance(ratio_param, (np.ndarray, list)) and len(ratio_param) == 1):
                    raise KWTAError("ratio parameter ({}) for {} must be a single number".format(ratio_param, self))

            if ratio_param > 1 or ratio_param < 0:
                raise KWTAError("ratio parameter ({}) for {} must be between 0 and 1".format(ratio_param, self))

        if K_VALUE in target_set and target_set[K_VALUE] is not None:
            k_param = target_set[K_VALUE]
            if not isinstance(k_param, numbers.Real):
                if not (isinstance(k_param, (np.ndarray, list)) and len(k_param) == 1):
                    raise KWTAError("k-value parameter ({}) for {} must be a single number".format(k_param, self))
            if (isinstance(k_param, (np.ndarray, list)) and len(k_param) == 1):
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

        if THRESHOLD in target_set and target_set[THRESHOLD] is not None:
            threshold_param = target_set[THRESHOLD]
            if not isinstance(threshold_param, numbers.Real):
                if not (isinstance(threshold_param, (np.ndarray, list)) and len(threshold_param) == 1):
                    raise KWTAError("k-value parameter ({}) for {} must be a single number".
                                    format(threshold_param, self))

        # NOTE 7/10/17 CW: this version of KWTAMechanism executes scaling _before_ noise or integration is applied. This can be
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
        #     value of input transformed by TransferMechanism function in OutputPort[TransferOuput.RESULT].value
        #     mean of items in RESULT OutputPort[TransferOuput.MEAN].value
        #     variance of items in RESULT OutputPort[TransferOuput.VARIANCE].value
        #
        # Arguments:
        #
        # # CONFIRM:
        # variable (float): set to self.value (= self.input_value)
        # - params (dict):  runtime_params passed from Mechanism, used as one-time value for current execution:
        #     + NOISE (float)
        #     + INTEGRATION_RATE (float)
        #     + RANGE ([float, float])
        # - time_scale (TimeScale): specifies "temporal granularity" with which mechanism is executed
        # - context (str)
        #
        # Returns the following values in self.value (2D np.array) and in
        #     the value of the corresponding OutputPort in the self.output_ports list:
        #     - activation value (float)
        #     - mean activation value (float)
        #     - standard deviation of activation values (float)
        #
        # :param self:
        # :param variable (float)
        # :param params: (dict)
        # :param time_scale: (TimeScale)
        # :param context: (str)
        # :rtype self.output_port.value: (number)
        # """
        #
        # # NOTE: This was heavily based on 6/20/17 devel branch version of _execute from TransferMechanism.py
        # # Thus, any errors in that version should be fixed in this version as well.
        #
        # # FIX: ??CALL check_args()??
        #
        # # FIX: IS THIS CORRECT?  SHOULD THIS BE SET TO INITIAL_VALUE
        # # FIX:     WHICH SHOULD BE DEFAULTED TO 0.0??
        # # Use self.defaults.variable to initialize state of input
        #
        #
        # if INITIALIZING in context:
        #     self.previous_input = self.defaults.variable
        #
        # if self.decay is not None and self.decay != 1.0:
        #     self.previous_input *= self.decay
        #
        # # FIX: NEED TO GET THIS TO WORK WITH CALL TO METHOD:
        # time_scale = self.time_scale
        #
        # #region ASSIGN PARAMETER VALUES
        #
        # integration_rate = self.integration_rate
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
        #                                     self.defaults.variable,
        #                                     initializer = self.initial_value,
        #                                     noise = self.noise,
        #                                     rate = self.integration_rate
        #                                     )
        #
        #     current_input = self.integrator_function.execute(variable,
        #                                                 # Should we handle runtime params?
        #                                                      # params={INITIALIZER: self.previous_input,
        #                                                      #         INTEGRATION_TYPE: ADAPTIVE,
        #                                                      #         NOISE: self.noise,
        #                                                      #         RATE: self.integration_rate}
        #                                                      # context=context
        #                                                      # name=IntegratorFunction.componentName + '_for_' + self.name
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
        #     current_input = self.input_port.value + noise
        # else:
        #     raise MechanismError("time_scale not specified for KWTAMechanism")
        #
        # # this is the primary line that's different in KWTAMechanism compared to TransferMechanism
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
    #         size = len(mech.defaults.variable[0])
    #         matrix = get_matrix(matrix, size, size)
    #
    #     return AutoAssociativeProjection(sender=mech,
    #                                      receiver=mech.input_ports[mech.indexOfInhibitionInputPort],
    #                                      matrix=matrix,
    #                                      name=mech.name + ' recurrent projection')

    # @property
    # def k_value(self):
    #     return super(KWTAMechanism, self.__class__).k_value.fget(self)
    #
    # @k_value.setter
    # def k_value(self, setting):
    #     super(KWTAMechanism, self.__class__).k_value.fset(self, setting)
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
