# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# *********************************************** KohoneMechanism ******************************************************

"""

Overview
--------

A KohonenMechanism is a subclass of `RecurrentTransferMechanism` that implements a `Kohonen network
<http://www.scholarpedia.org/article/Kohonen_network>`_ (`brief explanation
<https://www.cs.bham.ac.uk/~jlw/sem2a2/Web/Kohonen.htm>`_; `nice demo <https://www.youtube.com/watch?v=QvI6L-KqsT4>`_).

.. _Kohonen_Creation:

Creating a KohonenMechanism
---------------------------

A KohonenMechanism can be created directly by calling its constructor.

.. _Kohonen_Structure:

Structure
---------

XXX

.. _Kohonen_Execution:

Execution
---------

XXX

.. _Kohonen_Reference:

Class Reference
---------------

"""

import logging
import numbers
import warnings
from collections import Iterable

import numpy as np
import typecheck as tc

from psyneulink.components.functions.function import Linear, Gaussian, Kohonen, OneHot
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.globals.keywords import FULL_CONNECTIVITY_MATRIX, INITIALIZING, MAX_VAL, RESULT
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.globals.utilities import is_numeric_or_none
from psyneulink.library.mechanisms.processing.transfer.recurrenttransfermechanism import RecurrentTransferMechanism

__all__ = [
    'KohonenMechanism', 'KohonenError',
]

logger = logging.getLogger(__name__)

class KohonenError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

class KohonenMechanism(TransferMechanism):
    """
    KohonenMechanism(                        \
    default_variable=None,                   \
    size=None,                               \
    function=Linear,                         \
    selection_function=OneHot(mode=MAX_VAL), \
    distance_function=Gaussian               \
    learning_function=Kohonen,               \
    matrix=None,                             \
    initial_value=None,                      \
    noise=0.0,                               \
    integration_rate=1.0,                    \
    clip=None,                               \
    params=None,                             \
    name=None,                               \
    prefs=None)

    Subclass of `TransferMechanism` that learns a `self-organized <https://en.wikipedia.org/wiki/Self-organizing_map>`_
    map of its input.

    Arguments
    ---------

    default_variable : number, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the input to the mechanism to use if none is provided in a call to its
        `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` method;
        also serves as a template to specify the length of `variable <KWTA.variable>` for
        `function <KWTA.function>`, and the `primary OutputState <OutputState_Primary>`
        of the mechanism.

    size : int, list or np.ndarray of ints
        specifies variable as array(s) of zeros if **variable** is not passed as an argument;
        if **variable** is specified, it takes precedence over the specification of **size**.
        As an example, the following mechanisms are equivalent::
            T1 = TransferMechanism(size = [3, 2])
            T2 = TransferMechanism(default_variable = [[0, 0, 0], [0, 0]])

    function : TransferFunction : default Linear
        specifies the function used to transform the input;  can be `Linear`, `Logistic`, `Exponential`,
        or a custom function.

    selection_function : SelectionFunction, function or method : default OneHot(mode=MAX_VAL)
        specifes the function used to select the element of the input used to train the `matrix
        <MappingProjection.matrix>` of afferent `MappingProjection` to the Mechanism.

    distance_function : Function, function or method : default Gaussian
        specifes the function used to determine the distance of each element from the one identified by
        `selection_function <KohonenMechanism.selection_function>`.

    learning_function : LearningFunction, function or method
        specifies function used by `learning_mechanism <KohonenMechanism.learning_mechanism>` to update `matrix
        <MappingProjection.matrix>` of `learned_projection <KohonenMechanism.learned_projection>.

    matrix : list, np.ndarray, np.matrix, matrix keyword, or AutoAssociativeProjection : default FULL_CONNECTIVITY_MATRIX
        specifies the matrix to use for creating a `recurrent AutoAssociativeProjection <Recurrent_Transfer_Structure>`,
        or a AutoAssociativeProjection to use. If **auto** or **hetero** arguments are specified, the **matrix** argument
        will be ignored in favor of those arguments.

    initial_value :  value, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the starting value for time-averaged input (only relevant if
        `integration_rate <KWTA.integration_rate>` is not 1.0).
        COMMENT:
            Transfer_DEFAULT_BIAS SHOULD RESOLVE TO A VALUE
        COMMENT

    noise : float or function : default 0.0
        a value added to the result of the `function <KWTA.function>` or to the result of `integrator_function
        <KWTA.integrator_function>`, depending on whether `integrator_mode <KWTA.integrator_mode>` is True or False. See
        `noise <KWTA.noise>` for more details.

    integration_rate : float : default 0.5
        the smoothing factor for exponential time averaging of input when `integrator_mode <KWTA.integrator_mode>` is set
        to True ::

         result = (integration_rate * current input) +
         (1-integration_rate * result on previous time_step)

    clip : list [float, float] : default None (Optional)
        specifies the allowable range for the result of `function <KWTA.function>` the item in index 0 specifies the
        minimum allowable value of the result, and the item in index 1 specifies the maximum allowable value; any
        element of the result that exceeds the specified minimum or maximum value is set to the value of
        `clip <KWTA.clip>` that it exceeds.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the mechanism, its function, and/or a custom function and its parameters.  Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default see `name <KWTA Mechanism.name>`
        specifies the name of the KWTA Mechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the KWTA Mechanism; see `prefs <KWTA Mechanism.prefs>` for details.

    context : str : default componentType+INITIALIZING
        string used for contextualization of instantiation, hierarchical calls, executions, etc.

    Attributes
    ----------

    variable : value
        the input to Mechanism's `function <KohonenMechanism.variable>`.

    function : Function
        the Function used to transform the input.

    selection_function : SelectionFunction, function or method : default OneHot(mode=MAX_VAL)
        determines the function used to select the element of the input used to train the `matrix
        <MappingProjection.matrix>` of the `learned_projection <KohonenMechanism.learned_projection>`.

    distance_function : Function, function or method : default Gaussian
        determines the function used to determine the distance of each element from the one identified by
        `selection_function <KohonenMechanism.selection_function>` and the corresponding size of the changes in
        the weight of the MappingProjection

    matrix : 2d np.array
        `matrix <AutoAssociativeProjection.matrix>` parameter of the `learned_projection
        <KohonenMechanism.learned_projection>`.

    learning_enabled : bool
        indicates whether `learning is enabled <Kohonen_Learning>`;  see `learning_enabled
        <Kohonen.learning_enabled>` for additional details.

    learned_projection : MappingProjection
        `MappingProjection` that projects to the Mechanism and is trained by its `learning_mechanism
        <KohonenMechanism.learning_mechanism>`.

    learning_function : LearningFunction, function or method
        function used by `learning_mechanism <KohonenMechanism.learning_mechanism>` to update `matrix
        <MappingProjection.matrix>` of `learned_projection <KohonenMechanism.learned_projection>.

    learning_mechanism : LearningMechanism
        created automatically if `learning is specified <KohonenMechanism_Learning>`, and used to train the
        `learned_projection <KohonenMechanism.learned_projection>`.

    initial_value :  value, list or np.ndarray : Transfer_DEFAULT_BIAS
        determines the starting value for time-averaged input
        (only relevant if `integration_rate <KWTA.integration_rate>` parameter is not 1.0).
        COMMENT:
            Transfer_DEFAULT_BIAS SHOULD RESOLVE TO A VALUE
        COMMENT

    noise : float or function : default 0.0
        When `integrator_mode <KWTA.integrator_mode>` is set to True, noise is passed into the `integrator_function
        <KWTA.integrator_function>`. Otherwise, noise is added to the output of the `function <KWTA.function>`.

        If noise is a list or array, it must be the same length as `variable <KWTA.default_variable>`.

        If noise is specified as a single float or function, while `variable <KWTA.variable>` is a list or array,
        noise will be applied to each variable element. In the case of a noise function, this means that the function
        will be executed separately for each variable element.

        .. note::
            In order to generate random noise, we recommend selecting a probability distribution function
            (see `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value from
            its distribution on each execution. If noise is specified as a float or as a function with a fixed output, then
            the noise will simply be an offset that remains the same across all executions.

    integration_rate : float : default 0.5
        the smoothing factor for exponential time averaging of input when `integrator_mode <KWTA.integrator_mode>` is set
        to True::

          result = (integration_rate * current input) + (1-integration_rate * result on previous time_step)

    clip : list [float, float] : default None (Optional)
        specifies the allowable range for the result of `function <KWTA.function>`

        the item in index 0 specifies the minimum allowable value of the result, and the item in index 1 specifies the
        maximum allowable value; any element of the result that exceeds the specified minimum or maximum value is set to
         the value of `clip <KWTA.clip>` that it exceeds.

    integrator_function:
        When *integrator_mode* is set to True, the KWTA executes its `integrator_function <KWTA.integrator_function>`,
        which is the `AdaptiveIntegrator`. See `AdaptiveIntegrator <AdaptiveIntegrator>` for more details on what it computes.
        Keep in mind that the `integration_rate <KWTA.integration_rate>` parameter of the `KWTA` corresponds to the
        `rate <KWTAIntegrator.rate>` of the `KWTAIntegrator`.

    integrator_mode:
        **When integrator_mode is set to True:**

        the variable of the mechanism is first passed into the following equation:

        .. math::
            value = previous\\_value(1-smoothing\\_factor) + variable \\cdot smoothing\\_factor + noise

        The result of the integrator function above is then passed into the `mechanism's function <KWTA.function>`. Note that
        on the first execution, *initial_value* sets previous_value.

        **When integrator_mode is set to False:**

        The variable of the mechanism is passed into the `function of the mechanism <KWTA.function>`. The mechanism's
        `integrator_function <KWTA.integrator_function>` is skipped entirely, and all related arguments (*noise*, *leak*,
        *initial_value*, and *time_step_size*) are ignored.

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

    output_states : Dict[str, OutputState]
        an OrderedDict with the following `OutputStates <OutputState>`:

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

        * **result** of the ``function`` calculation (value of TRANSFER_RESULT OutputState);
        * **mean** of the result (``value`` of TRANSFER_MEAN OutputState)
        * **variance** of the result (``value`` of TRANSFER_VARIANCE OutputState);
        * **energy** of the result (``value`` of ENERGY OutputState);
        * **entropy** of the result (if the ENTROPY OutputState is present).

    name : str
        the name of the KWTA Mechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the KWTA Mechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    Returns
    -------
    instance of KWTA : KWTA

    """

    componentType = KOHONEN_MECHANISM

    class ClassDefaults(TransferMechanism.ClassDefaults):
        function = Linear

    paramClassDefaults = TransferMechanism.paramClassDefaults.copy()
    paramClassDefaults.update({'function': Linear})  # perhaps hacky? not sure (7/10/17 CW)

    standard_output_states = TransferMechanism.standard_output_states.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 function=Linear,
                 selection_function=OneHot(mode=MAX_VAL),
                 distance_function=Gaussian,
                 enable_learning=True,
                 learning_function=Kohonen,
                 matrix=FULL_CONNECTIVITY_MATRIX,
                 initial_value=None,
                 noise: is_numeric_or_none = 0.0,
                 integration_rate: is_numeric_or_none = 0.5,
                 integrator_mode=False,
                 clip=None,
                 input_states:tc.optional(tc.any(list, dict)) = None,
                 output_states:tc.optional(tc.any(str, Iterable))=RESULT,
                 params=None,
                 name=None,
                 prefs: is_pref_set = None,
                 context=componentType + INITIALIZING,
                 ):
        # Default output_states is specified in constructor as a string rather than a list
        # to avoid "gotcha" associated with mutable default arguments
        # (see: bit.ly/2uID3s3 and http://docs.python-guide.org/en/latest/writing/gotchas/)
        if output_states is None:
            output_states = [RESULT]

        params = self._assign_args_to_param_dicts(input_states=input_states,
                                                  integrator_mode=integrator_mode,
                                                  selection_function=selection_function,
                                                  distance_function=distance_function,
                                                  learning_function=learning_function,
                                                  enable_learning=enable_learning)

        super().__init__(default_variable=default_variable,
                         size=size,
                         input_states=input_states,
                         function=function,
                         matrix=matrix,
                         integrator_mode=integrator_mode,
                         initial_value=initial_value,
                         noise=noise,
                         integration_rate=integration_rate,
                         clip=clip,
                         output_states=output_states,
                         params=params,
                         name=name,
                         prefs=prefs)

    def _parse_function_variable(self, variable, context=None):
        if variable.dtype.char == "U":
            raise KohonenError(
                "input ({0}) to {1} was a string, which is not supported for {2}".format(
                    variable, self, self.__class__.__name__
                )
            )

        return self._kwta_scale(variable)

    # adds indexOfInhibitionInputState to the attributes of KWTA
    def _instantiate_attributes_before_function(self, function=None, context=None):

        super()._instantiate_attributes_before_function(function=function, context=context)

        # this index is saved so the KWTA mechanism knows which input state represents inhibition
        # (it will be wrong if the user deletes an input state: currently, deleting input states is not supported,
        # so it shouldn't be a problem)
        self.indexOfInhibitionInputState = len(self.input_states) - 1

    def _kwta_scale(self, current_input, context=None):
        k_value = self.get_current_mechanism_param("k_value")
        threshold = self.get_current_mechanism_param("threshold")
        average_based = self.get_current_mechanism_param("average_based")
        ratio = self.get_current_mechanism_param("ratio")
        inhibition_only = self.get_current_mechanism_param("inhibition_only")

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
                raise KWTAError("k value ({}) is greater than the length of the first input ({}) for KWTA mechanism {}".
                                format(k, current_input[0], self.name))
            else:
                final_diff = sorted_diffs[k] * ratio + sorted_diffs[k-1] * (1 - ratio)



        if inhibition_only and final_diff > 0:
            final_diff = 0

        new_input = np.array(current_input[0] + final_diff)
        if (sum(new_input > threshold) > k) and not average_based:
            warnings.warn("KWTA scaling was not successful: the result was too high. The original input was {}, "
                          "and the KWTA-scaled result was {}".format(current_input, new_input))
        new_input = list(new_input)
        for i in range(1, len(current_input)):
            new_input.append(current_input[i])
        return np.atleast_2d(new_input)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate shape and size of matrix.
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
