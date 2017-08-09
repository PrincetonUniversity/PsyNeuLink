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

A KWTA is a subclass of `RecurrentTransferMechanism` that implements a single-layered recurrent network with
k-winners-take-all (kWTA) behavior.

.. _KWTA_Creation:

Creating a KWTA
---------------

Similar to a `RecurrentTransferMechanism`, a KWTA mechanism can be created directly by calling its constructor, or by
using the `mechanism() <Mechanism.mechanism>` function and specifying KWTA as its **mech_spec** argument. Just as with a
`RecurrentTransferMechanism`, the **auto**, **hetero**, and/or **matrix** arguments are used to specify the matrix
used by the recurrent projection. The **k_value**, **threshold**, and **ratio** arguments are KWTA's characteristic
attributes. The **k_value** argument specifies the `k_value` and `int_k` attributes of the KWTA (which are currently
not supposed to be changed), and the number (or proportion) of 



"""

import builtins
import numbers
import warnings

import numpy as np
import typecheck as tc

from PsyNeuLink.Components.Functions.Function import Logistic, get_matrix
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism_Base
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.RecurrentTransferMechanism import RecurrentTransferMechanism
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.Projections.PathwayProjections.AutoAssociativeProjection import AutoAssociativeProjection, get_auto_matrix, get_hetero_matrix
from PsyNeuLink.Globals.Keywords import AUTO, HETERO, FULL_CONNECTIVITY_MATRIX, INITIALIZING, K_VALUE, KWTA, MATRIX, RATIO, RESULT, THRESHOLD
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set, kpVerbosePref
from PsyNeuLink.Globals.Utilities import is_numeric_or_none
from PsyNeuLink.Scheduling.TimeScale import CentralClock, TimeScale
import logging

logger = logging.getLogger(__name__)

class KWTAError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

class KWTA(RecurrentTransferMechanism):
    """Subclass of `RecurrentTransferMechanism` that dynamically regulates the "activity" of its elements.
    """

    componentType = KWTA

    paramClassDefaults = RecurrentTransferMechanism.paramClassDefaults.copy()
    paramClassDefaults.update({'function': Logistic})  # perhaps hacky? not sure (7/10/17 CW)

    standard_output_states = RecurrentTransferMechanism.standard_output_states.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_states: tc.optional(tc.any(list, dict)) = None,
                 function=Logistic,
                 initial_value=None,
                 matrix=None,  # None defaults to a hollow uniform inhibition matrix
                 auto: is_numeric_or_none=None,
                 hetero: is_numeric_or_none=None,
                 decay: tc.optional(tc.any(int, float)) = 1.0,
                 noise: is_numeric_or_none = 0.0,
                 time_constant: is_numeric_or_none = 1.0,
                 k_value: is_numeric_or_none = 0.5,
                 threshold: is_numeric_or_none = 0,
                 ratio: is_numeric_or_none = 0.5,
                 inhibition_only=True,
                 range=None,
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
                                                  inhibition_only=inhibition_only)

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

        self.int_k = k

    def _kwta_scale(self, current_input, context=None):
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
        k = self.int_k

        diffs = self.threshold - current_input[0]

        sorted_diffs = sorted(diffs)

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
        if (sum(new_input > self.threshold) > k):
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

        self.variable = self._kwta_scale(self.variable, context=context)

        return super()._execute(variable=self.variable,
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
        # # Use self.variable to initialize state of input
        #
        #
        # if INITIALIZING in context:
        #     self.previous_input = self.variable
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
        #                                     self.variable,
        #                                     initializer = self.initial_value,
        #                                     noise = self.noise,
        #                                     rate = self.time_constant
        #                                     )
        #
        #     current_input = self.integrator_function.execute(self.variable,
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
    #         size = len(mech.variable[0])
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
