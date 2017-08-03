# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# **************************************** KWTA *************************************************

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
                 gain=1,
                 bias=0,
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
                 range=None,
                 output_states: tc.optional(tc.any(list, dict)) = [RESULT],
                 time_scale=TimeScale.TRIAL,
                 params=None,
                 name=None,
                 prefs: is_pref_set = None,
                 context=componentType + INITIALIZING,
                 ):

        kwta_log_function = Logistic(gain=gain, bias=bias) # the user doesn't get to choose the function of the KWTA

        if default_variable is None and size is None:
            default_variable = self.variableClassDefault

        # IMPLEMENTATION NOTE: somewhat redundant with the call to _handle_size() in Component: but this allows us
        # to append zeros to default_variable immediately below, rather than wait to do it in
        # _instantiate_attributes_before_function or elsewhere: the longer we wait, the more bugs are likely to exist
        default_variable = self._handle_size(size, default_variable)

        # append an array of zeros to default_variable, to represent the zero outside input to the inhibition vector.
        d = list(default_variable)
        d.append(np.zeros(len(default_variable[0])))
        default_variable = np.array(d)

        # region set up the additional input_state that will represent inhibition
        if isinstance(input_states, dict):
            input_states = [input_states]

        if input_states is None or len(input_states) == 0:
            input_states = ["Default_input_state"]

        if isinstance(input_states, list) and len(input_states) > 1:
            # probably useless since self never has an attribute `prefs` at this point?
            if hasattr(self, 'prefs') and hasattr(self.prefs, kpVerbosePref) and self.prefs.verbosePref:
                warnings.warn("kWTA adjusts only the FIRST input state. If you have multiple input states, "
                              "only the primary one will be adjusted to have k values above the threshold.")

        input_states.append("Inhibition_input_state")
        # endregion

        params = self._assign_args_to_param_dicts(input_states=input_states,
                                                  k_value=k_value,
                                                  threshold=threshold,
                                                  ratio=ratio)
        # this defaults the matrix to be a hollow uniform inhibition matrix
        if matrix is None:
            if auto is None:
                auto = 0
            if hetero is None:
                hetero = -1

        super().__init__(default_variable=default_variable,
                         size=size,
                         input_states=input_states,
                         function=kwta_log_function,
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

    # def _instantiate_input_states(self, context=None):
    #     # this code is copied heavily from InputState.py, devel branch 6/26/17
    #     # the reason for this is to override the param-check that causes InputState to throw an exception
    #     # because the number of input_states is different from the length of the mechanism's "variable"
    #     owner = self
    #
    #     # extendedSelfVariable = list(self.variable)
    #     # extendedSelfVariable.append(np.ones(self.size[0]))
    #     # extendedSelfVariable = np.array(extendedSelfVariable)
    #
    #     from PsyNeuLink.Components.States.State import _instantiate_state_list
    #     state_list = _instantiate_state_list(owner=owner,
    #                                          state_list=owner.input_states,
    #                                          state_type=InputState,
    #                                          state_param_identifier=INPUT_STATE,
    #                                          constraint_value=self.variable,
    #                                          constraint_value_name="kwta-extended function variable",
    #                                          context=context)
    #
    #     # FIX: 5/23/17:  SHOULD APPEND THIS TO LIST OF EXISTING INPUT_STATES RATHER THAN JUST ASSIGN;
    #     #                THAT WAY CAN USE INCREMENTALLY IN COMPOSITION
    #     # if context and 'COMMAND_LINE' in context:
    #     #     if owner.input_states:
    #     #         owner.input_states.extend(state_list)
    #     #     else:
    #     #         owner.input_states = state_list
    #     # else:
    #     #     if owner._input_states:
    #     #         owner._input_states.extend(state_list)
    #     #     else:
    #     #         owner._input_states = state_list
    #
    #     # FIX: This is a hack to avoid recursive calls to assign_params, in which output_states never gets assigned
    #     # FIX: Hack to prevent recursion in calls to setter and assign_params
    #     if context and 'COMMAND_LINE' in context:
    #         owner.input_states = state_list
    #     else:
    #         owner._input_states = state_list
    #
    #     # Check that number of input_states and their variables are consistent with owner.variable,
    #     #    and adjust the latter if not
    #     for i in builtins.range(len(owner.input_states)):
    #         input_state = owner.input_states[i]
    #         try:
    #             variable_item_is_OK = iscompatible(self.variable[i], input_state.value)
    #             if not variable_item_is_OK:
    #                 break
    #         except IndexError:
    #             variable_item_is_OK = False
    #             break
    #
    #     if not variable_item_is_OK:
    #         old_variable = owner.variable
    #         new_variable = []
    #         for state_name, state in owner.input_states:
    #             new_variable.append(state.value)
    #         owner.variable = np.array(new_variable)
    #         if owner.verbosePref:
    #             warnings.warn("Variable for {} ({}) has been adjusted "
    #                           "to match number and format of its input_states: ({})".
    #                           format(old_variable, append_type_to_name(owner), owner.variable))

    # adds indexOfInhibitionInputState to the attributes of KWTA
    def _instantiate_attributes_before_function(self, context=None):

        super()._instantiate_attributes_before_function(context=context)

        # this index is saved so the KWTA mechanism knows which input state represents inhibition
        # (it will be wrong if the user deletes an input state: currently, deleting input states is not supported,
        # so it shouldn't be a problem)
        self.indexOfInhibitionInputState = len(self.input_states) - 1

        try:
            int_k_value = int(self.k_value[0])
        except:
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

        a = get_auto_matrix(self.auto, self.size[0])
        h = get_hetero_matrix(self.hetero, self.size[0])
        mat = a + h
        flat_mat = mat.flatten()
        if not (flat_mat <= 0).all() and not (flat_mat >= 0).all():
            raise KWTAError("matrix {} for {} should be non-positive, or "
                            "non-negative. Mixing positive and negative values can create non-supported "
                            "inhibition vectors".format(mat, self))

    # this function returns the KWTA-scaled current_input, which is scaled based on
    # self.k_value, self.threshold, self.ratio, and of course the inhibition vector
    def _kwta_scale(self, current_input, context=None):

        k = self.int_k

        inhibVector = self.input_states[self.indexOfInhibitionInputState].value  # inhibVector is the inhibition input
        inhibVector = self._validate_inhib(inhib=inhibVector, inp=current_input, context=context)

        if not isinstance(current_input, np.ndarray):
            logger.warning("input ({}) of type {} was not a numpy array: this may cause unexpected KWTA behavior".
                           format(current_input, current_input.__class__.__name__))

        sortedInput = sorted(current_input, reverse=True)  # sortedInput is the values of current_input, sorted

        # current_input[indices[i - 1]] is the i-th largest element of current_input
        indices = []
        for i in builtins.range(int(self.size[0])):
            j = 0
            w = np.where(current_input == sortedInput[i])
            while w[0][j] in indices:
                j += 1
            indices.append(np.where(current_input == sortedInput[i])[0][j])
        indices = np.array(indices)

        # scales[i] is the scale on inhibition that would put the (i+1)-th largest
        # element in current_input at the threshold
        scales = np.zeros(int(self.size[0]))
        for i in builtins.range(int(self.size[0])):
            inhib = inhibVector[indices[i]]
            if inhib == 0:
                pass
            else:
                scales[i] = (self.threshold - current_input[indices[i]]) / inhib

        # ratio determines where between the two scales our final scale will lie
        # for most situations where the inhibition vector is negative, a lower ratio means more inhibition
        sk = sorted(scales, reverse=True)[k]
        skMinusOne = sorted(scales, reverse=True)[k - 1]
        final_scale = sk * self.ratio + skMinusOne * (1 - self.ratio)

        out = current_input + final_scale * inhibVector

        if (sum(out > self.threshold) > k) or (sum(out < self.threshold) > len(out) - k):
            warnings.warn("KWTA scaling was not fully successful. The input was {}, the inhibition vector was {}, "
                          "and the KWTA-scaled input was {}".format(current_input, inhibVector, out))

        return out

    def _validate_inhib(self, inhib, inp, context=None):
        inhib = np.array(inhib)
        if (inhib == 0).all():
            if type(context) == str and INITIALIZING not in context:
                logger.info("inhib vector ({}) was all zeros (while input was ({})), so inhibition will be uniform".
                      format(inhib, inp))
            inhib = np.ones(int(self.size[0]))
        if (inhib > 0).all():
            inhib = -1 * inhib
        if (inhib == 0).any():
            raise KWTAError("inhibition vector ({}) for {} contained some, but not all, zeros: not "
                            "currently supported".format(inhib, self))
        if (inhib > 0).any():
            raise KWTAError("inhibition vector ({}) for {} was not all positive or all negative: not "
                            "currently supported".format(inhib, self))
        if len(inhib) != len(inp):
            raise KWTAError("The inhibition vector ({}) for {} is of a different length than the"
                            " current primary input vector ({}).".format(inhib, self, input))

        return inhib

    # # deprecated: used when variable was length 1.
    # def execute(self,
    #             input=None,
    #             runtime_params=None,
    #             clock=CentralClock,
    #             time_scale=TimeScale.TRIAL,
    #             ignore_execution_id = False,
    #             context=None):
    #     context = context or NO_CONTEXT
    #     if EXECUTING not in context and EVC_SIMULATION not in context:
    #         if input is None:
    #             input = self.variableInstanceDefault
    #         if isinstance(input, list):
    #             input.append(np.zeros(self.size[0]))
    #         elif isinstance(input, np.ndarray):
    #             new_input = list(input)
    #             new_input.append(np.zeros(self.size[0]))
    #             input = np.array(new_input)
    #
    #     return super().execute(input=input,
    #                            runtime_params=runtime_params,
    #                            clock=clock,
    #                            time_scale=time_scale,
    #                            ignore_execution_id=ignore_execution_id,
    #                            context=context)

    # this is the exact same as _execute in TransferMechanism, except that this _execute calls _kwta_scale()
    # and implements decay as self.previous_input *= self.decay

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate shape and size of matrix and decay.
        """

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        if RATIO in target_set:
            ratio_param = target_set[RATIO]
            if not isinstance(ratio_param, numbers.Real):
                if not (isinstance(ratio_param, (np.ndarray, list)) and len(ratio_param) == 1):
                    raise KWTAError("ratio parameter ({}) for {} must be a single number".format(ratio_param, self))

        if K_VALUE in target_set:
            k_param = target_set[K_VALUE]
            if not isinstance(k_param, numbers.Real):
                if not (isinstance(k_param, (np.ndarray, list)) and len(k_param) == 1):
                    raise KWTAError("k-value parameter ({}) for {} must be a single number".format(k_param, self))

        if THRESHOLD in target_set:
            threshold_param = target_set[THRESHOLD]
            if not isinstance(threshold_param, numbers.Real):
                if not (isinstance(threshold_param, (np.ndarray, list)) and len(threshold_param) == 1):
                    raise KWTAError(
                        "k-value parameter ({}) for {} must be a single number".format(threshold_param, self))

        # Validate MATRIX
        if MATRIX in target_set:

            matrix_param = target_set[MATRIX]
            size = len(self.variable[0])
            if isinstance(matrix_param, MappingProjection):
                matrix = matrix_param.matrix
            else:
                matrix = get_matrix(matrix_param, size, size)
            if matrix is not None:
                flat_mat = matrix.flatten()
                if not (flat_mat <= 0).all() and not (flat_mat >= 0).all():
                    raise KWTAError("matrix {} (from matrix specification {}) for {} should be non-positive, or "
                                    "non-negative. Mixing positive and negative values can create non-supported "
                                    "inhibition vectors".format(matrix, matrix_param, self))

            # 8/1/17 CW: if matrix is not all non-negative or non-positive, then it may result in inappropriate
            # inhibition vectors since KWTA currently does not support inhibVectors that are part positive part negative

    def execute(self,
                input=None,
                runtime_params=None,
                clock=CentralClock,
                time_scale=TimeScale.TRIAL,
                ignore_execution_id = False,
                context=None):
        if isinstance(input, str) or (isinstance(input, (list, np.ndarray)) and isinstance(input[0], str)):
            raise KWTAError("input ({}) to {} was a string, which is not supported for {}".
                            format(input, self, self.__class__.__name__))
        if input is not None:
            input = list(np.atleast_2d(input))
            if (input is not None) and len(input) == len(self.variable) - 1:
                input.append(np.zeros(self.size[0]))
            input = np.array(input)

        return super().execute(input=input, runtime_params=runtime_params, clock=clock, time_scale=time_scale,
                               ignore_execution_id=ignore_execution_id, context=context)
    def _execute(self,
                variable=None,
                runtime_params=None,
                clock=CentralClock,
                time_scale = TimeScale.TRIAL,
                context=None):

        self.variable[0] = self._kwta_scale(self.variable[0], context=context)

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

    @tc.typecheck
    def _instantiate_recurrent_projection(self,
                                          mech: Mechanism_Base,
                                          matrix=FULL_CONNECTIVITY_MATRIX,
                                          context=None):
        """Instantiate a MappingProjection from mech to itself

        """

        if isinstance(matrix, str):
            size = len(mech.variable[0])
            matrix = get_matrix(matrix, size, size)

        return AutoAssociativeProjection(sender=mech,
                                         receiver=mech.input_states[mech.indexOfInhibitionInputState],
                                         matrix=matrix,
                                         name=mech.name + ' recurrent projection')
