# Princeton University licenses this file to You under the Apache License,
# Version 2.0 (the "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
from typing import Iterable

import numpy as np
import typecheck as tc

from psyneulink.components.functions.function import LinearCombination
from psyneulink.components.mechanisms.mechanism import Mechanism_Base
from psyneulink.components.mechanisms.processing.objectivemechanism import \
    OUTCOME
from psyneulink.components.states.outputstate import OutputState
from psyneulink.globals.keywords import INITIALIZING, \
    PREDICTION_ERROR_MECHANISM, VARIABLE, SAMPLE
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set, \
    kpReportOutputPref
from psyneulink.globals.preferences.preferenceset import PreferenceEntry, \
    PreferenceLevel, kwPreferenceSetName
from psyneulink.globals.utilities import is_numeric
from psyneulink.library.mechanisms.processing.objective.comparatormechanism \
    import ComparatorMechanism, ComparatorMechanismError, MSE
from psyneulink.scheduling.timescale import CentralClock


class PredictionErrorMechanismError(ComparatorMechanismError):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class PredictionErrorMechanism(ComparatorMechanism):
    """
    PredictionErrorMechanism(                                \
        sample,                                              \
        target,                                              \
        input_states=[SAMPLE,TARGET]                         \
        function=LinearCombination,                          \
        output_states=[OUTCOME, MSE],                        \
        params=None,                                         \
        name=None,                                           \
        prefs=None)

    Calculates the prediction error between the predicted reward and the target
    """
    componentType = PREDICTION_ERROR_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    classPreferences = {
        kwPreferenceSetName: 'PredictionErrorMechanismCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
    }

    class ClassDefaults(ComparatorMechanism.ClassDefaults):
        variable = None

    paramClassDefaults = ComparatorMechanism.paramClassDefaults.copy()
    standard_output_states = ComparatorMechanism.standard_output_states.copy()

    @tc.typecheck
    def __init__(self,
                 sample: tc.optional(tc.any(OutputState, Mechanism_Base, dict,
                                            is_numeric,
                                            str)) = None,
                 target: tc.optional(tc.any(OutputState, Mechanism_Base, dict,
                                            is_numeric,
                                            str)) = None,
                 function=LinearCombination(weights=[[-1], [1]]),
                 output_states: tc.optional(tc.any(str, Iterable)) = OUTCOME,
                 learning_rate=0.3,
                 gamma=.99,
                 max_time_steps=0,
                 params=None,
                 name=None,
                 prefs: is_pref_set = None,
                 context=componentType + INITIALIZING):
        input_states = [sample, target]
        params = self._assign_args_to_param_dicts(sample=sample,
                                                  target=target,
                                                  function=function,
                                                  input_states=input_states,
                                                  output_states=output_states,
                                                  learning_rate=learning_rate,
                                                  gamma=gamma,
                                                  max_time_steps=max_time_steps,
                                                  params=params)

        super().__init__(sample=sample,
                         target=target,
                         input_states=input_states,
                         function=function,
                         output_states=output_states,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)

        print("input state for PEM = {}".format(self.input_state))
        print("PEM value = {}".format(self.function))
        # self.standard_output_states[OUTCOME].

        # self.integrator_function = AdaptiveIntegrator(
        #         initializer=0,
        #         owner=self,
        #         context=context)
        # self.delta_integrator_function = AdaptiveIntegrator(
        #         initializer=0,
        #         owner=self,
        #         context=context
        # )
        # self.prev_val = self.integrator_function.previous_value
        self.t = 0
        # self._max_time_steps = max_time_steps
        # self.max_t = max_time_steps or 0
        # self.load = True
        # self.max_t_default = True if self.max_t != 0 else False
        # self.prev_delta = 0
        # NOTE: assume gamma and lambda are 1

    def _execute(self, variable=None, runtime_params=None, clock=CentralClock,
                 time_scale=None, context=None):
        # TODO: update to take sample/reward from variable
        # sample = x(t) in Montague
        from globals.keywords import SAMPLE, TARGET, INITIALIZING, INPUT_STATES

        sample = self.paramsCurrent[INPUT_STATES][SAMPLE].value
        reward = self.paramsCurrent[INPUT_STATES][TARGET].value

        delta = np.zeros_like(sample)
        if clock.trial == 0:
            sample = sample * 0

        # FIXME: correct the value size
        # -- call function with sample[0:t-1], sample[1:t]
        # -- add reward to returned value
        sample_prev_t = self.gamma * sample[0:len(sample) - 1]
        sample_next_t = sample[1:len(sample)]

        new_sample = self.function(variable=[sample_prev_t, sample_next_t])

        for t in range(0, len(sample) - 1):
            delta[t] = reward[t + 1] + new_sample[t]

        # for t in range(0, len(sample) - 1):
        #     delta[t] = reward[t + 1] + self.function(sample[t + 1],
        #                                              sample[t])

        # v_t = sample
        # if self.t == 0:
        #     v_t_minus_1 = 0
        # else:
        #     if self.load:
        #         if not self.max_t_default:
        #             np.append(self.e, 0)
        #             np.append(self.utility_matrix, v_t)
        #         self.utility_matrix[self.t] = v_t
        #
        #     v_t_minus_1 = self.utility_matrix[self.t - 1]
        #
        #     # v_t = self.get_value(v_t, self.t, self.learning_rate,
        #     #                      self.prev_delta)
        #     # v_t_minus_1 = self.get_value(
        #     #     self.integrator_function.previous_value, self.t,
        #     #     self.learning_rate, self.prev_delta)
        # v_t *= self.gamma
        # values = [v_t_minus_1, v_t]
        #
        # output_vector = self.function(variable=values)
        # delta_t = reward[0] + output_vector[0]
        # if np.any(sample):
        #     self.e[self.t] += 1
        #     self._update_utility_matrix(self.learning_rate, delta_t)
        #     self.prev_delta = delta_t
        #     self._update_eligibility_trace()
        #
        # self.t += 1
        #
        # if self.load:
        #     if not self.max_t_default:
        #         if reward > 0:
        #             self.load = False
        #             self.max_t = self.t
        #             self.t = 0
        #     else:
        #         if self.t == self.max_t:
        #             self.load = False
        #             self.t = 0
        #
        #
        # else:
        #
        #     # delta_t = self.eligibility_trace[self.t + 1] +
        # output_vector[0]
        #     # self.t += 1
        #
        #     if self.t == self.max_t:
        #         self.t = 0
        print("delta value = {}".format(delta))
        return delta

    def get_value(self, sample_value, t, alpha, prev_delta):
        return sample_value * self.e[t] * alpha * prev_delta

    def _update_eligibility_trace(self, lambda_=0.5, gamma=.99):
        self.e = self.e * lambda_ * gamma

    def reset(self):
        self.e.fill(0)
