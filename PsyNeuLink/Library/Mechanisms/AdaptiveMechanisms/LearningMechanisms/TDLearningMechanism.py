# Princeton University licenses this file to You under the Apache License,
# Version 2.0 (the "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import numpy as np

import typecheck as tc

from PsyNeuLink import CentralClock
from PsyNeuLink.Components.Functions.Function import ModulationParam, \
    is_function_type, _is_modulation_param, TDLearning
from PsyNeuLink.Components.Mechanisms import Mechanism
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanism \
    .LearningMechanism import LearningMechanismError, LearningMechanism
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceLevel
from PsyNeuLink.Globals.Utilities import parameter_spec
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.ObjectiveMechanisms \
    .PredictionErrorMechanism import PredictionErrorMechanism
from PsyNeuLink.Scheduling.TimeScale import TimeScale


class TDLearningMechanismError(LearningMechanismError):
    def __init__(self, error_msg):
        super().__init__(error_msg)


class TDLearningMechanism(LearningMechanism):
    """
    TDLearningMechanism(                    \
        variable,                           \
        error_source,                       \
        function=None,                      \
        learning_rate=None,                 \
        learning_signals=LEARNING_SIGNAL,   \
        modulation=ModulationParam.ADDITIVE,\
        params=None,                        \
        name=None,                          \
        prefs=None)                         \

    Implements a Mechanism that modifies the `matrix <MappingProjection.matrix>`
    parameter of a `MappingProjection` using a temporal difference algorithm.

    """
    className = "TDLearningMechanism"
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    @tc.typecheck
    def __init__(self,
                 variable: tc.any(list, np.ndarray),
                 size=None,
                 error_source: tc.optional(Mechanism)=PredictionErrorMechanism,
                 function: is_function_type=TDLearning,
                 learning_signals: tc.optional(list)=None,
                 modulation: tc.optional(_is_modulation_param)=ModulationParam.ADDITIVE,
                 learning_rate: tc.optional(parameter_spec)=None,
                 params=None,
                 name=None,
                 prefs: is_pref_set = None,
                 context=None):
        params = self._assign_args_to_param_dicts(error_source=error_source,
                                                  function=function,
                                                  learning_signals=learning_signals,
                                                  modulation=modulation,
                                                  learning_rate=learning_rate,
                                                  params=params)

        super().__init__(variable=variable, size=size, params=params, name=name,
                         prefs=prefs, context=context)

        # just for the time being until EventBoundaryMechanisms are created
        self.T = 0
        self.num_timesteps = 0
        self.load = True
        # activation_input --> sample (aka x)
        # activation_output --> reward
        # error_signal --> output of PredictionErrorMechanism

    def _execute(self, variable=None, runtime_params=None, clock=CentralClock,
                 time_scale=TimeScale.TRIAL, context=None):
        if variable:
            sample = variable[0]
            reward = variable[1]
        error_signal = self.error_signal

        return self.function(variable=self.variable, params=self.params,
                             context=context)
