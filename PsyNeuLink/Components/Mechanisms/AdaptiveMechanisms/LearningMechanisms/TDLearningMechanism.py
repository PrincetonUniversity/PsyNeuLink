# Princeton University licenses this file to You under the Apache License,
# Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may
# obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union, List

import numpy as np
from PsyNeuLink.Globals.TimeScale import CentralClock, TimeScale

from PsyNeuLink.Components.Functions.Function import QLearning, \
    ModulationParam, \
    PreferenceSet
from PsyNeuLink.Components.Functions.Function import Sarsa
from PsyNeuLink.Components.ShellClasses import Mechanism

from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceLevel

from PsyNeuLink.Globals.Keywords import LEARNING_MECHANISM

from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanisms \
    .LearningMechanism import LearningMechanism, Function, \
    LearningMechanismError


class TDLearningMechanismError(LearningMechanismError):
    def __init__(self, error_msg):
        super(TDLearningMechanismError, self).__init__(error_msg)


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
    parameter of a `MappingProjection` using a temporal difference algorithm. It
    is essentially a convenience LearningMechanism that defaults to using
    `SARSA <Function.Sarsa>` as the function attribute.

    """
    componentType = LEARNING_MECHANISM
    className = "TDLearningMechanism"
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    def __init__(self,
                 variable: Union[List, np.ndarray],
                 size: Union[int, List, np.ndarray] = None,
                 error_source: Mechanism = None,
                 function: Union[Function, str] = None,
                 learning_signals: List = None,
                 modulation: ModulationParam = ModulationParam.ADDITIVE,
                 learning_rate: float = None,
                 discount_factor: float = None,
                 reward: float = None,
                 params=None,
                 name: str = None,
                 prefs: PreferenceSet = None,
                 context=None):
        if not function:
            function = Sarsa
        elif isinstance(function, str):
            if function == "QLearning":
                function = QLearning
            elif function == "Sarsa":
                function = Sarsa
            else:
                raise TDLearningMechanismError("Invalid function value "
                                               "specified. Valid values are "
                                               "'Sarsa', 'QLearning', or an "
                                               "instance of another Function")

        params = self._assign_args_to_param_dicts(error_source=error_source,
                                                  function=function,
                                                  learning_signals=learning_signals,
                                                  learning_rate=learning_rate,
                                                  discount_factor=discount_factor,
                                                  reward=reward,
                                                  params=params)
        self.learning_rate = learning_rate
        super().__init__(variable,
                         size=size,
                         modulation=modulation,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)

        # TODO: Override _execute() to handle QLearning and Sarsa
