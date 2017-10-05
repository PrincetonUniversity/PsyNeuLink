# Princeton University licenses this file to You under the Apache License,
# Version 2.0 (the "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
from typing import List, Union

import numpy as np
from PsyNeuLink.Components.Mechanisms.Mechanism import MechanismError

from PsyNeuLink.Components import Mechanism_Base
from PsyNeuLink.Components.Functions.Function import PreferenceSet, \
    ModulationParam
from PsyNeuLink.Components.Functions.TDLearning import TDLearning
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.AdaptiveMechanism \
    import \
    AdaptiveMechanismError, AdaptiveMechanism_Base
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanisms \
    .LearningMechanism import Function, LearningMechanism, \
    LearningMechanismError
from PsyNeuLink.Globals.Keywords import LEARNING_MECHANISM, MECHANISM
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceLevel


class TDLearningMechanismError(MechanismError):
    def __init__(self, error_msg):
        super().__init__(error_msg)


class TDLearningMechanism(Mechanism_Base):
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
    componentType = MECHANISM
    className = "TDLearningMechanism"
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    def __init__(self,
                 variable: Union[List, np.ndarray],
                 size: Union[int, List, np.ndarray] = None,
                 error_source=None,
                 function: Function = TDLearning,
                 params=None,
                 name: str = None,
                 prefs: PreferenceSet = None,
                 context=None):
        print("size = {}".format(size))
        print("variable = {}".format(variable))
        params = self._assign_args_to_param_dicts(error_source=error_source,
                                                  function=function,
                                                  params=params)

        if context is None:
            context = self

        self._size = 0

        super().__init__(variable=variable, size=size, params=params, name=name,
                         prefs=prefs, context=context)

        # TODO: get rid of magic numbers
        self.function = function
        self.weight_change_matrix = np.zeros(len(variable[0]))
        self.context = context
        self.name = "TDLearning Mechanism"

    def _execute(self, **kwargs):
        return self.function(variable=self.variable, params=self.params,
                             context=self.context)
