# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *******************************************  FeatureInputState *****************************************************
#
"""
FeatureInputState is a subclass of InputState that permits assignment of a function other than a CombinationFunction.
See InputState for documentation.
"""

import numbers
import warnings

import collections
import numpy as np
import typecheck as tc

from psyneulink.core.components.functions.combinationfunctions import LinearCombination
from psyneulink.core.components.states.inputstate import InputState
from psyneulink.core.components.states.outputstate import OutputState
from psyneulink.core.components.states.state import StateError, State_Base, _instantiate_state_list, state_type_keywords
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import EXPONENT, FEATURE_INPUT_STATE, GATING_SIGNAL, INPUT_STATE, INPUT_STATE_PARAMS, LEARNING_SIGNAL, MAPPING_PROJECTION, MATRIX, MECHANISM, OPERATION, OUTPUT_STATE, OUTPUT_STATES, PROCESS_INPUT_STATE, PRODUCT, PROJECTIONS, PROJECTION_TYPE, REFERENCE_VALUE, SENDER, SIZE, SUM, SYSTEM_INPUT_STATE, VALUE, VARIABLE, WEIGHT
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import append_type_to_name, is_numeric, iscompatible

__all__ = [
    'FeatureInputState', 'FeatureInputStateError', 'state_type_keywords',
]

state_type_keywords = state_type_keywords.update({FEATURE_INPUT_STATE})

WEIGHT_INDEX = 1
EXPONENT_INDEX = 2

DEFER_VARIABLE_SPEC_TO_MECH_MSG = "FeatureInputState variable not yet defined, defer to Mechanism"

class FeatureInputStateError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class FeatureInputState(InputState):
    """
    FeatureInputState(                             \
        owner=None,                                \
        variable=None,                             \
        size=None,                                 \
        function=LinearCombination(operation=SUM), \
        combine=None,                              \
        projections=None,                          \
        weight=None,                               \
        exponent=None,                             \
        internal_only=False,                       \
        params=None,                               \
        name=None,                                 \
        prefs=None)

    Subclass of `InputState <InputState>` that receives only one `PathwayProjection <PathwayProjection>`, and may apply
    any `Function` to its variable.  See `InputState` for documentation of all arguments and attributes.
    """

    #region CLASS ATTRIBUTES

    componentType = INPUT_STATE
    paramsType = INPUT_STATE_PARAMS

    stateAttributes = State_Base.stateAttributes | {WEIGHT, EXPONENT}

    connectsWith = [OUTPUT_STATE,
                    PROCESS_INPUT_STATE,
                    SYSTEM_INPUT_STATE,
                    LEARNING_SIGNAL,
                    GATING_SIGNAL]
    connectsWithAttribute = [OUTPUT_STATES]
    projectionSocket = SENDER
    modulators = [GATING_SIGNAL]


    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'FeatureInputStateCustomClassPreferences',
    #     kp<pref>: <setting>...}

    # Note: the following enforce encoding as 1D np.ndarrays (one variable/value array per state)
    variableEncodingDim = 1
    valueEncodingDim = 1

    class Parameters(State_Base.Parameters):
        """
            Attributes
            ----------

                exponent
                    see `exponent <FeatureInputState.exponent>`

                    :default value: None
                    :type:

                function
                    see `function <FeatureInputState.function>`

                    :default value: `LinearCombination`(offset=0.0, operation=sum, scale=1.0)
                    :type: `Function`

                internal_only
                    see `internal_only <FeatureInputState.internal_only>`

                    :default value: False
                    :type: bool

                weight
                    see `weight <FeatureInputState.weight>`

                    :default value: None
                    :type:

        """
        function = Parameter(LinearCombination(operation=SUM), stateful=False, loggable=False)
        weight = Parameter(None, modulable=True)
        exponent = Parameter(None, modulable=True)
        combine = None
        internal_only = Parameter(False, stateful=False, loggable=False)

    paramClassDefaults = State_Base.paramClassDefaults.copy()
    paramClassDefaults.update({PROJECTION_TYPE: MAPPING_PROJECTION,
                               MECHANISM: None,     # These are used to specifiy FeatureInputStates by projections to them
                               OUTPUT_STATES: None  # from the OutputStates of a particular Mechanism (see docs)
                               })
    #endregion

    @tc.typecheck
    def __init__(self,
                 owner=None,
                 reference_value=None,
                 variable=None,
                 size=None,
                 function=None,
                 projections=None,
                 combine:tc.optional(tc.enum(SUM,PRODUCT))=None,
                 weight=None,
                 exponent=None,
                 internal_only:bool=False,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        super(FeatureInputState, self).__init__(owner,
                                                reference_value=reference_value,
                                                variable=variable,
                                                size=size,
                                                function=function,
                                                projections=projections,
                                                combine=combine,
                                                weight=weight,
                                                exponent=exponent,
                                                internal_only=internal_only,
                                                params=params,
                                                name=name,
                                                prefs=prefs,
                                                context=context)

    def _validate_function(self, function):
        '''Override InputState._validate_function to permit any function'''
        pass



