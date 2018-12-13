# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ****************************************  DNDMechanism ***************************************************************

"""
"""

import numpy as np

from psyneulink.core.components.functions.function import Function
from psyneulink.core.components.functions.integratorfunctions import DND
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.core.globals.keywords import NAME, SIZE
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.context import ContextFlags

__all__ = ['DNDMechanism']


class DNDMechanism(ProcessingMechanism_Base):
    """The differentiable neural dictionary (DND) class. This enables episodic recall in a neural network.

    Parameters
    ----------

    Attributes
    ----------

    """

    class Params(ProcessingMechanism_Base.Params):
        variable = [[0],[0]]

    def __init__(self,
                 key_size=1,
                 value_size=1,
                 function:Function=DND,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None):

        # Template for dict entries
        default_variable = [np.zeros(key_size), np.zeros(value_size)]

        input_states = [{NAME:'KEY INPUT', SIZE:key_size},
                        {NAME:'VALUE INPUT', SIZE:value_size}]

        params = self._assign_args_to_param_dicts(function=function,
                                                  input_states=input_states,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR
                         )

    def _instantiate_attributes_after_function(self, context=None):
        super()._instantiate_attributes_after_function(context=context)
        self.dict = self.function_object.dict
