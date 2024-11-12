# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **********************************************  Mechanism ***********************************************************

"""
**[DOCUMENTATION STILL UNDER CONSTRUCTION]**

"""
import numpy as np
from beartype import beartype

from psyneulink._typing import Optional, Union

from psyneulink.core.components.mechanisms.mechanism import Mechanism_Base
from psyneulink.core.globals.defaults import SystemDefaultInputValue
from psyneulink.core.globals.keywords import DEFAULT_PROCESSING_MECHANISM
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.preferences.basepreferenceset import ValidPrefSet
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel

# **************************************** DefaultProcessingMechanism ******************************************************

__all__ = []


# class DefaultProcessingMechanism_Base(ProcessingMechanism_Base):
class DefaultProcessingMechanism_Base(Mechanism_Base):
    """Subclass of `ProcessingMechanism <ProcessingMechanism>` used to implement SystemDefaultInputMechanism,
    DefaultControlMechanism, and SystemDefaultOutputMechanism.

    Description:
        Implements "dummy" Mechanism used to implement default input, control signals, and outputs to other mechanisms

    Class attributes:
        + componentType (str): System Default Mechanism
    """

    componentName = DEFAULT_PROCESSING_MECHANISM
    componentType = DEFAULT_PROCESSING_MECHANISM
    onlyFunctionOnInit = True

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # Any preferences specified below will override those specified in SUBTYPE_DEFAULT_PREFERENCES
    # Note: only need to specify setting;  level will be assigned to SUBTYPE automatically
    # classPreferences = {
    #     PREFERENCE_SET_NAME: 'DefaultProcessingMechanismClassPreferences',
    #     PREFERENCE_KEYWORD<pref>: <setting>...}

    class Parameters(Mechanism_Base.Parameters):
        variable = Parameter(np.array([SystemDefaultInputValue]), constructor_argument='default_variable')

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 input_shapes=None,
                 params=None,
                 name=None,
                 prefs:   Optional[ValidPrefSet] = None,
                 function=None,
                 **kwargs
                 ):
        """Add Linear as default function, assign default name, and call super.__init__

        :param default_variable: (value)
        :param input_shapes: (int or list/array of ints)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        """

        super(DefaultProcessingMechanism_Base, self).__init__(default_variable=default_variable,
                                                              input_shapes=input_shapes,
                                                              function=function,
                                                              params=params,
                                                              name=name,
                                                              prefs=prefs,

                                                              **kwargs)
