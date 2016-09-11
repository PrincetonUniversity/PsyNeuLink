# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *************************************  Stimulus Prediction Mechanism *************************************************
#

import numpy as np
from numpy import sqrt, abs, tanh, exp
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.ProcessingMechanism import *

# AdaptiveIntegrator parameter keywords:
DEFAULT_RATE = 0.5

class AdaptiveIntegratorMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class AdaptiveIntegratorMechanism(ProcessingMechanism_Base):
# DOCUMENT:
    """Generate output based on Wiener filter of sequence of inputs

    Description:
        - DOCUMENT:

    Instantiation:
        - DOCUMENT:

    Initialization arguments:
         DOCUMENT:

    Parameters:
        DOCUMENT:  learningRate

    MechanismRegistry:
        All instances of SigmoidLayer are registered in MechanismRegistry, which maintains an entry for the subclass,
          a count for all instances of it, and a dictionary of those instances

    Naming:
        Instances of SigmoidLayer can be named explicitly (using the name='<name>' argument).
        If this argument is omitted, it will be assigned "SigmoidLayer" with a hyphenated, indexed suffix ('SigmoidLayer-n')

    Class attributes:
        + functionType (str): SigmoidLayer
        + classPreference (PreferenceSet): SigmoidLayer_PreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE
        + variableClassDefault (value):  SigmoidLayer_DEFAULT_BIAS
        + paramClassDefaults (dict): {kwTimeScale: TimeScale.TRIAL,
                                      FUNCTION_PARAMS:{kwSigmoidLayer_Unitst: kwSigmoidLayer_NetInput
                                                                 kwSigmoidLayer_Gain: SigmoidLayer_DEFAULT_GAIN
                                                                 kwSigmoidLayer_Bias: SigmoidLayer_DEFAULT_BIAS}}
        + paramNames (dict): names as above

    Class methods:
        None

    Instance attributes: none
        + variable - input to mechanism's execute method (default:  SigmoidLayer_DEFAULT_NET_INPUT)
        + params DOCUMENT:
        + learningRate DOCUMENT:
        + name (str) - if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet) - if not specified as an arg, a default set is created by copying SigmoidLayer_PreferenceSet

    Instance methods:
    """

    functionType = "SigmoidLayer"

    classPreferenceLevel = PreferenceLevel.TYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'AdaptiveIntegratorMechanismCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE)}

    # Sets template for variable (input)
    variableClassDefault = [[0]]

    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        kwTimeScale: TimeScale.TRIAL,
        kwOutputStates:[kwPredictionMechanismOutput]
    })

    # Set default input_value to default bias for SigmoidLayer
    paramNames = paramClassDefaults.keys()

    from PsyNeuLink.Functions.Utility import Integrator

    def __init__(self,
                 default_input_value=NotImplemented,
                 function=Integrator(rate=0.5,
                                     weighting=Integrator.Weightings.TIME_AVERAGED),
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented,
                 context=NotImplemented):
        """Assign type-level preferences, default input value (SigmoidLayer_DEFAULT_BIAS) and call super.__init__

        :param default_input_value: (value)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        """

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self.assign_args_to_param_dicts(function=function, params=params)

        # if default_input_value is NotImplemented:
        #     default_input_value = SigmoidLayer_DEFAULT_NET_INPUT

        super(AdaptiveIntegratorMechanism, self).__init__(variable=default_input_value,
                                  params=params,
                                  name=name,
                                  prefs=prefs,
                                  context=self)

        # IMPLEMENT: INITIALIZE LOG ENTRIES, NOW THAT ALL PARTS OF THE MECHANISM HAVE BEEN INSTANTIATED




