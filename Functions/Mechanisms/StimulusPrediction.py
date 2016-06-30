#
# *************************************  Stimulus Prediction Mechanism *************************************************
#

import numpy as np
from numpy import sqrt, abs, tanh, exp
from Functions.Mechanisms.Mechanism import *

# SigmoidLayer parameter keywords:
DEFAULT_LEARNING_RATE = 1

class StimulusPredictionMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class StimulusPredictionMechanism(Mechanism_Base):
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
                                      kwExecuteMethodParams:{kwSigmoidLayer_Unitst: kwSigmoidLayer_NetInput, kwControlSignal
                                                                 kwSigmoidLayer_Gain: SigmoidLayer_DEFAULT_GAIN, kwControlSignal
                                                                 kwSigmoidLayer_Bias: SigmoidLayer_DEFAULT_BIAS, kwControlSignal}}
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
        kwPreferenceSetName: 'StimulusPredictionMechanismCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE)}

    # Sets template for variable (input)
    variableClassDefault = [[0],[0]]

    from Functions.Utility import Integrator
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        kwTimeScale: TimeScale.TRIAL,
        kwExecuteMethod: Integrator,
        kwExecuteMethodParams:{
            Integrator.kwWeighting: Integrator.Weightings.DELTA_RULE,
            Integrator.kwRate: DEFAULT_LEARNING_RATE
        },
    })

    # Set default input_value to default bias for SigmoidLayer
    paramNames = paramClassDefaults.keys()

    def __init__(self,
                 default_input_value=NotImplemented,
                 params=NotImplemented,
                 name=NotImplemented,
                 prefs=NotImplemented):
        """Assign type-level preferences, default input value (SigmoidLayer_DEFAULT_BIAS) and call super.__init__

        :param default_input_value: (value)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        """

        # Assign functionType to self.name as default;
        #  will be overridden with instance-indexed name in call to super
        if name is NotImplemented:
            self.name = self.functionType
        else:
            self.name = name
        self.functionName = self.functionType

        # if default_input_value is NotImplemented:
        #     default_input_value = SigmoidLayer_DEFAULT_NET_INPUT

        super(StimulusPredictionMechanism, self).__init__(variable=default_input_value,
                                  params=params,
                                  name=name,
                                  prefs=prefs,
                                  context=self)

        # IMPLEMENT: INITIALIZE LOG ENTRIES, NOW THAT ALL PARTS OF THE MECHANISM HAVE BEEN INSTANTIATED

    # def execute(self,
    #             variable=NotImplemented,
    #             params=NotImplemented,
    #             time_scale = TimeScale.TRIAL,
    #             context=NotImplemented):
    #     """
    #     """



