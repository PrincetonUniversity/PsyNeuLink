import typecheck as tc

from PsyNeuLink.Components.Functions.Function import LinearCombination
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism_Base

from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism \
    import \
    OUTCOME
from PsyNeuLink.Components.States import OutputState
from PsyNeuLink.Globals.Keywords import PREDICTION_ERROR_MECHANISM, SAMPLE, \
    TARGET
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set
from PsyNeuLink.Globals.Utilities import is_numeric
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.ObjectiveMechanisms \
    .ComparatorMechanism import \
    MSE, ComparatorMechanism


class PredictionErrorMechanism(ComparatorMechanism):
    componentType = PREDICTION_ERROR_MECHANISM
    paramClassDefaults = ComparatorMechanism.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 sample: tc.optional(
                     tc.any(OutputState, Mechanism_Base, dict, is_numeric,
                            str)) = None,
                 target: tc.optional(
                     tc.any(OutputState, Mechanism_Base, dict, is_numeric,
                            str)) = None,
                 input_states=[SAMPLE, TARGET],
                 function=LinearCombination(),
                 output_states: tc.optional(tc.any(list, dict)) = [OUTCOME, MSE],
                 params=None,
                 name=None,
                 prefs: is_pref_set = None,
                 context=None):
        super(PredictionErrorMechanism, self).__init__(sample=sample,
                                                       target=target,
                                                       input_states=input_states,
                                                       function=function,
                                                       output_states=output_states,
                                                       params=params, name=name,
                                                       prefs=prefs,
                                                       context=context)
