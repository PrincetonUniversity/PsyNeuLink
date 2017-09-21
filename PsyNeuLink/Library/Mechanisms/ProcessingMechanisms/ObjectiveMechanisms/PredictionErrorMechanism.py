import typecheck as tc

from PsyNeuLink import CentralClock
from PsyNeuLink.Components.Functions.Function import AdaptiveIntegrator, \
    TDDeltaFunction
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism_Base
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism \
    import OUTCOME
from PsyNeuLink.Components.States import OutputState
from PsyNeuLink.Globals.Keywords import PREDICTION_ERROR_MECHANISM, SAMPLE, \
    TARGET, INITIALIZING, kwPreferenceSetName
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set, \
    kpReportOutputPref
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceLevel, \
    PreferenceEntry
from PsyNeuLink.Globals.Utilities import is_numeric
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.ObjectiveMechanisms \
    .ComparatorMechanism import ComparatorMechanism, MSE


class PredictionErrorMechanism(ComparatorMechanism):
    componentType = PREDICTION_ERROR_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    classPreferences = {
        kwPreferenceSetName: 'PredictionErrorMechanismCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
    }
    paramClassDefaults = ComparatorMechanism.paramClassDefaults.copy()

    # @tc.typecheck
    def __init__(self,
                 sample: tc.any(OutputState, Mechanism_Base, dict, is_numeric,
                                str) = None,
                 target: tc.any(OutputState, Mechanism_Base, dict, is_numeric,
                                str) = None,
                 # reward=None,
                 input_states=[SAMPLE, TARGET],
                 function=TDDeltaFunction(),
                 output_states: tc.optional(tc.any(list, dict)) = [OUTCOME,
                                                                   MSE],
                 params=None,
                 name=None,
                 prefs: is_pref_set = None,
                 context=componentType + INITIALIZING):
        params = self._assign_args_to_param_dicts(sample=sample,
                                                  target=target,
                                                  # reward=reward,
                                                  function=function,
                                                  input_states=input_states,
                                                  output_states=output_states,
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

    def _execute(self, variable=None, runtime_params=None, clock=CentralClock,
                 time_scale=None, context=None):
        print("Calling Prediction Error Mechanism _execute()")

        self.integrator_function = AdaptiveIntegrator(
            initializer=0,
            owner=self,
            context=context)
        values = [self.integrator_function.execute(variable=self.sample,
                                                   params=runtime_params,
                                                   context=context)[0],
                  self.integrator_function.previous_value]
        print("values: {}".format(values))
        output_vector = self.function(variable=values,
                                      params=runtime_params)
        return output_vector
