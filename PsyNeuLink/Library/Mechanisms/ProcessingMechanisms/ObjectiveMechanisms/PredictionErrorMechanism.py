import typecheck as tc

from PsyNeuLink import CentralClock
from PsyNeuLink.Components.Functions.Function import AdaptiveIntegrator, \
    TDDeltaFunction
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism_Base
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism \
    import OUTCOME
from PsyNeuLink.Components.States import OutputState
from PsyNeuLink.Globals.Keywords import PREDICTION_ERROR_MECHANISM, SAMPLE, \
    TARGET
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set
from PsyNeuLink.Globals.Utilities import is_numeric
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.ObjectiveMechanisms \
    .ComparatorMechanism import ComparatorMechanism, MSE


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
                 reward=None,
                 input_states=[SAMPLE, TARGET],
                 function=TDDeltaFunction(),
                 output_states: tc.optional(tc.any(list, dict)) = [OUTCOME,
                                                                   MSE],
                 params=None,
                 name=None,
                 prefs: is_pref_set = None,
                 context=None):
        params = self._assign_args_to_param_dicts(sample=sample, target=target,
                                                  reward=reward,
                                                  input_states=input_states,
                                                  output_states=output_states,
                                                  params=params)
        super(PredictionErrorMechanism, self).__init__(sample=sample,
                                                       target=target,
                                                       input_states=input_states,
                                                       function=function,
                                                       output_states=output_states,
                                                       params=params, name=name,
                                                       prefs=prefs,
                                                       context=context)

    def _execute(self, variable=None, runtime_params=None, clock=CentralClock,
                 time_scale=None, context=None):
        self.integrator_function = AdaptiveIntegrator(default_variable=variable,
                                                      params=runtime_params,
                                                      owner=self)

        values = [self.integrator_function.execute(variable=variable,
                                                   params=runtime_params,
                                                   context=context),
                  self.integrator_function.previous_value]
        output_vector = self.function(default_variable=values,
                                      params=runtime_params)
        return output_vector
