import typecheck as tc

from PsyNeuLink import CentralClock
from PsyNeuLink.Components.Functions.Function import AdaptiveIntegrator, \
    TDDeltaFunction
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism_Base
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism \
    import OUTCOME
from PsyNeuLink.Components.States.OutputState import OutputState
from PsyNeuLink.Globals.Keywords import FUNCTION_PARAMS, INITIALIZING, \
    INPUT_STATES, PREDICTION_ERROR_MECHANISM, SAMPLE, T, TARGET, \
    kwPreferenceSetName, REWARD
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set, \
    kpReportOutputPref
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceEntry, \
    PreferenceLevel
from PsyNeuLink.Globals.Utilities import is_numeric
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.ObjectiveMechanisms \
    .ComparatorMechanism import ComparatorMechanism, MSE


class PredictionErrorMechanism(ComparatorMechanism):
    """
    PredictionErrorMechanism(                                \
        sample,                                              \
        target,                                              \
        input_states=[SAMPLE,TARGET]                         \
        function=LinearCombination(weights=[[-1],[1]],       \
        output_states=[OUTCOME]                              \
        params=None,                                         \
        name=None,                                           \
        prefs=None)

    Subclass of `ComparatorMechanism` that calculates the prediction error
    between the sample and the target
    """
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

        self.integrator_function = AdaptiveIntegrator(
            initializer=0,
            owner=self,
            context=context)
        self.prev_val = self.integrator_function.previous_value
        self.t = 1

    def _execute(self, variable=None, runtime_params=None, clock=CentralClock,
                 time_scale=None, context=None):
        print("Calling Prediction Error Mechanism _execute()")
        print("variable = {}".format(variable))

        if self.paramsCurrent:
            sample = self.paramsCurrent[INPUT_STATES][SAMPLE].value
            reward = self.paramsCurrent[INPUT_STATES][TARGET].value
        else:
            sample = self.sample
            reward = self.target

        try:
            # integrator_prev_val = self.integrator_function.previous_value
            # TODO: AdaptiveIntegrator needs to remember the last value of V
            print("prev_val = {}".format(self.prev_val))
            self.integrator_function.execute(variable=sample, context=context)[0]
            self.prev_val = self.integrator_function.previous_value
            values = [variable[0], self.prev_val]
        except AttributeError:
            values = [0, 0]

        try:
            t = self.t
        except AttributeError:
            self.t = t = 1
        output_vector = self.function(variable=values,
                                      params={T: t,
                                              REWARD: reward})
        self.prev_val = values[0]
        return output_vector
