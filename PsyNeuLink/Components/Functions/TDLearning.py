import numpy as np

from PsyNeuLink.Components.Component import ComponentError

from PsyNeuLink.Components.Mechanisms import Mechanism

from PsyNeuLink import TDLEARNING_FUNCTION, LEARNING_ACTIVATION_INPUT, \
    LEARNING_ACTIVATION_OUTPUT, LEARNING_ERROR_OUTPUT, INITIALIZING
from PsyNeuLink.Components.Functions.Function import LearningFunction, \
    Function_Base


class TDLearning(LearningFunction):
    componentName = TDLEARNING_FUNCTION
    variableClassDefault = [[0], [0], [0]]
    default_learning_rate = 0.05
    paramClassDefaults = Function_Base.paramClassDefaults

    def __init__(self,
                 default_variable=variableClassDefault,
                 learning_rate: float = default_learning_rate,
                 params=None,
                 owner: Mechanism = None,
                 prefs=None,
                 context='Component Init'):
        params = self._assign_args_to_param_dicts(learning_rate=learning_rate,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        self.learning_rate = learning_rate
        self.functionOutputType = None

    def _validate_variable(self, variable, context=None):
        super()._validate_variable(variable, context)

        if len(self.variable) != 3:
            raise ComponentError("Variable for {} ({}) must have three items "
                                 "(input, output, and error arrays)".format(
                self.name, self.variable))

        self.activation_input = self.variable[LEARNING_ACTIVATION_INPUT]
        self.activation_output = self.variable[LEARNING_ACTIVATION_OUTPUT]
        self.error_signal = self.variable[LEARNING_ERROR_OUTPUT]

        if len(self.error_signal) != 1:
            raise ComponentError("Error term for {} (the third item of its "
                                 "variable arg) must be an array with a single "
                                 "element for {}".format(self.name,
                                                         self.error_signal))

        if INITIALIZING not in context:
            if np.count_nonzero(self.activation_output) != 1:
                raise ComponentError("First item ({}) of variable for {} must "
                                     "be an array with a single non-zero "
                                     "value".format(
                                                    self.variable[LEARNING_ACTIVATION_OUTPUT],
                                                    self.componentName))

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        self._check_args(variable=variable, params=params, context=context)

        output = self.activation_output
        error = self.error_signal
        learning_rate = self.learning_rate

        error_array = np.where(output, learning_rate * error, 0)

        weight_change_matrix = np.diag(error_array)