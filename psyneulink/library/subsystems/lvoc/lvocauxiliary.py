# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *************************************************  EVCAuxiliary ******************************************************

"""
Auxiliary functions for `EVCControlMechanism`.

"""
import warnings

import numpy as np
import typecheck as tc

from psyneulink.components.functions.function import Function_Base, BayesGLM, EPSILON
from psyneulink.components.mechanisms.processing.objectivemechanism import OUTCOME
from psyneulink.globals.context import ContextFlags
from psyneulink.globals.defaults import MPI_IMPLEMENTATION, defaultControlAllocation
from psyneulink.globals.keywords import kwPreferenceSetName
from psyneulink.globals.utilities import is_numeric
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set, kpReportOutputPref
from psyneulink.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel

__all__ = [
    'CONTROL_SIGNAL_GRADIENT_ASCENT_FUNCTION', 'CONTROLLER',
    'LVOCAuxiliaryError', 'LVOCAuxiliaryFunction',
    'kwEVCAuxFunction', 'kwEVCAuxFunctionType', 'kwValueFunction',
    'OUTCOME', 'PY_MULTIPROCESSING'
]

PY_MULTIPROCESSING = False

if PY_MULTIPROCESSING:
    from multiprocessing import Pool


if MPI_IMPLEMENTATION:
    from mpi4py import MPI

kwEVCAuxFunction = "EVC AUXILIARY FUNCTION"
kwEVCAuxFunctionType = "EVC AUXILIARY FUNCTION TYPE"
kwValueFunction = "EVC VALUE FUNCTION"
CONTROL_SIGNAL_GRADIENT_ASCENT_FUNCTION = "LVOC CONTROL SIGNAL GRADIENT ASCENT FUNCTION"
CONTROLLER = 'controller'


class LVOCAuxiliaryError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class LVOCAuxiliaryFunction(Function_Base):
    """Base class for EVC auxiliary functions
    """
    componentType = kwEVCAuxFunctionType

    class ClassDefaults(Function_Base.ClassDefaults):
        variable = None

    classPreferences = {
        kwPreferenceSetName: 'ValueFunctionCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
       }

    @tc.typecheck
    def __init__(self,
                 function,
                 variable=None,
                 params=None,
                 owner=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(params=params)
        self.aux_function = function

        super().__init__(default_variable=variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context,
                         function=function,
                         )


class ControlSignalGradientAscent(LVOCAuxiliaryFunction):
    """Use gradient ascent to determine allocation_policy with the maximum `EVC <LVOCControlMechanism_LVOC>`.

    This is the default `function <LVOCControlMechanism.function>` for an LVOCControlMechanism. It identifies the
    `allocation_policy` with the maximum `EVC <EVCControlMechanism_EVC>` as follows:

    - updates the distributions of weights for the prediction_vector using BayesGLM()
    - draws a sample from the new weight distributions
    - calls gradient_ascent to determine the allocation_policy with the maximum EVC for the new weights

     by conducting gradient ascent over
    `allocation_policies <LVOCControlMechanism.allocation_policies>`

    The ControlSignalGradientAscent function returns the `allocation_policy` that yields the maximum EVC.

    """

    componentName = CONTROL_SIGNAL_GRADIENT_ASCENT_FUNCTION

    tc.typecheck
    def __init__(self,
                 default_variable=None,
                 params=None,
                 prediction_weights_priors:is_numeric=0.0,
                 prediction_variances_priors:is_numeric=1.0,
                 convergence_criterion:tc.any(int,float)=.001,
                 max_iterations:int=1000,
                 udpate_rate:tc.any(int,float) = 0.01,
                 function=None,
                 owner=None):
        function = function or self.function
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(prediction_weights_priors=prediction_weights_priors,
                                                  prediction_variances_priors=prediction_variances_priors,
                                                  convergence_criterion=convergence_criterion,
                                                  max_iterations=max_iterations,
                                                  udpate_rate=udpate_rate,
                                                  params=params)
        super().__init__(function=function,
                         owner=owner,
                         context=ContextFlags.CONSTRUCTOR)

    def function(self,
                 controller=None,
                 variable=None,
                 runtime_params=None,
                 params=None,
                 context=None):
        """Gradient ascent over allocation_policies (combinations of control_signals) to find one that maximizes EVC

        variable should have two items:  current prediction_vector and outcome
        """

        if (self.context.initialization_status == ContextFlags.INITIALIZING or
                self.owner.context.initialization_status == ContextFlags.INITIALIZING):
            return defaultControlAllocation

        if controller is None:
            raise LVOCAuxiliaryError("Call to ControlSignalGradientAscent() missing controller argument")

        predictors = variable[0]
        outcome = variable[1]

        # Initialize attributes
        if not hasattr(self, 'prediction_vector'):

            # Numbers of terms in prediction_vector
            self.num_predictors = len(predictors)
            self.num_control_signals = self.num_costs = len(controller.control_signals)
            self.num_interactions = self.num_predictors * self.num_control_signals
            len_prediction_vector = self.num_predictors + self.num_interactions + self.num_control_signals + self.num_costs

            # Indices for fields of prediction_vector
            self.pred = slice(0, self.num_predictors)
            self.intrxn= slice(self.num_predictors, self.num_predictors+self.num_interactions)
            self.ctl = slice(self.intrxn.stop, self.intrxn.stop + self.num_control_signals)
            self.cst = slice(self.ctl.stop, len_prediction_vector)

            self.prediction_vector = np.zeros(len_prediction_vector)

            update_weight = BayesGLM(num_predictors=len(self.prediction_vector),
                                     mu_prior=self.prediction_weights_priors,
                                     sigma_prior=self.prediction_variances_priors)

        # Populate fields (subvectors) of prediction_vector
        self.prediction_vector[self.pred] = np.array(predictors)
        self.prediction_vector[self.ctl] = np.array([c.value for c in controller.control_signals]).reshape(-1)
        self.prediction_vector[self.intrxn]= \
            np.array(self.prediction_vector[self.pred] *
                     self.prediction_vector[self.ctl].reshape(self.num_control_signals,1)).reshape(-1)
        self.prediction_vector[self.cst] = \
            np.array([0 if c.cost is None else c.cost for c in controller.control_signals]).reshape(-1) * -1
        # FIX: VALIDATE THAT FIELDS OF prediction_vector HAVE BEEN UPDATED

        # Get sample of weights:
        update_weight.function([np.atleast_2d(self.prediction_vector), np.atleast_2d(outcome)])
        prediction_weights = update_weight.sample_weights()

        # Compute allocation_policy using gradient_ascent
        allocation_policy = self.gradient_ascent(controller.control_signals,
                                                      self.prediction_vector,
                                                      prediction_weights)

        return allocation_policy

    def gradient_ascent(self, control_signals, prediction_vector, prediction_weights):
        '''Determine next set of ControlSignal values, compute their costs, and update prediction_vector with both

        Iterate over prediction_vector, for each iteration:
            - updating control_signal, control_signal x predictor and control_cost terms
            - multiplying the vector by the prediction weights
            - computing the sum and gradients
          - continue to iterate until sum asymptotes
          - return allocation_policy and full prediction_vector
        '''

        convergence_metric = self.convergence_criterion + EPSILON
        previous_lvoc = np.finfo(np.float128).max

        predictors = prediction_vector[0:self.num_predictors]

        # Get interaction weights and reshape so that there is one row per control_signal
        #    containing the terms for the interaction of that control_signal with each of the predictors
        interaction_weights = prediction_weights[self.intrxn].reshape(self.num_control_signals,self.num_predictors)
        # multiply interactions terms by predictors (since those don't change during the gradient ascent)
        interaction_weights_x_predictors = interaction_weights * predictors

        control_signal_values = prediction_vector[self.ctl]
        control_signal_weights = prediction_weights[self.ctl]

        gradient_constants = np.zeros(self.num_control_signals)
        for i in range(self.num_control_signals):
            gradient_constants[i] = control_signal_weights[i]
            gradient_constants[i] += np.sum(interaction_weights_x_predictors[i])

        costs = prediction_vector[self.cst]
        cost_weights = prediction_weights[self.cst]

        # TEST PRINT:
        print('\n\npredictors: ', predictors,
              '\ncontrol_signals: ', control_signal_values,
              '\ncontrol_costs: ', costs,
              '\nprediction_weights: ', prediction_weights)
        # TEST PRINT END:

        # Perform gradient ascent until convergence criterion is reached
        j=0
        while convergence_metric > self.convergence_criterion:
            # initialize gradient arrray (one gradient for each control signal)
            gradient = np.copy(gradient_constants)
            cost_gradient = np.zeros(self.num_costs)

            for i, control_signal_value in enumerate(control_signal_values):

                # Recompute costs and add to gradient
                cost_function_derivative = control_signals[i].intensity_cost_function.__self__.derivative
                cost_gradient[i] = -(cost_function_derivative(control_signal_value) * cost_weights[i])
                gradient[i] += cost_gradient[i]

                # Update control_signal_value with gradient
                control_signal_values[i] = control_signal_value + self.udpate_rate * gradient[i]

                # Update cost based on new control_signal_value
                costs[i] = -(control_signals[i].intensity_cost_function(control_signal_value))

            # Assign new values of interaction terms, control_signals and costs to prediction_vector
            prediction_vector[self.intrxn]= np.array(prediction_vector[self.pred] *
                                                     prediction_vector[self.ctl].reshape(self.num_control_signals,1)).\
                                                     reshape(-1)
            prediction_vector[self.ctl] = control_signal_values
            prediction_vector[self.cst] = costs

            # Compute current LVOC using current features, weights and new control signals
            current_lvoc = self.compute_lvoc(prediction_vector, prediction_weights)

            # Compute convergence metric with updated control signals
            convergence_metric = np.abs(current_lvoc - previous_lvoc)

            # TEST PRINT:
            print('\niteration ', j,
                  '\nprevious_lvoc: ', previous_lvoc,
                  '\ncurrent_lvoc: ',current_lvoc ,
                  '\nconvergence_metric: ',convergence_metric,
                  '\npredictors: ', predictors,
                  '\ncontrol_signal_values: ', control_signal_values,
                  '\ninteractions: ', interaction_weights_x_predictors,
                  '\ncosts: ', costs)
            # TEST PRINT END

            j+=1
            if j > self.max_iterations:
                warnings.warn("{} failed to converge after {} iterations").format(self.name, self.max_iterations)
                break

            previous_lvoc = current_lvoc

        return control_signal_values

    def compute_lvoc(self, v, w):
        return np.sum(v * w)
