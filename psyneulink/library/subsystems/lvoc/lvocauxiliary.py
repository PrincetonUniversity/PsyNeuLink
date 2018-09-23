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

            self.num_predictors = len(predictors)
            self.num_control_signals = self.num_costs = len(controller.control_signals)
            self.num_interactions = self.num_predictors * self.num_control_signals
            len_prediction_vector = self.num_predictors + self.num_interactions + self.num_control_signals + self.num_costs

            self.prediction_vector = np.zeros(len_prediction_vector)

            # Indices for fields of prediction_vector
            self.intrxn_start = self.num_predictors
            self.intrxn_end = self.num_predictors+self.num_interactions
            self.ctl_start = self.intrxn_end
            self.ctl_end = self.intrxn_end + self.num_control_signals
            self.costs_start = self.ctl_end
            self.costs_end = len_prediction_vector

            update_weight = BayesGLM(num_predictors=len(self.prediction_vector),
                                     mu_prior=self.prediction_weights_priors,
                                     sigma_prior=self.prediction_variances_priors)

        # Populate prediction_vector
        self.prediction_vector[0:self.num_predictors] = predictors

        # Initialize with latest ControlSignal values and costs to
        self.prediction_vector[self.ctl_start:self.ctl_end] = [c.value for c in controller.control_signals]
        self.prediction_vector[self.costs_start:self.costs_end] = [0 if c.cost is None else c.cost
                                                                   for c in controller.control_signals]

        # Get sample of weights:
        update_weight.function([np.atleast_2d(self.prediction_vector), np.atleast_2d(outcome)])
        prediction_weights = update_weight.sample_weights()

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

        convergence_metric = np.finfo(np.float128).max # some large value; this metric is computed every iteration
        previous_lvoc = 0
        predictors = prediction_vector[0:self.num_predictors]
        # FIX: ??WHERE SHOULD THESE BE USED:
        predictor_weights = prediction_weights[0:self.num_predictors]

        control_signal_values = prediction_vector[self.ctl_start:self.ctl_end]
        control_signal_weights = prediction_weights[self.ctl_start:self.ctl_end]

        # get interaction weights and reshape so that there is one row per control_signal
        #    containing the interaction terms for that control_signal with all of the predictors
        interaction_weights = prediction_weights[self.intrxn_start:self.intrxn_end].reshape(self.num_control_signals,
                                                                                            self.num_predictors)
        interactions = np.zeros_like(interaction_weights)
        # multiply interactions terms by predictors (since those don't change during the gradient ascent)
        interaction_weights_x_predictors = interaction_weights * predictors

        costs = prediction_vector[self.costs_start:self.costs_end]
        cost_weights = prediction_weights[self.costs_start:self.costs_end]

        # TEST PRINT:
        print('\n\npredictors: ', predictors,
              '\ncontrol_signals: ', control_signal_values,
              '\ncontrol_costs: ', costs,
              '\nprediction_weights: ', prediction_weights)
        j = 0
        # TEST PRINT END:

        # perform gradient ascent until convergence criterion is reached
        while convergence_metric > self.convergence_criterion:
            # initialize gradient arrray (one gradient for each control signal)
            gradient = np.zeros(self.num_control_signals)

            # # recompute predictor-control interaction terms [c1*p1, c1*p2, c1*p3... c2*p1, c2*p2...] in each iteration
            # interactions_by_ctl_sig = np.array(predictors * control_signal_values.reshape(self.num_control_signals,1))
            # interactions = interactions_by_ctl_sig.reshape(-1)

            for i, control_signal_value in enumerate(control_signal_values):

                # FIX: ?IS THIS NEEDED:
                gradient[i] += np.sum(predictors * predictor_weights)

                # Add gradient with respect to control_signal itself
                gradient[i] += control_signal_weights[i] * control_signal_value

                # Add gradient with respect to control_signal-predictor interaction; as noted above,
                #    interaction_weights[i] is a vector of interaction terms for a control_signal with all predictors
                interactions[i] = interaction_weights_x_predictors[i] * control_signal_values[i]
                gradient[i] += np.sum(interactions)

                # FIX: ??IS THE "-" CORRECT HERE:
                # compute gradient for control cost term
                costs[i] = -(control_signals[i].intensity_cost_function(control_signal_value) * cost_weights[i])
                gradient[i] += costs[i]

                # update control signal with gradient
                control_signal_values[i] = control_signal_value + self.udpate_rate * gradient[i]

            prediction_vector[self.intrxn_start:self.intrxn_end] = interactions.reshape(-1)

            # Compute current LVOC using current features, weights and new control signals
            current_lvoc = self.compute_lvoc(prediction_vector, prediction_weights)
            # compute convergence metric with updated control signals
            convergence_metric = current_lvoc - previous_lvoc

            # TEST PRINT:
            print('\niteration ', j,
                  '\nprevious_lvoc: ', previous_lvoc,
                  '\ncurrent_lvoc: ',current_lvoc ,
                  '\nconvergence_metric: ',convergence_metric,
                  '\npredictors: ', predictors,
                  '\ncontrol_signal_values: ', control_signal_values,
                  '\ninteractions: ', interactions,
                  '\ncosts: ', costs)
            j+=1
            # TEST PRINT END

            previous_lvoc = current_lvoc

        return control_signal_values

    def compute_lvoc(self, v, w):
        return np.sum(v * w)




#         ---------------
#
#         # GRADIENCE ASCENT  FROM SEBASTIAN
#         # parameters
#         convergenceCriterion = some small value, e.g. 0.001
#         max_iterations = 1000
#         udpate_rate = 0.01
#         # initial values
#         convergenceMetric = some large value # this metric is computed every iteration
#         previousLVOC = 0
#         control_signals = array with control signal intensities of previous trial
#         # perform gradient ascent until convergence criterion is reached
#         while (convergenceMetric > convergenceCriterion) {
#             # initialize gradient
#             # there is a separate gradient for each control signal
#             gradient = zeros(1, num_control_signals);
#             for each control_signal_scalar in control_signals {
#                 # compute gradient based on feature-control interaction terms.
#                 for each regressor in control_signal_regressors {
#                     # get feature of feature-control interaction term
#                     feature = regressor(1) # for terms that only include the control signal, the feature is always set to 1
#                     # get sampled weight of feature-control interaction term
#                     weight = regressor(2)
#                     # compute gradient for that interaction term with respect to control signal
#                     gradient += feature * weight
#                 }
#             # compute gradient for control cost term
#             # assuming that control costs are of the form exp(a * control_signal_scalar + b) where a and b are parameters of the cost function
#             gradient += a * exp(a * control_signal_scalar + b)
#             # update control signal with gradient
#             control_signals(indexing current control signal) = control_signal_scalar + udpate_rate * gradient
#             }
#             # FIX: return control_signals
#             # compute convergence metric with updated control signals
#             currentLVOC = compute LVOC using current features, weights and new control signals
#             convergenceMetric = currentLVOC - previousLVOC
#             currentLVOC = previousLVOC
# }
