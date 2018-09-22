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
import warnings

from psyneulink.components.functions.function import Buffer, Function_Base, BayesGLM, EPSILON
from psyneulink.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.components.mechanisms.processing.objectivemechanism import OUTCOME
from psyneulink.globals.context import ContextFlags
from psyneulink.globals.defaults import MPI_IMPLEMENTATION, defaultControlAllocation
from psyneulink.globals.keywords import \
    COMBINE_OUTCOME_AND_COST_FUNCTION, COST_FUNCTION, FUNCTION, FUNCTION_PARAMS, \
    NOISE, PREDICTION_MECHANISM, RATE, kwPreferenceSetName
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
                 convergence_criterion:tc.any(int,float)=EPSILON,
                 function=None,
                 owner=None):
        function = function or self.function
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(convergence_criterion=convergence_criterion,
                                                  params=params)
        super().__init__(function=function,
                         owner=owner,
                         context=ContextFlags.CONSTRUCTOR)

        # FIX: CONSTRUCT prediction_vector TO ACCOMODATE:
        #        PREDICTORS, PREDICTORS X CONTROL_SIGNAL VALUES, CONTROL_SIGNAL VALUES, AND CONTROL_SIGNAL_COSTS
        #
        #        ASSIGN ATTRIBUTES TO OBJECT WITH TUPLES = FIELDS (START INDEX AND NUM ITEMS) FOR EACH OF THESE

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

        # FIX: MOVE THIS TO __init__ OR _instantiate_XXX METHOD
        # num_predictors = len(np.array(variable).reshape(-1))
        self.num_predictors = len(predictors)
        self.num_control_signals = self.num_costs = len(controller.control_signals)
        self.num_interactions = self.num_predictors * self.num_control_signals
        len_prediction_vector = self.num_predictors + self.num_interactions + self.num_control_signals + self.num_costs
        # FIX: END MOVE

        prediction_vector = np.zeros(len_prediction_vector)

        # FIX: GET RID OF THESE AND REPLACE WITH APPENDS OR CONCATENATES BELOW
        # Indices for fields of prediction_vector
        self.intrxn_start = self.num_predictors
        self.intrxn_end = self.num_predictors+self.num_interactions
        self.ctl_start = self.intrxn_end
        self.ctl_end = self.intrxn_end + self.num_control_signals
        self.costs_start = self.ctl_end
        self.costs_end = len_prediction_vector

        initial_ctl_sig_values = [c.value for c in controller.control_signals]
        initial_ctl_sig_costs = [0 if c.cost is None else c.cost for c in controller.control_signals]

        # Populate prediction_vector
        prediction_vector[0:num_predictors] = predictors
        # Ineractions: [c1*p1, c1*p2, c1*p3... c2*p1, c2*p2...]
        interactions = (np.array(predictors*initial_ctl_sig_values).reshape(self.num_interactions,1)).reshape(-1)
        prediction_vector[self.intrxn_start:self.intrxn_end]=interactions
        # Initialize gradient_ascent with latest ControlSignal values and costs to
        prediction_vector[self.ctl_start:self.ctl_end] = initial_ctl_sig_values
        prediction_vector[self.costs_start:self.costs_end] = initial_ctl_sig_costs

        # FIX: SHOULD DO THIS IN __init__ (BUT NEED TO KNOW num_predictors)
        # Get sample of weights:
        update_weight = BayesGLM(num_predictors=len_prediction_vector,
                                 mu_prior=controller.prediction_weights_priors,
                                 sigma_prior=controller.prediction_variances_priors)

        update_weight.function([np.atleast_2d(prediction_vector), np.atleast_2d(outcome)])
        prediction_weights = update_weight.sample_weights()

        continue_ascent = True
        lvoc = self.compute_lvoc(prediction_vector, prediction_weights)
        while continue_ascent :
            # Update control_signal_values, control_signal_costs and interactions
            prediction_vector = self.gradient_ascent(prediction_vector, prediction_weights)
            new_lvoc = self.compute_lvoc(prediction_vector)
            continue_ascent = new_lvoc-lvoc > self.convergence_criterion

        allocation_policy = prediction_vector[self.ctl_start:self.ctl_end]

        return allocation_policy

    def compute_lvoc(v, w):
        return np.sum(v * w)

    # FIX: DO GRADIENT ASCENT HERE:
    # - iterate over prediction_vector, for each iteration:
    #    - updating control_signal, control_signal x predictor and control_cost terms
    #    - multiplying the vector by the prediction weights
    #    - computing the sum and gradients
    # - continue to iterate until sum asymptotes
    # - return allocation_policy and full prediction_vector
    def gradient_ascent(self, prediction_vector, prediction_weights):
        # Detertermine next set of ControlSignal values, compute their costs, and update prediction_vector with both

        # # FIX: REPLACE WITH PROPER GRADIENT ASCENT COMPUTATION:
        # new_control_signal_values = initial_ctl_sig_values
        # new_control_signal_costs = initial_ctl_sig_costs
        # # FIX: END REPLACE

        # Assign new values
        predictors = prediction_vector[0:self.num_predictors]
        interactions = (np.array(predictors*new_control_signal_values).reshape(self.num_interactions,1)).reshape(-1)
        prediction_vector[self.intrxn_start:self.intrxn_end] = interactions
        prediction_vector[self.ctl_start:self.ctl_end] = new_control_signal_values
        prediction_vector[self.costs_start:self.costs_end] = new_control_signal_costs

        ---------------

        # GRADIENCE ASCENT  FROM SEBASTIAN
        # parameters
        convergenceCriterion = some small value, e.g. 0.001
        maximumIterations = 1000
        updateRate = 0.01
        # initial values
        convergenceMetric = some large value # this metric is computed every iteration
        previousLVOC = 0
        control_signal_vector = array with control signal intensities of previous trial
        # perform gradient ascent until convergence criterion is reached
        while (convergenceMetric > convergenceCriterion) {
            # initialize gradient
            # there is a separate gradient for each control signal
            gradient = zeros(1, num_control_signals);
            for each control_signal_scalar in control_signal_vector {
                # compute gradient based on feature-control interaction terms.
                for each regressor in control_signal_regressors {
                    # get feature of feature-control interaction term
                    feature = regressor(1) # for terms that only include the control signal, the feature is always set to 1
                    # get sampled weight of feature-control interaction term
                    weight = regressor(2)
                    # compute gradient for that interaction term with respect to control signal
                    gradient += feature * weight
                }
            # compute gradient for control cost term
            # assuming that control costs are of the form exp(a * control_signal_scalar + b) where a and b are parameters of the cost function
            gradient += a * exp(a * control_signal_scalar + b)
            # update control signal with gradient
            control_signal_vector(indexing current control signal) = control_signal_scalar + updateRate * gradient
            }
            # FIX: return control_signal_vector
            # compute convergence metric with updated control signals
            currentLVOC = compute LVOC using current features, weights and new control signals
            convergenceMetric = currentLVOC - previousLVOC
            currentLVOC = previousLVOC
}
