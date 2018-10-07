# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *************************************************  LVOCAuxiliary *****************************************************

"""
Auxiliary functions for `LVOCControlMechanism`.

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
    'CONTROL_SIGNAL_GRADIENT_ASCENT_FUNCTION', 'CONTROLLER', 'LearnAllocationPolicy',
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
    """Base class for LVOC auxiliary functions
    """
    componentType = kwEVCAuxFunctionType

    # class ClassDefaults(Function_Base.ClassDefaults):
    class Params(Function_Base.Params):
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


class LearnAllocationPolicy(LVOCAuxiliaryFunction):
    """LearnAllocationPolicy(   \
    learning_function=BayesGLM,       \
    prediction_weights_priors=0.0,    \
    prediction_variances_priors=1.0,  \
    udpate_rate=0.01,                 \
    convergence_criterion=.001,       \
    max_iterations=1000,              \
    params=None,                      \
    name=None,                        \
    prefs=None,                       \
    **kwags)

    Use **learning_function** to improve prediction of outcome of `LVOCControlMechanism's <LVOCControlMechanism>`
    `objective_mechanism <LVOCControlMechanism.objective_mechanism>` from LVOCControlMechanism's `predictors
    <LVOCControlMechanism.predictors>`, `control_signals <LVOCControlMechanism.control_signals>`, their interactions,
    and the `costs <ControlSignal.cost>` of the `control_signals <LVOCControlMechanism.control_signals>`, and then call
    `gradient_ascent <LearnAllocationPolicy.gradient_ascent>` method with updated prediction_weights to
    determine the `allocation_policy <LVOCControlMechanism.allocation_policy>` that maximizes `EVC
    <LVOCControlMechanism_EVC>`.

    This is the default `function <LVOCControlMechanism.function>` for an `LVOCControlMechanism`. It identifies the
    `allocation_policy <LVOCControlMechanism.allocation_policy>` with the maximum `EVC <EVCControlMechanism_EVC>` using
    the following procedure:

    - update the distributions (means and variances) of weights for `prediction_vector
      <LearnAllocationPolicy.prediction_vector>` using the function specified in the **learning_function**
      argument of the constructor (and assigned as the `update_prediction_weights_function
      <LearnAllocationPolicy.update_prediction_weights_function>` attribute) in order to better predict the
      outcome of the LVOCControlMechanism's `objective_mechanism <LVOCControlMechanism.objective_mechanism>`.

    - draw a sample from the new weight distributions by calling the sample method of the
      `update_prediction_weights_function <LearnAllocationPolicy.update_prediction_weights_function>`.

    - call `gradient_ascent <LearnAllocationPolicy.gradient_ascent>` to determine the allocation_policy with
      the maximum `EVC <LVOCControlMechanism_EVC>` given the new prediction weights.

    - return the `allocation_policy` that yields the maximum EVC.

    Arguments
    ---------

    learning_function : Function class : default BayesGLM
        assigned to the `update_prediction_weights_function
        <LearnAllocationPolicy.update_prediction_weights_function>` attribute and used to update distributions
        (means and variances) of weights for `prediction_vector <LearnAllocationPolicy.prediction_vector>`.
        The variable of its function must be a 2d array that contains two items -- a prediction vector and an outcome
        -- used to update the weights; and it must have a *sample* method that returns a set of prediction weights.

    prediction_weights_priors : int, float or 1d array of numbers : default 0.0
        specifies the value(s) used by `update_prediction_weights_function
        <LearnAllocationPolicy.update_prediction_weights_function>` to initialze the means of the
        prediction weights; if a single number is specified, that is used to initialize all of the means;
        if an array is specified, it must the anticipated length of the `prediction_vector
        <LearnAllocationPolicy.prediction_vector>`.

    prediction_variances_priors : int, float or 1d array of numbers : default 1.0
        specifies the value(s) used by `update_prediction_weights_function
        <LearnAllocationPolicy.update_prediction_weights_function>` to initialze the variances of the
        prediction weights; if a single number is specified, that is used to initialize all of the variances;
        if an array is specified, it must the anticipated length of the `prediction_vector
        <LearnAllocationPolicy.prediction_vector>`.

    udpate_rate : int or float : default 0.01
        specifies the amount by which the `value <ControlSignal.value>` of each `ControlSignal` in the
        `allocation_policy <LVOCControlMechanism.allocation_policy>` is modified in each iteration of the
        `gradient_ascent <LearnAllocationPolicy.gradient_ascent>` function.

    convergence_criterion : int or float : default .001
        specifies the change in estimate of the `EVC <LVOCControlMechanism_EVC>` below which the `gradient_ascent
        <LearnAllocationPolicy.gradient_ascent>` function should terminate and return an `allocation_policy
        <LVOCControlMechanism.allocation_policy>`.

    max_iterations : int : default 1000
        specifies the maximum number of iterations `gradient_ascent <LearnAllocationPolicy.gradient_ascent>`
        function is allowed to execute; if exceeded, a warning is issued and the function terminates, and returns the
        last `allocation_policy <LVOCControlMechanism.allocation_policy>` evaluated.

    Attributes
    ----------

    prediction_weights_priors : int, float or 1d array of numbers
        determines the value(s) used by `update_prediction_weights_function
        <LearnAllocationPolicy.update_prediction_weights_function>` to initialze the means of the
        prediction weights; if a single number is specified, that is used to initialize all of the means;
        if an array is specified, it must the anticipated length of the `prediction_vector
        <LearnAllocationPolicy.prediction_vector>`.

    prediction_variances_priors : int, float or 1d array
        determines the value(s) used by `update_prediction_weights_function
        <LearnAllocationPolicy.update_prediction_weights_function>` to initialze the variances of the
        prediction weights; if a single number is specified, that is used to initialize all of the variances;
        if an array is specified, it must the anticipated length of the `prediction_vector
        <LearnAllocationPolicy.prediction_vector>`.

    udpate_rate : int or float : default 0.01
        determines the amount by which the `value <ControlSignal.value>` of each `ControlSignal` in the
        `allocation_policy <LVOCControlMechanism.allocation_policy>` is modified in each iteration of the
        `gradient_ascent <LearnAllocationPolicy.gradient_ascent>` function.

    convergence_criterion : int or float
        determines the change in estimate of the `EVC <LVOCControlMechanism_EVC>` below which the `gradient_ascent
        <LearnAllocationPolicy.gradient_ascent>` function should terminate and return an `allocation_policy
        <LVOCControlMechanism.allocation_policy>`.

    max_iterations : int
        determines the maximum number of iterations `gradient_ascent <LearnAllocationPolicy.gradient_ascent>`
        function is allowed to execute; if exceeded, a warning is issued and the function terminates, and returns the
        last `allocation_policy <LVOCControlMechanism.allocation_policy>` evaluated.

    prediction_vector : ndarray
        array containing, in order, the LVOCControlMechanism's `predictor_values
        <LVOCControlMechanism.predictor_vaues>`, interaction terms with  `control_signals
        <LVOCControlMechanism.control_signals>`, the `values <ControlSignal.value>` of its `control_signals
        <LVOCControlMechanism.control_signals>`, and their `costs <ControlSignal.cost>`.

    num_predictors : int
        the number of elements in the predictors field of the `prediction_vector
        <LearnAllocationPolicy.prediction_vector>` (same as the number of items in the
        LVOCControlMechanism's `predictor_values <LVOCControlMechanism.predictor_values>` attribute).

    num_interactions : int
        the number of elements in the interactions field of the `prediction_vector
        <LearnAllocationPolicy.prediction_vector>` (= `num_predictiors
        <LearnAllocationPolicy.num_predictors>` * `num_control_signals
        <LearnAllocationPolicy.num_control_signals>`).

    num_control_signals : int
        the number of elements in the control_signals field of the `prediction_vector
        <LearnAllocationPolicy.prediction_vector>` (same as the number of ControlSignals in the
        LVOCControlMechanism's `control_signals <LVOCControlMechanism.control_signals>` attribute).

    num_costs : int
        the number of elements in the costs field of the `prediction_vector
        <LearnAllocationPolicy.prediction_vector>` (same as num_control_signals
        <LearnAllocationPolicy.num_control_signals>).

    update_prediction_weights_function : Function
        the function used to upated the prediction weights, based on the current value of the `prediction_vector
        <LearnAllocationPolicy.prediction_vector>` and the outcome received by the LVOCControlMechanism
        from its `objective_mechanism <LVOCControlMechanism.objective_mechanism>`.

    """

    componentName = CONTROL_SIGNAL_GRADIENT_ASCENT_FUNCTION

    tc.typecheck
    def __init__(self,
                 default_variable=None,
                 params=None,
                 learning_function=BayesGLM,
                 prediction_weights_priors:is_numeric=0.0,
                 prediction_variances_priors:is_numeric=1.0,
                 udpate_rate:tc.any(int,float) = 0.01,
                 convergence_criterion:tc.any(int,float)=.001,
                 max_iterations:int=1000,
                 function=None,
                 owner=None):

        function = function or self.function
        self.learning_function = learning_function

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(prediction_weights_priors=prediction_weights_priors,
                                                  prediction_variances_priors=prediction_variances_priors,
                                                  udpate_rate=udpate_rate,
                                                  convergence_criterion=convergence_criterion,
                                                  max_iterations=max_iterations,
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
        """Update prediction_weights to better predct outcome of `LVOCControlMechanism's <LVOCControlMechanism>`
        `objective_mechanism <LVOCControlMechanism.objective_mechanism>` from prediction_vector, then optimize
        `allocation_policy <LVOCControlMechanism>` given new prediction_weights.

        variable should have two items:  current prediction_vector and outcome
        Call `learning_function <LearnAllocationPolicy.learning_function>` to update prediction_weights.
        Call `gradient_ascent` to optimize `allocation_policy <LVOCControlMechahism.allocation_policy>` given new
        prediction_weights.
        """

        if (self.context.initialization_status == ContextFlags.INITIALIZING or
                self.owner.context.initialization_status == ContextFlags.INITIALIZING):
            return defaultControlAllocation

        if controller is None:
            raise LVOCAuxiliaryError("Call to LearnAllocationPolicy() missing controller argument")

        predictors = variable[0]
        outcome = variable[1]

        # Initialize attributes
        # IMPLEMENTATION NOTE:  This has to happen here rather than in __init__, as it requires
        #                       the LVOCControlMechanism to have fully instantiated its predictors
        #                       (the values of which are passed in here as variable[0]),
        #                       which is not the case when LearnAllocationPolicy is initialized.
        if not hasattr(self, 'prediction_vector'):

            # Numbers of terms in prediction_vector
            self.num_predictors = len(predictors)
            self.num_control_signals = self.num_costs = len(controller.control_signals)
            self.num_interactions = self.num_predictors * self.num_control_signals
            len_prediction_vector = \
                self.num_predictors + self.num_interactions + self.num_control_signals + self.num_costs

            # Indices for fields of prediction_vector
            self.pred = slice(0, self.num_predictors)
            self.intrxn= slice(self.num_predictors, self.num_predictors+self.num_interactions)
            self.ctl = slice(self.intrxn.stop, self.intrxn.stop + self.num_control_signals)
            self.cst = slice(self.ctl.stop, len_prediction_vector)

            self.prediction_vector = np.zeros(len_prediction_vector)

            # FIX: INSTEAD OF INITIALZING FUNCTION HERE, RE-INITIALIZE ITS VARIABLE TO PROPER SIZE
            #      OR MAYBE JUST CALL FUNCTION WITH "CONTEXT = INITIALIZING" SO THAT *ITS* FUNCTION CAN DO THE SIZING

            if isinstance(self.learning_function, type):
                self.learning_function = self.learning_function(
                        num_predictors=len(self.prediction_vector),
                        mu_prior=self.prediction_weights_priors,
                        sigma_prior=self.prediction_variances_priors
                )

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
        self.update_prediction_weights_function.function(
                [np.atleast_2d(self.prediction_vector), np.atleast_2d(outcome)]
        )
        prediction_weights = self.update_prediction_weights_function.sample_weights()

        # Compute allocation_policy using gradient_ascent
        allocation_policy = self.gradient_ascent(controller.control_signals,
                                                      self.prediction_vector,
                                                      prediction_weights)

        return allocation_policy

    def gradient_ascent(self, control_signals, prediction_vector, prediction_weights):
        '''Determine the `allocation_policy <LVOCControlMechanism.allocation_policy>` that maximizes the `EVC
        <LVOCControlMechanism_EVC>`.

        Iterate over prediction_vector; for each iteration: \n
        - compute gradients based on current control_signal values and their costs (in prediction_vector);
        - compute new control_signal values based on gradients;
        - update prediction_vector with new control_signal values and the interaction terms and costs based on those;
        - use prediction_weights and updated prediction_vector to compute new `EVC <LVOCControlMechanism_EVC>`.

        Continue to iterate until difference between new and old EVC is less than `convergence_criterion
        <LearnAllocationPolicy.convergence_criterion>` or number of iterations exceeds `max_iterations
        <LearnAllocationPolicy.max_iterations>`.

        Return control_signals field of prediction_vector (used by LVOCControlMechanism as its `allocation_vector
        <LVOCControlMechanism.allocation_policy>`).

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
        # print('\n\npredictors: ', predictors,
        #       '\ncontrol_signals: ', control_signal_values,
        #       '\ncontrol_costs: ', costs,
        #       '\nprediction_weights: ', prediction_weights)
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
            # print('\niteration ', j,
            #       '\nprevious_lvoc: ', previous_lvoc,
            #       '\ncurrent_lvoc: ',current_lvoc ,
            #       '\nconvergence_metric: ',convergence_metric,
            #       '\npredictors: ', predictors,
            #       '\ncontrol_signal_values: ', control_signal_values,
            #       '\ninteractions: ', interaction_weights_x_predictors,
            #       '\ncosts: ', costs)
            # TEST PRINT END

            j+=1
            if j > self.max_iterations:
                warnings.warn("{} failed to converge after {} iterations".format(self.name, self.max_iterations))
                break

            previous_lvoc = current_lvoc

        return control_signal_values

    def compute_lvoc(self, v, w):
        return np.sum(v * w)
