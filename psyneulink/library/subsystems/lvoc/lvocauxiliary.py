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
    'AVERAGE_INPUTS', 'CONTROL_SIGNAL_GRADIENT_ASCENT_FUNCTION', 'CONTROLLER',
    'LVOCAuxiliaryError', 'LVOCAuxiliaryFunction', 'WINDOW_SIZE',
    'kwEVCAuxFunction', 'kwEVCAuxFunctionType', 'kwValueFunction',
    'INPUT_SEQUENCE', 'OUTCOME', 'PY_MULTIPROCESSING',
    'TIME_AVERAGE_INPUT', 'FILTER_FUNCTION'
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
    `allocation_policy` with the maximum `EVC <EVCControlMechanism_EVC>` by a conducting gradient ascent over
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
        num_predictors = len(predictors)
        num_control_signals = num_costs = len(controller.control_signals)
        num_interactions = num_predictors * num_control_signals
        len_prediction_vector = num_predictors + num_interactions + num_control_signals + num_costs
        # FIX: END MOVE

        prediction_vector = np.zeros(len_prediction_vector)

        # FIX: GET RID OF THESE AND REPLACE WITH APPENDS OR CONCATENATES BELOW
        # Indices for fields of prediction_vector
        intrxn_start = num_predictors
        intrxn_end = num_predictors+num_interactions
        ctl_start = intrxn_end
        ctl_end = intrxn_end + num_control_signals
        costs_start = ctl_end
        costs_end = num_predictors

        initial_ctl_sig_values = [c.value for c in controller.control_signals]
        initial_ctl_sig_costs = [0 if c.cost_options is None else c.cost for c in controller.control_signals]

        # Populate prediction_vector
        prediction_vector[0:num_predictors] = predictors
        # Ineractions: [c1*p1, c1*p2, c1*p3... c2*p1, c2*p2...]
        interactions = (np.array(predictors*initial_ctl_sig_values).reshape(num_control_signals,1)).reshape(-1)
        prediction_vector[intrxn_start:intrxn_end]=interactions
        # Initialize gradient_ascent with latest ControlSignal values and costs to
        prediction_vector[ctl_start:ctl_end] = initial_ctl_sig_values
        prediction_vector[costs_start:costs_end] = initial_ctl_sig_costs

        # FIX: REMOVE WHEN VALIDATED:
        assert len(prediction_vector[0:num_predictors]) == len(predictors)
        assert len(prediction_vector[intrxn_start:intrxn_end]) == len(interactions)
        assert len(prediction_vector[ctl_start:ctl_end]) == len(controller.control_signals)
        assert len(prediction_vector[costs_start:costs_end]) == len(controller.control_signals)

        # FIX: DO GRADIENT ASCENT HERE:
        # - iterate over prediction_vector, for each iteration:
        #    - updating control_signal, control_signal x predictor and control_cost terms
        #    - multiplying the vector by the prediction weights
        #    - computing the sum and gradients
        # - continue to iterate until sum asymptotes
        # - return allocation_policy and full prediction_vector
        def gradient_ascent():
            # Detertermine next set of ControlSignal values, compute their costs, and update prediction_vector with both
            # FIX: REPLACE WITH PROPER GRADIENT ASCENT COMPUTATION:
            new_control_signal_values = initial_ctl_sig_values
            new_control_signal_costs = initial_ctl_sig_costs
            # FIX: END REPLACE
            interactions = (np.array(predictors*new_control_signal_values).reshape(num_control_signals,1)).reshape(-1)
            prediction_vector[intrxn_start:intrxn_end] = interactions
            prediction_vector[ctl_start:ctl_end] = new_control_signal_values
            prediction_vector[costs_start:costs_end] = new_control_signal_costs

        def compute_lvoc():
            return np.sum(prediction_vector * prediction_weights)
        continue_ascent = True
        lvoc = compute_lvoc()
        while continue_ascent :
            # Update control_signal_values, control_signal_costs and interactions
            gradient_ascent()
            new_lvoc = compute_lvoc()
            continue_ascent = new_lvoc-lvoc > self.convergence_criterion

        allocation_policy = prediction_vector[ctl_start:ctl_end]

        return allocation_policy


class UpdateWeights(LVOCAuxiliaryFunction):
    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 num_predictors:int=1,
                 prediction_weights_priors:tc.optional(is_numeric)=None,
                 prediction_variances_priors:tc.optional(is_numeric)=None,
                 # function:tc.optional(is_callable)=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs):

        function = BayesGLM(num_predictors=num_predictors,
                            mu_prior=prediction_weights_priors,
                            sigma_prior=prediction_variances_priors)

        super().__init__(default_variable=default_variable,
                         size=size,
                         function=function,
                         params=params,
                         name=name,
                         prefs=prefs,
                         **kwargs,
                         context=ContextFlags.CONSTRUCTOR)

    def _execute(self, variable=None, runtime_params=None, context=None):
        '''Execute function to get outcome, call _predictor_update_function to update wts, and return sample of wts'''
        super()._execute(variable=variable, runtime_params=runtime_params, context=context)
        return self.function.sample_weights()

    def _parse_function_variable(self, variable, context=None):
        dependent_vars = np.atleast_2d(variable)
        prediction_vector = np.atleast_2d(variable)
        return prediction_vector, dependent_vars




AVERAGE_INPUTS = 'AVERAGE_INPUTS'
INPUT_SEQUENCE = 'INPUT_SEQUENCE'
TIME_AVERAGE_INPUT = 'TIME_AVERAGE_INPUT'
input_types = {TIME_AVERAGE_INPUT, AVERAGE_INPUTS, INPUT_SEQUENCE}

WINDOW_SIZE = 'window_size'
FILTER_FUNCTION = 'filter_function'
