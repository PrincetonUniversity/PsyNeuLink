# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************  GymForagerCFA ***************************************************

"""

Contents
--------

  * `GymForagerCFA_Overview`
  * `GymForagerCFA_Class_Reference`


.. GymForagerCFA_Overview:

Overview
--------

A `GymForagerCFA` is a subclass of `RegressionCFA` that uses the `gym-forager
<https://github.com/IntelPNI/gym-forager>`_ as an agent to predict the `net_outcome <ControlMechanism.net_outcome>`
for a `Composition` (or part of one) controlled by an `OptimizationControlMechanism`.

It instantiates an agent with an interface to the gym-forager envirnoment and a specified action-space.

At present its `adapt <CompositionFunctionApproximator.adapt>` method is not implemented.

Its `evaluate <CompositionFunctionApproximator.evaluate>` method calls the DQN to generate an action, and then
calls the gym-forager agent with a specified action and returns the reward associated with that action.


    Parameterizes weights of a `update_weights <RegressorCFA.update_weights>` used by its `evaluate
    <CompositionFunctionApproximator.evaluate>` method to predict the `net_outcome <ControlMechanism.net_outcome>`
    for a `Composition` (or part of one) controlled by an `OptimiziationControlMechanism`, from a set of `feature_values
    <OptimizationControlMechanism.feature_values>` and a `control_allocation <ControlMechanism.control_allocation>`
    provided by the OptimiziationControlMechanism.

    The `feature_values <OptimiziationControlMechanism.feature_values>` and `control_allocation
    <ControlMechanism.control_allocation>` passed to the RegressorCFA's `adapt <RegressorCFA.adapt>` method,
    and provided as the input to its `update_weights <RegressorCFA.update_weights>`, are represented in the
    `vector <PredictionVector.vector>` attribute of a `PredictionVector` assigned to the RegressorCFA`s
    `prediction_vector <RegressorCFA.prediction_vector>` attribute.  The  `feature_values
    <OptimizationControlMechanism.feature_values>` are assigned to the features field of the
    `prediction_vector <RegressorCFA.prediction_vector>`, and the `control_allocation
    <ControlMechanism_control_allocation>` is assigned to the control_allocation field of the `prediction_vector
    <RegressorCFA.prediction_vector>`.  The `prediction_vector <RegressorCFA.prediction_vector>` may also contain
    fields for the `costs ControlMechanism.costs` associated with the `control_allocation
    <ControlMechanism.control_allocation>` and for interactions among those terms.

    The `regression_weights <RegressorCFA.regression_weights>` returned by the `update_weights
    <RegressorCFA.update_weights>` are used by the RegressorCFA's `evaluate <RegressorCFA.evaluate>` method to
    predict the `net_outcome <ControlMechanism.net_outcome>` from the
    `prediction_vector <RegressorCFA.prediction_vector>`.



COMMENT:
.. note::
  The RegressionCFA's `update_weights <RegressionCFA.update_weights>` is provided the `feature_values
  <OptimizationControlMechanism.feature_values>` and `net_outcome <ControlMechanism.net_outcome>` from the
  *previous* trial to update its parameters.  Those are then used to determine the `control_allocation
  <ControlMechanism.control_allocation>` predicted to yield the greatest `EVC <OptimizationControlMechanism_EVC>`
  based on the `feature_values <OptimizationControlMechanism.feature_values>` for the current trial.
COMMENT

.. _GymForagerCFA_Class_Reference:

Class Reference
---------------

"""

import numpy as np
import typecheck as tc

import gym_forager

from psyneulink.library.compositions.regressioncfa import RegressionCFA
from psyneulink.core.components.functions.learningfunctions import BayesGLM
from psyneulink.core.globals.keywords import DEFAULT_VARIABLE
from psyneulink.core.globals.parameters import Parameter

__all__ = ['GymForagerCFA']


class GymForagerCFAError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class GymForagerCFA(RegressionCFA):
    """
    GymForagerCFA(
        name=None,
        update_weights=BayesGLM,
        prediction_terms=None)

    Subclass of `RegressionCFA` that implements a CompositionFunctionApproximator as the
    `agent_rep <OptimizationControlmechanism.agent>` of an `OptimizationControlmechanism`.

    See `RegressionCFA <rRegressionCFA_Class_Reference>` for arguments and attributes.

    """

    class Parameters(RegressionCFA.Parameters):
        update_weights = Parameter(BayesGLM, stateful=False, loggable=False)

    def __init__(self,
                 name=None,
                 update_weights=BayesGLM,
                 prediction_terms:tc.optional(list)=None):

        self.update_weights = update_weights
        self._instantiate_prediction_terms(prediction_terms)

        super().__init__(name=name, update_weights=update_weights)


    def initialize(self, features_array, control_signals):
        """Assign owner and instantiate `prediction_vector <RegressorCFA.prediction_vector>`

        Must be called before RegressorCFA's methods can be used.
        """

        prediction_terms = self.prediction_terms
        self.prediction_vector = self.PredictionVector(features_array, control_signals, prediction_terms)

        # Assign parameters to update_weights
        update_weights_default_variable = [self.prediction_vector.vector, np.zeros(1)]
        if isinstance(self.update_weights, type):
            self.update_weights = \
                self.update_weights(default_variable=update_weights_default_variable)
        else:
            self.update_weights.reset({DEFAULT_VARIABLE: update_weights_default_variable})

    def adapt(self, feature_values, control_allocation, net_outcome, context=None):
        """Update `regression_weights <RegressorCFA.regression_weights>` so as to improve prediction of
        **net_outcome** from **feature_values** and **control_allocation**.
        """
        prediction_vector = self.parameters.prediction_vector._get(context)
        previous_state = self.parameters.previous_state._get(context)

        if previous_state is not None:
            # Update regression_weights
            regression_weights = self.update_weights([previous_state, net_outcome], context=context)
            # Update vector with current feature_values and control_allocation and store for next trial
            prediction_vector.update_vector(control_allocation, feature_values, context)
            previous_state = prediction_vector.vector
        else:
            # Initialize vector and control_signals on first trial
            # Note:  initialize vector to 1's so that learning_function returns specified priors
            # FIX: 11/9/19 LOCALLY MANAGE STATEFULNESS OF ControlSignals AND costs
            # prediction_vector.reference_variable = control_allocation
            previous_state = np.full_like(prediction_vector.vector, 0)
            regression_weights = self.update_weights([previous_state, 0], context=context)

        self._set_multiple_parameter_values(
            context,
            previous_state=previous_state,
            prediction_vector=prediction_vector,
            regression_weights=regression_weights,
        )

    # FIX: RENAME AS _EXECUTE_AS_REP ONCE SAME IS DONE FOR COMPOSITION
    # def evaluate(self, control_allocation, num_samples, reset_stateful_functions_to, feature_values, context):
    def evaluate(self, feature_values, control_allocation, num_estimates, context):
        """Update prediction_vector <RegressorCFA.prediction_vector>`,
        then multiply by regression_weights.

        Uses the current values of `regression_weights <RegressorCFA.regression_weights>` together with
        values of **control_allocation** and **feature_values** arguments to generate predicted `net_outcome
        <OptimiziationControlMechanism.net_outcome>`.

        .. note::
            If this method is assigned as the `objective_funtion of a `GradientOptimization` `Function`,
            it is differentiated using `autograd <https://github.com/HIPS/autograd>`_\\.grad().
        """
        predicted_outcome=0

        prediction_vector = self.parameters.prediction_vector._get(context)

        count = num_estimates if num_estimates else 1
        for i in range(count):
            terms = self.prediction_terms
            vector = prediction_vector.compute_terms(control_allocation, context=context)
            # FIX: THIS SHOULD GET A SAMPLE RATHER THAN JUST USE THE ONE RETURNED FROM ADAPT METHOD
            #      OR SHOULD MULTIPLE SAMPLES BE DRAWN AND AVERAGED AT END OF ADAPT METHOD?
            #      I.E., AVERAGE WEIGHTS AND THEN OPTIMIZE OR OTPIMZE FOR EACH SAMPLE OF WEIGHTS AND THEN AVERAGE?

            weights = self.parameters.regression_weights._get(context)
            net_outcome = 0

            for term_label, term_value in vector.items():
                if term_label in terms:
                    pv_enum_val = term_label.value
                    item_idx = prediction_vector.idx[pv_enum_val]
                    net_outcome += np.sum(term_value.reshape(-1) * weights[item_idx])
            predicted_outcome+=net_outcome
        predicted_outcome/=count
        return predicted_outcome
