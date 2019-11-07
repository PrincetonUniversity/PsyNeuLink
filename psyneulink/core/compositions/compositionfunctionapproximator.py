# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *************************************  CompositionFunctionApproximator ***********************************************

# FIX: SHOULD BE IMPLEMENTED AS ABSTRACT BASE CLASS (ABC)

"""

Contents
--------

  * `CompositionFunctionApproximator_Overview`
  * `CompositionFunctionApproximator_Class_Reference`


.. _CompositionFunctionApproximator_Overview:

Overview
--------

A `CompositionFunctionApproximator` is an abstract subclass of `Composition` that, over calls to its `adapt
<CompositionFunctionApproximator.adapt>` method, parameterizes its `function <Composition.function>` to predict the
`net_outcome <ControlMechanism.net_outcome>` of the Composition (or part of one) controlled by an
`OptimizationControlMechanism`, for a given set of `feature_values <OptimizationControlMechanism.feature_values>`
and a `control_allocation <ControlMechanism.control_allocation>` provided by the OptimizationControlMechanism.
Its `evaluate <CompositionFunctionApproximator.evaluate>` method calls its `function
<CompositionFunctionApproximator.function>` to generate and return the predicted `net_outcome
<ControlMechanism.net_outcome>` for a given set of `feature_values
<OptimizationControlMechanism.feature_values>`, `control_allocation <ControlMechanism.control_allocation>` and
`num_estimates <OptimizationControlMechanism.num_estimates>`

COMMENT:
.. note::
  The CompositionFunctionApproximator's `adapt <CompositionFunctionApproximator.adapt>` method is provided the
  `feature_values <OptimizationControlMechanism.feature_values>` and `net_outcome <ControlMechanism.net_outcome>`
  from the *previous* trial to update its parameters.  Those are then used to determine the `control_allocation
  <ControlMechanism.control_allocation>` predicted to yield the greatest `EVC <OptimizationControlMechanism_EVC>`
  based on the `feature_values <OptimizationControlMechanism.feature_values>` for the current trial.
COMMENT


.. _CompositionFunctionApproximator_Class_Reference:

Class Reference
---------------

"""

from psyneulink.core.compositions.composition import Composition
from psyneulink.core.globals.context import Context

__all__ = ['CompositionFunctionApproximator']


class CompositionFunctionApproximatorError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class CompositionFunctionApproximator(Composition):
    """Subclass of `Composition` that implements a FunctionApproximator as the `agent_rep
    <OptimizationControlmechanism.agent>` of an `OptimizationControlmechanism`.

    Parameterizes `its function <CompositionFunctionApproximator.function>` to predict a `net_outcome
    <Controlmechanism.net_outcome>` for a set of `feature_values <OptimizationControlmechanism.feature_values>`
    and a `control_allocation <ControlMechanism.control_allocation>` provided by an `OptimizationControlmechanism`.

    See `Composition <Composition_Class_Reference>` for additional arguments and attributes.

    Arguments
    ---------

    param_defaults : LearningFunction, function or method
        specifies the function parameterized by the CompositionFunctionApproximator's `adapt
        <CompositionFunctionApproximator.adapt>` method, and used by its `evaluate
        <CompositionFunctionApproximator.evaluate>` method to generate and return a predicted `net_outcome
        <ControlMechanism.net_outcome>` for a set of `feature_values <OptimizationControlMechanism.feature_values>`
        and a `control_allocation <OptimizationControlMechanism>` provided by an `OptimizationControlMechanism`.

    Attributes
    ----------

    function : LearningFunction, function or method
        parameterized by the CompositionFunctionApproximator's <adapt <CompositionFunctionApproximator.adapt>`
        method, and used by its `evaluate <CompositionFunctionApproximator.evaluate>` method to generate and return
        a predicted `net_outcome <ControlMechanism.net_outcome>` for a set of `feature_values
        <OptimizationControlMechanism.feature_values>` and a `control_allocation <OptimizationControlMechanism>`
        provided by an `OptimizationControlMechanism`.

    prediction_parameters : 1d array
        parameters adjusted by `adapt <CompositionFunctionApproximator.adapt>` method, and used by `function
        <FunctionAppproximator.function>` to predict the `net_outcome <ControlMechanism.net_outcome>`
        for a given set of `feature_values <OptimizationControlMechanism.feature_values>` and `control_allocation
        <ControlMechanism.control_allocation>`.

    """

    def __init__(self, name=None, **param_defaults):
       # self.function = function
        super().__init__(name=name, **param_defaults)

    def adapt(self, feature_values, control_allocation, net_outcome, context=None):
        """Adjust parameters of `function <FunctionAppproximator.function>` to improve prediction of `target
        <FunctionAppproximator.target>` from `input <FunctionAppproximator.input>`.
        """
        raise CompositionFunctionApproximatorError("Subclass of {} ({}) must implement {} method.".
                                                   format(CompositionFunctionApproximator.__name__,
                                                          self.__class__.__name__, repr('adapt')))

    def evaluate(self, feature_values, control_allocation, num_estimates, base_context=Context(execution_id=None), context=None):
        """Return `target <FunctionAppproximator.target>` predicted by `function <FunctionAppproximator.function> for
        **input**, using current set of `prediction_parameters <FunctionAppproximator.prediction_parameters>`.
        """
        # FIX: AUGMENTTO USE num_estimates
        return self.function(feature_values, control_allocation, context=context)

    @property
    def prediction_parameters(self):
        raise CompositionFunctionApproximatorError("Subclass of {} ({}) must implement {} attribute.".
                                                   format(CompositionFunctionApproximator.__name__, self.__class__.__name__,
                                                          repr('prediction_parameters')))

    @property
    def runs_simulations(self):
        return False
