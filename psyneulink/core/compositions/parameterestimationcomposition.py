# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *************************************  CompositionFunctionApproximator ***********************************************

"""

Contents
--------

  * `ParameterEstimationComposition_Overview`
  * `ParameterEstimationComposition_Supported_Optimzers`
  * `ParameterEstimationComposition_Class_Reference`


.. _ParameterEstimationComposition_Overview:

Overview
--------

A `ParameterEstimationComposition` is a subclass of `Composition` that is used to estimate parameters of
another `Composition` required to satisfy some objective function. The latter can be a fit to a set of empirical
data, using an external optimizer ("estimation"), or an optimization function that specifies some other conditions
that the results of the composition should satisfy when run on the inputs ("optimization").  The **target** argument
ParameterEstimationComposition's constructor is used to specify the Composition for which the parameters are to be
estimated; its **data** argument is used to specify the inputs over which the parameters should be estimated,
its **parameters** argument is used to specify the parameters of the **target** Composition to be estimated,
and its **optimization_function** is used to specify either an optimizer in another supported Python package (see
`ParameterEstimationComposition_Supported_Optimizers`), or a PsyNeuLink `OptimizationFunction`.  The optimized set
of parameters are returned when executing the  ParameterEstimationComposition`s `run <Composition.run>` method,
and stored in its `results <ParameterEstimationComposition.results>` attribute.

COMMENT:
 (which are passed to the `OptimizationControlMechanism`\\'s **state_feature_values** argument), 
the parameters of the  

its **optimization_function** are provided

If it is used for "estimation", then an external
optimizer   
 and either an external optimizer as its 
**optimization** 
COMMENT

.. _ParameterEstimationComposition_Class_Reference:

Class Reference
---------------

"""

from psyneulink.core.compositions.composition import Composition
from psyneulink.core.globals.context import Context

__all__ = ['ParameterEstimationComposition']


class ParameterEstimationCompositionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class ParameterEstimationComposition(Composition):
    """Subclass of `Composition` that estimates specified parameters of a target Composition to optimize a function
    over a set of data.

    Automatically implements an OptimizationControlMechanism as its `controller <Composition.controller>`,
    that is constructed using arguments to the ParameterEstimationComposition's constructor as described below.

    assigns
    its arguments using those
    the Composition for which the parameters are to be optimized, specified in its
    **target** argument, as the OptimizationControlMechanism' `agent_rep <OptimizationControlMechanism.agent>`;
    the

    See `Composition <Composition_Class_Reference>` for additional arguments and attributes.

    Arguments
    ---------

    target : Composition
        specifies the `Composition` for which parameters are to be estimated.

    data : array
        specifies the inputs to the `target <ParameterEstimationComposition.target>` Composition over which its
        parameters are to be estimated.

    parameters : list[Parameters]
        specifies the parameters of the `target <ParameterEstimationComposition.target>` Composition to be estimated.

    optimization_function : OptimizationFunction, function or method
        specifies the function used to estimate the parameters of the `target <ParameterEstimationComposition.target>`
        Composition.

    Attributes
    ----------

    target : Composition
        specifies the `Composition` for which parameters are to be estimated.  This is assigned as the **agent_rep**
        argument of the constructor for the ParameterEstimationComposition's `OptimizationControlMechanism`.

    data : array
        specifies the inputs to the `target <ParameterEstimationComposition.target>` Composition over which its
        parameters are to be estimated. This is assigned as the **state_feature_values** argument of the constructor
        for the ParameterEstimationComposition's `OptimizationControlMechanism`.

    parameters : list[Parameters]
        determines the parameters of the `target <ParameterEstimationComposition.target>` Composition to be estimated.
        This is assigned as the **control_signals** argument of the constructor for the ParameterEstimationComposition's
        `OptimizationControlMechanism`.

    optimization_function : OptimizationFunction, function or method
        determines the function used to estimate the parameters of the `target <ParameterEstimationComposition.target>`
        Composition.  This is assigned as the `function <OptimizationControlMechanism.function>` of the
        ParameterEstimationComposition's `OptimizationControlMechanism`.

    results : array
        contains the values of the `parameters <ParameterEstimationComposition.parameters>` of the
         `target <ParameterEstimationComposition.target>` Composition that best satisfy the
        `optimization_function <ParameterEstimationComposition.optimization_function>` given the `data
        <ParameterEstimationComposition.data>`.  This is the same as the final set of `control_signals
        <ControlMechanism.control_signals>` for the ParameterEstimationComposition's `OptimizationControlMechanism`.
    """

    def __init__(self, name=None, **param_defaults):
       # self.function = function
        super().__init__(name=name, **param_defaults)

    def adapt(self,
              feature_values,
              control_allocation,
              net_outcome,
              context=None):
        """Adjust parameters of `function <FunctionAppproximator.function>` to improve prediction of `target
        <FunctionAppproximator.target>` from `input <FunctionAppproximator.input>`.
        """
        raise ParameterEstimationCompositionError("Subclass of {} ({}) must implement {} method.".
                                                   format(ParameterEstimationComposition.__name__,
                                                          self.__class__.__name__, repr('adapt')))

    def evaluate(self,
                 feature_values,
                 control_allocation,
                 num_estimates,
                 num_trials_per_estimate,
                 base_context=Context(execution_id=None),
                 context=None):
        """Return `target <FunctionAppproximator.target>` predicted by `function <FunctionAppproximator.function> for
        **input**, using current set of `prediction_parameters <FunctionAppproximator.prediction_parameters>`.
        """
        # FIX: AUGMENT TO USE num_estimates and num_trials_per_estimate
        return self.function(feature_values, control_allocation, context=context)
