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
  * `ParameterEstimationComposition_Data_Fitting`
  * `ParameterEstimationComposition_Optimization`
  * `ParameterEstimationComposition_Supported_Optimizers`
  * `ParameterEstimationComposition_Class_Reference`


.. _ParameterEstimationComposition_Overview:

Overview
--------

A `ParameterEstimationComposition` is a subclass of `Composition` that is used to estimate parameters of
another `Composition` (its `target <ParameterEstimationComposition.target>` in order to fit that to a set of
data (`ParameterEstimationComposition_Data_Fitting`) or to optimize the `results <Composition.results>` of the
`target <ParameterEstimationComposition.target>` according to an `objective_function`
(`ParameterEstimationComposition_Optimization`). In either case, when the ParameterEstimationComposition is `run
<Composition.run>` with a given set of `inputs <Composition_Execution_Inputs>`, it returns the set of parameters
that it estimates best satisfy either of those conditions using its `optimization_function
<ParameterEstimationComposition.optimization_function>`.

.. _ParameterEstimationComposition_Data_Fitting:

Data Fitting
------------

The ParameterEstimationComposition can be used to find a set of parameters for another "target" Composition such that,
when that Composition is run with a given set of inputs, its results best match a specified set of empirical data. A
ParameterEstimationComposition can be configured for data fitting by specifying the arguments of its constructor as
follows:

    * **target** - specifies the `Composition` for which parameters are to be estimated.

    * **parameters** - list[Parameters]
        specifies the parameters of the `target <ParameterEstimationComposition.target>` Composition to be estimated.

    * **data** - specifies the inputs to the `target <ParameterEstimationComposition.target>` Composition over which its
        parameters are to be estimated.

    * **objective_function** -- ??this is automatically specified as MLE??

    * **optimization_function**  OptimizationFunction, function or method
        specifies the function used to estimate the parameters of the `target <ParameterEstimationComposition.target>`
        Composition.



   * **target** - specifies the `Composition` for which parameters are to be estimated.

   * **data** - specifies the inputs to the `target <ParameterEstimationComposition.target>` Composition over which its
        parameters are to be estimated. This is assigned as the **state_feature_values** argument of the constructor
        for the ParameterEstimationComposition's `OptimizationControlMechanism`.

    parameters : list[Parameters]
        determines the parameters of the `target <ParameterEstimationComposition.target>` Composition to be estimated.
        This is assigned as the **control_signals** argument of the constructor for the ParameterEstimationComposition's
        `OptimizationControlMechanism`.

    outcome_variables : list[Composition output nodes]
        determines the `OUTPUT` `Nodes <Composition_Nodes>` of the `target <ParameterEstimationComposition.target>`
        Composition, the `values <Mechanism_Base.value>` of which are either compared to the **data** when the
        ParameterEstimationComposition is used for `data fitting <ParameterEstimationComposition_Data_Fitting>`,
        or evaluated by the ParameterEstimationComposition's `optimization_function
        <ParameterEstimationComposition.optimization_function>` when it is used for `parameter optimization
        <ParameterEstimationComposition_Optimization>`.

    optimization_function : OptimizationFunction, function or method
        determines the function used to estimate the parameters of the `target <ParameterEstimationComposition.target>`
        Composition.  This is assigned as the `function <OptimizationControlMechanism.function>` of the
        ParameterEstimationComposition's `OptimizationControlMechanism`.


    optimized_parameters : list
        contains the values of the `parameters <ParameterEstimationComposition.parameters>` of the
         `target <ParameterEstimationComposition.target>` Composition that best fit the **data** when the
         ParameterEstimationComposition is used for `data fitting <ParameterEstimationComposition_Data_Fitting>`,
         or that optimize performance of the `target <ParameterEstimationComposition.target>` according to the
         `optimization_function <ParameterEstimationComposition.optimization_function>` when the
         ParameterEstimationComposition is used for `parameter `optimization_function
         <ParameterEstimationComposition.optimization_function>`.  This is the same as the final set of `values
         <ControlSignal.value>` for the `control_signals <ControlMechanism.control_signals>` of the
         ParameterEstimationComposition's `OptimizationControlMechanism`.

    results : list[list[list]]
        containts the `output_values <Mechanism_Base.output_values>` of the `OUTPUT` `Nodes <Composition_Nodes>`
        in the ``target <ParameterEstimationComposition.target>` Composition for every `TRIAL <TimeScale.TRIAL>`
        executed (see `Composition.results` for more details).

of the `target <ParameterEstimationComposition.target>` Composition that
best fit the data

The  **target**
argument ParameterEstimationComposition's constructor is used to specify the Composition for which the parameters
are to be estimated; its **data** argument is used to specify the data to be fit for parameter *estimation*,
its **parameters** argument is used to specify the parameters of the **target** Composition to be estimated,
and its **optimization_function** is used to specify either an optimizer in another supported Python package (see
`ParameterEstimationComposition_Supported_Optimizers`), or a PsyNeuLink `OptimizationFunction`.  The optimized set
of parameters are returned when executing the  ParameterEstimationComposition`s `run <Composition.run>` method,
and stored in its `results <ParameterEstimationComposition.results>` attribute.


.. _ParameterEstimationComposition_Optimization

Parameter Optimization
----------------------


.. _ParameterEstimationComposition_Supported_Optimizers:

Supported Optimizers
--------------------

TBD


.. _ParameterEstimationComposition_Class_Reference:

Class Reference
---------------

"""

import numpy as np

from psyneulink.core.components.mechanisms.modulatory.control.optimizationcontrolmechanism import \
    OptimizationControlMechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.ports.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.compositions.composition import Composition, NodeRole
from psyneulink.core.globals.context import Context
from psyneulink.core.globals.sampleiterator import SampleSpec
from psyneulink.core.globals.utilities import convert_to_list

__all__ = ['ParameterEstimationComposition']


class ParameterEstimationCompositionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class ParameterEstimationComposition(Composition):
    """Subclass of `Composition` that estimates specified parameters of a target Composition to optimize a function
    over a set of data.

    Automatically implements an `OptimizationControlMechanism` as its `controller <Composition.controller>`,
    that is constructed using arguments to the ParameterEstimationComposition's constructor as described below.

    Arguments
    ---------

    target : Composition
        specifies the `Composition` for which parameters are to be `fit to data
        <ParameterEstimationComposition_Data_Fitting>` or `optimized <ParameterEstimationComposition_Optimization>`.

    parameters : dict[Parameter:Union[Iterator, Function, List, value]
        specifies the parameters of the `target <ParameterEstimationComposition.target>` Composition used to `fit
        it to data <ParameterEstimationComposition_Data_Fitting>` or `optimize its performance
        <ParameterEstimationComposition_Optimization>` according to the `optimization_function
        <ParameterEstimationComposition.optimization_function>`, and either the range of values to be evaluated for
        each, or priors that define a distribution.

    outcome_variables : list[Composition output nodes]
        specifies the `OUTPUT` `Nodes <Composition_Nodes>` of the `target <ParameterEstimationComposition.target>`
        Composition, the `values <Mechanism_Base.value>` of which are either compared to the **data** when the
        ParameterEstimationComposition is used for `data fitting <ParameterEstimationComposition_Data_Fitting>`
        or used by the `optimization_function <ParameterEstimationComposition.optimization_function>` when the
        ParameterEstimationComposition is used for `parameter optimization
        <ParameterEstimationComposition_Optimization>`.

    data : array : default None
        specifies the data to to be fit when the ParameterEstimationComposition is used for
        `data fitting <ParameterEstimationComposition_Data_Fitting>`.

    objective_function : ObjectiveFunction, function or method
        specifies the function that must be optimized (maximized or minimized) when the ParameterEstimationComposition
        is used for `parameter optimization <ParameterEstimationComposition_Optimization>`.

    optimization_function : OptimizationFunction, function or method
        specifies the function used to evaluate the `fit to data <ParameterEstimationComposition_Data_Fitting>` or
        `optimize <ParameterEstimationComposition_Optimization>` the parameters of the `target
        <ParameterEstimationComposition.target>` Composition.

    num_estimates : int : default 1
        specifies the number of estimates made for a each combination of `parameter <ParameterEstimationComposition>`
        values (see `num_estimates <ParameterEstimationComposition.num_estimates>` for additional information).

    initial_seed : int : default None
        specifies the seed used to initialize the random number generator at construction.
        If it is not specified then then the seed is set to a random value on construction (see `initial_seed
        <ParameterEstimationComposition.initial_seed>` for additional information).

    same_seed_for_all_parameter_combinations :  bool : default False
        specifies whether the random number generator is re-initialized to the same value when estimating each
        combination of `parameter <ParameterEstimationComposition>` values (see `constant_seed_across_search
        <ParameterEstimationComposition.constant_seed_across_search>` for additional information).


    Attributes
    ----------

    target : Composition
        determines the `Composition` for which parameters are used to `fit data
        <ParameterEstimationComposition_Data_Fitting>` or `optimize its performance
        <ParameterEstimationComposition_Optimization>` as defined by the `objective_function
        <ParameterEstimationComposition.objective_function>`.  This is assigned as the `agent_rep
        <OptimizationControlMechanism.agent_rep>` attribute of the ParameterEstimationComposition's
        `OptimizationControlMechanism`.

    parameters : list[Parameters]
        determines the parameters of the `target <ParameterEstimationComposition.target>` Composition used to `fit
        it to data <ParameterEstimationComposition_Data_Fitting>` or `optimize its performance
        <ParameterEstimationComposition_Optimization>` according to the `optimization_function
        <ParameterEstimationComposition.optimization_function>`. These are assigned to the **control** argument of
        the constructor for the ParameterEstimationComposition's `OptimizationControlMechanism`, that is used to
        construct its `control_signals <OptimizationControlMechanism.control_signals>`.

    parameter_ranges_or_priors : List[Union[Iterator, Function, ist or Value]
        determines the range of values evaluated for each `parameter <ParameterEstimationComposition.parameters>`.
        These are assigned as the `allocation_samples <ControlSignal.allocation_samples>` for the `ControlSignal`
        assigned to the ParameterEstimationComposition` `OptimizationControlMechanism` corresponding to each of the
        specified `parameters <ParameterEstimationComposition.parameters>`.

    outcome_variables : list[Composition Output Nodes]
        determines the `OUTPUT` `Nodes <Composition_Nodes>` of the `target <ParameterEstimationComposition.target>`
        Composition, the `values <Mechanism_Base.value>` of which are either compared to the **data** when the
        ParameterEstimationComposition is used for `data fitting <ParameterEstimationComposition_Data_Fitting>`,
        or evaluated by the ParameterEstimationComposition's `optimization_function
        <ParameterEstimationComposition.optimization_function>` when it is used for `parameter optimization
        <ParameterEstimationComposition_Optimization>`.

    data : array
        determines the data to be fit by the model specified by the `target <ParameterEstimationComposition.target>`
        Composition when the ParameterEstimationComposition is used for `data fitting
        <ParameterEstimationComposition_Data_Fitting>`. These are passed to the optimizer specified
        by the `optimization_function <ParameterEstimationComposition.optimization_function>` specified as the
        `function <OptimizationControlMechanism.function> of the ParameterEstimationComposition's
        `OptimizationControlMechanism`.
        FIX: NEEDS TO BE ORGANIZED IN A WAY THAT IS COMPATIBLE WITH outcome_variables [SAME NUMBER OF ITEMS IN OUTER
             DIMENSION, WITH CORRESPONDING TYPES]

    objective_function : ObjectiveFunction, function or method
        FIX: IS EITHER:  THE NAME OF THE EXTERNAL OPTIMIZER FOR DATA FITTING OR A FUNCTION THAT IS OPTIMIZED FOR
            PARAMETER OPTIMIZATION [DAVE WANTS TO CALL THIS THE OPTIMZATION FUNCTION]
            Elfi, PYMC, S

    optimization_function : OptimizationFunction, function or method
        determines the function used to estimate the parameters of the `target <ParameterEstimationComposition.target>`
        Composition that either best fit the **data** when the ParameterEstimationComposition is used for `data
        fitting <ParameterEstimationComposition_Data_Fitting>`, or that achieve some maximum or minimum value of the
        the `optimization_function <ParameterEstimationComposition.optimization_function>` when the
        ParameterEstimationComposition is used for `parameter optimization
        <ParameterEstimationComposition_Optimization>`.  This is assigned as the `function
        <OptimizationControlMechanism.function>` of the ParameterEstimationComposition's `OptimizationControlMechanism`.
        FIX: DAVE'S OptimizationFunction [DAVE WANTS TO CALL THIS THE OBJECTIVE_FUNTION]

    num_estimates : int : default 1
        determines the number of estimates (i.e., calls to the OptimizationControlMechanism's `evaluation_function
        <OptimizationControlMechanism.evaluation_function>`) made for a each combination of `parameter
        <ParameterEstimationComposition>` values.

    initial_seed : int : default None
        determines the seed used to initialize the random number generator at construction.
        If it is not specified then then the seed is set to a random value on construction, and different runs of a
        script containing the ParameterEstimationComposition will yield different results, which should be roughly
        comparable if the estimation process is stable.  If **initial_seed** is specified, then running the script
        should yield identical results for the estimation process, which can be useful for debugging.

    same_seed_for_all_parameter_combinations :  bool : default False
        determines whether the random number generator used to select seeds for each estimate is re-initialized to the
        same value for each combination of `parameter <ParameterEstimationComposition>` values. If it is
        True, then any differences in the estimates made for each combination of parameter values will be determined
        exclusively by the effect of the *parameters* on the execution of the `target
        <ParameterEstimationComposition.target>` Composition, and not to any randomness intrinsic to the
        Composition itself (e.g., any of its Components).  This can be confirmed by identical results for repeated
        executions of the OptimizationControlMechanism's `evaluation_function
        <OptimizationControlMechanism.evaluation_function>` with the same set of parameter values.
        If *same_seed_for_all_parameter_combinations* is False, then each time a combination of parameter
        values is estimated, it will use a different set of seeds. This can be confirmed by different results for
        repeated executions of the OptimizationControlMechanism's `evaluation_function
        <OptimizationControlMechanism.evaluation_function>` with the same set of parameter values. Modest differences
        suggest stability of the estimation process across combination of parameter values, while substantial
        differences indicate instability, which may be helped by increasing `num_estimates
        <ParameterEstimationComposition.num_estimates>`.

    optimized_parameters : list
        contains the values of the `parameters <ParameterEstimationComposition.parameters>` of the
         `target <ParameterEstimationComposition.target>` Composition that best fit the **data** when the
         ParameterEstimationComposition is used for `data fitting <ParameterEstimationComposition_Data_Fitting>`,
         or that optimize performance of the `target <ParameterEstimationComposition.target>` according to the
         `optimization_function <ParameterEstimationComposition.optimization_function>` when the
         ParameterEstimationComposition is used for `parameter `optimization_function
         <ParameterEstimationComposition.optimization_function>`.  This is the same as the final set of `values
         <ControlSignal.value>` for the `control_signals <ControlMechanism.control_signals>` of the
         ParameterEstimationComposition's `OptimizationControlMechanism`.
         # FIX: ADD MENTION OF ARRAY ONCE THAT IS IMPLEMENTED FOR DISTRIBUTION OF PARAMETER VALUES

    results : list[list[list]]
        containts the `output_values <Mechanism_Base.output_values>` of the `OUTPUT` `Nodes <Composition_Nodes>`
        in the ``target <ParameterEstimationComposition.target>` Composition for every `TRIAL <TimeScale.TRIAL>`
        executed (see `Composition.results` for more details).
         # FIX: ADD MENTION OF OUTER DIMENSION ONCE THAT IS IMPLEMENTED FOR DISTRIBUTION OF PARAMETER VALUES
    """

    def __init__(self,
                 target, # agent_rep
                 parameters, # OCM control_signals
                 outcome_variables,  # OCM monitor_for_control
                 optimization_function, # function of OCM
                 data=None, # arg of OCM function
                 objective_function=None, # function of OCM ObjectiveMechanism
                 num_estimates=1, # num seeds per parameter combination (i.e., of OCM allocation_samples)
                 initial_seed=None,
                 same_seed_for_all_parameter_combinations=False,
                 name=None,
                 **param_defaults):

        self._validate_params(name, target, data, objective_function, outcome_variables)

        pem = self._instantiate_pem(target=target,
                                    parameters=parameters,
                                    outcome_variables=outcome_variables,
                                    data=data,
                                    objective_function=objective_function,
                                    optimization_function=optimization_function,
                                    num_estimates=num_estimates,
                                    initial_seed=initial_seed,
                                    same_seed_for_all_parameter_combinations=same_seed_for_all_parameter_combinations
                                    )

        super().__init__(name=name, nodes=target, controller=pem, **param_defaults)

    def _validate_params(self, name, target, data, objective_function, outcome_variables):

        # # Ensure parameters are in target composition
        if data and objective_function:
            raise ParameterEstimationCompositionError(f"Both 'data' and 'objective_function' args were specified for "
                                                      f"'{name or self.__class__.__name__}'; must choose one "
                                                      f"('data' for fitting or 'objective_function' for optimization).")

        # FIX: IMPLEMENT KATHERINE'S METHOD WHEN AVAILABLE
        # # Ensure parameters are in target composition
        # bad_params = [p for p in parameters if p not in target.parameters]
        # if bad_params:
        #     raise ParameterEstimationCompositionError(f"The following parameters "
        #                                               f"were not found in '{target.name}': {bad_params}.")

        # Ensure outcome_variables are OutputPorts in target
        bad_ports = [p for p in outcome_variables if not [p is not node and p not in node.output_ports for node in
                                                          target.nodes]]
        if bad_ports:
            raise ParameterEstimationCompositionError(f"The following outcome_variables were not found as "
                                                      f"nodes or OutputPorts in '{target.name}': {bad_ports}.")

    def _instantiate_pem(self,
                         target,
                         parameters,
                         outcome_variables,
                         data,
                         objective_function,
                         optimization_function,
                         num_estimates,
                         initial_seed,
                         same_seed_for_all_parameter_combinations
                         ):

        # # FIX: NEED TO GET CORRECT METHOD FOR "find_random_params"
        # random_params = target.find_random_params()

        def random_integer_generator():
            rng = np.random.RandomState()
            rng.seed(initial_seed)
            return rng.random_integers(num_estimates)

        random_seeds = SampleSpec(num=num_estimates, function=random_integer_generator)

        # randomization_control_signal = ControlSignal(modulates=random_params,
        #                                              allocation_samples=random_seeds)
        # parameters = convert_to_list(parameters).append(randomization_control_signal)

        if data:
            objective_function = objective_function(data)

        # Get ControlSignals for parameters to be searched
        control_signals = []
        for p,a in parameters.items():
            control_signals.append(ControlSignal(modulates=p,
                                                 allocation_samples=a))

        return OptimizationControlMechanism(control_signals=control_signals,
                                            objective_mechanism=ObjectiveMechanism(monitor=outcome_variables,
                                                                                   function=objective_function),
                                            function=optimization_function)

    # FIX: USE THIS IF ASSIGNING PEM AS OUTPUT Node DOESN'T WORK
    def execute(self, **kwargs):
        super().execute(kwargs)
        optimized_control_allocation = self.controller.output_values
        self.results[-1] = optimized_control_allocation
        return optimized_control_allocation

    # FIX: IF DATA WAS SPECIFIED, CHECK THAT INPUTS ARE APPROPRIATE FOR THOSE DATA.
    def run(self):
        pass

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
        # FIX: THIS NEEDS TO BE A DEQUE THAT TRACKS ALL THE CONTROL_SIGNAL VALUES OVER num_estimates FOR PARAM DISTRIB
        # FIX: AUGMENT TO USE num_estimates and num_trials_per_estimate
        # FIX: AUGMENT TO USE same_seed_for_all_parameter_combinations PARAMETER
        return self.function(feature_values, control_allocation, context=context)

    @property
    def optimized_parameter_values(self):
        return self.controller.output_values