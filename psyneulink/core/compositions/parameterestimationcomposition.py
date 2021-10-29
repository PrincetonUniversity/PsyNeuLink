# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *************************************  CompositionFunctionApproximator ***********************************************

# FIX: SEED FOR noise PARAMETER OF TransferMechanism GETS ASSIGNED TO THE MECHANISM,
#      BUT THERE DOES NOT SEEM TO BE A PARAMETER PORT ASSIGNED TO IT FOR THAT
# FIX: ADD Parameters Class DEFINTION FOR ParameterEstimationComposition
# FIX: CHANGE REFERENCES TO <`parameter <ParameterEstimationComposition.parameters>` values> AND THE LIKE TO
#      <`parameter values <ParameterEstimationComposition.parameter_ranges_or_priors>`>
# FIX: ADD TESTS:
#      - FOR ERRORS IN parameters AND outcome_variables SPECIFICATIONS
#      - GENERATES CORRECT SEED ITERATOR, control_signals AND THEIR projections
#      - EVENTUALLY, EXECUTION IN BOTH DATA FITTING AND OPTIMIZATION MODES

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

COMMENT:
    ADD MENTION THAT THIS ALLOWS FITTING AND OPTIMIZING "LIKELIHOOD-FREE" MODELS.
COMMENT

A `ParameterEstimationComposition` is a subclass of `Composition` that is used to estimate parameters of
another `Composition` (its `target <ParameterEstimationComposition.target>` in order to fit that to a set of
data (`ParameterEstimationComposition_Data_Fitting`) or to optimize the `net_outcome <ControlMechanism.net_outcome>`
of the `target <ParameterEstimationComposition.target>` Compositoin according to an `objective_function`
(`ParameterEstimationComposition_Optimization`). In either case, when the ParameterEstimationComposition is `run
<Composition.run>` with a given set of `inputs <Composition_Execution_Inputs>`, it returns the set of
parameter values in `optimized_parameter_values <ParameterEstimationComposition.optimized_parameter_values>`
that it estimates best satisfy either of those conditions, and the results of running the `target 
<ParameterEstimationComposition.target>` with those parameters in its `results <ParameterEstimationComposition.results>`
attribute.  The arguments below are the primary ones used to configure a ParameterEstimationComposition for either 
purpose `ParameterEstimationComposition_Data_Fitting` or `ParameterEstimationComposition_Optimization`), followed by
sections that describe arguments specific to each.

    * **target** - specifies the `Composition` for which parameters are to be estimated.

    * **parameters** - specifies the parameters of the `target <ParameterEstimationComposition.target>` Composition
      to be estimated.  These are specified in a dict, in which the key of each entry specifies a parameter to
      estimate, and its value either a range of values to sample for that parameter or a distribution from which
      to draw them.

    * **outcome_variables** - specifies the `OUTPUT` `Nodes <Composition_Nodes>` of the `target
      <ParameterEstimationComposition.target>` Composition, the `values <Mechanism_Base.value>` of which are used
      to evaluate the fit of the different combination of parameter values sampled.

    * **num_estimates** - specifies the number of independent samples that are estimated for a given combination of
      parameter values.


.. _ParameterEstimationComposition_Data_Fitting:

Data Fitting
------------

The ParameterEstimationComposition can be used to find a set of parameters for the `target
<ParameterEstimationComposition.target>` such that, when that Composition is run with a given set of inputs,
its results best match a specified set of empirical data.  This requires the following additional arguments to be
specified:

    .. __ParameterEstimationComposition_Data:

    * **data** - specifies the data to which the `outcome_variables <ParameterEstimationComposition.outcome_variables>`
      are fit in the estimation process.  They must be in a format that aligns the specification of
      `outcome_variables <ParameterEstimationComposition.outcome_variables>`.
      COMMENT:
          FIX:  GET MORE FROM DAVE HERE
      COMMENT

    * **optimization_function** - specifies the function used to compare the `values <Mechanism_Base.value>` of the
      `outcome_variables <ParameterEstimationComposition.outcome_variables>` with the **data**, and search over values
      of the `parameters <ParameterEstimationComposition.parameters>` that maximize the fit. This must be either
      `ParameterEstimationFunction` or a subclass of that.  By default, ParameterEstimationFunction uses maximum
      likelihood estimation (MLE) to compare the `outcome_variables <ParameterEstimationComposition.outcome_variables>`
      and the data, and
      COMMENT:
           FIX: GET MORE FROM DAVE HERE
      COMMENT
      for searching over parameter combinations.

.. _ParameterEstimationComposition_Optimization:

Parameter Optimization
----------------------

    .. _ParameterEstimationComposition_Objective_Function:

    * **objective_function** - specifies a function used to evaluate the `values <Mechanism_Base.value>` of the
      `outcome_variables <ParameterEstimationComposition.outcome_variables>`, according to which combinations of
      `parameters <ParameterEstimationComposition.parameters>` are assessed.

    * **optimization_function** - specifies the function used to search over values of the `parameters
      <ParameterEstimationComposition.parameters>` in order to optimize the **objective_function**.  It can be any
      `OptimizationFunction` that accepts an `objective_function <OptimizationFunction>` as an argument or specifies
      one by default.  By default `GridSearch` is used which exhaustively evaluates all combinations of  `parameter
      <ParameterEstimationComposition.parameters>` values, and returns the set that either maximizes or minimizes the
      **objective_function**.

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
from psyneulink.core.components.ports.port import PortError
from psyneulink.core.components.ports.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.compositions.composition import Composition
from psyneulink.core.globals.context import Context
from psyneulink.core.globals.utilities import convert_to_list
from psyneulink.core.globals.sampleiterator import SampleSpec

__all__ = ['ParameterEstimationComposition']

RANDOMIZATION_SEED_CONTROL_SIGNAL_NAME = 'RANDOMIZATION SEEDS'

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
        each, or priors that define a distribution over those.

    outcome_variables : list[Composition output nodes]
        specifies the `OUTPUT` `Nodes <Composition_Nodes>` of the `target <ParameterEstimationComposition.target>`
        Composition, the `values <Mechanism_Base.value>` of which are either compared to the **data** when the
        ParameterEstimationComposition is used for `data fitting <ParameterEstimationComposition_Data_Fitting>`
        or used by the `optimization_function <ParameterEstimationComposition.optimization_function>` when the
        ParameterEstimationComposition is used for `parameter optimization
        <ParameterEstimationComposition_Optimization>`.

    data : array : default None
        specifies the data to to be fit when the ParameterEstimationComposition is used for
        `data fitting <ParameterEstimationComposition_Data_Fitting>`;  structure must conform to format of
        **outcome_variables** (see `data <ParameterEstimationComposition,data>` for additional information).

    objective_function : ObjectiveFunction, function or method
        specifies the function used to evaluate the `net_outcome <ControlMechanism.net_outcome>` of the `target
        <ParameterEstimationComposition.target>` Composition when the ParameterEstimationComposition
        is used for `parameter optimization <ParameterEstimationComposition_Optimization>`
        (see `objective_function <ParameterEstimationComposition.objective_function>` for additional information).

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
        combination of `parameter <ParameterEstimationComposition.parameters>` values (see `constant_seed_across_search
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
        construct the `control_signals <OptimizationControlMechanism.control_signals>` used to modulate each parameter
        that is being fit.

    .. technical_note::
        A `ControlSignal` is added to the `control_signals <OptimizationControlMechanism.control_signals>` of the
        ParameterEstimationComposition's `OptimizationControlMechanism`, named *RANDOMIZATION_SEED_CONTROL_SIGNAL_NAME*,
        to modulate the seeds used to randomize each estimate of the `net_outcome <ControlMechanism.net_outcome>`
        for each run of the `target <ParameterEstimationComposition.target>` Composition (i.e., call to its `evaluate
        <Composition.evaluate>` method). That ControlSignal sends a `ControlProjection` to every `Parameter` of every
        `Component` in the `target <ParameterEstimationComposition.target>` Composition that is labelled "seed",
        each of which corresponds to a Parameter that uses a random number generator to assign its value (i.e.,
        as its `function <ParameterPort.function>`.  This ControlSignal is used to change the seeds for all Parameters
        that use random values at the start of each run of the `target <ParameterEstimationComposition.target>`
        Composition used to estimate a given `control_allocation <ControlMechanism.control_allocation>` of the other
        ControlSignals (i.e., the ones for the `parameters <ParameterEstimationComposition.parameters>` being fit).
        The `initial_seed <ParameterEstimationComposition.initial_seed>` `same_seed_for_all_parameter_combinations
        <ParameterEstimationComposition.same_seed_for_all_parameter_combinations>` attributes can be used to further
        refine this behavior.

    parameter_ranges_or_priors : List[Union[Iterator, Function, ist or Value]
        determines the range of values evaluated for each `parameter <ParameterEstimationComposition.parameters>`.
        These are assigned as the `allocation_samples <ControlSignal.allocation_samples>` for the `ControlSignal`
        assigned to the ParameterEstimationComposition's `OptimizationControlMechanism` corresponding to each of the
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
        <ParameterEstimationComposition_Data_Fitting>`.  These must be structured in form that aligns with the
        specified `outcome_variables <ParameterEstimationComposition_Data_Fitting.outcome_variables>` (see `data
        <ParameterEstimationComposition_Data>` for additional details). The data are passed to the optimizer
        used by `optimization_function <ParameterEstimationComposition.optimization_function>`.

    objective_function : ObjectiveFunction, function or method
        determines the function used to evaluate the `results <Composition.results>` of the `target
        <ParameterEstimationComposition.target>` Composition under each set of `parameter
        <ParameterEstimationComposition.parameters>` values when the ParameterEstimationComposition
        is used for `parameter optimization <ParameterEstimationComposition_Optimization>`.  It is passed to the
        ParameterEstimationComposition's `OptimizationControlMechanism` as the function of its `objective_mechanism
        <OptimizationControlMechanism.objective_mechanism>`, that is used to compute the `net_outcome
        <ControlMechanism.net_outcome>` for of the `target <ParameterEstimationComposition.target>` Composition each
        time it is `run <Composition.run>` (see `ParameterEstimationComposition_Objective_Function` for additional
        details).

    optimization_function : OptimizationFunction
        determines the function used to estimate the parameters of the `target <ParameterEstimationComposition.target>`
        Composition that either best fit the **data** when the ParameterEstimationComposition is used for `data
        fitting <ParameterEstimationComposition_Data_Fitting>`, or that achieve some maximum or minimum value of the
        the `optimization_function <ParameterEstimationComposition.optimization_function>` when the
        ParameterEstimationComposition is used for `parameter optimization
        <ParameterEstimationComposition_Optimization>`.  This is assigned as the `function
        <OptimizationControlMechanism.function>` of the ParameterEstimationComposition's `OptimizationControlMechanism`.

    num_estimates : int : default 1
        determines the number of estimates of the `net_outcome <ControlMechanism.net_outcome>` of the `target
        <ParameterEstimationComposition.target>` Composition (i.e., number of calls to its `evaluate
        <Composition.evaluate>` method) for each combination of `parameter <ParameterEstimationComposition.parameters>`
        values (i.e., `control_allocation <ControlMechanism.control_allocation>`) evaluated.

    initial_seed : int : default None
        determines the seed used to initialize the random number generator at construction.
        If it is not specified then then the seed is set to a random value on construction, and different runs of a
        script containing the ParameterEstimationComposition will yield different results, which should be roughly
        comparable if the estimation process is stable.  If **initial_seed** is specified, then running the script
        should yield identical results for the estimation process, which can be useful for debugging.

    same_seed_for_all_parameter_combinations :  bool : default False
        determines whether the random number generator used to select seeds for each estimate of the `target
        <ParameterEstimationComposition.target>` Composition's `net_outcome <ControlMechanism.net_outcome>` is
        re-initialized to the same value for each combination of `parameter <ParameterEstimationComposition>` values
        evaluated. If same_seed_for_all_parameter_combinations is True, then any differences in the estimates made
        of `net_outcome <ControlMechanism.net_outcome>` for each combination of parameter values will reflect
        exclusively the influence of the *parameters* on the execution of the `target
        <ParameterEstimationComposition.target>` Composition, and *not* any variability intrinsic to the execution of
        the Composition itself (e.g., any of its Components). This can be confirmed by identical results for repeated
        executions of the OptimizationControlMechanism's `evaluation_function
        <OptimizationControlMechanism.evaluation_function>` with the same set of parameter values (i.e.,
        `control_allocation <ControlMechanism.control_allocation>`control_allocation).
        If *same_seed_for_all_parameter_combinations* is False, then each time a combination of parameter
        values is estimated, it will use a different set of seeds. This can be confirmed by differing results for
        repeated executions of the OptimizationControlMechanism's `evaluation_function
        <OptimizationControlMechanism.evaluation_function>` with the same set of parameter values (`control_allocation
        <ControlMechanism.control_allocation>`control_allocation). Small differences in results suggest stability of
        the estimation process across combinations of parameter values, while substantial differences indicate
        instability, which may be helped by increasing `num_estimates <ParameterEstimationComposition.num_estimates>`.

    optimized_parameter_values : list
        contains the values of the `parameters <ParameterEstimationComposition.parameters>` of the
         `target <ParameterEstimationComposition.target>` Composition that best fit the **data** when the
         ParameterEstimationComposition is used for `data fitting <ParameterEstimationComposition_Data_Fitting>`,
         or that optimize performance of the `target <ParameterEstimationComposition.target>` according to the
         `optimization_function <ParameterEstimationComposition.optimization_function>` when the
         ParameterEstimationComposition is used for `parameter optimization
         <ParameterEstimationComposition_Optimization>`.  If `parameter values
         <ParameterEstimationComposition.parameter_ranges_or_priors>` are specified as ranges of values, then
         each item of `optimized_parameter_values` is the optimized value of the corresponding `parameter
         <ParameterEstimationComposition.parameter>`. If `parameter values
         <ParameterEstimationComposition.parameter_ranges_or_priors>` are specified as priors, then each item of
         `optimized_parameter_values` is an array containing the values of the corresponding `parameter
         <ParameterEstimationComposition.parameter>` the distribution of which were determined to be optimal.

    results : list[list[list]]
        contains the `output_values <Mechanism_Base.output_values>` of the `OUTPUT` `Nodes <Composition_Nodes>`
        in the ``target <ParameterEstimationComposition.target>` Composition for every `TRIAL <TimeScale.TRIAL>`
        executed (see `Composition.results` for more details). If the ParameterEstimationComposition is used for
        `data fitting <ParameterEstimationComposition_Data_Fitting>`, and `parameter values
        <ParameterEstimationComposition.parameter_ranges_or_priors>` are specified as ranges of values, then
        each item of `results <Composition.results>` is an array of `output_values <Mechanism_Base.output_values>`
        (sampled over `num_estimates <ParameterEstimationComposition.num_estimates>`) obtained for the single
        optimized combination of `parameter <ParameterEstimationComposition.parameters>` values contained in the
        corresponding item of `optimized_parameter_values <ParameterEstimationComposition.optimized_parameter_values>`.
        If `parameter values <ParameterEstimationComposition.parameter_ranges_or_priors>` are specified as priors,
        then each item of `results` is an array of `output_values <Mechanism_Base.output_values>` (sampled over
        `num_estimates <ParameterEstimationComposition.num_estimates>`), each of which corresponds to a combination
        of `parameter <ParameterEstimationComposition.parameters>` values that were used to generate those results;
        it is the *distribution* of those `parameter <ParameterEstimationComposition.parameters>` values that were
        found to best fit the data.
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

        self._validate_params(locals())

        self.optimized_parameter_values = []

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

    def _validate_params(self, params):

        if params['data'] and params['objective_function']:
            raise ParameterEstimationCompositionError(f"Both 'data' and 'objective_function' args were specified for "
                                                      f"'{params['name'] or self.__class__.__name__}'; must choose one "
                                                      f"('data' for fitting or 'objective_function' for optimization).")

        # FIX: REMOVE ALL THIS, AND LET IT BE HANDLED BY CONSTRUCTION (PROBLEM IS, ERROR MESSAGES ARE MORE OPAQUE)

        # FIX: IMPLEMENT RECURSIVELY FOR NESTED COMPS
        # Ensure that a ControlSignal can be created for all parameters specified
        bad_params = []
        from psyneulink.core.components.ports.port import _parse_port_spec
        for param_spec in list(params['parameters'].keys()):
            try:
                _parse_port_spec(owner=params['target'], port_type=ControlSignal, port_spec=param_spec)
            except:
                bad_params.append(param_spec)
        if bad_params:
            raise ParameterEstimationCompositionError(f"The following parameter specifications "
                                                      f"were not valid for '{params['target'].name}': {bad_params}.")

        # FIX: IMPLEMENT RECURSIVELY FOR NESTED COMPS
        # Ensure outcome_variables are OutputPorts in target
        bad_ports = [p for p in params['outcome_variables'] if not [p is not node and p not in node.output_ports for
                                                                 node in
                                                          params['target'].nodes]]
        if bad_ports:
            raise ParameterEstimationCompositionError(f"The following outcome_variables were not found as nodes or "
                                                      f"OutputPorts in '{params['target'].name}': {bad_ports}.")

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

        # Construct iterator for seeds used to randomize estimates
        def random_integer_generator():
            rng = np.random.RandomState()
            rng.seed(initial_seed)
            return rng.random_integers(num_estimates)
        random_seeds = SampleSpec(num=num_estimates, function=random_integer_generator)

        # FIX: DOES target.all_dependent_parameters LOOK FOR PARAMS RECURSIVELY IN NESTED COMPOSITIONS??
        # FIX: noise PARAM OF TRANSFERMECHANISM IS MARKED AS SEED WHEN ASSIGNED A DISTRIBUTUION FUNCTION,
        #                BUT IT HAS NO PARAMETER PORT BECAUSE THAT PRESUMABLY IS FOR THE INTEGRATOR FUNCTION,
        #                BUT THAT IS NOT FOUND BY target.all_dependent_parameters
        # Get parameters of target that use seeds (i.e., implement a random value)
        seed_params = []
        for params_dict in target.all_dependent_parameters('seed').values():
            seed_params.extend([p._port for p in list(params_dict.values())])

        # Construct ControlSignal to modify seeds over estimates
        seed_control_signal = ControlSignal(name=RANDOMIZATION_SEED_CONTROL_SIGNAL_NAME,
                                            modulates=seed_params,
                                            allocation_samples=random_seeds)

        # FIX: WHAT iS THIS DOING?
        if data:
            objective_function = objective_function(data)

        # Get ControlSignals for parameters to be searched
        control_signals = []
        for param,allocation in parameters.items():
            control_signals.append(ControlSignal(modulates=param,
                                                 allocation_samples=allocation))
        # Add ControlSignal for seeds to end of list of parameters to be controlled by pem
        convert_to_list(control_signals).append(seed_control_signal)

        return OptimizationControlMechanism(control_signals=control_signals,
                                            objective_mechanism=ObjectiveMechanism(monitor=outcome_variables,
                                                                                   function=objective_function),
                                            function=optimization_function)

    # FIX: IF DATA WAS SPECIFIED, CHECK THAT INPUTS ARE APPROPRIATE FOR THOSE DATA.
    def run(self):
        pass

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
        # FIX: THE FOLLOWING MOSTLY NEEDS TO BE HANDLED BY OptimizationFunction.evaluation_function AND/OR grid_evaluate
        # FIX:   THIS NEEDS TO BE A DEQUE THAT TRACKS ALL THE CONTROL_SIGNAL VALUES OVER num_estimates FOR PARAM DISTRIB
        # FIX:   AUGMENT TO USE num_estimates and num_trials_per_estimate
        # FIX:   AUGMENT TO USE same_seed_for_all_parameter_combinations PARAMETER
        return self.function(feature_values, control_allocation, context=context)