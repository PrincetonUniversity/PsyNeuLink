# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *************************************  ParameterEstimationComposition ************************************************

# FIX: SEED FOR noise PARAMETER OF TransferMechanism GETS ASSIGNED TO THE MECHANISM,
#      BUT THERE DOES NOT SEEM TO BE A PARAMETER PORT ASSIGNED TO IT FOR THAT
# FIX: ADD Parameters Class DEFINITION FOR ParameterEstimationComposition
# FIX: CHANGE REFERENCES TO <`parameter <ParameterEstimationComposition.parameters>` values> AND THE LIKE TO
#      <`parameter values <ParameterEstimationComposition.parameter_ranges_or_priors>`>
# FIX: ADD TESTS:
#      - FOR ERRORS IN parameters AND outcome_variables SPECIFICATIONS
#      - GENERATES CORRECT SEED ITERATOR, control_signals AND THEIR projections
#      - EVENTUALLY, EXECUTION IN BOTH DATA FITTING AND OPTIMIZATION MODES
# FIX: SHOULD PASS ANY ARGS OF RUN METHOD (OTHER THAN num_trial) TO evaluate METHOD OF model COMPOSITION
#  NUM_TRIALS?)

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

A `ParameterEstimationComposition` is a subclass of `Composition` that is used to estimate specified `parameters
<ParameterEstimationComposition.parameters>` of a `model <ParameterEstimationComposition.model>` Composition,
in order to fit the `outputs <ParameterEstimationComposition.outcome_variables>`
of the `model <ParameterEstimationComposition.model>` to a set of data (`ParameterEstimationComposition_Data_Fitting`),
or to optimize its `net_outcome <ControlMechanism.net_outcome>` according to an `objective_function`
(`ParameterEstimationComposition_Optimization`). In either case, when the ParameterEstimationComposition is `run
<Composition.run>` with a given set of `inputs <Composition_Execution_Inputs>`, it returns the set of
parameter values in its `optimized_parameter_values <ParameterEstimationComposition.optimized_parameter_values>`
attribute that it estimates best satisfy either of those conditions, and the results of running the `model
<ParameterEstimationComposition.model>` with those parameters in its `results <ParameterEstimationComposition.results>`
attribute.  The arguments below are the primary ones used to configure a ParameterEstimationComposition for either
`ParameterEstimationComposition_Data_Fitting` or `ParameterEstimationComposition_Optimization`, followed by
sections that describe arguments specific to each.

    .. _ParameterEstimationComposition_Model:

    * **model** - this is a convenience argument that can be used to specify a `Composition` other than the
      ParameterEstimationComposition itself as the model. Alternatively, the model to be fit can be constructed
      within the ParameterEstimationComposition itself, using the **nodes** and/or **pathways** arguments of its
      constructor (see `Composition_Constructor` for additional details).   The **model** argument
      or the **nodes**, **pathways**, and/or **projections** arguments must be specified, but not both.

      .. note::
         Neither the **controller** nor any of its associated arguments can be specified in the constructor for a
         ParameterEstimationComposition; this is constructed automatically using the arguments described below.

    * **parameters** - specifies the parameters of the `model <ParameterEstimationComposition.model>` to be
      estimated.  These are specified in a dict, in which the key of each entry specifies a parameter to estimate,
      and its value either a range of values to sample for that parameter or a distribution from which to draw them.

    * **outcome_variables** - specifies the `OUTPUT` `Nodes <Composition_Nodes>` of the `model
      <ParameterEstimationComposition.model>`, the `values <Mechanism_Base.value>` of which are used
      to evaluate the fit of the different combinations of parameter values sampled.

    * **num_estimates** - specifies the number of independent samples that are estimated for a given combination of
      parameter values.


.. _ParameterEstimationComposition_Data_Fitting:

Data Fitting
------------

The ParameterEstimationComposition can be used to find a set of parameters for the `model
<ParameterEstimationComposition.model>` such that, when it is run with a given set of inputs, its results
best match a specified set of empirical data.  This requires the following additional arguments to be specified:

    .. _ParameterEstimationComposition_Data:

    * **data** - specifies the data to which the `outcome_variables <ParameterEstimationComposition.outcome_variables>`
      are fit in the estimation process.  They must be in a format that aligns with the specification of
      the `outcome_variables <ParameterEstimationComposition.outcome_variables>`.
      COMMENT:
          FIX:  GET MORE FROM DAVE HERE
      COMMENT

    * **optimization_function** - specifies the function used to compare the `values <Mechanism_Base.value>` of the
      `outcome_variables <ParameterEstimationComposition.outcome_variables>` with the **data**, and search over values
      of the `parameters <ParameterEstimationComposition.parameters>` that maximize the fit. This must be either a
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
      `parameters <ParameterEstimationComposition.parameters>` are assessed.  The shape of the `variable
      <Component.variable>` of the **objective_function** (i.e., its first positional argument) must be the same as
      an array containing the `value <OutputPort.value>` of the OutputPort corresponding to each  item specified in
      `outcome_variables <ParameterEstimationComposition.outcome_variables>`.

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

Structure
---------

.. technical_note::
   ParameterEstimationComposition uses an `PEC_OCM` as its `controller <Composition.controller>` -- a specialized
   subclass of `OptimizationControlMechanism` that intercepts inputs provided to the `run
   <ParameterEstimationComposition.run>` method of the ParameterEstimationComposition, and assigns them directly
   to the `state_feature_values` of the PEC_OCM when it executes.

.. _ParameterEstimationComposition_Class_Reference:

Class Reference
---------------

"""
import numpy as np
import pandas as pd

from psyneulink.core.components.mechanisms.modulatory.control.optimizationcontrolmechanism import \
    OptimizationControlMechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.ports.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.compositions.composition import Composition, NodeRole
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import BEFORE, OVERRIDE
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.utilities import convert_to_list
from psyneulink.core.scheduling.time import TimeScale


__all__ = ['ParameterEstimationComposition']

COMPOSITION_SPECIFICATION_ARGS = {'nodes', 'pathways', 'projections'}
CONTROLLER_SPECIFICATION_ARGS = {'controller',
                                 'enable_controller',
                                 'controller_mode',
                                 'controller_time_scale',
                                 'controller_condition',
                                 'retain_old_simulation_data'}


class ParameterEstimationCompositionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


def _initial_seed_getter(owning_component, context=None):
    try:
        return owning_component.controler.parameters.initial_seed._get(context)
    except:
        return None

def _initial_seed_setter(value, owning_component, context=None):
    owning_component.controler.parameters.initial_seed.set(value, context)
    return value

def _same_seed_for_all_parameter_combinations_getter(owning_component, context=None):
    try:
        return owning_component.controler.parameters.same_seed_for_all_allocations._get(context)
    except:
        return None

def _same_seed_for_all_parameter_combinations_setter(value, owning_component, context=None):
    owning_component.controler.parameters.same_seed_for_all_allocations.set(value, context)
    return value


class ParameterEstimationComposition(Composition):
    """
    Composition(                           \
        parameters,
        outcome_variables,
        model=None,
        data=None,
        objective_function=None,
        optimization_function=None,
        num_estimates=1,
        number_trials_per_estimate=None,
        initial_seed=None,
        same_seed_for_all_parameter_combinations=False
        )

    Subclass of `Composition` that estimates specified parameters either to fit the results of a Composition
    to a set of data or to optimize a specified function.

    Automatically implements an `OptimizationControlMechanism` as its `controller <Composition.controller>`,
    that is constructed using arguments to the ParameterEstimationComposition's constructor as described below.

    The following arguments are those specific to ParmeterEstimationComposition; see `Composition` for additional
    arguments

    Arguments
    ---------

    parameters : dict[Parameter:Union[Iterator, Function, List, value]
        specifies the parameters of the `model <ParameterEstimationComposition.model>` used for
        `ParameterEstimationComposition_Data_Fitting` or `ParameterEstimationComposition_Optimization`, and either
        the range of values to be evaluated for each parameter, or priors that define a distribution over those.

    outcome_variables : list[Composition output nodes]
        specifies the `OUTPUT` `Nodes <Composition_Nodes>` of the `model <ParameterEstimationComposition.model>`,
        the `values <Mechanism_Base.value>` of which are either compared to a specified **data** when the
        ParameterEstimationComposition is used for `ParameterEstimationComposition_Data_Fitting`, or used by the
        `optimization_function <ParameterEstimationComposition.optimization_function>` for
        `ParameterEstimationComposition_Optimization`.

    model : Composition : default None
        specifies an external `Composition` for which parameters are to be `fit to data
        <ParameterEstimationComposition_Data_Fitting>` or `optimized <ParameterEstimationComposition_Optimization>`
        according to a specified `objective_function <ParameterEstimationComposition.objective_function>`.
        If **model** is None (the default), the ParameterEstimationComposition itself is used (see
        `model <ParameterEstimationComposition_Model>` for additional information).

    data : array : default None
        specifies the data to be fit when the ParameterEstimationComposition is used for
        `ParameterEstimationComposition_Data_Fitting`;  structure must conform to format of
        **outcome_variables** (see `data <ParameterEstimationComposition.data>` for additional information).

    data_categorical_dims : Union[Iterable] : default None
        specifies the dimensions of the data that are categorical. If a list of boolean values is provided, it is
        assumed to be a mask for the categorical data dimensions and must have the same length as columns in data. If
        it is an iterable of integers, it is assumed to be a list of the categorical dimensions indices. If it is None,
        all data dimensions are assumed to be continuous.

    objective_function : ObjectiveFunction, function or method
        specifies the function used to evaluate the `net_outcome <ControlMechanism.net_outcome>` of the `model
        <ParameterEstimationComposition.model>` when the ParameterEstimationComposition is used for
        `ParameterEstimationComposition_Optimization` (see `objective_function
        <ParameterEstimationComposition.objective_function>` for additional information).

    optimization_function : OptimizationFunction, function or method
        specifies the function used to evaluate the `fit to data <ParameterEstimationComposition_Data_Fitting>`
        or `optimize <ParameterEstimationComposition_Optimization>` the parameters of the `model
        <ParameterEstimationComposition.model>` according to a specified `objective_function
        <ParameterEstimationComposition.objective_function>`; the shape of its `variable <Component.variable>` of the
        `objective_function (i.e., its first positional argument) must be the same as an array containing the `value
        <OutputPort.value>` of the OutputPort corresponding to each item specified in `outcome_variables
        <ParameterEstimationComposition.outcome_variables>`.

    num_estimates : int : default 1
        specifies the number of estimates made for a each combination of `parameter <ParameterEstimationComposition>`
        values (see `num_estimates <ParameterEstimationComposition.num_estimates>` for additional information);
        it is passed to the ParameterEstimationComposition's `controller <Composition.controller>` to set its
        `num_estimates <OptimizationControlMechanism.num_estimates>` Parameter.

    num_trials_per_estimate : int : default None
        specifies an exact number of trials to execute for each run of the `model
        <ParameterEstimationComposition.model>` when estimating each combination of `parameter
        <ParameterEstimationComposition.parameters>` values (see `num_trials_per_estimate
        <ParameterEstimationComposition.num_trials_per_estimate>` for additional information).

    initial_seed : int : default None
        specifies the seed used to initialize the random number generator at construction; it is passed to the
        ParameterEstimationComposition's `controller <Composition.controller>` to set its `initial_seed
        <OptimizationControlMechanism.initial_seed>` Parameter.

    same_seed_for_all_parameter_combinations :  bool : default False
        specifies whether the random number generator is re-initialized to the same value when estimating each
        combination of `parameter <ParameterEstimationComposition.parameters>` values; it is passed to the
        ParameterEstimationComposition's `controller <Composition.controller>` to set its
        `same_seed_for_all_allocations <OptimizationControlMechanism.same_seed_for_all_allocations>` Parameter.


    Attributes
    ----------

    model : Composition
        identifies the `Composition` used for `ParameterEstimationComposition_Data_Fitting` or
        `ParameterEstimationComposition_Optimization`.  If the **model** argument of the
        ParameterEstimationComposition's constructor is not specified, `model` returns the
        ParameterEstimationComposition itself.

    parameters : list[Parameters]
        determines the parameters of the `model <ParameterEstimationComposition.model>` used for
        `ParameterEstimationComposition_Data_Fitting` or `ParameterEstimationComposition_Optimization`
        (see `control <OptimizationControlMechanism.control>` for additional details).

    parameter_ranges_or_priors : List[Union[Iterator, Function, ist or Value]
        determines the range of values evaluated for each `parameter <ParameterEstimationComposition.parameters>`.
        These are assigned as the `allocation_samples <ControlSignal.allocation_samples>` for the `ControlSignal`
        assigned to the ParameterEstimationComposition's `OptimizationControlMechanism` corresponding to each of the
        specified `parameters <ParameterEstimationComposition.parameters>`.

    outcome_variables : list[Composition Output Nodes]
        determines the `OUTPUT` `Nodes <Composition_Nodes>` of the `model <ParameterEstimationComposition.model>`,
        the `values <Mechanism_Base.value>` of which are either compared to the **data** when the
        ParameterEstimationComposition is used for `ParameterEstimationComposition_Data_Fitting`, or evaluated by the
        ParameterEstimationComposition's `optimization_function <ParameterEstimationComposition.optimization_function>`
        when it is used for `ParameterEstimationComposition_Optimization`.

    data : array
        determines the data to be fit by the `model <ParameterEstimationComposition.model>` when the
        ParameterEstimationComposition is used for `ParameterEstimationComposition_Data_Fitting`.
        These must be structured in form that aligns with the specified `outcome_variables
        <ParameterEstimationComposition.outcome_variables>` (see `data
        <ParameterEstimationComposition_Data>` for additional details). The data are passed to the optimizer
        used by `optimization_function <ParameterEstimationComposition.optimization_function>`.  Returns
        None if the model is being used for `ParameterEstimationComposition_Optimization`.

    objective_function : ObjectiveFunction, function or method
        determines the function used to evaluate the `results <Composition.results>` of the `model
        <ParameterEstimationComposition.model>` under each set of `parameter
        <ParameterEstimationComposition.parameters>` values when the ParameterEstimationComposition is used for
        `ParameterEstimationComposition_Optimization`.  It is passed to the ParameterEstimationComposition's
        `OptimizationControlMechanism` as the function of its `objective_mechanism
        <ControlMechanism.objective_mechanism>`, that is used to compute the `net_outcome
        <ControlMechanism.net_outcome>` for of the `model <ParameterEstimationComposition.model>` each time it is
        `run <Composition.run>` (see `objective_function <ParameterEstimationComposition_Objective_Function>`
        for additional details).

    optimization_function : OptimizationFunction
        determines the function used to estimate the parameters of the `model <ParameterEstimationComposition.model>`
        that either best fit the `data <ParameterEstimationComposition.data>` when the ParameterEstimationComposition
        is used for `ParameterEstimationComposition_Data_Fitting`, or that achieve some maximum or minimum value of
        the the `optimization_function <ParameterEstimationComposition.optimization_function>` when the
        ParameterEstimationComposition is used for `ParameterEstimationComposition_Optimization`.  This is assigned as
        the `function <OptimizationControlMechanism.function>` of the ParameterEstimationComposition's
        `OptimizationControlMechanism`.

    num_estimates : int
        determines the number of estimates of the `net_outcome <ControlMechanism.net_outcome>` of the `model
        <ParameterEstimationComposition.model>` (i.e., number of calls to its `evaluate <Composition.evaluate>`
        method) for a given combination of `parameter <ParameterEstimationComposition.parameters>` values (i.e.,
        `control_allocation <ControlMechanism.control_allocation>`) evaluated.

    num_trials_per_estimate : int or None
        imposes an exact number of trials to be executed in each run of `model <ParameterEstimationComposition.model>`
        used to evaluate its `net_outcome <ControlMechanism.net_outcome>` by a call to its
        OptimizationControlMechanism's `evaluate_agent_rep <OptimizationControlMechanism.evaluate_agent_rep>` method.
        If it is None (the default), then either the number of **inputs** or the value specified for **num_trials** in
        the ParameterEstimationComposition's `run <ParameterEstimationComposition.run>` method used to determine the
        number of trials executed (see `number of trials <Composition_Execution_Num_Trials>` for additional
        information).

        .. _note::
           The **num_trials_per_estimate** is distinct from the **num_trials** argument of the
           ParameterEstimationComposition's `run <Composition.run>` method.  The latter determines how many full fits
           of the `model <ParameterEstimationComposition.model>` are carried out (that is, how many times the
           ParameterEstimationComposition *itself* is run), whereas **num_trials_per_estimate** determines how many
           trials are run for a given combination of `parameter <ParameterEstimationComposition.parameters>` values
           *within* each fit.

    initial_seed : int or None
        contains the seed used to initialize the random number generator at construction, that is stored on the
        ParameterEstimationComposition's `controller <Composition.controller>`, and setting it sets the value
        of that Parameter (see `initial_seed <OptimizationControlMechanism.initial_seed>` for additional details).

    same_seed_for_all_parameter_combinations :  bool
        contains the setting for determining whether the random number generator used to select seeds for each
        estimate of the `model <ParameterEstimationComposition.model>`\\'s `net_outcome
        <ControlMechanism.net_outcome>` is re-initialized to the same value for each combination of `parameter
        <ParameterEstimationComposition>` values evaluated.  Its values is stored on the
        ParameterEstimationComposition's `controller <Composition.controller>`, and setting it sets the value
        of that Parameter (see `same_seed_for_all_allocations
        <OptimizationControlMechanism.same_seed_for_all_allocations>` for additional details).

    optimized_parameter_values : list
        contains the values of the `parameters <ParameterEstimationComposition.parameters>` of the `model
        <ParameterEstimationComposition.model>` that best fit the `data <ParameterEstimationComposition.data>` when
        the ParameterEstimationComposition is used for `ParameterEstimationComposition_Data_Fitting`,
        or that optimize performance of the `model <ParameterEstimationComposition.model>` according to the
        `optimization_function <ParameterEstimationComposition.optimization_function>` when the
        ParameterEstimationComposition is used for `ParameterEstimationComposition_Optimization`.  If `parameter values
        <ParameterEstimationComposition.parameter_ranges_or_priors>` are specified as ranges of values, then
        each item of `optimized_parameter_values` is the optimized value of the corresponding `parameter
        <ParameterEstimationComposition.parameter>`. If `parameter values
        <ParameterEstimationComposition.parameter_ranges_or_priors>` are specified as priors, then each item of
        `optimized_parameter_values` is an array containing the values of the corresponding `parameter
        <ParameterEstimationComposition.parameters>` the distribution of which were determined to be optimal.

    results : list[list[list]]
        contains the `output_values <Mechanism_Base.output_values>` of the `OUTPUT` `Nodes <Composition_Nodes>`
        in the `model <ParameterEstimationComposition.model>` for every `TRIAL <TimeScale.TRIAL>` executed (see
        `Composition.results` for more details). If the ParameterEstimationComposition is used for
        `ParameterEstimationComposition_Data_Fitting`, and `parameter values
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

    class Parameters(Composition.Parameters):
        """
            Attributes
            ----------

                initial_seed
                    see `input_specification <ParameterEstimationComposition.initial_seed>`

                    :default value: None
                    :type: ``int``

                same_seed_for_all_parameter_combinations
                    see `input_specification <ParameterEstimationComposition.same_seed_for_all_parameter_combinations>`

                    :default value: False
                    :type: ``bool``

        """
        # FIX: 11/32/21 CORRECT INITIAlIZATIONS?
        initial_seed = Parameter(None, loggable=False, pnl_internal=True,
                                 getter=_initial_seed_getter,
                                 setter=_initial_seed_setter)
        same_seed_for_all_parameter_combinations = Parameter(False, loggable=False, pnl_internal=True,
                                                             getter=_same_seed_for_all_parameter_combinations_getter,
                                                             setter=_same_seed_for_all_parameter_combinations_setter)

    @handle_external_context()
    @check_user_specified
    def __init__(self,
                 parameters, # OCM control_signals
                 outcome_variables,  # OCM monitor_for_control
                 optimization_function, # function of OCM
                 model=None,
                 data=None,
                 data_categorical_dims=None,
                 objective_function=None, # function of OCM ObjectiveMechanism
                 num_estimates=1, # num seeds per parameter combination (i.e., of OCM allocation_samples)
                 num_trials_per_estimate=None, # num trials per run of model for each combination of parameters
                 initial_seed=None,
                 same_seed_for_all_parameter_combinations=None,
                 name=None,
                 context=None,
                 **kwargs):

        self._validate_params(locals())

        # IMPLEMENTATION NOTE: this currently assigns pec as ocm.agent_rep (rather than model) to satisfy LLVM
        # Assign model as nested Composition of PEC
        if not model:
            # If model has not been specified, specification(s) in kwargs are used
            # (note: _validate_params() ensures that either model or nodes and/or pathways are specified, but not both)
            if 'nodes' in kwargs:
                nodes = convert_to_list(kwargs['nodes'])
                # A single Composition specified in nodes argument, so use as model
                if len(nodes) == 1 and isinstance(nodes[0], Composition):
                    model = nodes[0]

            elif 'pathways' in kwargs:
                pways = convert_to_list(kwargs['pathways'])
                # A single Composition specified in pathways arg, so use as model
                if len(pways) == 1 and isinstance(pways[0], Composition):
                    model = pways[0]
            else:
                # Use arguments provided to PEC in **nodes**, **pathways** and/or **projections** to construct model
                model = Composition(**kwargs, name='model')

            # Assign model as single node of PEC
            kwargs.update({'nodes': model})

        # Assign model as nested composition in PEC and self.model as self
        kwargs.update({'nodes': model})
        self.model = model

        self.optimized_parameter_values = []

        super().__init__(name=name,
                         controller_mode=BEFORE,
                         controller_time_scale=TimeScale.RUN,
                         enable_controller=True,
                         **kwargs)

        context = Context(source=ContextFlags.COMPOSITION, execution_id=None)

        # Assign parameters

        # Store the data used to fit the model, None if in OptimizationMode (the default)
        self.data = data
        self.data_categorical_dims = data_categorical_dims

        self.outcome_variables = outcome_variables

        # This internal list variable keeps track of the specific indices within the composition's output correspond
        # to the specified outcome variables. This is used in data fitting to subset the only the correct columns of the
        # simulation results for likelihood estimation.
        self._outcome_variable_indices = []

        if self.data is not None:
            self._validate_data()

        # If there is data being passed, then we are in data fitting mode and we need the OCM to return the full results
        # from a simulation of a composition.
        if self.data is not None:
            return_results = True
        else:
            return_results = False

        # Store the parameters specified for fitting
        self.fit_parameters = parameters

        # Implement OptimizationControlMechanism and assign as PEC controller
        # (Note: Implement after Composition itself, so that:
        #     - Composition's components are all available (limits need for deferred_inits)
        #     - search for seed params in _instantiate_ocm doesn't include pem itself or its functions)
        # IMPLEMENTATION NOTE: self is assigned as agent_rep to satisfy requirements of LLVM
        # TBI: refactor so that agent_rep = model
        ocm = self._instantiate_ocm(agent_rep = self,
                                    parameters=parameters,
                                    outcome_variables=outcome_variables,
                                    data=self.data,
                                    objective_function=objective_function,
                                    optimization_function=optimization_function,
                                    num_estimates=num_estimates,
                                    num_trials_per_estimate=num_trials_per_estimate,
                                    initial_seed=initial_seed,
                                    same_seed_for_all_parameter_combinations=same_seed_for_all_parameter_combinations,
                                    return_results=return_results,
                                    context=context)
        self.add_controller(ocm, context)

        # If we are using data fitting mode.
        # We need to ensure the aggregation function is set to None on the OptimizationFunction so that calls to
        # evaluate do not aggregate results of simulations. We want all results for all simulations so we can compute
        # the likelihood ourselves.
        if self.data is not None:
            ocm.function.parameters.aggregation_function._set(None, context)

        # The call run on PEC might lead to the run method again recursively for simulation. We need to keep track of
        # this to avoid infinite recursion.
        self._run_called = False

    def _validate_data(self):
        """Check if user supplied data to fit is valid for data fitting mode."""

        # If the data is not in numpy format (could be a pandas dataframe) convert it to numpy. Cast all values to
        # floats and keep track of categorical dimensions with a mask
        if isinstance(self.data, pd.DataFrame):
            self._data_numpy = self.data.to_numpy().astype(float)

            # Get which dimensions are categorical, and store the mask
            self.data_categorical_dims = [True if isinstance(t, pd.CategoricalDtype) or t == bool else False
                                          for t in self.data.dtypes]
        elif isinstance(self.data, np.ndarray) and self.data.ndim == 2:
            self._data_numpy = self.data

            # If no categorical dimensions are specified, assume all dimensions are continuous
            if self.data_categorical_dims is None:
                self.data_categorical_dims = [False for _ in range(self.data.shape[1])]
            else:
                # If the user specified a list of categorical dimensions, turn it into a mask
                x = np.array(self.data_categorical_dims)
                if x.dtype == int:
                    self.data_categorical_dims = np.arange(self.data.shape[1]).astype(bool)
                    self.data_categorical_dims[x] = True

        else:
            raise ValueError("Invalid format for data passed to OptimizationControlMechanism. Please ensure data is "
                             "either a 2D numpy array or a pandas dataframe. Each row represents a single experimental "
                             "trial.")

        if not isinstance(self.nodes[0], Composition):
            raise ValueError("PEC is data fitting mode requires the PEC to have a single node that is a composition!")

        # Make sure the output ports specified as outcome variables are present in the output ports of the inner
        # composition.
        in_comp = self.nodes[0]
        in_comp_ports = list(in_comp.output_CIM.port_map.keys())
        self._outcome_variable_indices = []
        for outcome_var in self.outcome_variables:
            try:
                self._outcome_variable_indices.append(in_comp_ports.index(outcome_var))
            except ValueError:
                raise ValueError(f"Could not find outcome variable {outcome_var.full_name} in the output ports of "
                                 f"the composition being fitted to data ({self.nodes[0]}). A current limitation of the "
                                 f"PEC data fitting API is that any output port of composition that should be fit to "
                                 f"data must be set as and output of the composition.")

        if len(self.outcome_variables) != self.data.shape[-1]:
            raise ValueError(f"The number of columns in the data to fit must match the length of outcome variables! "
                             f"data.colums = {self.data.columns}, outcome_variables = {self.outcome_variables}")

    def _validate_params(self, args):

        kwargs = args.pop('kwargs')
        pec_name = f"{self.__class__.__name__} '{args.pop('name',None)}'" or f'a {self.__class__.__name__}'

        # FIX: 11/3/21 - WRITE TESTS FOR THESE ERRORS IN test_parameter_estimation_composition.py

        # Must specify either model or a COMPOSITION_SPECIFICATION_ARGS
        if not (args['model'] or [arg for arg in kwargs if arg in COMPOSITION_SPECIFICATION_ARGS]):
        # if not ((args['model'] or args['nodes']) for arg in kwargs if arg in COMPOSITION_SPECIFICATION_ARGS):
            raise ParameterEstimationCompositionError(f"Must specify either 'model' or the "
                                                      f"'nodes', 'pathways', and/or `projections` ars "
                                                      f"in the constructor for {pec_name}.")

        # Can't specify both model and COMPOSITION_SPECIFICATION_ARGUMENTS
        # if (args['model'] and [arg for arg in kwargs if arg in COMPOSITION_SPECIFICATION_ARGS]):
        if args['model'] and kwargs.pop('nodes',None):
            raise ParameterEstimationCompositionError(f"Can't specify both 'model' and the "
                                                      f"'nodes', 'pathways', or 'projections' args "
                                                      f"in the constructor for {pec_name}.")

        # Disallow specification of PEC controller args
        ctlr_spec_args_found = [arg for arg in CONTROLLER_SPECIFICATION_ARGS if arg in list(kwargs.keys())]
        if ctlr_spec_args_found:
            plural = len(ctlr_spec_args_found) > 1
            raise ParameterEstimationCompositionError(f"Cannot specify the following controller arg"
                                                      f"{'s' if plural else ''} for {pec_name}: "
                                                      f"'{', '.join(ctlr_spec_args_found)}'; "
                                                      f"{'these are' if plural else 'this is'} "
                                                      f"set automatically.")

        # Disallow simultaneous specification of
        #     data (for data fitting; see _ParameterEstimationComposition_Data_Fitting)
        #          and objective_function (for optimization; see _ParameterEstimationComposition_Optimization)
        if args['data'] is not None and args['objective_function'] is not None:
            raise ParameterEstimationCompositionError(f"Both 'data' and 'objective_function' args were "
                                                      f"specified for {pec_name}; must choose one "
                                                      f"('data' for fitting or 'objective_function' for optimization).")

    def _instantiate_ocm(self,
                         agent_rep,
                         parameters,
                         outcome_variables,
                         data,
                         objective_function,
                         optimization_function,
                         num_estimates,
                         num_trials_per_estimate,
                         initial_seed,
                         same_seed_for_all_parameter_combinations,
                         return_results,
                         context=None
                         ):

        # # Parse **parameters** into ControlSignals specs
        control_signals = []
        for param, allocation in parameters.items():
            control_signals.append(ControlSignal(modulates=param,
                                                 # In parameter fitting (when data is present) we always want to
                                                 # override the fitting parameters with the search values.
                                                 modulation=OVERRIDE if self.data is not None else None,
                                                 allocation_samples=allocation))

        # If objective_function has been specified, create and pass ObjectiveMechanism to ocm
        objective_mechanism = ObjectiveMechanism(monitor=outcome_variables,
                                                 function=objective_function) if objective_function else None

        # FIX: NEED TO BE SURE CONSTRUCTOR FOR MLE optimization_function HAS data ATTRIBUTE
        if data is not None:
            optimization_function.data = self._data_numpy
            optimization_function.data_categorical_dims = self.data_categorical_dims
            optimization_function.outcome_variable_indices = self._outcome_variable_indices

        return PEC_OCM(
            agent_rep=agent_rep,
            monitor_for_control=outcome_variables,
            allow_probes=True,
            objective_mechanism=objective_mechanism,
            function=optimization_function,
            control_signals=control_signals,
            num_estimates=num_estimates,
            num_trials_per_estimate=num_trials_per_estimate,
            initial_seed=initial_seed,
            same_seed_for_all_allocations=same_seed_for_all_parameter_combinations,
            context=context,
            return_results=return_results,
        )

    @handle_external_context()
    def run(self, *args, **kwargs):

        # Clear any old results from the composition
        if self.results is not None:
            self.results.clear()

        context = kwargs.get('context', None)
        self._assign_execution_ids(context)

        # Capture the input passed to run and pass it on to the OCM
        assert self.controller is not None
        self.controller._cache_pec_inputs(kwargs.get('inputs', None if not args else args[0]))
        # We need to set the inputs for the composition during simulation, by assigning the inputs dict passed in
        # PEC run() to its controller's state_feature_values (this is in order to accomodate multi-trial inputs
        # without having the PEC provide them one-by-one to the simulated composition. This assumes that the inputs
        # dict has the inputs specified in the same order as the state features (i.e., as specified by
        # PEC.get_input_format()), and rearranges them so that each node gets a full trial's worth of inputs.
        inputs_dict = self.controller.parameters.state_feature_values._get(context)
        # inputs_dict = self.controller._get_pec_inputs()

        for state_input_port, value in zip(self.controller.state_input_ports, inputs_dict.values()):
            state_input_port.parameters.value._set(value, context)
        # Need to pass restructured inputs dict to run
        # kwargs['inputs'] = {self.nodes[0]: list(inputs_dict.values())}
        kwargs.pop('inputs', None)
        # Run the composition as normal
        return super(ParameterEstimationComposition, self).run(*args, **kwargs)

    @handle_external_context()
    def log_likelihood(self, *args, inputs=None, context=None) -> float:
        """
        Compute the log-likelihood of the data given the specified parameters of the model.

        Arguments
        ---------
        *args :
            Positional args, one for each paramter of the model. These must correspond directly to the parameters that
            have been specified in the `parameters` argument of the constructor.

        Returns
        -------
        The sum of the log-likelihoods of the data given the specified parameters of the model.

        """

        if self.controller is None:
            raise ParameterEstimationCompositionError(f"The controller for ParameterEstimationComposition {self.name} "
                                                      f"has not been instantiated yet. Cannot compute log-likelihood.")

        if self.controller.function is None:
            raise ParameterEstimationCompositionError(f"The function of the controller for "
                                                      f"ParameterEstimationComposition {self.name} has not been "
                                                      f"instantiated yet. Cannot compute log-likelihood.")

        if self.data is None:
            raise ParameterEstimationCompositionError(f"The data for ParameterEstimationComposition {self.name} "
                                                      f"has not been defined. Cannot compute log-likelihood.")

        if len(args) != len(self.fit_parameters):
            raise ParameterEstimationCompositionError(f"The number of parameters specified in the call to "
                                                      f"log_likelihood does not match the number of parameters "
                                                      f"specified in the constructor of ParameterEstimationComposition.")

        if not hasattr(self.controller.function, 'log_likelihood'):
            of = self.controller.function
            raise ParameterEstimationCompositionError(f"The function ({of}) for the controller of "
                                                      f"ParameterEstimationComposition {self.name} does not appear to "
                                                      f"have a log_likelihood function.")

        context.composition = self

        # Capture the inputs and pass it on to the OCM
        assert self.controller is not None
        self.controller._cache_pec_inputs(inputs)

        # Try to get the log-likelihood from controllers optimization_function, if it hasn't defined this function yet
        # then it will raise an error.
        return self.controller.function.log_likelihood(*args, context=context)

    def _complete_init_of_partially_initialized_nodes(self, context):
        pass


def _pec_ocm_state_feature_values_getter(owning_component=None, context=None)->dict:
    """Return the complete input values passed to the last call of run for the Composition that the PEC_OCM controls.
    This method is used by the PEC_OCM to get the complete input dictionary for all trials cached in _pec.input_values,
    in order to pass them on to the agent_rep during simulation.
    """
    pec_ocm = owning_component

    if pec_ocm.initialization_status == ContextFlags.INITIALIZING or pec_ocm._pec_input_values == None:
        return {}

    if not isinstance(pec_ocm.composition, ParameterEstimationComposition):
        raise ParameterEstimationCompositionError(
            f"A PEC_OCM can only be used with a ParmeterEstimationComposition")

    return pec_ocm._pec_input_values


class PEC_OCM(OptimizationControlMechanism):
    """OptimizationControlMechanism specialized for use with ParameterEstimationComposition
    Assign inputs passed to run method of ParameterEstimationComposition directly as values of
      PEC_OCM's state_input_ports (this allows a full set of trials' worth of inputs to be used in each
      run of the Composition being estimated or optimized.
    _cache_pec_inputs(): called by PEC to cache inputs passed to its run method
    _pec_ocm_state_feature_values_getter(): overrides state_feature_values_getter of OptimizationControlMechanism
      to return input dict for simulation that incluces all trials' worth of inputs for each node.
    """
    class Parameters(OptimizationControlMechanism.Parameters):
        """
        Attributes
        ----------
            state_feature_values
                overrides `state_feature_values <OptimizationControlMechanism.state_feature_values` to
                assign inputs provided to run() method of ParameterEstimationComposition, and cached in
                pec_ocm._pec_input_values, that returns inputs reformatted to provide full set of trials'
                worth of inputs to each node of Composition being estimated or optimized.
                :default value: {}
                :type: dict
        """
        state_feature_values = Parameter(None, getter=_pec_ocm_state_feature_values_getter,
                                         user=False, pnl_internal=True, read_only=True)

    def __init__(self, *args, **kwargs):
        self._pec_input_values = None
        super().__init__(*args, **kwargs)

    def _cache_pec_inputs(self, inputs_dict:dict)->dict:
        """Cache input values passed to the last call of run for the composition that this OCM controls.
        This method is used by the ParamterEstimationComposition in its run() method.
        If inputs_dict is of the form specified by ParemeterEstimationComposition.get_input_format()
          ({model: inputs_array}, in which each item in the outer dimension of inputs_array is a trial's
          worth of inputs, with one input for each of the pec_ocm.state_input_ports) then inputs_dict is
          simply assigned to _pec_input_values.
        If inputs_dict is formatted as the input to model (i.e., of the form model.get_input_format(),
          it is refactored to the format required as input to the ParemeterEstimationComposition described above.
        """

        model = self.composition.model

        # If inputs_dict has model as its only entry, then check that its format is OK to pass to pec.run()
        if len(inputs_dict) == 1 and model in inputs_dict:
            if len(inputs_dict) != self.num_state_input_ports:
                raise ParameterEstimationCompositionError(f"The array in the dict specified for the 'inputs' arg of "
                                                          f"ParameterEstimationMechanism.run() is badly formatted: "
                                                          f"the outer dimension should be equal to the number of inputs "
                                                          f"to '{model.name}' ")

        else:
            # Restructure inputs as nd array with each row (outer dim) a trial's worth of inputs
            #    and each item in the row (inner dim) the input to a node (or input_port) for that trial
            if len(inputs_dict) != self.num_state_input_ports:
                raise ParameterEstimationCompositionError(f"The dict specified in the `input` arg of "
                                                          f"ParameterEstimationMechanism.run() is badly formatted: "
                                                          f"the number of entries should equal the number of inputs to "
                                                          f"'{model.name}' ")
            trial_seqs = list(inputs_dict.values())
            num_trials = len(trial_seqs[0])
            input_values = [[] for _ in range(num_trials)]
            for trial in range(num_trials):
                for trial_seq in trial_seqs:
                    if len(trial_seq) != num_trials:
                        raise ParameterEstimationCompositionError(f"The dict specified in the `input` arg of "
                                                                  f"ParameterEstimationMechanism.run() is badly formatted: "
                                                                  f"every entry must have the same number of inputs.")
                    # input_values[trial].append(np.array([trial_seq[trial].tolist()]))
                    input_values[trial].extend(trial_seq[trial])
            inputs_dict = {model: input_values}

            self._pec_input_values = inputs_dict
