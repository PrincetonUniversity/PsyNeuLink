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

A `ParameterEstimationComposition` is a subclass of `Composition` that is used to estimate specified `parameters
<ParameterEstimationComposition.parameters>` of a `model <ParameterEstimationComposition.model>` Composition,
in order to fit the `outputs <ParameterEstimationComposition.outcome_variables>`
of the `model <ParameterEstimationComposition.model>` to a set of data (`ParameterEstimationComposition_Data_Fitting`)
via likelihood maximization using kernel density estimation (KDE), or to optimize a user provided scalar
`objective_function` (`ParameterEstimationComposition_Optimization`). In either case, when the
ParameterEstimationComposition is `run <Composition.run>` with a given set of `inputs <Composition_Execution_Inputs>`,
it returns the set of parameter values in its `optimized_parameter_values
<ParameterEstimationComposition.optimized_parameter_values>` attribute that it estimates best satisfy either of those
conditions. The `results <ParameterEstimationComposition.results>` attribute are also set to the optimal parameter
values.  The arguments below are used to configure a ParameterEstimationComposition for either
`ParameterEstimationComposition_Data_Fitting` or `ParameterEstimationComposition_Optimization`, followed by sections
that describe arguments specific to each.

    .. _ParameterEstimationComposition_Model:

    * **model** - specifies the `Composition` whose `parameters
      <ParameterEstimationComposition.parameters>` are to be estimated.

      .. note::
         Neither the **controller** nor any of its associated arguments can be specified in the constructor for a
         ParameterEstimationComposition; this is constructed automatically using the arguments described below.

    * **parameters** - specifies the `parameters <ParameterEstimationComposition.parameters>` of the `model
      <ParameterEstimationComposition.model>` to be estimated.  These are specified in a dict, in which the key
      of each entry specifies a parameter to estimate, and its value is a list values to sample for that
      parameter.

    * **outcome_variables** - specifies the `OUTPUT` `Nodes <Composition_Nodes>` of the `model
      <ParameterEstimationComposition.model>`, the `values <Mechanism_Base.value>` of which are used to evaluate the
      fit of the different combinations of `parameter <ParameterEstimationComposition.parameters>` values sampled. An
      important limitation of the PEC is that the `outcome_variables <ParameterEstimationComposition.outcome_variables>`
      must be a subset of the output ports of the `model <ParameterEstimationComposition.model>`'s terminal Mechanism.

    * **optimization_function** - specifies the function used to search over the combinations of `parameter
      <ParameterEstimationComposition.parameters>` values to be estimated. This must be either an instance of
      `PECOptimizationFunction` or a string name of one of the supported optimizers.

    * **num_estimates** - specifies the number of independent samples that are estimated for a given combination of
      `parameter <ParameterEstimationComposition.parameters>` values.

    * **num_trials_per_estimate** - specifies the number of trials executed when the `model <ParameterEstimationComposition.model>`
      is run for each estimate of a combination of `parameter <ParameterEstimationComposition.parameters>` values.
      Typically, this can be left unspecified and the `model <Composition>` will be run until all trials of inputs are
      exhausted.

.. _ParameterEstimationComposition_Data_Fitting:

Data Fitting
------------

The ParameterEstimationComposition can be used to find a set of parameters for the `model
<ParameterEstimationComposition.model>` such that, when it is run with a given set of inputs, its results
best match (maximum likelihood) a specified set of empirical data. This requires that the **data** argument be
specified:

    .. _ParameterEstimationComposition_Data:

    * **data** - specifies the data to which the `outcome_variables <ParameterEstimationComposition.outcome_variables>`
      are fit in the estimation process.  They must be in a format that aligns with the specification of
      the `outcome_variables <ParameterEstimationComposition.outcome_variables>`. The parameter data should be a
      pandas DataFrame where each column corresponds to one of the
      `outcome_variables <ParameterEstimationComposition.outcome_variables>`. If one of the outcome variables should be
      treated as a categorical variable (e.g. a decision value in a two-alternative forced choice task modeled by a
      DDM), the it should be specified as a pandas Categorical variable.

    .. technical_note::
    * **objective_function** - A function that computes the sum of the log likelihood of the data is automatically
      assigned for data fitting purposes and should not need to be specified. This function uses a kernel density
      estimation of the data to compute the likelihood of the data given the model. If you would like to use your own
      estimation of the likelhood, see `ParameterEstimationComposition_Optimization` below.

    .. warning::
       The **objective_function** argument should NOT be specified for data fitting; specifying both the
       **data** and **objective_function** arguments generates an error.

.. _ParameterEstimationComposition_Optimization:

Parameter Optimization
----------------------

The ParameterEstimationComposition can be used to find a set of parameters for the `model
<ParameterEstimationComposition.model>` such that, when it is run with a given set of inputs, its results
either maximize or minimize the **objective_function**, as determined by the **optimization_function**. This
requires that the **objective_function** argument be specified:

    .. _ParameterEstimationComposition_Objective_Function:

    * **objective_function** - specifies a function used to evaluate the `values <Mechanism_Base.value>` of the
      `outcome_variables <ParameterEstimationComposition.outcome_variables>`, according to which combinations of
      `parameters <ParameterEstimationComposition.parameters>` are assessed; this must be an `Callable`
      that takes a 3d array as its only argument, the shape of which will be (**num_estimates**, **num_trials**,
      number of **outcome_variables**).  The function should specify how to aggregate the value of each
      **outcome_variable** over **num_estimates** and/or **num_trials** if either is greater than 1.

    .. warning::
       The **data** argument should NOT be specified for parameter optimization;  specifying both the
       **objective_function** and the **data** arguments generates an error.

.. _ParameterEstimationComposition_Supported_Optimizers:

Supported Optimizers
--------------------

- `DifferentialEvolution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html>`_

Structure
---------

.. technical_note::
   ParameterEstimationComposition uses a `PEC_OCM` as its `controller <Composition.controller>` -- a specialized
   subclass of `OptimizationControlMechanism` that intercepts inputs provided to the `run
   <ParameterEstimationComposition.run>` method of the ParameterEstimationComposition, and assigns them directly
   to the `state_feature_values` of the PEC_OCM when it executes.

.. _ParameterEstimationComposition_Class_Reference:

Class Reference
---------------

"""
import warnings

import numpy as np
import pandas as pd

from beartype import beartype

from psyneulink._typing import Optional, Union, Dict, List, Callable, Literal, Mapping

import psyneulink.core.llvm as pnllvm
from psyneulink.core.globals.utilities import ContentAddressableList, convert_to_np_array
from psyneulink.core.components.shellclasses import Mechanism
from psyneulink.core.compositions.composition import Composition, CompositionError, NodeRole
from psyneulink.core.components.ports.port import Port_Base
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.modulatory.control.optimizationcontrolmechanism import (
    OptimizationControlMechanism,
)
from psyneulink.core.components.functions.nonstateful.fitfunctions import (
    PECOptimizationFunction,
    simulation_likelihood,
)
from psyneulink.core.components.ports.modulatorysignals.controlsignal import (
    ControlSignal,
)
from psyneulink.core.globals.context import (
    Context,
    ContextFlags,
    handle_external_context,
)
from psyneulink.core.globals.keywords import BEFORE, OVERRIDE
from psyneulink.core.globals.parameters import Parameter, SharedParameter, check_user_specified
from psyneulink.core.globals.utilities import convert_all_elements_to_np_array, convert_to_list
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.globals.defaults import defaultControlAllocation



__all__ = ["ParameterEstimationComposition", "ParameterEstimationCompositionError"]

COMPOSITION_SPECIFICATION_ARGS = {"nodes", "pathways", "projections"}
CONTROLLER_SPECIFICATION_ARGS = {
    "controller",
    "enable_controller",
    "controller_mode",
    "controller_time_scale",
    "controller_condition",
    "retain_old_simulation_data",
}


class ParameterEstimationCompositionError(CompositionError):
    pass


class ParameterEstimationComposition(Composition):
    """
    Subclass of `Composition` that estimates specified parameters either to fit the results of a Composition
    to a set of data or to optimize a specified function.

    Automatically implements an `OptimizationControlMechanism` as its `controller <Composition.controller>`,
    that is constructed using arguments to the ParameterEstimationComposition's constructor as described below.

    The following arguments are those specific to ParmeterEstimationComposition; see `Composition` for additional
    arguments

    Arguments
    ---------

    parameters :
        specifies the `parameters <ParameterEstimationComposition.parameters>` of the `model
        <ParameterEstimationComposition.model>` to be estimated.  These are specified in a dict, in which the key
        of each entry specifies a parameter to estimate, and its value is a list values to sample for that
        parameter.

    depends_on :
        A dictionary that specifies which parameters depend on a condition. The keys of the dictionary are the
        specified identically to the keys of the parameters dictionary. The values are a string that specifies a
        column in the data that the parameter depends on. The values of this column must be categorical. Each unique
        value will represent a condition and will result in a separate parameter being estimated for it. The number of
        unique values should be small because each unique value will result in a separate parameter being estimated.

    outcome_variables :
        specifies the `OUTPUT` `Nodes <Composition_Nodes>` of the `model
        <ParameterEstimationComposition.model>`, the `values <Mechanism_Base.value>` of which are used to evaluate the
        fit of the different combinations of `parameter <ParameterEstimationComposition.parameters>` values sampled. An
        important limitation of the PEC is that the `outcome_variables <ParameterEstimationComposition.outcome_variables>`
        must be a subset of the output ports of the `model <ParameterEstimationComposition.model>`'s terminal `Mechanism`.

    model :
        specifies an external `Composition` for which parameters are to be `fit to data
        <ParameterEstimationComposition_Data_Fitting>` or `optimized <ParameterEstimationComposition_Optimization>`
        according to a specified `objective_function <ParameterEstimationComposition.objective_function>`.

    data :
        specifies the data to which the `outcome_variables <ParameterEstimationComposition.outcome_variables>`
        are fit in the estimation process.  They must be in a format that aligns with the specification of
        the `outcome_variables <ParameterEstimationComposition.outcome_variables>`. The parameter data should be a
        pandas DataFrame where each column corresponds to one of the
        `outcome_variables <ParameterEstimationComposition.outcome_variables>`. If one of the outcome variables should be
        treated as a categorical variable (e.g. a decision value in a two-alternative forced choice task modeled by a
        DDM), the it should be specified as a pandas Categorical variable.

    data_categorical_dims : Union[Iterable] : default None
        specifies the dimensions of the data that are categorical. If a list of boolean values is provided, it is
        assumed to be a mask for the categorical data dimensions and must have the same length as columns in data. If
        it is an iterable of integers, it is assumed to be a list of the categorical dimensions indices. If it is None,
        all data dimensions are assumed to be continuous. Alternatively, if data is a pandas DataFrame, then the columns
        which have Category dtype are assumed to be categorical.

    objective_function : ObjectiveFunction, function or method
        specifies the function used by **optimization_function** (see `objective_function
        <ParameterEstimationComposition.objective_function>` for additional information);  the shape of its `variable
        <Component.variable>` argument (i.e., its first positional argument) must be the same as an
        array containing the `value <OutputPort.value>` of the OutputPort corresponding to each item specified in
        `outcome_variables <ParameterEstimationComposition.outcome_variables>`.

    optimization_function : OptimizationFunction, function or method : default or MaximumLikelihood or GridSearch
        specifies the function used to search over the combinations of `parameter
        <ParameterEstimationComposition.parameters>` values to be estimated. This must be either an instance of
        `PECOptimizationFunction` or a string name of one of the supported optimizers.

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
        <ParameterEstimationComposition.parameters>` values.  It is passed to the ParameterEstimationComposition's
        `OptimizationControlMechanism` as the function of its `objective_mechanism
        <ControlMechanism.objective_mechanism>`, that is used to compute the `net_outcome
        <ControlMechanism.net_outcome>` for of the `model <ParameterEstimationComposition.model>` each time it is
        `run <Composition.run>` (see `objective_function <ParameterEstimationComposition_Objective_Function>`
        for additional details).

    optimization_function : OptimizationFunction
        determines the function used to estimate the parameters of the `model <ParameterEstimationComposition.model>`
        that either best fit the `data <ParameterEstimationComposition.data>` when the ParameterEstimationComposition
        is used for `ParameterEstimationComposition_Data_Fitting`, or that achieve some maximum or minimum value of
        the `optimization_function <ParameterEstimationComposition.optimization_function>` when the
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

        .. note::
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

    optimal_value : float
        contains the results returned by execution of `agent_rep <OptimizationControlMechanism.agent_rep>` for the
        parameter values in `optimized_parameter_values <ParameterEstimationComposition.optimized_parameter_values>`.

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
        initial_seed = SharedParameter(attribute_name='controller')
        same_seed_for_all_parameter_combinations = SharedParameter(attribute_name='controller')

    @handle_external_context()
    @check_user_specified
    @beartype
    def __init__(
        self,
        parameters: Dict,
        outcome_variables: Union[
            List[Mechanism], Mechanism, List[OutputPort], OutputPort
        ],
        optimization_function: Union[
            PECOptimizationFunction,
            Literal["differential_evolution"],
            Literal["grid_search"],
        ],
        model: Optional[Composition] = None,
        data: Optional[pd.DataFrame] = None,
        likelihood_include_mask: Optional[np.ndarray] = None,
        data_categorical_dims=None,
        objective_function: Optional[Callable] = None,
        num_estimates: int = 1,
        num_trials_per_estimate: Optional[int] = None,
        initial_seed: Optional[int] = None,
        same_seed_for_all_parameter_combinations: Optional[bool] = None,
        depends_on: Optional[Mapping] = None,
        name: Optional[str] = None,
        context: Optional[Context] = None,
        **kwargs,
    ):
        # We don't allow user specified controllers in PEC
        if "controller" in kwargs:
            raise ValueError(
                "controller argument cannot be specified in a ParameterEstimationComposition. PEC sets "
                "up its own controller for executing its parameter estimation process."
            )

        # If the number of trials per estimate is not specified and we are fitting to data then
        # get it from the data.
        if num_trials_per_estimate is None and data is not None:
            num_trials_per_estimate = len(data)

        self._validate_params(locals().copy())

        # IMPLEMENTATION NOTE: this currently assigns pec as ocm.agent_rep (rather than model) to satisfy LLVM
        # Assign model as nested Composition of PEC
        if not model:
            # If model has not been specified, specification(s) in kwargs are used
            # (note: _validate_params() ensures that either model or nodes and/or pathways are specified, but not both)
            if "nodes" in kwargs:
                nodes = convert_to_list(kwargs["nodes"])
                # A single Composition specified in nodes argument, so use as model
                if len(nodes) == 1 and isinstance(nodes[0], Composition):
                    model = nodes[0]

            elif "pathways" in kwargs:
                pways = convert_to_list(kwargs["pathways"])
                # A single Composition specified in pathways arg, so use as model
                if len(pways) == 1 and isinstance(pways[0], Composition):
                    model = pways[0]
            else:
                # Use arguments provided to PEC in **nodes**, **pathways** and/or **projections** to construct model
                model = Composition(**kwargs, name="model")

            # Assign model as single node of PEC
            kwargs.update({"nodes": model})

        # Assign model as nested composition in PEC and self.model as self
        kwargs.update({"nodes": model})
        self.model = model

        self.depends_on = depends_on

        # These will be assigned in _validate_date if depends_on is not None
        self.cond_levels = None
        self.cond_mask = None
        self.cond_data = None

        self.optimized_parameter_values = []

        self.pec_control_mechs = {}
        for (pname, mech), values in parameters.items():
            self.pec_control_mechs[(pname, mech)] = ControlMechanism(name=f"{pname}_control",
                                                                     control_signals=[(pname, mech)],
                                                                     modulation=OVERRIDE)
            self.model.add_node(self.pec_control_mechs[(pname, mech)])

        super().__init__(
            name=name,
            controller_mode=BEFORE,
            controller_time_scale=TimeScale.RUN,
            enable_controller=True,
            **kwargs,
        )

        context = Context(source=ContextFlags.COMPOSITION, execution_id=None)

        # Assign parameters

        # Store the data used to fit the model, None if in OptimizationMode (the default)
        self.data = data
        self.data_categorical_dims = data_categorical_dims

        if not isinstance(self.nodes[0], Composition):
            raise ValueError(
                "PEC requires the PEC to have a single node that is a composition!"
            )

        # This internal list variable keeps track of the specific indices within the composition's output correspond
        # to the specified outcome variables. This is used in data fitting to subset the only the correct columns of the
        # simulation results for likelihood estimation.
        # Make sure the output ports specified as outcome variables are present in the output ports of the inner
        # composition.
        self.outcome_variables = outcome_variables

        try:
            iter(self.outcome_variables)
        except TypeError:
            self.outcome_variables = [self.outcome_variables]

        self._outcome_variable_indices = []
        in_comp = self.nodes[0]
        for outcome_var in self.outcome_variables:
            try:
                if not isinstance(outcome_var, OutputPort):
                    outcome_var = outcome_var.output_port

                # Get the index of the outcome variable in the output ports of inner composition. To do this,
                # we must use the inner composition's portmap to get the CIM output port that corresponds to
                # the outcome variable
                index = in_comp.output_ports.index(in_comp.output_CIM.port_map[outcome_var][1])

                self._outcome_variable_indices.append(index)
            except KeyError:
                raise KeyError(
                    f"Could not find outcome variable {outcome_var.full_name} in the output ports of "
                    f"the composition being fitted to data ({self.nodes[0]}). A current limitation of the "
                    f"PEC data fitting API is that any output port of composition that should be fit to "
                    f"data must be set as and output of the composition."
                )

        # Validate data if it is provided, need to do this now because this method also checks if
        # the data is compatible with outcome variables determined above
        if self.data is not None:
            self._validate_data()

            if likelihood_include_mask is not None:

                # Make sure the length is correct
                if len(likelihood_include_mask) != len(self.data):
                    raise ValueError(
                        "Likelihood include mask must be the same length as the number of rows in the data!")

                # If the include mask is 2D, make it 1D
                if likelihood_include_mask.ndim == 2:
                    likelihood_include_mask = likelihood_include_mask.flatten()

                self.likelihood_include_mask = likelihood_include_mask

            else:
                self.likelihood_include_mask = np.ones(len(self.data), dtype=bool)

        # Store the parameters specified for fitting
        self.fit_parameters = parameters

        # Implement OptimizationControlMechanism and assign as PEC controller
        # (Note: Implement after Composition itself, so that:
        #     - Composition's components are all available (limits need for deferred_inits)
        #     - search for seed params in _instantiate_ocm doesn't include pem itself or its functions)
        # IMPLEMENTATION NOTE: self is assigned as agent_rep to satisfy requirements of LLVM
        # TBI: refactor so that agent_rep = model
        ocm = self._instantiate_ocm(
            agent_rep=self,
            parameters=parameters,
            outcome_variables=outcome_variables,
            data=self.data,
            objective_function=objective_function,
            optimization_function=optimization_function,
            num_estimates=num_estimates,
            num_trials_per_estimate=num_trials_per_estimate,
            initial_seed=initial_seed,
            same_seed_for_all_parameter_combinations=same_seed_for_all_parameter_combinations,
            context=context,
        )
        self.add_controller(ocm, context)

        # In both optimization mode and data fitting mode, the PEC does not need an aggregation function to
        # combine results across the randomized dimension. We need to ensure the aggregation function is set to None on
        # the OptimizationFunction so that calls to evaluate do not aggregate results of simulations. We want all
        # results for all simulations so we can compute the likelihood ourselves.
        ocm.function.parameters.aggregation_function._set(None, context)

        # The call run on PEC might lead to the run method again recursively for simulation. We need to keep track of
        # this to avoid infinite recursion.
        self._run_called = False

    def _validate_data(self):
        """Check if user supplied data to fit is valid for data fitting mode."""

        # If there is a depends_on attribute, the user is doing a conditional parameterization. The data must be a
        # pandas dataframe, and we must strip out any columns that parameters are marked to depend on. These columns
        # should be categorical or string columns.
        if self.depends_on:
            if not isinstance(self.data, pd.DataFrame):
                raise ValueError(
                    "If using conditional parameterization, the data must be a pandas dataframe."
                )

            # Check if the dependent columns are in the data
            for param, col in self.depends_on.items():
                if col not in self.data.columns:
                    raise ValueError(f"The data does not contain the column '{col}' that parameter '{param}' "
                                     f"is dependent on.")

                # If the column is string, convert to categorical
                if self.data[col].dtype == object:
                    self.data[col] = self.data[col].astype('category')

                # If the column is not categorical, return and error
                if not self.data[col].dtype.name == 'category':
                    raise ValueError(f"The column '{col}' that parameter '{param}' is dependent on must be a string or"
                                     f" categorical column.")

                # Make sure the column does not have too many unique values.
                if len(self.data[col].unique()) > 5:
                    warnings.warn(f"Column '{col}' has more than 5 unique values. Values = {self.data[col].unique()}. "
                                  f"Each unique value will be treated as a separate condition. This may lead to a "
                                  f"large number of parameters to estimate. Consider reducing the number of unique "
                                  f"values in this column.")

            # Get a separate copy of the dataframe with conditional columns
            self.cond_data = self.data[list(set(self.depends_on.values()))].copy()

            # For each value in depends_on, get the unique levels of the column. This will determine the number of
            # of conditional parameters that need to be estimated for that parameter.
            self.cond_levels = {param: self.cond_data[col].unique() for param, col in self.depends_on.items()}

            # We also need a mask to keep track of which trials are associated with which condition
            self.cond_mask = {}
            for param, col in self.depends_on.items():
                self.cond_mask[param] = {}
                for level in self.cond_levels[param]:
                    self.cond_mask[param][level] = self.cond_data[col] == level

            # Remove the dependent columns from the data
            self.data = self.data.drop(columns=self.depends_on.values())

        # If the data is not in numpy format (could be a pandas dataframe) convert it to numpy. Cast all values to
        # floats and keep track of categorical dimensions with a mask. This preprocessing is done to make the data
        # compatible with passing directly to simulation_likelihood function. This avoids having to do the same with
        # each call to the likelihood function during optimization.
        if isinstance(self.data, pd.DataFrame):
            self._data_numpy = self.data.to_numpy().astype(float)

            # Get which dimensions are categorical, and store the mask
            self.data_categorical_dims = [
                True if isinstance(t, pd.CategoricalDtype) or t == bool else False
                for t in self.data.dtypes
            ]
        elif isinstance(self.data, np.ndarray) and self.data.ndim == 2:
            self._data_numpy = self.data

            # If no categorical dimensions are specified, assume all dimensions are continuous
            if self.data_categorical_dims is None:
                self.data_categorical_dims = [False for _ in range(self.data.shape[1])]
            else:
                # If the user specified a list of categorical dimensions, turn it into a mask
                x = np.array(self.data_categorical_dims)
                if x.dtype == int:
                    self.data_categorical_dims = np.arange(self.data.shape[1]).astype(
                        bool
                    )
                    self.data_categorical_dims[x] = True

        else:
            raise ValueError(
                "Invalid format for data passed to OptimizationControlMechanism. Please ensure data is "
                "either a 2D numpy array or a pandas dataframe. Each row represents a single experimental "
                "trial."
            )

        if len(self.outcome_variables) != self.data.shape[-1]:
            raise ValueError(
                f"The number of columns in the data to fit must match the length of outcome variables! "
                f"data.colums = {self.data.columns}, outcome_variables = {self.outcome_variables}"
            )

    def _validate_params(self, args):
        kwargs = args.pop("kwargs")
        pec_name = (
            f"{self.__class__.__name__} '{args.pop('name', None)}'"
            or f"a {self.__class__.__name__}"
        )

        # FIX: 11/3/21 - WRITE TESTS FOR THESE ERRORS IN test_parameter_estimation_composition.py

        # Must specify either model or a COMPOSITION_SPECIFICATION_ARGS
        if not (
            args["model"]
            or [arg for arg in kwargs if arg in COMPOSITION_SPECIFICATION_ARGS]
        ):
            # if not ((args['model'] or args['nodes']) for arg in kwargs if arg in COMPOSITION_SPECIFICATION_ARGS):
            raise ParameterEstimationCompositionError(
                f"Must specify either 'model' or the "
                f"'nodes', 'pathways', and/or `projections` ars "
                f"in the constructor for {pec_name}."
            )

        # Can't specify both model and COMPOSITION_SPECIFICATION_ARGUMENTS
        # if (args['model'] and [arg for arg in kwargs if arg in COMPOSITION_SPECIFICATION_ARGS]):
        if args["model"] and kwargs.pop("nodes", None):
            raise ParameterEstimationCompositionError(
                f"Can't specify both 'model' and the "
                f"'nodes', 'pathways', or 'projections' args "
                f"in the constructor for {pec_name}."
            )

        # Disallow specification of PEC controller args
        ctlr_spec_args_found = [
            arg for arg in CONTROLLER_SPECIFICATION_ARGS if arg in list(kwargs.keys())
        ]
        if ctlr_spec_args_found:
            plural = len(ctlr_spec_args_found) > 1
            raise ParameterEstimationCompositionError(
                f"Cannot specify the following controller arg"
                f"{'s' if plural else ''} for {pec_name}: "
                f"'{', '.join(ctlr_spec_args_found)}'; "
                f"{'these are' if plural else 'this is'} "
                f"set automatically."
            )

        # Disallow simultaneous specification of
        #     data (for data fitting; see _ParameterEstimationComposition_Data_Fitting)
        #          and objective_function (for optimization; see _ParameterEstimationComposition_Optimization)
        if args["data"] is not None and args["objective_function"] is not None:
            raise ParameterEstimationCompositionError(
                f"Both 'data' and 'objective_function' args were "
                f"specified for {pec_name}; must choose one "
                f"('data' for fitting or 'objective_function' for optimization)."
            )

    def _instantiate_ocm(
        self,
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
        context=None,
    ):

        # For the PEC, the objective mechanism is not needed because in context of optimization of data fitting
        # we require all trials (and number of estimates) to compute the scalar objective value. In data fitting
        # this is usually and likelihood estimated by kernel density estimation using the simulated data and the
        # user provided data. For optimization, it is computed arbitrarily by the user provided objective_function
        # to the PEC. Either way, the objective_mechanism is not the appropriate place because it gets
        # executed on each trial's execution.
        objective_mechanism = None

        # if data is specified and objective_function is None, define maximum likelihood estimation objective function
        if data is not None and objective_function is None:
            # Create an objective function that computes the negative sum of the log likelihood of the data,
            # so we can perform maximum likelihood estimation. This will be our objective function in
            # data fitting mode.
            def f(sim_data):
                like = simulation_likelihood(
                    sim_data=sim_data,
                    exp_data=self._data_numpy,
                    categorical_dims=self.data_categorical_dims,
                )

                return np.sum(np.log(like[self.likelihood_include_mask]))

            objective_function = f

        if optimization_function is None:
            warnings.warn(
                "optimization_function argument to PEC was not specified, defaulting to gridsearch, this is slow!"
            )
            optimization_function = PECOptimizationFunction(
                method="gridsearch", objective_function=objective_function
            )
        elif type(optimization_function) == str:
            optimization_function = PECOptimizationFunction(
                method=optimization_function, objective_function=objective_function
            )
        elif not isinstance(optimization_function, PECOptimizationFunction):
            raise ParameterEstimationCompositionError(
                "optimization_function for PEC must either be either a valid "
                "string for a supported optimization method or an instance of "
                "PECOptimizationFunction."
            )
        else:
            optimization_function.set_pec_objective_function(objective_function)

        if data is not None:
            optimization_function.data_fitting_mode = True
        else:
            optimization_function.data_fitting_mode = False

        # I wish I had a cleaner way to do this. The optimization function doesn't have any way to figure out which
        # indices it needs from composition output. This needs to be passed down from the PEC.
        optimization_function.outcome_variable_indices = self._outcome_variable_indices

        control_signals = None
        outcome_variables = None
        ocm = PEC_OCM(
            agent_rep=agent_rep,
            monitor_for_control=outcome_variables,
            fit_parameters=parameters,
            depends_on=self.depends_on,
            cond_levels=self.cond_levels,
            cond_mask=self.cond_mask,
            allow_probes=True,
            objective_mechanism=objective_mechanism,
            function=optimization_function,
            control_signals=control_signals,
            num_estimates=num_estimates,
            num_trials_per_estimate=num_trials_per_estimate,
            initial_seed=initial_seed,
            same_seed_for_all_allocations=same_seed_for_all_parameter_combinations,
            context=context,
            return_results=True,
        )

        return ocm

    @handle_external_context()
    def run(self, *args, **kwargs):
        # Clear any old results from the composition
        if self.results is not None:
            self.results = []

        context = kwargs.get("context", None)
        self._assign_execution_ids(context)

        # Before we do anything, clear any compilation structures that have been generated. This is a workaround to
        # an issue that causes the PEC to fail to run in LLVM mode when the inner composition that we are fitting
        # has already been compiled.
        if self.controller.parameters.comp_execution_mode.get(context) != "Python":
            pnllvm.cleanup()

        # Capture the input passed to run and pass it on to the OCM
        assert self.controller is not None

        # Get the inputs
        inputs = kwargs.get("inputs", None if not args else args[0])

        # Since we are passing fitting\optimization parameters as inputs we need add them to the inputs
        if inputs:

            # Don't check inputs if we are within a call to evaluate_agent_rep, the inputs have already been checked and
            # cached on the PEC controller.
            if ContextFlags.PROCESSING not in context.flags:
                self.controller.check_pec_inputs(inputs)

            # Copy the inputs so we don't modify the original dict, note, we can't copy the keys because they
            # are object\mechanisms that are in the underlying composition.
            inputs = {k: v.copy() for k, v in inputs.items()}

            # Run parse input dict on the inputs, this will fill in missing input ports with default values. There
            # will be missing input ports because the user doesn't know about the control mechanism's input ports that
            # have been added by the PEC for the fitting parameters.
            if self.model in inputs and len(inputs) == 1:
                full_inputs = inputs
            else:
                full_inputs, num_trials = self.model._parse_input_dict(inputs, context)

            # Add the fitting parameters to the inputs, these will be modulated during fitting or optimization,
            # we just use a dummy value here for now (the first value in the range of the parameter)
            dummy_params = [v[0] for v in self.controller.function.fit_param_bounds.values()]
            self.controller.set_parameters_in_inputs(dummy_params, full_inputs)

        self.controller.set_pec_inputs_cache(full_inputs)

        # We need to set the inputs for the composition during simulation, by assigning the inputs dict passed in
        # PEC run() to its controller's state_feature_values (this is in order to accomodate multi-trial inputs
        # without having the PEC provide them one-by-one to the simulated composition. This assumes that the inputs
        # dict has the inputs specified in the same order as the state features (i.e., as specified by
        # PEC.get_input_format()), and rearranges them so that each node gets a full trial's worth of inputs.
        inputs_dict = self.controller.parameters.state_feature_values._get(context)

        if len(inputs_dict) == 0:
            raise ValueError("No inputs were specified for the PEC.")

        for state_input_port, value in zip(
            self.controller.state_input_ports, inputs_dict.values()
        ):
            value = convert_all_elements_to_np_array(value)
            state_input_port.parameters.value._set(value, context)

        kwargs.pop("inputs", None)

        # Turn off warnings about no inputs the PEC. This is because the PEC doesn't have any inputs itself, it
        # caches the inputs passed to it and passes them along to the inner composition during simulation.
        self.warned_about_run_with_no_inputs = True

        num_trials_per_estimate = len(inputs_dict[list(inputs_dict.keys())[0]])
        self.controller.parameters.num_trials_per_estimate.set(
            num_trials_per_estimate, context=context
        )

        # Run the composition as normal
        results = super(ParameterEstimationComposition, self).run(*args, **kwargs)

        # IMPLEMENTATION NOTE: has not executed OCM after first call
        if hasattr(self.controller, "optimal_control_allocation"):
            # Assign optimized_parameter_values and optimal_value    (remove randomization dimension)
            self.optimized_parameter_values = dict(zip(
                self.controller.function.fit_param_names,
                self.controller.optimal_control_allocation[:-1]
            ))
            self.optimal_value = self.controller.optimal_net_outcome

        return results

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
            raise ParameterEstimationCompositionError(
                f"The controller for ParameterEstimationComposition {self.name} "
                f"has not been instantiated yet. Cannot compute log-likelihood."
            )

        if self.controller.function is None:
            raise ParameterEstimationCompositionError(
                f"The function of the controller for "
                f"ParameterEstimationComposition {self.name} has not been "
                f"instantiated yet. Cannot compute log-likelihood."
            )

        if self.data is None:
            raise ParameterEstimationCompositionError(
                f"The data for ParameterEstimationComposition {self.name} "
                f"has not been defined. Cannot compute log-likelihood."
            )

        if len(args) != len(self.fit_parameters):
            raise ParameterEstimationCompositionError(
                f"The number of parameters specified in the call to "
                f"log_likelihood does not match the number of parameters "
                f"specified in the constructor of ParameterEstimationComposition."
            )

        if not hasattr(self.controller.function, "log_likelihood"):
            of = self.controller.function
            raise ParameterEstimationCompositionError(
                f"The function ({of}) for the controller of "
                f"ParameterEstimationComposition {self.name} does not appear to "
                f"have a log_likelihood function."
            )

        context.composition = self

        # Capture the inputs and pass it on to the OCM
        assert self.controller is not None
        self.controller.set_pec_inputs_cache(inputs)

        # Try to get the log-likelihood from controllers optimization_function, if it hasn't defined this function yet
        # then it will raise an error.
        return self.controller.function.log_likelihood(*args, context=context)

    def _complete_init_of_partially_initialized_nodes(self, context):
        pass


def _pec_ocm_state_feature_values_getter(owning_component=None, context=None) -> dict:
    """Return the complete input values passed to the last call of run for the Composition that the PEC_OCM controls.
    This method is used by the PEC_OCM to get the complete input dictionary for all trials cached in _pec.input_values,
    in order to pass them on to the agent_rep during simulation.
    """
    pec_ocm = owning_component

    if (
        pec_ocm.initialization_status == ContextFlags.INITIALIZING
        or pec_ocm._pec_input_values is None
    ):
        return {}

    if not isinstance(pec_ocm.composition, ParameterEstimationComposition):
        raise ParameterEstimationCompositionError(
            f"A PEC_OCM can only be used with a ParmeterEstimationComposition"
        )

    return pec_ocm._pec_input_values


class PEC_OCM(OptimizationControlMechanism):
    """
    OptimizationControlMechanism specialized for use with ParameterEstimationComposition
    Assign inputs passed to run method of ParameterEstimationComposition directly as values of
    PEC_OCM's state_input_ports (this allows a full set of trials' worth of inputs to be used in each
    run of the Composition being estimated or optimized.
    set_pec_inputs_cache(): called by PEC to cache inputs passed to its run method
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

        state_feature_values = Parameter(
            None,
            getter=_pec_ocm_state_feature_values_getter,
            user=False,
            pnl_internal=True,
            read_only=True,
        )

    def __init__(self, *args, **kwargs):
        self._pec_input_values = None

        self._pec_control_mech_indices = None

        if 'fit_parameters' in kwargs:
            self.fit_parameters = kwargs['fit_parameters']
            del kwargs['fit_parameters']
        else:
            raise ValueError("PEC_OCM requires that the PEC parameters be passed down to it.")

        if 'depends_on' in kwargs:
            self.depends_on = kwargs['depends_on']
            del kwargs['depends_on']
        else:
            self.depends_on = None

        if 'cond_levels' in kwargs:
            self.cond_levels = kwargs['cond_levels']
            del kwargs['cond_levels']

        if 'cond_mask' in kwargs:
            self.cond_mask = kwargs['cond_mask']
            del kwargs['cond_mask']

        super().__init__(*args, **kwargs)

    def _instantiate_output_ports(self, context=None):
        """Assign CostFunctions.DEFAULTS as default for cost_option of ControlSignals.
        """

        # The only control signal that we need for the PEC is the randomization control signal. All other parameter
        # values will be passed through the inputs. This allows for trial-wise conditional parameter values to be
        # passed to the composition being fit or optimized.
        output_ports = ContentAddressableList(component_type=Port_Base)
        self.parameters.output_ports._set(output_ports, context)
        self._create_randomization_control_signal(context)

    def check_pec_inputs(self, inputs_dict: dict):

        model = self.composition.model

        # Since we added control mechanisms to the composition, we need to make sure that we subtract off
        # the number of control mechanisms from the number of state input ports in the error message.
        num_state_input_ports = self.num_state_input_ports - len(self.fit_parameters)

        if not inputs_dict:
            pass

        # If inputs_dict has model as its only entry, then check that its format is OK to pass to pec.run()
        elif len(inputs_dict) == 1 and model in inputs_dict:
            if not all(
                    len(trial) == num_state_input_ports for trial in inputs_dict[model]
            ):
                raise ParameterEstimationCompositionError(
                    f"The array in the dict specified for the 'inputs' arg of "
                    f"{self.composition.name}.run() is badly formatted: "
                    f"the length of each item in the outer dimension (a trial's "
                    f"worth of inputs) must be equal to the number of inputs to "
                    f"'{model.name}' ({num_state_input_ports})."
                )

        else:

            # Restructure inputs as nd array with each row (outer dim) a trial's worth of inputs
            #    and each item in the row (inner dim) the input to a node (or input_port) for that trial
            if len(inputs_dict) != num_state_input_ports:

                raise ParameterEstimationCompositionError(
                    f"The dict specified in the `input` arg of "
                    f"{self.composition.name}.run() is badly formatted: "
                    f"the number of entries should equal the number of inputs "
                    f"to '{model.name}' ({num_state_input_ports})."
                )
            trial_seqs = list(inputs_dict.values())
            num_trials = len(trial_seqs[0])
            for trial in range(num_trials):
                for trial_seq in trial_seqs:
                    if len(trial_seq) != num_trials:
                        raise ParameterEstimationCompositionError(
                            f"The dict specified in the `input` arg of "
                            f"ParameterEstimationMechanism.run() is badly formatted: "
                            f"every entry must have the same number of inputs."
                        )

    def set_pec_inputs_cache(self, inputs_dict: dict) -> dict:
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

        if not inputs_dict or (len(inputs_dict) == 1 and model in inputs_dict):
            pass
        else:
            trial_seqs = list(inputs_dict.values())
            num_trials = len(trial_seqs[0])
            input_values = [[] for _ in range(num_trials)]
            for trial in range(num_trials):
                for trial_seq in trial_seqs:
                    input_values[trial].extend(trial_seq[trial])
            inputs_dict = {model: input_values}

        self._pec_input_values = inputs_dict


    def set_parameters_in_inputs(self, parameters, inputs):
        """
        Add the fitting parameters to the inputs passed to the model for each trial. Originally, the PEC used the
        OCM to modulate the parameters of the model. However, this did not allow for trial-wise conditional or varying
        parameter values. The current implementation passes the fitting parameters directly to the model as inputs.
        These inputs go to dummy control mechanisms that are added to the composition before fitting or optimization.
        The control mechanisms are then used to modulate the parameters of the model. This function has side effects
        because it modifies the inputs dictionary in place.

        Args:
            parameters (list): A list of fitting parameters that are to be passed to the model as inputs.
            inputs (dict): A dictionary of inputs that are passed to the model for each trial.

        """

        # Get the input indices for the control mechanisms that are used to modulate the fitting parameters
        if self._pec_control_mech_indices is None:
            self.composition.model._analyze_graph()
            input_nodes = [node for node, roles in self.composition.model.nodes_to_roles.items()
                           if NodeRole.INPUT in roles]
            self._pec_control_mech_indices = [input_nodes.index(m) for m in self.composition.pec_control_mechs.values()]

        # If the model is in the inputs, then inputs are passed as list of lists and we need to add the fitting
        # parameters to each trial as a concatenated list.
        if self.composition.model in inputs:

            in_arr = inputs[self.composition.model]

            if type(in_arr) is not np.ndarray:
                in_arr = convert_to_np_array(in_arr)

            # Make sure it is 3D (only if not ragged)
            if in_arr.dtype != object:
                in_arr = np.atleast_3d(in_arr)

            # If the inputs don't have columns for the fitting parameters, then we need to add them
            if in_arr.shape[1] != len(self.composition.input_ports):
                num_missing = len(self.composition.input_ports) - in_arr.shape[1]
                if in_arr.ndim == 3:
                    in_arr = np.hstack((in_arr, np.zeros((in_arr.shape[0], num_missing, 1))))
                elif in_arr.ndim == 2:
                    in_arr = np.hstack((in_arr, np.zeros((in_arr.shape[0], num_missing))))

            j = 0
            for i, (pname, mech) in enumerate(self.fit_parameters.keys()):
                mech_idx = self._pec_control_mech_indices[i]
                if not self.depends_on or (pname, mech) not in self.depends_on:
                    if in_arr.ndim == 3:
                        in_arr[:, mech_idx, 0] = parameters[j]
                    else:
                        for k in range(in_arr.shape[0]):
                            in_arr[k, mech_idx] = np.array([parameters[j]])
                    j += 1
                else:
                    for level in self.cond_levels[(pname, mech)]:
                        mask = self.cond_mask[(pname, mech)][level]
                        if in_arr.ndim == 3:
                            in_arr[mask, mech_idx, 0] = parameters[j]
                        else:
                            for k in range(in_arr.shape[0]):
                                if mask[k]:
                                    in_arr[k, mech_idx] = np.array([parameters[j]])
                        j += 1

            inputs[self.composition.model] = in_arr

        # Otherwise, assume the inputs are passed to each mechanism individually. Thus, we need to feed the
        # fitting parameters to the model to their respective control mechanisms
        else:
            j = 0
            for i, ((pname, mech), values) in enumerate(self.fit_parameters.items()):
                control_mech = self.composition.pec_control_mechs[(pname, mech)]
                if not self.depends_on or (pname, mech) not in self.depends_on:
                    inputs[control_mech] = np.ones_like(inputs[control_mech]) * parameters[j]
                    j += 1
                else:
                    inputs[control_mech] = np.zeros_like(inputs[control_mech])
                    for level in self.cond_levels[(pname, mech)]:
                        mask = self.cond_mask[(pname, mech)][level]
                        inputs[control_mech][mask] = parameters[j]
                        j += 1

    def _execute(self, variable=None, context=None, runtime_params=None)->np.ndarray:
        """Return control_allocation that optimizes net_outcome of agent_rep.evaluate().
        """

        if self.is_initializing:
            return [defaultControlAllocation]

        # Assign default control_allocation if it is not yet specified (presumably first trial)
        control_allocation = self.parameters.control_allocation._get(context)
        if control_allocation is None:
            control_allocation = [c.defaults.variable for c in self.control_signals]

        # Give the agent_rep a chance to adapt based on last trial's state_feature_values and control_allocation
        if hasattr(self.agent_rep, "adapt"):
            # KAM 4/11/19 switched from a try/except to hasattr because in the case where we don't
            # have an adapt method, we also don't need to call the net_outcome getter
            net_outcome = self.parameters.net_outcome._get(context)

            self.agent_rep.adapt(self.parameters.state_feature_values._get(context),
                                 control_allocation,
                                 net_outcome,
                                 context=context)

        # freeze the values of current context, because they can be changed in between simulations,
        # and the simulations must start from the exact spot
        frozen_context = self._get_frozen_context(context)

        alt_controller = None
        if self.agent_rep.controller is None:
            try:
                alt_controller = context.composition.controller
            except AttributeError:
                alt_controller = None

        self.agent_rep._initialize_as_agent_rep(
            frozen_context, base_context=context, alt_controller=alt_controller
        )

        # Get control_allocation that optimizes net_outcome using OptimizationControlMechanism's function
        # IMPLEMENTATION NOTE: skip ControlMechanism._execute since it is a stub method that returns input_values
        optimal_control_allocation, optimal_net_outcome, saved_samples, saved_values = \
                                                super(ControlMechanism,self)._execute(
                                                    variable=control_allocation,
                                                    num_estimates=self.parameters.num_estimates._get(context),
                                                    context=context,
                                                    runtime_params=runtime_params
                                                )

        # clean up frozen values after execution
        self.agent_rep._clean_up_as_agent_rep(frozen_context, alt_controller=alt_controller)

        if self.function.save_samples:
            self.saved_samples = saved_samples
        if self.function.save_values:
            self.saved_values = saved_values

        self.optimal_control_allocation = optimal_control_allocation
        self.optimal_net_outcome = optimal_net_outcome

        # Return optimal control_allocation formatted as 2d array
        return [defaultControlAllocation]
