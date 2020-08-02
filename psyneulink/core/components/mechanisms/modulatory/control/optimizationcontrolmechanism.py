# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  OptimizationControlMechanism *************************************************


"""

Overview
--------

An OptimizationControlMechanism is a `ControlMechanism <ControlMechanism>` that uses an `OptimizationFunction` to find
an optimal `control_allocation <ControlMechanism.control_allocation>` for a given `state
<OptimizationControlMechanism_State>`. The `OptimizationFunction` uses the OptimizationControlMechanism's
`evaluation_function` <OptimizationControlMechanism.evalutate_function>` to evaluate `control_allocation
<ControlMechanism.control_allocation>` samples, and then implements the one that yields the best predicted result.
The result returned by the `evaluation_function` <OptimizationControlMechanism.evalutate_function>` is ordinarily
the `net_outcome <ControlMechanism.net_outcome>` computed by the OptimizationControlMechanism for the `Composition`
(or part of one) that it controls, and its `ObjectiveFunction` seeks to maximize this, which corresponds to
maximizing the Expected Value of Control, as described below.

.. _OptimizationControlMechanism_EVC:

**Expected Value of Control**

The `net_outcome <ControlMechanism.net_outcome>` of an OptmizationControlMechanism, like any `ControlMechanism`
is computed as the difference between the `outcome <ControlMechanism.outcome>` computed by its `objective_mechanism
<ControlMechanism.objective_mechanism>` and the `costs <ControlMechanism.costs>` of its `control_signals
<ControlMechanism.control_signals>` for a given `state <OptimizationControlMechanism_State>` (i.e.,
set of `feature_values <OptimizationControlMechanism.feature_values>` and `control_allocation
<ControlMechanism.control_allocation>`.  If the `outcome <ControlMechanism.outcome>` is configured to measure the
value of processing (e.g., reward received, time taken to respond, or a combination of these, etc.),
and the `OptimizationFunction` assigned as the OptimizationControlMechanism's `function
<OptimizationControlMechanism.function>` is configured find the `control_allocation
<ControlMechanism.control_allocation>` that maximizes its `net_outcome <ControlMechanism.net_outcome>`,
then the OptimizationControlMechanism is said to be maximizing the `Expected Value of Control (EVC)
<https://www.ncbi.nlm.nih.gov/pubmed/23889930>`_.  That is, it implements a cost-benefit analysis that
weighs the `costs <ControlMechanism.costs>` of the ControlSignal `values <ControlSignal.value>` specified by a
`control_allocation <ControlMechanism.control_allocation>` against the `outcome <ControlMechanism.outcome>` expected
to result from it.  The costs are computed based on the `cost_options <ControlSignal.cost_options>` specified for
each of the OptimizationControlMechanism's `control_signals <ControlMechanism.control_signals>` and its
`combine_costs <ControlMechanism.combine_costs>` function.  The EVC is determined by its `compute_net_outcome
<ControlMechanism.compute_net_outcome>` function (assigned to its `net_outcome <ControlMechanism.net_outcome>`
attribute), which is computed for a given `state <OptimizationControlMechanism_State>` by the
OptimizationControlMechanism's `evaluation_function <OptimizationControlMechanism.evalutation_function>`.

COMMENT:
The table `below <OptimizationControlMechanism_Examples>` lists different
parameterizations of OptimizationControlMechanism that implement various models of EVC Optimization.

### THROUGHOUT DOCUMENT, REWORD AS "optimizing control_allocation" RATHER THAN "maximizing" / "greatest"

COMMENT

.. _OptimizationControlMechanism_Agent_Representation_Types:

**Agent Representation and Types of Optimization**

The defining characteristic of an OptimizationControlMechanism is its `agent representation
<OptimizationControlMechanism_Agent_Rep>`, that is used to determine the `net_outcome <ControlMechanism.net_outcome>
for a given `state <OptimizationControlMechanism_State>` and find the `control_allocation
<ControlMechanism.control_allocation>` that optimizes this.  The `agent_rep <OptimizationControlMechanism.agent_rep>`
can be either the `Composition` to which the OptimizationControlMechanism belongs (and controls) or another one that
is used to estimate the `net_outcome <ControlMechanism.net_outcome>` for that Composition.  This distinction
corresponds closely to the distinction between *model-based* and *model-free* optimization in the `machine learning
<HTML REF>`_ and `cognitive neuroscience <https://www.nature.com/articles/nn1560>`_ literatures, as described below.

.. _OptimizationControlMechanism_Model_Free_Model_Based:

.. _OptimizationControlMechanism_Model_Free:

*Model-Free Optimization*

This is achieved by assigning as the `agent_rep  <OptimizationControlMechanism.agent_rep>` a Composition other than the
one to which the OptimizationControlMechanism belongs (and for which it is the `controller <Composition.controller>`).
In each `TRIAL <TimeScale.TRIAL>`, the `agent_rep <OptimizationControlMechanism.agent_rep>` is given the chance to
adapt, by adjusting its parameters in order to improve its prediction of the `net_outcome
<ControlMechanism.net_outcome>` for the Composition (or part of one) that is controlled by the
OptimizationControlMechanism (based on the `state <OptimizationControlMechanism_State>` and `net_outcome
<ControlMechanism.net_outcome>` of the prior trial).  The `agent_rep <OptimizationControlMechanism.agent_rep>` is
then used to predict the `net_outcome <ControlMechanism.net_outcome>` for `control_allocation
<ControlMechanism.control_allocation>` samples to find the one that yields the best predicted `net_outcome
<ControlMechanism.net_outcome>` of processing on the upcoming trial, based on the current or (expected)
`feature_values <OptimizationControlMechanism.feature_values>` for that trial.

.. _OptimizationControlMechanism_Model_Based:

*Model-Based Optimization*

This is achieved by assigning as the `agent_rep  <OptimizationControlMechanism.agent_rep>` the Composition to which the
OptimizationControlMechanism belongs (and for which it is the `controller <Composition.controller>`). On each `TRIAL
<TimeScale.TRIAL>`, that Composition itself is used to simulate processing on the upcoming trial, based on the current
or (expected) `feature_values <OptimizationControlMechanism.feature_values>` for that trial, in order to find the
<ControlMechanism.control_allocation>` that yields the best net_outcome <ControlMechanism.net_outcome>` for that trial.

.. _OptimizationControlMechanism_Creation:

Creating an OptimizationControlMechanism
----------------------------------------

An OptimizationControlMechanism is created in the same was as any `ControlMechanism <ControlMechanism>`.
The following arguments of its constructor are specific to the OptimizationControlMechanism:

* **features** -- takes the place of the standard **input_ports** argument in the constructor for a Mechanism`,
  and specifies the values used by the OptimizationControlMechanism, together with a `control_allocation
  <ControlMechanism.control_allocation>`, to calculate a `net_outcome <ControlMechanism.net_outcome>`.  For
  `model-based optimzation <OptimizationControlMechanism_Model_Based>` these are also used as the inputs to the
  Composition (i.e., `agent_rep <OptimizationControlMechanism.agent_rep>`) when it's `evaluate <Composition.evaluate>`
  method is called (see `OptimizationControlMechanism_Features` below).  Features can be specified using any of the
  following, singly or combined in a list:

  * *InputPort specification* -- this can be any form of `InputPort specification <InputPort_Specification>`
    that resolves to an OutputPort from which the InputPort receives a Projection;  the `value
    <OutputPort.value>` of that OutputPort is used as the feature. Each of these InputPorts is marked as
    `internal_only <InputPorts.internal_only>` = `True`.

  Features can also be added to an existing OptimizationControlMechanism using its `add_features` method.  If the
  **features** argument is not specified, then the `input <Composition.input_values>` to the `Composition` on the last
  trial of its execution is used to predict the `net_outcome <ControlMechanism.net_outcome>` for the upcoming trial.

.. _OptimizationControlMechanism_Feature_Function:

* **feature_function** -- specifies `function <InputPort>` of the InputPort created for each item listed in
  **features**.  By default, this is the identity function, that assigns the current value of the feature
  to the OptimizationControlMechanism's `feature_values <OptimizationControlMechanism.feature_values>` attribute.
  However, other functions can be assigned, for example to maintain a record of past values, or integrate them over
  trials.
..
* **agent_rep** -- specifies the `Composition` used by the OptimizationControlMechanism's `evaluation_function
  <OptimizationControlMechanism.evaluation_function>` to calculate the predicted `net_outcome
  <ControlMechanism.net_outcome>` for a given `state <OptimizationControlMechanism_State>` (see `below
  <OptimizationControlMechanism_Agent_Rep>` for additional details). If it is not specified, then the
  `Composition` to which the OptimizationControlMechanism is assigned becomes its `agent_rep
  <OptimizationControlMechanism.agent_rep>`, and the OptimizationControlMechanism is assigned as that Composition's
  `controller <Composition.controller>`, implementing fully `model-based <OptimizationControlMechanism_Model_Based>`
  optimization.  If that Composition already has a `controller <Composition.controller>` specified,
  the OptimizationControlMechanism is disabled. If another Composition is specified, it must conform to the
  specifications for an `agent_rep <OptimizationControlMechanism.agent_rep>` as described `below
  <OptimizationControlMechanism_Agent_Rep>`.

.. _OptimizationControlMechanism_Structure:

Structure
---------

In addition to the standard Components associated with a `ControlMechanism`, including a `Projection <Projection>`
to its *OUTCOME* InputPort from its `objective_mechanism <ControlMechanism.objective_mechanism>`, and a
`function <OptimizationControlMechanism.function>` used to carry out the optimization process, it has several
other constiuents, as described below.

.. _OptimizationControlMechanism_ObjectiveMechanism:

*ObjectiveMechanism*
^^^^^^^^^^^^^^^^^^^^

Like any `ControlMechanism`, an OptimizationControlMechanism may be assigned an `objective_mechanism
<ControlMechanism.objective_mechanism>` that is used to evaluate the outcome of processing for a given trial (see
`ControlMechanism_Objective_ObjectiveMechanism). This passes the result to the OptimizationControlMechanism, which it
places in its `outcome <OptimizationControlMechanism.outcome>` attribute.  This is used by its `compute_net_outcome
<ControlMechanism.compute_net_outcome>` function, together with the `costs <ControlMechanism.costs>` of its
`control_signals <ControlMechanism.control_signals>`, to compute the `net_outcome <ControlMechanism.net_outcome>` of
processing for a given `state <OptimizationControlMechanism_State>`, and that is returned by `evaluation` method of the
OptimizationControlMechanism's `agent_rep <OptimizationControlMechanism.agent_rep>`.

.. note::
    The `objective_mechanism <ControlMechanism.objective_mechanism>` is distinct from, and should not be
    confused with the `objective_function <OptimizationFunction.objective_function>` parameter of the
    OptimizationControlMechanism's `function <OptimizationControlMechanism.function>`.  The `objective_mechanism
    <ControlMechanism.objective_mechanism>` evaluates the `outcome <ControlMechanism.outcome>` of processing
    without taking into account the `costs <ControlMechanism.costs>` of the OptimizationControlMechanism's
    `control_signals <ControlMechanism.control_signals>`.  In contrast, its `evaluation_function
    <OptimizationControlMechanism.evaluation_function>`, which is assigned as the
    `objective_function` parameter of its `function <OptimizationControlMechanism.function>`, takes the `costs
    <ControlMechanism.costs>` of the OptimizationControlMechanism's `control_signals <ControlMechanism.control_signals>`
    into account when calculating the `net_outcome` that it returns as its result.

.. _OptimizationControlMechanism_Features:

*Features*
^^^^^^^^^^

In addition to its `primary InputPort <InputPort_Primary>` (which typically receives a projection from the
*OUTCOME* OutputPort of the `objective_mechanism <ControlMechanism.objective_mechanism>`,
an OptimizationControlMechanism also has an `InputPort` for each of its features. By default, these are the current
`input <Composition.input_values>` for the Composition to which the OptimizationControlMechanism belongs.  However,
different values can be specified, as can a `feature_function <OptimizationControlMechanism_Feature_Function>` that
transforms these.  For OptimizationControlMechanisms that implement `model-free
<OptimizationControlMechanism_Model_Free>` optimization, its `feature_values
<OptimizationControlMechanism.feature_values>` are used by its `evaluation_function
<OptimizationControlMechanism.evaluation_function>` to predict the `net_outcome <ControlMechanism.net_outcome>` for a
given `control_allocation <ControlMechanism.control_allocation>`.  For OptimizationControlMechanisms that implement
`model-based <OptimizationControlMechanism_Model_Based>` optimization, the `feature_values
<OptimizationCozntrolMechanism.feature_values>` are used as the Composition's `input <Composition.input_values>` when
it is executed to evaluate the `net_outcome <ControlMechanism.net_outcome>` for a given
`control_allocation<ControlMechanism.control_allocation>`.

Features can be of two types:

* *Input Features* -- these are values received as input by other Mechanisms in the `Composition`. They are
  specified as `shadowed inputs <InputPort_Shadow_Inputs>` in the **features** argument of the
  OptimizationControlMechanism's constructor (see `OptimizationControlMechanism_Creation`).  An InputPort is
  created on the OptimziationControlMechanism for each feature, that receives a `Projection` paralleling
  the input to be shadowed.
..
* *Output Features* -- these are the `value <OutputPort.value>` of an `OutputPort` of some other `Mechanism <Mechanism>`
  in the Composition.  These too are specified in the **features** argument of the OptimizationControlMechanism's
  constructor (see `OptimizationControlMechanism_Creation`), and each is assigned a `Projection` from the specified
  OutputPort(s) to the InputPort of the OptimizationControlMechanism for that feature.

The current `value <InputPort.value>` of the InputPorts for the features are listed in the `feature_values
<OptimizationControlMechanism.feature_values>` attribute.

.. _OptimizationControlMechanism_State:

*State*
^^^^^^^

The state of the Composition (or part of one) controlled by an OptimizationControlMechanism is defined by a combination
of `feature_values <OptimizationControlMechanism.feature_values>` (see `above <OptimizationControlMechanism_Features>`)
and a `control_allocation <ControlMechanism.control_allocation>`.

.. _OptimizationControlMechanism_Agent_Rep:

*Agent Representation*
^^^^^^^^^^^^^^^^^^^^^^

The defining feature of an OptimizationControlMechanism is its agent representation, specified in the **agent_rep**
argument of its constructor and assigned to its `agent_rep <OptimizationControlMechanism.agent_rep>` attribute.  This
designates a representation of the `Composition` (or parts of one) that the OptimizationControlMechanism controls, that
is used to evaluate sample `control_allocations <ControlMechanism.control_allocation>` in order to find the one that
optimizes the `net_outcome <ControlMechanism.net_outcome>`. The `agent_rep <OptimizationControlMechanism.agent_rep>`
is always itself a `Composition`, that can be either the same one that the OptimizationControlMechanism controls or
another one that is used to estimate the `net_outcome <ControlMechanism.net_outcome>` for that Composition (see `above
<OptimizationControlMechanism_Agent_Representation_Types>`).  The `evaluate <Composition.evaluate>` method of the
Composition is assigned as the `evaluation_function <OptimizationControlMechanism.evaluation_function>` of the
OptimizationControlMechanism.  If the `agent_rep <OptimizationControlMechanism.agent_rep>` is not the Composition for
which the OptimizationControlMechanism is the controller, then it must meet the following requirements:

    * Its `evaluate <Composition.evaluate>` method must accept as its first three arguments, in order,
      values that correspond in shape to  the `feature_values <OptimizationControlMechanism.feature_values>`,
      `control_allocation <ControlMechanism.control_allocation>` and `num_estimates
      <OptimizationControlMechanism.num_estimates>` attributes of the OptimizationControlMechanism, respectively.
    ..
    * If it has an `adapt <Composition.adapt>` method, that must accept as its first three arguments, in order,
      values that corresopnd to the shape of the `feature_values <OptimizationControlMechanism.feature_values>`,
      `control_allocation <ControlMechanism.control_allocation>` and `net_outcome
      <OptimizationControlMechanism.net_outcome>` attributes of the OptimizationControlMechanism. respectively.

.. _OptimizationControlMechanism_Function:

*Function*
^^^^^^^^^^

The `function <OptimizationControlMechanism.function>` of an OptimizationControlMechanism is used to find the
`control_allocation <ControlMechanism.control_allocation>` that optimizes the `net_outcome
<ControlMechanism.net_outcome>` for the current (or expected) `features <OptimizationControlMechanism_State>`.
It is generally an `OptimizationFunction`, which in turn has `objective_function
<OptimizationFunction.objective_function>`, `search_function <OptimizationFunction.search_function>`
and `search_termination_function <OptimizationFunction.search_termination_function>` methods, as well as a `search_space
<OptimizationFunction.search_space>` attribute.  The OptimizationControlMechanism's `evaluation_function
<OptimizationControlMechanism.evaluation_function>` is automatically assigned as the
OptimizationFunction's `objective_function <OptimizationFunction.objective_function>`, and is used to
evaluate each `control_allocation <ControlMechanism.control_allocation>` sampled from the `search_space
<OptimizationFunction.search_space>` by the `search_function `search_function <OptimizationFunction.search_function>`
until the `search_termination_function <OptimizationFunction.search_termination_function>` returns `True`.
A custom function can be assigned as the `function <OptimizationControlMechanism.function>` of an
OptimizationControlMechanism, however it must meet the following requirements:

.. _OptimizationControlMechanism_Custom_Funtion:

  - It must accept as its first argument and return as its result an array with the same shape as the
    OptimizationControlMechanism's `control_allocation <ControlMechanism.control_allocation>`.
  ..
  - It must implement a `reset` method that accepts **objective_function** as a keyword argument and
    implements an attribute with the same name.

    COMMENT:
    - it must implement a `reset` method that accepts as keyword arguments **objective_function**,
      **search_function**, **search_termination_function**, and **search_space**, and implement attributes
      with corresponding names.
    COMMENT

If **function** argument is not specified, the `GridSearch` `OptimiziationFunction` is assigned as the default,
which evaluates the `net_outcome <ControlMechanism.net_outcome>` using the OptimizationControlMechanism's
`control_allocation_search_space <OptimizationControlMechanism.control_allocation_search_spaces>` as its
`search_space <OptimizationFunction.search_space>`, and returns the `control_allocation
<ControlMechanism.control_allocation>` that yields the greatest `net_outcome <ControlMechanism.net_outcome>`,
thus implementing a computation of `EVC <OptimizationControlMechanism_EVC>`.

COMMENT:
.. _OptimizationControlMechanism_Search_Functions:

*Search Function, Search Space and Search Termination Function*

Subclasses of OptimizationControlMechanism may implement their own `search_function
<OptimizationControlMechanism.search_function>` and `search_termination_function
<OptimizationControlMechanism.search_termination_function>` methods, as well as a
`control_allocation_search_space <OptimizationControlMechanism.control_allocation_search_space>` attribute, that are
passed as parameters to the `OptimizationFunction` when it is constructed.  These can be specified in the
constructor for an `OptimizationFunction` assigned as the **function** argument in the
OptimizationControlMechanism's constructor, as long as they are compatible with the requirements of
the OptimizationFunction and OptimizationControlMechanism.  If they are not specified, then defaults specified
either by the OptimizationControlMechanism or the OptimizationFunction are used.
COMMENT

.. _OptimizationControlMechanism_Execution:

Execution
---------

When an OptimizationControlMechanism is executed, it carries out the following steps:

  * Calls `adapt` method of its `agent_rep <OptimizationControlMechanism.agent_rep>` to give that a chance to
    modify its parameters in order to better predict the `net_outcome <ControlMechanism.net_outcome>` for a given `state
    <OptimizationControlMechanism_State>`, based the state and `net_outcome <ControlMechanism.net_outcome>` of the
    previous trial.
  ..
  * Calls `function <OptimizationControlMechanism.function>` to find the `control_allocation
    <ControlMechanism.control_allocation>` that optimizes `net_outcome <ControlMechanism.net_outcome>`.  The
    way in which it searches for the best `control_allocation <ControlMechanism.control_allocation>` is determined by
    the type of `OptimzationFunction` assigned to `function <OptimizationControlMechanism.function>`, whereas the way
    that it evaluates each one is determined by the OptimizationControlMechanism's `evaluation_function
    <OptimizationControlMechanism.evalutation_function>`.  More specifically:

    * The `function <OptimizationControlMechanism.function>` selects a sample `control_allocation
      <ControlMechanism.control_allocation>` (using its `search_function <OptimizationFunction.search_function>`
      to select one from its `search_space <OptimizationFunction.search_space>`), and evaluates the predicted
      `net_outcome <ControlMechanism.net_outcome>` for that `control_allocation
      <ControlMechanism.control_allocation>` using the OptimizationControlMechanism's `evaluation_function`
      <OptimizationControlMechanism.evalutation_function>` and the current `feature_values
      <OptimizationControlMechanism.feature_values>`.
    ..
    * It continues to evaluate the `net_outcome <ControlMechanism.net_outcome>` for `control_allocation
      <ControlMechanism.control_allocation>` samples until its `search_termination_function
      <OptimizationFunction.search_termination_function>` returns `True`.
    ..
    * Finally, it implements the `control_allocation <ControlMechanism.control_allocation>` that yielded the optimal
      `net_outcome <ControlMechanism.net_outcome>`.  This is used by the OptimizationControlMechanism's `control_signals
      <ControlMechanism.control_signals>` to compute their `values <ControlSignal.value>` which, in turn, are used by
      their `ControlProjections <ControlProjection>` to modulate the parameters they control when the Composition is
      next executed.

COMMENT:
.. _OptimizationControlMechanism_Examples:

Examples
--------

The table below lists `model-free <ModelFreeOptimizationControlMechanism>` and `model-based
<ModelBasedOptimizationControlMechanism` subclasses of OptimizationControlMechanisms.

.. table:: **Model-Free and Model-Based OptimizationControlMechanisms**

   +-------------------------+----------------------+----------------------+---------------------+---------------------+------------------------------+
   |                         |     *Model-Free*     |                           *Model-Based*                                                         |
   +-------------------------+----------------------+----------------------+---------------------+---------------------+------------------------------+
   |**Functions**            |`LVOCControlMechanism`| LVOMControlMechanism | MDPControlMechanism |`EVCControlMechanism`| ParameterEstimationMechanism |
   +-------------------------+----------------------+----------------------+---------------------+---------------------+------------------------------+
   |**learning_function**    |     `BayesGLM`       |        `pymc`        |    `BeliefUpdate`   |       *None*        |           `pymc`             |
   +-------------------------+----------------------+----------------------+---------------------+---------------------+------------------------------+
   |**function** *(primary)* |`GradientOptimization`|     `GridSearch`     |       `Sample`      |    `GridSearch`     |           `Sample`           |
   +-------------------------+----------------------+----------------------+---------------------+---------------------+------------------------------+
   |       *search_function* |  *follow_gradient*   |   *traverse_grid*    | *sample_from_dist*  |   *traverse_grid*   |      *sample_from_dist*      |
   +-------------------------+----------------------+----------------------+---------------------+---------------------+------------------------------+
   |    *objective_function* |    *compute_EVC*     |  *evaluate*,   |  *evaluate*,  |  *evaluate*,  |    *evaluate*,         |
   |                         |                      |  *compute_EVC*       |  *compute_EVC*      |  *compute_EVC*      |    *compute_likelihood*      |
   +-------------------------+----------------------+----------------------+---------------------+---------------------+------------------------------+
   |             *execution* | *iterate w/in trial* |  *once per trial*    | *iterate w/in trial*| *iterate w/in trial*|     *iterate w/in trial*     |
   +-------------------------+----------------------+----------------------+---------------------+---------------------+------------------------------+

The following models provide examples of implementing the OptimizationControlMechanisms in the table above:

`LVOCControlMechanism`\\:  `BustamanteStroopXORLVOCModel`
`EVCControlMechanism`\\:  `UmemotoTaskSwitchingEVCModel`
COMMENT



.. _OptimizationControlMechanism_Class_Reference:

Class Reference
---------------

"""
import copy
import numpy as np
import typecheck as tc

from collections.abc import Iterable

from psyneulink.core.components.component import DefaultsFlexibility
from psyneulink.core.components.functions.function import is_function_type, FunctionError
from psyneulink.core.components.functions.optimizationfunctions import \
    OBJECTIVE_FUNCTION, SEARCH_SPACE
from psyneulink.core.components.functions.combinationfunctions import LinearCombination
from psyneulink.core.components.functions.transferfunctions import CostFunctions
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import \
    ObjectiveMechanism, ObjectiveMechanismError
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.shellclasses import Function
from psyneulink.core.components.ports.inputport import InputPort, _parse_shadow_inputs
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.components.ports.port import _parse_port_spec
from psyneulink.core.globals.context import Context, ContextFlags
from psyneulink.core.globals.defaults import defaultControlAllocation
from psyneulink.core.globals.keywords import \
    DEFAULT_VARIABLE, EID_FROZEN, FUNCTION, INTERNAL_ONLY, NAME, \
    OPTIMIZATION_CONTROL_MECHANISM, OBJECTIVE_MECHANISM, OUTCOME, PRODUCT, PARAMS, \
    CONTROL, AUTO_ASSIGN_MATRIX
from psyneulink.core.globals.utilities import convert_to_np_array
from psyneulink.core.globals.parameters import Parameter, ParameterAlias
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.context import handle_external_context

from psyneulink.core import llvm as pnlvm

__all__ = [
    'OptimizationControlMechanism', 'OptimizationControlMechanismError',
    'AGENT_REP', 'FEATURES'
]

AGENT_REP = 'agent_rep'
FEATURES = 'features'


def _parse_feature_values_from_variable(variable):
    return np.array(np.array(variable[1:]).tolist())


class OptimizationControlMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class OptimizationControlMechanism(ControlMechanism):
    """OptimizationControlMechanism(         \
        objective_mechanism=None,            \
        monitor_for_control=None,            \
        objective_mechanism=None,            \
        origin_objective_mechanism=False     \
        terminal_objective_mechanism=False   \
        features=None,                       \
        feature_function=None,               \
        function=None,                       \
        agent_rep=None,                      \
        search_function=None,                \
        search_termination_function=None,    \
        search_space=None,                   \
        control_signals=None,                \
        modulation=MULTIPLICATIVE,           \
        combine_costs=np.sum,                \
        compute_reconfiguration_cost=None,   \
        compute_net_outcome=lambda x,y:x-y)

    Subclass of `ControlMechanism <ControlMechanism>` that adjusts its `ControlSignals <ControlSignal>` to optimize
    performance of the `Composition` to which it belongs.  See parent class for additional arguments.

    Arguments
    ---------

    features : Mechanism, OutputPort, Projection, dict, or list containing any of these
        specifies Components, the values of which are assigned to `feature_values
        <OptimizationControlMechanism.feature_values>` and used to predict `net_outcome <ControlMechanism.net_outcome>`.
        Any `InputPort specification <InputPort_Specification>` can be used that resolves to an `OutputPort` that
        projects to the InputPort.

    feature_function : Function or function : default None
        specifies the `function <InputPort.function>` for the `InputPort` assigned to each `feature
        <OptimizationControlMechanism_Features>`.

    agent_rep : None  : default Composition to which the OptimizationControlMechanism is assigned
        specifies the `Composition` used by the `evalution_function <OptimizationControlMechanism.evaluation_function>`
        to predict the `net_outcome <ControlMechanism.net_outcome>` for a given `state
        <OptimizationControlMechanism_State>`.  If a Composition is specified, it must be suitably configured
        (see `above <OptimizationControlMechanism_Agent_Rep>` for additional details). If it is not specified, the
        OptimizationControlMechanism is placed in `deferred_init` status until it is assigned as the `controller
        <Composition.controller>` of a Composition, at which that Composition is assigned as the `agent_rep
        <agent_rep <OptimizationControlMechanism.agent_rep`.

    search_function : function or method
        specifies the function assigned to `function <OptimizationControlMechanism.function>` as its
        `search_function <OptimizationFunction.search_function>` parameter, unless that is specified in a
        constructor for `function <OptimizationControlMechanism.function>`.  It must take as its arguments
        an array with the same shape as `control_allocation <ControlMechanism.control_allocation>` and an integer
        (indicating the iteration of the `optimization process <OptimizationFunction_Process>`), and return
        an array with the same shape as `control_allocation <ControlMechanism.control_allocation>`.

    search_termination_function : function or method
        specifies the function assigned to `function <OptimizationControlMechanism.function>` as its
        `search_termination_function <OptimizationFunction.search_termination_function>` parameter, unless that is
        specified in a constructor for `function <OptimizationControlMechanism.function>`.  It must take as its
        arguments an array with the same shape as `control_allocation <ControlMechanism.control_allocation>` and two
        integers (the first representing the `net_outcome <ControlMechanism.net_outcome>` for the current
        `control_allocation <ControlMechanism.control_allocation>`, and the second the current iteration of the
        `optimization process <OptimizationFunction_Process>`);  it must return `True` or `False`.

    search_space : list or ndarray
        specifies the `search_space <OptimizationFunction.search_space>` parameter for `function
        <OptimizationControlMechanism.function>`, unless that is specified in a constructor for `function
        <OptimizationControlMechanism.function>`.  Each item must have the same shape as `control_allocation
        <ControlMechanism.control_allocation>`.

    function : OptimizationFunction, function or method
        specifies the function used to optimize the `control_allocation <ControlMechanism.control_allocation>`;
        must take as its sole argument an array with the same shape as `control_allocation
        <ControlMechanism.control_allocation>`, and return a similar array (see `Function
        <OptimizationControlMechanism_Function>` for additional details).

    Attributes
    ----------

    feature_values : 2d array
        the current value of each of the OptimizationControlMechanism's `features
        <OptimizationControlMechanism_Features>` (each of which is a 1d array).

    agent_rep : Composition
        determines the `Composition` used by the `evalution_function <OptimizationControlMechanism.evaluation_function>`
        to predict the `net_outcome <ControlMechanism.net_outcome>` for a given `state
        <OptimizationControlMechanism_State>` (see `above <OptimizationControlMechanism_Agent_Rep>`for additional
        details).

    function : OptimizationFunction, function or method
        takes current `control_allocation <ControlMechanism.control_allocation>` (as initializer),
        uses its `search_function <OptimizationFunction.search_function>` to select samples of `control_allocation
        <ControlMechanism.control_allocation>` from its `search_space <OptimizationFunction.search_space>`,
        evaluates these using its `evaluation_function <OptimizationControlMechanism.evaluation_function>`, and returns
        the one that yields the optimal `net_outcome <ControlMechanism.net_outcome>` (see `Function
        <OptimizationControlMechanism_Function>` for additional details).

    evaluation_function : function or method
        returns `net_outcome <ControlMechanism.net_outcome>` for a given `state <OptimizationControlMechanism_State>`
        (i.e., combination of `feature_values <OptimizationControlMechanism.feature_values>` and `control_allocation
        <ControlMechanism.control_allocation>`. It is assigned as the `objective_function
        <OptimizationFunction.objective_function>` parameter of `function
        <OptimizationControlMechanism.function>`, and calls the `evaluate` method of the OptimizationControlMechanism's
        `agent_rep <OptimizationControlMechanism.agent_rep>` with a `control_allocation
        <ControlMechanism.control_allocation>`, the OptimizationControlMechanism's `num_estimates
        <OptimizationControlMechanism.num_estimates>` attribute, and the current `feature_values
        <OptimizationControlMechanism.feature_values>`.

    COMMENT:
    search_function : function or method
        `search_function <OptimizationFunction.search_function>` assigned to `function
        <OptimizationControlMechanism.function>`; used to select samples of `control_allocation
        <ControlMechanism.control_allocation>` to evaluate by `evaluation_function
        <OptimizationControlMechanism.evaluation_function>`.

    search_termination_function : function or method
        `search_termination_function <OptimizationFunction.search_termination_function>` assigned to
        `function <OptimizationControlMechanism.function>`;  determines when to terminate the
        `optimization process <OptimizationFunction_Process>`.
    COMMENT

    control_allocation_search_space : list of SampleIterators
        `search_space <OptimizationFunction.search_space>` assigned by default to `function
        <OptimizationControlMechanism.function>`, that determines the samples of
        `control_allocation <ControlMechanism.control_allocation>` evaluated by the `evaluation_function
        <OptimizationControlMechanism.evaluation_function>`.  This is a proprety that, unless overridden,
        returns a list of the `SampleIterators <SampleIterator>` generated from the `allocation_sample
        <ControlSignal.allocation_sample>` specifications for each of the OptimizationControlMechanism's
        `control_signals <ControlMechanism.control_signals>`.

    saved_samples : list
        contains all values of `control_allocation <ControlMechanism.control_allocation>` sampled by `function
        <OptimizationControlMechanism.function>` if its `save_samples <OptimizationFunction.save_samples>` parameter
        is `True`;  otherwise list is empty.

    saved_values : list
        contains values of `net_outcome <ControlMechanism.net_outcome>` associated with all samples of
        `control_allocation <ControlMechanism.control_allocation>` evaluated by by `function
        <OptimizationControlMechanism.function>` if its `save_values <OptimizationFunction.save_samples>` parameter
        is `True`;  otherwise list is empty.

    search_statefulness : bool : True
        if set to False, an `OptimizationControlMechanism`\\ 's `evaluation_function` will never run simulations; the
        evaluations will simply execute in the original `execution context <_Execution_Contexts>`.

        if set to True, `simulations <OptimizationControlMechanism_Execution>` will be created normally for each
        `control allocation <control_allocation>`.

    """

    componentType = OPTIMIZATION_CONTROL_MECHANISM
    # initMethod = INIT_FULL_EXECUTE_METHOD
    # initMethod = INIT_EXECUTE_METHOD_ONLY

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TYPE_DEFAULT_PREFERENCES
    # Note: only need to specify setting;  level will be assigned to Type automatically
    # classPreferences = {
    #     PREFERENCE_SET_NAME: 'DefaultControlMechanismCustomClassPreferences',
    #     PREFERENCE_KEYWORD<pref>: <setting>...}

    # FIX: ADD OTHER Parameters() HERE??
    class Parameters(ControlMechanism.Parameters):
        """
            Attributes
            ----------

                agent_rep
                    see `agent_rep <OptimizationControlMechanism_Agent_Rep>`

                    :default value: None
                    :type:

                comp_execution_mode
                    see `comp_execution_mode <OptimizationControlMechanism.comp_execution_mode>`

                    :default value: `PYTHON`
                    :type: ``str``

                control_allocation_search_space
                    see `control_allocation_search_space <OptimizationControlMechanism.control_allocation_search_space>`

                    :default value: None
                    :type:

                feature_function
                    see `feature_function <OptimizationControlMechanism_Feature_Function>`

                    :default value: None
                    :type:

                function
                    see `function <OptimizationControlMechanism_Function>`

                    :default value: None
                    :type:

                input_ports
                    see `input_ports <OptimizationControlMechanism.input_ports>`

                    :default value: ["{name: OUTCOME, params: {internal_only: True}}"]
                    :type: ``list``
                    :read only: True

                num_estimates
                    see `num_estimates <OptimizationControlMechanism.num_estimates>`

                    :default value: None
                    :type:

                saved_samples
                    see `saved_samples <OptimizationControlMechanism.saved_samples>`

                    :default value: None
                    :type:

                saved_values
                    see `saved_values <OptimizationControlMechanism.saved_values>`

                    :default value: None
                    :type:

                search_function
                    see `search_function <OptimizationControlMechanism.search_function>`

                    :default value: None
                    :type:

                search_statefulness
                    see `search_statefulness <OptimizationControlMechanism.search_statefulness>`

                    :default value: True
                    :type: ``bool``

                search_termination_function
                    see `search_termination_function <OptimizationControlMechanism.search_termination_function>`

                    :default value: None
                    :type:
        """
        function = Parameter(None, stateful=False, loggable=False)
        feature_function = Parameter(None, reference=True, stateful=False, loggable=False)
        search_function = Parameter(None, stateful=False, loggable=False)
        search_termination_function = Parameter(None, stateful=False, loggable=False)
        comp_execution_mode = Parameter('Python', stateful=False, loggable=False, pnl_internal=True)
        search_statefulness = Parameter(True, stateful=False, loggable=False)

        agent_rep = Parameter(None, stateful=False, loggable=False, pnl_internal=True, structural=True)

        feature_values = Parameter(_parse_feature_values_from_variable([defaultControlAllocation]), user=False, pnl_internal=True)

        input_ports = Parameter(
            [{NAME: OUTCOME, PARAMS: {INTERNAL_ONLY: True}}],
            stateful=False,
            loggable=False,
            read_only=True,
            structural=True,
            parse_spec=True,
            aliases='features',
            constructor_argument='features'
        )
        num_estimates = None
        # search_space = None
        control_allocation_search_space = None

        saved_samples = None
        saved_values = None

    @handle_external_context()
    @tc.typecheck
    def __init__(self,
                 agent_rep=None,
                 function=None,
                 features: tc.optional(tc.optional(tc.any(Iterable, Mechanism, OutputPort, InputPort))) = None,
                 feature_function: tc.optional(tc.optional(tc.any(is_function_type))) = None,
                 num_estimates = None,
                 search_function: tc.optional(tc.optional(tc.any(is_function_type))) = None,
                 search_termination_function: tc.optional(tc.optional(tc.any(is_function_type))) = None,
                 search_statefulness=None,
                 context=None,
                 **kwargs):
        """Implement OptimizationControlMechanism"""

        # If agent_rep hasn't been specified, put into deferred init
        if agent_rep is None:
            if context.source==ContextFlags.COMMAND_LINE:
                # Temporarily name InputPort
                self._assign_deferred_init_name(self.__class__.__name__, context)
                # Store args for deferred initialization
                self._store_deferred_init_args(**locals())

                # Flag for deferred initialization
                self.initialization_status = ContextFlags.DEFERRED_INIT
                return
            # If constructor is called internally (i.e., for controller of Composition),
            # agent_rep needs to be specified
            else:
                assert False, f"PROGRAM ERROR: 'agent_rep' arg should have been specified " \
                              f"in internal call to constructor for {self.name}."

        super().__init__(
            function=function,
            input_ports=features,
            features=features,
            feature_function=feature_function,
            num_estimates=num_estimates,
            search_statefulness=search_statefulness,
            search_function=search_function,
            search_termination_function=search_termination_function,
            agent_rep=agent_rep,
            **kwargs
        )

    def _validate_params(self, request_set, target_set=None, context=None):
        """Insure that specification of ObjectiveMechanism has projections to it"""

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        from psyneulink.core.compositions.composition import Composition
        if request_set[AGENT_REP] is None:
            raise OptimizationControlMechanismError(f"The {repr(AGENT_REP)} arg of an {self.__class__.__name__} must "
                                                    f"be specified and be a {Composition.__name__}")

        elif not (isinstance(request_set[AGENT_REP], Composition)
                  or (isinstance(request_set[AGENT_REP], type) and issubclass(request_set[AGENT_REP], Composition))):
            raise OptimizationControlMechanismError(f"The {repr(AGENT_REP)} arg of an {self.__class__.__name__} "
                                                    f"must be either a {Composition.__name__} or a sublcass of one")

    def _instantiate_input_ports(self, context=None):
        """Instantiate input_ports for Projections from features and objective_mechanism.

        Inserts InputPort specification for Projection from ObjectiveMechanism as first item in list of
        InputPort specifications generated in _parse_feature_specs from the **features** and
        **feature_function** arguments of the OptimizationControlMechanism constructor.
        """

        # Specify *OUTCOME* InputPort;  receives Projection from *OUTCOME* OutputPort of objective_mechanism
        outcome_input_port = {NAME:OUTCOME, PARAMS:{INTERNAL_ONLY:True}}

        # If any features were specified (assigned to self.input_ports in __init__):
        if self.input_ports:
            self.input_ports = _parse_shadow_inputs(self, self.input_ports)
            self.input_ports = self._parse_feature_specs(self.input_ports, self.feature_function)
            # Insert primary InputPort for outcome from ObjectiveMechanism;
            #     assumes this will be a single scalar value and must be named OUTCOME by convention of ControlSignal
            self.input_ports.insert(0, outcome_input_port),
        else:
            self.input_ports = [outcome_input_port]

        # Configure default_variable to comport with full set of input_ports
        self.defaults.variable, _ = self._handle_arg_input_ports(self.input_ports)

        super()._instantiate_input_ports(context=context)

        for i in range(1, len(self.input_ports)):
            port = self.input_ports[i]
            if len(port.path_afferents) > 1:
                raise OptimizationControlMechanismError(f"Invalid {type(input_port).__name__} on {self.name}. "
                                                        f"{port.name} should receive exactly one projection, "
                                                        f"but it receives {len(port.path_afferents)} projections.")

    def _instantiate_output_ports(self, context=None):
        """Assign CostFunctions.DEFAULTS as default for cost_option of ControlSignals.
        OptimizationControlMechanism requires use of at least one of the cost options
        """
        super()._instantiate_output_ports(context)

        for control_signal in self.control_signals:
            if control_signal.cost_options is None:
                control_signal.cost_options = CostFunctions.DEFAULTS
                control_signal._instantiate_cost_attributes(context)

    def _instantiate_control_signals(self, context):
        """Size control_allocation and assign modulatory_signals
        Set size of control_allocadtion equal to number of modulatory_signals.
        Assign each modulatory_signal sequentially to corresponding item of control_allocation.
        """
        from psyneulink.core.globals.keywords import OWNER_VALUE
        output_port_specs = list(enumerate(self.output_ports))
        for i, spec in output_port_specs:
            control_signal = self._instantiate_control_signal(spec, context=context)
            control_signal._variable_spec = (OWNER_VALUE, i)
            self.output_ports[i] = control_signal
        self.defaults.value = np.tile(control_signal.parameters.variable.default_value, (i + 1, 1))
        self.parameters.control_allocation._set(copy.deepcopy(self.defaults.value), context)

    def _instantiate_function(self, function, function_params=None, context=None):
        # this indicates a significant peculiarity of OCM, in that its function
        # corresponds to its value (control_allocation) rather than anything to
        # do with its variable. see _instantiate_attributes_after_function

        # Workaround this issue here, and explicitly allow the function's
        # default variable to be modified
        if isinstance(function, Function):
            function._variable_shape_flexibility = DefaultsFlexibility.FLEXIBLE

        super()._instantiate_function(function, function_params, context)

    def _instantiate_attributes_after_function(self, context=None):
        """Instantiate OptimizationControlMechanism's OptimizatonFunction attributes"""

        super()._instantiate_attributes_after_function(context=context)
        # Assign parameters to function (OptimizationFunction) that rely on OptimizationControlMechanism
        self.function.reset({DEFAULT_VARIABLE: self.control_allocation,
                                    OBJECTIVE_FUNCTION: self.evaluation_function,
                                    # SEARCH_FUNCTION: self.search_function,
                                    # SEARCH_TERMINATION_FUNCTION: self.search_termination_function,
                                    SEARCH_SPACE: self.control_allocation_search_space
                                    })

        if isinstance(self.agent_rep, type):
            self.agent_rep = self.agent_rep()

        from psyneulink.core.compositions.compositionfunctionapproximator import CompositionFunctionApproximator
        if (isinstance(self.agent_rep, CompositionFunctionApproximator)):
            self._initialize_composition_function_approximator(context)

    def _update_input_ports(self, runtime_params=None, context=None):
        """Update value for each InputPort in self.input_ports:

        Call execute method for all (MappingProjection) Projections in Port.path_afferents
        Aggregate results (using InputPort execute method)
        Update InputPort.value
        """
        # "Outcome"
        outcome_input_port = self.input_port
        outcome_input_port._update(params=runtime_params, context=context)
        port_values = [np.atleast_2d(outcome_input_port.parameters.value._get(context))]
        # MODIFIED 5/8/20 OLD:
        # FIX 5/8/20 [JDC]: THIS DOESN'T CALL SUPER, SO NOT IDEAL HOWEVER, REVISION BELOW CRASHES... NEEDS TO BE FIXED
        for i in range(1, len(self.input_ports)):
            port = self.input_ports[i]
            port._update(params=runtime_params, context=context)
            port_values.append(port.parameters.value._get(context))
        return convert_to_np_array(port_values)
        # # MODIFIED 5/8/20 NEW:
        # input_port_values = super()._update_input_ports(runtime_params, context)
        # port_values.append(input_port_values)
        # return np.array(port_values)
        # MODIFIED 5/8/20 END

    def _execute(self, variable=None, context=None, runtime_params=None):
        """Find control_allocation that optimizes result of `agent_rep.evaluate`  ."""

        if self.is_initializing:
            return [defaultControlAllocation]

        # # FIX: THESE NEED TO BE FOR THE PREVIOUS TRIAL;  ARE THEY FOR FUNCTION_APPROXIMATOR?
        self.parameters.feature_values._set(_parse_feature_values_from_variable(variable), context)

        # Assign default control_allocation if it is not yet specified (presumably first trial)
        control_allocation = self.parameters.control_allocation._get(context)
        if control_allocation is None:
            control_allocation = [c.defaults.variable for c in self.control_signals]
            self.parameters.control_allocation._set(control_allocation, context=None)

        # Give the agent_rep a chance to adapt based on last trial's feature_values and control_allocation
        if hasattr(self.agent_rep, "adapt"):
            # KAM 4/11/19 switched from a try/except to hasattr because in the case where we don't
            # have an adapt method, we also don't need to call the net_outcome getter
            net_outcome = self.parameters.net_outcome._get(context)

            self.agent_rep.adapt(_parse_feature_values_from_variable(variable),
                                 control_allocation,
                                 net_outcome,
                                 context=context)

        # freeze the values of current context, because they can be changed in between simulations,
        # and the simulations must start from the exact spot
        self.agent_rep._initialize_from_context(self._get_frozen_context(context),
                                                base_context=context,
                                                override=True)

        # Get control_allocation that optimizes net_outcome using OptimizationControlMechanism's function
        # IMPLEMENTATION NOTE: skip ControlMechanism._execute since it is a stub method that returns input_values
        optimal_control_allocation, optimal_net_outcome, saved_samples, saved_values = \
                                                super(ControlMechanism,self)._execute(variable=control_allocation,
                                                                                      context=context,
                                                                                      runtime_params=runtime_params,
                                                                                      )

        # clean up frozen values after execution
        self.agent_rep._delete_contexts(self._get_frozen_context(context))

        optimal_control_allocation = np.array(optimal_control_allocation).reshape((len(self.defaults.value), 1))
        if self.function.save_samples:
            self.saved_samples = saved_samples
        if self.function.save_values:
            self.saved_values = saved_values

        # Return optimal control_allocation
        return optimal_control_allocation

    def _get_frozen_context(self, context=None):
        return Context(execution_id=f'{context.execution_id}{EID_FROZEN}')

    def _set_up_simulation(self, base_context=Context(execution_id=None), control_allocation=None):
        sim_context = copy.copy(base_context)
        sim_context.execution_id = self.get_next_sim_id(base_context)

        if control_allocation is not None:
            sim_context.execution_id += f'-{control_allocation}'

        try:
            self.parameters.simulation_ids._get(base_context).append(sim_context.execution_id)
        except AttributeError:
            self.parameters.simulation_ids._set([sim_context.execution_id], base_context)

        self.agent_rep._initialize_from_context(sim_context, self._get_frozen_context(base_context), override=False)

        return sim_context

    def _tear_down_simulation(self, sim_context=None):
        if not self.agent_rep.parameters.retain_old_simulation_data._get():
            self.agent_rep._delete_contexts(sim_context, check_simulation_storage=True)

    def evaluation_function(self, control_allocation, context=None, return_results=False):
        """Compute `net_outcome <ControlMechanism.net_outcome>` for current set of `feature_values
        <OptimizationControlMechanism.feature_values>` and a specified `control_allocation
        <ControlMechanism.control_allocation>`.

        Assigned as the `objective_function <OptimizationFunction.objective_function>` for the
        OptimizationControlMechanism's `function <OptimizationControlMechanism.function>`.

        Calls `agent_rep <OptimizationControlMechanism.agent_rep>`\\'s `evalute` method.

        Returns a scalar that is the predicted `net_outcome <ControlMechanism.net_outcome>`
        for the current `feature_values <OptimizationControlMechanism.feature_values>`
        and specified `control_allocation <ControlMechanism.control_allocation>`.

        """
        # agent_rep is a Composition (since runs_simuluations = True)
        if self.agent_rep.runs_simulations:
            # KDM 5/20/19: crudely using default here because it is a stateless parameter
            # and there is a bug in setting parameter values on init, see TODO note above
            # call to self._instantiate_defaults around component.py:1115
            if self.defaults.search_statefulness:
                new_context = self._set_up_simulation(context, control_allocation)
            else:
                new_context = context

            old_composition = context.composition
            context.composition = self.agent_rep

            # We shouldn't get this far if execution mode is not Python
            exec_mode = self.parameters.comp_execution_mode._get(context)
            assert exec_mode == "Python"
            result = self.agent_rep.evaluate(self.parameters.feature_values._get(context),
                                             control_allocation,
                                             self.parameters.num_estimates._get(context),
                                             base_context=context,
                                             context=new_context,
                                             execution_mode=exec_mode,
                                             return_results=return_results)
            context.composition = old_composition

            if self.defaults.search_statefulness:
                self._tear_down_simulation(new_context)

            # If results of the simulation shoudld be returned then, do so. Agent Rep Evaluate will
            # return a tuple in this case where the first element is the outcome as usual and the
            # results of composision run are the second element.
            if return_results:
                return result[0], result[1]
            else:
                return result

        # agent_rep is a CompositionFunctionApproximator (since runs_simuluations = False)
        else:
            result = self.agent_rep.evaluate(self.parameters.feature_values._get(context),
                                             control_allocation,
                                             self.parameters.num_estimates._get(context),
                                             context=context
            )

        return result

    def _get_evaluate_input_struct_type(self, ctx):
        # We construct input from optimization function input
        return ctx.get_input_struct_type(self.function)

    def _get_evaluate_output_struct_type(self, ctx):
        # Returns a scalar that is the predicted net_outcome
        return ctx.float_ty

    def _get_evaluate_alloc_struct_type(self, ctx):
        return pnlvm.ir.ArrayType(ctx.float_ty,
                                  len(self.control_allocation_search_space))

    def _gen_llvm_net_outcome_function(self, *, ctx, tags=frozenset()):
        assert "net_outcome" in tags
        args = [ctx.get_param_struct_type(self).as_pointer(),
                ctx.get_state_struct_type(self).as_pointer(),
                self._get_evaluate_alloc_struct_type(ctx).as_pointer(),
                ctx.float_ty.as_pointer(),
                ctx.float_ty.as_pointer()]

        builder = ctx.create_llvm_function(args, self, str(self) + "_net_outcome")
        llvm_func = builder.function
        for p in llvm_func.args:
            p.attributes.add('nonnull')
        params, state, allocation_sample, objective_ptr, arg_out = llvm_func.args

        op_params = pnlvm.helpers.get_param_ptr(builder, self, params,
                                                "output_ports")
        op_states = pnlvm.helpers.get_state_ptr(builder, self, state,
                                                "output_ports", None)

        # calculate cost function
        total_cost = builder.alloca(ctx.float_ty)
        builder.store(ctx.float_ty(-0.0), total_cost)
        for i, op in enumerate(self.output_ports):
            op_i_params = builder.gep(op_params, [ctx.int32_ty(0),
                                                  ctx.int32_ty(i)])
            op_i_state = builder.gep(op_states, [ctx.int32_ty(0),
                                                 ctx.int32_ty(i)])

            op_f = ctx.import_llvm_function(op, tags=frozenset({"costs"}))

            op_in = builder.alloca(op_f.args[2].type.pointee)

            # copy allocation_sample, the input is 1-element array in a struct
            data_in = builder.gep(allocation_sample, [ctx.int32_ty(0),
                                                      ctx.int32_ty(i)])
            data_out = builder.gep(op_in, [ctx.int32_ty(0), ctx.int32_ty(0),
                                           ctx.int32_ty(0)])
            builder.store(builder.load(data_in), data_out)

            # Invoke cost function
            cost = builder.call(op_f, [op_i_params, op_i_state, op_in])

            # simplified version of combination fmax(cost, 0)
            ltz = builder.fcmp_ordered("<", cost, cost.type(0))
            cost = builder.select(ltz, cost.type(0), cost)

            # combine is not a PNL function
            assert self.combine_costs is np.sum
            val = builder.load(total_cost)
            val = builder.fadd(val, cost)
            builder.store(val, total_cost)

        # compute net outcome
        objective = builder.load(objective_ptr)
        net_outcome = builder.fsub(objective, builder.load(total_cost))
        builder.store(net_outcome, arg_out)

        builder.ret_void()
        return llvm_func

    def _gen_llvm_evaluate_function(self, *, ctx:pnlvm.LLVMBuilderContext,
                                             tags=frozenset()):
        assert "evaluate" in tags
        args = [ctx.get_param_struct_type(self.agent_rep).as_pointer(),
                ctx.get_state_struct_type(self.agent_rep).as_pointer(),
                self._get_evaluate_alloc_struct_type(ctx).as_pointer(),
                self._get_evaluate_output_struct_type(ctx).as_pointer(),
                self._get_evaluate_input_struct_type(ctx).as_pointer(),
                ctx.get_data_struct_type(self.agent_rep).as_pointer()]

        builder = ctx.create_llvm_function(args, self, str(self) + "_evaluate")
        llvm_func = builder.function
        for p in llvm_func.args:
            p.attributes.add('nonnull')

        comp_params, base_comp_state, allocation_sample, arg_out, arg_in, base_comp_data = llvm_func.args

        # Create a simulation copy of composition state
        comp_state = builder.alloca(base_comp_state.type.pointee, name="state_copy")
        builder.store(builder.load(base_comp_state), comp_state)

        # Create a simulation copy of composition data
        comp_data = builder.alloca(base_comp_data.type.pointee, name="data_copy")
        builder.store(builder.load(base_comp_data), comp_data)

        # Evaluate is called on composition controller
        assert self.composition.controller is self
        assert self.composition is self.agent_rep
        nodes_states = pnlvm.helpers.get_state_ptr(builder, self.composition,
                                                   comp_state, "nodes", None)
        nodes_params = pnlvm.helpers.get_param_ptr(builder, self.composition,
                                                   comp_params, "nodes")

        controller_idx = self.composition._get_node_index(self)
        controller_state = builder.gep(nodes_states, [ctx.int32_ty(0),
                                                      ctx.int32_ty(controller_idx)])
        controller_params = builder.gep(nodes_params, [ctx.int32_ty(0),
                                                       ctx.int32_ty(controller_idx)])

        # Get simulation function
        sim_f = ctx.import_llvm_function(self.agent_rep,
                                         tags=frozenset({"run", "simulation"}))

        # Apply allocation sample to simulation data
        assert len(self.output_ports) == len(allocation_sample.type.pointee)
        idx = self.agent_rep._get_node_index(self)
        ocm_out = builder.gep(comp_data, [ctx.int32_ty(0), ctx.int32_ty(0),
                                          ctx.int32_ty(idx)])
        for i, _ in enumerate(self.output_ports):
            idx = ctx.int32_ty(i)
            sample_ptr = builder.gep(allocation_sample, [ctx.int32_ty(0), idx])
            sample_dst = builder.gep(ocm_out, [ctx.int32_ty(0), idx, ctx.int32_ty(0)])
            if sample_ptr.type != sample_dst.type:
                assert len(sample_dst.type.pointee) == 1
                sample_dst = builder.gep(sample_dst, [ctx.int32_ty(0),
                                                      ctx.int32_ty(0)])
            builder.store(builder.load(sample_ptr), sample_dst)

        # Construct input
        comp_input = builder.alloca(sim_f.args[3].type.pointee, name="sim_input")

        for src_idx, ip in enumerate(self.input_ports):
            if ip.shadow_inputs is None:
                continue
            cim_in_port = self.agent_rep.input_CIM_ports[ip.shadow_inputs][0]
            dst_idx = self.agent_rep.input_CIM.input_ports.index(cim_in_port)

            src = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(src_idx)])
            # Destination is a struct of 2d arrays
            dst = builder.gep(comp_input, [ctx.int32_ty(0),
                                           ctx.int32_ty(dst_idx),
                                           ctx.int32_ty(0)])
            builder.store(builder.load(src), dst)


        # Determine simulation counts
        num_estimates_ptr = pnlvm.helpers.get_param_ptr(builder, self,
                                                        controller_params,
                                                        "num_estimates")

        num_estimates = builder.load(num_estimates_ptr, "num_estimates")

        # if num_estimates is 0, run 1 trial
        param_is_zero = builder.icmp_unsigned("==", num_estimates,
                                                    ctx.int32_ty(0))
        num_sims = builder.select(param_is_zero, ctx.int32_ty(1),
                                  num_estimates, "corrected_estimates")

        num_runs = builder.alloca(ctx.int32_ty, name="num_runs")
        builder.store(num_sims, num_runs)

        # We only provide one input
        num_inputs = builder.alloca(ctx.int32_ty, name="num_inputs")
        builder.store(num_inputs.type.pointee(1), num_inputs)

        # Simulations don't store output
        comp_output = sim_f.args[4].type(None)
        builder.call(sim_f, [comp_state, comp_params, comp_data, comp_input,
                             comp_output, num_runs, num_inputs])

        # Extract objective mechanism value
        idx = self.agent_rep._get_node_index(self.objective_mechanism)
        # Mechanisms' results are stored in the first substructure
        objective_os_ptr = builder.gep(comp_data, [ctx.int32_ty(0),
                                                   ctx.int32_ty(0),
                                                   ctx.int32_ty(idx)])
        # Objective mech output shape should be 1 single element 2d array
        objective_val_ptr = builder.gep(objective_os_ptr,
                                        [ctx.int32_ty(0), ctx.int32_ty(0),
                                         ctx.int32_ty(0)], "obj_val_ptr")

        net_outcome_f = ctx.import_llvm_function(self, tags=tags.union({"net_outcome"}))
        builder.call(net_outcome_f, [controller_params, controller_state,
                                     allocation_sample, objective_val_ptr,
                                     arg_out])

        builder.ret_void()

        return llvm_func

    def _gen_llvm_function(self, *, ctx:pnlvm.LLVMBuilderContext, tags:frozenset):
        if "net_outcome" in tags:
            return self._gen_llvm_net_outcome_function(ctx=ctx, tags=tags)
        if "evaluate" in tags:
            return self._gen_llvm_evaluate_function(ctx=ctx, tags=tags)

        is_comp = not isinstance(self.agent_rep, Function)
        if is_comp:
            extra_args = [ctx.get_param_struct_type(self.agent_rep).as_pointer(),
                          ctx.get_state_struct_type(self.agent_rep).as_pointer(),
                          ctx.get_data_struct_type(self.agent_rep).as_pointer()]
        else:
            extra_args = []

        f = super()._gen_llvm_function(ctx=ctx, extra_args=extra_args, tags=tags)
        if is_comp:
            for a in f.args[-len(extra_args):]:
                a.attributes.add('nonnull')

        return f

    def _gen_llvm_invoke_function(self, ctx, builder, function, params, context, variable, *, tags:frozenset):
        fun = ctx.import_llvm_function(function)
        fun_out = builder.alloca(fun.args[3].type.pointee)

        args = [params, context, variable, fun_out]
        # If we're calling compiled version of Composition.evaluate,
        # we need to pass extra arguments
        if len(fun.args) > 4:
            args += builder.function.args[-3:]
        builder.call(fun, args)

        return fun_out, builder

    def _gen_llvm_output_port_parse_variable(self, ctx, builder, params, context, value, port):
        i = self.output_ports.index(port)
        # Allocate the only member of the port input struct
        oport_input = builder.alloca(ctx.get_input_struct_type(port).elements[0])
        # FIXME: workaround controller signals occasionally being 2d
        dest_ptr = pnlvm.helpers.unwrap_2d_array(builder, oport_input)
        dest_ptr = builder.gep(dest_ptr, [ctx.int32_ty(0), ctx.int32_ty(0)])
        val_ptr = builder.gep(value, [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(i)])
        builder.store(builder.load(val_ptr), dest_ptr)
        return oport_input

    # @property
    # def feature_values(self):
    #     if hasattr(self.agent_rep, 'model_based_optimizer') and self.agent_rep.model_based_optimizer is self:
    #         return self.agent_rep._get_predicted_input()
    #     else:
    #         return np.array(np.array(self.variable[1:]).tolist())

    # FIX: THE FOLLOWING SHOULD BE MERGED WITH HANDLING OF PredictionMechanisms FOR ORIG MODEL-BASED APPROACH;
    # FIX: SHOULD BE GENERALIZED AS SOMETHING LIKE update_feature_values

    @tc.typecheck
    def add_features(self, features):
        """Add InputPorts and Projections to OptimizationControlMechanism for features used to
        predict `net_outcome <ControlMechanism.net_outcome>`

        **features** argument can use any of the forms of specification allowed for InputPort(s)
        """

        if features:
            features = self._parse_feature_specs(features=features,
                                                 context=Context(source=ContextFlags.COMMAND_LINE))
        self.add_ports(InputPort, features)

    @tc.typecheck
    def _parse_feature_specs(self, input_ports, feature_function, context=None):
        """Parse entries of features into InputPort spec dictionaries
        Set INTERNAL_ONLY entry of params dict of InputPort spec dictionary to True
            (so that inputs to Composition are not required if the specified state is on an INPUT Mechanism)
        Assign functions specified in **feature_function** to InputPorts for all features
        Return list of InputPort specification dictionaries
        """

        parsed_features = []

        if not isinstance(input_ports, list):
            input_ports = [input_ports]

        for spec in input_ports:
            spec = _parse_port_spec(owner=self, port_type=InputPort, port_spec=spec)    # returns InputPort dict
            spec[PARAMS][INTERNAL_ONLY] = True
            if feature_function:
                if isinstance(feature_function, Function):
                    feat_fct = copy.deepcopy(feature_function)
                else:
                    feat_fct = feature_function
                spec.update({FUNCTION: feat_fct})
            spec = [spec]   # so that extend works below

            parsed_features.extend(spec)

        return parsed_features

    @property
    def control_allocation_search_space(self):
        """Return list of SampleIterators for allocation_samples of control_signals"""
        return [c.allocation_samples for c in self.control_signals]

    @property
    def _model_spec_parameter_blacklist(self):
        # default_variable is hidden in constructor arguments,
        # and anyway assigning it is problematic because it is modified
        # several times when creating input ports, and assigning function that
        # fits the control allocation
        return super()._model_spec_parameter_blacklist.union({
            'variable',
        })

    # ******************************************************************************************************************
    # FIX:  THE FOLLOWING IS SPECIFIC TO CompositionFunctionApproximator AS agent_rep
    # ******************************************************************************************************************

    def _initialize_composition_function_approximator(self, context):
        """Initialize CompositionFunctionApproximator"""

        # CompositionFunctionApproximator needs to have access to control_signals to:
        # - to construct control_allocation_search_space from their allocation_samples attributes
        # - compute their values and costs for samples of control_allocations from control_allocation_search_space
        self.agent_rep.initialize(features_array=np.array(self.defaults.variable[1:]),
                                  control_signals = self.control_signals,
                                  context=context)
