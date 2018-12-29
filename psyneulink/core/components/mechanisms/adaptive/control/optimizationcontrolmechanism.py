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
The result returned by the `evaluation_function` <OptimizationControlMechanism.evalutate_function>` is ordinally
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
In each `trial`, the `agent_rep <OptimizationControlMechanism.agent_rep>` is given the chance to adapt, by adjusting its
parameters in order to improve its prediction of the `net_outcome <ControlMechanism.net_outcome>` for the Composition
(or part of one) that is controlled by the OptimizationControlMechanism (based on the `state
<OptimizationControlMechanism_State>` and `net_outcome <ControlMechanism.net_outcome>` of
the prior trial).  The `agent_rep <OptimizationControlMechanism.agent_rep>` is then used to predict the `net_outcome
<ControlMechanism.net_outcome>` for `control_allocation <ControlMechanism.control_allocation>` samples to find the one
that yields the best predicted `net_outcome <ControlMechanism.net_outcome>` of processing on the upcoming trial,
based on the current or (expected) `feature_values <OptimizationControlMechanism.feature_values>` for that trial.

.. _OptimizationControlMechanism_Model_Based:

*Model-Based Optimization*

This is achieved by assigning as the `agent_rep  <OptimizationControlMechanism.agent_rep>` the Composition to which
the OptimizationControlMechanism belongs (and for which it is the `controller <Composition.controller>`). On each
`trial`, that Composition itself is used to simulate processing on the upcoming trial, based on the current or
(expected) `feature_values <OptimizationControlMechanism.feature_values>` for that trial, in order to find the
<ControlMechanism.control_allocation>` that yields the best net_outcome <ControlMechanism.net_outcome>` for that trial.

.. _OptimizationControlMechanism_Creation:

Creating an OptimizationControlMechanism
----------------------------------------

An OptimizationControlMechanism is created in the same was as any `ControlMechanism <ControlMechanism>`.
The following arguments of its constructor are specific to the OptimizationControlMechanism:

* **features** -- takes the place of the standard **input_states** argument in the constructor for a
  Mechanism`, and specifies the values used by the OptimizationControlMechanism, together with a `control_allocation
  <ControlMechanism.control_allocation>`, to calculate a `net_outcome <ControlMechanism.net_outcome>`.  Features can be
  specified using any of the following, singly or combined in a list:

  * {*SHADOW_EXTERNAL_INPUTS*: <`ORIGIN` Mechanism, InputState for one, or list with either or both>} --
    InputStates of the same shapes as those listed are created on the ModelFreeOptimizationControlMechanism,
    and are connected to the corresponding input_CIM OutputStates by projections. The external input values
    that are passed through the input_CIM are used as the `features <ModelFreeOptimizationControlMechanism_Feature>`.
    If a Mechanism is included in the list, it refers to all of its InputStates.
  |
  * *InputState specification* -- this can be any form of `InputState specification <InputState_Specification>`
    that resolves to an OutputState from which the InputState receives a Projection;  the `value
    <OutputState.value>` of that OutputState is used as the feature. Each of these InputStates is marked as
    `internal_only <InputStates.internal_only>` = `True`.

  Features can also be added to an existing OptimizationControlMechanism using its `add_features` method.  If the
  **features** argument is not specified, then the `input <Composition.input_values>` to the `Composition` on the last
  trial of its execution is used to predict the `net_outcome <ControlMechanism.net_outcome>` for the upcoming trial.

.. _OptimizationControlMechanism_Feature_Function:

* **feature_function** -- specifies `function <InputState>` of the InputState created for each item listed in
  **features**.  By default, this is the identity function, that assigns the current value of the feature
  to the OptimizationControlMechanism's `feature_values <OptimizationControlMechanism.feature_values>` attribute.
  However, other functions can be assigned, for example to maintain a record of past values, or integrate them over
  trials.
..
* **agent_rep** -- specifies the `Composition` used by the OptimizationControlMechanism's `evaluation_function
  <OptimizationControlMechanism.evaluation_function>` to calculate the predicted `net_outcome
  <ControlMechanism.net_outcome>` for a given `state <OptimizationControlMechanism_State>` (see `below
  <OptimizationControlMechanism_Agent_Rep>` for additional details). If it is not specified, the
  `Composition` to which the OptimizationControlMechanism belongs is assigned, and the OptimizationControlMechanism
  is assigned as that Composition's `controller <Composition.controller>`, implementing `model-based
  <OptimizationControlMechanism_Model_Based>` optimization.  If that Composition already has a `controller
  <Composition.controller>` specified, the OptimizationControlMechanism is disable. If another Composition is
  specified, it must conform to the specifications for an `agent_rep <OptimizationControlMechanism.agent_rep>` as
  described `below <OptimizationControlMechanism_Agent_Rep>`.

.. _OptimizationControlMechanism_Structure:

Structure
---------

In addition to the standard Components associated with a `ControlMechanism`, including a `Projection <Projection>`
to its *OUTCOME* InputState from its `objective_mechanism <ControlMechanism.objective_mechanism>`, and a
`function <OptimizationControlMechanism.function>` used to carry out the optimization process, it has several
other constiuents, as described below.

.. _OptimizationControlMechanism_ObjectiveMechanism:

*ObjectiveMechanism*
^^^^^^^^^^^^^^^^^^^^

Like any `ControlMechanism`, an OptimizationControlMechanism has an associated `objective_mechanism
<ControlMechanism.objective_mechanism>` that is used to evaluate the outcome of processing for a given trial and pass
the result to the OptimizationControlMechanism, which it places in its `outcome <OptimizationControlMechanism.outcome>`
attribute.  This is used by its `compute_net_outcome <ControlMechanism.compute_net_outcome>` function, together with
the `costs <ControlMechanism.costs>` of its `control_signals <ControlMechanism.control_signals>`, to compute the
`net_outcome <ControlMechanism.net_outcome>` of processing for a given `state <OptimizationControlMechanism_State>`,
and that is returned by `evaluation` method of the OptimizationControlMechanism's `agent_rep
<OptimizationControlMechanism.agent_rep>`.

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

In addition to its `primary InputState <InputState_Primary>` (which receives a projection from the *OUTCOME*
OutpuState of the `objective_mechanism <ControlMechanism.objective_mechanism>`,
an OptimizationControlMechanism also has an `InputState` for each of its features. By default, these are the current
`input <Composition.input_values>` for the Composition to which the OptimizationControlMechanism belongs.  However,
different values can be specified, as can a `feature_function <OptimizationControlMechanism_Feature_Function>` that
transforms these.  For OptimizationControlMechanisms that implement `model-free
<OptimizationControlMechanism_Model_Free>` optimization, its `feature_values
<OptimizationControlMechanism.feature_values>` are used by its `evaluation_function
<OptimizationControlMechanism.evaluation_function>` to predict the `net_outcome <ControlMechanism.net_outcome>` for a
given `control_allocation <ControlMechanism.control_allocation>`.  For OptimizationControlMechanisms that implement
`model-based <OptimizationControlMechanism_Model_Based>` optimization, the `feature_values
<OptimizationControlMechanism.feature_values>` are used as the Composition's `input <Composition.input_values>` when
it is executed to evaluate the `net_outcome <ControlMechanism.net_outcome>` for a given
`control_allocation<ControlMechanism.control_allocation>`.

Features can be of two types:

* *Input Features* -- these are values received as input by `ORIGIN` Mechanisms of the `Composition`.
  They are specified in the **features** argument of the OptimizationControlMechanism's constructor (see
  `OptimizationControlMechanism_Creation`), in a dictionary containing a *SHADOW_EXTERNAL_INPUTS* entry,
  the value of which is one or more `ORIGIN` Mechanisms and/or their `InputStates
  <InputState>` to be shadowed.  For each, a `Projection` is automatically created that parallels ("shadows") the
  Projection from the Composition's `InputCIM` to the `ORIGIN` Mechanism, projecting from the same `OutputState` of
  the InputCIM to the InputState of the ModelFreeOptimizationControlMechanism assigned to that feature_predictor.
..
* *Output Features* -- these are the `value <OutputState.value>` of an `OutputState` of some other `Mechanism` in the
  Composition.  These too are specified in the **features** argument of the OptimizationControlMechanism's
  constructor (see `OptimizationControlMechanism_Creation`), and each is assigned a `Projection` from the specified
  OutputState(s) to the InputState of the OptimizationControlMechanism for that feature.

The current `value <InputState.value>` of the InputStates for the features are listed in the `feature_values
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
argument of its constructor and assigned to its `agent_rep <OptimizationControlMechanism.agent_rep>` attribute.
This designates a representation of the `Composition` (or parts of one) that the OptimizationControlMechanism controls,
that is used to evaluate sample `control_allocations <ControlMechanism.control_allocation>` in order to find the one
that optimizes the `net_outcome <ControlMechanism.net_outcome>`. The `agent_rep
<OptimizationControlMechanism.agent_rep>` is always itself a `Composition`, that can be either the same one that the
OptimizationControlMechanism controls or another one that is used to estimate the `net_outcome
<ControlMechanism.net_outcome>` for that Composition (see `above
<OptimizationControlMechanism_Agent_Representation_Types>`).  The `evaluate <Composition.evaluate>` method of the
Composition is assigned as the `evaluation_function <OptimizationControlMechanism.evaluation_function>` of the
OptimizationControlMechanism.  If the `agent_rep <OptimizationControlMechanism.agent_rep>` is not the Composition
for which the OptimizationControlMechanism is the controller, then it must meet the following requirements:

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
  - It must implement a `reinitialize` method that accepts **objective_function** as a keyword argument and
    implements an attribute with the same name.

    COMMENT:
    - it must implement a `reinitialize` method that accepts as keyword arguments **objective_function**,
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
import itertools
import numbers
import numpy as np
import typecheck as tc

from collections import Iterable, namedtuple
from typing import NamedTuple

from psyneulink.core.components.functions.function import Function_Base, ModulationParam, _is_modulation_param, is_function_type
from psyneulink.core.components.functions.optimizationfunctions import OBJECTIVE_FUNCTION, SEARCH_SPACE, SampleIterator
from psyneulink.core.components.mechanisms.adaptive.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.shellclasses import Function
from psyneulink.core.components.states.featureinputstate import FeatureInputState
from psyneulink.core.components.states.inputstate import InputState
from psyneulink.core.components.states.modulatorysignals.controlsignal import ControlSignal, ControlSignalCosts
from psyneulink.core.components.states.outputstate import OutputState
from psyneulink.core.components.states.parameterstate import ParameterState
from psyneulink.core.components.states.state import _parse_state_spec
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.defaults import defaultControlAllocation
from psyneulink.core.globals.keywords import DEFAULT_VARIABLE, FUNCTION, INTERNAL_ONLY, NAME, OPTIMIZATION_CONTROL_MECHANISM, OUTCOME, PARAMETER_STATES, PARAMS, VARIABLE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import is_iterable

__all__ = [
    'OptimizationControlMechanism', 'OptimizationControlMechanismError',
    'AGENT_REP', 'FEATURES', 'SHADOW_EXTERNAL_INPUTS'
]

AGENT_REP = 'agent_rep'
FEATURES = 'features'
SHADOW_EXTERNAL_INPUTS = 'SHADOW_EXTERNAL_INPUTS'


def _parse_feature_values_from_variable(variable):
    return np.array(np.array(variable[1:]).tolist())


class OptimizationControlMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class OptimizationControlMechanism(ControlMechanism):
    """OptimizationControlMechanism(            \
    objective_mechanism=None,                   \
    features=None,                              \
    feature_function=None,                      \
    agent_rep=None,                             \
    search_function=None,                       \
    search_termination_function=None,           \
    search_space=None,                          \
    function=None,                              \
    control_signals=None,                       \
    modulation=ModulationParam.MULTIPLICATIVE,  \
    params=None,                                \
    name=None,                                  \
    prefs=None)

    Subclass of `ControlMechanism <ControlMechanism>` that adjusts its `ControlSignals <ControlSignal>` to optimize
    performance of the `Composition` to which it belongs

    .. note::
       OptimizationControlMechanism is an abstract class and should NEVER be instantiated by a call to its constructor.
       It should be instantiated using the constructor for a subclass.

    Arguments
    ---------

    objective_mechanism : ObjectiveMechanism or List[OutputState specification]
        specifies either an `ObjectiveMechanism` to use for the OptimizationControlMechanism, or a list of the
        `OutputState <OutputState>`\\s it should monitor; if a list of `OutputState specifications
        <ObjectiveMechanism_Monitored_Output_States>` is used, a default ObjectiveMechanism is created and the list
        is passed to its **monitored_output_states** argument.

    features : Mechanism, OutputState, Projection, dict, or list containing any of these : default
    {SHADOW_EXTERNAL_INPUTS : ALL}
        specifies Components, the values of which are assigned to `feature_values
        <OptimizationControlMechanism.feature_values>` and used to predict `net_outcome <ControlMechanism.net_outcome>`.
        Any `InputState specification <InputState_Specification>` can be used that resolves to an `OutputState` that
        projects to the InputState. In addition, a dictionary with a *SHADOW_EXTERNAL_INPUTS* entry can be used to
        shadow inputs to the Composition's `ORIGIN` Mechanism(s) (see `above <OptimizationControlMechanism_Creation>`
        for details).

    feature_function : Function or function : default None
        specifies the `function <InputState.function>` for the `InputState` assigned to each `feature
        <OptimizationControlMechanism_Features>`.

    agent_rep : Composition  : default Composition to which the OptimizationControlMechanism belongs
        specifies the `Composition` used by the `evalution_function <OptimizationControlMechanism.evaluation_function>`
        to predictd the `net_outcome <ControlMechanism.net_outcome>` for a given `state
        <OptimizationControlMechanism_State>`.  If a Composition other than the default is assigned,
        it must be suitably configured (see `above <OptimizationControlMechanism_Agent_Rep>` for additional details).
        If the default is used, the OptimizationControlMechanism is assigned as the Composition's `controller
        <Composition.controller>` unless one has already been assigned, in which case the
        OptimizationControlMechanismit is disabled.

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

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for the
        OptimizationControlMechanism, its `function <OptimizationControlMechanism.function>`, and/or a custom function
        and its parameters.  Values specified for parameters in the dictionary override any assigned to those
        parameters in arguments of the constructor.

    name : str : default see `name <OptimizationControlMechanism.name>`
        specifies the name of the OptimizationControlMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the OptimizationControlMechanism; see `prefs
        <OptimizationControlMechanism.prefs>` for details.

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

    name : str
        name of the OptimizationControlMechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the OptimizationControlMechanism; if it is not specified in the **prefs** argument of
        the constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentType = OPTIMIZATION_CONTROL_MECHANISM
    # initMethod = INIT_FULL_EXECUTE_METHOD
    # initMethod = INIT_EXECUTE_METHOD_ONLY

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to Type automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'DefaultControlMechanismCustomClassPreferences',
    #     kp<pref>: <setting>...}

    # FIX: ADD OTHER Parameters() HERE??
    class Parameters(ControlMechanism.Parameters):
        """
            Attributes
            ----------

                agent_rep
                    see `agent_rep <OptimizationControlMechanism.agent_rep>`

                    :default value: None
                    :type:

                control_allocation_search_space
                    see `control_allocation_search_space <OptimizationControlMechanism.control_allocation_search_space>`

                    :default value: None
                    :type:

                feature_function
                    see `feature_function <OptimizationControlMechanism.feature_function>`

                    :default value: None
                    :type:

                features
                    see `features <OptimizationControlMechanism.features>`

                    :default value: None
                    :type:

                function
                    see `function <OptimizationControlMechanism.function>`

                    :default value: None
                    :type:

                num_estimates
                    see `num_estimates <OptimizationControlMechanism.num_estimates>`

                    :default value: 1
                    :type: int

                search_function
                    see `search_function <OptimizationControlMechanism.search_function>`

                    :default value: None
                    :type:

                search_termination_function
                    see `search_termination_function <OptimizationControlMechanism.search_termination_function>`

                    :default value: None
                    :type:

        """
        function = Parameter(None, stateful=False, loggable=False)
        feature_function = Parameter(None, stateful=False, loggable=False)
        search_function = Parameter(None, stateful=False, loggable=False)
        search_termination_function = Parameter(None, stateful=False, loggable=False)

        agent_rep = Parameter(None, stateful=False, loggable=False)

        feature_values = Parameter(_parse_feature_values_from_variable([defaultControlAllocation]), user=False)

        features = None
        num_estimates = 1
        # search_space = None
        control_allocation_search_space = None

    paramClassDefaults = ControlMechanism.paramClassDefaults.copy()
    paramClassDefaults.update({PARAMETER_STATES: NotImplemented}) # This suppresses parameterStates


    @tc.typecheck
    def __init__(self,
                 agent_rep=None,
                 features: tc.optional(tc.any(Iterable, Mechanism, OutputState, InputState)) = None,
                 feature_function: tc.optional(tc.any(is_function_type)) = None,
                 objective_mechanism: tc.optional(tc.any(ObjectiveMechanism, list)) = None,
                 origin_objective_mechanism=False, terminal_objective_mechanism=False,
                 function: tc.optional(tc.any(is_function_type)) = None, num_estimates: int = 1,
                 search_function: tc.optional(tc.any(is_function_type)) = None,
                 search_termination_function: tc.optional(tc.any(is_function_type)) = None,
                 control_signals: tc.optional(tc.any(is_iterable, ParameterState, ControlSignal)) = None,
                 modulation: tc.optional(_is_modulation_param) = ModulationParam.MULTIPLICATIVE, params=None, name=None,
                 prefs: is_pref_set = None, **kwargs):
        '''Abstract class that implements OptimizationControlMechanism'''

        if kwargs:
                for i in kwargs.keys():
                    raise OptimizationControlMechanismError("Unrecognized arg in constructor for {}: {}".
                                                            format(self.__class__.__name__, repr(i)))
        self.agent_rep = agent_rep
        self.search_function = search_function
        self.search_termination_function = search_termination_function

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(input_states=features,
                                                  feature_function=feature_function,
                                                  origin_objective_mechanism=origin_objective_mechanism,
                                                  terminal_objective_mechanism=terminal_objective_mechanism,
                                                  num_estimates=num_estimates,
                                                  params=params)

        super().__init__(system=None,
                         # monitor_for_control=monitor_for_control,
                         objective_mechanism=objective_mechanism,
                         function=function,
                         control_signals=control_signals,
                         modulation=modulation,
                         params=params,
                         name=name,
                         prefs=prefs)

    def _validate_params(self, request_set, target_set=None, context=None):
        '''Insure that specification of ObjectiveMechanism has projections to it'''

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        from psyneulink.core.compositions.composition import Composition
        if self.agent_rep is None:
            raise OptimizationControlMechanismError("The {} arg of an {} must specify a {}".
                                                    format(repr(AGENT_REP), self.__class__.__name__,
                                                           Composition.__name__))

        elif not (isinstance(self.agent_rep, Composition)
                  or (isinstance(self.agent_rep, type) and issubclass(self.agent_rep, Composition))):
            raise OptimizationControlMechanismError("The {} arg of an {} must be either a {}".
                                                    format(repr(AGENT_REP),self.__class__.__name__,
                                                           Composition.__name__))

    def _instantiate_input_states(self, context=None):
        """Instantiate input_states for Projections from features and objective_mechanism.

        Inserts InputState specification for Projection from ObjectiveMechanism as first item in list of
        InputState specifications generated in _parse_feature_specs from the **features** and
        **feature_function** arguments of the ModelFreeOptimizationControlMechanism constructor.
        """

        # Specify *OUTCOME* InputState;  receives Projection from *OUTCOME* OutputState of objective_mechanism
        outcome_input_state = {NAME:OUTCOME, PARAMS:{INTERNAL_ONLY:True}}

        # If any features were specified (assigned to self.input_states in __init__):
        if self.input_states:
            self.input_states = self._parse_feature_specs(self.input_states, self.feature_function)
            # Insert primary InputState for outcome from ObjectiveMechanism;
            #     assumes this will be a single scalar value and must be named OUTCOME by convention of ControlSignal
            self.input_states.insert(0, outcome_input_state),
        else:
            self.input_states = [outcome_input_state]

        # Configure default_variable to comport with full set of input_states
        self.defaults.variable, _ = self._handle_arg_input_states(self.input_states)

        super()._instantiate_input_states(context=context)

        for i in range(1, len(self.input_states)):
            state = self.input_states[i]
            if len(state.path_afferents) > 1:
                raise OptimizationControlMechanismError("Invalid FeatureInputState on {}. {} should receive exactly one"
                                                        " projection, but it receives {} projections."
                                                        .format(self.name, state.name, len(state.path_afferents)))

        # KAM Removed the exception below 11/6/2018 because it was rejecting valid
        # monitored_output_state spec on ObjectiveMechanism

        # if (OBJECTIVE_MECHANISM in request_set and
        #         isinstance(request_set[OBJECTIVE_MECHANISM], ObjectiveMechanism)
        #         and not request_set[OBJECTIVE_MECHANISM].path_afferents):
        #     raise OptimizationControlMechanismError("{} specified for {} ({}) must be assigned one or more {}".
        #                                             format(ObjectiveMechanism.__name__, self.name,
        #                                                    request_set[OBJECTIVE_MECHANISM],
        #                                                    repr(MONITORED_OUTPUT_STATES)))

    def _instantiate_control_signal(self, control_signal, context=None):
        '''Implement ControlSignalCosts.DEFAULTS as default for cost_option of ControlSignals
        OptimizationControlMechanism requires use of at least one of the cost options
        '''
        control_signal = super()._instantiate_control_signal(control_signal, context)

        if control_signal.cost_options is None:
            control_signal.cost_options = ControlSignalCosts.DEFAULTS
            control_signal._instantiate_cost_attributes()
        return control_signal

    def _instantiate_attributes_after_function(self, context=None):
        '''Instantiate OptimizationControlMechanism's OptimizatonFunction attributes'''

        super()._instantiate_attributes_after_function(context=context)
        # Assign parameters to function (OptimizationFunction) that rely on OptimizationControlMechanism
        self.function.reinitialize({DEFAULT_VARIABLE: self.control_allocation,
                                           OBJECTIVE_FUNCTION: self.evaluation_function,
                                           # SEARCH_FUNCTION: self.search_function,
                                           # SEARCH_TERMINATION_FUNCTION: self.search_termination_function,
                                           SEARCH_SPACE: self.control_allocation_search_space
                                           })

        # test_local_search_space = self._get_control_allocation_grid_space

        # self.search_function = self.function.search_function
        # self.search_termination_function = self.function.search_termination_function
        # self.search_space = self.function.search_space

        if isinstance(self.agent_rep, type):
            self.agent_rep = self.agent_rep()

        from psyneulink.core.compositions.compositionfunctionapproximator import CompositionFunctionApproximator
        if (isinstance(self.agent_rep, CompositionFunctionApproximator)):
            self._initialize_composition_function_approximator()

    def _execute(self, variable=None, execution_id=None, runtime_params=None, context=None):
        '''Find control_allocation that optimizes result of `agent_rep.evaluate`  .'''

        if (self.parameters.context.get(execution_id).initialization_status == ContextFlags.INITIALIZING):
            return defaultControlAllocation

        # # FIX: THESE NEED TO BE FOR THE PREVIOUS TRIAL;  ARE THEY FOR FUNCTION_APPROXIMATOR?
        # # FIX: SHOULD get_feature_values BE A METHOD OF THE agent_rep OR THE OCM?
        # # Get feature_values based on agent_rep
        # self.feature_values = self.agent_rep.get_feature_values(context=self.context)

        self.parameters.feature_values.set(_parse_feature_values_from_variable(variable), execution_id)

        # Assign default control_allocation if it is not yet specified (presumably first trial)
        control_allocation = self.parameters.control_allocation.get(execution_id)
        if control_allocation is None:
            control_allocation = [c.defaults.variable for c in self.control_signals]
            self.parameters.control_allocation.set(control_allocation, execution_id=None, override=True)

        # KAM Commented out below 12/5/18 to see if it is indeed no longer needed now that control signals are stateful

        # Assign default net_outcome if it is not yet specified (presumably first trial)
        # FIX: ??CAN GET RID OF THIS ONCE CONTROL SIGNALS ARE STATEFUL (_last_intensity SHOULD BE SET OR NOT NEEDED)
        costs = [c.compute_costs(c.parameters.variable.get(execution_id), execution_id=execution_id) for c in
                 self.control_signals]
        try:
            net_outcome = variable[0] - self.combine_costs(costs)
        except AttributeError:
            net_outcome = [0]
        # FIX: END
        #
        # Give the agent_rep a chance to adapt based on last trial's feature_values and control_allocation
        try:
            self.agent_rep.adapt(_parse_feature_values_from_variable(variable), control_allocation, net_outcome, execution_id=execution_id)
        except AttributeError as e:
            # If error is due to absence of adapt method, OK; otherwise, raise exception
            if not 'has no attribute \'adapt\'' in e.args[0]:
                raise AttributeError(e.args[0])

        # Get control_allocation that optmizes net_outcome using OptimizationControlMechanism's function
        # IMPLEMENTATION NOTE: skip ControlMechanism._execute since it is a stub method that returns input_values
        optimal_control_allocation, optimal_net_outcome, saved_samples, saved_values = \
                                                super(ControlMechanism,self)._execute(variable=control_allocation,
                                                                                      execution_id=execution_id,
                                                                                      runtime_params=runtime_params,
                                                                                      context=context)

        optimal_control_allocation = np.array(optimal_control_allocation).reshape((len(self.defaults.value), 1))

        # Give agent_rep a chance to clean up
        try:
            self.agent_rep._after_agent_rep_execution(context=context)
        except AttributeError as e:
            # If error is due to absence of adapt method, OK; otherwise, raise exception
            if not 'has no attribute \'_after_agent_rep_execution\'' in e.args[0]:
                raise AttributeError(e.args[0])
        # Return optimal control_allocation
        return optimal_control_allocation

    def _set_up_simulation(self, base_execution_id=None):
        sim_execution_id = self.get_next_sim_id(base_execution_id)

        try:
            self.parameters.simulation_ids.get(base_execution_id).append(sim_execution_id)
        except AttributeError:
            self.parameters.simulation_ids.set([sim_execution_id], base_execution_id)

        self.agent_rep._initialize_from_context(sim_execution_id, base_execution_id, override=False)

        return sim_execution_id

    def evaluation_function(self, control_allocation, execution_id=None):
        '''Compute `net_outcome <ControlMechanism.net_outcome>` for current set of `feature_values
        <OptimizationControlMechanism.feature_values>` and a specified `control_allocation
        <ControlMechanism.control_allocation>`.

        Assigned as the `objective_function <OptimizationFunction.objective_function>` for the
        OptimizationControlMechanism's `function <OptimizationControlMechanism.function>`.

        Calls `agent_rep <OptimizationControlMechanism.agent_rep>`\\'s `evalute` method.

        Returns a scalar that is the predicted `net_outcome <ControlMechanism.net_outcome>` (`net_outcome
        <ControlMechanism.net_outcome>`) for the current `feature_values <OptimizationControlMechanism.feature_values>`
        and specified `control_allocation <ControlMechanism.control_allocation>`.

        '''
        if self.agent_rep.runs_simulations:
            sim_execution_id = self._set_up_simulation(execution_id)

            result = self.agent_rep.evaluate(self.parameters.feature_values.get(execution_id),
                                             control_allocation,
                                             self.parameters.num_estimates.get(execution_id),
                                             base_execution_id=execution_id,
                                             execution_id=sim_execution_id,
                                             context=self.function.parameters.context.get(execution_id)
            )
        else:
            result = self.agent_rep.evaluate(self.parameters.feature_values.get(execution_id),
                                             control_allocation,
                                             self.parameters.num_estimates.get(execution_id),
                                             execution_id=execution_id,
                                             context=self.function.parameters.context.get(execution_id)
            )

        return result

    def apply_control_allocation(self, control_allocation, runtime_params, context, execution_id=None):
        '''Update `values <ControlSignal.value>` of `control_signals <ControlMechanism.control_signals>` based on
        specified `control_allocation <ControlMechanism.control_allocation>`.

        Called by `evaluate <Composition.evaluate>` method of `Composition` when it is assigned as `agent_rep
        <OptimizationControlMechanism.agent_rep>`.
        '''

        value = self.parameters.value.get(execution_id)
        if value is None:
            value = copy.deepcopy(self.defaults.value)

        for i in range(len(control_allocation)):
            value[i] = np.atleast_1d(control_allocation[i])

        self.parameters.value.set(value, execution_id)
        self._update_output_states(execution_id=execution_id, runtime_params=runtime_params,
                                   context=ContextFlags.COMPOSITION)

    # @property
    # def feature_values(self):
    #     if hasattr(self.agent_rep, 'model_based_optimizer') and self.agent_rep.model_based_optimizer is self:
    #         return self.agent_rep._get_predicted_input()
    #     else:
    #         return np.array(np.array(self.variable[1:]).tolist())

    # FIX: THE FOLLOWING SHOULD BE MERGED WITH HANDLING OF PredictionMechanisms FOR ORIG MODEL-BASED APPROACH;
    # FIX: SHOULD BE GENERALIZED AS SOMETHING LIKE update_feature_values

    tc.typecheck
    def add_features(self, features):
        '''Add InputStates and Projections to ModelFreeOptimizationControlMechanism for features used to
        predict `net_outcome <ControlMechanism.net_outcome>`

        **features** argument can use any of the forms of specification allowed for InputState(s),
        as well as a dictionary containing an entry with *SHADOW_EXTERNAL_INPUTS* as its key and a
        list of `ORIGIN` Mechanisms and/or their InputStates as its value.
        '''

        if features:
            features = self._parse_feature_specs(features=features,
                                                     context=ContextFlags.COMMAND_LINE)
        self.add_states(InputState, features)

    @tc.typecheck
    def _parse_feature_specs(self, features, feature_function, context=None):
        """Parse entries of features into InputState spec dictionaries

        For InputState specs in SHADOW_EXTERNAL_INPUTS ("shadowing" an Origin InputState):
            - Call _parse_shadow_input_spec

        For standard InputState specs:
            - Call _parse_state_spec
            - Set INTERNAL_ONLY entry of params dict of InputState spec dictionary to True

        Assign functions specified in **feature_function** to InputStates for all features

        Returns list of InputState specification dictionaries
        """

        parsed_features = []

        if not isinstance(features, list):
            features = [features]

        for spec in features:

            # e.g. {SHADOW_EXTERNAL_INPUTS: [A]}
            if isinstance(spec, dict):
                if SHADOW_EXTERNAL_INPUTS in spec:
                    #  composition looks for node.shadow_external_inputs and uses it to set external_origin_sources
                    self.shadow_external_inputs = spec[SHADOW_EXTERNAL_INPUTS]
                    spec = self._parse_shadow_inputs_spec(spec, feature_function)
                else:
                    raise OptimizationControlMechanismError("Incorrect specification ({}) "
                                                                     "in features argument of {}."
                                                                     .format(spec, self.name))
            # e.g. Mechanism, OutputState
            else:
                spec = _parse_state_spec(state_type=InputState, state_spec=spec)    # returns InputState dict
                spec[PARAMS][INTERNAL_ONLY] = True
                if feature_function:
                    spec[PARAMS][FUNCTION] = feature_function
                spec = [spec]   # so that extend works below

            parsed_features.extend(spec)

        for feature in parsed_features:
            if isinstance(feature, dict):
                feature['state_type'] = FeatureInputState
            else:
                if not isinstance(feature, FeatureInputState):
                    raise OptimizationControlMechanismError("{} has an invalid Feature: {}. Must be a FeatureInputState"
                                                            .format(self.name, feature))
        return parsed_features

    @tc.typecheck
    def _parse_shadow_inputs_spec(self, spec:dict, fct:tc.optional(Function)):
        ''' Return a list of InputState specifications for the inputs specified in value of dict

        For any other specification, specify an InputState with a Projection from the sender of any Projections
            that project to the specified item
        If FUNCTION entry, assign as Function for all InputStates specified in SHADOW_EXTERNAL_INPUTS
        '''

        input_state_specs = []

        shadow_spec = spec[SHADOW_EXTERNAL_INPUTS]

        if not isinstance(shadow_spec, list):
            shadow_spec = [shadow_spec]
        for item in shadow_spec:
            if isinstance(item, Mechanism):
                # Shadow all of the InputStates for the Mechanism
                input_states = item.input_states
            if isinstance(item, InputState):
                # Place in a list for consistency of handling below
                input_states = [item]
            # Shadow all of the Projections to each specified InputState
            input_state_specs.extend([
                {
                    #NAME:i.name + ' of ' + i.owner.name,
                    VARIABLE: i.variable}
                for i in input_states
            ])
        if fct:
            for i in input_state_specs:
                i.update({FUNCTION:fct})

        return input_state_specs

    @property
    def control_allocation_search_space(self):
        '''Return list of SampleIterators for allocation_samples of control_signals'''
        return [c.allocation_samples for c in self.control_signals]

    # ******************************************************************************************************************
    # FIX:  THE FOLLOWING IS SPECIFIC TO CompositionFunctionApproximator AS agent_rep
    # ******************************************************************************************************************

    def _initialize_composition_function_approximator(self):
        '''Initialize CompositionFunctionApproximator'''

        # CompositionFunctionApproximator needs to have access to control_signals to:
        # - to construct control_allocation_search_space from their allocation_samples attributes
        # - compute their values and costs for samples of control_allocations from control_allocation_search_space
        self.agent_rep.initialize(features_array=np.array(self.defaults.variable[1:]),
                                  control_signals = self.control_signals)

    @property
    def _dependent_components(self):
        from psyneulink.core.compositions.compositionfunctionapproximator import CompositionFunctionApproximator

        return list(itertools.chain(
            super()._dependent_components,
            [self.objective_mechanism],
            [self.agent_rep] if isinstance(self.agent_rep, CompositionFunctionApproximator) else [],
            [self.search_function] if isinstance(self.search_function, Function_Base) else [],
            [self.search_termination_function] if isinstance(self.search_termination_function, Function_Base) else [],
        ))
