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
an `control_allocation <ControlMechanism.control_allocation>` that maximizes the `Expected Value of Control (EVC)
<OptimizationControlMechanism_EVC>` for a given `state <OptimizationControlMechanism_State>`. The `OptimizationFunction`
uses the OptimizationControlMechanism's `evaluate_function` <OptimizationControlMechanism.evalutate_function>` to
evaluate the `EVC <OptimizationControlMechanism_EVC>` for samples of `control_allocation
<ControlMechanism.control_allocation>`, and then implements the one that yields the greatest EVC for the next
execution of the `Composition` (or part of one) that the OptimizationControlMechanism controls.

.. _OptimizationControlMechanism_EVC:

**Expected Value of Control**

All OptimizationControlMechanisms compute the `Expected Value of Control (EVC)
<https://www.ncbi.nlm.nih.gov/pubmed/23889930>`_ for a control_allocation <ControlMechanism.control_allocation>`
--  a cost-benefit analysis that weighs the `costs <ControlMechanism.costs>` of the `control_signals
<ControlMechanism.control_signals>` for the `control_allocation <ControlMechanism.control_allocation>` against the
`outcome <ControlMechanism.outcome>` expected to result from it.  The costs are computed based on the
`cost_options <ControlSignal.cost_options>` specified for each of the OptimizationControlMechanism's `control_signals
<ControlMechanism.control_signals>` and its `combine_costs <ControlMechanism.combine_costs>` function.  The EVC is
determined by its `compute_net_outcome <OptimizationControlMechanism.compute_net_outcome>` function (assigned to
its `net_outcome <ControlMechanism.net_outcome>` attribute), which is computed for a given `state
<OptimizationControlMechanism_State>` by the OptimizationControlMechanism's `evaluate_function`
<OptimizationControlMechanism.evalutate_function>`.

The table `below <OptimizationControlMechanism_Examples>` lists different
parameterizations of OptimizationControlMechanism that implement various models of EVC Optimization.

.. _OptimizationControlMechanism_Agent_Representation_Types:

**Agent Representation and Types of Optimization**

The defining characteristic of an OptimizationControlMechanism is its `agent representation
<OptimizationControlMechanism_Agent_Rep>`, that is used to determine the `EVC <OptimizationControlMechanism_EVC>` for a
given `state <OptimizationControlMechanism_State>` and find the `control_allocation
<ControlMechanism.control_allocation>` that maximizes this.  The `agent_rep <OptimizationControlMechanism.agent_rep>`
can be either the `Composition` to which the OptimizationControlMechanism belongs (and controls) or another one that
is used to estimate the EVC for that Composition.  This distinction corresponds closely to the
distnction between *model-based* and *model-free* optimization in the `machine learning <HTML REF>`_ and `cognitive
neuroscience <HTM REF>`_ literatures, as described below.

.. _OptimizationControlMechanism_Model_Free_Model_Based:

.. _OptimizationControlMechanism_Model_Free:

*Model-Free Optimization*

This is achieved by assigning as the `agent_rep  <OptimizationControlMechanism.agent_rep>` a Composition other than the
one to which the OptimizationControlMechanism belongs (and for which it is the `controller <Composition.controller>`).
In each `trial`, the `agent_rep <OptimizationControlMechanism.agent_rep>` is given the chance to adapt, by adjusting its
parameters in order to improve its prediction of the EVC for the Composition to which the OptimizationControlMechanism
belongs (based on the `state <OptimizationControlMechanism_State>` and `net_outcome <ControlMechanism.net_outcome>` of
the prior trial).  The agent_rep is then used to predict the `net_outcome <ControlMechanism.net_outcome>` of
processing on the upcoming trial, based on the current or (expected) `features <OptimizationControlMechanism.features>`
for that trial, in order to find the <ControlMechanism.control_allocation>` that yields the greatest `EVC
<OptimizationControlMechanism_EVC>` for that trial.

.. _OptimizationControlMechanism_Model_Based:

*Model-Based Optimization*

This is achieved by assigning as the `agent_rep  <OptimizationControlMechanism.agent_rep>` the Composition to which
the OptimizationControlMechanism belongs (and for which it is the `controller <Composition.controller>`). On each
`trial`, that Composition itself is used to simulate the `outcome <ControlMechanism.outcome>` of processing on the
upcoming trial, based on the current or (expected) values of the `features <OptimizationControlMechanism.features>`
for that trial, in order to find the <ControlMechanism.control_allocation>` that yields the greatest `EVC
<OptimizationControlMechanism_EVC>` for that trial.

.. _OptimizationControlMechanism_Creation:

Creating an OptimizationControlMechanism
----------------------------------------

An OptimizationControlMechanism is created in the same was as any `ControlMechanism <ControlMechanism>`.
The following arguments of its constructor are specific to the OptimizationControlMechanism:

  * **features** -- takes the place of the standard **input_states** argument in the constructor for a
    Mechanism`, and specifies the values used by the OptimizationControlMechanism, together with a
    `control_allocation <ControlMechanism.control_allocation>`, to calculate an `EVC
    <OptimizationControlMechanism_EVC>`.  Features can be specified using any of the following, singly or combined in a
    list:

        * {*SHADOW_EXTERNAL_INPUTS*: <`ORIGIN` Mechanism, InputState for one, or list with either or both>} --
          InputStates of the same shapes as those listed are created on the ModelFreeOptimizationControlMechanism,
          and are connected to the corresponding input_CIM OutputStates by projections. The external input values
          that are passed through the input_CIM are used as the `feature_predictors
          <ModelFreeOptimizationControlMechanism_Feature>`. If a Mechanism is included in the list, it refers to all
          of its InputStates.
        |
        * *InputState specification* -- this can be any form of `InputState specification <InputState_Specification>`
          that resolves to an OutputState from which the InputState receives a Projection;  the `value
          <OutputState.value>` of that OutputState is used as the feature. Each of these InputStates is marked as
          `internal_only <InputStates.internal_only>` = `True`.

    Features can also be added to an existing OptimizationControlMechanism using its `add_features` method.  If the
    **features** argument is not specified, then the `input <Composition.input>` to the `Composition` on the last
    trial of its execution is used to calculate the EVC <OptimizationControlMechanism_EVC>` for the upcoming trial.

  * **feature_function** -- specifies `function <InputState>` of the InputState created for each item listed in
    **feature_predictors**.  By default, this is the identity function, that assigns the current value of the feature
    to the OptimizationControlMechanism's `feature_values <OptimizationControlMechanism.feature_values>` attribute.
    However, other functions can be assigned, for example to maintain a record of past values, or integrate them over
    trials.

  * **agent_rep** -- specifies the `Composition` used by the OptimizationControlMechanism's `evaluation_function
    <OptimizationControlMechanism.evaluation_function>` to calculate the EVC <OptimizationControlMechanism_EVC>`
    (see `below <OptimizationControlMechanism_Agent_Rep>` for additional details). If it is not specified, the
    `Composition` to which the OptimizationControlMechanism belongs is assigned, and the OptimizationControlMechanism
    is assigned as that Composition's `controller <Composition.controller>`, implementing `model-based
    <OptimizationControlMechanism_Model_Based>` optimization.  If it is another Composition, it must conform to the
    specifications for an `agent_rep <OptimizationControlMechanism.agent_rep>` as described `below`
    <OptimizationControlMechanism_Agent_Rep>`.

.. _OptimizationControlMechanism_Structure:

Structure
---------

In addition to the standard Components associated with a `ControlMechanism`, including a `Projection <Projection>`
to its *OUTCOME* InputState from its `objective_mechanism <ControlMechanism.objective_mechanism>`, and a
`function <OptimizationControlMechanism.function>` used to carry out the optimization process, it has several
other constiuents, as described below.

.. _OptimizationControlMechanism_ObjectiveMechanism:

ObjectiveMechanism
^^^^^^^^^^^^^^^^^^

Like any `ControlMechanism`, an OptimizationControlMechanism has an associated `objective_mechanism
<ControlMechanism.objective_mechanism>` that is used to evaluate the outcome of processing for a given trial and pass
the result to the OptimizationControlMechanism, which it places in its `outcome <OptimizationControlMechanism.outcome>`
attribute.  This is used by its `compute_net_outcome <ControlMechanism.compute_net_outcome>` function, together with
the `costs <ControlMechanism.costs>` of its `control_signals <ControlMechanism.control_signals>`, to compute the
`net_outcome <ControlMechanism.net_outcome>` of processing for a given `state <OptimizationControlMechanism_State>`,
and by the `evaluation` method of the OptimizationControlMechanism's `agent_rep
<OptimizationControlMechanism.agent_rep>` to carry out the `EVC <OptimizationControlMechanism_EVC>` calculation.

.. note::
    The `objective_mechanism is distinct from, and should not be confused with the `objective_function
    <OptimizationFunction.objective_function>` parameter of the OptimizationControlMechanism's `function
    <OptimizationControlMechanism.function>`.  The `objective_mechanism
    <OptimizationControlMechanism.objective_mechanism>` evaluates the `outcome <ControlMechanism.outcome>` of processing
    without taking into account the `costs <ControlMechanism.costs>` of the OptimizationControlMechanism's
    `control_signals <ControlMechanism.control_signals>`.  In contrast, its `evaluate_function
    <OptimizationControlMechanism.evaluate_function>`, which is assigned as the
    `objective_function` parameter of its `function <OptimizationControlMechanism.function>`, takes the `costs
    <ControlMechanism.costs>` of the OptimizationControlMechanism's `control_signals <ControlMechanism.control_signals>`
    into account when calculating its `net_outcome` and corresponding `EVC <OptimizationControlMechanism_EVC>`.

.. _OptimizationControlMechanism_Features:

*Features*
~~~~~~~~~~

In addition to its `primary InputState <InputState_Primary>` (which receives a projection from the *OUTCOME*
OutpuState of the `objective_mechanism <OptimizationControlMechanism.objective_mechanism>`,
an OptimizationControlMechanism also has an `InputState` for each of its `features
<OptimizationControlMechanism.fetures>`.  By default, these are the current `input <Composition.input>` for the
Composition to which the OptimizationControlMechanism belongs.  However, different values can be specified, as
can a `feature_function <OptimizationControlMechanism.feature_function>`that transforms these.  For
OptimizationControlMechanisms that implement `model-free <OptimizationControlMechanism_Model_Free>` optimization,
its `features <OptimizationControlMechanism.features>` are used by its `evaluate_function
<OptimizationControlMechanism.evaluate_function>` to predict the `EVC <OptimizationControlMechanism_EVC>` for a given
`control_allocation <ControlMechanism.control_allocation>`.  For OptimizationControlMechanisms that implement
`model-based <OptimizationControlMechanism_Model_Based>` optimization, the `features
<OptimizationControlMechanism.features>` are used as the Composition's `input <Composition.input>` when it is
executed to evaluate the `EVC <OptimizationControlMechanism_EVC>` for a given
`control_allocation<ControlMechanism.control_allocation>`.

Features can be of two types:

* *Input Feature* -- this is a value received as input by an `ORIGIN` Mechanism in the `Composition`.
    These are specified in the **features** argument of the OptimizationControlMechanism's constructor (see
    `OptimizationControlMechanism_Creation`), in a dictionary containing a *SHADOW_EXTERNAL_INPUTS* entry,
    the value of which is one or more `ORIGIN` Mechanisms and/or their `InputStates
    <InputState>` to be shadowed.  For each, a `Projection` is automatically created that parallels ("shadows") the
    Projection from the Composition's `InputCIM` to the `ORIGIN` Mechanism, projecting from the same `OutputState` of
    the InputCIM to the InputState of the ModelFreeOptimizationControlMechanism assigned to that feature_predictor.

* *Output Feature* -- this is the `value <OutputState.value>` of an `OutputState` of some other `Mechanism` in the
    Composition.  These too are specified in the **feature_predictors** argument of the OptimizationControlMechanism's
    constructor (see `OptimizationControlMechanism_Creation`), and each is assigned a `Projection` from the specified
    OutputState(s) to the InputState of the OptimizationControlMechanism for that feature.

The current `value <InputState.value>` of the InputStates for the features are listed in the `feature_values
<OptimizationControlMechanism.feature_values>` attribute.

.. _OptimizationControlMechanism_State:

*State*

The state of the Composition (or part of one) controlled by an OptimizationControlMechanism is defined by a combination
of `feature_values <OptimizationControlMechanism.feature_values>` (see `above <OptimizationControlMechanism_Features>`)
and a `control_allocation <ControlMechanism.control_allocation>`.

.. _OptimizationControlMechanism_Agent_Rep:

*Agent Representation*

The defining feature of an OptimizationControlMechanism is its agent representation, specified in the **agent_rep**
argument of its constructor and assigned to its `agent_rep <OptimizationControlMechanism.agent_rep>` attribute.
This designates a representation of the `Composition` (or parts of one) that the OptimizationControlMechanism controls,
that is used to evaluate sample `control_allocations <ControlMechanism.control_allocation>` in order to find the one
that optimizes the `EVC <OptimizationControlMechanism_EVC>`. The `agent_rep  <OptimizationControlMechanism.agent_rep>`
is always itself a `Composition`, that can be either the same one that the OptimizationControlMechanism controls or
another one that is used to estimate the EVC for that Composition (see `above
<OptimizationControlMechanism_Agent_Representation_Types>`).  If it is another Composition, it must meet the following
requirem,ents:

    * Its `evaluate <Composition.evaluate>` method must accept as its first three arguments, in order,
      values that correspond in shape to  the `feature_values <OptimizationControlMechanism.feature_values>`,
      `control_allocation <ControlMechanism.control_allocation>` and `num_estimates
      <OptimizationControlMechanism.num_estimates>` attributes of the OptimizationControlMechanism, respectively.

    * If it has an `adapt <Composition.adapt>` method, that must accept as its first three arguments, in order,
      values that corresopnd to the shape of the `feature_values <OptimizationControlMechanism.feature_values>`,
      `control_allocation <ControlMechanism.control_allocation>` and `net_outcome
      <OptimizationControlMechanism.net_outcome>` attributes of the OptimizationControlMechanism. respectively.

.. _OptimizationControlMechanism_Function:

*Function*
^^^^^^^^^^

The `function <OptimizationControlMechanism.function>` of an OptimizationControlMechanism is used to find the
`control_allocation <ControlMechanism.control_allocation>` that yields the greatest `EVC
<OptimizationControlMechanism_EVC>` for the current (or expected) `features <OptimizationControlMechanism_State>`.
It is generally an `OptimizationFunction`, which in turn has `objective_function
<OptimizationFunction.objective_function>`, `search_function <OptimizationFunction.search_function>`
and `search_termination_function <OptimizationFunction.search_termination_function>` methods, as well as a `search_space
<OptimizationFunction.search_space>` attribute.  The OptimizationControlMechanism's `evaluate_function
<OptimizationControlMechanism.evaluate_function>` is automatically assigned as the
OptimizationFunction's `objective_function <OptimizationFunction.objective_function>`, and is used to
evaluate each `control_allocation <ControlMechanism.control_allocation>` sampled from the `search_space
<OptimizationFunction.search_space>` by the `search_function `search_function <OptimizationFunction.search_function>`
until the `search_termination_function <OptimizationFunction.search_termination_function>` returns `True`.
A custom function can be assigned as the `function <OptimizationControlMechanism.function>` of an
OptimizationControlMechanism, however it must meet the following requirements:

.. _OptimizationControlMechanism_Custom_Funtion:

    - it must accept as its first argument and return as its result an array with the same shape as the
      OptimizationControlMechanism's `control_allocation <ControlMechanism.control_allocation>`.

    - it must implement a `reinitialize` method that accepts **objective_function** as a keyword argument and
      implements an attribute with the same name.

    COMMENT:
    - it must implement a `reinitialize` method that accepts as keyword arguments **objective_function**,
      **search_function**, **search_termination_function**, and **search_space**, and implement attributes
      with corresponding names.
    COMMENT

If **function** argument is not specified, the `GridSearch` `OptimiziationFunction` is assigned as the default,
which evaluates the `EVC <OptimizationControlMechanism_EVC>` using the OptimizationControlMechanism's
`control_allocation_search_space <OptimizationControlMechanism.control_allocation_search_spaces>` as its
`search_space <OptimizationFunction.search_space>`.

COMMENT:
.. _OptimizationControlMechanism_Search_Functions:

*Search Function, Search Space and Search Termination Function*

The OptimizationControlMechanism may also implement `search_function <OptimizationControlMechanism.search_function>`
and `search_termination_function <OptimizationControlMechanism.search_termination_function>` methods, as well as a
`control_allocation_search_space <OptimizationControlMechanism.control_allocation_search_space>` attribute, that will
be passed as parameters to the `OptimizationFunction` when it is constructed.  These can be specified in the
constructor for an `OptimizationFunction` assigned as the **function** argument of the
OptimizationControlMechanism's constructor, as long as they are compatible with the requirements of
the OptimizationFunction and OptimizationControlMechanism.  If they are not specified, then defaults specified
either by the OptimizationControlMechanism or the OptimizationFunction are used.
COMMENT

.. _OptimizationControlMechanism_Execution:

Execution
---------

When an OptimizationControlMechanism is executed, it carries out the following steps:

  * It calls the `adapt` method of its `agent_rep <OptimizationControlMechanism.agent_rep>` to give that a chance to
    modify its parameters in order to better predict the `EVC <OptimizationControlMechanism_EVC>` for a given `state
    <OptimizationControlMechanism_State>`, based the state and `net_outcome <ControlMechanism.net_outcome>` of the
    previous trial.

  * It then calls its `function <OptimizationControlMechanism.function>` to find the `control_allocation
    <ControlMechanism.control_allocation>` that yields the greatest `EVC <OptimizationControlMechanism_EVC>`.  The
    way in which it searches for the best `control_allocation <ControlMechanism.control_allocation>` is determined by
    the type of `OptimzationFunction` assigned to `function <OptimizationControlMechanism.function>`, whereas the way
    that it evaluates each one is determined by the OptimizationControlMechanism's `evaluate_function`
    <OptimizationControlMechanism.evalutate_function>`.  More specifically:

    * The `function <OptimizationControlMechanism.function>` selects a sample `control_allocation
      <ControlMechanism.control_allocation>` (using its `search_function <OptimizationFunction.search_function>`
      to select one from its `search_space <OptimizationFunction.search_space>`), and evaluates the EVC for that
      `control_allocation <ControlMechanism.control_allocation>` using the OptimizationControlMechanism's
      `evaluate_function` <OptimizationControlMechanism.evalutate_function>` and the current `feature_values
      <OptimizationControlMechanism.feature_values>`.

    * It continues to evaluate the `EVC <OptimizationControlMechanism_EVC>` for `control_allocation
      <ControlMechanism.control_allocation>` samples until its `search_termination_function
      <OptimizationFunction.search_termination_function>` returns `True`.

    COMMENT
    * The `function <OptimizationControlMechanism.function>` selects a sample `control_allocation
      <ControlMechanism.control_allocation>` (usually using `search_function
      <OptimizationControlMechanism.search_function>` to select one from `control_allocation_search_space
      <OptimizationControlMechanism.control_allocation_search_space>`), and evaluates the EVC for that
      `control_allocation <ControlMechanism.control_allocation>` using the OptimizationControlMechanism's
      `evaluate_function` <OptimizationControlMechanism.evalutate_function>` and the current `feature_values
      <OptimizationControlMechanism.feature_values>`.

    * It continues to evaluate the `EVC <OptimizationControlMechanism_EVC>` for `control_allocation
      <ControlMechanism.control_allocation>` samples until the `search_termination_function
      <OptimizationControlMechanism.search_termination_function>` returns `True`.
    COMMENT

    * Finally, it implements the `control_allocation <ControlMechanism.control_allocation>` that yielded the greates
      `EVC <OptimizationControlMechanism_EVC>`.  This is used by the OptimizationControlMechanism's `control_signals
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
from collections import Iterable

import typecheck as tc

import numpy as np

from psyneulink.core.components.functions.function import \
    ModulationParam, _is_modulation_param, is_function_type, OBJECTIVE_FUNCTION, \
    SEARCH_SPACE
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.components.mechanisms.adaptive.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import \
    ObjectiveMechanism
from psyneulink.core.components.states.inputstate import InputState
from psyneulink.core.components.states.outputstate import OutputState

from psyneulink.core.components.states.parameterstate import ParameterState
from psyneulink.core.components.states.modulatorysignals.controlsignal import ControlSignalCosts, ControlSignal
from psyneulink.core.components.states.state import _parse_state_spec
from psyneulink.core.components.functions.function import Function
from psyneulink.core.globals.keywords import DEFAULT_VARIABLE, INTERNAL_ONLY, NAME, \
    OPTIMIZATION_CONTROL_MECHANISM, OUTCOME, PARAMS, PARAMETER_STATES, FUNCTION, VARIABLE
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import is_iterable
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.defaults import defaultControlAllocation

__all__ = [
    'OptimizationControlMechanism', 'OptimizationControlMechanismError',
    'AGENT_REP', 'FEATURES', 'SHADOW_EXTERNAL_INPUTS'
]

AGENT_REP = 'agent_rep'
FEATURES = 'features'
SHADOW_EXTERNAL_INPUTS = 'SHADOW_EXTERNAL_INPUTS'


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

    features : Mechanism, OutputState, Projection, dict, or list containing any of these
        specifies Components, the values of which are assigned to `feature_values
        <OptimizationControlMechanism.feature_values>` and used to estimate `EVC <OptimizationControlMechanism_EVC>`.
        Any `InputState specification <InputState_Specification>` can be used that resolves to an `OutputState` that
        projects to the InputState. In addition, a dictionary with a *SHADOW_EXTERNAL_INPUTS* entry can be used to
        shadow inputs to the Composition's `ORIGIN` Mechanism(s) (see  OptimizationControlMechanism_Creation` for
        details).

    feature_function : Function or function : default None
        specifies the `function <InputState.function>` for the `InputState` assigned to each `feature_predictor
        <OptimizationControlMechanism_Feature_Predictors>`.

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
        integers (the first representing the `EVC <OptimizationControlMechanism_EVC>` value for the current 
        `control_allocation <ControlMechanism.control_allocation>`, and the second the current iteration of the
        `optimization process <OptimizationFunction_Process>`;  it must return `True` or `False`.
        
    search_space : list or ndarray
        specifies the `search_space <OptimizationFunction.search_space>` parameter for `function 
        <OptimizationControlMechanism.function>`, unless that is specified in a constructor for `function 
        <OptimizationControlMechanism.function>`.  Each item must have the same shape as `control_allocation
        <ControlMechanism.control_allocation>`.
        
    function : OptimizationFunction, function or method
        specifies the function used to optimize the `control_allocation <ControlMechanism.control_allocation>`;
        must take as its sole argument an array with the same shape as `control_allocation
        <ControlMechanism.control_allocation>`, and return a similar array (see `Primary Function
        <OptimizationControlMechanism>` for additional details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for the
        Mechanism, its `learning_function <OptimizationControlMechanism.learning_function>`, and/or a custom function
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
        <OptimizationControlMechanism.features>` (each of which is a 1d array).

    function : OptimizationFunction, function or method
        takes current `control_allocation <ControlMechanism.control_allocation>` (as initializer),
        uses its `search_function <OptimizationFunction.search_function>` to select samples of `control_allocation
        <ControlMechanism.control_allocation>` from its `search_space <OptimizationControlMechanism.search_space>`,
        evaluates these using its `evaluation_function <OptimizationControlMechanism.evaluation_function>`, and returns
        the one that yields the greatest `EVC <OptimizationControlMechanism_EVC>`  (see `Primary Function
        <OptimizationControlMechanism_Function>` for additional details).

    evaluation_function : function or method
        returns `EVC <OptimizationControlMechanism_EVC>` for a given state (i.e., combination of `feature_values
        <OptimizationControlMechanism.feature_values>` and `control_allocation <ControlMechanism.control_allocation>`.
        It is assigned as the `objective_function <OptimizationFunction.objective_function>` parameter of `function
        <OptimizationControlMechanism.function>`, and calls the `evaluate` method of the OptimizationControlMechanism's
        `agent_rep <OptimizationControlMechanism.agent_rep>` with a `control_allocation
        <ControlMechanism.control_allocation>`, the OptimizationControlMechanism's `num_estimates
        <OptimizationControlMechanism.num_estimates>` attribute, and the current value of its `feature
        <OptimizationControlMechanism.features>`.

    COMMENT:
    evaluation_function : function or method
        specifies the function used to evaluate the `EVC <OptimizationControlMechanism_EVC>` for a given
        `control_allocation <ControlMechanism.control_allocation>`. It is assigned as the `objective_function
        <OptimizationFunction.objective_function>` parameter of `function  <OptimizationControlMechanism.function>`,
        unless that is specified in the constructor for an  OptimizationFunction assigned to the **function**
        argument of the OptimizationControlMechanism's constructor.  Often it is assigned directy to the
        OptimizationControlMechanism's `compute_EVC <OptimizationControlMechanism.compute_EVC>` method;  in some
        cases it may implement additional operations, but should always call `compute_EVC
        <OptimizationControlMechanism.compute_EVC>`. A custom function can be assigned, but it must take as its
        first argument an array with the same shape as the OptimizationControlMechanism's `control_allocation
        <ControlMechanism.control_allocation>`, and return the following four values: an array containing the
        `control_allocation <ControlMechanism.control_allocation>` that generated the optimal `EVC
        <OptimizationControlMechanism_EVC>`; an array containing that EVC value;  a list containing each
        `control_allocation <ControlMechanism.control_allocation>` sampled if `function
        <OptimizationControlMechanism.function>` has a `save_samples <OptimizationFunction.save_samples>` attribute
        and it is `True`, otherwise it should return an empty list; and a list containing the EVC values for each
        `control_allocation <ControlMechanism.control_allocation>` sampled if the function has a `save_values
        <OptimizationFunction.save_values>` attribute and it is `True`, otherwise it should return an empty list.
    COMMENT

    COMMENT:
    search_function : function or method
        `search_function <OptimizationFunction.search_function>` assigned to `function 
        <OptimizationControlMechanism.function>`; used to select samples of `control_allocation
        <ControlMechanism.control_allocation>` to evaluate by `evaluate_function
        <OptimizationControlMechanism.evaluation_function>`.

    search_termination_function : function or method
        `search_termination_function <OptimizationFunction.search_termination_function>` assigned to
        `function <OptimizationControlMechanism.function>`;  determines when to terminate the 
        `optimization process <OptimizationFunction_Process>`.
    COMMENT

    control_allocation_search_space : list or ndarray
        `search_space <OptimizationFunction.search_space>` assigned by default to `function
        <OptimizationControlMechanism.function, that determines the samples of
        `control_allocation <ControlMechanism.control_allocation>` evaluated by the `evaluation_function
        <OptimizationControlMechanism.evaluation_function>`.  This is a proprety that, unless overridden,
        returns a list containing every possible `control_allocation <ControlMechanism.control_allocation>`, as
        determined by the `Cartesian product <HTML REF>`_ of the `allocation_samples
        <ControlSignal.allocation_samples>` specified for each of the `OptimizationControlMechanism's
        `control_signals <ControlMechanism.control_signals>`).

    saved_samples : list
        contains all values of `control_allocation <ControlMechanism.control_allocation>` sampled by `function
        <OptimizationControlMechanism.function>` if its `save_samples <OptimizationFunction.save_samples>` parameter
        is `True`;  otherwise list is empty.

    saved_values : list
        contains values of EVC associated with all samples of `control_allocation <ControlMechanism.control_allocation>`
         evaluated by by `function <OptimizationControlMechanism.function>` if its `save_values 
         <OptimizationFunction.save_samples>` parameter is `True`;  otherwise list is empty.

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

    # FIX: ADD OTHER Params() HERE??
    class Params(ControlMechanism.Params):
        function = None

    paramClassDefaults = ControlMechanism.paramClassDefaults.copy()
    paramClassDefaults.update({PARAMETER_STATES: NotImplemented}) # This suppresses parameterStates


    @tc.typecheck
    def __init__(self,
                 agent_rep=None,
                 features:tc.optional(tc.any(Iterable, Mechanism, OutputState, InputState))=None,
                 feature_function:tc.optional(tc.any(is_function_type))=None,
                 # monitor_for_control:tc.optional(tc.any(is_iterable, Mechanism, OutputState))=None,
                 objective_mechanism:tc.optional(tc.any(ObjectiveMechanism, list))=None,
                 origin_objective_mechanism=False,
                 terminal_objective_mechanism=False,
                 function:tc.optional(tc.any(is_function_type))=None,
                 search_function:tc.optional(tc.any(is_function_type))=None,
                 search_termination_function:tc.optional(tc.any(is_function_type))=None,
                 search_space:tc.optional(tc.any(list, np.ndarray))=None,
                 control_signals:tc.optional(tc.any(is_iterable, ParameterState, ControlSignal))=None,
                 modulation:tc.optional(_is_modulation_param)=ModulationParam.MULTIPLICATIVE,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs):
        '''Abstract class that implements OptimizationControlMechanism'''

        if kwargs:
                for i in kwargs.keys():
                    raise OptimizationControlMechanismError("Unrecognized arg in constructor for {}: {}".
                                                            format(self.__class__.__name__, repr(i)))
        self.agent_rep = agent_rep
        self.search_function = search_function
        self.search_termination_function = search_termination_function
        self.search_space = search_space

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(input_states=features,
                                                  feature_function=feature_function,
                                                  origin_objective_mechanism=origin_objective_mechanism,
                                                  terminal_objective_mechanism=terminal_objective_mechanism,
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
        self.instance_defaults.variable, ignore = self._handle_arg_input_states(self.input_states)

        super()._instantiate_input_states(context=context)


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
        self.function_object.reinitialize({DEFAULT_VARIABLE: self.control_allocation,
                                           OBJECTIVE_FUNCTION: self.evaluation_function,
                                           # SEARCH_FUNCTION: self.search_function,
                                           # SEARCH_TERMINATION_FUNCTION: self.search_termination_function,
                                           SEARCH_SPACE: self.control_allocation_search_space
                                           })

        # self.search_function = self.function_object.search_function
        # self.search_termination_function = self.function_object.search_termination_function
        self.search_space = self.function_object.search_space

        from psyneulink.core.compositions.compositionfunctionapproximator import CompositionFunctionApproximator
        if (isinstance(self.agent_rep, CompositionFunctionApproximator)
                or (isinstance(self.agent_rep, type) and issubclass(self.agent_rep, CompositionFunctionApproximator))):
            self._instantiate_function_approximator_as_agent()

    def _get_control_allocation_search_space(self):

        control_signal_sample_lists = []
        for control_signal in self.control_signals:
            control_signal_sample_lists.append(control_signal.allocation_samples)

        # Construct control_allocation_search_space:  set of all permutations of ControlProjection allocations
        #                                     (one sample from the allocationSample of each ControlProjection)
        # Reference for implementation below:
        # http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
        control_allocation_search_space = \
            np.array(np.meshgrid(*control_signal_sample_lists)).T.reshape(-1,len(self.control_signals))

        # Insure that ControlSignal in each sample is in its own 1d array
        re_shape = (control_allocation_search_space.shape[0], control_allocation_search_space.shape[1], 1)
        return control_allocation_search_space.reshape(re_shape)

    def _execute(self, variable=None, runtime_params=None, context=None):
        '''Find control_allocation that optimizes result of `agent_rep.evaluate`  .'''

        if (self.context.initialization_status == ContextFlags.INITIALIZING):
            return defaultControlAllocation

        # # FIX: THESE NEED TO BE FOR THE PREVIOUS TRIAL;  ARE THEY FOR FUNCTION_APPROXIMATOR?
        # # FIX: SHOULD get_feature_values BE A METHOD OF THE agent_rep OR THE OCM?
        # # Get feature_values based on agent_rep
        # self.feature_values = self.agent_rep.get_feature_values(context=self.context)

        # Assign default control_allocation if it is not yet specified (presumably first trial)
        if self.control_allocation is None:
            self.value = [c.instance_defaults.variable for c in self.control_signals]

        # Assign default net_outcome if it is not yet specified (presumably first trial)
        # FIX: ??CAN GET RID OF THIS ONCE CONTROL SIGNALS ARE STATEFUL (_last_intensity SHOULD BE SET OR NOT NEEDED)
        try:
            net_outcome = self.net_outcome
        except AttributeError:
            net_outcome = [0]
        # FIX: END

        # Give the agent_rep a chance to adapt based on last trial's feature_values and control_allocation
        try:
            self.agent_rep.adapt(self.feature_values, self.control_allocation, net_outcome)
        except AttributeError as e:
            # If error is due to absence of adapt method, OK; otherwise, raise exception
            if not 'has no attribute \'adapt\'' in e.args[0]:
                raise AttributeError(e.args[0])

        # Get control_allocation that optmizes EVC using OptimizationControlMechanism's function
        # IMPLEMENTATION NOTE: skip ControlMechanism._execute since it is a stub method that returns input_values
        optimal_control_allocation, self.optimal_EVC, self.saved_samples, self.saved_values = \
                                        super(ControlMechanism, self)._execute(variable=self.control_allocation,
                                                                               runtime_params=runtime_params,
                                                                               context=context)
        # Give agent_rep a chance to clean up
        try:
            self.agent_rep._after_agent_rep_execution(context=context)
        except AttributeError as e:
            # If error is due to absence of adapt method, OK; otherwise, raise exception
            if not 'has no attribute \'_after_agent_rep_execution\'' in e.args[0]:
                raise AttributeError(e.args[0])

        # Return optimal control_allocation
        return optimal_control_allocation

    def evaluation_function(self, control_allocation):
        '''Compute metric for a given control_allocation.
        Assigned as the `objective_function <OptimizationFunction.objective_function>` parameter of the
        `ObjectiveFunction` assigned to the OptimizationControlMechanism's `function
        <OptimizationControlMechanism.function>`.

        Returns a scalar that is the predicted outcome of the `function_approximator
        <ModelFreeOptimizationControlMechanism.function_approximator>`.
        '''
        self.num_estimates = 1
        return self.agent_rep.evaluate(self.feature_values,
                                       control_allocation,
                                       self.num_estimates,
                                       context=self.function_object.context)

    def apply_control_allocation(self, control_allocation, runtime_params, context):
        '''Update ControlSignal values based on specified control_allocation'''
        for i in range(len(control_allocation)):
            if self.value is None:
                self.value = self.instance_defaults.value
            self.value[i] = np.atleast_1d(control_allocation[i])

        self._update_output_states(self.value, runtime_params=runtime_params, context=ContextFlags.COMPOSITION)

    @property
    def feature_values(self):
        if hasattr(self.agent_rep, 'model_based_optimizer') and self.agent_rep.model_based_optimizer is self:
            return self.agent_rep._get_predicted_input()
        else:
            return np.array(np.array(self.variable[1:]).tolist())

    # ******************************************************************************************************************
    # FIX:  THE FOLLOWING IS SPECIFIC TO MODEL-FREE (FUNCTION_APPROXIMATOR) IMPLEMENTATION
    # ******************************************************************************************************************

    def _instantiate_function_approximator_as_agent(self):
        '''Instantiate attributes for ModelFreeOptimizationControlMechanism's function_approximator'''

        if isinstance(self.agent_rep, type):
            self.agent_rep = self.agent_rep()
        # CompositionFunctionApproximator needs to have access to control_signals to:
        # - to construct control_allocation_search_space from their allocation_samples attributes
        # - compute their values and costs for samples of control_allocations from control_allocation_search_space
        self.agent_rep.initialize(features_array=np.array(self.instance_defaults.variable[1:]),
                                  control_signals = self.control_signals)

    # FIX: THIS SHOULD BE MERGED WITH HANDLING OF PredictionMechanisms FOR ORIG MODEL-BASED APPROACH;
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
        return self._get_control_allocation_search_space()




