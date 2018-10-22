# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *************************************************  LVOCControlMechanism ******************************************************

"""

Overview
--------

An LVOCControlMechanism is a `ControlMechanism <ControlMechanism>` that regulates it `ControlSignals <ControlSignal>` in
order to optimize the performance of the `Composition` to which it belongs.  It implements a form of the Learned Value
of Control model described in `Leider et al. <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi
.1006043&rev=2>`_, which learns to select the value for its `control_signals <LVOCControlMechanism.control_signals>`
(i.e., its `allocation_policy  <LVOCControlMechanism.allocation_policy>`) that maximzes its `EVC
<LVOCControlMechanism_EVC>` based on a set of `predictors <LVOCControlMechanism_Predictors>`.

.. _LVOCControlMechanism_EVC:

*Expected Value of Control (EVC)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **expected value of control (EVC)** is the outcome of executing the `composition`
to which the LVOCControlMechanism belongs under a given `allocation_policy <LVOCControlMechanism.allocation_policy>`,
as determined by its `objective_mechanism <LVOCControlMechanism.objective_mechanism>`, discounted by the `cost
<ControlSignal.cost> of the `control_signals <LVOCControlMechanism.control_signals>` under that `allocation_policy
<LVOCControlMechanism.allocation_policy>`.

The LVOCControlMechanism's `function <LVOCControlMechanism.function>` learns to predict the outcome of its
`objective_mechanism <LVOCControlMechanism.objective_mechanism>` from a weighted sum of its `predictors
<LVOCControlMechanism.predictors>`, `control_signals <LVOCControlMechanism.control_signals>`, interactions among
these, and the costs of the `control_signals <LVOCControlMechanism.control_signals>`.  This is referred to as the
"learned value of control," or LVOC.

.. _LVOCControlMechanism_Creation:

Creating an LVOCControlMechanism
------------------------

 An LVOCControlMechanism can be created in the same was as any `ControlMechanism`, with the exception that it cannot
 be assigned as the `controller <Composition.controller>` of a Composition.  The following arguments of its
 constructor are specific to the LVOCControlMechanism:

  * **predictors** -- this takes the place of the standard **input_states** argument in the constructor for a
    Mechanism`, and specifies the inputs that it learns to use to determine its `allocation_policy
    <LVOCControlMechanism.allocation_policy>` in each `trial` of execution.
    It can be specified using any of the following, singly or combined in a list:

        * {*SHADOW_EXTERNAL_INPUTS*: <`ORIGIN` Mechanism, InputState for one, or list with either or both>} --
          InputStates of the same shapes as those listed are created on the LVOC, and are connected to the
          corresponding input_CIM OutputStates by projections. The external input values that are passed through the
          input_CIM are used as the `predictors <LVOCControlMechanism.predictor>`. If a Mechanism is included in the
          list, it refers to all of its InputStates.

        COMMENT:
          the input of all items specified received from the `Composition` is used as predictors.
        COMMENT
        |
        * *InputState specification* -- this can be any form of `InputState specification <InputState_Specification>`
          that resolves to an OutputState from which the InputState receives a Projection;  the `value
          <OutputState.value>` of that OutputState is used as the `predictor <LVOCControlMechanism.predictor>`. Each of
          these InputStates is marked as internal_only.

    Predictors can also be added to an existing LVOCControlMechanism using its `add_predictors` method.

  * **predictor_function** specifies `function <InputState>` of the InputState created for each item listed in
    **predictors**.

.. _LVOCControlMechanism_Structure:

Structure
---------

.. _LVOCControlMechanism_Input:

*Input*
~~~~~~~

An LVOCControlMechanism has one `InputState` that receives a `Projection` from its `objective_mechanism
<LVOCControlMechanism.objective_mechanism>` (its primary InputState <InputState_Primary>`), and additional ones for
each of its predictors, as described below.

.. _LVOCControlMechanism_Predictors:

Predictors
^^^^^^^^^^

Predictors, together with the LVOCControlMechanism's `control_signals <LVOCControlMechanism.control_signals>`,
are used by its `function <LVOCControlMechanism.function>` to learn to predict the outcome of its
`objective_mechanism <LVOCControlMechanism.objective_mechanism>` and to determine its `allocation_policy
<LVOCControlMechanism.allocation_policy>`.

Predictors can be of two types:

* *Input Predictor* -- this is a value received as input by an `ORIGIN` Mechanism in the Composition.
    These are specified in the **predictors** argument of the LVOCControlMechanism's constructor (see
    `LVOCControlMechanism_Creation`), in a dictionary containing a *SHADOW_EXTERNAL_INPUTS* entry, the value of
    which is one or more `ORIGIN` Mechanisms and/or their InputStates to be shadowed.  For each, a Projection is
    automatically created that parallels ("shadows") the Projection from the Composition's `InputCIM` to the `ORIGIN`
    Mechanism, projecting from the same `OutputState` of the InputCIM to the the InputState of the
    LVOCControlMechanism assigned to that predictor.

* *Output Predictor* -- this is the `value <OutputState.value>` of an OutputState of some other Mechanism in the
    Composition.  These too are specified in the **predictors** argument of the LVOCControlMechanism's constructor
    (see `LVOCControlMechanism_Creation`), and each is assigned a Projection to the InputState of the
    LVOCControlMechanism for that predictor.

The current `values <InputState.value>` of the InputStates for the predictors are listed in the `predictor_values
<LVOCControlMechanism.predictor_values>` attribute.

.. _LVOCControlMechanism_ObjectiveMechanism:

ObjectiveMechanism
^^^^^^^^^^^^^^^^^^

Like any ControlMechanism, an LVOCControlMechanism receives its input from the *OUTCOME* `OutputState
<ObjectiveMechanism_Output>` of its `objective_mechanism <LVOCControlMechanism.objective_mechanism>`,
via a MappingProjection to its `primary InputState <InputState_Primary>`. By default, the ObjectiveMechanism's
function is a `LinearCombination` function with its `operation <LinearCombination.operation>` attribute assigned as
*PRODUCT*; this takes the product of the `value <OutputState.value>`\\s of the OutputStates that it monitors (listed
in its `monitored_output_states <ObjectiveMechanism.monitored_output_states>` attribute.  However, this can be
customized in a variety of ways:

    * by specifying a different `function <ObjectiveMechanism.function>` for the ObjectiveMechanism
      (see `Objective Mechanism Examples <ObjectiveMechanism_Weights_and_Exponents_Example>` for an example);
    ..
    * using a list to specify the OutputStates to be monitored  (and the `tuples format
      <InputState_Tuple_Specification>` to specify weights and/or exponents for them) in either the
      **monitor_for_control** or **objective_mechanism** arguments of the LVOCControlMechanism's constructor;
    ..
    * using the  **monitored_output_states** argument for an ObjectiveMechanism specified in the `objective_mechanism
      <LVOCControlMechanism.objective_mechanism>` argument of the LVOCControlMechanism's constructor;
    ..
    * specifying a different `ObjectiveMechanism` in the **objective_mechanism** argument of the LVOCControlMechanism's
      constructor.

    .. _LVOCControlMechanism_Objective_Mechanism_Function_Note:

    .. note::
       If a constructor for an `ObjectiveMechanism` is used for the **objective_mechanism** argument of the
       LVOCControlMechanism's constructor, then the default values of its attributes override any used by the
       LVOCControlMechanism for its `objective_mechanism <LVOCControlMechanism.objective_mechanism>`.  In particular,
       whereas an LVOCControlMechanism uses the same default `function <ObjectiveMechanism.function>` as an
       `ObjectiveMechanism` (`LinearCombination`), it uses *PRODUCT* rather than *SUM* as the default value of the
       `operation <LinearCombination.operation>` attribute of the function.  As a consequence, if the constructor for
       an ObjectiveMechanism is used to specify the LVOCControlMechanism's **objective_mechanism** argument,
       and the **operation** argument is not specified, *SUM* rather than *PRODUCT* will be used for the
       ObjectiveMechanism's `function <ObjectiveMechanism.function>`.  To ensure that *PRODUCT* is used, it must be
       specified explicitly in the **operation** argument of the constructor for the ObjectiveMechanism (see 1st
       example under `System_Control_Examples`).

The LVOCControlMechanism's `function <LVOCControlMechanism.function>` learns to predict the `value <OutputState.value>`
of the *OUTCOME* `OutputState` of the LVOCControlMechanism's `objective_mechanism
<LVOCControlMechanism.objective_mechanism>`, as described below.

.. _LVOCControlMechanism_Function:

*Function*
~~~~~~~~~~

The `function <LVOCControlMechanism.function>` of an LVOCControlMechanism learns how to weight its `predictors
<LVOCControlMechanism_Predictors>`, the `values <ControlSignal.value>` of its  `control_signals
<LVOCControlMechanism.control_signals>`, the interactions between these, and the `costs <ControlSignal.costs>` of the
`control_signals <LVOCControlMechanism.control_signals>`, to best predict the outcome of its `objective_mechanism
<LVOCControlMechanism.objective_mechanism>`.  Using those weights, and the current set of predictors, it then
searches for and returns the `allocation_policy <LVOCControlMechanism.allocation_policy>` that maximizes the `EVC
<LVOCControlMechanism_EVC>`.  By default, `function <LVOCControlMechanism.function>` is `BayesGLM`. However,
any function can be used that accepts a 2d array, the first item of which is an array of scalar values (the prediction
terms) and the second that is a scalar value (the outcome to be predicted), and returns an array with the same shape as
the LVOCControlMechanism's `allocation_policy <LVOCControlMechanism.allocation_policy>`.

.. note::
  The LVOCControlMechanism's `function <LVOCControlMechanism.function>` is provided the values of the `predictors
  <LVOCControlMechanmism.predictors>` and outcome of its `objective_mechanism
  <LVOCControlMechanmism.objective_mechanism>` from the *previous* trial to update the `prediction_weights
  `prediction_weights <LVOCControlMechanism.prediction_weights>`.  Those are then used to determine (and implement)
  the `allocation_policy <LVOCControlMechanism.allocation_policy>` that is predicted to generate the greatest `EVC
  <LVOCControlMechanism_EVC>` based on the `predictor_values <LVOCControlMechanism.predictor_values>` for the current
  trial.

.. _LVOCControlMechanism_ControlSignals:

*ControlSignals*
~~~~~~~~~~~~~~~~

The OutputStates of an LVOCControlMechanism (like any `ControlMechanism`) are a set of `ControlSignals
<ControlSignal>`, that are listed in its `control_signals <LVOCControlMechanism.control_signals>` attribute (as well as
its `output_states <ControlMechanism.output_states>` attribute).  Each ControlSignal is assigned a `ControlProjection`
that projects to the `ParameterState` for a parameter controlled by the LVOCControlMechanism.  Each ControlSignal is
assigned an item of the LVOCControlMechanism's `allocation_policy`, that determines its `allocation
<ControlSignal.allocation>` for a given `TRIAL` of execution.  The `allocation <ControlSignal.allocation>` is used by
a ControlSignal to determine its `intensity <ControlSignal.intensity>`, which is then assigned as the `value
<ControlProjection.value>` of the ControlSignal's ControlProjection.   The `value <ControlProjection>` of the
ControlProjection is used by the `ParameterState` to which it projects to modify the value of the parameter (see
`ControlSignal_Modulation` for description of how a ControlSignal modulates the value of a parameter it controls).
A ControlSignal also calculates a `cost <ControlSignal.cost>`, based on its `intensity <ControlSignal.intensity>`
and/or its time course. The `cost <ControlSignal.cost>` may be included in the evaluation carried out by the
LVOCControlMechanism's `function <LVOCControlMechanism.function>` for a given `allocation_policy`,
and that it uses to adapt the ControlSignal's `allocation <ControlSignal.allocation>` in the future.

.. _LVOCControlMechanism_Execution:

Execution
---------

When an LVOCControlMechanism is executed, it uses the values of its `predictors <LVOCControlMechanism_Predictors>`,
listed in its `predictor_values <LVOCControlMechanism.predictor_values>` attribute, to determines and implement the
`allocation_policy` for the current `trial` of execution of its `composition <LVOCControlMechanism.composition>`.
Specifically it executes the following steps:

  * Updates `prediction_vector <LVOCControlMechanism.prediction_vector>` with the current `predictors_values
    <LVOCControlMechanism.predictor_values>`, `control_signals <LVOCControlMechanism.control_signals>`,
    and their `costs <ControlSignal.cost>`.

  * Calls its `function <LVOCControlMechanism.function>` with the `prediction_vector
    <LVOCControlMechanism.prediction_vector>` and the outcome received from the
    LVOCControlMechanism's `objective_mechanism <LVOCControlMechanism.objective_mechanism>`, discounted by the
    `costs <ControlSignal.cost>` associated with each of its `control_signals <LVOCControlMechanism.control_signals>`,
    to update its `prediction_weights <LVOCControlMechanism.prediction_weights>`.

  * Calls its `gradient_ascent <LVOCControlMechanism.gradient_ascent>` function with `prediction_vector
    <LVOCControlMechanism.prediction_vector>` and `prediction_weights <LVOCControlMechanism.prediction_weights>`
    to determine the `allocation_policy <LVOCControlMechanism.alocation_policy>` that yields the greatest `EVC
    <LVOCControlMechanism_EVC>`, and returns that `allocation_policy <LVOCControlMechanism.allocation_policy>`.

The values specified by the `allocation_policy <LVOCControlMechanism.allocation_policy>` returned by the
LVOCControlMechanism's `function <LVOCControlMechanism.function>` are assigned as the `values <ControlSignal.values>`
of its `control_signals <LVOCControlMechanism.control_signals>`.

COMMENT:
.. _LVOCControlMechanism_Examples:

Example
-------
COMMENT

.. _LVOCControlMechanism_Class_Reference:

Class Reference
---------------

"""
import warnings

from collections import Iterable, deque

import numpy as np
import typecheck as tc

from psyneulink.core.components.functions.function import BayesGLM, EPSILON, ModulationParam, _is_modulation_param, is_function_type
from psyneulink.core.components.mechanisms.adaptive.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import MONITORED_OUTPUT_STATES, OUTCOME, ObjectiveMechanism
from psyneulink.core.components.shellclasses import Function
from psyneulink.core.components.states.inputstate import InputState
from psyneulink.core.components.states.modulatorysignals.controlsignal import ControlSignalCosts
from psyneulink.core.components.states.outputstate import OutputState
from psyneulink.core.components.states.parameterstate import ParameterState
from psyneulink.core.components.states.state import _parse_state_spec
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.defaults import defaultControlAllocation
from psyneulink.core.globals.keywords import FUNCTION, INTERNAL_ONLY, LVOCCONTROLMECHANISM, NAME, OBJECTIVE_MECHANISM, PARAMETER_STATES, PARAMS, VARIABLE
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import is_iterable

__all__ = [
    'LVOCControlMechanism', 'LVOCError', 'SHADOW_EXTERNAL_INPUTS',
]

SHADOW_EXTERNAL_INPUTS = 'SHADOW_EXTERNAL_INPUTS'
PREDICTION_WEIGHTS = 'PREDICTION_WEIGHTS'

class LVOCError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class LVOCControlMechanism(ControlMechanism):
    """LVOCControlMechanism(                        \
    predictors,                                     \
    predictor_function=None,                        \
    objective_mechanism=None,                       \
    origin_objective_mechanism=False,               \
    terminal_objective_mechanism=False,             \
    function=BayesGLM,                              \
    update_rate=0.1,                                \
    convergence_criterion=.001,                     \
    max_iterations=1000,                            \
    control_signals=None,                           \
    modulation=ModulationParam.MULTIPLICATIVE,      \
    params=None,                                    \
    name=None,                                      \
    prefs=None)

    Subclass of `ControlMechanism <ControlMechanism>` that optimizes the `ControlSignals <ControlSignal>` for a
    `Composition`.

    Arguments
    ---------

    predictors : Mechanism, OutputState, Projection, dict, or list containing any of these
        specifies the values that the LVOCControlMechanism learns to use for determining its `allocation_policy
        <LVOCControlMechanism.allocation_policy>`.  Any `InputState specification <InputState_Specification>`
        can be used that resolves to an `OutputState` that projects to the InputState.  In addition, a dictionary
        with a *SHADOW_EXTERNAL_INPUTS* entry can be used to shadow inputs to the Composition's `ORIGIN` Mechanism(s)
        (see `LVOCControlMechanism_Creation` for details).

    predictor_function : Function or function : default None
        specifies the `function <InputState.function>` for the `InputState` assigned to each `predictor
        <LVOCControlMechanism_Predictors>`.

    objective_mechanism : ObjectiveMechanism or List[OutputState specification] : default None
        specifies either an `ObjectiveMechanism` to use for the LVOCControlMechanism, or a list of the `OutputState
        <OutputState>`\\s it should monitor; if a list of `OutputState specifications
        <ObjectiveMechanism_Monitored_Output_States>` is used, a default ObjectiveMechanism is created and the list
        is passed to its **monitored_output_states** argument.

    function : LearningFunction or callable : BayesGLM
        specifies the function used to learn to predict the outcome of `objective_mechanism
        <LVOCControlMechanism.objective_mechanism>` minus the `costs <ControlSignal.cost>` of the
        `control_signals <LVOCControlMechanism.control_signals>` from the `prediction_vector
        <LVOCControlMechanism.prediction_vector>` (see `LVOCControlMechanism_Function` for details).

    update_rate : int or float : default 0.1
        specifies the amount by which the `value <ControlSignal.value>` of each `ControlSignal` in the
        `allocation_policy <LVOCControlMechanism.allocation_policy>` is modified in each iteration of the
        `gradient_ascent <LVOCControlMechanism.gradient_ascent>` method.

    convergence_criterion : int or float : default 0.001
        specifies the change in estimate of the `EVC <LVOCControlMechanism_EVC>` below which the `gradient_ascent
        <LVOCControlMechanism.gradient_ascent>` method should terminate and return an `allocation_policy
        <LVOCControlMechanism.allocation_policy>`.

    max_iterations : int : default 1000
        specifies the maximum number of iterations `gradient_ascent <LVOCControlMechanism.gradient_ascent>`
        method is allowed to execute; if exceeded, a warning is issued, and the method returns the
        last `allocation_policy <LVOCControlMechanism.allocation_policy>` evaluated.

    control_signals : ControlSignal specification or List[ControlSignal specification, ...]
        specifies the parameters to be controlled by the LVOCControlMechanism
        (see `ControlSignal_Specification` for details of specification).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for the
        Mechanism, its `function <LVOCControlMechanism.function>`, and/or a custom function and its parameters.  Values
        specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default see `name <LVOCControlMechanism.name>`
        specifies the name of the LVOCControlMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the LVOCControlMechanism; see `prefs <LVOCControlMechanism.prefs>` for details.

    Attributes
    ----------

    predictor_values : 1d ndarray
        the current `values <InputState.value>` of the InputStates used by `function <LVOCControlMechanism.function>`
        to determine `allocation_policy <LVOCControlMechanism.allocation_policy>` (see
        `LVOCControlMechanism_Predictors` for details about predictors).

    objective_mechanism : ObjectiveMechanism
        the 'ObjectiveMechanism' used by the LVOCControlMechanism to evaluate the performance of its `system
        <LVOCControlMechanism.system>`.  If a list of OutputStates is specified in the **objective_mechanism** argument
        of the LVOCControlMechanism's constructor, they are assigned as the `monitored_output_states
        <ObjectiveMechanism.monitored_output_states>` attribute for the `objective_mechanism
        <LVOCControlMechanism.objective_mechanism>` (see LVOCControlMechanism_ObjectiveMechanism for additional
        details).

    monitored_output_states : List[OutputState]
        list of the OutputStates monitored by `objective_mechanism <LVOCControlMechanism.objective_mechanism>`
        (and listed in its `monitored_output_states <ObjectiveMechanism.monitored_output_states>` attribute),
        and used to evaluate the performance of the LVOCControlMechanism's `system <LVOCControlMechanism.system>`.

    monitored_output_states_weights_and_exponents: List[Tuple[scalar, scalar]]
        a list of tuples, each of which contains the weight and exponent (in that order) for an OutputState in
        `monitored_outputStates`, listed in the same order as the outputStates are listed in `monitored_outputStates`.

    prediction_vector : 1d ndarray
        current values, respectively, of `predictors <LVOCControlMechanism_Predictors>`, interaction terms for
        predictors x control_signals, `control_signals <LVOCControlMechanism.control_signals>`, and `costs
        <ControlSignal.cost>` of control_signals.

    prediction_weights : 1d ndarray
        weights assigned to each term of `prediction_vector <LVOCControlMechanism.prediction_vectdor>`
        last returned by `function <LVOCControlMechanism.function>`.

    function : LearningFunction or callable
        takes `prediction_vector <LVOCControlMechanism.prediction_vector>` and outcome and returns an updated set of
        `prediction_weights <LVOCControlMechanism.prediction_weights>` (see `LVOCControlMechanism_Function`
        for additional details).

    update_rate : int or float
        determines the amount by which the `value <ControlSignal.value>` of each `ControlSignal` in the
        `allocation_policy <LVOCControlMechanism.allocation_policy>` is modified in each iteration of the
        `gradient_ascent <LVOCControlMechanism.gradient_ascent>` method.

    convergence_criterion : int or float
        determines the change in estimate of the `EVC <LVOCControlMechanism_EVC>` below which the `gradient_ascent
        <LVOCControlMechanism.gradient_ascent>` method should terminate and return an `allocation_policy
        <LVOCControlMechanism.allocation_policy>`.

    max_iterations : int
        determines the maximum number of iterations `gradient_ascent <LVOCControlMechanism.gradient_ascent>`
        method is allowed to execute; if exceeded, a warning is issued, and the method returns the
        last `allocation_policy <LVOCControlMechanism.allocation_policy>` evaluated.

    allocation_policy : 2d np.array : defaultControlAllocation
        determines the value assigned as the `variable <ControlSignal.variable>` for each `ControlSignal` and its
        associated `ControlProjection`.  Each item of the array must be a 1d array (usually containing a scalar)
        that specifies an `allocation` for the corresponding ControlSignal, and the number of items must equal the
        number of ControlSignals in the LVOCControlMechanism's `control_signals` attribute.

    control_signals : ContentAddressableList[ControlSignal]
        list of the LVOCControlMechanism's `ControlSignals <LVOCControlMechanism_ControlSignals>`, including any that it inherited
        from its `system <LVOCControlMechanism.system>` (same as the LVOCControlMechanism's `output_states
        <Mechanism_Base.output_states>` attribute); each sends a `ControlProjection` to the `ParameterState` for the
        parameter it controls

    name : str
        the name of the LVOCControlMechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the LVOCControlMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    """

    componentType = LVOCCONTROLMECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to Type automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'DefaultControlMechanismCustomClassPreferences',
    #     kp<pref>: <setting>...}

    class Params(ControlMechanism.Params):
        function = BayesGLM

    paramClassDefaults = ControlMechanism.paramClassDefaults.copy()
    paramClassDefaults.update({PARAMETER_STATES: NotImplemented}) # This suppresses parameterStates

    @tc.typecheck
    def __init__(self,
                 predictors:tc.optional(tc.any(Iterable, Mechanism, OutputState, InputState)),
                 predictor_function:tc.optional(tc.any(is_function_type))=None,
                 objective_mechanism:tc.optional(tc.any(ObjectiveMechanism, list))=None,
                 origin_objective_mechanism=False,
                 terminal_objective_mechanism=False,
                 function=BayesGLM,
                 update_rate=0.1,
                 convergence_criterion=0.001,
                 max_iterations=1000,
                 control_signals:tc.optional(tc.any(is_iterable, ParameterState))=None,
                 modulation:tc.optional(_is_modulation_param)=ModulationParam.MULTIPLICATIVE,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(input_states=predictors,
                                                  predictor_function=predictor_function,
                                                  convergence_criterion=convergence_criterion,
                                                  max_iterations=max_iterations,
                                                  update_rate=update_rate,
                                                  origin_objective_mechanism=origin_objective_mechanism,
                                                  terminal_objective_mechanism=terminal_objective_mechanism,
                                                  params=params)

        super().__init__(system=None,
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

        if (OBJECTIVE_MECHANISM in request_set and
                isinstance(request_set[OBJECTIVE_MECHANISM], ObjectiveMechanism)
                and not request_set[OBJECTIVE_MECHANISM].path_afferents):
            raise LVOCError("{} specified for {} ({}) must be assigned one or more {}".
                            format(ObjectiveMechanism.__name__, self.name,
                                   request_set[OBJECTIVE_MECHANISM], repr(MONITORED_OUTPUT_STATES)))

    def _instantiate_input_states(self, context=None):
        """Instantiate input_states for Projections from predictors and objective_mechanism.

        Inserts InputState specification for Projection from ObjectiveMechanism as first item in list of
        InputState specifications generated in _parse_predictor_specs from the **predictors** and
        **predictor_function** arguments of the LVOCControlMechanism constructor.
        """

        self.input_states = self._parse_predictor_specs(self.input_states, self.predictor_function)

        # Insert primary InputState for outcome from ObjectiveMechanism; assumes this will be a single scalar value
        self.input_states.insert(0, {NAME:OUTCOME, PARAMS:{INTERNAL_ONLY:True}}),

        # Configure default_variable to comport with full set of input_states
        self.instance_defaults.variable, ignore = self._handle_arg_input_states(self.input_states)

        super()._instantiate_input_states(context=context)

    tc.typecheck
    def add_predictors(self, predictors):
        '''Add InputStates and Projections to LVOCControlMechanism for predictors used to predict outcome

        **predictors** argument can use any of the forms of specification allowed for InputState(s),
            as well as a dictionary containing an entry with *SHADOW_EXTERNAL_INPUTS* as its key and a
            list of `ORIGIN` Mechanisms and/or their InputStates as its value.
        '''

        predictors = self._parse_predictor_specs(predictors=predictors,
                                                 context=ContextFlags.COMMAND_LINE)
        self.add_states(InputState, predictors)

    @tc.typecheck
    def _parse_predictor_specs(self, predictors, predictor_function, context=None):
        """Parse entries of predictors into InputState spec dictionaries

        For InputState specs in SHADOW_EXTERNAL_INPUTS ("shadowing" an Origin InputState):
            - Call _parse_shadow_input_spec

        For standard InputState specs:
            - Call _parse_state_spec
            - Set INTERNAL_ONLY entry of params dict of InputState spec dictionary to True

        Assign functions specified in **predictor_function** to InputStates for all predictors

        Returns list of InputState specification dictionaries
        """

        parsed_predictors = []

        if not isinstance(predictors, list):
            predictors = [predictors]

        for spec in predictors:

            # e.g. {SHADOW_EXTERNAL_INPUTS: [A]}
            if isinstance(spec, dict):
                if SHADOW_EXTERNAL_INPUTS in spec:
                    #  composition looks for node.shadow_external_inputs and uses it to set external_origin_sources
                    self.shadow_external_inputs = spec[SHADOW_EXTERNAL_INPUTS]
                    spec = self._parse_shadow_inputs_spec(spec, predictor_function)
                else:
                    raise LVOCError("Incorrect specification ({}) in predictors argument of {}."
                                    .format(spec, self.name))
            # e.g. Mechanism, OutputState
            else:
                spec = _parse_state_spec(state_type=InputState, state_spec=spec)    # returns InputState dict
                spec[PARAMS][INTERNAL_ONLY] = True
                if predictor_function:
                    spec[PARAMS][FUNCTION] = predictor_function
                spec = [spec]   # so that extend works below

            parsed_predictors.extend(spec)

        return parsed_predictors

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
            input_state_specs.extend([{#NAME:i.name + ' of ' + i.owner.name,
                                       VARIABLE: i.variable,
                            }
                                      for i in input_states])
        if fct:
            for i in input_state_specs:
                i.update({FUNCTION:fct})

        return input_state_specs

    def _instantiate_control_signal(self, control_signal, context=None):
        '''Implement ControlSignalCosts.DEFAULTS as default for cost_option of ControlSignals
        LVOCControlMechanism requires use of at least one of the cost options
        '''
        control_signal = super()._instantiate_control_signal(control_signal, context)

        if control_signal.cost_options is None:
            control_signal.cost_options = ControlSignalCosts.DEFAULTS
            control_signal._instantiate_cost_attributes()
        return control_signal

    def _execute(self, variable=None, runtime_params=None, context=None):
        """Determine `allocation_policy <LVOCControlMechanism.allocation_policy>` for current run of Composition

        Update `prediction_weights <LVOCControlMechanism.prediction_weights>` to better predict outcome of
        `LVOCControlMechanism's <LVOCControlMechanism>` `objective_mechanism <LVOCControlMechanism.objective_mechanism>`
        minus the summed costs of `control_signals <LVOCControlMechanism.control_signals>` from prediction_vector, and
        then determine `allocation_policy <LVOCControlMechanism>` that yields greatest `EVC <LVCOControlMechanism_EVC>`
        given the new `prediction_weights <LVOCControlMechanism.prediction_weights>`.

        variable should have two items:
          - variable[0]: current `prediction_vector <LVOCControlMechanism.prediction_vector> and
          - variable[1]: `value <OutputState.value>` of the *OUTCOME* OutputState of `objective_mechanism
            <LVOCControlMechanism.objective_mechanism>`.

        Call to super._execute calculates outcome from last trial, by subtracting the `costs <ControlSignal.costs>` for
        the `control_signal <LVOCControlMechanism.control_signals>` values used in the previous trial from the value
        received from the `objective_mechanism <LVOCControlMechanism.objective_mechanism>` (in variable[1]) reflecting
        performance on the previous trial.  It then calls the LVOCControlMechanism's `function
        <LVOCControlMechanism.function>` to update the `prediction_weights <LVOCControlMechanism.prediction_weights>`
        so as to better predict the outcome.

        Call to `gradient_ascent` optimizes `allocation_policy <LVOCControlMechahism.allocation_policy>` given new
        `prediction_weights <LVOCControlMechanism.prediction_weights>`.

        """

        if (self.context.initialization_status == ContextFlags.INITIALIZING):
            return defaultControlAllocation

        # Get sample of weights
        # IMPLEMENTATION NOTE: skip ControlMechanism._execute since it is a stub method that returns input_values
        self.prediction_weights = super(ControlMechanism, self)._execute(variable=variable,
                                                                         runtime_params=runtime_params,
                                                                         context=context
                                                                         )

        # Compute allocation_policy using gradient_ascent
        allocation_policy = self.gradient_ascent(self.control_signals,
                                                 self.prediction_vector,
                                                 self.prediction_weights)

        return allocation_policy.reshape((len(allocation_policy),1))

    def _parse_function_variable(self, variable, context=None):
        '''Update current prediction_vector, and return prediction vector and outcome from previous trial

        Determines prediction_vector for current trial, and buffers this in prediction_buffer;  also buffers
        costs of control_signals used in previous trial and buffers this in previous_costs.

        Computes outcome for previous trial by subtracting costs of control_signals from outcome received
        from objective_mechanism, both of which reflect values assigned in previous trial (since Projection from
        objective_mechanism is a feedback Projection, the value received from it corresponds to the one computed on
        the previous trial).
        # FIX: SHOULD REFERENCE RELEVANT DOCUMENTATION ON COMPOSITION REGARDING FEEDBACK CONNECTIONS)

        Returns prediction_vector and outcome from previous trial, to be used by function to update prediction_weights
        that will be used to predict the EVC for the current trial.

        '''

        # This is the value received from the objective_mechanism's OUTCOME OutputState:
        obj_mech_outcome = variable[0]

        # This is the current values of the predictors
        self.predictor_values = np.array(variable[1:]).reshape(-1)

        # Initialize attributes
        if context is ContextFlags.INSTANTIATE:
            # Numbers of terms in prediction_vector
            self.num_predictors = len(self.predictor_values)
            self.num_control_signals = self.num_costs = len(self.control_signals)
            self.num_interactions = self.num_predictors * self.num_control_signals
            len_prediction_vector = \
                self.num_predictors + self.num_interactions + self.num_control_signals + self.num_costs

            # Indices for fields of prediction_vector
            self.pred = slice(0, self.num_predictors)
            self.intrxn= slice(self.num_predictors, self.num_predictors+self.num_interactions)
            self.ctl = slice(self.intrxn.stop, self.intrxn.stop + self.num_control_signals)
            self.cst = slice(self.ctl.stop, len_prediction_vector)

            self.prediction_vector = np.zeros(len_prediction_vector)
            self.prediction_buffer = deque([self.prediction_vector], maxlen=2)
            self.previous_cost = np.zeros_like(obj_mech_outcome)

        else:
            # Populate fields (subvectors) of prediction_vector
            self.prediction_vector[self.pred] = self.predictor_values
            self.prediction_vector[self.ctl] = np.array([c.value for c in self.control_signals]).reshape(-1)
            self.prediction_vector[self.intrxn]= \
                np.array(self.prediction_vector[self.pred] *
                         self.prediction_vector[self.ctl].reshape(self.num_control_signals,1)).reshape(-1)
            self.prediction_vector[self.cst] = \
                np.array([0 if c.cost is None else c.cost for c in self.control_signals]).reshape(-1) * -1

            self.prediction_buffer.append(self.prediction_vector)
            self.previous_cost = np.sum(self.prediction_vector[self.cst])

        outcome = obj_mech_outcome + self.previous_cost # costs are assigned as negative above, so add them here

        return [self.prediction_buffer[0], outcome]

    def gradient_ascent(self, control_signals, prediction_vector, prediction_weights):
        '''Determine the `allocation_policy <LVOCControlMechanism.allocation_policy>` that maximizes the `EVC
        <LVOCControlMechanism_EVC>`.

        Iterate over prediction_vector; for each iteration: \n
        - compute gradients based on current control_signal values and their costs (in prediction_vector);
        - compute new control_signal values based on gradients;
        - update prediction_vector with new control_signal values and the interaction terms and costs based on those;
        - use prediction_weights and updated prediction_vector to compute new `EVC <LVOCControlMechanism_EVC>`.

        Continue to iterate until difference between new and old EVC is less than `convergence_criterion
        <LearnAllocationPolicy.convergence_criterion>` or number of iterations exceeds `max_iterations
        <LearnAllocationPolicy.max_iterations>`.

        Return control_signals field of prediction_vector (used by LVOCControlMechanism as its `allocation_vector
        <LVOCControlMechanism.allocation_policy>`).

        '''

        convergence_metric = self.convergence_criterion + EPSILON
        previous_lvoc = np.finfo(np.longdouble).max

        predictors = prediction_vector[0:self.num_predictors]

        # Get interaction weights and reshape so that there is one row per control_signal
        #    containing the terms for the interaction of that control_signal with each of the predictors
        interaction_weights = prediction_weights[self.intrxn].reshape(self.num_control_signals,self.num_predictors)
        # multiply interactions terms by predictors (since those don't change during the gradient ascent)
        interaction_weights_x_predictors = interaction_weights * predictors

        control_signal_values = prediction_vector[self.ctl]
        control_signal_weights = prediction_weights[self.ctl]

        gradient_constants = np.zeros(self.num_control_signals)
        for i in range(self.num_control_signals):
            gradient_constants[i] = control_signal_weights[i]
            gradient_constants[i] += np.sum(interaction_weights_x_predictors[i])

        costs = prediction_vector[self.cst]
        cost_weights = prediction_weights[self.cst]

        # TEST PRINT:
        print('\n\npredictors: ', predictors,
              '\ncontrol_signals: ', control_signal_values,
              '\ncontrol_costs: ', costs,
              '\nprediction_weights: ', prediction_weights)
        # TEST PRINT END:

        # Perform gradient ascent until convergence criterion is reached
        j=0
        while convergence_metric > self.convergence_criterion:
            # initialize gradient arrray (one gradient for each control signal)
            gradient = np.copy(gradient_constants)
            cost_gradient = np.zeros(self.num_costs)

            for i, control_signal_value in enumerate(control_signal_values):

                # Recompute costs and add to gradient
                cost_function_derivative = control_signals[i].intensity_cost_function.__self__.derivative
                cost_gradient[i] = -(cost_function_derivative(control_signal_value) * cost_weights[i])
                gradient[i] += cost_gradient[i]

                # Update control_signal_value with gradient
                control_signal_values[i] = control_signal_value + self.update_rate * gradient[i]

                # Update cost based on new control_signal_value
                costs[i] = -(control_signals[i].intensity_cost_function(control_signal_value))

            # Assign new values of interaction terms, control_signals and costs to prediction_vector
            prediction_vector[self.intrxn]= np.array(prediction_vector[self.pred] *
                                                     prediction_vector[self.ctl].reshape(self.num_control_signals,1)).\
                                                     reshape(-1)
            prediction_vector[self.ctl] = control_signal_values
            prediction_vector[self.cst] = costs

            # Compute current LVOC using current features, weights and new control signals
            current_lvoc = self.compute_lvoc(prediction_vector, prediction_weights)

            # Compute convergence metric with updated control signals
            convergence_metric = np.abs(current_lvoc - previous_lvoc)

            # TEST PRINT:
            print('\niteration ', j,
                  '\nprevious_lvoc: ', previous_lvoc,
                  '\ncurrent_lvoc: ',current_lvoc ,
                  '\nconvergence_metric: ',convergence_metric,
                  '\npredictors: ', predictors,
                  '\ncontrol_signal_values: ', control_signal_values,
                  '\ninteractions: ', interaction_weights_x_predictors,
                  '\ncosts: ', costs)
            # TEST PRINT END

            j+=1
            if j > self.max_iterations:
                warnings.warn("{} failed to converge after {} iterations".format(self.name, self.max_iterations))
                break

            previous_lvoc = current_lvoc

        return control_signal_values

    def compute_lvoc(self, v, w):
        return np.sum(v * w)
