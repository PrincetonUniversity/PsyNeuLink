# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *****************************************  LVOCControlMechanism ******************************************************

"""

Overview
--------

An LVOCControlMechanism is an `OptimizationControlMechanism` that learns to regulate its `ControlSignals
<ControlSignal>` in order to optimize the performance of the `Composition` to which it belongs.  It
implements a form of the Learned Value of Control model described in `Leider et al.
<https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006043&rev=2>`_, which learns to select the
value for its `control_signals <ControlMechanism.control_signals>` (i.e., its `allocation_policy
<ControlMechanism.allocation_policy>`) that maximzes its `EVC <OptimizationControlMechanism_EVC>` based on a set of
`predictors <LVOCControlMechanism_Feature_Predictors>`.

.. _LVOCControlMechanism_EVC:

*Expected Value of Control (EVC) and Learned Value of Control (LVOC)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `Expected Value of Control (EVC) <OptimizationControlMechanism_EVC>` is the predicted value of executing the
`Composition` to which the LVOCControlMechanism belongs under a given `allocation_policy
<ControlMechanism.allocation_policy>`, as determined by its `objective_function <ControlMechanism.objective_function>`.

The LVOCControlMechanism's `learning_function <LVOCControlMechanism.learning_function>` learns to estimate the EVC
from its `prediction_vector <LVOCControlMechanism.prediction_vector>` -- comprised of `feature_predictors
<LVOCControlMechanism.feature_predictors>`, an `allocation_policy <ControlMechanism.allocation_policy>`, interactions
among these, and the `costs <ControlMechanism.costs> of its `control_signals <ControlMechanism.control_signals>`.
by learning a set of `prediction_weights <LVOCControlMechanism.prediction_weights>` that can predict the
`net_outcome <ControlMechanism.net_outcome>` of processing for experienced values of those variables.   The
set of `prediction_weights <LVOCControlMechaism.prediction_weights>` it learns are referred to as  the **learned value
of control** (*LVOC*).

The LVOCControlMechanism's primary `function <LVOCControlMechanism.function>` optimizes the EVC for the current value
of its `feature_predictors <LVOCControlMechanism.feature_preditors>`, by using its `prediction_weights
LVOCControlMechanism.prediction_weights` (i.e., the LVOC) to estimate the EVC for samples of `allocation_policy
<ControlMechanism.allocation_policy>`, and returning the one that maximizes the EVC for the current `feature_predictors
<LVOCControlMechanism.feature_preditors>`.
  
  
.. _LVOCControlMechanism_Creation:

Creating an LVOCControlMechanism
--------------------------------

 An LVOCControlMechanism can be created in the same was as any `OptimizationControlMechanism`.  The following arguments
 of its constructor are specific to the LVOCControlMechanism:

  * **feature_predictors** -- takes the place of the standard **input_states** argument in the constructor for a
    Mechanism`, and specifies the inputs that it learns to use to determine its `allocation_policy
    <ControlMechanism.allocation_policy>` in each `trial` of execution.
    It can be specified using any of the following, singly or combined in a list:

        * {*SHADOW_EXTERNAL_INPUTS*: <`ORIGIN` Mechanism, InputState for one, or list with either or both>} --
          InputStates of the same shapes as those listed are created on the LVOC, and are connected to the
          corresponding input_CIM OutputStates by projections. The external input values that are passed through the
          input_CIM are used as the `feature_predictors <LVOCControlMechanism_Feature>`. If a Mechanism is included
          in the list, it refers to all of its InputStates.
        |
        * *InputState specification* -- this can be any form of `InputState specification <InputState_Specification>`
          that resolves to an OutputState from which the InputState receives a Projection;  the `value
          <OutputState.value>` of that OutputState is used as the `feature <LVOCControlMechanism.feature>`. Each of
          these InputStates is marked as internal_only.

    Feature_predictors can also be added to an existing LVOCControlMechanism using its `add_features` method.

  * **feature_function** -- specifies `function <InputState>` of the InputState created for each item listed in
    **feature_predictors**.  By default, this is the identity function, that provides the current value of the feature
    to the LVOCControlMechanism's `learning_function <LVOCControlMechanism.learning_function>`.  However,
    other functions can be assigned, to maintain a record of past values, or integrate them over trials.

  * **learning_function** -- specifies `LearningFunction` that learns to predict the `EVC <LVOCControlMechanism_EVC>`
    for a given `allocation_policy <ControlMechanism.allocation_policy>` from the terms specified in the
    **prediction_terms** argument.
    
  * **prediction_terms** -- specifies the terms used by the `learning_function <LVOCControlMechanism.learning_function>`
    and by the LVOCControlMechanism's primary `function <LVOCControlMechanism.function>` to determine the 
    `allocation_policy <ControlMechanism.allocation_policy>` that maximizes the `EVC <LVOCControlMechanism_EVC>`.


.. _LVOCControlMechanism_Structure:

Structure
---------

Same as an OptimizationControlMechanism, with the following exceptions.

.. _LVOCControlMechanism_Input:

*Input*
~~~~~~~

Like any `ControlMechanism`, an LVOCControlMechanism has a `primary InputState <InputState_Primary>` named *OUTCOME*
that receives a `Projection` from the *OUTCOME* `OutputState` of its `objective_mechanism
<ControlMechanism.objective_mechanism>`. However, it also has an additional InputState for each of its
feature_predictors, as described below.

.. _LVOCControlMechanism_Feature_Predictors:

Feature Predictors
^^^^^^^^^^^^^^^^^^

Features_Predictors, together with the LVOCControlMechanism's `control_signals <ControlMechanism.control_signals>`
and `costs <ControlMechanism.costs>` are assigned to its `prediction_vector <LVOCControlMechanism.prediction_vector>`,
from which its `learning_function <LVOCControlMechanism.learning_function>` learns to predict the `EVC
<LVOCControlMechanism_EVC>`.

Feature_Predictors can be of two types:

* *Input Feature Predictor* -- this is a value received as input by an `ORIGIN` Mechanism in the `Composition`.
    These are specified in the **feature_predictors** argument of the LVOCControlMechanism's constructor (see
    `LVOCControlMechanism_Creation`), in a dictionary containing a *SHADOW_EXTERNAL_INPUTS* entry, the value of
    which is one or more `ORIGIN` Mechanisms and/or their `InputStates <InputState>` to be shadowed.  For each, 
    a `Projection` is automatically created that parallels ("shadows") the Projection from the Composition's 
    `InputCIM` to the `ORIGIN` Mechanism, projecting from the same `OutputState` of the InputCIM to the InputState 
    of the LVOCControlMechanism assigned to that feature_predictor.

* *Output Feature Predictor* -- this is the `value <OutputState.value>` of an OutputState of some other Mechanism
    in the Composition.  These too are specified in the **feature_predictors** argument of the LVOCControlMechanism's
    constructor (see `LVOCControlMechanism_Creation`), and each is assigned a Projection from the specified 
    OutputState(s) to the InputState of the LVOCControlMechanism for that feature.

The current `values <InputState.value>` of the InputStates for the feature_predictors are listed in the 
`feature_values <LVOCControlMechanism.feature_values>` attribute.

*Functions*
~~~~~~~~~~~

.. _LVOCControlMechanism_Learning_Function:

Learning Function
^^^^^^^^^^^^^^^^^

The `learning_function <LVOCControlMechanism.learning_function>` of an LVOCControlMechanism learns how to weight its
`feature_predictors <LVOCControlMechanism_Feature_Predictors>`, `allocation_policy
<ControlMechanism.allocation_policy>`, the interactions between these, and the `costs <ControlMechanism.costs>` of its
`control_signals <ControlMechanism.control_signals>`, to best predict the `net_outcome <ControlMechanism.net_outcome>`
that results from their values.  Those weights, together with the current value of its `feature_predictors
<LVOCControlMechanism_Feature_Predictors>` (contained in its `feature_values <LVOCControlMechanism.feature_values>`
attribute, are used by the LVOCControlMechanism's primary `function <LVOCControlMechanism.function>` to estimate
the `EVC <_LVOCControlMechanism_EVC>` for those different samples of `allocation_policy
<ControlMechanism.allocation_policy>`. By  default, the `learning_function <LVOCControlMechanism.function>` is
`BayesGLM`. However, any function can be used that accepts a 2d array, the first item of which is an array of scalar
values (the prediction terms) and the second that is a scalar value (the outcome to be predicted), and returns an
array with the same shape as the LVOCControlMechanism's `allocation_policy <ControlMechanism.allocation_policy>`.

.. note::
  The LVOCControlMechanism's `function <LVOCControlMechanism.learning_function>` is provided the `feature_values
  <LVOCControlMechanism.feature_values>` and `net_outcome <ControlMechanism.net_outcome>` from the *previous* trial
  to update the `prediction_weights <LVOCControlMechanism.prediction_weights>`.  Those are then used to estimate
  (and implement) the `allocation_policy <ControlMechanism.allocation_policy>` that is  predicted to generate the
  greatest `EVC <LVOCControlMechanism_EVC>` based on the `feature_values <LVOCControlMechanism.feature_values>` for
  the current trial.

.. _LVOCControlMechanism_Function:

*Primary Function*
^^^^^^^^^^^^^^^^^^

The `function <LVOCControlMechanism.function>` of an LVOCControlMechanism uses the `prediction_weights
<LVOCControlMechanism.prediction_weights>` learned by its `learning_function <LVOCControlMechanism.learning_function>`,
together with the current `feature_values <LVOCControlMechanism.feature_values>`, to find the `allocation_policy
<ControlMechanism.allocation_policy>` that maximizes the estimated `EVC <LVOCControlMechanism_EVC>`.  It does this by
selecting samples of `allocation_policy <ControlMechanism.allocation_policy>` and evaluating each using its
`objective_function <OptimizationControlMechanism.objective_function>`.  The latter calls the LVOCControlMechanism's
`compute_EVC <LVOCControlMechanism.compute_EVC>` method, which uses `prediction_weights
<LVOCControlMechanism.prediction_weights>` and current `feature_values <LVOCControlMechanism.feature_values>`
to estimate the EVC for each `allocation_policy <ControlMechanism.allocation_policy>` sampled.  The `function
<LVOCControlMechanism.function>` returns the `allocation_policy <ControlMechanism.allocation_policy>` that yields
the greatest `EVC <LVOCControlMechanism_EVC>.

The default for `function <LVOCControlMechanism.function>` is the `GradientOptimization` Function, which uses
gradient ascent to select samples of `allocation_policy <ControlMechanism.allocation_policy>` that yield
progessively better values of `EVC <LVOCControlMechanism_EVC>`. However, any `OptimizationFunction` can be used in
its place.  A custom function can also be used, however it must meet the requirements for the `function
<OptimizationControlMechanism.function>` of an `OptimizationControlFunction`, as described `here
<OptimizationControlMechanism_Custom_Funtion>`.

.. _LVOCControlMechanism_ControlSignals:

.. _LVOCControlMechanism_Execution:

Execution
---------

When an LVOCControlMechanism is executed, it first calls its `learning_function
<LVOCControlMechanism.learning_function>`, which uses information from the previous trial to update its
`prediction_weights <LVOCControlMechanism.prediction_weights>`.  It then calls its `function
<LVOCControlMechanism.function>`, which uses those weights to predict the `allocadtion_policy
<ControlMechanism.allocation_policy>` that will yield the greatet `EVC <LVOCControlMechanism_EVC>`, and then
implements that for the next `trial` of execution.  Specifically, it executes the following steps:

  * Calls `learning_function <LVOCControlMechanism.learning_function>` with the `prediction_vector
    <LVOCControlMechanism.prediction_vector>` (containing the `feature_values
    <LVOCControlMechanism.feature_values>`, `allocation_policy <ControlMechanism.allocation_policy>`, and associated
    `costs <ControlMechanism.cost>`) for the previous trial, together with the `net_outcome
    <ControlMechanism.net_outcome>` for that trial, and updates its `prediction_weights
    <LVOCControlMechanism.prediction_weights>`.

  * Updates `prediction_vector <LVOCControlMechanism.prediction_vector>` with the `features_values
    <LVOCControlMechanism.feature_values>` for the current trial.

  * Calls `function <LVOCControlMechanism.function>`, which uses the current `feature_values
    <LVOCControlMechanism.feature_values>` and `prediction_weights <LVOCControlMechanism.prediction_weights>` to
    determine the `allocation_policy <LVOCControlMechanism.alocation_policy>` that yields the greatest `EVC
    <LVOCControlMechanism_EVC>` (see `above <LVOCControlMechanism_Learning_Function>` for details).

The values in the `allocation_policy <ControlMechanism.allocation_policy>` returned by `function
<LVOCControlMechanism.function>` are assigned as the `variables <ControlSignal.variables>` of its `control_signals
<ControlMechanism.control_signals>`, from which they compute their `values <ControlSignal.value>`.

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
from itertools import product
import typecheck as tc
# from aenum import AutoNumberEnum, auto
from enum import Enum

import numpy as np

from psyneulink.core.components.functions.function import \
    ModulationParam, _is_modulation_param, BayesGLM, is_function_type, GradientOptimization, OBJECTIVE_FUNCTION, \
    SEARCH_SPACE
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.components.mechanisms.adaptive.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.adaptive.control.optimizationcontrolmechanism import \
    OptimizationControlMechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import \
    ObjectiveMechanism, MONITORED_OUTPUT_STATES
from psyneulink.core.components.states.state import _parse_state_spec
from psyneulink.core.components.states.inputstate import InputState
from psyneulink.core.components.states.outputstate import OutputState
from psyneulink.core.components.states.parameterstate import ParameterState
from psyneulink.core.components.states.modulatorysignals.controlsignal import ControlSignalCosts, ControlSignal
from psyneulink.core.components.shellclasses import Function
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import \
    DEFAULT_VARIABLE, INTERNAL_ONLY, PARAMS, LVOC_CONTROL_MECHANISM, NAME, \
    PARAMETER_STATES, VARIABLE, OBJECTIVE_MECHANISM, OUTCOME, FUNCTION, ALL, CONTROL_SIGNALS
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.defaults import defaultControlAllocation
from psyneulink.core.globals.utilities import ContentAddressableList, is_iterable, powerset, tensor_power

__all__ = [
    'LVOC', 'LVOCControlMechanism', 'LVOCError', 'SHADOW_EXTERNAL_INPUTS', 'PREDICTION_TERMS', 'PV'
]

LVOC = 'LVOC'
FEATURE_PREDICTORS = 'feature_predictors'
SHADOW_EXTERNAL_INPUTS = 'SHADOW_EXTERNAL_INPUTS'
PREDICTION_WEIGHTS = 'PREDICTION_WEIGHTS'
PREDICTION_TERMS = 'prediction_terms'
PREDICTION_WEIGHT_PRIORS = 'prediction_weight_priors'


class PV(Enum):
# class PV(AutoNumberEnum):
    '''PV()
    Specifies terms used to compute `prediction_vector <LVOCControlMechanism.prediction_vector>`.

    Attributes
    ----------

    F
        Main effect of `feature_predictors <LVOCControlMechanism_Feature_Predictors>`.
    C
        Main effect of `values <ControlSignal.value>` of `control_signals <ControlMechanism.control_signals>`.
    FF
        Interaction among `feature_predictors <LVOCControlMechanism_Feature_Predictors>`.
    CC
        Interaction among `values <ControlSignal.value>` of `control_signals <ControlMechanism.control_signals>`.
    FC
        Interaction between `feature_predictors <LVOCControlMechanism_Feature_Predictors>` and
        `values <ControlSignal.value>` of `control_signals <ControlMechanism.control_signals>`.
    FFC
        Interaction between interactions of `feature_predictors <LVOCControlMechanism_Feature_Predictors>` and
        `values <ControlSignal.value>` of `control_signals <ControlMechanism.control_signals>`.
    FCC
        Interaction between `feature_predictors <LVOCControlMechanism_Feature_Predictors>` and interactions among
        `values <ControlSignal.value>` of `control_signals <ControlMechanism.control_signals>`.
    FFCC
        Interaction between interactions of `feature_predictors <LVOCControlMechanism_Feature_Predictors>` and
        interactions among `values <ControlSignal.value>` of `control_signals <ControlMechanism.control_signals>`.
    COST
        Main effect of `costs <ControlSignal.cost>` of `control_signals <ControlMechanism.control_signals>`.
    '''
    # F =    auto()
    # C =    auto()
    # FF =   auto()
    # CC =   auto()
    # FC =   auto()
    # FFC =  auto()
    # FCC =  auto()
    # FFCC = auto()
    # COST = auto()
    F =    0
    C =    1
    FF =   2
    CC =   3
    FC =   4
    FFC =  5
    FCC =  6
    FFCC = 7
    COST = 8

class LVOCError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class LVOCControlMechanism(OptimizationControlMechanism):
    """LVOCControlMechanism(                               \
    feature_predictors,                                    \
    feature_function=None,                                 \
    objective_mechanism=None,                              \
    origin_objective_mechanism=False,                      \
    terminal_objective_mechanism=False,                    \
    learning_function=BayesGLM,                            \
    prediction_terms=[PV.F, PV.C, PV.FC, PV.COST]          \
    function=GradientOptimization,                         \
    control_signals=None,                                  \
    modulation=ModulationParam.MULTIPLICATIVE,             \
    params=None,                                           \
    name=None,                                             \
    prefs=None)

    Subclass of `OptimizationControlMechanism` that learns to optimize its `ControlSignals <ControlSignal>`.

    Arguments
    ---------

    feature_predictors : Mechanism, OutputState, Projection, dict, or list containing any of these
        specifies values to assign to `prediction_vector <LVOCControlMechanism.prediction_vector>`,
        that are used to estimate `EVC <LVOCControlMechanism_EVC>`.  Any `InputState specification
        <InputState_Specification>` can be used that resolves to an `OutputState` that projects to the InputState.
        In addition, a dictionary with a *SHADOW_EXTERNAL_INPUTS* entry can be used to shadow inputs to the
        Composition's `ORIGIN` Mechanism(s) (see `LVOCControlMechanism_Creation` for details).

    feature_function : Function or function : default None
        specifies the `function <InputState.function>` for the `InputState` assigned to each `feature_predictor
        <LVOCControlMechanism_Feature_Predictors>`.

    objective_mechanism : ObjectiveMechanism or List[OutputState specification] : default None
        specifies either an `ObjectiveMechanism` to use for the LVOCControlMechanism, or a list of the `OutputState
        <OutputState>`\\s it should monitor; if a list of `OutputState specifications
        <ObjectiveMechanism_Monitored_Output_States>` is used, a default ObjectiveMechanism is created and the list
        is passed to its **monitored_output_states** argument.

    learning_function : LearningFunction, function or method : default BayesGLM
        specifies the function used to learn to estimate `EVC <LVOCControlMechanism_EVC>` from the `prediction_vector
        <LVOCControlMechanism.prediction_vector>` and `net_outcome <ControlMechanism.net_outcome>` (see
        `LVOCControlMechanism_Learning_Function` for details).

    prediction_terms : List[PV] : default [PV.F, PV.C, PV.FC, PV.COST]
        specifies terms to be included in `prediction_vector <LVOCControlMechanism.prediction_vector>`.
        items must be members of the `PV` Enum.  If the keyword *ALL* is specified, then all of the terms are used;
        if `None` is specified, the default values will automatically be assigned.

    function : OptimizationFunction, function or method : default GradientOptimization
        specifies the function used to find the `allocation_policy` that maximizes `EVC <LVOCControlMechanism_EVC>`>`;
        must take as its sole argument an array with the same shape as `allocation_policy
        <ControlMechanism.allocation_policy>`, and return a similar array (see `Primary Function
        <LVOCControlMechanism_Function>` for additional details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for the
        Mechanism, its `learning_function <LVOCControlMechanism.learning_function>`, and/or a custom function and its 
        parameters.  Values specified for parameters in the dictionary override any assigned to those parameters in 
        arguments of the constructor.

    name : str : default see `name <LVOCControlMechanism.name>`
        specifies the name of the LVOCControlMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the LVOCControlMechanism; see `prefs <LVOCControlMechanism.prefs>` for details.

    Attributes
    ----------

    feature_values : 1d ndarray
        the current `values <InputState.value>` of `feature_predictors LVOCControlMechanism_Feature_Predictors`.

    prediction_terms : List[PV]
        identifies terms included in `prediction_vector <LVOCControlMechanism.prediction_vector.vector>`;
        items are members of the `PV` enum; the default is [`F <PV.F>`, `C <PV.C>` `FC <PV.FC>`, `COST <PV.COST>`].

    prediction_vector : PredictionVector
        object with `vector <PredictionVector.vector>` containing current values of `feature_predictors
        <LVOCControlMechanism_Feature_Predictors>` `allocation_policy <ControlMechanism.allocation_policy>`,
        their interactions, and `costs <ControlMechanism.costs>` of `control_signals <ControlMechanism.control_signals>`
        as specified in `prediction_terms <LVOCControlMechanism.prediction_terms>`, as well as an `update_vector`
        <PredictionVector.update_vector>` method used to update their values, and attributes for accessing their values.

        COMMENT:
        current values, respectively, of `feature_predictors <LVOCControlMechanism_Feature_Predictors>`,
        interaction terms for feature_predictors x control_signals, `control_signals
        <ControlMechanism.control_signals>`, and `costs <ControlSignal.cost>` of control_signals.
        COMMENT

    prediction_weights : 1d ndarray
        weights assigned to each term of `prediction_vector <LVOCControlMechanism.prediction_vectdor>`
        last returned by `learning_function <LVOCControlMechanism.learning_function>`.

    learning_function : LearningFunction, function or method
        takes `prediction_vector <LVOCControlMechanism.prediction_vector>` and `net_outcome
        <ControlMechanism.net_outcome>` and returns an updated set of `prediction_weights
        <LVOCControlMechanism.prediction_weights>` (see `LVOCControlMechanism_Learning_Function`
        for additional details).

    function : OptimizationFunction, function or method
        takes current `allocation_policy <ControlMechanism.allocation_policy>` (as initializer) and, using the current
        `feature_values <LVOCControlMechanism.feature_values>`, `prediction_weights
        <LVOCControlMechanism.prediction_vector>` and `compute_EVC <LVOCControlMechanism.compute_EVC>`, returns an
        `allocation_policy` that maximizes the `EVC <LVOCControlMechanism_EVC>` (see `Primary Function
        <LVOCControlMechanism_Function>` for additional details).

    name : str
        the name of the LVOCControlMechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the LVOCControlMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentType = LVOC_CONTROL_MECHANISM
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
        function = GradientOptimization

    paramClassDefaults = OptimizationControlMechanism.paramClassDefaults.copy()
    paramClassDefaults.update({PARAMETER_STATES: NotImplemented}) # This suppresses parameterStates

    @tc.typecheck
    def __init__(self,
                 feature_predictors:tc.optional(tc.any(Iterable, Mechanism, OutputState, InputState))=None,
                 feature_function:tc.optional(tc.any(is_function_type))=None,
                 objective_mechanism:tc.optional(tc.any(ObjectiveMechanism, list))=None,
                 origin_objective_mechanism=False,
                 terminal_objective_mechanism=False,
                 learning_function=BayesGLM,
                 prediction_terms:tc.optional(list)=None,
                 function=GradientOptimization,
                 control_signals:tc.optional(tc.any(is_iterable, ParameterState, ControlSignal))=None,
                 modulation:tc.optional(_is_modulation_param)=ModulationParam.MULTIPLICATIVE,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs):

        # Avoid mutable default:
        prediction_terms = prediction_terms or [PV.F,PV.C,PV.FC, PV.COST]
        if ALL in prediction_terms:
            prediction_terms = list(PV.__members__.values())

        if feature_predictors is None:
            # Included for backward compatibility
            if 'predictors' in kwargs:
                feature_predictors = kwargs['predictors']
                del(kwargs['predictors'])
            else:
                raise LVOCError("{} arg for {} must be specified".format(repr(FEATURE_PREDICTORS),
                                                                         self.__class__.__name__))
        if kwargs:
                for i in kwargs.keys():
                    raise LVOCError("Unrecognized arg in constructor for {}: {}".format(self.__class__.__name__,
                                                                                        repr(i)))

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(input_states=feature_predictors,
                                                  feature_function=feature_function,
                                                  prediction_terms=prediction_terms,
                                                  origin_objective_mechanism=origin_objective_mechanism,
                                                  terminal_objective_mechanism=terminal_objective_mechanism,
                                                  params=params)

        super().__init__(objective_mechanism=objective_mechanism,
                         learning_function=learning_function,
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

        if PREDICTION_TERMS in request_set:
            if not all(term in PV for term in request_set[PREDICTION_TERMS]):
                raise LVOCError("One or more items in list specified for {} arg of {} is not a member of the {} enum".
                                format(repr(PREDICTION_TERMS), self.name, PV.__class__.__name__))

    def _instantiate_input_states(self, context=None):
        """Instantiate input_states for Projections from features and objective_mechanism.

        Inserts InputState specification for Projection from ObjectiveMechanism as first item in list of
        InputState specifications generated in _parse_feature_specs from the **feature_predictors** and
        **feature_function** arguments of the LVOCControlMechanism constructor.
        """

        self.input_states = self._parse_feature_specs(self.input_states, self.feature_function)

        # Insert primary InputState for outcome from ObjectiveMechanism;
        #     assumes this will be a single scalar value and must be named OUTCOME by convention of ControlSignal
        self.input_states.insert(0, {NAME:OUTCOME, PARAMS:{INTERNAL_ONLY:True}}),

        # Configure default_variable to comport with full set of input_states
        self.instance_defaults.variable, ignore = self._handle_arg_input_states(self.input_states)

        super()._instantiate_input_states(context=context)

    tc.typecheck
    def add_features(self, feature_predictors):
        '''Add InputStates and Projections to LVOCControlMechanism for feature_predictors used to predict outcome

        **feature_predictors** argument can use any of the forms of specification allowed for InputState(s),
            as well as a dictionary containing an entry with *SHADOW_EXTERNAL_INPUTS* as its key and a
            list of `ORIGIN` Mechanisms and/or their InputStates as its value.
        '''

        feature_predictors = self._parse_feature_specs(feature_predictors=feature_predictors,
                                                 context=ContextFlags.COMMAND_LINE)
        self.add_states(InputState, feature_predictors)

    @tc.typecheck
    def _parse_feature_specs(self, feature_predictors, feature_function, context=None):
        """Parse entries of feature_predictors into InputState spec dictionaries

        For InputState specs in SHADOW_EXTERNAL_INPUTS ("shadowing" an Origin InputState):
            - Call _parse_shadow_input_spec

        For standard InputState specs:
            - Call _parse_state_spec
            - Set INTERNAL_ONLY entry of params dict of InputState spec dictionary to True

        Assign functions specified in **feature_function** to InputStates for all feature_predictors

        Returns list of InputState specification dictionaries
        """

        parsed_features = []

        if not isinstance(feature_predictors, list):
            feature_predictors = [feature_predictors]

        for spec in feature_predictors:

            # e.g. {SHADOW_EXTERNAL_INPUTS: [A]}
            if isinstance(spec, dict):
                if SHADOW_EXTERNAL_INPUTS in spec:
                    #  composition looks for node.shadow_external_inputs and uses it to set external_origin_sources
                    self.shadow_external_inputs = spec[SHADOW_EXTERNAL_INPUTS]
                    spec = self._parse_shadow_inputs_spec(spec, feature_function)
                else:
                    raise LVOCError("Incorrect specification ({}) in feature_predictors argument of {}."
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

    def _instantiate_control_signal(self, control_signal, context=None):
        '''Implement ControlSignalCosts.DEFAULTS as default for cost_option of ControlSignals
        LVOCControlMechanism requires use of at least one of the cost options
        '''
        control_signal = super()._instantiate_control_signal(control_signal, context)

        if control_signal.cost_options is None:
            control_signal.cost_options = ControlSignalCosts.DEFAULTS
            control_signal._instantiate_cost_attributes()
        return control_signal

    def _instantiate_attributes_after_function(self, context=None):
        '''Assign LVOCControlMechanism's objective_function'''

        self.objective_function = self.compute_EVC
        super()._instantiate_attributes_after_function(context=context)

    def _instantiate_learning_function(self):
        '''Instantiate attributes for LVOCControlMechanism's learning_function'''

        self.feature_values = np.array(self.instance_defaults.variable[1:])

        self.prediction_vector = self.PredictionVector(self.feature_values,
                                                       self.control_signals,
                                                       self.prediction_terms)
        # Assign parameters to learning_function
        learning_function_default_variable = [self.prediction_vector.vector, np.zeros(1)]
        if isinstance(self.learning_function, type):
            self.learning_function = self.learning_function(default_variable=learning_function_default_variable)
        else:
            self.learning_function.reinitialize({DEFAULT_VARIABLE: learning_function_default_variable})


    def _execute(self, variable=None, runtime_params=None, context=None):
        """Find allocation_policy that optimizes EVC.

        Items of variable should be:
          - self.outcome: `value <OutputState.value>` of the *OUTCOME* OutputState of `objective_mechanism
            <ControlMechanism.objective_mechanism>`.
          - variable[n]: current value of `feature_predictor <LVOCControlMechanism_Feature_Predictors>`\\[n]

        Executes the following steps:
        - calculate net_outcome from previous trial (value of objective_mechanism - costs of control_signals)
        - call learning_function with net_outcome and prediction_vector from previous trial to update prediction_weights
        - update prediction_vector
        - execute primary (optimization) function to get allocation_policy that maximizes EVC (and corresponding EVC)
        - return allocation_policy
        """

        if (self.context.initialization_status == ContextFlags.INITIALIZING):
            return defaultControlAllocation

        if not self.current_execution_count:
            # Initialize prediction_vector and control_signals on first trial
            # Note:  initialize prediction_vector to 1's so that learning_function returns specified priors
            self._previous_prediction_vector = np.full_like(self.prediction_vector.vector, 0)
            self.prediction_weights = self.learning_function.function([self._previous_prediction_vector, 0])
        else:
            # Update prediction_weights
            self.prediction_weights = self.learning_function.function([self._previous_prediction_vector,
                                                                       self.net_outcome])

            # Update prediction_vector with current feature_values and control_signals and store for next trial
            self.feature_values = np.array(np.array(variable[1:]).tolist())
            self.prediction_vector.update_vector(self.allocation_policy, self.feature_values)
            self._previous_prediction_vector = self.prediction_vector.vector

        # # TEST PRINT
        # print ('\nexecution_count: ', self.current_execution_count)
        # print ('\outcome: ', self.outcome)
        # # print ('prediction_weights: ', self.prediction_weights)
        # # TEST PRINT END

        # Compute allocation_policy using LVOCControlMechanism's optimization function
        # IMPLEMENTATION NOTE: skip ControlMechanism._execute since it is a stub method that returns input_values
        allocation_policy, self.evc_max, self.saved_samples, self.saved_values = \
                                        super(ControlMechanism, self)._execute(variable=self.allocation_policy,
                                                                               runtime_params=runtime_params,
                                                                               context=context)
        # # # TEST PRINT
        # print ('EXECUTION COUNT: ', self.current_execution_count)
        # print ('ALLOCATION POLICY: ', allocation_policy)
        # print ('ALLOCATION POLICY: ', self.evc_max)
        # print ('\n------------------------------------------------')
        # # # TEST PRINT END

        return allocation_policy

    class PredictionVector():
        '''Maintain lists and vector of prediction terms.

        Lists are indexed by the `PV` Enum, and vector fields are indexed by slices listed in the `idx
        <PredicitionVector.idx>` attribute.

        Arguments
        ---------

        feature_values : 2d nparray
            arrays of features to assign as the `PV.F` term of `terms <PredictionVector.terms>`.

        control_signal_variables : List[ControlSignal.variable]
            list containing `variables <ControlSignal.variable>` of `ControlSignals <ControlSignal>`;
            assigned as the `PV.C` term of `terms <PredictionVector.terms>`.

        specified_terms : List[PV]
            terms to include in `vector <PredictionVector.vector>`;
            entries must be members of the `PV` Enum.

        Attributes
        ----------

        specified_terms : List[PV]
            terms included as predictors, specified using members of the `PV` Enum.

        terms : List[ndarray]
            current value of ndarray terms, some of which are used to compute other terms. Only entries for terms in
            `specified_terms <specified_terms>` are assigned values; others are assigned `None`.

        num : List[int]
            number of arrays in outer dimension (axis 0) of each ndarray in `terms <PredictionVector.terms>`.
            Only entries for terms in `specified_terms <PredictionVector.specified_terms>` are assigned values;
            others are assigned `None`.

        num_elems : List[int]
            number of elements in flattened array for each ndarray in `terms <PredictionVector.terms>`.
            Only entries for terms in `specified_terms <PredictionVector.specified_terms>` are assigned values;
            others are assigned `None`.

        self.labels : List[str]
            label of each item in `terms <PredictionVector.terms>`. Only entries for terms in  `specified_terms
            <PredictionVector.specified_terms>` are assigned values; others are assigned `None`.

        vector : ndarray
            contains the flattened array for all ndarrays in `terms <PredictionVector.terms>`.  Contains only
            the terms specified in `specified_terms <PredictionVector.specified_terms>`.  Indices for the fields
            corresponding to each term are listed in `idx <PredictionVector.idx>`.

        idx : List[slice]
            indices of `vector <PredictionVector.vector>` for the flattened version of each nd term in
            `terms <PredictionVector.terms>`. Only entries for terms in `specified_terms
            <PredictionVector.specified_terms>` are assigned values; others are assigned `None`.

        '''

        def __init__(self, feature_values, control_signals, specified_terms):

            # Get variable for control_signals specified in contructor
            control_signal_variables = []
            for c in control_signals:
                if isinstance(c, ControlSignal):
                    try:
                        v = c.variable
                    except:
                        v = c.instance_defaults.variable
                elif isinstance(c, type):
                    if issubclass(c, ControlSignal):
                        v = c.class_defaults.variable
                    else:  # If a class other than ControlSignal was specified, typecheck should have found it
                        raise LVOCError("PROGRAM ERROR: unrecognized specification for {} arg of {}: {}".
                                        format(repr(CONTROL_SIGNALS), self.name, c))
                else:
                    state_spec_dict = _parse_state_spec(state_type=ControlSignal, owner=self, state_spec=c)
                    v = state_spec_dict[VARIABLE]
                    v = v or ControlSignal.class_defaults.variable
                control_signal_variables.append(v)
            self.control_signal_functions = [c.function for c in control_signals]
            self._compute_costs = [c.compute_costs for c in control_signals]

            def get_intrxn_labels(x):
                return list([s for s in powerset(x) if len(s)>1])

            def error_for_too_few_terms(term):
                spec_type = {'FF':'feature_predictors', 'CC':'control_signals'}
                raise LVOCError("Specification of {} for {} arg of {} requires at least two {} be specified".
                                format('PV.'+term, repr(PREDICTION_TERMS), self.name, spec_type(term)))

            F = PV.F.value
            C = PV.C.value
            FF = PV.FF.value
            CC = PV.CC.value
            FC = PV.FC.value
            FFC = PV.FFC.value
            FCC = PV.FCC.value
            FFCC = PV.FFCC.value
            COST = PV.COST.value

            # RENAME THIS AS SPECIFIED_TERMS
            self.specified_terms = specified_terms
            self.terms = [None] * len(PV)
            self.idx =  [None] * len(PV)
            self.num =  [None] * len(PV)
            self.num_elems =  [None] * len(PV)
            self.labels = [None] * len(PV)

            # MAIN EFFECT TERMS (unflattened)

            # Feature_predictors
            self.terms[F] = f = feature_values
            self.num[F] = len(f)  # feature_predictors are arrays
            self.num_elems[F] = len(f.reshape(-1)) # num of total elements assigned to prediction_vector.vector
            self.labels[F] = ['f'+str(i) for i in range(0,len(f))]

            # Placemarker until control_signals are instantiated
            self.terms[C] = c = np.array([[0]] * len(control_signal_variables))
            self.num[C] = len(c)
            self.num_elems[C] = len(c.reshape(-1))
            self.labels[C] = ['c'+str(i) for i in range(0,len(control_signal_variables))]

            # Costs
            # Placemarker until control_signals are instantiated
            self.terms[COST] = cst = np.array([[0]] * len(control_signal_variables))
            self.num[COST] = self.num[C]
            self.num_elems[COST] = len(cst.reshape(-1))
            self.labels[COST] = ['cst'+str(i) for i in range(0,self.num[COST])]

            # INTERACTION TERMS (unflattened)

            # Interactions among feature vectors
            if any(term in specified_terms for term in [PV.FF, PV.FFC, PV.FFCC]):
                if len(f) < 2:
                    self.error_for_too_few_terms('FF')
                self.terms[FF] = ff = np.array(tensor_power(f, levels=range(2,len(f)+1)))
                self.num[FF] = len(ff)
                self.num_elems[FF] = len(ff.reshape(-1))
                self.labels[FF]= get_intrxn_labels(self.labels[F])

            # Interactions among values of control_signals
            if any(term in specified_terms for term in [PV.CC, PV.FCC, PV.FFCC]):
                if len(c) < 2:
                    self.error_for_too_few_terms('CC')
                self.terms[CC] = cc = np.array(tensor_power(c, levels=range(2,len(c)+1)))
                self.num[CC]=len(cc)
                self.num_elems[CC] = len(cc.reshape(-1))
                self.labels[CC] = get_intrxn_labels(self.labels[C])

            # feature-control interactions
            if any(term in specified_terms for term in [PV.FC, PV.FCC, PV.FFCC]):
                self.terms[FC] = fc = np.tensordot(f, c, axes=0)
                self.num[FC] = len(fc.reshape(-1))
                self.num_elems[FC] = len(fc.reshape(-1))
                self.labels[FC] = list(product(self.labels[F], self.labels[C]))

            # feature-feature-control interactions
            if any(term in specified_terms for term in [PV.FFC, PV.FFCC]):
                if len(f) < 2:
                    self.error_for_too_few_terms('FF')
                self.terms[FFC] = ffc = np.tensordot(ff, c, axes=0)
                self.num[FFC] = len(ffc.reshape(-1))
                self.num_elems[FFC] = len(ffc.reshape(-1))
                self.labels[FFC] = list(product(self.labels[FF], self.labels[C]))

            # feature-control-control interactions
            if any(term in specified_terms for term in [PV.FCC, PV.FFCC]):
                if len(c) < 2:
                    self.error_for_too_few_terms('CC')
                self.terms[FCC] = fcc = np.tensordot(f, cc, axes=0)
                self.num[FCC] = len(fcc.reshape(-1))
                self.num_elems[FCC] = len(fcc.reshape(-1))
                self.labels[FCC] = list(product(self.labels[F], self.labels[CC]))

            # feature-feature-control-control interactions
            if PV.FFCC in specified_terms:
                if len(f) < 2:
                    self.error_for_too_few_terms('FF')
                if len(c) < 2:
                    self.error_for_too_few_terms('CC')
                self.terms[FFCC] = ffcc = np.tensordot(ff, cc, axes=0)
                self.num[FFCC] = len(ffcc.reshape(-1))
                self.num_elems[FFCC] = len(ffcc.reshape(-1))
                self.labels[FFCC] = list(product(self.labels[FF], self.labels[CC]))

            # Construct "flattened" prediction_vector based on specified terms, and assign indices (as slices)
            i=0
            for t in range(len(PV)):
                if t in [t.value for t in specified_terms]:
                    self.idx[t] = slice(i, i + self.num_elems[t])
                    i += self.num_elems[t]

            self.vector = np.zeros(i)

        def update_vector(self, variable, feature_values=None):
            '''Update vector with flattened versions of values returned from `compute_terms
            <LVOCControlMechanism.PredictionVector.compute_terms>`.

            Updates `vector <PredictionVector.vector>` used by LVOCControlMechanism as its `prediction_vector
            <LVOCControlMechanism.prediction_vector>`, with current values of variable (i.e., `variable
            <LVOCControlMechanism.variable>`) and, optionally, and feature_vales (i.e., `feature_values
            <LVOCControlMechanism.feature_values>`.

            This method is passed to `function <LVOCControlMechanism.function>` as its **update_function** (see
            `Primary Function <LVOCControlMechanism_Function>`.
            '''

            if feature_values is not None:
                self.terms[PV.F.value] = np.array(feature_values)
            computed_terms = self.compute_terms(np.array(variable))

            # Assign flattened versions of specified terms to vector
            for k, v in computed_terms.items():
                if k in self.specified_terms:
                    self.vector[self.idx[k.value]] = v.reshape(-1)

        def compute_terms(self, control_signal_variables):
            '''Calculate interaction terms.
            Results are returned in a dict; entries are keyed using names of terms listed in the `PV` Enum.
            Values of entries are nd arrays.
            '''

            terms = self.specified_terms
            computed_terms = {}

            # No need to calculate features, so just get values
            computed_terms[PV.F] = f = self.terms[PV.F.value]

            # Compute value of each control_signal from its variable
            c = [None] * len(control_signal_variables)
            for i, var in enumerate(control_signal_variables):
                c[i] = self.control_signal_functions[i](var)
            computed_terms[PV.C] = c = np.array(c)

            # Compute costs for new control_signal values
            if PV.COST in terms:
                # computed_terms[PV.COST] = -(np.exp(0.25*c-3))
                # computed_terms[PV.COST] = -(np.exp(0.25*c-3) + (np.exp(0.25*np.abs(c-self.control_signal_change)-3)))
                costs = [None] * len(c)
                for i, val in enumerate(c):
                    costs[i] = -(self._compute_costs[i](val))
                computed_terms[PV.COST] = np.array(costs)

            # Compute terms interaction that are used
            if any(term in terms for term in [PV.FF, PV.FFC, PV.FFCC]):
                computed_terms[PV.FF] = ff = np.array(tensor_power(f, range(2, self.num[PV.F.value]+1)))
            if any(term in terms for term in [PV.CC, PV.FCC, PV.FFCC]):
                computed_terms[PV.CC] = cc = np.array(tensor_power(c, range(2, self.num[PV.C.value]+1)))
            if any(term in terms for term in [PV.FC, PV.FCC, PV.FFCC]):
                computed_terms[PV.FC] = np.tensordot(f, c, axes=0)
            if any(term in terms for term in [PV.FFC, PV.FFCC]):
                computed_terms[PV.FFC] = np.tensordot(ff, c, axes=0)
            if any(term in terms for term in [PV.FCC, PV.FFCC]):
                computed_terms[PV.FCC] = np.tensordot(f,cc,axes=0)
            if PV.FFCC in terms:
                computed_terms[PV.FFCC] = np.tensordot(ff,cc,axes=0)

            return computed_terms

    def compute_EVC(self, variable):
        '''Update interaction terms and then multiply by prediction_weights

        Serves as `objective_function <OptimizationControlMechanism.objective_function>` for LVOCControlMechanism.

        Uses the current values of `prediction_weights <LVOCControlMechanism.prediction_weights>`
        and `feature_values <LVOCControlMechanism.feature_values>`, together with the `allocation_policy
        <ControlMechanism.allocation_policy>` (provided in variable) to evaluate the `EVC <LVOCControlMechanism_EVC>`.

        .. note::
            If `GradientOptimization` is used as the LVOCControlMechanism's `function <LVOCControlMechanism.function>`,
            this method (including its call to `PredictionVector.compute_terms`) is differentiated using `autograd
            <https://github.com/HIPS/autograd>`_\\.grad().
        '''

        terms = self.prediction_terms
        vector = self.prediction_vector.compute_terms(variable)
        weights = self.prediction_weights
        evc = 0

        for term_label, term_value in vector.items():
            if term_label in terms:
                pv_enum_val = term_label.value
                item_idx = self.prediction_vector.idx[pv_enum_val]
                evc += np.sum(term_value.reshape(-1) * weights[item_idx])

        return evc


# OLD ******************************************************************************************************************
# Manual computation of derivatives

    # def gradient_ascent(self, control_signals, prediction_vector, prediction_weights):
    #     '''Determine the `allocation_policy <LVOCControlMechanism.allocation_policy>` that maximizes the `EVC
    #     <LVOCControlMechanism_EVC>`.
    #
    #     Iterate over prediction_vector; for each iteration: \n
    #     - compute gradients based on current control_signal values and their costs (in prediction_vector);
    #     - compute new control_signal values based on gradients;
    #     - update prediction_vector with new control_signal values and the interaction terms and costs based on those;
    #     - use prediction_weights and updated prediction_vector to compute new `EVC <LVOCControlMechanism_EVC>`.
    #
    #     Continue to iterate until difference between new and old EVC is less than `convergence_threshold
    #     <LearnAllocationPolicy.convergence_threshold>` or number of iterations exceeds `max_iterations
    #     <LearnAllocationPolicy.max_iterations>`.
    #
    #     Return control_signals field of prediction_vector (used by LVOCControlMechanism as its `allocation_vector
    #     <LVOCControlMechanism.allocation_policy>`).
    #     '''
    #
    #     pv = prediction_vector.vector
    #     idx = prediction_vector.idx
    #     # labels = prediction_vector.labels
    #     num_c = prediction_vector.num_c
    #     num_cst = prediction_vector.num_cst
    #     # num_intrxn = prediction_vector.num_interactions
    #
    #     convergence_metric = self.convergence_threshold + EPSILON
    #     previous_value = np.finfo(np.longdouble).max
    #     prev_control_signal_values = np.full(num_c, np.finfo(np.longdouble).max)
    #
    #     feature_predictors = self.feature_values.reshape(-1)
    #
    #     control_signal_values = [np.array(c.value) for c in self.control_signals]
    #
    #     costs = [np.array(c.cost) for c in self.control_signals]
    #     if PV.COST in self.prediction_terms:
    #         cost_weights = prediction_weights[idx.cst]
    #
    #     # COMPUTE DERIVATIVES THAT ARE CONSTANTS
    #     #    Do it here so don't have to do it in each iteration of the while loop
    #
    #     gradient_constants = np.zeros(num_c)
    #
    #     # Derivative for control_signals
    #     if PV.C in self.prediction_terms:
    #         # d(c*wt)/(dc) = wt
    #         gradient_constants += np.array(prediction_weights[idx.c])
    #
    #     # FIX: NEEDS TO BE CHECKED THAT THESE COMPUTE SAME VALUES AS _partial_derivative
    #     # Derivatives for fc interactions:
    #     if PV.FC in self.prediction_terms:
    #         # Get weights for fc interaction term and reshape so that there is one row per control_signal
    #         #    containing the terms for the interaction of that control_signal with each of the feature_predictors
    #         fc_weights = prediction_weights[idx.fc].reshape(num_c, prediction_vector.num_f_elems)
    #         fc_weights_x_features = fc_weights * feature_predictors
    #         for i in range(num_c):
    #             gradient_constants[i] += np.sum(fc_weights_x_features[i])
    #
    #     # Derivatives for ffc interactions:
    #     if PV.FFC in self.prediction_terms:
    #         # Get weights for ffc interaction term and reshape so that there is one row per control_signal
    #         #    containing the terms for the interaction of that control_signal with each of the feature interactions
    #         ffc_weights = prediction_weights[idx.ffc].reshape(num_c, prediction_vector.num_ff_elems)
    #         ffc_weights_x_ff = ffc_weights * prediction_vector.ff.reshape(-1)
    #         for i in range(num_c):
    #             gradient_constants[i] += np.sum(ffc_weights_x_ff[i])
    #
    #     # TEST PRINT:
    #     print(
    #             '\nprediction_weights: ', prediction_weights,
    #           )
    #     self.test_print(prediction_vector)
    #     # TEST PRINT END:
    #
    #     # Perform gradient ascent on d(control_signals)/dEVC until convergence criterion is reached
    #     j=0
    #     while convergence_metric > self.convergence_threshold:
    #         # initialize gradient arrray (one gradient for each control signal)
    #         gradient = np.copy(gradient_constants)
    #
    #         for i, control_signal_value in enumerate(control_signal_values):
    #
    #             # Derivative of cc interaction term with respect to current control_signal_value
    #             if PV.CC in self.prediction_terms:
    #                 gradient[i] += prediction_vector._partial_derivative(PV.CC, prediction_weights, i,
    #                                                                      control_signal_value)
    #
    #             # Derivative of ffcc interaction term with respect to current control_signal_value
    #             if PV.FFCC in self.prediction_terms:
    #                 gradient[i] += prediction_vector._partial_derivative(PV.FFCC, prediction_weights, i,
    #                                                                      control_signal_value)
    #
    #             # Derivative for costs (since costs depend on control_signals)
    #             if PV.COST in self.prediction_terms:
    #                 cost_function_derivative = control_signals[i].intensity_cost_function.__self__.derivative
    #                 gradient[i] += np.sum(cost_function_derivative(control_signal_value) * cost_weights[i])
    #
    #             # Update control_signal_value with gradient
    #             control_signal_values[i] = control_signal_value + self.step_size * gradient[i]
    #
    #             # Update cost based on new control_signal_value
    #             costs[i] = control_signals[i].intensity_cost_function(control_signal_value)
    #
    #         # Only updatre terms with control_signal in them
    #         terms = [term for term in self.prediction_terms if 'c' in term.value]
    #         prediction_vector._update(self.feature_values, control_signal_values, costs, terms)
    #
    #         # Compute current LVOC using current feature_predictors, weights and new control signals
    #         current_lvoc = self.compute_lvoc(pv, prediction_weights)
    #
    #         if self.convergence_criterion == LVOC:
    #             # Compute convergence metric with updated control signals
    #             convergence_metric = np.abs(current_lvoc - previous_value)
    #         else:
    #             convergence_metric = np.max(np.abs(np.array(control_signal_values) -
    #                                                np.array(prev_control_signal_values)))
    #
    #         # TEST PRINT:
    #         print(
    #                 '\niteration {}-{}'.format(self.current_execution_count-1, j),
    #                 '\nprevious_value: ', previous_value,
    #                 '\ncurrent_lvoc: ',current_lvoc ,
    #                 '\nconvergence_metric: ',convergence_metric,
    #         )
    #         self.test_print(prediction_vector)
    #         # TEST PRINT END
    #
    #         j+=1
    #         if j > self.max_iterations:
    #             warnings.warn("{} failed to converge after {} iterations".format(self.name, self.max_iterations))
    #             break
    #
    #         previous_value = current_lvoc
    #         prev_control_signal_values = control_signal_values
    #
    #     return control_signal_values
    #
    # def _partial_derivative(self, term_label, pw, ctl_idx, ctl_val):
    #     '''Compute derivative of interaction (term) for prediction vector (pv) and prediction_weights (pw)
    #     with respect to control_signal i'''
    #
    #     # Get label and value of control signal with respect to which the derivative is being taken
    #     ctl_label = self.prediction_vector.labels.c[ctl_idx]
    #
    #     # Get labels and values of terms, and weights
    #     t_labels = getattr(self.prediction_vector.labels, term_label.value)
    #     terms = getattr(self.prediction_vector, term_label.value)
    #     wts_idx = getattr(self.prediction_vector.idx, term_label.value)
    #     # Reshape weights to match termss
    #     weights = pw[wts_idx].reshape(np.array(terms).shape)
    #
    #     gradient = 0
    #
    #     # Compute derivative for terms that contain control signal
    #     for t_label, term, wts in zip(t_labels,terms,weights):
    #         if ctl_label in t_label:
    #             gradient += np.sum((term/ctl_val)*wts)
    #
    #     return gradient
