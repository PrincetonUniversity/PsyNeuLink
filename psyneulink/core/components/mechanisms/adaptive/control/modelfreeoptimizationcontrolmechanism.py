# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **********************************  ModelFreeOptimizationControlMechanism ********************************************

"""

Overview
--------

A ModelFreeOptimizationControlMechanism is an `OptimizationControlMechanism` that uses a `FunctionApproximator`
to regulate its `ControlSignals <ControlSignal>` in order to optimize the processing of the `Components` monitored
by the ModelFreeOptimizationControlMechanism's `objective_mechanism
<ModelFreeOptimizationControlMechanism.objective_mechanism>`, as evaluated by its `evaluation_function
<ModelFreeOptimizationControlMechanism.evaluation_function>`.  An example of its use is the Learned Value of Control
model described in `Leider et al. <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006043&rev
=2>`_, which learns to select the value for its `control_signals <ControlMechanism.control_signals>` (i.e.,
its `allocation_policy <ControlMechanism.allocation_policy>`) that maximzes its `EVC <OptimizationControlMechanism_EVC>`
based on a set of `predictors <ModelFreeOptimizationControlMechanism_Feature_Predictors>`.

.. _ModelFreeOptimizationControlMechanism_EVC:

*ModelFreeOptimizationControlMechanism and the Expected Value of Control (EVC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `Expected Value of Control (EVC) <OptimizationControlMechanism_EVC>` is the predicted value of executing the
`Components` montiored by the ModelFreeOptimizationControlMechanism's `objective_mechanism
<ModelFreeOptimizationControlMechanism.objective_mechanism>`, and executed under a given `allocation_policy
<ControlMechanism.allocation_policy>`, as determined by its `evaluation_function 
<OptimizationControlMechanism_EVC.evaluaton_function>`.

The ModelFreeOptimizationControlMechanism's `function_approximator 
<ModelFreeOptimizationControlMechanism.function_approximator>` is parameterized, over `trials <trial>` to estimate the 
`EVC <OptimizationControlMechanism_EVC>` from its current `feature_values
<ModelFreeOptimizationControlMechanism.features_values>` (comprised of `feature_predictors
<ModelFreeOptimizationControlMechanism.feature_predictors>`), an `allocation_policy
<ControlMechanism.allocation_policy>`, interactions among these, and the `costs <ControlMechanism.costs> of its 
`control_signals <ControlMechanism.control_signals>`, by learning to predict the `net_outcome 
<ControlMechanism.net_outcome>` of processing for experienced values of those factors.


.. _ModelFreeOptimizationControlMechanism_Creation:

Creating a ModelFreeOptimizationControlMechanism
-------------------------------------------------

 It can be created in the same was as any `OptimizationControlMechanism`.  The following arguments
 of its constructor are specific to the ModelFreeOptimizationControlMechanism:

  * **feature_predictors** -- takes the place of the standard **input_states** argument in the constructor for a
    Mechanism`, and specifies the inputs that it learns to use to determine its `allocation_policy
    <ControlMechanism.allocation_policy>` in each `trial` of execution.
    It can be specified using any of the following, singly or combined in a list:

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

    Feature_predictors can also be added to an existing ModelFreeOptimizationControlMechanism using its `add_features`
    method.

  * **feature_function** -- specifies `function <InputState>` of the InputState created for each item listed in
    **feature_predictors**.  By default, this is the identity function, that provides the current value of the feature
    to the ModelFreeOptimizationControlMechanism's `function_approximator
    <ModelFreeOptimizationControlMechanism.function_approximator>`.  However, other functions can be assigned,
    for example to maintain a record of past values, or integrate them over trials.

  * **function_approximator** -- this specifies the `FunctionApproximator` that is parameterized over trials to
    predict the `EVC <ModelFreeOptimizationControlMechanism_EVC>` for a given `allocation_policy
    <ControlMechanism.allocation_policy>` from the terms specified in its **prediction_terms** argument.
    

.. _ModelFreeOptimizationControlMechanism_Structure:

Structure
---------

Same as an `OptimizationControlMechanism`, with the following exceptions.

.. _ModelFreeOptimizationControlMechanism_Input:

*Input*
~~~~~~~

Like any `ControlMechanism`, an ModelFreeOptimizationControlMechanism has a `primary InputState <InputState_Primary>`
named *OUTCOME* that receives a `Projection` from the *OUTCOME* `OutputState` of its `objective_mechanism
<ControlMechanism.objective_mechanism>`. However, it also has an additional InputState for each of its
feature_predictors, as described below.

.. _ModelFreeOptimizationControlMechanism_Feature_Predictors:

Feature Predictors
^^^^^^^^^^^^^^^^^^

Features_Predictors, together with the ModelFreeOptimizationControlMechanism's `control_signals
<ControlMechanism.control_signals>` and `costs <ControlMechanism.costs>` are assigned to its `feature_values
<ModelFreeOptimizationControlMechanism.feature_values>` attribute, that is `function_approximator
<ModelFreeOptimizationControlMechanism>.function_approximator` uses to predict the `EVC
<ModelFreeOptimizationControlMechanism_EVC>`.

Feature_Predictors can be of two types:

* *Input Feature Predictor* -- this is a value received as input by an `ORIGIN` Mechanism in the `Composition`.
    These are specified in the **feature_predictors** argument of the ModelFreeOptimizationControlMechanism's
    constructor (see `ModelFreeOptimizationControlMechanism_Creation`), in a dictionary containing a
    *SHADOW_EXTERNAL_INPUTS* entry, the value of which is one or more `ORIGIN` Mechanisms and/or their `InputStates
    <InputState>` to be shadowed.  For each, a `Projection` is automatically created that parallels ("shadows") the
    Projection from the Composition's `InputCIM` to the `ORIGIN` Mechanism, projecting from the same `OutputState` of
    the InputCIM to the InputState of the ModelFreeOptimizationControlMechanism assigned to that feature_predictor.

* *Output Feature Predictor* -- this is the `value <OutputState.value>` of an OutputState of some other Mechanism
    in the Composition.  These too are specified in the **feature_predictors** argument of the
    ModelFreeOptimizationControlMechanism's constructor (see `ModelFreeOptimizationControlMechanism_Creation`), and
    each is assigned a Projection from the specified OutputState(s) to the InputState of the
    ModelFreeOptimizationControlMechanism for that feature.

The current `values <InputState.value>` of the InputStates for the feature_predictors are listed in the 
`feature_values <ModelFreeOptimizationControlMechanism.feature_values>` attribute.

*Functions*
~~~~~~~~~~~

.. _ModelFreeOptimizationControlMechanism_Function_Approximator:

Function Approximator
^^^^^^^^^^^^^^^^^^^^^

The `function_approximator <ModelFreeOptimizationControlMechanism.function_approximator>` attribute of a
ModelFreeOptimizationControlMechanism is a `FunctionApproximator` that it parameterizes over trials (using the
FunctionApproximator's `parameterization_function <FunctionApproximator.parameterization_function>`) to predict
the `net_outcome <ControlMechanism.net_outcome>` of processing of the Components monitored by
its `objective_mechanism <ModelFreeOptimizationControlMechanism.objective_mechanism>`, from
combinations of `feature_values <ModelFreeOptimizationControlMechanism.feature_values>` and `allocation_policy
<ControlMechanism.allocation_policy>` it has experienced.  By default, the `parameterization_function
<FunctionApproximator.parameterization_function>` is `BayesGLM`. However, any function can be used that accepts a 2d
array, the first item of which is an array of scalar values (the prediction terms) and the second that is a scalar
value (the outcome to be predicted), and returns an array with the same shape as the first item.

The FunctionApproximator's `make_prediction <FunctionApproximator.make_prediction>` function uses the current
parameters of the `FunctionApproximator` to predict the `net_outcome <ControlMechanism.net_outcome>` of processing for
a sample `allocation_policy <ControlMechanism.allocation_policy>`, given the current `feature_values
<ModelFreeOptimizationControlMechanism.feature_values>` and the `costs <ControlMechanism.costs>` of the
`allocation_policy <ControlMechanism.allocation_policy>`.

.. note::
  The ModelFreeOptimizationControlMechanism_Feature_Predictor's `function_approximator
  <ModelFreeOptimizationControlMechanism.function_approximator>` is provided the `feature_values
  <ModelFreeOptimizationControlMechanism.feature_values>` and `net_outcome <ControlMechanism.net_outcome>` from the
  *previous* trial to update its parameters.  Those are then used to estimate
  (and implement) the `allocation_policy <ControlMechanism.allocation_policy>` that is predicted to generate the
  greatest `EVC <ModelFreeOptimizationControlMechanism_EVC>` based on the `feature_values
  <ModelFreeOptimizationControlMechanism.feature_values>` for the current trial.

.. _ModelFreeOptimizationControlMechanism_Function:

*Function*
^^^^^^^^^^

The `function <ModelFreeOptimizationControlMechanism.function>` of a ModelFreeOptimizationControlMechanism uses the
`make_prediction <FunctionApproximator.make_prediction>` method of its `function_approximator
<ModelFreeOptimizationControlMechanism.function_approximator>`, to find the `allocation_policy
<ControlMechanism.allocation_policy>` that yields the greatest predicted `EVC
<ModelFreeOptimizationControlMechanism_EVC>` given the current `feature_values
<ModelFreeOptimizationControlMechanism.feature_values>`. The default for `function
<ModelFreeOptimizationControlMechanism.function>` is the `GradientOptimization` Function, which uses
gradient ascent to select samples of `allocation_policy <ControlMechanism.allocation_policy>` that yield
progessively better values of `EVC <ModelFreeOptimizationControlMechanism_EVC>`. However, any `OptimizationFunction`
can be used in its place.  A custom function can also be used, however it must meet the requirements for the `function
<OptimizationControlMechanism.function>` of an `OptimizationControlFunction`, as described `here
<OptimizationControlMechanism_Custom_Funtion>`.

.. _ModelFreeOptimizationControlMechanism_Execution:

Execution
---------

When a ModelFreeOptimizationControlMechanism is executed, its `function
<ModelFreeOptimizationControlMechanism.function>` first calls its `function_approximator
<ModelFreeOptimizationControlMechanism.function_approximator>` to update its parameters based on the
`feature_values <ModelFreeOptimizationControlMechanism.feature_values>` and `net_outcome
<ControlMechanism.net_outcome>` of the last `trial`.  It then uses the `make_prediction
<FunctionApproximator.make_prediction>` of the `function_approximator
<ModelFreeOptimizationControlMechanism.function_approximator>` to find the `allocadtion_policy
<ControlMechanism.allocation_policy>` that predicts the greatet `EVC <ModelFreeOptimizationControlMechanism_EVC>`
for the current `feature_values <ModelFreeOptimizationControlMechanism.feature_values>`, and implements that for the
next `trial` of execution.  Specifically, it executes the following steps:
The values in the `allocation_policy <ControlMechanism.allocation_policy>` returned by `function
<ModelFreeOptimizationControlMechanism.function>` are assigned as the `variables <ControlSignal.variables>` of its
`control_signals <ControlMechanism.control_signals>`, from which they compute their `values <ControlSignal.value>`.

COMMENT:
.. _ModelFreeOptimizationControlMechanism_Examples:

Example
-------
COMMENT

.. _ModelFreeOptimizationControlMechanism_Class_Reference:

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
    ModulationParam, _is_modulation_param, BayesGLM, is_function_type, GradientOptimization
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
    DEFAULT_VARIABLE, INTERNAL_ONLY, PARAMS, NAME, \
    PARAMETER_STATES, VARIABLE, OBJECTIVE_MECHANISM, OUTCOME, FUNCTION, ALL, CONTROL_SIGNALS, \
    MODEL_FREE_OPTIMIZATION_CONTROL_MECHANISM
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.defaults import defaultControlAllocation
from psyneulink.core.globals.utilities import is_iterable, powerset, tensor_power

__all__ = [
    'ModelFreeOptimizationControlMechanism', 'SHADOW_EXTERNAL_INPUTS', 'PREDICTION_TERMS', 'PV', 'FunctionApproximator'
]


FEATURE_PREDICTORS = 'feature_predictors'
SHADOW_EXTERNAL_INPUTS = 'SHADOW_EXTERNAL_INPUTS'
PREDICTION_WEIGHTS = 'PREDICTION_WEIGHTS'
PREDICTION_TERMS = 'prediction_terms'
PREDICTION_WEIGHT_PRIORS = 'prediction_weight_priors'


class PV(Enum):
# class PV(AutoNumberEnum):
    '''PV()
    Specifies terms used to compute `vector <PredictionVector.vector>` attribute of `PredictionVector`.

    Attributes
    ----------

    F
        Main effect of `feature_predictors <ModelFreeOptimizationControlMechanism_Feature_Predictors>`.
    C
        Main effect of `values <ControlSignal.value>` of `control_signals <ControlMechanism.control_signals>`.
    FF
        Interaction among `feature_predictors <ModelFreeOptimizationControlMechanism_Feature_Predictors>`.
    CC
        Interaction among `values <ControlSignal.value>` of `control_signals <ControlMechanism.control_signals>`.
    FC
        Interaction between `feature_predictors <ModelFreeOptimizationControlMechanism_Feature_Predictors>` and
        `values <ControlSignal.value>` of `control_signals <ControlMechanism.control_signals>`.
    FFC
        Interaction between interactions of `feature_predictors
        <ModelFreeOptimizationControlMechanism_Feature_Predictors>` and `values <ControlSignal.value>` of
        `control_signals <ControlMechanism.control_signals>`.
    FCC
        Interaction between `feature_predictors <ModelFreeOptimizationControlMechanism_Feature_Predictors>` and
        interactions among `values <ControlSignal.value>` of `control_signals <ControlMechanism.control_signals>`.
    FFCC
        Interaction between interactions of `feature_predictors
        <ModelFreeOptimizationControlMechanism_Feature_Predictors>` and interactions among `values
        <ControlSignal.value>` of `control_signals <ControlMechanism.control_signals>`.
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


class FunctionApproximator():
    '''Parameterizes a `parameterization_function <FunctionApproximator.parameterization_function>` to predict an
    outcome from an input.

    The input is represented in the `vector <PredictionVector.vector>` attribute of a `PredictionVector`
    (assigned to the FunctionApproximator`s `prediction_vector <FunctionApproximator.prediction_vector>`) attribute,
    and the `make_prediction <FunctionApproximator.make_prediction>` method is used to predict the outcome from the
    prediction_vector.

    When used with a ModelBasedOptimizationControlMechanism, the input is the ModelBasedOptimizationControlMechanism's
    `allocation_policy <ControlMechanism_allocation_policy>` (assigned to the variable field of the prediction_vector)
    and its `feature_values <ModelBasedOptimizationControlMechanism.feature_values>` assigned to the
    features field of the prediction_vector).  The prediction vector may also contain fields for the `costs
    ControlMechanism.costs` associated with the `allocation_policy <ControlMechanism.allocation_policy>` and
    for interactions among those terms.

    [Placeholder for Composition with learning]

    '''
    def __init__(self, owner=None,
                 parameterization_function=BayesGLM,
                 prediction_terms:tc.optional(list)=None):
        '''

        Arguments
        ---------

        owner : ModelFreeOptimizationControlMechanism : default None
            ModelFreeOptimizationControlMechanism to which the FunctionApproximator is assiged.

        parameterization_function : LearningFunction, function or method : default BayesGLM
            used to parameterize the FunctionApproximator.  It must take a 2d array as its first argument,
            the first item of which is an array the same length of the `vector <PredictionVector.prediction_vector>`
            attribute of its `prediction_vector <FunctionApproximator.prediction_vector>`, and the second item a
            1d array containing a scalar value that it tries predict.

        prediction_terms : List[PV] : default [PV.F, PV.C, PV.COST]
            terms to be included in (and thereby determines the length of) the `vector
            <PredictionVector.prediction_vector>` attribute of the  `prediction_vector
            <FunctionApproximator.prediction_vector>`;  items are members of the `PV` enum; the default is [`F
            <PV.F>`, `C <PV.C>` `FC <PV.FC>`, `COST <PV.COST>`].  if `None` is specified, the default
            values will automatically be assigned.

        Attributes
        ----------

        owner : ModelFreeOptimizationControlMechanism
            `ModelFreeOptimizationControlMechanism` to which the `FunctionApproximator` belongs;  assigned as the
            `objective_function <OptimizationFunction.objective_function>` parameter of the `OptimizationFunction`
            assigned to that Mechanism's `function <ModelFreeOptimizationControlMechanism.function>`.

        parameterization_function : LearningFunction, function or method
            used to parameterize the FunctionApproximator;  its result is assigned as the
            `prediction_weights <FunctionApproximator.prediction_weights>` attribute.

        prediction_terms : List[PV]
            terms included in `vector <PredictionVector.prediction_vector>` attribute of the
            `prediction_vector <FunctionApproximator.prediction_vector>`;  items are members of the `PV` enum; the
            default is [`F <PV.F>`, `C <PV.C>` `FC <PV.FC>`, `COST <PV.COST>`].

        prediction_vector : PredictionVector
            represents and manages values in its `vector <PredictionVector.vector>` attribute that are used by
            `make_prediction <FunctionApproximator.make_prediction>`, along with `prediction_weights
            <FunctionApproximator.prediction_weights>` to make its prediction.  The values contained in
            the `vector <PredictionVector.vector>` attribute are determined by `prediction_terms
            <FunctionApproximator.prediction_terms>`.

        prediction_weights : 1d array
            result of `parameterization_function <FunctionApproximator.parameterization_function>, used by
            `make_prediction <FunctionApproximator.make_prediction>` method to generate its prediction.
        '''

        self.parameterization_function = parameterization_function
        self._instantiate_prediction_terms(prediction_terms)
        if owner:
            self.initialize(owner=owner)

    def _instantiate_prediction_terms(self, prediction_terms):

        # # MODIFIED 11/9/18 OLD:
        # prediction_terms = prediction_terms or [PV.F,PV.C,PV.FC, PV.COST]
        # if ALL in prediction_terms:
        #     prediction_terms = list(PV.__members__.values())
        # MODIFIED 11/9/18 NEW: [JDC]
        # FIX: FOR SOME REASON prediction_terms ARE NOT GETTING PASSED INTACT (ARE NOT RECOGNIZED IN AS MEMBERS OF PV)
        #      AND SO THEY'RE FAILING IN _validate_params
        #      EVEN THOUGH THEY ARE FINE UNDER EXACTLY THE SAME CONDITIONS IN LVOCCONTROLMECHANISM
        #      THIS IS A HACK TO FIX THE PROBLEM:
        if prediction_terms:
            if ALL in prediction_terms:
                self.prediction_terms = list(PV.__members__.values())
            else:
                terms = prediction_terms.copy()
                self.prediction_terms = []
                for term in terms:
                    self.prediction_terms.append(PV[term.name])
        # MODIFIED 11/9/18 END
            for term in self.prediction_terms:
                if not term in PV:
                    raise ModelFreeOptimizationControlMechanismError("{} specified in {} arg of {} "
                                                                     "is not a member of the {} enum".
                                                                     format(repr(term.name),
                                                                            repr(PREDICTION_TERMS),
                                                                            self.__class__.__name__, PV.__name__))
        else:
            self.prediction_terms = [PV.F,PV.C,PV.COST]

    def initialize(self, owner):
        '''Assign owner and instantiate `prediction_vector <FunctionApproximator.prediction_vector>`

        Must be called before FunctionApproximator's methods can be used if its `owner <FunctionApproximator.owner>`
        was not specified in its constructor.
        '''

        self.owner = owner
        self.prediction_vector = self.PredictionVector(self.owner.feature_values,
                                                       self.owner.control_signals,
                                                       self.prediction_terms)
        # Assign parameters to parameterization_function
        parameterization_function_default_variable = [self.prediction_vector.vector, np.zeros(1)]
        if isinstance(self.parameterization_function, type):
            self.parameterization_function = \
                self.parameterization_function(default_variable=parameterization_function_default_variable)
        else:
            self.parameterization_function.reinitialize({DEFAULT_VARIABLE:
                                                             parameterization_function_default_variable})

    def before_execution(self, context):
        '''Call `parameterization_function <FunctionApproximator.parameterization_function>` prior to calls to
        `make_prediction <FunctionApproximator.make_prediction>`.'''

        feature_values = np.array(np.array(self.owner.variable[1:]).tolist())
        try:
            # Update prediction_weights
            variable = self.owner.value
            outcome = self.owner.net_outcome
            self.prediction_weights = self.parameterization_function.function([self._previous_state,
                                                                               outcome])
            # Update vector with owner's current variable and feature_values and  and store for next trial
            # Note: self.owner.value = last allocation_policy used by owner
            self.prediction_vector.update_vector(variable, feature_values, variable)
            self._previous_state = self.prediction_vector.vector
        except AttributeError:
            # Initialize vector and control_signals on first trial
            # Note:  initialize vector to 1's so that learning_function returns specified priors
            # FIX: 11/9/19 LOCALLY MANAGE STATEFULNESS OF ControlSignals AND costs
            self.prediction_vector.reference_variable = self.owner.allocation_policy
            self._previous_state = np.full_like(self.prediction_vector.vector, 0)
            self.prediction_weights = self.parameterization_function.function([self._previous_state, 0])
        return feature_values

    # def make_prediction(self, allocation_policy, num_samples, reinitialize_values, feature_values, context):
    def make_prediction(self, variable, num_samples, feature_values, context):
        '''Update terms of prediction_vector <FunctionApproximator.prediction_vector>` and then multiply by
        prediction_weights.

        Uses the current values of `prediction_weights <FunctionApproximator.prediction_weights>`
        together with values of **variable** and **feature_values** arguments to generate a predicted outcome.

        .. note::
            If this method is assigned as the `objective_funtion of a `GradientOptimization` `Function`,
            it is differentiated using `autograd <https://github.com/HIPS/autograd>`_\\.grad().
        '''

        predicted_outcome=0
        for i in range(num_samples):
            terms = self.prediction_terms
            vector = self.prediction_vector.compute_terms(variable )
            weights = self.prediction_weights
            net_outcome = 0

            for term_label, term_value in vector.items():
                if term_label in terms:
                    pv_enum_val = term_label.value
                    item_idx = self.prediction_vector.idx[pv_enum_val]
                    net_outcome += np.sum(term_value.reshape(-1) * weights[item_idx])
            predicted_outcome+=net_outcome
        predicted_outcome/=num_samples
        return predicted_outcome

    def after_execution(self, context):
        pass

    class PredictionVector():
        '''Maintain a `vector <PredictionVector.vector>` of terms for a regression model specified by a list of
        `specified_terms <PredictionVector.specified_terms>`.

        Terms are maintained in lists indexed by the `PV` Enum and, in "flattened" form within fields of a 1d
        array in `vector <PredictionVector.vector>` indexed by slices listed in the `idx <PredicitionVector.idx>`
        attribute.

        Arguments
        ---------

        feature_values : 2d nparray
            arrays of features to assign as the `PV.F` term of `terms <PredictionVector.terms>`.

        control_signals : List[ControlSignal]
            list containing the `ControlSignals <ControlSignal>` of an `OptimizationControlMechanism`;
            the `variable <ControlSignal.variable>` of each is assigned as the `PV.C` term of `terms
            <PredictionVector.terms>`.

        specified_terms : List[PV]
            terms to include in `vector <PredictionVector.vector>`; entries must be members of the `PV` Enum.

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
                        assert False, "PROGRAM ERROR: unrecognized specification for {} arg of {}: {}".\
                                                      format(repr(CONTROL_SIGNALS), self.name, c)
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
                raise ModelFreeOptimizationControlMechanismError("Specification of {} for {} arg of {} "
                                                                 "requires at least two {} be specified".
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
            self.num_elems[F] = len(f.reshape(-1)) # num of total elements assigned to vector
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

            # Construct "flattened" vector based on specified terms, and assign indices (as slices)
            i=0
            for t in range(len(PV)):
                if t in [t.value for t in specified_terms]:
                    self.idx[t] = slice(i, i + self.num_elems[t])
                    i += self.num_elems[t]

            self.vector = np.zeros(i)

        def __call__(self, terms:tc.any(PV, list))->tc.any(PV, tuple):
            '''Return subvector(s) for specified term(s)'''
            if not isinstance(terms, list):
                return self.idx[terms.value]
            else:
                return tuple([self.idx[pv_member.value] for pv_member in terms])

        # FIX: 11/9/19 LOCALLY MANAGE STATEFULNESS OF ControlSignals AND costs
        def update_vector(self, variable, feature_values=None, reference_variable=None):
            '''Update vector with flattened versions of values returned from the `compute_terms
            <PredictionVector.compute_terms>` method of the `prediction_vector
            <FunctionApproximator.prediction_vector>`.

            Updates `vector <PredictionVector.vector>` with current values of variable and, optionally,
            and feature_values.

            '''

            # FIX: 11/9/19 LOCALLY MANAGE STATEFULNESS OF ControlSignals AND costs
            if reference_variable is not None:
                self.reference_variable = reference_variable
            self.reference_variable = reference_variable

            if feature_values is not None:
                self.terms[PV.F.value] = np.array(feature_values)
            # FIX: 11/9/19 LOCALLY MANAGE STATEFULNESS OF ControlSignals AND costs
            computed_terms = self.compute_terms(np.array(variable), self.reference_variable)

            # Assign flattened versions of specified terms to vector
            for k, v in computed_terms.items():
                if k in self.specified_terms:
                    self.vector[self.idx[k.value]] = v.reshape(-1)

        def compute_terms(self, control_signal_variables, ref_variables=None):
            '''Calculate interaction terms.

            Results are returned in a dict; entries are keyed using names of terms listed in the `PV` Enum.
            Values of entries are nd arrays.
            '''

            # FIX: 11/9/19 LOCALLY MANAGE STATEFULNESS OF ControlSignals AND costs
            ref_variables = ref_variables or self.reference_variable
            self.reference_variable = ref_variables

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
                    costs[i] = -(self._compute_costs[i](val, ref_variables[i]))
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


class ModelFreeOptimizationControlMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ModelFreeOptimizationControlMechanism(OptimizationControlMechanism):
    """ModelFreeOptimizationControlMechanism(                                                  \
    feature_predictors,                                                                        \
    feature_function=None,                                                                     \
    objective_mechanism=None,                                                                  \
    origin_objective_mechanism=False,                                                          \
    terminal_objective_mechanism=False,                                                        \
    function_approximator=FunctionApproximator(parameterization_function=BayesGLM,             \
                                               prediction_terms=[PV.F, PV.C, PV.FC, PV.COST]), \
    num_samples=1,                                                                             \
    function=GradientOptimization,                                                             \
    control_signals=None,                                                                      \
    modulation=ModulationParam.MULTIPLICATIVE,                                                 \
    params=None,                                                                               \
    name=None,                                                                                 \
    prefs=None)

    Subclass of `OptimizationControlMechanism` that learns to optimize its `ControlSignals <ControlSignal>`.

    Arguments
    ---------

    feature_predictors : Mechanism, OutputState, Projection, dict, or list containing any of these
        specifies values to assign to `feature_values <ModelFreeOptimizationControlMechanism.feature_values>`,
        that are used to estimate `EVC <ModelFreeOptimizationControlMechanism_EVC>`.  Any `InputState specification
        <InputState_Specification>` can be used that resolves to an `OutputState` that projects to the InputState.
        In addition, a dictionary with a *SHADOW_EXTERNAL_INPUTS* entry can be used to shadow inputs to the
        Composition's `ORIGIN` Mechanism(s) (see `ModelFreeOptimizationControlMechanism_Creation` for details).

    feature_function : Function or function : default None
        specifies the `function <InputState.function>` for the `InputState` assigned to each `feature_predictor
        <ModelFreeOptimizationControlMechanism_Feature_Predictors>`.

    objective_mechanism : ObjectiveMechanism or List[OutputState specification] : default None
        specifies either an `ObjectiveMechanism` to use for the ModelFreeOptimizationControlMechanism, or a list of the
        `OutputState <OutputState>`\\s it should monitor; if a list of `OutputState specifications
        <ObjectiveMechanism_Monitored_Output_States>` is used, a default ObjectiveMechanism is created and the list
        is passed to its **monitored_output_states** argument.

    function_approximator : FunctionApproximator : default FunctionApproximator(parameterization_function=BayesGLM)
        specifies the FunctionApproximator that is parameterized to predict the `net_outcome
        <ControlMechanism.net_outcome>` of processing by the Components monitored by the
        ModelFreeOptimizationControlMechanism's `objective_mechanism
        <ModelFreeOptimizationControlMechanism.objective_mechanism>` from the `current
        <ModelFreeOptimizationControlMechanism.feature_values>` and `allocation_policy
        <ControlMechanism.allocation_policy>` (see `ModelFreeOptimizationControlMechanism_Function_Approximator` for
        details).

    function : OptimizationFunction, function or method : default GradientOptimization
        specifies the function used to find the `allocation_policy` that maximizes `EVC
        <ModelFreeOptimizationControlMechanism_EVC>`>`; must take as its sole argument an array with the same shape
        as `allocation_policy <ControlMechanism.allocation_policy>`, and return a similar array (see `Function
        <ModelFreeOptimizationControlMechanism_Function>` for additional details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for the
        Mechanism, its `function_approximator <ModelFreeOptimizationControlMechanism.function_approximator>`,
        and/or a custom function and its parameters.  Values specified for parameters in the dictionary override any
        assigned to those parameters in arguments of the constructor.

    name : str : default see `name <ModelFreeOptimizationControlMechanism.name>`
        specifies the name of the ModelFreeOptimizationControlMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the ModelFreeOptimizationControlMechanism;
        see `prefs <ModelFreeOptimizationControlMechanism.prefs>` for details.

    Attributes
    ----------

    feature_values : 1d ndarray
        the current `values <InputState.value>` of `feature_predictors
        ModelFreeOptimizationControlMechanism_Feature_Predictors`.

    function_approxmiator : FunctionApproximator
        used to predict `EVC <ModelFreeOptimizationControlMechanism_EVC>` for a given `feature_values
        <ModelFreeOptimizationControlMechanism.feature_values>` and `allocation_policy
        <ControlMechanism.allocation_policy>` (see `ModelFreeOptimizationControlMechanism_Function_Approximator`
        for additional details).

    function : OptimizationFunction, function or method
        takes current `allocation_policy <ControlMechanism.allocation_policy>` (as initializer) and, using the current
        `feature_values <ModelFreeOptimizationControlMechanism.feature_values>` and the `make_prediction
        <FunctionApproximator.make_prediction>` method of the `function_approximator
        <ModelFreeOptimizationControlMechanism.function_approximator>`, returns an
        `allocation_policy` that maximizes the `EVC <ModelFreeOptimizationControlMechanism_EVC>` (see `Function
        <ModelFreeOptimizationControlMechanism_Function>` for additional details).

    name : str
        the name of the ModelFreeOptimizationControlMechanism; if it is not specified in the **name** argument of the
        constructor, a default is assigned by MechanismRegistry (see `Naming` for conventions used for default and
        duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the ModelFreeOptimizationControlMechanism; if it is not specified in the **prefs**
        argument of the constructor, a default is assigned using `classPreferences` defined in __init__.py (see
        :doc:`PreferenceSet <LINK>` for details).
    """

    componentType = MODEL_FREE_OPTIMIZATION_CONTROL_MECHANISM
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
                 function_approximator:FunctionApproximator=FunctionApproximator(parameterization_function=BayesGLM),
                 function=GradientOptimization,
                 control_signals:tc.optional(tc.any(is_iterable, ParameterState, ControlSignal))=None,
                 modulation:tc.optional(_is_modulation_param)=ModulationParam.MULTIPLICATIVE,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs):

        if feature_predictors is None:
            # Included for backward compatibility
            if 'predictors' in kwargs:
                feature_predictors = kwargs['predictors']
                del(kwargs['predictors'])
            else:
                raise ModelFreeOptimizationControlMechanismError("{} arg for {} must be specified".
                                                                 format(repr(FEATURE_PREDICTORS),
                                                                        self.__class__.__name__))
        self.function_approximator = function_approximator

        if kwargs:
                for i in kwargs.keys():
                    raise ModelFreeOptimizationControlMechanismError("Unrecognized arg in constructor for {}: {}".
                                                                     format(self.__class__.__name__, repr(i)))

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(input_states=feature_predictors,
                                                  feature_function=feature_function,
                                                  origin_objective_mechanism=origin_objective_mechanism,
                                                  terminal_objective_mechanism=terminal_objective_mechanism,
                                                  params=params)

        super().__init__(objective_mechanism=objective_mechanism,
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
            raise ModelFreeOptimizationControlMechanismError("{} specified for {} ({}) must be assigned one or more {}".
                            format(ObjectiveMechanism.__name__, self.name,
                                   request_set[OBJECTIVE_MECHANISM], repr(MONITORED_OUTPUT_STATES)))

    def _instantiate_input_states(self, context=None):
        """Instantiate input_states for Projections from features and objective_mechanism.

        Inserts InputState specification for Projection from ObjectiveMechanism as first item in list of
        InputState specifications generated in _parse_feature_specs from the **feature_predictors** and
        **feature_function** arguments of the ModelFreeOptimizationControlMechanism constructor.
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
        '''Add InputStates and Projections to ModelFreeOptimizationControlMechanism for feature_predictors used to
        predict `net_outcome <ControlMechanism.net_outcome>`

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
                    raise ModelFreeOptimizationControlMechanismError("Incorrect specification ({}) "
                                                                     "in feature_predictors argument of {}."
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
        ModelFreeOptimizationControlMechanism requires use of at least one of the cost options
        '''
        control_signal = super()._instantiate_control_signal(control_signal, context)

        if control_signal.cost_options is None:
            control_signal.cost_options = ControlSignalCosts.DEFAULTS
            control_signal._instantiate_cost_attributes()
        return control_signal

    def _instantiate_attributes_after_function(self, context=None):
        '''Instantiate ModelFreeOptimizationControlMechanism's function_approximator'''
        super()._instantiate_attributes_after_function(context=context)
        self._instantiate_function_approximator()

    def _instantiate_function_approximator(self):
        '''Instantiate attributes for ModelFreeOptimizationControlMechanism's function_approximator'''

        self.feature_values = np.array(self.instance_defaults.variable[1:])

        # Assign parameters to learning_function
        if isinstance(self.function_approximator, type):
            self.function_approximator = self.function_approximator(owner=self)
        else:
            self.function_approximator.initialize(owner=self)

    def _execute(self, variable=None, runtime_params=None, context=None):
        """Find allocation_policy that optimizes EVC.

        Items of variable should be:

              - self.outcome: `value <OutputState.value>` of the *OUTCOME* OutputState of `objective_mechanism
                <ControlMechanism.objective_mechanism>`.

              - variable[n]: current value of `feature_predictor
                <ModelFreeOptimizationControlMechanism_Feature_Predictors>`

        Executes the following steps:

            - call `before_execution <FunctionApproximator.before_execution>` method of `function_approximator
              <ModelFreeOptimizationControlMechanism.function_approximator>`, which calls its `parameterization_function
              <FunctionApproximator.parameterization_function>` to update its parameters;

            - call super()._execute(), which calls the `OptimizationFunction` assigned as the
              ModelFreeOptimizationControlMechanism's `function <ModelFreeOptimizationControlMechanism.function>` that
              finds the `allocation_policy <ControlMechanism.allocation_policy>` predictive of the greatest `EVC
              <ModelFreeOptimizationControlMechanism_EVC>`;

            - call `after_execution <FunctionApproximator.after_execution>` method of `function_approximator
              <ModelFreeOptimizationControlMechanism.function_approximator>`;

            - return allocation_policy.
        """

        if (self.context.initialization_status == ContextFlags.INITIALIZING):
            return defaultControlAllocation
        assert variable == self.variable, 'PROGRAM ERROR: variable != self.variable for MFOCM'
        if self.allocation_policy is None:
            self.value = [c.instance_defaults.variable for c in self.control_signals]

        self.feature_values = self.function_approximator.before_execution(context=self.context)

        # TEST PRINT
        print ('\n------------------------------------------------')
        print ('BEFORE EXECUTION:')
        print ('\tEXECUTION COUNT: ', self.current_execution_count)
        print ('\tPREDICTION WEIGHTS', self.function_approximator.prediction_weights)
        # TEST PRINT END

        # Compute allocation_policy using ModelFreeOptimizationControlMechanism's optimization function
        # IMPLEMENTATION NOTE: skip ControlMechanism._execute since it is a stub method that returns input_values
        allocation_policy, self.net_outcome_max, self.saved_samples, self.saved_values = \
                                            super(ControlMechanism, self)._execute(variable=self.allocation_policy,
                                                                                   runtime_params=runtime_params,
                                                                                   context=context)
        # # TEST PRINT
        print ('\nAFTER EXECUTION:')
        print ('\tEXECUTION COUNT: ', self.current_execution_count)
        print ('\tALLOCATION POLICY: ', allocation_policy)
        print ('\tNET_OUTCOME MAX: ', self.net_outcome_max)
        # # TEST PRINT END

        self.function_approximator.after_execution(context=context)

        return allocation_policy

    def evaluation_function(self, allocation_policy):
        '''Compute outcome for a given allocation_policy.

        Assigned as the `objective_function <OptimizationFunction.objective_function>` parameter of the
        `ObjectiveFunction` assigned to the ModelFreeOptimizationControlMechanism's `function
        <ModelFreeOptimizationControlMechanism.function>`.

        Returns a scalar that is the predicted outcome of the `function_approximator
        <ModelFreeOptimizationControlMechanism.function_approximator>`.
        '''

        num_samples = 1
        return self.function_approximator.make_prediction(allocation_policy,
                                                          num_samples,
                                                          self.feature_values,
                                                          context=self.function_object.context)


# OLD ******************************************************************************************************************
# Manual computation of derivatives

    # def gradient_ascent(self, control_signals, current_state, prediction_weights):
    #     '''Determine the `allocation_policy <LVOCControlMechanism.allocation_policy>` that maximizes the `EVC
    #     <ModelFreeOptimizationControlMechanism_EVC>`.
    #
    #     Iterate over current_state; for each iteration: \n
    #     - compute gradients based on current control_signal values and their costs (in current_state);
    #     - compute new control_signal values based on gradients;
    #     - update current_state with new control_signal values and the interaction terms and costs based on those;
    #     - use prediction_weights and updated current_state to compute new `EVC <ModelFreeOptimizationControlMechanism_EVC>`.
    #
    #     Continue to iterate until difference between new and old EVC is less than `convergence_threshold
    #     <LearnAllocationPolicy.convergence_threshold>` or number of iterations exceeds `max_iterations
    #     <LearnAllocationPolicy.max_iterations>`.
    #
    #     Return control_signals field of current_state (used by LVOCControlMechanism as its `allocation_vector
    #     <LVOCControlMechanism.allocation_policy>`).
    #     '''
    #
    #     pv = current_state.vector
    #     idx = current_state.idx
    #     # labels = current_state.labels
    #     num_c = current_state.num_c
    #     num_cst = current_state.num_cst
    #     # num_intrxn = current_state.num_interactions
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
    #         fc_weights = prediction_weights[idx.fc].reshape(num_c, current_state.num_f_elems)
    #         fc_weights_x_features = fc_weights * feature_predictors
    #         for i in range(num_c):
    #             gradient_constants[i] += np.sum(fc_weights_x_features[i])
    #
    #     # Derivatives for ffc interactions:
    #     if PV.FFC in self.prediction_terms:
    #         # Get weights for ffc interaction term and reshape so that there is one row per control_signal
    #         #    containing the terms for the interaction of that control_signal with each of the feature interactions
    #         ffc_weights = prediction_weights[idx.ffc].reshape(num_c, current_state.num_ff_elems)
    #         ffc_weights_x_ff = ffc_weights * current_state.ff.reshape(-1)
    #         for i in range(num_c):
    #             gradient_constants[i] += np.sum(ffc_weights_x_ff[i])
    #
    #     # TEST PRINT:
    #     print(
    #             '\nprediction_weights: ', prediction_weights,
    #           )
    #     self.test_print(current_state)
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
    #                 gradient[i] += current_state._partial_derivative(PV.CC, prediction_weights, i,
    #                                                                      control_signal_value)
    #
    #             # Derivative of ffcc interaction term with respect to current control_signal_value
    #             if PV.FFCC in self.prediction_terms:
    #                 gradient[i] += current_state._partial_derivative(PV.FFCC, prediction_weights, i,
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
    #         current_state._update(self.feature_values, control_signal_values, costs, terms)
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
    #         self.test_print(current_state)
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
    #     ctl_label = self.current_state.labels.c[ctl_idx]
    #
    #     # Get labels and values of terms, and weights
    #     t_labels = getattr(self.current_state.labels, term_label.value)
    #     terms = getattr(self.current_state, term_label.value)
    #     wts_idx = getattr(self.current_state.idx, term_label.value)
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
