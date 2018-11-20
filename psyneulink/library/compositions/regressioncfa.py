# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************  RegressionCFA ***************************************************

"""

RegressionCFA
^^^^^^^^^^^^^

A `CompositionFunctionApproximator` is an abstract subclass of `Composition` that parameterizes itself over trials to predict the
`net_outcome <ControlMechanism.net_outcome>` for a `Composition` (or part of one) controlled by an
`OptimizationControlMechanism`.

It uses its `adapt <CompositionFunctionApproximator.adapt>` method to parameterize itself, based on `feature_values
<OptimizationControlMechanism.feature_values>`, a `control_allocation <ControlMechanism.control_allocation>`,
and a `net_outcome <ControlMechanism.net_outcome>` passed to it from the OptimizationControlMechanism.

It uses its `evaluate <CompositionFunctionApproximator.evaluate>` method to generate and return a predicted `net_outcome
<ControlMechanism.net_outcome>` for a given set of `feature_values  <OptimizationControlMechanism.feature_values>`
and `control_allocation <ControlMechanism.control_allocation>`.

COMMENT:
.. note::
  The ModelFreeOptimizationControlMechanism_Feature_Predictor's `function_approximator
  <ModelFreeOptimizationControlMechanism.function_approximator>` is provided the `feature_values
  <ModelFreeOptimizationControlMechanism.feature_values>` and `net_outcome <ControlMechanism.net_outcome>` from the
  *previous* trial to update its parameters.  Those are then used to estimate
  (and implement) the `control_allocation <ControlMechanism.control_allocation>` that is predicted to generate the
  greatest `EVC <ModelFreeOptimizationControlMechanism_EVC>` based on the `feature_values
  <ModelFreeOptimizationControlMechanism.feature_values>` for the current trial.
COMMENT

"""
from itertools import product
import typecheck as tc
# from aenum import AutoNumberEnum, auto
from enum import Enum

import numpy as np

from psyneulink.core.components.functions.function import BayesGLM
from psyneulink.core.components.states.state import _parse_state_spec
from psyneulink.core.components.states.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.globals.keywords import DEFAULT_VARIABLE, ALL, CONTROL_SIGNALS, VARIABLE
from psyneulink.core.globals.utilities import powerset, tensor_power
from psyneulink.core.compositions.compositionfunctionapproximator import \
    CompositionFunctionApproximator

__all__ = ['PREDICTION_TERMS', 'PV', 'RegressionCFA']

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


class RegressionCFAError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class RegressionCFA(CompositionFunctionApproximator):
    '''Parameterizes a `parameterization_function <CompositionFunctionApproximator.parameterization_function>` to predict an
    outcome from an input.

    The input is represented in the `vector <PredictionVector.vector>` attribute of a `PredictionVector`
    (assigned to the CompositionFunctionApproximator`s `prediction_vector <CompositionFunctionApproximator.prediction_vector>`) attribute,
    and the `evaluate <CompositionFunctionApproximator.evaluate>` method is used to predict the outcome from the
    prediction_vector.

    When used with an OptimizationControlMechanism, the input is the ModelBasedOptimizationControlMechanism's
    `control_allocation <ControlMechanism_control_allocation>` (assigned to the variable field of the prediction_vector)
    and its `feature_values <ModelBasedOptimizationControlMechanism.feature_values>` assigned to the
    features field of the prediction_vector).  The prediction vector may also contain fields for the `costs
    ControlMechanism.costs` associated with the `control_allocation <ControlMechanism.control_allocation>` and
    for interactions among those terms.

    [Placeholder for Composition with learning]

    '''
    def __init__(self,
                 name=None,
                 parameterization_function=BayesGLM,
                 prediction_terms:tc.optional(list)=None):
        '''

        Arguments
        ---------

        owner : ModelFreeOptimizationControlMechanism : default None
            ModelFreeOptimizationControlMechanism to which the CompositionFunctionApproximator is assiged.

        parameterization_function : LearningFunction, function or method : default BayesGLM
            used to parameterize the CompositionFunctionApproximator.  It must take a 2d array as its first argument,
            the first item of which is an array the same length of the `vector <PredictionVector.prediction_vector>`
            attribute of its `prediction_vector <CompositionFunctionApproximator.prediction_vector>`, and the second item a
            1d array containing a scalar value that it tries predict.

        prediction_terms : List[PV] : default [PV.F, PV.C, PV.COST]
            terms to be included in (and thereby determines the length of) the `vector
            <PredictionVector.prediction_vector>` attribute of the  `prediction_vector
            <CompositionFunctionApproximator.prediction_vector>`;  items are members of the `PV` enum; the default is [`F
            <PV.F>`, `C <PV.C>` `FC <PV.FC>`, `COST <PV.COST>`].  if `None` is specified, the default
            values will automatically be assigned.

        Attributes
        ----------

        parameterization_function : LearningFunction, function or method
            used to parameterize the CompositionFunctionApproximator;  its result is assigned as the
            `prediction_weights <CompositionFunctionApproximator.prediction_weights>` attribute.

        prediction_terms : List[PV]
            terms included in `vector <PredictionVector.prediction_vector>` attribute of the
            `prediction_vector <CompositionFunctionApproximator.prediction_vector>`;  items are members of the `PV` enum; the
            default is [`F <PV.F>`, `C <PV.C>` `FC <PV.FC>`, `COST <PV.COST>`].

        prediction_vector : PredictionVector
            represents and manages values in its `vector <PredictionVector.vector>` attribute that are used by
            `evaluate <CompositionFunctionApproximator.evaluate>`, along with `prediction_weights
            <CompositionFunctionApproximator.prediction_weights>` to make its prediction.  The values contained in
            the `vector <PredictionVector.vector>` attribute are determined by `prediction_terms
            <CompositionFunctionApproximator.prediction_terms>`.

        prediction_weights : 1d array
            result of `parameterization_function <CompositionFunctionApproximator.parameterization_function>, used by
            `evaluate <CompositionFunctionApproximator.evaluate>` method to generate its prediction.
        '''

        self.parameterization_function = parameterization_function
        self._instantiate_prediction_terms(prediction_terms)

        super().__init__(name=name)

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
                    raise RegressionCFAError("{} specified in {} arg of {} is not a member of the {} enum".
                                                    format(repr(term.name),repr(PREDICTION_TERMS),
                                                           self.__class__.__name__, PV.__name__))
        else:
            self.prediction_terms = [PV.F,PV.C,PV.COST]

    # def initialize(self, owner):
    def initialize(self, features_array, control_signals):
        '''Assign owner and instantiate `prediction_vector <CompositionFunctionApproximator.prediction_vector>`

        Must be called before CompositionFunctionApproximator's methods can be used if its `owner <CompositionFunctionApproximator.owner>`
        was not specified in its constructor.
        '''

        prediction_terms = self.prediction_terms
        self.prediction_vector = self.PredictionVector(features_array, control_signals, prediction_terms)

        # Assign parameters to parameterization_function
        parameterization_function_default_variable = [self.prediction_vector.vector, np.zeros(1)]
        if isinstance(self.parameterization_function, type):
            self.parameterization_function = \
                self.parameterization_function(default_variable=parameterization_function_default_variable)
        else:
            self.parameterization_function.reinitialize({DEFAULT_VARIABLE:
                                                             parameterization_function_default_variable})

    def adapt(self, feature_values, control_allocation, outcome):

        try:
            # Update prediction_weights
            self.prediction_weights = self.parameterization_function.function([self._previous_state, outcome])

            # Update vector with current feature_values and control_allocation and store for next trial
            self.prediction_vector.update_vector(control_allocation, feature_values, control_allocation)
            self._previous_state = self.prediction_vector.vector

        except AttributeError:
            # Initialize vector and control_signals on first trial
            # Note:  initialize vector to 1's so that learning_function returns specified priors
            # FIX: 11/9/19 LOCALLY MANAGE STATEFULNESS OF ControlSignals AND costs
            self.prediction_vector.reference_variable = control_allocation
            self._previous_state = np.full_like(self.prediction_vector.vector, 0)
            self.prediction_weights = self.parameterization_function.function([self._previous_state, 0])

    # FIX: RENAME AS _EXECUTE_AS_REP ONCE SAME IS DONE FOR COMPOSITION
    # def evaluate(self, control_allocation, num_samples, reinitialize_values, feature_values, context):
    def evaluate(self, feature_values, control_allocation, num_estimates, context):
        '''Update prediction_vector <CompositionFunctionApproximator.prediction_vector>`, then multiply by prediction_weights.

        Uses the current values of `prediction_weights <CompositionFunctionApproximator.prediction_weights>` together with
        values of **control_allocation** and **feature_values** arguments to generate a predicted outcome.

        .. note::
            If this method is assigned as the `objective_funtion of a `GradientOptimization` `Function`,
            it is differentiated using `autograd <https://github.com/HIPS/autograd>`_\\.grad().
        '''

        predicted_outcome=0
        for i in range(num_estimates):
            terms = self.prediction_terms
            vector = self.prediction_vector.compute_terms(control_allocation )
            weights = self.prediction_weights
            net_outcome = 0

            for term_label, term_value in vector.items():
                if term_label in terms:
                    pv_enum_val = term_label.value
                    item_idx = self.prediction_vector.idx[pv_enum_val]
                    net_outcome += np.sum(term_value.reshape(-1) * weights[item_idx])
            predicted_outcome+=net_outcome
        predicted_outcome/=num_estimates
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

            # Get variable for control_signals specified in constructor
            control_allocation = []
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
                control_allocation.append(v)
            self.control_signal_functions = [c.function for c in control_signals]
            self._compute_costs = [c.compute_costs for c in control_signals]

            def get_intrxn_labels(x):
                return list([s for s in powerset(x) if len(s)>1])

            def error_for_too_few_terms(term):
                spec_type = {'FF':'feature_predictors', 'CC':'control_signals'}
                raise RegressionCFAError("Specification of {} for {} arg of {} "
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
            self.terms[C] = c = np.array([[0]] * len(control_allocation))
            self.num[C] = len(c)
            self.num_elems[C] = len(c.reshape(-1))
            self.labels[C] = ['c'+str(i) for i in range(0,len(control_allocation))]

            # Costs
            # Placemarker until control_signals are instantiated
            self.terms[COST] = cst = np.array([[0]] * len(control_allocation))
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
            <CompositionFunctionApproximator.prediction_vector>`.

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

        def compute_terms(self, control_allocation, ref_variables=None):
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
            c = [None] * len(control_allocation)
            for i, var in enumerate(control_allocation):
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
