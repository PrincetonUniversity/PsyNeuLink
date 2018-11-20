# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ***********************************  ModelBasedOptimizationControlMechanism ******************************************

"""

Overview
--------

A ModelBasedOptimizationControlMechanism is a subclass of `OptimizationControlMechanism <OptimizationControlMechanism>`
that uses the Composition's `evaluate <Composition.evaluate>` method as the `evaluation_function
<OptimizationFunction.evaluation_function>` of its `OptimizationFunction` in order to find an optimal `control_allocation
<ControlMechanism.control_allocation>`.

.. _ModelBasedOptimizationControlMechanism_Creation:

Creating a ModelBasedOptimizationControlMechanism
----------------------------------------

A ModelBasedOptimizationControlMechanism can be created in the same was as any `ControlMechanism <ControlMechanism>`.
The only constraints are:

    (1) an `OptimizationFunction` (or one that has the same structure) must be specified as the **function** argument
        of its constructor.

    (2) the `evaluation_function <OptimizationControlMechanism.evaluation_function>` must be the
        `composition's <ModelBasedOptimizationControlMechanism.composition>` `evaluate
        <Composition.evaluate>` method.

In addition, a **learning_function** can be specified (see `Optimization Control Mechanism Learning Function
<OptimizationControlMechanism_Learning_Function>`)

.. _ModelBasedOptimizationControlMechanism_Structure:

Structure
---------

A ModelBasedOptimizationControlMechanism is based on the same structure as an `OptimizationControlMechanism
<OptimizationControlMechanism>`. A ModelBasedOptimizationControlMechanism more specifically seeks to optimize the
`net_outcome <ControlMechanism.net_outcome>` by running simulations of its `composition
<ModelBasedOptimizationControlMechanism.composition>` with various control allocation policies.

As a result, a ModelBasedOptimizationControlMechanism differs from an `OptimizationControlMechanism
<OptimizationControlMechanism>` in the following ways:

    - it has a `composition <ModelBasedOptimizationControlMechanism.composition>`, which is the `Composition` to which
      it belongs

    - its `evaluation_function <OptimizationControlMechanism.evaluation_function>` is or includes its `composition's
      <ModelBasedOptimizationControlMechanism.composition>` `evaluate <Composition.evaluate>` method

.. _ModelBasedOptimizationControlMechanism_Execution:

Execution
---------

There are three parts to the execution of a ModelBasedOptimizationControlMechanism.

First, it calls its `composition's <ModelBasedOptimizationControlMechanism.composition>`
`before_simulations <Composition.before_simulations>` method, which prepares the Composition for simulations by
updating predicted inputs and storing current values.

Next, the ModelBasedOptimizationControlMechanism goes through the standard `OptimizationControlMechanism` execution.

Finally, it calls its `composition's <ModelBasedOptimizationControlMechanism.composition>`
`after_simulations <Composition.after_simulations>` method, which reinstates the values that were stored before the
simulations.

COMMENT:
.. _ModelBasedOptimizationControlMechanism_Examples:

Example
-------
COMMENT

.. _ModelBasedOptimizationControlMechanism_Class_Reference:

Class Reference
---------------

"""
import typecheck as tc

import numpy as np

from psyneulink.core.components.functions.function import ModulationParam, _is_modulation_param, is_function_type
from psyneulink.core.components.mechanisms.adaptive.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.adaptive.control.optimizationcontrolmechanism import OptimizationControlMechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.states.parameterstate import ParameterState
from psyneulink.core.components.states.modulatorysignals.controlsignal import ControlSignal

from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.defaults import defaultControlAllocation
from psyneulink.core.globals.keywords import PARAMETER_STATES, OPTIMIZATION_CONTROL_MECHANISM, \
    MODEL_BASED_OPTIMIZATION_CONTROL_MECHANISM
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import is_iterable

__all__ = [
    'ModelBasedOptimizationControlMechanism', 'ModelBasedOptimizationControlMechanismError'
]


class ModelBasedOptimizationControlMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ModelBasedOptimizationControlMechanism(OptimizationControlMechanism):
    """ModelBasedOptimizationControlMechanism(                       \
    objective_mechanism=None,                              \
    learning_function=None,                                \
    function=None,                                         \
    search_function=None,                                  \
    search_termination_function=None,                      \
    search_space=None,                                     \
    control_signals=None,                                  \
    modulation=ModulationParam.MULTIPLICATIVE,             \
    composition=None                                       \
    params=None,                                           \
    name=None,                                             \
    prefs=None)

    Subclass of `OptimizationControlMechanism <OptimizationControlMechanism>` that uses a `evaluate
    <ModelBasedOptimizationControlMechanism.evaluate>` method in order to adjusts its `ControlSignals
    <ControlSignal>` and optimize performance of the `Composition` to which it belongs.

    Arguments
    ---------

    objective_mechanism : ObjectiveMechanism or List[OutputState specification]
        specifies either an `ObjectiveMechanism` to use for the OptimizationControlMechanism, or a list of the
        `OutputState <OutputState>`\\s it should monitor; if a list of `OutputState specifications
        <ObjectiveMechanism_Monitored_Output_States>` is used, a default ObjectiveMechanism is created and the list
        is passed to its **monitored_output_states** argument.

    learning_function : LearningFunction, function or method
        specifies a function used to learn to predict the `EVC <OptimizationControlMechanism_EVC>` from
        the current `current_state <OptimizationControlMechanism.current_state>` (see
        `OptimizationControlMechanism_Learning_Function` for details).

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

    composition : Composition : default None
        the `Composition` to which this ModelBasedOptimizationControlMechanism belongs

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for the
        Mechanism, its `learning_function <OptimizationControlMechanism.learning_function>`, and/or a custom function and its parameters.  Values
        specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default see `name <OptimizationControlMechanism.name>`
        specifies the name of the OptimizationControlMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the OptimizationControlMechanism; see `prefs
        <OptimizationControlMechanism.prefs>` for details.

    Attributes
    ----------
    current_state : 1d ndarray
        array passed to `learning_function <OptimizationControlMechanism.learning_function>` if that is implemented.

    prediction_weights : 1d ndarray
        weights assigned to each term of `current_state <OptimizationControlMechanism.current_state>`
        by `learning_function <OptimizationControlMechanism.learning_function>`.

    learning_function : LearningFunction, function or method
        takes `current_state <OptimizationControlMechanism.current_state>` as its first argument, and
        `net_outcome <ControlMechanism.net_outcome>` as its second argument, and returns an updated set of
        `prediction_weights <OptimizationControlMechanism.prediction_weights>` (see
        `OptimizationControlMechanism_Learning_Function` for additional details).

    function : OptimizationFunction, function or method
        takes current `control_allocation <ControlMechanism.control_allocation>` (as initializer),
        uses its `search_function <OptimizationFunction.search_function>` to select samples of `control_allocation
        <ControlMechanism.control_allocation>` from its `search_space <OptimizationControlMechanism.search_space>`,
        evaluates these using its `evaluation_function <OptimizationControlMechanism.evaluation_function>`, and returns
        the one that yields the greatest `EVC <OptimizationControlMechanism_EVC>`  (see `Primary Function
        <OptimizationControlMechanism_Function>` for additional details).

    composition : Composition : default None
        the `Composition` to which this ModelBasedOptimizationControlMechanism belongs

    evaluation_function : function or method
        calls the `composition's <ModelBasedOptimizationControlMechanism.composition>` `evaluate
        <Composition.evaluate>` method in order to evaluate a particular allocation policy

    search_function : function or method
        `search_function <OptimizationFunction.search_function>` assigned to `function
        <OptimizationControlMechanism.function>`; used to select samples of `control_allocation
        <ControlMechanism.control_allocation>` to evaluate by `evaluation_function
        <OptimizationControlMechanism.evaluation_function>`.

    search_termination_function : function or method
        `search_termination_function <OptimizationFunction.search_termination_function>` assigned to
        `function <OptimizationControlMechanism.function>`;  determines when to terminate the
        `optimization process <OptimizationFunction_Process>`.

    control_allocation_search_space : list or ndarray
        `search_space <OptimizationFunction.search_space>` assigned to `function
        <OptimizationControlMechanism.function>`;  determines the samples of
        `control_allocation <ControlMechanism.control_allocation>` evaluated by the `evaluation_function
        <OptimizationControlMechanism.evaluation_function>`.

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

    componentType = MODEL_BASED_OPTIMIZATION_CONTROL_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE

    class Params(ControlMechanism.Params):
        function = None

    paramClassDefaults = ControlMechanism.paramClassDefaults.copy()
    paramClassDefaults.update({PARAMETER_STATES: NotImplemented})  # This suppresses parameterStates

    @tc.typecheck
    def __init__(self,
                 objective_mechanism: tc.optional(tc.any(ObjectiveMechanism, list)) = None,
                 origin_objective_mechanism=False,
                 terminal_objective_mechanism=False,
                 learning_function=None,
                 function: tc.optional(tc.any(is_function_type)) = None,
                 search_function: tc.optional(tc.any(is_function_type)) = None,
                 search_termination_function: tc.optional(tc.any(is_function_type)) = None,
                 search_space: tc.optional(tc.any(list, np.ndarray)) = None,
                 control_signals: tc.optional(tc.any(is_iterable, ParameterState, ControlSignal)) = None,
                 modulation: tc.optional(_is_modulation_param) = ModulationParam.MULTIPLICATIVE,
                 composition=None,
                 params=None,
                 name=None,
                 prefs: is_pref_set = None,
                 **kwargs):
        '''Subclass of `OptimizationControlMechanism <OptimizationControlMechanism>` that uses a `evaluate
           <ModelBasedOptimizationControlMechanism.evaluate>` method in order to adjusts its `ControlSignals
           <ControlSignal>` and optimize performance of the `Composition` to which it belongs.'''

        if kwargs:
            for i in kwargs.keys():
                raise ModelBasedOptimizationControlMechanismError("Unrecognized arg in constructor for {}: {}".
                                                        format(self.__class__.__name__, repr(i)))
        self.learning_function = learning_function
        self.search_function = search_function
        self.search_termination_function = search_termination_function
        self.search_space = search_space
        self.composition = composition

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(origin_objective_mechanism=origin_objective_mechanism,
                                                  terminal_objective_mechanism=terminal_objective_mechanism,
                                                  params=params)

        super().__init__(objective_mechanism=objective_mechanism,
                         function=function,
                         control_signals=control_signals,
                         modulation=modulation,
                         params=params,
                         name=name,
                         prefs=prefs)

    def apply_control_signal_values(self, control_allocation, runtime_params, context):
        '''Assign specified control_allocation'''
        for i in range(len(control_allocation)):
            if self.value is None:
                self.value = self.instance_defaults.value
            self.value[i] = np.atleast_1d(control_allocation[i])

        self._update_output_states(self.value, runtime_params=runtime_params, context=ContextFlags.COMPOSITION)

    def _execute(self, variable=None, runtime_params=None, context=None):
        '''Find control_allocation that optimizes evaluation_function.'''

        if (self.context.initialization_status == ContextFlags.INITIALIZING):
            return defaultControlAllocation

        self.predicted_input = self.composition.before_simulations()

        # Compute control_allocation using MBOCM's optimization function
        control_allocation, self.evc_max, self.saved_samples, self.saved_values = \
                                        super(ControlMechanism, self)._execute(variable=self.control_allocation,
                                                                               runtime_params=runtime_params,
                                                                               context=context)

        self.composition.after_simulations()

        return control_allocation

    def evaluation_function(self, control_allocation):
        '''Compute outcome for a given control_allocation.'''
        # returns net_control_allocation_outcomes
        num_trials = 1
        return self.composition.evaluate(control_allocation,
                                         self.predicted_input,
                                         num_trials,
                                         context=self.function_object.context)

