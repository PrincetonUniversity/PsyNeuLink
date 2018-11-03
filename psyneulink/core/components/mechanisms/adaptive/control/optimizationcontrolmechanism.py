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

An OptimizationControlMechanism is an abstract class for subclasses of `ControlMechanism <ControlMechanism>` that
uses an `OptimizationFunction` to find an `allocation_policy <ControlMechanism.allocation_policy>` -- 
a `variable <ControlSignal.variable>` for each of its `ControlSignals <ControlSignal>` -- that optimizes the
value of its `objective_function <OptimizationControlMechanism.objective_function>`.  The `objective_function 
<OptimizationControlMechanism.objective_function>` is used by the OptimizationControlMechanism's primary 
`function <OptimizationControlMechanism.function>` to determine the `Expected Value of Control (EVC) 
<OptimizationControlMechanisms_EVC>` for a given  `allocation_policy <ControlMechanism.allocation_policy>`. 
There are two broad types of OptimizationControlMechanisms, that can be described loosely as "model-free" and  
"model-based," based on how their `objective_function <OptimizationControlMechanism.objective_function>` determines 
the EVC.

.. _OptimizationControlMechanism_Model_Free:

*Model Free OptimizationControlMechanisms*

OptimizationControlMechanism that are model-free use a `learning_function
<OptimizationControlMechanism_Learning_Function>` to generate a set of `prediction_weights
<OptimizationControlMechanism.prediction_weights>` that can predict, for the current state, 
the `outcome <ControlMechanism.outcome>` resulting from different allocation_policies.  The current state is 
represented in a `prediction_vector <OptimizationControlMechanism.prediction_vector>`.  In each trial, 
the `learning_function <OptimizationControlMechanism.learning_function>` updates the 
`prediction_weights <OptimizationControlMechanism.prediction_weights>` based on the `prediction_vector 
<OptimizationControlMechanism.prediction_vector>` and the `outcome <ControlMechanism.outcome>` for the 
previous trial.  The updated weights can be used by the `objective_function  
<OptimizationControlMechanism.objective_function>` to predict the EVC for the current `prediction_vector 
<OptimizationControlMechanism.prediction_vector>` and a given `allocation_policy 
<ControlMechanism.allocation_policy>`  The OptimizationControlMechainsm's primary `function 
<OptimizationControlMechanism.function>` exploits this to find the `allocation_policy 
<ControlMechanism.allocation_policy>` that yields the *best* EVC for the current `prediction_vector 
<OptimizationControlMechanism.prediction_vector>`, and then implements that for the next `trial` of execution.  

.. _OptimizationControlMechanism_Model_Free:

*Model-Based OptimizationControlMechanisms*

OptimizationControlMechanisms that are model-based are called `controllers <Composition.controllers>`.   These use the  
`ModelBasedControlMechanism` subclass, that has a `run_simulation <ModelBasedControlMechanism.run_simulation>` method.  
Their `objective_function <OptimizationControlMechanism.objective_function>` uses this to empirically determine the 
`EVC <OptimizationControlMechanisms_EVC>` for a given `allocation_policy 
<ControlMechanism.allocation_policy>`, by running one or more simulations of the `Composition` to which 
the ModelBasedControlMechanism belongs. ModelBasedControlMechanisms may or may not also use a `learning_function 
<OptimizationControlMechanism_Learning_Function>` in combination with their `run_simulation 
<ModelBasedControlMechanism.run_simulation>` method (e.g., for efficiency, or to handle factors that influence
`outcome <ControlMechanism.outcome>` and/or `costs <ControlMechanism.costs>` in different ways).  Like model-free 
OptimizationControlMechanisms, their `function <OptimizationControlMechanism.function>` uses `objective_function 
<OptimizationControlMechanism.objective_function` to identify the `allocation_policy 
<ControlMechanism.allocation_policy>` that yields the greatest EVC, and then implement that for the next 
`trial` of execution.
  
.. _OptimizationControlMechanisms_EVC:

All OptimizationControlMechanisms seek to maximize their Expected Value of Control (EVC) --  a cost-benefit analysis 
that weighs the `cost <ControlSignal.cost> of its `control_signals` for an `allocation_policy 
<ControlMechanism.allocation_policy>` against the `outcome <OptimizationControlMechanism.outcome>` 
resulting from that policy.  The EVC for an `allocation_policy <ControlMechanism.allocation_policy>`
is computed by `objective_function <OptimizationControlMechanism.objective_function>` using the 
OptimizationControlMechanism's `compute_EVC <OptimizationControlMechanism.compute_EVC>` method and some combination
of its `outcome <OptimizationControlMechanism.outcome>`, `costs` attribute, and current `predicition_vector
<OptimizationControlMechanism.prediction_vector>` attributes, depending on whether it is model-free or model-based 
and the particular subclass.

.. _OptimizationControlMechanism_Creation:

Creating an OptimizationControlMechanism
----------------------------------------

An OptimizationControlMechanism can be created in the same was as any `ControlMechanism <ControlMechanism>`.  The only
constraint is that an `OptimizationFunction` (or one that has the same structure) must be specified as the **function**
argument of its constructor.  In addition, a **learning_function** can be specified (see `below
<OptimizationControlMechanism_Learning_Function>`)

.. _OptimizationControlMechanism_Structure:

Structure
---------

An OptimizationControlMechanism has the same structure as a `ControlMechanism`, including a `Projection <Projection>`
to its *OUTCOME* InputState from its `objective_mechanism <ControlMechanism.objective_mechanism>`.  In
addition to its primary `function <OptimizationControlMechanism.function>`, it may also have a `learning_function
<OptimizationControlMechanism.learning_function>`, both of which are described below.

.. _OptimizationControlMechanism_Learning_Function:

Learning Function
^^^^^^^^^^^^^^^^^

An OptimizationControlMechanism may have a `learning_function <OptimizationControlMechanism.learning_function>`
used to generate a model that attempts to predict the value of its `objective_function
<OptimizationControlMechanism.objective_function>` for a given `allocation_policy <ControlMechanism.allocation_policy>`
from a `prediction_vector <OptimizationControlMechanism.prediction_vector>`; it is up to the subclass of the
OptimizationControlMechanism to determine the contents of `prediction_vector
<OptimizationControlMechanism.prediction_vector>`, as well as the `objective_function
<OptimizationControlMechanism.objective_function>`. The `learning_function
<OptimizationControlMechanism.learning_function>` takes as its first argument the `prediction_vector
<OptimizationControlMechanism.prediction_vector>`), and as its second argument the value of it seeks to predict
(typically, the OptimizationControlMechanism's `net_oucome <OptimizationControlMechanism.net_outcome>` attribute.
It returns an array with one weight for each element of `prediction_vector
<OptimizationControlMechanism.prediction_vector>`, that is assigned as the OptimizationControlMechanism's
`prediction_weights <OptimizationControlMechanism.prediction_weights>`) attribute.  This is can be used by its
primary `function <OptimizationControlMechanism.function>` in seeking an `allocation_policy
<ControlMechanism.allocation_policy>` that yields the best value of the `objective_function
<OptimizationControlMechanism.objective_function>` (see `below <OptimizationControlMechanism_Function>).

.. _OptimizationControlMechanism_Function:

*Primary Function*
^^^^^^^^^^^^^^^^^^

The `function <OptimizationControlMechanism.function>` of an OptimizationControlMechanism is generally an
`OptimizationFunction`, which in turn has `objective_function <OptimizationFunction.objective_function>`,
`search_function <OptimizationFunction.search_function>` and `search_termination_function
<OptimizationFunction.search_termination_function>` methods, as well as a `search_space
<OptimizationFunction.search_space>` attribute.  The `objective_function <OptimizationFunction.objective_function>`
is used to evaluate each `allocation_policy <ControlMechanism.allocation_policy>` generated by the `search_function
<OptimizationFunction.search_function>`, and return the one that yields the greatest `EVC 
<OptimizationControlMechanism_EVC>` based on the `compute_EVC <OptimizationControlMechanism.compute_EVC>` method.

An OptimizationControlMechanism must implement an `objective_function <OptimizationControlMechanism>` method that
is passed to the `OptimizationFunction` as its `objective_function <OptimizationFunction.objective_function>` parameter.
The OptimizationControlMechanism may also implement `search_function <OptimizationControlMechanism.search_function>`
and `search_termination_function <OptimizationControlMechanism.search_termination_function>` methods, as well as a
`search_space <OptimizationControlMechanism.search_space>` attribute, that will also be passed as parameters to the
`OptimizationFunction` when it is constructed.  Any or all of these assignments can be overriden by specifying the
relevant parameters in a constructor for the `OptimizationFunction` assigned as the **function** argument of the
OptimizationControlMechanism's constructor, as long as they are compatible with the requirements of the
OptimizationFunction and OptimizationControlMechanism.  A custom function can also be assigned as the `function
<OptimizationControlMechanism.function>` of an OptimizationControlMechanism, however it must meet the following
requirements:

.. _OptimizationControlMechanism_Custom_Funtion:

    - it must accept as its first argument and return as its result an array with the same shape as the
      OptimizationControlMechanism's `allocation_policy <ControlMechanism.allocation_policy>`.

    - it must implement a :method:`reinitialize` method that accepts as keyword arguments **objective_function**,
      **search_function**, **search_termination_function**, and **search_space**, and implement attributes
      with corresponding names.

.. _OptimizationControlMechanism_Execution:

Execution
---------

When an OptimizationControlMechanism executes, it calls its `learning_function
<OptimizationControlMechanism.learning_function>` if it has one, to udpate its `prediction_weights 
OptimizationControlMechanism.prediction_weights>`. It then calls its primary `function 
<OptimizationControlMechanism.function>` to find the `allocation_policy <ControlMechanism.allocation_policy>` that 
yields the greatest `EVC <OptimizationControlMechanism_EVC>`.  The `function <OptimizationControlMechanism.function>` 
does this by selecting a sample `allocation_policy <ControlMechanism.allocation_policy>` (usually using  
`search_function <OptimizationControlMechanism.search_function>` to select one from `control_signal_search_space 
<OptimizationControlMechanism.control_signal_search_space>`, and evaluating the EVC for that `allocation_policy 
<ControlMechanism.allocation_policy>` using the `objective_function <OptimizationControlMechanism.objective_function>`.
The latter does so either by using the current `prediction_vector OptimizationControlMechanism.prediction_vector` and
`prediction_weights <OptimizationControlMechanism.prediction_weights>` to predict the EVC (model-free
OptimizationControlMechanisms), or by calling the OptimizationControlMechanism's `run_simulation
<ModelBasedControlMechanism.run_simulation>` to "empirically" generate the `outcome
<OptimizationControlMechanism.outcome>` for the `allocation_policy <ControlMechanism.allocation_policy>` and then
evaluting the EVC for the resulting `outcome <OptimizationControlMechanism.outcome>` and `costs
<OptimizationControlMechanism.costs>` (model-based `controller <Composition.controller>`\\s).  In either case, one or
more allocation_policies are evaluated, and the one that yields the greatest EVC is returned.  The values of that
`allocation_policy <ControlMechanism.allocation_policy>` are assigned as the `variables <ControlSignal.variable>` of
its `control_signals <ControlMechanism.control_signals>`, from which they compute their `values <ControlSignal.value>`.

COMMENT:
.. _OptimizationControlMechanism_Examples:

Example
-------
COMMENT

.. _OptimizationControlMechanism_Class_Reference:

Class Reference
---------------

"""
import typecheck as tc

import numpy as np

from psyneulink.core.components.functions.function import \
    ModulationParam, _is_modulation_param, is_function_type, OBJECTIVE_FUNCTION, \
    SEARCH_SPACE, SEARCH_FUNCTION, SEARCH_TERMINATION_FUNCTION
from psyneulink.core.components.mechanisms.adaptive.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import \
    ObjectiveMechanism, MONITORED_OUTPUT_STATES
from psyneulink.core.components.states.parameterstate import ParameterState
from psyneulink.core.components.states.modulatorysignals.controlsignal import ControlSignalCosts, ControlSignal
from psyneulink.core.globals.keywords import \
    DEFAULT_VARIABLE, PARAMETER_STATES, OBJECTIVE_MECHANISM, OPTIMIZATION_CONTROL_MECHANISM
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import is_iterable

__all__ = [
    'OptimizationControlMechanism', 'OptimizationControlMechanismError'
]

class OptimizationControlMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

    
class OptimizationControlMechanism(ControlMechanism):
    """OptimizationControlMechanism(                       \
    objective_mechanism=None,                              \
    learning_function=None,                                \
    objective_function=None,                               \
    search_function=None,                                  \
    search_termination_function=None,                      \
    search_space=None,                                     \
    function=None,                                         \
    control_signals=None,                                  \
    modulation=ModulationParam.MULTIPLICATIVE,             \
    params=None,                                           \
    name=None,                                             \
    prefs=None)

    Subclass of `ControlMechanism <ControlMechanism>` that adjusts its `ControlSignals <ControlSignal>` to optimize
    performance of the `Composition` to which it belongs

    Arguments
    ---------

    objective_mechanism : ObjectiveMechanism or List[OutputState specification]
        specifies either an `ObjectiveMechanism` to use for the OptimizationControlMechanism, or a list of the 
        `OutputState <OutputState>`\\s it should monitor; if a list of `OutputState specifications
        <ObjectiveMechanism_Monitored_Output_States>` is used, a default ObjectiveMechanism is created and the list
        is passed to its **monitored_output_states** argument.

    learning_function : LearningFunction, function or method
        specifies a function used to learn to predict the `EVC <OptimizationControlMechanism_EVC>` from
        the current `prediction_vector <OptimizationControlMechanism.prediction_vector>` (see
        `OptimizationControlMechanism_Learning_Function` for details).

    objective_function : function or method
        specifies the function used to evaluate the `EVC <OptimizationControlMechanism_EVC>` for a given
        `allocation_policy <ControlMechanism.allocation_policy>`. It is assigned as the `objective_function 
        <OptimizationFunction.objective_function>` parameter of `function  <OptimizationControlMechanism.function>`, 
        unless that is specified in the constructor for an  OptimizationFunction assigned to the **function** 
        argument of the OptimizationControlMechanism's constructor.  Often it is assigned directy to the 
        OptimizationControlMechanism's `compute_EVC <OptimizationControlMechanism.compute_EVC>` method;  in some 
        cases it may implement additional operations, but should always call `compute_EVC 
        <OptimizationControlMechanism.compute_EVC>`. A custom function can be assigned, but it must take as its 
        first argument an array with the same shape as the OptimizationControlMechanism's `allocation_policy 
        <ControlMechanism.allocation_policy>`, and return the following four values: an array containing the 
        `allocation_policy <ControlMechanism.allocation_policy>` that generated the optimal EVC 
        <OptimizationControlMechanism_EVC>; an array containing that EVC value;  a list containing each 
        `allocation_policy <ControlMechanism.allocation_policy>` sampled if `function 
        <OptimizationControlMechanism.function>` has a `save_samples <OptimizationFunction.save_samples>` attribute 
        and it is `True`, otherwise it should return an empty list; and a list containing the EVC values for each 
        `allocation_policy <ControlMechanism.allocation_policy>` sampled if the function has a `save_values 
        <OptimizationFunction.save_values>` attribute and it is `True`, otherwise it should return an empty list.

    search_function : function or method
        specifies the function assigned to `function <OptimizationControlMechanism.function>` as its 
        `search_function <OptimizationFunction.search_function>` parameter, unless that is specified in a 
        constructor for `function <OptimizationControlMechanism.function>`.  It must take as its arguments 
        an array with the same shape as `allocation_policy <ControlMechanism.allocation_policy>` and an integer
        (indicating the iteration of the `optimization process <OptimizationFunction_Process>`), and return 
        an array with the same shape as `allocation_policy <ControlMechanism.allocation_policy>`.

    search_termination_function : function or method
        specifies the function assigned to `function <OptimizationControlMechanism.function>` as its 
        `search_termination_function <OptimizationFunction.search_termination_function>` parameter, unless that is 
        specified in a constructor for `function <OptimizationControlMechanism.function>`.  It must take as its 
        arguments an array with the same shape as `allocation_policy <ControlMechanism.allocation_policy>` and two 
        integers (the first representing the `EVC <OptimizationControlMechanism_EVC>` value for the current 
        `allocation_policy <ControlMechanism.allocation_policy>`, and the second the current iteration of the 
        `optimization process <OptimizationFunction_Process>`;  it must return `True` or `False`.
        
    search_space : list or ndarray
        specifies the `search_space <OptimizationFunction.search_space>` parameter for `function 
        <OptimizationControlMechanism.function>`, unless that is specified in a constructor for `function 
        <OptimizationControlMechanism.function>`.  Each item must have the same shape as `allocation_policy 
        <ControlMechanism.allocation_policy>`.
        
    function : OptimizationFunction, function or method
        specifies the function used to optimize the `allocation_policy <ControlMechanism.allocation_policy>`;  
        must take as its sole argument an array with the same shape as `allocation_policy 
        <ControlMechanism.allocation_policy>`, and return a similar array (see `Primary Function 
        <OptimizationControlMechanism>` for additional details).

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

    prediction_vector : 1d ndarray
        array passed to `learning_function <OptimizationControlMechanism.learning_function>` if that is implemented.

    prediction_weights : 1d ndarray
        weights assigned to each term of `prediction_vector <OptimizationControlMechanism.prediction_vector>`
        by `learning_function <OptimizationControlMechanism.learning_function>`.

    learning_function : LearningFunction, function or method
        takes `prediction_vector <OptimizationControlMechanism.prediction_vector>` as its first argument, and 
        `net_outcome <ControlMechanism.net_outcome>` as its second argument, and returns an updated set of 
        `prediction_weights <OptimizationControlMechanism.prediction_weights>` (see  
        `OptimizationControlMechanism_Learning_Function` for additional details).

    function : OptimizationFunction, function or method
        takes current `allocation_policy <ControlMechanism.allocation_policy>` (as initializer),
        evaluates samples of `allocation_policy <ControlMechanism.allocation_policy>`, and returns 
        the one that yields the greatest EVC <OptimizationControlMechanism_EVC>`  (see 
        `Primary Function  <OptimizationControlMechanism_Function>` for additional details).

    objective_function : function or method
        `objective_function <OptimizationFunction.objective_function>` assigned to `function 
        <OptimizationControlMechanism.function>`;  often this is simply the OptimizationControlMechanism's
        `compute_EVC <OptimizationControlMechanism.compute_EVC>` method, but it should always call that.  
        
    search_function : function or method
        `search_function <OptimizationFunction.search_function>` assigned to `function 
        <OptimizationControlMechanism.function>`; used to select samples of `allocation_policy 
        <ControlMechanism.allocation_policy>` to evaluate by `objective_function 
        <OptimizationControlMechanism.objective_function>`.

    search_termination_function : function or method
        `search_termination_function <OptimizationFunction.search_termination_function>` assigned to
        `function <OptimizationControlMechanism.function>`;  determines when to terminate the 
        `optimization process <OptimizationFunction_Process>`.
        
    search_space : list or ndarray
        `search_space <OptimizationFunction.search_space>` assigned to `function 
        <OptimizationControlMechanism.function>`;  determines the samples of 
        `allocation_policy <ControlMechanism.allocation_policy>` evaluated by the `objective_function 
        <OptimizationControlMechanism.objective_function>`.

    saved_samples : list
        contains all values of `allocation_policy <ControlMechanism.allocation_policy>` sampled by `function
        <OptimizationControlMechanism.function>` if its `save_samples <OptimizationFunction.save_samples>` parameter
        is `True`;  otherwise list is empty.

    saved_values : list
        contains values of EVC associated with all samples of `allocation_policy <ControlMechanism.allocation_policy>` 
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
                 objective_mechanism:tc.optional(tc.any(ObjectiveMechanism, list))=None,
                 origin_objective_mechanism=False,
                 terminal_objective_mechanism=False,
                 learning_function=None,
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
        self.learning_function = learning_function
        self.search_function = search_function
        self.search_termination_function = search_termination_function
        self.search_space = search_space

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(origin_objective_mechanism=origin_objective_mechanism,
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
            raise OptimizationControlMechanismError("{} specified for {} ({}) must be assigned one or more {}".
                                                    format(ObjectiveMechanism.__name__, self.name,
                                                           request_set[OBJECTIVE_MECHANISM],
                                                           repr(MONITORED_OUTPUT_STATES)))

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
        '''Instantiate OptimizationControlMechanism attributes and assign parameters to learning_function & function'''

        super()._instantiate_attributes_after_function(context=context)

        if self.learning_function:
            self._instantiate_learning_function()

        # Assign parameters to function (OptimizationFunction) that rely on OptimizationControlMechanism
        self.function_object.reinitialize({DEFAULT_VARIABLE: self.allocation_policy,
                                           OBJECTIVE_FUNCTION: self.objective_function,
                                           SEARCH_FUNCTION: self.search_function,
                                           SEARCH_TERMINATION_FUNCTION: self.search_termination_function,
                                           SEARCH_SPACE: self.get_control_signal_search_space()})
        
        self.objective_function = self.function_object.objective_function
        self.search_function = self.function_object.search_function
        self.search_termination_function = self.function_object.search_termination_function
        self.search_space = self.function_object.search_space

    def get_control_signal_search_space(self):

        control_signal_sample_lists = []
        for control_signal in self.control_signals:
            control_signal_sample_lists.append(control_signal.allocation_samples)

        # Construct control_signal_search_space:  set of all permutations of ControlProjection allocations
        #                                     (one sample from the allocationSample of each ControlProjection)
        # Reference for implementation below:
        # http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
        self.control_signal_search_space = \
            np.array(np.meshgrid(*control_signal_sample_lists)).T.reshape(-1,len(self.control_signals))

        # Insure that ControlSignal in each sample is in its own 1d array
        re_shape = (self.control_signal_search_space.shape[0], self.control_signal_search_space.shape[1], 1)

        return self.control_signal_search_space.reshape(re_shape)

    def _execute(self, variable=None, runtime_params=None, context=None):
        '''Find allocation_policy that optimizes objective_function.'''

        raise OptimizationControlMechanismError("PROGRAM ERROR: {} must implement its own {} method".
                                                format(self.__class__.__name__, repr('_execute')))
    def objective_function(self, allocation_policy):
        '''Compute outcome for a given allocation_policy.'''

        raise OptimizationControlMechanismError("PROGRAM ERROR: {} must implement an {} method".
                                                format(self.__class__.__name__, repr('objective_function')))

    def compute_EVC(self):
        '''Computes `EVC <OptimizationControlMechanism_EVC> for a given allocation_policy.'''

        raise OptimizationControlMechanismError("PROGRAM ERROR: {} must implement an {} method".
                                                format(self.__class__.__name__, repr('compute_EVC')))

