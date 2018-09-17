# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *********************************************  BayesGLMObjectiveMechanism ********************************************

"""

Overview
--------

A BayesGLMObjectiveMechanism is an `ObjectiveMechanism <ObjectiveMechanism>` that uses the `BayesGLM` function
to evaluate the `values <OutputState.value>` of its `monitored_output_states <BayseGLM.monitored_output_states>`,
and updates a set of `predictor_weights` to improve its prediction of the outcome of that evaluation.

COMMENT:
.. _BayesGLMObjectiveMechanism_Creation:

Creating a BayesGLMObjectiveMechanism
------------------------------

.. _BayesGLMObjectiveMechanism_Creation_Monitored_Output_States:

*Monitored OutputStates*
~~~~~~~~~~~~~~~~~~~~~~~~

.. _BayesGLMObjectiveMechanism_Structure:


Structure
---------

.. _BayesGLMObjectiveMechanism_Input:

*Input*
~~~~~~~

.. _BayesGLMObjectiveMechanism_Function:

*Function*
~~~~~~~~~~

.. _BayesGLMObjectiveMechanism_Output:

*Output*
~~~~~~~~

.. _BayesGLMObjectiveMechanism_Execution:

Execution
---------


.. _BayesGLMObjectiveMechanism_Examples:

Examples
--------

.. _BayesGLMObjectiveMechanism_Class_Reference:

Class Reference
---------------

"""

import typecheck as tc
import numpy as np

from collections import Iterable

from psyneulink.components.mechanisms.processing.objectivemechanism import *
from psyneulink.components.functions.function import LinearCombination, BayesGLM
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.globals.context import ContextFlags
from psyneulink.globals.utilities import is_numeric
from psyneulink.globals.keywords import NAME, VARIABLE, OWNER_VALUE

PREDICTOR_WEIGHTS = 'PREDICTOR_WEIGHTS'
PREDICTOR_VARIANCES = 'PREDICTOR_VARIANCES'

class BayesGLMObjectiveMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

# monitored_output_states is an alias to input_states argument, which can
# still be used in a spec dict
class BayesGLMObjectiveMechanism(ObjectiveMechanism):
    """
    BayesGLMObjectiveMechanism(       \
        monitored_output_states,      \
        num_predictors,               \
        predictor_weights_priors,     \
        predictor_variance_priors,    \
        default_variable,             \
        size,                         \
        function=LinearCombination,   \
        output_states=OUTCOME,        \
        params=None,                  \
        name=None,                    \
        prefs=None)

    Subclass of `ObjectiveMechanism` that evaluates the value(s) of its `monitored_output_states
    <BayesGLMObjectiveMechanism>` and then updates a set of `predictor_weights
    <BayesGLMObjectiveMechanism.predictor_weights>` to improves its prediction of the outcome of the evaluation.

    Arguments
    ---------

    monitored_output_states : List[`OutputState`, `Mechanism`, str, value, dict, `MonitoredOutputStatesOption`] or dict
        specifies the OutputStates, the `values <OutputState.value>` of which will be monitored, and evaluated by
        the ObjectiveMechanism's `function <ObjectiveMechanism>` (see `ObjectiveMechanism_Monitored_Output_States`
        for details of specification).

    default_variable : number, list or np.ndarray : default monitored_output_states
        specifies the format of the `variable <ObjectiveMechanism.variable>` for the `InputStates` of the
        ObjectiveMechanism (see `Mechanism InputState specification <Mechanism_InputState_Specification>` for details).

    size : int, list or np.ndarray of ints
        specifies default_variable as array(s) of zeros if **default_variable** is not passed as an argument;
        if **default_variable** is specified, it takes precedence over the specification of **size**.
        As an example, the following mechanisms are equivalent::
            T1 = TransferMechanism(size = [3, 2])
            T2 = TransferMechanism(default_variable = [[0, 0, 0], [0, 0]])

    num_predictors : int
        specifies number of predictors used to predict the value returned by `function <BayesGLMObjectieMechanism>`.

    predictor_weights_priors : int or 1d array
        specifies initial value used for mean of the distribution for each predictor. If it is an int, the same value
        is used for all predictors.  An array can be used to specify a different prior for each predictor, in which
        case the length of the array must equal `num_predictors <BayesGLMObjectiveMechanism.num_predictors>`.

    predictor_variance_priors :
        specifies initial value used for variance of the distribution for each predictor. If it is an int,
        the same value is used for all predictors.  An array can be used to specify a different prior for each
        predictor, in which case the length of the array must equal `num_predictors
        <BayesGLMObjectiveMechanism.num_predictors>`.

    COMMENT:
    input_states :  List[InputState, value, str or dict] or Dict[] : default None
        specifies the names and/or formats to use for the values of the InputStates that receive the input from the
        OutputStates specified in the monitored_output_states** argument; if specified, there must be one for each item
        specified in the **monitored_output_states** argument.
    COMMENT

    function: CombinationFunction, ObjectiveFunction, function or method : default LinearCombination
        specifies the function used to evaluate the values listed in :keyword:`monitored_output_states`
        (see `function <LearningMechanism.function>` for details.

    output_states :  List[OutputState, value, str or dict] or Dict[] : default [OUTCOME]
        specifies the OutputStates for the Mechanism;

    role: Optional[LEARNING, CONTROL]
        specifies if the ObjectiveMechanism is being used for learning or control (see `role` for details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for the
        Mechanism, its `function <Mechanism_Base.function>`, and/or a custom function and its parameters. Values
        specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default see `name <ObjectiveMechanism.name>`
        specifies the name of the ObjectiveMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the ObjectiveMechanism; see `prefs <ObjectiveMechanism.prefs>` for details.


    Attributes
    ----------

    variable : 2d ndarray : default array of values of OutputStates in monitor_output_states
        the input to Mechanism's `function <TransferMechanism.function>`.

    monitored_output_states : ContentAddressableList[OutputState]
        determines the OutputStates, the `values <OutputState.value>` of which are monitored, and evaluated by the
        ObjectiveMechanism's `function <ObjectiveMechanism.function>`.  Each item in the list refers to an
        `OutputState` containing the value to be monitored, with a `MappingProjection` from it to the
        corresponding `InputState` listed in the `input_states <ObjectiveMechanism.input_states>` attribute.

    monitored_output_states_weights_and_exponents : List[Tuple(float, float)]
        each tuple in the list contains a weight and exponent associated with a corresponding InputState listed in the
        ObjectiveMechanism's `input_states <ObjectiveMechanism.input_states>` attribute;  these are used by its
        `function <ObjectiveMechanism.function>` to parametrize the contribution that the values of each of the
        OuputStates monitored by the ObjectiveMechanism makes to its output (see `ObjectiveMechanism_Function`)

    input_states : ContentAddressableList[InputState]
        contains the InputStates of the ObjectiveMechanism, each of which receives a `MappingProjection` from the
        OutputStates specified in its `monitored_output_states <ObjectiveMechanism.monitored_output_states>` attribute.

    function : CombinationFunction, ObjectiveFunction, function, or method
        the function used to evaluate the values monitored by the ObjectiveMechanism.  The function can be
        any PsyNeuLink `CombinationFunction` or a Python function that takes a 2d array with an arbitrary number of
        items or a number equal to the number of items in the ObjectiveMechanism's variable (i.e., its number of
        input_states) and returns a 1d array.

    role : None, LEARNING or CONTROL
        specifies whether the ObjectiveMechanism is used for learning in a Process or System (in conjunction with a
        `ObjectiveMechanism`), or for control in a System (in conjunction with a `ControlMechanism <ControlMechanism>`).

    value : 1d np.array
        the output of the evaluation carried out by the ObjectiveMechanism's `function <ObjectiveMechanism.function>`.

    output_state : OutputState
        contains the `primary OutputState <OutputState_Primary>` of the ObjectiveMechanism; the default is
        its *OUTCOME* `OutputState <ObjectiveMechanism_Output>`, the value of which is equal to the
        `value <ObjectiveMechanism.value>` attribute of the ObjectiveMechanism.

    output_states : ContentAddressableList[OutputState]
        by default, contains only the *OUTCOME* (`primary <OutputState_Primary>`) OutputState of the ObjectiveMechanism.

    output_values : 2d np.array
        contains one item that is the value of the *OUTCOME* `OutputState <ObjectiveMechanism_Output>`.

    name : str
        the name of the ObjectiveMechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the ObjectiveMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).


    """

    @tc.typecheck
    def __init__(self,
                 monitored_output_states=None,
                 default_variable=None,
                 size=None,
                 num_predictors:int=1,
                 predictor_weights_priors:tc.optional(is_numeric)=None,
                 predictor_variance_priors:tc.optional(is_numeric)=None,
                 function=LinearCombination,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs):

        monitored_output_states = [{NAME:'CURRENT_PREDICTOR_WEIGHTS',
                                   VARIABLE:[0]*num_predictors}] + monitored_output_states

        self._predictor_update_function = BayesGLM(num_predictors=num_predictors,
                                                   mu_prior=predictor_weights_priors,
                                                   sigma_prior=predictor_variance_priors)

        super().__init__(monitored_output_states=monitored_output_states,
                         default_variable=default_variable,
                         size=size,
                         function=function,
                         output_states=[{NAME:PREDICTOR_WEIGHTS, VARIABLE:(OWNER_VALUE,0)},
                                        {NAME:PREDICTOR_VARIANCES, VARIABLE:(OWNER_VALUE,1)}],
                         params=params,
                         name=name,
                         prefs=prefs,
                         **kwargs,
                         context=ContextFlags.CONSTRUCTOR)

    def _execute(self, variable=None, runtime_params=None, context=None):
        self.outcome = super()._execute(variable, runtime_params, context)
        predictors = np.atleast_2d(variable[0])
        dependent_vars = np.atleast_2d(self.outcome)
        return self._predictor_update_function.function(variable=[predictors,dependent_vars])

    def _parse_function_variable(self, variable, context=None):
        '''Return all but first item of variable
        First item of variable is reserved for current predictor weights used by update_function
        '''
        # return variable[1:]
        return np.array([i.astype(float) for i in variable[1:]])

