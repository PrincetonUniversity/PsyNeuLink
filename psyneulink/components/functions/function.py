#
# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ***********************************************  Function ************************************************************

"""
Example function:
  * `ArgumentTherapy`

Combination Functions:
  * `Reduce`
  * `LinearCombination`
  * `CombineMeans`
  * `PredictionErrorDelta`

TransferMechanism Functions:
  * `Linear`
  * `Exponential`
  * `Logistic`
  * `OneHot`
  * `SoftMax`
  * `LinearMatrix`

Integrator Functions:
  * `Integrator`
  * `SimpleIntegrator`
  * `ConstantIntegrator`
  * `AdaptiveIntegrator`
  * `DriftDiffusionIntegrator`
  * `OrnsteinUhlenbeckIntegrator`
  * `AccumulatorIntegrator`
  * `FHNIntegrator`
  * `AGTUtilityIntegrator`
  * `BogaczEtAl`
  * `NavarroAndFuss`

Distribution Functions:
  * `NormalDist`
  * `UniformToNormalDist`
  * `ExponentialDist`
  * `UniformDist`
  * `GammaDist`
  * `WaldDist`

Objective Functions:
  * `Stability`
  * `Distance`

Learning Functions:
  * `Hebbian`
  * `Reinforcement`
  * `BackPropagation`
  * `TDLearning`

Custom Function:
  * `UserDefinedFunction`

.. _Function_Overview:

Overview
--------

A Function is a `Component <Component>` that "packages" a function (in its `function <Function_Base.function>` method)
for use by other Components.  Every Component in PsyNeuLink is assigned a Function; when that Component is executed, its
Function's `function <Function_Base.function>` is executed.  The `function <Function_Base.function>` can be any callable
operation, although most commonly it is a mathematical operation (and, for those, almost always uses a call to one or
more numpy functions).  There are two reasons PsyNeuLink packages functions in a Function Component:

* **Manage parameters** -- parameters are attributes of a Function that either remain stable over multiple calls to the
  function (e.g., the `gain <Logistic.gain>` or `bias <Logistic.bias>` of a `Logistic` function, or the learning rate
  of a learning function); or, if they change, they do so less frequently or under the control of different factors
  than the function's variable (i.e., its input).  As a consequence, it is useful to manage these separately from the
  function's variable, and not have to provide them every time the function is called.  To address this, every
  PsyNeuLink Function has a set of attributes corresponding to the parameters of the function, that can be specified at
  the time the Function is created (in arguments to its constructor), and can be modified independently
  of a call to its :keyword:`function`. Modifications can be directly (e.g., in a script), or by the operation of other
  PsyNeuLink Components (e.g., `AdaptiveMechanisms`) by way of `ControlProjections <ControlProjection>`.
..
* **Modularity** -- by providing a standard interface, any Function assigned to a Components in PsyNeuLink can be
  replaced with other PsyNeuLink Functions, or with user-written custom functions so long as they adhere to certain
  standards (the PsyNeuLink :ref:`Function API <LINK>`).

.. _Function_Creation:

Creating a Function
-------------------

A Function can be created directly by calling its constructor.  Functions are also created automatically whenever
any other type of PsyNeuLink Component is created (and its :keyword:`function` is not otherwise specified). The
constructor for a Function has an argument for its `variable <Function_Base.variable>` and each of the parameters of
its `function <Function_Base.function>`.  The `variable <Function_Base.variable>` argument is used both to format the
input to the `function <Function_Base.function>`, and assign its default value.  The arguments for each parameter can
be used to specify the default value for that parameter; the values can later be modified in various ways as described
below.

.. _Function_Structure:

Structure
---------

.. _Function_Core_Attributes:

Core Attributes
~~~~~~~~~~~~~~~

Every Function has the following core attributes:

* `variable <Function_Base.variable>` -- provides the input to the Function's `function <Function_Base.function>`.
..
* `function <Function_Base.function>` -- determines the computation carried out by the Function; it must be a
  callable object (that is, a python function or method of some kind). Unlike other PsyNeuLink `Components
  <Component>`, it *cannot* be (another) Function object (it can't be "turtles" all the way down!). If the Function
  has been assigned to another `Component`, then its `function <Function_Base.function>` is also assigned as the
  the `function <Component.function>` attribute of the Component to which it has been assigned (i.e., its
  `owner <Function_Base.owner>`.

A Function also has an attribute for each of the parameters of its `function <Function_Base.function>`.

Owner
~~~~~

If a Function has been assigned to another `Component`, then it also has an `owner <Function_Base.owner>` attribute
that refers to that Component.  The Function itself is assigned as the Component's
`function_object <Component.function_object>` attribute.  Each of the Function's attributes is also assigned
as an attribute of the `owner <Function_Base.owner>`, and those are each associated with with a
`parameterState <ParameterState>` of the `owner <Function_Base.owner>`.  Projections to those parameterStates can be
used by `ControlProjections <ControlProjection>` to modify the Function's parameters.


COMMENT:
.. _Function_Output_Type_Conversion:

If the `function <Function_Base.function>` returns a single numeric value, and the Function's class implements
FunctionOutputTypeConversion, then the type of value returned by its `function <Function>` can be specified using the
`functionOutputType` attribute, by assigning it one of the following `FunctionOutputType` values:
    * FunctionOutputType.RAW_NUMBER: return "exposed" number;
    * FunctionOutputType.NP_1D_ARRAY: return 1d np.array
    * FunctionOutputType.NP_2D_ARRAY: return 2d np.array.

To implement FunctionOutputTypeConversion, the Function's FUNCTION_OUTPUT_TYPE_CONVERSION parameter must set to True,
and function type conversion must be implemented by its `function <Function_Base.function>` method
(see `Linear` for an example).
COMMENT

.. _Function_Modulatory_Params:

Modulatory Parameters
~~~~~~~~~~~~~~~~~~~~~

Some classes of Functions also implement a pair of modulatory parameters: `multiplicative_param` and `additive_param`.
Each of these is assigned the name of one of the function's parameters. These are used by `ModulatorySignals
<ModulatorySignal>` to modulate the output of the function (see `figure <ModulatorySignal_Detail_Figure>`).  For
example, they are used by `GatingSignals <GatingSignal>` to modulate the `function <State_Base.function>` of an
`InputState` or `OutputState`, and thereby its `value <State_Base.value>`; and by the `ControlSignal(s) <ControlSignal>`
of an `LCControlMechanism` to modulate the `multiplicative_param` of the `function <TransferMechanism.function>` of a
`TransferMechanism`.


.. _Function_Execution:

Execution
---------

Functions are not executable objects, but their `function <Function_Base.function>` can be called.   This can be done
directly.  More commonly, however, they are called when their `owner <Function_Base.owner>` is executed.  The parameters
of the `function <Function_Base.function>` can be modified when it is executed, by assigning a
`parameter specification dictionary <ParameterState_Specification>` to the **params** argument in the
call to the `function <Function_Base.function>`.  For `Mechanisms <Mechanism>`, this can also be done by specifying
`runtime_params <Mechanism_Runtime_Parameters>` for the Mechanism when it is `executed <Mechanism_Base.execute>`.

Class Reference
---------------

"""

import numbers
import warnings

from collections import namedtuple
from enum import Enum, IntEnum
from random import randint

import numpy as np
import typecheck as tc

from psyneulink.components.component import ComponentError, DefaultsFlexibility, function_type, method_type, parameter_keywords
from psyneulink.components.shellclasses import Function
from psyneulink.globals.context import ContextFlags
from psyneulink.globals.keywords import ACCUMULATOR_INTEGRATOR_FUNCTION, ADAPTIVE_INTEGRATOR_FUNCTION, ALL, ARGUMENT_THERAPY_FUNCTION, AUTO_ASSIGN_MATRIX, AUTO_DEPENDENT, BACKPROPAGATION_FUNCTION, BETA, BIAS, COMBINATION_FUNCTION_TYPE, COMBINE_MEANS_FUNCTION, CONSTANT_INTEGRATOR_FUNCTION, CONTEXT, CORRELATION, CROSS_ENTROPY, CUSTOM_FUNCTION, DECAY, DIFFERENCE, DISTANCE_FUNCTION, DISTANCE_METRICS, DIST_FUNCTION_TYPE, DIST_MEAN, DIST_SHAPE, DRIFT_DIFFUSION_INTEGRATOR_FUNCTION, DistanceMetrics, ENERGY, ENTROPY, EUCLIDEAN, EXAMPLE_FUNCTION_TYPE, EXECUTING, EXPONENTIAL_DIST_FUNCTION, EXPONENTIAL_FUNCTION, EXPONENTS, FHN_INTEGRATOR_FUNCTION, FULL_CONNECTIVITY_MATRIX, FUNCTION, FUNCTION_OUTPUT_TYPE, FUNCTION_OUTPUT_TYPE_CONVERSION, FUNCTION_PARAMS, GAIN, GAMMA_DIST_FUNCTION, HEBBIAN_FUNCTION, HIGH, HOLLOW_MATRIX, IDENTITY_MATRIX, INCREMENT, INITIALIZER, INITIALIZING, INPUT_STATES, INTEGRATOR_FUNCTION, INTEGRATOR_FUNCTION_TYPE, INTERCEPT, LEARNING, LEARNING_FUNCTION_TYPE, LEARNING_RATE, LINEAR_COMBINATION_FUNCTION, LINEAR_FUNCTION, LINEAR_MATRIX_FUNCTION, LOGISTIC_FUNCTION, LOW, MATRIX, MATRIX_KEYWORD_NAMES, MATRIX_KEYWORD_VALUES, MAX_ABS_INDICATOR, MAX_ABS_VAL, MAX_INDICATOR, MAX_VAL, NOISE, NORMALIZING_FUNCTION_TYPE, NORMAL_DIST_FUNCTION, OBJECTIVE_FUNCTION_TYPE, OFFSET, ONE_HOT_FUNCTION, OPERATION, ORNSTEIN_UHLENBECK_INTEGRATOR_FUNCTION, OUTPUT_STATES, OUTPUT_TYPE, PARAMETER_STATE_PARAMS, PARAMS, PEARSON, PREDICTION_ERROR_DELTA_FUNCTION, PROB, PROB_INDICATOR, PRODUCT, RANDOM_CONNECTIVITY_MATRIX, RATE, RECEIVER, REDUCE_FUNCTION, RL_FUNCTION, SCALE, SIMPLE_INTEGRATOR_FUNCTION, SLOPE, SOFTMAX_FUNCTION, STABILITY_FUNCTION, STANDARD_DEVIATION, SUM, TDLEARNING_FUNCTION, TIME_STEP_SIZE, TRANSFER_FUNCTION_TYPE, UNIFORM_DIST_FUNCTION, USER_DEFINED_FUNCTION, USER_DEFINED_FUNCTION_TYPE, UTILITY_INTEGRATOR_FUNCTION, VARIABLE, WALD_DIST_FUNCTION, WEIGHTS, kwComponentCategory, kwPreferenceSetName
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set, kpReportOutputPref, kpRuntimeParamStickyAssignmentPref
from psyneulink.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel
from psyneulink.globals.registry import register_category
from psyneulink.globals.utilities import is_distance_metric, is_iterable, is_matrix, is_numeric, iscompatible, np_array_less_than_2d, parameter_spec

__all__ = [
    'AccumulatorIntegrator', 'AdaptiveIntegrator', 'ADDITIVE', 'ADDITIVE_PARAM',
    'AdditiveParam', 'AGTUtilityIntegrator', 'ArgumentTherapy',
    'AUTOASSOCIATIVE', 'BackPropagation', 'BogaczEtAl', 'BOUNDS',
    'CombinationFunction', 'CombineMeans', 'ConstantIntegrator', 'DISABLE',
    'DISABLE_PARAM', 'Distance', 'DistributionFunction', 'DRIFT_RATE',
    'DRIFT_RATE_VARIABILITY', 'DriftDiffusionIntegrator', 'EPSILON',
    'ERROR_MATRIX', 'Exponential', 'ExponentialDist', 'FHNIntegrator',
    'Function_Base', 'function_keywords', 'FunctionError', 'FunctionOutputType',
    'FunctionRegistry', 'GammaDist', 'get_matrix', 'get_param_value_for_function',
    'get_param_value_for_keyword', 'Hebbian', 'Integrator',
    'IntegratorFunction', 'is_Function', 'is_function_type', 'kwBogaczEtAl',
    'kwNavarrosAndFuss', 'LCAIntegrator', 'LEARNING_ACTIVATION_FUNCTION',
    'LEARNING_ACTIVATION_INPUT', 'LEARNING_ACTIVATION_OUTPUT',
    'LEARNING_ERROR_OUTPUT', 'LearningFunction', 'Linear', 'LinearCombination',
    'LinearMatrix', 'Logistic', 'max_vs_avg', 'max_vs_next', 'MODE', 'ModulatedParam',
    'ModulationParam', 'MULTIPLICATIVE', 'MULTIPLICATIVE_PARAM',
    'MultiplicativeParam', 'NavarroAndFuss', 'NF_Results', 'NON_DECISION_TIME',
    'NormalDist', 'ObjectiveFunction', 'OrnsteinUhlenbeckIntegrator',
    'OneHot', 'OVERRIDE', 'OVERRIDE_PARAM', 'PERTINACITY', 'PredictionErrorDeltaFunction',
    'PROPENSITY', 'Reduce', 'Reinforcement', 'ReturnVal', 'SimpleIntegrator',
    'SoftMax', 'Stability', 'STARTING_POINT', 'STARTING_POINT_VARIABILITY',
    'TDLearning', 'THRESHOLD', 'TransferFunction', 'THRESHOLD_VARIABILITY',
    'UniformDist', 'UniformToNormalDist', 'UserDefinedFunction', 'WaldDist', 'WT_MATRIX_RECEIVERS_DIM',
    'WT_MATRIX_SENDERS_DIM'
]

EPSILON = np.finfo(float).eps

FunctionRegistry = {}

function_keywords = {FUNCTION_OUTPUT_TYPE, FUNCTION_OUTPUT_TYPE_CONVERSION}


class FunctionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class FunctionOutputType(IntEnum):
    RAW_NUMBER = 0
    NP_1D_ARRAY = 1
    NP_2D_ARRAY = 2


# Typechecking *********************************************************************************************************

# TYPE_CHECK for Function Instance or Class
def is_Function(x):
    if not x:
        return False
    elif isinstance(x, Function):
        return True
    elif issubclass(x, Function):
        return True
    else:
        return False


def is_function_type(x):
    if not x:
        return False
    elif isinstance(x, (Function, function_type, method_type)):
        return True
    elif issubclass(x, Function):
        return True
    else:
        return False


# Modulatory Parameters ************************************************************************************************

ADDITIVE_PARAM = 'additive_param'
MULTIPLICATIVE_PARAM = 'multiplicative_param'
OVERRIDE_PARAM = 'OVERRIDE'
DISABLE_PARAM = 'DISABLE'


class MultiplicativeParam():
    attrib_name = MULTIPLICATIVE_PARAM
    name = 'MULTIPLICATIVE'
    init_val = 1
    reduce = lambda x : np.product(np.array(x), axis=0)


class AdditiveParam():
    attrib_name = ADDITIVE_PARAM
    name = 'ADDITIVE_PARAM'
    init_val = 0
    reduce = lambda x : np.sum(np.array(x), axis=0)

# IMPLEMENTATION NOTE:  USING A namedtuple DOESN'T WORK, AS CAN'T COPY PARAM IN Component._validate_param
# ModulationType = namedtuple('ModulationType', 'attrib_name, name, init_val, reduce')


class ModulationParam():
    """Specify parameter of a `Function <Function>` for `modulation <ModulatorySignal_Modulation>` by a ModulatorySignal

    COMMENT:
        Each term specifies a different type of modulation used by a `ModulatorySignal <ModulatorySignal>`.  The first
        two refer to classes that define the following terms:
            * attrib_name (*ADDITIVE_PARAM* or *MULTIPLICATIVE_PARAM*):  specifies which meta-parameter of the function
              to use for modulation;
            * name (str): name of the meta-parameter
            * init_val (int or float): value with which to initialize the parameter being modulated if it is not otherwise
              specified
            * reduce (function): the manner by which to aggregate multiple ModulatorySignals of that type, if the
              `ParameterState` receives more than one `ModulatoryProjection <ModulatoryProjection>` of that type.
    COMMENT

    Attributes
    ----------

    MULTIPLICATIVE
        assign the `value <ModulatorySignal.value>` of the ModulatorySignal to the *MULTIPLICATIVE_PARAM*
        of the State's `function <State_Base.function>`

    ADDITIVE
        assign the `value <ModulatorySignal.value>` of the ModulatorySignal to the *ADDITIVE_PARAM*
        of the State's `function <State_Base.function>`

    OVERRIDE
        assign the `value <ModulatorySignal.value>` of the ModulatorySignal directly to the State's
        `value <State_Base.value>` (ignoring its `variable <State_Base.variable>` and `function <State_Base.function>`)

    DISABLE
        ignore the ModulatorySignal when calculating the State's `value <State_Base.value>`
    """
    MULTIPLICATIVE = MultiplicativeParam
    # MULTIPLICATIVE = ModulationType(MULTIPLICATIVE_PARAM,
    #                                 'MULTIPLICATIVE',
    #                                 1,
    #                                 lambda x : np.product(np.array(x), axis=0))
    ADDITIVE = AdditiveParam
    # ADDITIVE = ModulationType(ADDITIVE_PARAM,
    #                           'ADDITIVE_PARAM',
    #                           0,
    #                           lambda x : np.sum(np.array(x), axis=0))
    OVERRIDE = OVERRIDE_PARAM
    DISABLE = DISABLE_PARAM

MULTIPLICATIVE = ModulationParam.MULTIPLICATIVE
ADDITIVE = ModulationParam.ADDITIVE
OVERRIDE = ModulationParam.OVERRIDE
DISABLE = ModulationParam.DISABLE


def _is_modulation_param(val):
    if val in ModulationParam.__dict__.values():
        return True
    else:
        return False

ModulatedParam = namedtuple('ModulatedParam', 'meta_param, function_param, function_param_val')


def _get_modulated_param(owner, mod_proj):
    """Return ModulationParam object, function param name and value of param modulated by ModulatoryProjection
    """

    from psyneulink.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base

    if not isinstance(mod_proj, ModulatoryProjection_Base):
        raise FunctionError('mod_proj ({0}) is not a ModulatoryProjection_Base'.format(mod_proj))

    # Get function "meta-parameter" object specified in the Projection sender's modulation attribute
    function_mod_meta_param_obj = mod_proj.sender.modulation

    # Get the actual parameter of owner.function_object to be modulated
    function_param_name = owner.function_object.params[function_mod_meta_param_obj.attrib_name]

    # Get the function parameter's value
    function_param_value = owner.function_object.params[function_param_name]
    # MODIFIED 6/9/17 OLD:
    # if function_param_value is None:
    #     function_param_value = function_mod_meta_param_obj.init_val
    # MODIFIED 6/9/17 END

    # Return the meta_parameter object, function_param name, and function_param_value
    return ModulatedParam(function_mod_meta_param_obj, function_param_name, function_param_value)


# *******************************   get_param_value_for_keyword ********************************************************

def get_param_value_for_keyword(owner, keyword):
    """Return the value for a keyword used by a subclass of Function

    Parameters
    ----------
    owner : Component
    keyword : str

    Returns
    -------
    value

    """
    try:
        function_val = owner.params[FUNCTION]
        if function_val is None:
            # paramsCurrent will go directly to an attribute value first before
            # returning what's actually in its dictionary, so fall back
            try:
                keyval = owner.params.data[FUNCTION].keyword(owner, keyword)
            except KeyError:
                keyval = None
        else:
            keyval = function_val.keyword(owner, keyword)
        return keyval
    except FunctionError as e:
        # assert(False)
        # prefs is not always created when this is called, so check
        try:
            owner.prefs
            has_prefs = True
        except AttributeError:
            has_prefs = False

        if has_prefs and owner.prefs.verbosePref:
            print("{} of {}".format(e, owner.name))
        # return None
        else:
            raise FunctionError(e)
    except AttributeError:
        # prefs is not always created when this is called, so check
        try:
            owner.prefs
            has_prefs = True
        except AttributeError:
            has_prefs = False

        if has_prefs and owner.prefs.verbosePref:
            print("Keyword ({}) not recognized for {}".format(keyword, owner.name))
        return None


def get_param_value_for_function(owner, function):
    try:
        return owner.paramsCurrent[FUNCTION].param_function(owner, function)
    except FunctionError as e:
        if owner.prefs.verbosePref:
            print("{} of {}".format(e, owner.name))
        return None
    except AttributeError:
        if owner.prefs.verbosePref:
            print("Function ({}) can't be evaluated for {}".format(function, owner.name))
        return None


class Function_Base(Function):
    """
    Function_Base(           \
         default_variable,   \
         params=None,        \
         owner=None,         \
         name=None,          \
         prefs=None          \
    )

    Implement abstract class for Function category of Component class

    COMMENT:
        Description:
            Functions are used to "wrap" functions used used by other components;
            They are defined here (on top of standard libraries) to provide a uniform interface for managing parameters
             (including defaults)
            NOTE:   the Function category definition serves primarily as a shell, and as an interface to the Function
                       class, to maintain consistency of structure with the other function categories;
                    it also insures implementation of .function for all Function Components
                    (as distinct from other Function subclasses, which can use a FUNCTION param
                        to implement .function instead of doing so directly)
                    Function Components are the end of the recursive line; as such:
                        they don't implement functionParams
                        in general, don't bother implementing function, rather...
                        they rely on Function_Base.function which passes on the return value of .function

        Variable and Parameters:
        IMPLEMENTATION NOTE:  ** DESCRIBE VARIABLE HERE AND HOW/WHY IT DIFFERS FROM PARAMETER
            - Parameters can be assigned and/or changed individually or in sets, by:
              - including them in the initialization call
              - calling the _instantiate_defaults method (which changes their default values)
              - including them in a call the function method (which changes their values for just for that call)
            - Parameters must be specified in a params dictionary:
              - the key for each entry should be the name of the parameter (used also to name associated Projections)
              - the value for each entry is the value of the parameter

        Return values:
            The functionOutputType can be used to specify type conversion for single-item return values:
            - it can only be used for numbers or a single-number list; other values will generate an exception
            - if self.functionOutputType is set to:
                FunctionOutputType.RAW_NUMBER, return value is "exposed" as a number
                FunctionOutputType.NP_1D_ARRAY, return value is 1d np.array
                FunctionOutputType.NP_2D_ARRAY, return value is 2d np.array
            - it must be enabled for a subclass by setting params[FUNCTION_OUTPUT_TYPE_CONVERSION] = True
            - it must be implemented in the execute method of the subclass
            - see Linear for an example

        MechanismRegistry:
            All Function functions are registered in FunctionRegistry, which maintains a dict for each subclass,
              a count for all instances of that type, and a dictionary of those instances

        Naming:
            Function functions are named by their componentName attribute (usually = componentType)

        Class attributes:
            + componentCategory: kwComponentCategory
            + className (str): kwMechanismFunctionCategory
            + suffix (str): " <className>"
            + registry (dict): FunctionRegistry
            + classPreference (PreferenceSet): ComponentPreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.CATEGORY
            + paramClassDefaults (dict): {FUNCTION_OUTPUT_TYPE_CONVERSION: :keyword:`False`}

        Class methods:
            none

        Instance attributes:
            + componentType (str):  assigned by subclasses
            + componentName (str):   assigned by subclasses
            + variable (value) - used as input to function's execute method
            + paramInstanceDefaults (dict) - defaults for instance (created and validated in Components init)
            + paramsCurrent (dict) - set currently in effect
            + value (value) - output of execute method
            + name (str) - if not specified as an arg, a default based on the class is assigned in register_category
            + prefs (PreferenceSet) - if not specified as an arg, default is created by copying ComponentPreferenceSet

        Instance methods:
            The following method MUST be overridden by an implementation in the subclass:
            - execute(variable, params)
            The following can be implemented, to customize validation of the function variable and/or params:
            - [_validate_variable(variable)]
            - [_validate_params(request_set, target_set, context)]
    COMMENT

    Arguments
    ---------

    variable : value : default ClassDefaults.variable
        specifies the format and a default value for the input to `function <Function>`.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).


    Attributes
    ----------

    variable: value
        format and default value can be specified by the :keyword:`variable` argument of the constructor;  otherwise,
        they are specified by the Function's :keyword:`ClassDefaults.variable`.

    function : function
        called by the Function's `owner <Function_Base.owner>` when it is executed.

    COMMENT:
    functionOutputTypeConversion : Bool : False
        specifies whether `function output type conversion <Function_Output_Type_Conversion>` is enabled.

    functionOutputType : FunctionOutputType : None
        used to specify the return type for the `function <Function_Base.function>`;  `functionOuputTypeConversion`
        must be enabled and implemented for the class (see `FunctionOutputType <Function_Output_Type_Conversion>`
        for details).
    COMMENT

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    """

    componentCategory = kwComponentCategory
    className = componentCategory
    suffix = " " + className

    registry = FunctionRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY

    variableClassDefault_locked = False

    class ClassDefaults(Function.ClassDefaults):
        variable = np.array([0])

    # Note: the following enforce encoding as 1D np.ndarrays (one array per variable)
    variableEncodingDim = 1

    paramClassDefaults = Function.paramClassDefaults.copy()
    paramClassDefaults.update({
        FUNCTION_OUTPUT_TYPE_CONVERSION: False,  # Enable/disable output type conversion
        FUNCTION_OUTPUT_TYPE:None                # Default is to not convert
    })

    def __init__(self,
                 default_variable,
                 params,
                 function=None,
                 owner=None,
                 name=None,
                 prefs=None,
                 context=None):
        """Assign category-level preferences, register category, and call super.__init__

        Initialization arguments:
        - default_variable (anything): establishes type for the variable, used for validation
        - params_default (dict): assigned as paramInstanceDefaults
        Note: if parameter_validation is off, validation is suppressed (for efficiency) (Function class default = on)

        :param default_variable: (anything but a dict) - value to assign as self.instance_defaults.variable
        :param params: (dict) - params to be assigned to paramInstanceDefaults
        :param log: (ComponentLog enum) - log entry types set in self.componentLog
        :param name: (string) - optional, overrides assignment of default (componentName of subclass)
        :return:
        """

        if context != ContextFlags.CONSTRUCTOR:
            raise FunctionError("Direct call to abstract class Function() is not allowed; use a Function subclass")

        self._functionOutputType = None
        # self.name = self.componentName

        register_category(entry=self,
                          base_class=Function_Base,
                          registry=FunctionRegistry,
                          name=name,
                          context=context)
        self.owner = owner

        super().__init__(default_variable=default_variable,
                         function=function,
                         param_defaults=params,
                         name=name,
                         prefs=prefs)

    def _parse_arg_generic(self, arg_val):
        if isinstance(arg_val, list):
            return np.asarray(arg_val)
        else:
            return arg_val

    def _validate_parameter_spec(self, param, param_name, numeric_only=True):
        """Validates function param
        Replace direct call to parameter_spec in tc, which seems to not get called by Function __init__()'s"""
        if not parameter_spec(param, numeric_only):
            owner_name = 'of ' + self.owner_name if self.owner else ""
            raise FunctionError("{} is not a valid specification for the {} argument of {}{}".
                                format(param, param_name, self.__class__.__name__, owner_name))

    def get_current_function_param(self, param_name):
        if param_name == "variable":
            raise FunctionError("The method 'get_current_function_param' is intended for retrieving the current value "
                                "of a function parameter. 'variable' is not a function parameter. If looking for {}'s "
                                "default variable, try {}.instance_defaults.variable.".format(self.name, self.name))
        try:
            return self.owner._parameter_states[param_name].value
        except (AttributeError, TypeError):
            try:
                return self._parameter_states[param_name].value
            except (AttributeError, TypeError):

                return getattr(self, param_name)

    @property
    def functionOutputType(self):
        if hasattr(self, FUNCTION_OUTPUT_TYPE_CONVERSION):
            return self._functionOutputType
        return None

    @functionOutputType.setter
    def functionOutputType(self, value):

        # Initialize backing field if it has not yet been set
        #    ??or if FunctionOutputTypeConversion is False?? <- FIX: WHY?? [IS THAT A SIDE EFFECT OR PREVIOUSLY USING
        #                                                       FIX: self.paramsCurrent[FUNCTION_OUTPUT_TYPE_CONVERSION]
        #                                                       FIX: TO DECIDE IF ATTRIBUTE EXISTS?
        if value is None and (not hasattr(self, FUNCTION_OUTPUT_TYPE_CONVERSION)
                              or not self.FunctionOutputTypeConversion):
            self._functionOutputType = value
            return

        # Attempt to set outputType but conversion not enabled
        if value and not self.paramsCurrent[FUNCTION_OUTPUT_TYPE_CONVERSION]:
            raise FunctionError("output conversion is not enabled for {0}".format(self.__class__.__name__))

        # Bad outputType specification
        if value and not isinstance(value, FunctionOutputType):
            raise FunctionError("value ({0}) of functionOutputType attribute must be FunctionOutputType for {1}".
                                format(self.functionOutputType, self.__class__.__name__))

        # Can't convert from arrays of length > 1 to number
        if (len(self.instance_defaults.variable) > 1 and (self.functionOutputType is FunctionOutputType.RAW_NUMBER)):
            raise FunctionError(
                "{0} can't be set to return a single number since its variable has more than one number".
                format(self.__class__.__name__))
        self._functionOutputType = value

    def show_params(self):
        print("\nParams for {} ({}):".format(self.name, self.componentName))
        for param_name, param_value in sorted(self.user_params.items()):
            print("\t{}: {}".format(param_name, param_value))
        print('')

    @property
    def owner_name(self):
        try:
            return self.owner.name
        except AttributeError:
            return '<no owner>'

# *****************************************   EXAMPLE FUNCTION   *******************************************************

PROPENSITY = "PROPENSITY"
PERTINACITY = "PERTINACITY"


class ArgumentTherapy(Function_Base):
    """
    ArgumentTherapy(                   \
         variable,                     \
         propensity=Manner.CONTRARIAN, \
         pertinacity=10.0              \
         params=None,                  \
         owner=None,                   \
         name=None,                    \
         prefs=None                    \
         )

    .. _ArgumentTherapist:

    Return `True` or :keyword:`False` according to the manner of the therapist.

    Arguments
    ---------

    variable : boolean or statement that resolves to one : default ClassDefaults.variable
        assertion for which a therapeutic response will be offered.

    propensity : Manner value : default Manner.CONTRARIAN
        specifies preferred therapeutic manner

    pertinacity : float : default 10.0
        specifies therapeutic consistency

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).


    Attributes
    ----------

    variable : boolean
        assertion to which a therapeutic response is made.

    propensity : Manner value : default Manner.CONTRARIAN
        determines therapeutic manner:  tendency to agree or disagree.

    pertinacity : float : default 10.0
        determines consistency with which the manner complies with the propensity.

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).


    """

    # Function componentName and type (defined at top of module)
    componentName = ARGUMENT_THERAPY_FUNCTION
    componentType = EXAMPLE_FUNCTION_TYPE

    classPreferences = {
        kwPreferenceSetName: 'ExampleClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
        kpRuntimeParamStickyAssignmentPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
    }

    # Variable class default
    # This is used both to type-cast the variable, and to initialize instance_defaults.variable
    variableClassDefault_locked = False

    # Mode indicators
    class Manner(Enum):
        OBSEQUIOUS = 0
        CONTRARIAN = 1

    # Param class defaults
    # These are used both to type-cast the params, and as defaults if none are assigned
    #  in the initialization call or later (using either _instantiate_defaults or during a function call)

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({FUNCTION_OUTPUT_TYPE_CONVERSION: True,
                               PARAMETER_STATE_PARAMS: None
                               # PROPENSITY: Manner.CONTRARIAN,
                               # PERTINACITY:  10
                               })

    def __init__(self,
                 default_variable=None,
                 propensity=10.0,
                 pertincacity=Manner.CONTRARIAN,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(propensity=propensity,
                                                  pertinacity=pertincacity,
                                                  params=params)

        # This validates variable and/or params_list if assigned (using _validate_params method below),
        #    and assigns them to paramsCurrent and paramInstanceDefaults;
        #    otherwise, assigns paramClassDefaults to paramsCurrent and paramInstanceDefaults
        # NOTES:
        #    * paramsCurrent can be changed by including params in call to function
        #    * paramInstanceDefaults can be changed by calling assign_default
        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

        self.functionOutputType = None

    def _validate_variable(self, variable, context=None):
        """Validates variable and returns validated value

        This overrides the class method, to perform more detailed type checking
        See explanation in class method.
        Note: this method (or the class version) is called only if the parameter_validation attribute is `True`

        :param variable: (anything but a dict) - variable to be validated:
        :param context: (str)
        :return variable: - validated
        """

        if type(variable) == type(self.ClassDefaults.variable) or \
                (isinstance(variable, numbers.Number) and isinstance(self.ClassDefaults.variable, numbers.Number)):
            return variable
        else:
            raise FunctionError("Variable must be {0}".format(type(self.ClassDefaults.variable)))

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validates variable and /or params and assigns to targets

        This overrides the class method, to perform more detailed type checking
        See explanation in class method.
        Note: this method (or the class version) is called only if the parameter_validation attribute is `True`

        :param request_set: (dict) - params to be validated
        :param target_set: (dict) - destination of validated params
        :return none:
        """

        message = ""

        # Check params
        for param_name, param_value in request_set.items():

            # Check that specified parameter is legal
            if param_name not in request_set.keys():
                message += "{0} is not a valid parameter for {1}".format(param_name, self.name)

            if param_name == PROPENSITY:
                if isinstance(param_value, ArgumentTherapy.Manner):
                    # target_set[self.PROPENSITY] = param_value
                    pass  # This leaves param in request_set, clear to be assigned to target_set in call to super below
                else:
                    message = "Propensity must be of type Example.Mode"
                continue

            # Validate param
            if param_name == PERTINACITY:
                if isinstance(param_value, numbers.Number) and 0 <= param_value <= 10:
                    # target_set[PERTINACITY] = param_value
                    pass  # This leaves param in request_set, clear to be assigned to target_set in call to super below
                else:
                    message += "Pertinacity must be a number between 0 and 10"
                continue

        if message:
            raise FunctionError(message)

        super()._validate_params(request_set, target_set, context)

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """
        Returns a boolean that is (or tends to be) the same as or opposite the one passed in.

        Arguments
        ---------

        variable : boolean : default ClassDefaults.variable
           an assertion to which a therapeutic response is made.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        therapeutic response : boolean

        """
        variable = self._update_variable(self._check_args(variable, params, context))

        # Compute the function
        statement = variable
        propensity = self.get_current_function_param(PROPENSITY)
        pertinacity = self.get_current_function_param(PERTINACITY)
        whim = randint(-10, 10)

        if propensity == self.Manner.OBSEQUIOUS:
            return whim < pertinacity

        elif propensity == self.Manner.CONTRARIAN:
            return whim > pertinacity

        else:
            raise FunctionError("This should not happen if parameter_validation == True;  check its value")


# region ****************************************   FUNCTIONS   ********************************************************
# endregion

# region **********************************  USER-DEFINED FUNCTION  ****************************************************
# endregion

class UserDefinedFunction(Function_Base):
    """
    UserDefinedFunction(        \
         custom_function=None,  \
         default_variable=None, \
         params=None,           \
         owner=None,            \
         name=None,             \
         prefs=None             \
    )

    .. _UDF_Description:

    A UserDefinedFunction (UDF) is used to "wrap" a Python function or method, including a lamdba function,
    as a PsyNeuLink `Function`, so that it can be used as the `function <Component.function>` of a `Component
    <Component>`.  This is done automatically if a Python function or method is assigned as the `function
    <Component.function>` attribute of a Component.  A Python function or method can also be wrapped on its own,
    by calling the UserDefinedFunction constructor, and assigning the Python function or method as its
    **custom_function** argument.  The Python function or method must obey the following conventions to be treated
    correctly as a UserDefinedFunction (UDF):

    .. _UDF_Variable:

    * It must have **at least one argument** (that can be a positional or a keyword argument);  this will be treated
      as the `variable <UserDefinedFunction.variable>` attribute of the UDF's `function <UserDefinedFunction.function>`.
      When the UDF calls the function or method that it wraps, an initial attempt is made to do so with **variable**
      as the name of the first argument; if that fails, it is called positionally.  The argument is always passed as a
      2d np.array, that may contain one or more items (elements in axis 0), depending upon the Component to which the
      UDF is assigned.  It is the user's responsibility to insure that the number of items expected in the first
      argument of the function or method is compatible with the circumstances in which it will be called.
    ..
    .. _UDF_Additional_Arguments:

    * It may have have **any number of additional arguments** (positional and/or keyword);  these are treated as
      parameters of the UDF, and can be modulated by `ModulatorySignals <ModulatorySignal>` like the parameters of
      ordinary PsyNeuLink `Functions <Function>`.  If the UDF is assigned to (or automatically created for) a
      `Mechanism` or `Projection <Projection>`, these parameters are each automatically assigned a `ParameterState`
      so that they can be modulated by `ControlSignals <ControlSignal>` or `LearningSignals <LearningSignal>`,
      respectively.  If the UDF is assigned to (or automatically created for) an `InputState` or `OutputState`,
      and any of the parameters are specified as `Function_Modulatory_Params` (see `below <UDF_Modulatory_Params>`),
      then they can be modulated by `GatingSignals <GatingSignal>`. The function or method wrapped by the UDF is called
      with these parameters by their name and with their current values (i.e., as determined by any
      `ModulatorySignals <ModulatorySignal>` assigned to them).
    ..
    .. _UDF_Params_Context:

    * It may include **context** and **params** arguments;  these are not required, but can be included to receive
      information about the current conditions of execution.  When the function or method is called, an initial attempt
      is made to do so with these arguments; if that fails, it is called again without them.
    ..
    .. _UDF_Modulatory_Params:

    * The parameters of a UDF can be specified as `Function_Modulatory_Params` in a `parameter specification dictionary
      <ParameterState_Specification>` assigned to the **params** argument of the constructor for either the Python
      function or method, or of an explicitly defined UDF (see `examples below <UDF_Modulatory_Params_Examples>`).
      It can include either or both of the following two entries:
         *MULTIPLICATIVE_PARAM*: <parameter name>\n
         *ADDITIVE_PARAM*: <parameter name>
      These are used only when the UDF is assigned as the `function <State_Base.function>` of an InputState or
      OutputState that receives one more more `GatingProjections <GatingProjection>`.

      COMMENT:
      # IMPLEMENT INTERFACE FOR OTHER ModulationParam TYPES (i.e., for ability to add new custom ones)
      COMMENT

    .. tip::
       The format of the `variable <UserDefinedFunction.variable>` passed to the `custom_function
       <UserDefinedFunction.custom_function>` function can be verified by adding a ``print(variable)`` or
       ``print(type(variable))`` statement to the function.

    Examples
    --------

    **Assigning a custom function to a Mechanism**

    .. _UDF_Lambda_Function_Examples:

    The following example assigns a simple lambda function that returns the sum of the elements of a 1d array) to a
    `TransferMechanism`::

        >>> import psyneulink as pnl
        >>> my_mech = pnl.ProcessingMechanism(default_variable=[[0,0,0]],
        ...                                   function=lambda x:sum(x[0]))
        >>> my_mech.execute(input = [1, 2, 3])
        array([[6]])

    Note that the function treats its argument, x, as a 2d array, and accesses its first item for the calculation.
    This is because  the `variable <Mechanism_Base.variable>` of ``my_mech`` is defined in the **size** argument of
    its constructor as having a single item (a 1d array of length 3;  (see `size <Component.size>`).  In the
    following example, a function is defined for a Mechanism in which the variable has two items, that are summed by
    the function::

        >>> my_mech = pnl.ProcessingMechanism(default_variable=[[0],[0]],
        ...                                   function=lambda x: x[0] + x[1])
        >>> my_mech.execute(input = [[1],[2]])
        array([[3]])

    .. _UDF_Defined_Function_Examples:

    The **function** argument can also be assigned a function defined in Python::

        >>> def my_fct(variable):
        ...     return variable[0] + variable[1]
        >>> my_mech = pnl.ProcessingMechanism(default_variable=[[0],[0]],
        ...                                   function=my_fct)

    This will produce the same result as the last example.  This can be useful for assigning the function to more than
    one Component.

    More complicated functions, including ones with more than one parameter can also be used;  for example::

        >>> def my_sinusoidal_fct(input=[[0],[0]],
        ...                       phase=0,
        ...                       amplitude=1):
        ...    frequency = input[0]
        ...    t = input[1]
        ...    return amplitude * np.sin(2 * np.pi * frequency * t + phase)
        >>> my_wave_mech = pnl.ProcessingMechanism(default_variable=[[0],[0]],
        ...                                        function=my_sinusoidal_fct)

    Note that in this example, ``input`` is used as the name of the first argument, instead of ``variable``
    as in the examples above. The name of the first argument of a function to be "wrapped" as a UDF does not matter;
    in general it is good practice to use ``variable``, as the `variable <Component.variable>` of the Component
    to which the UDF is assigned is what is passed to the function as its first argument.  However, if it is helpful to
    name it something else, that is fine.

    Notice also that ``my_sinusoidal_fct`` takes two values in its ``input`` argument, that it assigns to the
    ``frequency`` and ``t`` variables of the function.  While  it could have been specified more compactly as a 1d array
    with two elements (i.e. [0,0]), it is specified in the example as a 2d array with two items to make it clear that
    it matches the format of the **default_variable** for the ProcessingMechanism to which it will be assigned,
    which requires it be formatted this way (since the `variable <Component.variable>` of all Components are converted
    to a 2d array).

    ``my_sinusoidal_fct`` also has two other arguments, ``phase`` and ``amplitude``.   When it is assigned to
    ``my_wave_mech``, those parameters are assigned to `ParameterStates <ParameterState>` of ``my_wave_mech``, which
    that be used to modify their values by `ControlSignals <ControlSignal>` (see `example below <_
    UDF_Control_Signal_Example>`).

    .. _UDF_Explicit_Creation_Examples:

    In all of the examples above, a UDF was automatically created for the functions assigned to the Mechanism.  A UDF
    can also be created explicitly, as follows:

        >>> my_sinusoidal_UDF = pnl.UserDefinedFunction(custom_function=my_sinusoidal_fct)
        >>> my_wave_mech = pnl.ProcessingMechanism(default_variable=[[0],[0]],
        ...                                        function=my_sinusoidal_UDF)

    When the UDF is created explicitly, parameters of the function can be included as arguments to its constructor,
    to assign them default values that differ from the those in the definition of the function, or for parameters
    that don't define default values.  For example::

        >>> my_sinusoidal_UDF = pnl.UserDefinedFunction(custom_function=my_sinusoidal_fct,
        ...                                  phase=10,
        ...                                  amplitude=3)
        >>> my_wave_mech = pnl.ProcessingMechanism(default_variable=[[0],[0]],
        ...                                        function=my_sinusoidal_UDF)

    assigns ``my_sinusoidal_fct`` as the `function <Mechanism_Base.function>` for ``my_mech``, but with the default
    values of its ``phase`` and ``amplitude`` parameters assigned new values.  This can be useful for assigning the
    same function to different Mechanisms with different default values.

    .. _UDF_Control_Signal_Example:

    Explicitly defining the UDF can also be used to `specify control <ControlSignal_Specification>` for parameters of
    the function, as in the following example::

        >>> my_mech = pnl.ProcessingMechanism(default_variable=[[0],[0]],
        ...                                   function=UserDefinedFunction(custom_function=my_sinusoidal_fct,
        ...                                                                amplitude=(1.0, pnl.CONTROL)))

    This specifies that the default value of the ``amplitude`` parameter of ``my_sinusoidal_fct`` be ``1.0``, but
    its value should be modulated by a `ControlSignal`.

    COMMENT:
    Note:  if a function explicitly defined in a UDF does not assign a default value to its first argument (i.e.,
    it is a positional argument), then the UDF that must define the variable, as in:

    Note:  if the function does not assign a default value to its first argument i.e., it is a positional arg),
    then if it is explicitly wrapped in a UDF that must define the variable, as in:
        xxx my_mech = pnl.ProcessingMechanism(default_variable=[[0],[0]],
        ...                                   function=UserDefinedFunction(default_variable=[[0],[0]],
        ...                                                                custom_function=my_sinusoidal_fct,
        ...                                                                amplitude=(1.0, pnl.CONTROL)))

    This is required so that the format of the variable can be checked for compatibilty with other Components
    with which it interacts.

    .. note::
       Built-in Python functions and methods (including numpy functions) cannot be assigned to a UDF

    COMMENT

    Custom functions can be as elaborate as desired, and can even include other PsyNeuLink `Functions <Function>`
    indirectly, such as::

        >>> import psyneulink as pnl
        >>> L = pnl.Logistic(gain = 2)
        >>> def my_fct(variable):
        ...     return L.function(variable) + 2
        >>> my_mech = pnl.ProcessingMechanism(size = 3, function = my_fct)
        >>> my_mech.execute(input = [1, 2, 3])  #doctest: +SKIP
        array([[2.88079708, 2.98201379, 2.99752738]])


    .. _UDF_Assign_to_State_Examples:

    **Assigning of a custom function to a State**

    A custom function can also be assigned as the `function <State_Base.function>` of an `InputState` or `OutputState`.
    For example, the following assigns ``my_sinusoidal_fct`` to the `function <OutputState.function>` of an OutputState
    of ``my_mech``, rather the Mechanism's `function <Mechanism_Base.function>`::

        >>> my_wave_mech = pnl.ProcessingMechanism(size=1,
        ...                                        function=pnl.Linear,
        ...                                        output_states=[{pnl.NAME: 'SINUSOIDAL OUTPUT',
        ...                                                       pnl.VARIABLE: [(pnl.OWNER_VALUE, 0),pnl.EXECUTION_COUNT],
        ...                                                       pnl.FUNCTION: my_sinusoidal_fct}])

    For details on how to specify a function of an OutputState, see `OutputState Customization <OutputState_Customization>`.
    Below is an example plot of the output of the 'SINUSOIDAL OUTPUT' `OutputState` from my_wave_mech above, as the
    execution count increments, when the input to the mechanism is 0.005 for 1000 runs::

.. figure:: _static/sinusoid_005.png
   :alt: Sinusoid function
   :scale: 50 %

.. _UDF_Modulatory_Params_Examples:

    The parameters of a custom function assigned to an InputState or OutputState can also be used for `gating
    <GatingMechanism_Specifying_Gating>`.  However, this requires that its `Function_Modulatory_Params` be specified
    (see `above <UDF_Modulatory_Params>`). This can be done by including a **params** argument in the definition of
    the function itself::

        >>> def my_sinusoidal_fct(input=[[0],[0]],
        ...                      phase=0,
        ...                      amplitude=1,
        ...                      params={pnl.ADDITIVE_PARAM:'phase',
        ...                              pnl.MULTIPLICATIVE_PARAM:'amplitude'}):
        ...    frequency = input[0]
        ...    t = input[1]
        ...    return amplitude * np.sin(2 * np.pi * frequency * t + phase)

    or in the explicit creation of a UDF::

        >>> my_sinusoidal_UDF = pnl.UserDefinedFunction(custom_function=my_sinusoidal_fct,
        ...                                             phase=0,
        ...                                             amplitude=1,
        ...                                             params={pnl.ADDITIVE_PARAM:'phase',
        ...                                                     pnl.MULTIPLICATIVE_PARAM:'amplitude'})


    The ``phase`` and ``amplitude`` parameters of ``my_sinusoidal_fct`` can now be used as the
    `Function_Modulatory_Params` for gating any InputState or OutputState to which the function is assigned (see
    `GatingMechanism_Specifying_Gating` and `GatingSignal_Examples`).

    **Class Definition:**


    Arguments
    ---------

    COMMENT:
        CW 1/26/18: Again, commented here is the old version, because I'm afraid I may have missed some functionality.
        custom_function : function
        specifies function to "wrap." It can be any function, take any arguments (including standard ones,
        such as :keyword:`params` and :keyword:`context`) and return any value(s), so long as these are consistent
        with the context in which the UserDefinedFunction will be used.
    COMMENT
    custom_function : function
        specifies the function to "wrap." It can be any function or method, including a lambda function;
        see `above <UDF_Description>` for additional details.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the function.
        This can be used to define an `additive_param <UserDefinedFunction.additive_param>` and/or
        `multiplicative_param <UserDefinedFunction.multiplicative_param>` for the UDF, by including one or both
        of the following entries:\n
          *ADDITIVE_PARAM*: <param_name>\n
          *MULTIPLICATIVE_PARAM*: <param_name>\n
        Values specified for parameters in the dictionary override any assigned to those parameters in arguments of
        the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable: value
        format and default value of the function "wrapped" by the UDF.

    custom_function : function
        the user-specified function: called by the Function's `owner <Function_Base.owner>` when it is executed.

    additive_param : str
        this contains the name of the additive_param, if one has been specified for the UDF
        (see `above <UDF_Modulatory_Params>` for details).

    multiplicative_param : str
        this contains the name of the multiplicative_param, if one has been specified for the UDF
        (see `above <UDF_Modulatory_Params>` for details).

    COMMENT:
    functionOutputTypeConversion : Bool : False
        specifies whether `function output type conversion <Function_Output_Type_Conversion>` is enabled.

    functionOutputType : FunctionOutputType : None
        used to specify the return type for the `function <Function_Base.function>`;  `functionOuputTypeConversion`
        must be enabled and implemented for the class (see `FunctionOutputType <Function_Output_Type_Conversion>`
        for details).
    COMMENT

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the Function; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    """
    componentName = USER_DEFINED_FUNCTION
    componentType = USER_DEFINED_FUNCTION_TYPE

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        FUNCTION_OUTPUT_TYPE_CONVERSION: False,
        PARAMETER_STATE_PARAMS: None,
        CUSTOM_FUNCTION: None,
        MULTIPLICATIVE_PARAM:None,
        ADDITIVE_PARAM:None
    })

    @tc.typecheck
    def __init__(self,
                 custom_function=None,
                 default_variable=None,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 **kwargs):

        def get_cust_fct_args(custom_function):
            """Get args of custom_function
            Return:
                - value of first arg (to be used as default_variable for UDF)
                - dict with all others (to be assigned as params of UDF)
                - dict with default values (from function definition, else set to None)
            """
            from inspect import signature, _empty
            try:
                arg_names = custom_function.__code__.co_varnames
            except AttributeError:
                raise FunctionError("Can't get __code__ for custom_function")
            args = {}
            defaults = {}
            for arg_name, arg in signature(custom_function).parameters.items():
                # Use definition from the function as default;
                #    this allows UDF to assign a value for this instance (including a MODULATORY spec)
                #    while assigning an actual value to paramClassDefaults (in _assign_args_to_params_dicts);
                if arg.default is _empty:
                    defaults[arg_name] = None
                else:
                    defaults[arg_name] = arg.default
                # If arg is specified in the constructor for the UDF, assign that as its value
                if arg_name in kwargs:
                    args[arg_name] = kwargs[arg_name]
                # Otherwise, use the default value from the definition of the function
                else:
                    args[arg_name] = defaults[arg_name]

            # Assign default value of first arg as variable and remove from dict
            variable = args[arg_names[0]]
            if variable is _empty:
                variable = None
            del args[arg_names[0]]

            return variable, args, defaults

        # Get variable and names of other any other args for custom_function and assign to cust_fct_params
        if params is not None and CUSTOM_FUNCTION in params:
            custom_function = params[CUSTOM_FUNCTION]
        try:
            cust_fct_variable, self.cust_fct_params, defaults = get_cust_fct_args(custom_function)
        except FunctionError:
            raise FunctionError("Assignment of a built-in function or method ({}) to a {} is not supported".
                                format(custom_function, self.__class__.__name__))

        if PARAMS in self.cust_fct_params:
            if self.cust_fct_params[PARAMS]:
                if params:
                    params.update(self.cust_fct_params)
                else:
                    params = self.cust_fct_params[PARAMS]
            del self.cust_fct_params[PARAMS]

        if CONTEXT in self.cust_fct_params:
            if self.cust_fct_params[CONTEXT]:
                context = self.cust_fct_params[CONTEXT]
            del self.cust_fct_params[CONTEXT]

        # Assign variable to default_variable if default_variable was not specified
        if default_variable is None:
            default_variable = cust_fct_variable
        elif cust_fct_variable and not iscompatible(default_variable, cust_fct_variable):
            raise FunctionError("Value passed as \'default_variable\' for {} ({}) of {} ({}) "
                                "conflicts with specification of first argument in constructor for {} itself ({})".
                                format(self.__class__.__name__, custom_function.__name__,
                                       owner.name, default_variable, custom_function.__name__, cust_fct_variable))


        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(custom_function=custom_function,
                                                  params=params,
                                                  defaults=defaults,
                                                  **self.cust_fct_params
                                                  )

        super().__init__(default_variable=default_variable,
                         function=custom_function,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

        self.functionOutputType = None

    def function(self, **kwargs):

        # Update value of parms in cust_fct_params
        for param in self.cust_fct_params:
            # First check for value passed in params as runtime param:
            if PARAMS in kwargs and kwargs[PARAMS] is not None and param in kwargs[PARAMS]:
                self.cust_fct_params[param] = kwargs[PARAMS][param]
            else:
            # Otherwise, get current value from ParameterState (in case it is being modulated by ControlSignal(s)
                self.cust_fct_params[param] = self.get_current_function_param(param)
        kwargs.update(self.cust_fct_params)

        try:
            # Try calling with full list of args (including context and params)
            return self.custom_function(**kwargs)
        except TypeError:
            # Try calling with just variable and cust_fct_params
            return self.custom_function(kwargs[VARIABLE], **self.cust_fct_params)


# region **********************************  COMBINATION FUNCTIONS  ****************************************************
# endregion


class CombinationFunction(Function_Base):
    """Function that combines multiple items, yielding a result with the same shape as its operands

    All CombinationFunctions must have two attributes - multiplicative_param and additive_param -
        each of which is assigned the name of one of the function's parameters;
        this is for use by ModulatoryProjections (and, in particular, GatingProjections,
        when the CombinationFunction is used as the function of an InputState or OutputState).

    """
    componentType = COMBINATION_FUNCTION_TYPE

    class ClassDefaults(Function_Base.ClassDefaults):
        variable = np.array([0, 0])

    # IMPLEMENTATION NOTE: THESE SHOULD SHOULD BE REPLACED WITH ABC WHEN IMPLEMENTED
    def __init__(self, default_variable,
                 params,
                 owner,
                 prefs,
                 context):

        if not hasattr(self, MULTIPLICATIVE_PARAM):
            raise FunctionError("PROGRAM ERROR: {} must implement a {} attribute".
                                format(self.__class__.__name__, MULTIPLICATIVE_PARAM))

        if not hasattr(self, ADDITIVE_PARAM):
            raise FunctionError("PROGRAM ERROR: {} must implement an {} attribute".
                                format(self.__class__.__name__, ADDITIVE_PARAM))

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

    @property
    def multiplicative(self):
        return getattr(self, self.multiplicative_param)

    @multiplicative.setter
    def multiplicative(self, val):
        setattr(self, self.multiplicative_param, val)

    @property
    def additive(self):
        return getattr(self, self.additive_param)

    @additive.setter
    def additive(self, val):
        setattr(self, self.additive_param, val)


class Reduce(CombinationFunction):  # ------------------------------------------------------------------------
    # FIX: CONFIRM THAT 1D KWEIGHTS USES EACH ELEMENT TO SCALE CORRESPONDING VECTOR IN VARIABLE
    # FIX  CONFIRM THAT LINEAR TRANSFORMATION (OFFSET, SCALE) APPLY TO THE RESULTING ARRAY
    # FIX: CONFIRM RETURNS LIST IF GIVEN LIST, AND SIMLARLY FOR NP.ARRAY
    """
    Reduce(                                       \
         default_variable=ClassDefaults.variable, \
         weights=None,                            \
         exponents=None,                          \
         operation=SUM,                           \
         scale=1.0,                               \
         offset=0.0,                              \
         params=None,                             \
         owner=None,                              \
         prefs=None,                              \
    )

    .. _Reduce:

    Combine values in each of one or more arrays into a single value for each array, with optional weighting and/or
    exponentiation of each item within an array prior to combining, and scaling and/or offset of result.

    Returns a scalar value for each array of the input.

    COMMENT:
        IMPLEMENTATION NOTE: EXTEND TO MULTIDIMENSIONAL ARRAY ALONG ARBITRARY AXIS
    COMMENT

    Arguments
    ---------

    default_variable : list or np.array : default ClassDefaults.variable
        specifies a template for the value to be transformed and its default value;  all entries must be numeric.

    weights : 1d or 2d np.array : default None
        specifies values used to multiply the elements of each array in `variable  <LinearCombination.variable>`.
        If it is 1d, its length must equal the number of items in `variable <LinearCombination.variable>`;
        if it is 2d, the length of each item must be the same as those in `variable <LinearCombination.variable>`,
        and there must be the same number of items as there are in `variable <LinearCombination.variable>`
        (see `weights <LinearCombination.weights>` for details)

    exponents : 1d or 2d np.array : default None
        specifies values used to exponentiate the elements of each array in `variable  <LinearCombination.variable>`.
        If it is 1d, its length must equal the number of items in `variable <LinearCombination.variable>`;
        if it is 2d, the length of each item must be the same as those in `variable <LinearCombination.variable>`,
        and there must be the same number of items as there are in `variable <LinearCombination.variable>`
        (see `exponents <LinearCombination.exponents>` for details)

    operation : SUM or PRODUCT : default SUM
        specifies whether to sum or multiply the elements in `variable <Reduce.function.variable>` of
        `function <Reduce.function>`.

    scale : float
        specifies a value by which to multiply each element of the output of `function <Reduce.function>`
        (see `scale <Reduce.scale>` for details)

    offset : float
        specifies a value to add to each element of the output of `function <Reduce.function>`
        (see `offset <Reduce.offset>` for details)

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    default_variable : list or np.array
        contains array(s) to be reduced.

    operation : SUM or PRODUCT
        determines whether elements of each array in `variable <Reduce.function.variable>` of
        `function <Reduce.function>` are summmed or multiplied.

    scale : float
        value is applied multiplicatively to each element of the array after applying the `operation <Reduce.operation>`
        (see `scale <Reduce.scale>` for details);  this done before applying the `offset <Reduce.offset>`
        (if it is specified).

    offset : float
        value is added to each element of the array after applying the `operation <Reduce.operation>`
        and `scale <Reduce.scale>` (if it is specified).

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """
    componentName = REDUCE_FUNCTION

    multiplicative_param = SCALE
    additive_param = OFFSET

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 # weights: tc.optional(parameter_spec)=None,
                 # exponents: tc.optional(parameter_spec)=None,
                 weights=None,
                 exponents=None,
                 default_variable=None,
                 operation: tc.enum(SUM, PRODUCT) = SUM,
                 scale: parameter_spec = 1.0,
                 offset: parameter_spec = 0.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(weights=weights,
                                                  exponents=exponents,
                                                  operation=operation,
                                                  scale=scale,
                                                  offset=offset,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

    def _validate_variable(self, variable, context=None):
        """Insure that list or array is 1d and that all elements are numeric

        Args:
            variable:
            context:
        """
        variable = self._update_variable(super()._validate_variable(variable=variable, context=context))
        if not is_numeric(variable):
            raise FunctionError("All elements of {} must be scalar values".
                                format(self.__class__.__name__))
        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate weghts, exponents, scale and offset parameters

        Check that WEIGHTS and EXPONENTS are lists or np.arrays of numbers with length equal to variable.
        Check that SCALE and OFFSET are scalars.

        Note: the checks of compatibility with variable are only performed for validation calls during execution
              (i.e., from check_args(), since during initialization or COMMAND_LINE assignment,
              a parameter may be re-assigned before variable assigned during is known
        """

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if WEIGHTS in target_set and target_set[WEIGHTS] is not None:
            self._validate_parameter_spec(target_set[WEIGHTS], WEIGHTS, numeric_only=True)
            target_set[WEIGHTS] = np.atleast_1d(target_set[WEIGHTS])
            if self.context.execution_phase & (ContextFlags.EXECUTING | ContextFlags.LEARNING):
                if len(target_set[WEIGHTS]) != len(self.instance_defaults.variable):
                    raise FunctionError("Number of weights ({0}) is not equal to number of elements in variable ({1})".
                                        format(len(target_set[WEIGHTS]), len(self.instance_defaults.variable)))

        if EXPONENTS in target_set and target_set[EXPONENTS] is not None:
            self._validate_parameter_spec(target_set[EXPONENTS], EXPONENTS, numeric_only=True)
            target_set[EXPONENTS] = np.atleast_1d(target_set[EXPONENTS])
            if self.context.execution_phase & (ContextFlags.EXECUTING | ContextFlags.LEARNING):
                if len(target_set[EXPONENTS]) != len(self.instance_defaults.variable):
                    raise FunctionError("Number of exponents ({0}) does not equal number of elements in variable ({1})".
                                        format(len(target_set[EXPONENTS]), len(self.instance_defaults.variable)))

        if SCALE in target_set and target_set[SCALE] is not None:
            scale = target_set[SCALE]
            if not isinstance(scale, numbers.Number):
                raise FunctionError("{} param of {} ({}) must be a scalar".format(SCALE, self.name, scale))

        if OFFSET in target_set and target_set[OFFSET] is not None:
            offset = target_set[OFFSET]
            if not isinstance(offset, numbers.Number):
                raise FunctionError("{} param of {} ({}) must be a scalar".format(OFFSET, self.name, offset))

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """
        Calculate sum or product of the elements for each array in `variable <Reduce.variable>`,
        apply `scale <Reduce.scale>` and/or `offset <Reduce.offset>`, and return array of resulting values.

        Arguments
        ---------

        variable : list or np.array : default ClassDefaults.variable
           a list or np.array of numeric values.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        Sum or product of arrays in variable : np.array
            in an array that is one dimension less than `variable <Reduce.variable>`.


        """

        # Validate variable and assign to variable, and validate params
        variable = self._update_variable(self._check_args(variable=variable,
                                                                     params=params,
                                                                     context=context))

        weights = self.get_current_function_param(WEIGHTS)
        exponents = self.get_current_function_param(EXPONENTS)
        operation = self.get_current_function_param(OPERATION)
        scale = self.get_current_function_param(SCALE)
        offset = self.get_current_function_param(OFFSET)


        # FIX FOR EFFICIENCY: CHANGE THIS AND WEIGHTS TO TRY/EXCEPT // OR IS IT EVEN NECESSARY, GIVEN VALIDATION ABOVE??
        # Apply exponents if they were specified
        if exponents is not None:
            # Avoid divide by zero warning:
            #    make sure there are no zeros for an element that is assigned a negative exponent
            # Allow during initialization because 0s are common in default_variable argument
            if self.context.initialization_status == ContextFlags.INITIALIZING:
                with np.errstate(divide='raise'):
                    try:
                        variable = self._update_variable(variable ** exponents)
                    except FloatingPointError:
                        variable = self._update_variable(np.ones_like(variable))
            else:
                # if this fails with FloatingPointError it should not be caught outside of initialization
                variable = self._update_variable(variable ** exponents)

        # Apply weights if they were specified
        if weights is not None:
            variable = self._update_variable(variable * weights)

        # Calculate using relevant aggregation operation and return
        if operation is SUM:
            # result = np.sum(np.atleast_2d(variable), axis=0) * scale + offset
            result = np.sum(np.atleast_2d(variable), axis=1) * scale + offset
        elif operation is PRODUCT:
            result = np.product(np.atleast_2d(variable), axis=1) * scale + offset
        else:
            raise FunctionError("Unrecognized operator ({0}) for Reduce function".
                                format(self.get_current_function_param(OPERATION)))
        return result


class LinearCombination(CombinationFunction):  # ------------------------------------------------------------------------
    """
    LinearCombination(     \
         default_variable, \
         weights=None,     \
         exponents=None,   \
         operation=SUM,    \
         scale=None,       \
         offset=None,      \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _LinearCombination:

    Linearly combine arrays of values, with optional weighting and/or exponentiation of each array prior to combining,
    and scaling and/or offset of result.

    Combines the arrays in the items of the `variable <LinearCombination.variable>` argument.  Each array can be
    individually weighted and/or exponentiated; they can combined additively or multiplicatively; and the resulting
    array can be multiplicatively transformed and/or additively offset.

    COMMENT:
        Description:
            Combine corresponding elements of arrays in variable arg, using arithmetic operation determined by OPERATION
            Use optional SCALE and OFFSET parameters to linearly transform the resulting array
            Returns a list or 1D array of the same length as the individual ones in the variable

            Notes:
            * If variable contains only a single array, it is simply linearly transformed using SCALE and OFFSET
            * If there is more than one array in variable, they must all be of the same length
            * WEIGHTS and EXPONENTS can be:
                - 1D: each array in variable is scaled by the corresponding element of WEIGHTS or EXPONENTS
                - 2D: each array in variable is scaled by (Hadamard-wise) corresponding array of WEIGHTS or EXPONENTS
        Initialization arguments:
         - variable (value, np.ndarray or list): values to be combined;
             can be a list of lists, or a 1D or 2D np.array;  a 1D np.array is always returned
             if it is a list, it must be a list of numbers, lists, or np.arrays
             all items in the list or 2D np.array must be of equal length
             + WEIGHTS (list of numbers or 1D np.array): multiplies each item of variable before combining them
                  (default: [1,1])
             + EXPONENTS (list of numbers or 1D np.array): exponentiates each item of variable before combining them
                  (default: [1,1])
         - params (dict) can include:
             + WEIGHTS (list of numbers or 1D np.array): multiplies each variable before combining them (default: [1,1])
             + OFFSET (value): added to the result (after the arithmetic operation is applied; default is 0)
             + SCALE (value): multiples the result (after combining elements; default: 1)
             + OPERATION (Operation Enum) - method used to combine terms (default: SUM)
                  SUM: element-wise sum of the arrays in variable
                  PRODUCT: Hadamard Product of the arrays in variable

        LinearCombination.function returns combined values:
        - single number if variable was a single number
        - list of numbers if variable was list of numbers
        - 1D np.array if variable was a single np.variable or np.ndarray
    COMMENT

    Arguments
    ---------

    variable : 1d or 2d np.array : default ClassDefaults.variable
        specifies a template for the arrays to be combined.  If it is 2d, all items must have the same length.

    weights : scalar or 1d or 2d np.array : default None
        specifies values used to multiply the elements of each array in **variable**.
        If it is 1d, its length must equal the number of items in `variable <LinearCombination.variable>`;
        if it is 2d, the length of each item must be the same as those in `variable <LinearCombination.variable>`,
        and there must be the same number of items as there are in `variable <LinearCombination.variable>`
        (see `weights <LinearCombination.weights>` for details of how weights are applied).

    exponents : scalar or 1d or 2d np.array : default None
        specifies values used to exponentiate the elements of each array in `variable  <LinearCombination.variable>`.
        If it is 1d, its length must equal the number of items in `variable <LinearCombination.variable>`;
        if it is 2d, the length of each item must be the same as those in `variable <LinearCombination.variable>`,
        and there must be the same number of items as there are in `variable <LinearCombination.variable>`
        (see `exponents <LinearCombination.exponents>` for details of how exponents are applied).

    operation : SUM or PRODUCT : default SUM
        specifies whether the `function <LinearCombination.function>` takes the elementwise (Hadamarad)
        sum or product of the arrays in `variable  <LinearCombination.variable>`.

    scale : float or np.ndarray : default None
        specifies a value by which to multiply each element of the result of `function <LinearCombination.function>`
        (see `scale <LinearCombination.scale>` for details)

    offset : float or np.ndarray : default None
        specifies a value to add to each element of the result of `function <LinearCombination.function>`
        (see `offset <LinearCombination.offset>` for details)

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : 1d or 2d np.array
        contains the arrays to be combined by `function <LinearCombination>`.  If it is 1d, the array is simply
        linearly transformed by and `scale <LinearCombination.scale>` and `offset <LinearCombination.scale>`.
        If it is 2d, the arrays (all of which must be of equal length) are weighted and/or exponentiated as
        specified by `weights <LinearCombination.weights>` and/or `exponents <LinearCombination.exponents>`
        and then combined as specified by `operation <LinearCombination.operation>`.

    weights : scalar or 1d or 2d np.array
        if it is a scalar, the value is used to multiply all elements of all arrays in `variable
        <LinearCombination.variable>`; if it is a 1d array, each element is used to multiply all elements in the
        corresponding array of `variable <LinearCombination.variable>`;  if it is a 2d array, then each array is
        multiplied elementwise (i.e., the Hadamard Product is taken) with the corresponding array of `variable
        <LinearCombinations.variable>`. All `weights` are applied before any exponentiation (if it is specified).

    exponents : scalar or 1d or 2d np.array
        if it is a scalar, the value is used to exponentiate all elements of all arrays in `variable
        <LinearCombination.variable>`; if it is a 1d array, each element is used to exponentiate the elements of the
        corresponding array of `variable <LinearCombinations.variable>`;  if it is a 2d array, the element of each
        array is used to exponentiate the corresponding element of the corresponding array of `variable
        <LinearCombination.variable>`. In either case, all exponents are applied after application of the `weights
        <LinearCombination.weights>` (if any are specified).

    operation : SUM or PRODUCT
        determines whether the `function <LinearCombination.function>` takes the elementwise (Hadamard) sum or
        product of the arrays in `variable  <LinearCombination.variable>`.

    scale : float or np.ndarray
        value is applied multiplicatively to each element of the array after applying the
        `operation <LinearCombination.operation>` (see `scale <LinearCombination.scale>` for details);
        this done before applying the `offset <LinearCombination.offset>` (if it is specified).

    offset : float or np.ndarray
        value is added to each element of the array after applying the `operation <LinearCombination.operation>`
        and `scale <LinearCombination.scale>` (if it is specified).

    COMMENT:
    function : function
        applies the `weights <LinearCombination.weights>` and/or `exponents <LinearCombinations.weights>` to the
        arrays in `variable <LinearCombination.variable>`, then takes their sum or product (as specified by
        `operation <LinearCombination.operation>`), and finally applies `scale <LinearCombination.scale>` and/or
        `offset <LinearCombination.offset>`.

    functionOutputTypeConversion : Bool : False
        specifies whether `function output type conversion <Function_Output_Type_Conversion>` is enabled.

    functionOutputType : FunctionOutputType : None
        used to specify the return type for the `function <Function_Base.function>`;  `functionOuputTypeConversion`
        must be enabled and implemented for the class (see `FunctionOutputType <Function_Output_Type_Conversion>`
        for details).
    COMMENT

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = LINEAR_COMBINATION_FUNCTION

    classPreferences = {
        kwPreferenceSetName: 'LinearCombinationCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
        kpRuntimeParamStickyAssignmentPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
    }

    multiplicative_param = SCALE
    additive_param = OFFSET

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 # weights: tc.optional(parameter_spec)=None,
                 # exponents: tc.optional(parameter_spec)=None,
                 weights=None,
                 exponents=None,
                 operation: tc.enum(SUM, PRODUCT)=SUM,
                 scale=None,
                 offset=None,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(weights=weights,
                                                  exponents=exponents,
                                                  operation=operation,
                                                  scale=scale,
                                                  offset=offset,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

        if self.weights is not None:
            self.weights = np.atleast_2d(self.weights).reshape(-1, 1)
        if self.exponents is not None:
            self.exponents = np.atleast_2d(self.exponents).reshape(-1, 1)

    def _validate_variable(self, variable, context=None):
        """Insure that all items of list or np.ndarray in variable are of the same length

        Args:
            variable:
            context:
        """
        variable = self._update_variable(super()._validate_variable(variable=variable, context=context))
        # FIX: CONVERT TO AT LEAST 1D NP ARRAY IN INIT AND EXECUTE, SO ALWAYS NP ARRAY
        # FIX: THEN TEST THAT SHAPES OF EVERY ELEMENT ALONG AXIS 0 ARE THE SAME
        # FIX; PUT THIS IN DOCUMENTATION
        if isinstance(variable, (list, np.ndarray)):
            if isinstance(variable, np.ndarray) and not variable.ndim:
                return variable
            length = 0
            for i in range(len(variable)):
                if i == 0:
                    continue
                if isinstance(variable[i - 1], numbers.Number):
                    old_length = 1
                else:
                    old_length = len(variable[i - 1])
                if isinstance(variable[i], numbers.Number):
                    new_length = 1
                else:
                    new_length = len(variable[i])
                if old_length != new_length:
                    raise FunctionError("Length of all arrays in variable {0} "
                                        "for {1} must be the same".format(variable,
                                                                          self.__class__.__name__))
        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate weghts, exponents, scale and offset parameters

        Check that WEIGHTS and EXPONENTS are lists or np.arrays of numbers with length equal to variable
        Check that SCALE and OFFSET are either scalars or np.arrays of numbers with length and shape equal to variable

        Note: the checks of compatibility with variable are only performed for validation calls during execution
              (i.e., from check_args(), since during initialization or COMMAND_LINE assignment,
              a parameter may be re-assigned before variable assigned during is known
        """

        # FIX: MAKE SURE THAT IF OPERATION IS SUBTRACT OR DIVIDE, THERE ARE ONLY TWO VECTORS

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if WEIGHTS in target_set and target_set[WEIGHTS] is not None:
            self._validate_parameter_spec(target_set[WEIGHTS], WEIGHTS, numeric_only=True)
            target_set[WEIGHTS] = np.atleast_2d(target_set[WEIGHTS]).reshape(-1, 1)
            if self.context.execution_phase & (ContextFlags.EXECUTING | ContextFlags.LEARNING):
                if len(target_set[WEIGHTS]) != len(self.instance_defaults.variable):
                    raise FunctionError("Number of weights ({0}) is not equal to number of items in variable ({1})".
                                        format(len(target_set[WEIGHTS]), len(self.instance_defaults.variable)))

        if EXPONENTS in target_set and target_set[EXPONENTS] is not None:
            self._validate_parameter_spec(target_set[EXPONENTS], EXPONENTS, numeric_only=True)
            target_set[EXPONENTS] = np.atleast_2d(target_set[EXPONENTS]).reshape(-1, 1)
            if self.context.execution_phase & (ContextFlags.PROCESSING | ContextFlags.LEARNING):
                if len(target_set[EXPONENTS]) != len(self.instance_defaults.variable):
                    raise FunctionError("Number of exponents ({0}) does not equal number of items in variable ({1})".
                                        format(len(target_set[EXPONENTS]), len(self.instance_defaults.variable)))

        if SCALE in target_set and target_set[SCALE] is not None:
            scale = target_set[SCALE]
            if isinstance(scale, numbers.Number):
                pass
            elif isinstance(scale, np.ndarray):
                target_set[SCALE] = np.array(scale)
            else:
                raise FunctionError("{} param of {} ({}) must be a scalar or an np.ndarray".
                                    format(SCALE, self.name, scale))
            scale_is_a_scalar = isinstance(scale, numbers.Number) or (len(scale) == 1) and isinstance(scale[0], numbers.Number)
            if self.context.execution_phase & (ContextFlags.PROCESSING | ContextFlags.LEARNING):
                if not scale_is_a_scalar:
                    err_msg = "Scale is using Hadamard modulation but its shape and/or size (scale shape: {}, size:{})" \
                              " do not match the variable being modulated (variable shape: {}, size: {})".\
                        format(scale.shape, scale.size, self.instance_defaults.variable.shape,
                               self.instance_defaults.variable.size)
                    if len(self.instance_defaults.variable.shape) == 0:
                        raise FunctionError(err_msg)
                    if (scale.shape != self.instance_defaults.variable.shape) and \
                        (scale.shape != self.instance_defaults.variable.shape[1:]):
                        raise FunctionError(err_msg)

        if OFFSET in target_set and target_set[OFFSET] is not None:
            offset = target_set[OFFSET]
            if isinstance(offset, numbers.Number):
                pass
            elif isinstance(offset, np.ndarray):
                target_set[OFFSET] = np.array(offset)
            else:
                raise FunctionError("{} param of {} ({}) must be a scalar or an np.ndarray".
                                    format(OFFSET, self.name, offset))
            offset_is_a_scalar = isinstance(offset, numbers.Number) or (len(offset) == 1) and isinstance(offset[0], numbers.Number)
            if self.context.execution_phase & (ContextFlags.PROCESSING | ContextFlags.LEARNING):
                if not offset_is_a_scalar:
                    err_msg = "Offset is using Hadamard modulation but its shape and/or size (offset shape: {}, size:{})" \
                              " do not match the variable being modulated (variable shape: {}, size: {})".\
                        format(offset.shape, offset.size, self.instance_defaults.variable.shape,
                               self.instance_defaults.variable.size)
                    if len(self.instance_defaults.variable.shape) == 0:
                        raise FunctionError(err_msg)
                    if (offset.shape != self.instance_defaults.variable.shape) and \
                        (offset.shape != self.instance_defaults.variable.shape[1:]):
                        raise FunctionError(err_msg)

            # if not operation:
            #     raise FunctionError("Operation param missing")
            # if not operation == self.Operation.SUM and not operation == self.Operation.PRODUCT:
            #     raise FunctionError("Operation param ({0}) must be Operation.SUM or Operation.PRODUCT".
            #     format(operation))

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """
        Apply `weights <LinearCombination.weights>` and/or `exponents <LinearCombinations.weights>` to the
        arrays in `variable <LinearCombination.variable>`, then take their sum or product (as specified by
        `operation <LinearCombination.operation>`), apply `scale <LinearCombination.scale>` and/or `offset
        <LinearCombination.offset>`, and return the resulting array.

        COMMENT: [SHORTER VERSION]
            Linearly combine multiple arrays, optionally weighted and/or exponentiated, and return optionally scaled
            and/or offset array (see :ref:`above <LinearCombination>` for details of param specifications`).
        COMMENT

        Arguments
        ---------

        variable : 1d or 2d np.array : default ClassDefaults.variable
           a single numeric array, or multiple arrays to be combined; if it is 2d, all arrays must have the same length.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        combined array : 1d np.array
            the result of linearly combining the arrays in `variable <LinearCombination.variable>`.

        """

        # Validate variable and assign to variable, and validate params
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        weights = self.get_current_function_param(WEIGHTS)
        exponents = self.get_current_function_param(EXPONENTS)
        # if self.context.initialization_status == ContextFlags.INITIALIZED:
        #     if weights is not None and weights.shape != variable.shape:
        #         weights = weights.reshape(variable.shape)
        #     if exponents is not None and exponents.shape != variable.shape:
        #         exponents = exponents.reshape(variable.shape)
        operation = self.get_current_function_param(OPERATION)
        scale = self.get_current_function_param(SCALE)
        offset = self.get_current_function_param(OFFSET)

        # QUESTION:  WHICH IS LESS EFFICIENT:
        #                A) UNECESSARY ARITHMETIC OPERATIONS IF SCALE AND/OR OFFSET ARE 1.0 AND 0, RESPECTIVELY?
        #                   (DOES THE COMPILER KNOW NOT TO BOTHER WITH MULT BY 1 AND/OR ADD 0?)
        #                B) EVALUATION OF IF STATEMENTS TO DETERMINE THE ABOVE?
        # IMPLEMENTATION NOTE:  FOR NOW, ASSUME B) ABOVE, AND ASSIGN DEFAULT "NULL" VALUES TO offset AND scale
        if offset is None:
            offset = 0.0

        if scale is None:
            scale = 1.0

        # IMPLEMENTATION NOTE: CONFIRM: SHOULD NEVER OCCUR, AS _validate_variable NOW ENFORCES 2D np.ndarray
        # If variable is 0D or 1D:
        if np_array_less_than_2d(variable):
            return (variable * scale) + offset

        # FIX FOR EFFICIENCY: CHANGE THIS AND WEIGHTS TO TRY/EXCEPT // OR IS IT EVEN NECESSARY, GIVEN VALIDATION ABOVE??
        # Apply exponents if they were specified
        if exponents is not None:
            # Avoid divide by zero warning:
            #    make sure there are no zeros for an element that is assigned a negative exponent
            # Allow during initialization because 0s are common in default_variable argument
            if self.context.initialization_status == ContextFlags.INITIALIZING:
                with np.errstate(divide='raise'):
                    try:
                        variable = self._update_variable(variable ** exponents)
                    except FloatingPointError:
                        variable = self._update_variable(np.ones_like(variable))
            else:
                # if this fails with FloatingPointError it should not be caught outside of initialization
                variable = self._update_variable(variable ** exponents)

        # Apply weights if they were specified
        if weights is not None:
            variable = self._update_variable(variable * weights)

        # CW 3/19/18: a total hack, e.g. to make scale=[4.] turn into scale=4. Used b/c the `scale` ParameterState
        # changes scale's format: e.g. if you write c = pnl.LinearCombination(scale = 4), print(c.scale) returns [4.]
        if isinstance(scale, (list, np.ndarray)):
            if len(scale) == 1 and isinstance(scale[0], numbers.Number):
                scale = scale[0]
        if isinstance(offset, (list, np.ndarray)):
            if len(offset) == 1 and isinstance(offset[0], numbers.Number):
                offset = offset[0]

        # CALCULATE RESULT USING RELEVANT COMBINATION OPERATION AND MODULATION
        if operation is SUM:
            combination = np.sum(variable, axis=0)
        elif operation is PRODUCT:
            combination = np.product(variable, axis=0)
        else:
            raise FunctionError("Unrecognized operator ({0}) for LinearCombination function".
                                format(operation.self.Operation.SUM))
        if isinstance(scale, numbers.Number):
            product = combination * scale
            # Scalar scale and offset
            if isinstance(offset, numbers.Number):
                result = product + offset
            # Scalar scale and Hadamard offset
            else:
                result = np.sum([product, offset], axis=0)
        else:
            hadamard_product = np.product([combination, scale], axis=0)
            # Hadamard scale, scalar offset
            if isinstance(offset, numbers.Number):
                result = hadamard_product + offset
            # Hadamard scale and offset
            else:
                result = np.sum([hadamard_product, offset], axis=0)

        return result

    @property
    def offset(self):
        if not hasattr(self, '_offset'):
            return None
        else:
            return self._offset

    @offset.setter
    def offset(self, val):
        self._offset = val

    @property
    def scale(self):
        if not hasattr(self, '_scale'):
            return None
        else:
            return self._scale

    @scale.setter
    def scale(self, val):
        self._scale = val


class CombineMeans(CombinationFunction):  # ------------------------------------------------------------------------
    # FIX: CONFIRM THAT 1D KWEIGHTS USES EACH ELEMENT TO SCALE CORRESPONDING VECTOR IN VARIABLE
    # FIX  CONFIRM THAT LINEAR TRANSFORMATION (OFFSET, SCALE) APPLY TO THE RESULTING ARRAY
    # FIX: CONFIRM RETURNS LIST IF GIVEN LIST, AND SIMLARLY FOR NP.ARRAY
    """
    CombineMeans(            \
         default_variable, \
         weights=None,     \
         exponents=None,   \
         operation=SUM,    \
         scale=None,       \
         offset=None,      \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _CombineMeans:

    Linearly combines the means of one or more arrays of values with optional scaling and/or offset applied to result.

    Takes the mean of the array in each item of its `variable <CombineMeans.variable>` argument, and combines them
    as specified by the `operation <CombineMeans.operation>` parameter, taking either their sum (the default) or their
    product.  The mean of each array can be individually weighted and/or exponentiated prior to being combined,
    and the resulting scalar can be multiplicatively transformed and/or additively offset.

    COMMENT:
        Description:
            Take means of elements of each array in variable arg,
                and combine using arithmetic operation determined by OPERATION
            Use optional SCALE and OFFSET parameters to linearly transform the resulting array
            Returns a scalar

            Notes:
            * WEIGHTS and EXPONENTS can be:
                - 1D: each array in variable is scaled by the corresponding element of WEIGHTS or EXPONENTS
                - 2D: each array in variable is scaled by (Hadamard-wise) corresponding array of WEIGHTS or EXPONENTS
        Initialization arguments:
         - variable (value, np.ndarray or list): values to be combined;
             can be a list of lists, or a 1D or 2D np.array;  a scalar is always returned
             if it is a list, it must be a list of numbers, lists, or np.arrays
             if WEIGHTS or EXPONENTS are specified, their length along the outermost dimension (axis 0)
                 must equal the number of items in the variable
         - params (dict) can include:
             + WEIGHTS (list of numbers or 1D np.array): multiplies each item of variable before combining them
                  (default: [1,1])
             + EXPONENTS (list of numbers or 1D np.array): exponentiates each item of variable before combining them
                  (default: [1,1])
             + OFFSET (value): added to the result (after the arithmetic operation is applied; default is 0)
             + SCALE (value): multiples the result (after combining elements; default: 1)
             + OPERATION (Operation Enum) - method used to combine the means of the arrays in variable (default: SUM)
                  SUM: sum of the means of the arrays in variable
                  PRODUCT: product of the means of the arrays in variable

        CombineMeans.function returns a scalar value
    COMMENT

    Arguments
    ---------

    variable : 1d or 2d np.array : default ClassDefaults.variable
        specifies a template for the arrays to be combined.  If it is 2d, all items must have the same length.

    weights : 1d or 2d np.array : default None
        specifies values used to multiply the elements of each array in `variable  <CombineMeans.variable>`.
        If it is 1d, its length must equal the number of items in `variable <CombineMeans.variable>`;
        if it is 2d, the length of each item must be the same as those in `variable <CombineMeans.variable>`,
        and there must be the same number of items as there are in `variable <CombineMeans.variable>`
        (see `weights <CombineMeans.weights>` for details)

    exponents : 1d or 2d np.array : default None
        specifies values used to exponentiate the elements of each array in `variable  <CombineMeans.variable>`.
        If it is 1d, its length must equal the number of items in `variable <CombineMeans.variable>`;
        if it is 2d, the length of each item must be the same as those in `variable <CombineMeans.variable>`,
        and there must be the same number of items as there are in `variable <CombineMeans.variable>`
        (see `exponents <CombineMeans.exponents>` for details)

    operation : SUM or PRODUCT : default SUM
        specifies whether the `function <CombineMeans.function>` takes the sum or product of the means of the arrays in
        `variable  <CombineMeans.variable>`.

    scale : float or np.ndarray : default None
        specifies a value by which to multiply the result of `function <CombineMeans.function>`
        (see `scale <CombineMeans.scale>` for details)

    offset : float or np.ndarray : default None
        specifies a value to add to the result of `function <CombineMeans.function>`
        (see `offset <CombineMeans.offset>` for details)

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : 1d or 2d np.array
        contains the arrays to be combined by `function <CombineMeans>`.  If it is 1d, the array is simply
        linearly transformed by and `scale <CombineMeans.scale>` and `offset <CombineMeans.scale>`.
        If it is 2d, the arrays (all of which must be of equal length) are weighted and/or exponentiated as
        specified by `weights <CombineMeans.weights>` and/or `exponents <CombineMeans.exponents>`
        and then combined as specified by `operation <CombineMeans.operation>`.

    weights : 1d or 2d np.array : default NOne
        if it is 1d, each element is used to multiply all elements in the corresponding array of
        `variable <CombineMeans.variable>`;    if it is 2d, then each array is multiplied elementwise
        (i.e., the Hadamard Product is taken) with the corresponding array of `variable <CombineMeanss.variable>`.
        All :keyword:`weights` are applied before any exponentiation (if it is specified).

    exponents : 1d or 2d np.array : default None
        if it is 1d, each element is used to exponentiate the elements of the corresponding array of
        `variable <CombineMeans.variable>`;  if it is 2d, the element of each array is used to exponentiate
        the corresponding element of the corresponding array of `variable <CombineMeans.variable>`.
        In either case, exponentiating is applied after application of the `weights <CombineMeans.weights>`
        (if any are specified).

    operation : SUM or PRODUCT : default SUM
        determines whether the `function <CombineMeans.function>` takes the elementwise (Hadamard) sum or
        product of the arrays in `variable  <CombineMeans.variable>`.

    scale : float or np.ndarray : default None
        value is applied multiplicatively to each element of the array after applying the
        `operation <CombineMeans.operation>` (see `scale <CombineMeans.scale>` for details);
        this done before applying the `offset <CombineMeans.offset>` (if it is specified).

    offset : float or np.ndarray : default None
        value is added to each element of the array after applying the `operation <CombineMeans.operation>`
        and `scale <CombineMeans.scale>` (if it is specified).

    COMMENT:
    function : function
        applies the `weights <CombineMeans.weights>` and/or `exponents <CombineMeanss.weights>` to the
        arrays in `variable <CombineMeans.variable>`, then takes their sum or product (as specified by
        `operation <CombineMeans.operation>`), and finally applies `scale <CombineMeans.scale>` and/or
        `offset <CombineMeans.offset>`.

    functionOutputTypeConversion : Bool : False
        specifies whether `function output type conversion <Function_Output_Type_Conversion>` is enabled.

    functionOutputType : FunctionOutputType : None
        used to specify the return type for the `function <Function_Base.function>`;  `functionOuputTypeConversion`
        must be enabled and implemented for the class (see `FunctionOutputType <Function_Output_Type_Conversion>`
        for details).
    COMMENT

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = COMBINE_MEANS_FUNCTION

    classPreferences = {
        kwPreferenceSetName: 'CombineMeansCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
        kpRuntimeParamStickyAssignmentPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
    }

    multiplicative_param = SCALE
    additive_param = OFFSET

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 # weights:tc.optional(parameter_spec)=None,
                 # exponents:tc.optional(parameter_spec)=None,
                 weights=None,
                 exponents=None,
                 operation: tc.enum(SUM, PRODUCT)=SUM,
                 scale=None,
                 offset=None,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(weights=weights,
                                                  exponents=exponents,
                                                  operation=operation,
                                                  scale=scale,
                                                  offset=offset,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

        if self.weights is not None:
            self.weights = np.atleast_2d(self.weights).reshape(-1, 1)
        if self.exponents is not None:
            self.exponents = np.atleast_2d(self.exponents).reshape(-1, 1)

    def _validate_variable(self, variable, context=None):
        """Insure that all items of variable are numeric
        """
        variable = self._update_variable(super()._validate_variable(variable=variable, context=context))
        # if any(not is_numeric(item) for item in variable):
        #     raise FunctionError("All items of the variable for {} must be numeric".format(self.componentName))
        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate weights, exponents, scale and offset parameters

        Check that WEIGHTS and EXPONENTS are lists or np.arrays of numbers with length equal to variable
        Check that SCALE and OFFSET are either scalars or np.arrays of numbers with length and shape equal to variable

        Note: the checks of compatibility with variable are only performed for validation calls during execution
              (i.e., from check_args(), since during initialization or COMMAND_LINE assignment,
              a parameter may be re-assigned before variable assigned during is known
        """

        # FIX: MAKE SURE THAT IF OPERATION IS SUBTRACT OR DIVIDE, THERE ARE ONLY TWO VECTORS

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if WEIGHTS in target_set and target_set[WEIGHTS] is not None:
            target_set[WEIGHTS] = np.atleast_2d(target_set[WEIGHTS]).reshape(-1, 1)
            if self.context.execution_phase & (ContextFlags.PROCESSING | ContextFlags.LEARNING):
                if len(target_set[WEIGHTS]) != len(self.instance_defaults.variable):
                    raise FunctionError("Number of weights ({0}) is not equal to number of items in variable ({1})".
                                        format(len(target_set[WEIGHTS]), len(self.instance_defaults.variable.shape)))

        if EXPONENTS in target_set and target_set[EXPONENTS] is not None:
            target_set[EXPONENTS] = np.atleast_2d(target_set[EXPONENTS]).reshape(-1, 1)
            if self.context.execution_phase & (ContextFlags.PROCESSING | ContextFlags.LEARNING):
                if len(target_set[EXPONENTS]) != len(self.instance_defaults.variable):
                    raise FunctionError("Number of exponents ({0}) does not equal number of items in variable ({1})".
                                        format(len(target_set[EXPONENTS]), len(self.instance_defaults.variable.shape)))

        if SCALE in target_set and target_set[SCALE] is not None:
            scale = target_set[SCALE]
            if isinstance(scale, numbers.Number):
                pass
            elif isinstance(scale, np.ndarray):
                target_set[SCALE] = np.array(scale)
            else:
                raise FunctionError("{} param of {} ({}) must be a scalar or an np.ndarray".
                                    format(SCALE, self.name, scale))
            if self.context.execution_phase & (ContextFlags.PROCESSING | ContextFlags.LEARNING):
                if (isinstance(scale, np.ndarray) and
                        (scale.size != self.instance_defaults.variable.size or
                         scale.shape != self.instance_defaults.variable.shape)):
                    raise FunctionError("Scale is using Hadamard modulation "
                                        "but its shape and/or size (shape: {}, size:{}) "
                                        "do not match the variable being modulated (shape: {}, size: {})".
                                        format(scale.shape, scale.size, self.instance_defaults.variable.shape, self.instance_defaults.variable.size))

        if OFFSET in target_set and target_set[OFFSET] is not None:
            offset = target_set[OFFSET]
            if isinstance(offset, numbers.Number):
                pass
            elif isinstance(offset, np.ndarray):
                target_set[OFFSET] = np.array(offset)
            else:
                raise FunctionError("{} param of {} ({}) must be a scalar or an np.ndarray".
                                    format(OFFSET, self.name, offset))
            if self.context.execution_phase & (ContextFlags.PROCESSING | ContextFlags.LEARNING):
                if (isinstance(offset, np.ndarray) and
                        (offset.size != self.instance_defaults.variable.size or
                         offset.shape != self.instance_defaults.variable.shape)):
                    raise FunctionError("Offset is using Hadamard modulation "
                                        "but its shape and/or size (shape: {}, size:{}) "
                                        "do not match the variable being modulated (shape: {}, size: {})".
                                        format(offset.shape, offset.size, self.instance_defaults.variable.shape, self.instance_defaults.variable.size))

            # if not operation:
            #     raise FunctionError("Operation param missing")
            # if not operation == self.Operation.SUM and not operation == self.Operation.PRODUCT:
            #     raise FunctionError("Operation param ({0}) must be Operation.SUM or Operation.PRODUCT".
            #     format(operation))


    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """Calculate and combine means of items in `variable <CombineMean.variable>`.

        Take mean of each item of `variable <CombineMean.variable>`;
        Apply `weights <CombineMeans.weights>` and/or `exponents <CombineMeanss.weights>` (if specified) to the means;
        Take their sum or product, as specified by `operation <CombineMeans.operation>`;
        Apply `scale <CombineMeans.scale>` (if specified) multiplicatively to the result;
        Apply `offset <CombineMeans.offset>` (if specified) to the result;
        Return scalar

        Arguments
        ---------

        variable : 1d or 2d np.array : default ClassDefaults.variable
           a single numeric array, or multiple arrays to be combined; if it is 2d, all arrays must have the same length.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        combined array : 1d np.array
            the result of linearly combining the arrays in `variable <CombineMeans.variable>`.

        """

        # Validate variable and assign to variable, and validate params
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        exponents = self.get_current_function_param(EXPONENTS)
        weights = self.get_current_function_param(WEIGHTS)
        operation = self.get_current_function_param(OPERATION)
        offset = self.get_current_function_param(OFFSET)
        scale = self.get_current_function_param(SCALE)

        # QUESTION:  WHICH IS LESS EFFICIENT:
        #                A) UNECESSARY ARITHMETIC OPERATIONS IF SCALE AND/OR OFFSET ARE 1.0 AND 0, RESPECTIVELY?
        #                   (DOES THE COMPILER KNOW NOT TO BOTHER WITH MULT BY 1 AND/OR ADD 0?)
        #                B) EVALUATION OF IF STATEMENTS TO DETERMINE THE ABOVE?
        # IMPLEMENTATION NOTE:  FOR NOW, ASSUME B) ABOVE, AND ASSIGN DEFAULT "NULL" VALUES TO offset AND scale
        if offset is None:
            offset = 0.0

        if scale is None:
            scale = 1.0

        # IMPLEMENTATION NOTE: CONFIRM: SHOULD NEVER OCCUR, AS _validate_variable NOW ENFORCES 2D np.ndarray
        # If variable is 0D or 1D:
        # if np_array_less_than_2d(variable):
        #     return (variable * scale) + offset

        means = np.array([[None]]*len(variable))
        for i, item in enumerate(variable):
            means[i] = np.mean(item)

        # FIX FOR EFFICIENCY: CHANGE THIS AND WEIGHTS TO TRY/EXCEPT // OR IS IT EVEN NECESSARY, GIVEN VALIDATION ABOVE??
        # Apply exponents if they were specified
        if exponents is not None:
            # Avoid divide by zero warning:
            #    make sure there are no zeros for an element that is assigned a negative exponent
            if (self.context.initialization_status == ContextFlags.INITIALIZING and
                    any(not any(i) and j < 0 for i, j in zip(variable, exponents))):
                means = np.ones_like(means)
            else:
                means = means ** exponents

        # Apply weights if they were specified
        if weights is not None:
            means = means * weights

        # CALCULATE RESULT USING RELEVANT COMBINATION OPERATION AND MODULATION

        if operation is SUM:
            result = np.sum(means, axis=0) * scale + offset

        elif operation is PRODUCT:
            result = np.product(means, axis=0) * scale + offset

        else:
            raise FunctionError("Unrecognized operator ({0}) for CombineMeans function".
                                format(self.get_current_function_param(OPERATION)))
        return result

    @property
    def offset(self):
        if not hasattr(self, '_offset'):
            return None
        else:
            return self._offset

    @offset.setter
    def offset(self, val):
        self._offset = val

    @property
    def scale(self):
        if not hasattr(self, '_scale'):
            return None
        else:
            return self._scale

    @scale.setter
    def scale(self, val):
        self._scale = val


GAMMA = 'gamma'
class PredictionErrorDeltaFunction(CombinationFunction):
    """
    Function that calculates the temporal difference prediction error
    """
    componentName = PREDICTION_ERROR_DELTA_FUNCTION

    classPreferences = {
        kwPreferenceSetName: 'PredictionErrorDeltaCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
        kpRuntimeParamStickyAssignmentPref: PreferenceEntry(False,
                                                            PreferenceLevel.INSTANCE)
    }

    class ClassDefaults(CombinationFunction.ClassDefaults):
        variable = np.array([[1], [1]])

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    multiplicative_param = None
    additive_param = None

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 gamma: tc.optional(float) = 1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts
        # (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(gamma=gamma,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

        self.gamma = gamma

    def _validate_variable(self, variable, context=None):
        """
        Insure that all items of variable are numeric

        Parameters
        ----------
        variable
        context

        Returns
        -------
        variable if all items are numeric
        """
        variable = self._update_variable(super()._validate_variable(variable=variable, context=context))

        if isinstance(variable, (list, np.ndarray)):
            if isinstance(variable, np.ndarray) and not variable.ndim:
                return variable
            length = 0
            for i in range(1, len(variable)):
                if i == 0:
                    continue
                if isinstance(variable[i - 1], numbers.Number):
                    old_length = 1
                else:
                    old_length = len(variable[i - 1])
                if isinstance(variable[i], numbers.Number):
                    new_length = 1
                else:
                    new_length = len(variable[i])
                if old_length != new_length:
                    raise FunctionError("Length of all arrays in variable {} "
                                        "for {} must be the same".format(variable,
                                                                         self.__class__.__name__))
        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """
        Checks that WEIGHTS is a list or np.array of numbers with length equal
        to variable.

        Note: the checks of compatibility with variable are only performed for
        validation calls during execution (i.e. from `check_args()`), since
        during initialization or COMMAND_LINE assignment, a parameter may be
        re-assigned before variable assigned during is known

        Parameters
        ----------
        request_set
        target_set
        context

        Returns
        -------
        None
        """
        super()._validate_params(request_set,
                                 target_set=target_set,
                                 context=context)

        if GAMMA in target_set and target_set[GAMMA] is not None:
            self._validate_parameter_spec(target_set[GAMMA] ,GAMMA, numeric_only=True)

        if WEIGHTS in target_set and target_set[WEIGHTS] is not None:
            self._validate_parameter_spec(target_set[WEIGHTS] ,WEIGHTS, numeric_only=True)
            target_set[WEIGHTS] = np.atleast_2d(target_set[WEIGHTS]).reshape(-1,1)
            if self.context.execution_phase & (ContextFlags.EXECUTING):
                if len(target_set[WEIGHTS]) != len(
                        self.instance_defaults.variable):
                    raise FunctionError("Number of weights {} is not equal to "
                                        "number of items in variable {}".format(
                        len(target_set[WEIGHTS]),
                        len(self.instance_defaults.variable.shape)))

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """
        Calculates the prediction error using the arrays in `variable
        <PredictionErrorDeltaFunction.variable>` and returns the resulting
        array.

        Parameters
        ----------
        variable : 2d np.array : default ClassDefaults.variable
            a 2d array representing the sample and target values to be used to
            calculate the temporal difference delta values. Both arrays must
            have the same length

        params : Dict[param keyword, param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that
            specifies the parameters for the function. Values specified for
            parameters in the dictionary override any assigned to those
            parameters in arguments of the constructor.


        Returns
        -------
        delta values : 1d np.array
            the result of
                :math: `\\delta(t) = r(t) + \\gamma sample(t) - sample(t - 1)`

        """
        variable = self._update_variable(self._check_args(variable=variable,
                                                          params=params,
                                                          context=context))
        gamma = self.get_current_function_param(GAMMA)
        sample = variable[0]
        reward = variable[1]
        delta = np.zeros(sample.shape)

        for t in range(1, len(sample)):
            delta[t] = reward[t] + gamma * sample[t] - sample[t - 1]
        return delta


# region ***********************************  TRANSFER FUNCTIONS  ***********************************************
# endregion

BOUNDS = 'bounds'

class TransferFunction(Function_Base):
    """Function that transforms variable but maintains its shape

    All TransferFunctions must have the following attributes:

    `bounds` -- specifies the lower and upper limits of the result;  if there are none, the attribute is set to
    `None`;  if it has at least one bound, the attribute is set to a tuple specifying the lower and upper bounds,
    respectively, with `None` as the entry for no bound.

    `multiplicative_param` and `additive_param` -- each of these is assigned the name of one of the function's
    parameters and used by `ModulatoryProjections <ModulatoryProjection>` to modulate the output of the
    TransferFunction's function (see `Function_Modulatory_Params`).

    """
    componentType = TRANSFER_FUNCTION_TYPE

    # IMPLEMENTATION NOTE: THESE SHOULD SHOULD BE REPLACED WITH ABC WHEN IMPLEMENTED
    def __init__(self, default_variable,
                 params,
                 owner,
                 prefs,
                 context):

        if not hasattr(self, BOUNDS):
            raise FunctionError("PROGRAM ERROR: {} must implement a {} attribute".
                                format(self.__class__.__name__, BOUNDS))

        if not hasattr(self, MULTIPLICATIVE_PARAM):
            raise FunctionError("PROGRAM ERROR: {} must implement a {} attribute".
                                format(self.__class__.__name__, MULTIPLICATIVE_PARAM))

        if not hasattr(self, ADDITIVE_PARAM):
            raise FunctionError("PROGRAM ERROR: {} must implement an {} attribute".
                                format(self.__class__.__name__, ADDITIVE_PARAM))

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

    @property
    def multiplicative(self):
        return getattr(self, self.multiplicative_param)

    @multiplicative.setter
    def multiplicative(self, val):
        setattr(self, self.multiplicative_param, val)

    @property
    def additive(self):
        return getattr(self, self.additive_param)

    @additive.setter
    def additive(self, val):
        setattr(self, self.additive_param, val)


class Linear(TransferFunction):  # -------------------------------------------------------------------------------------
    """
    Linear(                \
         default_variable, \
         slope=1.0,        \
         intercept=0.0,    \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _Linear:

    Linearly transform variable.

    Note: default values for `slope` and `intercept` implement the IDENTITY_FUNCTION

    Arguments
    ---------

    variable : number or np.array : default ClassDefaults.variable
        specifies a template for the value to be transformed.

    slope : float : default 1.0
        specifies a value by which to multiply `variable <Linear.variable>`.

    intercept : float : default 0.0
        specifies a value to add to each element of `variable <Linear.variable>` after applying `slope <Linear.slope>`.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : number or np.array
        contains value to be transformed.

    slope : float
        value by which each element of `variable <Linear.variable>` is multiplied before applying the
        `intercept <Linear.intercept>` (if it is specified).

    intercept : float
        value added to each element of `variable <Linear.variable>` after applying the `slope <Linear.slope>`
        (if it is specified).

    bounds : None

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = LINEAR_FUNCTION

    bounds = None
    multiplicative_param = SLOPE
    additive_param = INTERCEPT

    classPreferences = {
        kwPreferenceSetName: 'LinearClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
        kpRuntimeParamStickyAssignmentPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
    }

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        FUNCTION_OUTPUT_TYPE_CONVERSION: True,
        PARAMETER_STATE_PARAMS: None
    })

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 slope: parameter_spec = 1.0,
                 intercept: parameter_spec = 0.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(slope=slope,
                                                  intercept=intercept,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

        # self.functionOutputType = None

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """
        Return: `slope <Linear.slope>` * `variable <Linear.variable>` + `intercept <Linear.intercept>`.

        Arguments
        ---------

        variable : number or np.array : default ClassDefaults.variable
           a single value or array to be transformed.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        linear transformation of variable : number or np.array

        """

        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))
        slope = self.get_current_function_param(SLOPE)
        intercept = self.get_current_function_param(INTERCEPT)
        outputType = self.functionOutputType

        # MODIFIED 11/9/17 NEW:
        try:
        # By default, result should be returned as np.ndarray with same dimensionality as input
            result = variable * slope + intercept
        except TypeError:
            # If variable is an array with mixed sizes or types, try item-by-item operation
            if variable.dtype == object:
                result = np.zeros_like(variable)
                for i, item in enumerate(variable):
                    result[i] = variable[i] * slope + intercept
            else:
                raise FunctionError("Unrecognized type for {} of {} ({})".format(VARIABLE, self.name, variable))
        # MODIFIED 11/9/17 END


        # region Type conversion (specified by outputType):
        # Convert to 2D array, irrespective of variable type:
        if outputType is FunctionOutputType.NP_2D_ARRAY:
            result = np.atleast_2d(result)

        # Convert to 1D array, irrespective of variable type:
        # Note: if 2D array (or higher) has more than two items in the outer dimension, generate exception
        elif outputType is FunctionOutputType.NP_1D_ARRAY:
            # If variable is 2D
            if variable.ndim == 2:
                # If there is only one item:
                if len(variable) == 1:
                    result = result[0]
                else:
                    raise FunctionError("Can't convert result ({0}: 2D np.ndarray object with more than one array)"
                                        " to 1D array".format(result))
            elif len(variable) == 1:
                result = result
            elif len(variable) == 0:
                result = np.atleast_1d(result)
            else:
                raise FunctionError("Can't convert result ({0} to 1D array".format(result))

        # Convert to raw number, irrespective of variable type:
        # Note: if 2D or 1D array has more than two items, generate exception
        elif outputType is FunctionOutputType.RAW_NUMBER:
            # If variable is 2D
            if variable.ndim == 2:
                # If there is only one item:
                if len(variable) == 1 and len(variable[0]) == 1:
                    result = result[0][0]
                else:
                    raise FunctionError("Can't convert result ({0}) with more than a single number to a raw number".
                                        format(result))
            elif len(variable) == 1:
                if len(variable) == 1:
                    result = result[0]
                else:
                    raise FunctionError("Can't convert result ({0}) with more than a single number to a raw number".
                                        format(result))
            else:
                return result
        # endregion

        return result

    def derivative(self, input=None, output=None):
        """
        derivative()

        Derivative of `function <Linear.function>`.

        Returns
        -------

        derivative :  number
            current value of `slope <Linear.slope>`.

        """

        return self.get_current_function_param(SLOPE)


class Exponential(TransferFunction):  # --------------------------------------------------------------------------------
    """
    Exponential(           \
         default_variable, \
         scale=1.0,        \
         rate=1.0,         \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _Exponential:

    Exponentially transform variable.

    Arguments
    ---------

    variable : number or np.array : default ClassDefaults.variable
        specifies a template for the value to be transformed.

    rate : float : default 1.0
        specifies a value by which to multiply `variable <Exponential.variable>` before exponentiation.

    scale : float : default 1.0
        specifies a value by which to multiply the exponentiated value of `variable <Exponential.variable>`.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : number or np.array
        contains value to be transformed.

    rate : float
        value by which `variable <Exponential.variable>` is multiplied before exponentiation.

    scale : float
        value by which the exponentiated value is multiplied.

    bounds : (0, None)

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = EXPONENTIAL_FUNCTION

    bounds = (0, None)
    multiplicative_param = RATE
    additive_param = SCALE

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate: parameter_spec = 1.0,
                 scale: parameter_spec = 1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  scale=scale,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """
        Return: `scale <Exponential.scale>`
        :math:`*` e**(`rate <Exponential.rate>` :math:`*` `variable <Linear.variable>`).

        Arguments
        ---------

        variable : number or np.array : default ClassDefaults.variable
           a single value or array to be exponentiated.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        exponential transformation of variable : number or np.array

        """

        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))
        rate = self.get_current_function_param(RATE)
        scale = self.get_current_function_param(SCALE)

        return scale * np.exp(rate * variable)

    def derivative(self, input, output=None):
        """
        derivative(input)

        Derivative of `function <Exponential.function>`.

        Returns
        -------

        derivative :  number
            `rate <Exponential.rate>` * input.

        """
        return self.get_current_function_param(RATE) * input


class Logistic(TransferFunction):  # ------------------------------------------------------------------------------------
    """
    Logistic(              \
         default_variable, \
         gain=1.0,         \
         bias=0.0,         \
         offset=0.0,       \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _Logistic:

    Logistically transform variable.

    Arguments
    ---------

    variable : number or np.array : default ClassDefaults.variable
        specifies a template for the value to be transformed.

    gain : float : default 1.0
        specifies a value by which to multiply `variable <Logistic.variable>` before logistic transformation

    bias : float : default 0.0
        specifies a value to add to each element of `variable <Logistic.variable>` before applying `gain <Logistic.gain>`
        and before logistic transformation.

    offset : float : default 0.0
        specifies a value to add to each element of `variable <Logistic.variable>` after applying `gain <Logistic.gain>`
        but before logistic transformation.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : number or np.array
        contains value to be transformed.

    gain : float : default 1.0
        value by which each element of `variable <Logistic.variable>` is multiplied before applying the
        `bias <Logistic.bias>` (if it is specified).

    bias : float : default 0.0
        value added to each element of `variable <Logistic.variable>` after applying the `gain <Logistic.gain>`
        (if it is specified).

    bounds : (0,1)

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = LOGISTIC_FUNCTION
    parameter_keywords.update({GAIN, BIAS})

    bounds = (0,1)
    multiplicative_param = GAIN
    additive_param = BIAS

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 gain: parameter_spec = 1.0,
                 bias: parameter_spec = 0.0,
                 offset: parameter_spec = 0.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(gain=gain,
                                                  bias=bias,
                                                  offset=offset,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """
        Return:

        .. math::

            \\frac{1}{1 + e^{ - gain ( variable - bias ) + offset}}

        Arguments
        ---------

        variable : number or np.array : default ClassDefaults.variable
           a single value or array to be transformed.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        logistic transformation of variable : number or np.array

        """

        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))
        gain = self.get_current_function_param(GAIN)
        bias = self.get_current_function_param(BIAS)
        offset = self.get_current_function_param(OFFSET)

        return 1 / (1 + np.exp(-gain*(variable-bias) + offset))

    def derivative(self, output, input=None):
        """
        derivative(output)

        Derivative of `function <Logistic.function>`.

        Returns
        -------

        derivative :  number
            output * (1 - output).

        """
        return output * (1 - output)


MODE = 'mode'
class OneHot(TransferFunction):  # -------------------------------------------------------------------------------------
    """
    OneHot(                \
         default_variable, \
         mode=MAX_VAL,     \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _OneHot:

    Return an array with one non-zero value.

    The `mode <OneHot.mode>` parameter determines the nature of the non-zero value:

    Arguments
    ---------

    variable : 2d np.array : default ClassDefaults.variable
        First (possibly only) item specifies a template for the array to be transformed;  if `mode <OneHot.mode>` is
        *PROB* then a 2nd item must be included that is a probability distribution with same length as 1st item.

    mode : MAX_VAL, MAX_ABS_VAL, MAX_INDICATOR, or PROB : default MAX_VAL
        specifies the nature of the single non-zero value in the array returned by `function <OneHot.function>`
        (see `mode <OneHot.mode>` for details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    bounds : None

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : number or np.array
        1st item contains value to be transformed;  if `mode <OneHot.mode>` is *PROB*, 2nd item is a probability
        distribution, each element of which specifies the probability for selecting the corresponding element of the
        1st item.

    mode : MAX_VAL, MAX_ABS_VAL, MAX_INDICATOR, or PROB : default MAX_VAL
        determines the nature of the single non-zero value in the array returned by `function <OneHot.function>`:
            * **MAX_VAL**: element with the maximum signed value in the original array;
            * *MAX_ABS_VAL**: element with the maximum absolute value;
            * **MAX_INDICATOR**: 1 in place of the element with the maximum signed value;
            * **MAX_ABS_INDICATOR**: 1 in place of the element with the maximum absolute value;
            * **PROB**: probabilistically chosen element based on probabilities passed in second item of
            * **PROB_INDICATOR**: same as *PROB* but chosen item is assigned a value of 1.

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = ONE_HOT_FUNCTION

    bounds = None
    multiplicative_param = None
    additive_param = None

    classPreferences = {
        kwPreferenceSetName: 'OneHotClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
        kpRuntimeParamStickyAssignmentPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
    }

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        FUNCTION_OUTPUT_TYPE_CONVERSION: False,
        PARAMETER_STATE_PARAMS: None
    })

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 mode: tc.enum(MAX_VAL, MAX_ABS_VAL, MAX_INDICATOR, MAX_ABS_INDICATOR, PROB, PROB_INDICATOR)=MAX_VAL,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(mode=mode,
                                                  params=params)

        if mode in {PROB, PROB_INDICATOR} and default_variable is None:
            default_variable = [[0],[0]]

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

        # self.functionOutputType = None

    def _validate_params(self, request_set, target_set=None, context=None):

        if request_set[MODE] in {PROB, PROB_INDICATOR}:
            if not self.variable.ndim == 2:
                raise FunctionError("If {} for {} {} is set to {}, variable must be 2d array".
                                    format(MODE, self.__class__.__name__, Function.__name__, PROB))
            values = self.variable[0]
            prob_dist = self.variable[1]
            if len(values)!=len(prob_dist):
                raise FunctionError("If {} for {} {} is set to {}, the two items of its variable must be of equal "
                                    "length (len item 1 = {}; len item 2 = {}".
                                    format(MODE, self.__class__.__name__, Function.__name__, PROB,
                                           len(values), len(prob_dist)))
            if not all((elem>=0 and elem<=1) for elem in prob_dist)==1:
                raise FunctionError("If {} for {} {} is set to {}, the 2nd item of its variable ({}) must be an "
                                    "array of elements each of which is in the (0,1) interval".
                                    format(MODE, self.__class__.__name__, Function.__name__, PROB, prob_dist))
            if self.context.initialization_status == ContextFlags.INITIALIZING:
                return
            if not np.sum(prob_dist)==1:
                raise FunctionError("If {} for {} {} is set to {}, the 2nd item of its variable ({}) must be an "
                                    "array of probabilities that sum to 1".
                                    format(MODE, self.__class__.__name__, Function.__name__, PROB, prob_dist))

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """
        Return array of len(`variable <Linear.variable>`) with single non-zero value specified by `mode <OneHot.mode>`.

        Arguments
        ---------

        variable : 2d np.array : default ClassDefaults.variable
           1st item is an array to be transformed;  if `mode <OneHot.mode>` is *PROB*, 2nd item must be an array of
           probabilities (i.e., elements between 0 and 1) of equal length to the 1st item.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        array with single non-zero value : np.array

        """

        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        if self.mode is MAX_VAL:
            max_value = np.max(variable)
            return np.where(variable == max_value, max_value, 0)

        if self.mode is MAX_ABS_VAL:
            max_value = np.max(np.absolute(variable))
            return np.where(variable == max_value, max_value, 0)

        elif self.mode is MAX_INDICATOR:
            max_value = np.max(variable)
            return np.where(variable == max_value, 1, 0)

        elif self.mode is MAX_ABS_INDICATOR:
            max_value = np.max(np.absolute(variable))
            return np.where(variable == max_value, 1, 0)

        elif self.mode in {PROB, PROB_INDICATOR}:
            # 1st item of variable should be data, and 2nd a probability distribution for choosing
            v = variable[0]
            prob_dist = variable[1]
            # if not prob_dist.any() and INITIALIZING in context:
            if not prob_dist.any():
                return v
            cum_sum = np.cumsum(prob_dist)
            random_value = np.random.uniform()
            chosen_item = next(element for element in cum_sum if element > random_value)
            chosen_in_cum_sum = np.where(cum_sum == chosen_item, 1, 0)
            if self.mode is PROB:
                result = v * chosen_in_cum_sum
            else:
                result = np.ones_like(v) * chosen_in_cum_sum
            return result
            # chosen_item = np.random.choice(v, 1, p=prob_dist)
            # one_hot_indicator = np.where(v == chosen_item, 1, 0)
            # return v * one_hot_indicator


class NormalizingFunction(Function_Base):
    """Function that adjusts a set of values
    """
    componentType = NORMALIZING_FUNCTION_TYPE

    # IMPLEMENTATION NOTE: THESE SHOULD SHOULD BE REPLACED WITH ABC WHEN IMPLEMENTED
    def __init__(self, default_variable,
                 params,
                 owner,
                 prefs,
                 context):

        if not hasattr(self, MULTIPLICATIVE_PARAM):
            raise FunctionError("PROGRAM ERROR: {} must implement a {} attribute".
                                format(self.__class__.__name__, MULTIPLICATIVE_PARAM))

        if not hasattr(self, ADDITIVE_PARAM):
            raise FunctionError("PROGRAM ERROR: {} must implement an {} attribute".
                                format(self.__class__.__name__, ADDITIVE_PARAM))

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

    @property
    def multiplicative(self):
        return getattr(self, self.multiplicative_param)

    @multiplicative.setter
    def multiplicative(self, val):
        setattr(self, self.multiplicative_param, val)

    @property
    def additive(self):
        return getattr(self, self.additive_param)

    @additive.setter
    def additive(self, val):
        setattr(self, self.additive_param, val)


class SoftMax(NormalizingFunction):
    """
    SoftMax(               \
         default_variable, \
         gain=1.0,         \
         output=ALL,       \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _SoftMax:

    SoftMax transform of variable (see `The Softmax function and its derivative
    <http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/>`_ for a nice discussion).

    Arguments
    ---------

    default_variable : 1d np.array : default ClassDefaults.variable
        specifies a template for the value to be transformed.

    gain : float : default 1.0
        specifies a value by which to multiply `variable <Linear.variable>` before SoftMax transformation.

    output : ALL, MAX_VAL, MAX_INDICATOR, or PROB : default ALL
        specifies the format of array returned by `function <SoftMax.function>`
        (see `output <SoftMax.output>` for details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : 1d np.array
        contains value to be transformed.

    gain : float
        value by which `variable <Logistic.variable>` is multiplied before the SoftMax transformation;  determines
        the "sharpness" of the distribution.

    output : ALL, MAX_VAL, MAX_INDICATOR, or PROB
        determines how the SoftMax-transformed values of the elements in `variable <SoftMax.variable>` are reported
        in the array returned by `function <SoftMax.function>`:
            * **ALL**: array of all SoftMax-transformed values (the default);
            * **MAX_VAL**: SoftMax-transformed value for the element with the maximum such value, 0 for all others;
            * **MAX_INDICATOR**: 1 for the element with the maximum SoftMax-transformed value, 0 for all others;
            * **PROB**: probabilistically chosen element based on SoftMax-transformed values after normalizing the
              sum of values to 1 (i.e., their `Luce Ratio <https://en.wikipedia.org/wiki/Luce%27s_choice_axiom>`_),
              0 for all others.

    bounds : None if `output <SoftMax.output>` == MAX_VAL, else (0,1) : default (0,1)

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = SOFTMAX_FUNCTION

    bounds = (0,1)
    multiplicative_param = GAIN
    additive_param = None

    class ClassDefaults(NormalizingFunction.ClassDefaults):
        variable = 0

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 gain: parameter_spec = 1.0,
                 output: tc.enum(ALL, MAX_VAL, MAX_INDICATOR, PROB) = ALL,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(gain=gain,
                                                  output=output,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

    def _instantiate_function(self, function, function_params=None, context=None):

        self.one_hot_function = None
        output_type = self.get_current_function_param(OUTPUT_TYPE)
        bounds = None

        if not output_type is ALL:
            self.one_hot_function = OneHot(mode=output_type).function

        super()._instantiate_function(function, function_params=function_params, context=context)

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """
        Return: e**(`gain <SoftMax.gain>` * `variable <SoftMax.variable>`) /
        sum(e**(`gain <SoftMax.gain>` * `variable <SoftMax.variable>`)),
        filtered by `ouptput <SoftMax.output>` specification.

        Arguments
        ---------

        variable : 1d np.array : default ClassDefaults.variable
           an array to be transformed.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        SoftMax transformation of variable : number or np.array

        """

        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        # Assign the params and return the result
        output_type = self.get_current_function_param(OUTPUT_TYPE)
        gain = self.get_current_function_param(GAIN)

        # Compute softmax and assign to sm

        # Modulate variable by gain
        v = gain * variable
        # Shift by max to avoid extreme values:
        v = v - np.max(v)
        # Exponentiate
        v = np.exp(v)
        # Normalize (to sum to 1)
        sm = v / np.sum(v, axis=0)

        # Generate one-hot encoding based on selected output_type

        if output_type in {MAX_VAL, MAX_INDICATOR}:
            return self.one_hot_function(sm)
        elif output_type in {PROB, PROB_INDICATOR}:
            return self.one_hot_function([variable, sm])
        else:
            return sm

    def derivative(self, output, input=None):
        """
        derivative(output)

        Calculate the derivative of `function <SoftMax.function>`.  If OUTPUT_TYPE for the SoftMax Function is ALL,
        return Jacobian matrix (derivative for each element of the output array with respect to each of the others):
            COMMENT:
                D[j]/S[i] = S[i](d[i,j] - S[j]) where d[i,j]=1 if i==j; d[i,j]=0 if i!=j.
            COMMENT
            D\\ :sub:`j`\\ S\\ :sub:`i` = S\\ :sub:`i`\\ (𝜹\\ :sub:`i,j` - S\\ :sub:`j`),
            where 𝜹\\ :sub:`i,j`\\ =1 if i=j and 𝜹\\ :sub:`i,j`\\ =0 if i≠j.
        If OUTPUT_TYPE is MAX_VAL or MAX_INDICATOR, return 1d array of the derivatives of the maximum
        value with respect to the others (calculated as above). If OUTPUT_TYPE is PROB, raise an exception
        (since it is ambiguous as to which element would have been chosen by the SoftMax function)

        Returns
        -------

        derivative :  1d or 2d np.array (depending on OUTPUT_TYPE of SoftMax)
            derivative of values returns by SoftMax.

        """

        output_type = self.params[OUTPUT_TYPE]
        size = len(output)
        sm = self.function(output, params={OUTPUT_TYPE: ALL})

        if output_type is ALL:
            # Return full Jacobian matrix of derivatives
            derivative = np.empty([size, size])
            for j in range(size):
                for i, val in zip(range(size), output):
                    if i==j:
                        d = 1
                    else:
                        d = 0
                    derivative[j,i] = sm[i] * (d - sm[j])

        elif output_type in {MAX_VAL, MAX_INDICATOR}:
            # Return 1d array of derivatives for max element (i.e., the one chosen by SoftMax)
            derivative = np.empty(size)
            # Get the element of output returned as non-zero when output_type is not ALL
            index_of_max = int(np.where(output==np.max(output))[0])
            max_val = sm[index_of_max]
            for i in range(size):
                if i==index_of_max:
                    d = 1
                else:
                    d = 0
                derivative[i] = sm[i] * (d - max_val)

        else:
            raise FunctionError("Can't assign derivative for SoftMax function{} since OUTPUT_TYPE is PROB "
                                "(and therefore the relevant element is ambiguous)".format(self.owner_name))

        return derivative


class LinearMatrix(TransferFunction):  # -------------------------------------------------------------------------------
    """
    LinearMatrix(          \
         default_variable, \
         matrix=None,      \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _LinearMatrix:

    Matrix transform of variable:

        `function <LinearMatrix.function>` returns dot product of `variable <LinearMatrix.variable>` and
        `matrix <LinearMatrix.matrix>`.

    COMMENT:  [CONVERT TO FIGURE]
        ----------------------------------------------------------------------------------------------------------
        MATRIX FORMAT <shape: (3,5)>
                                         INDICES:
                                     Output elements:
                              0       1       2       3       4
                         0  [0,0]   [0,1]   [0,2]   [0,3]   [0,4]
        Input elements:  1  [1,0]   [1,1]   [1,2]   [1,3]   [1,4]
                         2  [2,0]   [2,1]   [2,2]   [2,3]   [2,4]

        matrix.shape => (input/rows, output/cols)

        ----------------------------------------------------------------------------------------------------------
        ARRAY FORMAT
                                                                            INDICES
                                          [ [      Input 0 (row0)       ], [       Input 1 (row1)      ]... ]
                                          [ [ out0,  out1,  out2,  out3 ], [ out0,  out1,  out2,  out3 ]... ]
        matrix[input/rows, output/cols]:  [ [ row0,  row0,  row0,  row0 ], [ row1,  row1,  row1,  row1 ]... ]
                                          [ [ col0,  col1,  col2,  col3 ], [ col0,  col1,  col2,  col3 ]... ]
                                          [ [[0,0], [0,1], [0,2], [0,3] ], [[1,0], [1,1], [1,2], [1,3] ]... ]

        ----------------------------------------------------------------------------------------------------------
    COMMENT


    Arguments
    ---------

    variable : list or 1d np.array : default ClassDefaults.variable
        specifies a template for the value to be transformed; length must equal the number of rows of `matrix
        <LinearMatrix.matrix>`.

    matrix : number, list, 1d or 2d np.ndarray, np.matrix, function, or matrix keyword : default IDENTITY_MATRIX
        specifies matrix used to transform `variable <LinearMatrix.variable>`
        (see `matrix <LinearMatrix.matrix>` for specification details).

        When LinearMatrix is the `function <Projection.function>` of a projection:

            - the matrix specification must be compatible with the variables of the `sender <Projection.sender>` and
              `receiver <Projection.receiver>`

            - a matrix keyword specification generates a matrix based on the sender and receiver shapes

        When LinearMatrix is instantiated on its own, or as the function of `Mechanism` or `State`:

            - the matrix specification must be compatible with the function's own `variable <LinearMatrix.variable>`

            - if matrix is not specified, a square identity matrix is generated based on the number of columns in
              `variable <LinearMatrix.variable>`

            - matrix keywords are not valid matrix specifications

    bounds : None

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : 1d np.array
        contains value to be transformed.

    matrix : 2d np.array
        matrix used to transform `variable <LinearMatrix.variable>`.
        Can be specified as any of the following:
            * number - used as the filler value for all elements of the :keyword:`matrix` (call to np.fill);
            * list of arrays, 2d np.array or np.matrix - assigned as the value of :keyword:`matrix`;
            * matrix keyword - see `MatrixKeywords` for list of options.
        Rows correspond to elements of the input array (outer index), and
        columns correspond to elements of the output array (inner index).

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = LINEAR_MATRIX_FUNCTION

    bounds = None
    multiplicative_param = None
    additive_param = None

    DEFAULT_FILLER_VALUE = 0

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    # def is_matrix_spec(m):
    #     if m is None:
    #         return True
    #     if m in MATRIX_KEYWORD_VALUES:
    #         return True
    #     if isinstance(m, (list, np.ndarray, np.matrix, function_type)):
    #         return True
    #     return False

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 matrix:tc.optional(is_matrix) = None,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(matrix=matrix,
                                                  params=params)

        # Note: this calls _validate_variable and _validate_params which are overridden below;
        #       the latter implements the matrix if required
        # super(LinearMatrix, self).__init__(default_variable=default_variable,
        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

        self._matrix = self.instantiate_matrix(self.paramsCurrent[MATRIX])

    # def _validate_variable(self, variable, context=None):
    #     """Insure that variable passed to LinearMatrix is a max 2D np.array
    #
    #     :param variable: (max 2D np.array)
    #     :param context:
    #     :return:
    #     """
    #     variable = self._update_variable(super()._validate_variable(variable, context))
    #
    #     # Check that variable <= 2D
    #     try:
    #         if not variable.ndim <= 2:
    #             raise FunctionError("variable ({0}) for {1} must be a numpy.ndarray of dimension at most 2".format(variable, self.__class__.__name__))
    #     except AttributeError:
    #         raise FunctionError("PROGRAM ERROR: variable ({0}) for {1} should be a numpy.ndarray".
    #                                 format(variable, self.__class__.__name__))
    #
    #     return variable


    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate params and assign to targets

        This overrides the class method, to perform more detailed type checking (see explanation in class method).
        Note: this method (or the class version) is called only if the parameter_validation attribute is `True`

        :param request_set: (dict) - params to be validated
        :param target_set: (dict) - destination of validated params
        :param context: (str)
        :return none:
        """

        super()._validate_params(request_set, target_set, context)

        param_set = target_set
        # proxy for checking whether the owner is a projection
        if hasattr(self.owner, "receiver"):
            sender = self.instance_defaults.variable
            # Note: this assumes variable is a 1D np.array, as enforced by _validate_variable
            sender_len = sender.size

            # FIX: RELABEL sender -> input AND receiver -> output
            # FIX: THIS NEEDS TO BE CLEANED UP:
            #      - AT LEAST CHANGE THE NAME FROM kwReceiver TO output_template OR SOMETHING LIKE THAT
            #      - MAKE ARG?  OR ADD OTHER PARAMS:  E.G., FILLER?
            #      - OR REFACTOR TO INCLUDE AS MATRIX SPEC:
            #          IF MATRIX IS 1D, USE AS OUTPUT TEMPLATE
            #          IF ALL ITS VALUES ARE 1'S => FULL CONNECTIVITY MATRIX
            #          IF ALL ITS VALUES ARE 0'S => RANDOM CONNECTIVITY MATRIX
            #          NOTE:  NO NEED FOR IDENTITY MATRIX, AS THAT WOULD BE SQUARE SO NO NEED FOR OUTPUT TEMPLATE
            #      - DOCUMENT WHEN DONE
            # MODIFIED 3/26/17 OLD:
            # Check for and validate kwReceiver first, since it may be needed to validate and/or construct the matrix
            # First try to get receiver from specification in params
            if RECEIVER in param_set:
                self.receiver = param_set[RECEIVER]
                # Check that specification is a list of numbers or an np.array
                if ((isinstance(self.receiver, list) and all(
                        isinstance(elem, numbers.Number) for elem in self.receiver)) or
                        isinstance(self.receiver, np.ndarray)):
                    self.receiver = np.atleast_1d(self.receiver)
                else:
                    raise FunctionError("receiver param ({0}) for {1} must be a list of numbers or an np.array".
                                        format(self.receiver, self.name))
            # No receiver, so use sender as template (assuming square -- e.g., identity -- matrix)
            else:
                if (self.owner and self.owner.prefs.verbosePref) or self.prefs.verbosePref:
                    print("Identity matrix requested but kwReceiver not specified; sender length ({0}) will be used".
                          format(sender_len))
                self.receiver = param_set[RECEIVER] = sender

            receiver_len = len(self.receiver)

            # Check rest of params
            message = ""
            for param_name, param_value in param_set.items():

                # Receiver param already checked above
                if param_name is RECEIVER:
                    continue

                # Not currently used here
                if param_name in function_keywords:
                    continue

                if param_name is AUTO_DEPENDENT:
                    continue

                # Matrix specification param
                elif param_name == MATRIX:

                    # A number (to be used as a filler), so OK
                    if isinstance(param_value, numbers.Number):
                        continue

                    # np.matrix or np.ndarray provided, so validate that it is numeric and check dimensions
                    elif isinstance(param_value, (list, np.ndarray, np.matrix)):
                        # get dimensions specified by:
                        #   variable (sender): width/cols/outer index
                        #   kwReceiver param: height/rows/inner index

                        weight_matrix = np.matrix(param_value)
                        if 'U' in repr(weight_matrix.dtype):
                            raise FunctionError("Non-numeric entry in MATRIX "
                                                "specification ({}) for the {} "
                                                "function of {}".format(param_value,
                                                                        self.name,
                                                                        self.owner_name))

                        if weight_matrix.ndim != 2:
                            raise FunctionError("The matrix provided for the {} function of {} must be 2d (it is {}d".
                                                format(weight_matrix.ndim, self.name, self.owner_name))

                        matrix_rows = weight_matrix.shape[0]
                        matrix_cols = weight_matrix.shape[1]

                        # Check that number of rows equals length of sender vector (variable)
                        if matrix_rows != sender_len:
                            raise FunctionError("The number of rows ({}) of the "
                                                "matrix provided for {} function "
                                                "of {} does not equal the length "
                                                "({}) of the sender vector "
                                                "(variable)".format(matrix_rows,
                                                                    self.name,
                                                                    self.owner_name,
                                                                    sender_len))

                    # Auto, full or random connectivity matrix requested (using keyword):
                    # Note:  assume that these will be properly processed by caller
                    #        (e.g., MappingProjection._instantiate_receiver)
                    elif param_value in MATRIX_KEYWORD_VALUES:
                        continue

                    # Identity matrix requested (using keyword), so check send_len == receiver_len
                    elif param_value in {IDENTITY_MATRIX, HOLLOW_MATRIX}:
                        # Receiver length doesn't equal sender length
                        if not (self.receiver.shape == sender.shape and self.receiver.size == sender.size):
                            # if self.owner.prefs.verbosePref:
                            #     print ("Identity matrix requested, but length of receiver ({0})"
                            #            " does not match length of sender ({1});  sender length will be used".
                            #            format(receiver_len, sender_len))
                            # # Set receiver to sender
                            # param_set[kwReceiver] = sender
                            raise FunctionError("{} requested for the {} function of {}, "
                                                "but length of receiver ({}) does not match length of sender ({})".
                                                format(param_value, self.name, self.owner_name, receiver_len,
                                                       sender_len))
                        continue

                    # list used to describe matrix, so convert to 2D np.array and pass to validation of matrix below
                    elif isinstance(param_value, list):
                        try:
                            param_value = np.atleast_2d(param_value)
                        except (ValueError, TypeError) as error_msg:
                            raise FunctionError(
                                "Error in list specification ({}) of matrix for the {} function of {}: {})".
                                # format(param_value, self.__class__.__name__, error_msg))
                                format(param_value, self.name, self.owner_name, error_msg))

                    # string used to describe matrix, so convert to np.matrix and pass to validation of matrix below
                    elif isinstance(param_value, str):
                        try:
                            param_value = np.matrix(param_value)
                        except (ValueError, TypeError) as error_msg:
                            raise FunctionError("Error in string specification ({}) of the matrix "
                                                "for the {} function of {}: {})".
                                                # format(param_value, self.__class__.__name__, error_msg))
                                                format(param_value, self.name, self.owner_name, error_msg))

                    # function so:
                    # - assume it uses random.rand()
                    # - call with two args as place markers for cols and rows
                    # -  validate that it returns an np.array or np.matrix
                    elif isinstance(param_value, function_type):
                        test = param_value(1, 1)
                        if not isinstance(test, (np.ndarray, np.matrix)):
                            raise FunctionError("A function is specified for the matrix of the {} function of {}: {}) "
                                                "that returns a value ({}) that is neither a matrix nor an array".
                                                # format(param_value, self.__class__.__name__, test))
                                                format(self.name, self.owner_name, param_value, test))

                    elif param_value is None:
                        raise FunctionError("TEMP ERROR: param value is None.")

                    else:
                        raise FunctionError("Value of {} param ({}) for the {} function of {} "
                                            "must be a matrix, a number (for filler), or a matrix keyword ({})".
                                            format(param_name,
                                                   param_value,
                                                   self.name,
                                                   self.owner_name,
                                                   MATRIX_KEYWORD_NAMES))
                else:
                    message += "Unrecognized param ({}) specified for the {} function of {}\n".\
                        format(param_name, self.componentName, self.owner_name)
                    continue
            if message:
                raise FunctionError(message)

        # owner is a mechanism, state
        # OR function was defined on its own (no owner)
        else:
            if MATRIX in param_set:
                param_value = param_set[MATRIX]

                # numeric value specified; verify that it is compatible with variable
                if isinstance(param_value, (float, list, np.ndarray, np.matrix)):
                    if np.size(np.atleast_2d(param_value),0)!=np.size(np.atleast_2d(self.instance_defaults.variable),1):
                        raise FunctionError("Specification of matrix and/or default_variable for {} is not valid. The "
                                            "shapes of variable {} and matrix {} are not compatible for multiplication".
                                            format(self.name, np.shape(np.atleast_2d(self.instance_defaults.variable)),
                                                   np.shape(np.atleast_2d(param_value))))

                # keyword matrix specified - not valid outside of a projection
                elif param_value in MATRIX_KEYWORD_VALUES:
                    raise FunctionError("{} is not a valid specification for the matrix parameter of {}. Keywords "
                                        "may only be used to specify the matrix parameter of a Projection's "
                                        "LinearMatrix function. When the LinearMatrix function is implemented in a "
                                        "mechanism, such as {}, the correct matrix cannot be determined from a "
                                        "keyword. Instead, the matrix must be fully specified as a float, list, "
                                        "np.ndarray, or np.matrix".
                                        format(param_value, self.name, self.owner.name))

                # The only remaining valid option is matrix = None (sorted out in instantiate_attribs_before_fn)
                elif param_value is not None:
                    raise FunctionError("Value of the matrix param ({}) for the {} function of {} "
                                        "must be a matrix, a number (for filler), or a matrix keyword ({})".
                                        format(param_value,
                                               self.name,
                                               self.owner_name,
                                               MATRIX_KEYWORD_NAMES))

    def _instantiate_attributes_before_function(self, function=None, context=None):
        if self.matrix is None and not hasattr(self.owner, "receiver"):
            variable_length = np.size(np.atleast_2d(self.instance_defaults.variable), 1)
            self.matrix = np.identity(variable_length)
        self.matrix = self.instantiate_matrix(self.matrix)

    def instantiate_matrix(self, specification, context=None):
        """Implements matrix indicated by specification

         Specification is derived from MATRIX param (passed to self.__init__ or self.function)

         Specification (validated in _validate_params):
            + single number (used to fill self.matrix)
            + matrix keyword (see get_matrix)
            + 2D list or np.ndarray of numbers

        :return matrix: (2D list)
        """
        from psyneulink.components.projections.projection import Projection
        if isinstance(self.owner, Projection):
            # Matrix provided (and validated in _validate_params); convert to np.array
            if isinstance(specification, np.matrix):
                return np.array(specification)

            sender = self.instance_defaults.variable
            sender_len = sender.shape[0]
            try:
                receiver = self.receiver
            except:
                raise FunctionError("Can't instantiate matrix specification ({}) for the {} function of {} "
                                    "since its receiver has not been specified".
                                    format(specification, self.name, self.owner_name))
                # receiver = sender
            receiver_len = receiver.shape[0]

            matrix = get_matrix(specification, rows=sender_len, cols=receiver_len, context=context)

            # This should never happen (should have been picked up in validate_param or above)
            if matrix is None:
                raise FunctionError("MATRIX param ({}) for the {} function of {} must be a matrix, a function "
                                    "that returns one, a matrix specification keyword ({}), or a number (filler)".
                                    format(specification, self.name, self.owner_name, MATRIX_KEYWORD_NAMES))
            else:
                return matrix
        else:
            return np.array(specification)

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """
        Return: `variable <LinearMatrix.variable>` • `matrix <LinearMatrix.matrix>`

        Arguments
        ---------
        variable : list or 1d np.array
            array to be transformed;  length must equal the number of rows of 'matrix <LinearMatrix.matrix>`.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        ---------

        dot product of variable and matrix : 1d np.array
            length of the array returned equals the number of columns of `matrix <LinearMatrix.matrix>`.

        """

        # Note: this calls _validate_variable and _validate_params which are overridden above;
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))
        matrix = self.get_current_function_param(MATRIX)
        return np.dot(variable, matrix)

    @staticmethod
    def keyword(obj, keyword):

        from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
        rows = None
        cols = None
        # use of variable attribute here should be ok because it's using it as a format/type
        if isinstance(obj, MappingProjection):
            if isinstance(obj.sender.value, numbers.Number):
                rows = 1
            else:
                rows = len(obj.sender.value)
            if isinstance(obj.receiver.instance_defaults.variable, numbers.Number):
                cols = 1
            else:
                cols = len(obj.receiver.instance_defaults.variable)
        matrix = get_matrix(keyword, rows, cols)

        if matrix is None:
            raise FunctionError("Unrecognized keyword ({}) specified for the {} function of {}".
                                format(keyword, obj.name, obj.owner_name))
        else:
            return matrix

    def param_function(owner, function):
        sender_len = len(owner.sender.value)
        receiver_len = len(owner.receiver.instance_defaults.variable)
        return function(sender_len, receiver_len)


# def is_matrix_spec(m):
#     if m is None:
#         return True
#     if isinstance(m, (list, np.ndarray, np.matrix, function_type)):
#         return True
#     if m in MATRIX_KEYWORD_VALUES:
#         return True
#     return False

def get_matrix(specification, rows=1, cols=1, context=None):
    """Returns matrix conforming to specification with dimensions = rows x cols or None

     Specification can be a matrix keyword, filler value or np.ndarray

     Specification (validated in _validate_params):
        + single number (used to fill self.matrix)
        + matrix keyword:
            + AUTO_ASSIGN_MATRIX: IDENTITY_MATRIX if it is square, othwerwise FULL_CONNECTIVITY_MATRIX
            + IDENTITY_MATRIX: 1's on diagonal, 0's elsewhere (must be square matrix), otherwise generates error
            + HOLLOW_MATRIX: 0's on diagonal, 1's elsewhere (must be square matrix), otherwise generates error
            + FULL_CONNECTIVITY_MATRIX: all 1's
            + RANDOM_CONNECTIVITY_MATRIX (random floats uniformly distributed between 0 and 1)
        + 2D list or np.ndarray of numbers

     Returns 2D np.array with length=rows in dim 0 and length=cols in dim 1, or none if specification is not recognized
    """

    # Matrix provided (and validated in _validate_params); convert to np.array
    if isinstance(specification, (list, np.matrix)):
        specification = np.array(specification)

    if isinstance(specification, np.ndarray):
        if specification.ndim == 2:
            return specification
        # FIX: MAKE THIS AN np.array WITH THE SAME DIMENSIONS??
        elif specification.ndim < 2:
            return np.atleast_2d(specification)
        else:
            raise FunctionError("Specification of np.array for matrix ({}) is more than 2d".
                                format(specification))

    if specification == AUTO_ASSIGN_MATRIX:
        if rows == cols:
            specification = IDENTITY_MATRIX
        else:
            specification = FULL_CONNECTIVITY_MATRIX

    if specification == FULL_CONNECTIVITY_MATRIX:
        return np.full((rows, cols), 1.0)

    if specification == IDENTITY_MATRIX:
        if rows != cols:
            raise FunctionError("Sender length ({}) must equal receiver length ({}) to use {}".
                                format(rows, cols, specification))
        return np.identity(rows)

    if specification == HOLLOW_MATRIX:
        if rows != cols:
            raise FunctionError("Sender length ({}) must equal receiver length ({}) to use {}".
                                format(rows, cols, specification))
        return 1-np.identity(rows)

    if specification == RANDOM_CONNECTIVITY_MATRIX:
        return np.random.rand(rows, cols)

    # Function is specified, so assume it uses random.rand() and call with sender_len and receiver_len
    if isinstance(specification, function_type):
        return specification(rows, cols)

    # (7/12/17 CW) this is a PATCH (like the one in MappingProjection) to allow users to
    # specify 'matrix' as a string (e.g. r = RecurrentTransferMechanism(matrix='1 2; 3 4'))
    if type(specification) == str:
        try:
            return np.array(np.matrix(specification))
        except (ValueError, NameError, TypeError):
            # np.matrix(specification) will give ValueError if specification is a bad value (e.g. 'abc', '1; 1 2')
            #                          [JDC] actually gives NameError if specification is a string (e.g., 'abc')
            pass

    # Specification not recognized
    return None


# region ***********************************  INTEGRATOR FUNCTIONS *****************************************************

#  Integrator
#  DDM_BogaczEtAl
#  DDM_NavarroAndFuss

class IntegratorFunction(Function_Base):
    componentType = INTEGRATOR_FUNCTION_TYPE

# • why does integrator return a 2d array?
# • are rate and noise converted to 1d np.array?  If not, correct docstring
# • can noise and initializer be an array?  If so, validated in validate_param?

class Integrator(IntegratorFunction):  # -------------------------------------------------------------------------------
    """
    Integrator(                 \
        default_variable=None,  \
        rate=1.0,               \

        noise=0.0,              \
        time_step_size=1.0,     \
        initializer,     \
        params=None,            \
        owner=None,             \
        prefs=None,             \
        )

    .. _Integrator:

    Integrate current value of `variable <Integrator.variable>` with its prior value.

    Arguments
    ---------

    default_variable : number, list or np.array : default ClassDefaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float, list or 1d np.array : default 1.0
        specifies the rate of integration.  If it is a list or array, it must be the same length as
        `variable <Integrator.default_variable>` (see `rate <Integrator.rate>` for details).

    noise : float, PsyNeuLink Function, list or 1d np.array : default 0.0
        specifies random value to be added in each call to `function <Integrator.function>`. (see
        `noise <Integrator.noise>` for details).

    time_step_size : float : default 0.0
        determines the timing precision of the integration process

    initializer float, list or 1d np.array : default 0.0
        specifies starting value for integration.  If it is a list or array, it must be the same length as
        `default_variable <Integrator.default_variable>` (see `initializer <Integrator.initializer>` for details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : number or np.array
        current input value some portion of which (determined by `rate <Integrator.rate>`) that will be
        added to the prior value;  if it is an array, each element is independently integrated.

    rate : float or 1d np.array
        determines the rate of integration based on current and prior values.  If integration_type is set to ADAPTIVE,
        all elements must be between 0 and 1 (0 = no change; 1 = instantaneous change). If it has a single element, it
        applies to all elements of `variable <Integrator.variable>`;  if it has more than one element, each element
        applies to the corresponding element of `variable <Integrator.variable>`.

    noise : float, function, list, or 1d np.array
        specifies random value to be added in each call to `function <Integrator.function>`.

        If noise is a list or array, it must be the same length as `variable <Integrator.default_variable>`. If noise is
        specified as a single float or function, while `variable <Integrator.variable>` is a list or array,
        noise will be applied to each variable element. In the case of a noise function, this means that the function
        will be executed separately for each variable element.

        Note that in the case of DIFFUSION, noise must be specified as a float (or list or array of floats) because this
        value will be used to construct the standard DDM probability distribution. For all other types of integration,
        in order to generate random noise, we recommend that you instead select a probability distribution function
        (see `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value from
        its distribution on each execution. If noise is specified as a float or as a function with a fixed output (or a
        list or array of these), then the noise will simply be an offset that remains the same across all executions.

    initializer : 1d np.array or list
        determines the starting value for integration (i.e., the value to which
        `previous_value <Integrator.previous_value>` is set.

        If initializer is a list or array, it must be the same length as `variable <Integrator.default_variable>`. If
        initializer is specified as a single float or function, while `variable <Integrator.variable>` is a list or
        array, initializer will be applied to each variable element. In the case of an initializer function, this means
        that the function will be executed separately for each variable element.

    previous_value : 1d np.array : default ClassDefaults.variable
        stores previous value with which `variable <Integrator.variable>` is integrated.

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = INTEGRATOR_FUNCTION

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    # paramClassDefaults.update({INITIALIZER: ClassDefaults.variable})
    paramClassDefaults.update({
        NOISE: None,
        RATE: None
    })

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate: parameter_spec = 1.0,
                 noise=0.0,
                 initializer=None,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context=None):

        if initializer is None:
            if params is not None and INITIALIZER in params and params[INITIALIZER] is not None:
                # This is only needed as long as a new copy of a function is created
                # whenever assigning the function to a mechanism.
                # The old values are compiled and passed in through params argument.
                initializer = params[INITIALIZER]
            else:
                initializer = self.ClassDefaults.variable

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  initializer=initializer,
                                                  noise=noise,
                                                  params=params)


        # Assign here as default, for use in initialization of function
        self.previous_value = initializer
        # does not actually get set in _assign_args_to_param_dicts but we need it as an instance_default
        params[INITIALIZER] = initializer

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        self.initializer = initializer

        # Reassign to kWInitializer in case default value was overridden
        self.previous_value = self.initializer

        self.auto_dependent = True

    def _validate(self):
        self._validate_rate(self.instance_defaults.rate)
        super()._validate()

    def _validate_params(self, request_set, target_set=None, context=None):

        # Handle list or array for rate specification
        if RATE in request_set:
            rate = request_set[RATE]

            if isinstance(rate, (list, np.ndarray)) and not iscompatible(rate, self.instance_defaults.variable):
                if len(rate) != 1 and len(rate) != np.array(self.instance_defaults.variable).size:
                    # If the variable was not specified, then reformat it to match rate specification
                    #    and assign ClassDefaults.variable accordingly
                    # Note: this situation can arise when the rate is parametrized (e.g., as an array) in the
                    #       Integrator's constructor, where that is used as a specification for a function parameter
                    #       (e.g., for an IntegratorMechanism), whereas the input is specified as part of the
                    #       object to which the function parameter belongs (e.g., the IntegratorMechanism); in that
                    #       case, the Integrator gets instantiated using its ClassDefaults.variable ([[0]]) before
                    #       the object itself, thus does not see the array specification for the input.
                    if self._default_variable_flexibility is DefaultsFlexibility.FLEXIBLE:
                        self._instantiate_defaults(variable=np.zeros_like(np.array(rate)), context=context)
                        if self.verbosePref:
                            warnings.warn(
                                "The length ({}) of the array specified for the rate parameter ({}) of {} "
                                "must match the length ({}) of the default input ({});  "
                                "the default input has been updated to match".format(
                                    len(rate),
                                    rate,
                                    self.name,
                                    np.array(self.instance_defaults.variable).size
                                ),
                                self.instance_defaults.variable,
                            )
                    else:
                        raise FunctionError(
                            "The length of the array specified for the rate parameter of {} ({})"
                            "must match the length of the default input ({}).".format(
                                len(rate),
                                # rate,
                                self.name,
                                np.array(self.instance_defaults.variable).size,
                                # self.instance_defaults.variable,
                            )
                        )
                # OLD:
                # self.paramClassDefaults[RATE] = np.zeros_like(np.array(rate))

                # KAM changed 5/15 b/c paramClassDefaults were being updated and *requiring* future integrator functions
                # to have a rate parameter of type ndarray/list

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        # if INITIALIZER in target_set:
        #     print(target_set)
        #     self._validate_initializer(target_set[INITIALIZER])

        if NOISE in target_set:
            noise = target_set[NOISE]
            if isinstance(noise, DistributionFunction):
                noise.owner = self
                target_set[NOISE] = noise._execute
            self._validate_noise(target_set[NOISE])

    def _validate_rate(self, rate):
        # kmantel: this duplicates much code in _validate_params above, but that calls _instantiate_defaults
        # which I don't think is the right thing to do here, but if you don't call it in _validate_params
        # then a lot of things don't get instantiated properly
        if rate is not None:
            if isinstance(rate, list):
                rate = np.asarray(rate)

            rate_type_msg = 'The rate parameter of {0} must be a number or an array/list of at most 1d (you gave: {1})'
            if isinstance(rate, np.ndarray):
                # kmantel: current test_gating test depends on 2d rate
                #   this should be looked at but for now this restriction is removed
                # if rate.ndim > 1:
                #     raise FunctionError(rate_type_msg.format(self.name, rate))
                pass
            elif not isinstance(rate, numbers.Number):
                raise FunctionError(rate_type_msg.format(self.name, rate))

            if isinstance(rate, np.ndarray) and not iscompatible(rate, self.instance_defaults.variable):
                if len(rate) != 1 and len(rate) != np.array(self.instance_defaults.variable).size:
                    if self._default_variable_flexibility is DefaultsFlexibility.FLEXIBLE:
                        self.instance_defaults.variable = np.zeros_like(np.array(rate))
                        if self.verbosePref:
                            warnings.warn(
                                "The length ({}) of the array specified for the rate parameter ({}) of {} "
                                "must match the length ({}) of the default input ({});  "
                                "the default input has been updated to match".format(
                                    len(rate),
                                    rate,
                                    self.name,
                                    np.array(self.instance_defaults.variable).size
                                ),
                                self.instance_defaults.variable,
                            )
                        self._instantiate_value()
                        self._default_variable_flexibility = DefaultsFlexibility.INCREASE_DIMENSION
                    else:
                        raise FunctionError(
                            "The length of the array specified for the rate parameter of {} ({})"
                            "must match the length of the default input ({}).".format(
                                len(rate),
                                # rate,
                                self.name,
                                np.array(self.instance_defaults.variable).size,
                                # self.instance_defaults.variable,
                            )
                        )

    # Ensure that the noise parameter makes sense with the input type and shape; flag any noise functions that will
    # need to be executed

    def _validate_noise(self, noise):
        # Noise is a list or array
        if isinstance(noise, (np.ndarray, list)):
            if len(noise) == 1:
                pass
            # Variable is a list/array
            elif (not iscompatible(np.atleast_2d(noise), self.instance_defaults.variable)
                  and not iscompatible(np.atleast_1d(noise), self.instance_defaults.variable) and len(noise) > 1):
                raise FunctionError(
                        "Noise parameter ({}) does not match default variable ({}). Noise parameter of {} "
                        "must be specified as a float, a function, or an array of the appropriate shape ({})."
                            .format(noise, self.instance_defaults.variable, self.name, np.shape(np.array(self.instance_defaults.variable))))
            else:
                for i in range(len(noise)):
                    if isinstance(noise[i], DistributionFunction):
                        noise[i] = noise[i]._execute
                    if not isinstance(noise[i], (float, int)) and not callable(noise[i]):
                        raise FunctionError("The elements of a noise list or array must be floats or functions. "
                                            "{} is not a valid noise element for {}".format(noise[i], self.name))

        # Otherwise, must be a float, int or function
        elif not isinstance(noise, (float, int)) and not callable(noise):
            raise FunctionError(
                "Noise parameter ({}) for {} must be a float, function, or array/list of these."
                    .format(noise, self.name))


    def _try_execute_param(self, param, var):

        # param is a list; if any element is callable, execute it
        if isinstance(param, (np.ndarray, list)):
            # NOTE: np.atleast_2d will cause problems if the param has "rows" of different lengths
            param = np.atleast_2d(param)
            for i in range(len(param)):
                for j in range(len(param[i])):
                    if callable(param[i][j]):
                        param[i][j] = param[i][j]()
        # param is one function
        elif callable(param):
            # NOTE: np.atleast_2d will cause problems if the param has "rows" of different lengths
            new_param = []
            for row in np.atleast_2d(var):
                new_row = []
                for item in row:
                    new_row.append(param())
                new_param.append(new_row)
            param = new_param

        return param

    def _euler(self, previous_value, previous_time, slope, time_step_size):

        if callable(slope):
            slope = slope(previous_time, previous_value)

        return previous_value + slope*time_step_size

    def _runge_kutta_4(self, previous_value, previous_time, slope, time_step_size):

        if callable(slope):
            slope_approx_1 = slope(previous_time,
                                        previous_value)

            slope_approx_2 = slope(previous_time + time_step_size/2,
                                        previous_value + (0.5 * time_step_size * slope_approx_1))

            slope_approx_3 = slope(previous_time + time_step_size/2,
                                        previous_value + (0.5 * time_step_size * slope_approx_2))

            slope_approx_4 = slope(previous_time + time_step_size,
                                        previous_value + (time_step_size * slope_approx_3))

            value = previous_value \
                    + (time_step_size/6)*(slope_approx_1 + 2*(slope_approx_2 + slope_approx_3) + slope_approx_4)

        else:
            value = previous_value + time_step_size*slope

        return value

    def reinitialize(self, new_previous_value=None, **kwargs):
        """
            Effectively begins accumulation over again at the specified value.

            Sets

            - `previous_value <Integrator.previous_value>`
            - `initializer <Integrator.initial_value>`
            - `value <Integrator.value>`

            to the quantity specified.

            For specific types of Integrator functions, additional values, such as initial time, must be specified, and
            additional attributes are reset.

            If no arguments are specified, then the instance default for `initializer <Integrator.initializer>` is used.
        """
        if new_previous_value is None:
            new_previous_value = self.instance_defaults.initializer
        self._initializer = new_previous_value
        self.value = new_previous_value
        self.previous_value = new_previous_value
        return self.value

    def function(self, *args, **kwargs):
        raise FunctionError("Integrator is not meant to be called explicitly")

class SimpleIntegrator(
    Integrator):  # --------------------------------------------------------------------------------
    """
    SimpleIntegrator(                 \
        default_variable=None,  \
        rate=1.0,               \
        noise=0.0,              \
        initializer,            \
        params=None,            \
        owner=None,             \
        prefs=None,             \
        )

    .. _SimpleIntegrator:

    Integrate current value of `variable <SimpleIntegrator.variable>` with its prior value:

    `previous_value <SimpleIntegrator.previous_value>` + \
    `rate <SimpleIntegrator.rate>` *`variable <variable.SimpleIntegrator.variable>` + \
    `noise <SimpleIntegrator.noise>`;

    Arguments
    ---------

    default_variable : number, list or np.array : default ClassDefaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float, list or 1d np.array : default 1.0
        specifies the rate of integration.  If it is a list or array, it must be the same length as
        `variable <SimpleIntegrator.default_variable>` (see `rate <SimpleIntegrator.rate>` for details).

    noise : float, PsyNeuLink Function, list or 1d np.array : default 0.0
        specifies random value to be added in each call to `function <SimpleIntegrator.function>`. (see
        `noise <SimpleIntegrator.noise>` for details).

    initializer float, list or 1d np.array : default 0.0
        specifies starting value for integration.  If it is a list or array, it must be the same length as
        `default_variable <SimpleIntegrator.default_variable>` (see `initializer <SimpleIntegrator.initializer>`
        for details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : number or np.array
        current input value some portion of which (determined by `rate <SimpleIntegrator.rate>`) will be
        added to the prior value;  if it is an array, each element is independently integrated.

    rate : float or 1d np.array
        determines the rate of integration based on current and prior values. If it has a single element, it applies
        to all elements of `variable <SimpleIntegrator.variable>`;  if it has more than one element, each element
        applies to the corresponding element of `variable <SimpleIntegrator.variable>`.

    noise : float, function, list, or 1d np.array
        specifies random value to be added in each call to `function <SimpleIntegrator.function>`.

        If noise is a list or array, it must be the same length as `variable <SimpleIntegrator.default_variable>`.

        If noise is specified as a single float or function, while `variable <SimpleIntegrator.variable>` is a list or
        array, noise will be applied to each variable element. In the case of a noise function, this means that the
        function will be executed separately for each variable element.


        .. note::
            In order to generate random noise, we recommend selecting a probability distribution function (see
            `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value from
            its distribution on each execution. If noise is specified as a float or as a function with a fixed output,
            then the noise will simply be an offset that remains the same across all executions.

    initializer : float, 1d np.array or list
        determines the starting value for integration (i.e., the value to which
        `previous_value <SimpleIntegrator.previous_value>` is set.

        If initializer is a list or array, it must be the same length as `variable <SimpleIntegrator.default_variable>`.

    previous_value : 1d np.array : default ClassDefaults.variable
        stores previous value with which `variable <SimpleIntegrator.variable>` is integrated.

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = SIMPLE_INTEGRATOR_FUNCTION

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    # paramClassDefaults.update({INITIALIZER: ClassDefaults.variable})
    paramClassDefaults.update({
        NOISE: None,
        RATE: None
    })

    multiplicative_param = RATE
    additive_param = OFFSET

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate: parameter_spec=1.0,
                 noise=0.0,
                 offset=None,
                 initializer=None,
                 params: tc.optional(dict)=None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  initializer=initializer,
                                                  noise=noise,
                                                  offset=offset,
                                                  params=params)

        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)

        self.auto_dependent = True

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """
        Return: `variable <Linear.slope>` combined with `previous_value <SimpleIntegrator.previous_value>`
        according to `previous_value <SimpleIntegrator.previous_value>` + `rate <SimpleIntegrator.rate>` *`variable
        <variable.SimpleIntegrator.variable>` + `noise <SimpleIntegrator.noise>`;

        Arguments
        ---------

        variable : number, list or np.array : default ClassDefaults.variable
           a single value or array of values to be integrated.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        updated value of integral : 2d np.array

        """

        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        rate = np.array(self.get_current_function_param(RATE)).astype(float)

        offset = self.get_current_function_param(OFFSET)
        if offset is None:
            offset = 0.0

        # execute noise if it is a function
        noise = self._try_execute_param(self.get_current_function_param(NOISE), variable)
        previous_value = self.previous_value
        new_value = variable

        value = previous_value + (new_value * rate) + noise

        adjusted_value = value + offset

        # If this NOT an initialization run, update the old value
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if self.context.initialization_status != ContextFlags.INITIALIZING:
            self.previous_value = adjusted_value

        return adjusted_value

class LCAIntegrator(
    Integrator):  # --------------------------------------------------------------------------------
    """
    LCAIntegrator(                  \
        default_variable=None,      \
        noise=0.0,                  \
        initializer=0.0,            \
        rate=1.0,                   \
        offset=None,                \
        time_step_size=0.1,         \
        params=None,                \
        owner=None,                 \
        prefs=None,                 \
        )

    .. _LCAIntegrator:

    Integrate current value of `variable <LCAIntegrator.variable>` with its prior value:

    .. math::

        rate \\cdot previous\\_value + variable + noise \\sqrt{time\\_step\\_size}

    COMMENT:
    `rate <LCAIntegrator.rate>` * `previous_value <LCAIntegrator.previous_value>` + \
    `variable <variable.LCAIntegrator.variable>` + \
    `noise <LCAIntegrator.noise>`;
    COMMENT

    Arguments
    ---------

    default_variable : number, list or np.array : default ClassDefaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float, list or 1d np.array : default 1.0
        scales the contribution of `previous_value <LCAIntegrator.previous_value>` to the accumulation of the
        `value <LCAIntegrator.value>` on each time step

    noise : float, PsyNeuLink Function, list or 1d np.array : default 0.0
        specifies random value to be added in each call to `function <LCAIntegrator.function>`. (see
        `noise <LCAIntegrator.noise>` for details).

    initializer : float, list or 1d np.array : default 0.0
        specifies starting value for integration.  If it is a list or array, it must be the same length as
        `default_variable <LCAIntegrator.default_variable>` (see `initializer <LCAIntegrator.initializer>` for details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : number or np.array
        current input value some portion of which (determined by `rate <LCAIntegrator.rate>`) will be
        added to the prior value;  if it is an array, each element is independently integrated.

    rate : float or 1d np.array
        scales the contribution of `previous_value <LCAIntegrator.previous_value>` to the
        accumulation of the `value <LCAIntegrator.value>` on each time step. If rate has a single element, it
        applies to all elements of `variable <LCAIntegrator.variable>`;  if rate has more than one element, each element
        applies to the corresponding element of `variable <LCAIntegrator.variable>`.

    noise : float, function, list, or 1d np.array
        specifies a value to be added in each call to `function <LCAIntegrator.function>`.

        If noise is a list or array, it must be the same length as `variable <LCAIntegrator.default_variable>`.

        If noise is specified as a single float or function, while `variable <LCAIntegrator.variable>` is a list or
        array, noise will be applied to each variable element. In the case of a noise function, this means that the
        function will be executed separately for each variable element.

        .. note::
            In order to generate random noise, we recommend selecting a probability distribution function (see
            `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value from
            its distribution on each execution. If noise is specified as a float or as a function with a fixed output,
            then the noise will simply be an offset that remains the same across all executions.

    initializer : float, 1d np.array or list
        determines the starting value for integration (i.e., the value to which
        `previous_value <LCAIntegrator.previous_value>` is set.

        If initializer is a list or array, it must be the same length as `variable <LCAIntegrator.default_variable>`.

    previous_value : 1d np.array : default ClassDefaults.variable
        stores previous value with which `variable <LCAIntegrator.variable>` is integrated.

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = SIMPLE_INTEGRATOR_FUNCTION

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    # paramClassDefaults.update({INITIALIZER: ClassDefaults.variable})
    paramClassDefaults.update({
        NOISE: None,
        RATE: None
    })

    multiplicative_param = RATE
    additive_param = OFFSET

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate: parameter_spec=1.0,
                 noise=0.0,
                 offset=None,
                 initializer=None,
                 time_step_size=0.1,
                 params: tc.optional(dict)=None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  initializer=initializer,
                                                  noise=noise,
                                                  time_step_size=time_step_size,
                                                  offset=offset,
                                                  params=params)

        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)

        self.auto_dependent = True

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """
        Return:

        .. math::

            rate \\cdot previous\\_value + variable + noise \\sqrt{time\\_step\\_size}

        Arguments
        ---------

        variable : number, list or np.array : default ClassDefaults.variable
           a single value or array of values to be integrated.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        updated value of integral : 2d np.array

        """

        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        rate = np.atleast_1d(self.get_current_function_param(RATE))
        initializer = self.get_current_function_param(INITIALIZER)  # unnecessary?
        time_step_size = self.get_current_function_param(TIME_STEP_SIZE)
        offset = self.get_current_function_param(OFFSET)

        if offset is None:
            offset = 0.0

        # execute noise if it is a function
        noise = self._try_execute_param(self.get_current_function_param(NOISE), variable)
        previous_value = self.previous_value
        new_value = variable

        # Gilzenrat: previous_value + (-previous_value + variable)*self.time_step_size + noise --> rate = -1
        value = previous_value + (rate*previous_value + new_value)*time_step_size + noise*(time_step_size**0.5)

        adjusted_value = value + offset

        # If this NOT an initialization run, update the old value
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if self.context.initialization_status != ContextFlags.INITIALIZING:
            self.previous_value = adjusted_value

        return adjusted_value

class ConstantIntegrator(Integrator):  # -------------------------------------------------------------------------------
    """
    ConstantIntegrator(                 \
        default_variable=None,          \
        rate=1.0,                       \
        noise=0.0,                      \
        scale: parameter_spec = 1.0,    \
        offset: parameter_spec = 0.0,   \
        initializer,                    \
        params=None,                    \
        owner=None,                     \
        prefs=None,                     \
        )

    .. _ConstantIntegrator:

    Integrates prior value by adding `rate <Integrator.rate>` and `noise <Integrator.noise>`. (Ignores
    `variable <Integrator.variable>`).

    `previous_value <ConstantIntegrator.previous_value>` + `rate <ConstantIntegrator.rate>` +
    `noise <ConstantIntegrator.noise>`

    Arguments
    ---------

    default_variable : number, list or np.array : default ClassDefaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float, list or 1d np.array : default 1.0
        specifies the rate of integration.  If it is a list or array, it must be the same length as
        `variable <ConstantIntegrator.default_variable>` (see `rate <ConstantIntegrator.rate>` for details).

    noise : float, PsyNeuLink Function, list or 1d np.array : default 0.0
        specifies random value to be added in each call to `function <ConstantIntegrator.function>`. (see
        `noise <ConstantIntegrator.noise>` for details).

    initializer float, list or 1d np.array : default 0.0
        specifies starting value for integration.  If it is a list or array, it must be the same length as
        `default_variable <ConstantIntegrator.default_variable>` (see `initializer <ConstantIntegrator.initializer>`
        for details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : number or np.array
        **Ignored** by the ConstantIntegrator function. Refer to LCAIntegrator or AdaptiveIntegrator for integrator
         functions that depend on both a prior value and a new value (variable).

    rate : float or 1d np.array
        determines the rate of integration.

        If it has a single element, that element is added to each element of
        `previous_value <ConstantIntegrator.previous_value>`.

        If it has more than one element, each element is added to the corresponding element of
        `previous_value <ConstantIntegrator.previous_value>`.

    noise : float, function, list, or 1d np.array
        specifies random value to be added in each call to `function <ConstantIntegrator.function>`.

        If noise is a list or array, it must be the same length as `variable <ConstantIntegrator.default_variable>`.

        If noise is specified as a single float or function, while `variable <ConstantIntegrator.variable>` is a list
        or array, noise will be applied to each variable element. In the case of a noise function, this means that
        the function will be executed separately for each variable element.

        .. note::
            In order to generate random noise, we recommend selecting a probability distribution function (see
            `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value from
            its distribution on each execution. If noise is specified as a float or as a function with a fixed output,
            then the noise will simply be an offset that remains the same across all executions.

    initializer : float, 1d np.array or list
        determines the starting value for integration (i.e., the value to which
        `previous_value <ConstantIntegrator.previous_value>` is set.

        If initializer is a list or array, it must be the same length as
        `variable <ConstantIntegrator.default_variable>`.

    previous_value : 1d np.array : default ClassDefaults.variable
        stores previous value to which `rate <ConstantIntegrator.rate>` and `noise <ConstantIntegrator.noise>` will be
        added.

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = CONSTANT_INTEGRATOR_FUNCTION

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    # paramClassDefaults.update({INITIALIZER: ClassDefaults.variable})
    paramClassDefaults.update({
        NOISE: None,
        RATE: None,
        OFFSET: None,
        SCALE: None,
    })

    multiplicative_param = SCALE
    additive_param = RATE

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 # rate: parameter_spec = 1.0,
                 rate=0.0,
                 noise=0.0,
                 offset=0.0,
                 scale = 1.0,
                 initializer=None,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  initializer=initializer,
                                                  noise=noise,
                                                  scale = scale,
                                                  offset=offset,
                                                  params=params)

        # Assign here as default, for use in initialization of function
        self.previous_value = initializer

        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)

        # Reassign to initializer in case default value was overridden

        self.auto_dependent = True

    def _validate_rate(self, rate):
        # unlike other Integrators, variable does not need to match rate

        if isinstance(rate, list):
            rate = np.asarray(rate)

        rate_type_msg = 'The rate parameter of {0} must be a number or an array/list of at most 1d (you gave: {1})'
        if isinstance(rate, np.ndarray):
            # kmantel: current test_gating test depends on 2d rate
            #   this should be looked at but for now this restriction is removed
            # if rate.ndim > 1:
            #     raise FunctionError(rate_type_msg.format(self.name, rate))
            pass
        elif not isinstance(rate, numbers.Number):
            raise FunctionError(rate_type_msg.format(self.name, rate))

        if self._default_variable_flexibility is DefaultsFlexibility.FLEXIBLE:
            self.instance_defaults.variable = np.zeros_like(np.array(rate))
            self._instantiate_value()
            self._default_variable_flexibility = DefaultsFlexibility.INCREASE_DIMENSION

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """
        Return: the sum of `previous_value <ConstantIntegrator.previous_value>`, `rate <ConstantIntegrator.rate>`, and
        `noise <ConstantIntegrator.noise>`.

        Arguments
        ---------

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        updated value of integral : 2d np.array

        """
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        rate = np.array(self.rate).astype(float)
        offset = self.get_current_function_param(OFFSET)
        scale = self.get_current_function_param(SCALE)
        noise = self._try_execute_param(self.noise, variable)

        previous_value = np.atleast_2d(self.previous_value)

        value = previous_value + rate + noise

        adjusted_value = value*scale + offset

        # If this NOT an initialization run, update the old value
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if self.context.initialization_status != ContextFlags.INITIALIZING:
            self.previous_value = adjusted_value

        return adjusted_value

class AdaptiveIntegrator(Integrator):  # -------------------------------------------------------------------------------
    """
    AdaptiveIntegrator(                 \
        default_variable=None,          \
        rate=1.0,                       \
        noise=0.0,                      \
        scale: parameter_spec = 1.0,    \
        offset: parameter_spec = 0.0,   \
        initializer,                    \
        params=None,                    \
        owner=None,                     \
        prefs=None,                     \
        )

    .. _AdaptiveIntegrator:

    Computes an exponentially weighted moving average.

    (1 - `rate <AdaptiveIntegrator.rate>`) * `previous_value <AdaptiveIntegrator.previous_value>` + `rate
    <AdaptiveIntegrator.rate>` * `variable <AdaptiveIntegrator.variable>` + `noise <AdaptiveIntegrator.noise>`


    Arguments
    ---------

    default_variable : number, list or np.array : default ClassDefaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float, list or 1d np.array : default 1.0
        specifies the smoothing factor of the EWMA.  If it is a list or array, it must be the same length as
        `variable <AdaptiveIntegrator.default_variable>` (see `rate <AdaptiveIntegrator.rate>` for details).

    noise : float, PsyNeuLink Function, list or 1d np.array : default 0.0
        specifies random value to be added in each call to `function <AdaptiveIntegrator.function>`. (see
        `noise <AdaptiveIntegrator.noise>` for details).

    initializer float, list or 1d np.array : default 0.0
        specifies starting value for integration.  If it is a list or array, it must be the same length as
        `default_variable <AdaptiveIntegrator.default_variable>` (see `initializer <AdaptiveIntegrator.initializer>`
        for details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : number or np.array
        current input value some portion of which (determined by `rate <AdaptiveIntegrator.rate>`) will be
        added to the prior value;  if it is an array, each element is independently integrated.

    rate : float or 1d np.array
        determines the smoothing factor of the EWMA. All rate elements must be between 0 and 1 (rate = 0 --> no change,
        `variable <AdaptiveAdaptiveIntegrator.variable>` is ignored; rate = 1 -->
        `previous_value <AdaptiveIntegrator.previous_value>` is ignored).

        If rate is a float, it is applied to all elements of `variable <AdaptiveAdaptiveIntegrator.variable>` (and
        `previous_value <AdaptiveIntegrator.previous_value>`); if it has more than one element, each element is applied
        to the corresponding element of `variable <AdaptiveAdaptiveIntegrator.variable>` (and
        `previous_value <AdaptiveIntegrator.previous_value>`).

    noise : float, function, list, or 1d np.array
        specifies random value to be added in each call to `function <AdaptiveIntegrator.function>`.

        If noise is a list or array, it must be the same length as `variable <AdaptiveIntegrator.default_variable>`.

        If noise is specified as a single float or function, while `variable <AdaptiveIntegrator.variable>` is a list
        or array, noise will be applied to each variable element. In the case of a noise function, this means that
        the function will be executed separately for each variable element.

        .. note::
            In order to generate random noise, we recommend selecting a probability distribution function
            (see `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value from
            its distribution on each execution. If noise is specified as a float or as a function with a fixed output, then
            the noise will simply be an offset that remains the same across all executions.

    initializer : float, 1d np.array or list
        determines the starting value for time-averaging (i.e., the value to which
        `previous_value <AdaptiveIntegrator.previous_value>` is originally set).

        If initializer is a list or array, it must be the same length as
        `variable <AdaptiveIntegrator.default_variable>`.

    previous_value : 1d np.array : default ClassDefaults.variable
        stores previous value with which `variable <AdaptiveIntegrator.variable>` is integrated.

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = ADAPTIVE_INTEGRATOR_FUNCTION

    multiplicative_param = RATE
    additive_param = OFFSET

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    # paramClassDefaults.update({INITIALIZER: ClassDefaults.variable})
    paramClassDefaults.update({
        NOISE: None,
        RATE: None
    })

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate: parameter_spec = 1.0,
                 noise=0.0,
                 offset= 0.0,
                 time_step_size=0.02,
                 initializer=None,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  initializer=initializer,
                                                  noise=noise,
                                                  offset=offset,
                                                  time_step_size=time_step_size,
                                                  params=params)

        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)

        self.auto_dependent = True

    def _validate_params(self, request_set, target_set=None, context=None):

        # Handle list or array for rate specification
        if RATE in request_set:
            rate = request_set[RATE]
            if isinstance(rate, (list, np.ndarray)):
                if len(rate) != 1 and len(rate) != np.array(self.instance_defaults.variable).size:
                    # If the variable was not specified, then reformat it to match rate specification
                    #    and assign ClassDefaults.variable accordingly
                    # Note: this situation can arise when the rate is parametrized (e.g., as an array) in the
                    #       AdaptiveIntegrator's constructor, where that is used as a specification for a function
                    #       parameter (e.g., for an IntegratorMechanism), whereas the input is specified as part of the
                    #       object to which the function parameter belongs (e.g., the IntegratorMechanism);
                    #       in that case, the Integrator gets instantiated using its ClassDefaults.variable ([[0]])
                    #       before the object itself, thus does not see the array specification for the input.
                    if self._default_variable_flexibility is DefaultsFlexibility.FLEXIBLE:
                        self._instantiate_defaults(variable=np.zeros_like(np.array(rate)), context=context)
                        if self.verbosePref:
                            warnings.warn(
                                "The length ({}) of the array specified for the rate parameter ({}) of {} "
                                "must match the length ({}) of the default input ({});  "
                                "the default input has been updated to match".format(
                                    len(rate),
                                    rate,
                                    self.name,
                                    np.array(self.instance_defaults.variable).size
                                ),
                                self.instance_defaults.variable
                            )
                    else:
                        raise FunctionError(
                            "The length ({}) of the array specified for the rate parameter ({}) of {} "
                            "must match the length ({}) of the default input ({})".format(
                                len(rate),
                                rate,
                                self.name,
                                np.array(self.instance_defaults.variable).size,
                                self.instance_defaults.variable,
                            )
                        )
                        # OLD:
                        # self.paramClassDefaults[RATE] = np.zeros_like(np.array(rate))

                        # KAM changed 5/15 b/c paramClassDefaults were being updated and *requiring* future integrator
                        # function to have a rate parameter of type ndarray/list

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if RATE in target_set:
            # cannot use _validate_rate here because it assumes it's being run after instantiation of the object
            rate_value_msg = "The rate parameter ({}) (or all of its elements) of {} must be between 0.0 and 1.0 because it is an AdaptiveIntegrator"
            if isinstance(rate, np.ndarray) and rate.ndim > 0:
                for r in rate:
                    if r < 0.0 or r > 1.0:
                        raise FunctionError(rate_value_msg.format(rate, self.name))
            else:
                if rate < 0.0 or rate > 1.0:
                    raise FunctionError(rate_value_msg.format(rate, self.name))

        if NOISE in target_set:
            noise = target_set[NOISE]
            if isinstance(noise, DistributionFunction):
                noise.owner = self
                target_set[NOISE] = noise._execute
            self._validate_noise(target_set[NOISE])
        # if INITIALIZER in target_set:
        #     self._validate_initializer(target_set[INITIALIZER])

    def _validate_rate(self, rate):
        super()._validate_rate(rate)

        if isinstance(rate, list):
            rate = np.asarray(rate)

        rate_value_msg = "The rate parameter ({}) (or all of its elements) of {} must be between 0.0 and 1.0 because it is an AdaptiveIntegrator"
        if isinstance(rate, np.ndarray) and rate.ndim > 0:
            for r in rate:
                if r < 0.0 or r > 1.0:
                    raise FunctionError(rate_value_msg.format(rate, self.name))
        else:
            if rate < 0.0 or rate > 1.0:
                raise FunctionError(rate_value_msg.format(rate, self.name))

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """
        Return: some fraction of `variable <AdaptiveIntegrator.variable>` combined with some fraction of `previous_value
        <AdaptiveIntegrator.previous_value>`.

        Arguments
        ---------

        variable : number, list or np.array : default ClassDefaults.variable
           a single value or array of values to be integrated.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        updated value of integral : 2d np.array

        """
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        rate = np.array(self.get_current_function_param(RATE)).astype(float)
        offset = self.get_current_function_param(OFFSET)
        # execute noise if it is a function
        noise = self._try_execute_param(self.get_current_function_param(NOISE), variable)

        previous_value = np.atleast_2d(self.previous_value)

        value = (1-rate)*previous_value + rate*variable + noise
        adjusted_value = value + offset

        # If this NOT an initialization run, update the old value
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if self.context.initialization_status != ContextFlags.INITIALIZING:
            self.previous_value = adjusted_value
        return adjusted_value

class DriftDiffusionIntegrator(
    Integrator):  # --------------------------------------------------------------------------------
    """
    DriftDiffusionIntegrator(           \
        default_variable=None,          \
        rate=1.0,                       \
        noise=0.0,                      \
        scale= 1.0,                     \
        offset= 0.0,                    \
        time_step_size=1.0,             \
        t0=0.0,                         \
        decay=0.0,                      \
        threshold=1.0                   \
        initializer,                    \
        params=None,                    \
        owner=None,                     \
        prefs=None,                     \
        )

    .. _DriftDiffusionIntegrator:

    Accumulates evidence over time based on a stimulus, rate, previous position, and noise. Stops accumulating at a
    threshold.

    Arguments
    ---------

    default_variable : number, list or np.array : default ClassDefaults.variable
        specifies the stimulus component of drift rate -- the drift rate is the product of variable and rate

    rate : float, list or 1d np.array : default 1.0
        specifies the attentional component of drift rate -- the drift rate is the product of variable and rate

    noise : float, PsyNeuLink Function, list or 1d np.array : default 0.0
        scales the random value to be added in each call to `function <DriftDiffusionIntegrator.function>`. (see
        `noise <DriftDiffusionIntegrator.noise>` for details).

    time_step_size : float : default 0.0
        determines the timing precision of the integration process (see `time_step_size
        <DriftDiffusionIntegrator.time_step_size>` for details.

    t0 : float
        determines the start time of the integration process and is used to compute the RESPONSE_TIME output state of
        the DDM Mechanism.

    initializer : float, list or 1d np.array : default 0.0
        specifies starting value for integration.  If it is a list or array, it must be the same length as
        `default_variable <DriftDiffusionIntegrator.default_variable>` (see `initializer
        <DriftDiffusionIntegrator.initializer>` for details).

    threshold : float : default 0.0
        specifies the threshold (boundaries) of the drift diffusion process (i.e., at which the
        integration process is assumed to terminate).

        Once the magnitude of the decision variable has exceeded the threshold, the function will simply return the
        threshold magnitude (with the appropriate sign) for that execution and any future executions.

        If the function is in a `DDM mechanism <DDM>`, crossing the threshold will also switch the `is_finished`
        attribute from False to True. This attribute may be important for the `Scheduler <Scheduler>` when using
         `Conditions <Condition>` such as `WhenFinished <WhenFinished>`.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : number or np.array
        current input value, which represents the stimulus component of drift.

    rate : float or 1d np.array
        specifies the attentional component of drift rate -- the drift rate is the product of variable and rate

    noise : float, function, list, or 1d np.array
        scales the random value to be added in each call to `function <DriftDiffusionIntegrator.function> according to
        the standard DDM probability distribution.

        On each call to `function <DriftDiffusionIntegrator.function>, :math:`\\sqrt{time\\_step\\_size \\cdot noise}
        \\cdot Sample\\,From\\,Normal\\,distribution` is added to the accumulated evidence.

        Noise must be specified as a float (or list or array of floats).

    time_step_size : float
        determines the timing precision of the integration process and is used to scale the `noise
        <DriftDiffusionIntegrator.noise>` parameter according to the standard DDM probability distribution.

    t0 : float
        determines the start time of the integration process and is used to compute the RESPONSE_TIME output state of
        the DDM Mechanism.

    initializer : float, 1d np.array or list
        determines the starting value for integration (i.e., the value to which
        `previous_value <DriftDiffusionIntegrator.previous_value>` is set.

        If initializer is a list or array, it must be the same length as
        `variable <DriftDiffusionIntegrator.default_variable>`.

    previous_time : float
        stores previous time at which the function was executed and accumulates with each execution according to
        `time_step_size <DriftDiffusionIntegrator.default_time_step_size>`.

    previous_value : 1d np.array : default ClassDefaults.variable
        stores previous value with which `variable <DriftDiffusionIntegrator.variable>` is integrated.

    threshold : float : default 0.0
        when used properly determines the threshold (boundaries) of the drift diffusion process (i.e., at which the
        integration process is assumed to terminate).

        If the system is assembled as follows, then the DriftDiffusionIntegrator function stops accumulating when its
        value reaches +threshold or -threshold

            (1) the function is used in the `DDM mechanism <DDM>`

            (2) the mechanism is part of a `System <System>` with a `Scheduler <Scheduler>` which applies the
            `WhenFinished <WhenFinished>` `Condition <Condition>` to the mechanism

        Otherwise, `threshold <DriftDiffusionIntegrator.threshold>` does not influence the function at all.

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = DRIFT_DIFFUSION_INTEGRATOR_FUNCTION

    multiplicative_param = RATE
    additive_param = OFFSET

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    # paramClassDefaults.update({INITIALIZER: ClassDefaults.variable})
    paramClassDefaults.update({
        NOISE: None,
        RATE: None
    })

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate: parameter_spec = 1.0,
                 noise=0.0,
                 offset: parameter_spec = 0.0,
                 time_step_size=1.0,
                 t0=0.0,
                 initializer=None,
                 threshold=100.0,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  time_step_size=time_step_size,
                                                  t0=t0,
                                                  initializer=initializer,
                                                  threshold=threshold,
                                                  noise=noise,
                                                  offset=offset,
                                                  params=params)

        # Assign here as default, for use in initialization of function
        self.previous_value = initializer
        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)

        self.previous_time = self.t0
        self.auto_dependent = True

    def _validate_noise(self, noise):
        if not isinstance(noise, float):
            raise FunctionError(
                "Invalid noise parameter for {}. DriftDiffusionIntegrator requires noise parameter to be a float. Noise"
                " parameter is used to construct the standard DDM noise distribution".format(self.name))

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """
        Return: One time step of evidence accumulation according to the Drift Diffusion Model

        ..  math::

            previous\\_value + rate \\cdot variable \\cdot time\\_step\\_size + \\sqrt{time\\_step\\_size \\cdot noise}
            \\cdot Sample\\,from\\,Normal\\,Distribution

        Arguments
        ---------

        variable : number, list or np.array : default ClassDefaults.variable
            specifies the stimulus component of drift rate -- the drift rate is the product of variable and rate

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        updated value of integral : 2d np.array

        """
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        rate = np.array(self.get_current_function_param(RATE)).astype(float)
        offset = self.get_current_function_param(OFFSET)
        noise = self.get_current_function_param(NOISE)
        threshold = self.get_current_function_param(THRESHOLD)
        time_step_size = self.get_current_function_param(TIME_STEP_SIZE)

        previous_value = np.atleast_2d(self.previous_value)

        value = previous_value + rate * variable * time_step_size  \
                + np.sqrt(time_step_size * noise) * np.random.normal()

        if np.all(abs(value) < threshold):
            adjusted_value = value + offset
        elif np.all(value >= threshold):
            adjusted_value = np.atleast_2d(threshold)
        elif np.all(value <= -threshold):
            adjusted_value = np.atleast_2d(-threshold)

        # If this NOT an initialization run, update the old value and time
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if self.context.initialization_status != ContextFlags.INITIALIZING:
            self.previous_value = adjusted_value
            self.previous_time += time_step_size

        # FIX?
        # Current output format is [[[decision_variable]], time]
        return adjusted_value

    def reinitialize(self, new_previous_value=None, new_previous_time=None):
        """
        In effect, begins accumulation over again at the original starting point and time, or new ones.

        Sets

        - `previous_value <DriftDiffusionIntegrator.previous_value>`
        - `initializer <DriftDiffusionIntegrator.initializer>`
        - `value <DriftDiffusionIntegrator.value>`

        to the value specified in the first argument.

        Sets `previous_time <DriftDiffusionIntegrator.previous_time>` to the value specified in the second argument.

        If no arguments are specified, then the instance defaults for `initializer
        <DriftDiffusionIntegrator.initializer>` and `t0 <DriftDiffusionIntegrator.t0>` are used.
        """
        if new_previous_value is None:
            new_previous_value = self.instance_defaults.initializer
        if new_previous_time is None:
            new_previous_time = self.instance_defaults.t0
        self._initializer = new_previous_value
        self.value = new_previous_value
        self.previous_value = new_previous_value
        self.previous_time = new_previous_time
        return np.atleast_1d(new_previous_value), np.atleast_1d(new_previous_time)

class OrnsteinUhlenbeckIntegrator(
    Integrator):  # --------------------------------------------------------------------------------
    """
    OrnsteinUhlenbeckIntegrator(        \
        default_variable=None,          \
        rate=1.0,                       \
        noise=0.0,                      \
        offset= 0.0,                    \
        time_step_size=1.0,             \
        t0=0.0,                         \
        decay=1.0,                      \
        initializer=0.0,                \
        params=None,                    \
        owner=None,                     \
        prefs=None,                     \
        )

    .. _OrnsteinUhlenbeckIntegrator:

    Accumulate evidence overtime based on a stimulus, noise, decay, and previous position.

    Arguments
    ---------

    default_variable : number, list or np.array : default ClassDefaults.variable
        specifies a template for  the stimulus component of drift rate -- the drift rate is the product of variable and
        rate

    rate : float, list or 1d np.array : default 1.0
        specifies  the attentional component of drift rate -- the drift rate is the product of variable and rate

    noise : float, PsyNeuLink Function, list or 1d np.array : default 0.0
        scales random value to be added in each call to `function <OrnsteinUhlenbeckIntegrator.function>`. (see
        `noise <OrnsteinUhlenbeckIntegrator.noise>` for details).

    time_step_size : float : default 0.0
        determines the timing precision of the integration process (see `time_step_size
        <OrnsteinUhlenbeckIntegrator.time_step_size>` for details.

    t0 : float : default 0.0
        represents the starting time of the model and is used to compute
        `previous_time <OrnsteinUhlenbeckIntegrator.previous_time>`

    initializer float, list or 1d np.array : default 0.0
        specifies starting value for integration.  If it is a list or array, it must be the same length as
        `default_variable <OrnsteinUhlenbeckIntegrator.default_variable>` (see `initializer
        <OrnsteinUhlenbeckIntegrator.initializer>` for details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : number or np.array
        represents the stimulus component of drift. The product of
        `variable <OrnsteinUhlenbeckIntegrator.variable>` and `rate <OrnsteinUhlenbeckIntegrator.rate>` is multiplied
        by `time_step_size <OrnsteinUhlenbeckIntegrator.time_step_size>` to model the accumulation of evidence during
        one step.

    rate : float or 1d np.array
        represents the attentional component of drift. The product of `rate <OrnsteinUhlenbeckIntegrator.rate>` and
        `variable <OrnsteinUhlenbeckIntegrator.variable>` is multiplied by
        `time_step_size <OrnsteinUhlenbeckIntegrator.time_step_size>` to model the accumulation of evidence during
        one step.

    noise : float, function, list, or 1d np.array
        scales the random value to be added in each call to `function <OrnsteinUhlenbeckIntegrator.function>`

        Noise must be specified as a float (or list or array of floats) because this
        value will be used to construct the standard DDM probability distribution.

    time_step_size : float
        determines the timing precision of the integration process and is used to scale the `noise
        <OrnsteinUhlenbeckIntegrator.noise>` parameter appropriately.

    initializer : float, 1d np.array or list
        determines the starting value for integration (i.e., the value to which
        `previous_value <OrnsteinUhlenbeckIntegrator.previous_value>` is originally set.)

        If initializer is a list or array, it must be the same length as `variable
        <OrnsteinUhlenbeckIntegrator.default_variable>`.

    previous_value : 1d np.array : default ClassDefaults.variable
        stores previous value with which `variable <OrnsteinUhlenbeckIntegrator.variable>` is integrated.

    previous_time : float
        stores previous time at which the function was executed and accumulates with each execution according to
        `time_step_size <OrnsteinUhlenbeckIntegrator.default_time_step_size>`.

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = ORNSTEIN_UHLENBECK_INTEGRATOR_FUNCTION

    multiplicative_param = RATE
    additive_param = OFFSET

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    # paramClassDefaults.update({INITIALIZER: ClassDefaults.variable})
    paramClassDefaults.update({
        NOISE: None,
        RATE: None
    })

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate: parameter_spec = 1.0,
                 noise=0.0,
                 offset: parameter_spec = 0.0,
                 time_step_size=1.0,
                 t0=0.0,
                 decay=1.0,
                 initializer=None,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  time_step_size=time_step_size,
                                                  decay = decay,
                                                  initializer=initializer,
                                                  t0=t0,
                                                  noise=noise,
                                                  offset=offset,
                                                  params=params)

        # Assign here as default, for use in initialization of function
        self.previous_value = initializer

        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)

        self.previous_time = self.t0
        self.auto_dependent = True

    def _validate_noise(self, noise):
        if not isinstance(noise, float):
            raise FunctionError(
                "Invalid noise parameter for {}. OrnsteinUhlenbeckIntegrator requires noise parameter to be a float. "
                "Noise parameter is used to construct the standard DDM noise distribution".format(self.name))

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """
        Return: One time step of evidence accumulation according to the Ornstein Uhlenbeck Model

        previous_value + decay * (previous_value -  rate * variable) + :math:`\\sqrt{time_step_size * noise}` * random
        sample from Normal distribution


        Arguments
        ---------

        variable : number, list or np.array : default ClassDefaults.variable
           the stimulus component of drift rate in the Drift Diffusion Model.


        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        updated value of integral : 2d np.array

        """

        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))
        rate = np.array(self.get_current_function_param(RATE)).astype(float)
        offset = self.get_current_function_param(OFFSET)
        time_step_size = self.get_current_function_param(TIME_STEP_SIZE)
        decay = self.get_current_function_param(DECAY)
        noise = self.get_current_function_param(NOISE)

        previous_value = np.atleast_2d(self.previous_value)

        # dx = (lambda*x + A)dt + c*dW
        value = previous_value + (decay * previous_value - rate * variable) * time_step_size + np.sqrt(
            time_step_size * noise) * np.random.normal()

        # If this NOT an initialization run, update the old value and time
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        adjusted_value = value + offset

        if self.context.initialization_status != ContextFlags.INITIALIZING:
            self.previous_value = adjusted_value
            self.previous_time += time_step_size

        return adjusted_value

    def reinitialize(self, new_previous_value=None, new_previous_time=None):
        """
        In effect, begins accumulation over again at the original starting point and time, or new ones.

        Sets

        - `previous_value <OrnsteinUhlenbeckIntegrator.previous_value>`
        - `initializer <OrnsteinUhlenbeckIntegrator.initializer>`
        - `value <OrnsteinUhlenbeckIntegrator.value>`

        to the value specified in the first argument.

        Sets `previous_time <OrnsteinUhlenbeckIntegrator.previous_time>` to the value specified in the second argument.

        If no arguments are specified, then the instance defaults for `initializer
        <OrnsteinUhlenbeckIntegrator.initializer>` and `t0 <OrnsteinUhlenbeckIntegrator.t0>` are used.
        """
        if new_previous_value is None:
            new_previous_value = self.instance_defaults.initializer
        if new_previous_time is None:
            new_previous_time = self.instance_defaults.t0
        self._initializer = new_previous_value
        self.value = new_previous_value
        self.previous_value = new_previous_value
        self.previous_time = new_previous_time
        return self.value

class FHNIntegrator(Integrator):  # --------------------------------------------------------------------------------
    """
    FHNIntegrator(                      \
        default_variable=1.0,           \
        scale: parameter_spec = 1.0,    \
        offset: parameter_spec = 0.0,   \
        initial_w=0.0,                  \
        initial_v=0.0,                  \
        time_step_size=0.05,          \
        t_0=0.0,                        \
        a_v=-1/3,                       \
        b_v=0.0,                        \
        c_v=1.0,                        \
        d_v=0.0,                        \
        e_v=-1.0,                       \
        f_v=1.0,                        \
        threshold=-1.0                  \
        time_constant_v=1.0,            \
        a_w=1.0,                        \
        b_w=-0.8,                       \
        c_w=0.7,                        \
        mode=1.0,                       \
        uncorrelated_activity=0.0       \
        time_constant_w = 12.5,         \
        integration_method="RK4"        \
        params=None,                    \
        owner=None,                     \
        prefs=None,                     \
        )

    .. _FHNIntegrator:

    The FHN Integrator function in PsyNeuLink implements the Fitzhugh-Nagumo model using a choice of Euler or 4th Order
    Runge-Kutta numerical integration.

    In order to support several common representations of the model, the FHNIntegrator includes many parameters, some of
    which would not be sensible to use in combination. The equations of the Fitzhugh-Nagumo model are expressed below in
    terms of all of the parameters exposed in PsyNeuLink:

    **Fast, Excitatory Variable:**


    .. math::

        \\frac{dv}{dt} = \\frac{a_v v^{3} + b_v v^{2} (1+threshold) - c_v v\\, threshold + d_v + e_v\\, previous_w + f_v\\, variable)}{time\\, constant_v}


    **Slow, Inactivating Variable:**


    .. math::

        \\frac{dw}{dt} = \\frac{a_w\\, mode\\, previous_v + b_w w + c_w +
                    uncorrelated\\,activity\\,(1-mode)}{time\\, constant_w}

    *The three formulations that the FHNIntegrator was designed to allow are:*

    (1) **Fitzhugh-Nagumo Model**

        **Fast, Excitatory Variable:**

        .. math::

            \\frac{dv}{dt} = v - \\frac{v^3}{3} - w + I_{ext}

        **Slow, Inactivating Variable:**

        .. math::

            \\frac{dw}{dt} = \\frac{v + a - bw}{T}

        :math:`\\frac{dw}{dt}` often has the following parameter values:

        .. math::

            \\frac{dw}{dt} = 0.08\\,(v + 0.7 - 0.8 w)

        *How to implement this model in PsyNeuLink:*

            In PsyNeuLink, the default parameter values of the FHNIntegrator function implement the above equations.


    (2) **Modified FHN Model**

        **Fast, Excitatory Variable:**

        .. math::

            \\frac{dv}{dt} = v(a-v)(v-1) - w + I_{ext}

        **Slow, Inactivating Variable:**

        .. math::

            \\frac{dw}{dt} = bv - cw

        `Mahbub Khan (2013) <http://pcwww.liv.ac.uk/~bnvasiev/Past%20students/Mahbub_549.pdf>`_ provides a nice summary
        of why this formulation is useful.

        *How to implement this model in PsyNeuLink:*

            In order to implement the modified FHN model, the following PsyNeuLink parameter values must be set in the
            equation for :math:`\\frac{dv}{dt}`:

            +-----------------+-----+-----+-----+-----+-----+-----+---------------+
            |**PNL Parameter**| a_v | b_v | c_v | d_v | e_v | f_v |time_constant_v|
            +-----------------+-----+-----+-----+-----+-----+-----+---------------+
            |**Value**        |-1.0 |1.0  |1.0  |0.0  |-1.0 |1.0  |1.0            |
            +-----------------+-----+-----+-----+-----+-----+-----+---------------+

            When the parameters above are set to the listed values, the PsyNeuLink equation for :math:`\\frac{dv}{dt}`
            reduces to the Modified FHN formulation, and the remaining parameters in the :math:`\\frac{dv}{dt}` equation
            correspond as follows:

            +--------------------------+---------------------------------------+---------------------------------------+
            |**PNL Parameter**         |`threshold <FHNIntegrator.threshold>`  |`variable <FHNIntegrator.variable>`    |
            +--------------------------+---------------------------------------+---------------------------------------+
            |**Modified FHN Parameter**|a                                      |:math:`I_{ext}`                        |
            +--------------------------+---------------------------------------+---------------------------------------+

            In order to implement the modified FHN model, the following PsyNeuLink parameter values must be set in the
            equation for :math:`\\frac{dw}{dt}`:

            +-----------------+-----+------+---------------+----------------------+
            |**PNL Parameter**|c_w  | mode |time_constant_w|uncorrelated_activity |
            +-----------------+-----+------+---------------+----------------------+
            |**Value**        | 0.0 | 1.0  |1.0            | 0.0                  |
            +-----------------+-----+------+---------------+----------------------+

            When the parameters above are set to the listed values, the PsyNeuLink equation for :math:`\\frac{dw}{dt}`
            reduces to the Modified FHN formulation, and the remaining parameters in the :math:`\\frac{dw}{dt}` equation
            correspond as follows:

            +--------------------------+---------------------------------------+---------------------------------------+
            |**PNL Parameter**         |`a_w <FHNIntegrator.a_w>`              |*NEGATIVE* `b_w <FHNIntegrator.b_w>`   |
            +--------------------------+---------------------------------------+---------------------------------------+
            |**Modified FHN Parameter**|b                                      |c                                      |
            +--------------------------+---------------------------------------+---------------------------------------+

    (3) **Modified FHN Model as implemented in** `Gilzenrat (2002) <http://www.sciencedirect.com/science/article/pii/S0893608002000552?via%3Dihub>`_

        **Fast, Excitatory Variable:**

        [Eq. (6) in `Gilzenrat (2002) <http://www.sciencedirect.com/science/article/pii/S0893608002000552?via%3Dihub>`_ ]

        .. math::

            \\tau_v \\frac{dv}{dt} = v(a-v)(v-1) - u + w_{vX_1}\\, f(X_1)

        **Slow, Inactivating Variable:**

        [Eq. (7) & Eq. (8) in `Gilzenrat (2002) <http://www.sciencedirect.com/science/article/pii/S0893608002000552?via%3Dihub>`_ ]

        .. math::

            \\tau_u \\frac{du}{dt} = Cv + (1-C)\\, d - u

        *How to implement this model in PsyNeuLink:*

            In order to implement the Gilzenrat 2002 model, the following PsyNeuLink parameter values must be set in the
            equation for :math:`\\frac{dv}{dt}`:

            +-----------------+-----+-----+-----+-----+-----+
            |**PNL Parameter**| a_v | b_v | c_v | d_v | e_v |
            +-----------------+-----+-----+-----+-----+-----+
            |**Value**        |-1.0 |1.0  |1.0  |0.0  |-1.0 |
            +-----------------+-----+-----+-----+-----+-----+

            When the parameters above are set to the listed values, the PsyNeuLink equation for :math:`\\frac{dv}{dt}`
            reduces to the Gilzenrat formulation, and the remaining parameters in the :math:`\\frac{dv}{dt}` equation
            correspond as follows:

            +-----------------------+-------------------------------------+-----------------------------------+-------------------------+----------------------------------------------------+
            |**PNL Parameter**      |`threshold <FHNIntegrator.threshold>`|`variable <FHNIntegrator.variable>`|`f_v <FHNIntegrator.f_v>`|`time_constant_v <FHNIntegrator.time_constant_v>`   |
            +-----------------------+-------------------------------------+-----------------------------------+-------------------------+----------------------------------------------------+
            |**Gilzenrat Parameter**|a                                    |:math:`f(X_1)`                     |:math:`w_{vX_1}`         |:math:`T_{v}`                                       |
            +-----------------------+-------------------------------------+-----------------------------------+-------------------------+----------------------------------------------------+

            In order to implement the Gilzenrat 2002 model, the following PsyNeuLink parameter values must be set in the
            equation for :math:`\\frac{dw}{dt}`:

            +-----------------+-----+-----+-----+
            |**PNL Parameter**| a_w | b_w | c_w |
            +-----------------+-----+-----+-----+
            |**Value**        | 1.0 |-1.0 |0.0  |
            +-----------------+-----+-----+-----+

            When the parameters above are set to the listed values, the PsyNeuLink equation for :math:`\\frac{dw}{dt}`
            reduces to the Gilzenrat formulation, and the remaining parameters in the :math:`\\frac{dw}{dt}` equation
            correspond as follows:

            +--------------------------+---------------------------------------+-------------------------------------------------------------+----------------------------------------------------+
            |**PNL Parameter**         |`mode <FHNIntegrator.mode>`            |`uncorrelated_activity <FHNIntegrator.uncorrelated_activity>`|`time_constant_v <FHNIntegrator.time_constant_w>`   |
            +--------------------------+---------------------------------------+-------------------------------------------------------------+----------------------------------------------------+
            |**Gilzenrat Parameter**   |C                                      |d                                                            |:math:`T_{u}`                                       |
            +--------------------------+---------------------------------------+-------------------------------------------------------------+----------------------------------------------------+

    Arguments
    ---------

    default_variable : number, list or np.array : default ClassDefaults.variable
        specifies a template for the external stimulus

    initial_w : float, list or 1d np.array : default 0.0
        specifies starting value for integration of dw/dt.  If it is a list or array, it must be the same length as
        `default_variable <FHNIntegrator.default_variable>`

    initial_v : float, list or 1d np.array : default 0.0
        specifies starting value for integration of dv/dt.  If it is a list or array, it must be the same length as
        `default_variable <FHNIntegrator.default_variable>`

    time_step_size : float : default 0.1
        specifies the time step size of numerical integration

    t_0 : float : default 0.0
        specifies starting value for time

    a_v : float : default -1/3
        coefficient on the v^3 term of the dv/dt equation

    b_v : float : default 0.0
        coefficient on the v^2 term of the dv/dt equation

    c_v : float : default 1.0
        coefficient on the v term of the dv/dt equation

    d_v : float : default 0.0
        constant term in the dv/dt equation

    e_v : float : default -1.0
        coefficient on the w term in the dv/dt equation

    f_v : float : default  1.0
        coefficient on the external stimulus (`variable <FHNIntegrator.variable>`) term in the dv/dt equation

    time_constant_v : float : default 1.0
        scaling factor on the dv/dt equation

    a_w : float : default 1.0,
        coefficient on the v term of the dw/dt equation

    b_w : float : default -0.8,
        coefficient on the w term of the dv/dt equation

    c_w : float : default 0.7,
        constant term in the dw/dt equation

    threshold : float : default -1.0
        specifies a value of the input below which the LC will tend not to respond and above which it will

    mode : float : default 1.0
        coefficient which simulates electrotonic coupling by scaling the values of dw/dt such that the v term
        (representing the input from the LC) increases when the uncorrelated_activity term (representing baseline
        activity) decreases

    uncorrelated_activity : float : default 0.0
        constant term in the dw/dt equation

    time_constant_w : float : default 12.5
        scaling factor on the dv/dt equation

    integration_method: str : default "RK4"
        selects the numerical integration method. Currently, the choices are: "RK4" (4th Order Runge-Kutta) or "EULER"
        (Forward Euler)

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : number or np.array
        External stimulus

    previous_v : 1d np.array : default ClassDefaults.variable
        stores accumulated value of v during integration

    previous_w : 1d np.array : default ClassDefaults.variable
        stores accumulated value of w during integration

    previous_t : float
        stores accumulated value of time, which is incremented by time_step_size on each execution of the function

    owner : Component
        `component <Component>` to which the Function has been assigned.

    initial_w : float, list or 1d np.array : default 0.0
        specifies starting value for integration of dw/dt.  If it is a list or array, it must be the same length as
        `default_variable <FHNIntegrator.default_variable>`

    initial_v : float, list or 1d np.array : default 0.0
        specifies starting value for integration of dv/dt.  If it is a list or array, it must be the same length as
        `default_variable <FHNIntegrator.default_variable>`

    time_step_size : float : default 0.1
        specifies the time step size of numerical integration

    t_0 : float : default 0.0
        specifies starting value for time

    a_v : float : default -1/3
        coefficient on the v^3 term of the dv/dt equation

    b_v : float : default 0.0
        coefficient on the v^2 term of the dv/dt equation

    c_v : float : default 1.0
        coefficient on the v term of the dv/dt equation

    d_v : float : default 0.0
        constant term in the dv/dt equation

    e_v : float : default -1.0
        coefficient on the w term in the dv/dt equation

    f_v : float : default  1.0
        coefficient on the external stimulus (`variable <FHNIntegrator.variable>`) term in the dv/dt equation

    time_constant_v : float : default 1.0
        scaling factor on the dv/dt equation

    a_w : float : default 1.0
        coefficient on the v term of the dw/dt equation

    b_w : float : default -0.8
        coefficient on the w term of the dv/dt equation

    c_w : float : default 0.7
        constant term in the dw/dt equation

    threshold : float : default -1.0
        coefficient that scales both the v^2 [ (1+threshold)*v^2 ] and v [ (-threshold)*v ] terms in the dv/dt equation
        under a specific formulation of the FHN equations, the threshold parameter behaves as a "threshold of
        excitation", and has the following relationship with variable (the external stimulus):

            - when the external stimulus is below the threshold of excitation, the system is either in a stable state,
              or will emit a single excitation spike, then reach a stable state. The behavior varies depending on the
              magnitude of the difference between the threshold and the stimulus.

            - when the external stimulus is equal to or above the threshold of excitation, the system is
              unstable, and will emit many excitation spikes

            - when the external stimulus is too far above the threshold of excitation, the system will emit some
              excitation spikes before reaching a stable state.

    mode : float : default 1.0
        coefficient which simulates electrotonic coupling by scaling the values of dw/dt such that the v term
        (representing the input from the LC) increases when the uncorrelated_activity term (representing baseline
        activity) decreases

    uncorrelated_activity : float : default 0.0
        constant term in the dw/dt equation

    time_constant_w : float : default 12.5
        scaling factor on the dv/dt equation

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).
    """

    MODE = 'mode'

    componentName = FHN_INTEGRATOR_FUNCTION

    class ClassDefaults(Integrator.ClassDefaults):
        variable = np.array([1.0])
        initializer = np.array([1.0])

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({INITIALIZER: ClassDefaults.variable})
    paramClassDefaults.update({
        NOISE: None,
        RATE: None,
        INCREMENT: None,
    })


    multiplicative_param = SCALE
    additive_param = OFFSET

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 offset=0.0,
                 scale=1.0,
                 initial_w=0.0,
                 initial_v=0.0,
                 time_step_size=0.05,
                 t_0=0.0,
                 a_v=-1/3,
                 b_v=0.0,
                 c_v=1.0,
                 d_v=0.0,
                 e_v=-1.0,
                 f_v=1.0,
                 time_constant_v=1.0,
                 a_w=1.0,
                 b_w=-0.8,
                 c_w=0.7,
                 threshold=-1.0,
                 time_constant_w=12.5,
                 mode=1.0,
                 uncorrelated_activity=0.0,
                 integration_method="RK4",
                 params: tc.optional(dict)=None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(default_variable=default_variable,
                                                  offset=offset,
                                                  scale=scale,
                                                  initial_v=initial_v,
                                                  initial_w=initial_w,
                                                  time_step_size=time_step_size,
                                                  t_0=t_0,
                                                  a_v=a_v,
                                                  b_v=b_v,
                                                  c_v=c_v,
                                                  d_v=d_v,
                                                  e_v=e_v,
                                                  f_v=f_v,
                                                  time_constant_v=time_constant_v,
                                                  a_w=a_w,
                                                  b_w=b_w,
                                                  c_w=c_w,
                                                  threshold=threshold,
                                                  mode=mode,
                                                  uncorrelated_activity=uncorrelated_activity,
                                                  integration_method=integration_method,
                                                  time_constant_w=time_constant_w,
                                                  params=params,
                                                  )

        self.previous_v = self.initial_v
        self.previous_w = self.initial_w
        self.previous_time = self.t_0

        super().__init__(
            default_variable=default_variable,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)

        self.auto_dependent = True

    def _validate_params(self, request_set, target_set=None, context=None):
        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)
        if self.integration_method not in {"RK4", "EULER"}:
            raise FunctionError("Invalid integration method ({}) selected for {}. Choose 'RK4' or 'EULER'".
                                format(self.integration_method, self.name))

    def _euler_FHN(self, variable, previous_value_v, previous_value_w, previous_time, slope_v, slope_w, time_step_size, a_v,
                   threshold, b_v, c_v, d_v, e_v, f_v, time_constant_v, mode, a_w, b_w, c_w, uncorrelated_activity,
                   time_constant_w):

        slope_v_approx = slope_v(variable,
                                 previous_time,
                                 previous_value_v,
                                 previous_value_w,
                                 a_v,
                                 threshold,
                                 b_v,
                                 c_v,
                                 d_v,
                                 e_v,
                                 f_v,
                                 time_constant_v)

        slope_w_approx = slope_w(variable,
                                 previous_time,
                                 previous_value_w,
                                 previous_value_v,
                                 mode,
                                 a_w,
                                 b_w,
                                 c_w,
                                 uncorrelated_activity,
                                 time_constant_w)

        new_v = previous_value_v + time_step_size*slope_v_approx
        new_w = previous_value_w + time_step_size*slope_w_approx

        return new_v, new_w

    def _runge_kutta_4_FHN(self, variable, previous_value_v, previous_value_w, previous_time, slope_v, slope_w, time_step_size,
                           a_v, threshold, b_v, c_v, d_v, e_v, f_v, time_constant_v, mode, a_w, b_w, c_w,
                           uncorrelated_activity, time_constant_w):

        # First approximation
        # v is approximately previous_value_v
        # w is approximately previous_value_w

        slope_v_approx_1 = slope_v(variable,
                                   previous_time,
                                   previous_value_v,
                                   previous_value_w,
                                   a_v,
                                   threshold,
                                   b_v,
                                   c_v,
                                   d_v,
                                   e_v,
                                   f_v,
                                   time_constant_v)

        slope_w_approx_1 = slope_w(variable,
                                   previous_time,
                                   previous_value_w,
                                   previous_value_v,
                                   mode,
                                   a_w,
                                   b_w,
                                   c_w,
                                   uncorrelated_activity,
                                   time_constant_w)
        # Second approximation
        # v is approximately previous_value_v + 0.5 * time_step_size * slope_w_approx_1
        # w is approximately previous_value_w + 0.5 * time_step_size * slope_w_approx_1

        slope_v_approx_2 = slope_v(variable,
                                   previous_time + time_step_size/2,
                                   previous_value_v + (0.5 * time_step_size * slope_v_approx_1),
                                   previous_value_w + (0.5 * time_step_size * slope_w_approx_1),
                                   a_v,
                                   threshold,
                                   b_v,
                                   c_v,
                                   d_v,
                                   e_v,
                                   f_v,
                                   time_constant_v)

        slope_w_approx_2 = slope_w(variable,
                                   previous_time + time_step_size/2,
                                   previous_value_w + (0.5 * time_step_size * slope_w_approx_1),
                                   previous_value_v + (0.5 * time_step_size * slope_v_approx_1),
                                   mode,
                                   a_w,
                                   b_w,
                                   c_w,
                                   uncorrelated_activity,
                                   time_constant_w)

        # Third approximation
        # v is approximately previous_value_v + 0.5 * time_step_size * slope_v_approx_2
        # w is approximately previous_value_w + 0.5 * time_step_size * slope_w_approx_2

        slope_v_approx_3 = slope_v(variable,
                                   previous_time + time_step_size/2,
                                   previous_value_v + (0.5 * time_step_size * slope_v_approx_2),
                                   previous_value_w + (0.5 * time_step_size * slope_w_approx_2),
                                   a_v,
                                   threshold,
                                   b_v,
                                   c_v,
                                   d_v,
                                   e_v,
                                   f_v,
                                   time_constant_v)

        slope_w_approx_3 = slope_w(variable,
                                   previous_time + time_step_size/2,
                                   previous_value_w + (0.5 * time_step_size * slope_w_approx_2),
                                   previous_value_v + (0.5 * time_step_size * slope_v_approx_2),
                                   mode,
                                   a_w,
                                   b_w,
                                   c_w,
                                   uncorrelated_activity,
                                   time_constant_w)

        # Fourth approximation
        # v is approximately previous_value_v + time_step_size * slope_v_approx_3
        # w is approximately previous_value_w + time_step_size * slope_w_approx_3

        slope_v_approx_4 = slope_v(variable,
                                   previous_time + time_step_size,
                                   previous_value_v + (time_step_size * slope_v_approx_3),
                                   previous_value_w + (time_step_size * slope_v_approx_3),
                                   a_v,
                                   threshold,
                                   b_v,
                                   c_v,
                                   d_v,
                                   e_v,
                                   f_v,
                                   time_constant_v)

        slope_w_approx_4 = slope_w(variable,
                                   previous_time + time_step_size,
                                   previous_value_w + (time_step_size * slope_v_approx_3),
                                   previous_value_v + (time_step_size * slope_v_approx_3),
                                   mode,
                                   a_w,
                                   b_w,
                                   c_w,
                                   uncorrelated_activity,
                                   time_constant_w)

        new_v = previous_value_v \
                + (time_step_size/6)*(slope_v_approx_1 + 2*(slope_v_approx_2 + slope_v_approx_3) + slope_v_approx_4)
        new_w = previous_value_w \
                + (time_step_size/6)*(slope_w_approx_1 + 2*(slope_w_approx_2 + slope_w_approx_3) + slope_w_approx_4)

        return new_v, new_w

    def dv_dt(self, variable, time, v, w, a_v, threshold, b_v, c_v, d_v, e_v, f_v, time_constant_v):

        val= (a_v*(v**3) + (1+threshold)*b_v*(v**2) + (-threshold)*c_v*v + d_v
                + e_v*self.previous_w + f_v*variable)/time_constant_v

        # Standard coefficients - hardcoded for testing
        # val = v - (v**3)/3 - w + variable

        # Gilzenrat paper - hardcoded for testing
        # val = (v*(v-0.5)*(1-v) - w + variable)/0.01

        return val

    def dw_dt(self, variable, time, w, v, mode, a_w, b_w, c_w, uncorrelated_activity, time_constant_w):
        val = (mode*a_w*self.previous_v + b_w*w + c_w +
                (1-mode)*uncorrelated_activity)/time_constant_w

        # Standard coefficients - hardcoded for testing
        # val = (v + 0.7 - 0.8*w)/12.5

        #Gilzenrat paper - hardcoded for testing

        # val = (v - 0.5*w)

        return val

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """
        Return: current v, current w

        The model is defined by the following system of differential equations:

            *time_constant_v* :math:`* \\frac{dv}{dt} =`

                *a_v* :math:`* v^3 + (1 + threshold) *` *b_v* :math:`* v^2 + (- threshold) *` *c_v*
                :math:`* v^2 +` *d_v* :math:`+` *e_v* :math:`* w +` *f_v* :math:`* I_{ext}`

            *time_constant_w* :math:`* dw/dt =`

                :math:`mode *` *a_w* :math:`* v +` *b_w* :math:`* w +` *c_w*
                :math:`+ (1 - self.mode) *` *self.uncorrelated_activity*


        Arguments
        ---------

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        current value of v , current value of w : float, list, or np.array

        """

        a_v = self.get_current_function_param("a_v")
        b_v = self.get_current_function_param("b_v")
        c_v = self.get_current_function_param("c_v")
        d_v = self.get_current_function_param("d_v")
        e_v = self.get_current_function_param("e_v")
        f_v = self.get_current_function_param("f_v")
        time_constant_v = self.get_current_function_param("time_constant_v")
        threshold = self.get_current_function_param("threshold")
        a_w = self.get_current_function_param("a_w")
        b_w = self.get_current_function_param("b_w")
        c_w = self.get_current_function_param("c_w")
        uncorrelated_activity = self.get_current_function_param("uncorrelated_activity")
        time_constant_w = self.get_current_function_param("time_constant_w")
        mode = self.get_current_function_param("mode")
        integration_method = self.get_current_function_param("integration_method")
        time_step_size = self.get_current_function_param(TIME_STEP_SIZE)

        if integration_method == "RK4":
            approximate_values = self._runge_kutta_4_FHN(variable,
                                                         self.previous_v,
                                                         self.previous_w,
                                                         self.previous_time,
                                                         self.dv_dt,
                                                         self.dw_dt,
                                                         time_step_size,
                                                         a_v,
                                                         threshold,
                                                         b_v,
                                                         c_v,
                                                         d_v,
                                                         e_v,
                                                         f_v,
                                                         time_constant_v,
                                                         mode,
                                                         a_w,
                                                         b_w,
                                                         c_w,
                                                         uncorrelated_activity,
                                                         time_constant_w)


        elif integration_method == "EULER":
            approximate_values = self._euler_FHN(variable,
                                                 self.previous_v,
                                                 self.previous_w,
                                                 self.previous_time,
                                                 self.dv_dt,
                                                 self.dw_dt,
                                                 time_step_size,
                                                 a_v,
                                                 threshold,
                                                 b_v,
                                                 c_v,
                                                 d_v,
                                                 e_v,
                                                 f_v,
                                                 time_constant_v,
                                                 mode,
                                                 a_w,
                                                 b_w,
                                                 c_w,
                                                 uncorrelated_activity,
                                                 time_constant_w)
        else:
            raise FunctionError("Invalid integration method ({}) selected for {}".
                                format(integration_method, self.name))

        if self.context.initialization_status != ContextFlags.INITIALIZING:
            self.previous_v = approximate_values[0]
            self.previous_w = approximate_values[1]
            self.previous_time += time_step_size

        return self.previous_v, self.previous_w, self.previous_time

    def reinitialize(self, new_previous_v=None, new_previous_w=None, new_previous_time=None):
        """
        Effectively begins accumulation over again at the specified v, w, and time.

        Sets

        - `previous_v <DriftDiffusionIntegrator.previous_v>`
        - `initial_v <DriftDiffusionIntegrator.initial_v>`

        to the quantity specified in the first argument.

        Sets

        - `previous_w <DriftDiffusionIntegrator.previous_w>`
        - `initial_w <DriftDiffusionIntegrator.initial_w>`

        to the quantity specified in the second argument.

        Sets `previous_time <DriftDiffusionIntegrator.previous_time>` to the quantity specified in the third argument.

        If no arguments are specified, then the instance defaults for `initial_v <FHNIntegrator.initial_v>`, `initial_w
        <FHNIntegrator.initial_w>` and `t_0 <FHNIntegrator.t_0>` are used.
        """
        if new_previous_v is None:
            new_previous_v = self.instance_defaults.initial_v
        if new_previous_w is None:
            new_previous_w = self.instance_defaults.initial_w
        if new_previous_time is None:
            new_previous_time = self.instance_defaults.t_0
        self._initial_v = new_previous_v
        self.previous_v = new_previous_v
        self._initial_w = new_previous_w
        self.previous_w = new_previous_w
        self.previous_time = new_previous_time
        self.value = new_previous_v, new_previous_w, new_previous_time
        return [new_previous_v], [new_previous_w], [new_previous_time]

class AccumulatorIntegrator(Integrator):  # --------------------------------------------------------------------------------
    """
    AccumulatorIntegrator(              \
        default_variable=None,          \
        rate=1.0,                       \
        noise=0.0,                      \
        scale: parameter_spec = 1.0,    \
        offset: parameter_spec = 0.0,   \
        initializer,                    \
        params=None,                    \
        owner=None,                     \
        prefs=None,                     \
        )

    .. _AccumulatorIntegrator:

    Integrates prior value by multiplying `previous_value <AccumulatorIntegrator.previous_value>` by `rate
    <Integrator.rate>` and adding `increment <AccumulatorIntegrator.increment>` and  `noise
    <AccumulatorIntegrator.noise>`. Ignores `variable <Integrator.variable>`).

    Arguments
    ---------

    default_variable : number, list or np.array : default ClassDefaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float, list or 1d np.array : default 1.0
        specifies the multiplicative decrement of `previous_value <AccumulatorIntegrator.previous_value>` (i.e.,
        the rate of exponential decay).  If it is a list or array, it must be the same length as
        `variable <AccumulatorIntegrator.default_variable>`.

    increment : float, list or 1d np.array : default 0.0
        specifies an amount to be added to `previous_value <AccumulatorIntegrator.previous_value>` in each call to
        `function <AccumulatorIntegrator.function>` (see `increment <AccumulatorIntegrator.increment>` for details).
        If it is a list or array, it must be the same length as `variable <AccumulatorIntegrator.default_variable>`
        (see `increment <AccumulatorIntegrator.increment>` for details).

    noise : float, PsyNeuLink Function, list or 1d np.array : default 0.0
        specifies random value to be added to `prevous_value <AccumulatorIntegrator.previous_value>` in each call to
        `function <AccumulatorIntegrator.function>`. If it is a list or array, it must be the same length as
        `variable <AccumulatorIntegrator.default_variable>` (see `noise <AccumulatorIntegrator.noise>` for details).

    initializer float, list or 1d np.array : default 0.0
        specifies starting value for integration.  If it is a list or array, it must be the same length as
        `default_variable <AccumulatorIntegrator.default_variable>` (see `initializer
        <AccumulatorIntegrator.initializer>` for details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : number or np.array
        **Ignored** by the AccumulatorIntegrator function. Refer to LCAIntegrator or AdaptiveIntegrator for
        integrator functions that depend on both a prior value and a new value (variable).

    rate : float or 1d np.array
        determines the multiplicative decrement of `previous_value <AccumulatorIntegrator.previous_value>` (i.e., the
        rate of exponential decay) in each call to `function <AccumulatorIntegrator.function>`.  If it is a list or
        array, it must be the same length as `variable <AccumulatorIntegrator.default_variable>` and each element is
        used to multiply the corresponding element of `previous_value <AccumulatorIntegrator.previous_value>` (i.e.,
        it is used for Hadamard multiplication).  If it is a scalar or has a single element, its value is used to
        multiply all the elements of `previous_value <AccumulatorIntegrator.previous_value>`.

    increment : float, function, list, or 1d np.array
        determines the amount added to `previous_value <AccumulatorIntegrator.previous_value>` in each call to
        `function <AccumulatorIntegrator.function>`.  If it is a list or array, it must be the same length as
        `variable <AccumulatorIntegrator.default_variable>` and each element is added to the corresponding element of
        `previous_value <AccumulatorIntegrator.previous_value>` (i.e., it is used for Hadamard addition).  If it is a
        scalar or has a single element, its value is added to all the elements of `previous_value
        <AccumulatorIntegrator.previous_value>`.

    noise : float, function, list, or 1d np.array
        determines a random value to be added in each call to `function <AccumulatorIntegrator.function>`.
        If it is a list or array, it must be the same length as `variable <AccumulatorIntegrator.default_variable>` and
        each element is added to the corresponding element of `previous_value <AccumulatorIntegrator.previous_value>`
        (i.e., it is used for Hadamard addition).  If it is a scalar or has a single element, its value is added to all
        the elements of `previous_value <AccumulatorIntegrator.previous_value>`.  If it is a function, it will be
        executed separately and added to each element.

        .. note::

            In order to generate random noise, a probability distribution function should be selected (see
            `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value from
            its distribution on each execution. If noise is specified as a float or as a function with a fixed output,
            then the noise will simply be an offset that remains the same across all executions.

    initializer : float, 1d np.array or list
        determines the starting value for integration (i.e., the value to which `previous_value
        <AccumulatorIntegrator.previous_value>` is set. If initializer is a list or array, it must be the same length
        as `variable <AccumulatorIntegrator.default_variable>`.

    previous_value : 1d np.array : default ClassDefaults.variable
        stores previous value to which `rate <AccumulatorIntegrator.rate>` and `noise <AccumulatorIntegrator.noise>`
        will be added.

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = ACCUMULATOR_INTEGRATOR_FUNCTION

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    # paramClassDefaults.update({INITIALIZER: ClassDefaults.variable})
    paramClassDefaults.update({
        NOISE: None,
        RATE: None,
        INCREMENT: None,
    })

    # multiplicative param does not make sense in this case
    multiplicative_param = RATE
    additive_param = INCREMENT

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 # rate: parameter_spec = 1.0,
                 rate=None,
                 noise=0.0,
                 increment=None,
                 initializer=None,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  initializer=initializer,
                                                  noise=noise,
                                                  increment=increment,
                                                  params=params)

        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)


        self.auto_dependent = True

    def _accumulator_check_args(self, variable=None, params=None, target_set=None, context=None):
        """validate params and assign any runtime params.

        Called by AccumulatorIntegrator to validate params
        Validation can be suppressed by turning parameter_validation attribute off
        target_set is a params dictionary to which params should be assigned;
           otherwise, they are assigned to paramsCurrent;

        Does the following:
        - assign runtime params to paramsCurrent
        - validate params if PARAM_VALIDATION is set

        :param params: (dict) - params to validate
        :target_set: (dict) - set to which params should be assigned (default: self.paramsCurrent)
        :return:
        """

        # PARAMS ------------------------------------------------------------

        # If target_set is not specified, use paramsCurrent
        if target_set is None:
            target_set = self.paramsCurrent

        # # MODIFIED 11/27/16 OLD:
        # # If parameter_validation is set, the function was called with params,
        # #   and they have changed, then validate requested values and assign to target_set
        # if self.prefs.paramValidationPref and params and not params is None and not params is target_set:
        #     # self._validate_params(params, target_set, context=FUNCTION_CHECK_ARGS)
        #     self._validate_params(request_set=params, target_set=target_set, context=context)

        # If params have been passed, treat as runtime params and assign to paramsCurrent
        #   (relabel params as runtime_params for clarity)
        runtime_params = params
        if runtime_params and runtime_params is not None:
            for param_name in self.user_params:
                # Ignore input_states and output_states -- they should not be modified during run
                # IMPLEMENTATION NOTE:
                #    FUNCTION_RUNTIME_PARAM_NOT_SUPPORTED:
                #        At present, assignment of ``function`` as runtime param is not supported
                #        (this is because paramInstanceDefaults[FUNCTION] could be a class rather than an bound method;
                #        i.e., not yet instantiated;  could be rectified by assignment in _instantiate_function)
                if param_name in {FUNCTION, INPUT_STATES, OUTPUT_STATES}:
                    continue
                # If param is specified in runtime_params, then assign it
                if param_name in runtime_params:
                    self.paramsCurrent[param_name] = runtime_params[param_name]
                # Otherwise, (re-)assign to paramInstanceDefaults
                #    this insures that any params that were assigned as runtime on last execution are reset here
                #    (unless they have been assigned another runtime value)
                elif not self.runtimeParamStickyAssignmentPref:
                    if param_name is FUNCTION_PARAMS:
                        for function_param in self.function_object.user_params:
                            self.function_object.paramsCurrent[function_param] = \
                                self.function_object.paramInstanceDefaults[function_param]
                        continue
                    self.paramsCurrent[param_name] = self.paramInstanceDefaults[param_name]
            self.runtime_params_in_use = True

        # Otherwise, reset paramsCurrent to paramInstanceDefaults
        elif self.runtime_params_in_use and not self.runtimeParamStickyAssignmentPref:
            # Can't do the following since function could still be a class ref rather than abound method (see below)
            # self.paramsCurrent = self.paramInstanceDefaults
            for param_name in self.user_params:
                # IMPLEMENTATION NOTE: FUNCTION_RUNTIME_PARAM_NOT_SUPPORTED
                #    At present, assignment of ``function`` as runtime param is not supported
                #        (this is because paramInstanceDefaults[FUNCTION] could be a class rather than an bound method;
                #        i.e., not yet instantiated;  could be rectified by assignment in _instantiate_function)
                if param_name is FUNCTION:
                    continue
                if param_name is FUNCTION_PARAMS:
                    for function_param in self.function_object.user_params:
                        self.function_object.paramsCurrent[function_param] = \
                            self.function_object.paramInstanceDefaults[function_param]
                    continue
                self.paramsCurrent[param_name] = self.paramInstanceDefaults[param_name]

            self.runtime_params_in_use = False

        # If parameter_validation is set and they have changed, then validate requested values and assign to target_set
        if self.prefs.paramValidationPref and params and not params is target_set:
            try:
                self._validate_params(variable=variable, request_set=params, target_set=target_set, context=context)
            except TypeError:
                self._validate_params(request_set=params, target_set=target_set, context=context)

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """
        Return: `previous_value <ConstantIntegrator.previous_value>` combined with `rate <ConstantIntegrator.rate>` and
        `noise <ConstantIntegrator.noise>`.

        Arguments
        ---------

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        updated value of integral : 2d np.array

        """
        self._accumulator_check_args(variable, params=params, context=context)

        rate = self.get_current_function_param(RATE)
        increment = self.get_current_function_param(INCREMENT)
        noise = self._try_execute_param(self.get_current_function_param(NOISE), variable)

        if rate is None:
            rate = 1.0

        if increment is None:
            increment = 0.0

        previous_value = np.atleast_2d(self.previous_value)

        value = previous_value * rate + noise + increment

        # If this NOT an initialization run, update the old value
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if self.context.initialization_status != ContextFlags.INITIALIZING:
            self.previous_value = value
        return value


class AGTUtilityIntegrator(Integrator):  # --------------------------------------------------------------------------------
    """
    AGTUtilityIntegrator(                    \
        default_variable=None,            \
        rate=1.0,                         \
        noise=0.0,                        \
        scale: parameter_spec = 1.0,      \
        offset: parameter_spec = 0.0,     \
        initializer,                      \
        initial_short_term_utility = 0.0, \
        initial_long_term_utility = 0.0,  \
        short_term_gain = 1.0,            \
        long_term_gain =1.0,              \
        short_term_bias = 0.0,            \
        long_term_bias=0.0,               \
        short_term_rate=1.0,              \
        long_term_rate=1.0,               \
        params=None,                      \
        owner=None,                       \
        prefs=None,                       \
        )

    .. _AGTUtilityIntegrator:

    Computes an exponentially weighted moving average on the variable using two sets of parameters:

    short_term_utility =

       (1 - `short_term_rate <AGTUtilityIntegrator.short_term_rate>`) :math:`*` `previous_short_term_utility
       <AGTUtilityIntegrator.previous_short_term_utility>` + `short_term_rate <AGTUtilityIntegrator.short_term_rate>`
       :math:`*` `variable <AGTUtilityIntegrator.variable>`

    long_term_utility =

       (1 - `long_term_rate <AGTUtilityIntegrator.long_term_rate>`) :math:`*` `previous_long_term_utility
       <AGTUtilityIntegrator.previous_long_term_utility>` + `long_term_rate <AGTUtilityIntegrator.long_term_rate>`
       :math:`*` `variable <AGTUtilityIntegrator.variable>`

    then takes the logistic of each utility value, using the corresponding (short term and long term) gain and bias.

    Finally, computes a single value which combines the two values according to:

    value = [1-short_term_utility_logistic]*long_term_utility_logistic

    Arguments
    ---------

    rate : float, list or 1d np.array : default 1.0
        specifies the overall smoothing factor of the EWMA used to combine the long term and short term utility values

    noise : float, PsyNeuLink Function, list or 1d np.array : default 0.0
        TBI?

    initial_short_term_utility : float : default 0.0
        specifies starting value for integration of short_term_utility

    initial_long_term_utility : float : default 0.0
        specifies starting value for integration of long_term_utility

    short_term_gain : float : default 1.0
        specifies gain for logistic function applied to short_term_utility

    long_term_gain : float : default 1.0
        specifies gain for logistic function applied to long_term_utility

    short_term_bias : float : default 0.0
        specifies bias for logistic function applied to short_term_utility

    long_term_bias : float : default 0.0
        specifies bias for logistic function applied to long_term_utility

    short_term_rate : float : default 1.0
        specifies smoothing factor of EWMA filter applied to short_term_utility

    long_term_rate : float : default 1.0
        specifies smoothing factor of EWMA filter applied to long_term_utility

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : number or np.array
        current input value used in both the short term and long term EWMA computations

    noise : float, PsyNeuLink Function, list or 1d np.array : default 0.0
        TBI?

    initial_short_term_utility : float : default 0.0
        specifies starting value for integration of short_term_utility

    initial_long_term_utility : float : default 0.0
        specifies starting value for integration of long_term_utility

    short_term_gain : float : default 1.0
        specifies gain for logistic function applied to short_term_utility

    long_term_gain : float : default 1.0
        specifies gain for logistic function applied to long_term_utility

    short_term_bias : float : default 0.0
        specifies bias for logistic function applied to short_term_utility

    long_term_bias : float : default 0.0
        specifies bias for logistic function applied to long_term_utility

    short_term_rate : float : default 1.0
        specifies smoothing factor of EWMA filter applied to short_term_utility

    long_term_rate : float : default 1.0
        specifies smoothing factor of EWMA filter applied to long_term_utility

    previous_short_term_utility : 1d np.array
        stores previous value with which `variable <AGTUtilityIntegrator.variable>` is integrated using the EWMA filter and
        short term parameters

    previous_long_term_utility : 1d np.array
        stores previous value with which `variable <AGTUtilityIntegrator.variable>` is integrated using the EWMA filter and
        long term parameters

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = UTILITY_INTEGRATOR_FUNCTION

    multiplicative_param = RATE
    additive_param = OFFSET

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    # paramClassDefaults.update({INITIALIZER: ClassDefaults.variable})
    paramClassDefaults.update({
        NOISE: None,
        RATE: None
    })

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate: parameter_spec = 1.0,
                 noise=0.0,
                 offset=0.0,
                 initializer=None,
                 initial_short_term_utility=0.0,
                 initial_long_term_utility=0.0,
                 short_term_gain=1.0,
                 long_term_gain=1.0,
                 short_term_bias=0.0,
                 long_term_bias=0.0,
                 short_term_rate=0.9,
                 long_term_rate=0.1,
                 operation="s*l",
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  initializer=initializer,
                                                  noise=noise,
                                                  offset=offset,
                                                  initial_short_term_utility=initial_short_term_utility,
                                                  initial_long_term_utility=initial_long_term_utility,
                                                  short_term_gain=short_term_gain,
                                                  long_term_gain=long_term_gain,
                                                  short_term_bias=short_term_bias,
                                                  long_term_bias=long_term_bias,
                                                  short_term_rate=short_term_rate,
                                                  long_term_rate=long_term_rate,
                                                  operation=operation,
                                                  params=params)

        self.previous_long_term_utility = self.initial_long_term_utility
        self.previous_short_term_utility = self.initial_short_term_utility

        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)

        self.auto_dependent = True

    def _validate_params(self, request_set, target_set=None, context=None):

        # Handle list or array for rate specification
        if RATE in request_set:
            rate = request_set[RATE]
            if isinstance(rate, (list, np.ndarray)):
                if len(rate) != 1 and len(rate) != np.array(self.instance_defaults.variable).size:
                    # If the variable was not specified, then reformat it to match rate specification
                    #    and assign ClassDefaults.variable accordingly
                    # Note: this situation can arise when the rate is parametrized (e.g., as an array) in the
                    #       AGTUtilityIntegrator's constructor, where that is used as a specification for a function parameter
                    #       (e.g., for an IntegratorMechanism), whereas the input is specified as part of the
                    #       object to which the function parameter belongs (e.g., the IntegratorMechanism);
                    #       in that case, the Integrator gets instantiated using its ClassDefaults.variable ([[0]]) before
                    #       the object itself, thus does not see the array specification for the input.
                    if self._default_variable_flexibility is DefaultsFlexibility.FLEXIBLE:
                        self._instantiate_defaults(variable=np.zeros_like(np.array(rate)), context=context)
                        if self.verbosePref:
                            warnings.warn(
                                "The length ({}) of the array specified for the rate parameter ({}) of {} "
                                "must match the length ({}) of the default input ({});  "
                                "the default input has been updated to match".format(
                                    len(rate),
                                    rate,
                                    self.name,
                                    np.array(self.instance_defaults.variable).size
                                ),
                                self.instance_defaults.variable
                            )
                    else:
                        raise FunctionError(
                            "The length ({}) of the array specified for the rate parameter ({}) of {} "
                            "must match the length ({}) of the default input ({})".format(
                                len(rate),
                                rate,
                                self.name,
                                np.array(self.instance_defaults.variable).size,
                                self.instance_defaults.variable,
                            )
                        )
                        # OLD:
                        # self.paramClassDefaults[RATE] = np.zeros_like(np.array(rate))

                        # KAM changed 5/15 b/c paramClassDefaults were being updated and *requiring* future integrator functions
                        # to have a rate parameter of type ndarray/list

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if RATE in target_set:
            if isinstance(target_set[RATE], (list, np.ndarray)):
                for r in target_set[RATE]:
                    if r < 0.0 or r > 1.0:
                        raise FunctionError("The rate parameter ({}) (or all of its elements) of {} must be "
                                            "between 0.0 and 1.0 when integration_type is set to ADAPTIVE.".
                                            format(target_set[RATE], self.name))
            else:
                if target_set[RATE] < 0.0 or target_set[RATE] > 1.0:
                    raise FunctionError(
                        "The rate parameter ({}) (or all of its elements) of {} must be between 0.0 and "
                        "1.0 when integration_type is set to ADAPTIVE.".format(target_set[RATE], self.name))

        if NOISE in target_set:
            noise = target_set[NOISE]
            if isinstance(noise, DistributionFunction):
                noise.owner = self
                target_set[NOISE] = noise._execute
            self._validate_noise(target_set[NOISE])
            # if INITIALIZER in target_set:
            #     self._validate_initializer(target_set[INITIALIZER])

        if OPERATION in target_set:
            if not target_set[OPERATION] in {'s*l', 's+l', 's-l', 'l-s'}:
                raise FunctionError("\'{}\' arg for {} must be one of the following: {}".
                                    format(OPERATION, self.name, {'s*l', 's+l', 's-l', 'l-s'}))

    def _EWMA_filter(self, a, rate, b):

        return (1 - rate) * a + rate * b

    def _logistic(self, variable, gain, bias):

        return 1 / (1 + np.exp(-(gain * variable) + bias))

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """
        Return: some fraction of `variable <AGTUtilityIntegrator.variable>` combined with some fraction of `previous_value
        <AGTUtilityIntegrator.previous_value>`.

        Arguments
        ---------

        variable : number, list or np.array : default ClassDefaults.variable
           a single value or array of values to be integrated.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        updated value of integral : 2d np.array

        """
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))
        rate = np.array(self.get_current_function_param(RATE)).astype(float)
        # execute noise if it is a function
        noise = self._try_execute_param(self.get_current_function_param(NOISE), variable)
        short_term_rate = self.get_current_function_param("short_term_rate")
        long_term_rate = self.get_current_function_param("long_term_rate")

        # Integrate Short Term Utility:
        short_term_utility = self._EWMA_filter(self.previous_short_term_utility,
                                               short_term_rate,
                                               variable)
        # Integrate Long Term Utility:
        long_term_utility = self._EWMA_filter(self.previous_long_term_utility,
                                              long_term_rate,
                                              variable)

        value = self.combine_utilities(short_term_utility, long_term_utility)

        if self.context.initialization_status != ContextFlags.INITIALIZING:
            self.previous_short_term_utility = short_term_utility
            self.previous_long_term_utility = long_term_utility

        return value

    def combine_utilities(self, short_term_utility, long_term_utility):
        short_term_gain = self.get_current_function_param("short_term_gain")
        short_term_bias = self.get_current_function_param("short_term_bias")
        long_term_gain = self.get_current_function_param("long_term_gain")
        long_term_bias = self.get_current_function_param("long_term_bias")
        operation = self.get_current_function_param(OPERATION)
        offset = self.get_current_function_param(OFFSET)

        short_term_utility_logistic = self._logistic(variable=short_term_utility,
                                                     gain=short_term_gain,
                                                     bias=short_term_bias)
        self.short_term_utility_logistic = short_term_utility_logistic

        long_term_utility_logistic = self._logistic(variable=long_term_utility,
                                                    gain=long_term_gain,
                                                    bias=long_term_bias)
        self.long_term_utility_logistic = long_term_utility_logistic

        if operation == "s*l":
            # Engagement in current task = [1—logistic(short term utility)]*[logistic{long - term utility}]
            value = (1-short_term_utility_logistic)*long_term_utility_logistic
        elif operation =="s-l":
            # Engagement in current task = [1—logistic(short term utility)] - [logistic{long - term utility}]
            value = (1-short_term_utility_logistic) - long_term_utility_logistic
        elif operation =="s+l":
            # Engagement in current task = [1—logistic(short term utility)] + [logistic{long - term utility}]
            value = (1 - short_term_utility_logistic) + long_term_utility_logistic
        elif operation =="l-s":
            # Engagement in current task = [logistic{long - term utility}] - [1—logistic(short term utility)]
            value = long_term_utility_logistic - (1-short_term_utility_logistic)

        return value + offset

    def reinitialize(self, short=None, long=None):

        """
        Effectively begins accumulation over again at the specified utilities.

        Sets

        - `previous_short_term_utility <AGTUtilityIntegrator.previous_short_term_utility>`
        - `initial_short_term_utility <AGTUtilityIntegrator.initial_short_term_utility>`

        to the quantity specified in the first argument.

        Sets

        - `previous_long_term_utility <AGTUtilityIntegrator.previous_long_term_utility>`
        - `initial_long_term_utility <AGTUtilityIntegrator.initial_long_term_utility>`

        to the quantity specified in the second argument.

        sets `value <AGTUtilityIntegrator.value>` by computing it based on the newly updated values for
        `previous_short_term_utility <AGTUtilityIntegrator.previous_short_term_utility>` and
        `previous_long_term_utility <AGTUtilityIntegrator.previous_long_term_utility>`.

        If no arguments are specified, then the instance defaults for `initial_short_term_utility
        <AGTUtilityIntegrator.initial_short_term_utility>` and `initial_long_term_utility
        <AGTUtilityIntegrator.initial_long_term_utility>` are used.
        """

        if short is None:
            short = self.instance_defaults.initial_short_term_utility
        if long is None:
            long = self.instance_defaults.initial_long_term_utility
        self._initial_short_term_utility = short
        self.previous_short_term_utility = short
        self._initial_long_term_utility = long
        self.previous_long_term_utility = long
        self.value = self.combine_utilities(short, long)
        return self.value
#
# Note:  For any of these that correspond to args, value must match the name of the corresponding arg in __init__()
DRIFT_RATE = 'drift_rate'
DRIFT_RATE_VARIABILITY = 'DDM_DriftRateVariability'
THRESHOLD = 'threshold'
THRESHOLD_VARIABILITY = 'DDM_ThresholdRateVariability'
STARTING_POINT = 'starting_point'
STARTING_POINT_VARIABILITY = "DDM_StartingPointVariability"
# NOISE = 'noise' -- Defined in Keywords
NON_DECISION_TIME = 't0'

# DDM solution options:
kwBogaczEtAl = "BogaczEtAl"
kwNavarrosAndFuss = "NavarroAndFuss"


# QUESTION: IF VARIABLE IS AN ARRAY, DOES IT RETURN AN ARRAY FOR EACH RETURN VALUE (RT, ER, ETC.)
class BogaczEtAl(
    IntegratorFunction):  # --------------------------------------------------------------------------------
    """
    BogaczEtAl(                 \
        default_variable=None,  \
        drift_rate=1.0,         \
        threshold=1.0,          \
        starting_point=0.0,     \
        t0=0.2                  \
        noise=0.5,              \
        params=None,            \
        owner=None,             \
        prefs=None              \
        )

    .. _BogaczEtAl:

    Return terminal value of decision variable, mean accuracy, and mean response time computed analytically for the
    drift diffusion process as described in `Bogacz et al (2006) <https://www.ncbi.nlm.nih.gov/pubmed/17014301>`_.

    Arguments
    ---------

    default_variable : number, list or np.array : default ClassDefaults.variable
        specifies a template for decision variable(s);  if it is list or array, a separate solution is computed
        independently for each element.

    drift_rate : float, list or 1d np.array : default 1.0
        specifies the drift_rate of the drift diffusion process.  If it is a list or array,
        it must be the same length as `default_variable <BogaczEtAl.default_variable>`.

    threshold : float, list or 1d np.array : default 1.0
        specifies the threshold (boundary) of the drift diffusion process.  If it is a list or array,
        it must be the same length as `default_variable <BogaczEtAl.default_variable>`.

    starting_point : float, list or 1d np.array : default 1.0
        specifies the initial value of the decision variable for the drift diffusion process.  If it is a list or
        array, it must be the same length as `default_variable <BogaczEtAl.default_variable>`.

    noise : float, list or 1d np.array : default 0.0
        specifies the noise term (corresponding to the diffusion component) of the drift diffusion process.
        If it is a float, it must be a number from 0 to 1.  If it is a list or array, it must be the same length as
        `default_variable <BogaczEtAl.default_variable>` and all elements must be floats from 0 to 1.

    t0 : float, list or 1d np.array : default 0.2
        specifies the non-decision time for solution. If it is a float, it must be a number from 0 to 1.  If it is a
        list or array, it must be the same length as  `default_variable <BogaczEtAl.default_variable>` and all
        elements must be floats from 0 to 1.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : number or 1d np.array
        holds initial value assigned to :keyword:`default_variable` argument;
        ignored by `function <BogaczEtal.function>`.

    drift_rate : float or 1d np.array
        determines the drift component of the drift diffusion process.

    threshold : float or 1d np.array
        determines the threshold (boundary) of the drift diffusion process (i.e., at which the integration
        process is assumed to terminate).

    starting_point : float or 1d np.array
        determines the initial value of the decision variable for the drift diffusion process.

    noise : float or 1d np.array
        determines the diffusion component of the drift diffusion process (used to specify the variance of a
        Gaussian random process).

    t0 : float or 1d np.array
        determines the assumed non-decision time to determine the response time returned by the solution.

    bias : float or 1d np.array
        normalized starting point:
        (`starting_point <BogaczEtAl.starting_point>` + `threshold <BogaczEtAl.threshold>`) /
        (2 * `threshold <BogaczEtAl.threshold>`)

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = kwBogaczEtAl

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 drift_rate:parameter_spec = 1.0,
                 starting_point: parameter_spec = 0.0,
                 threshold: parameter_spec = 1.0,
                 noise: parameter_spec = 0.5,
                 t0: parameter_spec = .200,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(drift_rate=drift_rate,
                                                  starting_point=starting_point,
                                                  threshold=threshold,
                                                  noise=noise,
                                                  t0=t0,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """
        Return: terminal value of decision variable (equal to threshold), mean accuracy (error rate; ER) and mean
        response time (RT)

        Arguments
        ---------

        variable : 2d np.array
            ignored.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------
        Decision variable, mean ER, mean RT : (float, float, float)

        """

        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        attentional_drift_rate = float(self.get_current_function_param(DRIFT_RATE))
        stimulus_drift_rate = float(variable)
        drift_rate = attentional_drift_rate * stimulus_drift_rate
        threshold = float(self.get_current_function_param(THRESHOLD))
        starting_point = float(self.get_current_function_param(STARTING_POINT))
        noise = float(self.get_current_function_param(NOISE))
        t0 = float(self.get_current_function_param(NON_DECISION_TIME))

        # drift_rate = float(self.drift_rate) * float(variable)
        # threshold = float(self.threshold)
        # starting_point = float(self.starting_point)
        # noise = float(self.noise)
        # t0 = float(self.t0)

        self.bias = bias = (starting_point + threshold) / (2 * threshold)

        # Prevents div by 0 issue below:
        if bias <= 0:
            bias = 1e-8
        if bias >= 1:
            bias = 1 - 1e-8

        # drift_rate close to or at 0 (avoid float comparison)
        if np.abs(drift_rate) < 1e-8:
            # back to absolute bias in order to apply limit
            bias_abs = bias * 2 * threshold - threshold
            # use expression for limit a->0 from Srivastava et al. 2016
            rt = t0 + (threshold ** 2 - bias_abs ** 2) / (noise ** 2)
            er = (threshold - bias_abs) / (2 * threshold)
        else:
            drift_rate_normed = np.abs(drift_rate)
            ztilde = threshold / drift_rate_normed
            atilde = (drift_rate_normed / noise) ** 2

            is_neg_drift = drift_rate < 0
            bias_adj = (is_neg_drift == 1) * (1 - bias) + (is_neg_drift == 0) * bias
            y0tilde = ((noise ** 2) / 2) * np.log(bias_adj / (1 - bias_adj))
            if np.abs(y0tilde) > threshold:
                y0tilde = -1 * (is_neg_drift == 1) * threshold + (is_neg_drift == 0) * threshold
            x0tilde = y0tilde / drift_rate_normed

            with np.errstate(over='raise', under='raise'):
                try:
                    rt = ztilde * np.tanh(ztilde * atilde) + \
                         ((2 * ztilde * (1 - np.exp(-2 * x0tilde * atilde))) / (
                         np.exp(2 * ztilde * atilde) - np.exp(-2 * ztilde * atilde)) - x0tilde) + t0
                    er = 1 / (1 + np.exp(2 * ztilde * atilde)) - \
                         ((1 - np.exp(-2 * x0tilde * atilde)) / (np.exp(2 * ztilde * atilde) - np.exp(-2 * ztilde * atilde)))

                except FloatingPointError:
                    # Per Mike Shvartsman:
                    # If ±2*ztilde*atilde (~ 2*z*a/(c^2) gets very large, the diffusion vanishes relative to drift
                    # and the problem is near-deterministic. Without diffusion, error rate goes to 0 or 1
                    # depending on the sign of the drift, and so decision time goes to a point mass on z/a – x0, and
                    # generates a "RuntimeWarning: overflow encountered in exp"
                    er = 0
                    rt = ztilde / atilde - x0tilde + t0

            # This last line makes it report back in terms of a fixed reference point
            #    (i.e., closer to 1 always means higher p(upper boundary))
            # If you comment this out it will report errors in the reference frame of the drift rate
            #    (i.e., reports p(upper) if drift is positive, and p(lower if drift is negative)
            er = (is_neg_drift == 1) * (1 - er) + (is_neg_drift == 0) * (er)

        return rt, er

    def derivative(self, output=None, input=None):
        """
        derivative(output, input)

        Calculate the derivative of :math:`\\frac{1}{reward rate}` with respect to the threshold (**output** arg)
        and drift_rate (**input** arg).  Reward rate (:math:`RR`) is assumed to be:

            :math:`RR = delay_{ITI} + \\frac{Z}{A} + ED`;

        the derivative of :math:`\\frac{1}{RR}` with respect to the `threshold <BogaczEtAl.threshold>` is:

            :math:`\\frac{1}{A} - \\frac{E}{A} - 2\\frac{A}{c^2}ED`;

        and the derivative of 1/RR with respect to the `drift_rate <BogaczEtAl.drift_rate>` is:

            :math:`-\\frac{Z}{A^2} + \\frac{Z}{A^2}E - \\frac{2Z}{c^2}ED`

        where:

            *A* = `drift_rate <BogaczEtAl.drift_rate>`,

            *Z* = `threshold <BogaczEtAl.threshold>`,

            *c* = `noise <BogaczEtAl.noise>`,

            *E* = :math:`e^{-2\\frac{ZA}{c^2}}`,

            *D* = :math:`delay_{ITI} + delay_{penalty} - \\frac{Z}{A}`,

            :math:`delay_{ITI}` is the intertrial interval and :math:`delay_{penalty}` is a penalty delay.


        Returns
        -------

        derivatives :  List[float, float)
            of :math:`\\frac{1}{RR}` with respect to `threshold <BogaczEtAl.threshold>` and `drift_rate
            <BogaczEtAl.drift_rate>`.

        """
        Z = output or self.get_current_function_param(THRESHOLD)
        A = input or self.get_current_function_param(DRIFT_RATE)
        c = self.get_current_function_param(NOISE)
        c_sq = c**2
        E = np.exp(-2*Z*A/c_sq)
        D_iti = 0
        D_pen = 0
        D = D_iti + D_pen
        # RR =  1/(D_iti + Z/A + (E*D))

        dRR_dZ = 1/A + E/A + (2*A/c_sq)*E*D
        dRR_dA = -Z/A**2 + (Z/A**2)*E - (2*Z/c_sq)*E*D

        return [dRR_dZ, dRR_dA]


# Results from Navarro and Fuss DDM solution (indices for return value tuple)
class NF_Results(IntEnum):
    MEAN_ER = 0
    MEAN_RT = 1
    MEAN_DT = 2
    COND_RTS = 3
    COND_VAR_RTS = 4
    COND_SKEW_RTS = 5


# ----------------------------------------------------------------------------
class NavarroAndFuss(IntegratorFunction):
    """
    NavarroAndFuss(                             \
        default_variable=None,                  \
        drift_rate=1.0,                         \
        threshold=1.0,                          \
        starting_point=0.0,                     \
        t0=0.2                                  \
        noise=0.5,                              \
        params=None,                            \
        owner=None,                             \
        prefs=None                              \
        )

    .. _NavarroAndFuss:

    Return terminal value of decision variable, mean accuracy, mean response time (RT), correct RT mean, correct RT
    variance and correct RT skew computed analytically for the drift diffusion process (Wiener diffusion model)
    as described in `Navarro and Fuss (2009) <http://www.sciencedirect.com/science/article/pii/S0022249609000200>`_.

    .. note::
       Use of this Function requires that the MatLab engine is installed.

    Arguments
    ---------

    default_variable : number, list or np.array : default ClassDefaults.variable
        specifies a template for decision variable(s);  if it is list or array, a separate solution is computed
        independently for each element.

    drift_rate : float, list or 1d np.array : default 1.0
        specifies the drift_rate of the drift diffusion process.  If it is a list or array,
        it must be the same length as `default_variable <BogaczEtAl.default_variable>`.

    threshold : float, list or 1d np.array : default 1.0
        specifies the threshold (boundary) of the drift diffusion process.  If it is a list or array,
        it must be the same length as `default_variable <BogaczEtAl.default_variable>`.

    starting_point : float, list or 1d np.array : default 1.0
        specifies the initial value of the decision variable for the drift diffusion process.  If it is a list or
        array, it must be the same length as `default_variable <BogaczEtAl.default_variable>`.

    noise : float, list or 1d np.array : default 0.0
        specifies the noise term (corresponding to the diffusion component) of the drift diffusion process.
        If it is a float, it must be a number from 0 to 1.  If it is a list or array, it must be the same length as
        `default_variable <BogaczEtAl.default_variable>` and all elements must be floats from 0 to 1.

    t0 : float, list or 1d np.array : default 0.2
        specifies the non-decision time for solution. If it is a float, it must be a number from 0 to 1.  If it is a
        list or array, it must be the same length as  `default_variable <BogaczEtAl.default_variable>` and all
        elements must be floats from 0 to 1.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : number or 1d np.array
        holds initial value assigned to :keyword:`default_variable` argument;
        ignored by `function <NovarroAndFuss.function>`.

    drift_rate : float or 1d np.array
        determines the drift component of the drift diffusion process.

    threshold : float or 1d np.array
        determines the threshold (bound) of the drift diffusion process (i.e., at which the integration
        process is assumed to terminate).

    starting_point : float or 1d np.array
        determines the initial value of the decision variable for the drift diffusion process.

    noise : float or 1d np.array
        determines the diffusion component of the drift diffusion process (used to specify the variance of a
        Gaussian random process).

    t0 : float or 1d np.array
        determines the assumed non-decision time to determine the response time returned by the solution.

    bias : float or 1d np.array
        normalized starting point:
        (`starting_point <BogaczEtAl.starting_point>` + `threshold <BogaczEtAl.threshold>`) /
        (2 * `threshold <BogaczEtAl.threshold>`)

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = kwNavarrosAndFuss

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 drift_rate: parameter_spec = 1.0,
                 starting_point: parameter_spec = 0.0,
                 threshold: parameter_spec = 1.0,
                 noise: parameter_spec = 0.5,
                 t0: parameter_spec = .200,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(drift_rate=drift_rate,
                                                  starting_point=starting_point,
                                                  threshold=threshold,
                                                  noise=noise,
                                                  t0=t0,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

    def _instantiate_function(self, function, function_params=None, context=None):
        import os
        import sys
        try:
            import matlab.engine
        except ImportError as e:
            raise ImportError(
                'python failed to import matlab. Ensure that MATLAB and the python API is installed. See'
                ' https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html'
                ' for more info'
            )

        ddm_functions_path = os.path.abspath(sys.modules['psyneulink'].__path__[0] + '/../Matlab/DDMFunctions')

        # must add the package-included MATLAB files to the engine path to run when not running from the path
        # MATLAB is very finnicky about the formatting here to actually add the path so be careful if you modify
        self.eng1 = matlab.engine.start_matlab("-r 'addpath(char(\"{0}\"))' -nojvm".format(ddm_functions_path))

        super()._instantiate_function(function=function, function_params=function_params, context=context)

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """
        Return: terminal value of decision variable, mean accuracy (error rate; ER), mean response time (RT),
        correct RT mean, correct RT variance and correct RT skew.  **Requires that the MatLab engine is installed.**

        Arguments
        ---------

        variable : 2d np.array
            ignored.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------
        Decision variable, mean ER, mean RT, correct RT mean, correct RT variance, correct RT skew : \
        (float, float, float, float, float, float)

        """

        self._check_args(variable=variable, params=params, context=context)

        drift_rate = float(self.get_current_function_param(DRIFT_RATE))
        threshold = float(self.get_current_function_param(THRESHOLD))
        starting_point = float(self.get_current_function_param(STARTING_POINT))
        noise = float(self.get_current_function_param(NOISE))
        t0 = float(self.get_current_function_param(NON_DECISION_TIME))

        # used to pass values in a way that the matlab script can handle
        ddm_struct = {
            'z': threshold,
            'c': noise,
            'T0': t0
        }

        results = self.eng1.ddmSimFRG(drift_rate, starting_point, ddm_struct, 1, nargout=6)

        return results


# region ************************************   DISTRIBUTION FUNCTIONS   ***********************************************

class DistributionFunction(Function_Base):
    componentType = DIST_FUNCTION_TYPE


class NormalDist(DistributionFunction):
    """
    NormalDist(                     \
             mean=0.0,              \
             standard_dev=1.0,      \
             params=None,           \
             owner=None,            \
             prefs=None             \
             )

    .. _NormalDist:

    Return a random sample from a normal distribution using numpy.random.normal

    Arguments
    ---------

    mean : float : default 0.0
        The mean or center of the normal distribution

    standard_dev : float : default 1.0
        Standard deviation of the normal distribution. Must be > 0.0

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    mean : float : default 0.0
        The mean or center of the normal distribution

    standard_dev : float : default 1.0
        Standard deviation of the normal distribution. Must be > 0.0

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    """

    componentName = NORMAL_DIST_FUNCTION

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 mean=0.0,
                 standard_dev=1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(mean=mean,
                                                  standard_dev=standard_dev,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

        self.functionOutputType = None

    def _validate_params(self, request_set, target_set=None, context=None):
        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        if STANDARD_DEVIATION in target_set:
            if target_set[STANDARD_DEVIATION] <= 0.0:
                raise FunctionError("The standard_dev parameter ({}) of {} must be greater than zero.".
                                            format(target_set[STANDARD_DEVIATION], self.name))


    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        # Validate variable and validate params
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        mean = self.get_current_function_param(DIST_MEAN)
        standard_deviation = self.get_current_function_param(STANDARD_DEVIATION)

        result = np.random.normal(mean, standard_deviation)

        return result


class UniformToNormalDist(DistributionFunction):
    """
    UniformToNormalDist(            \
             mean=0.0,              \
             standard_dev=1.0,      \
             params=None,           \
             owner=None,            \
             prefs=None             \
             )

    .. _UniformToNormalDist:

    Return a random sample from a normal distribution using first np.random.rand(1) to generate a sample from a uniform
    distribution, and then converting that sample to a sample from a normal distribution with the following equation:

    .. math::

        normal\\_sample = \\sqrt{2} \\cdot standard\\_dev \\cdot scipy.special.erfinv(2 \\cdot uniform\\_sample - 1)  + mean

    The uniform --> normal conversion allows for a more direct comparison with MATLAB scripts.

    .. note::

        This function requires `SciPy <https://pypi.python.org/pypi/scipy>`_.

    (https://github.com/jonasrauber/randn-matlab-python)

    Arguments
    ---------

    mean : float : default 0.0
        The mean or center of the normal distribution

    standard_dev : float : default 1.0
        Standard deviation of the normal distribution

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    mean : float : default 0.0
        The mean or center of the normal distribution

    standard_dev : float : default 1.0
        Standard deviation of the normal distribution

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    """

    componentName = NORMAL_DIST_FUNCTION

    class ClassDefaults(DistributionFunction.ClassDefaults):
        variable = [0]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 mean=0.0,
                 standard_dev=1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(mean=mean,
                                                  standard_dev=standard_dev,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

        self.functionOutputType = None

    def function(self,
                 variable=None,
                 params=None,
                 context=None):

        try:
            from scipy.special import erfinv
        except:
            raise FunctionError("The UniformToNormalDist function requires the SciPy package.")

        # Validate variable and validate params
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        mean = self.get_current_function_param(DIST_MEAN)
        standard_deviation = self.get_current_function_param(STANDARD_DEVIATION)

        sample = np.random.rand(1)[0]
        return ((np.sqrt(2) * erfinv(2 * sample - 1)) * standard_deviation) + mean

class ExponentialDist(DistributionFunction):
    """
    ExponentialDist(                \
             beta=1.0,              \
             params=None,           \
             owner=None,            \
             prefs=None             \
             )

    .. _ExponentialDist:

    Return a random sample from a exponential distribution using numpy.random.exponential

    Arguments
    ---------

    beta : float : default 1.0
        The scale parameter of the exponential distribution

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    beta : float : default 1.0
        The scale parameter of the exponential distribution

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    """
    componentName = EXPONENTIAL_DIST_FUNCTION

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 beta=1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(beta=beta,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

        self.functionOutputType = None

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        # Validate variable and validate params
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        beta = self.get_current_function_param(BETA)
        result = np.random.exponential(beta)

        return result


class UniformDist(DistributionFunction):
    """
    UniformDist(                      \
             low=0.0,             \
             high=1.0,             \
             params=None,           \
             owner=None,            \
             prefs=None             \
             )

    .. _UniformDist:

    Return a random sample from a uniform distribution using numpy.random.uniform

    Arguments
    ---------

    low : float : default 0.0
        Lower bound of the uniform distribution

    high : float : default 1.0
        Upper bound of the uniform distribution

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    low : float : default 0.0
        Lower bound of the uniform distribution

    high : float : default 1.0
        Upper bound of the uniform distribution

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    """
    componentName = UNIFORM_DIST_FUNCTION

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 low=0.0,
                 high=1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(low=low,
                                                  high=high,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

        self.functionOutputType = None

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        # Validate variable and validate params
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        low = self.get_current_function_param(LOW)
        high = self.get_current_function_param(HIGH)
        result = np.random.uniform(low, high)

        return result


class GammaDist(DistributionFunction):
    """
    GammaDist(\
             scale=1.0,\
             dist_shape=1.0,\
             params=None,\
             owner=None,\
             prefs=None\
             )

    .. _GammaDist:

    Return a random sample from a gamma distribution using numpy.random.gamma

    Arguments
    ---------

    scale : float : default 1.0
        The scale of the gamma distribution. Should be greater than zero.

    dist_shape : float : default 1.0
        The shape of the gamma distribution. Should be greater than zero.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    scale : float : default 1.0
        The dist_shape of the gamma distribution. Should be greater than zero.

    dist_shape : float : default 1.0
        The scale of the gamma distribution. Should be greater than zero.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    """

    componentName = GAMMA_DIST_FUNCTION

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 scale=1.0,
                 dist_shape=1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(scale=scale,
                                                  dist_shape=dist_shape,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

        self.functionOutputType = None

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        # Validate variable and validate params
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        scale = self.get_current_function_param(SCALE)
        dist_shape = self.get_current_function_param(DIST_SHAPE)

        result = np.random.gamma(dist_shape, scale)

        return result


class WaldDist(DistributionFunction):
    """
     WaldDist(             \
              scale=1.0,\
              mean=1.0,\
              params=None,\
              owner=None,\
              prefs=None\
              )

     .. _WaldDist:

     Return a random sample from a Wald distribution using numpy.random.wald

     Arguments
     ---------

     scale : float : default 1.0
         Scale parameter of the Wald distribution. Should be greater than zero.

     mean : float : default 1.0
         Mean of the Wald distribution. Should be greater than or equal to zero.

     params : Dict[param keyword: param value] : default None
         a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
         function.  Values specified for parameters in the dictionary override any assigned to those parameters in
         arguments of the constructor.

     owner : Component
         `component <Component>` to which to assign the Function.

     prefs : PreferenceSet or specification dict : default Function.classPreferences
         the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
         defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


     Attributes
     ----------

     scale : float : default 1.0
         Scale parameter of the Wald distribution. Should be greater than zero.

     mean : float : default 1.0
         Mean of the Wald distribution. Should be greater than or equal to zero.

     params : Dict[param keyword: param value] : default None
         a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
         function.  Values specified for parameters in the dictionary override any assigned to those parameters in
         arguments of the constructor.

     owner : Component
         `component <Component>` to which to assign the Function.

     prefs : PreferenceSet or specification dict : default Function.classPreferences
         the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
         defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


     """

    componentName = WALD_DIST_FUNCTION

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 scale=1.0,
                 mean=1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(scale=scale,
                                                  mean=mean,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

        self.functionOutputType = None

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        # Validate variable and validate params
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        scale = self.get_current_function_param(SCALE)
        mean = self.get_current_function_param(DIST_MEAN)

        result = np.random.wald(mean, scale)

        return result


# endregion

# region **************************************   OBJECTIVE FUNCTIONS **************************************************

class ObjectiveFunction(Function_Base):
    """Abstract class of `Function` used for evaluating states.
    """

    componentType = OBJECTIVE_FUNCTION_TYPE


class Stability(ObjectiveFunction):
    """
    Stability(                                  \
        default_variable=None,                  \
        matrix=HOLLOW_MATRIX,                   \
        metric=ENERGY                           \
        transfer_fct=None                       \
        normalize=False,                        \
        params=None,                            \
        owner=None,                             \
        prefs=None                              \
        )

    .. _Stability:

    Return the stability of `variable <Stability.variable>` based on a state transformation matrix.

    The value of `variable <Stability.variable>` is passed through the `matrix <Stability.matrix>`,
    transformed using the `transfer_fct <Stability.transfer_fct>` (if specified), and then compared with its initial
    value using the `distance metric <DistanceMetric>` specified by `metric <Stability.metric>`.  If `normalize
    <Stability.normalize>` is `True`, the result is normalized by the length of (number of elements in) `variable
    <Stability.variable>`.

COMMENT:
*** 11/11/17 - DELETE THIS ONE Stability IS STABLE:
    Stability s is calculated according as specified by `metric <Distance.metric>`, using the formulae below,
    where :math:`i` and :math:`j` are each elements of `variable <Stability.variable>`, *len* is its length,
    :math:`\\bar{v}` is its mean, :math:`\\sigma_v` is its standard deviation, and :math:`w_{ij}` is the entry of the
    weight matrix for the connection between entries :math:`i` and :math:`j` in `variable <Stability.variable>`.

    *ENTROPY*:

       :math:`s = -\\sum\\limits^{len}(i*log(j))`

    *DIFFERENCE*:

       :math:`s = \\sum\\limits^{len}(i-j)`

    *EUCLIDEAN*:

       :math:`s = \\sum\\limits^{len}\\sqrt{(i-j)^2}`

    *CORRELATION*:

       :math:`s = \\frac{\\sum\\limits^{len}(i-\\bar{i})(j-\\bar{j})}{(len-1)\\sigma_{i}\\sigma_{j}}`

    **normalize**:

       :math:`s = \\frac{s}{len}`
COMMENT


    Arguments
    ---------

    variable : list of numbers or 1d np.array : Default ClassDefaults.variable
        the array for which stability is calculated.

    matrix : list, np.ndarray, np.matrix, function keyword, or MappingProjection : default HOLLOW_MATRIX
        specifies the matrix of recurrent weights;  must be a square matrix with the same width as the
        length of `variable <Stability.variable>`.

    metric : keyword in DistanceMetrics : Default ENERGY
        specifies a `metric <DistanceMetrics>` from `DistanceMetrics` used to compute stability.

    transfer_fct : function or method : Default None
        specifies the function used to transform output of weight `matrix <Stability.matrix>`.

    normalize : bool : Default False
        specifies whether to normalize the stability value by the length of `variable <Stability.variable>`.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).
    Attributes
    ----------

    variable : 1d np.array
        array for which stability is calculated.

    matrix : list, np.ndarray, np.matrix, function keyword, or MappingProjection : default HOLLOW_MATRIX
        weight matrix from each element of `variable <Stability.variablity>` to each other;  if a matrix other
        than HOLLOW_MATRIX is assigned, it is convolved with HOLLOW_MATRIX to eliminate self-connections from the
        stability calculation.

    metric : keyword in DistanceMetrics
        metric used to compute stability; must be a `DistanceMetrics` keyword. The `Distance` Function is used to
        compute the stability of `variable <Stability.variable>` with respect to its value after its transformation
        by `matrix <Stability.matrix>` and `transfer_fct <Stability.transfer_fct>`.

    transfer_fct : function or method
        function used to transform output of weight `matrix <Stability.matrix>` prior to computing stability.

    normalize : bool
        if `True`, result of stability calculation is normalized by the length of `variable <Stability.variable>`.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).     """

    componentName = STABILITY_FUNCTION

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 matrix=HOLLOW_MATRIX,
                 # metric:is_distance_metric=ENERGY,
                 metric:tc.any(tc.enum(ENERGY, ENTROPY), is_distance_metric)=ENERGY,
                 transfer_fct:tc.optional(tc.any(function_type, method_type))=None,
                 normalize:bool=False,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(matrix=matrix,
                                                  metric=metric,
                                                  transfer_fct=transfer_fct,
                                                  normalize=normalize,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

        self.functionOutputType = None

    def _validate_variable(self, variable, context=None):
        """Validates that variable is 1d array
        """
        if len(np.atleast_2d(variable)) != 1:
            raise FunctionError("Variable for {} must contain a single array or list of numbers".format(self.name))
        return variable

    def _validate_params(self, variable, request_set, target_set=None, context=None):
        """Validate matrix param

        `matrix <Stability.matrix>` argument must be one of the following
            - 2d list, np.ndarray or np.matrix
            - ParameterState for one of the above
            - MappingProjection with a parameterStates[MATRIX] for one of the above

        Parse matrix specification to insure it resolves to a square matrix
        (but leave in the form in which it was specified so that, if it is a ParameterState or MappingProjection,
         its current value can be accessed at runtime (i.e., it can be used as a "pointer")
        """

        # Validate matrix specification
        if MATRIX in target_set:

            from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
            from psyneulink.components.states.parameterstate import ParameterState

            matrix = target_set[MATRIX]

            if isinstance(matrix, str):
                matrix = get_matrix(matrix)

            if isinstance(matrix, MappingProjection):
                try:
                    matrix = matrix._parameter_states[MATRIX].value
                    param_type_string = "MappingProjection's ParameterState"
                except KeyError:
                    raise FunctionError("The MappingProjection specified for the {} arg of {} ({}) must have a {} "
                                        "ParameterState that has been assigned a 2d array or matrix".
                                        format(MATRIX, self.name, matrix.shape, MATRIX))

            elif isinstance(matrix, ParameterState):
                try:
                    matrix = matrix.value
                    param_type_string = "ParameterState"
                except KeyError:
                    raise FunctionError("The value of the {} parameterState specified for the {} arg of {} ({}) "
                                        "must be a 2d array or matrix".
                                        format(MATRIX, MATRIX, self.name, matrix.shape))

            else:
                param_type_string = "array or matrix"

            matrix = np.array(matrix)
            if matrix.ndim != 2:
                raise FunctionError("The value of the {} specified for the {} arg of {} ({}) "
                                    "must be a 2d array or matrix".
                                    format(param_type_string, MATRIX, self.name, matrix))
            rows = matrix.shape[0]
            cols = matrix.shape[1]
            # MODIFIED 11/25/17 OLD:
            # size = len(np.squeeze(self.instance_defaults.variable))
            # MODIFIED 11/25/17 NEW:
            size = len(self.instance_defaults.variable)
            # MODIFIED 11/25/17 END

            if rows != size:
                raise FunctionError("The value of the {} specified for the {} arg of {} is the wrong size;"
                                    "it is {}x{}, but must be square matrix of size {}".
                                    format(param_type_string, MATRIX, self.name, rows, cols, size))

            if rows != cols:
                raise FunctionError("The value of the {} specified for the {} arg of {} ({}) "
                                    "must be a square matrix".
                                    format(param_type_string, MATRIX, self.name, matrix))

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)



    def _instantiate_attributes_before_function(self, function=None, context=None):
        """Instantiate matrix

        Specified matrix is convolved with HOLLOW_MATRIX
            to eliminate the diagonal (self-connections) from the calculation.
        The `Distance` Function is used for all calculations except ENERGY (which is not really a distance metric).
        If ENTROPY is specified as the metric, convert to CROSS_ENTROPY for use with the Distance Function.
        :param function:

        """

        # MODIFIED 11/25/17 OLD:
        # size = len(np.squeeze(self.instance_defaults.variable))
        # MODIFIED 11/25/17 NEW:
        size = len(self.instance_defaults.variable)
        # MODIFIED 11/25/17 END

        from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
        from psyneulink.components.states.parameterstate import ParameterState
        if isinstance(self.matrix,MappingProjection):
            self._matrix = self.matrix._parameter_states[MATRIX]
        elif isinstance(self.matrix,ParameterState):
            pass
        else:
            self._matrix = get_matrix(self.matrix, size, size)

        self._hollow_matrix = get_matrix(HOLLOW_MATRIX, size, size)

        # # MODIFIED 11/12/17 OLD:
        # if self.metric is ENTROPY:
        #     self._metric_fct = Distance(metric=CROSS_ENTROPY)
        # elif self.metric in DISTANCE_METRICS:
        #     self._metric_fct = Distance(metric=self.metric)
        # MODIFIED 11/12/17 NEW:
        if self.metric is ENTROPY:
            self._metric_fct = Distance(metric=CROSS_ENTROPY, normalize=self.normalize)
        elif self.metric in DISTANCE_METRICS._set():
            self._metric_fct = Distance(metric=self.metric, normalize=self.normalize)
        # MODIFIED 11/12/17 END


    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """Calculate the stability of `variable <Stability.variable>`.

        Compare the value of `variable <Stability.variable>` with its value after transformation by
        `matrix <Stability.matrix>` and `transfer_fct <Stability.transfer_fct>` (if specified), using the specified
        `metric <Stability.metric>`.  If `normalize <Stability.normalize>` is `True`, the result is divided
        by the length of `variable <Stability.variable>`.

        Returns
        -------

        stability : scalar

        """
        # Validate variable and validate params
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        from psyneulink.components.states.parameterstate import ParameterState
        if isinstance(self.matrix, ParameterState):
            matrix = self.matrix.value
        else:
            matrix = self.matrix

        current = variable
        if self.transfer_fct is not None:
            transformed = self.transfer_fct(np.dot(matrix * self._hollow_matrix, variable))
        else:
            transformed = np.dot(matrix * self._hollow_matrix, variable)

        # # MODIFIED 11/12/15 OLD:
        # if self.metric is ENERGY:
        #     result = -np.sum(current * transformed)/2
        # else:
        #     result = self._metric_fct.function(variable=[current,transformed], context=context)
        #
        # if self.normalize:
        #     if self.metric is ENERGY:
        #         result /= len(variable)**2
        #     else:
        #         result /= len(variable)
        # MODIFIED 11/12/15 NEW:
        result = self._metric_fct.function(variable=[current,transformed], context=context)
        # MODIFIED 11/12/15 END

        return result

# endregion

class Distance(ObjectiveFunction):
    """
    Distance(                                    \
       default_variable=None,                    \
       metric=EUCLIDEAN                          \
       normalize=False,                          \
       params=None,                              \
       owner=None,                               \
       prefs=None                                \
       )

    .. _Distance:

    Return the distance between the vectors in the two items of `variable <Distance.variable>` using the `distance
    metric <DistanceMetrics>` specified in the `metric <Stability.metric>` attribute.  If `normalize
    <Distance.normalize>` is `True`, the result is normalized by the length of (number of elements in) `variable
    <Stability.variable>`.

    Arguments
    ---------

    variable : 2d np.array with two items : Default ClassDefaults.variable
        the arrays between which the distance is calculated.

    metric : keyword in DistancesMetrics : Default EUCLIDEAN
        specifies a `distance metric <DistanceMetrics>` used to compute the distance between the two items in `variable
        <Distance.variable>`.

    normalize : bool : Default False
        specifies whether to normalize the distance by the length of `variable <Distance.variable>`.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : 2d np.array with two items
        contains the arrays between which the distance is calculated.

    metric : keyword in DistanceMetrics
        determines the `metric <DistanceMetrics>` used to compute the distance between the two items in `variable
        <Distance.variable>`.

    normalize : bool
        determines whether the distance is normalized by the length of `variable <Distance.variable>`.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).    """

    componentName = DISTANCE_FUNCTION

    class ClassDefaults(ObjectiveFunction.ClassDefaults):
        variable = np.array([[0], [0]])

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 metric:DistanceMetrics._is_metric=DIFFERENCE,
                 normalize:bool=False,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(metric=metric,
                                                  normalize=normalize,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

        self.functionOutputType = None

    def _validate_params(self, request_set, target_set=None, variable=None, context=None):
        """Validate that variable had two items of equal length

        """
        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        err_two_items = FunctionError("variable for {} ({}) must have two items".format(self.name, variable))

        try:
            if len(variable) != 2:
                raise err_two_items
        except TypeError:
            raise err_two_items

        try:
            if len(variable[0]) != len(variable[1]):
                raise FunctionError(
                    "The lengths of the items in the variable for {0} ({1},{2}) must be equal".format(
                        self.name,
                        variable[0],
                        variable[1]
                    )
                )
        except TypeError:
            if is_iterable(variable[0]) ^ is_iterable(variable[1]):
                raise FunctionError(
                    "The lengths of the items in the variable for {0} ({1},{2}) must be equal".format(
                        self.name,
                        variable[0],
                        variable[1]
                    )
                )

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """Calculate the distance between the two vectors in `variable <Stability.variable>`.

        Use the `distance metric <DistanceMetrics>` specified in `metric <Distance.metric>` to calculate the distance.
        If `normalize <Distance.normalize>` is `True`, the result is divided by the length of `variable
        <Distance.variable>`.

        Returns
        -------

        distance : scalar

        """
        # Validate variable and validate params
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        v1 = variable[0]
        v2 = variable[1]

        # Simple Hadamard difference of v1 and v2
        if self.metric is DIFFERENCE:
            result = np.sum(np.abs(v1 - v2))

        # Euclidean distance between v1 and v2
        elif self.metric is EUCLIDEAN:
            result = np.linalg.norm(v2-v1)

        # FIX: NEED SCIPY HERE
        # # Angle (cosine) of v1 and v2
        # elif self.metric is ANGLE:
        #     result = scipy.spatial.distance.cosine(v1,v2)

        # Correlation of v1 and v2
        elif self.metric is CORRELATION:
            result = np.correlate(v1, v2)

        # Pearson Correlation of v1 and v2
        elif self.metric is PEARSON:
            result = np.corrcoef(v1, v2)

        # Cross-entropy of v1 and v2
        elif self.metric is CROSS_ENTROPY:
            # FIX: VALIDATE THAT ALL ELEMENTS OF V1 AND V2 ARE 0 TO 1
            if self.context.initialization_status != ContextFlags.INITIALIZING:
                v1 = np.where(v1==0, EPSILON, v1)
                v2 = np.where(v2==0, EPSILON, v2)
            # MODIFIED CW 3/20/18: avoid divide by zero error by plugging in two zeros
            # FIX: unsure about desired behavior when v2 = 0 and v1 != 0
            # JDC: returns [inf]; leave, and let it generate a warning or error message for user
            result = np.where(np.logical_and(v1==0, v2==0), 0, -np.sum(v1*np.log(v2)))

        # Energy
        elif self.metric is ENERGY:
            result = -np.sum(v1*v2)/2

        if self.normalize:
            # # MODIFIED 11/12/17 OLD:
            # result /= len(variable[0])
            # MODIFIED 11/12/17 NEW:
            if self.metric is ENERGY:
                result /= len(v1)**2
            else:
                result /= len(v1)
            # MODIFIED 11/12/17 END

        return result

# endregion

# region **************************************   LEARNING FUNCTIONS ***************************************************

ReturnVal = namedtuple('ReturnVal', 'learning_signal, error_signal')

LEARNING_ACTIVATION_FUNCTION = 'activation_function'
LEARNING_ACTIVATION_INPUT = 0  # a(j)
# MATRIX = 1             # w
LEARNING_ACTIVATION_OUTPUT = 1  # a(i)
LEARNING_ERROR_OUTPUT = 2
AUTOASSOCIATIVE = 'AUTOASSOCIATIVE'

class LearningFunction(Function_Base):
    """Abstract class of `Function <Function>` used for learning.

    COMMENT:
    IMPLEMENTATION NOTE:
       The function method of a LearningFunction *must* include a **kwargs argument, which accomodates
       Function-specific parameters;  this is to accommodate the ability of LearningMechanisms to call
       the function of a LearningFunction with arguments that may not be implemented for all LearningFunctions
       (e.g., error_matrix for BackPropagation) -- these can't be included in the params argument, as those
       are validated against paramClassDefaults which will not recognize params specific to another Function.

    Attributes
    ----------

    variable : list or np.array
        most LearningFunctions take a list or 2d array that must contain three items:

        * the input to the parameter being modified (variable[LEARNING_ACTIVATION_INPUT]);
        * the output of the parameter being modified (variable[LEARNING_ACTIVATION_OUTPUT]);
        * the error associated with the output (variable[LEARNING_ERROR_OUTPUT]).

        However, the exact specification depends on the funtion's type.

    default_learning_rate : numeric
        the value used for the function's `learning_rate <LearningFunction.learning_rate>` parameter if none of the
        following are specified:  the `learning_rate <LearningMechanism.learning_rate>` for the `LearningMechanism` to
        which the function has been assigned, the `learning_rate <Process.learning_rate>` for any `Process` or
        the `learning_rate <System.learning_rate>` for any `System` to which that LearningMechanism belongs.
        The exact form of the value (i.e., whether it is a scalar or array) depends on the function's type.

    learning_rate : numeric
        generally used to multiply the result of the function before it is returned;  however, both the form of the
        value (i.e., whether it is a scalar or array) and how it is used depend on the function's type.

    Returns
    -------

    The number of items returned and their format depend on the function's type.

    Most return an array (used as the `learning_signal <LearningMechanism.learning_signal>` by a \
    `LearningMechanism`), and some also return a similarly formatted array containing either the \
    error received (in the third item of the `variable <LearningFunction.variable>`) or a \
    version of it modified by the function.

    """

    componentType = LEARNING_FUNCTION_TYPE

    class ClassDefaults(Function_Base.ClassDefaults):
        variable = np.array([0, 0, 0])

    # def __init__(self, default_variable, params, owner, prefs, context):
    #     super().__init__(default_variable=default_variable,
    #                      params=params,
    #                      owner=owner,
    #                      prefs=prefs,
    #                      context=context)

    #     self.learning_rate_dim = None
    #     if learning_rate is not None:
    #         self.learning_rate_dim = np.array(learning_rate).ndim

    #     self.return_val = return_val(None, None)


    def _validate_learning_rate(self, learning_rate, type=None):

        learning_rate = np.array(learning_rate).copy()
        learning_rate_dim = learning_rate.ndim

        self._validate_parameter_spec(learning_rate, LEARNING_RATE)

        if type is AUTOASSOCIATIVE:

            if learning_rate_dim == 1 and len(learning_rate) != len(self.instance_defaults.variable):
                raise FunctionError("Length of {} arg for {} ({}) must be the same as its variable ({})".
                                    format(LEARNING_RATE, self.name, len(learning_rate), len(self.instance_defaults.variable)))

            if learning_rate_dim == 2:
                shape = learning_rate.shape
                if shape[0] != shape[1] or shape[0] != len(self.instance_defaults.variable):
                    raise FunctionError("Shape of {} arg for {} ({}) must be square and "
                                        "of the same width as the length of its variable ({})".
                                        format(LEARNING_RATE, self.name, shape, len(self.instance_defaults.variable)))

            if learning_rate_dim > 2:
                raise FunctionError("{} arg for {} ({}) must be a single value of a 1d or 2d array".
                                    format(LEARNING_RATE, self.name, learning_rate))

        else:
            if learning_rate_dim:
                raise FunctionError("{} arg for {} ({}) must be a single value".
                                    format(LEARNING_RATE, self.name, learning_rate))

class Hebbian(LearningFunction):  # -------------------------------------------------------------------------------
    """
    Hebbian(                    \
        default_variable=None,  \
        learning_rate=None,     \
        params=None,            \
        name=None,              \
        prefs=None)

    Implements a function that calculates a matrix of weight changes using the Hebbian (correlational) learning rule.

    Arguments
    ---------

    variable : List[number] or 1d np.array : default ClassDefaults.variable
       specifies the activation values, the pair-wise products of which are used to generate the a weight change matrix.

    COMMENT:
    activation_function : Function or function : SoftMax
        specifies the `function <Mechanism_Base.function>` of the `Mechanism` that generated the array of activations
        in `variable <Hebbian.variable>`.
    COMMENT

    learning_rate : scalar or list, 1d or 2d np.array, or np.matrix of numeric values: default default_learning_rate
        specifies the learning rate used by the `function <Hebbian.function>`; supersedes any specification  for the
        `Process` and/or `System` to which the function's `owner <Function.owner>` belongs (see `learning_rate
        <Hebbian.learning_rate>` for details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the function.
        Values specified for parameters in the dictionary override any assigned to those parameters in arguments
        of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).
    Attributes
    ----------

    variable: 1d np.array
        activation values, the pair-wise products of which are used to generate the weight change matrix returned by
        the `function <Hebbian.function>`.

    COMMENT:
    activation_function : Function or function : SoftMax
        the `function <Mechanism_Base.function>` of the `Mechanism` that generated the array of activations in
        `variable <Hebbian.variable>`.
    COMMENT

    learning_rate : float, 1d or 2d np.array
        used by the `function <Hebbian.function>` to scale the weight change matrix returned by the `function
        <Hebbian.function>`.  If specified, it supersedes any learning_rate specified for the `Process
        <Process_Base_Learning>` and/or `System <System_Learning>` to which the function's `owner <Hebbian.owner>`
        belongs.  If it is a scalar, it is multiplied by the weight change matrix;  if it is a 1d np.array, it is
        multiplied Hadamard (elementwise) by the `variable` <Hebbian.variable>` before calculating the weight change
        matrix;  if it is a 2d np.array, it is multiplied Hadamard (elementwise) by the weight change matrix; if it is
        `None`, then the `learning_rate <Process.learning_rate>` specified for the Process to which the `owner
        <Hebbian.owner>` belongs is used;  and, if that is `None`, then the `learning_rate <System.learning_rate>`
        for the System to which it belongs is used. If all are `None`, then the `default_learning_rate
        <Hebbian.default_learning_rate>` is used.

    default_learning_rate : float
        the value used for the `learning_rate <Hebbian.learning_rate>` if it is not otherwise specified.

    function : function
         calculates the pairwise product of all elements in the `variable <Hebbian.variable>`, and then
         scales that by the `learning_rate <Hebbian.learning_rate>` to generate the weight change matrix
         returned by the function.

    owner : Component
        `Mechanism <Mechanism>` to which the Function belongs.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).    """

    componentName = HEBBIAN_FUNCTION

    class ClassDefaults(LearningFunction.ClassDefaults):
        variable = np.array([0, 0])

    default_learning_rate = 0.05

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    def __init__(self,
                 default_variable=None,
                 # activation_function: tc.any(Linear, tc.enum(Linear)) = Linear,  # Allow class or instance
                 # learning_rate: tc.optional(parameter_spec) = None,
                 learning_rate=None,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(
                # activation_function=activation_function,
                learning_rate=learning_rate,
                params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

        self.functionOutputType = None

    def _validate_variable(self, variable, context=None):
        variable = self._update_variable(super()._validate_variable(variable, context))

        variable = np.squeeze(np.array(variable))

        if not is_numeric(variable):
            raise ComponentError("Variable for {} ({}) contains non-numeric entries".
                                 format(self.name, variable))
        if variable.ndim == 0:
            raise ComponentError("Variable for {} is a single number ({}) "
                                 "which doesn't make much sense for associative learning".
                                 format(self.name, variable))
        if variable.ndim > 1:
            raise ComponentError("Variable for {} ({}) must be a list or 1d np.array of numbers".
                                 format(self.name, variable))
        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate learning_rate
        """
        super()._validate_params(request_set=request_set, target_set=target_set, context=context)
        if LEARNING_RATE in target_set and target_set[LEARNING_RATE] is not None:
            self._validate_learning_rate(target_set[LEARNING_RATE], AUTOASSOCIATIVE)

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        """Calculate a matrix of weight changes from a 1d array of activity values using Hebbian learning function.

        The weight change matrix is calculated as:

           *learning_rate* * :math:`a_ia_j` if :math:`i \\neq j`, else :math:`0`

        where :math:`a_i` and :math:`a_j` are elements of `variable <Hebbian.variable>`.

        Arguments
        ---------

        variable : List[number] or 1d np.array : default ClassDefaults.variable
            array of activity values, the pairwise products of which are used to generate a weight change matrix.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the function.
            Values specified for parameters in the dictionary override any assigned to those parameters in arguments
            of the constructor.

        Returns
        -------

        weight change matrix : 2d np.array
            matrix of pairwise products of elements of `variable <Hebbian.variable>` scaled by the `learning_rate
            <HebbinaMechanism.learning_rate>`, with all diagonal elements = 0 (i.e., hollow matix).

        """

        self._check_args(variable=variable, params=params, context=context)

        # IMPLEMENTATION NOTE: have to do this here, rather than in validate_params for the following reasons:
        #                      1) if no learning_rate is specified for the Mechanism, need to assign None
        #                          so that the process or system can see it is free to be assigned
        #                      2) if neither the system nor the process assigns a value to the learning_rate,
        #                          then need to assign it to the default value
        # If learning_rate was not specified for instance or composition, use default value
        if self.learning_rate is None:
            learning_rate = self.default_learning_rate
        else:
            learning_rate = self.learning_rate

        # FIX: SHOULD PUT THIS ON SUPER (THERE, BUT NEEDS TO BE DEBUGGED)
        self.learning_rate_dim = None
        if learning_rate is not None:
            self.learning_rate_dim = np.array(learning_rate).ndim

        # MODIFIED 9/21/17 NEW:
        # FIX: SHOULDN'T BE NECESSARY TO DO THIS;  WHY IS IT GETTING A 2D ARRAY AT THIS POINT?
        if not isinstance(variable, np.ndarray):
            variable = np.array(variable)
        if variable.ndim > 1:
            variable = np.squeeze(variable)
        # MODIFIED 9/21/17 END

        # If learning_rate is a 1d array, multiply it by variable
        if self.learning_rate_dim == 1:
            variable = variable * learning_rate

        # Generate the column array from the variable
        # col = variable.reshape(len(variable),1)
        col = np.array(np.matrix(variable).T)

        # Calculate weight chhange matrix
        weight_change_matrix = variable * col
        # Zero diagonals (i.e., don't allow correlation of a unit with itself to be included)
        weight_change_matrix = weight_change_matrix * (1-np.identity(len(variable)))

        # If learning_rate is scalar or 2d, multiply it by the weight change matrix
        if self.learning_rate_dim in {0, 2}:
            weight_change_matrix = weight_change_matrix * learning_rate

        return weight_change_matrix


class Reinforcement(LearningFunction):  # -------------------------------------------------------------------------------
    """
    Reinforcement(                     \
        default_variable=None,         \
        learning_rate=None,            \
        params=None,                   \
        name=None,                     \
        prefs=None)

    Implements a function that returns an error term for a single item in an input array, scaled by the learning_rate.

    Reinforcement takes an array with a single non-zero value (`activation_output <Reinforcement.activation_output>`),
    and returns an array of the same length with the single non-zero value replaced by the `error_signal
    <Reinforcement.error_signal>` scaled by the `learning_rate <Reinforcement.learning_rate>`.
    The non-zero item in `activation_output <Reinforcement.activation_output>` can be thought of as the predicted
    likelihood of a stimulus or value of an action, and the `error_signal <Reinforcement.error_signal>` as the error in
    the prediction for that value.

    .. note::
       To preserve compatibility with other LearningFunctions:

       * the **variable** argument of both the constructor and calls to the Reinforcement `function
         <Reinforcement.function>` must have three items, although only the 2nd and 3rd items are used
         (for the `activation_output <Reinforcement.activation_output>` and `error_signal
         <Reinforcement.error_signal>` attributes, respectively);
       ..
       * the Reinforcement `function <Reinforcement.function>` returns two copies of the error array
         (the first is a "place-marker", where a matrix of weights changes is often returned).

    Arguments
    ---------

    variable : List or 2d np.array [length 3 in axis 0] : default ClassDefaults.variable
       template for the three items provided as the variable in the call to the `function <Reinforcement.function>`
       (in order):

           * `activation_input <Reinforcement.activation_input>` (1d np.array);

           * `activation_output <Reinforcement.activation_output>` (1d np.array with a single non-zero value);

           * `error_signal <Reinforcement.error_signal>`  (1d np.array with a single value).

    COMMENT:
    activation_function : Function or function : SoftMax
        specifies the function of the Mechanism that generates `activation_output <Reinforcement.activation_output>`.
    COMMENT

    learning_rate : float : default default_learning_rate
        supersedes any specification for the `Process` and/or `System` to which the function's
        `owner <Function.owner>` belongs (see `learning_rate <Reinforcement.learning_rate>` for details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable: 2d np.array
        specifies three values used as input to the `function <Reinforcement.function>`:

            * `activation_input <Reinforcement.activation_input>`,

            * `activation_output <Reinforcement.activation_output>`, and

            * `error_signal <Reinforcement.error_signal>`.

    activation_input : 1d np.array
        first item of `variable <Reinforcement.variable>`;  this is not used (it is implemented for compatibility
        with other `LearningFunctions <LearningFunction>`).

    activation_output : 1d np.array
        an array containing a single "prediction" or "action" value as one of its elements, the remainder of which
        are zero.

    error_signal : 1d np.array
        contains a single item, specifying the error associated with the non-zero item in `activation_output
        <Reinforcement.activation_output>`.

    COMMENT:
    activation_function : Function or function : SoftMax
        the function of the Mechanism that generates `activation_output <Reinforcement.activation_output>`; must
        return an array with a single non-zero value.
    COMMENT

    learning_rate : float
        the learning rate used by the function.  If specified, it supersedes any learning_rate specified for the
        `Process <Process_Base_Learning>` and/or `System <System_Learning>` to which the function's
        `owner <Reinforcement.owner>` belongs.  If it is `None`, then the `learning_rate <Process.learning_rate>`
        specified for the Process to which the `owner <Reinforcement.owner>` belongs is used;  and, if that is `None`,
        then the `learning_rate <System.learning_rate>` for the System to which it belongs is used. If all are
        `None`, then the `default_learning_rate <Reinforcement.default_learning_rate>` is used.

    default_learning_rate : float
        the value used for the `learning_rate <Reinforcement.learning_rate>` if it is not otherwise specified.

    function : function
         the function that computes the weight change matrix, and returns that along with the
         `error_signal <Reinforcement.error_signal>` received.

    owner : Component
        `Mechanism <Mechanism>` to which the Function belongs.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).    """

    componentName = RL_FUNCTION

    class ClassDefaults(LearningFunction.ClassDefaults):
        variable = np.array([[0], [0], [0]])

    default_learning_rate = 0.05

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    def __init__(self,
                 default_variable=None,
                 # activation_function: tc.any(SoftMax, tc.enum(SoftMax)) = SoftMax,  # Allow class or instance
                 # learning_rate: tc.optional(parameter_spec) = None,
                 learning_rate=None,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(# activation_function=activation_function,
                                                  learning_rate=learning_rate,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

        self.functionOutputType = None

    def _validate_variable(self, variable, context=None):
        variable = self._update_variable(super()._validate_variable(variable, context))

        if len(variable) != 3:
            raise ComponentError("Variable for {} ({}) must have three items (input, output and error arrays)".
                                 format(self.name, variable))

        # TODO: stateful - should these be stateful?
        self.activation_input = variable[LEARNING_ACTIVATION_INPUT]
        self.activation_output = variable[LEARNING_ACTIVATION_OUTPUT]
        self.error_signal = variable[LEARNING_ERROR_OUTPUT]

        if len(self.error_signal) != 1:
            raise ComponentError("Error term for {} (the third item of its variable arg) must be an array with a "
                                 "single element for {}".
                                 format(self.name, self.error_signal))

        # Allow initialization with zero but not during a run (i.e., when called from check_args())
        if self.context.initialization_status != ContextFlags.INITIALIZING:
            if np.count_nonzero(self.activation_output) != 1:
                raise ComponentError("Second item ({}) of variable for {} must be an array with a single non-zero value "
                                     "(if output Mechanism being trained uses softmax,"
                                     " its \'output\' arg may need to be set to to PROB)".
                                     format(variable[LEARNING_ACTIVATION_OUTPUT], self.componentName))

        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate learning_rate
        """
        super()._validate_params(request_set=request_set, target_set=target_set, context=context)
        if LEARNING_RATE in target_set and target_set[LEARNING_RATE] is not None:
            self._validate_learning_rate(target_set[LEARNING_RATE], AUTOASSOCIATIVE)

    def function(self,
                 variable=None,
                 params=None,
                 context=None,
                 **kwargs):
        """Return an error array for the specified item of activation_output scaled by the learning_rate.

        Returns a 1d error array with a single non-zero value in the same position as the non-zero item
        in `activation_output <Reinforcement.activation_output>` (2nd item of the **variable** argument),
        that is the `error_signal <Reinforcement.error_signal>` (3rd item of
        **variable** argument) scaled by the `learning_rate <Reinforement.learning_rate>`.

        .. note::
           In order to preserve compatibilty with other `LearningFunctions <LearningFunction>`:

           * **variable** must have three items, although only the 2nd and 3rd are used;
           ..
           * `function <Reinforcement.function>` returns two copies of the error array.

        Arguments
        ---------

        variable : List or 2d np.array [length 3 in axis 0] : default ClassDefaults.variable
           must have three items that are the values for (in order):

               * `activation_input <Reinforcement.activation_input>` (not used),

               * `activation_output <Reinforcement.activation_output>` (1d np.array with a single non-zero value),

               * `error_signal <Reinforcement.error_signal>` (1d np.array with a single item).

        params : Dict[param keyword: param value] : default None
           a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
           function.  Values specified for parameters in the dictionary override any assigned to those parameters in
           arguments of the constructor.

        Returns
        -------

        error array : List[1d np.array, 1d np.array]
            Two copies of a 1d array with a single non-zero error term.

        """

        self._check_args(variable=variable, params=params, context=context)

        output = self.activation_output
        error = self.error_signal
        learning_rate = self.get_current_function_param(LEARNING_RATE)
        # IMPLEMENTATION NOTE: have to do this here, rather than in validate_params for the following reasons:
        #                      1) if no learning_rate is specified for the Mechanism, need to assign None
        #                          so that the process or system can see it is free to be assigned
        #                      2) if neither the system nor the process assigns a value to the learning_rate,
        #                          then need to assign it to the default value
        # If learning_rate was not specified for instance or composition, use default value
        if learning_rate is None:
            learning_rate = self.default_learning_rate

        # Assign error term to chosen item of output array
        error_array = (np.where(output, learning_rate * error, 0))

        # Construct weight change matrix with error term in proper element
        weight_change_matrix = np.diag(error_array)

        return [error_array, error_array]


# Argument names:
ERROR_MATRIX = 'error_matrix'
WT_MATRIX_SENDERS_DIM = 0
WT_MATRIX_RECEIVERS_DIM = 1


class BackPropagation(LearningFunction):
    """
    BackPropagation(                                     \
        default_variable=None,                           \
        activation_derivative_fct=Logistic().derivative, \
        learning_rate=None,                              \
        params=None,                                     \
        name=None,                                       \
        prefs=None)

    Implements a `function <BackPropagation.function>` that calculate a matrix of weight changes using the
    `backpropagation <https://en.wikipedia.org/wiki/Backpropagation>`_
     (`Generalized Delta Rule <http://www.nature.com/nature/journal/v323/n6088/abs/323533a0.html>`_)
    learning algorithm.  The weight change matrix is computed as:

        *weight_change_matrix* = `learning_rate <BackPropagation.learning_rate>` * `activation_input
        <BackPropagation.activation_input>` * :math:`\\frac{\delta E}{\delta W}`

            where:

               :math:`\\frac{\delta E}{\delta W}` = :math:`\\frac{\delta E}{\delta A} * \\frac{\delta A}{\delta W}`

                 is the derivative of the `error_signal <BackPropagation.error_signal>` with respect to the weights;

               :math:`\\frac{\delta E}{\delta A}` = `error_matrix <BackPropagation.error_matrix>` :math:`\\cdot`
               `error_signal <BackPropagation.error_signal>`

                 is the derivative of the error with respect to `activation_output
                 <BackPropagation.activation_output>` (i.e., the weighted contribution to the `error_signal
                 <BackPropagation.error_signal>` of each unit that receives activity from the weight matrix being
                 learned); and

               :math:`\\frac{\delta A}{\delta W}` =
               `activation_derivative_fct <BackPropagation.activation_derivative_fct>`
               (*input =* `activation_input <BackPropagation.activation_input>`,
               *output =* `activation_output <BackPropagation.activation_output>`\\)

                 is the derivative of the activation function responsible for generating `activation_output
                 <BackPropagation.activation_output>` at the point that generates each of its entries.

    The values of `activation_input <BackPropagation.activation_input>`, `activation_output
    <BackPropagation.activation_output>` and  `error_signal <BackPropagation.error_signal>` are specified as
    items of the `variable <BackPropgation.variable>` both in the constructor for the BackPropagation Function,
    and in calls to its `function <BackPropagation.function>`.  Although `error_matrix <BackPropagation.error_matrix>`
    is not specified in the constructor, it is required as an argument of the `function <BackPropagation.function>`;
    it is assumed that it's value is determined in context at the time of execution (e.g., by a LearningMechanism that
    uses the BackPropagation LearningFunction).

    The BackPropagation `function <BackPropagation.function>` returns the *weight_change_matrix* as well as
    :math:`\\frac{\delta E}{\delta W}`.

    Arguments
    ---------

    variable : List or 2d np.array [length 3 in axis 0] : default ClassDefaults.variable
       specifies a template for the three items provided as the variable in the call to the
       `function <BackPropagation.function>` (in order):
       `activation_input <BackPropagation.activation_input>` (1d np.array),
       `activation_output <BackPropagation.activation_output>` (1d np.array),
       `error_signal <BackPropagation.error_signal>` (1d np.array).

    activation_derivative_fct : Function or function
        specifies the derivative for the function of the Mechanism that generates
        `activation_output <BackPropagation.activation_output>`.

    COMMENT:
    error_derivative : Function or function
        specifies the derivative for the function of the Mechanism that is the receiver of the
        `error_matrix <BackPropagation.error_matrix>`.
    COMMENT

    COMMENT:
    error_matrix : List, 2d np.array, np.matrix, ParameterState, or MappingProjection
        matrix, the output of which is used to calculate the `error_signal <BackPropagation.error_signal>`.
        If it is specified as a ParameterState it must be one for the `matrix <MappingProjection.matrix>`
        parameter of a `MappingProjection`;  if it is a MappingProjection, it must be one with a
        MATRIX parameterState.
    COMMENT

    learning_rate : float : default default_learning_rate
        supersedes any specification for the `Process` and/or `System` to which the function's
        `owner <Function.owner>` belongs (see `learning_rate <BackPropagation.learning_rate>` for details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable: 2d np.array
        contains the three values used as input to the `function <BackPropagation.function>`:
       `activation_input <BackPropagation.activation_input>`,
       `activation_output <BackPropagation.activation_output>`, and
       `error_signal <BackPropagation.error_signal>`.

    activation_input : 1d np.array
        the input to the matrix being modified; same as 1st item of `variable <BackPropagation.variable>`.

    activation_output : 1d np.array
        the output of the function for which the matrix being modified provides the input;
        same as 2nd item of `variable <BackPropagation.variable>`.

    activation_derivative_fct : Function or function
        the derivative for the function of the Mechanism that generates
        `activation_output <BackPropagation.activation_output>`.

    error_signal : 1d np.array
        the error signal for the next matrix (layer above) in the learning sequence, or the error computed from the
        target (training signal) and the output of the last Mechanism in the sequence;
        same as 3rd item of `variable <BackPropagation.variable>`.

    error_matrix : 2d np.array or ParameterState
        matrix, the input of which is `activation_output <BackPropagation.activation_output>` and the output of which
        is used to calculate the `error_signal <BackPropagation.error_signal>`; if it is a `ParameterState`,
        it refers to the MATRIX parameterState of the `MappingProjection` being learned.

    learning_rate : float
        the learning rate used by the function.  If specified, it supersedes any learning_rate specified for the
        `process <Process.learning_Rate>` and/or `system <System.learning_rate>` to which the function's  `owner
        <BackPropagation.owner>` belongs.  If it is `None`, then the learning_rate specified for the process to
        which the `owner <BackPropagation.owner>` belongs is used;  and, if that is `None`, then the learning_rate for
        the system to which it belongs is used. If all are `None`, then the
        `default_learning_rate <BackPropagation.default_learning_rate>` is used.

    default_learning_rate : float
        the value used for the `learning_rate <BackPropagation.learning_rate>` if it is not otherwise specified.

    function : function
         the function that computes the weight change matrix, and returns that along with the
         `error_signal <BackPropagation.error_signal>` received, weighted by the contribution made by each element of
         `activation_output <BackPropagation.activation_output>` as a function of the
         `error_matrix <BackPropagation.error_matrix>`.

    owner : Component
        `Mechanism <Mechanism>` to which the Function belongs.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).
    """

    componentName = BACKPROPAGATION_FUNCTION

    class ClassDefaults(LearningFunction.ClassDefaults):
        variable = np.array([[0], [0], [0]])

    default_learning_rate = 1.0

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 # default_variable:tc.any(list, np.ndarray),
                 activation_derivative_fct: tc.optional(tc.any(function_type, method_type)) = Logistic().derivative,
                 # learning_rate: tc.optional(parameter_spec) = None,
                 learning_rate=None,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):

        error_matrix=np.zeros((len(default_variable[LEARNING_ACTIVATION_OUTPUT]),
                               len(default_variable[LEARNING_ERROR_OUTPUT])))

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(activation_derivative_fct=activation_derivative_fct,
                                                  error_matrix=error_matrix,
                                                  learning_rate=learning_rate,
                                                  params=params)

        # self.return_val = ReturnVal(None, None)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

        self.functionOutputType = None

    def _validate_variable(self, variable, context=None):
        variable = self._update_variable(super()._validate_variable(variable, context))

        if len(variable) != 3:
            raise ComponentError("Variable for {} ({}) must have three items: "
                                 "activation_input, activation_output, and error_signal)".
                                 format(self.name, variable))

        # TODO: stateful - should these be stateful?
        self.activation_input = variable[LEARNING_ACTIVATION_INPUT]
        self.activation_output = variable[LEARNING_ACTIVATION_OUTPUT]
        self.error_signal = variable[LEARNING_ERROR_OUTPUT]

        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate learning_rate and error_matrix params

        `error_matrix` argument must be one of the following
            - 2d list, np.ndarray or np.matrix
            - ParameterState for one of the above
            - MappingProjection with a parameterStates[MATRIX] for one of the above

        Parse error_matrix specification and insure it is compatible with error_signal and activation_output

        Insure that the length of the error_signal matches the number of cols (receiver elements) of error_matrix
            (since it will be dot-producted to generate the weighted error signal)

        Insure that length of activation_output matches the number of rows (sender elements) of error_matrix
           (since it will be compared against the *result* of the dot product of the error_matrix and error_signal

        Note: error_matrix is left in the form in which it was specified so that, if it is a ParameterState
              or MappingProjection, its current value can be accessed at runtime (i.e., it can be used as a "pointer")
        """

        # # MODIFIED 3/22/17 OLD:
        # # This allows callers to specify None as learning_rate (e.g., _instantiate_learning_components)
        # if request_set[LEARNING_RATE] is None:
        #     request_set[LEARNING_RATE] = 1.0
        # # request_set[LEARNING_RATE] = request_set[LEARNING_RATE] or 1.0
        # # MODIFIED 3/22/17 END

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        if LEARNING_RATE in target_set and target_set[LEARNING_RATE] is not None:
            self._validate_learning_rate(target_set[LEARNING_RATE], AUTOASSOCIATIVE)

        # Validate error_matrix specification
        if ERROR_MATRIX in target_set:

            error_matrix = target_set[ERROR_MATRIX]

            from psyneulink.components.states.parameterstate import ParameterState
            from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
            if not isinstance(error_matrix, (list, np.ndarray, np.matrix, ParameterState, MappingProjection)):
                raise FunctionError("The {} arg for {} ({}) must be a list, 2d np.array, ParamaterState or "
                                    "MappingProjection".format(ERROR_MATRIX, self.__class__.__name__, error_matrix))

            if isinstance(error_matrix, MappingProjection):
                try:
                    error_matrix = error_matrix._parameter_states[MATRIX].value
                    param_type_string = "MappingProjection's ParameterState"
                except KeyError:
                    raise FunctionError("The MappingProjection specified for the {} arg of {} ({}) must have a {} "
                                        "paramaterState that has been assigned a 2d array or matrix".
                                        format(ERROR_MATRIX, self.__class__.__name__, error_matrix.shape, MATRIX))

            elif isinstance(error_matrix, ParameterState):
                try:
                    error_matrix = error_matrix.value
                    param_type_string = "ParameterState"
                except KeyError:
                    raise FunctionError("The value of the {} parameterState specified for the {} arg of {} ({}) "
                                        "must be a 2d array or matrix".
                                        format(MATRIX, ERROR_MATRIX, self.__class__.__name__, error_matrix.shape))

            else:
                param_type_string = "array or matrix"

            error_matrix = np.array(error_matrix)
            rows = error_matrix.shape[WT_MATRIX_SENDERS_DIM]
            cols = error_matrix.shape[WT_MATRIX_RECEIVERS_DIM]
            activity_output_len = len(self.activation_output)
            error_signal_len = len(self.error_signal)

            if error_matrix.ndim != 2:
                raise FunctionError("The value of the {} specified for the {} arg of {} ({}) "
                                    "must be a 2d array or matrix".
                                    format(param_type_string, ERROR_MATRIX, self.name, error_matrix))

            # The length of the sender outputState.value (the error signal) must be the
            #     same as the width (# columns) of the MappingProjection's weight matrix (# of receivers)

            # Validate that columns (number of receiver elements) of error_matrix equals length of error_signal
            if cols != error_signal_len:
                raise FunctionError("The width (number of columns, {}) of the \'{}\' arg ({}) specified for {} "
                                    "must match the length of the error signal ({}) it receives".
                                    format(cols, MATRIX, error_matrix.shape, self.name, error_signal_len))

            # Validate that rows (number of sender elements) of error_matrix equals length of activity_output,
            if rows != activity_output_len:
                raise FunctionError("The height (number of rows, {}) of \'{}\' arg specified for {} must match the "
                                    "length of the output {} of the activity vector being monitored ({})".
                                    format(rows, MATRIX, self.name, activity_output_len))

    def function(self,
                 variable=None,
                 error_matrix=None,
                 params=None,
                 context=None,
                 **kwargs):
        """Calculate and return a matrix of weight changes from arrays of inputs, outputs and error terms.

        Note that both variable and error_matrix must be specified for the function to execute.

        Arguments
        ---------

        variable : List or 2d np.array [length 3 in axis 0]
           must have three items that are the values for (in order):
           `activation_input <BackPropagation.activation_input>` (1d np.array),
           `activation_output <BackPropagation.activation_output>` (1d np.array),
           `error_signal <BackPropagation.error_signal>` (1d np.array).

        error_matrix : List, 2d np.array, np.matrix, ParameterState, or MappingProjection
            matrix of weights that were used to generate the `error_signal <BackPropagation.error_signal>` (3rd item
            of `variable <BackPropagation.variable>` from `activation_output <BackPropagation.activation_output>`;
            its dimensions must be the length of `activation_output <BackPropagation.activation_output>` (rows) x
            length of `error_signal <BackPropagation.error_signal>` (cols).

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        weight change matrix : 2d np.array
            the modifications to make to the matrix.

        weighted error signal : 1d np.array
            `error_signal <BackPropagation.error_signal>`, weighted by the contribution made by each element of
            `activation_output <BackPropagation.activation_output>` as a function of
            `error_matrix <BackPropagation.error_matrix>`.

        """

        self._check_args(variable=variable, params=params, context=context)

        # Manage error_matrix param
        # During init, function is called directly from Component (i.e., not from LearningMechanism execute() method),
        #     so need "placemarker" error_matrix for validation
        if self.context.initialization_status == ContextFlags.INITIALIZING and error_matrix is None:
            self.error_matrix = np.zeros((len(variable[LEARNING_ACTIVATION_OUTPUT]),
                                          len(variable[LEARNING_ERROR_OUTPUT])))
        # If error_matrix is specified, assign to self.error_matrix attribute for validation
        elif error_matrix is not None:
            from psyneulink.components.states.parameterstate import ParameterState
            # If it is specified as a ParameterState, get actual matrix (for execution below)
            if isinstance(error_matrix, ParameterState):
                self.error_matrix = params[ERROR_MATRIX].value
            else:
                self.error_matrix = error_matrix
        # Raise exception if error_matrix is not specified
        else:
            owner_string = ""
            if self.owner:
                owner_string = " of " + self.owner.name
            raise FunctionError("Call to {} function{} must include \'ERROR_MATRIX\' in params arg".
                                format(self.__class__.__name__, owner_string))

        # self._check_args(variable=variable, params=params, context=context)

        # Manage learning_rate
        # IMPLEMENTATION NOTE: have to do this here, rather than in validate_params for the following reasons:
        #                      1) if no learning_rate is specified for the Mechanism, need to assign None
        #                          so that the process or system can see it is free to be assigned
        #                      2) if neither the system nor the process assigns a value to the learning_rate,
        #                          then need to assign it to the default value
        # If learning_rate was not specified for instance or composition, use default value
        if self.learning_rate is None:
            learning_rate = self.default_learning_rate
        else:
            learning_rate = self.learning_rate

        # make activation_input a 1D row array
        activation_input = np.array(self.activation_input).reshape(len(self.activation_input), 1)

        # Derivative of error with respect to output activity (contribution of each output unit to the error above)
        dE_dA = np.dot(self.error_matrix, self.error_signal)

        # Derivative of the output activity
        dA_dW = self.activation_derivative_fct(input=self.activation_input, output=self.activation_output)

        # Chain rule to get the derivative of the error with respect to the weights
        dE_dW = dE_dA * dA_dW

        # Weight changes = delta rule (learning rate * activity * error)
        weight_change_matrix = learning_rate * activation_input * dE_dW

        return [weight_change_matrix, dE_dW]


class TDLearning(Reinforcement):
    """
    This class is used to implement temporal difference learning via the
    `Reinforcement` function. See `Reinforcement` for class details.
    """
    componentName = TDLEARNING_FUNCTION

    class ClassDefaults(Reinforcement.ClassDefaults):
        variable = [[0], [0]]

    def __init__(self,
                 default_variable=Reinforcement.ClassDefaults.variable,
                 # activation_function: tc.any(SoftMax, tc.enum(SoftMax))=SoftMax,
                 learning_rate=Reinforcement.default_learning_rate,
                 params=None,
                 owner=None,
                 prefs=None):
        """
        Dummy function used to implement TD Learning via Reinforcement Learning

        Parameters
        ----------
        default_variable
        learning_rate: float: default 0.05
        params
        owner
        prefs
        context
        """
        # params = self._assign_args_to_param_dicts(learning_rate=learning_rate,
                                                  # params=params)
        super().__init__(default_variable=default_variable,
                         # activation_function=activation_function,
                         learning_rate=learning_rate,
                         params=params,
                         owner=owner,
                         prefs=prefs)

    def _validate_variable(self, variable, context=None):
        variable = self._update_variable(super(Reinforcement, self)._validate_variable(variable, context))

        if len(variable) != 3:
            raise ComponentError("Variable for {} ({}) must have three items "
                                 "(input, output, and error arrays)".format(self.name,
                                                                            variable))

        self.activation_input = variable[LEARNING_ACTIVATION_INPUT]
        self.activation_output = variable[LEARNING_ACTIVATION_OUTPUT]
        self.error_signal = variable[LEARNING_ERROR_OUTPUT]

        if len(self.error_signal) != len(self.activation_output):
            raise ComponentError("Error term does not match the length of the"
                                 "sample sequence")

        return variable

    def function(self, variable=None, params=None, context=None, **kwargs):
        return super().function(variable=variable, params=params, context=context)


# FIX: IMPLEMENT AS Functions
def max_vs_next(x):
    x_part = np.partition(x, -2)
    max_val = x_part[-1]
    next = x_part[-2]
    return max_val - next


def max_vs_avg(x):
    x_part = np.partition(x, -2)
    max_val = x_part[-1]
    others = x_part[:-1]
    return max_val - np.mean(others)

#endregion
