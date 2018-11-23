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
Function
  * `Function_Base`

Example function:
  * `ArgumentTherapy`

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

Optimization Functions:
  * `OptimizationFunction`
  * `GradientOptimization`
  * `GridSearch`

Learning Functions:
  * `Kohonen`
  * `Hebbian`
  * `ContrastiveHebbian`
  * `Reinforcement`
  * `BayesGLM`
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

*Core Attributes*
~~~~~~~~~~~~~~~~~

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

*Owner*
~~~~~~~

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
`output_type` attribute, by assigning it one of the following `FunctionOutputType` values:
    * FunctionOutputType.RAW_NUMBER: return "exposed" number;
    * FunctionOutputType.NP_1D_ARRAY: return 1d np.array
    * FunctionOutputType.NP_2D_ARRAY: return 2d np.array.

To implement FunctionOutputTypeConversion, the Function's FUNCTION_OUTPUT_TYPE_CONVERSION parameter must set to True,
and function type conversion must be implemented by its `function <Function_Base.function>` method
(see `Linear` for an example).
COMMENT

.. _Function_Modulatory_Params:

*Modulatory Parameters*
~~~~~~~~~~~~~~~~~~~~~~~

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
call to the `function <Function_Base.function>`.

For `Mechanisms <Mechanism>`, this can also be done by specifying `runtime_params <Run_Runtime_Parameters>` in the `Run`
method of their `Composition`.

Class Reference
---------------

"""

import functools
import itertools
import numbers
import numpy as np
import typecheck as tc
import warnings

from collections import namedtuple
from enum import Enum, IntEnum
from llvmlite import ir
from random import randint

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.component import ComponentError, function_type, method_type
from psyneulink.core.components.functions.transferfunctions import Logistic, get_matrix
from psyneulink.core.components.shellclasses import Function, Mechanism
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.defaults import MPI_IMPLEMENTATION
from psyneulink.core.globals.keywords import ARGUMENT_THERAPY_FUNCTION, BACKPROPAGATION_FUNCTION, BETA, \
    CONTRASTIVE_HEBBIAN_FUNCTION, CORRELATION, CROSS_ENTROPY, DEFAULT_VARIABLE, DIFFERENCE, DISTANCE_FUNCTION, DISTANCE_METRICS, DIST_FUNCTION_TYPE, DIST_MEAN, DIST_SHAPE, \
    DistanceMetrics, ENERGY, ENTROPY, EUCLIDEAN, EXAMPLE_FUNCTION_TYPE, EXPONENTIAL, EXPONENTIAL_DIST_FUNCTION, \
    FUNCTION, FUNCTION_OUTPUT_TYPE, FUNCTION_OUTPUT_TYPE_CONVERSION, GAMMA_DIST_FUNCTION, GAUSSIAN, \
    GRADIENT_OPTIMIZATION_FUNCTION, GRID_SEARCH_FUNCTION, HEBBIAN_FUNCTION, HIGH, HOLLOW_MATRIX, \
    KOHONEN_FUNCTION, LEARNING_FUNCTION_TYPE, LEARNING_RATE, LINEAR, \
    LOW, MATRIX, MAX_ABS_DIFF, NAME, NORMAL_DIST_FUNCTION, OBJECTIVE_FUNCTION_TYPE, OPTIMIZATION_FUNCTION_TYPE, OWNER, PARAMETER_STATE_PARAMS, \
    RL_FUNCTION, SCALE, STABILITY_FUNCTION, STANDARD_DEVIATION, \
    TDLEARNING_FUNCTION, UNIFORM_DIST_FUNCTION, VALUE, VARIABLE, \
    WALD_DIST_FUNCTION, \
    kwComponentCategory, kwPreferenceSetName
from psyneulink.core.globals.parameters import Param
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set, kpReportOutputPref
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel
from psyneulink.core.globals.registry import register_category
from psyneulink.core.globals.utilities import call_with_pruned_args, is_distance_metric, is_iterable, is_numeric, \
    object_has_single_value, parameter_spec, safe_len, scalar_distance
from psyneulink.core.llvm import helpers


__all__ = [
    'ADDITIVE', 'ADDITIVE_PARAM', 'ASCENT',
    'AdditiveParam', 'ArgumentTherapy', 'AUTOASSOCIATIVE',
    'BackPropagation', 'BayesGLM', 'ContrastiveHebbian',
    'DESCENT', 'DISABLE', 'DISABLE_PARAM', 'Distance', 'DistributionFunction', 'EPSILON',
    'ERROR_MATRIX', 'ExponentialDist', 'Function_Base', 'function_keywords', 'FunctionError', 'FunctionOutputType', 'FunctionRegistry',
    'GammaDist', 'get_param_value_for_function', 'get_param_value_for_keyword',
    'GradientOptimization', 'GridSearch',
    'Hebbian', 'is_Function', 'is_function_type',
    'LEARNING_ACTIVATION_FUNCTION',
    'LEARNING_ACTIVATION_INPUT', 'LEARNING_ACTIVATION_OUTPUT',
    'LEARNING_ERROR_OUTPUT', 'LearningFunction', 'MAXIMIZE', 'max_vs_avg', 'max_vs_next', 'MINIMIZE', 'ModulatedParam',
    'ModulationParam', 'MULTIPLICATIVE', 'MULTIPLICATIVE_PARAM',
    'MultiplicativeParam', 'NormalDist', 'ObjectiveFunction', 'OptimizationFunction', 'OVERRIDE', 'OVERRIDE_PARAM', 'PERTINACITY', 'PROPENSITY',
    'Reinforcement', 'ReturnVal',
    'SEARCH_FUNCTION', 'SEARCH_SPACE', 'SEARCH_TERMINATION_FUNCTION', 'Stability', 'TDLearning', 'UniformDist', 'UniformToNormalDist', 'WaldDist', 'WT_MATRIX_RECEIVERS_DIM', 'WT_MATRIX_SENDERS_DIM'
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
    reduce = lambda x: np.product(np.array(x), axis=0)


class AdditiveParam():
    attrib_name = ADDITIVE_PARAM
    name = 'ADDITIVE'
    init_val = 0
    reduce = lambda x: np.sum(np.array(x), axis=0)


# class OverrideParam():
#     attrib_name = OVERRIDE_PARAM
#     name = 'OVERRIDE'
#     init_val = None
#     reduce = lambda x : None
#
# class DisableParam():
#     attrib_name = OVERRIDE_PARAM
#     name = 'DISABLE'
#     init_val = None
#     reduce = lambda x : None


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
    #                           'ADDITIVE',
    #                           0,
    #                           lambda x : np.sum(np.array(x), axis=0))
    OVERRIDE = OVERRIDE_PARAM
    # OVERRIDE = OverrideParam
    DISABLE = DISABLE_PARAM
    # DISABLE = DisableParam


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


def _get_modulated_param(owner, mod_proj, execution_context=None):
    """Return ModulationParam object, function param name and value of param modulated by ModulatoryProjection
    """

    from psyneulink.core.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base

    if not isinstance(mod_proj, ModulatoryProjection_Base):
        raise FunctionError('mod_proj ({0}) is not a ModulatoryProjection_Base'.format(mod_proj))

    # Get function "meta-parameter" object specified in the Projection sender's modulation attribute
    function_mod_meta_param_obj = mod_proj.sender.modulation

    # # MODIFIED 6/27/18 OLD
    # # Get the actual parameter of owner.function_object to be modulated
    # function_param_name = owner.function_object.params[function_mod_meta_param_obj.attrib_name]
    # # Get the function parameter's value
    # function_param_value = owner.function_object.params[function_param_name]
    # # MODIFIED 6/27/18 NEW:
    if function_mod_meta_param_obj in {OVERRIDE, DISABLE}:
        # function_param_name = function_mod_meta_param_obj
        from psyneulink.core.globals.utilities import Modulation
        function_mod_meta_param_obj = getattr(Modulation, function_mod_meta_param_obj)
        function_param_name = function_mod_meta_param_obj
        function_param_value = mod_proj.sender.parameters.value.get(execution_context)
    else:
        # Get the actual parameter of owner.function_object to be modulated
        function_param_name = owner.function_object.params[function_mod_meta_param_obj.attrib_name]
        # Get the function parameter's value
        function_param_value = owner.function_object.params[function_param_name]
    # # MODIFIED 6/27/18 NEWER:
    # from psyneulink.core.globals.utilities import Modulation
    # mod_spec = function_mod_meta_param_obj.attrib_name
    # if mod_spec == OVERRIDE_PARAM:
    #     function_param_name = mod_spec
    #     function_param_value = mod_proj.sender.value
    # elif mod_spec == DISABLE_PARAM:
    #     function_param_name = mod_spec
    #     function_param_value = None
    # else:
    #     # Get name of the actual parameter of owner.function_object to be modulated
    #     function_param_name = owner.function_object.params[mod_spec]
    #     # Get the function parameter's value
    #     function_param_value = owner.function_object.params[mod_spec]
    # MODIFIED 6/27/18 END

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

# Parameter Mixins *****************************************************************************************************

# KDM 6/21/18: Below is left in for consideration; doesn't really gain much to justify relaxing the assumption
# that every Params class has a single parent

# class ScaleOffsetParamMixin:
#     scale = Param(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
#     offset = Param(1.0, modulable=True, aliases=[ADDITIVE_PARAM])


# Function Definitions *************************************************************************************************


# KDM 8/9/18: below is added for future use when function methods are completely functional
# used as a decorator for Function methods
# def enable_output_conversion(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         result = func(*args, **kwargs)
#         return convert_output_type(result)
#     return wrapper


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
              - including them in a call the function method (which changes their values for just for that call)
            - Parameters must be specified in a params dictionary:
              - the key for each entry should be the name of the parameter (used also to name associated Projections)
              - the value for each entry is the value of the parameter

        Return values:
            The output_type can be used to specify type conversion for single-item return values:
            - it can only be used for numbers or a single-number list; other values will generate an exception
            - if self.output_type is set to:
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
    enable_output_type_conversion : Bool : False
        specifies whether `function output type conversion <Function_Output_Type_Conversion>` is enabled.

    output_type : FunctionOutputType : None
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

    class Params(Function.Params):
        variable = Param(np.array([0]), read_only=True)

    # Note: the following enforce encoding as 1D np.ndarrays (one array per variable)
    variableEncodingDim = 1

    paramClassDefaults = Function.paramClassDefaults.copy()
    paramClassDefaults.update({
        FUNCTION_OUTPUT_TYPE_CONVERSION: False,  # Enable/disable output type conversion
        FUNCTION_OUTPUT_TYPE: None  # Default is to not convert
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

        if self.context.initialization_status == ContextFlags.DEFERRED_INIT:
            self._assign_deferred_init_name(name, context)
            self.init_args[NAME] = name
            return


        self._output_type = None
        self.enable_output_type_conversion = False

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

    def get_current_function_param(self, param_name, execution_context=None):
        if param_name == "variable":
            raise FunctionError("The method 'get_current_function_param' is intended for retrieving the current value "
                                "of a function parameter. 'variable' is not a function parameter. If looking for {}'s "
                                "default variable, try {}.instance_defaults.variable.".format(self.name, self.name))
        try:
            return self.owner._parameter_states[param_name].parameters.value.get(execution_context)
        except (AttributeError, TypeError):
            try:
                return getattr(self.parameters, param_name).get(execution_context)
            except AttributeError:
                raise FunctionError("{0} has no parameter '{1}'".format(self, param_name))

    def get_previous_value(self, execution_context=None):
        # temporary method until previous values are integrated for all parameters
        value = self.parameters.previous_value.get(execution_context)
        if value is None:
            value = self.parameters.previous_value.get()

        return value

    def convert_output_type(self, value, output_type=None):
        if output_type is None:
            if not self.enable_output_type_conversion or self.output_type is None:
                return value
            else:
                output_type = self.output_type

        value = np.asarray(value)

        # region Type conversion (specified by output_type):
        # Convert to 2D array, irrespective of value type:
        if output_type is FunctionOutputType.NP_2D_ARRAY:
            # KDM 8/10/18: mimicking the conversion that Mechanism does to its values, because
            # this is what we actually wanted this method for. Can be changed to pure 2D np array in
            # future if necessary

            converted_to_2d = np.atleast_2d(value)
            # If return_value is a list of heterogenous elements, return as is
            #     (satisfies requirement that return_value be an array of possibly multidimensional values)
            if converted_to_2d.dtype == object:
                pass
            # Otherwise, return value converted to 2d np.array
            else:
                value = converted_to_2d

        # Convert to 1D array, irrespective of value type:
        # Note: if 2D array (or higher) has more than two items in the outer dimension, generate exception
        elif output_type is FunctionOutputType.NP_1D_ARRAY:
            # If variable is 2D
            if value.ndim >= 2:
                # If there is only one item:
                if len(value) == 1:
                    value = value[0]
                else:
                    raise FunctionError("Can't convert value ({0}: 2D np.ndarray object with more than one array)"
                                        " to 1D array".format(value))
            elif value.ndim == 1:
                value = value
            elif value.ndim == 0:
                value = np.atleast_1d(value)
            else:
                raise FunctionError("Can't convert value ({0} to 1D array".format(value))

        # Convert to raw number, irrespective of value type:
        # Note: if 2D or 1D array has more than two items, generate exception
        elif output_type is FunctionOutputType.RAW_NUMBER:
            if object_has_single_value(value):
                value = float(value)
            else:
                raise FunctionError("Can't convert value ({0}) with more than a single number to a raw number".format(value))

        return value

    @property
    def output_type(self):
        return self._output_type

    @output_type.setter
    def output_type(self, value):
        # Bad outputType specification
        if value is not None and not isinstance(value, FunctionOutputType):
            raise FunctionError("value ({0}) of output_type attribute must be FunctionOutputType for {1}".
                                format(self.output_type, self.__class__.__name__))

        # Can't convert from arrays of length > 1 to number
        if (
            self.instance_defaults.variable is not None
            and safe_len(self.instance_defaults.variable) > 1
            and self.output_type is FunctionOutputType.RAW_NUMBER
        ):
            raise FunctionError(
                "{0} can't be set to return a single number since its variable has more than one number".
                format(self.__class__.__name__))

        # warn if user overrides the 2D setting for mechanism functions
        # may be removed when https://github.com/PrincetonUniversity/PsyNeuLink/issues/895 is solved properly
        # (meaning Mechanism values may be something other than 2D np array)
        try:
            # import here because if this package is not installed, we can assume the user is probably not dealing with compilation
            # so no need to warn unecessarily
            import llvmlite
            if (isinstance(self.owner, Mechanism) and (value == FunctionOutputType.RAW_NUMBER or value == FunctionOutputType.NP_1D_ARRAY)):
                warnings.warn(
                    'Functions that are owned by a Mechanism but do not return a 2D numpy array may cause unexpected behavior if '
                    'llvm compilation is enabled.'
                )
        except (AttributeError, ImportError):
            pass

        self._output_type = value

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

    def _get_context_initializer(self, execution_id):
        return tuple([])

    def _get_param_ids(self, execution_id=None):
        params = []

        for pc in self.parameters.names():
            # Filter out params not allowed in get_current_function_param
            if pc != 'function' and pc != 'value' and pc != 'variable':
                val = self.get_current_function_param(pc, execution_id)
                # or are not numeric (this includes aliases)
                if not isinstance(val, str):
                    params.append(pc)
        return params

    def _get_param_values(self, execution_id=None):
        param_init = []
        for p in self._get_param_ids():
            param = self.get_current_function_param(p, execution_id)
            if not np.isscalar(param) and param is not None:
                param = np.asfarray(param).flatten().tolist()
            param_init.append(param)

        return tuple(param_init)

    def _get_param_initializer(self, execution_id):
        return pnlvm._tupleize(self._get_param_values(execution_id))

    def bin_function(self,
                     variable=None,
                     execution_id=None,
                     params=None,
                     context=None):

        # TODO: Port this to llvm
        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        e = pnlvm.FuncExecution(self, execution_id)
        return e.execute(variable)

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
    paramClassDefaults.update({
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
                 execution_id=None,
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
        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        # Compute the function
        statement = variable
        propensity = self.get_current_function_param(PROPENSITY, execution_id)
        pertinacity = self.get_current_function_param(PERTINACITY, execution_id)
        whim = randint(-10, 10)

        if propensity == self.Manner.OBSEQUIOUS:
            value = whim < pertinacity

        elif propensity == self.Manner.CONTRARIAN:
            value = whim > pertinacity

        else:
            raise FunctionError("This should not happen if parameter_validation == True;  check its value")

        return self.convert_output_type(value)


# region ****************************************   FUNCTIONS   ********************************************************
# endregion

# region ************************************   DISTRIBUTION FUNCTIONS   ***********************************************

class DistributionFunction(Function_Base):
    componentType = DIST_FUNCTION_TYPE


class NormalDist(DistributionFunction):
    """
    NormalDist(                      \
             mean=0.0,               \
             standard_deviation=1.0, \
             params=None,            \
             owner=None,             \
             prefs=None              \
             )

    .. _NormalDist:

    Return a random sample from a normal distribution using numpy.random.normal

    Arguments
    ---------

    mean : float : default 0.0
        The mean or center of the normal distribution

    standard_deviation : float : default 1.0
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

    standard_deviation : float : default 1.0
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

    class Params(DistributionFunction.Params):
        mean = Param(0.0, modulable=True)
        standard_deviation = Param(1.0, modulable=True)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 mean=0.0,
                 standard_deviation=1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(mean=mean,
                                                  standard_deviation=standard_deviation,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)


    def _validate_params(self, request_set, target_set=None, context=None):
        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        if STANDARD_DEVIATION in target_set:
            if target_set[STANDARD_DEVIATION] <= 0.0:
                raise FunctionError("The standard_deviation parameter ({}) of {} must be greater than zero.".
                                    format(target_set[STANDARD_DEVIATION], self.name))

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        # Validate variable and validate params
        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        mean = self.get_current_function_param(DIST_MEAN, execution_id)
        standard_deviation = self.get_current_function_param(STANDARD_DEVIATION, execution_id)

        result = np.random.normal(mean, standard_deviation)

        return self.convert_output_type(result)


class UniformToNormalDist(DistributionFunction):
    """
    UniformToNormalDist(             \
             mean=0.0,               \
             standard_deviation=1.0, \
             params=None,            \
             owner=None,             \
             prefs=None              \
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

    standard_deviation : float : default 1.0
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

    standard_deviation : float : default 1.0
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

    class Params(DistributionFunction.Params):
        variable = Param(np.array([0]), read_only=True)
        mean = Param(0.0, modulable=True)
        standard_deviation = Param(1.0, modulable=True)

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 mean=0.0,
                 standard_deviation=1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(mean=mean,
                                                  standard_deviation=standard_deviation,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)


    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):

        try:
            from scipy.special import erfinv
        except:
            raise FunctionError("The UniformToNormalDist function requires the SciPy package.")

        # Validate variable and validate params
        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        mean = self.get_current_function_param(DIST_MEAN, execution_id)
        standard_deviation = self.get_current_function_param(STANDARD_DEVIATION, execution_id)

        sample = np.random.rand(1)[0]
        result = ((np.sqrt(2) * erfinv(2 * sample - 1)) * standard_deviation) + mean

        return self.convert_output_type(result)


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

    class Params(DistributionFunction.Params):
        beta = Param(1.0, modulable=True)

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


    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        # Validate variable and validate params
        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        beta = self.get_current_function_param(BETA, execution_id)
        result = np.random.exponential(beta)

        return self.convert_output_type(result)


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

    class Params(DistributionFunction.Params):
        low = Param(0.0, modulable=True)
        high = Param(1.0, modulable=True)

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


    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        # Validate variable and validate params
        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        low = self.get_current_function_param(LOW, execution_id)
        high = self.get_current_function_param(HIGH, execution_id)
        result = np.random.uniform(low, high)

        return self.convert_output_type(result)


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

    class Params(DistributionFunction.Params):
        scale = Param(1.0, modulable=True)
        dist_shape = Param(1.0, modulable=True)

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


    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        # Validate variable and validate params
        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        scale = self.get_current_function_param(SCALE, execution_id)
        dist_shape = self.get_current_function_param(DIST_SHAPE, execution_id)

        result = np.random.gamma(dist_shape, scale)

        return self.convert_output_type(result)


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

    class Params(DistributionFunction.Params):
        scale = Param(1.0, modulable=True)
        mean = Param(1.0, modulable=True)

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


    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        # Validate variable and validate params
        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        scale = self.get_current_function_param(SCALE, execution_id)
        mean = self.get_current_function_param(DIST_MEAN, execution_id)

        result = np.random.wald(mean, scale)

        return self.convert_output_type(result)


# endregion

# region **************************************  OBJECTIVE FUNCTIONS ***************************************************

class ObjectiveFunction(Function_Base):
    """Abstract class of `Function` used for evaluating states.
    """

    componentType = OBJECTIVE_FUNCTION_TYPE

    class Params(Function_Base.Params):
        normalize = False
        metric = Param(None, stateful=False)


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
*** 11/11/17 - DELETE THIS ONCE Stability IS STABLE:
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
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).
    """

    componentName = STABILITY_FUNCTION

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    class Params(ObjectiveFunction.Params):
        matrix = HOLLOW_MATRIX
        metric = Param(ENERGY, stateful=False)
        transfer_fct = None
        normalize = False

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 matrix=HOLLOW_MATRIX,
                 # metric:is_distance_metric=ENERGY,
                 metric: tc.any(tc.enum(ENERGY, ENTROPY), is_distance_metric) = ENERGY,
                 transfer_fct: tc.optional(tc.any(function_type, method_type)) = None,
                 normalize: bool = False,
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

            from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
            from psyneulink.core.components.states.parameterstate import ParameterState

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

        size = len(self.instance_defaults.variable)

        from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
        from psyneulink.core.components.states.parameterstate import ParameterState
        if isinstance(self.matrix, MappingProjection):
            self._matrix = self.matrix._parameter_states[MATRIX]
        elif isinstance(self.matrix, ParameterState):
            pass
        else:
            self._matrix = get_matrix(self.matrix, size, size)

        self._hollow_matrix = get_matrix(HOLLOW_MATRIX, size, size)

        default_variable = [self.instance_defaults.variable,
                            self.instance_defaults.variable]

        if self.metric is ENTROPY:
            self._metric_fct = Distance(default_variable=default_variable, metric=CROSS_ENTROPY, normalize=self.normalize)
        elif self.metric in DISTANCE_METRICS._set():
            self._metric_fct = Distance(default_variable=default_variable, metric=self.metric, normalize=self.normalize)

    def _get_param_struct_type(self, ctx):
        my_params = ctx.get_param_struct_type(super())
        metric_params = ctx.get_param_struct_type(self._metric_fct)
        transfer_params = ctx.get_param_struct_type(self.transfer_fct) if self.transfer_fct is not None else ir.LiteralStructType([])
        return ir.LiteralStructType([my_params, metric_params, transfer_params])

    def _get_param_initializer(self, execution_id):
        my_params = super()._get_param_initializer(execution_id)
        metric_params = self._metric_fct._get_param_initializer(execution_id)
        transfer_params = self.transfer_fct._get_param_initializer(execution_id) if self.transfer_fct is not None else tuple()
        return tuple([my_params, metric_params, transfer_params])

    def _gen_llvm_function_body(self, ctx, builder, params, state, arg_in, arg_out):
        # Dot product
        dot_out = builder.alloca(arg_in.type.pointee)
        my_params = builder.gep(params, [ctx.int32_ty(0), ctx.int32_ty(0)])
        matrix, builder = ctx.get_param_ptr(self, builder, my_params, MATRIX)

        # Convert array pointer to pointer to the fist element
        matrix = builder.gep(matrix, [ctx.int32_ty(0), ctx.int32_ty(0)])
        vec_in = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(0)])
        vec_out = builder.gep(dot_out, [ctx.int32_ty(0), ctx.int32_ty(0)])

        input_length = ctx.int32_ty(arg_in.type.pointee.count)
        output_length = ctx.int32_ty(arg_in.type.pointee.count)
        builtin = ctx.get_llvm_function('__pnl_builtin_vxm')
        builder.call(builtin, [vec_in, matrix, input_length, output_length, vec_out])

        # Prepare metric function
        metric_fun = ctx.get_llvm_function(self._metric_fct)
        metric_in = builder.alloca(metric_fun.args[2].type.pointee)

        # Transfer Function if configured
        trans_out = builder.gep(metric_in, [ctx.int32_ty(0), ctx.int32_ty(1)])
        if self.transfer_fct is not None:
            assert False
        else:
            builder.store(builder.load(dot_out), trans_out)

        # Copy original variable
        builder.store(builder.load(arg_in), builder.gep(metric_in, [ctx.int32_ty(0), ctx.int32_ty(0)]))

        # Distance Function
        metric_params = builder.gep(params, [ctx.int32_ty(0), ctx.int32_ty(1)])
        metric_state = state
        metric_out = arg_out
        builder.call(metric_fun, [metric_params, metric_state, metric_in, metric_out])
        return builder

    def function(self,
                 variable=None,
                 execution_id=None,
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
        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        matrix = self.get_current_function_param(MATRIX, execution_id)

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
        result = self._metric_fct.function(variable=[current, transformed], context=context)
        # MODIFIED 11/12/15 END

        return self.convert_output_type(result)

    @property
    def _dependent_components(self):
        return list(itertools.chain(
            super()._dependent_components,
            [self._metric_fct] if self._metric_fct is not None else [],
            [self.transfer_fct] if self.transfer_fct is not None else [],
        ))


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

    class Params(ObjectiveFunction.Params):
        variable = Param(np.array([[0], [0]]), read_only=True)
        metric = Param(DIFFERENCE, stateful=False)

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 metric: DistanceMetrics._is_metric = DIFFERENCE,
                 normalize: bool = False,
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

    def correlation(v1, v2):
        v1_norm = v1 - np.mean(v1)
        v2_norm = v2 - np.mean(v2)
        denom = np.sqrt(np.sum(v1_norm ** 2) * np.sum(v2_norm ** 2)) or EPSILON
        return np.sum(v1_norm * v2_norm) / denom

    def __gen_llvm_difference(self, builder, index, ctx, v1, v2, acc):
        ptr1 = builder.gep(v1, [index])
        ptr2 = builder.gep(v2, [index])
        val1 = builder.load(ptr1)
        val2 = builder.load(ptr2)

        sub = builder.fsub(val1, val2)
        ltz = builder.fcmp_ordered("<", sub, ctx.float_ty(0))
        abs_val = builder.select(ltz, builder.fsub(ctx.float_ty(0), sub), sub)
        acc_val = builder.load(acc)
        new_acc = builder.fadd(acc_val, abs_val)
        builder.store(new_acc, acc)

    def __gen_llvm_euclidean(self, builder, index, ctx, v1, v2, acc):
        ptr1 = builder.gep(v1, [index])
        ptr2 = builder.gep(v2, [index])
        val1 = builder.load(ptr1)
        val2 = builder.load(ptr2)

        sub = builder.fsub(val1, val2)
        sqr = builder.fmul(sub, sub)
        acc_val = builder.load(acc)
        new_acc = builder.fadd(acc_val, sqr)
        builder.store(new_acc, acc)

    def __gen_llvm_cross_entropy(self, builder, index, ctx, v1, v2, acc):
        ptr1 = builder.gep(v1, [index])
        ptr2 = builder.gep(v2, [index])
        val1 = builder.load(ptr1)
        val2 = builder.load(ptr2)

        log_f = ctx.module.declare_intrinsic("llvm.log", [ctx.float_ty])
        log = builder.call(log_f, [val2])
        prod = builder.fmul(val1, log)

        acc_val = builder.load(acc)
        new_acc = builder.fsub(acc_val, prod)
        builder.store(new_acc, acc)

    def __gen_llvm_energy(self, builder, index, ctx, v1, v2, acc):
        ptr1 = builder.gep(v1, [index])
        ptr2 = builder.gep(v2, [index])
        val1 = builder.load(ptr1)
        val2 = builder.load(ptr2)

        prod = builder.fmul(val1, val2)
        prod = builder.fmul(prod, ctx.float_ty(0.5))

        acc_val = builder.load(acc)
        new_acc = builder.fsub(acc_val, prod)
        builder.store(new_acc, acc)

    def __gen_llvm_correlate(self, builder, index, ctx, v1, v2, acc):
        ptr1 = builder.gep(v1, [index])
        ptr2 = builder.gep(v2, [index])
        val1 = builder.load(ptr1)
        val2 = builder.load(ptr2)

        # This should be conjugate, but we don't deal with complex numbers
        mul = builder.fmul(val1, val2)
        acc_val = builder.load(acc)
        new_acc = builder.fadd(acc_val, mul)
        builder.store(new_acc, acc)

    def __gen_llvm_max_diff(self, builder, index, ctx, v1, v2, max_diff_ptr):
        ptr1 = builder.gep(v1, [index])
        ptr2 = builder.gep(v2, [index])
        val1 = builder.load(ptr1)
        val2 = builder.load(ptr2)

        # Get the difference
        diff = builder.fsub(val1, val2)

        # Get absolute value
        fabs = ctx.module.declare_intrinsic("llvm.fabs", [ctx.float_ty])
        diff = builder.call(fabs, [diff])

        old_max = builder.load(max_diff_ptr)
        # Maxnum for some reason needs full function prototype
        fmax = ctx.module.declare_intrinsic("llvm.maxnum", [ctx.float_ty],
            ir.types.FunctionType(ctx.float_ty, [ctx.float_ty, ctx.float_ty]))

        max_diff = builder.call(fmax, [diff, old_max])
        builder.store(max_diff, max_diff_ptr)

    def __gen_llvm_pearson(self, builder, index, ctx, v1, v2, acc_x, acc_y, acc_xy, acc_x2, acc_y2):
        ptr1 = builder.gep(v1, [index])
        ptr2 = builder.gep(v2, [index])
        val1 = builder.load(ptr1)
        val2 = builder.load(ptr2)

        # Sum X
        acc_x_val = builder.load(acc_x)
        acc_x_val = builder.fadd(acc_x_val, val1)
        builder.store(acc_x_val, acc_x)

        # Sum Y
        acc_y_val = builder.load(acc_y)
        acc_y_val = builder.fadd(acc_y_val, val2)
        builder.store(acc_y_val, acc_y)

        # Sum XY
        acc_xy_val = builder.load(acc_xy)
        xy = builder.fmul(val1, val2)
        acc_xy_val = builder.fadd(acc_xy_val, xy)
        builder.store(acc_xy_val, acc_xy)

        # Sum X2
        acc_x2_val = builder.load(acc_x2)
        x2 = builder.fmul(val1, val1)
        acc_x2_val = builder.fadd(acc_x2_val, x2)
        builder.store(acc_x2_val, acc_x2)

        # Sum Y2
        acc_y2_val = builder.load(acc_y2)
        y2 = builder.fmul(val2, val2)
        acc_y2_val = builder.fadd(acc_y2_val, y2)
        builder.store(acc_y2_val, acc_y2)

    def _gen_llvm_function_body(self, ctx, builder, params, _, arg_in, arg_out):
        v1 = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(0)])
        v2 = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(1), ctx.int32_ty(0)])

        acc_ptr = builder.alloca(ctx.float_ty)
        builder.store(ctx.float_ty(0), acc_ptr)

        kwargs = {"ctx": ctx, "v1": v1, "v2": v2, "acc": acc_ptr}
        if self.metric == DIFFERENCE:
            inner = functools.partial(self.__gen_llvm_difference, **kwargs)
        elif self.metric == EUCLIDEAN:
            inner = functools.partial(self.__gen_llvm_euclidean, **kwargs)
        elif self.metric == CROSS_ENTROPY:
            inner = functools.partial(self.__gen_llvm_cross_entropy, **kwargs)
        elif self.metric == ENERGY:
            inner = functools.partial(self.__gen_llvm_energy, **kwargs)
        elif self.metric == MAX_ABS_DIFF:
            del kwargs['acc']
            max_diff_ptr = builder.alloca(ctx.float_ty)
            builder.store(ctx.float_ty("NaN"), max_diff_ptr)
            kwargs['max_diff_ptr'] = max_diff_ptr
            inner = functools.partial(self.__gen_llvm_max_diff, **kwargs)
        elif self.metric == CORRELATION:
            acc_x_ptr = builder.alloca(ctx.float_ty)
            acc_y_ptr = builder.alloca(ctx.float_ty)
            acc_xy_ptr = builder.alloca(ctx.float_ty)
            acc_x2_ptr = builder.alloca(ctx.float_ty)
            acc_y2_ptr = builder.alloca(ctx.float_ty)
            for loc in [acc_x_ptr, acc_y_ptr, acc_xy_ptr, acc_x2_ptr, acc_y2_ptr]:
                builder.store(ctx.float_ty(0), loc)
            del kwargs['acc']
            kwargs['acc_x'] = acc_x_ptr
            kwargs['acc_y'] = acc_y_ptr
            kwargs['acc_xy'] = acc_xy_ptr
            kwargs['acc_x2'] = acc_x2_ptr
            kwargs['acc_y2'] = acc_y2_ptr
            inner = functools.partial(self.__gen_llvm_pearson, **kwargs)
        else:
            raise RuntimeError('Unsupported metric')

        assert isinstance(arg_in.type.pointee, ir.ArrayType)
        assert isinstance(arg_in.type.pointee.element, ir.ArrayType)
        assert arg_in.type.pointee.count == 2

        input_length = arg_in.type.pointee.element.count
        vector_length = ctx.int32_ty(input_length)
        with helpers.for_loop_zero_inc(builder, vector_length, self.metric) as args:
            inner(*args)

        sqrt = ctx.module.declare_intrinsic("llvm.sqrt", [ctx.float_ty])
        fabs = ctx.module.declare_intrinsic("llvm.fabs", [ctx.float_ty])
        ret = builder.load(acc_ptr)
        if self.metric == EUCLIDEAN:
            ret = builder.call(sqrt, [ret])
        elif self.metric == MAX_ABS_DIFF:
            ret = builder.load(max_diff_ptr)
        elif self.metric == CORRELATION:
            n = ctx.float_ty(input_length)
            acc_xy = builder.load(acc_xy_ptr)
            acc_x = builder.load(acc_x_ptr)
            acc_y = builder.load(acc_y_ptr)
            acc_x2 = builder.load(acc_x2_ptr)
            acc_y2 = builder.load(acc_y2_ptr)

            # We'll need meanx,y below
            mean_x = builder.fdiv(acc_x, n)
            mean_y = builder.fdiv(acc_y, n)

            # Numerator: sum((x - mean(x))*(y - mean(y)) =
            # sum(x*y - x*mean(y) - y*mean(x) + mean(x)*mean(y)) =
            # sum(x*y) - sum(x)*mean(y) - sum(y)*mean(x) + mean(x)*mean(y)*n
            b = builder.fmul(acc_x, mean_y)
            c = builder.fmul(acc_y, mean_x)
            d = builder.fmul(mean_x, mean_y)
            d = builder.fmul(d, n)

            numerator = builder.fsub(acc_xy, b)
            numerator = builder.fsub(numerator, c)
            numerator = builder.fadd(numerator, d)

            # Denominator: sqrt(D_X * D_Y)
            # D_X = sum((x - mean(x))^2) = sum(x^2 - 2*x*mean(x) + mean(x)^2) =
            # sum(x^2) - 2 * sum(x) * mean(x) + n * mean(x)^2
            dxb = builder.fmul(acc_x, mean_x)
            dxb = builder.fadd(dxb, dxb)        # *2
            dxc = builder.fmul(mean_x, mean_x)  # ^2
            dxc = builder.fmul(dxc, n)

            dx = builder.fsub(acc_x2, dxb)
            dx = builder.fadd(dx, dxc)

            # Similarly for y
            dyb = builder.fmul(acc_y, mean_y)
            dyb = builder.fadd(dyb, dyb)        # *2
            dyc = builder.fmul(mean_y, mean_y)  # ^2
            dyc = builder.fmul(dyc, n)

            dy = builder.fsub(acc_y2, dyb)
            dy = builder.fadd(dy, dyc)

            # Denominator: sqrt(D_X * D_Y)
            denominator = builder.fmul(dx, dy)
            denominator = builder.call(sqrt, [denominator])

            corr = builder.fdiv(numerator, denominator)

            # ret =  1 - abs(corr)
            ret = builder.call(fabs, [corr])
            ret = builder.fsub(ctx.float_ty(1), ret)

        # MAX_ABS_DIFF ignores normalization
        if self.normalize and self.metric != MAX_ABS_DIFF and self.metric != CORRELATION:
            norm_factor = input_length
            if self.metric == ENERGY:
                norm_factor = norm_factor ** 2
            ret = builder.fdiv(ret, ctx.float_ty(norm_factor), name="normalized")
        builder.store(ret, arg_out)

        return builder

    def function(self,
                 variable=None,
                 execution_id=None,
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
        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        v1 = variable[0]
        v2 = variable[1]

        # Maximum of  Hadamard (elementwise) difference of v1 and v2
        if self.metric is MAX_ABS_DIFF:
            result = np.max(abs(v1 - v2))

        # Simple Hadamard (elementwise) difference of v1 and v2
        elif self.metric is DIFFERENCE:
            result = np.sum(np.abs(v1 - v2))

        # Euclidean distance between v1 and v2
        elif self.metric is EUCLIDEAN:
            result = np.linalg.norm(v2 - v1)

        # FIX: NEED SCIPY HERE
        # # Angle (cosine) of v1 and v2
        # elif self.metric is ANGLE:
        #     result = scipy.spatial.distance.cosine(v1,v2)

        # Correlation of v1 and v2
        elif self.metric is CORRELATION:
            # result = np.correlate(v1, v2)
            result = 1 - np.abs(Distance.correlation(v1, v2))
            return self.convert_output_type(result)

        # Cross-entropy of v1 and v2
        elif self.metric is CROSS_ENTROPY:
            # FIX: VALIDATE THAT ALL ELEMENTS OF V1 AND V2 ARE 0 TO 1
            if self.parameters.context.get(execution_id).initialization_status != ContextFlags.INITIALIZING:
                v1 = np.where(v1 == 0, EPSILON, v1)
                v2 = np.where(v2 == 0, EPSILON, v2)
            # MODIFIED CW 3/20/18: avoid divide by zero error by plugging in two zeros
            # FIX: unsure about desired behavior when v2 = 0 and v1 != 0
            # JDC: returns [inf]; leave, and let it generate a warning or error message for user
            result = -np.sum(np.where(np.logical_and(v1 == 0, v2 == 0), 0, v1 * np.log(v2)))

        # Energy
        elif self.metric is ENERGY:
            result = -np.sum(v1 * v2) / 2

        if self.normalize and not self.metric in {MAX_ABS_DIFF, CORRELATION}:
            if self.metric is ENERGY:
                result /= len(v1) ** 2
            else:
                result /= len(v1)

        return self.convert_output_type(result)

# endregion


# region **************************************   OPTIMIZATION FUNCTIONS ***********************************************


OBJECTIVE_FUNCTION = 'objective_function'
SEARCH_FUNCTION = 'search_function'
SEARCH_SPACE = 'search_space'
SEARCH_TERMINATION_FUNCTION = 'search_termination_function'
DIRECTION = 'direction'

class OptimizationFunction(Function_Base):
    """OptimizationFunction( \
         default_variable, objective_function, search_function, search_space, search_termination_function, \
         save_samples, save_values, max_iterations, params, owner, prefs, context)

    Abstract class of `Function <Function>` that returns the sample of a variable yielding the optimized value
    of an `objective_function <OptimizationFunction.objective_function>`.

    .. note::
       This information is for reference only -- OptimizationFunction cannot be called directly;
       only subclasses can be called.

    Provides an interface to subclasses and external optimization functions. The default `function
    <OptimizationFunction.function>` executes iteratively, evaluating samples from `search_space
    <OptimizationFunction.search_space>` using `objective_function <OptimizationFunction.objective_function>`
    until terminated by `search_termination_function <OptimizationFunction.search_termination_function>`.
    Subclasses can override this to implement their own optimization function or call an external one.

    .. _OptimizationFunction_Process:

    **Default Optimization Process**

    When `function <OptimizationFunction.function>` is executed, it iterates over the following steps:

        - get sample from `search_space <OptimizationFunction.search_space>` using `search_function
          <OptimizationFunction.search_function>`.
        ..
        - compute value of `objective_function <OptimizationFunction.objective_function>` using the sample;
        ..
        - evaluate `search_termination_function <OptimizationFunction.search_termination_function>`.

    The current iteration is contained in `iteration <OptimizationFunction.iteration>`. Iteration continues until all
    values of `search_space <OptimizationFunction.search_space>` have been evaluated (i.e., `search_termination_function
    <OptimizationFunction.search_termination_function>` returns `True`).  The `function <OptimizationFunction.function>`
    returns:

    - the last sample evaluated (which may or may not be the optimal value, depending on the `objective_function
      <OptimizationFunction.objective_function>`);

    - the value of `objective_function <OptimzationFunction.objective_function>` associated with the last sample;

    - two lists, that may contain all of the samples evaluated and their values, depending on whether `save_samples
      <OptimizationFunction.save_samples>` and/or `save_vales <OptimizationFunction.save_values>` are `True`,
      respectively.

    .. _OptimizationFunction_Defaults:

    .. note::

        An OptimizationFunction or any of its subclasses can be created by calling its constructor.  This provides
        runnable defaults for all of its arguments (see below). However these do not yield useful results, and are
        meant simply to allow the  constructor of the OptimziationFunction to be used to specify some but not all of
        its parameters when specifying the OptimizationFunction in the constructor for another Component. For
        example, an OptimizationFunction may use for its `objective_function <OptimizationFunction.objective_function>`
        or `search_function <OptimizationFunction.search_function>` a method of the Component to which it is being
        assigned;  however, those methods will not yet be available, as the Component itself has not yet been
        constructed. This can be handled by calling the OptimizationFunction's `reinitialization
        <OptimizationFunction.reinitialization>` method after the Component has been instantiated, with a parameter
        specification dictionary with a key for each entry that is the name of a parameter and its value the value to
        be assigned to the parameter.  This is done automatically for Mechanisms that take an ObjectiveFunction as
        their `function <Mechanism.function>` (such as the `EVCControlMechanism`, `LVOCControlMechanism` and
        `ParamterEstimationControlMechanism`), but will require it be done explicitly for Components for which that
        is not the case. A warning is issued if defaults are used for the arguments of an OptimizationFunction or
        its subclasses;  this can be suppressed by specifying the relevant argument(s) as `NotImplemnted`.

    COMMENT:
    NOTES TO DEVELOPERS:
    - Constructors of subclasses should include **kwargs in their constructor method, to accomodate arguments required
      by some subclasses but not others (e.g., search_space needed by `GridSearch` but not `GradientOptimization`) so
      that subclasses are meant to be used interchangeably by OptimizationMechanisms.

    - Subclasses with attributes that depend on one of the OptimizationFunction's parameters should implement the
      `reinitialize <OptimizationFunction.reinitialize>` method, that calls super().reinitialize(*args) and then
      reassigns the values of the dependent attributes accordingly.  If an argument is not needed for the subclass,
      `NotImplemented` should be passed as the argument's value in the call to super (i.e., the OptimizationFunction's
      constructor).
    COMMENT


    Arguments
    ---------

    default_variable : list or ndarray : default None
        specifies a template for (i.e., an example of the shape of) the samples used to evaluate the
        `objective_function <OptimizationFunction.objective_function>`.

    objective_function : function or method : default None
        specifies function used to evaluate sample in each iteration of the `optimization process
        <OptimizationFunction_Process>`; if it is not specified, a default function is used that simply returns
        the value passed as its `variable <OptimizationFunction.variable>` parameter (see `note
        <OptimizationFunction_Defaults>`).

    search_function : function or method : default None
        specifies function used to select a sample for `objective_function <OptimizationFunction.objective_function>`
        in each iteration of the `optimization process <OptimizationFunction_Process>`.  It **must be specified**
        if the `objective_function <OptimizationFunction.objective_function>` does not generate samples on its own
        (e.g., as does `GradientOptimization`).  If it is required and not specified, the optimization process
        executes exactly once using the value passed as its `variable <OptimizationFunction.variable>` parameter
        (see `note <OptimizationFunction_Defaults>`).

    search_space : list or np.ndarray : default None
        specifies samples used to evaluate `objective_function <OptimizationFunction.objective_function>`
        in each iteration of the `optimization process <OptimizationFunction_Process>`. It **must be specified**
        if the `objective_function <OptimizationFunction.objective_function>` does not generate samples on its own
        (e.g., as does `GradientOptimization`).  If it is required and not specified, the optimization process
        executes exactly once using the value passed as its `variable <OptimizationFunction.variable>` parameter
        (see `note <OptimizationFunction_Defaults>`).

    search_termination_function : function or method : None
        specifies function used to terminate iterations of the `optimization process <OptimizationFunction_Process>`.
        It **must be specified** if the `objective_function <OptimizationFunction.objective_function>` is not
        overridden.  If it is required and not specified, the optimization process executes exactly once
        (see `note <OptimizationFunction_Defaults>`).

    save_samples : bool
        specifies whether or not to save and return the values of the samples used to evalute `objective_function
        <OptimizationFunction.objective_function>` over all iterations of the `optimization process
        <OptimizationFunction_Process>`.

    save_values : bool
        specifies whether or not to save and return the values of `objective_function
        <OptimizationFunction.objective_function>` for samples evaluated in all iterations of the
        `optimization process <OptimizationFunction_Process>`.

    max_iterations : int : default 1000
        specifies the maximum number of times the `optimization process <OptimizationFunction_Process>` is allowed
        to iterate; if exceeded, a warning is issued and the function returns the last sample evaluated.


    Attributes
    ----------

    variable : ndarray
        first sample evaluated by `objective_function <OptimizationFunction.objective_function>` (i.e., one used to
        evaluate it in the first iteration of the `optimization process <OptimizationFunction_Process>`).

    objective_function : function or method
        used to evaluate the sample in each iteration of the `optimization process <OptimizationFunction_Process>`.

    search_function : function, method or None
        used to select a sample evaluated by `objective_function <OptimizationFunction.objective_function>`
        in each iteration of the `optimization process <OptimizationFunction_Process>`.  `NotImplemented` if
        the `objective_function <OptimizationFunction.objective_function>` generates its own samples.

    search_space : list or np.ndarray
        samples used to evaluate `objective_function <OptimizationFunction.objective_function>`
        in each iteration of the `optimization process <OptimizationFunction_Process>`;  `NotImplemented` if
        the `objective_function <OptimizationFunction.objective_function>` generates its own samples.

    search_termination_function : function or method
        used to terminate iterations of the `optimization process <OptimizationFunction_Process>`.

    iteration : int
        the current iteration of the `optimization process <OptimizationFunction_Process>`.

    max_iterations : int : default 1000
        specifies the maximum number of times the `optimization process <OptimizationFunction_Process>` is allowed
        to iterate; if exceeded, a warning is issued and the function returns the last sample evaluated.

    save_samples : bool
        determines whether or not to save the values of the samples used to evalute `objective_function
        <OptimizationFunction.objective_function>` over all iterations of the `optimization process
        <OptimizationFunction_Process>`.

    save_values : bool
        determines whether or not to save and return the values of `objective_function
        <OptimizationFunction.objective_function>` for samples evaluated in all iterations of the
        `optimization process <OptimizationFunction_Process>`.
    """

    componentType = OPTIMIZATION_FUNCTION_TYPE

    class Params(Function_Base.Params):
        variable = Param(np.array([0, 0, 0]), read_only=True)

        objective_function = Param(lambda x: 0, stateful=False, loggable=False)
        search_function = Param(lambda x: x, stateful=False, loggable=False)
        search_termination_function = Param(lambda x, y, z: True, stateful=False, loggable=False)
        search_space = Param([0], stateful=False, loggable=False)

        # these are created as parameter states, but should they be?
        save_samples = Param(False, modulable=True)
        save_values = Param(False, modulable=True)
        max_iterations = Param(None, modulable=True)

        saved_samples = Param([], read_only=True)
        saved_values = Param([], read_only=True)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 objective_function:tc.optional(is_function_type)=None,
                 search_function:tc.optional(is_function_type)=None,
                 search_space=None,
                 search_termination_function:tc.optional(is_function_type)=None,
                 save_samples:tc.optional(bool)=False,
                 save_values:tc.optional(bool)=False,
                 max_iterations:tc.optional(int)=None,
                 params=None,
                 owner=None,
                 prefs=None,
                 context=None):

        self._unspecified_args = []

        if objective_function is None:
            self.objective_function = lambda x:0
            self._unspecified_args.append(OBJECTIVE_FUNCTION)
        else:
            self.objective_function = objective_function

        if search_function is None:
            self.search_function = lambda x:x
            self._unspecified_args.append(SEARCH_FUNCTION)
        else:
            self.search_function = search_function

        if search_termination_function is None:
            self.search_termination_function = lambda x,y,z:True
            self._unspecified_args.append(SEARCH_TERMINATION_FUNCTION)
        else:
            self.search_termination_function = search_termination_function

        if search_space is None:
            self.search_space = [0]
            self._unspecified_args.append(SEARCH_SPACE)
        else:
            self.search_space = search_space

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(save_samples=save_samples,
                                                  save_values=save_values,
                                                  max_iterations=max_iterations,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

    def reinitialize(self, *args, execution_id=None):
        '''Reinitialize parameters of the OptimizationFunction

        Parameters to be reinitialized should be specified in a parameter specification dictionary, in which they key
        for each entry is the name of one of the following parameters, and its value is the value to be assigned to the
        parameter.  The following parameters can be reinitialized:

            * `default_variable <OptimizationFunction.default_variable>`
            * `objective_function <OptimizationFunction.objective_function>`
            * `search_function <OptimizationFunction.search_function>`
            * `search_termination_function <OptimizationFunction.search_termination_function>`
        '''

        if DEFAULT_VARIABLE in args[0]:
            self.instance_defaults.variable = args[0][DEFAULT_VARIABLE]
        if OBJECTIVE_FUNCTION in args[0] and args[0][OBJECTIVE_FUNCTION] is not None:
            self.objective_function = args[0][OBJECTIVE_FUNCTION]
            if OBJECTIVE_FUNCTION in self._unspecified_args:
                del self._unspecified_args[self._unspecified_args.index(OBJECTIVE_FUNCTION)]
        if SEARCH_FUNCTION in args[0] and args[0][SEARCH_FUNCTION] is not None:
            self.search_function = args[0][SEARCH_FUNCTION]
            if SEARCH_FUNCTION in self._unspecified_args:
                del self._unspecified_args[self._unspecified_args.index(SEARCH_FUNCTION)]
        if SEARCH_TERMINATION_FUNCTION in args[0] and args[0][SEARCH_TERMINATION_FUNCTION] is not None:
            self.search_termination_function = args[0][SEARCH_TERMINATION_FUNCTION]
            if SEARCH_TERMINATION_FUNCTION in self._unspecified_args:
                del self._unspecified_args[self._unspecified_args.index(SEARCH_TERMINATION_FUNCTION)]
        if SEARCH_SPACE in args[0] and args[0][SEARCH_SPACE] is not None:
            self.parameters.search_space.set(args[0][SEARCH_SPACE], execution_id)
            if SEARCH_SPACE in self._unspecified_args:
                del self._unspecified_args[self._unspecified_args.index(SEARCH_SPACE)]

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None,
                 **kwargs):
        '''Find the sample that yields the optimal value of `objective_function
        <OptimizationFunction.objective_function>`.

        See `optimization process <OptimizationFunction_Process>` for details.

        Returns
        -------

        optimal sample, optimal value, saved_samples, saved_values : array, array, list, list
            first array contains sample that yields the optimal value of the `optimization process
            <OptimizationFunction_Process>`, and second array contains the value of `objective_function
            <OptimizationFunction.objective_function>` for that sample.  If `save_samples
            <OptimizationFunction.save_samples>` is `True`, first list contains all the values sampled in the order
            they were evaluated; otherwise it is empty.  If `save_values <OptimizationFunction.save_values>` is `True`,
            second list contains the values returned by `objective_function <OptimizationFunction.objective_function>`
            for all the samples in the order they were evaluated; otherwise it is empty.
        '''

        if self._unspecified_args and self.parameters.context.get(execution_id).initialization_status == ContextFlags.INITIALIZED:
            warnings.warn("The following arg(s) were not specified for {}: {} -- using default(s)".
                          format(self.name, ', '.join(self._unspecified_args)))
            self._unspecified_args = []

        sample = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        current_sample = sample
        current_value = call_with_pruned_args(self.objective_function, current_sample, execution_id=execution_id)

        samples = []
        values = []

        # Initialize variables used in while loop
        iteration = 0

        # Set up progress bar
        _show_progress = False
        if hasattr(self, OWNER) and self.owner and self.owner.prefs.reportOutputPref:
            _show_progress = True
            _progress_bar_char = '.'
            _progress_bar_rate_str = ""
            _search_space_size = len(self.search_space)
            _progress_bar_rate = int(10 ** (np.log10(_search_space_size)-2))
            if _progress_bar_rate > 1:
                _progress_bar_rate_str = str(_progress_bar_rate) + " "
            print("\n{} executing optimization process (one {} for each {}of {} samples): ".
                  format(self.owner.name, repr(_progress_bar_char), _progress_bar_rate_str, _search_space_size))
            _progress_bar_count = 0

        # Iterate optimization process
        while call_with_pruned_args(self.search_termination_function, current_sample, current_value, iteration, execution_id=execution_id):

            if _show_progress:
                increment_progress_bar = (_progress_bar_rate < 1) or not (_progress_bar_count % _progress_bar_rate)
                if increment_progress_bar:
                    print(_progress_bar_char, end='', flush=True)
                _progress_bar_count +=1

            # Get next sample of sample
            new_sample = call_with_pruned_args(self.search_function, current_sample, iteration, execution_id=execution_id)

            # Compute new value based on new sample
            new_value = call_with_pruned_args(self.objective_function, new_sample, execution_id=execution_id)

            iteration += 1
            max_iterations = self.parameters.max_iterations.get(execution_id)
            if max_iterations and iteration > max_iterations:
                warnings.warn("{} failed to converge after {} iterations".format(self.name, max_iterations))
                break

            current_sample = new_sample
            current_value = new_value

            if self.parameters.save_samples.get(execution_id):
                samples.append(new_sample)
                self.parameters.saved_samples.set(samples, execution_id, override=True)
            if self.parameters.save_values.get(execution_id):
                values.append(current_value)
                self.parameters.saved_values.set(values, execution_id, override=True)

        return new_sample, new_value, samples, values


ASCENT = 'ascent'
DESCENT = 'descent'


class GradientOptimization(OptimizationFunction):
    """
    GradientOptimization(            \
        default_variable=None,       \
        objective_function=None,     \
        direction=ASCENT,            \
        step_size=1.0,               \
        annealing_function=None,     \
        convergence_criterion=VALUE, \
        convergence_threshold=.001,  \
        max_iterations=1000,         \
        save_samples=False,          \
        save_values=False,           \
        params=None,                 \
        owner=None,                  \
        prefs=None                   \
        )

    Return sample that yields optimized value of `objective_function
    <GradientOptimization.objective_function>`.

    .. _GradientOptimization_Process:

    **Optimization Process**

    When `function <GradientOptimization.function>` is executed, it iterates over the folowing steps:

        - `compute gradient <GradientOptimization_Gradient_Calculation>` using the `gradient_function
          <GradientOptimization.gradient_function>`;
        ..
        - adjust `variable <GradientOptimization.variable>` based on the gradient, in the specified
          `direction <GradientOptimization.direction>` and by an amount specified by `step_size
          <GradientOptimization.step_size>` and possibly `annealing_function
          <GradientOptimization.annealing_function>`;
        ..
        - compute value of `objective_function <GradientOptimization.objective_function>` using the adjusted value of
          `variable <GradientOptimization.variable>`;
        ..
        - adjust `step_size <GradientOptimization.udpate_rate>` using `annealing_function
          <GradientOptimization.annealing_function>`, if specified, for use in the next iteration;
        ..
        - evaluate `convergence_criterion <GradientOptimization.convergence_criterion>` and test whether it is below
          the `convergence_threshold <GradientOptimization.convergence_threshold>`.

    The current iteration is contained in `iteration <GradientOptimization.iteration>`. Iteration continues until
    `convergence_criterion <GradientOptimization.convergence_criterion>` falls below `convergence_threshold
    <GradientOptimization.convergence_threshold>` or the number of iterations exceeds `max_iterations
    <GradientOptimization.max_iterations>`.  The `function <GradientOptimization.function>` returns the last sample
    evaluated by `objective_function <GradientOptimization.objective_function>` (presumed to be the optimal one),
    the value of the function, as well as lists that may contain all of the samples evaluated and their values,
    depending on whether `save_samples <OptimizationFunction.save_samples>` and/or `save_vales
    <OptimizationFunction.save_values>` are `True`, respectively.

    .. _GradientOptimization_Gradient_Calculation:

    **Gradient Calculation**

    The gradient is evaluated by `gradient_function <GradientOptimization.gradient_function>`,
    which is the derivative of the `objective_function <GradientOptimization.objective_function>`
    with respect to `variable <GradientOptimization.variable>` at its current value:
    :math:`\\frac{d(objective\\_function(variable))}{d(variable)}`

    `Autograd's <https://github.com/HIPS/autograd>`_ `grad <autograd.grad>` method is used to
    generate `gradient_function <GradientOptimization.gradient_function>`.


    Arguments
    ---------

    default_variable : list or ndarray : default None
        specifies a template for (i.e., an example of the shape of) the samples used to evaluate the
        `objective_function <GradientOptimization.objective_function>`.

    objective_function : function or method
        specifies function used to evaluate `variable <GradientOptimization.variable>`
        in each iteration of the `optimization process  <GradientOptimization_Process>`;
        it must be specified and it must return a scalar value.

    direction : ASCENT or DESCENT : default ASCENT
        specifies the direction of gradient optimization: if *ASCENT*, movement is attempted in the positive direction
        (i.e., "up" the gradient);  if *DESCENT*, movement is attempted in the negative direction (i.e. "down"
        the gradient).

    step_size : int or float : default 1.0
        specifies the rate at which the `variable <GradientOptimization.variable>` is updated in each
        iteration of the `optimization process <GradientOptimization_Process>`;  if `annealing_function
        <GradientOptimization.annealing_function>` is specified, **step_size** specifies the intial value of
        `step_size <GradientOptimization.step_size>`.

    annealing_function : function or method : default None
        specifies function used to adapt `step_size <GradientOptimization.step_size>` in each
        iteration of the `optimization process <GradientOptimization_Process>`;  must take accept two parameters —
        `step_size <GradientOptimization.step_size>` and `iteration <GradientOptimization_Process>`, in that
        order — and return a scalar value, that is used for the next iteration of optimization.

    convergence_criterion : *VARIABLE* or *VALUE* : default *VALUE*
        specifies the parameter used to terminate the `optimization process <GradientOptimization_Process>`.
        *VARIABLE*: process terminates when the most recent sample differs from the previous one by less than
        `convergence_threshold <GradientOptimization.convergence_threshold>`;  *VALUE*: process terminates when the
        last value returned by `objective_function <GradientOptimization.objective_function>` differs from the
        previous one by less than `convergence_threshold <GradientOptimization.convergence_threshold>`.

    convergence_threshold : int or float : default 0.001
        specifies the change in value of `convergence_criterion` below which the optimization process is terminated.

    max_iterations : int : default 1000
        specifies the maximum number of times the `optimization process<GradientOptimization_Process>` is allowed to
        iterate; if exceeded, a warning is issued and the function returns the last sample evaluated.

    save_samples : bool
        specifies whether or not to save and return all of the samples used to evaluate `objective_function
        <GradientOptimization.objective_function>` in the `optimization process<GradientOptimization_Process>`.

    save_values : bool
        specifies whether or not to save and return the values of `objective_function
        <GradientOptimization.objective_function>` for all samples evaluated in the `optimization
        process<GradientOptimization_Process>`

    Attributes
    ----------

    variable : ndarray
        sample used as the starting point for the `optimization process <GradientOptimization_Process>` (i.e., one
        used to evaluate `objective_function <GradientOptimization.objective_function>` in the first iteration).

    objective_function : function or method
        function used to evaluate `variable <GradientOptimization.variable>`
        in each iteration of the `optimization process <GradientOptimization_Process>`;
        it must be specified and it must return a scalar value.

    gradient_function : function
        function used to compute the gradient in each iteration of the `optimization process
        <GradientOptimization_Process>` (see `Gradient Calculation <GradientOptimization_Gradient_Calculation>` for
        details).

    direction : ASCENT or DESCENT
        direction of gradient optimization:  if *ASCENT*, movement is attempted in the positive direction
        (i.e., "up" the gradient);  if *DESCENT*, movement is attempted in the negative direction (i.e. "down"
        the gradient).

    step_size : int or float
        determines the rate at which the `variable <GradientOptimization.variable>` is updated in each
        iteration of the `optimization process <GradientOptimization_Process>`;  if `annealing_function
        <GradientOptimization.annealing_function>` is specified, `step_size <GradientOptimization.step_size>`
        determines the initial value.

    annealing_function : function or method
        function used to adapt `step_size <GradientOptimization.step_size>` in each iteration of the `optimization
        process <GradientOptimization_Process>`;  if `None`, no call is made and the same `step_size
        <GradientOptimization.step_size>` is used in each iteration.

    iteration : int
        the currention iteration of the `optimization process <GradientOptimization_Process>`.

    convergence_criterion : VARIABLE or VALUE
        determines parameter used to terminate the `optimization process<GradientOptimization_Process>`.
        *VARIABLE*: process terminates when the most recent sample differs from the previous one by less than
        `convergence_threshold <GradientOptimization.convergence_threshold>`;  *VALUE*: process terminates when the
        last value returned by `objective_function <GradientOptimization.objective_function>` differs from the
        previous one by less than `convergence_threshold <GradientOptimization.convergence_threshold>`.

    convergence_threshold : int or float
        determines the change in value of `convergence_criterion` below which the `optimization process
        <GradientOptimization_Process>` is terminated.

    max_iterations : int
        determines the maximum number of times the `optimization process<GradientOptimization_Process>` is allowed to
        iterate; if exceeded, a warning is issued and the function returns the last sample evaluated.

    save_samples : bool
        determines whether or not to save and return all of the samples used to evaluate `objective_function
        <GradientOptimization.objective_function>` in the `optimization process<GradientOptimization_Process>`.

    save_values : bool
        determines whether or not to save and return the values of `objective_function
        <GradientOptimization.objective_function>` for all samples evaluated in the `optimization
        process<GradientOptimization_Process>`
    """

    componentName = GRADIENT_OPTIMIZATION_FUNCTION

    class Params(OptimizationFunction.Params):
        variable = Param([[0], [0]], read_only=True)

        # these should be removed and use switched to .get_previous()
        previous_variable = Param([[0], [0]], read_only=True)
        previous_value = Param([[0], [0]], read_only=True)

        annealing_function = Param(None, stateful=False, loggable=False)

        step_size = Param(1.0, modulable=True)
        convergence_threshold = Param(.001, modulable=True)
        max_iterations = Param(1000, modulable=True)

        direction = ASCENT
        convergence_criterion = VALUE

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 objective_function:tc.optional(is_function_type)=None,
                 direction:tc.optional(tc.enum(ASCENT, DESCENT))=ASCENT,
                 step_size:tc.optional(tc.any(int, float))=1.0,
                 annealing_function:tc.optional(is_function_type)=None,
                 convergence_criterion:tc.optional(tc.enum(VARIABLE, VALUE))=VALUE,
                 convergence_threshold:tc.optional(tc.any(int, float))=.001,
                 max_iterations:tc.optional(int)=1000,
                 save_samples:tc.optional(bool)=False,
                 save_values:tc.optional(bool)=False,
                 params=None,
                 owner=None,
                 prefs=None,
                 **kwargs):

        search_function = self._follow_gradient
        search_termination_function = self._convergence_condition
        self.gradient_function = None

        if direction is ASCENT:
            self.direction = 1
        else:
            self.direction = -1
        self.annealing_function = annealing_function

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(step_size=step_size,
                                                  convergence_criterion=convergence_criterion,
                                                  convergence_threshold=convergence_threshold,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         objective_function=objective_function,
                         search_function=search_function,
                         search_space=NotImplemented,
                         search_termination_function=search_termination_function,
                         max_iterations=max_iterations,
                         save_samples=save_samples,
                         save_values=save_values,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

    def reinitialize(self, *args):
        super().reinitialize(*args)
        if OBJECTIVE_FUNCTION in args[0]:
            try:
                from autograd import grad
                self.gradient_function = grad(self.objective_function)
            except:
                warnings.warn("Unable to use autograd with {} specified for {} Function: {}.".
                              format(repr(OBJECTIVE_FUNCTION), self.__class__.__name__,
                                     args[0][OBJECTIVE_FUNCTION].__name__))

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None,
                 **kwargs):
        '''Return the sample that yields the optimal value of `objective_function
        <GradientOptimization.objective_function>`, and possibly all samples evaluated and their corresponding values.

        Optimal value is defined by `direction <GradientOptimization.direction>`:
        - if *ASCENT*, returns greatest value
        - if *DESCENT*, returns least value

        Returns
        -------

        optimal sample, optimal value, saved_samples, saved_values : ndarray, list, list
            first array contains sample that yields the highest or lowest value of `objective_function
            <GradientOptimization.objective_function>`, depending on `direction <GradientOptimization.direction>`,
            and the second array contains the value of the function for that sample.
            If `save_samples <GradientOptimization.save_samples>` is `True`, first list contains all the values
            sampled in the order they were evaluated; otherwise it is empty.  If `save_values
            <GradientOptimization.save_values>` is `True`, second list contains the values returned by
            `objective_function <GradientOptimization.objective_function>` for all the samples in the order they were
            evaluated; otherwise it is empty.
        '''

        optimal_sample, optimal_value, all_samples, all_values = super().function(variable=variable,execution_id=execution_id, params=params, context=context)
        return_all_samples = return_all_values = []
        if self.parameters.save_samples.get(execution_id):
            return_all_samples = all_samples
        if self.parameters.save_values.get(execution_id):
            return_all_values = all_values
        # return last_variable
        return optimal_sample, optimal_value, return_all_samples, return_all_values

    def _follow_gradient(self, variable, sample_num, execution_id=None):

        if self.gradient_function is None:
            return variable

        # Update step_size
        if sample_num == 0:
            _current_step_size = self.parameters.step_size.get(execution_id)
        elif self.annealing_function:
            _current_step_size = call_with_pruned_args(self.annealing_function, self._current_step_size, sample_num, execution_id=execution_id)

        # Compute gradients with respect to current variable
        _gradients = call_with_pruned_args(self.gradient_function, variable, execution_id=execution_id)

        # Update variable based on new gradients
        return variable + self.parameters.direction.get(execution_id) * _current_step_size * np.array(_gradients)

    def _convergence_condition(self, variable, value, iteration, execution_id=None):
        previous_variable = self.parameters.previous_variable.get(execution_id)
        previous_value = self.parameters.previous_value.get(execution_id)

        if iteration is 0:
            # self._convergence_metric = self.convergence_threshold + EPSILON
            self.parameters.previous_variable.set(variable, execution_id, override=True)
            self.parameters.previous_value.set(value, execution_id, override=True)
            return True

        # Evaluate for convergence
        if self.convergence_criterion == VALUE:
            convergence_metric = np.abs(value - previous_value)
        else:
            convergence_metric = np.max(np.abs(np.array(variable) -
                                               np.array(previous_variable)))

        self.parameters.previous_variable.set(variable, execution_id, override=True)
        self.parameters.previous_value.set(value, execution_id, override=True)

        return convergence_metric > self.parameters.convergence_threshold.get(execution_id)


MAXIMIZE = 'maximize'


MINIMIZE = 'minimize'


class GridSearch(OptimizationFunction):
    """
    GridSearch(                      \
        default_variable=None,       \
        objective_function=None,     \
        direction=MAXIMIZE,          \
        max_iterations=1000,         \
        save_samples=False,          \
        save_values=False,           \
        params=None,                 \
        owner=None,                  \
        prefs=None                   \
        )

    Search over all samples in `search_space <GridSearch.search_space>` for the one that optimizes the value of
    `objective_function <GridSearch.objective_function>`.

    .. _GridSearch_Process:

    **Grid Search Process**

    When `function <GridSearch.function>` is executed, it iterates over the folowing steps:

        - get next sample from `search_space <GridSearch.search_space>`;
        ..
        - compute value of `objective_function <GridSearch.objective_function>` for that sample;

    The current iteration is contained in `iteration <GridSearch.iteration>`. Iteration continues until all values of
    `search_space <GridSearch.search_space>` have been evaluated, or `max_iterations <GridSearch.max_iterations>` is
    execeeded.  The function returns the sample that yielded either the highest (if `direction <GridSearch.direction>`
    is *MAXIMIZE*) or lowest (if `direction <GridSearch.direction>` is *MINIMIZE*) value of the `objective_function
    <GridSearch.objective_function>`, along with the value for that sample, as well as lists containing all of the
    samples evaluated and their values if either `save_samples <GridSearch.save_samples>` or `save_values
    <GridSearch.save_values>` is `True`, respectively.

    Arguments
    ---------

    default_variable : list or ndarray : default None
        specifies a template for (i.e., an example of the shape of) the samples used to evaluate the
        `objective_function <GridSearch.objective_function>`.

    objective_function : function or method
        specifies function used to evaluate sample in each iteration of the `optimization process <GridSearch_Process>`;
        it must be specified and must return a scalar value.

    search_space : list or array
        specifies samples used to evaluate `objective_function <GridSearch.objective_function>`.

    direction : MAXIMIZE or MINIMIZE : default MAXIMIZE
        specifies the direction of optimization:  if *MAXIMIZE*, the highest value of `objective_function
        <GridSearch.objective_function>` is sought;  if *MINIMIZE*, the lowest value is sought.

    max_iterations : int : default 1000
        specifies the maximum number of times the `optimization process<GridSearch_Process>` is allowed to iterate;
        if exceeded, a warning is issued and the function returns the optimal sample of those evaluated.

    save_samples : bool
        specifies whether or not to return all of the samples used to evaluate `objective_function
        <GridSearch.objective_function>` in the `optimization process <GridSearch_Process>`
        (i.e., a copy of the `search_space <GridSearch.search_space>`.

    save_values : bool
        specifies whether or not to save and return the values of `objective_function <GridSearch.objective_function>`
        for all samples evaluated in the `optimization process <GridSearch_Process>`.

    Attributes
    ----------

    variable : ndarray
        first sample evaluated by `objective_function <GridSearch.objective_function>` (i.e., one used to evaluate it
        in the first iteration of the `optimization process <GridSearch_Process>`).

    objective_function : function or method
        function used to evaluate sample in each iteration of the `optimization process <GridSearch_Process>`.

    search_space : list or array
        contains samples used to evaluate `objective_function <GridSearch.objective_function>` in iterations of the
        `optimization process <GridSearch_Process>`.

    direction : MAXIMIZE or MINIMIZE : default MAXIMIZE
        determines the direction of optimization:  if *MAXIMIZE*, the greatest value of `objective_function
        <GridSearch.objective_function>` is sought;  if *MINIMIZE*, the least value is sought.

    iteration : int
        the currention iteration of the `optimization process <GridSearch_Process>`.

    max_iterations : int
        determines the maximum number of times the `optimization process<GridSearch_Process>` is allowed to iterate;
        if exceeded, a warning is issued and the function returns the optimal sample of those evaluated.

    save_samples : True
        determines whether or not to save and return all samples evaluated by the `objective_function
        <GridSearch.objective_function>` in the `optimization process <GridSearch_Process>` (if the process
        completes, this should be identical to `search_space <GridSearch.search_space>`.

    save_values : bool
        determines whether or not to save and return the value of `objective_function
        <GridSearch.objective_function>` for all samples evaluated in the `optimization process <GridSearch_Process>`.
    """

    componentName = GRID_SEARCH_FUNCTION

    class Params(OptimizationFunction.Params):
        variable = Param([[0], [0]], read_only=True)

        # these are created as parameter states, but should they be?
        save_samples = Param(True, modulable=True)
        save_values = Param(True, modulable=True)

        direction = MAXIMIZE

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 objective_function:tc.optional(is_function_type)=None,
                 # search_space:tc.optional(tc.any(list, np.ndarray))=None,
                 search_space=None,
                 direction:tc.optional(tc.enum(MAXIMIZE, MINIMIZE))=MAXIMIZE,
                 save_values:tc.optional(bool)=False,
                 params=None,
                 owner=None,
                 prefs=None,
                 **kwargs):

        search_function = self._traverse_grid
        search_termination_function = self._grid_complete
        self._return_values = save_values

        self.direction = direction

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(params=params)

        super().__init__(default_variable=default_variable,
                         objective_function=objective_function,
                         search_function=search_function,
                         search_space=search_space,
                         search_termination_function=search_termination_function,
                         save_samples=True,
                         save_values=True,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None,
                 **kwargs):
        '''Return the sample that yields the optimal value of `objective_function <GridSearch.objective_function>`,
        and possibly all samples evaluated and their corresponding values.

        Optimal value is defined by `direction <GridSearch.direction>`:
        - if *MAXIMIZE*, returns greatest value
        - if *MINIMIZE*, returns least value

        Returns
        -------

        optimal sample, optimal value, saved_samples, saved_values : ndarray, list, list
            first array contains sample that yields the highest or lowest value of `objective_function
            <GridSearch.objective_function>`, depending on `direction <GridSearch.direction>`, and the
            second array contains the value of the function for that sample. If `save_samples
            <GridSearch.save_samples>` is `True`, first list contains all the values sampled in the order they were
            evaluated; otherwise it is empty.  If `save_values <GridSearch.save_values>` is `True`, second list
            contains the values returned by `objective_function <GridSearch.objective_function>` for all the samples
            in the order they were evaluated; otherwise it is empty.
        '''

        return_all_samples = return_all_values = []

        if MPI_IMPLEMENTATION:

            from mpi4py import MPI

            Comm = MPI.COMM_WORLD
            rank = Comm.Get_rank()
            size = Comm.Get_size()

            self.search_space = np.atleast_2d(self.search_space)

            chunk_size = (len(self.search_space) + (size-1)) // size
            start = chunk_size * rank
            end = chunk_size * (rank+1)
            if start > len(self.search_space):
                start = len(self.search_space)
            if end > len(self.search_space):
                end = len(self.search_space)

            # # TEST PRINT
            # print("\nContext: {}".format(self.context.flags_string))
            # print("search_space length: {}".format(len(self.search_space)))
            # print("Rank: {}\tSize: {}\tChunk size: {}".format(rank, size, chunk_size))
            # print("START: {0}\tEND: {1}\tPROCESSED: {2}".format(start,end,end-start))

            # FIX:  INITIALIZE TO FULL LENGTH AND ASSIGN DEFAULT VALUES (MORE EFFICIENT):
            samples = np.array([[]])
            sample_optimal = np.empty_like(self.search_space[0])
            values = np.array([])
            value_optimal = float('-Infinity')
            sample_value_max_tuple = (sample_optimal, value_optimal)

            # Set up progress bar
            _show_progress = False
            if hasattr(self, OWNER) and self.owner and self.owner.prefs.reportOutputPref:
                _show_progress = True
                _progress_bar_char = '.'
                _progress_bar_rate_str = ""
                _search_space_size = len(self.search_space)
                _progress_bar_rate = int(10 ** (np.log10(_search_space_size)-2))
                if _progress_bar_rate > 1:
                    _progress_bar_rate_str = str(_progress_bar_rate) + " "
                print("\n{} executing optimization process (one {} for each {}of {} samples): ".
                      format(self.owner.name, repr(_progress_bar_char), _progress_bar_rate_str, _search_space_size))
                _progress_bar_count = 0

            for sample in self.search_space[start:end,:]:

                if _show_progress:
                    increment_progress_bar = (_progress_bar_rate < 1) or not (_progress_bar_count % _progress_bar_rate)
                    if increment_progress_bar:
                        print(_progress_bar_char, end='', flush=True)
                    _progress_bar_count +=1

                # Evaluate objective_function for current sample
                value = self.objective_function(sample)

                # Evaluate for optimal value
                if self.direction is MAXIMIZE:
                    value_optimal = max(value, value_optimal)
                elif self.direction is MINIMIZE:
                    value_optimal = min(value, value_optimal)
                else:
                    assert False, "PROGRAM ERROR: bad value for {} arg of {}: {}".\
                        format(repr(DIRECTION),self.name,self.direction)

                # FIX: PUT ERROR HERE IF value AND/OR value_max ARE EMPTY (E.G., WHEN EXECUTION_ID IS WRONG)
                # If value is optimal, store corresponing sample
                if value == value_optimal:
                    # Keep track of state values and allocation policy associated with EVC max
                    sample_optimal = sample
                    sample_value_max_tuple = (sample_optimal, value_optimal)

                # Save samples and/or values if specified
                if self.save_values:
                    # FIX:  ASSIGN BY INDEX (MORE EFFICIENT)
                    values = np.append(values, np.atleast_1d(value), axis=0)
                if self.save_samples:
                    if len(samples[0])==0:
                        samples = np.atleast_2d(sample)
                    else:
                        samples = np.append(samples, np.atleast_2d(sample), axis=0)

            # Aggregate, reduce and assign global results
            # combine max result tuples from all processes and distribute to all processes
            max_tuples = Comm.allgather(sample_value_max_tuple)
            # get tuple with "value_max of maxes"
            max_value_of_max_tuples = max(max_tuples, key=lambda max_tuple: max_tuple[1])
            # get value_optimal, state values and allocation policy associated with "max of maxes"
            return_optimal_sample = max_value_of_max_tuples[0]
            return_optimal_value = max_value_of_max_tuples[1]

            # if self._return_samples:
            #     return_all_samples = np.concatenate(Comm.allgather(samples), axis=0)
            if self._return_values:
                return_all_values = np.concatenate(Comm.allgather(values), axis=0)

        else:
            last_sample, last_value, all_samples, all_values = super().function(
                variable=variable,
                execution_id=execution_id,
                params=params,
                context=context
            )
            return_optimal_value = max(all_values)
            return_optimal_sample = all_samples[all_values.index(return_optimal_value)]
            # if self._return_samples:
            #     return_all_samples = all_samples
            if self._return_values:
                return_all_values = all_values

        return return_optimal_sample, return_optimal_value, return_all_samples, return_all_values

    def _traverse_grid(self, variable, sample_num, execution_id=None):
        return self.parameters.search_space.get(execution_id)[sample_num]

    def _grid_complete(self, variable, value, iteration, execution_id=None):
        return iteration != len(self.parameters.search_space.get(execution_id))


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
    COMMENT

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

    class Params(Function_Base.Params):
        variable = Param(np.array([0, 0, 0]), read_only=True)
        learning_rate = Param(0.05, modulable=True)

    def _validate_learning_rate(self, learning_rate, type=None):

        learning_rate = np.array(learning_rate).copy()
        learning_rate_dim = learning_rate.ndim

        self._validate_parameter_spec(learning_rate, LEARNING_RATE)

        if type is AUTOASSOCIATIVE:

            if learning_rate_dim == 1 and len(learning_rate) != len(self.instance_defaults.variable):
                raise FunctionError("Length of {} arg for {} ({}) must be the same as its variable ({})".
                                    format(LEARNING_RATE, self.name, len(learning_rate),
                                           len(self.instance_defaults.variable)))

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


class Kohonen(LearningFunction):  # -------------------------------------------------------------------------------
    """
    Kohonen(                       \
        default_variable=None,     \
        learning_rate=None,        \
        distance_measure=GAUSSIAN, \
        params=None,               \
        name=None,                 \
        prefs=None)

    Implements a function that calculates a matrix of weight changes using the Kohenen (SOM) learning rule.
    This modifies the weights to each element in proportion to their difference from the current input pattern
    and the distance of that element from the one with the weights most similar to the current input pattern.

    Arguments
    ---------

    variable: List[array(float64), array(float64), 2d np.array[[float64]]] : default ClassDefaults.variable
        input pattern, array of activation values, and matrix used to calculate the weights changes.

    learning_rate : scalar or list, 1d or 2d np.array, or np.matrix of numeric values: default default_learning_rate
        specifies the learning rate used by the `function <Kohonen.function>`; supersedes any specification  for the
        `Process` and/or `System` to which the function's `owner <Function.owner>` belongs (see `learning_rate
        <Kohonen.learning_rate>` for details).

    distance_measure : GAUSSIAN, LINEAR, EXPONENTIAL, SINUSOID or function
        specifies the method used to calculate the distance of each element in `variable <Kohonen.variable>`\[2]
        from the one with the greatest value.

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

    variable: List[array(float64), array(float64), 2d np.array[[float64]]]
        input pattern, array of activation values, and weight matrix  used to generate the weight change matrix
        returned by `function <Kohonen.function>`.

    learning_rate : float, 1d or 2d np.array
        used by the `function <Kohonen.function>` to scale the weight change matrix returned by the `function
        <Kohonen.function>`.  If specified, it supersedes any learning_rate specified for the `Process
        <Process_Base_Learning>` and/or `System <System_Learning>` to which the function's `owner <Kohonen.owner>`
        belongs.  If it is a scalar, it is multiplied by the weight change matrix;  if it is a 1d np.array, it is
        multiplied Hadamard (elementwise) by the `variable` <Kohonen.variable>` before calculating the weight change
        matrix;  if it is a 2d np.array, it is multiplied Hadamard (elementwise) by the weight change matrix; if it is
        `None`, then the `learning_rate <Process.learning_rate>` specified for the Process to which the `owner
        <Kohonen.owner>` belongs is used;  and, if that is `None`, then the `learning_rate <System.learning_rate>`
        for the System to which it belongs is used. If all are `None`, then the `default_learning_rate
        <Kohonen.default_learning_rate>` is used.

    default_learning_rate : float
        the value used for the `learning_rate <Kohonen.learning_rate>` if it is not otherwise specified.

    function : function
         calculates a matrix of weight changes from: i) the difference between an input pattern (variable
         <Kohonen.variable>`\[0]) and the weights in a weigh matrix (`variable <Kohonen.variable>`\[2]) to each
         element of an activity array (`variable <Kohonen.variable>`\[1]); and ii) the distance of each element of
         the activity array (variable <Kohonen.variable>`\[1])) from the one with the weights most similar to the
         input array (variable <Kohonen.variable>`\[0])) using `distance_measure <Kohonen.distance_measure>`.

    owner : Component
        `Mechanism <Mechanism>` to which the Function belongs.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).    """

    componentName = KOHONEN_FUNCTION

    class Params(LearningFunction.Params):
        variable = Param([[0, 0], [0, 0], [[0, 0], [0, 0]]], read_only=True)
        distance_function = Param(GAUSSIAN, stateful=False)

        def _validate_distance_function(self, distance_function):
            options = {GAUSSIAN, LINEAR, EXPONENTIAL}
            if distance_function in options:
                # returns None indicating no error message (this is a valid assignment)
                return None
            else:
                # returns error message
                return 'not one of {0}'.format(options)

    default_learning_rate = 0.05

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    def __init__(self,
                 default_variable=None,
                 # learning_rate: tc.optional(parameter_spec) = None,
                 learning_rate=None,
                 distance_function:tc.any(tc.enum(GAUSSIAN, LINEAR, EXPONENTIAL), is_function_type)=GAUSSIAN,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(distance_function=distance_function,
                                                  learning_rate=learning_rate,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)


    def _validate_variable(self, variable, context=None):
        variable = super()._validate_variable(variable, context)

        # variable = np.squeeze(np.array(variable))

        name = self.name
        if self.owner and self.owner.name:
            name = name + " for {}".format(self.owner.name)

        if not is_numeric(variable):
            raise ComponentError("Variable for {} ({}) contains non-numeric entries".
                                 format(name, variable))

        if len(variable)!=3:
            raise FunctionError("variable for {} has {} items ({}) but must have three:  "
                                "input pattern (1d array), activity array (1d array) and matrix (2d array)"
                                "".format(name, len(variable), variable))

        input = np.array(variable[0])
        activity = np.array(variable[1])
        matrix = np.array(variable[2])

        if input.ndim != 1:
            raise FunctionError("First item of variable ({}) for {} must be a 1d array".
                                format(input, name))

        if activity.ndim != 1:
            raise FunctionError("Second item of variable ({}) for {} must be a 1d array".
                                format(activity, name))

        if matrix.ndim != 2:
            raise FunctionError("Third item of variable ({}) for {} must be a 2d array or matrix".
                                format(activity, name))

        if len(input) != len(activity):
            raise FunctionError("Length of first ({}) and second ({}) items of variable for {} must be the same".
                                format(len(input), len(activity), name))

        #     VALIDATE THAT len(variable[0])==len(variable[1])==len(variable[2].shape)
        if (len(input) != matrix.shape[0]) or (matrix.shape[0] != matrix.shape[1]):
            raise FunctionError("Third item of variable for {} ({}) must be a square matrix the dimension of which "
                                "must be the same as the length ({}) of the first and second items of the variable".
                                format(name, matrix, len(input)))

        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate learning_rate
        """
        super()._validate_params(request_set=request_set, target_set=target_set, context=context)
        if LEARNING_RATE in target_set and target_set[LEARNING_RATE] is not None:
            self._validate_learning_rate(target_set[LEARNING_RATE], AUTOASSOCIATIVE)

    def _instantiate_attributes_before_function(self, function=None, context=None):
        super()._instantiate_attributes_before_function(function, context)

        if isinstance(self.distance_function, str):
            self.measure=self.distance_function
            self.distance_function = scalar_distance

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """Calculate a matrix of weight changes from an array of activity values and a weight matrix that generated
        them using the Kohonen learning rule.

        The weight change matrix is calculated as:

           *learning_rate* * :math:`distance_j' * *variable[0]*-:math:`w_j`

        where :math:`distance_j` is the distance of the jth element of `variable <Kohonen.variable>`\[1] from the
        element with the weights most similar to activity array in `variable <Kohonen.variable>`\[1],
        and :math:`w_j` is the column of the matrix in `variable <Kohonen.variable>`\[2] that corresponds to
        the jth element of the activity array `variable <Kohonen.variable>`\[1].

        .. _note::
           the array of activities in `variable <Kohonen.variable>`\[1] is assumed to have been generated by the
           dot product of the input pattern in `variable <Kohonen.variable>`\[0] and the matrix in `variable
           <Kohonen.variable>`\[2], and thus the element with the greatest value in `variable <Kohonen.variable>`\[1]
           can be assumed to be the one with weights most similar to the input pattern.

        Arguments
        ---------

        variable : np.array or List[1d array, 1d array, 2d array] : default ClassDefaults.variable
           input pattern, array of activation values, and matrix used to calculate the weights changes.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the function.
            Values specified for parameters in the dictionary override any assigned to those parameters in arguments
            of the constructor.

        Returns
        -------

        weight change matrix : 2d np.array
            matrix of weight changes scaled by difference of the current weights from the input pattern and the
            distance of each element from the one with the weights most similar to the input pattern.

        """

        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        # IMPLEMENTATION NOTE: have to do this here, rather than in validate_params for the following reasons:
        #                      1) if no learning_rate is specified for the Mechanism, need to assign None
        #                          so that the process or system can see it is free to be assigned
        #                      2) if neither the system nor the process assigns a value to the learning_rate,
        #                          then need to assign it to the default value
        # If learning_rate was not specified for instance or composition, use default value
        learning_rate = self.get_current_function_param(LEARNING_RATE, execution_id)
        if learning_rate is None:
            learning_rate = self.defaults.learning_rate

        # FIX: SHOULD PUT THIS ON SUPER (THERE, BUT NEEDS TO BE DEBUGGED)
        learning_rate_dim = None
        if learning_rate is not None:
            learning_rate_dim = np.array(learning_rate).ndim

        # If learning_rate is a 1d array, multiply it by variable
        if learning_rate_dim == 1:
            variable = variable * learning_rate

        input_pattern = np.array(np.matrix(variable[0]).T)
        activities = np.array(np.matrix(variable[1]).T)
        matrix = variable[2]
        measure = self.distance_function

        # Calculate I-w[j]
        input_cols = np.repeat(input_pattern,len(input_pattern),1)
        differences = matrix - input_cols

        # Calculate distances
        index_of_max = list(activities).index(max(activities))
        distances = np.zeros_like(activities)
        for i, item in enumerate(activities):
            distances[i]=self.distance_function(self.measure, abs(i-index_of_max))
        distances = 1-np.array(np.matrix(distances).T)

        # Multiply distances by differences and learning_rate
        weight_change_matrix = distances * differences * learning_rate

        return self.convert_output_type(weight_change_matrix)


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

    class Params(LearningFunction.Params):
        variable = Param(np.array([0, 0]), read_only=True)

    default_learning_rate = 0.05

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    def __init__(self,
                 default_variable=None,
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


    def _validate_variable(self, variable, context=None):
        variable = super()._validate_variable(variable, context)

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
                 execution_id=None,
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
            <HebbianMechanism.learning_rate>`, with all diagonal elements = 0 (i.e., hollow matix).

        """

        self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        # IMPLEMENTATION NOTE: have to do this here, rather than in validate_params for the following reasons:
        #                      1) if no learning_rate is specified for the Mechanism, need to assign None
        #                          so that the process or system can see it is free to be assigned
        #                      2) if neither the system nor the process assigns a value to the learning_rate,
        #                          then need to assign it to the default value
        # If learning_rate was not specified for instance or composition, use default value
        learning_rate = self.get_current_function_param(LEARNING_RATE, execution_id)
        if learning_rate is None:
            learning_rate = self.defaults.learning_rate

        # FIX: SHOULD PUT THIS ON SUPER (THERE, BUT NEEDS TO BE DEBUGGED)
        learning_rate_dim = None
        if learning_rate is not None:
            learning_rate_dim = np.array(learning_rate).ndim

        # MODIFIED 9/21/17 NEW:
        # FIX: SHOULDN'T BE NECESSARY TO DO THIS;  WHY IS IT GETTING A 2D ARRAY AT THIS POINT?
        if not isinstance(variable, np.ndarray):
            variable = np.array(variable)
        if variable.ndim > 1:
            variable = np.squeeze(variable)
        # MODIFIED 9/21/17 END

        # If learning_rate is a 1d array, multiply it by variable
        if learning_rate_dim == 1:
            variable = variable * learning_rate

        # Generate the column array from the variable
        # col = variable.reshape(len(variable),1)
        col = np.array(np.matrix(variable).T)

        # Calculate weight chhange matrix
        weight_change_matrix = variable * col
        # Zero diagonals (i.e., don't allow correlation of a unit with itself to be included)
        weight_change_matrix = weight_change_matrix * (1 - np.identity(len(variable)))

        # If learning_rate is scalar or 2d, multiply it by the weight change matrix
        if learning_rate_dim in {0, 2}:
            weight_change_matrix = weight_change_matrix * learning_rate

        return self.convert_output_type(weight_change_matrix)


class ContrastiveHebbian(LearningFunction):  # -------------------------------------------------------------------------
    """
    ContrastiveHebbian(         \
        default_variable=None,  \
        learning_rate=None,     \
        params=None,            \
        name=None,              \
        prefs=None)

    Implements a function that calculates a matrix of weight changes using the `ContrastiveHebbian learning rule
    <https://www.sciencedirect.com/science/article/pii/B978148321448150007X>`_.

    Arguments
    ---------

    variable : List[number] or 1d np.array : default ClassDefaults.variable
       specifies the activation values, the pair-wise products of which are used to generate the a weight change matrix.

    COMMENT:
    activation_function : Function or function : SoftMax
        specifies the `function <Mechanism_Base.function>` of the `Mechanism` that generated the array of activations
        in `variable <ContrastiveHebbian.variable>`.
    COMMENT

    learning_rate : scalar or list, 1d or 2d np.array, or np.matrix of numeric values: default default_learning_rate
        specifies the learning rate used by the `function <ContrastiveHebbian.function>`; supersedes any specification
        for the `Process` and/or `System` to which the function's `owner <ContrastiveHebbian.owner>` belongs (see
        `learning_rate <ContrastiveHebbian.learning_rate>` for details).

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
        the `function <ContrastiveHebbian.function>`.

    COMMENT:
    activation_function : Function or function : SoftMax
        the `function <Mechanism_Base.function>` of the `Mechanism` that generated the array of activations in
        `variable <ContrastiveHebbian.variable>`.
    COMMENT

    learning_rate : float, 1d or 2d np.array
        used by the `function <ContrastiveHebbian.function>` to scale the weight change matrix returned by the `function
        <ContrastiveHebbian.function>`.  If specified, it supersedes any learning_rate specified for the `Process
        <Process_Base_Learning>` and/or `System <System_Learning>` to which the function's `owner
        <ContrastiveHebbian.owner>` belongs.  If it is a scalar, it is multiplied by the weight change matrix;  if it
        is a 1d np.array, it is multiplied Hadamard (elementwise) by the `variable` <ContrastiveHebbian.variable>`
        before calculating the weight change matrix;  if it is a 2d np.array, it is multiplied Hadamard (elementwise) by
        the weight change matrix; if it is `None`, then the `learning_rate <Process.learning_rate>` specified for the
        Process to which the `owner <ContrastiveHebbian.owner>` belongs is used;  and, if that is `None`, then the
        `learning_rate <System.learning_rate>` for the System to which it belongs is used. If all are `None`, then the
        `default_learning_rate <ContrastiveHebbian.default_learning_rate>` is used.

    default_learning_rate : float
        the value used for the `learning_rate <ContrastiveHebbian.learning_rate>` if it is not otherwise specified.

    function : function
         calculates the pairwise product of all elements in the `variable <ContrastiveHebbian.variable>`, and then
         scales that by the `learning_rate <ContrastiveHebbian.learning_rate>` to generate the weight change matrix
         returned by the function.

    owner : Component
        `Mechanism <Mechanism>` to which the Function belongs.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).    """

    componentName = CONTRASTIVE_HEBBIAN_FUNCTION

    class Params(LearningFunction.Params):
        variable = Param(np.array([0, 0]), read_only=True)

    default_learning_rate = 0.05

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    def __init__(self,
                 default_variable=None,
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


    def _validate_variable(self, variable, context=None):
        variable = super()._validate_variable(variable, context)

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
                 execution_id=None,
                 params=None,
                 context=None):
        """Calculate a matrix of weight changes from a 1d array of activity values using ContrastiveHebbian
        learning function.

        The weight change matrix is calculated as:

        COMMENT:
        THE FOLOWING NEEDS TO BE REPLACED WITH CONTRASTIVE HEBBIAN LEARNING RULE:

           *learning_rate* * :math:`a_ia_j` if :math:`i \\neq j`, else :math:`0`

        where :math:`a_i` and :math:`a_j` are elements of `variable <ContrastiveHebbian.variable>`.
        COMMENT

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
            matrix of pairwise products of elements of `variable <ContrastiveHebbian.variable>` scaled by the
            `learning_rate <ContrastiveHebbian.learning_rate>`, with all diagonal elements = 0 (i.e., hollow matix).

        """

        self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        # IMPLEMENTATION NOTE: have to do this here, rather than in validate_params for the following reasons:
        #                      1) if no learning_rate is specified for the Mechanism, need to assign None
        #                          so that the process or system can see it is free to be assigned
        #                      2) if neither the system nor the process assigns a value to the learning_rate,
        #                          then need to assign it to the default value
        # If learning_rate was not specified for instance or composition, use default value
        learning_rate = self.get_current_function_param(LEARNING_RATE, execution_id)
        if learning_rate is None:
            learning_rate = self.defaults.learning_rate

        # FIX: SHOULD PUT THIS ON SUPER (THERE, BUT NEEDS TO BE DEBUGGED)
        learning_rate_dim = None
        if learning_rate is not None:
            learning_rate_dim = np.array(learning_rate).ndim

        # MODIFIED 9/21/17 NEW:
        # FIX: SHOULDN'T BE NECESSARY TO DO THIS;  WHY IS IT GETTING A 2D ARRAY AT THIS POINT?
        if not isinstance(variable, np.ndarray):
            variable = np.array(variable)
        if variable.ndim > 1:
            variable = np.squeeze(variable)
        # MODIFIED 9/21/17 END

        # If learning_rate is a 1d array, multiply it by variable
        if learning_rate_dim == 1:
            variable = variable * learning_rate

        # IMPLEMENTATION NOTE:  THE FOLLOWING NEEDS TO BE REPLACED BY THE CONTRASTIVE HEBBIAN LEARNING RULE:

        # Generate the column array from the variable
        # col = variable.reshape(len(variable),1)
        col = np.array(np.matrix(variable).T)

        # Calculate weight chhange matrix
        weight_change_matrix = variable * col
        # Zero diagonals (i.e., don't allow correlation of a unit with itself to be included)
        weight_change_matrix = weight_change_matrix * (1 - np.identity(len(variable)))

        # If learning_rate is scalar or 2d, multiply it by the weight change matrix
        if learning_rate_dim in {0, 2}:
            weight_change_matrix = weight_change_matrix * learning_rate

        return self.convert_output_type(weight_change_matrix)


def _activation_input_getter(owning_component=None, execution_id=None):
    return owning_component.parameters.variable.get(execution_id)[LEARNING_ACTIVATION_INPUT]


def _activation_output_getter(owning_component=None, execution_id=None):
    return owning_component.parameters.variable.get(execution_id)[LEARNING_ACTIVATION_OUTPUT]


def _error_signal_getter(owning_component=None, execution_id=None):
    return owning_component.parameters.variable.get(execution_id)[LEARNING_ERROR_OUTPUT]


class Reinforcement(LearningFunction):  # -----------------------------------------------------------------------------
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

    default_variable : List or 2d np.array [length 3 in axis 0] : default ClassDefaults.variable
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

    class Params(LearningFunction.Params):
        variable = Param(np.array([[0], [0], [0]]), read_only=True)
        activation_input = Param([0], read_only=True, getter=_activation_input_getter)
        activation_output = Param([0], read_only=True, getter=_activation_output_getter)
        error_signal = Param([0], read_only=True, getter=_error_signal_getter)

    default_learning_rate = 0.05

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    def __init__(self,
                 default_variable=None,
                 # learning_rate: tc.optional(parameter_spec) = None,
                 learning_rate=None,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(  # activation_function=activation_function,
            learning_rate=learning_rate,
            params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

    @property
    def output_type(self):
        return self._output_type

    @output_type.setter
    def output_type(self, value):
        # disabled because it happens during normal execution, may be confusing
        # warnings.warn('output_type conversion disabled for {0}'.format(self.__class__.__name__))
        self._output_type = None

    def _validate_variable(self, variable, context=None):
        variable = super()._validate_variable(variable, context)

        if len(variable) != 3:
            raise ComponentError("Variable for {} ({}) must have three items (input, output and error arrays)".
                                 format(self.name, variable))

        if len(variable[LEARNING_ERROR_OUTPUT]) != 1:
            raise ComponentError("Error term for {} (the third item of its variable arg) must be an array with a "
                                 "single element for {}".
                                 format(self.name, variable[LEARNING_ERROR_OUTPUT]))

        # Allow initialization with zero but not during a run (i.e., when called from check_args())
        if self.context.initialization_status != ContextFlags.INITIALIZING:
            if np.count_nonzero(variable[LEARNING_ACTIVATION_OUTPUT]) != 1:
                raise ComponentError(
                    "Second item ({}) of variable for {} must be an array with a single non-zero value "
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
                 execution_id=None,
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

        self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        output = self.get_current_function_param('activation_output', execution_id)
        error = self.get_current_function_param('error_signal', execution_id)
        learning_rate = self.get_current_function_param(LEARNING_RATE, execution_id)
        # IMPLEMENTATION NOTE: have to do this here, rather than in validate_params for the following reasons:
        #                      1) if no learning_rate is specified for the Mechanism, need to assign None
        #                          so that the process or system can see it is free to be assigned
        #                      2) if neither the system nor the process assigns a value to the learning_rate,
        #                          then need to assign it to the default value
        # If learning_rate was not specified for instance or composition, use default value
        if learning_rate is None:
            learning_rate = self.defaults.learning_rate

        # Assign error term to chosen item of output array
        error_array = (np.where(output, learning_rate * error, 0))

        # Construct weight change matrix with error term in proper element
        weight_change_matrix = np.diag(error_array)

        return [error_array, error_array]


class BayesGLM(LearningFunction):
    """
    BayesGLM(                   \
        default_variable=None,  \
        mu_0=0,                 \
        sigma_0=1,              \
        gamma_shape_0=1,        \
        gamma_size_0=1,         \
        params=None,            \
        prefs=None)

    Implements Bayesian linear regression that fits means and distributions of weights to predict dependent variable(s)
    in `variable <BayesGLM.variable>`\\[1] from predictor vector(s) in `variable <BayesGLM.variable>`\\[0].

    Uses a normal linear model variable[1] = variable[0]\Theta + \epsilon, with normal-gamma prior distribution
    and returns a vector of prediction weights sampled from the multivariate normal-gamma distribution.
    [Based on Falk Lieder's BayesianGLM.m, adapted for Python by Yotam Sagiv, and for PsyNeuLink by Jon Cohen;
    useful reference: `Bayesian Inference <http://www2.stat.duke.edu/~sayan/Sta613/2017/read/chapter_9.pdf>`_.]

    .. hint::
       The **mu_0** or **sigma_0** arguments of the consructor can be used in place of **default_variable** to define
       the size of the predictors array and, correspondingly, the weights array returned by the function (see
       **Parameters** below).

    Arguments
    ---------

    default_variable : 3d array : default None
        first item of axis 0 must be a 2d array with one or more 1d arrays to use as predictor vectors, one for
        each sample to be fit;  second item must be a 2d array of equal length to the first item, with a 1d array
        containing a scalar that is the dependent (to-be-predicted) value for the corresponding sample in the first
        item.  If `None` is specified, but either **mu_0** or **sigma_0 is specified, then the they are used to
        determine the shape of `variable <BayesGLM.variable>`.  If neither **mu_0** nor **sigma_0** are specified,
        then the shape of `variable <BayesGLM.variable>` is determined by the first call to its `function
        <BayesGLM.function>`, as are `mu_prior <BayesGLM.mu_prior>`, `sigma_prior <BayesGLM.mu_prior>`,
        `gamma_shape_prior <BayesGLM.gamma_shape_prior>` and `gamma_size_prior <BayesGLM.gamma_size_prior>`.

    mu_0 : int, float or 1d array : default 0
        specifies initial value of `mu_prior <BayesGLM.mu_prior>` (the prior for the mean of the distribution for
        the prediction weights returned by the function).  If a scalar is specified, the same value will be used
        for all elements of `mu_prior <BayesGLM.mu_prior>`;  if it is an array, it must be the same length as
        the predictor array(s) in axis 0 of **default_variable**.  If **default_variable** is not specified, the
        specification for **mu_0** is used to determine the shape of `variable <BayesGLM.variable>` and
        `sigma_prior <BayesGLM.sigma_prior>`.

    sigma_0 : int, float or 1d array : default 0
        specifies initial value of `sigma_prior <BayesGLM.Lambda_prior>` (the prior for the variance of the distribution
        for the prediction weights returned by the function).  If a scalar is specified, the same value will be used for
        all elements of `Lambda_prior <BayesGLM.Lambda_prior>`;  if it is an array, it must be the same length as the
        predictor array(s) in axis 0 of **default_variable**.  If neither **default_variable** nor **mu_0** is
        specified, the specification for **sigma_0** is used to determine their shapes.

    gamma_shape_0 : int or float : default 1
        specifies the shape of the gamma distribution from which samples of the weights are drawn (see documentation
        for `numpy.random.gamma <https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.gamma.html>`_.

    gamma_size_0 : int or float : default 1
        specifies the size of the gamma distribution from which samples of the weights are drawn (see documentation for
        `numpy.random.gamma <https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.gamma.html>`_.

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

    variable : 3d array
        samples used to update parameters of prediction weight distributions.
        variable[0] is a 2d array of predictor vectors, all of the same length;
        variable[1] is a 2d array of scalar dependent variables, one for each predictor vector.

    mu_0 : int, float or 2d np.array
        determines the initial prior(s) for the means of the distributions of the prediction weights;
        if it is a scalar, that value is assigned as the priors for all means.

    mu_prior : 2d np.array
        current priors for the means of the distributions of the predictions weights.

    mu_n : 2d np.array
        current means for the distributions of the prediction weights.

    sigma_0 : int, float or 2d np.array
        value used to determine the initial prior(s) for the variances of the distributions of the prediction
        weights; if it is a scalar, that value is assigned as the priors for all variances.

    Lambda_prior :  2d np.array
        current priors for the variances of the distributions of the predictions weights.

    Lambda_n :  2d np.array
        current variances for the distributions of the prediction weights.

    gamma_shape_0 : int or float
        determines the initial value used for the shape parameter of the gamma distribution used to sample the
        prediction weights.

    gamma_shape_prior : int or float
        current prior for the shape parameter of the gamma distribution used to sample the prediction weights.

    gamma_shape_n : int or float
        current value of the shape parameter of the gamma distribution used to sample the prediction weights.

    gamma_size_0 : int or float
        determines the initial value used for the size parameter of the gamma distribution used to sample the
        prediction weights.

    gamma_size_prior : int or float
        current prior for the size parameter of the gamma distribution used to sample the prediction weights.

    gamma_size_n : 2d array with single scalar value
        current value of the size parameter of the gamma distribution used to sample the prediction weights.

    function : function
        updates mean (`mu_n <BayesGLM.mu_n>`) and variance (`Lambda_n <BayesGLM.Lambda_n>`) of weight distributions
        to improve prediction of of dependent variable sample(s) in `variable <BayesGLM.variable>`\\[1] from
        predictor vector(s) in `variable <BayesGLM.variable>`\\[1].  Returns a vector of weights `weights_sample
        <BayesGLM.weights_sample>`) sampled from the weight disributions.

    weights_sample : 1d np.array
        last sample of prediction weights drawn in call to `sample_weights <BayesGLM.sample_weights>` and returned by
        `function <BayesGLM.function>`.

    owner : Component
        `Mechanism <Mechanism>` to which the Function belongs.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).
    """
    class Params(LearningFunction.Params):
        variable = Param([np.array([0, 0, 0]), np.array([0])], read_only=True)
        value = Param(np.array([0]), read_only=True, aliases=['sample_weights'])

        Lambda_0 = 0
        Lambda_prior = 0
        Lambda_n = 0

        mu_0 = 0
        mu_prior = 0
        mu_n = 0

        sigma_0 = 1

        gamma_shape_0 = 1
        gamma_shape_n = 1
        gamma_shape_prior = 1

        gamma_size_0 = 1
        gamma_size_n = 1
        gamma_size_prior = 1

    def __init__(self,
                 default_variable = None,
                 mu_0=0,
                 sigma_0=1,
                 gamma_shape_0=1,
                 gamma_size_0=1,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):

        self.user_specified_default_variable = default_variable

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(mu_0=mu_0,
                                                  sigma_0=sigma_0,
                                                  gamma_shape_0=gamma_shape_0,
                                                  gamma_size_0=gamma_size_0,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

    def _handle_default_variable(self, default_variable=None, size=None):

        # If default_variable was not specified by user...
        if default_variable is None and size in {None, NotImplemented}:
            #  but mu_0 and/or sigma_0 was specified as an array...
            if isinstance(self.mu_0, (list, np.ndarray)) or isinstance(self.sigma_0, (list, np.ndarray)):
                # if both are specified, make sure they are the same size
                if (isinstance(self.mu_0, (list, np.ndarray))
                        and isinstance(self.sigma_0, (list, np.ndarray))
                        and len(self.mu_0) != len(self.self.sigma_0)):
                    raise FunctionError("Length of {} ({}) does not match length of {} ({}) for {}".
                                        format(repr('mu_0'), len(self.mu_0),
                                                    repr('sigma_0'), len(self.self.sigma_0),
                                                         self.__class.__.__name__))
                # allow their size to determine the size of variable
                if isinstance(self.mu_0, (list, np.ndarray)):
                    default_variable = [np.zeros_like(self.mu_0), np.zeros((1,1))]
                else:
                    default_variable = [np.zeros_like(self.sigma_0), np.zeros((1,1))]

        return super()._handle_default_variable(default_variable=default_variable, size=size)

    def initialize_priors(self):
        '''Set the prior parameters (`mu_prior <BayesGLM.mu_prior>`, `Lamba_prior <BayesGLM.Lambda_prior>`,
        `gamma_shape_prior <BayesGLM.gamma_shape_prior>`, and `gamma_size_prior <BayesGLM.gamma_size_prior>`)
        to their initial (_0) values, and assign current (_n) values to the priors'''

        variable = np.array(self.instance_defaults.variable)
        variable = self.instance_defaults.variable
        if np.array(variable).dtype != object:
            variable = np.atleast_2d(variable)

        n = len(variable[0])

        if isinstance(self.mu_0, (int, float)):
            self.mu_prior = np.full((n, 1),self.mu_0)
        else:
            if len(self.mu_0) != n:
                raise FunctionError("Length of mu_0 ({}) does not match number of predictors ({})".
                                    format(len(self.mu_0), n))
            self.mu_prior = np.array(self.mu_0).reshape(len(self._mu_0),1)

        if isinstance(self.sigma_0, (int, float)):
            Lambda_0 = (1 / (self.sigma_0 ** 2)) * np.eye(n)
        else:
            if len(self.sigma_0) != n:
                raise FunctionError("Length of sigma_0 ({}) does not match number of predictors ({})".
                                    format(len(self.sigma_0), n))
            Lambda_0 = (1 / (np.array(self.sigma_0) ** 2)) * np.eye(n)
        self.Lambda_prior = Lambda_0

        # before we see any data, the posterior is the prior
        self.mu_n = self.mu_prior
        self.Lambda_n = self.Lambda_prior
        self.gamma_shape_n = self.gamma_shape_0
        self.gamma_size_n = self.gamma_size_0

    def reinitialize(self, *args):
        # If variable passed during execution does not match default assigned during initialization,
        #    reassign default and re-initialize priors
        if DEFAULT_VARIABLE in args[0]:
            self.instance_defaults.variable = np.array([np.zeros_like(args[0][DEFAULT_VARIABLE][0]),
                                                        np.zeros_like(args[0][DEFAULT_VARIABLE][1])])
            self.initialize_priors()

    def function(
        self,
        variable=None,
        execution_id=None,
        params=None,
        context=None
    ):
        '''Use predictor(s) and dependent variable(s) in `variable <BayesGLM.variable>` to update weight distribution
        parameters `mu_n <BayesGLM.mu_n>`, `Lambda_n <BayesGLM.Lambda_n>`, `gamma_shape_n <BayesGLM.gamma_shape_n>`,
        and `gamma_size_n <BayesGLM.gamma_size_n>`, and return an array of weights sampled from the distributions.

        Arguments
        ---------

        variable : 2d or 3d array : default ClassDefaults.variable
           if it is a 2d array, the first item must be a 1d array of scalar predictors, and the second must
           be a 1d array containing the dependent variable to be predicted by the predictors;
           if it is a 3d array, the first item in the outermost dimension must be 2d array containing one or more
           1d arrays of scalar predictors, and the second item be a 2d array containing 1d arrays each of which
           contains a scalar dependent variable for the corresponding predictor vector.

        params : Dict[param keyword: param value] : default None
           a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
           function.  Values specified for parameters in the dictionary override any assigned to those parameters in
           arguments of the constructor.

        Returns
        -------

        sample weights : 1d np.array
            array of weights drawn from updated weight distributions.

        '''

        if self.parameters.context.get(execution_id).initialization_status == ContextFlags.INITIALIZING:
            self.initialize_priors()

        # # MODIFIED 10/26/18 OLD:
        # # If variable passed during execution does not match default assigned during initialization,
        # #    reassign default and re-initialize priors
        # elif np.array(variable).shape != self.instance_defaults.variable.shape:
        #     self.instance_defaults.variable = np.array([np.zeros_like(variable[0]),np.zeros_like(variable[1])])
        #     self.initialize_priors()
        # MODIFIED 10/26/18 END

        # Today's prior is yesterday's posterior
        Lambda_prior = self.get_current_function_param('Lambda_n', execution_id)
        mu_prior = self.get_current_function_param('mu_n', execution_id)
        gamma_shape_prior = self.get_current_function_param('gamma_shape_n', execution_id)
        gamma_size_prior = self.get_current_function_param('gamma_size_n', execution_id)

        variable = self._check_args(
            [np.atleast_2d(variable[0]), np.atleast_2d(variable[1])],
            execution_id,
            params,
            context
        )
        predictors = variable[0]
        dependent_vars = variable[1]

        # online update rules as per the given reference
        Lambda_n = (predictors.T @ predictors) + Lambda_prior
        mu_n = np.linalg.inv(Lambda_n) @ ((predictors.T @ dependent_vars) + (Lambda_prior @ mu_prior))
        gamma_shape_n = gamma_shape_prior + dependent_vars.shape[1]
        gamma_size_n = gamma_size_prior + (dependent_vars.T @ dependent_vars) \
            + (mu_prior.T @ Lambda_prior @ mu_prior) \
            - (mu_n.T @ Lambda_n @ mu_n)

        self.parameters.Lambda_prior.set(Lambda_prior, execution_id)
        self.parameters.mu_prior.set(mu_prior, execution_id)
        self.parameters.gamma_shape_prior.set(gamma_shape_prior, execution_id)
        self.parameters.gamma_size_prior.set(gamma_size_prior, execution_id)

        self.parameters.Lambda_n.set(Lambda_n, execution_id)
        self.parameters.mu_n.set(mu_n, execution_id)
        self.parameters.gamma_shape_n.set(gamma_shape_n, execution_id)
        self.parameters.gamma_size_n.set(gamma_size_n, execution_id)

        return self.sample_weights(gamma_shape_n, gamma_size_n, mu_n, Lambda_n)

    def sample_weights(self, gamma_shape_n, gamma_size_n, mu_n, Lambda_n):
        '''Draw a sample of prediction weights from the distributions parameterized by `mu_n <BayesGLM.mu_n>`,
        `Lambda_n <BayesGLM.Lambda_n>`, `gamma_shape_n <BayesGLM.gamma_shape_n>`, and `gamma_size_n
        <BayesGLM.gamma_size_n>`.'''
        phi = np.random.gamma(gamma_shape_n / 2, gamma_size_n / 2)
        return np.random.multivariate_normal(mu_n.reshape(-1,), phi * np.linalg.inv(Lambda_n))


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

    class Params(LearningFunction.Params):
        variable = Param(np.array([[0], [0], [0]]), read_only=True)
        learning_rate = Param(1.0, modulable=True)

        activation_input = Param([0], read_only=True, getter=_activation_input_getter)
        activation_output = Param([0], read_only=True, getter=_activation_output_getter)
        error_signal = Param([0], read_only=True, getter=_error_signal_getter)

        error_matrix = Param(None, read_only=True)

    default_learning_rate = 1.0

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 activation_derivative_fct: tc.optional(tc.any(function_type, method_type)) = Logistic().derivative,
                 # learning_rate: tc.optional(parameter_spec) = None,
                 learning_rate=None,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):

        error_matrix = np.zeros((len(default_variable[LEARNING_ACTIVATION_OUTPUT]),
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

    @property
    def output_type(self):
        return self._output_type

    @output_type.setter
    def output_type(self, value):
        # disabled because it happens during normal execution, may be confusing
        # warnings.warn('output_type conversion disabled for {0}'.format(self.__class__.__name__))
        self._output_type = None

    def _validate_variable(self, variable, context=None):
        variable = super()._validate_variable(variable, context)

        if len(variable) != 3:
            raise ComponentError("Variable for {} ({}) must have three items: "
                                 "activation_input, activation_output, and error_signal)".
                                 format(self.name, variable))

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

            from psyneulink.core.components.states.parameterstate import ParameterState
            from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
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
            activity_output_len = len(self.defaults.variable[LEARNING_ACTIVATION_OUTPUT])
            error_signal_len = len(self.defaults.variable[LEARNING_ERROR_OUTPUT])

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
                 execution_id=None,
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

        self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        # Manage error_matrix param
        # During init, function is called directly from Component (i.e., not from LearningMechanism execute() method),
        #     so need "placemarker" error_matrix for validation
        if error_matrix is None:
            if self.parameters.context.get(execution_id).initialization_status == ContextFlags.INITIALIZING:
                error_matrix = np.zeros(
                    (len(variable[LEARNING_ACTIVATION_OUTPUT]), len(variable[LEARNING_ERROR_OUTPUT]))
                )
            # Raise exception if error_matrix is not specified
            else:
                owner_string = ""
                if self.owner:
                    owner_string = " of " + self.owner.name
                raise FunctionError("Call to {} function{} must include \'ERROR_MATRIX\' in params arg".
                                    format(self.__class__.__name__, owner_string))

        self.parameters.error_matrix.set(error_matrix, execution_id, override=True)
        # self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        # Manage learning_rate
        # IMPLEMENTATION NOTE: have to do this here, rather than in validate_params for the following reasons:
        #                      1) if no learning_rate is specified for the Mechanism, need to assign None
        #                          so that the process or system can see it is free to be assigned
        #                      2) if neither the system nor the process assigns a value to the learning_rate,
        #                          then need to assign it to the default value
        # If learning_rate was not specified for instance or composition, use default value
        learning_rate = self.get_current_function_param(LEARNING_RATE, execution_id)
        if learning_rate is None:
            learning_rate = self.defaults.learning_rate

        # make activation_input a 1D row array
        activation_input = self.get_current_function_param('activation_input', execution_id)
        activation_input = np.array(activation_input).reshape(len(activation_input), 1)

        # Derivative of error with respect to output activity (contribution of each output unit to the error above)
        dE_dA = np.dot(error_matrix, self.get_current_function_param('error_signal', execution_id))

        # Derivative of the output activity
        activation_output = self.get_current_function_param('activation_output', execution_id)
        dA_dW = self.activation_derivative_fct(input=activation_input, output=activation_output)

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

    def __init__(self,
                 default_variable=None,
                 learning_rate=None,
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
        variable = super(Reinforcement, self)._validate_variable(variable, context)

        if len(variable) != 3:
            raise ComponentError("Variable for {} ({}) must have three items "
                                 "(input, output, and error arrays)".format(self.name,
                                                                            variable))

        if len(variable[LEARNING_ERROR_OUTPUT]) != len(variable[LEARNING_ACTIVATION_OUTPUT]):
            raise ComponentError("Error term does not match the length of the"
                                 "sample sequence")

        return variable


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

# endregion
