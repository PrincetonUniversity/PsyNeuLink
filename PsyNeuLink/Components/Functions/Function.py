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

TransferMechanism Functions:
  * `Linear`
  * `Exponential`
  * `Logistic`
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
  * `BogaczEtAl`
  * `NavarroAndFuss`

Distribution Functions:
  * `NormalDist`
  * `ExponentialDist`
  * `UniformDist`
  * `GammaDist`
  * `WaldDist`

Objective Functions:
  * `Stability`
  * `Distance`

Learning Functions:
  * `Reinforcement`
  * `BackPropagation`

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
Each of these is assigned the name of one of the function's parameters. These are used by `ModulatoryProjections
<ModulatoryProjection>` to modulate the output of the function.  For example, they are used by `GatingProjections
<GatingProjection>` to modulate the `function <State_Base.function>` (and thereby the `value <State_Base.value>`) of
an `InputState` or `OutputState`; and by the `ControlProjection(s) <ControlProjection>` of an `LCMechanism` to
modulate the `function <TransferMechanism.function>` of a `TransferMechanism`.


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

# __all__ = ['Reduce',
#            'LinearCombination',
#            'Linear',
#            'Exponential',
#            'Logistic',
#            'SoftMax',
#            'Integrator',
#            'LinearMatrix',
#            'NormalDist',
#            'ExponentialDist',
#            'UniformDist`',
#            'GammaDist',
#            'WaldDist',
#            'Stability`,
#            'Distance`,
#            'Reinforcement',
#            'BackPropagation',
#            'FunctionError',
#            "FunctionOutputType"]
import numbers
import warnings

from collections import namedtuple
from enum import Enum, IntEnum
from random import randint

import numpy as np
import typecheck as tc

from numpy import abs, exp, tanh

from PsyNeuLink.Components.Component import Component, ComponentError, function_type, method_type, parameter_keywords
from PsyNeuLink.Components.ShellClasses import Function
from PsyNeuLink.Globals.Keywords import FHN_INTEGRATOR_FUNCTION, ACCUMULATOR_INTEGRATOR_FUNCTION, \
    ADAPTIVE_INTEGRATOR_FUNCTION, ALL, ANGLE, \
    ARGUMENT_THERAPY_FUNCTION, AUTO_ASSIGN_MATRIX, AUTO_DEPENDENT, BACKPROPAGATION_FUNCTION, BETA, BIAS, \
    COMBINATION_FUNCTION_TYPE, CONSTANT_INTEGRATOR_FUNCTION, CORRELATION, CROSS_ENTROPY, \
    DECAY, DIFFERENCE, DISTANCE_FUNCTION, DISTANCE_METRICS, DIST_FUNCTION_TYPE, DIST_MEAN, DIST_SHAPE, \
    DRIFT_DIFFUSION_INTEGRATOR_FUNCTION, ENERGY, ENTROPY, EUCLIDEAN, EXAMPLE_FUNCTION_TYPE, EXECUTING, \
    EXPONENTIAL_DIST_FUNCTION, EXPONENTIAL_FUNCTION, EXPONENTS, FULL_CONNECTIVITY_MATRIX, FUNCTION, \
    FUNCTION_OUTPUT_TYPE, FUNCTION_OUTPUT_TYPE_CONVERSION, FUNCTION_PARAMS, GAIN, GAMMA_DIST_FUNCTION, \
    HIGH, HOLLOW_MATRIX, IDENTITY_MATRIX, INCREMENT, INITIALIZER, INITIALIZING, INPUT_STATES, INTEGRATOR_FUNCTION, \
    INTEGRATOR_FUNCTION_TYPE, INTERCEPT, LEARNING_FUNCTION_TYPE, LINEAR_COMBINATION_FUNCTION, LINEAR_FUNCTION, \
    LINEAR_MATRIX_FUNCTION, LOGISTIC_FUNCTION, LOW, MATRIX, MATRIX_KEYWORD_NAMES, MATRIX_KEYWORD_VALUES, \
    MAX_INDICATOR, MAX_VAL, NOISE, NORMAL_DIST_FUNCTION, OBJECTIVE_FUNCTION_TYPE, OFFSET, OPERATION, \
    ORNSTEIN_UHLENBECK_INTEGRATOR_FUNCTION, OUTPUT_STATES, OUTPUT_TYPE, PARAMETER_STATE_PARAMS, PEARSON, \
    PROB, PRODUCT, RANDOM_CONNECTIVITY_MATRIX, RATE, RECEIVER, REDUCE_FUNCTION, RL_FUNCTION, SCALE, \
    SIMPLE_INTEGRATOR_FUNCTION, SLOPE, SOFTMAX_FUNCTION, STABILITY_FUNCTION, STANDARD_DEVIATION, SUM, \
    TIME_STEP_SIZE, TRANSFER_FUNCTION_TYPE, UNIFORM_DIST_FUNCTION, USER_DEFINED_FUNCTION, USER_DEFINED_FUNCTION_TYPE, \
    WALD_DIST_FUNCTION, WEIGHTS, kwComponentCategory, kwPreferenceSetName
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set, kpReportOutputPref, kpRuntimeParamStickyAssignmentPref
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceEntry, PreferenceLevel
from PsyNeuLink.Globals.Registry import register_category
from PsyNeuLink.Globals.Utilities import AutoNumber, is_distance_metric, is_matrix, is_numeric, iscompatible, np_array_less_than_2d, parameter_spec
from PsyNeuLink.Scheduling.TimeScale import TimeScale

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
    elif isinstance(x, (Function, function_type)):
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

from PsyNeuLink.Components.Projections.ModulatoryProjections.ModulatoryProjection import ModulatoryProjection_Base
@tc.typecheck
def _get_modulated_param(owner, mod_proj:ModulatoryProjection_Base):
    """Return ModulationParam object, function param name and value of param modulated by ModulatoryProjection
    """

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
        return owner.paramsCurrent[FUNCTION].keyword(owner, keyword)
    except FunctionError as e:
        # assert(False)
        if owner.prefs.verbosePref:
            print("{} of {}".format(e, owner.name))
        # return None
        else:
            raise FunctionError(e)
    except AttributeError:
        if owner.prefs.verbosePref:
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

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


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

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the **prefs** argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentCategory = kwComponentCategory
    className = componentCategory
    suffix = " " + className

    registry = FunctionRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY

    variableClassDefault_locked = False

    # Note: the following enforce encoding as 1D np.ndarrays (one array per variable)
    variableEncodingDim = 1

    paramClassDefaults = Component.paramClassDefaults.copy()
    paramClassDefaults.update({
        FUNCTION_OUTPUT_TYPE_CONVERSION: False,  # Enable/disable output type conversion
        FUNCTION_OUTPUT_TYPE:None                # Default is to not convert
    })

    def __init__(self,
                 default_variable,
                 params,
                 owner=None,
                 name=None,
                 prefs=None,
                 context='Function_Base Init'):
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

        self._functionOutputType = None
        # self.name = self.componentName

        register_category(entry=self,
                          base_class=Function_Base,
                          registry=FunctionRegistry,
                          name=name,
                          context=context)
        self.owner = owner
        if self.owner is not None:
            self.owner_name = ' ' + self.owner.name
        else:
            self.owner_name = ''

        super().__init__(default_variable=default_variable,
                         param_defaults=params,
                         name=name,
                         prefs=prefs,
                         context=context)

    def execute(self, variable=None, params=None, context=None):
        return self.function(variable=variable, params=params, context=context)

    @property
    def functionOutputType(self):
        # # MODIFIED 6/11/17 OLD:
        # if self.paramsCurrent[FUNCTION_OUTPUT_TYPE_CONVERSION]:
        # MODIFIED 6/11/17 NEW:
        if hasattr(self, FUNCTION_OUTPUT_TYPE_CONVERSION):
        # MODIFIED 6/11/17 END
            return self._functionOutputType
        return None

    @functionOutputType.setter
    def functionOutputType(self, value):

        # Initialize backing field if it has not yet been set
        #    ??or if FunctionOutputTypeConversion is False?? <- FIX: WHY?? [IS THAT A SIDE EFFECT OR PREVIOUSLY USING
        #                                                       FIX: self.paramsCurrent[FUNCTION_OUTPUT_TYPE_CONVERSION]
        #                                                       FIX: TO DECIDE IF ATTRIBUTE EXISTS?
        # # MODIFIED 6/11/17 OLD:
        # if not value and not self.paramsCurrent[FUNCTION_OUTPUT_TYPE_CONVERSION]:
        # MODIFIED 6/11/17 NEW:
        if value is None and (not hasattr(self, FUNCTION_OUTPUT_TYPE_CONVERSION)
                              or not self.FunctionOutputTypeConversion):
        # MODIFIED 6/11/17 END
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

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    variable : boolean
        assertion to which a therapeutic response is made.

    propensity : Manner value : default Manner.CONTRARIAN
        determines therapeutic manner:  tendency to agree or disagree.

    pertinacity : float : default 10.0
        determines consistency with which the manner complies with the propensity.

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the **prefs** argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    # Function componentName and type (defined at top of module)
    componentName = ARGUMENT_THERAPY_FUNCTION
    componentType = EXAMPLE_FUNCTION_TYPE

    class ClassDefaults(Function_Base.ClassDefaults):
        variable = 0

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
                 default_variable=ClassDefaults.variable,
                 propensity=10.0,
                 pertincacity=Manner.CONTRARIAN,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context=componentName + INITIALIZING):

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
                         context=context)

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
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """
        Returns a boolean that is (or tends to be) the same as or opposite the one passed in.

        Arguments
        ---------

        variable : boolean : default ClassDefaults.variable
           an assertion to which a therapeutic response is made.

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------

        therapeutic response : boolean

        """
        variable = self._update_variable(self._update_variable(self._check_args(variable, params, context)))

        # Compute the function
        statement = variable
        propensity = self.paramsCurrent[PROPENSITY]
        pertinacity = self.paramsCurrent[PERTINACITY]
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
    Function_Base(           \
         function,           \
         variable=None,      \
         params=None,        \
         owner=None,         \
         name=None,          \
         prefs=None          \
    )

    Implement user-defined Function.

    This is used to "wrap" custom functions in the PsyNeuLink `Function API <LINK>`.
    It is automatically invoked and applied to any function that is assigned to the `function <Component.function>`
    attribute of a PsyNeuLink component (other than a Function itself).  The function can take any arguments and
    return any values.  However, if UserDefinedFunction is used to create a custom version of another PsyNeuLink
    `Function <Function>`, then it must conform to the requirements of that Function's type.

    .. note::
       Currently the arguments for the `function <UserDefinedFunction.function>` of a UserDefinedFunction are NOT
       assigned as attributes of the UserDefinedFunction object or its owner, nor to its :keyword:`user_params` dict.

    Arguments
    ---------

    function : function
        specifies function to "wrap." It can be any function, take any arguments (including standard ones,
        such as :keyword:`params` and :keyword:`context`) and return any value(s), so long as these are consistent
        with the context in which the UserDefinedFunction will be used.

    variable : value : default ClassDefaults.variable
        specifies the format and a default value for the input to `function <Function>`.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


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

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the **prefs** argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """
    componentName = USER_DEFINED_FUNCTION
    componentType = USER_DEFINED_FUNCTION_TYPE

    class ClassDefaults(Function_Base.ClassDefaults):
        variable = [0]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        FUNCTION_OUTPUT_TYPE_CONVERSION: False,
        PARAMETER_STATE_PARAMS: None
    })

    @tc.typecheck
    def __init__(self,
                 function,
                 variable=None,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context=componentName + INITIALIZING):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(params=params)
        self.user_defined_function = function

        super().__init__(default_variable=variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        self.functionOutputType = None

        # IMPLEMENT: PARSE ARGUMENTS FOR user_defined_function AND ASSIGN TO user_params

    def function(self,
                 **kwargs):
        return self.user_defined_function(**kwargs)


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
    Reduce(                                     \
         default_variable=ClassDefaults.variable, \
         operation=SUM,                         \
         scale=1.0,                             \
         offset=0.0,                            \
         params=None,                           \
         owner=None,                            \
         prefs=None,                            \
    )

    .. _Reduce:

    Combine values in each of one or more arrays into a single value for each array.
    Use optional SCALE and OFFSET parameters to linearly transform the resulting value for each array.
    Returns a scalar value for each array of the input.

    COMMENT:
        IMPLEMENTATION NOTE: EXTEND TO MULTIDIMENSIONAL ARRAY ALONG ARBITRARY AXIS
    COMMENT

    Arguments
    ---------

    default_variable : list or np.array : default ClassDefaults.variable
        specifies a template for the value to be transformed and its default value;  all entries must be numeric.

    operation : SUM or PRODUCT : default SUM
        specifies whether to sum or multiply the elements in `variable <Reduce.function.variable>` of
        `function <Reduce.function>`.

    scale : float
        specifies a value by which to multiply each element of the output of `function <Reduce.function>`
        (see `scale <Reduce.scale>` for details)

    offset : float
        specifies a value to add to each element of the output of `function <Reduce.function>`
        (see `offset <Reduce.offset>` for details)

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


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

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the **prefs** argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """
    componentName = REDUCE_FUNCTION

    multiplicative_param = SCALE
    additive_param = OFFSET

    class ClassDefaults(CombinationFunction.ClassDefaults):
        variable = [0, 0]
    # variableClassDefault_locked = True

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=ClassDefaults.variable,
                 operation: tc.enum(SUM, PRODUCT) = SUM,
                 scale: parameter_spec = 1.0,
                 offset: parameter_spec = 0.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context=componentName + INITIALIZING):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(operation=operation,
                                                  scale=scale,
                                                  offset=offset,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

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

    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """
        Calculate sum or product of the elements for each array in `variable <Reduce.variable>`,
        apply `scale <Reduce.scale>` and/or `offset <Reduce.offset>`, and return array of resulting values.

        Arguments
        ---------

        variable : list or np.array : default ClassDefaults.variable
           a list or np.array of numeric values.

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------

        Sum or product of arrays in variable : np.array
            in an array that is one dimension less than `variable <Reduce.variable>`.


        """

        # Validate variable and assign to variable, and validate params
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        operation = self.paramsCurrent[OPERATION]
        scale = self.paramsCurrent[SCALE]
        offset = self.paramsCurrent[OFFSET]

        # Calculate using relevant aggregation operation and return
        if (operation is SUM):
            result = np.sum(variable) * scale + offset
        elif operation is PRODUCT:
            result = np.product(variable) * scale + offset
        else:
            raise FunctionError("Unrecognized operator ({0}) for Reduce function".
                                format(self.paramsCurrent[OPERATION].self.Operation.SUM))
        return result


class LinearCombination(CombinationFunction):  # ------------------------------------------------------------------------
    # FIX: CONFIRM THAT 1D KWEIGHTS USES EACH ELEMENT TO SCALE CORRESPONDING VECTOR IN VARIABLE
    # FIX  CONFIRM THAT LINEAR TRANSFORMATION (OFFSET, SCALE) APPLY TO THE RESULTING ARRAY
    # FIX: CONFIRM RETURNS LIST IF GIVEN LIST, AND SIMLARLY FOR NP.ARRAY
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

    Linearly combine arrays of values with optional integration_type, exponentiation, scaling and/or offset.

    Combines the arrays in the items of the `variable <LinearCombination.variable>` argument.  Each array can be
    individually weighted and/or exponentiated; they can combined additively or multiplicatively; and the resulting
    array can be multiplicatively transformed and/or additively offset.

    COMMENT:
        Description:
            Combine corresponding elements of arrays in variable arg, using arithmetic operation determined by OPERATION
            Use optional INTEGRATION_TYPE argument to weight contribution of each array to the combination
            Use optional SCALE and OFFSET parameters to linearly transform the resulting array
            Returns a list or 1D array of the same length as the individual ones in the variable

            Notes:
            * If variable contains only a single array, it is simply linearly transformed using SCALE and OFFSET
            * If there is more than one array in variable, they must all be of the same length
            * WEIGHTS can be:
                - 1D: each array in the variable is scaled by the corresponding element of WEIGHTS)
                - 2D: each array in the variable is multiplied by (Hadamard Product) the corresponding array in kwWeight
        Initialization arguments:
         - variable (value, np.ndarray or list): values to be combined;
             can be a list of lists, or a 1D or 2D np.array;  a 1D np.array is always returned
             if it is a list, it must be a list of numbers, lists, or np.arrays
             all items in the list or 2D np.array must be of equal length
             the length of WEIGHTS (if provided) must equal the number of arrays (2nd dimension; default is 2)
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

    operation : SUM or PRODUCT
        specifies whether the `function <LinearCombination.function>` takes the elementwise (Hadamarad)
        sum or product of the arrays in `variable  <LinearCombination.variable>`.

    scale : float or np.ndarray : default None
        specifies a value by which to multiply each element of the output of `function <LinearCombination.function>`
        (see `scale <LinearCombination.scale>` for details)

    offset : float or np.ndarray : default None
        specifies a value to add to each element of the output of `function <LinearCombination.function>`
        (see `offset <LinearCombination.offset>` for details)

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    variable : 1d or 2d np.array
        contains the arrays to be combined by `function <LinearCombination>`.  If it is 1d, the array is simply
        linearly transformed by and `scale <LinearCombination.scale>` and `offset <LinearCombination.scale>`.
        If it is 2d, the arrays (all of which must be of equal length) are weighted and/or exponentiated as
        specified by `weights <LinearCombination.weights>` and/or `exponents <LinearCombination.exponents>`
        and then combined as specified by `operation <LinearCombination.operation>`.

    weights : 1d or 2d np.array
        if it is 1d, each element is used to multiply all elements in the corresponding array of
        `variable <LinearCombination.variable>`;    if it is 2d, then each array is multiplied elementwise
        (i.e., the Hadamard Product is taken) with the corresponding array of `variable <LinearCombinations.variable>`.
        All :keyword:`weights` are applied before any exponentiation (if it is specified).

    exponents : 1d or 2d np.array
        if it is 1d, each element is used to exponentiate the elements of the corresponding array of
        `variable <LinearCombinations.variable>`;  if it is 2d, the element of each array is used to exponentiate
        the correspnding element of the corresponding array of `variable <LinearCombination.variable>`.
        In either case, exponentiating is applied after application of the `weights <LinearCombination.weights>`
        (if any are specified).

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

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the **prefs** argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = LINEAR_COMBINATION_FUNCTION

    classPreferences = {
        kwPreferenceSetName: 'LinearCombinationCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
        kpRuntimeParamStickyAssignmentPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
    }

    multiplicative_param = SCALE
    additive_param = OFFSET

    class ClassDefaults(CombinationFunction.ClassDefaults):
        variable = [2, 2]
    # variableClassDefault_locked = True

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=ClassDefaults.variable,
                 # IMPLEMENTATION NOTE - these don't check whether every element of np.array is numerical:
                 # weights:tc.optional(tc.any(int, float, tc.list_of(tc.any(int, float)), np.ndarray))=None,
                 # exponents:tc.optional(tc.any(int, float, tc.list_of(tc.any(int, float)), np.ndarray))=None,
                 # MODIFIED 2/10/17 OLD: [CAUSING CRASHING FOR SOME REASON]
                 # # weights:is_numeric_or_none=None,
                 # # exponents:is_numeric_or_none=None,
                 # weights=None,
                 # exponents=None,
                 weights:tc.optional(parameter_spec)=None,
                 exponents:tc.optional(parameter_spec)=None,
                 operation: tc.enum(SUM, PRODUCT)=SUM,
                 # scale=1.0,
                 # offset=0.0,
                 # scale:tc.optional(parameter_spec)=1.0,
                 # offset:tc.optional(parameter_spec)=0.0,
                 # scale:is_numeric_or_none=None,
                 # offset:is_numeric_or_none=None,
                 scale=None,
                 offset=None,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context=componentName + INITIALIZING):

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
                         context=context)

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
                    raise FunctionError("Length of all arrays in variable {0} for {1} must be the same".
                                        format(variable, self.__class__.__name__))
        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate weghts, exponents, scale and offset parameters

        Check that WEIGHTS and EXPONENTS are lists or np.arrays of numbers with length equal to variable
        Check that SCALE and OFFSET are either scalars or np.arrays of numbers with length and shape equal to variable

        Note: the checks of compatiability with variable are only performed for validation calls during execution
              (i.e., from check_args(), since during initialization or COMMAND_LINE assignment,
              a parameter may be re-assigned before variable assigned during is known
        """

        # FIX: MAKE SURE THAT IF OPERATION IS SUBTRACT OR DIVIDE, THERE ARE ONLY TWO VECTORS

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if WEIGHTS in target_set and target_set[WEIGHTS] is not None:
            target_set[WEIGHTS] = np.atleast_2d(target_set[WEIGHTS]).reshape(-1, 1)
            if EXECUTING in context:
                if len(target_set[WEIGHTS]) != len(self.instance_defaults.variable):
                    raise FunctionError("Number of weights ({0}) is not equal to number of items in variable ({1})".
                                        format(len(target_set[WEIGHTS]), len(self.instance_defaults.variable.shape)))

        if EXPONENTS in target_set and target_set[EXPONENTS] is not None:
            target_set[EXPONENTS] = np.atleast_2d(target_set[EXPONENTS]).reshape(-1, 1)
            if EXECUTING in context:
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
            if EXECUTING in context:
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
            if EXECUTING in context:
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
                 time_scale=TimeScale.TRIAL,
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

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------

        combined array : 1d np.array
            the result of linearly combining the arrays in `variable <LinearCombination.variable>`.

        """

        # Validate variable and assign to variable, and validate params
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        exponents = self.exponents
        weights = self.weights
        operation = self.operation
        # QUESTION:  WHICH IS LESS EFFICIENT:
        #                A) UNECESSARY ARITHMETIC OPERATIONS IF SCALE AND/OR OFFSET ARE 1.0 AND 0, RESPECTIVELY?
        #                   (DOES THE COMPILER KNOW NOT TO BOTHER WITH MULT BY 1 AND/OR ADD 0?)
        #                B) EVALUATION OF IF STATEMENTS TO DETERMINE THE ABOVE?
        # IMPLEMENTATION NOTE:  FOR NOW, ASSUME B) ABOVE, AND ASSIGN DEFAULT "NULL" VALUES TO offset AND scale
        if self.offset is None:
            offset = 0.0
        else:
            offset = self.offset
        if self.scale is None:
            scale = 1.0
        else:
            scale = self.scale

        # IMPLEMENTATION NOTE: CONFIRM: SHOULD NEVER OCCUR, AS _validate_variable NOW ENFORCES 2D np.ndarray
        # If variable is 0D or 1D:
        if np_array_less_than_2d(variable):
            return (variable * scale) + offset

        # FIX FOR EFFICIENCY: CHANGE THIS AND WEIGHTS TO TRY/EXCEPT // OR IS IT EVEN NECESSARY, GIVEN VALIDATION ABOVE??
        # Apply exponents if they were specified
        if exponents is not None:
            # Avoid divide by zero warning:
            #    make sure there are no zeros for an element that is assigned a negative exponent
            if INITIALIZING in context and any(not any(i) and j < 0 for i, j in zip(variable, exponents)):
                variable = self._update_variable(np.ones_like(variable))
            else:
                variable = self._update_variable(variable ** exponents)

        # Apply weights if they were specified
        if weights is not None:
            variable = self._update_variable(variable * weights)

        # CALCULATE RESULT USING RELEVANT COMBINATION OPERATION AND MODULATION

        if (operation is SUM):
            if isinstance(scale, numbers.Number):
                # Scalar scale and offset
                if isinstance(offset, numbers.Number):
                    result = np.sum(variable, axis=0) * scale + offset
                # Scalar scale and Hadamard offset
                else:
                    result = np.sum(np.append([variable * scale], [offset], axis=0), axis=0)
            else:
                # Hadamard scale, scalar offset
                if isinstance(offset, numbers.Number):
                    result = np.product([np.sum([variable], axis=0), scale], axis=0)
                # Hadamard scale and offset
                else:
                    hadamard_product = np.product([np.sum([variable], axis=0), scale], axis=0)
                    result = np.sum(np.append([hadamard_product], [offset], axis=0), axis=0)

        elif (operation is PRODUCT):
            product = np.product(variable, axis=0)
            if isinstance(scale, numbers.Number):
                # Scalar scale and offset
                if isinstance(offset, numbers.Number):
                    result = product * scale + offset
                # Scalar scale and Hadamard offset
                else:
                    result = np.sum(np.append([product], [offset], axis=0), axis=0) + offset
            else:
                # Hadamard scale, scalar offset
                if isinstance(offset, numbers.Number):
                    result = np.product(np.append([product], [scale], axis=0), axis=0) + offset
                # Hadamard scale and offset
                else:
                    hadamard_product = np.product(np.append([product], [scale], axis=0), axis=0)
                    result = np.sum(np.append([hadamard_product], [offset], axis=0), axis=0)

        else:
            raise FunctionError("Unrecognized operator ({0}) for LinearCombination function".
                                format(self.paramsCurrent[OPERATION].self.Operation.SUM))
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

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


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

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the **prefs** argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

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

    class ClassDefaults(TransferFunction.ClassDefaults):
        variable = [0]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        FUNCTION_OUTPUT_TYPE_CONVERSION: True,
        PARAMETER_STATE_PARAMS: None
    })

    @tc.typecheck
    def __init__(self,
                 default_variable=ClassDefaults.variable,
                 slope: parameter_spec = 1.0,
                 intercept: parameter_spec = 0.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context=componentName + INITIALIZING):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(slope=slope,
                                                  intercept=intercept,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        # self.functionOutputType = None

    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """
        Return: `slope <Linear.slope>` * `variable <Linear.variable>` + `intercept <Linear.intercept>`.

        Arguments
        ---------

        variable : number or np.array : default ClassDefaults.variable
           a single value or array to be transformed.

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------

        linear transformation of variable : number or np.array

        """

        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))
        slope = self.paramsCurrent[SLOPE]
        intercept = self.paramsCurrent[INTERCEPT]
        outputType = self.functionOutputType

        # By default, result should be returned as np.ndarray with same dimensionality as input
        result = variable * slope + intercept

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

        return self.slope


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

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    variable : number or np.array
        contains value to be transformed.

    rate : float
        value by which `variable <Exponential.variable>` is multiplied before exponentiation.

    scale : float
        value by which the exponentiated value is multiplied.

    bounds : (0, None)

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the **prefs** argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = EXPONENTIAL_FUNCTION

    bounds = (0, None)
    multiplicative_param = RATE
    additive_param = SCALE


    class ClassDefaults(TransferFunction.ClassDefaults):
        variable = 0

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=ClassDefaults.variable,
                 rate: parameter_spec = 1.0,
                 scale: parameter_spec = 1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context=componentName + INITIALIZING):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  scale=scale,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """
        Return: `scale <Exponential.scale>` * e**(`rate <Exponential.rate>` * `variable <Linear.variable>`).

        Arguments
        ---------

        variable : number or np.array : default ClassDefaults.variable
           a single value or array to be exponentiated.

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------

        exponential transformation of variable : number or np.array

        """

        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        # Assign the params and return the result
        rate = self.paramsCurrent[RATE]
        scale = self.paramsCurrent[SCALE]

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
        return self.rate * input


class Logistic(TransferFunction):  # ------------------------------------------------------------------------------------
    """
    Logistic(              \
         default_variable, \
         gain=1.0,         \
         bias=0.0,         \
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
        specifies a value by which to multiply `variable <Linear.variable>` before logistic transformation

    bias : float : default 0.0
        specifies a value to add to each element of `variable <Linear.variable>` after applying `gain <Linear.gain>`
        but before logistic transformation.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    variable : number or np.array
        contains value to be transformed.

    gain : float
        value by which each element of `variable <Logistic.variable>` is multiplied before applying the
        `bias <Linear.bias>` (if it is specified).

    bias : float
        value added to each element of `variable <Logistic.variable>` after applying the `gain <Logistic.gain>`
        (if it is specified).

    bounds : (0,1)

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the **prefs** argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = LOGISTIC_FUNCTION
    parameter_keywords.update({GAIN, BIAS})

    bounds = (0,1)
    multiplicative_param = GAIN
    additive_param = BIAS

    class ClassDefaults(TransferFunction.ClassDefaults):
        variable = 0

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=ClassDefaults.variable,
                 gain: parameter_spec = 1.0,
                 bias: parameter_spec = 0.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context='Logistic Init'):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(gain=gain,
                                                  bias=bias,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """
        Return: 1 / (1 + e**( (`gain <Logistic.gain>` * `variable <Logistic.variable>`) + `bias <Logistic.bias>`))

        Arguments
        ---------

        variable : number or np.array : default ClassDefaults.variable
           a single value or array to be transformed.

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------

        logistic transformation of variable : number or np.array

        """

        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))
        gain = self.paramsCurrent[GAIN]
        bias = self.paramsCurrent[BIAS]

        try:
            return_val = 1 / (1 + np.exp(-(gain * variable) + bias))
        except (Warning):
            # handle RuntimeWarning: overflow in exp
            return_val = 0

        return return_val

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

# ------------------------------------------------------------------------------------
class SoftMax(TransferFunction):
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

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    variable : 1d np.array
        contains value to be transformed.

    gain : float
        value by which `variable <Logistic.variable>` is multiplied before the SoftMax transformation;  determines
        the "sharpness" of the distribution.

    output : ALL, MAX_VAL, MAX_INDICATOR, or PROB
        determines how the SoftMax-transformed values of the elements in `variable <SoftMax.variable>` are reported
        in the array returned by `function <SoftMax.funtion>`:
            * **ALL**: array of all SoftMax-transformed values (the default);
            * **MAX_VAL**: SoftMax-transformed value for the element with the maximum such value, 0 for all others;
            * **MAX_INDICATOR**: 1 for the element with the maximum SoftMax-transformed value, 0 for all others;
            * **PROB**: probabilistically chosen element based on SoftMax-transformed values after normalizing sum of
              values to 1, 0 for all others.

    bounds : None if `output <SoftMax.output>` == MAX_VAL, else (0,1) : default (0,1)

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the **prefs** argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = SOFTMAX_FUNCTION

    bounds = (0,1)
    multiplicative_param = GAIN
    additive_param = None

    class ClassDefaults(TransferFunction.ClassDefaults):
        variable = 0

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=ClassDefaults.variable,
                 gain: parameter_spec = 1.0,
                 output: tc.enum(ALL, MAX_VAL, MAX_INDICATOR, PROB) = ALL,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context='SoftMax Init'):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(gain=gain,
                                                  output=output,
                                                  params=params)
        if output is MAX_VAL:
            bounds = None

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """
        Return: e**(`gain <SoftMax.gain>` * `variable <SoftMax.variable>`) /
        sum(e**(`gain <SoftMax.gain>` * `variable <SoftMax.variable>`)),
        filtered by `ouptput <SoftMax.output>` specification.

        Arguments
        ---------

        variable : 1d np.array : default ClassDefaults.variable
           an array to be transformed.

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------

        SoftMax transformation of variable : number or np.array

        """

        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        # Assign the params and return the result
        output_type = self.params[OUTPUT_TYPE]
        gain = self.params[GAIN]

        # Modulate variable by gain
        v = gain * variable
        # Shift by max to avoid extreme values:
        v = v - np.max(v)
        # Exponentiate
        v = np.exp(v)
        # Normalize (to sum to 1)
        sm = v / np.sum(v, axis=0)

        # For the element that is max of softmax, set it's value to its softmax value, set others to zero
        if output_type is MAX_VAL:
            max_value = np.max(sm)
            sm = np.where(sm == max_value, max_value, 0)

        # For the element that is max of softmax, set its value to 1, set others to zero
        elif output_type is MAX_INDICATOR:
            # sm = np.where(sm == np.max(sm), 1, 0)
            max_value = np.max(sm)
            sm = np.where(sm == max_value, 1, 0)

        # Choose a single element probabilistically based on softmax of their values;
        #    leave that element's value intact, set others to zero
        elif output_type is PROB:
            cum_sum = np.cumsum(sm)
            random_value = np.random.uniform()
            chosen_item = next(element for element in cum_sum if element > random_value)
            chosen_in_cum_sum = np.where(cum_sum == chosen_item, 1, 0)
            sm = variable * chosen_in_cum_sum

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
            raise FunctionError("Can't calculate derivative for SoftMax function{} since OUTPUT_TYPE is PROB "
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

    bounds : None

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


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

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the **prefs** argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = LINEAR_MATRIX_FUNCTION

    bounds = None
    multiplicative_param = None
    additive_param = None

    DEFAULT_FILLER_VALUE = 0

    class ClassDefaults(TransferFunction.ClassDefaults):
        variable = [0]  # Sender vector

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
                 default_variable=ClassDefaults.variable,
                 matrix:tc.optional(is_matrix) = None,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context=componentName + INITIALIZING):

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
                         context=context)

        self._matrix = self.instantiate_matrix(self.paramsCurrent[MATRIX])

    def _validate_variable(self, variable, context=None):
        """Insure that variable passed to LinearMatrix is a max 2D np.array

        :param variable: (max 2D np.array)
        :param context:
        :return:
        """
        variable = self._update_variable(super()._validate_variable(variable, context))

        # Check that variable <= 2D
        try:
            if not variable.ndim <= 2:
                raise FunctionError("variable ({0}) for {1} must be a numpy.ndarray of dimension at most 2".format(variable, self.__class__.__name__))
        except AttributeError:
            raise FunctionError("PROGRAM ERROR: variable ({0}) for {1} should be a numpy.ndarray".
                                    format(variable, self.__class__.__name__))

        return variable


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
        sender = self.instance_defaults.variable
        # Note: this assumes variable is a 1D np.array, as enforced by _validate_variable
        sender_len = sender.size

        # FIX: RELABEL sender -> input AND receiver -> output
        # FIX: THIS NEEDS TO BE CLEANED UP:
        #      - AT LEAST CHANGE THE NAME FROM kwReceiver TO output_template OR SOMETHING LIKE THAT
        #      - MAKE ARG?  OR ADD OTHER PARAMS:  E.G., FILLER?
        #      - OR REFACTOR TO INCLUDE AS MATRIX SPEC:
        #                  IF MATRIX IS 1D, USE AS OUTPUT TEMPLATE
        #                     IF ALL ITS VALUES ARE 1'S => FULL CONNECTIVITY MATRIX
        #                     IF ALL ITS VALUES ARE 0'S => RANDOM CONNECTIVITY MATRIX
        #                     NOTE:  NO NEED FOR IDENTITY MATRIX, AS THAT WOULD BE SQUARE SO NO NEED FOR OUTPUT TEMPLATE
        #      - DOCUMENT WHEN DONE
        # MODIFIED 3/26/17 OLD:
        # Check for and validate kwReceiver first, since it may be needed to validate and/or construct the matrix
        # First try to get receiver from specification in params
        if RECEIVER in param_set:
            self.receiver = param_set[RECEIVER]
            # Check that specification is a list of numbers or an np.array
            if ((isinstance(self.receiver, list) and all(isinstance(elem, numbers.Number) for elem in self.receiver)) or
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
        # # MODIFIED 3/26/17 NEW:
        # self.receiver = param_set[kwReceiver] = sender
        # MODIFIED 3/26/17 END

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
                        raise FunctionError("Non-numeric entry in MATRIX specification ({}) for the {} function of {}".
                                            format(param_value), self.name, self.owner.name)

                    if weight_matrix.ndim != 2:
                        raise FunctionError("The matrix provided for the {} function of {} must be 2d (it is {}d".
                                            format(weight_matrix.ndim, self.name, self.owner.name))

                    matrix_rows = weight_matrix.shape[0]
                    matrix_cols = weight_matrix.shape[1]

                    # Check that number of rows equals length of sender vector (variable)
                    if matrix_rows != sender_len:
                        raise FunctionError("The number of rows ({}) of the matrix provided for {} function of {} "
                                            "does not equal the length ({}) of the sender vector (variable)".
                                            format(matrix_rows, self.name, self.owner.name, sender_len))

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
                                            format(param_value, self.name, self.owner.name, receiver_len, sender_len))
                    continue

                # list used to describe matrix, so convert to 2D np.array and pass to validation of matrix below
                elif isinstance(param_value, list):
                    try:
                        param_value = np.atleast_2d(param_value)
                    except (ValueError, TypeError) as error_msg:
                        raise FunctionError("Error in list specification ({}) of matrix for the {} function of {}: {})".
                                            # format(param_value, self.__class__.__name__, error_msg))
                                            format(param_value, self.name, self.owner.name, error_msg))

                # string used to describe matrix, so convert to np.matrix and pass to validation of matrix below
                elif isinstance(param_value, str):
                    try:
                        param_value = np.matrix(param_value)
                    except (ValueError, TypeError) as error_msg:
                        raise FunctionError("Error in string specification ({}) of the matrix "
                                            "for the {} function of {}: {})".
                                            # format(param_value, self.__class__.__name__, error_msg))
                                            format(param_value, self.name, self.owner.name, error_msg))

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
                                            format(self.name, self.owner.name, param_value, test))

                else:
                    raise FunctionError("Value of {} param ({}) for the {} function of {} "
                                        "must be a matrix, a number (for filler), or a matrix keyword ({})".
                                        format(param_name,
                                               param_value,
                                               self.name,
                                               self.owner.name,
                                               MATRIX_KEYWORD_NAMES))
            else:
                message += "Unrecognized param ({}) specified for the {} function of {}\n".format(param_name,
                                                                                                self.componentName,
                                                                                                self.owner.name)
                continue

        if message:
            raise FunctionError(message)

    def _instantiate_attributes_before_function(self, context=None):
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
                                format(specification, self.name, self.owner.name))
            # receiver = sender
        receiver_len = receiver.shape[0]

        matrix = get_matrix(specification, rows=sender_len, cols=receiver_len, context=context)

        # This should never happen (should have been picked up in validate_param or above)
        if matrix is None:
            raise FunctionError("MATRIX param ({}) for the {} function of {} must be a matrix, a function that returns "
                                "one, a matrix specification keyword ({}), or a number (filler)".
                                format(specification, self.name, self.owner.name, MATRIX_KEYWORD_NAMES))
        else:
            return matrix

    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """
        Return: `variable <LinearMatrix.variable>` • `matrix <LinearMatrix.matrix>`

        Arguments
        ---------
        variable : list or 1d np.array
            array to be transformed;  length must equal the number of rows of 'matrix <LinearMatrix.matrix>`.

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        ---------

        dot product of variable and matrix : 1d np.array
            length of the array returned equals the number of columns of `matrix <LinearMatrix.matrix>`.

        """

        # Note: this calls _validate_variable and _validate_params which are overridden above;
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        return np.dot(variable, self.matrix)

    def keyword(self, keyword):

        from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
        rows = None
        cols = None
        # use of variable attribute here should be ok because it's using it as a format/type
        if isinstance(self, MappingProjection):
            rows = len(self.sender.value)
            cols = len(self.receiver.instance_defaults.variable)
        matrix = get_matrix(keyword, rows, cols)

        if matrix is None:
            raise FunctionError("Unrecognized keyword ({}) specified for the {} function of {}".
                                format(keyword, self.name, self.owner.name))
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
        except ValueError:
            # np.matrix(specification) will give ValueError if specification is a bad value (e.g. 'abc', '1; 1 2')
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

class Integrator(IntegratorFunction):  # --------------------------------------------------------------------------------
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
        determines the timing precision of the integration process when `integration_type <Integrator.integration_type>`
        is set to DIFFUSION (see `time_step_size <Integrator.time_step_size>` for details.

    initializer float, list or 1d np.array : default 0.0
        specifies starting value for integration.  If it is a list or array, it must be the same length as
        `default_variable <Integrator.default_variable>` (see `initializer <Integrator.initializer>` for details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    variable : number or np.array
        current input value some portion of which (determined by `rate <Integrator.rate>`) that will be
        added to the prior value;  if it is an array, each element is independently integrated.

    integration_type : [**NEEDS TO BE SPECIFIED**] : default [**NEEDS TO BE SPECIFIED**]
        [**NEEDS TO BE SPECIFIED**]

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

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the **prefs** argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = INTEGRATOR_FUNCTION
    class ClassDefaults(IntegratorFunction.ClassDefaults):
        variable = [[0]]

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
                 initializer=ClassDefaults.variable,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context="Integrator Init"):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  initializer=initializer,
                                                  noise=noise,
                                                  params=params)


        # Assign here as default, for use in initialization of function
        self.previous_value = self.paramClassDefaults[INITIALIZER]

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        # Reassign to kWInitializer in case default value was overridden
        # self.previous_value = self.initializer

        self.auto_dependent = True


    def _validate_params(self, request_set, target_set=None, context=None):

        # Handle list or array for rate specification
        if RATE in request_set:
            rate = request_set[RATE]
            if isinstance(rate, (list, np.ndarray)) and not iscompatible(rate, self.instance_defaults.variable):
                if len(rate) != np.array(self.instance_defaults.variable).size:
                    # If the variable was not specified, then reformat it to match rate specification
                    #    and assign ClassDefaults.variable accordingly
                    # Note: this situation can arise when the rate is parametrized (e.g., as an array) in the
                    #       Integrator's constructor, where that is used as a specification for a function parameter
                    #       (e.g., for an IntegratorMechanism), whereas the input is specified as part of the
                    #       object to which the function parameter belongs (e.g., the IntegratorMechanism);
                    #       in that case, the Integrator gets instantiated using its ClassDefaults.variable ([[0]]) before
                    #       the object itself, thus does not see the array specification for the input.
                    if self._variable_not_specified:
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

        # if INITIALIZER in target_set:
        #     print(target_set)
        #     self._validate_initializer(target_set[INITIALIZER])

        if NOISE in target_set:
            self._validate_noise(target_set[NOISE], self.instance_defaults.variable)

    # Ensure that the noise parameter makes sense with the input type and shape; flag any noise functions that will
    # need to be executed

    def _validate_noise(self, noise, var):
        # Noise is a list or array
        if isinstance(noise, (np.ndarray, list)) and len(noise) != 1:
            # Variable is a list/array
            if isinstance(var, (np.ndarray, list)):
                if len(noise) != np.array(var).size:
                    # Formatting noise for proper display in error message
                    try:
                        formatted_noise = list(map(lambda x: x.__qualname__, noise))
                    except AttributeError:
                        formatted_noise = noise
                    raise FunctionError(
                        "The length ({}) of the array specified for the noise parameter ({}) of {} "
                        "must match the length ({}) of the default input ({}). If noise is specified as"
                        " an array or list, it must be of the same size as the input."
                        .format(len(noise), formatted_noise, self.name, np.array(var).size,
                                var))
                else:
                    for noise_item in noise:
                        if not isinstance(noise_item, (float, int)) and not callable(noise_item):
                            raise FunctionError(
                                "The elements of a noise list or array must be floats or functions.")


            # Variable is not a list/array
            else:
                raise FunctionError("The noise parameter ({}) for {} may only be a list or array if the "
                                    "default input value is also a list or array.".format(noise, self.name))

            # # Elements of list/array have different types
            # if not all(isinstance(x, type(noise[0])) for x in noise):
            #     raise FunctionError("All elements of noise list/array ({}) for {} must be of the same type. "
            #                         .format(noise, self.name))

        elif not isinstance(noise, (float, int, np.ndarray)) and not callable(noise):
            raise FunctionError(
                "Noise parameter ({}) for {} must be a float, function, or array/list of these."
                    .format(noise, self.name))

    def _try_execute_param(self, param, var):

        # param is a list; if any element is callable, execute it
        if isinstance(param, (np.ndarray, list)):
            for i in range(len(param)):
                if callable(param[i]):
                    param[i] = param[i]()
        # param is one function
        elif callable(param):
            # if the variable is a list/array, execute the param function separately for each element
            if isinstance(var, (np.ndarray, list)):
                if isinstance(var[0], (np.ndarray, list)):
                    new_param = []
                    for i in var[0]:
                        new_param.append(param())
                    param = new_param
                else:
                    new_param = []
                    for i in var:
                        new_param.append(param())
                    param = new_param
            # if the variable is not a list/array, execute the param function
            else:
                param = param()
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


    def function(self, *args, **kwargs):
        raise FunctionError("Integrator is not meant to be called explicitly")

    @property
    def reset_initializer(self):
        return self._initializer

    @reset_initializer.setter
    def reset_initializer(self, val):
        self._initializer = val
        self.previous_value = val

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
        `default_variable <SimpleIntegrator.default_variable>` (see `initializer <SimpleIntegrator.initializer>` for details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    variable : number or np.array
        current input value some portion of which (determined by `rate <SimpleIntegrator.rate>`) will be
        added to the prior value;  if it is an array, each element is independently integrated.

    rate : float or 1d np.array
        determines the rate of integration based on current and prior values. If it has a single element, it
        applies to all elements of `variable <SimpleIntegrator.variable>`;  if it has more than one element, each element
        applies to the corresponding element of `variable <SimpleIntegrator.variable>`.

    noise : float, function, list, or 1d np.array
        specifies random value to be added in each call to `function <SimpleIntegrator.function>`.

        If noise is a list or array, it must be the same length as `variable <SimpleIntegrator.default_variable>`.

        If noise is specified as a single float or function, while `variable <SimpleIntegrator.variable>` is a list or array,
        noise will be applied to each variable element. In the case of a noise function, this means that the function
        will be executed separately for each variable element.


        .. note::
            In order to generate random noise, we recommend selecting a probability distribution function
            (see `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value from
            its distribution on each execution. If noise is specified as a float or as a function with a fixed output, then
            the noise will simply be an offset that remains the same across all executions.

    initializer : float, 1d np.array or list
        determines the starting value for integration (i.e., the value to which
        `previous_value <SimpleIntegrator.previous_value>` is set.

        If initializer is a list or array, it must be the same length as `variable <SimpleIntegrator.default_variable>`.

    previous_value : 1d np.array : default ClassDefaults.variable
        stores previous value with which `variable <SimpleIntegrator.variable>` is integrated.

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the **prefs** argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = SIMPLE_INTEGRATOR_FUNCTION

    class ClassDefaults(Integrator.ClassDefaults):
        variable = [[0]]

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
                 initializer=ClassDefaults.variable,
                 params: tc.optional(dict)=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context="SimpleIntegrator Init"):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  initializer=initializer,
                                                  noise=noise,
                                                  offset=offset,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        self.previous_value = self.initializer
        self.auto_dependent = True

    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """
        Return: `variable <Linear.slope>` combined with `previous_value <SimpleIntegrator.previous_value>`
        according to `previous_value <SimpleIntegrator.previous_value>` + `rate <SimpleIntegrator.rate>` *`variable
        <variable.SimpleIntegrator.variable>` + `noise <SimpleIntegrator.noise>`;

        Arguments
        ---------

        variable : number, list or np.array : default ClassDefaults.variable
           a single value or array of values to be integrated.

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        updated value of integral : 2d np.array

        """

        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        rate = np.array(self.paramsCurrent[RATE]).astype(float)

        if self.offset is None:
            offset = 0.0
        else:
            offset = self.offset

        # execute noise if it is a function
        noise = self._try_execute_param(self.noise, variable)

        # try:
        #     previous_value = self._initializer
        # except (TypeError, KeyError):
        previous_value = self.previous_value

        # previous_value = np.atleast_2d(previous_value)
        new_value = variable


        # if params and VARIABLE in params:
        #     new_value = params[VARIABLE]

        # Compute function based on integration_type param

        value = previous_value + (new_value * rate) + noise

        adjusted_value = value + offset
        # If this NOT an initialization run, update the old value
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if not context or not INITIALIZING in context:
            self.previous_value = adjusted_value

        return adjusted_value

class ConstantIntegrator(
    Integrator):  # --------------------------------------------------------------------------------
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
        `default_variable <ConstantIntegrator.default_variable>` (see `initializer <ConstantIntegrator.initializer>` for details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    variable : number or np.array
        **Ignored** by the ConstantIntegrator function. Refer to SimpleIntegrator or AdaptiveIntegrator for integrator
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

        If noise is specified as a single float or function, while `variable <ConstantIntegrator.variable>` is a list or array,
        noise will be applied to each variable element. In the case of a noise function, this means that the function
        will be executed separately for each variable element.

        .. note::
            In order to generate random noise, we recommend selecting a probability distribution function
            (see `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value from
            its distribution on each execution. If noise is specified as a float or as a function with a fixed output, then
            the noise will simply be an offset that remains the same across all executions.

    initializer : float, 1d np.array or list
        determines the starting value for integration (i.e., the value to which
        `previous_value <ConstantIntegrator.previous_value>` is set.

        If initializer is a list or array, it must be the same length as `variable <ConstantIntegrator.default_variable>`.

    previous_value : 1d np.array : default ClassDefaults.variable
        stores previous value to which `rate <ConstantIntegrator.rate>` and `noise <ConstantIntegrator.noise>` will be
        added.

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the **prefs** argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = CONSTANT_INTEGRATOR_FUNCTION

    class ClassDefaults(Integrator.ClassDefaults):
        variable = [[0]]

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
                 initializer=ClassDefaults.variable,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context="ConstantIntegrator Init"):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  initializer=initializer,
                                                  noise=noise,
                                                  scale = scale,
                                                  offset=offset,
                                                  params=params)

        # Assign here as default, for use in initialization of function
        self.previous_value = self.paramClassDefaults[INITIALIZER]

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        # Reassign to initializer in case default value was overridden
        self.previous_value = self.initializer

        self.auto_dependent = True

    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """
        Return: the sum of `previous_value <ConstantIntegrator.previous_value>`, `rate <ConstantIntegrator.rate>`, and
        `noise <ConstantIntegrator.noise>`.

        Arguments
        ---------

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------

        updated value of integral : 2d np.array

        """
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        rate = np.array(self.rate).astype(float)
        offset = self.offset
        scale = self.scale

        # CAVEAT: why was self.variable never used in this function previously?
        # execute noise if it is a function
        noise = self._try_execute_param(self.noise, variable)


        # try:
        #     previous_value = params[INITIALIZER]
        # except (TypeError, KeyError):
        previous_value = self.previous_value

        previous_value = np.atleast_2d(previous_value)

        value = previous_value + rate + noise


        adjusted_value = value*scale + offset
        # If this NOT an initialization run, update the old value
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if not context or not INITIALIZING in context:
            self.previous_value = adjusted_value

        return adjusted_value

class AdaptiveIntegrator(
    Integrator):  # --------------------------------------------------------------------------------
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

    (1 - `rate <AdaptiveIntegrator.rate>`) * `previous_value <AdaptiveIntegrator.previous_value>` + `rate <AdaptiveIntegrator.rate>` *
    `variable <AdaptiveIntegrator.variable>` + `noise <AdaptiveIntegrator.noise>`


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
        `default_variable <AdaptiveIntegrator.default_variable>` (see `initializer <AdaptiveIntegrator.initializer>` for details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


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

        If noise is specified as a single float or function, while `variable <AdaptiveIntegrator.variable>` is a list or array,
        noise will be applied to each variable element. In the case of a noise function, this means that the function
        will be executed separately for each variable element.

        .. note::
            In order to generate random noise, we recommend selecting a probability distribution function
            (see `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value from
            its distribution on each execution. If noise is specified as a float or as a function with a fixed output, then
            the noise will simply be an offset that remains the same across all executions.

    initializer : float, 1d np.array or list
        determines the starting value for time-averaging (i.e., the value to which
        `previous_value <AdaptiveIntegrator.previous_value>` is originally set).

        If initializer is a list or array, it must be the same length as `variable <AdaptiveIntegrator.default_variable>`.

    previous_value : 1d np.array : default ClassDefaults.variable
        stores previous value with which `variable <AdaptiveIntegrator.variable>` is integrated.

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the **prefs** argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = ADAPTIVE_INTEGRATOR_FUNCTION

    class ClassDefaults(Integrator.ClassDefaults):
        variable = [[0]]

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
                 initializer=ClassDefaults.variable,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context="AdaptiveIntegrator Init"):

        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  initializer=initializer,
                                                  noise=noise,
                                                  offset=offset,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        self.previous_value = self.initializer

        self.auto_dependent = True

    def _validate_params(self, request_set, target_set=None, context=None):

        # Handle list or array for rate specification
        if RATE in request_set:
            rate = request_set[RATE]
            if isinstance(rate, (list, np.ndarray)):
                if len(rate) != np.array(self.instance_defaults.variable).size:
                    # If the variable was not specified, then reformat it to match rate specification
                    #    and assign ClassDefaults.variable accordingly
                    # Note: this situation can arise when the rate is parametrized (e.g., as an array) in the
                    #       AdaptiveIntegrator's constructor, where that is used as a specification for a function parameter
                    #       (e.g., for an IntegratorMechanism), whereas the input is specified as part of the
                    #       object to which the function parameter belongs (e.g., the IntegratorMechanism);
                    #       in that case, the Integrator gets instantiated using its ClassDefaults.variable ([[0]]) before
                    #       the object itself, thus does not see the array specification for the input.
                    if self._variable_not_specified:
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
            self._validate_noise(target_set[NOISE], self.instance_defaults.variable)
        # if INITIALIZER in target_set:
        #     self._validate_initializer(target_set[INITIALIZER])


    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """
        Return: some fraction of `variable <AdaptiveIntegrator.variable>` combined with some fraction of `previous_value
        <AdaptiveIntegrator.previous_value>`.

        Arguments
        ---------

        variable : number, list or np.array : default ClassDefaults.variable
           a single value or array of values to be integrated.

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------

        updated value of integral : 2d np.array

        """
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        rate = np.array(self.paramsCurrent[RATE]).astype(float)
        offset = self.paramsCurrent[OFFSET]
        # execute noise if it is a function
        noise = self._try_execute_param(self.noise, variable)


        # try:
        #     previous_value = params[INITIALIZER]
        # except (TypeError, KeyError):
        previous_value = self.previous_value

        previous_value = np.atleast_2d(previous_value)
        new_value = variable

        value = (1 - rate) * previous_value + rate * new_value + noise


        adjusted_value = value + offset
        # If this NOT an initialization run, update the old value
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if not context or not INITIALIZING in context:
            self.previous_value = adjusted_value

        return adjusted_value

class DriftDiffusionIntegrator(
    Integrator):  # --------------------------------------------------------------------------------
    """
    DriftDiffusionIntegrator(           \
        default_variable=None,          \
        rate=1.0,                       \
        noise=0.0,                      \
        scale: parameter_spec = 1.0,    \
        offset: parameter_spec = 0.0,   \
        time_step_size=1.0,             \
        t0=0.0,                         \
        decay=0.0,                      \
        initializer,                    \
        params=None,                    \
        owner=None,                     \
        prefs=None,                     \
        )

    .. _DriftDiffusionIntegrator:

    Accumulate evidence overtime based on a stimulus, previous position, and noise.

    Arguments
    ---------

    default_variable : number, list or np.array : default ClassDefaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float, list or 1d np.array : default 1.0
        specifies the rate of integration.  If it is a list or array, it must be the same length as
        `variable <DriftDiffusionIntegrator.default_variable>` (see `rate <DriftDiffusionIntegrator.rate>` for details).

    noise : float, PsyNeuLink Function, list or 1d np.array : default 0.0
        specifies random value to be added in each call to `function <DriftDiffusionIntegrator.function>`. (see
        `noise <DriftDiffusionIntegrator.noise>` for details).

    time_step_size : float : default 0.0
        determines the timing precision of the integration process (see `time_step_size
        <DriftDiffusionIntegrator.time_step_size>` for details.

    t0 : float
        determines the start time of the integration process and is used to compute the RESPONSE_TIME output state of
        the DDM Mechanism.

    initializer float, list or 1d np.array : default 0.0
        specifies starting value for integration.  If it is a list or array, it must be the same length as
        `default_variable <DriftDiffusionIntegrator.default_variable>` (see `initializer <DriftDiffusionIntegrator.initializer>` for details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    variable : number or np.array
        current input value, which represents the stimulus component of drift.

    rate : float or 1d np.array
        determines the rate of integration based on current and prior values.  If integration_type is set to ADAPTIVE,
        all elements must be between 0 and 1 (0 = no change; 1 = instantaneous change). If it has a single element, it
        applies to all elements of `variable <DriftDiffusionIntegrator.variable>`;  if it has more than one element, each element
        applies to the corresponding element of `variable <DriftDiffusionIntegrator.variable>`.

    noise : float, function, list, or 1d np.array
        scales the random value to be added in each call to `function <DriftDiffusionIntegrator.function>

        Noise must be specified as a float (or list or array of floats) because this
        value will be used to construct the standard DDM probability distribution.

    time_step_size : float
        determines the timing precision of the integration process and is used to scale the `noise
        <DriftDiffusionIntegrator.noise>` parameter appropriately.

    t0 : float
        determines the start time of the integration process and is used to compute the RESPONSE_TIME output state of
        the DDM Mechanism.

    initializer : float, 1d np.array or list
        determines the starting value for integration (i.e., the value to which
        `previous_value <DriftDiffusionIntegrator.previous_value>` is set.

        If initializer is a list or array, it must be the same length as `variable <DriftDiffusionIntegrator.default_variable>`.

    previous_time : float
        stores previous time at which the function was executed and accumulates with each execution according to
        `time_step_size <DriftDiffusionIntegrator.default_time_step_size>`.

    previous_value : 1d np.array : default ClassDefaults.variable
        stores previous value with which `variable <DriftDiffusionIntegrator.variable>` is integrated.

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the **prefs** argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = DRIFT_DIFFUSION_INTEGRATOR_FUNCTION

    class ClassDefaults(Integrator.ClassDefaults):
        variable = [[0]]

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
                 initializer=ClassDefaults.variable,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context="DriftDiffusionIntegrator Init"):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  time_step_size=time_step_size,
                                                  t0=t0,
                                                  initializer=initializer,
                                                  noise=noise,
                                                  offset=offset,
                                                  params=params)

        # Assign here as default, for use in initialization of function
        self.previous_value = self.paramClassDefaults[INITIALIZER]

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        # Reassign to kWInitializer in case default value was overridden
        self.previous_value = self.initializer
        self.previous_time = self.t0
        self.auto_dependent = True

    def _validate_noise(self, noise, var):
        if not isinstance(noise, float):
            raise FunctionError(
                "Invalid noise parameter for {}. DriftDiffusionIntegrator requires noise parameter to be a float. Noise"
                " parameter is used to construct the standard DDM noise distribution".format(self.name))

    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """
        Return: One time step of evidence accumulation according to the Drift Diffusion Model

        previous_value + rate * variable * time_step_size + :math:`\\sqrt{time_step_size * noise}` * random
        sample from Normal distribution

        Arguments
        ---------

        variable : number, list or np.array : default ClassDefaults.variable
           the stimulus component of drift rate in the Drift Diffusion Model.

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        updated value of integral : 2d np.array

        """
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        rate = np.array(self.paramsCurrent[RATE]).astype(float)
        offset = self.paramsCurrent[OFFSET]

        time_step_size = self.paramsCurrent[TIME_STEP_SIZE]

        noise = self.noise

        # try:
        #     previous_value = params[INITIALIZER]
        # except (TypeError, KeyError):
        previous_value = self.previous_value

        previous_value = np.atleast_2d(previous_value)
        new_value = variable

        value = previous_value + rate * new_value * time_step_size  \
                + np.sqrt(time_step_size * noise) * np.random.normal()

        adjusted_value = value + offset
        # If this NOT an initialization run, update the old value and time
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if not context or not INITIALIZING in context:
            self.previous_value = adjusted_value
            self.previous_time += time_step_size

        return adjusted_value

class OrnsteinUhlenbeckIntegrator(
    Integrator):  # --------------------------------------------------------------------------------
    """
    OrnsteinUhlenbeckIntegrator(                 \
        default_variable=None,          \
        rate=1.0,                       \
        noise=0.0,                      \
        scale: parameter_spec = 1.0,    \
        offset: parameter_spec = 0.0,   \
        time_step_size=1.0,             \
        t0=0.0,                         \
        initializer,                    \
        params=None,                    \
        owner=None,                     \
        prefs=None,                     \
        )

    .. _OrnsteinUhlenbeckIntegrator:

    Accumulate evidence overtime based on a stimulus, noise, decay, and previous position.

    Arguments
    ---------

    default_variable : number, list or np.array : default ClassDefaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float, list or 1d np.array : default 1.0
        specifies the rate of integration.  If it is a list or array, it must be the same length as
        `variable <OrnsteinUhlenbeckIntegrator.default_variable>` (see `rate <OrnsteinUhlenbeckIntegrator.rate>` for
        details).

    noise : float, PsyNeuLink Function, list or 1d np.array : default 0.0
        specifies random value to be added in each call to `function <OrnsteinUhlenbeckIntegrator.function>`. (see
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

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    variable : number or np.array
        current input value which represents the stimulus component of drift. The product of
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

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the **prefs** argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = ORNSTEIN_UHLENBECK_INTEGRATOR_FUNCTION

    class ClassDefaults(Integrator.ClassDefaults):
        variable = [[0]]

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
                 decay = 1.0,
                 initializer=ClassDefaults.variable,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context="OrnsteinUhlenbeckIntegrator Init"):

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
        self.previous_value = self.paramClassDefaults[INITIALIZER]

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        # Reassign to kWInitializer in case default value was overridden
        self.previous_value = self.initializer
        self.previous_time = self.t0

        self.auto_dependent = True

    def _validate_noise(self, noise, var):
        if not isinstance(noise, float):
            raise FunctionError(
                "Invalid noise parameter for {}. OrnsteinUhlenbeckIntegrator requires noise parameter to be a float. "
                "Noise parameter is used to construct the standard DDM noise distribution".format(self.name))

    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """
        Return: One time step of evidence accumulation according to the Ornstein Uhlenbeck Model

        previous_value + decay * (previous_value -  rate * variable) + :math:`\\sqrt{time_step_size * noise}` * random
        sample from Normal distribution


        Arguments
        ---------

        variable : number, list or np.array : default ClassDefaults.variable
           the stimulus component of drift rate in the Drift Diffusion Model.


        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------

        updated value of integral : 2d np.array

        """
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        rate = np.array(self.paramsCurrent[RATE]).astype(float)
        offset = self.paramsCurrent[OFFSET]

        time_step_size = self.paramsCurrent[TIME_STEP_SIZE]
        decay = self.paramsCurrent[DECAY]

        noise = self.noise

        # try:
        #     previous_value = params[INITIALIZER]
        # except (TypeError, KeyError):
        previous_value = self.previous_value

        previous_value = np.atleast_2d(previous_value)
        new_value = variable
        # dx = (lambda*x + A)dt + c*dW
        value = previous_value + decay * (previous_value -  rate * new_value) * time_step_size + np.sqrt(
            time_step_size * noise) * np.random.normal()

        # If this NOT an initialization run, update the old value and time
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        adjusted_value = value + offset

        if not context or not INITIALIZING in context:
            self.previous_value = adjusted_value
            self.previous_time += time_step_size

        return adjusted_value

class FHNIntegrator(
    Integrator):  # --------------------------------------------------------------------------------
    """
    FHNIntegrator(              \
        default_variable=None,          \
        rate=1.0,                       \
        noise=0.0,                      \
        scale: parameter_spec = 1.0,    \
        offset: parameter_spec = 0.0,   \
        initial_w=0.0,                  \
        initial_v=0.0,                  \
        time_step_size=0.1,             \
        t_0=0.0,                        \
        a_v=-1/3,                       \
        b_v=0.0,                        \
        c_v=1.0,                        \
        d_v=0.0,                        \
        e_v=-1.0,                       \
        f_v=1.0,                        \
        time_constant_v=1.0,            \
        a_w=1.0,                        \
        b_w=-0.8,                       \
        c_w=0.7,                        \
        mode=1.0,      \
        uncorrelated_activity=0.0       \
        time_constant_w = 12.5,         \
        params=None,                    \
        owner=None,                     \
        prefs=None,                     \
        )

    .. _FHNIntegrator:

    Implements the Fitzhugh-Nagumo model using the 4th order Runge Kutta method of numerical integration.

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

    mode : float : default 1.0
        coefficient which simulates electrotonic coupling by scaling the values of dw/dt such that the v term
        (representing the input from the LC) increases when the uncorrelated_activity term (representing baseline
        activity) decreases

    uncorrelated_activity : float : default 0.0
        constant term in the dw/dt equation

    time_constant_w : float : default 12.5
        scaling factor on the dv/dt equation

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).



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

    owner : Mechanism
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
        coefficient on the external stimulus ('variable <FHNIntegrator.variable>`) term in the dv/dt equation

    time_constant_v : float : default 1.0
        scaling factor on the dv/dt equation

    a_w : float : default 1.0,
        coefficient on the v term of the dw/dt equation

    b_w : float : default -0.8,
        coefficient on the w term of the dv/dt equation

    c_w : float : default 0.7,
        constant term in the dw/dt equation

    mode : float : default 1.0
        coefficient which simulates electrotonic coupling by scaling the values of dw/dt such that the v term
        (representing the input from the LC) increases when the uncorrelated_activity term (representing baseline
        activity) decreases

    uncorrelated_activity : float : default 0.0
        constant term in the dw/dt equation

    time_constant_w : float : default 12.5
        scaling factor on the dv/dt equation

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the **prefs** argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = FHN_INTEGRATOR_FUNCTION

    class ClassDefaults(Integrator.ClassDefaults):
        variable = [[0]]

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
                 default_variable=1.0,
                 offset=0.0,
                 scale=1.0,
                 initial_w=0.0,
                 initial_v=0.0,
                 time_step_size=0.1,
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
                 time_constant_w = 12.5,
                 mode = 1.0,
                 uncorrelated_activity = 0.0,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context="FHNIntegrator Init"):

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
                                                  mode=mode,
                                                  uncorrelated_activity=uncorrelated_activity,
                                                  time_constant_w=time_constant_w,
                                                  params=params)

        self.previous_v = self.initial_v
        self.previous_w = self.initial_w
        self.previous_t = self.t_0
        super().__init__(
            default_variable=default_variable,
            params=params,
            owner=owner,
            prefs=prefs,
            context=context)

        self.variable = self.default_variable
        self.auto_dependent = True





    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """
        Return: current v, current w

        The model is defined by the following system of differential equations:

            time_constant_v * dv/dt = a_v * v^3 + b_v * v^2 + c_v*v^2 + d_v + e_v * w + f_v * I_ext

            time_constant_w * dw/dt = mode * a_w * v + b_w * w + c_w + (1 - self.mode) * self.uncorrelated_activity




        Arguments
        ---------

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        current value of v , current value of w : float, list, or np.array

        """

        variable = self.variable

        def dv_dt(time, v):

            val= (self.a_v*(v**3) + self.b_v*(v**2) + self.c_v*v + self.d_v
                    + self.e_v*self.previous_w + self.f_v*variable)/self.time_constant_v
            return val
        def dw_dt(time, w):

            return (self.mode*self.a_w*self.previous_v + self.b_w*w + self.c_w +
                    (1-self.mode)*self.uncorrelated_activity)/self.time_constant_w

        new_v = self._runge_kutta_4(previous_time=self.previous_t,
                                    previous_value=self.previous_v,
                                    slope=dv_dt,
                                    time_step_size=self.time_step_size)*self.scale + self.offset

        new_w = self._runge_kutta_4(previous_time=self.previous_t,
                                    previous_value=self.previous_w,
                                    slope=dw_dt,
                                    time_step_size=self.time_step_size)*self.scale + self.offset

        if not context or INITIALIZING not in context:
            self.previous_v = new_v
            self.previous_w = new_w
            self.previous_t += self.time_step_size

        return new_v, new_w

class AccumulatorIntegrator(
    Integrator):  # --------------------------------------------------------------------------------
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

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    variable : number or np.array
        **Ignored** by the AccumulatorIntegrator function. Refer to SimpleIntegrator or AdaptiveIntegrator for
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

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the **prefs** argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = ACCUMULATOR_INTEGRATOR_FUNCTION

    class ClassDefaults(Integrator.ClassDefaults):
        variable = [[0]]

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
                 initializer=ClassDefaults.variable,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context="AccumulatorIntegrator Init"):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  initializer=initializer,
                                                  noise=noise,
                                                  increment=increment,
                                                  params=params)

        super().__init__(
            # default_variable=default_variable,
            params=params,
            owner=owner,
            prefs=prefs,
            context=context)

        self.previous_value = self.initializer
        self.instance_defaults.variable = self.initializer

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
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """
        Return: `previous_value <ConstantIntegrator.previous_value>` combined with `rate <ConstantIntegrator.rate>` and
        `noise <ConstantIntegrator.noise>`.

        Arguments
        ---------

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------

        updated value of integral : 2d np.array

        """
        self._accumulator_check_args(variable, params=params, context=context)

        # rate = np.array(self.rate).astype(float)
        # increment = self.increment

        if self.rate is None:
            rate = 1.0
        else:
            rate = self.rate

        if self.increment is None:
            increment = 0.0
        else:
            increment = self.increment

        # execute noise if it is a function
        noise = self._try_execute_param(self.noise, variable)

        # try:
        #     previous_value = params[INITIALIZER]
        # except (TypeError, KeyError):

        previous_value = np.atleast_2d(self.previous_value)

        value = previous_value * rate + noise + increment

        # If this NOT an initialization run, update the old value
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if not context or not INITIALIZING in context:
            self.previous_value = value
        return value


# Note:  For any of these that correspond to args, value must match the name of the corresponding arg in __init__()
DRIFT_RATE = 'drift_rate'
DRIFT_RATE_VARIABILITY = 'DDM_DriftRateVariability'
THRESHOLD = 'threshold'
TRESHOLD_VARIABILITY = 'DDM_ThresholdRateVariability'
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
    BogaczEtAl(                                 \
        default_variable=ClassDefaults.variable,  \
        drift_rate=1.0,                         \
        threshold=1.0,                          \
        starting_point=0.0,                     \
        t0=0.2                                  \
        noise=0.5,                              \
        params=None,                            \
        owner=None,                             \
        prefs=None                              \
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

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


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

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the **prefs** argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = kwBogaczEtAl

    class ClassDefaults(IntegratorFunction.ClassDefaults):
        variable = [[0]]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=ClassDefaults.variable,
                 drift_rate:parameter_spec = 1.0,
                 starting_point: parameter_spec = 0.0,
                 threshold: parameter_spec = 1.0,
                 noise: parameter_spec = 0.5,
                 t0: parameter_spec = .200,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context='Integrator Init'):

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
                         context=context)

    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """
        Return: terminal value of decision variable (equal to threshold), mean accuracy (error rate; ER) and mean
        response time (RT)

        Arguments
        ---------

        variable : 2d np.array
            ignored.

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------
        Decision variable, mean ER, mean RT : (float, float, float)

        """

        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        drift_rate = float(self.drift_rate) * float(variable)
        threshold = float(self.threshold)
        starting_point = float(self.starting_point)
        noise = float(self.noise)
        t0 = float(self.t0)

        self.bias = bias = (starting_point + threshold) / (2 * threshold)

        # Prevents div by 0 issue below:
        if bias <= 0:
            bias = 1e-8
        if bias >= 1:
            bias = 1 - 1e-8

        # drift_rate close to or at 0 (avoid float comparison)
        if abs(drift_rate) < 1e-8:
            # back to absolute bias in order to apply limit
            bias_abs = bias * 2 * threshold - threshold
            # use expression for limit a->0 from Srivastava et al. 2016
            rt = t0 + (threshold ** 2 - bias_abs ** 2) / (noise ** 2)
            er = (threshold - bias_abs) / (2 * threshold)
        else:
            drift_rate_normed = abs(drift_rate)
            ztilde = threshold / drift_rate_normed
            atilde = (drift_rate_normed / noise) ** 2

            is_neg_drift = drift_rate < 0
            bias_adj = (is_neg_drift == 1) * (1 - bias) + (is_neg_drift == 0) * bias
            y0tilde = ((noise ** 2) / 2) * np.log(bias_adj / (1 - bias_adj))
            if abs(y0tilde) > threshold:
                y0tilde = -1 * (is_neg_drift == 1) * threshold + (is_neg_drift == 0) * threshold
            x0tilde = y0tilde / drift_rate_normed

            import warnings
            warnings.filterwarnings('error')

            try:
                rt = ztilde * tanh(ztilde * atilde) + \
                     ((2 * ztilde * (1 - exp(-2 * x0tilde * atilde))) / (
                     exp(2 * ztilde * atilde) - exp(-2 * ztilde * atilde)) - x0tilde) + t0
                er = 1 / (1 + exp(2 * ztilde * atilde)) - \
                     ((1 - exp(-2 * x0tilde * atilde)) / (exp(2 * ztilde * atilde) - exp(-2 * ztilde * atilde)))

            except (Warning):
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

        Calculate the derivative of 1/(reward rate) with respect to the threshold (**output** arg)
        and drift_rate (**input** arg).  Reward rate (RR) is assumed to be:

            RR = (delay\\ :sub:`ITI` + Z/A + ED);

        the derivative of 1/RR with respect to the `threshold <BogaczEtAl.threshold>` is:

            1/A - E/A - (2A/c\\ :sup:`2`\\ )ED;

        and the derivative of 1/RR with respect to the `drift_rate <BogaczEtAl.drift_rate>` is:

            -Z/A\\ :sup:`2` + (Z/A\\ :sup:`2`\\ )E - (2Z/c\\ :sup:`2`\\ )ED

        where:

            A = `drift_rate <BogaczEtAl.drift_rate>`,

            Z = `threshold <BogaczEtAl.threshold>`,

            c = `noise <BogaczEtAl.noise>`,

            E = exp(-2ZA/\\ c\\ :sup:`2`\\ ), and

            D = delay\\ :sub:`ITI` + delay\\ :sub:`penalty` - Z/A

            delay\\ :sub:`ITI` is the intertrial interval and delay\\ :sub:`penalty` is a penalty delay.


        Returns
        -------

        derivatives :  List[float, float)
            of 1/RR with respect to `threshold <BogaczEtAl.threshold>` and `drift_rate <BogaczEtAl.drift_rate>`.

        """
        Z = output or self.threshold
        A = input or self.drift_rate
        c = self.noise
        c_sq = c**2
        E = exp(-2*Z*A/c_sq)
        D_iti = 0
        D_pen = 0
        D = D_iti + D_pen
        # RR =  1/(D_iti + Z/A + (E*D))

        dRR_dZ = 1/A + E/A + (2*A/c_sq)*E*D
        dRR_dA = -Z/A**2 + (Z/A**2)*E - (2*Z/c_sq)*E*D

        return [dRR_dZ, dRR_dA]


# Results from Navarro and Fuss DDM solution (indices for return value tuple)
class NF_Results(AutoNumber):
    RESULT = ()
    MEAN_ER = ()
    MEAN_DT = ()
    PLACEMARKER = ()
    MEAN_CORRECT_RT = ()
    MEAN_CORRECT_VARIANCE = ()
    MEAN_CORRECT_SKEW_RT = ()

# ----------------------------------------------------------------------------
class NavarroAndFuss(IntegratorFunction):
    """
    NavarroAndFuss(                             \
        default_variable=ClassDefaults.variable,  \
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

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


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

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the **prefs** argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = kwNavarrosAndFuss

    class ClassDefaults(IntegratorFunction.ClassDefaults):
        variable = [[0]]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=ClassDefaults.variable,
                 drift_rate: parameter_spec = 1.0,
                 starting_point: parameter_spec = 0.0,
                 threshold: parameter_spec = 1.0,
                 noise: parameter_spec = 0.5,
                 t0: parameter_spec = .200,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context='Integrator Init'):
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
                         context=context)

    def _instantiate_function(self, context=None):
        print("\nimporting matlab...")
        import matlab.engine
        self.eng1 = matlab.engine.start_matlab('-nojvm')
        print("matlab imported\n")

        super()._instantiate_function(context=context)

    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """
        Return: terminal value of decision variable, mean accuracy (error rate; ER), mean response time (RT),
        correct RT mean, correct RT variance and correct RT skew.  **Requires that the MatLab engine is installed.**

        Arguments
        ---------

        variable : 2d np.array
            ignored.

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------
        Decision variable, mean ER, mean RT, correct RT mean, correct RT variance, correct RT skew : \
        (float, float, float, float, float, float)

        """

        self._check_args(variable=variable, params=params, context=context)

        drift_rate = float(self.drift_rate)
        threshold = float(self.threshold)
        starting_point = float(self.starting_point)
        noise = float(self.noise)
        t0 = float(self.t0)

        # print("\nimporting matlab...")
        # import matlab.engine
        # eng1 = matlab.engine.start_matlab('-nojvm')
        # print("matlab imported\n")
        results = self.eng1.ddmSim(drift_rate, starting_point, threshold, noise, t0, 1, nargout=5)

        return results


# region ************************************   DISTRIBUTION FUNCTIONS   ***********************************************

class DistributionFunction(Function_Base):
    componentType = DIST_FUNCTION_TYPE


class NormalDist(DistributionFunction):
    """
    NormalDist(                      \
             mean=0.0,             \
             standard_dev=1.0,             \
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
        Standard deviation of the normal distribution

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    mean : float : default 0.0
        The mean or center of the normal distribution

    standard_dev : float : default 1.0
        Standard deviation of the normal distribution

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    """

    componentName = NORMAL_DIST_FUNCTION

    class ClassDefaults(DistributionFunction.ClassDefaults):
        variable = [0]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=ClassDefaults.variable,
                 mean=0.0,
                 standard_dev=1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context=componentName + INITIALIZING):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(mean=mean,
                                                  standard_dev=standard_dev,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        self.functionOutputType = None

    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        # Validate variable and validate params
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        mean = self.paramsCurrent[DIST_MEAN]
        standard_dev = self.paramsCurrent[STANDARD_DEVIATION]

        result = standard_dev * np.random.normal() + mean

        return result


class ExponentialDist(DistributionFunction):
    """
    ExponentialDist(                      \
             beta=1.0,             \
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

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    beta : float : default 1.0
        The scale parameter of the exponential distribution

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    """
    componentName = EXPONENTIAL_DIST_FUNCTION

    class ClassDefaults(DistributionFunction.ClassDefaults):
        variable = [0]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=ClassDefaults.variable,
                 beta=1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context=componentName + INITIALIZING):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(beta=beta,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        self.functionOutputType = None

    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        # Validate variable and validate params
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        beta = self.paramsCurrent[BETA]

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

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    low : float : default 0.0
        Lower bound of the uniform distribution

    high : float : default 1.0
        Upper bound of the uniform distribution

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    """
    componentName = UNIFORM_DIST_FUNCTION

    class ClassDefaults(DistributionFunction.ClassDefaults):
        variable = [0]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=ClassDefaults.variable,
                 low=0.0,
                 high=1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context=componentName + INITIALIZING):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(low=low,
                                                  high=high,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        self.functionOutputType = None

    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        # Validate variable and validate params
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        low = self.paramsCurrent[LOW]
        high = self.paramsCurrent[HIGH]

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

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    scale : float : default 1.0
        The dist_shape of the gamma distribution. Should be greater than zero.

    dist_shape : float : default 1.0
        The scale of the gamma distribution. Should be greater than zero.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    """

    componentName = GAMMA_DIST_FUNCTION

    class ClassDefaults(DistributionFunction.ClassDefaults):
        variable = [0]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=ClassDefaults.variable,
                 scale=1.0,
                 dist_shape=1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context=componentName + INITIALIZING):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(scale=scale,
                                                  dist_shape=dist_shape,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        self.functionOutputType = None

    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        # Validate variable and validate params
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        scale = self.paramsCurrent[SCALE]
        dist_shape = self.paramsCurrent[DIST_SHAPE]

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

     params : Optional[Dict[param keyword, param value]]
         a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
         function.  Values specified for parameters in the dictionary override any assigned to those parameters in
         arguments of the constructor.

     owner : Component
         `component <Component>` to which to assign the Function.

     prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
         the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
         defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


     Attributes
     ----------

     scale : float : default 1.0
         Scale parameter of the Wald distribution. Should be greater than zero.

     mean : float : default 1.0
         Mean of the Wald distribution. Should be greater than or equal to zero.

     params : Optional[Dict[param keyword, param value]]
         a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
         function.  Values specified for parameters in the dictionary override any assigned to those parameters in
         arguments of the constructor.

     owner : Component
         `component <Component>` to which to assign the Function.

     prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
         the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
         defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


     """

    componentName = WALD_DIST_FUNCTION

    class ClassDefaults(DistributionFunction.ClassDefaults):
        variable = [0]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=ClassDefaults.variable,
                 scale=1.0,
                 mean=1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context=componentName + INITIALIZING):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(scale=scale,
                                                  mean=mean,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        self.functionOutputType = None

    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        # Validate variable and validate params
        variable = self._update_variable(self._check_args(variable=variable, params=params, context=context))

        scale = self.paramsCurrent[SCALE]
        mean = self.paramsCurrent[DIST_MEAN]

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
        default_variable=ClassDefaults.variable,  \
        matrix=HOLLOW_MATRIX,                   \
        metric=ENERGY                           \
        transfer_fct=None                       \
        normalize=False,                        \
        params=None,                            \
        owner=None,                             \
        prefs=None                              \
        )

    .. _Stability:

    Return the stability of a vector based an a weight matrix from each element to every other element in the vector.
    The value of `variable <Stability.variable>` is passed through the `matrix <Stability.matrix>`, transformed
    using the `transfer_fct <Stability.transfer_fct>` (if specified), and then compared with its initial value
    using the specified `metric <Stability.metric>`.  If `normalize <Stability.normalize>` is specified, the result
    is normalized by the number of elements in the `variable <Stability.variable>`.

    Arguments
    ---------

    variable : list of numbers or 1d np.array : Default ClassDefaults.variable
        the array for which stabilty is calculated.

    matrix : list, np.ndarray, np.matrix, function keyword, or MappingProjection : default HOLLOW_MATRIX
        specifies the matrix of recurrent weights;  must be a square matrix with the same width as the
        length of `variable <Stability.variable>`.

    metric : ENERGY, ENTROPY or keyword in DISTANCE_METRICS : Default ENERGY
        specifies the metric used to compute stability.

    transfer_fct : function or method : Default None
        specifies the function used to transform output of weight `matrix <Stability.matrix>`.

    normalize : bool : Default False
        specifies whether to normalize the stability value by the length of `variable <Stability.variable>`.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).

    Attributes
    ----------

    variable : 1d np.array
        array for which stability is calculated.

    matrix : list, np.ndarray, np.matrix, function keyword, or MappingProjection : default HOLLOW_MATRIX
        weight matrix from each element of `variable <Stability.variablity>` to each other;  if a matrix other
        than HOLLOW_MATRIX is assigned, it is convolved with HOLLOW_MATRIX to eliminate self-connections from the
        stability calculation.

    metric : ENERGY, ENTROPY or keyword in DISTANCE_METRICS
        metric used to compute stability.  If ENTROPY or DISTANCE_METRICS keyword is used, the `Distance` Function
        is used to compute the stability of `variable <Stability.variable>` with respect to its value after
        transformation by `matrix <Stability.matrix>` and `transfer_fct <Stability.transfer_fct>`.

    transfer_fct : function or method
        function used to transform output of weight `matrix <Stability.matrix>` prior to computing stability.

    normalize : bool
        if `True`, result of stability calculation is normalized by the length of `variable <Stability.variable>`.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).
     """

    from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
    from PsyNeuLink.Components.States.ParameterState import ParameterState

    componentName = STABILITY_FUNCTION

    class ClassDefaults(ObjectiveFunction.ClassDefaults):
        variable = [0]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=ClassDefaults.variable,
                 matrix:tc.any(is_matrix, MappingProjection, ParameterState)=HOLLOW_MATRIX,
                 # metric:is_distance_metric=ENERGY,
                 metric:tc.any(tc.enum(ENERGY, ENTROPY), is_distance_metric)=ENERGY,
                 transfer_fct:tc.optional(tc.any(function_type, method_type))=None,
                 normalize:bool=False,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context=componentName + INITIALIZING):
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
                         context=context)

        self.functionOutputType = None

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate matrix param

        `matrix <Stability.matrix>` argument must be one of the following
            - 2d list, np.ndarray or np.matrix
            - ParameterState for one of the above
            - MappingProjection with a parameterStates[MATRIX] for one of the above

        Parse matrix specification to insure it resolves to a square matrix
        (but leave in the form in which it was specified so that, if it is a ParameterState or MappingProjection,
         its current value can be accessed at runtime (i.e., it can be used as a "pointer")
        """

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        # Validate error_matrix specification
        if MATRIX in target_set:

            from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
            from PsyNeuLink.Components.States.ParameterState import ParameterState

            matrix = target_set[MATRIX]

            if isinstance(matrix, MappingProjection):
                try:
                    matrix = matrix._parameter_states[MATRIX].value
                    param_type_string = "MappingProjection's ParameterState"
                except KeyError:
                    raise FunctionError("The MappingProjection specified for the {} arg of {} ({}) must have a {} "
                                        "paramaterState that has been assigned a 2d array or matrix".
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
            size = len(np.squeeze(self.instance_defaults.variable))

            if rows != size:
                raise FunctionError("The value of the {} specified for the {} arg of {} is the wrong size;"
                                    "it is {}x{}, but must be square matrix of size {}".
                                    format(param_type_string, MATRIX, self.name, rows, cols, size))

            if rows != cols:
                raise FunctionError("The value of the {} specified for the {} arg of {} ({}) "
                                    "must be a square matrix".
                                    format(param_type_string, MATRIX, self.name, matrix))


    def _instantiate_attributes_before_function(self, context=None):
        """Instantiate matrix

        Specified matrix specified is convolved with HOLLOW_MATRIX
            to eliminate the diagonal (self-connections) from the calculation.
        The `Distance` Function is used for all calculations except ENERGY (which is not really a distance metric).
        If ENTROPY is specified as the metric, convert to CROSS_ENTROPY for use with the Distance Function.

        """

        size = len(np.squeeze(self.instance_defaults.variable))

        from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
        from PsyNeuLink.Components.States.ParameterState import ParameterState
        if isinstance(self.matrix,MappingProjection):
            self._matrix = self.matrix._parameter_states[MATRIX]
        elif isinstance(self.matrix,ParameterState):
            pass
        else:
            self._matrix = get_matrix(self.matrix, size, size)

        self._hollow_matrix = get_matrix(HOLLOW_MATRIX,size, size)

        if self.metric is ENTROPY:
            self._metric_fct = Distance(metric=CROSS_ENTROPY)

        elif self.metric in DISTANCE_METRICS:
            self._metric_fct = Distance(metric=self.metric)


    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
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

        from PsyNeuLink.Components.States.ParameterState import ParameterState
        if isinstance(self.matrix, ParameterState):
            matrix = self.matrix.value
        else:
            matrix = self.matrix

        current = variable
        if self.transfer_fct is not None:
            transformed = self.transfer_fct(np.dot(matrix * self._hollow_matrix, variable))
        else:
            transformed = np.dot(matrix * self._hollow_matrix, variable)

        if self.metric is ENERGY:
            result = -np.sum(current * transformed)
        else:
            result = self._metric_fct.function(variable=[current,transformed], context=context)

        if self.normalize:
            result /= len(variable)

        return result

# endregion

class Distance(ObjectiveFunction):
    """
    Distance(                                  \
       default_variable=ClassDefaults.variable,  \
       metric=EUCLIDEAN                        \
       normalize=False,                        \
       params=None,                            \
       owner=None,                             \
       prefs=None                              \
       )

    .. _Distance:

    Return the distance between two vectors based on a specified metric.

    Arguments
    ---------

    variable : 2d np.array with two items : Default ClassDefaults.variable
        the arrays between which the distance is calculated.

    metric : keyword in DISTANCE_METRICS : Default EUCLIDEAN
        specifies the metric used to compute the distance between the two items in `variable <Distance.variable>`.

    normalize : bool : Default False
        specifies whether to normalize the distance by the length of `variable <Distance.variable>`.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    variable : 2d np.array with two items
        contains the arrays between which the distance is calculated.

    metric : keyword in DISTANCE_METRICS
        specifies the metric used to compute the distance between the two items in `variable <Distance.variable>`.

    normalize : bool
        specifies whether to normalize the distance by the length of `variable <Distance.variable>`.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).
    """

    componentName = DISTANCE_FUNCTION

    class ClassDefaults(ObjectiveFunction.ClassDefaults):
        variable = [[0],[0]]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=ClassDefaults.variable,
                 metric:tc.enum(PEARSON, EUCLIDEAN, DIFFERENCE, CROSS_ENTROPY, CORRELATION, ANGLE)=DIFFERENCE,
                 normalize:bool=False,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context=componentName + INITIALIZING):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(metric=metric,
                                                  normalize=normalize,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        self.functionOutputType = None

    def _validate_params(self, request_set, target_set=None, variable=None, context=None):
        """Validate that variable had two items of equal length

        """
        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        if len(variable) != 2:
            raise FunctionError("variable for {} ({}) must have two items".format(self.name, variable))

        if len(variable[0]) != len(variable[1]):
            raise FunctionError("The lengths of the items in the variable for {} ({},{}) must be equal".
                format(self.name, len(variable[0]), len(variable[1])))

    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """Calculate the distance between the two arrays in `variable <Stability.variable>`.

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

        # Cross-entropy of v1 and v2
        elif self.metric is CROSS_ENTROPY:
            # FIX: VALIDATE THAT ALL ELEMENTS OF V1 AND V2 ARE 0 TO 1
            if context is not None and INITIALIZING in context:
                v1 = np.where(v1==0, EPSILON, v1)
                v2 = np.where(v2==0, EPSILON, v2)
            result = -np.sum(v1*np.log(v2))

        # FIX: NEED SCIPY HERE
        # # Angle (cosyne) of v1 and v2
        # elif self.metric is ANGLE:
        #     result = scipy.spatial.distance.cosine(v1,v2)

        # Correlation of v1 and v2
        elif self.metric is CORRELATION:
            result = np.correlate(v1, v2)

        # Pearson Correlation of v1 and v2
        elif self.metric is PEARSON:
            result = np.corrcoef(v1, v2)


        if self.normalize:
            # if np.sum(denom):
            # result /= np.sum(x,y)
            result /= len(variable[0])

        return result

# endregion

# region **************************************   LEARNING FUNCTIONS ***************************************************

ReturnVal = namedtuple('ReturnVal', 'learning_signal, error_signal')

class LearningFunction(Function_Base):
    """Abstract class of `Function` used for learning.

    All LearningFunctions take three input values (specified in each of the three required items of the
    `variable` argument), and return two output values.

    Input values:
       * input to the parameter being modified (variable[0]);
       * output of the parameter being modified (variable[1]);
       * error associated with the output (variable[2]).

    Output values:
       * learning_signal: modifications calculated by the function that attempt to reduce the error;
       * error_signal: the error received, possibly modified by the function.

    """

    componentType = LEARNING_FUNCTION_TYPE

    # def __init__(self, default_variable, params, owner, prefs, context):
    #     super().__init__(default_variable=default_variable,
    #                      params=params,
    #                      owner=owner,
    #                      prefs=prefs,
    #                      context=context)
    #     self.return_val = return_val(None, None)


LEARNING_ACTIVATION_FUNCTION = 'activation_function'
LEARNING_ACTIVATION_INPUT = 0  # a(j)
# MATRIX = 1             # w
LEARNING_ACTIVATION_OUTPUT = 1  # a(i)
LEARNING_ERROR_OUTPUT = 2

class Reinforcement(
    LearningFunction):  # -------------------------------------------------------------------------------
    """
    Reinforcement(                                       \
        default_variable=ClassDefaults.variable,           \
        activation_function=SoftMax,                     \
        learning_rate=None,                              \
        params=None,                                     \
        name=None,                                       \
        prefs=None)

    Implements a function that calculates a diagonal matrix of weight changes using the reinforcement (delta)
    learning rule.

    COMMENT:
        Reinforcement learning rule
          [matrix]         [scalar]        [col array]
        delta_weight =  learning rate   *     error
          return     =  LEARNING_RATE  *  variable

        Reinforcement.function:
            variable must be a 1D np.array with three items (standard for learning functions)
                note: only the LEARNING_ACTIVATION_OUTPUT and LEARNING_ERROR_OUTPUT items are used by RL
            assumes matrix to which errors are applied is the identity matrix
                (i.e., set of "parallel" weights from input to output)
            LEARNING_RATE param must be a float
            returns matrix of weight changes

        Initialization arguments:
         - variable (list or np.array): must a single 1D np.array
         - params (dict): specifies
             + LEARNING_RATE: (float) - learning rate (default: 1.0)
    COMMENT

    Arguments
    ---------

    variable : List or 2d np.array [length 3] : default ClassDefaults.variable
       template for the three items provided as the variable in the call to the `function <Reinforcement.function>`
       (in order):
       `activation_input <Reinforcement.activation_input>` (1d np.array),
       `activation_output <Reinforcement.activation_output>` (1d np.array),
       `error_signal <Reinforcement.error_signal>` (1d np.array).

    activation_function : Function or function : SoftMax
        specifies the function of the Mechanism that generates `activation_output <Reinforcement.activation_output>`.

    learning_rate : float : default default_learning_rate
        supersedes any specification for the `Process` and/or `System` to which the function's
        `owner <Function.owner>` belongs (see `learning_rate <Reinforcement.learning_rate>` for details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    variable: 2d np.array
        specifies three values used as input to the `function <Reinforcement.function>`:
       `activation_input <Reinforcement.activation_input>`,
       `activation_output <Reinforcement.activation_output>`, and
       `error_signal <Reinforcement.error_signal>`.

    activation_input : 1d np.array
        first item of `variable <Reinforcement.variable>`;  this is not used (it is implemented for consistency
        with other `LearningFunctions <LearningFunction>`).

    activation_output : 1d np.array
        the output of the function for which the matrix being modified provides the input; must have a single non-zero
        value (corresponding to the selected "action").

    error_signal : 1d np.array
        the error signal associated with the `activation_output <Reinforcement.activation_output>`; must be the same
        length as `activation_output <Reinforcement.activation_output>` and must have a single non-zero value in the
        same position as the one in `activation_output <Reinforcement.activation_output>`.

    activation_function : Function or function : SoftMax
        the function of the Mechanism that generates `activation_output <Reinforcement.activation_output>`; must
        return and array with a single non-zero value.

    learning_rate : float
        the learning rate used by the function.  If specified, it supersedes any learning_rate specified for the
        `Process <Process_Base_Learning>` and/or `System <System_Learning>` to which the function's
        `owner <Reinforcement.owner>` belongs.  If it is `None`, then the `learning_rate <Process_Base.learning_rate>`
        specified for the Process to which the `owner <Reinforcement.owner>` belongs is used;  and, if that is `None`,
        then the `learning_rate <System_Base.learning_rate>` for the System to which it belongs is used. If all are
        `None`, then the `default_learning_rate <Reinforcement.default_learning_rate>` is used.

    default_learning_rate : float
        the value used for the `learning_rate <Reinforcement.learning_rate>` if it is not otherwise specified.

    function : function
         the function that computes the weight change matrix, and returns that along with the
         `error_signal <Reinforcement.error_signal>` received.

    owner : Mechanism
        `Mechanism <Mechanism>` to which the function belongs.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the **prefs** argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).
    """

    componentName = RL_FUNCTION

    class ClassDefaults(LearningFunction.ClassDefaults):
        variable = [[0], [0], [0]]

    default_learning_rate = 0.05

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    def __init__(self,
                 default_variable=ClassDefaults.variable,
                 activation_function: tc.any(SoftMax, tc.enum(SoftMax)) = SoftMax,  # Allow class or instance
                 learning_rate: tc.optional(parameter_spec) = None,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context='Component Init'):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(activation_function=activation_function,
                                                  learning_rate=learning_rate,
                                                  params=params)

        # self.return_val = ReturnVal(None, None)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

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

        # Allow initializion with zero but not during a run (i.e., when called from check_args())
        if not INITIALIZING in context:
            if np.count_nonzero(self.activation_output) != 1:
                raise ComponentError("First item ({}) of variable for {} must be an array with a single non-zero value "
                                     "(if output Mechanism being trained uses softmax,"
                                     " its \'output\' arg may need to be set to to PROB)".
                                     format(variable[LEARNING_ACTIVATION_OUTPUT], self.componentName))

        return variable

    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """Calculate a matrix of weight changes from a single (scalar) error term

        COMMENT:
            Assume output array has a single non-zero value chosen by the softmax function of the error_source
            Assume error is a single scalar value
            Assume weight matrix (for MappingProjection to error_source) is a diagonal matrix
                (one weight for corresponding pairs of elements in the input and output arrays)
            Adjust the weight corresponding to  chosen element of the output array, using error value and learning rate

            Note: assume variable is a 2D np.array with three items (input, output, error)
                  for compatibility with other learning functions (and calls from LearningProjection)

        COMMENT

        Arguments
        ---------

        variable : List or 2d np.array [length 3] : default ClassDefaults.variable
           must have three items that are the values for (in order):
           `activation_input <Reinforcement.activation_input>` (not used),
           `activation_output <Reinforcement.activation_output>` (1d np.array with a single non-zero value),
           `error_signal <Reinforcement.error_signal>` (1d np.array).

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------
        error signal : 1d np.array
            same as value received in `error_signal <Reinforcement.error_signal>` argument.

        diagonal weight change matrix : 2d np.array
            has a single non-zero entry in the same row and column as the one in
            `activation_output <Reinforcement.activation_output>` and `error_signal <Reinforcement.error_signal>`.
        """

        self._check_args(variable=variable, params=params, context=context)

        output = self.activation_output
        error = self.error_signal

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
        # # MODIFIED 3/22/17 NEWER:
        # learning_rate = self.learning_rate
        # MODIFIED 3/22/17 END

        # Assign error term to chosen item of output array
        error_array = (np.where(output, learning_rate * error, 0))

        # Construct weight change matrix with error term in proper element
        weight_change_matrix = np.diag(error_array)

        # self.return_val.error_signal = error_array
        # self.return_val.learning_signal = weight_change_matrix
        #
        # # return:
        # # - weight_change_matrix and error_array
        # return list(self.return_val)
        return [weight_change_matrix, error_array]


# Argument names:
ERROR_MATRIX = 'error_matrix'
WT_MATRIX_SENDERS_DIM = 0
WT_MATRIX_RECEIVERS_DIM = 1


class BackPropagation(LearningFunction):
    """
    BackPropagation(                                     \
        default_variable=ClassDefaults.variable,           \
        activation_derivative_fct=Logistic().derivative, \
        error_derivative_fct=Logistic().derivative,      \
        error_matrix=None,                               \
        learning_rate=None,                              \
        params=None,                                     \
        name=None,                                       \
        prefs=None)

    Implements a function that calculate a matrix of weight changes using the backpropagation
    (`Generalized Delta Rule <http://www.nature.com/nature/journal/v323/n6088/abs/323533a0.html>`_) learning algorithm.

    COMMENT:
        Description:
            Backpropagation learning algorithm (Generalized Delta Rule):
              [matrix]         [scalar]     [row array]              [row array/ col array]                [col array]
            delta_weight =  learning rate *   input      *            d(output)/d(input)                 *     error
              return     =  LEARNING_RATE * variable[0]  *  kwTransferFctDeriv(variable[1],variable[0])  *  variable[2]

    COMMENT

    Arguments
    ---------

    variable : List or 2d np.array [length 3] : default ClassDefaults.variable
       specifies a template for the three items provided as the variable in the call to the
       `function <BackPropagation.function>` (in order):
       `activation_input <BackPropagation.activation_input>` (1d np.array),
       `activation_output <BackPropagation.activation_output>` (1d np.array),
       `error_signal <BackPropagation.error_signal>` (1d np.array).

    activation_derivative : Function or function
        specifies the derivative for the function of the Mechanism that generates
        `activation_output <BackPropagation.activation_output>`.

    error_derivative : Function or function
        specifies the derivative for the function of the Mechanism that is the receiver of the
        `error_matrix <BackPropagation.error_matrix>`.

    error_matrix : List, 2d np.array, np.matrix, ParameterState, or MappingProjection
        matrix, the output of which is used to calculate the `error_signal <BackPropagation.error_signal>`.
        If it is specified as a ParameterState it must be one for the `matrix <MappingProjection.matrix>`
        parameter of a `MappingProjection`;  if it is a MappingProjection, it must be one with a
        MATRIX parameterState.

    learning_rate : float : default default_learning_rate
        supersedes any specification for the `Process` and/or `System` to which the function's
        `owner <Function.owner>` belongs (see `learning_rate <BackPropagation.learning_rate>` for details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    variable: 2d np.array
        contains the three values used as input to the `function <BackPropagation.function>`:
       `activation_input <BackPropagation.activation_input>`,
       `activation_output <BackPropagation.activation_output>`, and
       `error_signal <BackPropagation.error_signal>`.

    activation_input : 1d np.array
        the input to the matrix being modified; same as 1st item of `variable <BackPropagation.variable>.

    activation_output : 1d np.array
        the output of the function for which the matrix being modified provides the input;
        same as 2nd item of `variable <BackPropagation.variable>.

    error_signal : 1d np.array
        the error signal for the next matrix (layer above) in the learning sequence, or the error computed from the
        target (training signal) and the output of the last Mechanism in the sequence;
        same as 3rd item of `variable <BackPropagation.variable>.

    error_matrix : 2d np.array or ParameterState
        matrix, the output of which is used to calculate the `error_signal <BackPropagation.error_signal>`;
        if it is a `ParameterState`, it refers to the MATRIX parameterState of the `MappingProjection` being learned.

    learning_rate : float
        the learning rate used by the function.  If specified, it supersedes any learning_rate specified for the
        `process <Process.learning_Rate>` and/or `system <System.learning_rate>` to which the function's  `owner
        <BackPropagation.owner>` belongs.  If it is `None`, then the learning_rate specified for the process to
        which the `owner <BackPropagationowner>` belongs is used;  and, if that is `None`, then the learning_rate for
        the system to which it belongs is used. If all are `None`, then the
        `default_learning_rate <BackPropagation.default_learning_rate>` is used.

    default_learning_rate : float
        the value used for the `learning_rate <BackPropagation.learning_rate>` if it is not otherwise specified.

    function : function
         the function that computes the weight change matrix, and returns that along with the
         `error_signal <BackPropagation.error_signal>` received, weighted by the contribution made by each element of
         `activation_output <BackPropagation.activation_output>` as a function of the
         `error_matrix <BackPropagation.error_matrix>`.

    owner : Mechanism
        `Mechanism <Mechanism>` to which the function belongs.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the **prefs** argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = BACKPROPAGATION_FUNCTION

    class ClassDefaults(LearningFunction.ClassDefaults):
        variable = [[0], [0], [0]]

    default_learning_rate = 1.0

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=ClassDefaults.variable,
                 # default_variable:tc.any(list, np.ndarray),
                 activation_derivative_fct: tc.optional(tc.any(function_type, method_type)) = Logistic().derivative,
                 error_derivative_fct: tc.optional(tc.any(function_type, method_type)) = Logistic().derivative,
                 error_matrix=None,
                 learning_rate: tc.optional(parameter_spec) = None,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context='Component Init'):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(activation_derivative_fct=activation_derivative_fct,
                                                  error_derivative_fct=error_derivative_fct,
                                                  error_matrix=error_matrix,
                                                  learning_rate=learning_rate,
                                                  params=params)

        # self.return_val = ReturnVal(None, None)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

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
        """Validate error_matrix param

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

        # Validate error_matrix specification
        if ERROR_MATRIX in target_set:

            error_matrix = target_set[ERROR_MATRIX]

            from PsyNeuLink.Components.States.ParameterState import ParameterState
            from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
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
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """Calculate and return a matrix of weight changes from arrays of inputs, outputs and error terms

        Arguments
        ---------

        variable : List or 2d np.array [length 3] : default ClassDefaults.variable
           must have three items that are the values for (in order):
           `activation_input <BackPropagation.activation_input>` (1d np.array),
           `activation_output <BackPropagation.activation_output>` (1d np.array),
           `error_signal <BackPropagation.error_signal>` (1d np.array).

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------

        weighted error signal : 1d np.array
            `error_signal <BackPropagation.error_signal>`, weighted by the contribution made by each element of
            `activation_output <BackPropagation.activation_output>` as a function of
            `error_matrix <BackPropagation.error_matrix>`.

        weight change matrix : 2d np.array
            the modifications to make to the matrix.
        """

        self._check_args(variable=variable, params=params, context=context)

        from PsyNeuLink.Components.States.ParameterState import ParameterState
        if isinstance(self.error_matrix, ParameterState):
            error_matrix = self.error_matrix.value
        else:
            error_matrix = self.error_matrix

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
        dE_dA = np.dot(error_matrix, self.error_signal)

        # Derivative of the output activity
        dA_dW = self.activation_derivative_fct(input=self.activation_input, output=self.activation_output)

        # Chain rule to get the derivative of the error with respect to the weights
        dE_dW = dE_dA * dA_dW

        # Weight changes = delta rule (learning rate * activity * error)
        weight_change_matrix = learning_rate * activation_input * dE_dW

        # # TEST PRINT:
        # if context and not 'INIT' in context:
        #     print("\nBACKPROP for {}:\n    "
        #           "-input: {}\n    "
        #           "-error_signal (dE_DA): {}\n    "
        #           "-derivative (dA_dW): {}\n    "
        #           "-error_derivative (dE_dW): {}\n".
        #           format(self.owner.name, self.activation_input, dE_dA, dA_dW ,dE_dW))

        # self.return_val.error_signal = dE_dW
        # self.return_val.learning_signal = weight_change_matrix
        #
        # return list(self.return_val)

        return [weight_change_matrix, dE_dW]

# region *****************************************   OBJECTIVE FUNCTIONS ***********************************************
# endregion
# TBI

# region  *****************************************   REGISTER FUNCTIONS ***********************************************

# region


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
