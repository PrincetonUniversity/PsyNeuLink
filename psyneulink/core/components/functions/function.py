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


.. _Function_Overview:

Overview
--------

A Function is a `Component <Component>` that "packages" a function for use by other Components.
Every Component in PsyNeuLink is assigned a Function; when that Component is executed, its
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
  PsyNeuLink Components (e.g., `ModulatoryMechanisms`) by way of `ControlProjections <ControlProjection>`.
..
* **Modularity** -- by providing a standard interface, any Function assigned to a Components in PsyNeuLink can be
  replaced with other PsyNeuLink Functions, or with user-written custom functions so long as they adhere to certain
  standards (the PsyNeuLink `Function API <LINK>`).

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
  <Component>`, it *cannot* be (another) Function object (it can't be "turtles" all the way down!).

A Function also has an attribute for each of the parameters of its `function <Function_Base.function>`.

*Owner*
~~~~~~~

If a Function has been assigned to another `Component`, then it also has an `owner <Function_Base.owner>` attribute
that refers to that Component.  The Function itself is assigned as the Component's
`function <Component.function>` attribute.  Each of the Function's attributes is also assigned
as an attribute of the `owner <Function_Base.owner>`, and those are each associated with with a
`parameterPort <ParameterPort>` of the `owner <Function_Base.owner>`.  Projections to those parameterPorts can be
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
<ModulatorySignal>` to modulate the `function <Port_Base.function>` of a `Port <Port>` and thereby its `value
<Port_Base.value>` (see `ModulatorySignal_Modulation` and `figure <ModulatorySignal_Detail_Figure>` for additional
details). For example, a `ControlSignal` typically uses the `multiplicative_param` to modulate the value of a parameter
of a Mechanism's `function <Mechanism_Base.function>`, whereas a `LearningSignal` uses the `additive_param` to increment
the `value <ParamterPort.value>` of the `matrix <MappingProjection.matrix>` parameter of a `MappingProjection`.

COMMENT:
FOR DEVELOPERS:  'multiplicative_param` and `additive_param` are implemented as aliases to the relevant
parameters of a given Function, declared in its Parameters subclass declaration of the Function's declaration.
COMMENT


.. _Function_Execution:

Execution
---------

Functions are executable objects that can be called directly.  More commonly, however, they are called when
their `owner <Function_Base.owner>` is executed.  The parameters
of the `function <Function_Base.function>` can be modified when it is executed, by assigning a
`parameter specification dictionary <ParameterPort_Specification>` to the **params** argument in the
call to the `function <Function_Base.function>`.

For `Mechanisms <Mechanism>`, this can also be done by specifying `runtime_params <Composition_Runtime_Params>` in the
`Run` method of their `Composition`.

Class Reference
---------------

"""

import abc
import numbers
import types
import warnings
from enum import Enum, IntEnum

import numpy as np
import typecheck as tc

from psyneulink.core.components.component import ComponentError, DefaultsFlexibility
from psyneulink.core.components.shellclasses import Function, Mechanism
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import (
    ARGUMENT_THERAPY_FUNCTION, AUTO_ASSIGN_MATRIX, EXAMPLE_FUNCTION_TYPE, FULL_CONNECTIVITY_MATRIX,
    FUNCTION_COMPONENT_CATEGORY, FUNCTION_OUTPUT_TYPE, FUNCTION_OUTPUT_TYPE_CONVERSION, HOLLOW_MATRIX,
    IDENTITY_MATRIX, INVERSE_HOLLOW_MATRIX, NAME, PREFERENCE_SET_NAME, RANDOM_CONNECTIVITY_MATRIX
)
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import REPORT_OUTPUT_PREF, is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel
from psyneulink.core.globals.registry import register_category
from psyneulink.core.globals.utilities import (
    convert_to_np_array, get_global_seed, object_has_single_value, parameter_spec, safe_len
)

__all__ = [
    'ArgumentTherapy', 'EPSILON', 'Function_Base', 'function_keywords', 'FunctionError', 'FunctionOutputType',
    'FunctionRegistry', 'get_param_value_for_function', 'get_param_value_for_keyword', 'is_Function',
    'is_function_type', 'PERTINACITY', 'PROPENSITY'
]

EPSILON = np.finfo(float).eps

FunctionRegistry = {}

function_keywords = {FUNCTION_OUTPUT_TYPE, FUNCTION_OUTPUT_TYPE_CONVERSION}


class FunctionError(ComponentError):
    pass


class FunctionOutputType(IntEnum):
    RAW_NUMBER = 0
    NP_1D_ARRAY = 1
    NP_2D_ARRAY = 2
    DEFAULT = 3


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
    elif isinstance(x, (Function, types.FunctionType, types.MethodType, types.BuiltinFunctionType, types.BuiltinMethodType)):
        return True
    elif isinstance(x, type) and issubclass(x, Function):
        return True
    else:
        return False

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
        return owner.function.keyword(owner, keyword)
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
        return owner.function.param_function(owner, function)
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
# that every Parameters class has a single parent

# class ScaleOffsetParamMixin:
#     scale = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
#     offset = Parameter(1.0, modulable=True, aliases=[ADDITIVE_PARAM])


# Function Definitions *************************************************************************************************


# KDM 8/9/18: below is added for future use when function methods are completely functional
# used as a decorator for Function methods
# def enable_output_conversion(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         result = func(*args, **kwargs)
#         return convert_output_type(result)
#     return wrapper

# this should eventually be moved to a unified validation method
def _output_type_setter(value, owning_component):
    # Can't convert from arrays of length > 1 to number
    if (
        owning_component.defaults.variable is not None
        and safe_len(owning_component.defaults.variable) > 1
        and owning_component.output_type is FunctionOutputType.RAW_NUMBER
    ):
        raise FunctionError(
            f"{owning_component.__class__.__name__} can't be set to return a "
            "single number since its variable has more than one number."
        )

    # warn if user overrides the 2D setting for mechanism functions
    # may be removed when
    # https://github.com/PrincetonUniversity/PsyNeuLink/issues/895 is solved
    # properly(meaning Mechanism values may be something other than 2D np array)
    try:
        if (
            isinstance(owning_component.owner, Mechanism)
            and (
                value == FunctionOutputType.RAW_NUMBER
                or value == FunctionOutputType.NP_1D_ARRAY
            )
        ):
            warnings.warn(
                f'Functions that are owned by a Mechanism but do not return a '
                '2D numpy array may cause unexpected behavior if llvm '
                'compilation is enabled.'
            )
    except (AttributeError, ImportError):
        pass

    return value


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
            + componentCategory: FUNCTION_COMPONENT_CATEGORY
            + className (str): kwMechanismFunctionCategory
            + suffix (str): " <className>"
            + registry (dict): FunctionRegistry
            + classPreference (PreferenceSet): BasePreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.CATEGORY

        Class methods:
            none

        Instance attributes:
            + componentType (str):  assigned by subclasses
            + componentName (str):   assigned by subclasses
            + variable (value) - used as input to function's execute method
            + value (value) - output of execute method
            + name (str) - if not specified as an arg, a default based on the class is assigned in register_category
            + prefs (PreferenceSet) - if not specified as an arg, default is created by copying BasePreferenceSet

        Instance methods:
            The following method MUST be overridden by an implementation in the subclass:
            - execute(variable, params)
            The following can be implemented, to customize validation of the function variable and/or params:
            - [_validate_variable(variable)]
            - [_validate_params(request_set, target_set, context)]
    COMMENT

    Arguments
    ---------

    variable : value : default class_defaults.variable
        specifies the format and a default value for the input to `function <Function>`.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
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
        they are specified by the Function's :keyword:`class_defaults.variable`.

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
        the name of the Function; if it is not specified in the **name** argument of the constructor, a default is
        assigned by FunctionRegistry (see `Registry_Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see `Preferences`
        for details).

    """

    componentCategory = FUNCTION_COMPONENT_CATEGORY
    className = componentCategory
    suffix = " " + className

    registry = FunctionRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY

    _model_spec_id_parameters = 'args'

    _specified_variable_shape_flexibility = DefaultsFlexibility.INCREASE_DIMENSION

    class Parameters(Function.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <Function_Base.variable>`

                    :default value: numpy.array([0])
                    :type: ``numpy.ndarray``
                    :read only: True

                enable_output_type_conversion
                    see `enable_output_type_conversion <Function_Base.enable_output_type_conversion>`

                    :default value: False
                    :type: ``bool``

                output_type
                    see `output_type <Function_Base.output_type>`

                    :default value: FunctionOutputType.DEFAULT
                    :type: `FunctionOutputType`
        """
        variable = Parameter(np.array([0]), read_only=True, pnl_internal=True, constructor_argument='default_variable')

        output_type = Parameter(
            FunctionOutputType.DEFAULT,
            stateful=False,
            loggable=False,
            pnl_internal=True,
            valid_types=FunctionOutputType
        )
        enable_output_type_conversion = Parameter(False, stateful=False, loggable=False, pnl_internal=True)

    # Note: the following enforce encoding as 1D np.ndarrays (one array per variable)
    variableEncodingDim = 1

    @abc.abstractmethod
    def __init__(
        self,
        default_variable,
        params,
        owner=None,
        name=None,
        prefs=None,
        context=None,
        **kwargs
    ):
        """Assign category-level preferences, register category, and call super.__init__

        Initialization arguments:
        - default_variable (anything): establishes type for the variable, used for validation
        Note: if parameter_validation is off, validation is suppressed (for efficiency) (Function class default = on)

        :param default_variable: (anything but a dict) - value to assign as self.defaults.variable
        :param params: (dict) - params to be assigned as instance defaults
        :param log: (ComponentLog enum) - log entry types set in self.componentLog
        :param name: (string) - optional, overrides assignment of default (componentName of subclass)
        :return:
        """

        if self.initialization_status == ContextFlags.DEFERRED_INIT:
            self._assign_deferred_init_name(name, context)
            self._init_args[NAME] = name
            return

        register_category(entry=self,
                          base_class=Function_Base,
                          registry=FunctionRegistry,
                          name=name,
                          context=context)
        self.owner = owner

        super().__init__(
            default_variable=default_variable,
            param_defaults=params,
            name=name,
            prefs=prefs,
            **kwargs
        )

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __deepcopy__(self, memo):
        new = super().__deepcopy__(memo)
        # ensure copy does not have identical name
        register_category(new, Function_Base, new.name, FunctionRegistry)
        try:
            # HACK: Make sure any copies are re-seeded to avoid dependent RNG.
            new.random_state.seed([get_global_seed()])
        except:
            pass
        return new

    @handle_external_context()
    def function(self,
                 variable=None,
                 context=None,
                 params=None,
                 target_set=None,
                 **kwargs):
        # Validate variable and assign to variable, and validate params
        variable = self._check_args(variable=variable,
                                    context=context,
                                    params=params,
                                    target_set=target_set,
                                    )
        value = self._function(variable=variable,
                               context=context,
                               params=params,
                               **kwargs)
        self.most_recent_context = context
        self.parameters.value._set(value, context=context)
        return value

    @abc.abstractmethod
    def _function(
        self,
        variable=None,
        context=None,
        params=None,

    ):
        pass

    def _parse_arg_generic(self, arg_val):
        if isinstance(arg_val, list):
            return np.asarray(arg_val)
        else:
            return arg_val

    def _validate_parameter_spec(self, param, param_name, numeric_only=True):
        """Validates function param
        Replace direct call to parameter_spec in tc, which seems to not get called by Function __init__()'s
        """
        if not parameter_spec(param, numeric_only):
            owner_name = 'of ' + self.owner_name if self.owner else ""
            raise FunctionError(f"{param} is not a valid specification for "
                                f"the {param_name} argument of {self.__class__.__name__}{owner_name}.")

    def _get_current_function_param(self, param_name, context=None):
        if param_name == "variable":
            raise FunctionError(f"The method '_get_current_function_param' is intended for retrieving "
                                f"the current value of a function parameter. 'variable' is not a function parameter. "
                                f"If looking for {self.name}'s default variable, try {self.name}.defaults.variable.")
        try:
            return self.owner._parameter_ports[param_name].parameters.value._get(context)
        except (AttributeError, TypeError):
            try:
                return getattr(self.parameters, param_name)._get(context)
            except AttributeError:
                raise FunctionError(f"{self} has no parameter '{param_name}'.")

    def get_previous_value(self, context=None):
        # temporary method until previous values are integrated for all parameters
        value = self.parameters.previous_value._get(context)
        if value is None:
            value = self.parameters.previous_value._get(Context())

        return value

    def convert_output_type(self, value, output_type=None):
        if output_type is None:
            if not self.enable_output_type_conversion or self.output_type is None:
                return value
            else:
                output_type = self.output_type

        value = convert_to_np_array(value)

        # Type conversion (specified by output_type):

        # MODIFIED 6/21/19 NEW: [JDC]
        # Convert to same format as variable
        if isinstance(output_type, (list, np.ndarray)):
            shape = np.array(output_type).shape
            return np.array(value).reshape(shape)
        # MODIFIED 6/21/19 END

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
                    raise FunctionError(f"Can't convert value ({value}: 2D np.ndarray object "
                                        f"with more than one array) to 1D array.")
            elif value.ndim == 1:
                value = value
            elif value.ndim == 0:
                value = np.atleast_1d(value)
            else:
                raise FunctionError(f"Can't convert value ({value} to 1D array.")

        # Convert to raw number, irrespective of value type:
        # Note: if 2D or 1D array has more than two items, generate exception
        elif output_type is FunctionOutputType.RAW_NUMBER:
            if object_has_single_value(value):
                value = float(value)
            else:
                raise FunctionError(f"Can't convert value ({value}) with more than a single number to a raw number.")

        return value

    @property
    def owner_name(self):
        try:
            return self.owner.name
        except AttributeError:
            return '<no owner>'

    def _is_identity(self, context=None):
        # should return True in subclasses if the parameters for context are such that
        # the Function's output will be the same as its input
        # Used to bypass execute when unnecessary
        return False

    @property
    def _model_spec_parameter_blacklist(self):
        return super()._model_spec_parameter_blacklist.union({
            'multiplicative_param', 'additive_param',
        })

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

    variable : boolean or statement that resolves to one : default class_defaults.variable
        assertion for which a therapeutic response will be offered.

    propensity : Manner value : default Manner.CONTRARIAN
        specifies preferred therapeutic manner

    pertinacity : float : default 10.0
        specifies therapeutic consistency

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
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
        the name of the Function; if it is not specified in the **name** argument of the constructor, a default is
        assigned by FunctionRegistry (see `Registry_Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see `Preferences`
        for details).


    """

    # Function componentName and type (defined at top of module)
    componentName = ARGUMENT_THERAPY_FUNCTION
    componentType = EXAMPLE_FUNCTION_TYPE

    classPreferences = {
        PREFERENCE_SET_NAME: 'ExampleClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    }

    # Mode indicators
    class Manner(Enum):
        OBSEQUIOUS = 0
        CONTRARIAN = 1

    # Parameter class defaults
    # These are used both to type-cast the params, and as defaults if none are assigned
    #  in the initialization call or later (using either _instantiate_defaults or during a function call)

    def __init__(self,
                 default_variable=None,
                 propensity=10.0,
                 pertincacity=Manner.CONTRARIAN,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

        super().__init__(
            default_variable=default_variable,
            propensity=propensity,
            pertinacity=pertincacity,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _validate_variable(self, variable, context=None):
        """Validates variable and returns validated value

        This overrides the class method, to perform more detailed type checking
        See explanation in class method.
        Note: this method (or the class version) is called only if the parameter_validation attribute is `True`

        :param variable: (anything but a dict) - variable to be validated:
        :param context: (str)
        :return variable: - validated
        """

        if type(variable) == type(self.class_defaults.variable) or \
                (isinstance(variable, numbers.Number) and isinstance(self.class_defaults.variable, numbers.Number)):
            return variable
        else:
            raise FunctionError(f"Variable must be {type(self.class_defaults.variable)}.")

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

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """
        Returns a boolean that is (or tends to be) the same as or opposite the one passed in.

        Arguments
        ---------

        variable : boolean : default class_defaults.variable
           an assertion to which a therapeutic response is made.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        therapeutic response : boolean

        """
        # Compute the function
        statement = variable
        propensity = self._get_current_function_param(PROPENSITY, context)
        pertinacity = self._get_current_function_param(PERTINACITY, context)
        whim = np.random.randint(-10, 10)

        if propensity == self.Manner.OBSEQUIOUS:
            value = whim < pertinacity

        elif propensity == self.Manner.CONTRARIAN:
            value = whim > pertinacity

        else:
            raise FunctionError("This should not happen if parameter_validation == True;  check its value")

        return self.convert_output_type(value)



kwEVCAuxFunction = "EVC AUXILIARY FUNCTION"
kwEVCAuxFunctionType = "EVC AUXILIARY FUNCTION TYPE"
kwValueFunction = "EVC VALUE FUNCTION"
CONTROL_SIGNAL_GRID_SEARCH_FUNCTION = "EVC CONTROL SIGNAL GRID SEARCH FUNCTION"
CONTROLLER = 'controller'

class EVCAuxiliaryFunction(Function_Base):
    """Base class for EVC auxiliary functions
    """
    componentType = kwEVCAuxFunctionType

    class Parameters(Function_Base.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <Function_Base.variable>`

                    :default value: numpy.array([0])
                    :type: numpy.ndarray
                    :read only: True

        """
        variable = Parameter(None, pnl_internal=True, constructor_argument='default_variable')

    classPreferences = {
        PREFERENCE_SET_NAME: 'ValueFunctionCustomClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
       }

    @tc.typecheck
    def __init__(self,
                 function,
                 variable=None,
                 params=None,
                 owner=None,
                 prefs:is_pref_set=None,
                 context=None):
        self.aux_function = function

        super().__init__(default_variable=variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context,
                         function=function,
                         )


def get_matrix(specification, rows=1, cols=1, context=None):
    """Returns matrix conforming to specification with dimensions = rows x cols or None

     Specification can be a matrix keyword, filler value or np.ndarray

     Specification (validated in _validate_params):
        + single number (used to fill self.matrix)
        + matrix keyword:
            + AUTO_ASSIGN_MATRIX: IDENTITY_MATRIX if it is square, othwerwise FULL_CONNECTIVITY_MATRIX
            + IDENTITY_MATRIX: 1's on diagonal, 0's elsewhere (must be square matrix), otherwise generates error
            + HOLLOW_MATRIX: 0's on diagonal, 1's elsewhere (must be square matrix), otherwise generates error
            + INVERSE_HOLLOW_MATRIX: 0's on diagonal, -1's elsewhere (must be square matrix), otherwise generates error
            + FULL_CONNECTIVITY_MATRIX: all 1's
            + RANDOM_CONNECTIVITY_MATRIX (random floats uniformly distributed between 0 and 1)
        + 2D list or np.ndarray of numbers

     Returns 2D array with length=rows in dim 0 and length=cols in dim 1, or none if specification is not recognized
    """

    # Matrix provided (and validated in _validate_params); convert to array
    if isinstance(specification, (list, np.matrix)):
        return convert_to_np_array(specification)

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
        return 1 - np.identity(rows)

    if specification == INVERSE_HOLLOW_MATRIX:
        if rows != cols:
            raise FunctionError("Sender length ({}) must equal receiver length ({}) to use {}".
                                format(rows, cols, specification))
        return (1 - np.identity(rows)) * -1

    if specification == RANDOM_CONNECTIVITY_MATRIX:
        return np.random.rand(rows, cols)

    # Function is specified, so assume it uses random.rand() and call with sender_len and receiver_len
    if isinstance(specification, types.FunctionType):
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
