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
  * `BogaczEtAl`
  * `NavarroAndFuss`

Distribution Functions:
  * `NormalDist`
  * `ExponentialDist`
  * `UniformDist`
  * `GammaDist`
  * `WaldDist`

Learning Functions:
  * `Reinforcement`
  * `BackPropagation`

.. _Function_Overview:

Overview
--------

A Function is a `component <Component>` that "packages" a function (in its `function <Function_Base.function>` method)
for use by PsyNeuLink components.  Every `component <Component>` in PsyNeuLink is assigned a Function, and when that
component is executed, that Function's `function <Function_Base.function` is executed.  The
`function <Function_Base.function` can be any callable operation, although most commonly it is a mathematical operation
(and, for those, almost always uses calls to numpy function).

There are two reasons PsyNeuLink packages functions in a Function component. The first is to **manage parameters**.
Parameters are attributes of a function that either remain stable over multiple calls to the
function (e.g., the gain or bias of a logistic function, or the learning rate of a learning function);
or, if they change, they do less frequently or under the control of different factors than the function's variable
(i.e., its input).  As a consequence, it is useful to manage these separately from the function's variable,
and not have to provide them every time the function is called.  To address this, every PsyNeuLink Function has a
set of attributes corresponding to the parameters of the function, that can be specified at the time the Function is
created (in arguments to its constructor), and can be modified independently of a call to its :keyword:`function`.
Modifications can be directly (e.g., in a script), or by the operation of other PsyNeuLink components (e.g.,
`AdaptiveMechanisms`).  The second to reason PsyNeuLink uses Functions is for  **modularity**. By providing a standard
interface, any Function assigned to a components in PsyNeuLink can be replaced with other PsyNeuLink Functions, or with
user-written custom functions (so long as they adhere to certain standards (the PsyNeuLink `Function API <LINK>`).

.. _Function_Creation:

Creating a Function
-------------------

A Function can be created directly by calling its constructor.  Functions are also created automatically whenever
any other type of PsyNeuLink component is created (and its :keyword:`function` is not otherwise specified). The
constructor for a Function has an argument for its `variable <Function_Base.variable>` and each of the parameters of
its `function <Function_Base.function>`.  The `variable <Function_Base.variable>` argument is used both to format the
input to the `function <Function_Base.function>`, and assign its default value.  The arguments for each parameter can
be used to specify the default value for that parameter; the values can later be modified in various ways as described
below.

.. _Function_Structure:

Structure
---------

Every Function has a `variable <Function_Base.variable>` that provides the input to its
`function <Function_Base.function>` method.  It also has an attribute for each of the parameters of its `function
<Function_Base.function>`.   If a Function has been assigned to another component, then it also has an `owner
<Function_Base.owner>` attribute that refers to that component. Each of the Function's attributes is also assigned
as an attribute of the `owner <Function_Base.owner>`, and those are each associated with with a
`parameterState <ParameterState>` of the `owner <Function_Base.owner>`. Projections to those parameterStates can be
used to modify the Function's parameters.

COMMENT:
.. _Function_Output_Type_Conversion:

If the `function <Function_Base.function>` returns a single numeric value, and the Function's class implements
FunctionOutputTypeConversion, then the type of value returned by its `function <Function>` can be specified using the
`functionOutputType` attribute, by assigning it one of the following `FunctionOutputType` values:
    * FunctionOutputType.RAW_NUMBER: return "exposed" number;
    * FunctionOutputType.NP_1D_ARRAY: return 1d np.array
    * FunctionOutputType.NP_2D_ARRAY: return 2d np.array.

To implement FunctionOutputTypeConversion, the Function's kwFunctionOutputTypeConversion parameter must set to True,
and function type conversion must be implemented by its `function <Function_Base.function>` method
(see `Linear` for an example).
COMMENT

.. _Function_Execution:

Execution
---------

Functions are not executable objects, but their `function <Function_Base.function>` can be called.   This can be done
directly.  More commonly, however, they are called when their `owner <Function_Base.owner>` is executed.  The parameters
of the `function <Function_Base.function>` can be modified when it is executed, by assigning a
`parameter specification dictionary <Mechanism_Creation>` to the `Function.params` argument in the call to the
`function <Function_Base.function>`.  For `mechanisms <Mechanism>`, this can also be done by specifying `runtime_params
<Mechanism_Runtime_Parameters>` for the mechanism when it is `executed <Mechanism_Base.execute>`.

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
#            'Reinforcement',
#            'BackPropagation',
#            'FunctionError',
#            "FunctionOutputType"]

from functools import reduce
from operator import *
from random import randint
from numpy import sqrt, abs, tanh, exp
import numpy as np

import typecheck as tc

from PsyNeuLink.Components.ShellClasses import *
from PsyNeuLink.Globals.Registry import register_category
from PsyNeuLink.Globals.Keywords import *
from PsyNeuLink.Globals.Utilities import random_matrix

FunctionRegistry = {}


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
        if owner.prefs.verbosePref:
            print("{} of {}".format(e, owner.name))
        return None
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
         variable_default,   \
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
            NOTE:   the Function category definition serves primarily as a shell, and an interface to the Function class,
                       to maintain consistency of structure with the other function categories;
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
              - calling the _assign_defaults method (which changes their default values)
              - including them in a call the function method (which changes their values for just for that call)
            - Parameters must be specified in a params dictionary:
              - the key for each entry should be the name of the parameter (used also to name associated projections)
              - the value for each entry is the value of the parameter

        Return values:
            The functionOutputType can be used to specify type conversion for single-item return values:
            - it can only be used for numbers or a single-number list; other values will generate an exception
            - if self.functionOutputType is set to:
                FunctionOutputType.RAW_NUMBER, return value is "exposed" as a number
                FunctionOutputType.NP_1D_ARRAY, return value is 1d np.array
                FunctionOutputType.NP_2D_ARRAY, return value is 2d np.array
            - it must be enabled for a subclass by setting params[kwFunctionOutputTypeConversion] = True
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
            + paramClassDefaults (dict): {kwFunctionOutputTypeConversion: :keyword:`False`}

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

    variable : value : default variableClassDefault
        specifies the format and a default value for the input to `function <Function>`.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
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
        they are specified by the Function's :keyword:`variableClassDefault`.

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
        the `PreferenceSet` for function. Specified in the `prefs` argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentCategory = kwComponentCategory
    className = componentCategory
    suffix = " " + className

    registry = FunctionRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY

    variableClassDefault = None
    variableClassDefault_locked = False

    # Note: the following enforce encoding as 1D np.ndarrays (one array per variable)
    variableEncodingDim = 1

    paramClassDefaults = Component.paramClassDefaults.copy()
    paramClassDefaults.update({kwFunctionOutputTypeConversion: False})  # Enable/disable output type conversion

    def __init__(self,
                 variable_default,
                 params,
                 owner=None,
                 name=None,
                 prefs=None,
                 context='Function_Base Init'):
        """Assign category-level preferences, register category, and call super.__init__

        Initialization arguments:
        - variable_default (anything): establishes type for the variable, used for validation
        - params_default (dict): assigned as paramInstanceDefaults
        Note: if parameter_validation is off, validation is suppressed (for efficiency) (Function class default = on)

        :param variable_default: (anything but a dict) - value to assign as variableInstanceDefault
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

        super().__init__(variable_default=variable_default,
                         param_defaults=params,
                         name=name,
                         prefs=prefs,
                         context=context)

    def execute(self, variable=None, params=None, context=None):
        return self.function(variable=variable, params=params, context=context)

    @property
    def functionOutputType(self):
        if self.paramsCurrent[kwFunctionOutputTypeConversion]:
            return self._functionOutputType
        return None

    @functionOutputType.setter
    def functionOutputType(self, value):

        if not value and not self.paramsCurrent[kwFunctionOutputTypeConversion]:
            self._functionOutputType = value
            return

        # Attempt to set outputType but conversion not enabled
        if value and not self.paramsCurrent[kwFunctionOutputTypeConversion]:
            raise FunctionError("output conversion is not enabled for {0}".format(self.__class__.__name__))

        # Bad outputType specification
        if value and not isinstance(value, FunctionOutputType):
            raise FunctionError("value ({0}) of functionOutputType attribute must be FunctionOutputType for {1}".
                                format(self.functionOutputType, self.__class__.__name__))

        # Can't convert from arrays of length > 1 to number
        # if (self.variable.size  > 1 and (self.functionOutputType is FunctionOutputType.RAW_NUMBER)):
        if (len(self.variable) > 1 and (self.functionOutputType is FunctionOutputType.RAW_NUMBER)):
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

    Return :keyword:`True` or :keyword:`False` according to the manner of the therapist.

    Arguments
    ---------

    variable : boolean or statement that resolves to one : default variableClassDefault
        assertion for which a therapeutic response will be offered.

    propensity : Manner value : default Manner.CONTRARIAN
        specifies preferred therapeutic manner

    pertinacity : float : default 10.0
        specifies therapeutic consistency

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
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
        the `PreferenceSet` for function. Specified in the `prefs` argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

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
    # This is used both to type-cast the variable, and to initialize variableInstanceDefault
    variableClassDefault = 0
    variableClassDeafult_locked = False

    # Mode indicators
    class Manner(Enum):
        OBSEQUIOUS = 0
        CONTRARIAN = 1

    # Param class defaults
    # These are used both to type-cast the params, and as defaults if none are assigned
    #  in the initialization call or later (using either _assign_defaults or during a function call)

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({kwFunctionOutputTypeConversion: True,
                               PARAMETER_STATE_PARAMS: None
                               # PROPENSITY: Manner.CONTRARIAN,
                               # PERTINACITY:  10
                               })

    def __init__(self,
                 variable_default=variableClassDefault,
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
        super().__init__(variable_default=variable_default,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        self.functionOutputType = None

    def _validate_variable(self, variable, context=None):
        """Validates variable and assigns validated values to self.variable

        This overrides the class method, to perform more detailed type checking
        See explanation in class method.
        Note: this method (or the class version) is called only if the parameter_validation attribute is :keyword:`True`

        :param variable: (anything but a dict) - variable to be validated:
        :param context: (str)
        :return none:
        """

        if type(variable) == type(self.variableClassDefault) or \
                (isinstance(variable, numbers.Number) and isinstance(self.variableClassDefault, numbers.Number)):
            self.variable = variable
        else:
            raise FunctionError("Variable must be {0}".format(type(self.variableClassDefault)))

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validates variable and /or params and assigns to targets

        This overrides the class method, to perform more detailed type checking
        See explanation in class method.
        Note: this method (or the class version) is called only if the parameter_validation attribute is :keyword:`True`

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

        variable : boolean : default variableClassDefault
           an assertion to which a therapeutic response is made.

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------

        therapeutic response : boolean

        """
        self._check_args(variable, params, context)

        # Compute the function

        # Use self.variable (rather than variable), as it has been validated (and default assigned, if necessary)
        statement = self.variable
        propensity = self.paramsCurrent[PROPENSITY]
        pertinacity = self.paramsCurrent[PERTINACITY]
        whim = randint(-10, 10)

        if propensity == self.Manner.OBSEQUIOUS:
            return whim < pertinacity

        elif propensity == self.Manner.CONTRARIAN:
            return whim > pertinacity

        else:
            raise FunctionError("This should not happen if parameter_validation == True;  check its value")


# region ****************************************   FUNCTIONS   *********************************************************
# endregion

# region **********************************  USER-DEFINED FUNCTION  *****************************************************
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

    variable : value : default variableClassDefault
        specifies the format and a default value for the input to `function <Function>`.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
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
        they are specified by the Function's :keyword:`variableClassDefault`.

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
        the `PreferenceSet` for function. Specified in the `prefs` argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """
    componentName = USER_DEFINED_FUNCTION
    componentType = USER_DEFINED_FUNCTION_TYPE

    variableClassDefault = [0]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        kwFunctionOutputTypeConversion: False,
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

        super().__init__(variable_default=variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        self.functionOutputType = None

        # IMPLEMENT: PARSE ARGUMENTS FOR user_defined_function AND ASSIGN TO user_params

    def function(self,
                 **kwargs):
        return self.user_defined_function(**kwargs)


# region **********************************  COMBINATION FUNCTIONS  *****************************************************
# endregion


class CombinationFunction(Function_Base):
    componentType = COMBINATION_FUNCTION_TYPE


class Reduce(CombinationFunction):  # ------------------------------------------------------------------------
    # FIX: CONFIRM THAT 1D KWEIGHTS USES EACH ELEMENT TO SCALE CORRESPONDING VECTOR IN VARIABLE
    # FIX  CONFIRM THAT LINEAR TRANSFORMATION (OFFSET, SCALE) APPLY TO THE RESULTING ARRAY
    # FIX: CONFIRM RETURNS LIST IF GIVEN LIST, AND SIMLARLY FOR NP.ARRAY
    """
    Reduce(                                     \
         variable_default=variableClassDefault, \
         operation=SUM,                         \
         params=None,                           \
         owner=None,                            \
         prefs=None,                            \
    )

    .. _Reduce:

    Combine values in each of one or more arrays into a single value for each array.

    COMMENT:
        IMPLEMENTATION NOTE: EXTEND TO MULTIDIMENSIONAL ARRAY ALONG ARBITRARY AXIS
    COMMENT

    Arguments
    ---------

    variable_default : list or np.array : default variableClassDefault
        specifies a template for the value to be transformed and its default value;  all entries must be numeric.

    operation : SUM or PRODUCT : default SUM
        specifies whether to sum or multiply the elements in `variable <Reduce.function.variable>` of
        `function <Reduce.function>`.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
        the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
        defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    variable_default : list or np.array
        contains array(s) to be reduced.

    operation : SUM or PRODUCT
        determines whether elements of each array in `variable <Reduce.function.variable>` of
        `function <Reduce.function>` are summmed or multiplied.

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the `prefs` argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """
    componentName = REDUCE_FUNCTION

    variableClassDefault = [0, 0]
    # variableClassDefault_locked = True

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
                 operation: tc.enum(SUM, PRODUCT) = SUM,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context=componentName + INITIALIZING):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(operation=operation,
                                                  params=params)

        super().__init__(variable_default=variable_default,
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
        super()._validate_variable(variable=variable, context=context)
        if not is_numeric(variable):
            raise FunctionError("All elements of {} must be scalar values".
                                format(self.__class__.__name__))

    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """
        Returns a scalar value for each array in `variable <Reduce.variable>` that is either the sum or
        product of the elements in that array.

        Arguments
        ---------

        variable : list or np.array : default variableClassDefault
           a list or np.array of numeric values.

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------

        Sum or product of arrays in variable : np.array
            in an array that is one dimension less than `variable <Reduce.variable>`.


        """

        # Validate variable and assign to self.variable, and validate params
        self._check_args(variable=variable, params=params, context=context)

        operation = self.paramsCurrent[OPERATION]

        # Calculate using relevant aggregation operation and return
        if (operation is SUM):
            result = np.sum(self.variable)
        elif operation is PRODUCT:
            result = np.product(self.variable)
        else:
            raise FunctionError("Unrecognized operator ({0}) for Reduce function".
                                format(self.paramsCurrent[OPERATION].self.Operation.SUM))
        return result


class LinearCombination(
    CombinationFunction):  # ------------------------------------------------------------------------
    # FIX: CONFIRM THAT 1D KWEIGHTS USES EACH ELEMENT TO SCALE CORRESPONDING VECTOR IN VARIABLE
    # FIX  CONFIRM THAT LINEAR TRANSFORMATION (OFFSET, SCALE) APPLY TO THE RESULTING ARRAY
    # FIX: CONFIRM RETURNS LIST IF GIVEN LIST, AND SIMLARLY FOR NP.ARRAY
    """
    LinearCombination(     \
         variable_default, \
         weights=None,     \
         exponents=None,   \
         scale=1.0,        \
         offset=0.0,       \
         operation=SUM,    \
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

    variable : 1d or 2d np.array : default variableClassDefault
        specifies a template for the arrays to be combined.  If it is 2d, all items must have the same length.

    weights : 1d or 2d np.array
        specifies values used to multiply the elements of each array in `variable  <LinearCombination.variable>`.
        If it is 1d, its length must equal the number of items in `variable <LinearCombination.variable>`;
        if it is 2d, the length of each item must be the same as those in `variable <LinearCombination.variable>`,
        and there must be the same number of items as there are in `variable <LinearCombination.variable>`
        (see `weights <LinearCombination.weights>` for details)

    exponents : 1d or 2d np.array
        specifies values used to exponentiate the elements of each array in `variable  <LinearCombination.variable>`.
        If it is 1d, its length must equal the number of items in `variable <LinearCombination.variable>`;
        if it is 2d, the length of each item must be the same as those in `variable <LinearCombination.variable>`,
        and there must be the same number of items as there are in `variable <LinearCombination.variable>`
        (see `exponents <LinearCombination.exponents>` for details)

    scale : float
        specifies a value by which to multiply each element of the output of `function <LinearCombination.function>`
        (see `scale <LinearCombination.scale>` for details)

    offset : float
        specifies a value to add to each element of the output of `function <LinearCombination.function>`
        (see `offset <LinearCombination.offset>` for details)

    operation : SUM or PRODUCT
        specifies whether the `function <LinearCombination.function>` takes the elementwise (Hadamarad)
        sum or product of the arrays in `variable  <LinearCombination.variable>`.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
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

    scale : float
        value is multiplied by each element of the array after applying the `operation <LinearCombination.operation>`
        (see `scale <LinearCombination.scale>` for details);  this done before applying the
        `offset <LinearCombination.offset>` (if it is specified).

    offset : float
        value is added to each element of the array after applying the `operation <LinearCombination.operation>`
        and `scale <LinearCombination.scale>` (if it is specified).

    operation : SUM or PRODUCT
        determines whether the `function <LinearCombination.function>` takes the elementwise (Hadamard) sum or
        product of the arrays in `variable  <LinearCombination.variable>`.

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
        the `PreferenceSet` for function. Specified in the `prefs` argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = LINEAR_COMBINATION_FUNCTION

    classPreferences = {
        kwPreferenceSetName: 'LinearCombinationCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
        kpRuntimeParamStickyAssignmentPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
    }

    variableClassDefault = [2, 2]
    # variableClassDefault_locked = True

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
                 # IMPLEMENTATION NOTE - these don't check whether every element of np.array is numerical:
                 # weights:tc.optional(tc.any(int, float, tc.list_of(tc.any(int, float)), np.ndarray))=None,
                 # exponents:tc.optional(tc.any(int, float, tc.list_of(tc.any(int, float)), np.ndarray))=None,
                 # MODIFIED 2/10/17 OLD: [CAUSING CRASHING FOR SOME REASON]
                 # # weights:is_numeric_or_none=None,
                 # # exponents:is_numeric_or_none=None,
                 # MODIFIED 2/10/17 NEW:
                 weights=None,
                 exponents=None,
                 scale: parameter_spec = 1.0,
                 offset: parameter_spec = 0.0,
                 # MODIFIED 2/10/17 END
                 operation: tc.enum(SUM, PRODUCT) = SUM,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context=componentName + INITIALIZING):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(scale=scale,
                                                  offset=offset,
                                                  weights=weights,
                                                  exponents=exponents,
                                                  operation=operation,
                                                  params=params)

        super().__init__(variable_default=variable_default,
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
        super()._validate_variable(variable=variable, context=context)
        # FIX: CONVERT TO AT LEAST 1D NP ARRAY IN INIT AND EXECUTE, SO ALWAYS NP ARRAY
        # FIX: THEN TEST THAT SHAPES OF EVERY ELEMENT ALONG AXIS 0 ARE THE SAME
        # FIX; PUT THIS IN DOCUMENTATION
        if isinstance(variable, (list, np.ndarray)):
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

    def _validate_params(self, request_set, target_set=None, context=None):
        """Insure that WEIGHTS and EXPONENTS are lists or np.arrays of numbers with length equal to variable

        Args:
            request_set:
            target_set:
            context:

        Returns:

        """

        # FIX: MAKE SURE THAT IF OPERATION IS SUBTRACT OR DIVIDE, THERE ARE ONLY TWO VECTORS

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if target_set[WEIGHTS] is not None:
            target_set[WEIGHTS] = np.atleast_2d(target_set[WEIGHTS]).reshape(-1, 1)
        if target_set[EXPONENTS] is not None:
            target_set[EXPONENTS] = np.atleast_2d(target_set[EXPONENTS]).reshape(-1, 1)

            # if not operation:
            #     raise FunctionError("Operation param missing")
            # if not operation == self.Operation.SUM and not operation == self.Operation.PRODUCT:
            #     raise FunctionError("Operation param ({0}) must be Operation.SUM or Operation.PRODUCT".format(operation))

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

        variable : 1d or 2d np.array : default variableClassDefault
           a single numeric array, or multiple arrays to be combined; if it is 2d, all arrays must have the same length.

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------

        combined array : 1d np.array
            the result of linearly combining the arrays in `variable <LinearCombination.variable>`.

        """

        # Validate variable and assign to self.variable, and validate params
        self._check_args(variable=variable, params=params, context=context)

        exponents = self.paramsCurrent[EXPONENTS]
        weights = self.paramsCurrent[WEIGHTS]
        operation = self.paramsCurrent[OPERATION]
        offset = self.paramsCurrent[OFFSET]
        scale = self.paramsCurrent[SCALE]

        # IMPLEMENTATION NOTE: CONFIRM: SHOULD NEVER OCCUR, AS _validate_variable NOW ENFORCES 2D np.ndarray
        # If variable is 0D or 1D:
        if np_array_less_than_2d(self.variable):
            return (self.variable * scale) + offset

        # FIX FOR EFFICIENCY: CHANGE THIS AND WEIGHTS TO TRY/EXCEPT // OR IS IT EVEN NECESSARY, GIVEN VALIDATION ABOVE??
        # Apply exponents if they were specified
        if exponents is not None:
            if len(exponents) != len(self.variable):
                raise FunctionError("Number of exponents ({0}) does not equal number of items in variable ({1})".
                                    format(len(exponents), len(self.variable.shape)))
            # Avoid divide by zero warning:
            #    make sure there are no zeros for an element that is assigned a negative exponent
            if INITIALIZING in context and any(not any(i) and j < 0 for i, j in zip(self.variable, exponents)):
                self.variable = np.ones_like(self.variable)
            else:
                self.variable = self.variable ** exponents

        # Apply weights if they were specified
        if weights is not None:
            if len(weights) != len(self.variable):
                raise FunctionError("Number of weights ({0}) is not equal to number of items in variable ({1})".
                                    format(len(weights), len(self.variable.shape)))
            else:
                self.variable = self.variable * weights

        # Calculate using relevant aggregation operation and return
        if (operation is SUM):
            result = sum(self.variable) * scale + offset
        elif operation is PRODUCT:
            result = reduce(mul, self.variable, 1)
        else:
            raise FunctionError("Unrecognized operator ({0}) for LinearCombination function".
                                format(self.paramsCurrent[OPERATION].self.Operation.SUM))
        # FIX: CONFIRM THAT RETURNS LIST IF GIVEN A LIST
        return result


# region ***********************************  TRANSFER FUNCTIONS  ***********************************************
# endregion

class TransferFunction(Function_Base):
    componentType = TRANFER_FUNCTION_TYPE


class Linear(
    TransferFunction):  # --------------------------------------------------------------------------------------
    """
    Linear(                \
         variable_default, \
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

    variable : number or np.array : default variableClassDefault
        specifies a template for the value to be transformed.

    slope : float : default 1.0
        specifies a value by which to multiply `variable <Linear.variable>`.

    intercept : float : default 0.0
        specifies a value to add to each element of `variable <Linear.variable>` after applying `slope <Linear.slope>`.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
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

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the `prefs` argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = LINEAR_FUNCTION

    classPreferences = {
        kwPreferenceSetName: 'LinearClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
        kpRuntimeParamStickyAssignmentPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
    }

    variableClassDefault = [0]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        kwFunctionOutputTypeConversion: True,
        PARAMETER_STATE_PARAMS: None
    })

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
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

        super().__init__(variable_default=variable_default,
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
        """
        Return: `slope <Linear.slope>` * `variable <Linear.variable>` + `intercept <Linear.intercept>`.

        Arguments
        ---------

        variable : number or np.array : default variableClassDefault
           a single value or array to be transformed.

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------

        linear transformation of variable : number or np.array

        """

        self._check_args(variable, params, context)
        slope = self.paramsCurrent[SLOPE]
        intercept = self.paramsCurrent[INTERCEPT]
        outputType = self.functionOutputType

        # By default, result should be returned as np.ndarray with same dimensionality as input
        result = self.variable * slope + intercept

        # region Type conversion (specified by outputType):
        # Convert to 2D array, irrespective of variable type:
        if outputType is FunctionOutputType.NP_2D_ARRAY:
            result = np.atleast2d(result)

        # Convert to 1D array, irrespective of variable type:
        # Note: if 2D array (or higher) has more than two items in the outer dimension, generate exception
        elif outputType is FunctionOutputType.NP_1D_ARRAY:
            # If variable is 2D
            if self.variable.ndim == 2:
                # If there is only one item:
                if len(self.variable) == 1:
                    result = result[0]
                else:
                    raise FunctionError("Can't convert result ({0}: 2D np.ndarray object with more than one array)"
                                        " to 1D array".format(result))
            elif len(self.variable) == 1:
                result = result
            elif len(self.variable) == 0:
                result = np.atleast_1d(result)
            else:
                raise FunctionError("Can't convert result ({0} to 1D array".format(result))

        # Convert to raw number, irrespective of variable type:
        # Note: if 2D or 1D array has more than two items, generate exception
        elif outputType is FunctionOutputType.RAW_NUMBER:
            # If variable is 2D
            if self.variable.ndim == 2:
                # If there is only one item:
                if len(self.variable) == 1 and len(self.variable[0]) == 1:
                    result = result[0][0]
                else:
                    raise FunctionError("Can't convert result ({0}) with more than a single number to a raw number".
                                        format(result))
            elif len(self.variable) == 1:
                if len(self.variable) == 1:
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


class Exponential(
    TransferFunction):  # ---------------------------------------------------------------------------------
    """
    Exponential(           \
         variable_default, \
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

    variable : number or np.array : default variableClassDefault
        specifies a template for the value to be transformed.

    rate : float : default 1.0
        specifies a value by which to multiply `variable <Exponential.variable>` before exponentiation.

    scale : float : default 1.0
        specifies a value by which to multiply the exponentiated value of `variable <Exponential.variable>`.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
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
        value by which the exponentiated value is multipled.

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the `prefs` argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = EXPONENTIAL_FUNCTION

    variableClassDefault = 0

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
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

        super().__init__(variable_default=variable_default,
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

        variable : number or np.array : default variableClassDefault
           a single value or array to be exponentiated.

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------

        exponential transformation of variable : number or np.array

        """

        self._check_args(variable, params, context)

        # Assign the params and return the result
        rate = self.paramsCurrent[RATE]
        scale = self.paramsCurrent[SCALE]

        return scale * np.exp(rate * self.variable)

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


class Logistic(
    TransferFunction):  # ------------------------------------------------------------------------------------
    """
    Logistic(              \
         variable_default, \
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

    variable : number or np.array : default variableClassDefault
        specifies a template for the value to be transformed.

    gain : float : default 1.0
        specifies a value by which to multiply `variable <Linear.variable>` before logistic transformation

    bias : float : default 0.0
        specifies a value to add to each element of `variable <Linear.variable>` after applying `gain <Linear.gain>`
        but before logistic transformation.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
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

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the `prefs` argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = LOGISTIC_FUNCTION
    parameter_keywords.update({GAIN, BIAS})

    variableClassDefault = 0

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
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

        super().__init__(variable_default=variable_default,
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

        variable : number or np.array : default variableClassDefault
           a single value or array to be transformed.

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------

        logistic transformation of variable : number or np.array

        """

        self._check_args(variable, params, context)
        gain = self.paramsCurrent[GAIN]
        bias = self.paramsCurrent[BIAS]

        return 1 / (1 + np.exp(-(gain * self.variable) + bias))

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


class SoftMax(
    TransferFunction):  # -------------------------------------------------------------------------------------
    """
    SoftMax(               \
         variable_default, \
         gain=1.0,         \
         output=ALL,       \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _SoftMax:

    SoftMax transform of variable.

    Arguments
    ---------

    variable : 1d np.array : default variableClassDefault
        specifies a template for the value to be transformed.

    gain : float : default 1.0
        specifies a value by which to multiply `variable <Linear.variable>` before softmax transformation.

    output : ALL, MAX_VAL, MAX_INDICATOR, or PROB : default ALL
        specifies the format of array returned by `function <SoftMax.function>`
        (see `output <SoftMax.output>` for details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
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
        value by which `variable <Logistic.variable>` is multiplied before the softmax transformation;  determines
        the "sharpness" of the distribution.

    output : ALL, MAX_VAL, MAX_INDICATOR, or PROB
        determines how the softmax-transformed values of the elements in `variable <SoftMax.variable>` are reported
        in the array returned by `function <SoftMax.funtion>`:
            * **ALL**: array of all softmax-transformed values;
            * **MAX_VAL**: softmax-transformed value for the element with the maximum such value, 0 for all others;
            * **MAX_INDICATOR**: 1 for the element with the maximum softmax-transformed value, 0 for all others;
            * **PROB**: probabilistically chosen element based on softmax-transformed values after normalizing sum of
              values to 1, 0 for all others.

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the `prefs` argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = SOFTMAX_FUNCTION

    variableClassDefault = 0

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
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

        super().__init__(variable_default=variable_default,
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

        variable : 1d np.array : default variableClassDefault
           an array to be transformed.

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------

        softmax transformation of variable : number or np.array

        """

        self._check_args(variable, params, context)

        # Assign the params and return the result
        output = self.params[OUTPUT_TYPE]
        gain = self.params[GAIN]

        # print('\ninput: {}'.format(self.variable))

        # Get numerator
        sm = np.exp(gain * self.variable)

        # Normalize
        sm = sm / np.sum(sm, axis=0)

        # For the element that is max of softmax, set it's value to its softmax value, set others to zero
        if output is MAX_VAL:
            max_value = np.max(sm)
            sm = np.where(sm == max_value, max_value, 0)

        # For the element that is max of softmax, set its value to 1, set others to zero
        elif output is MAX_INDICATOR:
            # sm = np.where(sm == np.max(sm), 1, 0)
            max_value = np.max(sm)
            sm = np.where(sm == max_value, 1, 0)

        # Choose a single element probabilistically based on softmax of their values;
        #    leave that element's value intact, set others to zero
        elif output is PROB:
            cum_sum = np.cumsum(sm)
            random_value = np.random.uniform()
            chosen_item = next(element for element in cum_sum if element > random_value)
            chosen_in_cum_sum = np.where(cum_sum == chosen_item, 1, 0)
            sm = self.variable * chosen_in_cum_sum

        return sm

    def derivative(self, output, input=None):
        """
        derivative(output)

        Derivative of `function <SoftMax.function>`.

        Returns
        -------

        derivative :  number
            output - maximum value.

        """
        # FIX: ??CORRECT:
        indicator = self.function(input, params={MAX_VAL: True})
        return output - indicator


class LinearMatrix(TransferFunction):  # -------------------------------------------------------------------------------
    """
    LinearMatrix(          \
         variable_default, \
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
        MATRIX FORMAT
                                         INDICES:
                                     Output elements:
                              0       1       2       3
                         0  [0,0]   [0,1]   [0,2]   [0,3]
        Input elements:  1  [1,0]   [1,1]   [1,2]   [1,3]
                         2  [2,0]   [2,1]   [2,2]   [2,3]

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

    variable : list or 1d np.array : default variableClassDefault
        specifies a template for the value to be transformed; length must equal the number of rows of `matrix
        <LinearMatrix.matrix>`.

    matrix : number, list, 1d or 2d np.ndarray, np.matrix, function, or matrix keyword : default IDENTITY_MATRIX
        specifies matrix used to transform `variable <LinearMatrix.variable>`
        (see `matrix <LinearMatrix.matrix>` for specification details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
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
        the `PreferenceSet` for function. Specified in the `prefs` argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = LINEAR_MATRIX_FUNCTION

    DEFAULT_FILLER_VALUE = 0

    variableClassDefault = [DEFAULT_FILLER_VALUE]  # Sender vector

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    def matrix_spec(m):
        if m is None:
            return True
        if m in MATRIX_KEYWORD_VALUES:
            return True
        if isinstance(m, (list, np.ndarray, np.matrix, function_type)):
            return True
        return False

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
                 matrix: matrix_spec = None,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context=componentName + INITIALIZING):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(matrix=matrix,
                                                  params=params)

        # Note: this calls _validate_variable and _validate_params which are overridden below;
        #       the latter implements the matrix if required
        # super(LinearMatrix, self).__init__(variable_default=variable_default,
        super().__init__(variable_default=variable_default,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        self.matrix = self.instantiate_matrix(self.paramsCurrent[MATRIX])

    def _validate_variable(self, variable, context=None):
        """Insure that variable passed to LinearMatrix is a 1D np.array

        :param variable: (1D np.array)
        :param context:
        :return:
        """
        super()._validate_variable(variable, context)

        # Check that self.variable == 1D
        try:
            is_not_1D = not self.variable.ndim is 1

        except AttributeError:
            raise FunctionError("PROGRAM ERROR: variable ({0}) for {1} should be an np.ndarray".
                                format(self.variable, self.__class__.__name__))
        else:
            if is_not_1D:
                raise FunctionError("variable ({0}) for {1} must be a 1D np.ndarray".
                                    format(self.variable, self.__class__.__name__))

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate params and assign to targets

        This overrides the class method, to perform more detailed type checking (see explanation in class method).
        Note: this method (or the class version) is called only if the parameter_validation attribute is :keyword:`True`

        :param request_set: (dict) - params to be validated
        :param target_set: (dict) - destination of validated params
        :param context: (str)
        :return none:
        """

        super()._validate_params(request_set, target_set, context)
        param_set = target_set
        sender = self.variable
        # Note: this assumes self.variable is a 1D np.array, as enforced by _validate_variable
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
        if kwReceiver in param_set:
            self.receiver = param_set[kwReceiver]
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
            self.receiver = param_set[kwReceiver] = sender
        # # MODIFIED 3/26/17 NEW:
        # self.receiver = param_set[kwReceiver] = sender
        # MODIFIED 3/26/17 END

        receiver_len = len(self.receiver)

        # Check rest of params
        message = ""
        for param_name, param_value in param_set.items():

            # Receiver param already checked above
            if param_name is kwReceiver:
                continue

            # Not currently used here
            if param_name is kwFunctionOutputTypeConversion:
                continue

            # Matrix specification param
            elif param_name == MATRIX:

                # A number (to be used as a filler), so OK
                if isinstance(param_value, numbers.Number):
                    continue

                # np.matrix or np.ndarray provided, so validate that it is numeric and check dimensions
                elif isinstance(param_value, (np.ndarray, np.matrix)):
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
                elif param_value is IDENTITY_MATRIX:
                    # Receiver length doesn't equal sender length
                    if not (self.receiver.shape == sender.shape and self.receiver.size == sender.size):
                        # if self.owner.prefs.verbosePref:
                        #     print ("Identity matrix requested, but length of receiver ({0})"
                        #            " does not match length of sender ({1});  sender length will be used".
                        #            format(receiver_len, sender_len))
                        # # Set receiver to sender
                        # param_set[kwReceiver] = sender
                        raise FunctionError("Identity matrix requested for the {} function of {}, "
                                            "but length of receiver ({}) does not match length of sender ({})".
                                            format(self.name, self.owner.name, receiver_len, sender_len))
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
                message += "Unrecognized param ({}) specified for the {} function of {}".format(param_name,
                                                                                                self.name,
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

        sender = self.variable
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
                                format(specification, self.name, self.owner.name, matrix_keywords))
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
            a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
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
        self._check_args(variable, params, context=context)

        return np.dot(self.variable, self.matrix)

    def keyword(self, keyword):

        from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
        rows = None
        cols = None
        if isinstance(self, MappingProjection):
            rows = len(self.sender.value)
            cols = len(self.receiver.variable)
        matrix = get_matrix(keyword, rows, cols)

        if matrix is None:
            raise FunctionError("Unrecognized keyword ({}) specified for the {} function of {}".
                                format(keyword, self.name, self.owner.name))
        else:
            return matrix

    def param_function(owner, function):
        sender_len = len(owner.sender.value)
        receiver_len = len(owner.receiver.variable)
        return function(sender_len, receiver_len)


def get_matrix(specification, rows=1, cols=1, context=None):
    """Returns matrix conforming to specification with dimensions = rows x cols or None

     Specification can be a matrix keyword, filler value or np.ndarray

     Specification (validated in _validate_params):
        + single number (used to fill self.matrix)
        + matrix keyword:
            + AUTO_ASSIGN_MATRIX: IDENTITY_MATRIX if it is square, othwerwise FULL_CONNECTIVITY_MATRIX
            + IDENTITY_MATRIX: 1's on diagonal, 0's elsewhere (must be square matrix), otherwise generates error
            + FULL_CONNECTIVITY_MATRIX: all 1's
            + RANDOM_CONNECTIVITY_MATRIX (random floats uniformly distributed between 0 and 1)
        + 2D list or np.ndarray of numbers

     Returns 2D np.array with length=rows in dim 0 and length=cols in dim 1, or none if specification is not recognized
    """

    # Matrix provided (and validated in _validate_params); convert to np.array
    if isinstance(specification, np.matrix):
        return np.array(specification)

    if isinstance(specification, np.ndarray):
        if specification.ndim == 2:
            return specification
        # FIX: MAKE THIS AN np.array WITH THE SAME DIMENSIONS??
        elif specification.ndim < 2:
            return np.atleast_2d(specification)
        else:
            raise FunctionError("Specification of np.array for matrix ({}) is more than 2d".
                                format(specification))

    if specification is AUTO_ASSIGN_MATRIX:
        if rows == cols:
            specification = IDENTITY_MATRIX
        else:
            specification = FULL_CONNECTIVITY_MATRIX

    if specification == FULL_CONNECTIVITY_MATRIX:
        return np.full((rows, cols), 1.0)

    if specification == IDENTITY_MATRIX:
        if rows != cols:
            raise FunctionError("Sender length ({0}) must equal receiver length ({1}) to use identity matrix".
                                format(rows, cols))
        return np.identity(rows)

    if specification is RANDOM_CONNECTIVITY_MATRIX:
        return np.random.rand(rows, cols)

    # Function is specified, so assume it uses random.rand() and call with sender_len and receiver_len
    if isinstance(specification, function_type):
        return specification(rows, cols)

    # Specification not recognized
    return None


# region ***********************************  INTEGRATOR FUNCTIONS ******************************************************

#  Integrator
#  DDM_BogaczEtAl
#  DDM_NavarroAndFuss

class IntegratorFunction(Function_Base):
    componentType = INTEGRATOR_FUNCTION_TYPE


# FIX: IF RATE HAS TO BE BETWEEN 0 AND 1, VALIDATE_VARIABLE ACCORDINGLY
# SEARCH & REPLACE: old_value -> previous_value

# • why does integrator return a 2d array?
# • does rate have to be between 0 and 1 (if so, validate_variable)
# • does rate = 0 and rate = 1 have the same meaning for all integration_types?
# • should we change "integration_type" to "type"??
# • are rate and noise converted to 1d np.array?  If not, correct docstring
# • can noise and initializer be an array?  If so, validated in validate_param?
# • time_step_size?? (vs rate??)
# • can noise be a function now?


class Integrator(
    IntegratorFunction):  # --------------------------------------------------------------------------------
    """
    Integrator(                 \
        variable_default=None,  \
        rate=1.0,               \
        integration_type=CONSTANT,     \
        noise=0.0,              \
        time_step_size=1.0,     \
        params=None,            \
        owner=None,             \
        prefs=None,             \
        )

    .. _Integrator:

    Integrate current value of `variable <Integrator.variable>` with its prior value.

    Arguments
    ---------

    variable_default : number, list or np.array : default variableClassDefault
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float, list or 1d np.array : default 1.0
        specifies the rate of integration.  If it is a list or array, it must be the same length as
        `variable <Integrator.variable_default>` and all elements must be floats between 0 and 1
        (see `rate <Integrator.rate>` for details).

    integration_type : CONSTANT, SIMPLE, ADAPTIVE, DIFFUSION : default CONSTANT
        specifies type of integration (see `integration_type <Integrator.integration_type>` for details).

    noise : float, list or 1d np.array : default 0.0
        specifies random value to be added in each call to `function <Integrator.function>`.
        If it is a list or array, it must be the same length as `variable <Integrator.variable_default>` and all
        elements must be floats between 0 and 1.

    time_step_size : float : default 0.0
        determines the timing precision of the integration process when `integration_type <Integrator.integration_type>` is set to
        DIFFUSION (see `time_step_size <Integrator.time_step_size>` for details.

    initializer float, list or 1d np.array : default 0.0
        specifies starting value for integration.  If it is a list or array, it must be the same length as
        `variable_default <Integrator.variable_default>` (see `initializer <Integrator.initializer>` for details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
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

    rate : 1d np.array
        determines the rate of integration based on current and prior values.  All elements are between 0 and 1
        (0 = no change; 1 = instantaneous change). If it has a single element, it applies to all elements of
        `variable <Integrator.variable>`;  if it has more than one element, each element applies to the
        corresponding element of `variable <Integrator.variable>`.

    integration_type : CONSTANT, SIMPLE, ADAPTIVE, DIFFUSION
        specifies type of integration:
            * **CONSTANT**: `old_value <Integrator.old_value>` + `rate <Integrator.rate>` + `noise <Integrator.noise>`
              (ignores `variable <Integrator.variable>`);
            * **SIMPLE**: `old_value <Integrator.old_value>` + `rate <Integrator.rate>` *
              `variable <variable.Integrator.variable>` + `noise <Integrator.noise>`;
            * **ADAPTIVE**: (1-`rate <Integrator.rate>`) * `variable <Integrator.variable>` +
              (`rate <Integrator.rate>` * `old_value <Integrator.old_value>`) + `noise <Integrator.noise>`
              (`Weiner filter <https://en.wikipedia.org/wiki/Wiener_filter>`_ or
              `Delta rule <https://en.wikipedia.org/wiki/Delta_rule>`_);
            * **DIFFUSION**: `old_value <Integrator.old_value>` +
              (`rate <Integrator.rate>` * `old_value` * `time_step_size <Integrator.time_step_size>`) +
              √(`time_step_size <Integrator.time_step_size>` * `noise <Integrator.noise>` * Gaussian(0,1))
              (`Drift Diffusion Model
              <https://en.wikipedia.org/wiki/Two-alternative_forced_choice#Drift-diffusion_model>`_).

    noise : float or 1d np.array
        specifies random value to be added in each call to `function <Integrator.function>`.

    time_step_size : float
        determines the timing precision of the integration process when `integration_type <Integrator.integration_type>` is set to
        DIFFUSION (and used to scale the `noise <Integrator.noise>` parameter appropriately).

    initializer : float or 1d np.array
        determines the starting value for integration (i.e., the value to which `old_value <Integrator.old_value>`
        is set.  If it is assigned as a `runtime_param <LINK>` it resets `old_value <Integrator.old_value>` to the
        specified value (see `initializer <Integrator.initializer>` for details).

    old_value : 1d np.array : default variableClassDefault
        stores previous value with which `variable <Integrator.variable>` is integrated.

    owner : Mechanism
        `component <Component>` to which the Function has been assigned.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the `prefs` argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = INTEGRATOR_FUNCTION

    variableClassDefault = [[0]]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    # paramClassDefaults.update({INITIALIZER: variableClassDefault})
    paramClassDefaults.update({})

    @tc.typecheck
    def __init__(self,
                 variable_default=None,
                 rate: parameter_spec = 1.0,
                 integration_type: tc.enum(CONSTANT, SIMPLE, ADAPTIVE, DIFFUSION) = CONSTANT,
                 noise=0.0,
                 time_step_size=1.0,
                 initializer=variableClassDefault,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context="Integrator Init"):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  integration_type=integration_type,
                                                  time_step_size=time_step_size,
                                                  initializer=initializer,
                                                  params=params,
                                                  noise=noise,
                                                  )

        # Assign here as default, for use in initialization of function
        self.old_value = self.paramClassDefaults[INITIALIZER]

        super().__init__(variable_default=variable_default,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        # Reassign to kWInitializer in case default value was overridden
        self.old_value = self.initializer


        # self.noise = self.paramsCurrent[NOISE]

    def _validate_params(self, request_set, target_set=None, context=None):

        # Handle list or array for rate specification
        rate = request_set[RATE]
        if isinstance(rate, (list, np.ndarray)):
            if len(rate) != np.array(self.variable).size:
                # If the variable was not specified, then reformat it to match rate specification
                #    and assign variableClassDefault accordingly
                # Note: this situation can arise when the rate is parameterized (e.g., as an array)
                #       in the Integrator's constructor, where that is used as a specification for a function parameter
                #       (e.g., for an IntegratorMechanism), whereas the input is specified as part of the
                #       object to which the function parameter belongs (e.g., the IntegratorMechanism);
                #       in that case, the Integrator gets instantiated using its variableClassDefault ([[0]]) before
                #       the object itself, thus does not see the array specification for the input.
                if self._variable_not_specified:
                    self._assign_defaults(variable=np.zeros_like(np.array(rate)), context=context)
                    if self.verbosePref:
                        warnings.warn("The length ({}) of the array specified for the rate parameter ({}) of {} must "
                                      "matach the length ({}) of the default input ({});  the default input has been "
                                      "updated to match".
                                      format(len(rate), rate, self.name, np.array(self.variable).size), self.variable)
                else:
                    raise FunctionError("The length ({}) of the array specified for the rate parameter ({}) of {} "
                                        "must match the length ({}) of the default input ({})".
                                        format(len(rate), rate, self.name, np.array(self.variable).size, self.variable))

            self.paramClassDefaults[RATE] = np.zeros_like(np.array(rate))

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        noise = target_set[NOISE]

        time_step_size = target_set[TIME_STEP_SIZE]

        # Validate NOISE:
        # If the noise is a float, continue; if it is function, set self.noise_function to True
        # (flags noise to be executed before passing it to integrator )
        # Otherwise, error
        if isinstance(noise, float):
            self.noise_function = False
        elif callable(noise):
            self.noise_function = True
        else:
            raise FunctionError("noise parameter ({}) for {} must be a float or a function".
                                format(noise, self.name))

        # Make sure initializer is compatible with variable
        try:
            if not iscompatible(target_set[INITIALIZER], self.variableClassDefault):
                raise FunctionError("INITIALIZER param {0} for {1} must be same type as variable {2}".
                                    format(target_set[INITIALIZER],
                                           self.__class__.__name__,
                                           self.variable))
        except KeyError:
            pass

    def function(self,
                 variable=None,
                 params=None,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """
        Return: some fraction of `variable <Linear.slope>` combined with some fraction of `old_value
        <Integrator.old_value>` (see `integration_type <Integrator.integration_type>`).

        Arguments
        ---------

        variable : number, list or np.array : default variableClassDefault
           a single value or array of values to be integrated.

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------

        updated value of integral : 2d np.array

        """

        # FIX:  CONVERT TO NP?
        # FIX:  NEED TO CONVERT OLD_VALUE TO NP ARRAY

        self._check_args(variable=variable, params=params, context=context)

        rate = np.array(self.paramsCurrent[RATE]).astype(float)
        integration_type = self.paramsCurrent[INTEGRATION_TYPE]

        time_step_size = self.paramsCurrent[TIME_STEP_SIZE]

        # if noise is a function, execute it
        if self.noise_function:
            noise = self.noise()
        else:
            noise = self.noise

        try:
            old_value = params[INITIALIZER]
        except (TypeError, KeyError):
            old_value = self.old_value

        old_value = np.atleast_2d(old_value)
        new_value = self.variable

        # Compute function based on integration_type param
        if integration_type is CONSTANT:
            value = old_value + rate + noise
        elif integration_type is SIMPLE:
            value = old_value + (new_value * rate) + noise
        elif integration_type is ADAPTIVE:
            value = (1 - rate) * old_value + rate * new_value + noise
        elif integration_type is DIFFUSION:
            value = old_value + rate * old_value * time_step_size + np.sqrt(time_step_size * noise) * np.random.normal()
        else:
            value = new_value

        # If this NOT an initialization run, update the old value
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if not context or not INITIALIZING in context:
            self.old_value = value

        return value

# Trying commenting out DDMIntegrator entirely and forcing the integration_type to be DIFFUSION in the DDM *mechanism*
# # FIX: SHOULD THIS EVEN ALLOW A INTEGRATION_TYPE PARAM IF IT REQUIRES THAT IT BE DIFFUSION??
# class DDMIntegrator(
#     Integrator):  # -------------------------------------------------------------------------------------
#     """
#     DDMIntegrator(                 \
#         variable_default=None,  \
#         rate=1.0,               \
#         noise=0.5,              \
#         time_step_size=1.0,     \
#         params=None,            \
#         owner=None,             \
#         prefs:is_pref_set=None, \
#         )
#
#     .. _DDMIntegrator:
#
#     Implement drift diffusion integration process.
#     It is a subclass of the `Integrator` Function that enforce use of the DIFFUSION `integration_type <Integrator.integration_type>`.
#
#     Arguments
#     ---------
#
#     variable_default : number, list or np.array : default variableClassDefault
#         specifies a template for the value to be integrated;  if it is list or array, each element is independently
#         integrated.
#
#     rate : float, list or 1d np.array : default 1.0
#         specifies the rate of integration (drift component of a drift diffusion process).  If it is a list or array,
#         it must be the same length as `variable <DDMIntegrator.variable_default>` and all elements must be floats
#         between 0 and 1 (see `rate <Integrator.rate>` for details).
#
#     noise : float, list or 1d np.array : default 0.0
#         specifies random value to be added in each call to `function <Integrator.function>` (corresponds to the
#         diffusion component of the drift diffusion process). If it is a list or array, it must be the same length as
#         `variable <DDMIntegrator.variable>` and all elements must be floats between 0 and 1
#         (see `noise <DDMIntegrator.rate>` for details).
#
#     time_step_size : float : default 0.0
#         determines the timing precision of the integration process when `integration_type <Integrator.integration_type>` is set to
#         DIFFUSION (see `time_step_size <Integrator.time_step_size>` for details.
#
#     initializer float, list or 1d np.array : default 0.0
#         specifies starting value for integration.  If it is a list or array, it must be the same length as
#         `variable_default <Integrator.variable_default>` (see `initializer <DDMIntegrator.initializer>` for details).
#
#     params : Optional[Dict[param keyword, param value]]
#         a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
#         function.  Values specified for parameters in the dictionary override any assigned to those parameters in
#         arguments of the constructor.
#
#     owner : Component
#         `component <Component>` to which to assign the Function.
#
#     prefs : Optional[PreferenceSet or specification dict : Function.classPreferences]
#         the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
#         defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).
#
#
#     Attributes
#     ----------
#
#     variable : number or np.array
#         current input value some portion of which (determined by `rate <DDMIntegrator.rate>` that will be added to
#         prior value.  If it is an array, each element is independently integrated.
#
#     rate : 1d np.array
#         determines the rate of integration based on current and prior values (corresponds to the drift component
#         of the drift diffusion process).  All elements are between 0 and 1 (0 = no change; 1 = instantaneous change).
#         If it has a single element, it applies to all elements of `variable <Integrator.variable>`;  if it has more
#         than one element, each element applies to the corresponding element of `variable <Integrator.variable>`.
#
#     noise : float or 1d np.array
#         determines the random value to be added in each call to `function <Integrator.function>` (corresponds to the
#         diffusion component of the drift diffusion process).
#
#     time_step_size : float
#         determines the timing precision of the integration process when `integration_type <Integrator.integration_type>` is set to
#         DIFFUSION (and used to scale the `noise <Integrator.noise>` parameter appropriately).
#
#     initializer : float or 1d np.array
#         determines the starting value for integration (i.e., the value to which `old_value <Integrator.old_value>`
#         is set.  If it is assigned as a `runtime_param <LINK>` it resets `old_value <Integrator.old_value>` to the
#         specified value (see `initializer <Integrator.initializer>` for details).
#
#     old_value : 1d np.array : default variableClassDefault
#         stores previous value with which `variable <Integrator.variable>` is integrated.
#
#     owner : Mechanism
#         `component <Component>` to which the Function has been assigned.
#
#     prefs : PreferenceSet or specification dict : Projection.classPreferences
#         the `PreferenceSet` for function. Specified in the `prefs` argument of the constructor for the function;
#         if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
#         (see :doc:`PreferenceSet <LINK>` for details).
#
#     """
#
#     componentName = DDM_INTEGRATOR_FUNCTION
#
#     @tc.typecheck
#     def __init__(self,
#                  variable_default=None,
#                  rate=1.0,
#                  noise=0.5,
#                  time_step_size=1.0,
#                  params: tc.optional(dict) = None,
#                  owner=None,
#                  prefs: is_pref_set = None,
#                  context="DDMIntegrator Init"):
#
#         # Assign here as default, for use in initialization of function
#         self.old_value = self.paramClassDefaults[INITIALIZER]
#         integration_type = DIFFUSION
#
#         # Assign args to params and functionParams dicts (kwConstants must == arg names)
#         params = self._assign_args_to_param_dicts(
#             integration_type=integration_type,
#             params=params,
#             noise=noise,
#             rate=rate,
#             time_step_size=time_step_size)
#
#         super().__init__(variable_default=variable_default,
#                          params=params,
#                          owner=owner,
#                          prefs=prefs,
#                          context=context)
#
#         # Reassign to kWInitializer in case default value was overridden
#         self.old_value = [self.paramsCurrent[INITIALIZER]]
#
#     def _validate_params(self, request_set, target_set=None, context=None):
#
#         super()._validate_params(request_set=request_set,
#                                  target_set=target_set,
#                                  context=context)
#
#         noise = target_set[NOISE]
#
#         if (isinstance(noise, float) == False):
#             raise FunctionError("noise parameter ({}) for {} must be a float.".
#                                 format(noise, self.name))
#
#         integration_type = target_set[INTEGRATION_TYPE]
#         if (integration_type != "diffusion"):
#             raise FunctionError("integration_type parameter ({}) for {} must be diffusion. "
#                                 "For alternate methods of accumulation, use the Integrator function".
#                                 format(integration_type, self.name))


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
        variable_default=variableCLassDefault,  \
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

    variable_default : number, list or np.array : default variableClassDefault
        specifies a template for decision variable(s);  if it is list or array, a separate solution is computed
        independently for each element.

    drift_rate : float, list or 1d np.array : default 1.0
        specifies the drift_rate of the drift diffusion process.  If it is a list or array,
        it must be the same length as `variable_default <BogaczEtAl.variable_default>`.

    threshold : float, list or 1d np.array : default 1.0
        specifies the threshold (boundary) of the drift diffusion process.  If it is a list or array,
        it must be the same length as `variable_default <BogaczEtAl.variable_default>`.

    starting_point : float, list or 1d np.array : default 1.0
        specifies the initial value of the decision variable for the drift diffusion process.  If it is a list or
        array, it must be the same length as `variable_default <BogaczEtAl.variable_default>`.

    noise : float, list or 1d np.array : default 0.0
        specifies the noise term (corresponding to the diffusion component) of the drift diffusion process.
        If it is a float, it must be a number from 0 to 1.  If it is a list or array, it must be the same length as
        `variable_default <BogaczEtAl.variable_default>` and all elements must be floats from 0 to 1.

    t0 : float, list or 1d np.array : default 0.2
        specifies the non-decision time for solution. If it is a float, it must be a number from 0 to 1.  If it is a
        list or array, it must be the same length as  `variable_default <BogaczEtAl.variable_default>` and all
        elements must be floats from 0 to 1.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
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
        current value of the decision variable.  If it is an array, each element is integrated independently.

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
        the `PreferenceSet` for function. Specified in the `prefs` argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = kwBogaczEtAl

    variableClassDefault = [[0]]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
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

        super().__init__(variable_default=variable_default,
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
        Return: terminal value of decision variable, mean accuracy (error rate; ER) and mean response time (RT)

        Arguments
        ---------

        variable : ignored
            uses value of `drift_rate`, `threshold`, `starting_point`, `noise` and `t0` parameters for calculation.

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------
        Decision variable, mean ER, mean RT : (float, float, float)

        """

        self._check_args(variable=variable, params=params, context=context)

        drift_rate = float(self.drift_rate)
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
            if abs(y0tilde) > threshold:    y0tilde = -1 * (is_neg_drift == 1) * threshold + (
                                                                                             is_neg_drift == 0) * threshold
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


# Results from Navarro and Fuss DDM solution (indices for return value tuple)
class NF_Results(AutoNumber):
    RESULT = ()
    MEAN_ER = ()
    MEAN_DT = ()
    PLACEMARKER = ()
    MEAN_CORRECT_RT = ()
    MEAN_CORRECT_VARIANCE = ()
    MEAN_CORRECT_SKEW_RT = ()


class NavarroAndFuss(
    IntegratorFunction):  # ----------------------------------------------------------------------------
    """
    NavarroAndFuss(                             \
        variable_default=variableCLassDefault,  \
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

    variable_default : number, list or np.array : default variableClassDefault
        specifies a template for decision variable(s);  if it is list or array, a separate solution is computed
        independently for each element.

    drift_rate : float, list or 1d np.array : default 1.0
        specifies the drift_rate of the drift diffusion process.  If it is a list or array,
        it must be the same length as `variable_default <BogaczEtAl.variable_default>`.

    threshold : float, list or 1d np.array : default 1.0
        specifies the threshold (boundary) of the drift diffusion process.  If it is a list or array,
        it must be the same length as `variable_default <BogaczEtAl.variable_default>`.

    starting_point : float, list or 1d np.array : default 1.0
        specifies the initial value of the decision variable for the drift diffusion process.  If it is a list or
        array, it must be the same length as `variable_default <BogaczEtAl.variable_default>`.

    noise : float, list or 1d np.array : default 0.0
        specifies the noise term (corresponding to the diffusion component) of the drift diffusion process.
        If it is a float, it must be a number from 0 to 1.  If it is a list or array, it must be the same length as
        `variable_default <BogaczEtAl.variable_default>` and all elements must be floats from 0 to 1.

    t0 : float, list or 1d np.array : default 0.2
        specifies the non-decision time for solution. If it is a float, it must be a number from 0 to 1.  If it is a
        list or array, it must be the same length as  `variable_default <BogaczEtAl.variable_default>` and all
        elements must be floats from 0 to 1.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
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
        current value of the decision variable.  If it is an array, each element is integrated independently.

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
        the `PreferenceSet` for function. Specified in the `prefs` argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = kwNavarrosAndFuss

    variableClassDefault = [[0]]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
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

        super().__init__(variable_default=variable_default,
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

        variable : ignored
            uses value of `drift_rate`, `threshold`, `starting_point`, `noise` and `t0` parameters for calculation.

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
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


# region ************************************   DISTRIBUTION FUNCTIONS   ************************************************

class DistributionFunction(Function_Base):
    componentType = DIST_FUNCTION_TYPE


class NormalDist(DistributionFunction):
    """Return a random sample from a normal distribution.

    Description:
        Draws samples from a normal distribution of the specified mean and variance using numpy.random.normal

    Initialization arguments:
        - mean (float)
        - standard_dev (float)

    """
    componentName = NORMAL_DIST_FUNCTION

    variableClassDefault = [0]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
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

        super().__init__(variable_default=variable_default,
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
        # Validate variable and assign to self.variable, and validate params
        self._check_args(variable=variable, params=params, context=context)

        mean = self.paramsCurrent[MEAN]
        standard_dev = self.paramsCurrent[STANDARD_DEV]

        result = standard_dev * np.random.normal() + mean

        return result


class ExponentialDist(DistributionFunction):
    """Return a random sample from an exponential distribution.

    Description:
        Draws samples from an exponential distribution of the specified beta using numpy.random.exponential

    Initialization arguments:
        - beta (float)

    """
    componentName = EXPONENTIAL_DIST_FUNCTION

    variableClassDefault = [0]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
                 beta=1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context=componentName + INITIALIZING):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(beta=beta,
                                                  params=params)

        super().__init__(variable_default=variable_default,
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
        # Validate variable and assign to self.variable, and validate params
        self._check_args(variable=variable, params=params, context=context)

        beta = self.paramsCurrent[BETA]

        result = np.random.exponential(beta)

        return result


class UniformDist(DistributionFunction):
    """Return a random sample from a uniform distribution.

    Description:
        Draws samples from a uniform distribution of the specified low and high values using numpy.random.uniform

    Initialization arguments:
        - low (float)
        - high (float)

    """
    componentName = UNIFORM_DIST_FUNCTION

    variableClassDefault = [0]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
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

        super().__init__(variable_default=variable_default,
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
        # Validate variable and assign to self.variable, and validate params
        self._check_args(variable=variable, params=params, context=context)

        low = self.paramsCurrent[LOW]
        high = self.paramsCurrent[HIGH]

        result = np.random.uniform(low, high)

        return result


class GammaDist(DistributionFunction):
    """Return a random sample from a gamma distribution.

    Description:
        Draws samples from a gamma distribution of the specified mean and variance using numpy.random.gamma

    Initialization arguments:
        - scale (float)
        - shape (float)

    """
    componentName = GAMMA_DIST_FUNCTION

    variableClassDefault = [0]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
                 scale=1.0,
                 shape=1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context=componentName + INITIALIZING):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(scale=scale,
                                                  shape=shape,
                                                  params=params)

        super().__init__(variable_default=variable_default,
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
        # Validate variable and assign to self.variable, and validate params
        self._check_args(variable=variable, params=params, context=context)

        scale = self.paramsCurrent[SCALE]
        shape = self.paramsCurrent[SHAPE]

        result = np.random.gamma(shape, scale)

        return result


class WaldDist(DistributionFunction):
    """Return a random sample from a wald distribution.

    Description:
        Draws samples from a wald distribution of the specified mean and variance using numpy.random.wald

    Initialization arguments:
        - mean (float)
        - scale (float)

    """
    componentName = GAMMA_DIST_FUNCTION

    variableClassDefault = [0]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
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

        super().__init__(variable_default=variable_default,
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
        # Validate variable and assign to self.variable, and validate params
        self._check_args(variable=variable, params=params, context=context)

        scale = self.paramsCurrent[SCALE]
        mean = self.paramsCurrent[MEAN]

        result = np.random.wald(mean, scale)

        return result


# endregion

# region **************************************   LEARNING FUNCTIONS ****************************************************


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


LEARNING_ACTIVATION_FUNCTION = 'activation_function'
LEARNING_ACTIVATION_INPUT = 0  # a(j)
# MATRIX = 1             # w
LEARNING_ACTIVATION_OUTPUT = 1  # a(i)
LEARNING_ERROR_OUTPUT = 2


class Reinforcement(
    LearningFunction):  # -------------------------------------------------------------------------------
    """
    Reinforcement(                                       \
        variable_default=variableClassDefault,           \
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
          return     =  LEARNING_RATE  *  self.variable

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

    variable : List or 2d np.array [length 3] : default variableClassDefault
       template for the three items provided as the variable in the call to the `function <Reinforcement.function>`
       (in order):
       `activation_input <Reinforcement.activation_input>` (1d np.array),
       `activation_output <Reinforcement.activation_output>` (1d np.array),
       `error_signal <Reinforcement.error_signal>` (1d np.array).

    activation_function : Function or function : SoftMax
        specifies the function of the mechanism that generates `activation_output <Reinforcement.activation_output>`.

    learning_rate : float : default default_learning_rate
        supercedes any specification for the `process <Process>` and/or `system <System>` to which the function's
        `owner <Function.owner>` belongs (see `learning_rate <Reinforcement.learning_rate>` for details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
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
        the function of the mechanism that generates `activation_output <Reinforcement.activation_output>`; must
        return and array with a single non-zero value.

    learning_rate : float
        the learning rate used by the function.  If specified, it supercedes any learning_rate specified for the
        `process <Process.learning_Rate>` and/or `system <System.learning_rate>` to which the function's  `owner
        <Reinforcement.owner>` belongs.  If it is `None`, then the learning_rate specified for the process to
        which the `owner <Reinforcement.owner>` belongs is used;  and, if that is `None`, then the learning_rate for the
        system to which it belongs is used. If all are `None`, then the
        `default_learning_rate <Reinforcement.default_learning_rate>` is used.

    default_learning_rate : float
        the value used for the `learning_rate <Reinforcement.learning_rate>` if it is not otherwise specified.

    function : function
         the function that computes the weight change matrix, and returns that along with the
         `error_signal <Reinforcement.error_signal>` received.

    owner : Mechanism
        `mechanism <Mechanism>` to which the function belongs.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the `prefs` argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).
    """

    componentName = RL_FUNCTION

    variableClassDefault = [[0], [0], [0]]

    default_learning_rate = 0.05

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    def __init__(self,
                 variable_default=variableClassDefault,
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

        super().__init__(variable_default=variable_default,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        self.functionOutputType = None

    def _validate_variable(self, variable, context=None):
        super()._validate_variable(variable, context)

        if len(self.variable) != 3:
            raise ComponentError("Variable for {} ({}) must have three items (input, output and error arrays)".
                                 format(self.name, self.variable))

        self.activation_input = self.variable[LEARNING_ACTIVATION_INPUT]
        self.activation_output = self.variable[LEARNING_ACTIVATION_OUTPUT]
        self.error_signal = self.variable[LEARNING_ERROR_OUTPUT]

        if len(self.error_signal) != 1:
            raise ComponentError("Error term for {} (the third item of its variable arg) must be an array with a "
                                 "single element for {}".
                                 format(self.name, self.error_signal))

        # Allow initializion with zero but not during a run (i.e., when called from check_args())
        if not INITIALIZING in context:
            if np.count_nonzero(self.activation_output) != 1:
                raise ComponentError("First item ({}) of variable for {} must be an array with a single non-zero value "
                                     "(if output mechanism being trained uses softmax,"
                                     " its \'output\' arg may need to be set to to PROB)".
                                     format(self.variable[LEARNING_ACTIVATION_OUTPUT], self.componentName))

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

        variable : List or 2d np.array [length 3] : default variableClassDefault
           must have three items that are the values for (in order):
           `activation_input <Reinforcement.activation_input>` (not used),
           `activation_output <Reinforcement.activation_output>` (1d np.array with a single non-zero value),
           `error_signal <Reinforcement.error_signal>` (1d np.array).

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------

        diagonal weight change matrix : 2d np.array
            has a single non-zero entry in the same row and column as the one in
            `activation_output <Reinforcement.activation_output>` and `error_signal <Reinforcement.error_signal>`.

        error signal : 1d np.array
            same as value received in `error_signal <Reinforcement.error_signal>` argument.
        """

        self._check_args(variable=variable, params=params, context=context)

        output = self.activation_output
        error = self.error_signal

        # IMPLEMENTATION NOTE: have to do this here, rather than in validate_params for the following reasons:
        #                      1) if no learning_rate is specified for the mechanism, need to assign None
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

        # return:
        # - weight_change_matrix and error_array
        return [weight_change_matrix, error_array]


# Argument names:
ERROR_MATRIX = 'error_matrix'
WT_MATRIX_SENDERS_DIM = 0
WT_MATRIX_RECEIVERS_DIM = 1


class BackPropagation(LearningFunction):
    """
    BackPropagation(                                     \
        variable_default=variableClassDefault,           \
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

    variable : List or 2d np.array [length 3] : default variableClassDefault
       specifies a template for the three items provided as the variable in the call to the
       `function <BackPropagation.function>` (in order):
       `activation_input <BackPropagation.activation_input>` (1d np.array),
       `activation_output <BackPropagation.activation_output>` (1d np.array),
       `error_signal <BackPropagation.error_signal>` (1d np.array).

    activation_derivative : Function or function
        specifies the derivative for the function of the mechanism that generates
        `activation_output <BackPropagation.activation_output>`.

    error_derivative : Function or function
        specifies the derivative for the function of the mechanism that is the receiver of the
        `error_matrix <BackPropagation.error_matrix>`.

    error_matrix : List, 2d np.array, np.matrix, ParameterState, or MappingProjection
        matrix, the output of which is used to calculate the `error_signal <BackPropagation.error_signal>`.
        If it is specified as a ParameterState it must be one for the `matrix <MappingProjection.matrix>`
        parameter of a `MappingProjection`;  if it is a MappingProjection, it must be one with a
        MATRIX parameterState.

    learning_rate : float : default default_learning_rate
        supercedes any specification for the `process <Process>` and/or `system <System>` to which the function's
        `owner <Function.owner>` belongs (see `learning_rate <BackPropagation.learning_rate>` for details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
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
        target (training signal) and the output of the last mechanism in the sequence;
        same as 3rd item of `variable <BackPropagation.variable>.

    error_matrix : 2d np.array or ParameterState
        matrix, the output of which is used to calculate the `error_signal <BackPropagation.error_signal>`;
        if it is a `ParameterState`, it refers to the MATRIX parameterState of the `MappingProjection` being learned.

    learning_rate : float
        the learning rate used by the function.  If specified, it supercedes any learning_rate specified for the
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
        `mechanism <Mechanism>` to which the function belongs.

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for function. Specified in the `prefs` argument of the constructor for the function;
        if it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    componentName = BACKPROPAGATION_FUNCTION

    variableClassDefault = [[0], [0], [0]]

    default_learning_rate = 1.0

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
                 # variable_default:tc.any(list, np.ndarray),
                 activation_derivative_fct: tc.optional(tc.any(function_type, method_type)) = Logistic().derivative,
                 error_derivative_fct: tc.optional(tc.any(function_type, method_type)) = Logistic().derivative,
                 # error_matrix:tc.optional(tc.any(list, np.ndarray, np.matrix, ParameterState, MappingProjection))=None,
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

        super().__init__(variable_default=variable_default,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        self.functionOutputType = None

    def _validate_variable(self, variable, context=None):
        super()._validate_variable(variable, context)

        if len(self.variable) != 3:
            raise ComponentError("Variable for {} ({}) must have three items: "
                                 "activation_input, activation_output, and error_signal)".
                                 format(self.name, self.variable))

        self.activation_input = self.variable[LEARNING_ACTIVATION_INPUT]
        self.activation_output = self.variable[LEARNING_ACTIVATION_OUTPUT]
        self.error_signal = self.variable[LEARNING_ERROR_OUTPUT]

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
        try:
            error_matrix = target_set[ERROR_MATRIX]
        except KeyError:
            raise FunctionError("PROGRAM ERROR:  No specification for {} in {}".
                                format(ERROR_MATRIX, self.name))

        from PsyNeuLink.Components.States.ParameterState import ParameterState
        from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
        if not isinstance(error_matrix, (list, np.ndarray, np.matrix, ParameterState, MappingProjection)):
            raise FunctionError("The {} arg for {} must be a list, 2d np.array, ParamaterState or "
                                "MappingProjection".format(ERROR_MATRIX, self.name))

        if isinstance(error_matrix, MappingProjection):
            try:
                error_matrix = error_matrix.parameterStates[MATRIX].value
                param_type_string = "MappingProjection's ParameterState"
            except KeyError:
                raise FunctionError("The MappingProjection specified for the {} arg of {} ({}) must have a {} "
                                    "paramaterState that has been assigned a 2d array or matrix".
                                    format(ERROR_MATRIX, self.name, error_matrix.shape, MATRIX))

        elif isinstance(error_matrix, ParameterState):
            try:
                error_matrix = error_matrix.value
                param_type_string = "ParameterState"
            except KeyError:
                raise FunctionError("The value of the {} parameterState specified for the {} arg of {} ({}) "
                                    "must be a 2d array or matrix".
                                    format(MATRIX, ERROR_MATRIX, self.name, error_matrix.shape))

        else:
            param_type_string = "array or matrix"

        error_matrix = np.array(error_matrix)
        rows = error_matrix.shape[WT_MATRIX_SENDERS_DIM]
        cols = error_matrix.shape[WT_MATRIX_RECEIVERS_DIM]
        activity_output_len = len(self.activation_output)
        error_signal_len = len(self.error_signal)

        if error_matrix.ndim != 2:
            raise FunctionError("The value of the {} specified for the {} arg of {} ({}) must be a 2d array or matrix".
                                format(param_type_string, ERROR_MATRIX, self.name, error_matrix))

        # The length of the sender outputState.value (the error signal) must be the
        #     same as the width (# columns) of the MappingProjection's weight matrix (# of receivers)

        # Validate that columns (number of receiver elements) of error_matrix equals length of error_signal
        if cols != error_signal_len:
            raise FunctionError("The width (number of columns, {}) of the \'{}\' arg ({}) specified for {} "
                                "must match the length of the error signal ({}) it receives".
                                format(cols, MATRIX, error_matrix.shape, self.name, cols))

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

        variable : List or 2d np.array [length 3] : default variableClassDefault
           must have three items that are the values for (in order):
           `activation_input <BackPropagation.activation_input>` (1d np.array),
           `activation_output <BackPropagation.activation_output>` (1d np.array),
           `error_signal <BackPropagation.error_signal>` (1d np.array).

        params : Optional[Dict[param keyword, param value]]
            a `parameter dictionary <ParameterState_Specifying_Parameters>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the function is executed on the time_step or trial time scale.

        Returns
        -------

        weight change matrix : 2d np.array
            the modifications to make to the matrix.

        weighted error signal : 1d np.array
            `error_signal <BackPropagation.error_signal>`, weighted by the contribution made by each element of
            `activation_output <BackPropagation.activation_output>` as a function of
            `error_matrix <BackPropagation.error_matrix>`.

        """

        from PsyNeuLink.Components.States.ParameterState import ParameterState
        self._check_args(variable=variable, params=params, context=context)
        if isinstance(self.error_matrix, ParameterState):
            error_matrix = self.error_matrix.value
        else:
            error_matrix = self.error_matrix

        # IMPLEMENTATION NOTE: have to do this here, rather than in validate_params for the following reasons:
        #                      1) if no learning_rate is specified for the mechanism, need to assign None
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

        return [weight_change_matrix, dE_dW]

# region *****************************************   OBJECTIVE FUNCTIONS ************************************************
# endregion
# TBI

# region  *****************************************   REGISTER FUNCTIONS ************************************************
