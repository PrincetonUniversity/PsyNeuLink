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
  * :class:`Contradiction`

Combination Components:
  * :class:`LinearCombination`

Transfer Components:
  * :class:`Linear`
  * :class:`Exponential`
  * :class:`Logistic`
  * :class:`SoftMax`
  * :class:`LinearMatrix`

Integrator Components:
  * :class:`Integrator`
  * :class:`BogaczEtAl`
  * :class:`NavarroAndFuss`

Learning Components:
  * :class:`Reinforcement`
  * :class:`BackPropagation`

"""


# __all__ = ['LinearCombination',
#            'Linear',
#            'Exponential',
#            'Logistic',
#            'SoftMax',
#            'Integrator',
#            'LinearMatrix',
#            'ReinforcementLearning',
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
    try:
        return owner.paramsCurrent[FUNCTION].keyword(owner, keyword)
    except FunctionError as e:
        if owner.prefs.verbosePref:
            print ("{} of {}".format(e, owner.name))
        return None
    except AttributeError:
        if owner.prefs.verbosePref:
            print ("Keyword ({}) not recognized for {}".format(keyword, owner.name))
        return None

def get_param_value_for_function(owner, function):
    try:
        return owner.paramsCurrent[FUNCTION].param_function(owner, function)
    except FunctionError as e:
        if owner.prefs.verbosePref:
            print ("{} of {}".format(e, owner.name))
        return None
    except AttributeError:
        if owner.prefs.verbosePref:
            print ("Function ({}) can't be evaluated for {}".format(function, owner.name))
        return None

def optional_parameter_spec(param):
    if not param:
        return True
    return parameter_spec(param)

def parameter_spec(param):
    # if is_numerical(param):
    if isinstance(param, numbers.Number):
        return True
    if isinstance(param, (tuple, function_type, ParamValueProjection)):
        return True
    return False


class Function_Base(Function):
    """Implement abstract class for Component category of Function class

    Description:
        Function functions are ones used by other function categories;
        They are defined here (on top of standard libraries) to provide a uniform interface for managing parameters
         (including defaults)
        NOTE:   the Function category definition serves primarily as a shell, and an interface to the Function class,
                   to maintain consistency of structure with the other function categories;
                it also insures implementation of .function for all Function Components
                (as distinct from other Function subclasses, which can use a FUNCTION param
                    to implement .function instead of doing so directly)
                Function Components are the end of the recursive line: as such:
                    they don't implement functionParams
                    in general, don't bother implementing function, rather...
                    they rely on Function.function which passes on the return value of .function

    Instantiation:
        A function can be instantiated in one of several ways:
IMPLEMENTATION NOTE:  *** DOCUMENTATION

    Variable and Parameters:
IMPLEMENTATION NOTE:  ** DESCRIBE VARIABLE HERE AND HOW/WHY IT DIFFERS FROM PARAMETER
        - Parameters can be assigned and/or changed individually or in sets, by:
          - including them in the initialization call
          - calling the assign_defaults method (which changes their default values)
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
        + name (str) - if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet) - if not specified as an arg, default is created by copying ComponentPreferenceSet

    Instance methods:
        The following method MUST be overridden by an implementation in the subclass:
        - execute(variable, params)
        The following can be implemented, to customize validation of the function variable and/or params:
        - [_validate_variable(variable)]
        - [_validate_params(request_set, target_set, context)]
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
    paramClassDefaults.update({kwFunctionOutputTypeConversion: False}) # Enable/disable output type conversion


    def __init__(self,
                 variable_default,
                 params,
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

        # This is assigned by owner in Function._instantiate_function()
        self.owner = None

        super().__init__(variable_default=variable_default,
                                           param_defaults=params,
                                           name=name,
                                           prefs=prefs,
                                           context=context)

    def execute(self, variable=NotImplemented, params=None, context=None):
        return self.function(variable=variable, params=params, context=context)

    @property
    def functionOutputType(self):
        if self.paramsCurrent[kwFunctionOutputTypeConversion]:
            return self._functionOutputType
        return None

    @functionOutputType.setter
    def functionOutputType(self, value):

        # Attempt to set outputType but conversion not enabled
        if value and not self.paramsCurrent[kwFunctionOutputTypeConversion]:
            raise FunctionError("output conversion is not enabled for {0}".format(self.__class__.__name__))

        # Bad outputType specification
        if value and not isinstance(value, FunctionOutputType):
            raise FunctionError("value ({0}) of functionOutputType attribute must be FunctionOutputType for {1}".
                               format(self.functionOutputType, self.__class__.__name__))

        # Can't convert from arrays of length > 1 to number
        # if (self.variable.size  > 1 and (self.functionOutputType is FunctionOutputType.RAW_NUMBER)):
        if (len(self.variable)  > 1 and (self.functionOutputType is FunctionOutputType.RAW_NUMBER)):
            raise FunctionError("{0} can't be set to return a single number since its variable has more than one number".
                               format(self.__class__.__name__))
        self._functionOutputType = value


# *****************************************   EXAMPLE FUNCTION   *******************************************************


class Contradiction(Function_Base): # Example
    """Example function for use as template for function construction

    Iniialization arguments:
     - variable (boolean or statement resolving to boolean)
     - params (dict) specifying the:
         + propensity (kwPropensity: a mode specifying the manner of responses (tendency to agree or disagree)
         + pertinacity (kwPertinacity: the consistency with which the manner complies with the propensity

    Contradiction.function returns :keyword:`True` or :keyword:`False`
    """

    # Function componentName and type (defined at top of module)
    componentName = kwContradiction
    componentType = kwExampleFunction

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
    #  in the initialization call or later (using either assign_defaults or during a function call)
    kwPropensity = "PROPENSITY"
    kwPertinacity = "PERTINACITY"
    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({kwPropensity: Manner.CONTRARIAN,
                          kwPertinacity:  10,
                          })

    def __init__(self,
                 variable_default=variableClassDefault,
                 params=None,
                 prefs:is_pref_set=None,
                 context=componentName+INITIALIZING):
        # This validates variable and/or params_list if assigned (using _validate_params method below),
        #    and assigns them to paramsCurrent and paramInstanceDefaults;
        #    otherwise, assigns paramClassDefaults to paramsCurrent and paramInstanceDefaults
        # NOTES:
        #    * paramsCurrent can be changed by including params in call to function
        #    * paramInstanceDefaults can be changed by calling assign_default
        super(Contradiction, self).__init__(variable_default=variable_default,
                                            params=params,
                                            prefs=prefs,
                                            context=context)

    def function(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale=TimeScale.TRIAL,
                context=None):
        """Returns a boolean that is (or tends to be) the same as or opposite the one passed in

        Returns :keyword:`True` or :keyword:`False`, that is either the same or opposite the statement passed in as the
        variable
        The propensity parameter must be set to be Manner.OBSEQUIOUS or Manner.CONTRARIAN, which
            determines whether the response is (or tends to be) the same as or opposite to the statement
        The pertinacity parameter determines the consistency with which the response conforms to the manner

        :param variable: (boolean) Statement to probe
        :param params: (dict) with entires specifying
                       kwPropensity: Contradiction.Manner - contrarian or obsequious (default: CONTRARIAN)
                       kwPertinacity: float - obstinate or equivocal (default: 10)
        :return response: (boolean)
        """
        self._check_args(variable, params, context)

        # Compute the function

        # Use self.variable (rather than variable), as it has been validated (and default assigned, if necessary)
        statement = self.variable
        propensity = self.paramsCurrent[self.kwPropensity]
        pertinacity = self.paramsCurrent[self.kwPertinacity]
        whim = randint(-10, 10)

        if propensity == self.Manner.OBSEQUIOUS:
            return whim < pertinacity

        elif propensity == self.Manner.CONTRARIAN:
            return whim > pertinacity

        else:
            raise FunctionError("This should not happen if parameter_validation == True;  check its value")

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
                (isinstance(variable, numbers.Number) and  isinstance(self.variableClassDefault, numbers.Number)):
            self.variable = variable
        else:
            raise FunctionError("Variable must be {0}".format(type(self.variableClassDefault)))

    def _validate_params(self, request_set, target_set=NotImplemented, context=None):
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

            if param_name == self.kwPropensity:
                if isinstance(param_value, Contradiction.Manner):
                    # target_set[self.kwPropensity] = param_value
                    pass # This leaves param in request_set, clear to be assigned to target_set in call to super below
                else:
                    message = "Propensity must be of type Example.Mode"
                continue

            # Validate param
            if param_name == self.kwPertinacity:
                if isinstance(param_value, numbers.Number) and 0 <= param_value <= 10:
                    # target_set[self.kwPertinacity] = param_value
                    pass # This leaves param in request_set, clear to be assigned to target_set in call to super below
                else:
                    message += "Pertinacity must be a number between 0 and 10"
                continue

        if message:
            raise FunctionError(message)

        super(Contradiction, self)._validate_params(request_set, target_set, context)


#region ****************************************   FUNCTIONS   *********************************************************
#endregion

#region **********************************  COMBINATION FUNCTIONS  *****************************************************
#endregion

class CombinationFunction(Function_Base):
    componentType = kwCombinationFunction


class Reduce(CombinationFunction): # ------------------------------------------------------------------------
# FIX: CONFIRM THAT 1D KWEIGHTS USES EACH ELEMENT TO SCALE CORRESPONDING VECTOR IN VARIABLE
# FIX  CONFIRM THAT LINEAR TRANSFORMATION (OFFSET, SCALE) APPLY TO THE RESULTING ARRAY
# FIX: CONFIRM RETURNS LIST IF GIVEN LIST, AND SIMLARLY FOR NP.ARRAY
    """Combines an array of values into a single value

    Description:
        Combine elements of an array in variable arg, using arithmetic operation determined by OPERATION.
        Returns a scalar value.

    Initialization arguments:
     - variable (list or np.ndarray of numbers): values to be combined;
     - params (dict) can include:
         + OPERATION (Operation Enum) - method used to combine terms (default: SUM)
              SUM: element-wise sum of the arrays in variable
              PRODUCT: Hadamard Product of the arrays in variable

    Reduce.function returns combined values:
    - single scalar value
    """
    componentName = kwReduce

    # # Operation indicators
    # class Operation(Enum):
    #     SUM = 0
    #     PRODUCT = 1
    #     SUBTRACT = 2
    #     DIVIDE = 3
    #
    variableClassDefault = [0, 0]
    # variableClassDefault_locked = True

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
                 operation:tc.enum(SUM, PRODUCT)=SUM,
                 params=None,
                 prefs:is_pref_set=None,
                 context=componentName+INITIALIZING):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(operation=operation,
                                                  params=params)

        super().__init__(variable_default=variable_default,
                         params=params,
                         prefs=prefs,
                         context=context)

    def _validate_variable(self, variable, context=None):
        """Insure that all items of list or np.ndarray in variable are of the same length

        Args:
            variable:
            context:
        """
        super()._validate_variable(variable=variable, context=context)
        if not is_numerical(variable):
            raise FunctionError("All elements of {} must be scalar values".
                                format(self.__class__.__name__))


    def function(self,
                variable=NotImplemented,
                params=NotImplemented,
                context=None):
        """Combine a list or array of values

        Returns a scalar value

        :var variable: (list or np.array of numbers) - values to calculate (default: [0, 0]:
        :params: (dict) with entries specifying:
                           OPERATION: LinearCombination.Operation - operation to perform (default: SUM):
        :return: (scalar)
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


kwLinearCombinationInitializer = "Initializer"


class LinearCombination(CombinationFunction): # ------------------------------------------------------------------------
# FIX: CONFIRM THAT 1D KWEIGHTS USES EACH ELEMENT TO SCALE CORRESPONDING VECTOR IN VARIABLE
# FIX  CONFIRM THAT LINEAR TRANSFORMATION (OFFSET, SCALE) APPLY TO THE RESULTING ARRAY
# FIX: CONFIRM RETURNS LIST IF GIVEN LIST, AND SIMLARLY FOR NP.ARRAY
    """Linearly combine arrays of values with optional weighting, offset, and/or scaling

    Description:
        Combine corresponding elements of arrays in variable arg, using arithmetic operation determined by OPERATION
        Use optional WEIGHTING argument to weight contribution of each array to the combination
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
         + WEIGHTS (list of numbers or 1D np.array): multiplies each variable before combining them (default: [1, 1])
         + OFFSET (value): added to the result (after the arithmetic operation is applied; default is 0)
         + SCALE (value): multiples the result (after combining elements; default: 1)
         + OPERATION (Operation Enum) - method used to combine terms (default: SUM)
              SUM: element-wise sum of the arrays in variable
              PRODUCT: Hadamard Product of the arrays in variable

    LinearCombination.function returns combined values:
    - single number if variable was a single number
    - list of numbers if variable was list of numbers
    - 1D np.array if variable was a single np.variable or np.ndarray
    """

    componentName = kwLinearCombination

    # # Operation indicators
    # class Operation(Enum):
    #     SUM = 0
    #     PRODUCT = 1
    #     SUBTRACT = 2
    #     DIVIDE = 3
    #
    variableClassDefault = [2, 2]
    # variableClassDefault_locked = True

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
                 scale:parameter_spec=1.0,
                 offset:parameter_spec=0.0,
                 # IMPLEMENTATION NOTE - these don't check whether every element of np.array is numerical:
                 # exponents:tc.optional(tc.any(int, float, tc.list_of(tc.any(int, float)), np.ndarray))=None,
                 # weights:tc.optional(tc.any(int, float, tc.list_of(tc.any(int, float)), np.ndarray))=None,
                 exponents:is_numerical_or_none=None,
                 weights:is_numerical_or_none=None,
                 operation:tc.enum(SUM, PRODUCT, DIFFERENCE, QUOTIENT)=SUM,
                 params=None,
                 prefs:is_pref_set=None,
                 context=componentName+INITIALIZING):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(scale=scale,
                                                 offset=offset,
                                                 exponents=exponents,
                                                 weights=weights,
                                                 operation=operation,
                                                 params=params)

        super(LinearCombination, self).__init__(variable_default=variable_default,
                                                params=params,
                                                prefs=prefs,
                                                context=context)

        if not self.exponents is None:
            self.exponents = np.atleast_2d(self.exponents).reshape(-1,1)
        if not self.weights is None:
            self.weights = np.atleast_2d(self.weights).reshape(-1,1)


# MODIFIED 6/12/16 NEW:
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
            length=0
            for i in range(len(variable)):
                if i==0:
                    continue
                if isinstance(variable[i-1], numbers.Number):
                    old_length = 1
                else:
                    old_length = len(variable[i-1])
                if isinstance(variable[i], numbers.Number):
                    new_length = 1
                else:
                    new_length = len(variable[i])
                if old_length != new_length:
                    raise FunctionError("Length of all arrays in variable {0} for {1} must be the same".
                                       format(variable, self.__class__.__name__))


    def _validate_params(self, request_set, target_set=NotImplemented, context=None):
        """Insure that EXPONENTS and WEIGHTS are lists or np.arrays of numbers with length equal to variable

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

        # exponents = target_set[EXPONENTS]
        # weights = target_set[WEIGHTS]
        # operation = target_set[OPERATION]

#         # IMPLEMENTATION NOTE: checking is now taken care of by typecheck;  now only need to convert
#         # Make sure exponents is a list of numbers or an np.ndarray
# # FIX: CHANGE THIS AND WEIGHTS TO TRY/EXCEPT
#         if not exponents is None and not exponents is NotImplemented:
#             if ((isinstance(exponents, list) and all(isinstance(elem, numbers.Number) for elem in exponents)) or
#                     isinstance(exponents, np.ndarray)):
#                 # convert to 2D np.ndarrray (to distribute over 2D self.variable array)
#                 target_set[EXPONENTS] = np.atleast_2d(target_set[EXPONENTS]).reshape(-1,1)
#             else:
#                 raise FunctionError("EXPONENTS param ({0}) for {1} must be a list of numbers or an np.array".
#                                format(exponents, self.name))
#
#         # Make sure weights is a list of numbers or an np.ndarray
#         if not weights is None and not weights is NotImplemented:
#             if ((isinstance(weights, list) and all(isinstance(elem, numbers.Number) for elem in weights)) or
#                     isinstance(weights, np.ndarray)):
#                 # convert to 2D np.ndarrray (to distribute over 2D self.variable array)
#                 target_set[WEIGHTS] = np.atleast_2d(target_set[WEIGHTS]).reshape(-1,1)
#             else:
#                 raise FunctionError("WEIGHTS param ({0}) for {1} must be a list of numbers or an np.array".
#                                format(weights, self.name))

        if not target_set[EXPONENTS] is None:
            target_set[EXPONENTS] = np.atleast_2d(target_set[EXPONENTS]).reshape(-1,1)
        if not target_set[WEIGHTS] is None:
            target_set[WEIGHTS] = np.atleast_2d(target_set[WEIGHTS]).reshape(-1,1)

        # if not operation:
        #     raise FunctionError("Operation param missing")
        # if not operation == self.Operation.SUM and not operation == self.Operation.PRODUCT:
        #     raise FunctionError("Operation param ({0}) must be Operation.SUM or Operation.PRODUCT".format(operation))
# MODIFIED 6/12/16 END

    def function(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale=TimeScale.TRIAL,
                context=None):
        """Linearly combine a list of values, and optionally offset and/or scale them

# DOCUMENT:
        Handles 1-D or 2-D arrays of numbers
        Convert to np.array
        All elements must be numeric
        If linear (single number or 1-D array of numbers) just apply scale and offset
        If 2D (array of arrays), apply exponents to each array
        If 2D (array of arrays), apply weights to each array
        Operators:  SUM AND PRODUCT
        -------------------------------------------
        OLD:
        Variable must be a list of items:
            - each item can be a number or a list of numbers
        Corresponding elements of each item in variable are combined based on OPERATION param:
            - SUM adds corresponding elements
            - PRODUCT multiples corresponding elements
        An initializer (kwLinearCombinationInitializer) can be provided as the first item in variable;
            it will be populated with a number of elements equal to the second item,
            each element of which is determined by OPERATION param:
            - for SUM, initializer will be a list of 0's
            - for PRODUCT, initializer will be a list of 1's
        Returns a list of the same length as the items in variable,
            each of which is the combination of their corresponding elements specified by OPERATION

        :var variable: (list of numbers) - values to calculate (default: [0, 0]:
        :params: (dict) with entries specifying:
                           EXPONENTS (2D np.array): exponentiate each value in the variable array (default: none)
                           WEIGHTS (2D np.array): multiply each value in the variable array (default: none):
                           OFFSET (scalar) - additive constant (default: 0):
                           SCALE: (scalar) - scaling factor (default: 1)
                           OPERATION: LinearCombination.Operation - operation to perform (default: SUM):
        :return: (1D np.array)
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
        if not exponents is None and not exponents is NotImplemented:
            if len(exponents) != len(self.variable):
                raise FunctionError("Number of exponents ({0}) does not equal number of items in variable ({1})".
                                   format(len(exponents), len(self.variable.shape)))
            # Avoid divide by zero warning:
            #    make sure there no zeros for an element that is assigned a negative exponent
            if INITIALIZING in context and any(not i and j<0 for i,j in zip(self.variable, exponents)):
                self.variable = np.ones_like(self.variable)
            else:
                self.variable = self.variable ** exponents

        # Apply weights if they were specified
        if not weights is None and not weights is NotImplemented:
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


#region ***********************************  TRANSFER FUNCTIONS  *******************************************************
#endregion

class TransferFunction(Function_Base):
    componentType = kwTransferFunction


class Linear(TransferFunction): # --------------------------------------------------------------------------------------
    """Calculate a linear transform of input variable (SLOPE, INTERCEPT)

    Initialization arguments:
     - variable (number): transformed by linear function: slope * variable + intercept
     - params (dict): specifies
         + slope (SLOPE: value) - slope (default: 1)
         + intercept (INTERCEPT: value) - intercept (defaul: 0)

    Linear.function returns scalar result
    """

    componentName = kwLinear

    variableClassDefault = [0]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
                               kwFunctionOutputTypeConversion: True})

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
                 slope:parameter_spec=1,
                 intercept:parameter_spec=0,
                 params=None,
                 prefs:is_pref_set=None,
                 context=componentName+INITIALIZING):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(slope=slope,
                                                 intercept=intercept,
                                                 params=params)

        super(Linear, self).__init__(variable_default=variable_default,
                                     params=params,
                                     prefs=prefs,
                                     context=context)

        self.functionOutputType = None

    def function(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale=TimeScale.TRIAL,
                context=None):
        """Calculate single value (defined by slope and intercept)

        :var variable: (number) - value to be "plotted" (default: 0
        :parameter params: (dict) with entries specifying:
                           SLOPE: number - slope (default: 1)
                           INTERCEPT: number - intercept (default: 0)
        :return number:
        """

        self._check_args(variable, params, context)

        slope = self.paramsCurrent[SLOPE]
        intercept = self.paramsCurrent[INTERCEPT]
        outputType = self.functionOutputType

        # By default, result should be returned as np.ndarray with same dimensionality as input
        result = self.variable * slope + intercept

        #region Type conversion (specified by outputType):
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
        #endregion

        return result

    def derivative(self, output, input=None):
        """Derivative of the softMax sigmoid function
        """
        # FIX: ??CORRECT:
        return self.slope
        # raise FunctionError("Derivative not yet implemented for {}".format(self.componentName))


class Exponential(TransferFunction): # ---------------------------------------------------------------------------------
    """Calculate an exponential transform of input variable  (RATE, SCALE)

    Initialization arguments:
     - variable (number):
         + scalar value to be transformed by exponential function: scale * e**(rate * x)
     - params (dict): specifies
         + rate (RATE: coeffiencent on variable in exponent (default: 1)
         + scale (SCALE: coefficient on exponential (default: 1)

    Exponential.function returns scalar result
    """

    componentName = kwExponential

    # # Params
    # RATE = "rate"
    # SCALE = "scale"

    variableClassDefault = 0

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
                 rate:parameter_spec=1.0,
                 scale:parameter_spec=1.0,
                 params=None,
                 prefs:is_pref_set=None,
                 context=componentName + INITIALIZING):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                 scale=scale,
                                                 params=params)

        super(Exponential, self).__init__(variable_default=variable_default,
                                          params=params,
                                          prefs=prefs,
                                          context=context)
        TEST = True

    def function(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale=TimeScale.TRIAL,
                context=None):
        """Exponential function

        :var variable: (number) - value to be exponentiated (default: 0
        :parameter params: (dict) with entries specifying:
                           RATE: number - rate (default: 1)
                           SCALE: number - scale (default: 1)
        :return number:
        """

        self._check_args(variable, params, context)

        # Assign the params and return the result
        rate = self.paramsCurrent[RATE]
        scale = self.paramsCurrent[SCALE]

        return scale * np.exp(rate * self.variable)

    def derivative(self, output, input=None):
        """Derivative of the softMax sigmoid function
        """
        # FIX: ??CORRECT:
        return output
        # raise FunctionError("Derivative not yet implemented for {}".format(self.componentName))


class Logistic(TransferFunction): # ------------------------------------------------------------------------------------
    """Calculate the logistic transform of input variable  (GAIN, BIAS)

    Initialization arguments:
     - variable (number):
         + scalar value to be transformed by logistic function: 1 / (1 + e**(gain*variable + bias))
     - params (dict): specifies
         + gain (GAIN): coeffiencent on exponent (default: 1)
         + bias (BIAS): additive constant in exponent (default: 0)

    Logistic.function returns scalar result
    """

    componentName = kwLogistic

    variableClassDefault = 0

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
                 gain:parameter_spec=1.0,
                 bias:parameter_spec=0.0,
                 params=None,
                 prefs:is_pref_set=None,
                 context='Logistic Init'):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(gain=gain,
                                                 bias=bias,
                                                 params=params)

        super().__init__(variable_default=variable_default,
                         params=params,
                         prefs=prefs,
                         context=context)

    def function(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale=TimeScale.TRIAL,
                context=None):
        """Logistic sigmoid function

        :var variable: (number) - value to be transformed by logistic function (default: 0)
        :parameter params: (dict) with entries specifying:
                           GAIN: number - gain (default: 1)
                           BIAS: number - rate (default: 0)
        :return number:
        """

        self._check_args(variable, params, context)

        # Assign the params and return the result
        gain = self.paramsCurrent[GAIN]
        bias = self.paramsCurrent[BIAS]

        return 1 / (1 + np.exp(-(gain * self.variable) + bias))

    def derivative(self, output, input=None):
        """Derivative of the logistic signmoid function
        """
        return output*(1-output)


class SoftMax(TransferFunction): # -------------------------------------------------------------------------------------
    """Calculate the softMax transform of input variable  (GAIN, BIAS)

    Initialization arguments:
     - variable (number):
         + scalar value to be transformed by softMax function: e**(gain * variable) / sum(e**(gain * variable))
     - params (dict): specifies
         + gain (GAIN): coeffiencent on exponent (default: 1)
         + output (OUTPUT_TYPE): determines how to populate the return array (default: ALL)
             ALL: array each element of which is the softmax value of the elements in the input array
             MAX_VAL: array with a scalar for the element with the maximum softmax value, and zeros elsewhere
             MAX_INDICATOR: array with a one for the element with the maximum softmax value, and zeros elsewhere
             PROB: probabilistially picks an element based on their softmax values to pass through; all others are zero
         # + max (kwMax): only reports max value, all others set to 0 (default: :keyword:`False`)

    SoftMax.function returns scalar result
    """

    componentName = kwSoftMax

    variableClassDefault = 0

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
                 gain:parameter_spec=1.0,
                 output:tc.enum(ALL, MAX_VAL, MAX_INDICATOR, PROB)=ALL,
                 params:tc.optional(dict)=None,
                 prefs:is_pref_set=None,
                 context='SoftMax Init'):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(gain=gain,
                                                 output=output,
                                                 params=params)

        super().__init__(variable_default=variable_default,
                         params=params,
                         prefs=prefs,
                         context=context)

    def function(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale=TimeScale.TRIAL,
                context=None):
        """SoftMax sigmoid function

        :var variable: (number) - value to be transformed by softMax function (default: 0)
        :parameter params: (dict) with entries specifying:
                           GAIN: number - gain (default: 1)
                           BIAS: number - rate (default: 0)
        :return number:
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
            # sm = np.where(sm == np.max(sm), 1, 0)
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
            chosen_item = next(element for element in cum_sum if element>random_value)
            chosen_in_cum_sum = np.where(cum_sum == chosen_item, 1, 0)
            sm = self.variable * chosen_in_cum_sum

        return sm

    def derivative(self, output, input=None):
        """Derivative of the softMax sigmoid function
        """
        # FIX: ??CORRECT:
        indicator = self.function(input, params={MAX_VAL:True})
        return output - indicator
        # raise FunctionError("Derivative not yet implemented for {}".format(self.componentName))


def matrix_spec(m):
    if m is None:
        return True
    if m in {IDENTITY_MATRIX, FULL_CONNECTIVITY_MATRIX, RANDOM_CONNECTIVITY_MATRIX}:
        return True
    if isinstance(m, (list, np.ndarray, np.matrix, function_type)):
        return True
    return False


class LinearMatrix(TransferFunction):  # -----------------------------------------------------------------------------------
    """Map sender vector to receiver vector using a linear weight matrix  (kwReceiver, MATRIX)

    Use a weight matrix to convert a sender vector into a receiver vector:
    - each row of the mapping corresponds to an element of the sender vector (outer index)
    - each column of the mapping corresponds to an element of the receiver vector (inner index):

    ----------------------------------------------------------------------------------------------------------
    MATRIX FORMAT
                                     INDICES:
                                 Receiver elements:
                            0       1       2       3
                        0  [0,0]   [0,1]   [0,2]   [0,3]
    Sender elements:    1  [1,0]   [1,1]   [1,2]   [1,3]
                        2  [2,0]   [2,1]   [2,2]   [2,3]

    matrix.shape => (sender/rows, receiver/cols)

    ----------------------------------------------------------------------------------------------------------
    ARRAY FORMAT
                                                                        INDICES
                                           [ [      Sender 0 (row0)      ], [       Sender 1 (row1)      ]... ]
                                           [ [ rec0,  rec1,  rec2,  rec3 ], [ rec0,  rec1,  rec2,  rec3  ]... ]
    matrix[senders/rows, receivers/cols]:  [ [ row0,  row0,  row0,  row0 ], [ row1,  row1,  row1,  row1  ]... ]
                                           [ [ col0,  col1,  col2,  col3 ], [ col0,  col1,  col2,  col3  ]... ]
                                           [ [[0,0], [0,1], [0,2], [0,3] ], [[1,0], [1,1], [1,2], [1,3] ]... ]

    ----------------------------------------------------------------------------------------------------------

    Initialization arguments:
    - variable (2D np.ndarray containing exactly two sub-arrays:  sender and receiver vectors
    - params (dict) specifying:
         + filler (kwFillerValue: number) value used to initialize all entries in matrix (default: 0)
         + identity (kwkwIdentityMapping: boolean): constructs identity matrix (default: :keyword:`False`)

    Create a matrix in self.matrix that is used in calls to LinearMatrix.function.

    Returns sender 2D array linearly transformed by self.matrix
    """

    componentName = kwLinearMatrix

    DEFAULT_FILLER_VALUE = 0

    variableClassDefault = [DEFAULT_FILLER_VALUE]  # Sender vector

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
                 matrix:matrix_spec=None,
                 params=None,
                 prefs:is_pref_set=None,
                 context=componentName + INITIALIZING):
        """Transforms variable (sender vector) using matrix specified by params, and returns receiver vector

        Variable = sender vector (list of numbers)

        :param variable_default: (list) - list of numbers (default: [0]
        :param params: (dict) with entries specifying:
                                kwReceiver: value - list of numbers, determines width (cols) of matrix (defalut: [0])
                                MATRIX: value - value used to initialize matrix;  can be one of the following:
                                    + single number - used to fill self.matrix (default: DEFAULT_FILLER_VALUE)
                                    + matrix - assigned to self.matrix
                                    + kwIdentity - create identity matrix (diagonal elements = 1;  all others = 0)
        :return none
        """

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(matrix=matrix,
                                                 params=params)

        # Note: this calls _validate_variable and _validate_params which are overridden below;
        #       the latter implements the matrix if required
        # super(LinearMatrix, self).__init__(variable_default=variable_default,
        super().__init__(variable_default=variable_default,
                         params=params,
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
            raise FunctionError("PROGRAM ACTIVATION_ERROR: variable ({0}) for {1} should be an np.ndarray".
                               format(self.variable, self.__class__.__name__))
        else:
            if is_not_1D:
                raise FunctionError("variable ({0}) for {1} must be a 1D np.ndarray".
                                   format(self.variable, self.__class__.__name__))

    def _validate_params(self, request_set, target_set=NotImplemented, context=None):
        """Validate params and assign to targets

        This overrides the class method, to perform more detailed type checking (see explanation in class method).
        Note: this method (or the class version) is called only if the parameter_validation attribute is :keyword:`True`

        :param request_set: (dict) - params to be validated
        :param target_set: (dict) - destination of validated params
        :param context: (str)
        :return none:
        """

        super(LinearMatrix, self)._validate_params(request_set, target_set, context)
        param_set = target_set
        sender = self.variable
        # Note: this assumes self.variable is a 1D np.array, as enforced by _validate_variable
        sender_len = sender.size


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
                print ("Identity matrix requested but kwReceiver not specified; sender length ({0}) will be used".
                       format(sender_len))
            self.receiver = param_set[kwReceiver] = sender

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

# FIX: IMPLEMENT AUTO_ASSIGN_MATRIX HERE: PASS, AS SHOULD HAVE BEEN HANDLED BY CALLER (E.G., MAPPING._instantiate_receiver)
# FIX: IMPLEMENT RANDOM_CONNECTIVITY_MATRIX?
                #np.matrix or np.ndarray provided, so validate that it is numeric and check dimensions
                elif isinstance(param_value, (np.ndarray, np.matrix)):
                    # get dimensions specified by:
                    #   variable (sender): width/cols/outer index
                    #   kwReceiver param: height/rows/inner index

                    weight_matrix = np.matrix(param_value)
                    if 'U' in repr(weight_matrix.dtype):
                        raise FunctionError("Non-numeric entry in MATRIX specification ({0})".format(param_value))

                    matrix_rows = weight_matrix.shape[0]
                    matrix_cols = weight_matrix.shape[1]

                    # Check that number of rows equals length of sender vector (variable)
                    if matrix_rows != sender_len:
                        raise FunctionError("The number of rows ({0}) of the matrix provided does not equal the "
                                            "length ({1}) of the sender vector (variable)".
                                            format(matrix_rows, sender_len))
                    # MODIFIED 9/21/16:
                    #  IF MATRIX IS SPECIFIED, NO NEED TO VALIDATE RECEIVER_LEN (AND MAY NOT EVEN KNOW IT YET)
                    #  SINCE _instantiate_function() IS GENERALLY CALLED BEFORE _instantiate_receiver()
                    # # Check that number of columns equals length of specified receiver vector (kwReceiver)
                    # if matrix_cols != receiver_len:
                    #     raise FunctionError("The number of columns ({}) of the matrix provided for {} "
                    #                        "does not equal the length ({}) of the reciever vector (kwReceiver param)".
                    #                         format(matrix_cols, self.name, receiver_len))

                # Auto, full or random connectivity matrix requested (using keyword):
                # Note:  assume that these will be properly processed by caller (e.g., Mapping._instantiate_receiver)
                elif param_value in {AUTO_ASSIGN_MATRIX, FULL_CONNECTIVITY_MATRIX, RANDOM_CONNECTIVITY_MATRIX}:
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
                        raise FunctionError("Identity matrix requested, but length of receiver ({0})"
                                           " does not match length of sender ({1})".format(receiver_len, sender_len))
                    continue

                # list used to describe matrix, so convert to 2D np.array and pass to validation of matrix below
                elif isinstance(param_value, list):
                    try:
                        param_value = np.atleast_2d(param_value)
                    except (ValueError, TypeError) as error_msg:
                        raise FunctionError("Error in list specification ({0}) of matrix for {1}: {2})".
                                           format(param_value, self.__class__.__name__, error_msg))

                # string used to describe matrix, so convert to np.matrix and pass to validation of matrix below
                elif isinstance(param_value, str):
                    try:
                        param_value = np.matrix(param_value)
                    except (ValueError, TypeError) as error_msg:
                        raise FunctionError("Error in string specification ({0}) of matrix for {1}: {2})".
                                           format(param_value, self.__class__.__name__, error_msg))

                

                # function so:
                # - assume it uses random.rand()
                # - call with two args as place markers for cols and rows
                # -  validate that it returns an np.array or np.matrix
                elif isinstance(param_value, function_type):
                    test = param_value(1,1)
                    if not isinstance(test, (np.ndarray, np.matrix)):
                        raise FunctionError("A function is specified for matrix for {1}: {2}) "
                                           "that returns a value ({}) that is neither a matrix nor array ".
                               format(param_value, self.__class__.__name__, test))

                else:
                    raise FunctionError("Value of {0} param ({1}) must be a matrix, a number (for filler), "
                                       "or a matrix keyword ({2}, {3})".
                                        format(param_name, param_value, IDENTITY_MATRIX, FULL_CONNECTIVITY_MATRIX))
            else:
                message += "Param {0} not recognized by {1} function".format(param_name, self.componentName)
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
            raise FunctionError("Can't instantiate matrix specification ({}) for {} "
                               "since its receiver has not been specified".
                               format(specification, self.__class__.__name__))
            # receiver = sender
        receiver_len = receiver.shape[0]

        matrix = get_matrix(specification, rows=sender_len, cols=receiver_len, context=context)

        # This should never happen (should have been picked up in validate_param or above)
        if matrix is None:
            raise FunctionError("MATRIX param ({0}) must be a matrix, a function that returns one, "
                               "a matrix specification keyword, or a number (filler)".
                                format(specification))
        else:
            return matrix

    def function(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale=TimeScale.TRIAL,
                context=None):
        """Transforms variable vector using either self.matrix or specification in params

        :var variable: (list) - vector of numbers with length equal of height (number of rows, inner index) of matrix
        :parameter params: (dict) with entries specifying:
                            MATRIX: value - used to override self.matrix implemented by __init__;  must be one of:
                                                 + 2D matrix - two-item list, each of which is a list of numbers with
                                                              length that matches the length of the vector in variable
                                                 + kwIdentity - specifies use of identity matrix (dimensions of vector)
                                                 + number - used to fill matrix of same dimensions as self.matrix
        :return list of numbers: vector with length = width (number of columns, outer index) of matrix
        """

        # Note: this calls _validate_variable and _validate_params which are overridden above;
        self._check_args(variable, params, context=context)

        return np.dot(self.variable, self.matrix)

    def keyword(self, keyword):

        # # MODIFIED 10/29/16 OLD:
        # matrix = get_matrix(keyword)
        # MODIFIED 10/29/16 NEW:
        from PsyNeuLink.Components.Projections.Mapping import Mapping
        rows = None
        cols = None
        if isinstance(self, Mapping):
            rows = len(self.sender.value)
            cols = len(self.receiver.variable)
        matrix = get_matrix(keyword, rows, cols)
        # MODIFIED 10/29/16 END

        if matrix is None:
            raise FunctionError("Unrecognized keyword ({}) specified for LinearMatrix Function Function".format(keyword))
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
            raise FunctionError("Specification of np.array for matrix ({}) in {} was more than 2d".
                               format(specification,self.name))

    if specification is AUTO_ASSIGN_MATRIX:
        if rows == cols:
            specification = IDENTITY_MATRIX
        else:
            specification = FULL_CONNECTIVITY_MATRIX

    if specification == FULL_CONNECTIVITY_MATRIX:
        return np.full((rows, cols),1.0)

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


#region ***********************************  INTEGRATOR FUNCTIONS ******************************************************

#  Integrator
#  DDM_BogaczEtAl
#  DDM_NavarroAndFuss

class IntegratorFunction(Function_Base):
    componentType = kwIntegratorFunction


class Integrator(IntegratorFunction): # --------------------------------------------------------------------------------------
    """Calculate an accumulated and/or time-averaged value for input variable using a specified accumulation method

    Initialization arguments:
     - variable: new input value, to be combined with old value at rate and using method specified by params
     - params (dict): specifying:
         + kwInitializer (value): initial value to which to set self.oldValue (default: variableClassDefault)
             - must be same type and format as variable
             - can be specified as a runtime parameter, which resets oldValue to one specified
             Note: self.oldValue stores previous value with which new value is integrated
         + SCALE (value): rate of accumuluation based on weighting of new vs. old value (default: 1)
         + WEIGHTING (Weightings Enum): method of accumulation (default: LINEAR):
                LINEAR -- returns old_value incremented by rate parameter (simple accumulator)
                SCALED -- returns old_value incremented by rate * new_value
                TIME_AVERAGED -- returns rate-weighted average of old and new values  (Delta rule, Wiener filter)
                                rate = 0:  no change (returns old_value)
                                rate 1:    instantaneous change (returns new_value)

    Class attributes:
    - oldValue (value): stores previous value with which value provided in variable is integrated

    Integrator.function returns scalar result
    """

    componentName = kwIntegrator

    variableClassDefault = [[0]]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({kwInitializer: variableClassDefault})


    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
                 rate:parameter_spec=1.0,
                 weighting:tc.enum(LINEAR, SCALED, TIME_AVERAGED)=LINEAR,
                 params:tc.optional(dict)=None,
                 prefs:is_pref_set=None,
                 context='Integrator Init'):

        # Assign here as default, for use in initialization of function
        self.oldValue = self.paramClassDefaults[kwInitializer]

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                 weighting=weighting,
                                                 params=params)

        super(Integrator, self).__init__(variable_default=variable_default,
                                         params=params,
                                         prefs=prefs,
                                         context=context)

        # Reassign to kWInitializer in case default value was overridden
        self.oldValue = self.paramsCurrent[kwInitializer]

    def _validate_params(self, request_set, target_set=NotImplemented, context=None):
        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)
        try:
            if not iscompatible(target_set[kwInitializer],self.variableClassDefault):
                raise FunctionError("kwInitializer param {0} for {1} must be same type as variable {2}".
                                   format(target_set[kwInitializer],
                                          self.__class__.__name__,
                                          self.variable))
        except KeyError:
            pass

    # def function(self, old_value, new_value, param_list=NotImplemented):

    def function(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale=TimeScale.TRIAL,
                context=None):
        """Integrator function

        :var variable: (list) - old_value and new_value (default: [0, 0]:
        :parameter params: (dict) with entries specifying:
                        RATE: number - rate of accumulation as relative weighting of new vs. old value  (default = 1)
                        WEIGHTING: Integrator.Weightings - type of weighting (default = Weightings.LINEAR)
        :return number:
        """

# FIX:  CONVERT TO NP?

# FIX:  NEED TO CONVERT OLD_VALUE TO NP ARRAY

        self._check_args(variable, params, context)

        rate = float(self.paramsCurrent[RATE])
        weighting = self.paramsCurrent[WEIGHTING]

        try:
            old_value = params[kwInitializer]
        except (TypeError, KeyError):
            old_value = self.oldValue

        old_value = np.atleast_2d(old_value)

        new_value = self.variable

        # Compute function based on weighting param
        if weighting is LINEAR:
            value = old_value + rate
            # return value
        elif weighting is SCALED:
            value = old_value + (new_value * rate)
            # return value
        elif weighting is TIME_AVERAGED:
            # return (1-rate)*old_value + rate*new_value
            value = (1-rate)*old_value + rate*new_value
        else:
            # return new_value
            value = new_value

        self.oldValue = value
        return value


# region DDM
#
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


class BogaczEtAl(IntegratorFunction): # --------------------------------------------------------------------------------
    """Compute analytic solution to DDM process and return mean response time and accuracy.

    Description:
        generates mean response time (RT) and mean error rate (ER) as described in:
            Bogacz, R., Brown, E., Moehlis, J., Holmes, P., & Cohen, J. D. (2006). The physics of optimal
            decision making: a formal analysis of models of performance in two-alternative forced-choice
            tasks.  Psychological review, 113(4), 700.

    Initialization arguments:
        variable (float): set to self.value (== self.inputValue)
        - params (dict):  runtime_params passed from Mechanism, used as one-time value for current execution:
            + drift_rate (DRIFT_RATE: float)
            + threshold (THRESHOLD: float)
            + bias (kwDDM_Bias: float)
            + noise (NOISE: float)
            + t0 (NON_DECISION_TIME: float)
        - time_scale (TimeScale): specifies "temporal granularity" with which mechanism is executed
        - context (str)

        Returns the following values in self.value (2D np.array) and in
            the value of the corresponding outputState in the self.outputStates dict:
            - decision variable (float)
            - mean error rate (float)
            - mean RT (float)
            - correct mean RT (float) - Navarro and Fuss only
            - correct mean ER (float) - Navarro and Fuss only
    """

    componentName = kwBogaczEtAl

    variableClassDefault = [[0]]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
                 drift_rate:parameter_spec=1.0,
                 starting_point:parameter_spec=0.0,
                 threshold:parameter_spec=1.0,
                 noise:parameter_spec=0.5,
                 t0:parameter_spec=.200,
                 params=None,
                 prefs:is_pref_set=None,
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
                         prefs=prefs,
                         context=context)

    def function(self,
                 variable=NotImplemented,
                 params=NotImplemented,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """DDM function

        :var variable: (list)
        :parameter params: (dict) with entries specifying:
                        drift_rate...
        """

        self._check_args(variable=variable, params=params, context=context)

# FIX: USE self.driftRate ETC ONCE ParamsDict Implementation is done:
        drift_rate = float(self.paramsCurrent[DRIFT_RATE])
        threshold = float(self.paramsCurrent[THRESHOLD])
        starting_point = float(self.paramsCurrent[STARTING_POINT])
        noise = float(self.paramsCurrent[NOISE])
        t0 = float(self.paramsCurrent[NON_DECISION_TIME])

        bias = (starting_point + threshold) / (2 * threshold)
        # Prevents div by 0 issue below:
        if bias <= 0:
            bias = 1e-8
        if bias >= 1:
            bias = 1-1e-8

        # drift_rate close to or at 0 (avoid float comparison)
        if abs(drift_rate) < 1e-8:
            # back to absolute bias in order to apply limit
            bias_abs = bias * 2 * threshold - threshold
            # use expression for limit a->0 from Srivastava et al. 2016
            rt = t0 + (threshold**2 - bias_abs**2)/(noise**2)
            er = (threshold - bias_abs)/(2*threshold)
        else:
            drift_rate_normed = abs(drift_rate)
            ztilde = threshold/drift_rate_normed
            atilde = (drift_rate_normed/noise)**2

            is_neg_drift = drift_rate<0
            bias_adj = (is_neg_drift==1)*(1 - bias) + (is_neg_drift==0)*bias
            y0tilde = ((noise**2)/2) * np.log(bias_adj / (1 - bias_adj))
            if abs(y0tilde) > threshold:    y0tilde = -1*(is_neg_drift==1)*threshold + (is_neg_drift==0)*threshold
            x0tilde = y0tilde/drift_rate_normed

            import warnings
            warnings.filterwarnings('error')

            try:
                rt = ztilde * tanh(ztilde * atilde) + \
                     ((2*ztilde*(1-exp(-2*x0tilde*atilde)))/(exp(2*ztilde*atilde)-exp(-2*ztilde*atilde))-x0tilde) + t0
                er = 1/(1+exp(2*ztilde*atilde)) - ((1-exp(-2*x0tilde*atilde))/(exp(2*ztilde*atilde)-exp(-2*ztilde*atilde)))

            except (Warning):
                # Per Mike Shvartsman:
                # If ±2*ztilde*atilde (~ 2*z*a/(c^2) gets very large, the diffusion vanishes relative to drift
                # and the problem is near-deterministic. Without diffusion, error rate goes to 0 or 1
                # depending on the sign of the drift, and so decision time goes to a point mass on z/a – x0, and
                # generates a "RuntimeWarning: overflow encountered in exp"
                er = 0
                rt = ztilde/atilde - x0tilde + t0

            # This last line makes it report back in terms of a fixed reference point
            #    (i.e., closer to 1 always means higher p(upper boundary))
            # If you comment this out it will report errors in the reference frame of the drift rate
            #    (i.e., reports p(upper) if drift is positive, and p(lower if drift is negative)
            er = (is_neg_drift==1)*(1 - er) + (is_neg_drift==0)*(er)

        return rt, er


# Results from Navarro and Fuss DDM solution (indices for return value tuple)
class NF_Results(AutoNumber):
    MEAN_ER = ()
    MEAN_DT = ()
    PLACEMARKER = ()
    MEAN_CORRECT_RT = ()
    MEAN_CORRECT_VARIANCE = ()
    MEAN_CORRECT_SKEW_RT = ()

class NavarroAndFuss(IntegratorFunction): # --------------------------------------------------------------------------------
    """Compute analytic solution to distribution of DDM responses (mean and variance of response time and accuracy).

    Description:
        generates distributions of response time (RT) and error rate (ER) as described in:
            Navarro, D. J., and Fuss, I. G. "Fast and accurate calculations for first-passage times in
            Wiener diffusion models." Journal of Mathematical Psychology 53.4 (2009): 222-230.

    Initialization arguments:
        variable (float): set to self.value (== self.inputValue)
        - params (dict):  runtime_params passed from Mechanism, used as one-time value for current execution:
            + drift_rate (DRIFT_RATE: float)
            + threshold (THRESHOLD: float)
            + bias (kwDDM_Bias: float)
            + noise (NOISE: float)
            + t0 (NON_DECISION_TIME: float)
        - time_scale (TimeScale): specifies "temporal granularity" with which mechanism is executed
        - context (str)

        Returns the following values in self.value (2D np.array) and in
            the value of the corresponding outputState in the self.outputStates dict:
            - decision variable (float)
            - mean error rate (float)
            - mean RT (float)
            - correct mean RT (float) - Navarro and Fuss only
            - correct mean ER (float) - Navarro and Fuss only
    """

    componentName = kwNavarrosAndFuss

    variableClassDefault = [[0]]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
                 drift_rate:parameter_spec=1.0,
                 starting_point:parameter_spec=0.0,
                 threshold:parameter_spec=1.0,
                 noise:parameter_spec=0.5,
                 t0:parameter_spec=.200,
                 params=None,
                 prefs:is_pref_set=None,
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
                         prefs=prefs,
                         context=context)

    def _instantiate_function(self, context=None):

        print("\nimporting matlab...")
        import matlab.engine
        self.eng1 = matlab.engine.start_matlab('-nojvm')
        print("matlab imported\n")

        super()._instantiate_function(context=context)

    def function(self,
                 variable=NotImplemented,
                 params=NotImplemented,
                 time_scale=TimeScale.TRIAL,
                 context=None):
        """DDM function

        :var variable: (list)
        :parameter params: (dict) with entries specifying:
                        drift_rate...
        """

        self._check_args(variable=variable, params=params, context=context)

# FIX: USE self.driftRate ETC ONCE ParamsDict Implementation is done:
        drift_rate = float(self.paramsCurrent[DRIFT_RATE])
        threshold = float(self.paramsCurrent[THRESHOLD])
        starting_point = float(self.paramsCurrent[STARTING_POINT])
        noise = float(self.paramsCurrent[NOISE])
        t0 = float(self.paramsCurrent[NON_DECISION_TIME])

        # print("\nimporting matlab...")
        # import matlab.engine
        # eng1 = matlab.engine.start_matlab('-nojvm')
        # print("matlab imported\n")
        results = self.eng1.ddmSim(drift_rate, starting_point, threshold, noise, t0, 1, nargout=5)

        return results




#region ************************************   DISTRIBUTION FUNCTIONS   ************************************************

# TBI

#region **************************************   LEARNING FUNCTIONS ****************************************************


class LearningFunction(Function_Base):
    componentType = kwLearningFunction


LEARNING_RATE = "learning_rate"
ACTIVATION_FUNCTION = 'activation_function'
MATRIX_INPUT = 0
ACTIVATION_OUTPUT = 1
ACTIVATION_ERROR = 2


class Reinforcement(LearningFunction): # -------------------------------------------------------------------------------
    """Calculate matrix of weight changes using the reinforcement (delta) learning rule

    Reinforcement learning rule
      [matrix]         [scalar]        [col array]
    delta_weight =  learning rate   *     error
      return     =  LEARNING_RATE  *  self.variable

    Reinforcement.function:
        variable must be a 1D np.array of error terms
        assumes matrix to which errors are applied is the identity matrix
            (i.e., set of "parallel" weights from input to output)
        LEARNING_RATE param must be a float
        returns matrix of weight changes

    Initialization arguments:
     - variable (list or np.array): must a single 1D np.array
     - params (dict): specifies
         + LEARNING_RATE: (float) - learning rate (default: 1.0)
    """

    componentName = kwRL

    variableClassDefault = [[0],[0],[0]]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    def __init__(self,
                 variable_default=variableClassDefault,
                 activation_function:tc.any(SoftMax, tc.enum(SoftMax))=SoftMax, # Allow class or instance
                 learning_rate:parameter_spec=1,
                 params=None,
                 prefs:is_pref_set=None,
                 context='Component Init'):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(activation_function=activation_function,
                                                 learning_rate=learning_rate,
                                                 params=params)

        super().__init__(variable_default=variable_default,
                         params=params,
                         prefs=prefs,
                         context=context)

        self.functionOutputType = None


    def _validate_variable(self, variable, context=None):
        super()._validate_variable(variable, context)

        if len(self.variable) != 3:
            raise ComponentError("Variable for {} ({}) must have three items (input, output and error arrays)".
                                format(self.name, self.variable))

        # FIX: GETS CALLED BY _check_args W/O KWINIT IN CONTEXT
        if not INITIALIZING in context:
            if np.count_nonzero(self.variable[ACTIVATION_OUTPUT]) != 1:
                raise ComponentError("First item ({}) of variable for {} must be an array with a single non-zero value "
                                    "(if output mechanism being trained uses softmax,"
                                    " its output arg may need to be set to to PROB)".
                                    format(self.variable[ACTIVATION_OUTPUT], self.componentName))
            if len(self.variable[ACTIVATION_ERROR]) != 1:
                raise ComponentError("Error term ({}) for {} must be an array with a single element or a scalar value "
                                    "(variable of Comparator mechanism may need to be specified as an array of length 1)".
                                    format(self.name, self.variable[ACTIVATION_ERROR]))


    def function(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale=TimeScale.TRIAL,
                context=None):
        """Calculate a matrix of weight changes from a single (scalar) error term

        Assume output array has a single non-zero value chosen by the softmax function of the error_source
        Assume error is a single scalar value
        Assume weight matrix (for Mapping projection to error_source) is a diagonal matrix
            (one weight for corresponding pairs of elements in the input and output arrays)
        Adjust the weight corresponding to the chosen element of the output array, using error value and learning rate

        Note: assume variable is a 2D np.array with three items (input, output, error)
              for compatibility with other learning functions (and calls from LearningSignal)

        :var variable: 2D np.array with three items (input array, output array, error array)
        :parameter params: (dict) with entry specifying:
                           LEARNING_RATE: (float) - (default: 1)
        :return matrix:
        """

        self._check_args(variable=variable, params=params, context=context)

        output = self.variable[ACTIVATION_OUTPUT]
        error = self.variable[ACTIVATION_ERROR]
        learning_rate = self.paramsCurrent[LEARNING_RATE]

        # Assign error term to chosen item of output array
        error_array = (np.where(output, learning_rate * error, 0))

        # Construct weight change matrix with error term in proper element
        weight_change_matrix = np.diag(error_array)

        return weight_change_matrix


class BackPropagation(LearningFunction): # ---------------------------------------------------------------------------------
    """Calculate matrix of weight changes using the backpropagation (Generalized Delta Rule) learning algorithm

    Backpropagation learning algorithm (Generalized Delta Rule):
      [matrix]         [scalar]       [row array]              [row array/ col array]                 [col array]
    delta_weight =  learning rate   *    input      *            d(output)/d(input)                 *     error
      return     =  LEARNING_RATE  *  variable[0]  *  kwTransferFctDeriv(variable[1],variable[0])  *  variable[2]

    BackPropagation.function:
        variable must be a list or np.array with three items:
            - input (e.g, array of activities of sender units)
            - output (array of activities of receiver units)
            - error (array of errors for receiver units)
        LEARNING_RATE param must be a float
        kwTransferFunctionDerivative param must be a function reference for dReceiver/dSender
        returns matrix of weight changes

    Initialization arguments:
     - variable (list or np.array): must have three 1D elements
     - params (dict): specifies
         + LEARNING_RATE: (float) - learning rate (default: 1.0)
         + kwTransferFunctionDerivative - (function) derivative of transfer function (default: derivative of logistic)
    """

    componentName = kwBackProp

    variableClassDefault = [[0],[0],[0]]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
                 activation_function:tc.any(Logistic, tc.enum(Logistic))=Logistic, # Allow class or instance
                 learning_rate:parameter_spec=1,
                 params=None,
                 prefs:is_pref_set=None,
                 context='Component Init'):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(activation_function=activation_function,
                                                 learning_rate=learning_rate,
                                                 params=params)

        super().__init__(variable_default=variable_default,
                         params=params,
                         prefs=prefs,
                         context=context)

        self.functionOutputType = None


    def _validate_variable(self, variable, context=None):
        super()._validate_variable(variable, context)

        if len(self.variable) != 3:
            raise ComponentError("Variable for {} ({}) must have three items (input, output and error arrays)".
                                format(self.name, self.variable))
        if len(self.variable[ACTIVATION_ERROR]) != len(self.variable[ACTIVATION_OUTPUT]):
            raise ComponentError("Length of error term ({}) for {} must match length of the output array ({})".
                                format(self.variable[ACTIVATION_ERROR], self.name, self.variable[ACTIVATION_OUTPUT]))


    def _instantiate_function(self, context=None):
        """Get derivative of activation function being used
        """
        self.derivativeFunction = self.paramsCurrent[ACTIVATION_FUNCTION].derivative
        super()._instantiate_function(context=context)

    def function(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale=TimeScale.TRIAL,
                context=None):
        """Calculate and return a matrix of weight changes from an array of inputs, outputs and error terms

        :var variable: (list or np.array) len = 3 (input, output, error)
        :parameter params: (dict) with entries specifying:
                           LEARNING_RATE: (float) - (default: 1)
                           kwTransferFunctionDerivative (function) - derivative of function that generated values
                                                                     (default: derivative of logistic function)
        :return number:
        """

        self._check_args(variable, params, context)

        input = np.array(self.variable[MATRIX_INPUT]).reshape(len(self.variable[MATRIX_INPUT]),1)  # make input a 1D row array
        output = np.array(self.variable[ACTIVATION_OUTPUT]).reshape(1,len(self.variable[ACTIVATION_OUTPUT])) # make output a 1D column array
        error = np.array(self.variable[ACTIVATION_ERROR]).reshape(1,len(self.variable[ACTIVATION_ERROR]))  # make error a 1D column array
        learning_rate = self.paramsCurrent[LEARNING_RATE]
        derivative = self.derivativeFunction(input=input, output=output)

        weight_change_matrix = learning_rate * input * derivative * error

        return weight_change_matrix

# *****************************************   OBJECTIVE FUNCTIONS ******************************************************

# TBI

#  *****************************************   REGISTER FUNCTIONS   ****************************************************
