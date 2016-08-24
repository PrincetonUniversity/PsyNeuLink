# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ***********************************************  Utility *************************************************************
#

__all__ = ['LinearCombination',
           'Linear',
           'Exponential',
           'Logistic',
           'Integrator',
           'LinearMatrix',
           'BackPropagation',
           'UtilityError',
           "UtilityFunctionOutputType"]

from functools import reduce
from operator import *
from random import randint

import numpy as np

from PsyNeuLink.Functions.ShellClasses import *
from PsyNeuLink.Globals.Registry import register_category

UtilityRegistry = {}


class UtilityError(Exception):
     def __init__(self, error_value):
         self.error_value = error_value

     def __str__(self):
         return repr(self.error_value)


class UtilityFunctionOutputType(IntEnum):
    RAW_NUMBER = 0
    NP_1D_ARRAY = 1
    NP_2D_ARRAY = 2


# class Utility_Base(Function):
class Utility_Base(Utility):
    """Implement abstract class for Utility category of Function class

    Description:
        Utility functions are ones used by other function categories;
        They are defined here (on top of standard libraries) to provide a uniform interface for managing parameters
         (including defaults)
        NOTE:   the Utility category definition serves primarily as a shell, and an interface to the Function class,
                   to maintain consistency of structure with the other function categories;
                it also insures implementation of .function for all Utility Functions
                (as distinct from other Function subclasses, which can use a kwFunction param
                    to implement .function instead of doing so directly)
                Utility Functions are the end of the recursive line: as such, they don't implement functionParams

    Instantiation:
        A utility function can be instantiated in one of several ways:
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
            UtilityFunctionOutputType.RAW_NUMBER, return value is "exposed" as a number
            UtilityFunctionOutputType.NP_1D_ARRAY, return value is 1d np.array
            UtilityFunctionOutputType.NP_2D_ARRAY, return value is 2d np.array
        - it must be enabled for a subclass by setting params[kwFunctionOutputTypeConversion] = True
        - it must be implemented in the execute method of the subclass
        - see Linear for an example

    MechanismRegistry:
        All Utility functions are registered in UtilityRegistry, which maintains a dict for each subclass,
          a count for all instances of that type, and a dictionary of those instances

    Naming:
        Utility functions are named by their functionName attribute (usually = functionType)

    Class attributes:
        + functionCategory: kwUtilityeFunctionCategory
        + className (str): kwMechanismFunctionCategory
        + suffix (str): " <className>"
        + registry (dict): UtilityRegistry
        + classPreference (PreferenceSet): UtilityPreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.CATEGORY
        + paramClassDefaults (dict): {kwFunctionOutputTypeConversion: False}

    Class methods:
        none

    Instance attributes:
        + functionType (str):  assigned by subclasses
        + functionName (str):   assigned by subclasses
        + variable (value) - used as input to function's execute method
        + paramInstanceDefaults (dict) - defaults for instance (created and validated in Functions init)
        + paramsCurrent (dict) - set currently in effect
        + value (value) - output of execute method
        + name (str) - if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet) - if not specified as an arg, default is created by copying UtilityPreferenceSet

    Instance methods:
        The following method MUST be overridden by an implementation in the subclass:
        - execute(variable, params)
        The following can be implemented, to customize validation of the function variable and/or params:
        - [validate_variable(variable)]
        - [validate_params(request_set, target_set, context)]
    """

    functionCategory = kwUtilityFunctionCategory
    className = functionCategory
    suffix = " " + className

    registry = UtilityRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY

    variableClassDefault = None
    variableClassDefault_locked = False

    # Note: the following enforce encoding as 1D np.ndarrays (one array per variable)
    variableEncodingDim = 1

    paramClassDefaults = Function.paramClassDefaults.copy()
    paramClassDefaults.update({kwFunctionOutputTypeConversion: False}) # Enable/disable output type conversion

    def __init__(self,
                 variable_default,
                 params,
                 name=NotImplemented,
                 prefs=NotImplemented,
                 context=NotImplemented):
        """Assign category-level preferences, register category, and call super.__init__

        Initialization arguments:
        - variable_default (anything): establishes type for the variable, used for validation
        - params_default (dict): assigned as paramInstanceDefaults
        Note: if parameter_validation is off, validation is suppressed (for efficiency) (Function class default = on)

        :param variable_default: (anything but a dict) - value to assign as variableInstanceDefault
        :param params: (dict) - params to be assigned to paramInstanceDefaults
        :param log: (FunctionLog enum) - log entry types set in self.functionLog
        :param name: (string) - optional, overrides assignment of default (functionName of subclass)
        :return:
        """
        self._functionOutputType = None
        self.name = self.functionName

        register_category(self, Utility_Base, UtilityRegistry, context=context)

        super(Utility_Base, self).__init__(variable_default=variable_default,
                                           param_defaults=params,
                                           name=name,
                                           prefs=prefs,
                                           context=context)

    @property
    def functionOutputType(self):
        if self.paramsCurrent[kwFunctionOutputTypeConversion]:
            return self._functionOutputType
        return None

    @functionOutputType.setter
    def functionOutputType(self, value):

        # Attempt to set outputType but conversion not enabled
        if value and not self.paramsCurrent[kwFunctionOutputTypeConversion]:
            raise UtilityError("output conversion is not enabled for {0}".format(self.__class__.__name__))

        # Bad outputType specification
        if value and not isinstance(value, UtilityFunctionOutputType):
            raise UtilityError("value ({0}) of functionOutputType attribute must be UtilityFunctionOutputType for {1}".
                               format(self.functionOutputType, self.__class__.__name__))

        # Can't convert from arrays of length > 1 to number
        # if (self.variable.size  > 1 and (self.functionOutputType is UtilityFunctionOutputType.RAW_NUMBER)):
        if (len(self.variable)  > 1 and (self.functionOutputType is UtilityFunctionOutputType.RAW_NUMBER)):
            raise UtilityError("{0} can't be set to return a single number since its variable has more than one number".
                               format(self.__class__.__name__))
        self._functionOutputType = value


# *****************************************   EXAMPLE FUNCTION   *******************************************************


class Contradiction(Utility_Base): # Example
    """Example function for use as template for function construction

    Iniialization arguments:
     - variable (boolean or statement resolving to boolean)
     - params (dict) specifying the:
         + propensity (kwPropensity: a mode specifying the manner of responses (tendency to agree or disagree)
         + pertinacity (kwPertinacity: the consistency with which the manner complies with the propensity

    Contradiction.function returns True or False
    """

    # Function functionName and type (defined at top of module)
    functionName = kwContradiction
    functionType = kwExampleFunction

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
    paramClassDefaults = Utility_Base.paramClassDefaults.copy()
    paramClassDefaults.update({kwPropensity: Manner.CONTRARIAN,
                          kwPertinacity:  10,
                          })

    def __init__(self,
                 variable_default=variableClassDefault,
                 params=None,
                 prefs=NotImplemented,
                 context=NotImplemented):
        # This validates variable and/or params_list if assigned (using validate_params method below),
        #    and assigns them to paramsCurrent and paramInstanceDefaults;
        #    otherwise, assigns paramClassDefaults to paramsCurrent and paramInstanceDefaults
        # NOTES:
        #    * paramsCurrent can be changed by including params in call to function
        #    * paramInstanceDefaults can be changed by calling assign_default
        super(Contradiction, self).__init__(variable_default=variable_default,
                                            params=params,
                                            prefs=prefs,
                                            context=context)

    def execute(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale=TimeScale.TRIAL,
                context=NotImplemented):
        """Returns a boolean that is (or tends to be) the same as or opposite the one passed in

        Returns True or False, that is either the same or opposite the statement passed in as the variable
        The propensity parameter must be set to be Manner.OBSEQUIOUS or Manner.CONTRARIAN, which
            determines whether the response is (or tends to be) the same as or opposite to the statement
        The pertinacity parameter determines the consistency with which the response conforms to the manner

        :param variable: (boolean) Statement to probe
        :param params: (dict) with entires specifying
                       kwPropensity: Contradiction.Manner - contrarian or obsequious (default: CONTRARIAN)
                       kwPertinacity: float - obstinate or equivocal (default: 10)
        :return response: (boolean)
        """
        self.check_args(variable, params, context)

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
            raise UtilityError("This should not happen if parameter_validation == True;  check its value")

    def validate_variable(self, variable, context=NotImplemented):
        """Validates variable and assigns validated values to self.variable

        This overrides the class method, to perform more detailed type checking
        See explanation in class method.
        Note:  this method (or the class version) is called only if the parameter_validation attribute is True

        :param variable: (anything but a dict) - variable to be validated:
        :param context: (str)
        :return none:
        """

        if type(variable) == type(self.variableClassDefault) or \
                (isinstance(variable, numbers.Number) and  isinstance(self.variableClassDefault, numbers.Number)):
            self.variable = variable
        else:
            raise UtilityError("Variable must be {0}".format(type(self.variableClassDefault)))

    def validate_params(self, request_set, target_set=NotImplemented, context=NotImplemented):
        """Validates variable and /or params and assigns to targets

        This overrides the class method, to perform more detailed type checking
        See explanation in class method.
        Note:  this method (or the class version) is called only if the parameter_validation attribute is True

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
            raise UtilityError(message)

        super(Contradiction, self).validate_params(request_set, target_set, context)


# *****************************************   UTILITY FUNCTIONS   ******************************************************

#  COMBINATION FUNCTIONS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  LinearCombination
#  [Polynomial -- TBI]

kwLinearCombinationInitializer = "Initializer"

class LinearCombination(Utility_Base): # ------------------------------------------------------------------------------------------
# FIX: CONFIRM THAT 1D KWEIGHTS USES EACH ELEMENT TO SCALE CORRESPONDING VECTOR IN VARIABLE
# FIX  CONFIRM THAT LINEAR TRANSFORMATION (kwOffset, kwScale) APPLY TO THE RESULTING ARRAY
# FIX: CONFIRM RETURNS LIST IF GIVEN LIST, AND SIMLARLY FOR NP.ARRAY
    """Linearly combine arrays of values with optional weighting, offset, and/or scaling

    Description:
        Combine corresponding elements of arrays in variable arg, using arithmetic operation determined by kwOperation
        Use optional kwWeighiting argument to weight contribution of each array to the combination
        Use optional kwScale and kwOffset parameters to linearly transform the resulting array
        Returns a list or 1D array of the same length as the individual ones in the variable

        Notes:
        * If variable contains only a single array, it is simply linearly transformed using kwScale and kwOffset
        * If there is more than one array in variable, they must all be of the same length
        * kwWeights can be:
            - 1D: each array in the variable is scaled by the corresponding element of kwWeights)
            - 2D: each array in the variable is multipled by (Hadamard Product) the corresponding array in kwWeight

    Initialization arguments:
     - variable (value, np.ndarray or list): values to be combined;
         can be a list of lists, or a 1D or 2D np.array;  a 1D np.array is always returned
         if it is a list, it must be a list of numbers, lists, or np.arrays
         all items in the list or 2D np.array must be of equal length
         the length of kwWeights (if provided) must equal the number of arrays (2nd dimension; default is 2)
     - params (dict) can include:
         + kwWeights (list of numbers or 1D np.array): multiplies each variable before combining them (default: [1, 1])
         + kwOffset (value): added to the result (after the arithmetic operation is applied; default is 0)
         + kwScale (value): multiples the result (after combining elements; default: 1)
         + kwOperation (Operation Enum) - method used to combine terms (default: SUM)
              SUM: element-wise sum of the arrays in variable
              PRODUCT: Hadamard Product of the arrays in variable

    LinearCombination.execute returns combined values:
    - single number if variable was a single number
    - list of numbers if variable was list of numbers
    - 1D np.array if variable was a single np.variable or np.ndarray
    """

    functionName = kwLinearCombination
    functionType = kwCombinationFunction

    # Operation indicators
    class Operation(Enum):
        SUM = 0
        PRODUCT = 1

    variableClassDefault = [2, 2]
    # variableClassDefault_locked = True

    paramClassDefaults = Utility_Base.paramClassDefaults.copy()
    # paramClassDefaults.update({kwExponents: NotImplemented,
    #                            kwWeights: NotImplemented,
    #                            kwOffset: 0,
    #                            kwScale: 1,
    #                            kwOperation: Operation.SUM})

    def __init__(self,
                 variable_default=variableClassDefault,
                 scale=1.0,
                 offset=0.0,
                 exponents=NotImplemented,
                 weights=NotImplemented,
                 operation=Operation.SUM,
                 params=None,
                 prefs=NotImplemented,
                 context=NotImplemented):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self.assign_args_to_param_dicts(scale=scale,
                                                 offset=offset,
                                                 exponents=exponents,
                                                 weights=weights,
                                                 operation=operation,
                                                 params=params)

        super(LinearCombination, self).__init__(variable_default=variable_default,
                                                params=params,
                                                prefs=prefs,
                                                context=context)

# MODIFIED 6/12/16 NEW:
    def validate_variable(self, variable, context=NotImplemented):
        """Insure that all items of list or np.ndarray in variable are of the same length

        Args:
            variable:
            context:
        """
        super(Utility_Base, self).validate_variable(variable=variable,
                                                    context=context)
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
                    raise UtilityError("Length of all arrays in variable {0} for {1} must be the same".
                                       format(variable, self.__class__.__name__))

    def validate_params(self, request_set, target_set=NotImplemented, context=NotImplemented):
        """Insure that kwExponents and kwWeights are lists or np.arrays of numbers with length equal to variable

        Args:
            request_set:
            target_set:
            context:

        Returns:

        """
        super(Utility_Base, self).validate_params(request_set=request_set,
                                                  target_set=target_set,
                                                  context=context)

        exponents = target_set[kwExponents]
        weights = target_set[kwWeights]
        operation = target_set[kwOperation]

        # Make sure exponents is a list of numbers or an np.ndarray
# FIX: CHANGE THIS AND WEIGHTS TO TRY/EXCEPT
        if not exponents is None and not exponents is NotImplemented:
            if ((isinstance(exponents, list) and all(isinstance(elem, numbers.Number) for elem in exponents)) or
                    isinstance(exponents, np.ndarray)):
                # convert to 2D np.ndarrray (to distribute over 2D self.variable array)
                target_set[kwExponents] = np.atleast_2d(target_set[kwExponents]).reshape(-1,1)
            else:
                raise UtilityError("kwExponents param ({0}) for {1} must be a list of numbers or an np.array".
                               format(exponents, self.name))

        # Make sure weights is a list of numbers or an np.ndarray
        if not weights is None and not weights is NotImplemented:
            if ((isinstance(weights, list) and all(isinstance(elem, numbers.Number) for elem in weights)) or
                    isinstance(weights, np.ndarray)):
                # convert to 2D np.ndarrray (to distribute over 2D self.variable array)
                target_set[kwWeights] = np.atleast_2d(target_set[kwWeights]).reshape(-1,1)
            else:
                raise UtilityError("kwWeights param ({0}) for {1} must be a list of numbers or an np.array".
                               format(weights, self.name))

        if not operation:
            raise UtilityError("Operation param missing")
        if not operation == self.Operation.SUM and not operation == self.Operation.PRODUCT:
            raise UtilityError("Operation param ({0}) must be Operation.SUM or Operation.PRODUCT".format(operation))
# MODIFIED 6/12/16 END

    def execute(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale=TimeScale.TRIAL,
                context=NotImplemented):
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
        Corresponding elements of each item in variable are combined based on kwOperation param:
            - SUM adds corresponding elements
            - PRODUCT multiples corresponding elements
        An initializer (kwLinearCombinationInitializer) can be provided as the first item in variable;
            it will be populated with a number of elements equal to the second item,
            each element of which is determined by kwOperation param:
            - for SUM, initializer will be a list of 0's
            - for PRODUCT, initializer will be a list of 1's
        Returns a list of the same length as the items in variable,
            each of which is the combination of their corresponding elements specified by kwOperation

        :var variable: (list of numbers) - values to calculate (default: [0, 0]:
        :params: (dict) with entries specifying:
                           kwExponents (2D np.array): exponentiate each value in the variable array (default: none)
                           kwWeights (2D np.array): multiply each value in the variable array (default: none):
                           kwOffset (scalar) - additive constant (default: 0):
                           kwScale: (scalar) - scaling factor (default: 1)
                           kwOperation: LinearCombination.Operation - operation to perform (default: SUM):
        :return: (1D np.array)
        """

        # Validate variable and assign to self.variable, and validate params
        self.check_args(variable=variable, params=params, context=context)

        exponents = self.paramsCurrent[kwExponents]
        weights = self.paramsCurrent[kwWeights]
        operation = self.paramsCurrent[kwOperation]
        offset = self.paramsCurrent[kwOffset]
        scale = self.paramsCurrent[kwScale]

        # IMPLEMENTATION NOTE: CONFIRM: SHOULD NEVER OCCUR, AS validate_variable NOW ENFORCES 2D np.ndarray
        # If variable is 0D or 1D:
        if np_array_less_than_2d(self.variable):
            return (self.variable * scale) + offset


# FIX: CHANGE THIS AND WEIGHTS TO TRY/EXCEPT // OR IS IT EVEN NECESSARY, GIVEN VALIDATION ABOVE??

        # Apply exponents if they were specified
        if not exponents is None and not exponents is NotImplemented:
            if len(exponents) != len(self.variable):
                raise UtilityError("Number of exponents ({0}) does not equal number of items in variable ({1})".
                                   format(len(exponents), len(self.variable.shape)))
            else:
                self.variable = self.variable ** exponents


        # Apply weights if they were specified
        if not weights is None and not weights is NotImplemented:
            if len(weights) != len(self.variable):
                raise UtilityError("Number of weights ({0}) is not equal to number of items in variable ({1})".
                                   format(len(weights), len(self.variable.shape)))
            else:
                self.variable = self.variable * weights

        # Calculate using relevant aggregation operation and return
        if (operation is self.Operation.SUM):
            result = sum(self.variable) * scale + offset
        elif operation is self.Operation.PRODUCT:
            result = reduce(mul, self.variable, 1)
        else:
            raise UtilityError("Unrecognized operator ({0}) for LinearCombination function".
                               format(self.paramsCurrent[kwOperation].self.Operation.SUM))
# FIX: CONFIRM THAT RETURNS LIST IF GIVEN A LIST
        return result

# Polynomial param indices
# TBI

# class Polynomial(Utility_Base): # ------------------------------------------------------------------------------------------
#     pass


#  TRANSFER FUNCTIONS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#  Linear
#  Exponential
#  Logistic
#  Integrator

class Linear(Utility_Base): # ----------------------------------------------------------------------------------------------
    """Calculate a linear transform of input variable (kwSlope, kwIntercept)

    Initialization arguments:
     - variable (number): transformed by linear function: slope * variable + intercept
     - params (dict): specifies
         + slope (kwSlope: value) - slope (default: 1)
         + intercept (kwIntercept: value) - intercept (defaul: 0)

    Linear.execute returns scalar result
    """

    functionName = kwLinear
    functionType = kwTransferFuncton

    # Params
    kwSlope = "slope"
    kwIntercept = "intercept"

    variableClassDefault = [0]

    paramClassDefaults = Utility_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
                               # kwSlope: 1,
                               # kwIntercept: 0,
                               kwFunctionOutputTypeConversion: True})

    def __init__(self,
                 variable_default=variableClassDefault,
                 slope=1,
                 intercept=0,
                 params=None,
                 prefs=NotImplemented,
                 context=NotImplemented):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self.assign_args_to_param_dicts(slope=slope,
                                                 intercept=intercept,
                                                 params=params)

        super(Linear, self).__init__(variable_default=variable_default,
                                     params=params,
                                     prefs=prefs,
                                     context=context)

        self.functionOutputType = None

    def execute(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale=TimeScale.TRIAL,
                context=NotImplemented):
        """Calculate single value (defined by slope and intercept)

        :var variable: (number) - value to be "plotted" (default: 0
        :parameter params: (dict) with entries specifying:
                           kwSlope: number - slope (default: 1)
                           kwIntercept: number - intercept (default: 0)
        :return number:
        """

        self.check_args(variable, params, context)

        slope = self.paramsCurrent[self.kwSlope]
        intercept = self.paramsCurrent[self.kwIntercept]
        outputType = self.functionOutputType

        # By default, result should be returned as np.ndarray with same dimensionality as input
        result = self.variable * slope + intercept

        #region Type conversion (specified by outputType):
        # Convert to 2D array, irrespective of variable type:
        if outputType is UtilityFunctionOutputType.NP_2D_ARRAY:
            result = np.atleast2d(result)

        # Convert to 1D array, irrespective of variable type:
        # Note: if 2D array (or higher) has more than two items in the outer dimension, generate exception
        elif outputType is UtilityFunctionOutputType.NP_1D_ARRAY:
            # If variable is 2D
            if self.variable.ndim == 2:
                # If there is only one item:
                if len(self.variable) == 1:
                    result = result[0]
                else:
                    raise UtilityError("Can't convert result ({0}: 2D np.ndarray object with more than one array)"
                                       " to 1D array".format(result))
            elif len(self.variable) == 1:
                result = result
            elif len(self.variable) == 0:
                result = np.atleast_1d(result)
            else:
                raise UtilityError("Can't convert result ({0} to 1D array".format(result))

        # Convert to raw number, irrespective of variable type:
        # Note: if 2D or 1D array has more than two items, generate exception
        elif outputType is UtilityFunctionOutputType.RAW_NUMBER:
            # If variable is 2D
            if self.variable.ndim == 2:
                # If there is only one item:
                if len(self.variable) == 1 and len(self.variable[0]) == 1:
                    result = result[0][0]
                else:
                    raise UtilityError("Can't convert result ({0}) with more than a single number to a raw number".
                                       format(result))
            elif len(self.variable) == 1:
                if len(self.variable) == 1:
                    result = result[0]
                else:
                    raise UtilityError("Can't convert result ({0}) with more than a single number to a raw number".
                                       format(result))
            else:
                return result
        #endregion

        return result

class Exponential(Utility_Base): # -------------------------------------------------------------------------------------
    """Calculate an exponential transform of input variable  (kwRate, kwScale)

    Initialization arguments:
     - variable (number):
         + scalar value to be transformed by exponential function: scale * e**(rate * x)
     - params (dict): specifies
         + rate (kwRate: coeffiencent on variable in exponent (default: 1)
         + scale (kwScale: coefficient on exponential (default: 1)

    Exponential.execute returns scalar result
    """

    functionName = kwExponential
    functionType = kwTransferFuncton

    # Params
    kwRate = "rate"
    kwScale = "scale"

    variableClassDefault = 0

    paramClassDefaults = Utility_Base.paramClassDefaults.copy()
    # paramClassDefaults.update({kwRate: 1,
    #                       kwScale: 1
    #                       })

    def __init__(self,
                 variable_default=variableClassDefault,
                 rate=1.0,
                 scale=1.0,
                 params=None,
                 prefs=NotImplemented,
                 context=NotImplemented):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self.assign_args_to_param_dicts(rate=rate,
                                                 scale=scale,
                                                 params=params)

        super(Exponential, self).__init__(variable_default=variable_default,
                                          params=params,
                                          prefs=prefs,
                                          context=context)
        TEST = True

    def execute(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale=TimeScale.TRIAL,
                context=NotImplemented):
        """Exponential function

        :var variable: (number) - value to be exponentiated (default: 0
        :parameter params: (dict) with entries specifying:
                           kwRate: number - rate (default: 1)
                           kwScale: number - scale (default: 1)
        :return number:
        """

        self.check_args(variable, params, context)

        # Assign the params and return the result
        rate = self.paramsCurrent[self.kwRate]
        scale = self.paramsCurrent[self.kwScale]

        return scale * np.exp(rate * self.variable)

class Logistic(Utility_Base): # -------------------------------------------------------------------------------------
    """Calculate the logistic transform of input variable  (kwGain, kwBias)

    Initialization arguments:
     - variable (number):
         + scalar value to be transformed by logistic function: 1 / (1 + e**(gain*variable + bias))
     - params (dict): specifies
         + gain (kwGain: coeffiencent on exponent (default: 1)
         + bias (kwBias: additive constant in exponent (default: 0)

    Logistic.execute returns scalar result
    """

    functionName = kwExponential
    functionType = kwTransferFuncton

    # Params
    kwGain = "gain"
    kwBias = "bias"

    variableClassDefault = 0

    paramClassDefaults = Utility_Base.paramClassDefaults.copy()
    # paramClassDefaults.update({kwGain: 1,
    #                       kwBias: 1
    #                       })

    def __init__(self,
                 variable_default=variableClassDefault,
                 gain=1.0,
                 bias=0.0,
                 params=None,
                 prefs=NotImplemented,
                 context=NotImplemented):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self.assign_args_to_param_dicts(gain=gain,
                                                 bias=bias,
                                                 params=params)

        super().__init__(variable_default=variable_default,
                         params=params,
                         prefs=prefs,
                         context=context)

    def execute(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale=TimeScale.TRIAL,
                context=NotImplemented):
        """Logistic sigmoid function

        :var variable: (number) - value to be transformed by logistic function (default: 0)
        :parameter params: (dict) with entries specifying:
                           kwGain: number - gain (default: 1)
                           kwBias: number - rate (default: 0)
        :return number:
        """

        self.check_args(variable, params, context)

        # Assign the params and return the result
        gain = self.paramsCurrent[self.kwGain]
        bias = self.paramsCurrent[self.kwBias]

        return 1 / (1 + np.exp(-(gain * self.variable) + bias))

    def derivative(self, output, input=None):
        """Derivative of the logistic signmoid function
        """
        return output*(np.ones_like(output)-output)

class Integrator(Utility_Base): # --------------------------------------------------------------------------------------
    """Calculate an accumulated and/or time-averaged value for input variable using a specified accumulation method

    Initialization arguments:
     - variable: new input value, to be combined with old value at rate and using method specified by params
     - params (dict): specifying:
         + kwInitializer (value): initial value to which to set self.oldValue (default: variableClassDefault)
             - must be same type and format as variable
             - can be specified as a runtime parameter, which resets oldValue to one specified
             Note: self.oldValue stores previous value with which new value is integrated
         + kwScale (value): rate of accumuluation based on weighting of new vs. old value (default: 1)
         + kwWeighting (Weightings Enum): method of accumulation (default: LINEAR):
                LINEAR -- returns old_value incremented by rate parameter (simple accumulator)
                SCALED -- returns old_value incremented by rate * new_value
                TIME_AVERAGED -- returns rate-weighted average of old and new values  (Delta rule, Wiener filter)
                                rate = 0:  no change (returns old_value)
                                rate 1:    instantaneous change (returns new_value)

    Class attributes:
    - oldValue (value): stores previous value with which value provided in variable is integrated

    Integrator.execute returns scalar result
    """

    class Weightings(AutoNumber):
    # class Weightings(IntEnum):
        LINEAR        = ()
        SCALED        = ()
        TIME_AVERAGED = ()

    functionName = kwIntegrator
    functionType = kwTransferFuncton

    # Params:
    kwRate = "rate"
    kwWeighting = "weighting"

    variableClassDefault = [[0]]

    paramClassDefaults = Utility_Base.paramClassDefaults.copy()
    paramClassDefaults.update({kwInitializer: variableClassDefault})

    def __init__(self,
                 variable_default=variableClassDefault,
                 rate=1.0,
                 weighting=Weightings.LINEAR,
                 params=None,
                 prefs=NotImplemented,
                 context=NotImplemented):

        # Assign here as default, for use in initialization of function
        self.oldValue = self.paramClassDefaults[kwInitializer]

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self.assign_args_to_param_dicts(rate=rate,
                                                 weighting=weighting,
                                                 params=params)

        super(Integrator, self).__init__(variable_default=variable_default,
                                         params=params,
                                         prefs=prefs,
                                         context=context)

        # Reassign to kWInitializer in case default value was overridden
        self.oldValue = self.paramsCurrent[kwInitializer]

    def validate_params(self, request_set, target_set=NotImplemented, context=NotImplemented):
        super(Utility_Base, self).validate_params(request_set=request_set,
                                                  target_set=target_set,
                                                  context=context)
        try:
            if not iscompatible(target_set[kwInitializer],self.variableClassDefault):
                raise UtilityError("kwInitializer param {0} for {1} must be same type as variable {2}".
                                   format(target_set[kwInitializer],
                                          self.__class__.__name__,
                                          self.variable))
        except KeyError:
            pass

    # def execute(self, old_value, new_value, param_list=NotImplemented):

    def execute(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale=TimeScale.TRIAL,
                context=NotImplemented):
        """Integrator function

        :var variable: (list) - old_value and new_value (default: [0, 0]:
        :parameter params: (dict) with entries specifying:
                        kwRate: number - rate of accumulation as relative weighting of new vs. old value  (default = 1)
                        kwWeighting: Integrator.Weightings - type of weighting (default = Weightings.LINEAR)
        :return number:
        """

# FIX:  CONVERT TO NP?

# FIX:  NEED TO CONVERT OLD_VALUE TO NP ARRAY

        self.check_args(variable, params, context)

        rate = float(self.paramsCurrent[self.kwRate])
        weighting = self.paramsCurrent[self.kwWeighting]

        try:
            old_value = params[kwInitializer]
        except (TypeError, KeyError):
            old_value = self.oldValue

        old_value = np.atleast_2d(old_value)

        new_value = self.variable

        # Compute function based on weighting param
        if weighting is self.Weightings.LINEAR:
            value = old_value + rate
            # return value
        elif weighting is self.Weightings.SCALED:
            value = old_value + (new_value * rate)
            # return value
        elif weighting is self.Weightings.TIME_AVERAGED:
            # return (1-rate)*old_value + rate*new_value
            value = (1-rate)*old_value + rate*new_value
        else:
            # return new_value
            value = new_value

        self.oldValue = value
        return value

class LinearMatrix(Utility_Base):  # -----------------------------------------------------------------------------------
    """Map sender vector to receiver vector using a linear weight matrix  (kwReceiver, kwMatrix)

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
         + identity (kwkwIdentityMapping: boolean): constructs identity matrix (default: False)

    Create a matrix in self.matrix that is used in calls to LinearMatrix.execute.

    Returns sender 2D array linearly transformed by self.matrix
    """

    functionName = kwLinearMatrix
    functionType = kwTransferFuncton

    DEFAULT_FILLER_VALUE = 0

    VALUE = 'Value'
    VECTOR = 'Vector'

    variableClassDefault = [DEFAULT_FILLER_VALUE]  # Sender vector

    paramClassDefaults = Utility_Base.paramClassDefaults.copy()
    # paramClassDefaults.update({kwMatrix: NotImplemented})


    def __init__(self,
                 variable_default=variableClassDefault,
                 matrix=NotImplemented,
                 params=None,
                 prefs=NotImplemented,
                 context=NotImplemented):
        """Transforms variable (sender vector) using matrix specified by params, and returns receiver vector

        Variable = sender vector (list of numbers)

        :param variable_default: (list) - list of numbers (default: [0]
        :param params: (dict) with entries specifying:
                                kwReceiver: value - list of numbers, determines width (cols) of matrix (defalut: [0])
                                kwMatrix: value - value used to initialize matrix;  can be one of the following:
                                    + single number - used to fill self.matrix (default: DEFAULT_FILLER_VALUE)
                                    + matrix - assigned to self.matrix
                                    + kwIdentity - create identity matrix (diagonal elements = 1;  all others = 0)
        :return none
        """

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self.assign_args_to_param_dicts(matrix=matrix,
                                                 params=params)

        # Note: this calls validate_variable and validate_params which are overridden below;
        #       the latter implements the matrix if required
        super(LinearMatrix, self).__init__(variable_default=variable_default,
                                           params=params,
                                           prefs=prefs,
                                           context=context)

        self.matrix = self.implement_matrix(self.paramsCurrent[kwMatrix])

    def validate_variable(self, variable, context=NotImplemented):
        """Insure that variable passed to LinearMatrix is a 1D np.array

        :param variable: (1D np.array)
        :param context:
        :return:
        """
        super(Utility_Base, self).validate_variable(variable, context)

        # Check that self.variable == 1D
        try:
            is_not_1D = not self.variable.ndim is 1

        except AttributeError:
            raise UtilityError("PROGRAM ERROR: variable ({0}) for {1} should be an np.ndarray".
                               format(self.variable, self.__class__.__name__))
        else:
            if is_not_1D:
                raise UtilityError("variable ({0}) for {1} must be a 1D np.ndarray".
                                   format(self.variable, self.__class__.__name__))

    def validate_params(self, request_set, target_set=NotImplemented, context=NotImplemented):
        """Validate params and assign to targets

        This overrides the class method, to perform more detailed type checking (see explanation in class method).
        Note:  this method (or the class version) is called only if the parameter_validation attribute is True

        :param request_set: (dict) - params to be validated
        :param target_set: (dict) - destination of validated params
        :param context: (str)
        :return none:
        """

        super(LinearMatrix, self).validate_params(request_set, target_set, context)
        param_set = target_set
        sender = self.variable
        # Note: this assumes self.variable is a 1D np.array, as enforced by validate_variable
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
                raise UtilityError("receiver param ({0}) for {1} must be a list of numbers or an np.array".
                                   format(self.receiver, self.name))
        # No receiver, so use sender as template (assuming square -- e.g., identity -- matrix)
        else:
            if self.prefs.verbosePref:
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
            elif param_name == kwMatrix:

                # A number (to be used as a filler), so OK
                if isinstance(param_value, numbers.Number):
                    continue

                # Full connectivity matrix requested (using keyword)
                elif param_value is kwFullConnectivityMatrix:
                    continue

                # Identity matrix requested (using keyword), so check send_len == receiver_len
                elif param_value is kwIdentityMatrix:
                    # Receiver length doesn't equal sender length
                    if not (self.receiver.shape == sender.shape and self.receiver.size == sender.size):
                        # if self.prefs.verbosePref:
                        #     print ("Identity matrix requested, but length of receiver ({0})"
                        #            " does not match length of sender ({1});  sender length will be used".
                        #            format(receiver_len, sender_len))
                        # # Set receiver to sender
                        # param_set[kwReceiver] = sender
                        raise UtilityError("Identity matrix requested, but length of receiver ({0})"
                                           " does not match length of sender ({1})".format(receiver_len, sender_len))
                    continue

                # list used to describe matrix, so convert to 2D np.array and pass to validation of matrix below
                elif isinstance(param_value, list):
                    try:
                        param_value = np.atleast_2d(param_value)
                    except (ValueError, TypeError) as error_msg:
                        raise UtilityError("Error in list specification ({0}) of matrix for {1}: {2})".
                                           format(param_value, self.__class__.__name__, error_msg))

                # string used to describe matrix, so convert to np.matrix and pass to validation of matrix below
                elif isinstance(param_value, str):
                    try:
                        param_value = np.matrix(param_value)
                    except (ValueError, TypeError) as error_msg:
                        raise UtilityError("Error in string specification ({0}) of matrix for {1}: {2})".
                                           format(param_value, self.__class__.__name__, error_msg))

                # np.matrix or np.ndarray provided, so validate that it is numeric and check dimensions
                if isinstance(param_value, (np.ndarray, np.matrix)):
                    # get dimensions specified by:
                    #   variable (sender): width/cols/outer index
                    #   kwReceiver param: height/rows/inner index

                    weight_matrix = np.matrix(param_value)
                    if 'U' in repr(weight_matrix.dtype):
                        raise UtilityError("Non-numeric entry in kwMatrix specification ({0})".format(param_value))

                    matrix_rows = weight_matrix.shape[0]
                    matrix_cols = weight_matrix.shape[1]

                    # Check that number of rows equals length of sender vector (variable)
                    if matrix_rows != sender_len:
                        raise UtilityError("The number of rows ({0}) of the matrix provided does not equal the "
                                            "length ({1}) of the sender vector (variable)".
                                            format(matrix_rows, sender_len))
                    # Check that number of columns equals length of specified receiver vector (kwReciever)
                    if matrix_cols != receiver_len:
                        raise UtilityError("The number of columns ({0}) of the matrix provided does not equal the "
                                            "length ({1}) of the reciever vector (kwReceiver param)".
                                            format(matrix_cols, receiver_len))

                # function so:
                # - assume it uses random.rand()
                # - call with two args as place markers for cols and rows
                # -  validate that it returns an np.array or np.matrix
                if isinstance(param_value, function_type):
                    test = param_value(1,1)
                    if not isinstance(test, (np.ndarray, np.matrix)):
                        raise UtilityError("A function is specified for matrix for {1}: {2}) "
                                           "that returns a value ({}) that is neither a matrix nor array ".
                               format(param_value, self.__class__.__name__, test))

                else:
                    raise UtilityError("Value of {0} param ({1}) must be a matrix, a number (for filler), "
                                       "or a matrix keyword ({2}, {3})".
                                        format(param_name, param_value, kwIdentityMatrix, kwFullConnectivityMatrix))
            else:
                message += "Param {0} not recognized by {1} function".format(param_name, self.functionName)
                continue

        if message:
            raise UtilityError(message)


    def instantiate_attributes_before_execute_method(self, context=NotImplemented):
        self.matrix = self.implement_matrix()

    def implement_matrix(self, specification=NotImplemented, context=NotImplemented):
        """Implements matrix indicated by specification

         Specification is derived from kwMatrix param (passed to self.__init__ or self.execute)

         Specification (validated in validate_params):
            + single number (used to fill self.matrix)
            + kwIdentity (create identity matrix: diagonal elements = 1,  all others = 0)
            + 2D matrix of numbers (list or np.ndarray of numbers)

        :return matrix: (2D list)
        """

        if specification is NotImplemented:
            specification = kwIdentityMatrix

        # Matrix provided (and validated in validate_params); convert to np.array
        if isinstance(specification, np.matrix):
            return np.array(specification)

        sender = self.variable
        sender_len = sender.shape[0]
        try:
            receiver = self.receiver
        except:
            raise UtilityError("No receiver specified for {0};  will set length equal to sender ({1})".
                               format(self.__class__.__name__, sender_len))
            receiver = sender
        receiver_len = receiver.shape[0]

        # Filler specified so use that
        if isinstance(specification, numbers.Number):
            return np.matrix([[specification for n in range(receiver_len)] for n in range(sender_len)])

        # Full connectivity matrix specified
        if specification == kwFullConnectivityMatrix:
            return np.full((sender_len, receiver_len),1.0)

        # Identity matrix specified
        if specification == kwIdentityMatrix:
            if sender_len != receiver_len:
                raise UtilityError("Sender length ({0}) must equal receiver length ({1}) to use identity matrix".
                                     format(sender_len, receiver_len))
            return np.identity(sender_len)

        # Function is specified, so assume it uses random.rand() and call with sender_len and receiver_len
        if isinstance(specification, function_type):
            return specification(sender_len, receiver_len)

        # This should never happen (should have been picked up in validate_param)
        raise UtilityError("kwMatrix param ({0}) must be a matrix, a function that returns one, "
                           "a matrix specification keyword, or a number (filler)".
                            format(specification))


    def execute(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale=TimeScale.TRIAL,
                context=NotImplemented):
        """Transforms variable vector using either self.matrix or specification in params

        :var variable: (list) - vector of numbers with length equal of height (number of rows, inner index) of matrix
        :parameter params: (dict) with entries specifying:
                            kwMatrix: value - used to override self.matrix implemented by __init__;  must be one of:
                                                 + 2D matrix - two-item list, each of which is a list of numbers with
                                                              length that matches the length of the vector in variable
                                                 + kwIdentity - specifies use of identity matrix (dimensions of vector)
                                                 + number - used to fill matrix of same dimensions as self.matrix
        :return list of numbers: vector with length = width (number of columns, outer index) of matrix
        """

        # Note: this calls validate_variable and validate_params which are overridden above;
        self.check_args(variable, params, context=context)

        return np.dot(self.variable, self.matrix)

    def keyword(keyword):
        if keyword is kwIdentityMatrix:
            return np.identity(1)
        if keyword is kwFullConnectivityMatrix:
            return np.full((1, 1),1.0)
        else:
            raise UtilityError("Unrecognized keyword ({}) specified for LinearMatrix Utility Function".format(keyword))

def enddummy():
    pass

# *****************************************   DISTRIBUTION FUNCTIONS   *************************************************

# TBI

# *****************************************   LEARNING FUNCTIONS *******************************************************

kwLearningRate = "learning_rate"
kwActivationFunction = 'activation_function'

class BackPropagation(Utility_Base): # ---------------------------------------------------------------------------------
    """Calculate matrix of weight changes using the backpropagation (Generalized Delta Rule) learning algorithm

    Backpropagation learning algorithm (Generalized Delta Rule):
      [matrix]         [scalar]       [row array]              [row array/ col array]                 [col array]
    delta_weight =  learning rate   *    input      *            d(output)/d(input)                 *     error
      return     =  kwLearningRate  *  variable[0]  *  kwTransferFctDeriv(variable[1],variable[0])  *  variable[2]

    BackPropagation.execute:
        variable must be a list or np.array with three items:
            - input (e.g, array of activities of sender units)
            - output (array of activities of receiver units)
            - error (array of errors for receiver units)
        kwLearningRate param must be a float
        kwTransferFunctionDerivative param must be a function reference for dReceiver/dSender
        returns matrix of weight changes

    Initialization arguments:
     - variable (list or np.array): must have three 1D elements
     - params (dict): specifies
         + kwLearningRate: (float) - learning rate (default: 1.0)
         + kwTransferFunctionDerivative - (function) derivative of transfer function (default: derivative of logistic)
    """

    functionName = kwBackProp
    functionType = kwLearningFunction

    variableClassDefault = [[0],[0],[0]]

    paramClassDefaults = Utility_Base.paramClassDefaults.copy()

    def __init__(self,
                 variable_default=variableClassDefault,
                 learning_rate=1,
                 activation_function=Logistic(),
                 params=None,
                 prefs=NotImplemented,
                 context=NotImplemented):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self.assign_args_to_param_dicts(learning_rate=learning_rate,
                                                 activation_function=activation_function,
                                                 params=params)

        super().__init__(variable_default=variable_default,
                         params=params,
                         prefs=prefs,
                         context=context)

        self.functionOutputType = None


    def validate_variable(self, variable, context=NotImplemented):
        super().validate_variable(variable, context)

        if not len(self.variable) == 3:
            raise FunctionError("Variable for BackProp ({}) must have three items".format(self.variable))

    def instantiate_execute_method(self, context=NotImplemented):
        """Get derivative of activation function being used
        """
        self.derivativeFunction = self.paramsCurrent[kwActivationFunction].derivative
        super().instantiate_execute_method(context=context)

    def execute(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale=TimeScale.TRIAL,
                context=NotImplemented):
        """Calculate a matrix of weight changes for an array inputs, outputs and error terms

        :var variable: (list or np.array) len = 3 (input, output, error)
        :parameter params: (dict) with entries specifying:
                           kwLearningRate: (float) - (default: 1)
                           kwTransferFunctionDerivative (function) - derivative of function that generated values
                                                                     (default: derivative of logistic function)
        :return number:
        """

        self.check_args(variable, params, context)

        input = np.array(self.variable[0]).reshape(len(self.variable[0]),1)  # makine input as 1D row array
        output = np.array(self.variable[1]).reshape(1,len(self.variable[1])) # make output a 1D column array
        error = np.array(self.variable[2]).reshape(1,len(self.variable[2]))  # make error a 1D column array
        learning_rate = self.paramsCurrent[kwLearningRate]
        derivative = self.derivativeFunction(input=input, output=output)

        return learning_rate * input * derivative * error

# *****************************************   OBJECTIVE FUNCTIONS ******************************************************

# TBI

#  *****************************************   REGISTER FUNCTIONS   ****************************************************
