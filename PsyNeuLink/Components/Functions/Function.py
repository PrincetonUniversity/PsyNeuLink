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
  * :class:`Contradiction`

Combination Components:
  * :class:`LinearCombination`

TransferMechanism Components:
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


class Function_Base(Function):
    """Implement abstract class for Function category of Component class

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
    componentName = CONTRADICTION_FUNCTION
    componentType = EXAMPLE_FUNCTION_TYPE

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
        super().__init__(variable_default=variable_default,
                         params=params,
                         prefs=prefs,
                         context=context)

    def function(self,
                variable=None,
                params=None,
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

        super()._validate_params(request_set, target_set, context)


#region ****************************************   FUNCTIONS   *********************************************************
#endregion

#region **********************************  USER-DEFINED FUNCTION  *****************************************************
#endregion

class UserDefinedFunction(Function_Base):
    """Implement user-defined function

    Initialization arguments:
     - variable

    Linear.function returns scalar result
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
                 prefs:is_pref_set=None,
                 context=componentName+INITIALIZING):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(params=params)
        self.user_defined_function = function

        super().__init__(variable_default=variable,
                         params=params,
                         prefs=prefs,
                         context=context)

        self.functionOutputType = None

        # IMPLEMENT: PARSE ARGUMENTS FOR user_defined_function AND ASSIGN TO user_params

    def function(self,
                 # variable=None,
                 # params=None,
                 # time_scale=TimeScale.TRIAL,
                 # context=None,
                 **kwargs):
        # raise FunctionError("Function must be provided for {}".format(self.componentType))
        return self.user_defined_function(**kwargs)


#region **********************************  COMBINATION FUNCTIONS  *****************************************************
#endregion


class CombinationFunction(Function_Base):
    componentType = COMBINATION_FUNCTION_TYPE


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
    componentName = REDUCE_FUNCTION

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
        if not is_numeric(variable):
            raise FunctionError("All elements of {} must be scalar values".
                                format(self.__class__.__name__))


    def function(self,
                variable=None,
                params=None,
                time_scale=TimeScale.TRIAL,
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
                 scale:parameter_spec=1.0,
                 offset:parameter_spec=0.0,
                 # IMPLEMENTATION NOTE - these don't check whether every element of np.array is numerical:
                 # weights:tc.optional(tc.any(int, float, tc.list_of(tc.any(int, float)), np.ndarray))=None,
                 # exponents:tc.optional(tc.any(int, float, tc.list_of(tc.any(int, float)), np.ndarray))=None,
                 # MODIFIED 2/10/17 OLD: [CAUSING CRASHING FOR SOME REASON]
                 # # weights:is_numeric_or_none=None,
                 # # exponents:is_numeric_or_none=None,
                 # MODIFIED 2/10/17 NEW:
                 weights=None,
                 exponents=None,
                 # MODIFIED 2/10/17 END
                 operation:tc.enum(SUM, PRODUCT, DIFFERENCE, QUOTIENT)=SUM,
                 params=None,
                 prefs:is_pref_set=None,
                 context=componentName+INITIALIZING):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(scale=scale,
                                                 offset=offset,
                                                 weights=weights,
                                                 exponents=exponents,
                                                 operation=operation,
                                                 params=params)

        super().__init__(variable_default=variable_default,
                         params=params,
                         prefs=prefs,
                         context=context)

        if self.weights is not None:
            self.weights = np.atleast_2d(self.weights).reshape(-1,1)
        if self.exponents is not None:
            self.exponents = np.atleast_2d(self.exponents).reshape(-1,1)

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
            target_set[WEIGHTS] = np.atleast_2d(target_set[WEIGHTS]).reshape(-1,1)
        if target_set[EXPONENTS] is not None:
            target_set[EXPONENTS] = np.atleast_2d(target_set[EXPONENTS]).reshape(-1,1)

        # if not operation:
        #     raise FunctionError("Operation param missing")
        # if not operation == self.Operation.SUM and not operation == self.Operation.PRODUCT:
        #     raise FunctionError("Operation param ({0}) must be Operation.SUM or Operation.PRODUCT".format(operation))

    def function(self,
                variable=None,
                params=None,
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
        if exponents is not None:
            if len(exponents) != len(self.variable):
                raise FunctionError("Number of exponents ({0}) does not equal number of items in variable ({1})".
                                   format(len(exponents), len(self.variable.shape)))
            # Avoid divide by zero warning:
            #    make sure there are no zeros for an element that is assigned a negative exponent
            if INITIALIZING in context and any(not any(i) and j<0 for i,j in zip(self.variable, exponents)):
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

#region ***********************************  TRANSFER FUNCTIONS  ***********************************************
#endregion

class TransferFunction(Function_Base):
    componentType = TRANFER_FUNCTION_TYPE


class Linear(TransferFunction): # --------------------------------------------------------------------------------------
    """Calculate a linear transform of input variable (SLOPE, INTERCEPT)

    Initialization arguments:
     - variable (number): transformed by linear function: slope * variable + intercept
     - params (dict): specifies
         + slope (SLOPE: value) - slope (default: 1)
         + intercept (INTERCEPT: value) - intercept (defaul: 0)

    Linear.function returns scalar result
    """

    componentName = LINEAR_FUNCTION

    # MODIFIED 11/29/16 NEW:
    classPreferences = {
        kwPreferenceSetName: 'LinearClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
        kpRuntimeParamStickyAssignmentPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
    }
    # MODIFIED 11/29/16 END


    variableClassDefault = [0]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
                               kwFunctionOutputTypeConversion: True,
                               PARAMETER_STATE_PARAMS: None
    })

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
                 slope:parameter_spec=1.0,
                 intercept:parameter_spec=0.0,
                 params=None,
                 prefs:is_pref_set=None,
                 context=componentName+INITIALIZING):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(slope=slope,
                                                  intercept=intercept,
                                                  params=params)

        super().__init__(variable_default=variable_default,
                                     params=params,
                                     prefs=prefs,
                                     context=context)

        self.functionOutputType = None

    def function(self,
                variable=None,
                params=None,
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

    componentName = EXPONENTIAL_FUNCTION

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

        super().__init__(variable_default=variable_default,
                                          params=params,
                                          prefs=prefs,
                                          context=context)
        TEST = True

    def function(self,
                variable=None,
                params=None,
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

    componentName = LOGISTIC_FUNCTION
    parameter_keywords.update({GAIN,BIAS})

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
                variable=None,
                params=None,
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

    componentName = SOFTMAX_FUNCTION

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
                variable=None,
                params=None,
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

    COMMENT:
    XXXX CONVERT TO FIGURE:
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
    COMMENT

    Initialization arguments:
    - variable (2D np.ndarray containing exactly two sub-arrays:  sender and receiver vectors
    - params (dict) specifying:
         + filler (kwFillerValue: number) value used to initialize all entries in matrix (default: 0)
         + identity (kwkwIdentityMapping: boolean): constructs identity matrix (default: :keyword:`False`)

    Create a matrix in self.matrix that is used in calls to LinearMatrix.function.

    Returns sender 2D array linearly transformed by self.matrix
    """

    componentName = LINEAR_MATRIX_FUNCTION

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

# FIX: IMPLEMENT AUTO_ASSIGN_MATRIX HERE: PASS, AS SHOULD HAVE BEEN HANDLED BY CALLER (E.G., MAPPING_PROJECTION._instantiate_receiver)
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
                # Note:  assume that these will be properly processed by caller
                #        (e.g., MappingProjection._instantiate_receiver)
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
                variable=None,
                params=None,
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
        from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
        rows = None
        cols = None
        if isinstance(self, MappingProjection):
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
    componentType = INTEGRATOR_FUNCTION_TYPE


class Integrator(IntegratorFunction): # --------------------------------------------------------------------------------------
    """Calculate an accumulated and/or time-averaged value for input variable using a specified accumulation method

    Initialization arguments:
     - variable: new input value, to be combined with old value at rate and using method specified by params
     - params (dict): specifying:
         + kwInitializer (value): initial value to which to set self.oldValue (default: variableClassDefault)
             - must be same type and format as variable
             - can be specified as a runtime parameter, which resets oldValue to one specified
             Note: self.oldValue stores previous value with which new value is integrated
         + RATE (value): rate of accumulation based on weighting of new vs. old value (default: 1)
         + WEIGHTING (Weightings Enum): method of accumulation (default: CONSTANT):
                CONSTANT -- returns old_value incremented by rate parameter (ignores input) with optional noise 
                SIMPLE -- returns old_value incremented by rate * new_value with optional noise
                ADAPTIVE -- returns rate-weighted average of old and new values  (Delta rule, Wiener filter) with optional noise
                                rate = 0:  no change (returns old_value)
                                rate 1:    instantaneous change (returns new_value)
                DIFFUSION -- returns old_value incremented by drift_rate * old_value * time_step_size and the standard DDM noise distribution 

    Class attributes:
    - oldValue (value): stores previous value with which value provided in variable is integrated

    Integrator.function returns scalar result
    """

    componentName = INTEGRATOR_FUNCTION

    variableClassDefault = [[0]]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({kwInitializer: variableClassDefault})

    @tc.typecheck
    def __init__(self,
                 variable_default=None,
                 rate:parameter_spec=1.0,
                 weighting:tc.enum(CONSTANT, SIMPLE, ADAPTIVE, DIFFUSION)=CONSTANT,
                 params:tc.optional(dict)=None,
                 prefs:is_pref_set=None,
                 noise=0.0,
                 time_step_size = 1.0, 
                 context="Integrator Init"):

        # Assign here as default, for use in initialization of function
        self.oldValue = self.paramClassDefaults[kwInitializer]

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                 weighting=weighting,
                                                 params=params,
                                                 noise=noise,
                                                 time_step_size=time_step_size)

        super().__init__(variable_default=variable_default,
                                         params=params,
                                         prefs=prefs,
                                         context=context)

        # Reassign to kWInitializer in case default value was overridden
        self.oldValue = [self.paramsCurrent[kwInitializer]]


        # self.noise = self.paramsCurrent[NOISE]

    def _validate_params(self, request_set, target_set=None, context=None):

        # MODIFIED 11/22/16 NEW:
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
                                        format(len(rate), rate, self.name,np.array(self.variable).size, self.variable))

            self.paramClassDefaults[RATE] = np.zeros_like(np.array(rate))
        # MODIFIED 11/22/16 END

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
            if not iscompatible(target_set[kwInitializer],self.variableClassDefault):
                raise FunctionError("kwInitializer param {0} for {1} must be same type as variable {2}".
                                   format(target_set[kwInitializer],
                                          self.__class__.__name__,
                                          self.variable))
        except KeyError:
            pass


    # def function(self, old_value, new_value, param_list=NotImplemented):

    def function(self,
                variable=None,
                params=None,
                time_scale=TimeScale.TRIAL,
                context=None):
        """Integrator function

        :var variable: (list) - old_value and new_value (default: [0, 0]:
        :parameter params: (dict) with entries specifying:
                        RATE: number - rate of accumulation as relative weighting of new vs. old value  (default = 1)
                        WEIGHTING: Integrator.Weightings - type of weighting (default = CONSTANT)
        :return number:
        """

# FIX:  CONVERT TO NP?
# FIX:  NEED TO CONVERT OLD_VALUE TO NP ARRAY

        self._check_args(variable=variable, params=params, context=context)

        rate = np.array(self.paramsCurrent[RATE]).astype(float)
        weighting = self.paramsCurrent[WEIGHTING]

        time_step_size = self.paramsCurrent[TIME_STEP_SIZE]

        #if noise is a function, execute it 
        if self.noise_function:
            noise = self.noise()
        else:
            noise = self.noise

        try:
            old_value = params[kwInitializer]
        except (TypeError, KeyError):
            old_value = self.oldValue

        old_value = np.atleast_2d(old_value)
        new_value = self.variable

        # Compute function based on weighting param
        if weighting is CONSTANT:
            value = old_value + rate + noise 
        elif weighting is SIMPLE:
            value = old_value + (new_value * rate) + noise 
        elif weighting is ADAPTIVE:
            value = (1-rate)*old_value + rate*new_value + noise 
        elif weighting is DIFFUSION: 
            value = old_value + rate*old_value*time_step_size + np.sqrt(time_step_size*noise)*np.random.normal()
        else:
            value = new_value
 
        # If this NOT an initialization run, update the old value
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if not INITIALIZING in context:
            self.oldValue = value

        return value

class DDMIntegrator(Integrator): # --------------------------------------------------------------------------------------
    """Calculate an accumulated value for input variable using a the DDM accumulation method. The DDMIntegrator only allows for 'DIFFUSION' weighting, and requires the noise parameter to be a float, as it is used to construct the standard DDM Gaussian.  

    Initialization arguments:
     - params (dict): specifying:
         + kwInitializer (value): initial value to which to set self.oldValue (default: variableClassDefault)
             - can be specified as a runtime parameter, which resets oldValue to one specified
             Note: self.oldValue stores previous value
        + drift_rate (DRIFT_RATE: float)
        + noise (NOISE: float)
        + time_step_size (TIME_STEP_SIZE: float)

    Class attributes:
    - oldValue (value): stores previous value with which value provided in variable is integrated

    Integrator.function returns scalar result
    """


    componentName = DDM_INTEGRATOR_FUNCTION

    @tc.typecheck
    def __init__(self,
                 variable_default=None,
                 weighting=DIFFUSION,
                 params:tc.optional(dict)=None,
                 prefs:is_pref_set=None,
                 noise=0.5,
                 rate = 1.0,
                 time_step_size = 1.0,
                 context="DDMIntegrator Init"):

        # Assign here as default, for use in initialization of function
        self.oldValue = self.paramClassDefaults[kwInitializer]

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(
                                                 weighting=weighting,
                                                 params=params,
                                                 noise=noise,
                                                 rate=rate,
                                                 time_step_size=time_step_size)

        super().__init__(variable_default=variable_default,
                                         params=params,
                                         prefs=prefs,
                                         context=context)

        # Reassign to kWInitializer in case default value was overridden
        self.oldValue = [self.paramsCurrent[kwInitializer]]
    def _validate_params(self, request_set, target_set=None, context=None):

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        noise = target_set[NOISE]

        if (isinstance(noise, float) == False):
            raise FunctionError("noise parameter ({}) for {} must be a float.".
                                format(noise, self.name))

        weighting = target_set[WEIGHTING]
        if (weighting != "diffusion"):
            raise FunctionError("weighting parameter ({}) for {} must be diffusion. For alternate methods of accumulation, use the Integrator function".
                                format(weighting, self.name))


    

#     # def keyword(self, keyword):
#     #     return keyword

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
            tasks.  Psychological review, 113(4), 700. (`PubMed entry <https://www.ncbi.nlm.nih.gov/pubmed/17014301>`_)

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
                 variable=None,
                 params=None,
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
                er = 1/(1+exp(2*ztilde*atilde)) - \
                                             ((1-exp(-2*x0tilde*atilde))/(exp(2*ztilde*atilde)-exp(-2*ztilde*atilde)))

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
    RESULT = ()
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
            (`ScienceDirect entry <http://www.sciencedirect.com/science/article/pii/S0022249609000200>`_)

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
                 variable=None,
                 params=None,
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
                 mean = 0.0,
                 standard_dev = 1.0, 
                 params=None,
                 prefs:is_pref_set=None,
                 context=componentName+INITIALIZING):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(mean=mean,
                                                  standard_dev = standard_dev,
                                                  params=params)

        super().__init__(variable_default=variable_default,
                         params=params,
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

        result = standard_dev*np.random.normal() + mean 

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
                 beta = 1.0,  
                 params=None,
                 prefs:is_pref_set=None,
                 context=componentName+INITIALIZING):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(beta = beta, 
                                                  params=params)

        super().__init__(variable_default=variable_default,
                         params=params,
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
                 low = 0.0,
                 high = 1.0, 
                 params=None,
                 prefs:is_pref_set=None,
                 context=componentName+INITIALIZING):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(low=low,
                                                  high=high,
                                                  params=params)

        super().__init__(variable_default=variable_default,
                         params=params,
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

        result = np.random.uniform(low,high)

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
                 scale = 1.0,  
                 shape = 1.0, 
                 params=None,
                 prefs:is_pref_set=None,
                 context=componentName+INITIALIZING):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(scale = scale,
                                                  shape = shape,  
                                                  params=params)

        super().__init__(variable_default=variable_default,
                         params=params,
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
                 scale = 1.0,  
                 mean = 1.0, 
                 params=None,
                 prefs:is_pref_set=None,
                 context=componentName+INITIALIZING):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(scale = scale,
                                                  mean = mean,   
                                                  params=params)

        super().__init__(variable_default=variable_default,
                         params=params,
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

#endregion 

#region **************************************   LEARNING FUNCTIONS ****************************************************


class LearningFunction(Function_Base):
    componentType = LEARNING_FUNCTION_TYPE


LEARNING_ACTIVATION_FUNCTION = 'activation_function'
LEARNING_ACTIVATION_INPUT = 0       # a(j)
# MATRIX = 1             # w
LEARNING_ACTIVATION_OUTPUT = 1  # a(i)
LEARNING_ERROR_OUTPUT = 2

class ErrorDerivative(LearningFunction):
    """Calculate the contribution of each sender to the error signal based on the weight matrix

    Description:

    Initialization arguments:
     - variable (1d np.array or list, 1d np.rarray or list): activity vector and error vector, respectively
     - matrix (2d np.array or List(list))
     - derivative (function or method)
    ErrorDerivative.function returns dot product of matrix and error * derivative of activity
    Returns 1d np.array (error_signal)
    """

    componentName = ERROR_DERIVATIVE_FUNCTION

    classPreferences = {
        kwPreferenceSetName: 'ErrorDerivativeCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
        kpRuntimeParamStickyAssignmentPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
    }

    variableClassDefault = [[0], [0]]
    # variableClassDefault_locked = True

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 variable_default,
                 # matrix=None,
                 # derivative:is_function_type,
                 derivative:tc.optional(tc.any(function_type, method_type))=None,
                 params=None,
                 prefs:is_pref_set=None,
                 context=componentName+INITIALIZING):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(derivative=derivative,
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

        # FIX: MAKE SURE VARIABLE ndim = 2
        variable = np.array(variable)
        # self.variable = np.atleast_2d(variable)

        # variable must have two 1d items
        if variable.shape[0] != 2:
            raise FunctionError("variable for {} ({}) must have two arrays (activity and error vectors)".
                               format(self.__class__.__name__, variable))

    def _validate_params(self, request_set, target_set=None, context=None):
        """Insure that dimensions of matrix are compatible with activity and error vectors in variable
        """
        super()._validate_params(request_set=request_set,
                              target_set=target_set,
                              context=context)

        if not isinstance(target_set['derivative'],(function_type, method_type)):
            raise FunctionError("\'derivative\' arg ({}) for {} must be an ndarray or matrix".
                                format(target_set[DERIVATIVE], self.__class__.__name__))


    def function(self,
                variable=None,
                params=None,
                time_scale=TimeScale.TRIAL,
                context=None):

        self._check_args(variable, params, context)

        activity = self.variable[0]
        error = self.variable[1]

        activity_derivative = self.derivative(input=None, output=activity)
        # FIX:  ??CORRECT:
        #     ??dE/dA         ??E            dA/dW
        error_derivative  =  error  *  activity_derivative

        if ('WeightedError' in self.name):
            print("\n{} ({}): ".format('WeightedError', self.name))
            print("- activity ({}): {}".format(len(activity), activity))
            print("- error ({}): {}".format(len(error), error))
            print("- calculated activity_derivative ({}): {}".format(len(activity_derivative), activity_derivative))
            print("- calculated error_derivative ({}): {}".format(len(error_derivative), error_derivative))

        return error_derivative



class Reinforcement(LearningFunction): # -------------------------------------------------------------------------------
    """Calculate matrix of weight changes using the reinforcement (delta) learning rule

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
    """

    componentName = RL_FUNCTION

    variableClassDefault = [[0],[0],[0]]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    def __init__(self,
                 variable_default=variableClassDefault,
                 activation_function:tc.any(SoftMax, tc.enum(SoftMax))=SoftMax, # Allow class or instance
                 learning_rate:tc.optional(parameter_spec)=1.0,
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
                                    " its output arg may need to be set to to PROB)".
                                    format(self.variable[LEARNING_ACTIVATION_OUTPUT], self.componentName))

    def _validate_params(self, request_set, target_set=None, context=None):

        # This allows callers to specify None as learning_rate (e.g., _instantiate_learning_components)
        request_set[LEARNING_RATE] = request_set[LEARNING_RATE] or 1.0
        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

    def function(self,
                variable=None,
                params=None,
                time_scale=TimeScale.TRIAL,
                context=None):
        """Calculate a matrix of weight changes from a single (scalar) error term

        Assume output array has a single non-zero value chosen by the softmax function of the error_source
        Assume error is a single scalar value
        Assume weight matrix (for MappingProjection to error_source) is a diagonal matrix
            (one weight for corresponding pairs of elements in the input and output arrays)
        Adjust the weight corresponding to the chosen element of the output array, using error value and learning rate

        Note: assume variable is a 2D np.array with three items (input, output, error)
              for compatibility with other learning functions (and calls from LearningProjection)

        :var variable: 2D np.array with three items (input array, output array, error array)
        :parameter params: (dict) with entry specifying:
                           LEARNING_RATE: (float) - (default: 1)
        :return matrix:
        """

        self._check_args(variable=variable, params=params, context=context)

        # input_thing = list(self.activation_input.squeeze())
        activation_input = self.activation_input[0]
        output = self.activation_output
        error = float(self.error_signal.squeeze())
        error_assignment = np.full(activation_input.size, self.learning_rate * error)
        null_assignment = np.zeros_like(activation_input)

        # Assign error term to chosen item of output array
        # error_array = np.atleast_1d(np.where(input, self.learning_rate * error, 0))
        error_array = (np.where(input, error_assignment, null_assignment))

        # Construct weight change matrix with error term in proper element
        weight_change_matrix = np.diag(error_array)

        return [weight_change_matrix, error_array]


# Argument names:
ERROR_MATRIX = 'error_matrix'
WT_MATRIX_SENDERS_DIM = 0
WT_MATRIX_RECEIVERS_DIM = 1

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
         + kwTransferFunctionDerivative - (function) derivative of TransferMechanism function (default: derivative of logistic)
    """

    componentName = BACKPROPAGATION_FUNCTION

    variableClassDefault = [[0],[0],[0]]

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    from PsyNeuLink.Components.States.ParameterState import ParameterState
    from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection

    @tc.typecheck
    def __init__(self,
                 variable_default=variableClassDefault,
                 # variable_default:tc.any(list, np.ndarray),
                 activation_derivative_fct :tc.optional(tc.any(function_type, method_type))=Logistic().derivative,
                 error_derivative_fct:tc.optional(tc.any(function_type, method_type))=Logistic().derivative,
                 # error_matrix:tc.optional(tc.any(list, np.ndarray, np.matrix, ParameterState, MappingProjection))=None,
                 error_matrix=None,
                 learning_rate:tc.optional(parameter_spec)=1.0,
                 params=None,
                 prefs:is_pref_set=None,
                 context='Component Init'):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(activation_derivative_fct =activation_derivative_fct,
                                                  error_derivative_fct=error_derivative_fct,
                                                  error_matrix=error_matrix,
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


        # This allows callers to specify None as learning_rate (e.g., _instantiate_learning_components)
        request_set[LEARNING_RATE] = request_set[LEARNING_RATE] or 1.0

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
        if rows!= activity_output_len:
            raise FunctionError("The height (number of rows, {}) of \'{}\' arg specified for {} must match the "
                                "length of the output {} of the activity vector being monitored ({})".
                                format(rows, MATRIX, self.name, activity_output_len))


    def function(self,
                variable=None,
                params=None,
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

        from PsyNeuLink.Components.States.ParameterState import ParameterState
        self._check_args(variable, params, context)
        if isinstance(self.error_matrix, ParameterState):
            error_matrix = self.error_matrix.value
        else:
            error_matrix = self.error_matrix

        # MODIFIED 3/5/17 OLD:

        # make activation_input a 1D row array
        activation_input = np.array(self.activation_input).reshape(len(self.activation_input),1)

        # Derivative of error with respect to output activity (contribution of each output unit to the error above)
        dE_dA = np.dot(error_matrix, self.error_signal)

        # Derivative of the output activity
        dA_dW  = self.activation_derivative_fct(input=self.activation_input, output=self.activation_output)

        # Chain rule to get the derivative of the error with respect to the weights
        dE_dW = dE_dA * dA_dW

        # Weight changes = delta rule (learning rate * activity * error)
        weight_change_matrix = self.learning_rate * activation_input * dE_dW

        # TEST PRINT:
        if context and not 'INIT' in context:
            print("\nBACKPROP for {}:\n    "
                  "-input: {}\n    "
                  "-error_signal (dE_DA): {}\n    "
                  "-derivative (dA_dW): {}\n    "
                  "-error_derivative (dE_dW): {}\n".
                  format(self.owner.name, self.activation_input, dE_dA, dA_dW ,dE_dW))

        return [weight_change_matrix, dE_dW]

        # # MODIFIED 3/4/17 NEW:  [TEST: REPRODUCE ERROR_CALC IN DEVEL ORIG]:
        # try:
        #     from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism import ObjectiveMechanism
        #     from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanism import LearningMechanism
        #     if self.owner.inputStates:
        #         error_mech = self.owner.inputStates['error_signal'].receivesFromProjections[0].sender.owner
        #         error_mech_name = error_mech.name
        #         if isinstance(error_mech, ObjectiveMechanism):
        #             error_mech_error = error_mech.outputState.value
        #             error_mech_out = error_mech.inputStates['SAMPLE'].value
        #             # TEST PRINT:
        #             print("\nTARGET_ERROR for {}:\n    -error_mech_output: {}\n    -error_mech_error: {}\n".
        #                   format(self.owner.name, error_mech_out, error_mech_error))
        #
        #                             # DEVEL:
        #                             # activity = self.variable[0]  <- error_mech_output
        #                             # error = self.variable[1]     <- error_signal
        #                             # matrix = self.matrix.value   <- error_matrix
        #                             #
        #                             # # MODIFIED 3/4/17 OLD:
        #                             # activity_derivative = self.derivative(output=activity)
        #                             # error_derivative = error * activity_derivative
        #
        #         elif isinstance(error_mech, LearningMechanism):
        #             # error_signal = self.error_signal
        #             # error_mech_act_out = error_mech.inputStates['activation_output'].value
        #             # # activity_derivative = self.activation_derivative_fct(output=error_mech_act_out)
        #             # activity_derivative = self.error_derivative_fct(output=error_mech_act_out)
        #             # error_derivative = error_signal * activity_derivative
        #             # error_mech_error = np.dot(self.error_matrix, error_derivative)
        #             activity = error_mech.inputStates['activation_output'].value
        #             error = self.error_signal
        #             matrix = error_matrix
        #             activity_derivative = self.error_derivative_fct(output=activity)
        #             error_derivative = error * activity_derivative
        #             weighted_error = np.dot(error_matrix, error_derivative)
        #             error_mech_error = weighted_error
        #
        #             # TEST PRINT:
        #             print("\nWEIGHTED_ERROR for {}:\n    "
        #                   "-error_mech_output: {}\n    "
        #                   "-error_mech_error: {}\n    "
        #                   "-error_derivative: {}\n    "
        #                   "-dot_product of error derivative: {}\n    "
        #                   "-error_matrix: {}\n".
        #                   format(self.owner.name,
        #                          activity,
        #                          error,
        #                          error_derivative,
        #                          weighted_error,
        #                          error_matrix))
        #         TEST = True
        # except AttributeError:
        #     error_mech_error = np.dot(error_matrix, self.error_signal)
        #
        # # make activation_input a 1D row array
        # activation_input = np.array(self.activation_input).reshape(len(self.activation_input),1)
        #
        # # Derivative of the output activity
        # derivative = self.activation_derivative_fct(input=self.activation_input, output=self.activation_output)
        #
        # weight_change_matrix = self.learning_rate * activation_input * derivative * error_mech_error
        #
        # if 'INIT' not in context:
        #     print("\nBACKPROP for {}:\n    "
        #           "-input: {}\n    "
        #           "-derivative: {}\n    "
        #           "-error_signal: {}\n    "
        #           "-weight_change_matrix {}: \n".
        #           format(self.owner.name,
        #                  self.activation_input,
        #                  derivative,
        #                  error_mech_error,
        #                  weight_change_matrix))
        #
        # return [weight_change_matrix, error_mech_error]

        # MODIFIED 3/4/17 END



#region *****************************************   OBJECTIVE FUNCTIONS ************************************************
#endregion
# TBI

#region  *****************************************   REGISTER FUNCTIONS ************************************************
