# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ***********************************************  Function ************************************************************
#

"""  FUNCTION MODULE

This module defines the Function abstract class

It also contains:

- arg_name definitions for primary function categories:
    Process
    Mechanism
        types:
            DDM
            [PDP]
    Projection
        types:
            Mapping
            ControlSignal
    [Learning]
        types:
            Vectorial
            Reinforcement
    Utility

- Definitions for a set of standard Utility function types:
    Example
        Contradiction
    Combination:
        LinearCombination
        [Polynomial]
    Transfer:
        Linear
        Exponential
        Integrator
        LinearMatrix
    [Distribution]

"""

from PsyNeuLink.Globals.Main import *
from PsyNeuLink.Globals.Preferences.FunctionPreferenceSet import *

class ResetMode(Enum):
    CURRENT_TO_INSTANCE_DEFAULTS = 0
    INSTANCE_TO_CLASS = 1
    ALL_TO_CLASS_DEFAULTS = 2

# functionSystemDefaultPreferencesDict = FunctionPreferenceSet()

# MODIFIED 8/31/16: ADD FOR PARAMSCURRENT->ATTRIBUTES  START
# Prototype for implementing params as objects rather than dicts
# class Params(object):
#     def __init__(self, **kwargs):
#         for arg in kwargs:
#             self.__setattr__(arg, kwargs[arg])

# Transitional type:
#    for implementing params as attributes that are accessible via current paramsDicts
#    (until params are fully implemented as objects)
from collections import UserDict
class ParamsDict(UserDict):
    def __init__(self, owner, dict=None):
        super().__init__()
        self.owner = owner
        if dict:
            self.update(dict)

    def __getitem__(self, key):

        # # WORKS:
        # return super().__getitem__(key)

        try:
            # Try to retrieve from attribute of owner object
            return getattr(self.owner, key)
        except AttributeError:
            # If the owner has no such attribute, get from params dict entry
            return super().__getitem__(key)
        except:
            TEST = True

    def __setitem__(self, key, item):

        # # WORKS:
        # super().__setitem__(key, item)

        setattr(self.owner, key, item)
        TEST = True
    # # ORIG:
    #     self.data[key] = item
# MODIFIED 8/31/16: ADD FOR PARAMSCURRENT->ATTRIBUTES  END

# Used as templates for requiredParamClassDefaultTypes for FUNCTION:
class Params(object):
    def __init__(self, **kwargs):
        for arg in kwargs:
            self.__setattr__(arg, kwargs[arg])


class dummy_class:
    def dummy_method(self):
        pass
def dummy_function():
    pass
method_type = type(dummy_class().dummy_method)
function_type = type(dummy_function)

class FunctionLog(IntEnum):
    NONE            = 0
    ALL = 0
    DEFAULTS = NONE


class FunctionError(Exception):
     def __init__(self, error_value):
         self.error_value = error_value

     def __str__(self):
         return repr(self.error_value)


# *****************************************   FUNCTION CLASS    ********************************************************


class Function(object):
    """Implement parent class for functions used by Process, Mechanism, State, and Projection class categories

        Every function is associated with:
         - child class functionName
         - type
         - input (self.variable)
         - execute (method): called to execute it;  it in turn calls self.function
         - function (method): carries out object's core computation
             it can be referenced either as self.function, self.params[FUNCTION] or self.paramsCurrent[FUNCTION]
         - function_object (Utility): the object to which function belongs (and that defines it's parameters)
         - output (value: self.value)
         - class and instance variable defaults
         - class and instance param defaults
        The function's execute method (<subclass>.execute is the function's primary method
            (e.g., it is the one called when process, mechanism, state and projections objects are updated);
            the following attributes for or associated with the method are defined for every function object:
                + execute (method) - the execute method itself
                + value (value) - the output of the execute method
            the latter is used for typing and/or templating other variables (e.g., self.variable):
                type checking is generally done using Main.iscompatible(); for iterables (lists, tuples, dicts):
                    if the template (the "reference" arg) has entries (e.g., [1, 2, 3]), comparisons will include length
                    if the template is empty (e.g., [], {}, or ()), length will not be checked
                    if the template has only numbers, then the candidate must as well


        The function itself can be called without any arguments (in which case it uses its instance defaults) or
            one or more variables (as defined by the subclass) followed by an optional params dictionary
        The variable(s) can be a function reference, in which case the function is called to resolve the value;
            however:  it must be "wrapped" as an item in a list, so that it is not called before being passed
                      it must of course return a variable of the type expected for the variable
        The default variableList is a list of default values, one for each of the variables defined in the child class
        The params argument is a dictionary; the key for each entry is the parameter name, associated with its value.
            + function subclasses can define the param FUNCTION:<method or Function class>
        The function can be called with a params argument, which should contain entries for one or more of its params;
            - those values will be assigned to paramsCurrent at run time (overriding previous values in paramsCurrent)
            - if the function is called without a variable and/or params argument, it uses paramInstanceDefaults
        The instance defaults can be assigned at initialization or using the assign_defaults class method;
            - if instance defaults are not assigned on initialization, the corresponding class defaults are assigned
        Parameters can be REQUIRED to be in paramClassDefaults (for which there is no default value to assign)
            - for all classes, by listing the name and type in requiredParamClassDefaultTypes dict of the Function class
            - in subclasses, by inclusion in requiredParamClassDefaultTypes (via copy and update) in class definition
            * NOTE: inclusion in requiredParamClasssDefault simply acts as a template;  it does NOT implement the param
        Each function child class must initialize itself by calling super(childfunctionName).__init__()
            with a default value for its variable, and optionally an instance default paramList.

        A subclass MUST either:
            - implement a <class>.function method OR
            - specify paramClassDefaults[FUNCTION:<Utility Function>];
            - this is checked in Function.instantiate_function()
            - if params[FUNCTION] is NOT specified, it is assigned to self.function (so that it can be referenced)
            - if params[FUNCTION] IS specified, it assigns it's value to self.function (superceding existing value):
                self.function is aliased to it (in Function.instantiate_function):
                    if FUNCTION is found on initialization:
                        if it is a reference to an instantiated function, self.function is pointed to it
                        if it is a class reference to a function:
                            it is instantiated using self.variable and FUNCTION_PARAMS (if they are there too)
                            this works, since validate_params is always called after validate_variable
                            so self.variable can be used to initialize function
                            to the method referenced by paramInstanceDefaults[FUNCTION] (see below)
                    if paramClassDefaults[FUNCTION] is not found, it's value is assigned to self.function
                    if neither paramClassDefaults[FUNCTION] nor self.function is found, an exception is raised
        - self.value is determined for self.execute which calls self.function in Function.instantiate_function

        NOTES:
            * In the current implementation, validation is:
              - top-level only (items in lists, tuples and dictionaries are not checked, nor are nested items)
              - for type only (it is oblivious to content)
              - forgiving (e.g., no distinction is made among numberical types)
            * However, more restrictive validation (e.g., recurisve, range checking, etc.) can be achieved
                by overriding the class validate_variable and validate_params methods

    Class attributes:
        + className
        + suffix - " " + className (used to create subclass and instance names)
        + functionCategory - category of function (i.e., process, mechanism, projection, learning, utility)
        + functionType - type of function within a category (e.g., transfer, distribution, mapping, controlSignal, etc.)
        + requiredParamClassDefaultTypes - dict of param names and types that all subclasses of Function must implement;

    Class methods:
        - validate_variable(variable)
        - validate_params(request_set, target_set, context)
        - assign_defaults(variable, request_set, assign_missing, target_set, default_set=NotImplemented
        - reset_params()
        - check_args(variable, params)
        - assign_args_to_param_dicts(params, param_names, function_param_names)

    Instance attributes:
        + name
        + functionName - name of particular function (linear, exponential, integral, etc.)
        + variableClassDefault (value)
        + variableClassDefault_np_info (ndArrayInfo)
        + variableInstanceDefault (value)
        + variable (value)
        + variable_np_info (ndArrayInfo)
        + function (method)
        + function_object (Utility)
        + paramClassDefaults:
            + FUNCTION
            + FUNCTION_PARAMS
        + paramInstanceDefaults
        + paramsCurrent
        # + parameter_validation
        + user_params
        + recording

    Instance methods:
        + function (implementation is optional; aliased to params[FUNCTION] by default)
    """

    #region CLASS ATTRIBUTES
    className = "FUNCTION"
    suffix = " " + className
# IMPLEMENTATION NOTE:  *** CHECK THAT THIS DOES NOT CAUSE ANY CHANGES AT SUBORDNIATE LEVELS TO PROPOGATE EVERYWHERE
    functionCategory = None
    functionType = None

    initMethod = INIT_FULL_EXECUTE_METHOD

    classPreferenceLevel = PreferenceLevel.SYSTEM
    # Any preferences specified below will override those specified in SystemDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to SYSTEM automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'FunctionCustomClassPreferences',
    #     kp<pref>: <setting>...}

    # Determines whether variableClassDefault can be changed (to match an variable in __init__ method)
    variableClassDefault_locked = False

    # Names and types of params required to be implemented in all subclass paramClassDefaults:
    # Notes:
    # *  entry values here do NOT implement the param; they are simply used as type specs for checking (in __init__)
    # * kwUtilityFunctionCategory (below) is used as placemarker for Function.Utility class; replaced in __init__ below
    #              (can't reference own class directly class block)
    requiredParamClassDefaultTypes = {}
    paramClassDefaults = {}
    #endregion

    def __init__(self,
                 variable_default,
                 param_defaults,
                 name=None,
                 prefs=None,
                 context=None):
        """Assign system-level default preferences, enforce required, validate and instantiate params and execute method

        Initialization arguments:
        - variable_default (anything): establishes type for the variable, used for validation
        - params_default (dict): assigned as paramInstanceDefaults
        Note: if parameter_validation is off, validation is suppressed (for efficiency) (Function class default = on)

        :param variable_default: (anything but a dict) - value to assign as variableInstanceDefault
        :param param_defaults: (dict) - params to be assigned to paramInstanceDefaults
        :param log: (FunctionLog enum) - log entry types set in self.functionLog
        :param name: (string) - optional, overrides assignment of default (functionName of subclass)
        :return:
        """

        # # MODIFIED 8/14/16 NEW:
        # # PROBLEM: variable has different name for different classes;  need to standardize name across classes
        # try:
        #     if self.value is kwDeferredInit:
        #         defer_init = True
        # except AttributeError:
        #     pass
        # else:
        #     if defer_init:
        #         self.init_args = locals().copy()
        #         del self.init_args['self']
        #         # del self.init_args['__class__']
        #         return
        context = context + kwInit + ": " + kwFunctionInit

        # These insure that subclass values are preserved, while allowing them to be referred to below
        self.variableInstanceDefault = NotImplemented
        self.paramClassDefaults = self.paramClassDefaults
        self.paramInstanceDefaults = {}

        self.functionName = self.functionType

        #endregion

        #region ENFORCE REGISRY
        if self.__class__.__bases__[0].__bases__[0].__bases__[0].__name__ is 'ShellClass':
            try:
                self.__class__.__bases__[0].registry
            except AttributeError:
                raise FunctionError("{0} is a category class and so must implement a registry".
                                    format(self.__class__.__bases__[0].__name__))
        #endregion

        #region ASSIGN PREFS

        # If a PreferenceSet was provided, assign to instance
        if isinstance(prefs, PreferenceSet):
            self.prefs = prefs
            # FIX:  CHECK LEVEL HERE??  OR DOES IT NOT MATTER, AS OWNER WILL BE ASSIGNED DYNAMICALLY??
        # Otherwise, if prefs is a specification dict instantiate it, or if it is NotImplemented assign defaults
        else:
            self.prefs = FunctionPreferenceSet(owner=self, prefs=prefs, context=context)
        #endregion

        # MODIFIED 9/11/16 NEW:
        # IMPLEMENTATION NOTE:  This is nice and all, but:
        #                       - property version only works for getter, and for class (can't access instance values)
        #                       - attribute version works for getter, but setter sets the attribute and not the pref
        #                       So, for now, hard coding property setters and getters for each preference (see below)
        # Assign prefs to attributes on object
        # for pref in self.prefs.prefsList:
            # # Generate attribute for each pref that returns value of the pref
            # PROBLEM: MAKING AN ASSIGNMENT TO THE ATTRIBUTE WILL NOT AFFECT THE PREFERENCE, JUST THIS ATTRIBUTE
            # setattr(self,
            #         underscore_to_camelCase(pref),
            #         getattr(getattr(self, 'prefs'), underscore_to_camelCase(pref)))
            # PROBLEM: THIS REQUIRES THAT THE PROPERTY IS PUT ON THE CLASS, WHICH GENERATES UNDESIRABLE BEHAVIORS
            #          ALSO, SETTER WON'T WORK PROPERLY HERE EITHER
            # # IMPLEMENT: WITHOUT SETTER:
            # setattr(type(self),
            #         underscore_to_camelCase(pref),
            #         property(lambda self: getattr(getattr(self, 'prefs'), underscore_to_camelCase(pref))))
            # # IMPLEMENT: WITH SETTER:
            # pref_name = underscore_to_camelCase(pref)
            # setattr(type(self),
            #         pref_name,
            #         property(lambda self: getattr(getattr(self, 'prefs'), pref_name),
            #                  lambda self, value: setattr(getattr(getattr(self, 'prefs'), pref_name),
            #                                              pref_name,
            #                                              value)))
        # MODIFIED 9/11/16 END

        #region ASSIGN LOG
        self.log = Log(owner=self)
        self.recording = False
        #endregion

        #region ENFORCE REQUIRED CLASS DEFAULTS

        # All subclasses must implement variableClassDefault
        # Do this here, as validate_variable might be overridden by subclass
        try:
            if self.variableClassDefault is NotImplemented:
                raise FunctionError("variableClassDefault must be given a value for {0}".format(self.functionName))
        except AttributeError:
            raise FunctionError("variableClassDefault must be defined for {0} or its base class".
                                format(self.functionName))
        #endregion

        #region CHECK FOR REQUIRED PARAMS

        # All subclasses must implement, in their paramClassDefaults, params of types specified in
        #     requiredClassParams (either above or in subclass defintion)
        # Do the check here, as validate_params might be overridden by subclass
        for required_param, type_requirements in self.requiredParamClassDefaultTypes.items():
            # # Replace 'Function' placemarker with class reference:
            # type_requirements = [self.__class__ if item=='Function' else item for item in type_requirements]

            # get type for kwUtilityFunctionCategory specification
            import PsyNeuLink.Functions.Utilities.Utility
            if kwUtilityFunctionCategory in type_requirements:
               type_requirements[type_requirements.index(kwUtilityFunctionCategory)] = \
                   type(PsyNeuLink.Functions.Utilities.Utility.Utility_Base)

            if required_param not in self.paramClassDefaults.keys():
                raise FunctionError("Param {0} must be in paramClassDefaults for {1}".
                                    format(required_param, self.name))

            # If the param does not match any of the types specified for it in type_requirements
            # (either as a subclass or instance of the specified subclass):
            try:
                required_param_value = self.paramClassDefaults[required_param]
                if inspect.isclass(required_param_value):
                    OK = (any(issubclass(required_param_value, type_spec) for type_spec in type_requirements))
                else:
                    OK = (any(isinstance(required_param_value, type_spec) for type_spec in type_requirements))
                if not OK:
                    type_names = format(" or ".join("{!s}".format(type.__name__) for (type) in type_requirements))
                    raise FunctionError("Value ({0}) of param {1} is not appropriate for {2};"
                                        "  requires one of the following types: {3}".
                                        format(required_param_value.__name__, required_param, self.name, type_names))
            except TypeError:
                pass
        #endregion

        #region ASSIGN DEFAULTS
        # Validate the set passed in and assign to paramInstanceDefaults
        # By calling with assign_missing, this also populates any missing params with ones from paramClassDefaults
        self.assign_defaults(variable=variable_default,
                             request_set=param_defaults, # requested set
                             assign_missing=True,        # assign missing params from classPreferences to instanceDefaults
                             target_set=self.paramInstanceDefaults, # destination set to which params are being assigned
                             default_set=self.paramClassDefaults,   # source set from which missing params are assigned
                             context=context
                             )
        #endregion

        #region SET CURRENT VALUES OF VARIABLE AND PARAMS
        self.variable = self.variableInstanceDefault
        self.paramsCurrent = self.paramInstanceDefaults
        #endregion

        #region VALIDATE FUNCTION (self.function and/or self.params[function, FUNCTION_PARAMS])
        self.validate_function(context=context)
        #endregion

        #region INSTANTIATE ATTRIBUTES BEFORE FUNCTION
        # Stub for methods that need to be executed before instantiating function
        #    (e.g., instantiate_sender and instantiate_receiver in Projection)
        self.instantiate_attributes_before_function(context=context)
        #endregion

        #region INSTANTIATE FUNCTION and assign output (by way of self.execute) to self.value
        self.instantiate_function(context=context)
        #endregion

        #region INSTANTIATE ATTRIBUTES AFTER FUNCTION
        # Stub for methods that need to be executed after instantiating function
        #    (e.g., instantiate_outputState in Mechanism)
        self.instantiate_attributes_after_function(context=context)
        #endregion

        # MODIFIED 6/28/16 COMMENTED OUT:
        # #region SET NAME
        # if name is NotImplemented:
        #     self.name = self.functionName + self.suffix
        # else:
        #     self.name = name
#endregion

    def deferred_init(self, context=None):
        """Use in subclasses that require deferred initialization
        """
        if self.value is kwDeferredInit:

            # Flag that object is now being initialized
            # Note: self.value will be resolved to the object's value as part of initialization
            #       (usually in instantiate_function)
            self.value = kwInit

            del self.init_args['self']

            # Delete function since super doesn't take it as an arg;
            #   the value is stored in paramClassDefaults in assign_ags_to_params_dicts,
            #   and will be restored in instantiate_function
            try:
                del self.init_args['function']
            except KeyError:
                pass

            try:
                del self.init_args['__class__']
            except KeyError:
                pass

            # Delete reference to dict created by paramsCurrent -> ParamsDict
            try:
                del self.init_args['__pydevd_ret_val_dict']
            except KeyError:
                pass

            # Complete initialization
            super(self.__class__,self).__init__(**self.init_args)

    def assign_args_to_param_dicts(self, **kwargs):
        """Assign args passed in __init__() to params

        Get args and their corresponding values from call to self.__init__()
        - get default values for all args and assign to class.paramClassDefaults if they have not already been
        - assign arg values to local copy of params dict
        - override those with any values specified in params dict passed as "params" arg
        """

        # Get args in call to __init__ and create access to default values
        sig = inspect.signature(self.__init__)
        default = lambda val : list(sig.parameters.values())[list(sig.parameters.keys()).index(val)].default

        def parse_arg(arg):
            # Resolves the string value of any args that use keywords as their name
            try:
                name = eval(arg)
            except NameError:
                name = arg
            if inspect.isclass(name):
                name = arg
            return name

        # ASSIGN DEFAULTS TO paramClassDefaults
        # Check if defaults have been assigned to paramClassDefaults, and if not do so
        for arg in kwargs:

            arg_name = parse_arg(arg)

            # The params arg (nor anything in it) is never a default
            # if arg is kwParams:
            if arg_name is kwParams:
                continue

            # Check if param exists in paramClassDefaults
            try:
                self.paramClassDefaults[arg]

            # param corresponding to arg is NOT in paramClassDefaults, so add it
            except:
                # If arg is function and it's default is not a class, set it to one
                if arg_name is FUNCTION and not inspect.isclass(default(arg)):
                    # Note: this is for compatibility with current implementation of instantiate_function()
                    # FIX: REFACTOR Function.instantiate_function TO USE COPY OF INSTANTIATED function
                    self.paramClassDefaults[arg] = default(arg).__class__

                    # Get params from instantiated function
                    self.paramClassDefaults[FUNCTION_PARAMS] = default(arg).user_params.copy()

                # Get defaults values for args listed in FUNCTION_PARAMS
                # Note:  is not an arg, but rather used to package args that belong to a non-instantiated function
                elif arg is FUNCTION_PARAMS:
                    self.paramClassDefaults[FUNCTION_PARAMS] = {}
                    for item in kwargs[arg]:
                        self.paramClassDefaults[FUNCTION_PARAMS][item] = default(item)
                else:
                    self.paramClassDefaults[arg] = default(arg)

            # param corresponding to arg IS already in paramClassDefaults, so ignore
            else:
                continue

        # ASSIGN ARG VALUES TO params dicts
        params = {}       # this is for final params that will be returned
        params_arg = {}   # this captures any values specified in a params arg, that are used to override arg values
        ignore_kwFunctionParams = False

        for arg in kwargs:

            # Put any values (presumably in a dict) passed in the "params" arg in params_arg
            if arg is kwParams:
                params_arg = kwargs[arg]
                continue

            arg_name = parse_arg(arg)

            # For function:
            if arg_name is FUNCTION:

                function = kwargs[arg]

                # function arg is a class
                if inspect.isclass(function):
                    params[FUNCTION] = function
                    continue

                # function arg is not a class (presumably an object), so convert it to one
                # Note: this is for compatibility with current implementation of instantiate_function()
                # FIX: REFACTOR Function.instantiate_function TO USE INSTANTIATED function
                else:
                    params[FUNCTION] = function.__class__
                    # Get params from instantiated function
                    # FIX: DOES THIS OVER-WRITE FUNCTION_PARAMS??
                    #      SHOULD IF THEY WERE DERIVED FROM PARAM_CLASS_DEFAULTS;
                    #      BUT SHOULDN'T IF THEY CAME FROM __init__ ARG (I.E., KWARGS)
                    # FIX: GIVE PRECEDENCE TO FUNCTION PARAMS SPECIFIED IN FUNCTION_PARAMS
                    # FIX:     OVER ONES AS ARGS FOR FUNCTION ITSELF
                    # FIX: DOES THE FOLLOWING WORK IN ALL CASES
                    # FIX:    OR DOES TO REINTRODUCE THE OVERWRITE PROBLEM WITH MULTIPLE CONTROL SIGNALS (IN EVC SCRIPT)
                    # FIX: AND, EVEN IF IT DOES, WHAT ABOUT ORDER EFFECTS:
                    # FIX:    CAN IT BE TRUSTED THAT function WILL BE PROCESSED BEFORE FUNCTION_PARAMS,
                    # FIX:     SO THAT FUNCTION_PARAMS WILL ALWAYS COME AFTER AND OVER-RWITE FUNCTION.USER_PARAMS
                    params[FUNCTION_PARAMS] = function.user_params.copy()
                    ignore_kwFunctionParams = True

            elif arg_name is FUNCTION_PARAMS:

                # If function was instantiated object, functionParams came from it, so ignore additional specification
                if ignore_kwFunctionParams:
                    TEST = True
                    if TEST:
                        pass
                    continue
                params[FUNCTION_PARAMS] = kwargs[arg]

            # For standard params, assign arg and its default value to paramClassDefaults
            else:
                params[arg] = kwargs[arg]

        # Override arg values with any specified in params dict (including FUNCTION_PARAMS)
        if params_arg:
            try:
                params[FUNCTION_PARAMS].update(params_arg[FUNCTION_PARAMS])
            except KeyError:
                pass
            params.update(params_arg)

        # Save user-accessible params
        self.user_params = params.copy()

        self.create_attributes_for_user_params(**self.user_params)

        # Return params only for args:
        return params

    def create_attributes_for_user_params(self, **kwargs):
        for arg in kwargs:
            self.__setattr__(arg, kwargs[arg])

    def check_args(self, variable, params=NotImplemented, target_set=NotImplemented, context=None):
        """Instantiate variable (if missing or callable) and validate variable and params if PARAM_VALIDATION is set

        Called by execute methods to validate variable and params
        Can be suppressed by turning parameter_validation attribute off

        :param variable: (anything but a dict) - variable to validate
        :param params: (dict) - params to validate
        :target_set: (dict) - set to which params should be assigned (default: self.paramsCurrent)
        :return:
        """

        # If function is called without any arguments, get default for variable
        if variable is NotImplemented:
            variable = self.variableInstanceDefault # assigned by the Function class init when initializing

        # If the variable is a function, call it
        if callable(variable):
            variable = variable()

        # If parameter_validation is set and the function was called with a variable
        if self.prefs.paramValidationPref and not variable is NotImplemented:
            if context:
                context = context + kwSeparatorBar + kwFunctionCheckArgs
            else:
                context = kwFunctionCheckArgs
            self.validate_variable(variable, context=context)
        else:
            self.variable = variable

        # If target_set is not specified, use paramsCurrent
        if target_set is NotImplemented:
            target_set = self.paramsCurrent

        # If parameter_validation is set, the function was called with params,
        #   and they have changed, then validate requested values and assign to target_set
        if self.prefs.paramValidationPref and params and not params is NotImplemented and not params is target_set:
            # self.validate_params(params, target_set, context=kwFunctionCheckArgs)
            self.validate_params(request_set=params, target_set=target_set, context=context)

    def assign_defaults(self,
                        variable=NotImplemented,
                        request_set=NotImplemented,
                        assign_missing=True,
                        target_set=NotImplemented,
                        default_set=NotImplemented,
                        context=None
                        ):
        """Validate variable and/or param defaults in requested set and assign values to params in target set

          Variable can be any type other than a dictionary (reserved for use as params)
          request_set must contain a dict of params to be assigned to target_set (??and paramInstanceDefaults??)
          If assign_missing option is set, then any params defined for the class
              but not included in the requested set are assigned values from the default_set;
              if request_set is NotImplemented, then all values in the target_set are assigned from the default_set
              if the default set is not specified, then paramInstanceDefaults is used (see below)
          If target_set and/or default_set is not specified, paramInstanceDefaults is used for whichever is missing
              NOTES:
              * this is the most common case, used for updating of instance defaults:
                  neither target_set nor default_set are specified, and params in request_set are (after validation)
                   assigned to paramInstanceDefaults; any params not specified in the request set will stay the same
                   (even if assign_missing is set)
              * individual instance default values can be set to class defaults by
                  calling with a request_set that has the values from paramInstanceDefaults to be preserved,
                  paramInstanceDefaults as target_set, and paramClassDefaults as default_set
              * all paramInstanceDefaults can be set to class ("factory") defaults by
                  calling with an empty request_set (or is NotImplemented), paramInstanceDefaults for target_set,
                  and paramClassDefaults as default_set (although reset_params does the same thing)
          Class defaults can not be passed as target_set
              IMPLEMENTATION NOTE:  for now, treating class defaults as hard coded;
                                    could be changed in the future simply by commenting out code below

        :param variable: (anything but a dict (variable) - value to assign as variableInstanceDefault
        :param request_set: (dict) - params to be assigned
        :param assign_missing: (bool) - controls whether missing params are set to default_set values (default: False)
        :param target_set: (dict) - param set to which assignments should be made
        :param default_set: (dict) - values used for params missing from request_set (only if assign_missing is True)
        :return:
        """

        # Make sure all args are legal
        if not variable is NotImplemented:
            if isinstance(variable,dict):
                raise FunctionError("Dictionary passed as variable; probably trying to use param set as first argument")
        if request_set and not request_set is NotImplemented:
            if not isinstance(request_set, dict):
                raise FunctionError("requested parameter set must be a dictionary")
        if not target_set is NotImplemented:
            if not isinstance(target_set, dict):
                raise FunctionError("target parameter set must be a dictionary")
        if not default_set is NotImplemented:
            if not isinstance(default_set, dict):
                raise FunctionError("default parameter set must be a dictionary")

        # IMPLEMENTATION NOTE:  REMOVE
        # # Enforce implementation of variableEncodingDim and valueEncodingDim:
        # try:
        #     self.variableEncodingDim
        # except AttributeError:
        #     raise FunctionError("{0} or its base class must implement variableEncodingDim".
        #                         format(self.__class__.__name__))
        # try:
        #     self.valueEncodingDim
        # except AttributeError:
        #     raise FunctionError("{0} or its base class must implement valueEncodingDim".
        #                         format(self.__class__.__name__))


        # VALIDATE VARIABLE

        # if variable has been passed then validate and, if OK, assign as variableInstanceDefault
        self.validate_variable(variable, context=context)
        if variable is NotImplemented:
            self.variableInstanceDefault = self.variableClassDefault
        else:
            # MODIFIED 6/9/16 (CONVERT TO np.ndarray)
            self.variableInstanceDefault = self.variable



        # If no params were passed, then done
        if request_set is NotImplemented and  target_set is NotImplemented and default_set is NotImplemented:
            return

        # GET AND VALIDATE PARAMS

        # Assign param defaults for target_set and default_set
        if target_set is NotImplemented:
            target_set = self.paramInstanceDefaults
        if target_set is self.paramClassDefaults:
            raise FunctionError("Altering paramClassDefaults not permitted")
        if default_set is NotImplemented:
            default_set = self.paramInstanceDefaults

        self.paramNames = self.paramInstanceDefaults.keys()

        # If assign_missing option is set,
        #  assign value from specified default set to any params missing from request set
        # Note:  do this before validating execute method and params, as some params may depend on others being present
        if assign_missing:
            if not request_set or request_set is NotImplemented:
                request_set = {}

            # FIX: DO ALL OF THIS IN VALIDATE PARAMS?
            # FIX:    ?? HOWEVER, THAT MEANS MOVING THE ENTIRE IF STATEMENT BELOW TO THERE
            # FIX:    BECAUSE OF THE NEED TO INTERCEPT THE ASSIGNMENT OF functionParams FROM paramClassDefaults
            # FIX:    ELSE DON'T KNOW WHETHER THE ONES IN request_set CAME FROM CALL TO __init__() OR paramClassDefaults
            # FIX: IF functionParams ARE SPECIFIED, NEED TO FLAG THAT function != defaultFunction
            # FIX:    TO SUPPRESS VALIDATION OF functionParams IN validate_params (THEY WON'T MATCH paramclassDefaults)
            # Check if function matches one in paramClassDefaults;
            #    if not, suppress assignment of functionParams from paramClassDefaults, as they don't match the function
            # Note: this still allows functionParams included as arg in call to __init__ to be assigned

            # REFERENCE: Conditions for assignment of default function and functionParams
            #     A) default function, default functionParams
            #         example: Projection.__inits__
            #     B) default function, no default functionParams
            #         example: none??
            #     C) no default function, default functionParams
            #         example: DDM
            #     D) no default function, no default functionParams
            #         example: System, Process, MonitoringMechanism, WeightedError

            self.assign_default_kwFunctionParams = True

            try:
                function = request_set[FUNCTION]
            except KeyError:
                # If there is no function specified, then allow functionParams
                # Note: this occurs for objects that have "hard-coded" functions (e.g., DDM)
                self.assign_default_kwFunctionParams = True
            else:
                # Get function class:
                if inspect.isclass(function):
                    function_class = function
                else:
                    function_class = function.__class__
                # Get default function (from ParamClassDefaults)
                try:
                    default_function = default_set[FUNCTION]
                except KeyError:
                    # This occurs if a function has been specified as an arg in the call to __init__()
                    #     but there is no function spec in paramClassDefaults;
                    # This will be caught, and an exception raised, in validate_params()
                    pass
                else:
                    # Get default function class
                    if inspect.isclass(function):
                        default_function_class = default_function
                    else:
                        default_function_class = default_function.__class__

                    # If function's class != default function's class, suppress assignment of default functionParams
                    if function_class != default_function_class:
                        self.assign_default_kwFunctionParams = False

            for param_name, param_value in default_set.items():
                if param_name is FUNCTION_PARAMS and not self.assign_default_kwFunctionParams:
                    continue
                request_set.setdefault(param_name, param_value)
                if isinstance(param_value, dict):
                    for dict_entry_name, dict_entry_value in param_value.items():
                        request_set[param_name].setdefault(dict_entry_name, dict_entry_value)


        # VALIDATE PARAMS

        # if request_set has been passed or created then validate and, if OK, assign to targets
        if request_set and request_set != NotImplemented:
            self.validate_params(request_set, target_set, context=context)
            # Variable passed validation, so assign as instance_default

    def reset_params(self, mode):
        """Reset current and/or instance defaults

        If called with:
            - CURRENT_TO_INSTANCE_DEFAULTS all current param settings are set to instance defaults
            - INSTANCE_TO_CLASS all instance defaults are set to class defaults
            - ALL_TO_CLASS_DEFAULTS all current and instance param settings are set to class defaults

        :param mode: (ResetMode) - determines which params are reset
        :return none:
        """
        if not isinstance(mode, ResetMode):
            raise FunctionError("Must be called with a valid ResetMode")

        if mode == ResetMode.CURRENT_TO_INSTANCE_DEFAULTS:
            self.params_current = self.paramInstanceDefaults.copy()
        elif mode == ResetMode.INSTANCE_TO_CLASS:
            self.paramInstanceDefaults = self.paramClassDefaults.copy()
        elif mode == ResetMode.ALL_TO_CLASS_DEFAULTS:
            self.params_current = self.paramClassDefaults.copy()
            self.paramInstanceDefaults = self.paramClassDefaults.copy()

    def validate_variable(self, variable, context=None):
        """Validate variable and assign validated values to self.variable

        Convert variableClassDefault specification and variable (if specified) to list of 1D np.ndarrays:

        VARIABLE SPECIFICATION:                                        ENCODING:
        Simple value variable:                                         0 -> [array([0])]
        Single state array (vector) variable:                         [0, 1] -> [array([0, 1])
        Multiple state variables, each with a single value variable:  [[0], [0]] -> [array[0], array[0]]

        Perform top-level type validation of variable against the variableClassDefault;
            if the type is OK, the value is assigned to self.variable (which should be used by the function)
        This can be overridden by a subclass to perform more detailed checking (e.g., range, recursive, etc.)
        It is called only if the parameter_validation attribute is True (which it is by default)

        IMPLEMENTATION NOTES:
           * future versions should add hierarchical/recursive content (e.g., range) checking
           * add request/target pattern?? (as per validate_params) and return validated variable?

        :param variable: (anything other than a dictionary) - variable to be validated:
        :param context: (str)
        :return none:
        """

        pre_converted_variable = variable
        pre_converted_variable_class_default = self.variableClassDefault

        # FIX: SAYS "list of np.ndarrays" BELOW, WHICH WOULD BE A 2D ARRAY, BUT CONVERSION BELOW ONLY INDUCES 1D ARRAY
        # FIX: NOTE:  VARIABLE (BELOW) IS CONVERTED TO ONLY 1D ARRAY
        # Convert variableClassDefault to list of np.ndarrays
        # self.variableClassDefault = convert_to_np_array(self.variableClassDefault, 1)

        # If variable is not specified, then assign to (np-converted version of) variableClassDefault and return
        if variable is NotImplemented:
            self.variable = self.variableClassDefault
            return

        # Otherwise, do some checking on variable before converting to np.ndarray

        # If variable is a ParamValueProjection tuple, get value:
        from PsyNeuLink.Functions.Mechanisms.Mechanism import ParamValueProjection
        if isinstance(variable, ParamValueProjection):
            variable = variable.value

        # If variable is callable (function or object reference), call it and assign return to value to variable
        # Note: check for list is necessary since function references must be passed wrapped in a list so that they are
        #       not called before being passed
        if isinstance(variable, list) and callable(variable[0]):
            variable = variable[0]()

        # Convert variable to np.ndarray
        # Note: this insures that self.variable will be AT LEAST 1D;  however, can also be higher:
        #       e.g., given a list specification of [[0],[0]], it will return a 2D np.array
        variable = convert_to_np_array(variable, 1)

        # If variableClassDefault is locked, then check that variable matches it
        if self.variableClassDefault_locked:
            # If variable type matches variableClassDefault
            #    then assign variable to self.variable
            # if (type(variable) == type(self.variableClassDefault) or
            #         (isinstance(variable, numbers.Number) and
            #              isinstance(self.variableClassDefault, numbers.Number))):
            if not variable.dtype is self.variableClassDefault.dtype:
                message = "Variable for {0} (in {1}) must be a {2}".\
                    format(self.functionName, context, pre_converted_variable_class_default.__class__.__name__)
                raise FunctionError(message)

        self.variable = variable

    def validate_params(self, request_set, target_set=NotImplemented, context=None):
        """Validate params and assign validated values to targets,

        This performs top-level type validation of params against the paramClassDefaults specifications:
            - checks that param is listed in paramClassDefaults
            - checks that param value is compatible with on in paramClassDefaults
            - if param is a dict, checks entries against corresponding entries paramClassDefaults
            - if all is OK, the value is assigned to the target_set (if it has been provided)
            - otherwise, an exception is raised

        This can be overridden by a subclass to perform more detailed checking (e.g., range, recursive, etc.)
        It is called only if the parameter_validation attribute is True (which it is by default)

        IMPLEMENTATION NOTES:
           * future versions should add recursive and content (e.g., range) checking
           * should method return validated param set?

        :param dict (request_set) - set of params to be validated:
        :param dict (target_set) - repository of params that have been validated:
        :return none:
        """

        for param_name, param_value in request_set.items():

            # Check that param is in paramClassDefaults (if not, it is assumed to be invalid for this object)
            try:
                self.paramClassDefaults[param_name]
            except KeyError:
                raise FunctionError("{0} is not a valid parameter for {1}".format(param_name, self.__class__.__name__))

            # The value of the param is None or NotImplemented in paramClassDefaults: suppress type checking
            # DOCUMENT:
            # IMPLEMENTATION NOTE: this can be used for params with multiple possible types,
            #                      until type lists are implemented (see below)
            if self.paramClassDefaults[param_name] is None or self.paramClassDefaults[param_name] is NotImplemented:
                if self.prefs.verbosePref:
                    print("{0} is specified as NotImplemented for {1} "
                          "which suppresses type checking".format(param_name, self.name))
                if not target_set is NotImplemented:
                    target_set[param_name] = param_value
                continue

            # MODIFIED 8/24/16:
            # If the value in paramClassDefault is a type, check if param value is an instance of it
            if inspect.isclass(self.paramClassDefaults[param_name]):
                if isinstance(param_value, self.paramClassDefaults[param_name]):
                    continue

            # If the value in paramClassDefault is an object, check if param value is the corresponding class
            # This occurs if the item specified by the param has not yet been implemented (e.g., a function)
            if inspect.isclass(param_value):
                if isinstance(self.paramClassDefaults[param_name], param_value):
                    continue

            from PsyNeuLink.Functions.Utilities.Utility import Utility_Base
            from PsyNeuLink.Functions.States.ParameterState import get_function_param
            if isinstance(self, Utility_Base):
                param_value = get_function_param(param_value)

            # Check if param value is of same type as one with the same name in paramClassDefaults;
            #    don't worry about length
            if iscompatible(param_value, self.paramClassDefaults[param_name], **{kwCompatibilityLength:0}):
                # If param is a dict, check that entry exists in paramClassDefaults
                # IMPLEMENTATION NOTE:
                #    - currently doesn't check compatibility of value with paramClassDefaults
                #      since params can take various forms (e.g., value, tuple, etc.)
                #    - re-instate once paramClassDefaults includes type lists (as per requiredClassParams)
                if isinstance(param_value, dict):

                    # If assign_default_kwFunctionParams is False, it means that function's class is
                    #     compatiable but different from the one in paramClassDefaults;
                    #     therefore, FUNCTION_PARAMS will not match paramClassDefaults;
                    #     instead, check that functionParams are compatible with the function's default params
                    if param_name is FUNCTION_PARAMS and not self.assign_default_kwFunctionParams:
                        # Get function:
                        try:
                            function = request_set[FUNCTION]
                        except KeyError:
                            # If no function is specified, self.assign_default_kwFunctionParams should be True
                            # (see assign_defaults above)
                            raise FunctionError("PROGRAM ERROR: No function params for {} so should be able to "
                                                "validate {}".format(self.name, FUNCTION_PARAMS))
                        else:
                            for entry_name, entry_value in param_value.items():
                                try:
                                    function.paramClassDefaults[entry_name]
                                except KeyError:
                                    raise FunctionError("{0} is not a valid entry in {1} for {2} ".
                                                        format(entry_name, param_name, self.name))
                                # add [entry_name] entry to [param_name] dict
                                else:
                                    try:
                                        target_set[param_name][entry_name] = entry_value
                                    # [param_name] dict not yet created, so create it
                                    except KeyError:
                                        target_set[param_name] = {}
                                        target_set[param_name][entry_name] = entry_value
                                    # target_set NotImplemented
                                    except TypeError:
                                        pass
                    else:
                        for entry_name, entry_value in param_value.items():
                            # Make sure [entry_name] entry is in [param_name] dict in paramClassDefaults
                            try:
                                self.paramClassDefaults[param_name][entry_name]
                            except KeyError:
                                raise FunctionError("{0} is not a valid entry in {1} for {2} ".
                                                    format(entry_name, param_name, self.name))
                            # TBI: (see above)
                            # if not iscompatible(entry_value,
                            #                     self.paramClassDefaults[param_name][entry_name],
                            #                     **{kwCompatibilityLength:0}):
                            #     raise FunctionError("{0} ({1}) in {2} of {3} must be a {4}".
                            #         format(entry_name, entry_value, param_name, self.name,
                            #                type(self.paramClassDefaults[param_name][entry_name]).__name__))
                            else:
                                # add [entry_name] entry to [param_name] dict
                                try:
                                    target_set[param_name][entry_name] = entry_value
                                # [param_name] dict not yet created, so create it
                                except KeyError:
                                    target_set[param_name] = {}
                                    target_set[param_name][entry_name] = entry_value
                                # target_set NotImplemented
                                except TypeError:
                                    pass

                                # if not target_set is NotImplemented:
                                #     target_set[param_name][entry_name] = entry_value


                elif not target_set is NotImplemented:
                    target_set[param_name] = param_value

            # Parameter is not a valid type
            else:
                raise FunctionError("Value of {0} ({1}) must be of type {2} ".
                                    format(param_name, param_value,
                                           type(self.paramClassDefaults[param_name]).__name__))

    def validate_function(self, context=None):
        """Check that either params[FUNCTION] and/or self.execute are implemented

        # FROM validate_params:
        # It also checks FUNCTION:
        #     if it is specified and is a type reference (rather than an instance),
        #     it instantiates the reference (using FUNCTION_PARAMS if present)
        #     and puts a reference to the instance in target_set[FUNCTION]
        #
        This checks for an execute method in params[FUNCTION].
        It checks for a valid method reference in paramsCurrent, then paramInstanceDefaults, then paramClassDefaults
        If a specification is not present or valid:
            - it checks self.execute and, if present, kwExecute is assigned to it
            - if self.execute is not present or valid, an exception is raised
        When completed, there is guaranteed to be a valid method in paramsCurrent[FUNCTION] and/or self.execute;
            otherwise, an exception is raised

        Notes:
            * no new assignments (to FUNCTION or self.execute) are made here, except:
                if paramsCurrent[kwMethod] specified is not valid,
                an attempt is made to replace with a valid entry from paramInstanceDefaults or paramClassDefaults
            * if FUNCTION is missing, it is assigned to self.execute (if it is present)
            * no instantiations are done here;
            * any assignment(s) to and/or instantiation(s) of self.execute and/or params[FUNCTION]
                is/are carried out in instantiate_function

        :return:
        """

        # Check if params[FUNCTION] is specified
        try:
            param_set = kwParamsCurrent
            function = self.check_kwFunction(param_set)
            if not function:
                param_set = kwParamInstanceDefaults
                function, param_set = self.check_kwFunction(param_set)
                if not function:
                    param_set = kwParamClassDefaults
                    function, param_set = self.check_kwFunction(param_set)

        except KeyError:
            # FUNCTION is not specified, so try to assign self.function to it
            try:
                function = self.function
            except AttributeError:
                # self.function is also missing, so raise exception
                raise FunctionError("{} must either implement a function method, specify one as the FUNCTION param in"
                                    " paramClassDefaults, or as the default for the function argument in its init".
                                    format(self.__class__.__name__, FUNCTION))
            else:
                # self.function is NotImplemented
                # IMPLEMENTATION NOTE:  This is a coding error;  self.function should NEVER be assigned NotImplemented
                if (function is NotImplemented):
                    raise("PROGRAM ERROR: either {0} must be specified or {1}.function must be implemented for {2}".
                          format(FUNCTION,self.__class__.__name__, self.name))
                # self.function is OK, so return
                elif (isinstance(function, Function) or
                        isinstance(function, function_type) or
                        isinstance(function, method_type)):
                    self.paramsCurrent[FUNCTION] = function
                    return
                # self.function is NOT OK, so raise exception
                else:
                    raise FunctionError("{0} not specified and {1}.function is not a Function object or class"
                                        "or valid method in {2}".
                                        format(FUNCTION, self.__class__.__name__, self.name))

        # paramsCurrent[FUNCTION] was specified, so process it
        else:
            # FUNCTION is valid:
            if function:
                # - if other than paramsCurrent, report (if in VERBOSE mode) and assign to paramsCurrent
                if param_set is not kwParamsCurrent:
                    if self.prefs.verbosePref:
                        print("{0} ({1}) is not a Function object or a valid method; {2} ({3}) will be used".
                              format(FUNCTION,
                                     self.paramsCurrent[FUNCTION],
                                     param_set, function))
                self.paramsCurrent[FUNCTION] = function

            # FUNCTION was not valid, so try to assign self.function to it;
            else:
                # Try to assign to self.function
                try:
                    function = self.function
                except AttributeError:
                    # self.function is not implemented, SO raise exception
                    raise FunctionError("{0} ({1}) is not a Function object or class or valid method, "
                                        "and {2}.function is not implemented for {3}".
                                        format(FUNCTION,
                                               self.paramsCurrent[FUNCTION],
                                               self.__class__.__name__,
                                               self.name))
                else:
                    # self.function is there and is:
                    # - OK, so just warn that FUNCTION was no good and that self.function will be used
                    if (isinstance(function, Function) or
                            isinstance(function, function_type) or
                            isinstance(function, method_type)):
                        if self.prefs.verbosePref:
                            print("{0} ({1}) is not a Function object or class or valid method; "
                                                "{2}.function will be used instead".
                                                format(FUNCTION,
                                                       self.paramsCurrent[FUNCTION],
                                                       self.__class__.__name__))
                    # - NOT OK, so raise exception (FUNCTION and self.function were both no good)
                    else:
                        raise FunctionError("Neither {0} ({1}) nor {2}.function is a Function object or class "
                                            "or a valid method in {3}".
                                            format(FUNCTION, self.paramsCurrent[FUNCTION],
                                                   self.__class__.__name__, self.name))

    def check_kwFunction(self, param_set):
        """Check FUNCTION param is a Function,
        """

        function = getattr(self, param_set)[FUNCTION]
        # If it is a Function object, OK so return

        if (isinstance(function, FUNCTION_BASE_CLASS) or
                isinstance(function, function_type) or
                isinstance(function, method_type)):
            return function
        # Try as a Function class reference
        else:
            try:
                is_subclass = issubclass(self.paramsCurrent[FUNCTION], FUNCTION_BASE_CLASS)
            # It is not a class reference, so return None
            except TypeError:
                return None
            else:
                # It IS a Function class reference, so return function
                if is_subclass:
                    return function
                # It is NOT a Function class reference, so return none
                else:
                    return None

    def instantiate_attributes_before_function(self, context=None):
        pass

    def instantiate_function(self, context=None):
        """Instantiate function defined in <subclass>.function or <subclass>.paramsCurrent[FUNCTION]

        Instantiate params[FUNCTION] if present, and assign it to self.function

        If params[FUNCTION] is present and valid,
            it is assigned as the function's execute method, overriding any direct implementation of self.function

        If FUNCTION IS in params:
            - if it is a Function object, it is simply assigned to self.function;
            - if it is a Function class reference:
                it is instantiated using self.variable and, if present, params[FUNCTION_PARAMS]
        If FUNCTION IS NOT in params:
            - if self.function IS implemented, it is assigned to params[FUNCTION]
            - if self.function IS NOT implemented: program error (should have been caught in validate_function)
        Upon successful completion:
            - self.function === self.paramsCurrent[FUNCTION]
            - self.execute should always return the output of self.function in the first item of its output array;
                 this is done by Function.execute;  any subclass override should do the same, so that...
            - self.value == value[0] returned by self.execute

        :param request_set:
        :return:
        """

        try:
            function = self.paramsCurrent[FUNCTION]

        # params[FUNCTION] is NOT implemented
        except KeyError:
            function = None

        # params[FUNCTION] IS implemented
        else:
            # If FUNCTION is an already instantiated method:
            if isinstance(function, method_type):
                if issubclass(type(function.__self__), FUNCTION_BASE_CLASS):
                    pass
                # If it is NOT a subclass of Function,
                # - issue warning if in VERBOSE mode
                # - pass through to try self.function below
                else:
                    if self.prefs.verbosePref:
                        print("{0} ({1}) is not a subclass of Function".
                              format(FUNCTION,
                                     self.paramsCurrent[FUNCTION].__class__.__name__,
                                     self.name))
                    function = None

            # If FUNCTION is a Function object, assign it to self.function (overrides hard-coded implementation)
            elif isinstance(function, FUNCTION_BASE_CLASS):
                self.function = function

            # If FUNCTION is a Function class:
            # - instantiate method using:
            #    - self.variable
            #    - params[FUNCTION_PARAMS] (if specified)
            # - issue warning if in VERBOSE mode
            # - assign to self.function and params[FUNCTION]
            elif inspect.isclass(function) and issubclass(function, FUNCTION_BASE_CLASS):
                #  Check if params[FUNCTION_PARAMS] is specified
                try:
                    function_param_specs = self.paramsCurrent[FUNCTION_PARAMS].copy()
                except KeyError:
                    # FUNCTION_PARAMS not specified, so nullify
                    function_param_specs = {}
                else:
                    # FUNCTION_PARAMS are bad (not a dict):
                    if not isinstance(function_param_specs, dict):
                        # - nullify FUNCTION_PARAMS
                        function_param_specs = {}
                        # - issue warning if in VERBOSE mode
                        if self.prefs.verbosePref:
                            print("{0} in {1} ({2}) is not a dict; it will be ignored".
                                                format(FUNCTION_PARAMS, self.name, function_param_specs))
                    # parse entries of FUNCTION_PARAMS dict
                    else:
                        # Get param value from any params specified as ParamValueProjection or (param, projection) tuple
                        from PsyNeuLink.Functions.Projections.Projection import Projection
                        from PsyNeuLink.Functions.Mechanisms.Mechanism import ParamValueProjection
                        for param_name, param_spec in function_param_specs.items():
                            if isinstance(param_spec, ParamValueProjection):
                                from PsyNeuLink.Functions.States.ParameterState import ParameterState
                                function_param_specs[param_name] =  param_spec.value
                            if (isinstance(param_spec, tuple) and len(param_spec) is 2 and
                                    (param_spec[1] in {MAPPING, CONTROL_SIGNAL, LEARNING_SIGNAL} or
                                         isinstance(param_spec[1], Projection) or
                                         (inspect.isclass(param_spec[1]) and issubclass(param_spec[1], Projection)))
                                ):
                                from PsyNeuLink.Functions.States.ParameterState import ParameterState
                                function_param_specs[param_name] =  param_spec[0]

                # Instantiate function from class specification
                function_instance = function(variable_default=self.variable,
                                             params=function_param_specs,
                                             # owner=self,
                                             context=context)
                self.paramsCurrent[FUNCTION] = function_instance.function
                # MODIFIED 8/31/16 NEW:
                # # FIX: ?? COMMENT OUT WHEN self.paramsCurrent[FUNCTION] NOW MAPS TO self.function
                # self.function = self.paramsCurrent[FUNCTION]

                # If in VERBOSE mode, report assignment
                if self.prefs.verbosePref:
                    object_name = self.name
                    if self.__class__.__name__ is not object_name:
                        object_name = object_name + " " + self.__class__.__name__
                    try:
                        object_name = object_name + " of " + self.owner.name
                    except AttributeError:
                        pass
                    print("{0} assigned as function for {1}".
                          format(self.paramsCurrent[FUNCTION].__self__.functionName,
                                 object_name))

            # If FUNCTION is NOT a Function class reference:
            # - issue warning if in VERBOSE mode
            # - pass through to try self.function below
            else:
                if self.prefs.verbosePref:
                    print("{0} ({1}) is not a subclass of Function".
                          format(FUNCTION,
                                 self.paramsCurrent[FUNCTION].__class__.__name__,
                                 self.name))
                function = None

        # params[FUNCTION] was not specified (in paramsCurrent, paramInstanceDefaults or paramClassDefaults)
        if not function:
            # Try to assign to self.function
            try:
                self.paramsCurrent[FUNCTION] = self.function
            # If self.function is also not implemented, raise exception
            # Note: this is a "sanity check," as this should have been checked in validate_function (above)
            except AttributeError:
                raise FunctionError("{0} ({1}) is not a Function object or class, "
                                    "and {2}.function is not implemented".
                                    format(FUNCTION, self.paramsCurrent[FUNCTION],
                                           self.__class__.__name__))
            # If self.function is implemented, warn if in VERBOSE mode
            else:
                if self.prefs.verbosePref:
                    print("{0} ({1}) is not a Function object or a specification for one; "
                                        "{1}.function ({}) will be used instead".
                                        format(FUNCTION,
                                               self.paramsCurrent[FUNCTION].__self__.functionName,
                                               self.name,
                                               self.function.__self__.name))

        # Now that function has been instantiated, call self.function
        # to assign its output (and type of output) to self.value
        if not context:
            context = "DIRECT CALL"
        # MODIFIED 8/29/16:  QUESTION:
        # FIX: ?? SHOULD THIS CALL self.execute SO THAT function IS EVALUATED IN CONTEXT,
        # FIX:    AS WELL AS HOW IT HANDLES RETURN VALUES (RE: outputStates AND self.value??
        # ANSWER: MUST BE self.execute AS THE VALUE OF AN OBJECT IS THE OUTPUT OF ITS EXECUTE METHOD, NOT ITS FUNCTION
        # self.value = self.function(context=context+kwSeparator+kwFunctionInit)
        self.value = self.execute(context=context)
        if self.value is None:
            raise FunctionError("Execute method for {} must return a value".format(self.name))

        self.function_object = self.function.__self__
        self.function_object.owner = self

    def instantiate_attributes_after_function(self, context=None):
        pass

    def execute(self, input=NotImplemented, params=NotImplemented, time_scale=NotImplemented, context=None):
        raise FunctionError("{} class must implement execute".format(self.__class__.__name__))

    def update_value(self, context=None):
        """Evaluate execute method
        """
        self.value = self.execute(context=context)

    @property
    def variable(self):
        return self._variable

    @variable.setter
    def variable(self, value):
        self._variable = value

    @property
    def prefs(self):
        # Whenever pref is accessed, use current owner as context (for level checking)
        self._prefs.owner = self
        return self._prefs

    @prefs.setter
    def prefs(self, pref_set):
        if (isinstance(pref_set, PreferenceSet)):
            # IMPLEMENTATION NOTE:
            # - Complements dynamic assignment of owner in getter (above)
            # - Needed where prefs are assigned before they've been gotten (e.g., in PreferenceSet.__init__()
            # - owner needs to be assigned for call to get_pref_setting_for_level below
            # MODIFIED 6/1/16
            try:
                pref_set.owner = self
            except:
                pass
            # MODIFIED 6/1/16 END
            self._prefs = pref_set
            if self.prefs.verbosePref:
                print ('PreferenceSet {0} assigned to {1}'.format(pref_set.name, self.name))
            # Make sure that every pref attrib in PreferenceSet is OK
            for pref_name, pref_entry in self.prefs.__dict__.items():
                if '_pref' in pref_name:
                    value, err_msg = self.prefs.get_pref_setting_for_level(pref_name, pref_entry.level)
                    if err_msg and self.prefs.verbosePref:
                        print(err_msg)
                    # FIX: VALUE RETURNED SHOULD BE OK, SO ASSIGN IT INSTEAD OF ONE IN pref_set??
                    # FIX: LEVEL SHOULD BE LOWER THAN REQUESTED;  REPLACE RAISE WITH WARNING TO THIS EFFECT
        else:
            raise FunctionError("Attempt to assign non-PreferenceSet {0} to {0}.prefs".
                                format(pref_set, self.name))

    @property
    def params(self):
        return self.paramsCurrent

    @property
    def user_params(self):
        return self._user_params

    @user_params.setter
    def user_params(self, new_params):
        self._user_params = new_params
        TEST = True

    @property
    def paramsCurrent(self):
        return self._paramsCurrent

    @paramsCurrent.setter
    def paramsCurrent(self, dict):

        try:
            self._paramsCurrent.update(dict)
        except AttributeError:
            self._paramsCurrent = ParamsDict(self, dict)

            # INSTANTIATE PARAMSCURRENT AS A USER DICT HERE (THAT IS CONFIGURED TO HAVE GETTERS AND SETTERS FOR ITS ENTRIES)
            #    AND COPY THE DICT PASSED IN INTO IT (RATHER THAN SIMPLY ASSIGNING IT;  OR, ASSIGN INITIAL PARAM DICTS
            #    TO THE SAME USER CLASS SO THAT THE ASSIGNMENT IS TO A VERSION OF THE USER DICT
            # WHEN THOSE ENTRIES ARE SET IN USER DICT, REFERENCE THEM USING GETTATTR AND SETATTR
            #    TO THE CORRESPONDING ATTRIBUTES OF THE OWNER OBJECT

    @property
    def verbosePref(self):
        return self.prefs.verbosePref

    @verbosePref.setter
    def verbosePref(self, setting):
        self.prefs.verbosePref = setting

    @property
    def paramValidationPref(self):
        return self.prefs.paramValidationPref

    @paramValidationPref.setter
    def paramValidationPref(self, setting):
        self.prefs.paramValidationPref = setting

    @property
    def reportOutputPref(self):
        return self.prefs.reportOutputPref

    @reportOutputPref.setter
    def reportOutputPref(self, setting):
        self.prefs.reportOutputPref = setting

    @property
    def logPref(self):
        return self.prefs.logPref

    @logPref.setter
    def logPref(self, setting):
        self.prefs.logPref = setting

    @property
    def functionRuntimeParamsPref(self):
        return self.params.functionRuntimeParamsPref

    @functionRuntimeParamsPref.setter
    def functionRuntimeParamsPref(self, setting):
        self.params.functionRuntimeParamsPref = setting


FUNCTION_BASE_CLASS = Function

def get_function_param(param):
    """Returns param value (first item) of either a ParamValueProjection or an unnamed (value, projection) tuple
    """
    from PsyNeuLink.Functions.Mechanisms.Mechanism import ParamValueProjection
    from PsyNeuLink.Functions.Projections.Projection import Projection
    if isinstance(param, ParamValueProjection):
        value =  param.value
    elif (isinstance(param, tuple) and len(param) is 2 and
            (param[1] in {MAPPING, CONTROL_SIGNAL, LEARNING_SIGNAL} or
                 isinstance(param[1], Projection) or
                 (inspect.isclass(param[1]) and issubclass(param[1], Projection)))
          ):
        value =  param[0]
    else:
        value = param

    return value