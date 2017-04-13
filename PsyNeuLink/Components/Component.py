# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************** Component  ************************************************************


"""

.. _Component_Overview:

Overview
--------

Component is the base class for all of the objects used to create compositions (`processes <Process>` or 
`systems <System>`) in PsyNeuLink.  It defines a common set of attributes possessed, and methods used by all 
component objects.

.. _Component_Creation:

Creating a Component
--------------------

A Component is never created directly.  However, its __init__ method is always called when a subclass is instantiated;
that, in turn, calls a standard set of methods (listed `below <Component_Methods>`) as part of the initialization 
procedure.

.. _Component_Structure:

Component Structure
-------------------

.. _Component_Attributes:

Component Attributes
~~~~~~~~~~~~~~~~~~~~

Every component has the following set of core attributes that govern its operation:

.. _Component_Variable:

* **variable** - the value of the `variable <Component.variable>` attribute is used as the input to its 
  `function <Component.function>`.  Specification of the variable in the constructor for a component determines both 
  its format (e.g., whether it's value is numeric, its dimensionality and shape if it is an array, etc.) as well as 
  its default value (the value used when the component is executed and no input is provided). 

.. _Component_Function:

* **function** - the `function <Component.function>` attribute determines the computation that a component carries out.  
  It is always a PsyNeuLink `Function <Function>` object (itself a PsyNeuLink component).
  
  .. note::
     The `function <Component.function>` of a component can be assigned either a `Function <Function>` object or any
     other callable object in python.  If the latter is assigned, it will be "wrapped" in a `UserDefinedFunction`.  
  
  All components have a default `function <Component.function>` (with a default set of parameters), that is used if it 
  is not otherwise specified.  The `function <Component.function>` can be specified in the 
  function argument of the constructor for the component, using one of the following: 

    * **class** - this must be a subclass of `Function <Function>`, as in the following example::   

        my_component = SomeComponent(function=SomeFunction)
      
      This will create a default instance of the specified subclass, using default values for its parameters. 
    |
    * **Function** - this can be either an existing `Function <Function>` object or the constructor for one, as in the
      following examples:
            
        my_component = SomeComponent(function=SomeFunction)
        
        or
        
        some_function = SomeFunction(some_param=1)
        my_component = SomeComponent(some_function)
        
      The specified Function will be used as a template to create a new Function object that is assigned to the 
      `function_object` attribute of the component, the `function <Function.function>` of which will be assigned as  
      the 'function <Component.function>` attribute of the component.
    
      .. note::
      
        In the current implementation of PsyNeuLink, if a `Function <Function>` object (or the constructor for one) is 
        used to specify the `function <Component.function>` attribute of a component, the Function object specified (or 
        created) is used to determine attributes of the Function object created for and assigned to the component, but  
        is not *itself* assigned to the component.  This is so that `Functions <Function>` can be used as templates for 
        more than one component, without being assigned simultaneously to multiple components.

  A `function <Component.function>` can also be specified in an entry of a 
  `parameter specification dictionary <Mechanism_Creation>` assigned to the `params <Component.params>` argument of the 
  constructor for the component, with the keyword FUNCTION as its key, and one of the specifications above as its 
  value, as in the following example::  

        my_component = SomeComponent(params={FUNCTION:SomeFunction(some_param=1)})

* **function_params** - the `function_params <Component.function>` attribute contains a dictionary of the parameters 
  for the component's `function <Component.function>` and their values.  Each entry is the name of a parameter, and
  its value the value of that parameter.  This dictionary is read-only.  The parameters for the 
  `function_params <Component.function>` can be specified when the component is created in one of the following ways:
  
  * in the **constructor** for a Function -- if that is used to specify the `function <Component.function>` argument,
    as in the following example::

        my_component = SomeComponent(function=SomeFunction(some_param=1, some_param=2)

  * in an argument of the **component's constructor** -- if all of the allowable functions for a component's
    `function <Component.function>` share some or all of their parameters in common, the shared paramters may appear 
    as arguments in the constructor of the component itself, which can be used to set their values.

  * in an entry of a `parameter specification dictionary <Mechanism_Creation>` assigned to the 
    `params <Component.params>` argument of the constructor for the component.  The entry must use the keyword 
    FUNCTION_PARAMS as its key, and its value must be a dictionary containing the parameters and their values.
    The key for each entry in the FUNCTION_PARAMS dictionary must be the name of a parameter, and its value the 
    parameter's value, as in the example below::
    
        my_component = SomeComponent(function=SomeFunction
                                     params={FUNCTION_PARAMS:{SOME_PARAM=1, SOME_OTHER_PARAM=2}})

  The parameters for a `function <Component.function>` can be modified after it has been created by assigning the 
  new value to the corresponding attribute of the component's `function_object`; for example, if the name of the
  parameter is mole, it can be modified as follows::
  
        my_component.function_object.mole = 6.0221409
  
  COMMENT:       
      See `ParameterState_Specifying_Parameters` for details of parameter specification.
  COMMENT

.. _Component_Function_Object:

* **function_object** - the `function_object` attribute refers to the `Function <Function>` assigned to the component; 
  its `function <Function.function>` is assigned to the `function <Component>` attribute of the component.  The 
  parameters of the Function can be modified by assigning values to the attributes corresponding to those parameters 
  (see `function_params` above).

.. _Component_User_Params:

* **user_params** - the `user_params` attribute contains a dictionary of all of the user-modifiable attributes for the
  the component.  This dictionary is read-only.  Changes to the value of an attribute must be made by assigning a 
  value to the attribute directly.
..  
COMMENT:
  INCLUDE IN DEVELOPERS' MANUAL
    * **paramClassDefaults**
    
    * **paramInstanceDefaults**
COMMENT

* **value** - the `value <Component.value>` attribute contains the result (return value) of the component's 
  `function <Component.function>` after the function is called.     
..
* **name** - the `name <Component.name>` attribute contains the name assigned to the component when it was created.  
  If it was not specified, a default is assigned by the registry for subclass (see :doc:`Registry <LINK>` for 
  conventions used in assigning default names and handling of duplicate names).
..
* **prefs** - the `prefs <Components.prefs>` attribute contains the `PreferenceSet` assigned to the component when
  it was created.  If it was not specified, a default is assigned using `classPreferences` defined in __init__.py
  Each individual preference is accessible as an attribute of the component, the name of which is the name of the
  preference (see `PreferenceSet <LINK>` for details).  

COMMENT:
* **log**
COMMENT


COMMENT:
   INCLUDE IN DEVELOPERS' MANUAL
      
    .. _Component_Methods:
    
    Component Methods
    ~~~~~~~~~~~~~~~~~
    
    There are two sets of methods that belong to every component: one set that is called when it is initialized; and 
    another set that can be called to perform various operations common to all components.  Each of these is described 
    briefly below.  All of these methods can be overridden by subclasses to implement customized operations, however   
    it is strongly recommended that the method be called on super() at some point, so that the standard operations are 
    carried out.  Whether customization operations should be performed before or after the call to super is discussed in 
    the descriptions below where relevant. 
    
    .. _Component_Initialization_Methods:
    
    Initialization Methods
    ^^^^^^^^^^^^^^^^^^^^^^
    
    These methods can be overridden by the subclass to customize the initialization process, but should always call the
    corresponding method of the Component base class (using ``super``) to insure full initialization.  There are two
    categories of initializion methods:  validation and instantiation.  
    
    
    .. _Component_Validation_Methods:
    
    * **Validation methods** perform a strictly *syntactic* check, to determine if a value being validated conforms 
    to the format expected for it by the component (i.e., the type of the value and, if it is iterable, the type its 
    elements and/or its length).  The value itself is not checked in any other way (e.g., whether it equals a particular 
    value or falls in a specified range).  If the validation fails, and exception is raised.  Validation methods never 
    make changes the actual value of an attribute, but they may change its format (e.g., from a list to an ndarray) to
    comply with requirements of the component.
    
      * `_validate_variable <Component._validate_variable>` validates the value provided to the keyword:`variable` 
        argument in the constructor for the component.  If it is overridden, customized validation should generally 
        performed *prior* to the call to super(), to allow final processing by the Component base class. 
        
      * `_validate_params <Component._validate_params>` validates the value of any parameters specified in the 
        constructor for the component (whether they are made directly in the argument for a parameter, or in a 
        `parameter specification dictionary <Mechanism_Creation>`.  If it is overridden by a subclass, customized 
        validation should generally be performed *after* the call to super().
    
    * **Instantiation methods** create, assign, and/or perform *semantic* checks on the values of component attributes.  
      Semantic checks may include value and/or range checks, as well as checks of formatting and/or value 
      compatibility with other attributes of the component and/or the attributes of other components (for example, the
      _instantiate_function method checks that the input of the component's `function <Comonent.function>` is compatible 
      with its `variable <Component.variable>`).
    
      * `_instantiate_defaults <Component._instantiate_defaults>` first calls the validation methods, and then  
        assigns the default values for all of the attributes of the instance of the component being created.
        
        _instantiate_attributes_before_function
        _instantiate_function
        _instantiate_attributes_after_function
    
    .. _Component_Callable_Methods:
    
    Callable Methods
    ^^^^^^^^^^^^^^^^
    
    initialize
    assign_params
    reset_params
    execute
COMMENT

.. _Component_Execution:

Execution
~~~~~~~~~

Calls the :keyword:`execute` method of the subclass that, in turn, calls its :keyword:`function`.

COMMENT:
   INCLUDE IN DEVELOPERS' MANUAL
    .. _Component_Class_Reference:
    
    Class Reference
    ---------------
COMMENT

COMMENT:

This module defines the Component abstract class

It also contains:

- arg_name definitions for primary component categories:
    Process
    Mechanism
        types:
            DDM
            [PDP]
    Projection
        types:
            MappingProjection
            ControlProjection
            LearningProjection
    Function
COMMENT

"""

from collections import OrderedDict
from PsyNeuLink.Globals.Utilities import *
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import *

class ResetMode(Enum):
    CURRENT_TO_INSTANCE_DEFAULTS = 0
    INSTANCE_TO_CLASS = 1
    ALL_TO_CLASS_DEFAULTS = 2

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
        # MODIFIED 4/10/17 NEW: REPLACE FUNCTION_PARAMS WITH READONLYDICT
        if FUNCTION_PARAMS in dict:
            self[FUNCTION_PARAMS] = ReadOnlyOrderedDict(name=FUNCTION_PARAMS)
            for param_name in sorted(list(dict[FUNCTION_PARAMS].keys())):
                self[FUNCTION_PARAMS].__additem__(param_name, dict[FUNCTION_PARAMS][param_name])
            TEST = True
        # MODIFIED 4/10/17 END

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
            pass

    def __setitem__(self, key, item):

        # # WORKS:
        # super().__setitem__(key, item)

        setattr(self.owner, key, item)
    # # ORIG:
    #     self.data[key] = item

parameter_keywords = set()

# Used as templates for requiredParamClassDefaultTypes for COMPONENT:
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

class ComponentLog(IntEnum):
    NONE            = 0
    ALL = 0
    DEFAULTS = NONE


class ComponentError(Exception):
     def __init__(self, error_value):
         self.error_value = error_value

     def __str__(self):
         return repr(self.error_value)

# *****************************************   COMPONENT CLASS  ********************************************************


class Component(object):
    """Implement parent class for Components used by Process, Mechanism, State, and Projection class categories

        Every component is associated with:
         - child class componentName
         - type
         - input (self.variable)
         - execute (method): called to execute it;  it in turn calls self.function
         - function (method): carries out object's core computation
             it can be referenced either as self.function, self.params[FUNCTION] or self.paramsCurrent[FUNCTION]
         - function_object (Function): the object to which function belongs (and that defines it's parameters)
         - output (value: self.value)
         - outputValue (return from self.execute: concatenated set of values of outputStates)
         - class and instance variable defaults
         - class and instance param defaults
        The components's execute method (<subclass>.execute is the component's primary method
            (e.g., it is the one called when process, mechanism, state and projections objects are updated);
            the following attributes for or associated with the method are defined for every component object:
                + execute (method) - the execute method itself
                + value (value) - the output of the execute method
            the latter is used for typing and/or templating other variables (e.g., self.variable):
                type checking is generally done using Utilities.iscompatible(); for iterables (lists, tuples, dicts):
                    if the template (the "reference" arg) has entries (e.g., [1, 2, 3]), comparisons will include length
                    if the template is empty (e.g., [], {}, or ()), length will not be checked
                    if the template has only numbers, then the candidate must as well


        The component itself can be called without any arguments (in which case it uses its instance defaults) or
            one or more variables (as defined by the subclass) followed by an optional params dictionary
        The variable(s) can be a function reference, in which case the function is called to resolve the value;
            however:  it must be "wrapped" as an item in a list, so that it is not called before being passed
                      it must of course return a variable of the type expected for the variable
        The default variableList is a list of default values, one for each of the variables defined in the child class
        The params argument is a dictionary; the key for each entry is the parameter name, associated with its value.
            + component subclasses can define the param FUNCTION:<method or Function class>
        The component can be called with a params argument, which should contain entries for one or more of its params;
            - those values will be assigned to paramsCurrent at run time (overriding previous values in paramsCurrent)
            - if the component is called without a variable and/or params argument, it uses paramInstanceDefaults
        The instance defaults can be assigned at initialization or using the _instantiate_defaults class method;
            - if instance defaults are not assigned on initialization, the corresponding class defaults are assigned
        Parameters can be REQUIRED to be in paramClassDefaults (for which there is no default value to assign)
            - for all classes, by listing the name and type in requiredParamClassDefaultTypes dict of the Function class
            - in subclasses, by inclusion in requiredParamClassDefaultTypes (via copy and update) in class definition
            * NOTE: inclusion in requiredParamClasssDefault simply acts as a template;  it does NOT implement the param
        Each component child class must initialize itself by calling super(childComponentName).__init__()
            with a default value for its variable, and optionally an instance default paramList.

        A subclass MUST either:
            - implement a <class>.function method OR
            - specify paramClassDefaults[FUNCTION:<Function>];
            - this is checked in Component._instantiate_function()
            - if params[FUNCTION] is NOT specified, it is assigned to self.function (so that it can be referenced)
            - if params[FUNCTION] IS specified, it assigns it's value to self.function (superceding existing value):
                self.function is aliased to it (in Component._instantiate_function):
                    if FUNCTION is found on initialization:
                        if it is a reference to an instantiated function, self.function is pointed to it
                        if it is a class reference to a function:
                            it is instantiated using self.variable and FUNCTION_PARAMS (if they are there too)
                            this works, since _validate_params is always called after _validate_variable
                            so self.variable can be used to initialize function
                            to the method referenced by paramInstanceDefaults[FUNCTION] (see below)
                    if paramClassDefaults[FUNCTION] is not found, it's value is assigned to self.function
                    if neither paramClassDefaults[FUNCTION] nor self.function is found, an exception is raised
        - self.value is determined for self.execute which calls self.function in Component._instantiate_function

        NOTES:
            * In the current implementation, validation is:
              - top-level only (items in lists, tuples and dictionaries are not checked, nor are nested items)
              - for type only (it is oblivious to content)
              - forgiving (e.g., no distinction is made among numberical types)
            * However, more restrictive validation (e.g., recurisve, range checking, etc.) can be achieved
                by overriding the class _validate_variable and _validate_params methods

    Class attributes:
        + className
        + suffix - " " + className (used to create subclass and instance names)
        + componentCategory - category of Component (i.e., process, mechanism, projection, learning, function)
        + componentType - type of component within a category
                             (e.g., TransferMechanism, MappingProjection, ControlProjection, etc.)
        + requiredParamClassDefaultTypes - dict of param names and types that all subclasses of Component must
        implement;

    Class methods:
        - _validate_variable(variable)
        - _validate_params(request_set, target_set, context)
        - _instantiate_defaults(variable, request_set, assign_missing, target_set, default_set=None
        - reset_params()
        - _check_args(variable, params)
        - _assign_args_to_param_dicts(params, param_names, function_param_names)

    Instance attributes:
        + name
        + componentName - name of particular Function (linear, exponential, integral, etc.)
        + variableClassDefault (value)
        + variableClassDefault_np_info (ndArrayInfo)
        + variableInstanceDefault (value)
        + _variable_not_specified
        + variable (value)
        + variable_np_info (ndArrayInfo)
        + function (method)
        + function_object (Function)
        + paramClassDefaults:
            + FUNCTION
            + FUNCTION_PARAMS
        + paramInstanceDefaults
        + paramsCurrent
        # + parameter_validation
        + user_params
        + runtime_params_in_use
        + recording

    Instance methods:
        + function (implementation is optional; aliased to params[FUNCTION] by default)
    """

    #region CLASS ATTRIBUTES
    className = "COMPONENT"
    suffix = " " + className
# IMPLEMENTATION NOTE:  *** CHECK THAT THIS DOES NOT CAUSE ANY CHANGES AT SUBORDNIATE LEVELS TO PROPOGATE EVERYWHERE
    componentCategory = None
    componentType = None

    initMethod = INIT_FULL_EXECUTE_METHOD

    classPreferenceLevel = PreferenceLevel.SYSTEM
    # Any preferences specified below will override those specified in SystemDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to SYSTEM automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'ComponentCustomClassPreferences',
    #     kp<pref>: <setting>...}

    # Determines whether variableClassDefault can be changed (to match an variable in __init__ method)
    variableClassDefault_locked = False


    # Names and types of params required to be implemented in all subclass paramClassDefaults:
    # Notes:
    # *  entry values here do NOT implement the param; they are simply used as type specs for checking (in __init__)
    # * kwComponentCategory (below) is used as placemarker for Component.Function class; replaced in __init__ below
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
        """Assign default preferences; enforce required params; validate and instantiate params and execute method

        Initialization arguments:
        - variable_default (anything): establishes type for the variable, used for validation
        - params_default (dict): assigned as paramInstanceDefaults
        Note: if parameter_validation is off, validation is suppressed (for efficiency) (Component class default = on)

        :param variable_default: (anything but a dict) - value to assign as variableInstanceDefault
        :param param_defaults: (dict) - params to be assigned to paramInstanceDefaults
        :param log: (ComponentLog enum) - log entry types set in self.componentLog
        :param name: (string) - optional, overrides assignment of default (componentName of subclass)
        :return:
        """

        # # MODIFIED 8/14/16 NEW:
        # # PROBLEM: variable has different name for different classes;  need to standardize name across classes
        # try:
        #     if self.value is DEFERRED_INITIALIZATION:
        #         defer_init = True
        # except AttributeError:
        #     pass
        # else:
        #     if defer_init:
        #         self.init_args = locals().copy()
        #         del self.init_args['self']
        #         # del self.init_args['__class__']
        #         return
        context = context + INITIALIZING + ": " + COMPONENT_INIT

        # These insure that subclass values are preserved, while allowing them to be referred to below
        self.variableInstanceDefault = None
        self.paramInstanceDefaults = {}

        # self.componentName = self.componentType
        try:
            self.componentName = self.componentName or self.componentType
        except AttributeError:
            self.componentName = self.componentType
        #endregion

        #region ENFORCE REGISRY
        if self.__class__.__bases__[0].__bases__[0].__bases__[0].__name__ is 'ShellClass':
            try:
                self.__class__.__bases__[0].registry
            except AttributeError:
                raise ComponentError("{0} is a category class and so must implement a registry".
                                    format(self.__class__.__bases__[0].__name__))
        #endregion

        #region ASSIGN PREFS

        # If a PreferenceSet was provided, assign to instance
        if isinstance(prefs, PreferenceSet):
            self.prefs = prefs
            # FIX:  CHECK LEVEL HERE??  OR DOES IT NOT MATTER, AS OWNER WILL BE ASSIGNED DYNAMICALLY??
        # Otherwise, if prefs is a specification dict instantiate it, or if it is None assign defaults
        else:
            self.prefs = ComponentPreferenceSet(owner=self, prefs=prefs, context=context)
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
        # Used by run to store return value of execute
        self.results = []
        #endregion


        #region ENFORCE REQUIRED CLASS DEFAULTS

        # All subclasses must implement variableClassDefault
        # Do this here, as _validate_variable might be overridden by subclass
        try:
            if self.variableClassDefault is NotImplemented:
                raise ComponentError("variableClassDefault for {} must be assigned a value or \'None\'".
                                     format(self.componentName))
        except AttributeError:
            raise ComponentError("variableClassDefault must be defined for {} or its base class".
                                format(self.componentName))
        #endregion

        #region CHECK FOR REQUIRED PARAMS

        # All subclasses must implement, in their paramClassDefaults, params of types specified in
        #     requiredClassParams (either above or in subclass defintion)
        # Do the check here, as _validate_params might be overridden by subclass
        for required_param, type_requirements in self.requiredParamClassDefaultTypes.items():
            # # Replace 'Function' placemarker with class reference:
            # type_requirements = [self.__class__ if item=='Function' else item for item in type_requirements]

            # get type for kwComponentCategory specification
            import PsyNeuLink.Components.Functions.Function
            if kwComponentCategory in type_requirements:
               type_requirements[type_requirements.index(kwComponentCategory)] = \
                   type(PsyNeuLink.Components.Functions.Function.Function_Base)

            if required_param not in self.paramClassDefaults.keys():
                raise ComponentError("Param {} must be in paramClassDefaults for {}".
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
                    raise ComponentError("Value ({}) of param {} is not appropriate for {};"
                                        "  requires one of the following types: {}".
                                        format(required_param_value.__name__, required_param, self.name, type_names))
            except TypeError:
                pass
        #endregion

        #region ASSIGN DEFAULTS
        # Validate the set passed in and assign to paramInstanceDefaults
        # By calling with assign_missing, this also populates any missing params with ones from paramClassDefaults
        self._instantiate_defaults(variable=variable_default,
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

        self.runtime_params_in_use = False
        #endregion

        #region VALIDATE FUNCTION (self.function and/or self.params[function, FUNCTION_PARAMS])
        self._validate_function(context=context)
        #endregion

        #region INSTANTIATE ATTRIBUTES BEFORE FUNCTION
        # Stub for methods that need to be executed before instantiating function
        #    (e.g., _instantiate_sender and _instantiate_receiver in Projection)
        self._instantiate_attributes_before_function(context=context)
        #endregion

        #region INSTANTIATE FUNCTION and assign output (by way of self.execute) to self.value
        self._instantiate_function(context=context)
        #endregion

        #region INSTANTIATE ATTRIBUTES AFTER FUNCTION
        # Stub for methods that need to be executed after instantiating function
        #    (e.g., instantiate_output_state in Mechanism)
        self._instantiate_attributes_after_function(context=context)
        #endregion

#endregion

    def _deferred_init(self, context=None):
        """Use in subclasses that require deferred initialization
        """
        if self.value is DEFERRED_INITIALIZATION:

            # Flag that object is now being initialized
            # Note: self.value will be resolved to the object's value as part of initialization
            #       (usually in _instantiate_function)
            self.value = INITIALIZING

            del self.init_args['self']

            # Delete function since super doesn't take it as an arg;
            #   the value is stored in paramClassDefaults in assign_ags_to_params_dicts,
            #   and will be restored in _instantiate_function
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

            # If name is None, mark as deferred so that name can be customized
            #    using info that has become available at time of deferred init
            self.init_args['name'] = self.init_args['name'] or ('deferred_init_' + self.className) or \
                                     DEFERRED_DEFAULT_NAME

            # Complete initialization
            super(self.__class__,self).__init__(**self.init_args)

    def _assign_args_to_param_dicts(self, **kwargs):
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
            if arg_name is kwParams:
                continue

            # Check if param exists in paramClassDefaults
            try:
                self.paramClassDefaults[arg]

            # param corresponding to arg is NOT in paramClassDefaults, so add it
            except:
                # If arg is FUNCTION and it's default is an instance (i.e., not a class)
                if arg_name is FUNCTION and not inspect.isclass(default(arg)):

                    # FIX: REFACTOR Component._instantiate_function TO USE COPY OF INSTANTIATED function

                    function = default(arg)
                    from PsyNeuLink.Components.Functions.Function import Function
                    from inspect import isfunction

                    # It is a PsyNeuLink Function
                    if isinstance(function, Function):
                        # Set it to the class (for compatibility with current implementation of _instantiate_function()
                        # and put its params in FUNCTION_PARAMS
                        self.paramClassDefaults[arg] = default(arg).__class__
                        self.paramClassDefaults[FUNCTION_PARAMS] = default(arg).user_params.copy()

                    # It is a generic function
                    elif isfunction(function):
                        # Assign to paramClassDefaults as is (i.e., don't convert to class), since class is generic
                        # (_instantiate_function also tests for this and leaves it as is)
                        self.paramClassDefaults[arg] = function
                    else:
                        raise ComponentError("Unrecognized object ({}) specified as function for {}".
                                             format(function, self.name))

                # Get defaults values for args listed in FUNCTION_PARAMS
                # Note:  is not an arg, but rather used to package args that belong to a non-instantiated function
                elif arg is FUNCTION_PARAMS:
                    self.paramClassDefaults[FUNCTION_PARAMS] = {}
                    for item in kwargs[arg]:
                        self.paramClassDefaults[FUNCTION_PARAMS][item] = default(item)
                else:
                    if inspect.isclass(default(arg)) and issubclass(default(arg),inspect._empty):
                        raise ComponentError("PROGRAM ERROR: \'{}\' parameter of {} must be assigned a default value "
                                             "in its constructor or in paramClassDefaults (it can be \'None\')".
                                             format(arg, self.__class__.__name__))
                    self.paramClassDefaults[arg] = default(arg)

            # param corresponding to arg IS already in paramClassDefaults, so ignore
            else:
                continue

        # ASSIGN ARG VALUES TO params dicts

        # IMPLEMENTATION NOTE:  Use OrderedDicts for params (as well as user_params and user_param_for_instantiation)
        #                       to insure a consistent order of retrieval (e.g., EVC ControlSignalGridSearch);
        params = OrderedDict() # this is for final params that will be returned;
        params_arg = {}        # this captures values specified in a params arg, that are used to override arg values
        ignore_FUNCTION_PARAMS = False

        # Sort kwargs so that params are entered in params OrderedDict in a consistent (alphabetical) order
        for arg in sorted(list(kwargs.keys())):

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
                    # Get copy of default params
                    # IMPLEMENTATION NOTE: this is needed so that function_params gets included in user_params and
                    #                      thereby gets instantiated as a property in _create_attributes_for_user_params
                    params[FUNCTION_PARAMS] = ReadOnlyOrderedDict(name=FUNCTION_PARAMS)
                    for param_name in sorted(list(function().user_params.keys())):
                        params[FUNCTION_PARAMS].__additem__(param_name, function().user_params[param_name])
                    continue

                # function arg is not a class (presumably an object)
                # FIX: REFACTOR Function._instantiate_function TO USE INSTANTIATED function
                else:
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

                    from PsyNeuLink.Components.Functions.Function import Function
                    from inspect import isfunction

                    # It is a PsyNeuLink Function
                    # IMPLEMENTATION NOTE:  REPLACE THIS WITH "CONTINUE" ONCE _instantiate_function IS REFACTORED TO
                    #                       TO ALLOW Function SPECIFICATION (VS. ONLY CLASS)
                    if isinstance(function, Function):
                        # Set it to the class (for compatibility with current implementation of _instantiate_function()
                        params[FUNCTION] = function.__class__
                        # Create ReadOnlyDict for FUNCTION_PARAMS and copy function's params into it
                        params[FUNCTION_PARAMS] = ReadOnlyOrderedDict(name=FUNCTION_PARAMS)
                        for param_name in sorted(list(function.user_params_for_instantiation.keys())):
                            params[FUNCTION_PARAMS].__additem__(param_name,
                                                                function.user_params_for_instantiation[param_name])

                    # It is a generic function
                    elif isfunction(function):
                        # Assign as is (i.e., don't convert to class), since class is generic
                        # (_instantiate_function also tests for this and leaves it as is)
                        params[FUNCTION] = function
                        # FIX: UNCOMMENT WHEN EVC IS GIVEN A PREF SET
                        # if self.verbosePref:
                        #     warnings.warn("{} is not a PsyNeuLink Function, "
                        #                   "therefore runtime_params cannot be used".format(default(arg).__name__))
                    else:
                        raise ComponentError("Unrecognized object ({}) specified as function for {}".
                                             format(fct, self.name))

                    ignore_FUNCTION_PARAMS = True

            elif arg_name is FUNCTION_PARAMS:

                # If function was instantiated object, FUNCTION_PARAMS came from it, so ignore additional specification
                if ignore_FUNCTION_PARAMS:
                    continue
                # # MODIFIED 4/9/17 OLD:
                # params[FUNCTION_PARAMS] = kwargs[arg]
                # MODIFIED 4/9/17 NEW:
                params[FUNCTION_PARAMS] = ReadOnlyOrderedDict(name=FUNCTION_PARAMS)
                for param_name in sorted(list(kwargs[arg].keys())):
                    params[FUNCTION_PARAMS].__additem__(param_name,kwargs[arg][param_name])
                # MODIFIED 4/9/17 END

            # For standard params, assign arg and its default value to paramClassDefaults
            else:
                params[arg] = kwargs[arg]

        # Add or override arg values with any specified in params dict (including FUNCTION_PARAMS)
        if params_arg:

            try:
                # Update params[FUNCTION_PARAMS] with any from param_arg[FUNCTION_PARAMS] (specified in the constructor)
                for param_name in params_arg[FUNCTION_PARAMS].keys():
                    params[FUNCTION_PARAMS].__additem__(param_name, params_arg[FUNCTION_PARAMS][param_name])
                # Convert params_arg[FUNCTION_PARAMS] to ReadOnlyOrderedDict and update it with params[FUNCTION_PARAMS];
                #    this is needed so that when params is updated below,
                #    it updates with the full and updated params[FUNCTION_PARAMS] (i.e, a complete set, from above)
                #    and not just whichever ones were in params_arg[FUNCTION_PARAMS]
                #    (i.e., if the user just specified a subset)
                if isinstance(params_arg[FUNCTION_PARAMS], dict):
                    function_params = params_arg[FUNCTION_PARAMS]
                    params_arg[FUNCTION_PARAMS] = ReadOnlyOrderedDict(name=FUNCTION_PARAMS)
                    for param_name in sorted(list(function_params.keys())):
                        params_arg[FUNCTION_PARAMS].__additem__(param_name, function_params[param_name])
                for param_name in sorted(list(params[FUNCTION_PARAMS].keys())):
                    params_arg[FUNCTION_PARAMS].__additem__(param_name, params[FUNCTION_PARAMS][param_name])
            except KeyError:
                pass

            params.update(params_arg)

        # Save user-accessible params
        # self.user_params = params.copy()
        self.user_params = ReadOnlyOrderedDict(name=USER_PARAMS)
        for param_name in sorted(list(params.keys())):
            self.user_params.__additem__(param_name, params[param_name])


        # Cache a (deep) copy of the user-specified values;  this is to deal with the following:
        #    • _create_attributes_for_user_params assigns properties to each param in user_params;
        #    • the setter for those properties (in make_property) also assigns its value to its entry user_params;
        #    • paramInstanceDefaults are assigned to paramsCurrent in Component.__init__ assigns
        #    • since paramsCurrent is a ParamsDict, it assigns the values of its entries to the corresponding attributes
        #         and the setter assigns those values to the user_params
        #    • therefore, assignments of paramInstance defaults to paramsCurrent in __init__ overwrites the
        #         the user-specified vaules (from the constructor args) in user_params
        self.user_params_for_instantiation = OrderedDict()
        from collections import Iterable
        for param_name in sorted(list(self.user_params.keys())):
            param_value = self.user_params[param_name]
            if isinstance(param_value, (str, np.ndarray, tuple)):
                self.user_params_for_instantiation[param_name] = param_value
            elif isinstance(param_value, Iterable):
                self.user_params_for_instantiation[param_name] = type(self.user_params[param_name])()
                # DICT
                if isinstance(param_value, dict):
                    for k, v in param_value.items():
                        self.user_params_for_instantiation[param_name][k] = v
                elif isinstance(param_value, ReadOnlyOrderedDict):
                    for k in sorted(list(param_value.keys())):
                        self.user_params_for_instantiation[param_name].__additem__(k,param_value[k])
                # SET
                elif isinstance(param_value, set):
                    for i in param_value:
                        self.user_params_for_instantiation[param_name].add(i)
                # OTHER ITERABLE
                else:
                    for i in range(len(param_value)):
                        self.user_params_for_instantiation[param_name].append(param_value[i])
            else:
                self.user_params_for_instantiation[param_name] = param_value

        # Provide opportunity for subclasses to filter final set of params in class-specific way
        # Note:  this is done here to preserve identity of user-specified params assigned to user_params above
        self._filter_params(params)

        # Create property on self for each parameter in user_params
        self._create_attributes_for_user_params(**self.user_params)

        # Return params only for args:
        return params

    def _filter_params(self, params):
        """This provides an opportunity for subclasses to modify the final set of params in a class-specific way;

        Note:
        The default (here) allows user-specified params to override entries in paramClassDefaults with the same name
        """
        pass

    def _create_attributes_for_user_params(self, **kwargs):

        from PsyNeuLink.Components.Functions.Function import Function, Function_Base
        for arg_name, arg_value in kwargs.items():
            if not any(hasattr(parent_class, arg_name) for parent_class in self.__class__.mro()):
                setattr(self.__class__, arg_name, make_property(arg_name, arg_value))
            setattr(self, '_'+arg_name, arg_value)


    def _check_args(self, variable, params=None, target_set=None, context=None):
        """validate variable and params, instantiate variable (if necessary) and assign any runtime params

        Called by functions to validate variable and params
        Validation can be suppressed by turning parameter_validation attribute off
        target_set is a params dictionary to which params should be assigned;
           otherwise, they are assigned to paramsCurrent;

        Does the following:
        - instantiate variable (if missing or callable)
        - validate variable if PARAM_VALIDATION is set
        - assign runtime params to paramsCurrent
        - validate params if PARAM_VALIDATION is set


        :param variable: (anything but a dict) - variable to validate
        :param params: (dict) - params to validate
        :target_set: (dict) - set to which params should be assigned (default: self.paramsCurrent)
        :return:
        """

        # If function is called without any arguments, get default for variable
        if variable is None:
            variable = self.variableInstanceDefault # assigned by the Function class init when initializing

        # If the variable is a function, call it
        if callable(variable):
            variable = variable()

        # Validate variable if parameter_validation is set and the function was called with a variable
        if self.prefs.paramValidationPref and not variable is None:
            if context:
                context = context + SEPARATOR_BAR + FUNCTION_CHECK_ARGS
            else:
                context = FUNCTION_CHECK_ARGS
            self._validate_variable(variable, context=context)
        else:
            self.variable = variable

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
                # IMPLEMENTATION NOTE: FUNCTION_RUNTIME_PARAM_NOT_SUPPORTED
                #    At present, assignment of ``function`` as runtime param is not supported
                #        (this is because paramInstanceDefaults[FUNCTION] could be a class rather than an bound method;
                #        i.e., not yet instantiated;  could be rectified by assignment in _instantiate_function)
                if param_name is FUNCTION:
                    continue
                # If param is specified in runtime_params, then assign it
                if param_name in runtime_params:
                    self.paramsCurrent[param_name] = runtime_params[param_name]
                # Otherwise, (re-)assign to paramInstanceDefaults
                #    this insures that any params that were assigned as runtime on last execution are reset here
                #    (unless they have been assigned another runtime value)
                elif not self.runtimeParamStickyAssignmentPref:
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
                self.paramsCurrent[param_name] = self.paramInstanceDefaults[param_name]
            self.runtime_params_in_use = False

        # If parameter_validation is set and they have changed, then validate requested values and assign to target_set
        if self.prefs.paramValidationPref and params and not params is target_set:
            self._validate_params(request_set=params, target_set=target_set, context=context)


    def _instantiate_defaults(self,
                        variable=None,
                        request_set=None,
                        assign_missing=True,
                        target_set=None,
                        default_set=None,
                        context=None
                        ):
        """Validate variable and/or param defaults in requested set and assign values to params in target set

          Variable can be any type other than a dictionary (reserved for use as params)
          request_set must contain a dict of params to be assigned to target_set (??and paramInstanceDefaults??)
          If assign_missing option is set, then any params defined for the class
              but not included in the requested set are assigned values from the default_set;
              if request_set is None, then all values in the target_set are assigned from the default_set
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
                  calling with an empty request_set (or =None), paramInstanceDefaults for target_set,
                  and paramClassDefaults as default_set (although reset_params does the same thing)
          Class defaults can not be passed as target_set
              IMPLEMENTATION NOTE:  for now, treating class defaults as hard coded;
                                    could be changed in the future simply by commenting out code below

          If not context:  instantiates function and any states specified in request set
                           (if they have changed from the previous value(s))

        :param variable: (anything but a dict (variable) - value to assign as variableInstanceDefault
        :param request_set: (dict) - params to be assigned
        :param assign_missing: (bool) - controls whether missing params are set to default_set values (default: False)
        :param target_set: (dict) - param set to which assignments should be made
        :param default_set: (dict) - values used for params missing from request_set (only if assign_missing is True)
        :return:
        """

        # Make sure all args are legal
        if variable is not None:
            if isinstance(variable,dict):
                raise ComponentError("Dictionary passed as variable; probably trying to use param set as 1st argument")
        if request_set:
            if not isinstance(request_set, dict):
                raise ComponentError("requested parameter set must be a dictionary")
        if target_set:
            if not isinstance(target_set, dict):
                raise ComponentError("target parameter set must be a dictionary")
        if default_set:
            if not isinstance(default_set, dict):
                raise ComponentError("default parameter set must be a dictionary")

        # IMPLEMENTATION NOTE:  REMOVE
        # # Enforce implementation of variableEncodingDim and valueEncodingDim:
        # try:
        #     self.variableEncodingDim
        # except AttributeError:
        #     raise ComponentError("{0} or its base class must implement variableEncodingDim".
        #                         format(self.__class__.__name__))
        # try:
        #     self.valueEncodingDim
        # except AttributeError:
        #     raise ComponentError("{0} or its base class must implement valueEncodingDim".
        #                         format(self.__class__.__name__))


        # VALIDATE VARIABLE (if not called from assign_params)
        if not COMMAND_LINE in context:
            # if variable has been passed then validate and, if OK, assign as variableInstanceDefault
            self._validate_variable(variable, context=context)
            if variable is None:
                self.variableInstanceDefault = self.variableClassDefault
            else:
                # MODIFIED 6/9/16 (CONVERT TO np.ndarray)
                self.variableInstanceDefault = self.variable

        # If no params were passed, then done
        if request_set is None and  target_set is None and default_set is None:
            return

        # GET AND VALIDATE PARAMS

        # Assign param defaults for target_set and default_set
        if target_set is None:
            target_set = self.paramInstanceDefaults
        if target_set is self.paramClassDefaults:
            raise ComponentError("Altering paramClassDefaults not permitted")

        # # MODIFIED 1/10/17 OLD:
        # if default_set is None:
        #     default_set = self.paramInstanceDefaults
        # # MODIFIED 1/10/17 NEW:
        # # If called from assign_params, restrict to user_params
        # #   as those are the only ones that should be modifiable
        # #   (and are included paramClassDefaults, which will be tested in validate_params)
        # if default_set is None:
        #     if COMMAND_LINE in context:
        #         default_set = self.user_params
        # # Otherwise, use paramInstanceDefaults (i.e., full set of implemented params)
        #     else:
        #         default_set = self.paramInstanceDefaults
        # MODIFIED 1/10/17 NEWER:
        # If called from assign_params, restrict to user_params
        #   as those are the only ones that should be modifiable
        #   (and are included paramClassDefaults, which will be tested in validate_params)
        if default_set is None:
            if COMMAND_LINE in context:
                default_set = {}
                for param_name in request_set:
                    default_set[param_name] = self.paramInstanceDefaults[param_name]
        # Otherwise, use paramInstanceDefaults (i.e., full set of implemented params)
            else:
                default_set = self.paramInstanceDefaults
        # MODIFIED 1/10/17 END


        # IMPLEMENT: IF not context, DO RECURSIVE UPDATE OF DEFAULT WITH REQUEST, THEN SKIP NEXT IF (MAKE IT elif)
        #            (update default_set with request_set)
        #            BUT STILL NEED TO ADDRESS POSSIBLE MISMATCH OF FUNCTION AND FUNCTION_PARAMS (PER BELOW)
        #            IF FUNCTION_PARAMS ARE NOT IN REQUEST SET, AS ONES FROM DEFAULT WILL BE FOR DIFFERENT FUNCTION
        #            AND SHOULD BE CHECKED ANYHOW
        #
        # FROM: http://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
        # import collections
        #
        # def update(d, u):
        #     for k, v in u.items():
        #         if isinstance(v, collections.Mapping):
        #             r = update(d.get(k, {}), v)
        #             d[k] = r
        #         else:
        #             d[k] = u[k]
        #     return d


        # If assign_missing option is set,
        #  assign value from specified default set to any params missing from request set
        # Note:  do this before validating execute method and params, as some params may depend on others being present
        if assign_missing:
            if not request_set:
                request_set = {}

            # FIX: DO ALL OF THIS IN VALIDATE PARAMS?
            # FIX:    ?? HOWEVER, THAT MEANS MOVING THE ENTIRE IF STATEMENT BELOW TO THERE
            # FIX:    BECAUSE OF THE NEED TO INTERCEPT THE ASSIGNMENT OF functionParams FROM paramClassDefaults
            # FIX:    ELSE DON'T KNOW WHETHER THE ONES IN request_set CAME FROM CALL TO __init__() OR paramClassDefaults
            # FIX: IF functionParams ARE SPECIFIED, NEED TO FLAG THAT function != defaultFunction
            # FIX:    TO SUPPRESS VALIDATION OF functionParams IN _validate_params (THEY WON'T MATCH paramclassDefaults)
            # Check if function matches one in paramClassDefaults;
            #    if not, suppress assignment of functionParams from paramClassDefaults, as they don't match the function
            # Note: this still allows functionParams included as arg in call to __init__ to be assigned

            # REFERENCE: Conditions for assignment of default function and functionParams
            #     A) default function, default functionParams
            #         example: Projection.__inits__
            #     B) default function, no default functionParams
            #         example: none??
            #     C) no default function, default functionParams
            #         example: ??DDM
            #     D) no default function, no default functionParams
            #         example: System, Process, MonitoringMechanism, WeightedErrorMechanism

            self.assign_default_FUNCTION_PARAMS = True

            try:
                # # MODIFIED 11/30/16 OLD:
                # function = request_set[FUNCTION]
                # MODIFIED 11/30/16 NEW:
                # Copy to keep record of request_set function for comparison below, after request_set has been updated
                import copy
                function = copy.deepcopy(request_set[FUNCTION])
                # MODIFIED 11/30/16 END
            except KeyError:
                # If there is no function specified, then allow functionParams
                # Note: this occurs for objects that have "hard-coded" functions
                self.assign_default_FUNCTION_PARAMS = True
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
                    # This will be caught, and an exception raised, in _validate_params()
                    pass
                else:
                    # Get default function class
                    if inspect.isclass(function):
                        default_function_class = default_function
                    else:
                        default_function_class = default_function.__class__

                    # If function's class != default function's class, suppress assignment of default functionParams
                    if function_class != default_function_class:
                        self.assign_default_FUNCTION_PARAMS = False

            # Sort to be sure FUNCTION is processed before FUNCTION_PARAMS,
            #    so that latter are evaluated in context of former
            for param_name, param_value in sorted(default_set.items()):

                # MODIFIED 11/30/16 NEW:
                # FUNCTION class has changed, so replace rather than update FUNCTION_PARAMS
                if param_name is FUNCTION:
                    try:
                        if function_class != default_function_class and COMMAND_LINE in context:
                            from PsyNeuLink.Components.Functions.Function import Function_Base
                            if isinstance(function, Function_Base):
                                request_set[FUNCTION] = function.__class__
                            default_set[FUNCTION_PARAMS] = function.user_params
                    # function not yet defined, so allow FUNCTION_PARAMS)
                    except UnboundLocalError:
                        pass
                # FIX: MAY NEED TO ALSO ALLOW assign_default_FUNCTION_PARAMS FOR COMMAND_LINE IN CONTEXT
                # MODIFIED 11/30/16 END

                if param_name is FUNCTION_PARAMS and not self.assign_default_FUNCTION_PARAMS:
                    continue

                # MODIFIED 11/29/16 NEW:
                # Don't replace requested entry with default
                if param_name in request_set:
                    continue
                # MODIFIED 11/29/16 END

                # Add to request_set any entries it is missing fron the default_set
                request_set.setdefault(param_name, param_value)
                # Update any values in a dict
                if isinstance(param_value, dict):
                    for dict_entry_name, dict_entry_value in param_value.items():
                        # MODIFIED 11/29/16 NEW:
                        # Don't replace requested entries
                        if dict_entry_name in request_set[param_name]:
                            continue
                        # MODIFIED 11/29/16 END
                        request_set[param_name].setdefault(dict_entry_name, dict_entry_value)



        # VALIDATE PARAMS

        # if request_set has been passed or created then validate and, if OK, assign to targets
        if request_set:
            self._validate_params(request_set, target_set, context=context)
            # Variable passed validation, so assign as instance_default


    @tc.typecheck
    def assign_params(self, request_set:tc.optional(dict)=None):
        """Validates specified params, adds them TO paramInstanceDefaults, and instantiates any if necessary

        Call _instantiate_defaults with context = COMMAND_LINE, and "validated_set" as target_set.
        Update paramInstanceDefaults with validated_set so that any instantiations (below) are done in proper context.
        Instantiate any items in request set that require it (i.e., function or states).

        """
        from PsyNeuLink.Components.Functions.Function import Function
        context=COMMAND_LINE

        if not request_set:
            if self.verbosePref:
                warnings.warn("No params specified")
            return

        import copy
        validated_set = {}

        self._instantiate_defaults(request_set=request_set,
                             target_set=validated_set,
                             # assign_missing=False,
                             context=context)

        self.paramInstanceDefaults.update(validated_set)
        self.paramsCurrent = self.paramInstanceDefaults


        # 4/8/17 FIX: THIS SHOULD NOW ASSIGN TO PARAMS DIRECTLY:
        for param_name, param_value in validated_set.items():
            setattr(self, param_name, param_value)
            # self.user_params[param_name]=param_value
            # self.user_params.__additem__(param_name, param_value)

        # FIX: THIS NEEDS TO BE HANDLED BETTER:
        # FIX: DEAL WITH INPUT_STATES AND PARAMETER_STATES DIRECTLY (RATHER THAN VIA instantiate_attributes_before...)
        # FIX: SAME FOR FUNCTIONS THAT NEED TO BE "WRAPPED"
        # FIX: FIGURE OUT HOW TO DEAL WITH INPUT_STATES
        # FIX: FOR PARAMETER_STATES:
        #        CALL THE FOLLOWING FOR EACH PARAM:
        # FIX: NEED TO CALL

        validated_set_param_names = list(validated_set.keys())

        if INPUT_STATES in validated_set_param_names:
            self._instantiate_attributes_before_function()

        # Give owner a chance to instantiate function and/or function params
        # (e.g., wrap in UserDefineFunction, as per EVCMechanism)
        elif any(isinstance(param_value, (function_type, Function)) or
                      (inspect.isclass(param_value) and issubclass(param_value, Function))
                 for param_value in validated_set.values()):
            self._instantiate_attributes_before_function()

        # NEED TO DO THIS NO MATTER WHAT, SINCE NEED PARAMETER STATES FOR ALL NEW PARAMS
        # AS IT IS NOW, _instantiate_parameter_states ignores existing parameterStates
        #               but this may cause it to ignore FUNCTION_PARAMS when FUNCTION has changed
        from PsyNeuLink.Components.States.ParameterState import _instantiate_parameter_state
        for param_name in validated_set_param_names:
            _instantiate_parameter_state(owner=self,
                                         param_name=param_name,
                                         param_value=validated_set[param_name],
                                         context=context)

        # If the objects function is being assigned, and it is a class, instantiate it as a Function object
        if FUNCTION in validated_set and inspect.isclass(self.function):
            self._instantiate_function(context=COMMAND_LINE)

        if OUTPUT_STATES in validated_set:
            self._instantiate_attributes_after_function()



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
            raise ComponentError("Must be called with a valid ResetMode")

        if mode == ResetMode.CURRENT_TO_INSTANCE_DEFAULTS:
            self.params_current = self.paramInstanceDefaults.copy()
        elif mode == ResetMode.INSTANCE_TO_CLASS:
            self.paramInstanceDefaults = self.paramClassDefaults.copy()
        elif mode == ResetMode.ALL_TO_CLASS_DEFAULTS:
            self.params_current = self.paramClassDefaults.copy()
            self.paramInstanceDefaults = self.paramClassDefaults.copy()

    def _validate_variable(self, variable, context=None):
        """Validate variable and assign validated values to self.variable

        Convert variableClassDefault specification and variable (if specified) to list of 1D np.ndarrays:

        VARIABLE SPECIFICATION:                                        ENCODING:
        Simple value variable:                                         0 -> [array([0])]
        Single state array (vector) variable:                         [0, 1] -> [array([0, 1])
        Multiple state variables, each with a single value variable:  [[0], [0]] -> [array[0], array[0]]

        Perform top-level type validation of variable against the variableClassDefault;
            if the type is OK, the value is assigned to self.variable (which should be used by the function)
        This can be overridden by a subclass to perform more detailed checking (e.g., range, recursive, etc.)
        It is called only if the parameter_validation attribute is :keyword:`True` (which it is by default)

        IMPLEMENTATION NOTES:
           * future versions should add hierarchical/recursive content (e.g., range) checking
           * add request/target pattern?? (as per _validate_params) and return validated variable?

        :param variable: (anything other than a dictionary) - variable to be validated:
        :param context: (str)
        :return none:
        """

        if inspect.isclass(variable):
            raise ComponentError("Assignment of class ({}) as a variable (for {}) is not allowed".
                                 format(variable.__name__, self.name))

        pre_converted_variable_class_default = self.variableClassDefault

        # FIX: SAYS "list of np.ndarrays" BELOW, WHICH WOULD BE A 2D ARRAY, BUT CONVERSION BELOW ONLY INDUCES 1D ARRAY
        # FIX: NOTE:  VARIABLE (BELOW) IS CONVERTED TO ONLY 1D ARRAY
        # Convert variableClassDefault to list of np.ndarrays
        # self.variableClassDefault = convert_to_np_array(self.variableClassDefault, 1)

        # If variable is not specified, then:
        #    - assign to (??now np-converted version of) variableClassDefault
        #    - mark as not having been specified
        #    - return
        self._variable_not_specified = False
        if variable is None:
            self.variable = self.variableClassDefault
            self._variable_not_specified = True
            return

        # Otherwise, do some checking on variable before converting to np.ndarray

        # If variable is a ParamValueProjection tuple, get value:
        from PsyNeuLink.Components.Mechanisms.Mechanism import ParamValueProjection
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
                    format(self.componentName, context, pre_converted_variable_class_default.__class__.__name__)
                raise ComponentError(message)

        self.variable = variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate params and assign validated values to targets,

        This performs top-level type validation of params against the paramClassDefaults specifications:
            - checks that param is listed in paramClassDefaults
            - checks that param value is compatible with on in paramClassDefaults
            - if param is a dict, checks entries against corresponding entries paramClassDefaults
            - if all is OK, the value is assigned to the target_set (if it has been provided)
            - otherwise, an exception is raised

        This can be overridden by a subclass to perform more detailed checking (e.g., range, recursive, etc.)
        It is called only if the parameter_validation attribute is :keyword:`True` (which it is by default)

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
                # MODIFIED 11/30/16 NEW:
                # function is a class, so function_params has not yet been implemented
                if param_name is FUNCTION_PARAMS and inspect.isclass(self.function):
                    continue
                # MODIFIED 11/30/16 END
                raise ComponentError("{0} is not a valid parameter for {1}".format(param_name, self.__class__.__name__))

            # The value of the param is None in paramClassDefaults: suppress type checking
            # DOCUMENT:
            # IMPLEMENTATION NOTE: this can be used for params with multiple possible types,
            #                      until type lists are implemented (see below)
            if self.paramClassDefaults[param_name] is None or self.paramClassDefaults[param_name] is NotImplemented:
                if self.prefs.verbosePref:
                    warnings.warn("{0} is specified as None for {1} which suppresses type checking".
                                  format(param_name, self.name))
                if target_set is not None:
                    target_set[param_name] = param_value
                continue

            # If the value in paramClassDefault is a type, check if param value is an instance of it
            if inspect.isclass(self.paramClassDefaults[param_name]):
                if isinstance(param_value, self.paramClassDefaults[param_name]):
                    # MODIFIED 2/14/17 NEW:
                    target_set[param_name] = param_value
                    # MODIFIED 2/14/17 END
                    continue
                # If the value is a Function class, allow any instance of Function class
                from PsyNeuLink.Components.Functions.Function import Function_Base
                if issubclass(self.paramClassDefaults[param_name], Function_Base):
                    # if isinstance(param_value, (function_type, Function_Base)):  <- would allow function of any kind
                    if isinstance(param_value, Function_Base):
                        # MODIFIED 2/14/17 NEW:
                        target_set[param_name] = param_value
                        # MODIFIED 2/14/17 END
                        continue

            # If the value in paramClassDefault is an object, check if param value is the corresponding class
            # This occurs if the item specified by the param has not yet been implemented (e.g., a function)
            if inspect.isclass(param_value):
                if isinstance(self.paramClassDefaults[param_name], param_value):
                    continue

            # If the value is a projection, projection class, or a keyword for one, for anything other than
            #    the FUNCTION param (which is not allowed to be specified as a projection)
            #    then simply assign value to paramClassDefault (implication of not specifying it explicitly);
            #    this also allows it to pass the test below and function execution to occur for initialization;
            from PsyNeuLink.Components.Projections.Projection import Projection, ProjectionRegistry
            # from PsyNeuLink.Components.Projections.ControlProjection import ControlProjection
            # from PsyNeuLink.Components.Projections.LearningProjection import LearningProjection
            if (((isinstance(param_value, str) and
                          param_value in {CONTROL_PROJECTION, LEARNING_PROJECTION, LEARNING}) or
                isinstance(param_value, Projection) or  # These should be just ControlProjection or LearningProjection
                inspect.isclass(param_value) and issubclass(param_value,(Projection)))
                and not param_name is FUNCTION):
                param_value = self.paramClassDefaults[param_name]

            # If self is a Function:
            #    if param is a tuple, get its value (since Functions can't take projection specifications)
            #    if param is a class ref for function, instantiate it as the function
            from PsyNeuLink.Components.Functions.Function import Function_Base
            from PsyNeuLink.Components.ShellClasses import ParamValueProjection
            if isinstance(self, Function_Base):
                if isinstance(param_value, (ParamValueProjection, tuple)):
                    # Get value and assign to param_value for compatibility check below
                    param_value = self._get_param_value_from_tuple(param_value)
                # Value is a class (presumably a Function), so instantiate it as value
                elif (inspect.isclass(param_value) and
                          issubclass(param_value, self.paramClassDefaults[param_name])):
                    # Assign instance to target and move on
                    #  (compatiblity check no longer needed and can't handle function)
                    target_set[param_name] = param_value()
                    continue

            # Value is a ParamValueProjection or 2-item tuple, so extract its value for validation below
            if isinstance(param_value, (ParamValueProjection, tuple)):
                param_value = self._get_param_value_from_tuple(param_value)

            # MODIFIED 12/11/16 OLD:  NO LONGER NEED AS "LISTIFICATION" NOW OCCURS IN assign_args_to_param_dicts
            # # If it is a state specification for a mechanism with a single item, convert to list format
            # if param_name in {INPUT_STATES, OUTPUT_STATES}:
            #     from PsyNeuLink.Components.States.State import State_Base
            #     if (isinstance(param_value, (str, State_Base, dict)) or
            #             is_numeric(param_value) or
            #             (inspect.isclass(param_value) and issubclass(param_value, State_Base))):
            #         param_value = [param_value]
            #         # request_set[param_name] = [param_value]

            # Check if param value is of same type as one with the same name in paramClassDefaults;
            #    don't worry about length

            if iscompatible(param_value, self.paramClassDefaults[param_name], **{kwCompatibilityLength:0}):
                # If param is a dict, check that entry exists in paramClassDefaults
                # IMPLEMENTATION NOTE:
                #    - currently doesn't check compatibility of value with paramClassDefaults
                #      since params can take various forms (e.g., value, tuple, etc.)
                #    - re-instate once paramClassDefaults includes type lists (as per requiredClassParams)
                if isinstance(param_value, dict):

                    # If assign_default_FUNCTION_PARAMS is False, it means that function's class is
                    #     compatible but different from the one in paramClassDefaults;
                    #     therefore, FUNCTION_PARAMS will not match paramClassDefaults;
                    #     instead, check that functionParams are compatible with the function's default params
                    if param_name is FUNCTION_PARAMS and not self.assign_default_FUNCTION_PARAMS:
                        # Get function:
                        try:
                            function = request_set[FUNCTION]
                        except KeyError:
                            # If no function is specified, self.assign_default_FUNCTION_PARAMS should be True
                            # (see _instantiate_defaults above)
                            raise ComponentError("PROGRAM ERROR: No function params for {} so should be able to "
                                                "validate {}".format(self.name, FUNCTION_PARAMS))
                        else:
                            for entry_name, entry_value in param_value.items():
                                try:
                                    function.paramClassDefaults[entry_name]
                                except KeyError:
                                    raise ComponentError("{0} is not a valid entry in {1} for {2} ".
                                                        format(entry_name, param_name, self.name))
                                # add [entry_name] entry to [param_name] dict
                                else:
                                    try:
                                        target_set[param_name][entry_name] = entry_value
                                    # [param_name] dict not yet created, so create it
                                    except KeyError:
                                        target_set[param_name] = {}
                                        target_set[param_name][entry_name] = entry_value
                                    # target_set None
                                    except TypeError:
                                        pass
                    else:
                        for entry_name, entry_value in param_value.items():
                            # Make sure [entry_name] entry is in [param_name] dict in paramClassDefaults
                            try:
                                self.paramClassDefaults[param_name][entry_name]
                            except KeyError:
                                raise ComponentError("{0} is not a valid entry in {1} for {2} ".
                                                    format(entry_name, param_name, self.name))
                            # TBI: (see above)
                            # if not iscompatible(entry_value,
                            #                     self.paramClassDefaults[param_name][entry_name],
                            #                     **{kwCompatibilityLength:0}):
                            #     raise ComponentError("{0} ({1}) in {2} of {3} must be a {4}".
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
                                # target_set None
                                except TypeError:
                                    pass

                elif target_set is not None:
                    # Copy any iterables so that deletions can be made to assignments belonging to the instance
                    from collections import Iterable
                    if not isinstance(param_value, Iterable) or isinstance(param_value, str):
                        target_set[param_name] = param_value
                    else:
                        target_set[param_name] = param_value.copy()
            # MODIFIED 4/3/17 KM adding list/array parameter for noise


            # If param is a function_type, allow any other function_type
            # MODIFIED 1/9/16 NEW:
            elif callable(param_value):
                target_set[param_name] = param_value
            # MODIFIED 1/9/16 END
            elif callable(param_value.function):
                target_set[param_name] = param_value
            # Parameter is not a valid type
            else:
                if type(self.paramClassDefaults[param_name]) is type:
                    type_name = 'the name of a subclass of ' + self.paramClassDefaults[param_name].__base__.__name__
                else:
                    type_name = 'an instance of  ' + self.paramClassDefaults[param_name].__name__
                raise ComponentError("Value of {0} param ({1}) must be {2} ".
                                    format(param_name, param_value, type_name))

    def _get_param_value_from_tuple(self, param_spec):
        """Returns param value (first item) of either a ParamValueProjection or an unnamed (value, projection) tuple
        """
        from PsyNeuLink.Components.Mechanisms.Mechanism import ParamValueProjection
        from PsyNeuLink.Components.Projections.Projection import Projection
        # from PsyNeuLink.Components.Projections.ControlProjection import ControlProjection
        # from PsyNeuLink.Components.Projections.LearningProjection import LearningProjection

        if isinstance(param_spec, ParamValueProjection):
            value =  param_spec.value
        elif (isinstance(param_spec, tuple) and len(param_spec) is 2 and
                (param_spec[1] in {CONTROL_PROJECTION, LEARNING_PROJECTION} or
                     isinstance(param_spec[1], Projection) or
                     (inspect.isclass(param_spec[1]) and issubclass(param_spec[1], Projection)))
              ):
            value =  param_spec[0]
        else:
            value = param_spec

        return value

    def _validate_function(self, context=None):
        """Check that either params[FUNCTION] and/or self.execute are implemented

        # FROM _validate_params:
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
                is/are carried out in _instantiate_function

        :return:
        """
        # Check if params[FUNCTION] is specified
        try:
            param_set = PARAMS_CURRENT
            function = self._check_FUNCTION(param_set)
            if not function:
                param_set = PARAM_INSTANCE_DEFAULTS
                function, param_set = self._check_FUNCTION(param_set)
                if not function:
                    param_set = PARAM_CLASS_DEFAULTS
                    function, param_set = self._check_FUNCTION(param_set)

        except KeyError:
            # FUNCTION is not specified, so try to assign self.function to it
            try:
                function = self.function
            except AttributeError:
                # self.function is also missing, so raise exception
                raise ComponentError("{} must either implement a function method, specify one as the FUNCTION param in"
                                    " paramClassDefaults, or as the default for the function argument in its init".
                                    format(self.__class__.__name__, FUNCTION))
            else:
                # self.function is None
                # IMPLEMENTATION NOTE:  This is a coding error;  self.function should NEVER be assigned None
                if function is None:
                    raise("PROGRAM ERROR: either {0} must be specified or {1}.function must be implemented for {2}".
                          format(FUNCTION,self.__class__.__name__, self.name))
                # self.function is OK, so return
                elif (isinstance(function, Component) or
                        isinstance(function, function_type) or
                        isinstance(function, method_type)):
                    self.paramsCurrent[FUNCTION] = function
                    return
                # self.function is NOT OK, so raise exception
                else:
                    raise ComponentError("{0} not specified and {1}.function is not a Function object or class"
                                        "or valid method in {2}".
                                        format(FUNCTION, self.__class__.__name__, self.name))

        # paramsCurrent[FUNCTION] was specified, so process it
        else:
            # FUNCTION is valid:
            if function:
                # - if other than paramsCurrent, report (if in VERBOSE mode) and assign to paramsCurrent
                if param_set is not PARAMS_CURRENT:
                    if self.prefs.verbosePref:
                        warnings.warn("{0} ({1}) is not a Function object or a valid method; {2} ({3}) will be used".
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
                    raise ComponentError("{0} ({1}) is not a Function object or class or valid method, "
                                        "and {2}.function is not implemented for {3}".
                                        format(FUNCTION,
                                               self.paramsCurrent[FUNCTION],
                                               self.__class__.__name__,
                                               self.name))
                else:
                    # self.function is there and is:
                    # - OK, so just warn that FUNCTION was no good and that self.function will be used
                    if (isinstance(function, Component) or
                            isinstance(function, function_type) or
                            isinstance(function, method_type)):
                        if self.prefs.verbosePref:
                            warnings.warn("{0} ({1}) is not a Function object or class or valid method; "
                                          "{2}.function will be used instead".
                                          format(FUNCTION,
                                                 self.paramsCurrent[FUNCTION],
                                                 self.__class__.__name__))
                    # - NOT OK, so raise exception (FUNCTION and self.function were both no good)
                    else:
                        raise ComponentError("Neither {0} ({1}) nor {2}.function is a Function object or class "
                                            "or a valid method in {3}".
                                            format(FUNCTION, self.paramsCurrent[FUNCTION],
                                                   self.__class__.__name__, self.name))

    def _check_FUNCTION(self, param_set):
        """Check FUNCTION param is a Function,
        """

        function = getattr(self, param_set)[FUNCTION]
        # If it is a Function object, OK so return

        if (isinstance(function, COMPONENT_BASE_CLASS) or
                isinstance(function, function_type) or
                isinstance(function, method_type) or
                callable(function)):

            return function
        # Try as a Function class reference
        else:
            try:
                is_subclass = issubclass(self.paramsCurrent[FUNCTION], COMPONENT_BASE_CLASS)
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

    def _instantiate_attributes_before_function(self, context=None):
        pass

    def _instantiate_function(self, context=None):
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
            - if self.function IS NOT implemented: program error (should have been caught in _validate_function)
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
                if issubclass(type(function.__self__), COMPONENT_BASE_CLASS):
                    pass
                # If it is NOT a subclass of Function,
                # - issue warning if in VERBOSE mode
                # - pass through to try self.function below
                else:
                    if self.prefs.verbosePref:
                        warnings.warn("{0} ({1}) is not a subclass of Function".
                                      format(FUNCTION,
                                             self.paramsCurrent[FUNCTION].__class__.__name__,
                                             self.name))

                    function = None

            # If FUNCTION is a Function object, assign it to self.function (overrides hard-coded implementation)
            elif isinstance(function, COMPONENT_BASE_CLASS):
                self.function = function

            # If FUNCTION is a Function class:
            # - instantiate method using:
            #    - self.variable
            #    - params[FUNCTION_PARAMS] (if specified)
            # - issue warning if in VERBOSE mode
            # - assign to self.function and params[FUNCTION]
            elif inspect.isclass(function) and issubclass(function, COMPONENT_BASE_CLASS):
                #  Check if params[FUNCTION_PARAMS] is specified
                try:
                    function_param_specs = self.paramsCurrent[FUNCTION_PARAMS].copy()
                except (KeyError, AttributeError):
                    # FUNCTION_PARAMS not specified, so nullify
                    function_param_specs = {}
                else:
                    # FUNCTION_PARAMS are bad (not a dict):
                    if not isinstance(function_param_specs, dict):
                        # - nullify FUNCTION_PARAMS
                        function_param_specs = {}
                        # - issue warning if in VERBOSE mode
                        if self.prefs.verbosePref:
                            warnings.warn("{0} in {1} ({2}) is not a dict; it will be ignored".
                                          format(FUNCTION_PARAMS, self.name, function_param_specs))
                    # parse entries of FUNCTION_PARAMS dict
                    else:
                        # Get param value from any params specified as ParamValueProjection or (param, projection) tuple
                        from PsyNeuLink.Components.Projections.Projection import Projection
                        from PsyNeuLink.Components.Mechanisms.Mechanism import ParamValueProjection
                        for param_name, param_spec in function_param_specs.items():
                            if isinstance(param_spec, ParamValueProjection):
                                from PsyNeuLink.Components.States.ParameterState import ParameterState
                                function_param_specs[param_name] =  param_spec.value
                            if (isinstance(param_spec, tuple) and len(param_spec) is 2 and
                                    (param_spec[1] in {MAPPING_PROJECTION, CONTROL_PROJECTION, LEARNING_PROJECTION} or
                                         isinstance(param_spec[1], Projection) or
                                         (inspect.isclass(param_spec[1]) and issubclass(param_spec[1], Projection)))
                                ):
                                from PsyNeuLink.Components.States.ParameterState import ParameterState
                                function_param_specs[param_name] =  param_spec[0]

                # Instantiate function from class specification
                function_instance = function(variable_default=self.variable,
                                             params=function_param_specs,
                                             # IMPLEMENTATION NOTE:
                                             #    Don't bother with this, since it has to be assigned explicitly below
                                             #    anyhow, for cases in which function already exists
                                             #    and would require every function to have the owner arg in its __init__
                                             owner=self,
                                             context=context)
                self.paramsCurrent[FUNCTION] = function_instance.function

                # If in VERBOSE mode, report assignment
                if self.prefs.verbosePref:
                    object_name = self.name
                    if self.__class__.__name__ is not object_name:
                        object_name = object_name + " " + self.__class__.__name__
                    try:
                        object_name = object_name + " of " + self.owner.name
                    except AttributeError:
                        pass
                    warnings.warn("{0} assigned as function for {1}".
                                  format(self.paramsCurrent[FUNCTION].__self__.componentName,
                                         object_name))

            # FUNCTION is a generic function (presumably user-defined), so "wrap" it in UserDefinedFunction:
            #   Note: calling UserDefinedFunction.function will call FUNCTION
            elif inspect.isfunction(function):
                

                from PsyNeuLink.Components.Functions.Function import UserDefinedFunction
                self.paramsCurrent[FUNCTION] = UserDefinedFunction(function=function, context=context).function

            # If FUNCTION is NOT a Function class reference:
            # - issue warning if in VERBOSE mode
            # - pass through to try self.function below
            else:
                if self.prefs.verbosePref:
                    warnings.warn("{0} ({1}) is not a subclass of Function".
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
            # Note: this is a "sanity check," as this should have been checked in _validate_function (above)
            except AttributeError:
                raise ComponentError("{0} ({1}) is not a Function object or class, "
                                    "and {2}.function is not implemented".
                                    format(FUNCTION, self.paramsCurrent[FUNCTION],
                                           self.__class__.__name__))
            # If self.function is implemented, warn if in VERBOSE mode
            else:
                if self.prefs.verbosePref:
                    warnings.warn("{0} ({1}) is not a Function object or a specification for one; "
                                  "{1}.function ({}) will be used instead".
                                  format(FUNCTION,
                                         self.paramsCurrent[FUNCTION].__self__.componentName,
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
        # self.value = self.function(context=context+kwSeparator+COMPONENT_INIT)
        self.value = self.execute(context=context)
        if self.value is None:
            raise ComponentError("PROGRAM ERROR: Execute method for {} must return a value".format(self.name))
        self._value_template = self.value

        self.paramInstanceDefaults[FUNCTION] = self.function

        # For all components other than a Function itself, assign function_object and function_params
        from PsyNeuLink.Components.Functions.Function import Function
        if not isinstance(self, Function):
            self.function_object = self.function.__self__
            if not self.function_object.owner:
                self.function_object.owner = self
            elif self.function_object.owner != self:
                raise ComponentError("Function being assigned to {} ({}) belongs to another component: {}".
                                     format(self.name, self.function_object.name, self.function_object.owner.name))

            # IMPLEMENT:  PROGRAMMATICALLY ADD GETTER AND SETTER PROPERTY FOR EACH FUNCTION_PARAM HERE
            #             SEE learning_rate IN LearningMechanism FOR EXAMPLE
            # # MODIFIED 4/1/17 OLD:
            # self.function_params = self.function_object.user_params
            # self.paramInstanceDefaults[FUNCTION_PARAMS] = self.function_params
            # MODIFIED 4/8/17 NEWER:
            self.function_params = ReadOnlyOrderedDict(name='function_params')
            for param_name in sorted(list(self.function_object.user_params_for_instantiation.keys())):
                self.function_params.__additem__(param_name,
                                                 self.function_object.user_params_for_instantiation[param_name])
            self.paramInstanceDefaults[FUNCTION_PARAMS] = self.function_params
            # MODIFIED 4/1/17 END

    def _instantiate_attributes_after_function(self, context=None):
        pass

    def initialize(self):
        raise ComponentError("{} class does not support initialize() method".format(self.__class__.__name__))

    def execute(self, input=None, params=None, time_scale=None, context=None):
        raise ComponentError("{} class must implement execute".format(self.__class__.__name__))

    def _update_value(self, context=None):
        """Evaluate execute method
        """
        self.value = self.execute(context=context)

    # @property
    # def variable(self):
    #     return self._variable
    #
    # @variable.setter
    # def variable(self, value):
    #     self._variable = value

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
                warnings.warn('PreferenceSet {0} assigned to {1}'.format(pref_set.name, self.name))
            # Make sure that every pref attrib in PreferenceSet is OK
            for pref_name, pref_entry in self.prefs.__dict__.items():
                if '_pref' in pref_name:
                    value, err_msg = self.prefs.get_pref_setting_for_level(pref_name, pref_entry.level)
                    if err_msg and self.prefs.verbosePref:
                        warnings.warn(err_msg)
                    # FIX: VALUE RETURNED SHOULD BE OK, SO ASSIGN IT INSTEAD OF ONE IN pref_set??
                    # FIX: LEVEL SHOULD BE LOWER THAN REQUESTED;  REPLACE RAISE WITH WARNING TO THIS EFFECT
        else:
            raise ComponentError("Attempt to assign non-PreferenceSet {0} to {0}.prefs".
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
    def runtimeParamModulationPref(self):
        return self.prefs.runtimeParamModulationPref

    @runtimeParamModulationPref.setter
    def runtimeParamModulationPref(self, setting):
        self.prefs.runtimeParamModulationPref = setting

    @property
    def runtimeParamStickyAssignmentPref(self):
        return self.prefs.runtimeParamStickyAssignmentPref

    @runtimeParamStickyAssignmentPref.setter
    def runtimeParamStickyAssignmentPref(self, setting):
        self.prefs.runtimeParamStickyAssignmentPref = setting

COMPONENT_BASE_CLASS = Component


# Autoprop
# per Bryn Keller

docs = {'foo': 'Foo controls the fooness, as modulated by the the bar',
        'bar': 'Bar none, the most important property'}

def make_property(name, default_value):
    backing_field = '_' + name

    def getter(self):
        return getattr(self, backing_field)

    def setter(self, val):

        if self.paramValidationPref:
            self.assign_params(request_set={backing_field:val})
        else:
            setattr(self, backing_field, val)


        # Update user_params dict with new value
        self.user_params.__additem__(name, val)

        # If component is a Function and has an owner, update function_params dict for owner
        from PsyNeuLink.Components.Functions.Function import Function_Base
        if isinstance(self, Function_Base) and self.owner:
            self.owner.function_params.__additem__(name, val)


    # Create the property
    prop = property(getter).setter(setter)

    # # Install some documentation
    # prop.__doc__ = docs[name]
    return prop
