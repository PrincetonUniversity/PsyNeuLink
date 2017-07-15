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

Component is the base class for all of the objects used to create `Compositions <Composition>` in PsyNeuLink.
It defines a common set of attributes possessed, and methods used by all Component objects.

.. _Component_Creation:

Creating a Component
--------------------

A Component is never created by calling the constructor for the Component base class.  However, its ``__init__()``
method is always called when a Component subclass is instantiated; that, in turn, calls a standard set of methods
(listed `below <Component_Methods>`) as part of the initialization procedure.

.. _Component_Deferred_Init:

If information necessary to complete initialization is not specified in the constructor (e.g, the **owner** for a
`State <State.owner>`, or the **sender** or **receiver** for a `Projection <Projection_Structure>`), then its full
initialization is deferred until its the information is available (e.g., the `State` is assigned to a `Mechanism`, or
a `Projection` is assigned its `sender <Projection.sender>` and `receiver <Projection.receiver>`).  This allows
Components to be created before all of the information they require is available (e.g., at the beginning of a script).
However, for the Component to be operational, initialization must be completed its `deferred_init` method must be
called.  This is usually done automatically when the Component is assigned to another Component to which it belongs
(e.g., assigning a State to a Mechanism) or to a Composition (e.g., a Projection to the `pathway <Process.pahtway>`)
of a `Process`), as appropriate.

.. _Component_Structure:

Component Structure
-------------------

.. _Component_Attributes:

Component Attributes
~~~~~~~~~~~~~~~~~~~~

Every Component has the following set of core attributes that govern its operation:

.. _Component_Variable:

* **variable** - the value of the `variable <Component.variable>` attribute is used as the input to its
  `function <Component.function>`.  Specification of the **variable** argument in the constructor for a Component
  determines both its format (e.g., whether its value is numeric, its dimensionality and shape if it is an array,
  etc.) as well as its default value (the value used when the Component is executed and no input is provided), and
  takes precedence over the specification of `size <Component_Size>`.

.. _Component_Size:

* **size** - the dimension of the `variable <Component.variable>` attribute.  The **size** argument of the
  constructor for a Component can be used as a convenient method for specifying the `variable <Component>`, attribute
  in which case it will be assigned as an array of zeros of the specified size.  For example, setting  **size** = 3 is
  equivalent to setting **variable** = [0, 0, 0] and setting **size** = [4, 3] is equivalent to setting
  **variable* = [[0, 0, 0, 0], [0, 0, 0]].

.. _Component_Function:

* **function** - the `function <Component.function>` attribute determines the computation that a Component carries out.
  It is always a PsyNeuLink `Function <Function>` object (itself a PsyNeuLink Component).

  .. note::
     The `function <Component.function>` of a Component can be assigned either a `Function <Function>` object or any
     other callable object in python.  If the latter is assigned, it will be "wrapped" in a `UserDefinedFunction`.

  All Components have a default `function <Component.function>` (with a default set of parameters), that is used if it
  is not otherwise specified.  The `function <Component.function>` can be specified in the
  function argument of the constructor for the Component, using one of the following:

    * **class** - this must be a subclass of `Function <Function>`, as in the following example::

        my_component = SomeComponent(function=SomeFunction)

      This will create a default instance of the specified subclass, using default values for its parameters.
    |
    * **Function** - this can be either an existing `Function <Function>` object or the constructor for one, as in the
      following examples::

        my_component = SomeComponent(function=SomeFunction)

        or

        some_function = SomeFunction(some_param=1)
        my_component = SomeComponent(some_function)

      The specified Function will be used as a template to create a new Function object that is assigned to the
      `function_object` attribute of the Component, the `function <Function.function>` of which will be assigned as
      the `function <Component.function>` attribute of the Component.

      .. note::

        In the current implementation of PsyNeuLink, if a `Function <Function>` object (or the constructor for one) is
        used to specify the `function <Component.function>` attribute of a Component, the Function object specified (or
        created) is used to determine attributes of the Function object created for and assigned to the Component, but
        is not *itself* assigned to the Component.  This is so that `Functions <Function>` can be used as templates for
        more than one Component, without being assigned simultaneously to multiple Components.

  A `function <Component.function>` can also be specified in an entry of a
  `parameter specification dictionary <ParameterState_Specifying_Parameters>` assigned to the
  **params** argument of the constructor for the Component, with the keyword *FUNCTION* as its key, and one of the
  specifications above as its value, as in the following example::

        my_component = SomeComponent(params={FUNCTION:SomeFunction(some_param=1)})

.. _Component_Function_Params:

* **function_params** - the `function_params <Component.function>` attribute contains a dictionary of the parameters
  for the Component's `function <Component.function>` and their values.  Each entry is the name of a parameter, and
  its value the value of that parameter.  This dictionary is read-only. Changes to the value of the function's
  parameters must be made by assigning a value to the corresponding attribute of the Component's
  `function_object <Component.function_object>` attribute (e.g., myMechanism.function_object.my_parameter),
  or in a FUNCTION_PARAMS dict using its `assign_params` method.  The parameters for the function can be specified
  when the Component is created in one of the following ways:

  * in the **constructor** for a Function -- if that is used to specify the `function <Component.function>` argument,
    as in the following example::

        my_component = SomeComponent(function=SomeFunction(some_param=1, some_param=2)

  * in an argument of the **Component's constructor** -- if all of the allowable functions for a Component's
    `function <Component.function>` share some or all of their parameters in common, the shared paramters may appear
    as arguments in the constructor of the Component itself, which can be used to set their values.

  * in an entry of a `parameter specification dictionary <ParameterState_Specifying_Parameters>` assigned to the
    **params** argument of the constructor for the Component.  The entry must use the keyword
    FUNCTION_PARAMS as its key, and its value must be a dictionary containing the parameters and their values.
    The key for each entry in the FUNCTION_PARAMS dictionary must be the name of a parameter, and its value the
    parameter's value, as in the example below::

        my_component = SomeComponent(function=SomeFunction
                                     params={FUNCTION_PARAMS:{SOME_PARAM=1, SOME_OTHER_PARAM=2}})

  See `ParameterState_Specifying_Parameters` for details concerning different ways in which the value of a parameter
  can be specified.

.. _Component_Function_Object:

* **function_object** - the `function_object` attribute refers to the `Function <Function>` assigned to the Component;
  The Function's `function <Function.function>` is assigned to the `function <Component>` attribute of the
  Component. The  parameters of the Function can be modified by assigning values to the attributes corresponding to
  those parameters (see `function_params <Component.function_params>` above).

.. _Component_User_Params:

* **user_params** - the `user_params` attribute contains a dictionary of all of the user-modifiable attributes for the
  the Component.  This dictionary is read-only.  Changes to the value of an attribute must be made by assigning a
  value to the attribute directly, or using the Component's `assign_params <Component.assign_params>` method.
..
COMMENT:
  INCLUDE IN DEVELOPERS' MANUAL
    * **paramClassDefaults**

    * **paramInstanceDefaults**
COMMENT

.. _Component_Value:

* **value** - the `value <Component.value>` attribute contains the result (return value) of the Component's
  `function <Component.function>` after the function is called.
..

.. _Component_Name:

* **name** - the `name <Component.name>` attribute contains the name assigned to the Component when it was created.
  If it was not specified, a default is assigned by the registry for subclass (see :doc:`Registry <LINK>` for
  conventions used in assigning default names and handling of duplicate names).
..

.. _Component_Prefs:
* **prefs** - the `prefs <Components.prefs>` attribute contains the `PreferenceSet` assigned to the Component when
  it was created.  If it was not specified, a default is assigned using `classPreferences` defined in ``__init__.py``
  Each individual preference is accessible as an attribute of the Component, the name of which is the name of the
  preference (see `PreferenceSet <LINK>` for details).

COMMENT:
* **log**
COMMENT

.. _Component_Methods:

Component Methods
~~~~~~~~~~~~~~~~~

COMMENT:
   INCLUDE IN DEVELOPERS' MANUAL

    There are two sets of methods that belong to every Component: one set that is called when it is initialized; and
    another set that can be called to perform various operations common to all Components.  Each of these is described
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
    to the format expected for it by the Component (i.e., the type of the value and, if it is iterable, the type its
    elements and/or its length).  The value itself is not checked in any other way (e.g., whether it equals a particular
    value or falls in a specified range).  If the validation fails, and exception is raised.  Validation methods never
    make changes the actual value of an attribute, but they may change its format (e.g., from a list to an ndarray) to
    comply with requirements of the Component.

      * `_validate_variable <Component._validate_variable>` validates the value provided to the keyword:`variable`
        argument in the constructor for the Component.  If it is overridden, customized validation should generally
        performed *prior* to the call to super(), to allow final processing by the Component base class.

      * `_validate_params <Component._validate_params>` validates the value of any parameters specified in the
        constructor for the Component (whether they are made directly in the argument for a parameter, or in a
        `parameter specification dictionary <ParameterState_Specifying_Parameters>`.  If it is overridden by a subclass,
        customized validation should generally be performed *after* the call to super().

    * **Instantiation methods** create, assign, and/or perform *semantic* checks on the values of Component attributes.
      Semantic checks may include value and/or range checks, as well as checks of formatting and/or value
      compatibility with other attributes of the Component and/or the attributes of other Components (for example, the
      _instantiate_function method checks that the input of the Component's `function <Comonent.function>` is compatible
      with its `variable <Component.variable>`).

      * `_handle_size <Component._handle_size>` converts the keyword:`variable` and keyword:`size` arguments
        to the correct dimensions (for keyword:`Mechanism`, this is a 2D array and 1D array, respectively).
        If keyword:`variable` was not passed as an argument, this method attempts to infer keyword:`variable`
        from the keyword:`size` argument, and vice versa if the keyword:`size` argument is missing.
        The _handle_size method then checks that the keyword:`size` and keyword:`variable` arguments are compatible.

      * `_instantiate_defaults <Component._instantiate_defaults>` first calls the validation methods, and then
        assigns the default values for all of the attributes of the instance of the Component being created.

        _instantiate_attributes_before_function
        _instantiate_function
        _instantiate_attributes_after_function

    .. _Component_Callable_Methods:

    Callable Methods
    ^^^^^^^^^^^^^^^^

    initialize
COMMENT

.. _Component_Assign_Params:

* **assign_params** - assign the value of one or more parameters of a Component.  Each parameter is specified
  as an entry in a `parameter specification dictionary <ParameterState_Specifying_Parameters>` in the **request_set**
  argument;  parameters for the Component's `function <Component.function>` are specified as entries in a
  *FUNCTION_PARAMS* dict within **request_set** dict.
..
* **reset_params** - reset the value of all user_params to a set of default values as specified in its **mode**
  argument, using a value of `ResetMode <Component_ResetMode>`.

.. _Component_Execution:

Execution
---------

Calls the :keyword:`execute` method of the subclass that, in turn, calls its :keyword:`function`.


.. _Component_Class_Reference:

Class Reference
---------------

COMMENT:

This module defines the Component abstract class

It also contains:

- arg_name definitions for primary Component categories:
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

from collections import OrderedDict, Iterable
from PsyNeuLink.Globals.Utilities import *
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import *

component_keywords = {NAME, VARIABLE, VALUE, FUNCTION, FUNCTION_PARAMS, PARAMS, PREFS_ARG, CONTEXT}

class ResetMode(Enum):
    """
    .. _Component_ResetMode:

    ResetModes used for **reset_params**:

    .. _CURRENT_TO_INSTANCE_DEFAULTS:

    *CURRENT_TO_INSTANCE_DEFAULTS*
      • resets all paramsCurrent values to paramInstanceDefaults values.

    .. _INSTANCE_TO_CLASS:

    *INSTANCE_TO_CLASS*
      • resets all paramInstanceDefaults values to paramClassDefaults values.

    .. _ALL_TO_CLASS_DEFAULTS:

    *ALL_TO_CLASS_DEFAULTS*
      • resets all paramsCurrent and paramInstanceDefaults values to paramClassDefafults values

    """
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
    """Create, set and get attribute of owner for each key in dict

    Creates and maintains an interface to attributes of a Component via a dict:
        - any assignment to an entry of the dict creates or updates the value of the attribute with the name of the key
        - any query retrieves the value of the attribute with the name of the key
    Dict itself is maintained in self.data

    Notes:
    * This provides functionality similar to the __dict__ attribute of a python object,
        but is restricted to the attributes relevant to its role as a PsyNeuLink Component.
    * It insures that any instantiation of a function_params attribute is a ReadOnlyOrderedDict

    """

    def __init__(self, owner, dict=None):
        super().__init__()
        self.owner = owner
        if dict:
            self.update(dict)
        # if there is a function_params entry in the dict, ensure its entry is created as a ReadOnlyOrderedDict
        if dict and FUNCTION_PARAMS in dict:
            self[FUNCTION_PARAMS] = ReadOnlyOrderedDict(name=FUNCTION_PARAMS)
            for param_name in sorted(list(dict[FUNCTION_PARAMS].keys())):
                self[FUNCTION_PARAMS].__additem__(param_name, dict[FUNCTION_PARAMS][param_name])

    def __getitem__(self, key):

        try:
            # Try to retrieve from attribute of owner object
            return getattr(self.owner, key)
        except AttributeError:
            # If the owner has no such attribute, get from params dict entry
            return super().__getitem__(key)

    def __setitem__(self, key, item):

        # if key is function_params, make sure it creates a ReadOnlyOrderedDict for the value of the entry
        if key is FUNCTION_PARAMS:
            if not isinstance(item, (dict, UserDict)):
                raise ComponentError("Attempt to assign non-dict ({}) to {} attribute of {}".
                                     format(item, FUNCTION_PARAMS, self.owner.name))
            function_params = ReadOnlyOrderedDict(name=FUNCTION_PARAMS)
            for param_name in sorted(list(item.keys())):
                function_params.__additem__(param_name, item[param_name])
            item = function_params

        # keep local dict of entries
        super().__setitem__(key, item)
        # assign value to attrib
        setattr(self.owner, key, item)


parameter_keywords = set()

# suppress_validation_preference_set = ComponentPreferenceSet(prefs = {
#     kpParamValidationPref: PreferenceEntry(False,PreferenceLevel.INSTANCE),
#     kpVerbosePref: PreferenceEntry(False,PreferenceLevel.INSTANCE),
#     kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE)})


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

    COMMENT:
        Every Component is associated with:
         - child class componentName
         - type
         - input (self.variable)
         - execute (method): called to execute it;  it in turn calls self.function
         - function (method): carries out object's core computation
             it can be referenced either as self.function, self.params[FUNCTION] or self.paramsCurrent[FUNCTION]
         - function_object (Function): the object to which function belongs (and that defines it's parameters)
         - output (value: self.value)
         - output_values (return from self.execute: concatenated set of values of outputStates)
         - class and instance variable defaults
         - class and instance param defaults
        The Components's execute method (<subclass>.execute is the Component's primary method
            (e.g., it is the one called when process, mechanism, state and projections objects are updated);
            the following attributes for or associated with the method are defined for every Component object:
                + execute (method) - the execute method itself
                + value (value) - the output of the execute method
            the latter is used for typing and/or templating other variables (e.g., self.variable):
                type checking is generally done using Utilities.iscompatible(); for iterables (lists, tuples, dicts):
                    if the template (the "reference" arg) has entries (e.g., [1, 2, 3]), comparisons will include length
                    if the template is empty (e.g., [], {}, or ()), length will not be checked
                    if the template has only numbers, then the candidate must as well


        The Component itself can be called without any arguments (in which case it uses its instance defaults) or
            one or more variables (as defined by the subclass) followed by an optional params dictionary
        The variable(s) can be a function reference, in which case the function is called to resolve the value;
            however:  it must be "wrapped" as an item in a list, so that it is not called before being passed
                      it must of course return a variable of the type expected for the variable
        The size argument is an int or array of ints, which specify the size of variable and set variable to be array(s)
            of zeros.
        The default variableList is a list of default values, one for each of the variables defined in the child class
        The params argument is a dictionary; the key for each entry is the parameter name, associated with its value.
            + Component subclasses can define the param FUNCTION:<method or Function class>
        The Component can be called with a params argument, which should contain entries for one or more of its params;
            - those values will be assigned to paramsCurrent at run time (overriding previous values in paramsCurrent)
            - if the Component is called without a variable and/or params argument, it uses paramInstanceDefaults
        The instance defaults can be assigned at initialization or using the _instantiate_defaults class method;
            - if instance defaults are not assigned on initialization, the corresponding class defaults are assigned
        Parameters can be REQUIRED to be in paramClassDefaults (for which there is no default value to assign)
            - for all classes, by listing the name and type in requiredParamClassDefaultTypes dict of the Function class
            - in subclasses, by inclusion in requiredParamClassDefaultTypes (via copy and update) in class definition
            * NOTE: inclusion in requiredParamClasssDefault simply acts as a template;  it does NOT implement the param
        Each Component child class must initialize itself by calling super(childComponentName).__init__()
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
        + componentType - type of Component within a category
                             (e.g., TransferMechanism, MappingProjection, ControlProjection, etc.)
        + requiredParamClassDefaultTypes - dict of param names & types that all subclasses of Component must implement;
        + prev_context - str (primarily used to track and prevent recursive calls to assign_params from setters)

        # Prevent recursive calls from setters
        if self.prev_context == context:
            return
        

    Class methods:
        - _handle_size(size, variable)
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
    COMMENT

    Attributes
    ----------

    variable : 2d np.array
        see `variable <Component_Function>`

    size : int or array of ints
        see `size <Component_Size>`

    function : Function, function or method
        see `variable <Component_Function>`

    function_params : Dict[param_name: param_value]
        see `function_params <Component_Function_Params>`

    function_object : Function
        see `function_object <Component_Function_Object>`

    user_params : Dict[param_name: param_value]
        see `user_params <Component_User_Params>`

    value : 2d np.array
        see `value <Component_Value>`

    name : str
        see `name <Component_Name>`

    prefs : PreferenceSet
        see `prefs <Component_Prefs>`

    """

    #CLASS ATTRIBUTES
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

    # IMPLEMENTATION NOTE: This is needed so that the State class can be used with ContentAddressableList,
    #                      which requires that the attribute used for addressing is on the class;
    #                      it is also declared as a property, so that any assignments are validated to be strings,
    #                      insuring that assignment by one instance will not affect the value of others.
    name = None

    # IMPLEMENTATION NOTE: Primarily used to track and prevent recursive calls to assign_params from setters.
    prev_context = None

    def __init__(self,
                 variable_default,
                 param_defaults,
                 size=NotImplemented,  # 7/5/17 CW: this is a hack to check whether the user has passed in a size arg
                 name=None,
                 prefs=None,
                 context=None):
        """Assign default preferences; enforce required params; validate and instantiate params and execute method

        Initialization arguments:
        - variable_default (anything): establishes type for the variable, used for validation
        - size (int or list/array of ints): if specified, establishes variable if variable was not already specified
        - params_default (dict): assigned as paramInstanceDefaults
        Note: if parameter_validation is off, validation is suppressed (for efficiency) (Component class default = on)

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

        # These ensure that subclass values are preserved, while allowing them to be referred to below
        self.variableInstanceDefault = None
        self.paramInstanceDefaults = {}

        self._auto_dependent = False
        self._role = None

        # self.componentName = self.componentType
        try:
            self.componentName = self.componentName or self.componentType
        except AttributeError:
            self.componentName = self.componentType

        # ENFORCE REGISRY
        if self.__class__.__bases__[0].__bases__[0].__bases__[0].__name__ is 'ShellClass':
            try:
                self.__class__.__bases__[0].registry
            except AttributeError:
                raise ComponentError("{0} is a category class and so must implement a registry".
                                    format(self.__class__.__bases__[0].__name__))

        # ASSIGN PREFS

        # If a PreferenceSet was provided, assign to instance
        if isinstance(prefs, PreferenceSet):
            self.prefs = prefs
            # FIX:  CHECK LEVEL HERE??  OR DOES IT NOT MATTER, AS OWNER WILL BE ASSIGNED DYNAMICALLY??
        # Otherwise, if prefs is a specification dict instantiate it, or if it is None assign defaults
        else:
            self.prefs = ComponentPreferenceSet(owner=self, prefs=prefs, context=context)

        # ASSIGN LOG

        self.log = Log(owner=self)
        self.recording = False
        # Used by run to store return value of execute
        self.results = []


        # ENFORCE REQUIRED CLASS DEFAULTS

        # All subclasses must implement variableClassDefault
        # Do this here, as _validate_variable might be overridden by subclass
        try:
            if self.variableClassDefault is NotImplemented:
                raise ComponentError("variableClassDefault for {} must be assigned a value or \'None\'".
                                     format(self.componentName))
        except AttributeError:
            raise ComponentError("variableClassDefault must be defined for {} or its base class".
                                format(self.componentName))

        # CHECK FOR REQUIRED PARAMS

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
                raise ComponentError("Param \'{}\' must be in paramClassDefaults for {}".
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

        # If 'variable_default' was not specified, _handle_size() tries to infer 'variable_default' based on 'size'
        variable_default = self._handle_size(size, variable_default)

        # VALIDATE VARIABLE AND PARAMS, AND ASSIGN DEFAULTS

        # Validate the set passed in and assign to paramInstanceDefaults
        # By calling with assign_missing, this also populates any missing params with ones from paramClassDefaults
        self._instantiate_defaults(variable=variable_default,
               request_set=param_defaults,            # requested set
               assign_missing=True,                   # assign missing params from classPreferences to instanceDefaults
               target_set=self.paramInstanceDefaults, # destination set to which params are being assigned
               default_set=self.paramClassDefaults,   # source set from which missing params are assigned
               context=context)

        # SET CURRENT VALUES OF VARIABLE AND PARAMS

        self.variable = self.variableInstanceDefault
        # self.variable = self.variableInstanceDefault.copy()

        # self.paramsCurrent = self.paramInstanceDefaults
        self.paramsCurrent = self.paramInstanceDefaults.copy()

        self.runtime_params_in_use = False

        # VALIDATE FUNCTION (self.function and/or self.params[function, FUNCTION_PARAMS])
        self._validate_function(context=context)

        # INSTANTIATE ATTRIBUTES BEFORE FUNCTION
        # Stub for methods that need to be executed before instantiating function
        #    (e.g., _instantiate_sender and _instantiate_receiver in Projection)
        self._instantiate_attributes_before_function(context=context)

        # INSTANTIATE FUNCTION
        #    - assign initial function parameter values from ParameterStates,
        #    - assign function's output to self.value (based on call of self.execute)
        self._instantiate_function(context=context)

        # INSTANTIATE ATTRIBUTES AFTER FUNCTION
        # Stub for methods that need to be executed after instantiating function
        #    (e.g., instantiate_output_state in Mechanism)
        self._instantiate_attributes_after_function(context=context)

    def __repr__(self):
        return '({0} {1})'.format(type(self).__name__, self.name)
        #return '{1}'.format(type(self).__name__, self.name)

    # IMPLEMENTATION NOTE: (7/7/17 CW) Due to System and Process being initialized with size at the moment (which will
    # be removed later), I’m keeping _handle_size in Component.py. I’ll move the bulk of the function to Mechanism
    # through an override, when Composition is done. For now, only State.py overwrites _handle_size().
    def _handle_size(self, size, variable):
        """ If variable is None, _handle_size tries to infer variable based on the **size** argument to the
            __init__() function. This method is overwritten in subclasses like Mechanism and State.
            If self is a Mechanism, it converts variable to a 2D array, (for a Mechanism, variable[i] represents
            the input from the i-th input state). If self is a State, variable is a 1D array and size is a length-1 1D
            array. It performs some validations on size and variable as well. This function is overridden in State.py.
            If size is NotImplemented (usually in the case of Projections/Functions), then this function passes without
            doing anything. Be aware that if size is NotImplemented, then variable is never cast to a particular shape.
        """
        # TODO: to get rid of the allLists bug, consider replacing np.atleast_2d with a similar method
        if size is not NotImplemented:

            # region Fill in and infer variable and size if they aren't specified in args
            # if variable is None and size is None:
            #     variable = self.variableClassDefault
            # 6/30/17 now handled in the individual subclasses' __init__() methods because each subclass has different
            # expected behavior when variable is None and size is None.

            def checkAndCastInt(x):
                if not isinstance(x, numbers.Number):
                    raise ComponentError("An element ({}) in size is not a number.".format(x))
                if x < 1:
                    raise ComponentError("An element ({}) in size is not a positive number.".format(x))
                try:
                    int_x = int(x)
                except:
                    raise ComponentError(
                        "Failed to convert an element ({}) in size argument for {} {} to an integer. size "
                        "should be a number, or iterable of numbers, which are integers or "
                        "can be converted to integers.".format(x, type(self), self.name))
                if int_x != x:
                    if hasattr(self, 'prefs') and hasattr(self.prefs, kpVerbosePref) and self.prefs.verbosePref:
                        warnings.warn("When an element ({}) in the size argument was cast to "
                                      "integer, its value changed to {}.".format(x, int_x))
                return int_x

            #region Convert variable (if given) to a 2D array, and size (if given) to a 1D integer array
            try:
                if variable is not None:
                    variable = np.atleast_2d(variable)
                    # 6/30/17 (CW): Previously, using variable or default_input_value to create
                    # input states of differing lengths (e.g. default_input_value = [[1, 2], [1, 2, 3]])
                    # caused a bug. The if statement below fixes this bug. This solution is ugly, though.
                    if isinstance(variable[0], list) or isinstance(variable[0], np.ndarray):
                        allLists = True
                        for i in range(len(variable[0])):
                            if isinstance(variable[0][i], (list, np.ndarray)):
                                variable[0][i] = np.array(variable[0][i])
                            else:
                                allLists = False
                                break
                        if allLists:
                            variable = variable[0]
            except:
                raise ComponentError("Failed to convert variable (of type {}) to a 2D array.".format(type(variable)))

            try:
                if size is not None:
                    size = np.atleast_1d(size)
                    if len(np.shape(size)) > 1:  # number of dimensions of size > 1
                        if hasattr(self, 'prefs') and hasattr(self.prefs, kpVerbosePref) and self.prefs.verbosePref:
                            warnings.warn(
                                "size had more than one dimension (size had {} dimensions), so only the first "
                                "element of its highest-numbered axis will be used".format(len(np.shape(size))))
                        while len(np.shape(size)) > 1:  # reduce the dimensions of size
                            size = size[0]
            except:
                raise ComponentError("Failed to convert size (of type {}) to a 1D array.".format(type(size)))

            if size is not None:
                size = np.array(list(map(checkAndCastInt, size)))  # convert all elements of size to int
            # endregion

            # region If variable is None, make it a 2D array of zeros each with length=size[i]
            # implementation note: for good coding practices, perhaps add setting to enable easy change of the default
            # value of variable (though it's an unlikely use case), which is an array of zeros at the moment
            if variable is None and size is not None:
                try:
                    variable = []
                    for s in size:
                        variable.append(np.zeros(s))
                    variable = np.array(variable)
                except:
                    raise ComponentError("variable (possibly default_input_value) was not specified, but PsyNeuLink "
                                         "was unable to infer variable from the size argument, {}. size should be"
                                         " an integer or an array or list of integers. Either size or "
                                         "variable must be specified.".format(size))
            # endregion

            # the two regions below (creating size if it's None and/or expanding it) are probably obsolete (7/7/17 CW)

            # region If size is None, then make it a 1D array of scalars with size[i] = length(variable[i])
            if size is None and variable is not None:
                size = []
                try:
                    for input_vector in variable:
                        size.append(len(input_vector))
                    size = np.array(size)
                except:
                    raise ComponentError(
                        "size was not specified, but PsyNeuLink was unable to infer size from "
                        "the variable argument, {}. variable can be an array,"
                        " list, a 2D array, a list of arrays, array of lists, etc. Either size or"
                        " variable must be specified.".format(variable))
            # endregion

            # region If length(size) = 1 and variable is not None, then expand size to length(variable)
            if size is not None and variable is not None:
                if len(size) == 1 and len(variable) > 1:
                    new_size = np.empty(len(variable))
                    new_size.fill(size[0])
                    size = new_size
            # endregion

            # endregion

            # the two lines below were used when size was a param and are likely obsolete (7/7/17 CW)
            # param_defaults['size'] = size  # 7/5/17 potentially buggy? Not sure (CW)
            # self.user_params_for_instantiation['size'] = None  # 7/5/17 VERY HACKY: See Changyan's Notes on this.

            # MODIFIED 6/28/17 (CW): Because size was changed to always be a 1D array, the check below was changed
            # to a for loop iterating over each element of variable and size
            # Both variable and size are specified
            if variable is not None and size is not None:
                # If they conflict, give warning
                if len(size) != len(variable):
                    if hasattr(self, 'prefs') and hasattr(self.prefs, kpVerbosePref) and self.prefs.verbosePref:
                        warnings.warn("The size arg of {} conflicts with the length "
                                      "of its variable arg ({}) at element {}: variable takes precedence".
                                      format(self.name, size[i], variable[i], i))
                else:
                    for i in range(len(size)):
                        if size[i] != len(variable[i]):
                            if hasattr(self, 'prefs') and hasattr(self.prefs, kpVerbosePref) and self.prefs.verbosePref:
                                warnings.warn("The size arg of {} ({}) conflicts with the length "
                                                 "of its variable arg ({}) at element {}: variable takes precedence".
                                                 format(self.name, size[i], variable[i], i))

        return variable

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
            self.init_args['name'] = (self.init_args['name'] or
                                      ('deferred_init_' + self.className) or
                                      DEFERRED_DEFAULT_NAME)

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
        # print(sig.parameters)
        # def default(val):
        #     print("type(self) is: ", type(self))
        #     print("sig is: ", sig)
        #     print("sig.parameters is: ", sig.parameters)
        #     print("val is: ", val)
        #     return list(sig.parameters.values())[list(sig.parameters.keys()).index(val)].default

        default = lambda val : list(sig.parameters.values())[list(sig.parameters.keys()).index(val)].default

        def parse_arg(arg):
            # Resolve the string value of any args that use keywords as their name
            try:
                name = eval(arg)
            except NameError:
                name = arg
            if inspect.isclass(name):
                name = arg
            return name

        def _convert_function_to_class(function, source):
            from PsyNeuLink.Components.Functions.Function import Function
            from inspect import isfunction
            fct_cls = None
            fct_params = None
            # It is a PsyNeuLink Function class
            if inspect.isclass(function) and issubclass(function, Function):
                fct_cls = function
            # It is an instantiateed PsyNeuLink Function class
            elif isinstance(function, Function):
                # Set it to the class (for compatibility with current implementation of _instantiate_function()
                # and put its params in FUNCTION_PARAMS
                fct_cls = function.__class__
                fct_params = function.user_params.copy()
            # It is a generic function
            elif isfunction(function):
                # Assign to paramClassDefaults as is (i.e., don't convert to class), since class is generic
                # (_instantiate_function also tests for this and leaves it as is)
                fct_cls = function
            else:
                if hasattr(self, 'name'):
                    name = self.name
                else:
                    name = self.__class__.__name__
                raise ComponentError("Unrecognized object ({}) specified in {} for {}".
                                     format(function, source, name))
            return (fct_cls, fct_params)

        # ASSIGN DEFAULTS TO paramClassDefaults
        # Check if defaults have been assigned to paramClassDefaults, and if not do so
        for arg in kwargs:

            arg_name = parse_arg(arg)


            # The params arg is never a default (nor is anything in it)
            if arg_name is PARAMS:
                continue

            # Check if param exists in paramClassDefaults
            try:
                self.paramClassDefaults[arg]

            # param corresponding to arg is NOT in paramClassDefaults, so add it
            except:
                # If arg is FUNCTION and it's default is an instance (i.e., not a class)
                if arg_name is FUNCTION and not inspect.isclass(default(arg)):
                    # FIX: REFACTOR Component._instantiate_function TO USE COPY OF INSTANTIATED function
                    fct_cls, fct_params = _convert_function_to_class(default(arg), 'function arg')
                    self.paramClassDefaults[arg] = fct_cls
                    if fct_params:
                        self.paramClassDefaults[FUNCTION_PARAMS] = fct_params

                # Get defaults values for args listed in FUNCTION_PARAMS
                # Note:  is not an arg, but rather used to package args that belong to a non-instantiated function
                elif arg is FUNCTION_PARAMS:
                    self.paramClassDefaults[FUNCTION_PARAMS] = {}
                    for item in kwargs[arg]:
                        self.paramClassDefaults[FUNCTION_PARAMS][item] = default(item)
                else:
                    default_arg = default(arg)
                    if inspect.isclass(default_arg) and issubclass(default_arg,inspect._empty):
                        raise ComponentError("PROGRAM ERROR: \'{}\' parameter of {} must be assigned a default value "
                                             "in its constructor or in paramClassDefaults (it can be \'None\')".
                                             format(arg, self.__class__.__name__))
                    self.paramClassDefaults[arg] = default_arg

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
            if arg is PARAMS:
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
                    #                      thereby gets instantiated as a property in _create_attributes_for_params
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
                        if self.verbosePref:
                            warnings.warn("{} is not a PsyNeuLink Function, "
                                          "therefore runtime_params cannot be used".format(default(arg).__name__))
                    else:
                        raise ComponentError("Unrecognized object ({}) specified as function for {}".
                                             format(function, self.name))

                    ignore_FUNCTION_PARAMS = True

            elif arg_name is FUNCTION_PARAMS:

                # If function was instantiated object, FUNCTION_PARAMS came from it, so ignore additional specification
                if ignore_FUNCTION_PARAMS:
                    continue
                params[FUNCTION_PARAMS] = ReadOnlyOrderedDict(name=FUNCTION_PARAMS)
                for param_name in sorted(list(kwargs[arg].keys())):
                    params[FUNCTION_PARAMS].__additem__(param_name,kwargs[arg][param_name])

            # If no input_states or output_states are specified, ignore
            #   (ones in paramClassDefaults will be assigned to paramsCurrent below (in params_class_defaults_only)
            elif arg in {INPUT_STATES, OUTPUT_STATES} and kwargs[arg] is None:
                continue

            # For all other params, assign arg and its default value to paramClassDefaults
            else:
                params[arg] = kwargs[arg]

        # Add or override arg values with any specified in params dict (including FUNCTION and FUNCTION_PARAMS)
        if params_arg:

            # If function was specified in the function arg of the constructor
            #    and also in the FUNCTION entry of a params dict in params arg of constructor:
            if params and FUNCTION in params and FUNCTION in params_arg:
                # Check if it is the same as the default or the one assigned in the function arg of the constructor
                if not is_same_function_spec(params[FUNCTION], params_arg[FUNCTION]):
                    fct_cls, fct_params = _convert_function_to_class(params_arg[FUNCTION],
                                                                     '{} entry of params dict'.format(FUNCTION))
                    params_arg[FUNCTION] = fct_cls
                    if fct_params:
                        params_arg[FUNCTION_PARAMS] = fct_params
                    # If it is not the same, delete any function params that have already been assigned
                    #    in params[] for the function specified in the function arg of the constructor
                    if FUNCTION_PARAMS in params:
                        for param in list(params[FUNCTION_PARAMS].keys()):
                            params[FUNCTION_PARAMS].__deleteitem__(param)
            try:
                # Replace any parameters for function specified in function arg of constructor
                #    with those specified either in FUNCTION_PARAMS entry of params dict
                #    or for an instantiated function specified in FUNCTION entry of params dict

                # First, if the function is instantiated, get the parameters from its user_params dict
                from PsyNeuLink.Components.Functions.Function import Function
                if FUNCTION in params_arg and isinstance(params_arg[FUNCTION], Function):
                    for param_name in params_arg[FUNCTION].user_params:
                        params[FUNCTION_PARAMS].__additem__(param_name, params_arg[FUNCTION].user_params[param_name])
                # Then get any specified in FUNCTION_PARAMS entry of the params dict
                #    (these will override any specified in the constructor for the function)
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
        #    • _create_attributes_for_params assigns properties to each param in user_params;
        #    • the setter for those properties (in make_property) also assigns its value to its entry user_params;
        #    • paramInstanceDefaults are assigned to paramsCurrent in Component.__init__ assigns
        #    • since paramsCurrent is a ParamsDict, it assigns the values of its entries to the corresponding attributes
        #         and the setter assigns those values to the user_params
        #    • therefore, assignments of paramInstance defaults to paramsCurrent in __init__ overwrites the
        #         the user-specified vaules (from the constructor args) in user_params
        self.user_params_for_instantiation = OrderedDict()
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
                elif isinstance(param_value, (ReadOnlyOrderedDict, ContentAddressableList)):
                    for k in sorted(list(param_value)):
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

        # FIX: 6/1/17 - MAKE SURE FUNCTIONS DON'T GET ASSIGNED AS PROPERTIES, SINCE THEY DON'T HAVE ParameterStates
        #                AND SO CAN'T RETURN A ParameterState.value AS THEIR VALUE

        # Provide opportunity for subclasses to filter final set of params in class-specific way
        # Note:  this is done here to preserve identity of user-specified params assigned to user_params above
        self._filter_params(params)

        # Create property on self for each parameter in user_params:
        #    these WILL be validated whenever they are assigned a new value
        self._create_attributes_for_params(make_as_properties=True, **self.user_params)

        # Create attribute on self for each parameter in paramClassDefaults not in user_params:
        #    these will NOT be validated when they are assigned a value.
        # IMPLEMENTATION NOTE:
        #    These must be created here, so that attributes in user_params that need to can reference them
        #    (e.g., TransferMechanism noise property references noise param of integrator_function,
        #           which is declared in paramClassDefaults);
        #    previously these were created when paramsCurrent is assigned (in __init__());  however because
        #    the order is not guaranteed, the user_param may be assigned before one from paramClassDefaults
        params_class_defaults_only = dict(item for item in self.paramClassDefaults.items()
                                          if not any(hasattr(parent_class, item[0])
                                                     for parent_class in self.__class__.mro()))
        self._create_attributes_for_params(make_as_properties=False, **params_class_defaults_only)

        # Return params only for args:
        return params

    def _filter_params(self, params):
        """This provides an opportunity for subclasses to modify the final set of params in a class-specific way.

        Note:
        The default (here) allows user-specified params to override entries in paramClassDefaults with the same name
        """
        pass

    def _create_attributes_for_params(self, make_as_properties=False, **kwargs):
        """Create property on parent class of object for all attributes passed in kwargs dict.

        If attribute or property already exists, do nothing.
        Create backing field for attribute with "_" prefixed to attribute name,
            and assign value provided in kwargs as its default value.
        """
        if make_as_properties:
            for arg_name, arg_value in kwargs.items():
                if not any(hasattr(parent_class, arg_name) for parent_class in self.__class__.mro()):
                    setattr(self.__class__, arg_name, make_property(arg_name, arg_value))
                setattr(self, '_'+arg_name, arg_value)
        else:
            for arg_name, arg_value in kwargs.items():
                setattr(self, arg_name, arg_value)


    def _check_args(self, variable, params=None, target_set=None, context=None):
        """validate variable and params, instantiate variable (if necessary) and assign any runtime params.

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
        # VARIABLE ------------------------------------------------------------

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


        # # GET VARIABLE FROM PARAM DICT IF SPECIFIED
        # #    (give precedence to that over variable arg specification)
        # if VARIABLE in request_set and request_set[VARIABLE] is not None:
        #     variable = request_set[VARIABLE]

        # ASSIGN SHAPE TO VARIABLE if specified

        elif hasattr(self, 'shape') and self.shape is not None:
            # IMPLEMENTATION NOTE 6/23/17 (CW): this test is currently unused by all components. To confirm this, we
            # may add an exception here (raise ComponentError("Oops this is actually used")), then run all tests.
            # thus, we should consider deleting this validation

            # Both variable and shape are specified
            if variable is not None:
                # If they conflict, raise exception, otherwise use variable (it specifies both shape and content)
                if self.shape != np.array(variable).shape:
                    raise ComponentError(
                        "The shape arg of {} ({}) conflicts with the shape of its variable arg ({})".
                        format(self.name, self.shape, np.array(variable).shape))
            # Variable is not specified, so set to array of zeros with specified shape
            else:
                variable = np.zeros(self.shape)

        # VALIDATE VARIABLE (if not called from assign_params)

        if not any(context_string in context for context_string in {COMMAND_LINE, SET_ATTRIBUTE}):
            # if variable has been passed then validate and, if OK, assign as variableInstanceDefault
            self._validate_variable(variable, context=context)
            if variable is None:
                self.variableInstanceDefault = self.variableClassDefault
            else:
                self.variableInstanceDefault = self.variable

        # If no params were passed, then done
        if request_set is None and target_set is None and default_set is None:
            return

        # GET AND VALIDATE PARAMS

        # Assign param defaults for target_set and default_set
        if target_set is None:
            target_set = self.paramInstanceDefaults
        if target_set is self.paramClassDefaults:
            raise ComponentError("Altering paramClassDefaults not permitted")

        if default_set is None:
            if any(context_string in context for context_string in {COMMAND_LINE, SET_ATTRIBUTE}):
                default_set = {}
                for param_name in request_set:
                    default_set[param_name] = self.paramInstanceDefaults[param_name]
        # Otherwise, use paramInstanceDefaults (i.e., full set of implemented params)
            else:
                default_set = self.paramInstanceDefaults

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
            #         example: System, Process, ??ComparatorMechanism, ??LearningMechanism

            self.assign_default_FUNCTION_PARAMS = True

            if FUNCTION in request_set:
                # Get function class:
                function = request_set[FUNCTION]
                if inspect.isclass(function):
                    function_class = function
                else:
                    function_class = function.__class__
                # Get default function (from ParamClassDefaults)
                if not FUNCTION in default_set:
                    # This occurs if a function has been specified as an arg in the call to __init__()
                    #     but there is no function spec in paramClassDefaults;
                    # This will be caught, and an exception raised, in _validate_params()
                    pass
                else:
                    default_function = default_set[FUNCTION]
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

        # if request_set has been passed or created then validate and, if OK, assign params to target_set
        if request_set:
            # MODIFIED 4/18/17 NEW:
            # For params that are a 2-item tuple, extract the value
            #    both for validation and assignment (tuples are left intact in user_params_for_instantiation dict
            #    which is used it instantiate the specified Components in the 2nd item of the tuple)
            # IMPLEMENTATION NOTE:  Do this here rather than in _validate_params, as it needs to be done before
            #                       any override of _validate_params, which (should not, but) may process params
            #                       before calling super()._validate_params
            for param_name, param_value in request_set.items():
                if isinstance(param_value, tuple):
                    param_value = self._get_param_value_from_tuple(param_value)
                    request_set[param_name] = param_value
            # MODIFIED 4/18/17 END NEW
            self._validate_params(request_set, target_set, context=context)

    def assign_params(self, request_set=None, context=None):
        """Validates specified params, adds them TO paramInstanceDefaults, and instantiates any if necessary

        Call _instantiate_defaults with context = COMMAND_LINE, and "validated_set" as target_set.
        Update paramInstanceDefaults with validated_set so that any instantiations (below) are done in proper context.
        Instantiate any items in request set that require it (i.e., function or states).

        """
        context = context or COMMAND_LINE

        self._assign_params(request_set=request_set, context=context)


    @tc.typecheck
    def _assign_params(self, request_set:tc.optional(dict)=None, context=None):

        from PsyNeuLink.Components.Functions.Function import Function

        # FIX: Hack to prevent recursion in calls to setter and assign_params
        # MODIFIED 5/6/17 NEW:
        # Prevent recursive calls from setters
        if self.prev_context == context:
            return
        self.prev_context = context
        # MODIFIED 5/6/17 END
        # import uuid
        # try:
        #     if self.prev_id == self.curr_id:
        #         return
        # except AttributeError:
        #     pass
        # self.curr_id = uuid.uuid4()

        if not request_set:
            if self.verbosePref:
                warnings.warn("No params specified")
            return

        import copy
        validated_set = {}

        self._instantiate_defaults(request_set=request_set,
                                   target_set=validated_set,
                                   # # MODIFIED 4/14/17 OLD:
                                   # assign_missing=False,
                                   # MODIFIED 4/14/17 NEW:
                                   assign_missing=False,
                                   # MODIFIED 4/14/17 END
                                   context=context)

        self.paramInstanceDefaults.update(validated_set)

        # Turn off paramValidationPref to prevent recursive loop
        #     (since setter for attrib of param calls assign_params if validationPref is True)
        #     and no need to validate, since that has already been done above (in _instantiate_defaults)

        pref_buffer = self.prefs._param_validation_pref
        self.paramValidationPref = PreferenceEntry(False, PreferenceLevel.INSTANCE)
        self.paramsCurrent.update(validated_set)
        # The following is so that:
        #    if the Component is a function and it is passed as an argument to a Component,
        #    then the parameters are available in self.user_params_for_instantiation
        #    (which is needed when the function is recreated from its class in _assign_args_to_params_dicts)
        self.user_params_for_instantiation.update(self.user_params)
        self.paramValidationPref = pref_buffer

        # FIX: THIS NEEDS TO BE HANDLED BETTER:
        # FIX: DEAL WITH INPUT_STATES AND PARAMETER_STATES DIRECTLY (RATHER THAN VIA instantiate_attributes_before...)
        # FIX: SAME FOR FUNCTIONS THAT NEED TO BE "WRAPPED"
        # FIX: FIGURE OUT HOW TO DEAL WITH INPUT_STATES
        # FIX: FOR PARAMETER_STATES:
        #        CALL THE FOLLOWING FOR EACH PARAM:
        # FIX: NEED TO CALL

        validated_set_param_names = list(validated_set.keys())

        # If an input_state is being added from the command line,
        #    must _instantiate_attributes_before_function to parse input_states specification
        # Otherwise, should not be run,
        #    as it induces an unecessary call to _instantatiate_parameter_states (during instantiate_input_states),
        #    that causes name-repetition problems when it is called as part of the standard init procedure
        if INPUT_STATES in validated_set_param_names and COMMAND_LINE in context:
            self._instantiate_attributes_before_function(context=COMMAND_LINE)

        # Give owner a chance to instantiate function and/or function params
        # (e.g., wrap in UserDefineFunction, as per EVCMechanism)
        elif any(isinstance(param_value, (function_type, Function)) or
                      (inspect.isclass(param_value) and issubclass(param_value, Function))
                 for param_value in validated_set.values()):
            self._instantiate_attributes_before_function()

        # If the object's function is being assigned, and it is a class, instantiate it as a Function object
        if FUNCTION in validated_set and inspect.isclass(self.function):
            self._instantiate_function(context=COMMAND_LINE)

        # FIX: WHY SHOULD IT BE CALLED DURING STANDRD INIT PROCEDURE?
        # # MODIFIED 5/5/17 OLD:
        # if OUTPUT_STATES in validated_set:
        # MODIFIED 5/5/17 NEW:  [THIS FAILS WITH A SPECIFICATION IN output_states ARG OF CONSTRUCTOR]
        if OUTPUT_STATES in validated_set and COMMAND_LINE in context:
        # MODIFIED 5/5/17 END
            self._instantiate_attributes_after_function(context=COMMAND_LINE)

    def reset_params(self, mode=ResetMode.INSTANCE_TO_CLASS):
        """Reset current and/or instance defaults

        If called with:
            - CURRENT_TO_INSTANCE_DEFAULTS all current param settings are set to instance defaults
            - INSTANCE_TO_CLASS all instance defaults are set to class defaults
            - ALL_TO_CLASS_DEFAULTS all current and instance param settings are set to class defaults

        :param mode: (ResetMode) - determines which params are reset
        :return none:
        """
        # if not isinstance(mode, ResetMode):
        #     raise ComponentError("Must be called with a valid ResetMode")
        #
        if not isinstance(mode, ResetMode):
            warnings.warn("No ResetMode specified for reset_params; CURRENT_TO_INSTANCE_DEFAULTS will be used")

        if mode == ResetMode.CURRENT_TO_INSTANCE_DEFAULTS:
            for param in self.paramsCurrent:
                # if param is FUNCTION_PARAMS:
                #     for function_param in param:
                #         self.paramsCurrent[FUNCTION_PARAMS].__additem__(
                #                 function_param,
                #                 self.paramInstanceDefaults[FUNCTION_PARAMS][
                #                     function_param])
                #     continue
                self.paramsCurrent[param] = self.paramInstanceDefaults[param]
            # self.params_current = self.paramInstanceDefaults.copy()
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
            if not param_name in self.paramClassDefaults:
                # these are always allowable since they are attribs of every Component
                if param_name in {VARIABLE, NAME, VALUE, PARAMS, SIZE}:  # added SIZE here (7/5/17, CW)
                    continue
                # function is a class, so function_params has not yet been implemented
                if param_name is FUNCTION_PARAMS and inspect.isclass(self.function):
                    continue
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
            # from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
            # from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection
            if (((isinstance(param_value, str) and
                          param_value in {CONTROL_PROJECTION, LEARNING_PROJECTION, LEARNING}) or
                isinstance(param_value, Projection) or  # These should be just ControlProjection or LearningProjection
                inspect.isclass(param_value) and issubclass(param_value,(Projection)))
                and not param_name is FUNCTION):
                param_value = self.paramClassDefaults[param_name]

            # If self is a Function and param is a class ref for function, instantiate it as the function
            from PsyNeuLink.Components.Functions.Function import Function_Base
            if (isinstance(self, Function_Base) and
                    inspect.isclass(param_value) and
                    issubclass(param_value, self.paramClassDefaults[param_name])):
                    # Assign instance to target and move on
                    #  (compatiblity check no longer needed and can't handle function)
                    target_set[param_name] = param_value()
                    continue

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

            # If param is a function_type (or it has a function attribute that is one), allow any other function_type
            elif callable(param_value):
                target_set[param_name] = param_value
            elif hasattr(param_value, FUNCTION) and callable(param_value.function):
                target_set[param_name] = param_value

            # Parameter is not a valid type
            else:
                if type(self.paramClassDefaults[param_name]) is type:
                    type_name = 'the name of a subclass of ' + self.paramClassDefaults[param_name].__base__.__name__
                else:
                    type_name = self.paramClassDefaults[param_name].__class__.__name__
                if param_name == 'matrix':
                    raise ComponentError("Value of {} param for {} ({}) must be a valid matrix specification".
                                    format(param_name, self.name, param_value))
                raise ComponentError("Value of {} param for {} ({}) must be compatible with {}".
                                    format(param_name, self.name, param_value, type_name))

    def _get_param_value_from_tuple(self, param_spec):
        """Returns param value (first item) of a (value, projection) tuple
        """
        from PsyNeuLink.Components.Projections.Projection import Projection
        # from PsyNeuLink.Components.Projections.Modulatory.ControlProjection import ControlProjection
        # from PsyNeuLink.Components.Projections.Modulatory.LearningProjection import LearningProjection
        from PsyNeuLink.Components.Projections.ModulatoryProjections.ModulatoryProjection import ModulatoryProjection_Base
        from PsyNeuLink.Components.States.ModulatorySignals.ModulatorySignal import ModulatorySignal
        ALLOWABLE_TUPLE_SPEC_KEYWORDS = {CONTROL_PROJECTION, LEARNING_PROJECTION, CONTROL, LEARNING}
        ALLOWABLE_TUPLE_SPEC_CLASSES = (ModulatoryProjection_Base, ModulatorySignal)

        # If the 2nd item is a CONTROL or LEARNING SPEC, return the first item as the value
        if (isinstance(param_spec, tuple) and len(param_spec) is 2 and
                # # MODIFIED 6/19/17 OLD:
                # (param_spec[1] in {CONTROL_PROJECTION, LEARNING_PROJECTION, CONTROL, LEARNING} or
                #      isinstance(param_spec[1], Projection) or
                #      (inspect.isclass(param_spec[1]) and issubclass(param_spec[1], Projection)))
                # MODIFIED 6/19/17 NEW:
                (param_spec[1] in ALLOWABLE_TUPLE_SPEC_KEYWORDS or
                     isinstance(param_spec[1], ALLOWABLE_TUPLE_SPEC_CLASSES) or
                         (inspect.isclass(param_spec[1]) and issubclass(param_spec[1], ALLOWABLE_TUPLE_SPEC_CLASSES)))
                # MODIFIED 6/19/17 END
            ):
            value =  param_spec[0]

        # Otherwise, just return the tuple
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
            function = self.function
        # self.function is NOT implemented
        except KeyError:
            function = None
        # self.function IS implemented
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
                                             self.function.__class__.__name__,
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
                    function_param_specs = self.function_params.copy()
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
                        # Get param value from any params specified in a tuple or a dict
                        from PsyNeuLink.Components.Projections.Projection import Projection
                        for param_name, param_spec in function_param_specs.items():
                            # Get param value from (param, projection) tuple
                            if (isinstance(param_spec, tuple) and len(param_spec) is 2 and
                                    (param_spec[1] in {MAPPING_PROJECTION, CONTROL_PROJECTION, LEARNING_PROJECTION} or
                                         isinstance(param_spec[1], Projection) or
                                         (inspect.isclass(param_spec[1]) and issubclass(param_spec[1], Projection)))
                                ):
                                from PsyNeuLink.Components.States.ParameterState import ParameterState
                                function_param_specs[param_name] =  param_spec[0]
                            # Get param value from VALUE entry of a parameter specification dictionary
                            elif isinstance(param_spec, dict) and VALUE in param_spec:
                                function_param_specs[param_name] =  param_spec[VALUE]

                # Instantiate function from class specification
                function_instance = function(variable_default=self.variable,
                                             params=function_param_specs,
                                             # IMPLEMENTATION NOTE:
                                             #    Don't bother with this, since it has to be assigned explicitly below
                                             #    anyhow, for cases in which function already exists
                                             #    and would require every function to have the owner arg in its __init__
                                             owner=self,
                                             context=context)
                self.function = function_instance.function

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
                                  format(self.function.__self__.componentName,
                                         object_name))

            # FUNCTION is a generic function (presumably user-defined), so "wrap" it in UserDefinedFunction:
            #   Note: calling UserDefinedFunction.function will call FUNCTION
            elif inspect.isfunction(function):


                from PsyNeuLink.Components.Functions.Function import UserDefinedFunction
                self.function = UserDefinedFunction(function=function, context=context).function

            # If FUNCTION is NOT a Function class reference:
            # - issue warning if in VERBOSE mode
            # - pass through to try self.function below
            else:
                if self.prefs.verbosePref:
                    warnings.warn("{0} ({1}) is not a subclass of Function".
                                  format(FUNCTION,
                                         self.function.__class__.__name__,
                                         self.name))
                function = None

        # params[FUNCTION] was not specified (in paramsCurrent, paramInstanceDefaults or paramClassDefaults)
        if not function:
            # Try to assign to self.function
            try:
                self.function = self.function
            # If self.function is also not implemented, raise exception
            # Note: this is a "sanity check," as this should have been checked in _validate_function (above)
            except AttributeError:
                raise ComponentError("{0} ({1}) is not a Function object or class, "
                                    "and {2}.function is not implemented".
                                    format(FUNCTION, self.function,
                                           self.__class__.__name__))
            # If self.function is implemented, warn if in VERBOSE mode
            else:
                if self.prefs.verbosePref:
                    warnings.warn("{0} ({1}) is not a Function object or a specification for one; "
                                  "{1}.function ({}) will be used instead".
                                  format(FUNCTION,
                                         self.function.__self__.componentName,
                                         self.name,
                                         self.function.__self__.name))

        # MAKE ASSIGNMENTS
        # Now that function has been instantiated:

        #  - assign to paramInstanceDefaults
        self.paramInstanceDefaults[FUNCTION] = self.function

        #  - for all Components other than a Function itself,
        #    assign function_object, function_params dict, and function's parameters from any ParameterStates
        from PsyNeuLink.Components.Functions.Function import Function
        if not isinstance(self, Function):
            self.function_object = self.function.__self__
            if not self.function_object.owner:
                self.function_object.owner = self
            elif self.function_object.owner != self:
                raise ComponentError("Function being assigned to {} ({}) belongs to another Component: {}".
                                     format(self.name, self.function_object.name, self.function_object.owner.name))
            # sort to maintain alphabetical order of function_params
            for param_name in sorted(list(self.function_object.user_params_for_instantiation.keys())):
                # assign to param to function_params dict
                self.function_params.__additem__(param_name,
                                                 self.function_object.user_params_for_instantiation[param_name])
                # # assign values from any ParameterStates the Component may (which it should) have
                # try:
                #     value_type = type(getattr(self.function_object, '_'+param_name))
                #     param_value = type_match(self._parameter_states[param_name].value, value_type)
                # except:
                #     pass
                # else:
                #     setattr(self.function_object, '_'+param_name, param_value)
            self.paramInstanceDefaults[FUNCTION_PARAMS] = self.function_params

        #  - call self.execute to get value, since the value of a Component is defined as what is returned by its
        #    execute method, not its function
        if not context:
            context = "DIRECT CALL"
        self.value = self.execute(context=context)
        if self.value is None:
            raise ComponentError("PROGRAM ERROR: Execute method for {} must return a value".format(self.name))
        try:
            # Could be mutable, so assign copy
            self._default_value = self.value.copy()
        except AttributeError:
            # Immutable, so just assign value
            self._default_value = self.value

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

    def _change_function(self, to_function):
        pass

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise ComponentError("Name assigned to {} ({}) must be a string constant".
                                 format(self.__class__.__name__, value))

        self._name = value

    @property
    def size(self):
        if not hasattr(self, 'variable'):
            return None
        s = []
        v = np.atleast_2d(self.variable)
        for i in range(len(v)):
            s.append(len(v[i]))
        return np.array(s)

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
        # try:
        #     return self._paramsCurrent
        # except AttributeError:
        #     self._paramsCurrent = ParamsDict(self)
        #     return self._paramsCurrent

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

    @property
    def auto_dependent(self):
        return self._auto_dependent

    @auto_dependent.setter
    def auto_dependent(self, value):
        """Assign auto_dependent status to Component and any of its owners up the hierarchy
        """
        owner = self
        while owner is not None:
            try:
                owner._auto_dependent = value
                owner = owner.owner

            except AttributeError:
                owner = None

COMPONENT_BASE_CLASS = Component


def make_property(name, default_value):
    backing_field = '_' + name

    def getter(self):
        try:
            # Get value of function param from ParameterState.value of owner
            #    case: request is for the value of a Function parameter for which the owner has a ParameterState
            #    example: slope or intercept parameter of a Linear Function)
            #    rationale: most common and therefore requires the greatest efficiency
            #    note: use backing_field[1:] to get name of parameter as index into _parameter_states)
            from PsyNeuLink.Components.Functions.Function import Function
            if not isinstance(self, Function):
                raise TypeError
            return self.owner._parameter_states[backing_field[1:]].value
        except (AttributeError, TypeError):
            try:
                # Get value of param from Component's own ParameterState.value
                #    case: request is for the value of a parameter of a Mechanism or Project that has a ParameterState
                #    example: matrix parameter of a MappingProjection)
                #    rationale: next most common case
                #    note: use backing_field[1:] to get name of parameter as index into _parameter_states)
                return self._parameter_states[backing_field[1:]].value
            except (AttributeError, TypeError):
                # Get value of param from Component's attribute
                #    case: request is for the value of an attribute for which the Component has no ParameterState
                #    rationale: least common case
                #    example: parameter of a Function belonging to a state (which don't themselves have ParameterStates)
                #    note: use backing_field since referencing property rather than item in _parameter_states)
                return getattr(self, backing_field)

    def setter(self, val):

        if self.paramValidationPref and hasattr(self, PARAMS_CURRENT):
            val_str = val.__class__.__name__
            curr_context = SET_ATTRIBUTE + ': ' + val_str + ' for ' + backing_field[1:] + ' of ' + self.name
            self._assign_params(request_set={backing_field[1:]:val}, context=curr_context)
        else:
            setattr(self, backing_field, val)

        # Update user_params dict with new value
        self.user_params.__additem__(name, val)

        # If Component is a Function and has an owner, update function_params dict for owner
        #    also, get parameter_state_owner if one exists
        from PsyNeuLink.Components.Functions.Function import Function_Base
        if isinstance(self, Function_Base) and self.owner:
            param_state_owner = self.owner
            self.owner.function_params.__additem__(name, val)
        else:
            param_state_owner = self

        # If the parameter is associated with a ParameterState, assign the value to the ParameterState's variable
        if hasattr(param_state_owner, '_parameter_states') and name in param_state_owner._parameter_states:
            param_state_owner._parameter_states[name].variable = val

    # Create the property
    prop = property(getter).setter(setter)

    # # Install some documentation
    # prop.__doc__ = docs[name]
    return prop
