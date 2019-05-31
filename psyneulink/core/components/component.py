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
(listed `below <Component_Methods>`) as part of the initialization procedure.  Every Component has a core set of
`configurable parameters <Parameters>` that can be specified in the arguments of the constructor, as well
as additional parameters and attributes that may be specific to particular Components, many of which can be modified
by the user, and some of which provide useful information about the Component (see `User_Modifiable_Parameters`
and `Informational Attributes` below).

.. _Component_Deferred_Init:

*Deferred Initialization*
~~~~~~~~~~~~~~~~~~~~~~~~~

If information necessary to complete initialization is not specified in the constructor (e.g, the **owner** for a
`State <State_Base.owner>`, or the **sender** or **receiver** for a `Projection <Projection_Structure>`), then its
full initialization is deferred until its the information is available (e.g., the `State <State>` is assigned to a
`Mechanism <Mechanism>`, or a `Projection <Projection>` is assigned its `sender <Projection_Base.sender>` and `receiver
<Projection_Base.receiver>`).  This allows Components to be created before all of the information they require is
available (e.g., at the beginning of a script). However, for the Component to be operational, its initialization must
be completed by a call to it `deferred_init` method.  This is usually done automatically when the Component is
assigned to another Component to which it belongs (e.g., assigning a State to a Mechanism) or to a Composition (e.g.,
a Projection to the `pathway <Process.pahtway>`) of a `Process`), as appropriate.

.. _Component_Structure:

Component Structure
-------------------

.. _Component_Structural_Attributes:

*Core Structural Attributes*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every Component has the following set of core structural attributes. These attributes are not meant to be changed by the
user once the component is constructed, with the one exception of `prefs <Component_Prefs>`.

.. _Component_Variable:

* **variable** - used as the input to its `function <Component_Function>`.  Specification of the **default_variable**
  argument in the constructor for a Component determines both its format (e.g., whether its value is numeric, its
  dimensionality and shape if it is an array, etc.) as well as its default value (the value used when the Component
  is executed and no input is provided), and takes precedence over the specification of `size <Component_Size>`.

  .. note::
    Internally, the attribute **variable** is not directly used as input to functions, to allow for parallelization.
    The attribute is maintained as a way for the user to monitor variable along the execution chain.
    During parallelization however, the attribute may not accurately represent the most current value of variable
    being used, due to asynchrony inherent to parallelization.

.. _Component_Size:

* **size** - the dimension of the `variable <Component.variable>` attribute.  The **size** argument of the
  constructor for a Component can be used as a convenient method for specifying the `variable <Component_Variable>`,
  attribute in which case it will be assigned as an array of zeros of the specified size.  For example,
  setting  **size** = 3 is equivalent to setting **variable** = [0, 0, 0] and setting **size** = [4, 3] is equivalent
  to setting **variable** = [[0, 0, 0, 0], [0, 0, 0]].

.. _Component_Function:

* **function** - determines the computation that a Component carries out. It is always a PsyNeuLink `Function <Function>`
  object (itself a PsyNeuLink Component).

  .. note::
     The `function <Component.function>` of a Component can be assigned either a `Function` object or any other
     callable object in python.  If the latter is assigned, it is "wrapped" in a `UserDefinedFunction`.

  All Components have a default `function <Component.function>` (with a default set of parameters), that is used if it
  is not otherwise specified.  The `function <Component.function>` can be specified in the
  **function** argument of the constructor for the Component, using one of the following:

    * **class** - this must be a subclass of `Function <Function>`, as in the following example::

        my_component = SomeComponent(function=SomeFunction)

      This will create a default instance of the specified subclass, using default values for its parameters.

    * **Function** - this can be either an existing `Function <Function>` object or the constructor for one, as in the
      following examples::

        my_component = SomeComponent(function=SomeFunction)

        or

        some_function = SomeFunction(some_param=1)
        my_component = SomeComponent(some_function)

      The specified Function will be used as a template to create a new Function object that is assigned to the
      `function` attribute of the Component.

      .. note::

        In the current implementation of PsyNeuLink, if a `Function <Function>` object (or the constructor for one) is
        used to specify the `function <Component.function>` attribute of a Component, the Function object specified (or
        created) will only *itself* be assigned to the Component if it does not already belong to another Component.
        Otherwise, it is copied, and the copy is assigned to the Component.
        This is so that `Functions <Function>` can be used as templates for
        more than one Component, without being assigned simultaneously to multiple Components.

  A `function <Component.function>` can also be specified in an entry of a
  `parameter specification dictionary <ParameterState_Specification>` assigned to the
  **params** argument of the constructor for the Component, with the keyword *FUNCTION* as its key, and one of the
  specifications above as its value, as in the following example::

        my_component = SomeComponent(params={FUNCTION:SomeFunction(some_param=1)})

.. _Component_Value:

* **value** - the `value <Component.value>` attribute contains the result (return value) of the Component's
  `function <Component.function>` after the function is called.
..

.. _Component_Log:

* **log** - the `log <Component.log>` attribute contains the Component's `Log`, that can be used to record its
  `value <Component.value>`, as well as that of Components that belong to it, during initialization, validation,
  execution and learning.  It also has four convenience methods -- `loggable_items <Log.loggable_items>`, `set_log_conditions
  <Log.set_log_conditions>`, `log_values <Log.log_values>` and `logged_items <Log.logged_items>` -- that provide access to the
  corresponding methods of its Log, used to identify, configure and track items for logging.
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

.. _Component_Reinitialize_When:

* **reinitialize_when** - the `reinitialize_when <Component.reinitialize_when>` attribute contains a `Condition`. When
  this condition is satisfied, the Component calls its `reinitialize <Component.reinitialize>` method. The
  `reinitialize <Component.reinitialize>` method is executed without arguments, meaning that the relevant function's
  `initializer<IntegratorFunction.initializer>` attribute (or equivalent -- initialization attributes vary among
  functions) is used for reinitialization. Keep in mind that the `reinitialize <Component.reinitialize>` method and
  `reinitialize_when <Component.reinitialize_when>` attribute only exist for Mechanisms that have `stateful
  <Parameter.stateful>` Parameters, or that have a `function <Mechanism.function>` with `stateful <Parameter.stateful>`
  Parameters.

  .. note::

        Currently, only Mechanisms reinitialize when their reinitialize_when Conditions are satisfied. Other types of
        Components do not reinitialize.

.. _User_Modifiable_Parameters:

*Parameters*
~~~~~~~~~~~~

.. _Component_Parameters:

A Component defines its `parameters <Parameters>` in its *parameters* attribute, which contains a collection of
`Parameter` objects, each of which stores a Parameter's values, `default values <Component.defaults>`, and various
`properties <Parameter_Attributes_Table>` of the parameter.

* `Parameters <Component.Parameters>` - a `Parameters class <Parameters>` defining parameters and their default values that
    are used for all Components, unless overridden.

* `user_params <Component.user_params>` - a dictionary that provides reference to all of the user-modifiable parameters
  of a Component. The dictionary is a ReadOnlyDict (a PsyNeuLink-defined subclass of the Python class `UserDict
  <https://docs.python.org/3.6/library/collections.html?highlight=userdict#collections.UserDict>`_). The
  value of an entry can be accessed in the standard manner (e.g., ``my_component.user_params[`PARAMETER NAME`]``);
  as can its full list of entries (e.g., ``my_component.user_params``).  However, because it is read-only,
  it cannot be used to make assignments.  Rather, changes to the value of a parameter must be made by assigning a
  value to the attribute for that parameter directly (e.g., ``my_component.my_parameter``), but using a dedicated
  method if one exists (e.g., `Mechanism_Base.add_states`), or by using the Component's `assign_params
  <Component.assign_params>` method.

  All of the parameters listed in the *user_params* dictionary can be modified by the user (as described above).  Some
  can also be modified by `ControlSignals <ControlSignal>` when a `System executes <System_Execution_Control>`. In
  general, only parameters that take numerical values and/or do not affect the structure, mode of operation,
  or format of the values associated with a Component can be subject to modulation.  For example, for a
  `TransferMechanism`, `clip <TransferMechanism.clip>`, `initial_value <TransferMechanism.initial_value>`,
  `integrator_mode <TransferMechanism.integrator_mode>`, `input_states <TransferMechanism.input_states>`,
  `output_states`, and `function <TransferMechanism.function>`, are all listed in user_params, and are user-modifiable,
  but are not subject to modulation; whereas `noise <TransferMechanism.noise>` and `integration_rate
  <TransferMechanism.integration_rate>`, as well as the parameters of the TransferMechanism's `function
  <TransferMechanism.function>` (listed in the *function_params* subdictionary) can all be subject to modulation.
  Parameters that are subject to modulation are associated with a `ParameterState` to which the ControlSignals
  can project (by way of a `ControlProjection`).

.. _Component_Function_Params:

* **function_params** - the `function_params <Component.function>` attribute contains a dictionary of the parameters
  for the Component's `function <Component.function>` and their values.  Each entry is the name of a parameter, and its
  value is the value of that parameter.  The dictionary uses a ReadOnlyDict (a PsyNeuLink-defined subclass of the Python
  class `UserList <https://docs.python.org/3.6/library/collections.html?highlight=userdict#collections.UserDict>`_). The
  value of an entry can be accessed in the standard manner (e.g., ``my_component.function_params[`PARAMETER NAME`]``);
  as can its  full list of its entries (e.g., ``my_component.function_params``).  However, because it is read-only,
  it cannot be used to make assignments. Rather, changes to the value of a function's parameters must be made by
  assigning a value to the corresponding attribute of the Component's `function <Component.function>`
  attribute (e.g., ``my_component.function.my_parameter``), or in a FUNCTION_PARAMS dict using its
  `assign_params` method.  The parameters for a function can be specified when the Component is created in one of
  the following ways:

      * in the **constructor** for a Function -- if that is used to specify the `function <Component.function>`
        argument, as in the following example::

            my_component = SomeComponent(function=SomeFunction(some_param=1, some_param=2)

      * in an argument of the **Component's constructor** -- if all of the allowable functions for a Component's
        `function <Component.function>` share some or all of their parameters in common, the shared paramters may appear
        as arguments in the constructor of the Component itself, which can be used to set their values.

      * in an entry of a `parameter specification dictionary <ParameterState_Specification>` assigned to the
        **params** argument of the constructor for the Component.  The entry must use the keyword
        FUNCTION_PARAMS as its key, and its value must be a dictionary containing the parameters and their values.
        The key for each entry in the FUNCTION_PARAMS dictionary must be the name of a parameter, and its value the
        parameter's value, as in the example below::

            my_component = SomeComponent(function=SomeFunction
                                         params={FUNCTION_PARAMS:{SOME_PARAM=1, SOME_OTHER_PARAM=2}})

  The parameters of functions for some Components may allow other forms of specification (see
  `ParameterState_Specification` for details concerning different ways in which the value of a
  parameter can be specified).

.. _Informational_Attributes:

*Informational Attributes*
~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to its `user-modifiable parameters <Component_User_Params>`, a Component has attributes that provide
information about its contents and/or state, but do not directly affect its operation.  Every Component has the
following two informational attributes:

.. _Component_Execution_Count:

* **execution_count** -- maintains a record of the number of times a Component has executed; it *excludes* the
  executions carried out during initialization and validation, but includes all other executions, whether they are of
  the Component on its own are as part of a `Composition`, and irresective of the `context <Context>` in which they
  are occur. The value can be changed "manually" or programmatically by assigning an integer value directly to the
  attribute.

.. _Component_Current_Execution_Time:

* **current_execution_time** -- maintains the `Time` of the last execution of the Component in the context of the
  `Composition`'s current `scheduler_processing <Composition.scheduler_processing`, and is stored as a `time
  <Context.time>` tuple of values indicating the `TimeScale.TRIAL`,  `TimeScale.PASS`, and `TimeScale.TIME_STEP` of the
  last execution.

COMMENT:
FIX: STATEMENT ABOVE ABOUT MODIFYING EXECUTION COUNT VIOLATES THIS DEFINITION, AS PROBABLY DO OTHER ATTRIBUTES
  * parameters are things that govern the operation of the Mechanism (including its function) and/or can be modified/modulated
  * attributes include parameters, but also read-only attributes that reflect but do not determine the operation (e.g., EXECUTION_COUNT)
COMMENT

..
COMMENT:
  FOR DEVELOPERS:
    * **paramClassDefaults**

    * **paramInstanceDefaults**
COMMENT


.. _Component_Methods:

*Component Methods*
~~~~~~~~~~~~~~~~~~~

COMMENT:
   FOR DEVELOPERS:

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
        `parameter specification dictionary <ParameterState_Specification>`.  If it is overridden by a subclass,
        customized validation should generally be performed *after* the call to super().

    * **Instantiation methods** create, assign, and/or perform *semantic* checks on the values of Component attributes.
      Semantic checks may include value and/or range checks, as well as checks of formatting and/or value
      compatibility with other attributes of the Component and/or the attributes of other Components (for example, the
      _instantiate_function method checks that the input of the Component's `function <Comonent.function>` is compatible
      with its `variable <Component.variable>`).

      * `_handle_size <Component._handle_size>` converts the `variable <Component.variable>` and `size <Component.size>`
        arguments to the correct dimensions (for `Mechanism <Mechanism>`, this is a 2D array and 1D
        array, respectively). If **variable** is not passed as an argument, this method attempts to infer `variable
        <Component.variable>` from the **size** argument, and vice versa if the **size** argument is missing.
        The _handle_size method then checks that the **size** and **variable** arguments are compatible.

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
  as an entry in a `parameter specification dictionary <ParameterState_Specification>` in the **request_set**
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
import copy
import inspect
import logging
import numbers
import types
import warnings

from abc import ABCMeta
from collections.abc import Iterable
from collections import OrderedDict, UserDict
from enum import Enum, IntEnum

import numpy as np
import typecheck as tc

from psyneulink.core import llvm as pnlvm
from psyneulink.core.globals.context import Context, ContextError, ContextFlags, INITIALIZATION_STATUS_FLAGS, _get_time, handle_external_context
from psyneulink.core.globals.keywords import \
    COMPONENT_INIT, CONTEXT, CONTROL_PROJECTION, DEFERRED_INITIALIZATION, \
    FUNCTION, FUNCTION_CHECK_ARGS, FUNCTION_PARAMS, INITIALIZING, INIT_FULL_EXECUTE_METHOD, INPUT_STATES, \
    LEARNING, LEARNING_PROJECTION, LOG_ENTRIES, MATRIX, MODULATORY_SPEC_KEYWORDS, NAME, OUTPUT_STATES, \
    PARAMS, PARAMS_CURRENT, PREFS_ARG, REINITIALIZE_WHEN, SIZE, USER_PARAMS, VALUE, VARIABLE, kwComponentCategory
from psyneulink.core.globals.log import LogCondition
from psyneulink.core.globals.parameters import Defaults, Parameter, ParameterAlias, ParameterError, ParametersBase
from psyneulink.core.globals.preferences.componentpreferenceset import ComponentPreferenceSet, kpVerbosePref
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel, PreferenceSet
from psyneulink.core.globals.registry import register_category
from psyneulink.core.globals.utilities import ContentAddressableList, ReadOnlyOrderedDict, convert_all_elements_to_np_array, convert_to_np_array, get_deepcopy_with_shared, is_instance_or_subclass, is_matrix, iscompatible, kwCompatibilityLength, prune_unused_args, unproxy_weakproxy
from psyneulink.core.scheduling.condition import Never

__all__ = [
    'Component', 'COMPONENT_BASE_CLASS', 'component_keywords', 'ComponentError', 'ComponentLog',
    'DefaultsFlexibility', 'DeferredInitRegistry', 'make_property', 'parameter_keywords', 'ParamsDict', 'ResetMode',
]

logger = logging.getLogger(__name__)

component_keywords = {NAME, VARIABLE, VALUE, FUNCTION, FUNCTION_PARAMS, PARAMS, PREFS_ARG, CONTEXT}

DeferredInitRegistry = {}


def get_deepcopy_with_shared_Components(shared_keys=None):
    return get_deepcopy_with_shared(shared_keys, (Component, ))


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


class DefaultsFlexibility(Enum):
    """
        Denotes how rigid an assignment to a default is. That is, how much, if at all
        it can be modified to suit the purpose of a method/owner/etc.

        e.g. when assigning a Function to a Mechanism:

            ``pnl.TransferMechanism(default_variable=[0, 0], function=pnl.Linear())``

            the Linear function is assigned a default variable ([0]) based on it's ClassDefault,
            which conflicts with the default variable specified by its future owner ([0, 0]). Since
            the default for Linear was not explicitly stated, we allow the TransferMechanism to
            reassign the Linear's default variable as needed (`FLEXIBLE`)

    Attributes
    ----------

    FLEXIBLE
        can be modified in any way

    RIGID
        cannot be modifed in any way

    INCREASE_DIMENSION
        can be wrapped in a single extra dimension

    """
    FLEXIBLE = 0
    RIGID = 1
    INCREASE_DIMENSION = 2


# Transitional type:
#    for implementing params as attributes that are accessible via current paramsDicts
#    (until params are fully implemented as objects)
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
        if key is FUNCTION:
            # hack because function is NOT stored as an attribute when this object wants it!
            return super().__getitem__(key)

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
        if key is not FUNCTION and key is not FUNCTION_PARAMS:
            # function is not stored as an attribute!
            # 7/14/18 - JDC BUG:  ISN'T SETTING VALUE OF AUTOASSOCIATIVEPROJECTION.MATRIX
            # suppresses read-only warnings because this is internal
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                setattr(self.owner, key, item)

parameter_keywords = set()

# suppress_validation_preference_set = ComponentPreferenceSet(prefs = {
#     kpParamValidationPref: PreferenceEntry(False,PreferenceLevel.INSTANCE),
#     kpVerbosePref: PreferenceEntry(False,PreferenceLevel.INSTANCE),
#     kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE)})


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


def _parameters_belongs_to_obj(obj):
    if obj is obj.parameters._owner:
        return True

    try:
        return obj is unproxy_weakproxy(obj.parameters._owner)
    except AttributeError:
        return False


def _get_backing_field(attr_name):
    return '_{0}'.format(attr_name)


def make_parameter_property(name):
    backing_field = _get_backing_field(name)

    def getter(self):
        if not _parameters_belongs_to_obj(self):
            # would refer to class parameters before an instance of Parameters is created for self
            return getattr(self, backing_field)
        else:
            return getattr(self.parameters, name)._get(self.most_recent_context)

    def setter(self, value):
        if not _parameters_belongs_to_obj(self):
            # set to backing field, which is what gets looked for and discarded when creating an object's parameters
            setattr(self, backing_field, value)
        else:
            getattr(self.parameters, name)._set(value, self.most_recent_context)

    return property(getter).setter(setter)


def _has_initializers_setter(value, owning_component=None, context=None):
    """
    Assign has_initializers status to Component and any of its owners up the hierarchy.
    """
    if value:
        # only update owner's attribute if setting to True, because there may be
        # other children that have initializers
        try:
            owning_component.owner.parameters.has_initializers._set(value, context)
        except AttributeError:
            # no owner
            pass

    return value

# *****************************************   COMPONENT CLASS  ********************************************************

class ComponentsMeta(ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.defaults = Defaults(owner=self)
        try:
            parent = self.__mro__[1].parameters
        except AttributeError:
            parent = None
        self.parameters = self.Parameters(owner=self, parent=parent)

        for param in self.parameters:
            if not hasattr(self, param.name):
                setattr(self, param.name, make_parameter_property(param.name))

    # consider removing this for explicitness
    # but can be useful for simplicity
    @property
    def class_defaults(self):
        return self.defaults


class Component(object, metaclass=ComponentsMeta):
    """Base class for Component.

    .. note::
       Component is an abstract class and should NEVER be instantiated by a direct call to its constructor.
       It should be instantiated using the constructor for a subclass.

    COMMENT:
        Every Component is associated with:
         - child class componentName
         - type
         - input
         - execute (method): called to execute it;  it in turn calls self.function
         - function (method): carries out object's core computation
             it can be referenced either as self.function, self.params[FUNCTION] or self.paramsCurrent[FUNCTION]
         - function (Function): the object to which function belongs (and that defines it's parameters)
         - output (value: self.value)
         - output_values (return from self.execute: concatenated set of values of output_states)
         - class and instance variable defaults
         - class and instance param defaults
        The Components's execute method (<subclass>.execute is the Component's primary method
            (e.g., it is the one called when Process, Mechanism, State and Projections objects are updated);
            the following attributes for or associated with the method are defined for every Component object:
                + execute (method) - the execute method itself
                + value (value) - the output of the execute method
            the latter is used for typing and/or templating other variables (e.g., self.defaults.variable):
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
                            it is instantiated using self.defaults.variable and FUNCTION_PARAMS (if they are there too)
                            this works, since _validate_params is always called after _validate_variable
                            so self.defaults.variable can be used to initialize function
                            to the method referenced by paramInstanceDefaults[FUNCTION] (see below)
                    if paramClassDefaults[FUNCTION] is not found, it's value is assigned to self.function
                    if neither paramClassDefaults[FUNCTION] nor self.function is found, an exception is raised

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
        + componentCategory - category of Component (i.e., Process, Mechanism, Projection, Function)
        + componentType - type of Component within a category
                             (e.g., TransferMechanism, MappingProjection, ControlProjection, etc.)
        + requiredParamClassDefaultTypes - dict of param names & types that all subclasses of Component must implement;

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
        + class_defaults.variable (value)
        + variableClassDefault_np_info (ndArrayInfo)
        + defaults.variable (value)
        + _default_variable_flexibility
        + variable (value)
        + variable_np_info (ndArrayInfo)
        + function (method)
        + function (Function)
        + paramClassDefaults:
            + FUNCTION
            + FUNCTION_PARAMS
        + paramInstanceDefaults
        + paramsCurrent
        # + parameter_validation
        + user_params
        +._runtime_params_reset

    Instance methods:
        + function (implementation is optional; aliased to params[FUNCTION] by default)
    COMMENT

    Attributes
    ----------

    variable : 2d np.array
        see `variable <Component_Variable>`

    size : int or array of ints
        see `size <Component_Size>`

    function : Function, function or method
        see `function <Component_Function>`

    function_params : Dict[param_name: param_value]
        see `function_params <Component_Function_Params>`

    user_params : Dict[param_name: param_value]
        see `user_params <Component_User_Params>`

    value : 2d np.array
        see `value <Component_Value>`

    log : Log
        see `log <Component_Log>`

    execution_count : int
        see `execution_count <Component_Execution_Count>`

    current_execution_time : tuple(`Time.RUN`, `Time.TRIAL`, `Time.PASS`, `Time.TIME_STEP`)
        see `current_execution_time <Component_Current_Execution_Time>`

    reinitialize_when : `Condition`
        see `reinitialize_when <Component_Reinitialize_When>`

    name : str
        see `name <Component_Name>`

    prefs : PreferenceSet
        see `prefs <Component_Prefs>`

    parameters
        see `parameters <Component_Parameters>`

    defaults
        an object that provides access to the default values of a `Component's` `parameters`

    initialization_status : field of flags attribute
        indicates the state of initialization of the Component;
        one and only one of the following flags is always set:

            * `DEFERRED_INIT <ContextFlags.DEFERRED_INIT>`
            * `INITIALIZING <ContextFlags.INITIALIZING>`
            * `VALIDATING <ContextFlags.VALIDATING>`
            * `INITIALIZED <ContextFlags.INITIALIZED>`
            * `REINITIALIZED <ContextFlags.REINITIALIZED>`
            * `UNINITIALIZED <ContextFlags.UNINITALIZED>`

    """

    #CLASS ATTRIBUTES
    className = "COMPONENT"
    suffix = " " + className
# IMPLEMENTATION NOTE:  *** CHECK THAT THIS DOES NOT CAUSE ANY CHANGES AT SUBORDNIATE LEVELS TO PROPOGATE EVERYWHERE
    componentCategory = None
    componentType = None

    standard_constructor_args = [REINITIALIZE_WHEN]

    class Parameters(ParametersBase):
        """
            The `Parameters` that are associated with all `Components`

            Attributes
            ----------

                variable
                    see `variable <Component.variable>`

                    :default value: numpy.array([0])
                    :type: numpy.ndarray
                    :read only: True

                value
                    see `value <Component.value>`

                    :default value: numpy.array([0])
                    :type: numpy.ndarray
                    :read only: True

                has_initializers
                    see `has_initializers <Component.has_initializers>`

                    :default value: False
                    :type: bool

                execution_count
                    see `execution_count <Component.execution_count>`

                    :default value: 0
                    :type: int
                    :read only: True

        """
        variable = Parameter(np.array([0]), read_only=True)
        value = Parameter(np.array([0]), read_only=True)
        has_initializers = Parameter(False, setter=_has_initializers_setter)
        # execution_count ios not stateful because it is a global counter;
        #    for context-specific counts should use schedulers which store this info
        execution_count = Parameter(0, read_only=True, loggable=False, stateful=False, fallback_default=True)

        def _parse_variable(self, variable):
            return variable

        def _validate_variable(self, variable):
            return None

    initMethod = INIT_FULL_EXECUTE_METHOD

    classPreferenceLevel = PreferenceLevel.SYSTEM
    # Any preferences specified below will override those specified in SystemDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to SYSTEM automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'ComponentCustomClassPreferences',
    #     kp<pref>: <setting>...}

    # Names and types of params required to be implemented in all subclass paramClassDefaults:
    # Notes:
    # *  entry values here do NOT implement the param; they are simply used as type specs for checking (in __init__)
    # * kwComponentCategory (below) is used as placemarker for Component.Function class; replaced in __init__ below
    #              (can't reference own class directly class block)
    requiredParamClassDefaultTypes = {}

    paramClassDefaults = {}

    exclude_from_parameter_states = [INPUT_STATES, OUTPUT_STATES]

    # IMPLEMENTATION NOTE: This is needed so that the State class can be used with ContentAddressableList,
    #                      which requires that the attribute used for addressing is on the class;
    #                      it is also declared as a property, so that any assignments are validated to be strings,
    #                      insuring that assignment by one instance will not affect the value of others.
    name = None

    _deepcopy_shared_keys = frozenset([
        'init_args',
    ])

    class _CompilationData(ParametersBase):
        parameter_struct = None
        state_struct = None

    def __init__(self,
                 default_variable,
                 param_defaults,
                 size=NotImplemented,  # 7/5/17 CW: this is a hack to check whether the user has passed in a size arg
                 function=None,
                 name=None,
                 prefs=None,
                 **kwargs):
        """Assign default preferences; enforce required params; validate and instantiate params and execute method

        Initialization arguments:
        - default_variable (anything): establishes type for the variable, used for validation
        - size (int or list/array of ints): if specified, establishes variable if variable was not already specified
        - params_default (dict): assigned as paramInstanceDefaults
        Note: if parameter_validation is off, validation is suppressed (for efficiency) (Component class default = on)

        """

        illegal_args = [arg for arg in kwargs.keys() if arg not in self.standard_constructor_args]
        if illegal_args:
            plural = ''
            if len(illegal_args)>1:
                plural = 's'
            raise ComponentError(f"Unrecognized argument{plural} in constructor for {self.name} "
                                 f"(type: {self.__class__.__name__}): {repr(', '.join(illegal_args))}")

        context = Context(
            source=ContextFlags.COMPONENT,
            execution_phase=ContextFlags.IDLE,
        )

        try:
            function_params = param_defaults[FUNCTION_PARAMS]
        except KeyError:
            function_params = None

        self._initialize_parameters(context=context, **param_defaults)

        v = self._handle_default_variable(default_variable, size)
        if v is None:
            default_variable = self.defaults.variable
        else:
            default_variable = v
            self.defaults.variable = default_variable

        # These ensure that subclass values are preserved, while allowing them to be referred to below
        self.paramInstanceDefaults = {}

        self.parameters.has_initializers._set(False, context)

        if kwargs and REINITIALIZE_WHEN in kwargs:
            self.reinitialize_when = kwargs[REINITIALIZE_WHEN]
        else:
            self.reinitialize_when = Never()

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
        try:
            # assign log conditions from preferences
            self.parameters.value.log_condition = self.prefs._log_pref.setting
        except AttributeError:
            pass

        # ASSIGN LOG
        from psyneulink.core.globals.log import Log
        self.log = Log(owner=self)
        # Used by run to store return value of execute
        self.results = []

        # CHECK FOR REQUIRED PARAMS

        # All subclasses must implement, in their paramClassDefaults, params of types specified in
        #     requiredClassParams (either above or in subclass defintion)
        # Do the check here, as _validate_params might be overridden by subclass
        for required_param, type_requirements in self.requiredParamClassDefaultTypes.items():
            # # Replace 'Function' placemarker with class reference:
            # type_requirements = [self.__class__ if item=='Function' else item for item in type_requirements]

            # get type for kwComponentCategory specification
            from psyneulink.core.components.functions.function import Function_Base
            if kwComponentCategory in type_requirements:
               type_requirements[type_requirements.index(kwComponentCategory)] = \
                   type(Function_Base)

            if required_param not in self.paramClassDefaults.keys():
                raise ComponentError("Parameter \'{}\' must be in paramClassDefaults for {}".
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

        if function is None:
            if FUNCTION in param_defaults and param_defaults[FUNCTION] is not None:
                function = param_defaults[FUNCTION]
            else:
                try:
                    function = self.class_defaults.function
                except AttributeError:
                    # assume function is a method on self
                    pass

        # VALIDATE VARIABLE AND PARAMS, AND ASSIGN DEFAULTS

        # TODO: the below overrides setting default values to None context,
        # at least in stateless parameters. Possibly more. Below should be
        # removed eventually

        # Validate the set passed in and assign to paramInstanceDefaults
        # By calling with assign_missing, this also populates any missing params with ones from paramClassDefaults
        self._instantiate_defaults(variable=default_variable,
               request_set=param_defaults,            # requested set
               assign_missing=True,                   # assign missing params from classPreferences to instanceDefaults
               target_set=self.paramInstanceDefaults, # destination set to which params are being assigned
               default_set=self.paramClassDefaults,   # source set from which missing params are assigned
               context=context,
               )

        self._runtime_params_reset = {}

        # KDM 9/9/19: this exists to deal with all the attribute setting in self.paramsCurrent - if not set
        # these will be included in logs as COMMAND_LINE settings. Remove this when self.paramsCurrent is removed
        self.most_recent_context = context

        # KDM: this is a poorly implemented hack that stops the .update call from
        # starting off a chain of assignment/validation calls that ends up
        # calling _instantiate_attributes_before_function and so attempting to create
        # ParameterStates twice in some cases
        self.paramsCurrent = {}
        orig_validation_pref = self.paramValidationPref
        self.paramValidationPref = PreferenceEntry(False, PreferenceLevel.INSTANCE)
        self.paramsCurrent.update(self.paramInstanceDefaults)
        self.paramValidationPref = orig_validation_pref

        # VALIDATE FUNCTION (self.function and/or self.params[function, FUNCTION_PARAMS])
        self._validate_function(function=function)

        # INSTANTIATE ATTRIBUTES BEFORE FUNCTION
        # Stub for methods that need to be executed before instantiating function
        #    (e.g., _instantiate_sender and _instantiate_receiver in Projection)
        # Allow _instantiate_attributes_before_function of subclass
        #    to modify/replace function arg provided in constructor (e.g. TransferWithCosts)
        function = self._instantiate_attributes_before_function(function=function, context=context) or function

        # INSTANTIATE FUNCTION
        #    - assign initial function parameter values from ParameterStates,
        #    - assign function's output to self.defaults.value (based on call of self.execute)
        self._instantiate_function(function=function, function_params=function_params, context=context)

        # SET CURRENT VALUES OF VARIABLE AND PARAMS
        # self.paramsCurrent = self.paramInstanceDefaults

        self._instantiate_value(context=context)

        # INSTANTIATE ATTRIBUTES AFTER FUNCTION
        # Stub for methods that need to be executed after instantiating function
        #    (e.g., instantiate_output_state in Mechanism)
        self._instantiate_attributes_after_function(context=context)

        self._validate(context=context)

        self.initialization_status = ContextFlags.INITIALIZED
        # MODIFIED 12/4/18 NEW [JDC]:
        from psyneulink.core.components.functions.function import Function_Base
        if (
            isinstance(self.function, Function_Base)
            and self.function.initialization_status != ContextFlags.DEFERRED_INIT
        ):
            self.function.initialization_status = ContextFlags.INITIALIZED
        # MODIFIED 12/4/18 END

        self._compilation_data = self._CompilationData(owner=self)

        self._update_parameter_components(context)

    def __repr__(self):
        return '({0} {1})'.format(type(self).__name__, self.name)
        #return '{1}'.format(type(self).__name__, self.name)

    def __deepcopy__(self, memo):
        fun = get_deepcopy_with_shared_Components(self._deepcopy_shared_keys)
        newone = fun(self, memo)

        if newone.parameters is not newone.class_parameters:
            # may be in DEFERRED INIT, so parameters/defaults belongs to class
            newone.parameters._owner = newone
            newone.defaults._owner = newone

        return newone

    # ------------------------------------------------------------------------------------------------------------------
    # Compilation support
    # ------------------------------------------------------------------------------------------------------------------
    def _get_compilation_state(self):
        try:
            stateful = self.stateful_attributes
        except AttributeError:
            stateful = []

        return (p for p in self.parameters if p.name in stateful or isinstance(p.get(), Component))

    def _get_state_ids(self):
        return [sp.name for sp in self._get_compilation_state()]

    def _get_state_values(self, context=None):
        def _state_values(x):
            return x._get_state_values(context) if isinstance(x, Component) else x
        return tuple(map(_state_values, (sp.get(context) for sp in self._get_compilation_state())))

    def _get_state_initializer(self, context):
        def _convert(x):
            if isinstance(x, np.random.RandomState):
                # Skip first element of random state (id string)
                return x.get_state()[1:]
            else:
                return x
        lists = (_convert(s) for s in self._get_state_values(context))
        return pnlvm._tupleize(lists)

    def _get_compilation_params(self, context=None):
        # Filter out known unused/invalid params
        black_list = {'variable', 'value', 'initializer'}
        try:
            # Don't list stateful params, the are included in context
            black_list.update(self.stateful_attributes)
        except AttributeError:
            pass
        def _is_compilation_param(p):
            if p.name not in black_list and not isinstance(p, ParameterAlias):
                val = p.get(context)
                # Check if the value is string (like integration_method)
                return not isinstance(val, (str, ComponentsMeta))
            return False

        return filter(_is_compilation_param, self.parameters)

    def _get_param_ids(self, context=None):
        return [p.name for p in self._get_compilation_params(context)]

    def _get_param_values(self, context=None):
        def _get_values(p):
            param = p.get(context)
            try:
                # Existence of parameter state changes the shape to array
                # the base value should remain the same though
                if p.name in self.owner.parameter_states:
                    param = [param]
            except AttributeError:
                pass
            if not np.isscalar(param) and param is not None:
                if p.name == 'matrix': # Flatten matrix
                    param = np.asfarray(param).flatten().tolist()
                elif isinstance(param, Component):
                    param = param._get_param_values(context)
                elif len(param) == 1 and hasattr(param[0], '__len__'): # Remove 2d. FIXME: Remove this
                    param = np.asfarray(param[0]).tolist()
            return param

        return tuple(map(_get_values, self._get_compilation_params(context)))

    def _get_param_initializer(self, context):
        return pnlvm._tupleize(self._get_param_values(context))

    def _gen_llvm_function(self, extra_args=[]):
        with pnlvm.LLVMBuilderContext.get_global() as ctx:
            args = [ctx.get_param_struct_type(self).as_pointer(),
                    ctx.get_state_struct_type(self).as_pointer(),
                    ctx.get_input_struct_type(self).as_pointer(),
                    ctx.get_output_struct_type(self).as_pointer()]
            builder = ctx.create_llvm_function(args + extra_args, self)
            llvm_func = builder.function

            llvm_func.attributes.add('alwaysinline')
            params, context, arg_in, arg_out = llvm_func.args[:len(args)]
            if len(extra_args) == 0:
                for p in params, context, arg_in, arg_out:
                    p.attributes.add('noalias')

            builder = self._gen_llvm_function_body(ctx, builder, params, context, arg_in, arg_out)
            builder.ret_void()

        return llvm_func

    # ------------------------------------------------------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------------------------------------------------------

    def _handle_default_variable(self, default_variable=None, size=None):
        """
            Finds whether default_variable can be determined using **default_variable** and **size**
            arguments.

            Returns
            -------
                a default variable if possible
                None otherwise
        """
        if self._default_variable_handled:
            return default_variable

        default_variable = self._parse_arg_variable(default_variable)

        if default_variable is None:
            default_variable = self._handle_size(size, default_variable)

            if default_variable is None or default_variable is NotImplemented:
                self._default_variable_handled = True
                return None
            else:
                self._default_variable_flexibility = DefaultsFlexibility.RIGID
        else:
            self._default_variable_flexibility = DefaultsFlexibility.RIGID

        self._default_variable_handled = True

        return convert_to_np_array(default_variable, dimension=1)

    # IMPLEMENTATION NOTE: (7/7/17 CW) Due to System and Process being initialized with size at the moment (which will
    # be removed later), I’m keeping _handle_size in Component.py. I’ll move the bulk of the function to Mechanism
    # through an override, when Composition is done. For now, only State.py overwrites _handle_size().
    def _handle_size(self, size, variable):
        """If variable is None, _handle_size tries to infer variable based on the **size** argument to the
            __init__() function. This method is overwritten in subclasses like Mechanism and State.
            If self is a Mechanism, it converts variable to a 2D array, (for a Mechanism, variable[i] represents
            the input from the i-th input state). If self is a State, variable is a 1D array and size is a length-1 1D
            array. It performs some validations on size and variable as well. This function is overridden in State.py.
            If size is NotImplemented (usually in the case of Projections/Functions), then this function passes without
            doing anything. Be aware that if size is NotImplemented, then variable is never cast to a particular shape.
        """
        if size is not NotImplemented:
            self._default_variable_flexibility = DefaultsFlexibility.RIGID
            # region Fill in and infer variable and size if they aren't specified in args
            # if variable is None and size is None:
            #     variable = self.class_defaults.variable
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

            if variable is not None:
                variable = np.array(variable)
                if variable.dtype == object:
                    # CAVEAT: assuming here that object dtype implies there are list objects (i.e. array with
                    # different sized arrays/lists inside like [[0, 1], [2, 3, 4]]), even though putting a None
                    # value in the array will give object dtype. This case doesn't really make sense in our
                    # context though, so ignoring this case in the interest of quickly fixing 3D variable behavior
                    variable = np.atleast_1d(variable)
                else:
                    variable = np.atleast_2d(variable)

                variable = convert_all_elements_to_np_array(variable)

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

            # implementation note: for good coding practices, perhaps add setting to enable easy change of the default
            # value of variable (though it's an unlikely use case), which is an array of zeros at the moment
            if variable is None and size is not None:
                try:
                    variable = []
                    for s in size:
                        variable.append(np.zeros(s))
                    variable = np.array(variable)
                except:
                    raise ComponentError("variable (possibly default_variable) was not specified, but PsyNeuLink "
                                         "was unable to infer variable from the size argument, {}. size should be"
                                         " an integer or an array or list of integers. Either size or "
                                         "variable must be specified.".format(size))

            # the two regions below (creating size if it's None and/or expanding it) are probably obsolete (7/7/17 CW)

            if size is None and variable is not None:
                size = []
                try:
                    for input_vector in variable:
                        size.append(len(input_vector))
                    size = np.array(size)
                except:
                    raise ComponentError(
                            "{}: size was not specified, and unable to infer it from the variable argument ({}) "
                            "-- it can be an array, list, a 2D array, a list of arrays, array of lists, etc. ".
                                format(self.name, variable))
            # endregion

            if size is not None and variable is not None:
                if len(size) == 1 and len(variable) > 1:
                    new_size = np.empty(len(variable))
                    new_size.fill(size[0])
                    size = new_size

            # the two lines below were used when size was a param and are likely obsolete (7/7/17 CW)
            # param_defaults['size'] = size  # 7/5/17 potentially buggy? Not sure (CW)
            # self.user_params_for_instantiation['size'] = None  # 7/5/17 VERY HACKY: See Changyan's Notes on this.

            # Both variable and size are specified
            if variable is not None and size is not None:
                # If they conflict, give warning
                if len(size) != len(variable):
                    if hasattr(self, 'prefs') and hasattr(self.prefs, kpVerbosePref) and self.prefs.verbosePref:
                        warnings.warn("The size arg of {} conflicts with the length "
                                      "of its variable arg ({}) at element {}: variable takes precedence".
                                      format(self.name, size, variable))
                else:
                    for i in range(len(size)):
                        if size[i] != len(variable[i]):
                            if hasattr(self, 'prefs') and hasattr(self.prefs, kpVerbosePref) and self.prefs.verbosePref:
                                warnings.warn("The size arg of {} ({}) conflicts with the length "
                                              "of its variable arg ({}) at element {}: variable takes precedence".
                                              format(self.name, size[i], variable[i], i))

        return variable

    @handle_external_context()
    def _deferred_init(self, context=None):
        """Use in subclasses that require deferred initialization
        """
        if self.initialization_status == ContextFlags.DEFERRED_INIT:

            # Flag that object is now being initialized
            #       (usually in _instantiate_function)
            self.initialization_status = ContextFlags.INITIALIZING

            del self.init_args['self']

            try:
                del self.init_args['__class__']
            except KeyError:
                pass

            # Delete reference to dict created by paramsCurrent -> ParamsDict
            try:
                del self.init_args['__pydevd_ret_val_dict']
            except KeyError:
                pass

            self.init_args['context'] = context

            # Complete initialization
            # MODIFIED 10/27/18 OLD:
            super(self.__class__,self).__init__(**self.init_args)
            # MODIFIED 10/27/18 NEW:  FOLLOWING IS NEEDED TO HANDLE FUNCTION DEFERRED INIT (JDC)
            # try:
            #     super(self.__class__,self).__init__(**self.init_args)
            # except:
            #     self.__init__(**self.init_args)
            # MODIFIED 10/27/18 END

            # If name was assigned, "[DEFERRED INITIALIZATION]" was appended to it, so remove it
            if DEFERRED_INITIALIZATION in self.name:
                self.name = self.name.replace("["+DEFERRED_INITIALIZATION+"]","")
            # Otherwise, allow class to replace std default name with class-specific one if it has a method for doing so
            else:
                self._assign_default_name()

            self.initialization_status = ContextFlags.INITIALIZED

    def _assign_deferred_init_name(self, name, context):

        name = "{} [{}]".format(name,DEFERRED_INITIALIZATION) if name \
          else "{} {}".format(DEFERRED_INITIALIZATION,self.__class__.__name__)

        # Register with ProjectionRegistry or create one
        register_category(entry=self,
                          base_class=Component,
                          name=name,
                          registry=DeferredInitRegistry,
                          context=context)

    def _assign_default_name(self, **kwargs):
        return

    def _assign_args_to_param_dicts(self, defaults=None, **kwargs):
        """Assign args passed in __init__() to params

        Get args and their corresponding values in call to constructor
        - get default values for all args and assign to class.paramClassDefaults if they have not already been
        - assign arg values to local copy of params dict
        - override those with any values specified in params dict passed as "params" arg

        Accepts defaults dict that, if provided, overrides any values assigned to arguments in self.__init__

        """

        # Get args in call to constructor and create dictionary of their default values (for use below)
        # Create dictionary of default values for args
        defaults_dict = {}
        for arg_name, arg in inspect.signature(self.__init__).parameters.items():
            defaults_dict[arg_name] = arg.default
        if defaults:
            defaults_dict.update(defaults)
        def default(val):
            try:
                return defaults_dict[val]
            except KeyError:
                # FIX: IF CUSTOM_FUNCTION IS IN PARAMS, TRY GETTING ITS ARGS
                # raise ComponentError("PROGRAM ERROR: \'{}\' not declared in {}.__init__() "
                #                      "but expected by its parent class ({}).".
                #                      format(val,
                #                             self.__class__.__name__,
                #                             self.__class__.__bases__[0].__name__))
                pass

        def parse_arg(arg):
            # Resolve the string value of any args that use keywords as their name
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

            # parse the argument either by specialized parser or generic
            try:
                kwargs[arg_name] = getattr(self, '_parse_arg_' + arg_name)(kwargs[arg_name])
            except AttributeError:
                kwargs[arg_name] = self._parse_arg_generic(kwargs[arg_name])

            # The params arg is never a default (nor is anything in it)
            if arg_name is PARAMS or arg_name is VARIABLE:
                continue

            # Check if param exists in paramClassDefaults
            try:
                self.paramClassDefaults[arg]

            # param corresponding to arg is NOT in paramClassDefaults, so add it
            except KeyError:
                # Get defaults values for args listed in FUNCTION_PARAMS
                # Note:  is not an arg, but rather used to package args that belong to a non-instantiated function
                if arg is FUNCTION_PARAMS:
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

            # param corresponding to arg IS already in paramClassDefaults
            else:
                # param has a value but paramClassDefaults is None, so assign param's value to paramClassDefaults
                if self.paramClassDefaults[arg] is None and arg in defaults_dict and defaults_dict[arg] is not None:
                    self.paramClassDefaults[arg] = defaults_dict[arg]
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
                    function_instance = function()
                    # Get copy of default params
                    # IMPLEMENTATION NOTE: this is needed so that function_params gets included in user_params and
                    #                      thereby gets instantiated as a property in _create_attributes_for_params
                    params[FUNCTION_PARAMS] = ReadOnlyOrderedDict(name=FUNCTION_PARAMS)
                    for param_name in sorted(list(function_instance.user_params.keys())):
                        params[FUNCTION_PARAMS].__additem__(param_name, function_instance.user_params[param_name])
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
                    from psyneulink.core.components.functions.function import Function

                    # It is a PsyNeuLink Function
                    # IMPLEMENTATION NOTE:  REPLACE THIS WITH "CONTINUE" ONCE _instantiate_function IS REFACTORED TO
                    #                       TO ALLOW Function SPECIFICATION (VS. ONLY CLASS)
                    if isinstance(function, Function):
                        # Set it to the class (for compatibility with current implementation of _instantiate_function()
                        params[FUNCTION] = function
                        # Create ReadOnlyDict for FUNCTION_PARAMS and copy function's params into it
                        params[FUNCTION_PARAMS] = ReadOnlyOrderedDict(name=FUNCTION_PARAMS)
                        for param_name in sorted(list(function.user_params_for_instantiation.keys())):
                            params[FUNCTION_PARAMS].__additem__(param_name,
                                                                function.user_params_for_instantiation[param_name])

                    # It is a generic function
                    # # MODIFIED 2/26/18 OLD:
                    # elif inspect.isfunction(function):
                    # MODIFIED 2/26/18 NEW:
                    elif (inspect.isfunction(function) or inspect.ismethod(function)):
                    # MODIFIED 2/26/18 END
                        # Assign as is (i.e., don't convert to class), since class is generic
                        # (_instantiate_function also tests for this and leaves it as is)
                        params[FUNCTION] = function
                        params[FUNCTION_PARAMS] = ReadOnlyOrderedDict(name=FUNCTION_PARAMS)
                        if hasattr(self, '_prefs') and self.verbosePref:
                            warnings.warn("{} is not a PsyNeuLink Function, "
                                          "therefore runtime_params cannot be used".format(default(arg).__name__))
                    else:
                        try:
                            params[FUNCTION] = self.class_defaults.function
                        except AttributeError:
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
            # if params and FUNCTION in params and FUNCTION in params_arg:
            #     # Check if it is the same as the default or the one assigned in the function arg of the constructor
            #     if not is_same_function_spec(params[FUNCTION], params_arg[FUNCTION]):
            #         # If it is not the same, delete any function params that have already been assigned
            #         #    in params[] for the function specified in the function arg of the constructor
            #         if FUNCTION_PARAMS in params:
            #             for param in list(params[FUNCTION_PARAMS].keys()):
            #                 params[FUNCTION_PARAMS].__deleteitem__(param)
            try:
                # Replace any parameters for function specified in function arg of constructor
                #    with those specified either in FUNCTION_PARAMS entry of params dict
                #    or for an instantiated function specified in FUNCTION entry of params dict

                # First, if the function is instantiated, get the parameters from its user_params dict
                from psyneulink.core.components.functions.function import Function
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

            # MODIFIED 6/29/18 OLD:
            params.update(params_arg)
            # # MODIFIED 6/29/18 NEW JDC:
            # for item in params_arg:
            #     if params_arg[item] is not None:
            #         params.update({item: params_arg[item]})
            # MODIFIED 6/29/18 END

        # Save user-accessible params
        # self.user_params = params.copy()
        self.user_params = ReadOnlyOrderedDict(name=USER_PARAMS)
        for param_name in sorted(list(params.keys())):
            # copy dicts because otherwise if you ever modify the dict specified
            # as the params argument, you will change the user_params of this object
            # and any other object instantiated with that dict
            new_param_val = params[param_name]
            if isinstance(new_param_val, dict):
                new_param_val = new_param_val.copy()
            elif isinstance(new_param_val, (dict, ReadOnlyOrderedDict)):
                # construct the ROOD key by key because disallows standard creation
                val_dict = new_param_val.copy()
                new_param_val = ReadOnlyOrderedDict()
                for k in val_dict:
                    new_param_val.__additem__(k, val_dict[k])

            self.user_params.__additem__(param_name, new_param_val)

        # Cache a (deep) copy of the user-specified values and put it in user_params_for_instantiation;
        #    this is to deal with the following:
        #    • _create_attributes_for_params assigns properties to each param in user_params;
        #    • the setter for those properties (in make_property) also assigns its value to its entry user_params;
        #    • paramInstanceDefaults are assigned to paramsCurrent in Component.__init__ assigns
        #    • since paramsCurrent is a ParamsDict, it assigns the values of its entries to the corresponding attributes
        #         and the setter assigns those values to the user_params
        #    • therefore, assignments of paramInstance defaults to paramsCurrent in __init__ overwrites the
        #         the user-specified values (from the constructor args) in user_params
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
            # getter returns backing field value
            # setter runs validation [_assign_params()], updates user_params

            for arg_name, arg_value in kwargs.items():
                if not any(hasattr(parent_class, arg_name) for parent_class in self.__class__.mro()):
                    # create property
                    setattr(self.__class__, arg_name, make_property(arg_name))
                # assign default value
                setattr(self, "_" + arg_name, arg_value)
        else:
            for arg_name, arg_value in kwargs.items():
                setattr(self, arg_name, arg_value)

    def _set_parameter_value(self, param, val, context=None):
        getattr(self.parameters, param)._set(val, context)
        if hasattr(self, "parameter_states"):
            if param in self.parameter_states:
                new_state_value = self.parameter_states[param].execute(
                    context=Context(execution_phase=ContextFlags.EXECUTING, execution_id=context.execution_id)
                )
                self.parameter_states[param].parameters.value._set(new_state_value, context)
        elif hasattr(self, "owner"):
            if hasattr(self.owner, "parameter_states"):
                if param in self.owner.parameter_states:
                    new_state_value = self.owner.parameter_states[param].execute(
                        context=Context(execution_phase=ContextFlags.EXECUTING, execution_id=context.execution_id)
                    )
                    self.owner.parameter_states[param].parameters.value._set(new_state_value, context)

    def _check_args(self, variable=None, params=None, context=None, target_set=None):
        """validate variable and params, instantiate variable (if necessary) and assign any runtime params.

        Called by functions to validate variable and params
        Validation can be suppressed by turning parameter_validation attribute off
        target_set is a params dictionary to which params should be assigned;
           otherwise, they are assigned to paramsCurrent;

        Does the following:
        - instantiate variable (if missing or callable)
        - validate variable if PARAM_VALIDATION is set
        - resets leftover runtime params back to original values (only if execute method was called directly)
        - sets runtime params
        - validate params if PARAM_VALIDATION is set

        :param variable: (anything but a dict) - variable to validate
        :param params: (dict) - params to validate
        :target_set: (dict) - set to which params should be assigned (default: self.paramsCurrent)
        :return:
        """
        # VARIABLE ------------------------------------------------------------

        # If function is called without any arguments, get default for variable
        if variable is None:
            try:
                # assigned by the Function class init when initializing
                variable = self.defaults.variable
            except AttributeError:
                variable = self.class_defaults.variable

        # If the variable is a function, call it
        if callable(variable):
            variable = variable()

        # Validate variable if parameter_validation is set and the function was called with a variable
        if self.prefs.paramValidationPref and variable is not None:
            variable = self._validate_variable(variable, context=context)

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

        # reset any runtime params that were leftover from a direct call to .execute (atypical)
        if context.execution_id in self._runtime_params_reset:
            for key in self._runtime_params_reset[context.execution_id]:
                self._set_parameter_value(key, self._runtime_params_reset[context.execution_id][key], context)
        self._runtime_params_reset[context.execution_id] = {}

        # If params have been passed, treat as runtime params
        runtime_params = params
        if isinstance(runtime_params, dict):
            for param_name in runtime_params:
                # (1) store current attribute value in _runtime_params_reset so that it can be reset later
                # (2) assign runtime param values to attributes (which calls validation via properties)
                # (3) update parameter states if needed
                if hasattr(self, param_name):
                    if param_name in {FUNCTION, INPUT_STATES, OUTPUT_STATES}:
                        continue
                    if context.execution_id not in self._runtime_params_reset:
                        self._runtime_params_reset[context.execution_id] = {}
                    self._runtime_params_reset[context.execution_id][param_name] = getattr(self.parameters, param_name)._get(context)
                    self._set_parameter_value(param_name, runtime_params[param_name], context)
        elif runtime_params:    # not None
            raise ComponentError("Invalid specification of runtime parameters for {}".format(self.name))

        self.parameters.variable._set(variable, context=context)
        return variable

    @handle_external_context()
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

        :param variable: (anything but a dict (variable) - value to assign as defaults.variable
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


        # FIX: 6/3/19 [JDC] SHOULD DEAL WITH THIS AND SHAPE BELOW
        # # GET VARIABLE FROM PARAM DICT IF SPECIFIED
        # #    (give precedence to that over variable arg specification)
        # if VARIABLE in request_set and request_set[VARIABLE] is not None:
        #     variable = request_set[VARIABLE]

        # ASSIGN SHAPE TO VARIABLE if specified

        if hasattr(self, 'shape') and self.shape is not None:
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

        if not (context.source & (ContextFlags.COMMAND_LINE | ContextFlags.PROPERTY)):
            # if variable has been passed then validate and, if OK, assign as self.defaults.variable
            variable = self._validate_variable(variable, context=context)

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
            if context.source & (ContextFlags.COMMAND_LINE | ContextFlags.PROPERTY):
                default_set = {}
                for param_name in request_set:
                    try:
                        default_set[param_name] = self.paramInstanceDefaults[param_name]
                    except KeyError:
                        pass
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

                # FUNCTION class has changed, so replace rather than update FUNCTION_PARAMS
                if param_name is FUNCTION:
                    try:
                        if function_class != default_function_class and context.source & ContextFlags.COMMAND_LINE:
                            from psyneulink.core.components.functions.function import Function_Base
                            if isinstance(function, Function_Base):
                                request_set[FUNCTION] = function.__class__
                            default_set[FUNCTION_PARAMS] = function.user_params
                    # function not yet defined, so allow FUNCTION_PARAMS)
                    except UnboundLocalError:
                        pass
                # FIX: MAY NEED TO ALSO ALLOW assign_default_FUNCTION_PARAMS FOR COMMAND_LINE IN CONTEXT

                if param_name is FUNCTION_PARAMS and not self.assign_default_FUNCTION_PARAMS:
                    continue

                # Don't replace requested entry with default
                if param_name in request_set:
                    continue

                # Add to request_set any entries it is missing fron the default_set
                request_set.setdefault(param_name, param_value)
                # Update any values in a dict
                if isinstance(param_value, dict):
                    for dict_entry_name, dict_entry_value in param_value.items():
                        # Don't replace requested entries
                        if dict_entry_name in request_set[param_name]:
                            continue
                        request_set[param_name].setdefault(dict_entry_name, dict_entry_value)

        # VALIDATE PARAMS

        # if request_set has been passed or created then validate and, if OK, assign params to target_set
        if request_set:
            # For params that are a 2-item tuple, extract the value; and get value of single item modulatory specs
            # Do this both for validation and assignment;
            #   tuples and modulatory specs are left intact in user_params_for_instantiation dict
            #   which are used to instantiate the specified Components
            # IMPLEMENTATION NOTE:  Do this here rather than in _validate_params, as it needs to be done before
            #                       any override of _validate_params, which (should not, but) may process params
            #                       before calling super()._validate_params
            for param_name, param_value in request_set.items():
                if isinstance(param_value, tuple):
                    param_value = self._get_param_value_from_tuple(param_value)
                elif isinstance(param_value, (str, Component, type)):
                    param_value = self._get_param_value_for_modulatory_spec(param_name, param_value)
                else:
                    continue
                request_set[param_name] = param_value

            try:
                self._validate_params(variable=variable,
                                      request_set=request_set,
                                      target_set=target_set,
                                      context=context)
            # variable not implemented by Mechanism subclass, so validate without it
            except TypeError:
                self._validate_params(request_set=request_set,
                                      target_set=target_set,
                                      context=context)

    def _initialize_parameters(self, context=None, **param_defaults):
        self.parameters = self.Parameters(owner=self, parent=self.class_parameters)

        # assign defaults based on pass in params and class defaults
        defaults = {
            k: copy.deepcopy(v) for (k, v) in self.class_defaults.values(show_all=True).items()
            if not isinstance(getattr(self.class_parameters, k), ParameterAlias)
        }
        try:
            function_params = param_defaults[FUNCTION_PARAMS]
        except KeyError:
            function_params = None

        if param_defaults is not None:
            # Exclude any function_params from the items to set on this Component
            # because these should just be pointers to the parameters of the same
            # name on this Component's function
            # Exclude any pass parameters whose value is None (assume this means "use the normal default")
            d = {
                k: v for (k, v) in param_defaults.items()
                if (
                    k not in defaults
                    or (
                        (function_params is None or k not in function_params)
                        and v is not None
                    )
                )
            }
            for p in d:
                try:
                    getattr(self.parameters, p)._user_specified = True
                except AttributeError:
                    pass
            defaults.update(d)

        self.defaults = Defaults(owner=self, **defaults)

        for p in self.parameters:
            # set default to None context to ensure it exists
            if p.getter is None and p._get(context) is None:
                try:
                    attr_name = '_{0}'.format(p.name)
                    attr_value = getattr(self, attr_name)
                    if attr_value is None:
                        attr_value = p.default_value

                    p._set(attr_value, context=context, skip_history=True)
                    delattr(self, attr_name)
                except AttributeError:
                    p._set(p.default_value, context=context, skip_history=True)

    @handle_external_context()
    def assign_params(self, request_set=None, context=None):
        """Validates specified params, adds them TO paramInstanceDefaults, and instantiates any if necessary

        Call _instantiate_defaults with context = COMMAND_LINE, and "validated_set" as target_set.
        Update paramInstanceDefaults with validated_set so that any instantiations (below) are done in proper context.
        Instantiate any items in request set that require it (i.e., function or states).

        """
        self._assign_params(request_set=request_set, context=context)

    @tc.typecheck
    @handle_external_context()
    def _assign_params(self, request_set:tc.optional(dict)=None, context=None):
        from psyneulink.core.components.functions.function import Function

        if not request_set:
            if self.verbosePref:
                warnings.warn("No params specified")
            return

        validated_set = {}

        self._instantiate_defaults(request_set=request_set,
                                   target_set=validated_set,
                                    assign_missing=False,
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
        if INPUT_STATES in validated_set_param_names and context.source & ContextFlags.COMMAND_LINE:
            self._instantiate_attributes_before_function(context=context)

        # If the object's function is being assigned, and it is a class, instantiate it as a Function object
        if FUNCTION in validated_set and inspect.isclass(self.function):
            self._instantiate_function(context=context)
        # FIX: WHY SHOULD IT BE CALLED DURING STANDRD INIT PROCEDURE?
        # # MODIFIED 5/5/17 OLD:
        # if OUTPUT_STATES in validated_set:
        # MODIFIED 5/5/17 NEW:  [THIS FAILS WITH A SPECIFICATION IN output_states ARG OF CONSTRUCTOR]
        if OUTPUT_STATES in validated_set and context.source & ContextFlags.COMMAND_LINE:
        # MODIFIED 5/5/17 END
            self._instantiate_attributes_after_function(context=context)

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

    def _initialize_from_context(self, context, base_context=Context(execution_id=None), override=True, visited=None):
        if visited is None:
            visited = set()

        for comp in self._dependent_components:
            if comp not in visited:
                visited.add(comp)
                comp._initialize_from_context(context, base_context, override, visited=visited)

        non_alias_params =  [p for p in self.stateful_parameters if not isinstance(p, ParameterAlias)]
        for param in non_alias_params:
            if param.setter is None:
                param._initialize_from_context(context, base_context, override)

        # attempt to initialize any params with setters (some params with setters may depend on the
        # initialization of other params)
        # this pushes the problem down one level so that if there are two such that they depend on each other,
        # it will still fail. in this case, it is best to resolve the problem in the setter with a default
        # initialization value
        for param in non_alias_params:
            if param.setter is not None:
                param._initialize_from_context(context, base_context, override)

    def _delete_contexts(self, *contexts, check_simulation_storage=False, visited=None):
        if visited is None:
            visited = set()

        for comp in self._dependent_components:
            if comp not in visited:
                visited.add(comp)
                comp._delete_contexts(*contexts, check_simulation_storage=check_simulation_storage, visited=visited)

        for param in self.stateful_parameters:
            if not check_simulation_storage or not param.retain_old_simulation_data:
                for context in contexts:
                    param.delete(context)

    def _set_all_parameter_properties_recursively(self, visited=None, **kwargs):
        if visited is None:
            visited = set()

        # sets a property of all parameters for this component and all its dependent components
        # used currently for disabling history, but setting logging could use this
        for param_name in self.parameters.names():
            parameter = getattr(self.parameters, param_name)
            for (k, v) in kwargs.items():
                try:
                    setattr(parameter, k, v)
                except ParameterError as e:
                    logger.warning(str(e) + ' Parameter has not been modified.')

        for comp in self._dependent_components:
            if comp not in visited:
                visited.add(comp)
                comp._set_all_parameter_properties_recursively(
                    visited=visited,
                    **kwargs
                )

    def _set_multiple_parameter_values(self, context, **kwargs):
        """
            Unnecessary, but can simplify multiple parameter assignments at once
            For every kwarg k, v pair, will attempt to set self.parameters.<k> to v for context
        """
        for (k, v) in kwargs.items():
            getattr(self.parameters, k)._set(v, context)

    # ------------------------------------------------------------------------------------------------------------------
    # Parsing methods
    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------
    # Argument parsers
    # ---------------------------------------------------------

    def _parse_arg_generic(self, arg_val):
        """
            Argument parser for any argument that does not have a specialized parser
        """
        return arg_val

    def _parse_arg_variable(self, variable):
        """
            Transforms **variable** into a form that Components expect. Used to allow
            users to pass input in convenient forms, like a single float when a list
            for input states is expected

            Returns
            -------
            The transformed **input**
        """
        if variable is None:
            return variable

        if not isinstance(variable, (list, np.ndarray)):
            variable = np.atleast_1d(variable)

        return convert_all_elements_to_np_array(variable)

    # ---------------------------------------------------------
    # Misc parsers
    # ---------------------------------------------------------

    def _parse_function_variable(self, variable, context=None):
        """
            Parses the **variable** passed in to a Component into a function_variable that can be used with the
            Function associated with this Component
        """
        return variable

    # ------------------------------------------------------------------------------------------------------------------
    # Validation methods
    # ------------------------------------------------------------------------------------------------------------------

    def _validate(self, context=None):
        """
            Eventually should contain all validation methods, occurs at end of Component.__init__
        """
        # 4/18/18 kmantel: below is a draft of what such a method should look like
        # it's beyond the scope of the current changes however

        # # currently allows chance to validate anything in constructor defaults
        # # when fleshed out, this should go over the new Parameters structure
        # for param, _ in self.get_param_class_defaults().items():
        #     try:
        #         # automatically call methods of the form _validate_<param name> with the attribute
        #         # as single argument. Sticking to this format can allow condensed and modular validation
        #         getattr(self, '_validate_' + param)(getattr(self, param))
        #     except AttributeError:
        #         pass
        self._validate_value()

    def _validate_variable(self, variable, context=None):
        """Validate variable and return validated variable

        Convert self.class_defaults.variable specification and variable (if specified) to list of 1D np.ndarrays:

        VARIABLE SPECIFICATION:                                        ENCODING:
        Simple value variable:                                         0 -> [array([0])]
        Single state array (vector) variable:                         [0, 1] -> [array([0, 1])]
        Multiple state variables, each with a single value variable:  [[0], [0]] -> [array[0], array[0]]

        Perform top-level type validation of variable against the self.class_defaults.variable;
            if the type is OK, the value is returned (which should be used by the function)
        This can be overridden by a subclass to perform more detailed checking (e.g., range, recursive, etc.)
        It is called only if the parameter_validation attribute is `True` (which it is by default)

        IMPLEMENTATION NOTES:
           * future versions should add hierarchical/recursive content (e.g., range) checking
           * add request/target pattern?? (as per _validate_params) and return validated variable?

        :param variable: (anything other than a dictionary) - variable to be validated:
        :param context: (str)
        :return variable: validated variable
        """

        if inspect.isclass(variable):
            raise ComponentError(f"Assignment of class ({variable.__name__}) "
                                 f"as a variable (for {self.name}) is not allowed.")

        # If variable is not specified, then:
        #    - assign to (??now np-converted version of) self.class_defaults.variable
        #    - mark as not having been specified
        #    - return
        if variable is None:
            try:
                return self.defaults.variable
            except AttributeError:
                return self.class_defaults.variable

        # Otherwise, do some checking on variable before converting to np.ndarray

        # If variable is callable (function or object reference), call it and assign return to value to variable
        # Note: check for list is necessary since function references must be passed wrapped in a list so that they are
        #       not called before being passed
        if isinstance(variable, list) and callable(variable[0]):
            variable = variable[0]()
        # NOTE (7/24/17 CW): the above two lines of code can be commented out without causing any current tests to fail
        # So we should either write tests for this piece of code, or remove it.
        # Convert variable to np.ndarray
        # Note: this insures that variable will be AT LEAST 1D;  however, can also be higher:
        #       e.g., given a list specification of [[0],[0]], it will return a 2D np.array
        variable = convert_to_np_array(variable, 1)

        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate params and assign validated values to targets,

        This performs top-level type validation of params against the paramClassDefaults specifications:
            - checks that param is listed in paramClassDefaults
            - checks that param value is compatible with on in paramClassDefaults
            - if param is a dict, checks entries against corresponding entries paramClassDefaults
            - if all is OK, the value is assigned to the target_set (if it has been provided)
            - otherwise, an exception is raised

        This can be overridden by a subclass to perform more detailed checking (e.g., range, recursive, etc.)
        It is called only if the parameter_validation attribute is `True` (which it is by default)

        IMPLEMENTATION NOTES:
           * future versions should add recursive and content (e.g., range) checking
           * should method return validated param set?

        :param dict (request_set) - set of params to be validated:
        :param dict (target_set) - repository of params that have been validated:
        :return none:
        """

        for param_name, param_value in request_set.items():
            # setattr(self, "_"+param_name, param_value)

            # Check that param is in paramClassDefaults (if not, it is assumed to be invalid for this object)
            if not param_name in self.paramClassDefaults:
                # these are always allowable since they are attribs of every Component
                if param_name in {VARIABLE, NAME, VALUE, PARAMS, SIZE, LOG_ENTRIES, FUNCTION_PARAMS}:
                    continue
                raise ComponentError(f"{param_name} is not a valid parameter for {self.__class__.__name__}.")

            # The value of the param is None in paramClassDefaults: suppress type checking
            # IMPLEMENTATION NOTE: this can be used for params with multiple possible types,
            #                      until type lists are implemented (see below)
            if self.paramClassDefaults[param_name] is None or self.paramClassDefaults[param_name] is NotImplemented:
                if self.prefs.verbosePref:
                    warnings.warn(f"{param_name} is specified as None for {self.name} which suppresses type checking.")
                if target_set is not None:
                    target_set[param_name] = param_value
                continue

            # If the value in paramClassDefault is a type, check if param value is an instance of it
            if inspect.isclass(self.paramClassDefaults[param_name]):
                if isinstance(param_value, self.paramClassDefaults[param_name]):
                    target_set[param_name] = param_value
                    continue
                # If the value is a Function class, allow any instance of Function class
                from psyneulink.core.components.functions.function import Function_Base
                if issubclass(self.paramClassDefaults[param_name], Function_Base):
                    # if isinstance(param_value, (function_type, Function_Base)):  <- would allow function of any kind
                    if isinstance(param_value, Function_Base):
                        target_set[param_name] = param_value
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
            from psyneulink.core.components.shellclasses import Projection
            if (((isinstance(param_value, str) and
                          param_value in {CONTROL_PROJECTION, LEARNING_PROJECTION, LEARNING}) or
                isinstance(param_value, Projection) or  # These should be just ControlProjection or LearningProjection
                inspect.isclass(param_value) and issubclass(param_value,(Projection)))
                and not param_name is FUNCTION):
                param_value = self.paramClassDefaults[param_name]

            # If self is a Function and param is a class ref for function, instantiate it as the function
            from psyneulink.core.components.functions.function import Function_Base
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
                    if param_name is FUNCTION_PARAMS:
                        if not self.assign_default_FUNCTION_PARAMS:
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
                            # if param_name != FUNCTION_PARAMS:
                            #     assert True
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
                    from collections.abc import Iterable
                    if not isinstance(param_value, Iterable) or isinstance(param_value, str):
                        target_set[param_name] = param_value
                    else:
                        target_set[param_name] = param_value.copy()

            # If param is a function_type (or it has a function attribute that is one), allow any other function_type
            elif callable(param_value):
                target_set[param_name] = param_value
            elif hasattr(param_value, FUNCTION) and callable(param_value.function):
                target_set[param_name] = param_value

            # It has already passed as the name of a valid param, so let it pass;
            #    value should be validated in subclass _validate_params override
            elif isinstance(param_name, str):
                # FIX: 10/3/17 - THIS IS A HACK;  IT SHOULD BE HANDLED EITHER
                # FIX:           MORE GENERICALLY OR LOCALLY (E.G., IN OVERRIDE OF _validate_params)
                if param_name == 'matrix':
                    if is_matrix(self.paramClassDefaults[param_name]):
                        # FIX:  ?? ASSIGN VALUE HERE, OR SIMPLY ALLOW AND ASSUME IT WILL BE PARSED ELSEWHERE
                        # param_value = self.paramClassDefaults[param_name]
                        # target_set[param_name] = param_value
                        target_set[param_name] = param_value
                    else:
                        raise ComponentError("Value of {} param for {} ({}) must be a valid matrix specification".
                                             format(param_name, self.name, param_value))
                target_set[param_name] = param_value

            # Parameter is not a valid type
            else:
                if type(self.paramClassDefaults[param_name]) is type:
                    type_name = 'the name of a subclass of ' + self.paramClassDefaults[param_name].__base__.__name__
                raise ComponentError("Value of {} param for {} ({}) is not compatible with {}".
                                    format(param_name, self.name, param_value, type_name))

    def _get_param_value_for_modulatory_spec(self, param_name, param_value):
        from psyneulink.core.globals.keywords import MODULATORY_SPEC_KEYWORDS
        if isinstance(param_value, str):
            param_spec = param_value
        elif isinstance(param_value, Component):
            param_spec = param_value.__class__.__name__
        elif isinstance(param_value, type):
            param_spec = param_value.__name__
        else:
            raise ComponentError("PROGRAM ERROR: got {} instead of string, Component, or Class".format(param_value))

        if not param_spec in MODULATORY_SPEC_KEYWORDS:
            return(param_value)

        try:
            param_default_value = self.paramClassDefaults[param_name]
            # Only assign default value if it is not None
            if param_default_value is not None:
                return param_default_value
            else:
                return param_value
        except:
            raise ComponentError("PROGRAM ERROR: Could not get default value for {} of {} (to replace spec as {})".
                                 format(param_name, self.name, param_value))

    def _get_param_value_from_tuple(self, param_spec):
        """Returns param value (first item) of a (value, projection) tuple;
        """
        from psyneulink.core.components.mechanisms.adaptive.adaptivemechanism import AdaptiveMechanism_Base
        from psyneulink.core.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base
        from psyneulink.core.components.states.modulatorysignals.modulatorysignal import ModulatorySignal

        ALLOWABLE_TUPLE_SPEC_KEYWORDS = MODULATORY_SPEC_KEYWORDS
        ALLOWABLE_TUPLE_SPEC_CLASSES = (ModulatoryProjection_Base, ModulatorySignal, AdaptiveMechanism_Base)

        # If the 2nd item is a CONTROL or LEARNING SPEC, return the first item as the value
        if (isinstance(param_spec, tuple) and len(param_spec) is 2 and
                not isinstance(param_spec[1], (dict, list, np.ndarray)) and
                (param_spec[1] in ALLOWABLE_TUPLE_SPEC_KEYWORDS or
                 isinstance(param_spec[1], ALLOWABLE_TUPLE_SPEC_CLASSES) or
                 (inspect.isclass(param_spec[1]) and issubclass(param_spec[1], ALLOWABLE_TUPLE_SPEC_CLASSES)))
            ):
            value =  param_spec[0]

        # Otherwise, just return the tuple
        else:
            value = param_spec

        return value

    def _validate_function(self, function):
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

        from psyneulink.core.components.shellclasses import Function

        # FUNCTION is not specified, so try to assign self.function to it
        if function is None:
            try:
                function = self.function
            except AttributeError:
                # self.function is also missing, so raise exception
                raise ComponentError("{} must either implement a function method, specify one as the FUNCTION param in"
                                    " paramClassDefaults, or as the default for the function argument in its init".
                                    format(self.__class__.__name__, FUNCTION))

        # self.function is None
        # IMPLEMENTATION NOTE:  This is a coding error;  self.function should NEVER be assigned None
        if function is None:
            raise ComponentError("PROGRAM ERROR: either {0} must be specified or {1}.function must be implemented for {2}".
                  format(FUNCTION,self.__class__.__name__, self.name))
        # self.function is OK, so return
        elif (
            isinstance(function, types.FunctionType)
            or isinstance(function, types.MethodType)
            or is_instance_or_subclass(function, Function)
        ):
            self.paramsCurrent[FUNCTION] = function
            return
        # self.function is NOT OK, so raise exception
        else:
            raise ComponentError("{0} not specified and {1}.function is not a Function object or class "
                                "or valid method in {2}".
                                format(FUNCTION, self.__class__.__name__, self.name))

    def _validate_value(self):
        pass

    def _instantiate_attributes_before_function(self, function=None, context=None):
        pass

    def _instantiate_function(self, function, function_params=None, context=None):
        """Instantiate function defined in <subclass>.function or <subclass>.paramsCurrent[FUNCTION]

        Instantiate params[FUNCTION] if present, and assign it to self.function

        If params[FUNCTION] is present and valid,
            it is assigned as the function's execute method, overriding any direct implementation of self.function

        If FUNCTION IS in params:
            - if it is a Function object, it is simply assigned to self.function;
            - if it is a Function class reference:
                it is instantiated using self.defaults.variable and, if present, params[FUNCTION_PARAMS]
        If FUNCTION IS NOT in params:
            - if self.function IS implemented, it is assigned to params[FUNCTION]
            - if self.function IS NOT implemented: program error (should have been caught in _validate_function)
        Upon successful completion:
            - self._function === self.paramsCurrent[FUNCTION]
            - self.execute should always return the output of self.function in the first item of its output array;
                 this is done by Function.execute;  any subclass override should do the same, so that...
            - value is value[0] returned by self.execute

        """
        from psyneulink.core.components.functions.userdefinedfunction import UserDefinedFunction
        from psyneulink.core.components.shellclasses import Function

        function_variable = copy.deepcopy(
            self._parse_function_variable(
                self.defaults.variable,
                # we can't just pass context here, because this specifically tries to bypass a
                # branch in TransferMechanism._parse_function_variable
                context=Context(source=ContextFlags.INSTANTIATE)
            )
        )

        # Specification is the function of a (non-instantiated?) Function class
        # KDM 11/12/18: parse an instance of a Function's .function method to itself
        # (not sure how worth it this is, but it existed in Scripts/Examples/Reinforcement-Learning REV)
        # purposely not attempting to parse a class Function.function
        # JDC 3/6/19:  ?what about parameter states for its parameters (see python function problem below)?
        if isinstance(function, types.MethodType):
            try:
                if isinstance(function.__self__, Function):
                    function = function.__self__
            except AttributeError:
                pass

        # Specification is a standard python function, so wrap as a UserDefnedFunction
        # Note:  parameter_states for function's parameters will be created in_instantiate_attributes_after_function
        if isinstance(function, types.FunctionType):
            self.function = UserDefinedFunction(default_variable=function_variable,
                                                custom_function=function,
                                                context=context)
            self.function_params = ReadOnlyOrderedDict(name=FUNCTION_PARAMS)
            for param_name in self.function.cust_fct_params:
                self.function_params.__additem__(param_name, self.function.cust_fct_params[param_name])

        # Specification is an already implemented Function
        elif isinstance(function, Function):
            if not iscompatible(function_variable, function.defaults.variable):
                owner_str = ''
                if hasattr(self, 'owner') and self.owner is not None:
                    owner_str = f' of {repr(self.owner.name)}'
                if function._default_variable_flexibility is DefaultsFlexibility.RIGID:
                    raise ComponentError(f'Variable format ({function.defaults.variable}) of {function.name} '
                                         f'is not compatible with the variable format ({function_variable}) '
                                         f'of {repr(self.name)}{owner_str} to which it is being assigned.')
                                         # f'Make sure variable for {function.name} is 2d.')
                elif function._default_variable_flexibility is DefaultsFlexibility.INCREASE_DIMENSION:
                    function_increased_dim = np.asarray([function.defaults.variable])
                    if not iscompatible(function_variable, function_increased_dim):
                        raise ComponentError(f'Variable format ({function.defaults.variable}) of {function.name} '
                                             f'is not compatible with the variable format ({function_variable})'
                                             f' of {repr(self.name)}{owner_str} to which it is being assigned.')
                                             # f'Make sure variable for {function.name} is 2d.')

            # class default functions should always be copied, otherwise anything this component
            # does with its function will propagate to anything else that wants to use
            # the default
            if function.owner is None and not function.is_pnl_inherent:
                self.function = function
            else:
                self.function = copy.deepcopy(function)

            # setting init status because many mechanisms change execution or validation behavior
            # during initialization
            self.function.initialization_status = ContextFlags.INITIALIZING

            self.function.defaults.variable = function_variable
            self.function._instantiate_value(context)

            self.function.initialization_status = ContextFlags.INITIALIZED

        # Specification is Function class
        # Note:  parameter_states for function's parameters will be created in_instantiate_attributes_after_function
        elif inspect.isclass(function) and issubclass(function, Function):
            kwargs_to_instantiate = function.class_defaults.values().copy()
            if function_params is not None:
                kwargs_to_instantiate.update(**function_params)
                # default_variable should not be in any function_params but sometimes it is
                kwargs_to_remove = ['default_variable']

                for arg in kwargs_to_remove:
                    try:
                        del kwargs_to_instantiate[arg]
                    except KeyError:
                        pass

                # matrix is determined from parameter state based on string value in function_params
                # update it here if needed
                if MATRIX in kwargs_to_instantiate:
                    try:
                        kwargs_to_instantiate[MATRIX] = self.parameter_states[MATRIX].defaults.value
                    except (AttributeError, KeyError, TypeError):
                        pass

            _, kwargs = prune_unused_args(function.__init__, args=[], kwargs=kwargs_to_instantiate)
            self.function = function(default_variable=function_variable, **kwargs)

        else:
            raise ComponentError(f'Unsupported function type: {type(function)}, function={function}.')

        self.function.owner = self

        # KAM added 6/14/18 for functions that do not pass their has_initializers status up to their owner via property
        # FIX: need comprehensive solution for has_initializers; need to determine whether states affect mechanism's
        # has_initializers status
        if self.function.has_initializers:
            self.has_initializers = True

        self._parse_param_state_sources()

    def _instantiate_attributes_after_function(self, context=None):
        if hasattr(self, "_parameter_states"):
            for param_state in self._parameter_states:
                setattr(self.__class__, "mod_"+param_state.name, make_property_mod(param_state.name))
                setattr(self.__class__, "get_mod_" + param_state.name, make_stateful_getter_mod(param_state.name))

    def _instantiate_value(self, context=None):
        #  - call self.execute to get value, since the value of a Component is defined as what is returned by its
        #    execute method, not its function
        try:
            value = self.execute(variable=self.defaults.variable, context=context)
        except TypeError as e:
            # don't hide other TypeErrors
            if "execute() got an unexpected keyword argument 'variable'" != str(e):
                raise

            try:
                value = self.execute(input=self.defaults.variable, context=context)
            except TypeError as e:
                if "execute() got an unexpected keyword argument 'input'" != str(e):
                    raise

                value = self.execute(context=context)
        if value is None:
            raise ComponentError(f"PROGRAM ERROR: Execute method for {self.name} must return a value.")

        self.parameters.value._set(value, context=context, skip_history=True)
        try:
            # Could be mutable, so assign copy
            self.defaults.value = value.copy()
        except AttributeError:
            # Immutable, so just assign value
            self.defaults.value = value

    def initialize(self, context=None):
        raise ComponentError("{} class does not support initialize() method".format(self.__class__.__name__))

    @handle_external_context(execution_id=NotImplemented)
    def reinitialize(self, *args, context=None):
        """
            If the component's execute method involves execution of an `IntegratorFunction` Function, this method
            effectively begins the function's accumulation over again at the specified value, and may update related
            values on the component, depending on the component type.  Otherwise, it simply reassigns the Component's
            value based on its default_variable.
        """
        from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import IntegratorFunction
        if isinstance(self.function, IntegratorFunction):
            if context is NotImplemented:
                context = self.most_recent_context
            new_value = self.function.reinitialize(*args, context=context)
            self.parameters.value.set(np.atleast_2d(new_value), context, override=True)
        else:
            raise ComponentError(f"Reinitializing {self.name} is not allowed because this Component is not stateful. "
                                 "(It does not have an accumulator to reinitialize).")

    @handle_external_context()
    def execute(self, variable=None, context=None, runtime_params=None):
        if context is None:
            try:
                context = self.owner.most_recent_context
            except AttributeError:
                context = self.most_recent_context

        value = self._execute(variable=variable, context=context, runtime_params=runtime_params)
        self.parameters.value._set(value, context=context)
        return value

    def _execute(self, variable=None, context=None, runtime_params=None, **kwargs):
        from psyneulink.core.components.functions.function import Function

        self.parameters.variable._set(variable, context=context)

        if isinstance(self, Function):
            pass # Functions don't have a Logs or maintain execution_counts or time
        else:
            if self.initialization_status & ~(ContextFlags.VALIDATING | ContextFlags.INITIALIZING):
                self._increment_execution_count()
            self._update_current_execution_time(context=context)

        # CALL FUNCTION

        # IMPLEMENTATION NOTE:  **kwargs is included to accommodate required arguments
        #                     that are specific to particular class of Functions
        #                     (e.g., error_matrix for LearningMechanism and controller for EVCControlMechanism)
        function_variable = self._parse_function_variable(variable, context=context)
        value = self.function(variable=function_variable, context=context, params=runtime_params, **kwargs)
        try:
            self.function.parameters.value._set(value, context)
        except AttributeError:
            pass

        self.most_recent_context = context
        return value

    def _parse_param_state_sources(self):
        try:
            for param_state in self._parameter_states:
                if param_state.source is FUNCTION:
                    param_state.source = self.function
        except AttributeError:
            pass

    def _increment_execution_count(self, count=1):
        self.parameters.execution_count.set(self.execution_count+count, override=True)
        return self.execution_count

    @property
    def current_execution_time(self):
        try:
            return self._current_execution_time
        except AttributeError:
            self._update_current_execution_time(self.most_recent_context.string)

    def get_current_execution_time(self, context=None):
        if context is None:
            return self.current_execution_time
        else:
            try:
                return context.composition.scheduler_processing.get_clock(context).time
            except AttributeError:
                return None
    # MODIFIED 9/22/19 END

    def _get_current_execution_time(self, context):
        from psyneulink.core.globals.context import _get_context
        return _get_time(self, context=context)

    def _update_current_execution_time(self, context):
        self._current_execution_time = self._get_current_execution_time(context=context)

    def _change_function(self, to_function):
        pass

    @property
    def name(self):
        try:
            return self._name
        except AttributeError:
            return 'unnamed {0}'.format(self.__class__)

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise ComponentError(f"Name assigned to {self.__class__.__name__} ({value}) must be a string constant.")

        self._name = value

    @property
    def size(self):
        s = []

        try:
            v = np.atleast_2d(self.defaults.variable)
        except AttributeError:
            return None

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
            # MODIFIED 6/1/16 END
                pass
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
    def initialization_status(self):
        try:
            return self._initialization_status
        except AttributeError:
            self._initialization_status = ContextFlags.INITIALIZING
        return self._initialization_status

    @initialization_status.setter
    def initialization_status(self, flag):
        """Check that a flag is one and only one status flag"""
        if flag in INITIALIZATION_STATUS_FLAGS:
            self._initialization_status = flag
        elif not flag:
            self._initialization_status = ContextFlags.UNINITIALIZED
        elif not (flag & ContextFlags.INITIALIZATION_MASK):
            raise ContextError("Attempt to assign a flag ({}) to initialization_status "
                               "that is not an initialization status flag".
                               format(str(flag)))
        else:
            raise ContextError("Attempt to assign more than one flag ({}) to initialization_status".
                               format(str(flag)))

    @property
    def is_initializing(self):
        try:
            owner_initializing = self.owner.initialization_status == ContextFlags.INITIALIZING
        except AttributeError:
            owner_initializing = False

        return self.initialization_status == ContextFlags.INITIALIZING or owner_initializing

    @property
    def log(self):
        try:
            return self._log
        except AttributeError:
            if self.initialization_status == ContextFlags.DEFERRED_INIT:
                raise ComponentError("Initialization of {} is deferred; try assigning {} after it is complete "
                                     "or appropriately configuring a system to which it belongs".
                                     format(self.name, 'log'))
            else:
                raise AttributeError

    @log.setter
    def log(self, log):
        self._log = log

    @property
    def loggable_items(self):
        """Diciontary of items that can be logged in the Component's `log <Component.log>` and their current `ContextFlags`.
        This is a convenience method that calls the `loggable_items <Log.loggable_items>` property of the Component's
        `log <Component.log>`.
        """
        return self.log.loggable_items

    def set_log_conditions(self, items, log_condition=LogCondition.EXECUTION):
        """
        set_log_conditions(          \
            items                    \
            log_condition=EXECUTION  \
        )

        Specifies items to be logged; these must be be `loggable_items <Component.loggable_items>` of the Component's
        `log <Component.log>`. This is a convenience method that calls the `set_log_conditions <Log.set_log_conditions>`
        method of the Component's `log <Component.log>`.
        """
        self.log.set_log_conditions(items=items, log_condition=log_condition)

    def log_values(self, entries):
        """
        log_values(              \
            entries              \
        )

        Specifies items to be logged; ; these must be be `loggable_items <Component.loggable_items>` of the Component's
        `log <Component.log>`. This is a convenience method that calls the `log_values <Log.log_values>` method
        of the Component's `log <Component.log>`.
        """
        self.log.log_values(entries)

    def _dict_summary(self):
        basic_attributes = ['name']

        # lifted from LLVM
        parameter_black_list = {'function', 'variable', 'value', 'context'}
        parameters_dict = {}

        for p in self.parameters:
            if p.user and p.name not in parameter_black_list:
                val = p._get(self.most_recent_context)

                if isinstance(val, np.ndarray):
                    val = f'numpy.array({val})'
                elif not isinstance(val, numbers.Number) and val is not None:
                    val = str(val)

                parameters_dict[p.name] = val

        function_dict = {'function': self.function._dict_summary()} if (hasattr(self, 'function') and isinstance(self.function, Component)) else {}

        return {
            **{attr: getattr(self, attr) for attr in basic_attributes},
            **{'parameters': parameters_dict},
            **function_dict,
            **{'type': self.__class__.__name__}
        }

    def json_summary(self):
        import json
        return json.dumps(self._dict_summary(), sort_keys=True, indent=4, separators=(',', ': '))

    @property
    def logged_items(self):
        """Dictionary of all items that have entries in the log, and their currently assigned `ContextFlags`\\s
        This is a convenience method that calls the `logged_items <Log.logged_items>` property of the Component's
        `log <Component.log>`.
        """
        return self.log.logged_items

    @property
    def _loggable_parameters(self):
        return [param.name for param in self.parameters if param.loggable and param.user]

    @property
    def _default_variable_flexibility(self):
        try:
            return self.__default_variable_flexibility
        except AttributeError:
            self.__default_variable_flexibility = DefaultsFlexibility.FLEXIBLE
            return self.__default_variable_flexibility

    @_default_variable_flexibility.setter
    def _default_variable_flexibility(self, value):
        self.__default_variable_flexibility = value

    @property
    def _default_variable_handled(self):
        try:
            return self.__default_variable_handled
        except AttributeError:
            self.__default_variable_handled = False
            return self.__default_variable_handled

    @_default_variable_handled.setter
    def _default_variable_handled(self, value):
        self.__default_variable_handled = value

    @classmethod
    def get_constructor_defaults(cls):
        return {arg_name: arg.default for (arg_name, arg) in inspect.signature(cls.__init__).parameters.items()}

    @classmethod
    def get_param_class_defaults(cls):
        try:
            return cls._param_class_defaults
        except AttributeError:
            excluded_keys = ['self', 'args', 'kwargs']

            cls._param_class_defaults = {}
            for klass in reversed(cls.__mro__):
                try:
                    cls._param_class_defaults.update({k: v for (k, v) in klass.get_constructor_defaults().items() if k not in excluded_keys})
                except AttributeError:
                    # skip before Component
                    pass

            return cls._param_class_defaults

    @property
    def function(self):
        # TODO: make sure all functions are stateless
        return self.parameters.function._get(Context())

    @function.setter
    def function(self, value):
        # TODO: currently no validation, should replicate from _instantiate_function
        self.parameters.function._set(value, Context())
        self._parse_param_state_sources()

    @property
    def function_params(self):
        return self.user_params[FUNCTION_PARAMS]

    @function_params.setter
    def function_params(self, val):
        self.user_params.__additem__(FUNCTION_PARAMS, val)

    @property
    def class_parameters(self):
        return self.__class__.parameters

    @property
    def stateful_parameters(self):
        """
            A list of all of this object's `parameters <Parameters>` whose values
            may change during runtime
        """
        return [param for param in self.parameters if param.stateful]

    @property
    def function_parameters(self):
        """
            The `parameters <Parameters>` object of this object's `function`
        """
        try:
            return self.function.parameters
        except AttributeError:
            return None

    @property
    def class_defaults(self):
        """
            Refers to the defaults of this object's class
        """
        return self.__class__.defaults

    @property
    def is_pnl_inherent(self):
        try:
            return self._is_pnl_inherent
        except AttributeError:
            self._is_pnl_inherent = False
            return self._is_pnl_inherent

    @property
    def _parameter_components(self):
        """
            Returns a set of Components that are values of this object's
            Parameters
        """
        try:
            return self.__parameter_components
        except AttributeError:
            self.__parameter_components = set()
            return self.__parameter_components

    def _update_parameter_components(self, context):
        # store all Components in Parameters to be used in
        # _dependent_components for _initialize_from_context
        for p in self.parameters:
            try:
                param_value = p.get(context)
                if isinstance(param_value, Component):
                    self._parameter_components.add(param_value)
            # ControlMechanism and GatingMechanism have Parameters that only
            # throw these errors
            except Exception as e:
                # cannot import the specific exceptions due to circularity
                if 'attribute is not implemented on' not in str(e):
                    raise

    @property
    def _dependent_components(self):
        """
            Returns a set of Components that will be executed if this Component is executed
        """
        return list(self._parameter_components)

    @property
    def most_recent_context(self):
        """
            used to set a default behavior for attributes that correspond to parameters
        """
        try:
            return self._most_recent_context
        except AttributeError:
            self._most_recent_context = Context(source=ContextFlags.COMMAND_LINE)
            return self._most_recent_context

    @most_recent_context.setter
    def most_recent_context(self, value):
        self._most_recent_context = value


COMPONENT_BASE_CLASS = Component


def make_property(name):
    backing_field = '_' + name

    def getter(self):
        return getattr(self, backing_field)

    def setter(self, val):
        if (
            hasattr(self, '_prefs')
            and self.paramValidationPref
            and hasattr(self, PARAMS_CURRENT)
            and hasattr(self, 'user_params_for_instantiation')
            and hasattr(self, 'user_params')
            and hasattr(self, 'paramInstanceDefaults')
        ):
            self._assign_params(request_set={name:val}, context=Context(source=ContextFlags.PROPERTY))
        else:
            setattr(self, backing_field, val)

        # Update user_params dict with new value
        # KAM COMMENTED OUT 3/2/18 -- we do not want to update user_params with the base value, only param state value
        # self.user_params.__additem__(name, val)

        # If Component is a Function and has an owner, update function_params dict for owner
        #    also, get parameter_state_owner if one exists
        from psyneulink.core.components.functions.function import Function_Base
        if isinstance(self, Function_Base) and hasattr(self, 'owner') and self.owner is not None:
            param_state_owner = self.owner
            # NOTE CW 1/26/18: if you're getting an error (such as "self.owner has no attribute function_params", or
            # "function_params" has no attribute __additem__ (this happens when it's a dict rather than a
            # ReadOnlyOrderedDict)) it may be caused by function_params not being included in paramInstanceDefaults,
            # which may be caused by _assign_args_to_param_dicts() bugs. LMK, if you're getting bugs here like that.
            # KAM COMMENTED OUT 3/2/18 --
            # we do not want to update function_params with the base value, only param state value
            # self.owner.function_params.__additem__(name, val)
        else:
            param_state_owner = self

        # If the parameter is associated with a ParameterState, assign the value to the ParameterState's variable
        # if hasattr(param_state_owner, '_parameter_states') and name in param_state_owner._parameter_states:
        #     param_state = param_state_owner._parameter_states[name]
        #
        #     # MODIFIED 7/24/17 CW: If the ParameterState's function has an initializer attribute (i.e. it's an
        #     # integrator function), then also reset the 'previous_value' and 'initializer' attributes by setting
        #     # 'reinitialize'
        #     if hasattr(param_state.function, 'initializer'):
        #         param_state.function.reinitialize = val

    # Create the property
    prop = property(getter).setter(setter)
    # # Install some documentation
    # prop.__doc__ = docs[name]
    return prop


def make_property_mod(param_name):

    def getter(self):
        try:
            return self._parameter_states[param_name].value
        except TypeError:
            raise ComponentError("{} does not have a '{}' ParameterState."
                                 .format(self.name, param_name))

    def setter(self, value):
        raise ComponentError("Cannot set to {}'s mod_{} directly because it is computed by the ParameterState."
                             .format(self.name, param_name))

    prop = property(getter).setter(setter)

    return prop


def make_stateful_getter_mod(param_name):

    def getter(self, context=None):
        try:
            return self._parameter_states[param_name].parameters.value.get(context)
        except TypeError:
            raise ComponentError("{} does not have a '{}' ParameterState."
                                 .format(self.name, param_name))

    return getter
