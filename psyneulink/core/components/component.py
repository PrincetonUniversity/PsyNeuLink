# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************** Component  ************************************************************


"""
Contents
--------

* `Component_Overview`
* `Component_Creation`
    * `Component_Deferred_Init`
* `Component_Structure`
    * `Component_Structural_Attributes`
        * `Variable <Component_Variable>`
        * `Function <Component_Function>`
        * `Value <Component_Value>`
        * `Log <Component_Log>`
        * `Name <Component_Name>`
        * `Preferences <Component_Prefs>`
    * `User_Modifiable_Parameters`
    COMMENT:
    * `Methods <Component_Methods>`
    COMMENT
* `Component_Execution`
    * `Component_Execution_Initialization`
    * `Component_Execution_Termination`
    * `Component_Execution_Count_and_Time`
* `Component_Class_Reference`


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
`Port <Port_Base.owner>`, or the **sender** or **receiver** for a `Projection <Projection_Structure>`), then its
full initialization is deferred until its the information is available (e.g., the `Port <Port>` is assigned to a
`Mechanism <Mechanism>`, or a `Projection <Projection>` is assigned its `sender <Projection_Base.sender>` and `receiver
<Projection_Base.receiver>`).  This allows Components to be created before all of the information they require is
available (e.g., at the beginning of a script). However, for the Component to be operational, its initialization must
be completed by a call to it `deferred_init` method.  This is usually done automatically when the Component is
assigned to another Component to which it belongs (e.g., assigning a Port to a Mechanism) or to a Composition (e.g.,
a Projection to the `pathway <Process.pahtway>`) of a `Process`), as appropriate.

.. _Component_Structure:

Component Structure
-------------------

.. _Component_Structural_Attributes:

*Core Structural Attributes*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every Component has the following set of core structural attributes. These attributes are not meant to be changed by the
user once the component is constructed, with the one exception of `prefs <Component_Prefs>`.

.. _Component_Type:

* **componentType** - species the type of Component.

.. _Component_Variable:

* **variable** - used as the input to its `function <Component_Function>`.  Specification of the **default_variable**
  argument in the constructor for a Component determines both its format (e.g., whether its value is numeric, its
  dimensionality and shape if it is an array, etc.) as well as its `default value <Component.defaults>` (the value
  used when the Componentis executed and no input is provided), and takes precedence over the specification of `size
  <Component_Size>`.

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

* **function** - determines the computation that a Component carries out. It is always a PsyNeuLink `Function
  <Function>` object (itself also a PsyNeuLink Component).

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
  `parameter specification dictionary <ParameterPort_Specification>` assigned to the
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
  If it was not specified, a default is assigned by the `registry <Registry>` for subclass (see `Registry_Naming` for
  conventions used in assigning default names and handling of duplicate names).
..

.. _Component_Prefs:

* **prefs** - the `prefs <Components.prefs>` attribute contains the `PreferenceSet` assigned to the Component when
  it was created.  If it was not specified, a default is assigned using `classPreferences` defined in ``__init__.py``
  Each individual preference is accessible as an attribute of the Component, the name of which is the name of the
  preference (see `Preferences` for details).


.. _User_Modifiable_Parameters:

*Parameters*
~~~~~~~~~~~~

.. _Component_Parameters:

A Component defines its `parameters <Parameters>` in its *parameters* attribute, which contains a collection of
`Parameter` objects, each of which stores a Parameter's values, `default values <Component.defaults>`, and various
`properties <Parameter_Attributes_Table>` of the parameter.

* `Parameters <Component.Parameters>` - a `Parameters class <Parameters>` defining parameters and their default values
   that are used for all Components, unless overridden.

   All of the parameters listed in the *parameters* class can be modified by the user (as described above).  Some
   can also be modified by `ControlSignals <ControlSignal>` when a `Composition executes <Composition_Execution>`.
   In general, only parameters that take numerical values and/or do not affect the structure, mode of operation,
   or format of the values associated with a Component can be subject to modulation.  For example, for a
   `TransferMechanism`, `clip <TransferMechanism.clip>`, `initial_value <TransferMechanism.initial_value>`,
   `integrator_mode <TransferMechanism.integrator_mode>`, `input_ports <Mechanism_Base.input_ports>`,
   `output_ports`, and `function <Mechanism_Base.function>`, are all listed in parameters, and are user-modifiable,
   but are not subject to modulation; whereas `noise <TransferMechanism.noise>` and `integration_rate
   <TransferMechanism.integration_rate>` can all be subject to modulation. Parameters that are subject to modulation
   have the `modulable <Parameter.modulable>` attribute set to True and are associated with a `ParameterPort` to which
   the ControlSignals can project (by way of a `ControlProjection`).

  COMMENT:
      FIX: ADD DISCUSSION ABOUT HOW TO ASSIGN DEFAULTS HERE 5/8/20
  COMMENT

.. _Component_Function_Params:

* **initial_function_parameters** - the `initial_function_parameters <Component.function>` attribute contains a
  dictionary of the parameters for the Component's `function <Component.function>` and their values, to be used to
  instantiate the function.  Each entry is the name of a parameter, and its value is the value of that parameter.
  The parameters for a function can be specified when the Component is created in one of the following ways:

      * in an argument of the **Component's constructor** -- if all of the allowable functions for a Component's
        `function <Component.function>` share some or all of their parameters in common, the shared paramters may appear
        as arguments in the constructor of the Component itself, which can be used to set their values.

      * in an entry of a `parameter specification dictionary <ParameterPort_Specification>` assigned to the
        **params** argument of the constructor for the Component.  The entry must use the keyword
        FUNCTION_PARAMS as its key, and its value must be a dictionary containing the parameters and their values.
        The key for each entry in the FUNCTION_PARAMS dictionary must be the name of a parameter, and its value the
        parameter's value, as in the example below::

            my_component = SomeComponent(function=SomeFunction
                                         params={FUNCTION_PARAMS:{SOME_PARAM=1, SOME_OTHER_PARAM=2}})

  The parameters of functions for some Components may allow other forms of specification (see
  `ParameterPort_Specification` for details concerning different ways in which the value of a
  parameter can be specified).

COMMENT:
    FIX: STATEMENT ABOVE ABOUT MODIFYING EXECUTION COUNT VIOLATES THIS DEFINITION, AS PROBABLY DO OTHER ATTRIBUTES
      * parameters are things that govern the operation of the Mechanism (including its function) and/or can be
        modified/modulated
      * attributes include parameters, but also read-only attributes that reflect but do not determine the operation
        (e.g., EXECUTION_COUNT)
COMMENT

.. _Component_Stateful_Parameters:

* **stateful_parameters** - a list containing all of the Component's `stateful parameters <Parameter_Statefulness>`.
  COMMENT:
     DESCRIPTION HERE
  COMMENT


COMMENT:
.. _Component_Methods:

*Component Methods*
~~~~~~~~~~~~~~~~~~~

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
        `parameter specification dictionary <ParameterPort_Specification>`.  If it is overridden by a subclass,
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

* **reset_params** - reset the value of all parameters to a set of default values as specified in its **mode**
  argument, using a value of `ResetMode <Component_ResetMode>`.

.. _Component_Execution:

Execution
---------

A Component is executed when its `execute` method is called, which in turn calls its `function <Component_Function>`.

.. _Component_Lazy_Updating:

*Lazy Updating*
~~~~~~~~~~~~~~~

In general, only `Compositions <Composition>` are executed from the command line (i.e., from the console or in a
script).  `Mechanisms <Mechanism>` can also be executed, although this is usually just for the purposes of demonstration
or debugging, and `Functions <Function>` can only be executed if they are standalone (that is, they do not belong to
another Component).  All other Components are executed only a Component that depends on them to do so.  This can be
one to which a Components belongs (such as the Mechanism to which a `Port` belongs) or that otherwise requires it to
execute (for example, a updating a `Port` requires its `afferent Projections <Port_Projections>` to `execute
<Port_Execution>`).  This is referred to as "lazy updating", since it means that most Components don't execute unless
and until they are required to do so.  While this reduces unecessary computation, it can sometimes be confusing. For
example, when `learning <Composition_Learning>` occurs in a Composition, the modification to the `matrix
<MappingProjection.matrix>` parameter of a `MappingProjection` that occurs on a given `TRIAL <TimeScale.TRIAL>`
does not acutally appear in its `value <ParameterPort>` until the next `TRIAL <TimeScale.TRIAL>`, since it requires
that the ParameterPort for the `matrix <MappingProjection.matrix>` be executed, which does not occur until the next
time the MappingProjection is executed (i.e., in the next `TRIAL <TimeScale.TRIAL>`).  Therefore, in tracking the
`value <Component.value>` of Components during execution, it is important to carefully consider the state of
execution of the Components to which they belong or on which they depend for execution.

The following attributes and methods control and provide information about the execution of a Component:

.. _Component_Execution_Initialization:

*Initialization*
~~~~~~~~~~~~~~~~

.. _Component_Reset_Stateful_Function_When:

* **reset_stateful_function_when** -- a `Condition` that determines when the Component's `reset <Component.reset>`
  method is called.  The `reset <Component.reset>` method and `reset_stateful_function_when
  <Component.reset_stateful_function_when>` attribute only exist for Mechanisms that have `stateful
  <Parameter.stateful>` `Parameters`, or that have a `function <Mechanism_Base.function>` with `stateful
  <Parameter.stateful>` Parameters.  When the `reset <Component.reset>` method is called, this is done without any
  arguments, so that the relevant `initializer <IntegratorFunction.initializer>` attributes (or their equivalents
  -- initialization attributes vary among functions) are used for reinitialization.
  COMMENT:
      WHAT ABOUT initializer ATTRIBUTE FOR NON-INTEGRATOR FUNCTIONS, AND FOR STATEFUL PARAMETERS ON MECHANISMS?
      WHY IS THIS ATTRIBUTE ON COMPONENT RATHER THAN MECHANISM?
  COMMENT

  .. note::

     `Mechanisms` <Mechanism>` are the only type of Component that reset when the `reset_stateful_function_when
     <Component.reset_stateful_function_when>` `Condition` is satisfied. Other Component types do not reset,
     although `Composition` has a `reset <Composition.reset>` method that can be used to reset all of its eligible
     Mechanisms (see `Composition_Reset`)

.. _Component_Execution_Termination:

*Termination*
~~~~~~~~~~~~~

.. _Component_Is_Finished:

* **is_finished()** -- method that determines whether execution of the Component is complete for a `TRIAL
  <TimeScale.TRIAL>`;  it is only used if `execute_until_finished <Component_Execute_Until_Finished>` is True.

.. _Component_Execute_Until_Finished:

* **execute_until_finished** -- determines whether the Component executes until its `is_finished` method returns True.
  If it is False, then the Component executes only once per call to its `execute <Component.execute>` method,
  irrespective of its `is_finished` method;  if it is True then, depending on how its class implements and handles its
  `is_finished` method, the Component may execute more than once per call to its `execute <Component.execute>` method.

.. _Component_Num_Executions_Before_Finished:

* **num_executions_before_finished** -- contains the number of times the Component has executed prior to finishing
  (and since it last finished);  depending upon the class, these may all be within a single call to the Component's
  `execute <Component.execute>` method, or extend over several calls.  It is set to 0 each time `is_finished` evalutes
  to True. Note that this is distinct from the `execution_count <Component_Execution_Count>` and `num_executions
  <Component_Num_Executions>` attributes.

.. _Component_Max_Executions_Before_Finished:

* **max_executions_before_finished** -- determines the maximum number of executions allowed before finishing
  (i.e., the maxmium allowable value of `num_executions_before_finished <Component.num_executions_before_finished>`).
  If it is exceeded, a warning message is generated.  Note that this only pertains to `num_executions_before_finished
  <Component_Num_Executions_Before_Finished>`, and not its `execution_count <Component_Execution_Count>`, which can be
  unlimited.

.. _Component_Execution_Count_and_Time:

*Count and Time*
~~~~~~~~~~~~~~~~

.. _Component_Execution_Count:

* **execution_count** -- maintains a record of the number of times a Component has executed since it was constructed,
  *excluding*  executions carried out during initialization and validation, but including all others whether they are
  of the Component on its own are as part of a `Composition`, and irresective of the `context <Context>` in which
  they are occur. The value can be changed "manually" or programmatically by assigning an integer
  value directly to the attribute.  Note that this is the distinct from the `num_executions <Component_Num_Executions>`
  and `num_executions_before_finished <Component_Num_Executions_Before_Finished>` attributes.

.. _Component_Num_Executions:

* **num_executions** -- maintains a record, in a `Time` object, of the number of times a Component has executed in a
  particular `context <Context>` and at different `TimeScales <TimeScale>`. The value cannot be changed. Note that this
  is the distinct from the `execution_count <Component_Execution_Count>` and `num_executions_before_finished
  <Component_Num_Executions_Before_Finished>` attributes.

.. _Component_Current_Execution_Time:

* **current_execution_time** -- maintains the `Time` of the last execution of the Component in the context of the
  `Composition`'s current `scheduler <Composition.scheduler`, and is stored as a `time
  <Context.time>` tuple of values indicating the `TimeScale.TRIAL`,  `TimeScale.PASS`, and `TimeScale.TIME_STEP` of the
  last execution.


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
import base64
import copy
import dill
import functools
import inspect
import logging
import numbers
import types
import warnings

from abc import ABCMeta
from collections.abc import Iterable
from enum import Enum, IntEnum

import numpy as np

from psyneulink.core import llvm as pnlvm
from psyneulink.core.globals.context import \
    Context, ContextError, ContextFlags, INITIALIZATION_STATUS_FLAGS, _get_time, handle_external_context
from psyneulink.core.globals.json import JSONDumpable
from psyneulink.core.globals.keywords import \
    CONTEXT, CONTROL_PROJECTION, DEFERRED_INITIALIZATION, EXECUTE_UNTIL_FINISHED, \
    FUNCTION, FUNCTION_PARAMS, INIT_FULL_EXECUTE_METHOD, INPUT_PORTS, \
    LEARNING, LEARNING_PROJECTION, MATRIX, MAX_EXECUTIONS_BEFORE_FINISHED, \
    MODEL_SPEC_ID_PSYNEULINK, MODEL_SPEC_ID_GENERIC, MODEL_SPEC_ID_TYPE, MODEL_SPEC_ID_PARAMETER_SOURCE, \
    MODEL_SPEC_ID_PARAMETER_VALUE, MODEL_SPEC_ID_INPUT_PORTS, MODEL_SPEC_ID_OUTPUT_PORTS, \
    MODULATORY_SPEC_KEYWORDS, NAME, OUTPUT_PORTS, OWNER, PARAMS, PREFS_ARG, \
    RESET_STATEFUL_FUNCTION_WHEN, VALUE, VARIABLE
from psyneulink.core.globals.log import LogCondition
from psyneulink.core.scheduling.time import Time, TimeScale
from psyneulink.core.globals.sampleiterator import SampleIterator
from psyneulink.core.globals.parameters import \
    Defaults, Parameter, ParameterAlias, ParameterError, ParametersBase, copy_parameter_value
from psyneulink.core.globals.preferences.basepreferenceset import BasePreferenceSet, VERBOSE_PREF
from psyneulink.core.globals.preferences.preferenceset import \
    PreferenceEntry, PreferenceLevel, PreferenceSet, _assign_prefs
from psyneulink.core.globals.registry import register_category
from psyneulink.core.globals.utilities import \
    ContentAddressableList, convert_all_elements_to_np_array, convert_to_np_array, get_deepcopy_with_shared,\
    is_instance_or_subclass, is_matrix, iscompatible, kwCompatibilityLength, prune_unused_args, \
    get_all_explicit_arguments, call_with_pruned_args
from psyneulink.core.scheduling.condition import Never

__all__ = [
    'Component', 'COMPONENT_BASE_CLASS', 'component_keywords', 'ComponentError', 'ComponentLog',
    'DefaultsFlexibility', 'DeferredInitRegistry', 'parameter_keywords', 'ResetMode',
]

logger = logging.getLogger(__name__)

component_keywords = {NAME, VARIABLE, VALUE, FUNCTION, FUNCTION_PARAMS, PARAMS, PREFS_ARG, CONTEXT}

DeferredInitRegistry = {}


class ResetMode(Enum):
    """

    .. _Component_ResetMode:

    ResetModes used for **reset_params**:

    .. _CURRENT_TO_INSTANCE_DEFAULTS:

    *CURRENT_TO_INSTANCE_DEFAULTS*
      • resets all current values to instance default values.

    .. _INSTANCE_TO_CLASS:

    *INSTANCE_TO_CLASS*
      • resets all instance default values to class default values.

    .. _ALL_TO_CLASS_DEFAULTS:

    *ALL_TO_CLASS_DEFAULTS*
      • resets all current values and instance default values to \
      class default values

    """
    CURRENT_TO_INSTANCE_DEFAULTS = 0
    INSTANCE_TO_CLASS = 1
    ALL_TO_CLASS_DEFAULTS = 2


class DefaultsFlexibility(Enum):
    """
        Denotes how rigid an assignment to a default is. That is, how much it can be modified, if at all,
        to suit the purpose of a method/owner/etc.

        e.g. when assigning a Function to a Mechanism:

            ``pnl.TransferMechanism(default_variable=[0, 0], function=pnl.Linear())``

            the Linear function is assigned a default variable ([0]) based on it's ClassDefault,
            which conflicts with the default variable specified by its future owner ([0, 0]). Since
            the default for Linear was not explicitly stated, we allow the TransferMechanism to
            reassign the Linear's default variable as needed (`FLEXIBLE`)

    Attributes
    ----------

    FLEXIBLE
        can be modified in any way.

    RIGID
        cannot be modifed in any way.

    INCREASE_DIMENSION
        can be wrapped in a single extra dimension.

    """
    FLEXIBLE = 0
    RIGID = 1
    INCREASE_DIMENSION = 2


parameter_keywords = set()

# suppress_validation_preference_set = BasePreferenceSet(prefs = {
#     PARAM_VALIDATION_PREF: PreferenceEntry(False,PreferenceLevel.INSTANCE),
#     VERBOSE_PREF: PreferenceEntry(False,PreferenceLevel.INSTANCE),
#     REPORT_OUTPUT_PREF: PreferenceEntry(True,PreferenceLevel.INSTANCE)})


class ComponentLog(IntEnum):
    NONE            = 0
    ALL = 0
    DEFAULTS = NONE


class ComponentError(Exception):
    def __init__(self, message, component=None):
        try:
            component_str = component.name
            try:
                if component.owner is not None:
                    component_str = f'{component_str} (owned by {component.owner.name})'
            except AttributeError:
                pass
        except AttributeError:
            component_str = None

        if component_str is not None:
            message = f'{component_str}: {message}'

        super().__init__(message)


def make_parameter_property(name):
    def getter(self):
        return getattr(self.parameters, name)._get(self.most_recent_context)

    def setter(self, value):
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

            try:
                if param.default_value.owner is None:
                    param.default_value.owner = param
            except AttributeError:
                pass

    # consider removing this for explicitness
    # but can be useful for simplicity
    @property
    def class_defaults(self):
        return self.defaults


class Component(JSONDumpable, metaclass=ComponentsMeta):
    """
    Component(                 \
        default_variable=None, \
        size=None,             \
        params=None,           \
        name=None,             \
        prefs=None,            \
        context=None           \
    )

    Base class for Component.

    The arguments below are ones that can be used in the constructor for any Component subclass.

    .. note::
       Component is an abstract class and should *never* be instantiated by a direct call to its constructor.
       It should be instantiated using the constructor for a subclass.

    COMMENT:
    FOR API DOCUMENTATION:
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
        The instance defaults can be assigned at initialization or using the _instantiate_defaults class method;
            - if instance defaults are not assigned on initialization, the corresponding class defaults are assigned
        Each Component child class must initialize itself by calling super(childComponentName).__init__()
            with a default value for its variable, and optionally an instance default paramList.

        A subclass MUST either:
            - implement a <class>.function method OR
            - specify a default Function
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
                            to the method referenced by self.defaults.function
                    if self.function is found, an exception is raised

        NOTES:
            * In the current implementation, validation is:
              - top-level only (items in lists, tuples and dictionaries are not checked, nor are nested items)
              - for type only (it is oblivious to content)
              - forgiving (e.g., no distinction is made among numberical types)
            * However, more restrictive validation (e.g., recurisve, range checking, etc.) can be achieved
                by overriding the class _validate_variable and _validate_params methods

    COMMENT

    Arguments
    ---------

    default_variable : scalar, list or array : default [[0]]
        specifies template for the input to the Component's `function <Component.function>`.

    size : int, list or np.ndarray of ints : default None
        specifies default_variable as array(s) of zeros if **default_variable** is not passed as an argument;
        if **default_variable** is specified, it takes precedence over the specification of **size**.

    COMMENT:
    param_defaults :   :  default None,
    COMMENT

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterPort_Specification>` that can be used to specify the parameters for
        the Component and/or a custom function and its parameters. Values specified for parameters in the dictionary
        override any assigned to those parameters in arguments of the constructor.

    name : str : for default see `name <Component_Name>`
        a string used for the name of the Component;  default is assigned by relevant `Registry` for Component
        (see `Registry_Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : default Component.classPreferences
        specifies the `PreferenceSet` for the Component (see `prefs <Component_Base.prefs>` for details).

    context : Context : default None
        specifies `context <Context>` in which Component is being initialized or executed.


    Attributes
    ----------

    variable : 2d np.array
        see `variable <Component_Variable>`

    size : int or array of ints
        see `size <Component_Size>`

    function : Function, function or method
        see `function <Component_Function>`

    value : 2d np.array
        see `value <Component_Value>`

    log : Log
        see `log <Component_Log>`

    execution_count : int
        see `execution_count <Component_Execution_Count>`

    num_executions : Time
        see `num_executions <_Component_Num_Executions>`

    current_execution_time : tuple(`Time.RUN`, `Time.TRIAL`, `Time.PASS`, `Time.TIME_STEP`)
        see `current_execution_time <Component_Current_Execution_Time>`

    execute_until_finished : bool
        see `execute_until_finished <Component_Execute_Until_Finished>`

    num_executions_before_finished : int
        see `num_executions_before_finished <Component_Num_Executions_Before_Finished>`

    max_executions_before_finished : bool
        see `max_executions_before_finished <Component_Max_Executions_Before_Finished>`

    stateful_parameters : list
        see `stateful_parameters <Component_Stateful_Parameters>`

    reset_stateful_function_when : `Condition`
        see `reset_stateful_function_when <Component_reset_stateful_function_when>`

    name : str
        see `name <Component_Name>`

    prefs : PreferenceSet
        see `prefs <Component_Prefs>`

    parameters :  Parameters
        see `parameters <Component_Parameters>` and `Parameters` for additional information.

    defaults : Defaults
        an object that provides access to the default values of a `Component's` `parameters`;
        see `parameter defaults <Parameter_Defaults>` for additional information.

    initialization_status : field of flags attribute
        indicates the state of initialization of the Component;
        one and only one of the following flags is always set:

            * `DEFERRED_INIT <ContextFlags.DEFERRED_INIT>`
            * `INITIALIZING <ContextFlags.INITIALIZING>`
            * `VALIDATING <ContextFlags.VALIDATING>`
            * `INITIALIZED <ContextFlags.INITIALIZED>`
            * `RESET <ContextFlags.RESET>`
            * `UNINITIALIZED <ContextFlags.UNINITALIZED>`

    COMMENT:
    FIX: THESE USED TO BE IN CONSTRUCTORS FOR ALL SUBCLASSES.  INTEGRATE WITH ABOVE
    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterPort_Specification>` that can be used to specify the parameters for
        the InputPort or its function, and/or a custom function and its parameters. Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default see `name <InputPort.name>`
        specifies the name of the InputPort; see InputPort `name <InputPort.name>` for details.

    prefs : PreferenceSet or specification dict : default Port.classPreferences
        specifies the `PreferenceSet` for the InputPort; see `prefs <InputPort.prefs>` for details.
    COMMENT

    """

    #CLASS ATTRIBUTES
    className = "COMPONENT"
    suffix = " " + className
# IMPLEMENTATION NOTE:  *** CHECK THAT THIS DOES NOT CAUSE ANY CHANGES AT SUBORDNIATE LEVELS TO PROPOGATE EVERYWHERE
    componentCategory = None
    componentType = None

    standard_constructor_args = [RESET_STATEFUL_FUNCTION_WHEN, EXECUTE_UNTIL_FINISHED, MAX_EXECUTIONS_BEFORE_FINISHED]

    # helper attributes for JSON model spec
    _model_spec_id_parameters = 'parameters'

    _model_spec_generic_type_name = NotImplemented
    """
        string describing this class's generic type in universal model
        specification,
        if it exists and is different than the class name
    """

    _model_spec_class_name_is_generic = False
    """
        True if the class name is the class's generic type in universal model specification,
        False otherwise
    """

    _specified_variable_shape_flexibility = DefaultsFlexibility.RIGID
    """
        The `DefaultsFlexibility` ._variable_shape_flexibility takes on
        when variable shape was manually specified
    """

    class Parameters(ParametersBase):
        """
            The `Parameters` that are associated with all `Components`

            Attributes
            ----------

                variable
                    see `variable <Component_Variable>`

                    :default value: numpy.array([0])
                    :type: ``numpy.ndarray``
                    :read only: True

                value
                    see `value <Component_Value>`

                    :default value: numpy.array([0])
                    :type: ``numpy.ndarray``
                    :read only: True

                execute_until_finished
                    see `execute_until_finished <Component_Execute_Until_Finished>`

                    :default value: True
                    :type: ``bool``

                execution_count
                    see `execution_count <Component_Execution_Count>`

                    :default value: 0
                    :type: ``int``
                    :read only: True

                num_executions
                    see `num_executions <_Component_Num_Executions>`

                    :default value:
                    :type: ``Time``
                    :read only: True

                has_initializers
                    see `has_initializers <Component.has_initializers>`

                    :default value: False
                    :type: ``bool``

                is_finished_flag
                    internal parameter used by some Component types to track previous status of is_finished() method,
                    or to set the status reported by the is_finished (see `is_finished <Component_Is_Finished>`

                    :default value: True
                    :type: ``bool``

                max_executions_before_finished
                    see `max_executions_before_finished <Component_Max_Executions_Before_Finished>`

                    :default value: 1000
                    :type: ``int``

                num_executions_before_finished
                    see `num_executions_before_finished <Component_Num_Executions_Before_Finished>`

                    :default value: 0
                    :type: ``int``
                    :read only: True
        """
        variable = Parameter(np.array([0]), read_only=True, pnl_internal=True, constructor_argument='default_variable')
        value = Parameter(np.array([0]), read_only=True, pnl_internal=True)
        has_initializers = Parameter(False, setter=_has_initializers_setter, pnl_internal=True)
        # execution_count is not stateful because it is a global counter;
        #    for context-specific counts should use schedulers which store this info
        execution_count = Parameter(0,
                                    read_only=True,
                                    loggable=False,
                                    stateful=False,
                                    fallback_default=True,
                                    pnl_internal=True)
        is_finished_flag = Parameter(True, loggable=False, stateful=True)
        execute_until_finished = True
        num_executions = Parameter(Time(), read_only=True, modulable=False, loggable=False)
        num_executions_before_finished = Parameter(0, read_only=True, modulable=False)
        max_executions_before_finished = Parameter(1000, modulable=False)

        def _parse_variable(self, variable):
            if variable is None:
                return variable

            try:
                return convert_to_np_array(variable)
            except ValueError:
                return convert_all_elements_to_np_array(variable)

        def _validate_variable(self, variable):
            return None

        def _parse_modulable(self, param_name, param_value):
            from psyneulink.core.components.functions.distributionfunctions import DistributionFunction
            # assume 2-tuple with class/instance as second item is a proper
            # modulatory spec, can possibly add in a flag on acceptable
            # classes in the future
            if (
                isinstance(param_value, tuple)
                and len(param_value) == 2
                and (
                    is_instance_or_subclass(param_value[1], Component)
                    or (
                        isinstance(param_value[1], str)
                        and param_value[1] in MODULATORY_SPEC_KEYWORDS
                    )
                )
            ):
                value = param_value[0]
            # assume a DistributionFunction is allowed to persist, for noise
            elif (
                (
                    is_instance_or_subclass(param_value, Component)
                    and not is_instance_or_subclass(
                        param_value,
                        DistributionFunction
                    )
                )
                or (
                    isinstance(param_value, str)
                    and param_value in MODULATORY_SPEC_KEYWORDS
                )
            ):
                value = getattr(self, param_name).default_value
            else:
                value = param_value

            if isinstance(value, list):
                value = np.asarray(value)

            return value

    initMethod = INIT_FULL_EXECUTE_METHOD

    classPreferenceLevel = PreferenceLevel.COMPOSITION
    # Any preferences specified below will override those specified in COMPOSITION_DEFAULT_PREFERENCES
    # Note: only need to specify setting;  level will be assigned to COMPOSITION automatically
    # classPreferences = {
    #     PREFERENCE_SET_NAME: 'ComponentCustomClassPreferences',
    #     PREFERENCE_KEYWORD<pref>: <setting>...}

    exclude_from_parameter_ports = [INPUT_PORTS, OUTPUT_PORTS]

    # IMPLEMENTATION NOTE: This is needed so that the Port class can be used with ContentAddressableList,
    #                      which requires that the attribute used for addressing is on the class;
    #                      it is also declared as a property, so that any assignments are validated to be strings,
    #                      insuring that assignment by one instance will not affect the value of others.
    name = None

    _deepcopy_shared_keys = frozenset([
        '_init_args',
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
                 reset_stateful_function_when=None,
                 prefs=None,
                 **kwargs):
        """Assign default preferences; enforce required params; validate and instantiate params and execute method

        Initialization arguments:
        - default_variable (anything): establishes type for the variable, used for validation
        - size (int or list/array of ints): if specified, establishes variable if variable was not already specified
        - params_default (dict): assigned as default
        Note: if parameter_validation is off, validation is suppressed (for efficiency) (Component class default = on)

        """
        self._handle_illegal_kwargs(**kwargs)

        context = Context(
            source=ContextFlags.COMPONENT,
            execution_phase=ContextFlags.IDLE,
        )

        try:
            function_params = copy.copy(param_defaults[FUNCTION_PARAMS])
        except (KeyError, TypeError):
            function_params = {}

        # allow override of standard arguments with arguments specified in
        # params (here, param_defaults) argument
        # (if there are duplicates, later lines override previous)
        parameter_values = {
            **{
                'function': function,
                'variable': default_variable
            },
            **kwargs,
            **(param_defaults if param_defaults is not None else {}),
        }

        self._initialize_parameters(
            context=context,
            **parameter_values
        )

        self.initial_function_parameters = {
            k: v for k, v in parameter_values.items() if k in self.parameters.names() and getattr(self.parameters, k).function_parameter
        }

        var = call_with_pruned_args(
            self._handle_default_variable,
            default_variable=default_variable,
            size=size,
            **parameter_values
        )
        if var is None:
            default_variable = self.defaults.variable
        else:
            default_variable = var
            self.defaults.variable = default_variable
            self.parameters.variable._user_specified = True

        # we must know the final variable shape before setting up parameter
        # Functions or they will mismatch
        self._instantiate_parameter_classes(context)
        self._validate_subfunctions()

        if reset_stateful_function_when is not None:
            self.reset_stateful_function_when = reset_stateful_function_when
        else:
            self.reset_stateful_function_when = Never()

        # self.componentName = self.componentType
        try:
            self.componentName = self.componentName or self.componentType
        except AttributeError:
            self.componentName = self.componentType

        # ENFORCE REGISRY
        if self.__class__.__bases__[0].__bases__[0].__bases__[0].__name__ == 'ShellClass':
            try:
                self.__class__.__bases__[0].registry
            except AttributeError:
                raise ComponentError("{0} is a category class and so must implement a registry".
                                    format(self.__class__.__bases__[0].__name__))

        # ASSIGN PREFS
        _assign_prefs(self, prefs, BasePreferenceSet)

        # ASSIGN LOG
        from psyneulink.core.globals.log import Log
        self.log = Log(owner=self)
        # Used by run to store return value of execute
        self.results = []

        if function is None:
            if (
                param_defaults is not None
                and FUNCTION in param_defaults
                and param_defaults[FUNCTION] is not None
            ):
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

        # Validate the set passed in
        self._instantiate_defaults(variable=default_variable,
               request_set=parameter_values,  # requested set
               assign_missing=True,                   # assign missing params from classPreferences to instanceDefaults
               target_set=self.defaults.values(), # destination set to which params are being assigned
               default_set=self.class_defaults.values(),   # source set from which missing params are assigned
               context=context,
               )

        self._runtime_params_reset = {}

        # KDM 11/12/19: this exists to deal with currently unknown attribute
        # setting - if not set these will be included in logs as COMMAND_LINE
        # settings. Remove this eventually
        self.most_recent_context = context

        # INSTANTIATE ATTRIBUTES BEFORE FUNCTION
        # Stub for methods that need to be executed before instantiating function
        #    (e.g., _instantiate_sender and _instantiate_receiver in Projection)
        # Allow _instantiate_attributes_before_function of subclass
        #    to modify/replace function arg provided in constructor (e.g. TransferWithCosts)
        function = self._instantiate_attributes_before_function(function=function, context=context) or function

        # INSTANTIATE FUNCTION
        #    - assign initial function parameter values from ParameterPorts,
        #    - assign function's output to self.defaults.value (based on call of self.execute)
        self._instantiate_function(function=function, function_params=function_params, context=context)

        self._instantiate_value(context=context)

        # INSTANTIATE ATTRIBUTES AFTER FUNCTION
        # Stub for methods that need to be executed after instantiating function
        #    (e.g., instantiate_output_port in Mechanism)
        self._instantiate_attributes_after_function(context=context)

        self._validate(context=context)

        self.initialization_status = ContextFlags.INITIALIZED

        self._compilation_data = self._CompilationData(owner=self)

        self._update_parameter_components(context)

    def __repr__(self):
        return '({0} {1})'.format(type(self).__name__, self.name)
        #return '{1}'.format(type(self).__name__, self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __deepcopy__(self, memo):
        if 'no_shared' in memo and memo['no_shared']:
            shared_types = tuple()
        else:
            shared_types = (Component, ComponentsMeta)

        fun = get_deepcopy_with_shared(
            self._deepcopy_shared_keys,
            shared_types
        )
        newone = fun(self, memo)

        if newone.parameters is not newone.class_parameters:
            # may be in DEFERRED INIT, so parameters/defaults belongs to class
            newone.parameters._owner = newone
            newone.defaults._owner = newone
            newone._compilation_data._owner = newone

        # by copying, this instance is no longer "inherent" to a single
        # 'import psyneulink' call
        newone._is_pnl_inherent = False

        return newone

    # ------------------------------------------------------------------------------------------------------------------
    # Compilation support
    # ------------------------------------------------------------------------------------------------------------------
    def _get_compilation_state(self):
        # FIXME: MAGIC LIST, Use stateful tag for this
        whitelist = {"previous_time", "previous_value", "previous_v",
                     "previous_w", "random_state", "is_finished_flag",
                     "num_executions_before_finished", "num_executions",
                     "execution_count", "value", "input_ports", "output_ports"}
        blacklist = { # References to other components
                     "objective_mechanism", "agent_rep", "projections"}
        # Only mechanisms use "value" state
        if not hasattr(self, 'ports'):
            blacklist.add("value")
        def _is_compilation_state(p):
            val = p.get()   # memoize for this function
            return val is not None and p.name not in blacklist and \
                   (p.name in whitelist or isinstance(val, Component))

        return filter(_is_compilation_state, self.parameters)

    def _get_state_ids(self):
        return [sp.name for sp in self._get_compilation_state()]

    @property
    def llvm_state_ids(self):
        ids = getattr(self, "_state_ids", None)
        if ids is None:
            ids = self._get_state_ids()
            setattr(self, "_state_ids", ids)
        return ids

    def _get_state_initializer(self, context):
        def _convert(p):
            x = p.get(context)
            if isinstance(x, np.random.RandomState):
                # Skip first element of random state (id string)
                val = pnlvm._tupleize(x.get_state()[1:])
            elif isinstance(x, Time):
                val = tuple(getattr(x, Time._time_scale_attr_map[t]) for t in TimeScale)
            elif isinstance(x, Component):
                return x._get_state_initializer(context)
            elif isinstance(x, ContentAddressableList):
                return tuple(p._get_state_initializer(context) for p in x)
            else:
                val = pnlvm._tupleize(x)

            return tuple(val for _ in range(p.history_min_length + 1))

        return tuple(map(_convert, self._get_compilation_state()))

    def _get_compilation_params(self):
        # FIXME: MAGIC LIST, detect used parameters automatically
        blacklist = {# Stateful parameters
                     "previous_time", "previous_value", "previous_v",
                     "previous_w", "random_state", "is_finished_flag",
                     "num_executions_before_finished", "num_executions",
                     "variable", "value", "saved_values", "saved_samples",
                     # Invalid types
                     "input_port_variables", "results", "simulation_results",
                     "monitor_for_control", "feature_values", "simulation_ids",
                     "input_labels_dict", "output_labels_dict",
                     "modulated_mechanisms", "grid",
                     "activation_derivative_fct", "input_specification",
                     # Reference to other components
                     "objective_mechanism", "agent_rep", "projections",
                     # Shape mismatch
                     "costs", "auto", "hetero",
                     # autodiff specific types
                     "pytorch_representation", "optimizer"}
        # Mechanism's need few extra entires:
        # * matrix -- is never used directly, and is flatened below
        # * integration rate -- shape mismatch with param port input
        if hasattr(self, 'ports'):
            blacklist.update(["matrix", "integration_rate"])
        def _is_compilation_param(p):
            if p.name not in blacklist and not isinstance(p, ParameterAlias):
                #FIXME: this should use defaults
                val = p.get()
                # Check if the value type is valid for compilation
                return not isinstance(val, (str, ComponentsMeta,
                                            type(max),
                                            type(_is_compilation_param),
                                            type(self._get_compilation_params)))
            return False

        return filter(_is_compilation_param, self.parameters)

    def _get_param_ids(self):
        return [p.name for p in self._get_compilation_params()]

    @property
    def llvm_param_ids(self):
        ids = getattr(self, "_param_ids", None)
        if ids is None:
            ids = self._get_param_ids()
            setattr(self, "_param_ids", ids)
        return ids

    def _is_param_modulated(self, p):
        try:
            if p.name in self.owner.parameter_ports:
                return True
        except AttributeError:
            pass
        try:
            if p.name in self.parameter_ports:
                return True
        except AttributeError:
            pass
        try:
            modulated_params = (
                getattr(self.parameters, p.sender.modulation).source
                for p in self.owner.mod_afferents)
            if p in modulated_params:
                return True
        except AttributeError:
            pass

        return False

    def _get_param_initializer(self, context):
        def _convert(x):
            if isinstance(x, Enum):
                return x.value
            elif isinstance(x, SampleIterator):
                if isinstance(x.generator, list):
                    return tuple(v for v in x.generator)
                else:
                    return (x.start, x.step, x.num)
            elif isinstance(x, Component):
                return x._get_param_initializer(context)

            try:   # This can't use tupleize and needs to recurse to handle
                   # 'search_space' list of SampleIterators
                return tuple(_convert(i) for i in x)
            except TypeError:
                return x if x is not None else tuple()

        def _get_values(p):
            param = p.get(context)
            # Modulated parameters change shape to array
            if np.isscalar(param) and self._is_param_modulated(p):
                return (param,)
            elif p.name == 'num_estimates':
                return 0 if param is None else param
            elif p.name == 'matrix': # Flatten matrix
                return tuple(np.asfarray(param).flatten())
            return _convert(param)

        return tuple(map(_get_values, self._get_compilation_params()))

    def _gen_llvm_function_reset(self, ctx, builder, *_, tags):
        assert "reset" in tags
        return builder

    def _gen_llvm_function(self, *, ctx:pnlvm.LLVMBuilderContext,
                                    extra_args=[], tags:frozenset):
        args = [ctx.get_param_struct_type(self).as_pointer(),
                ctx.get_state_struct_type(self).as_pointer(),
                ctx.get_input_struct_type(self).as_pointer(),
                ctx.get_output_struct_type(self).as_pointer()]
        builder = ctx.create_llvm_function(args + extra_args, self, tags=tags)

        params, state, arg_in, arg_out = builder.function.args[:len(args)]
        if len(extra_args) == 0:
            for p in params, state, arg_in, arg_out:
                p.attributes.add('noalias')

        if "reset" in tags:
            builder = self._gen_llvm_function_reset(ctx, builder, params, state,
                                                    arg_in, arg_out, tags=tags)
        else:
            builder = self._gen_llvm_function_body(ctx, builder, params, state,
                                                   arg_in, arg_out, tags=tags)
        builder.ret_void()
        return builder.function

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
        default_variable = self._parse_arg_variable(default_variable)

        if default_variable is None:
            default_variable = self._handle_size(size, default_variable)

            if default_variable is None or default_variable is NotImplemented:
                return None
            else:
                self._variable_shape_flexibility = self._specified_variable_shape_flexibility
        else:
            self._variable_shape_flexibility = self._specified_variable_shape_flexibility

        return convert_to_np_array(default_variable, dimension=1)

    # ELIMINATE SYSTEM
    # IMPLEMENTATION NOTE: (7/7/17 CW) Due to System and Process being initialized with size at the moment (which will
    # be removed later), I’m keeping _handle_size in Component.py. I’ll move the bulk of the function to Mechanism
    # through an override, when Composition is done. For now, only Port.py overwrites _handle_size().
    def _handle_size(self, size, variable):
        """If variable is None, _handle_size tries to infer variable based on the **size** argument to the
            __init__() function. This method is overwritten in subclasses like Mechanism and Port.
            If self is a Mechanism, it converts variable to a 2D array, (for a Mechanism, variable[i] represents
            the input from the i-th InputPort). If self is a Port, variable is a 1D array and size is a length-1 1D
            array. It performs some validations on size and variable as well. This function is overridden in Port.py.
            If size is NotImplemented (usually in the case of Projections/Functions), then this function passes without
            doing anything. Be aware that if size is NotImplemented, then variable is never cast to a particular shape.
        """
        if size is not NotImplemented:
            self._variable_shape_flexibility = self._specified_variable_shape_flexibility
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
                    if hasattr(self, 'prefs') and hasattr(self.prefs, VERBOSE_PREF) and self.prefs.verbosePref:
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
                        if hasattr(self, 'prefs') and hasattr(self.prefs, VERBOSE_PREF) and self.prefs.verbosePref:
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
                    variable = convert_to_np_array(variable)
                # TODO: fix bare except
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
                    if hasattr(self, 'prefs') and hasattr(self.prefs, VERBOSE_PREF) and self.prefs.verbosePref:
                        warnings.warn("The size arg of {} conflicts with the length "
                                      "of its variable arg ({}) at element {}: variable takes precedence".
                                      format(self.name, size, variable))
                else:
                    for i in range(len(size)):
                        if size[i] != len(variable[i]):
                            if hasattr(self, 'prefs') and hasattr(self.prefs, VERBOSE_PREF) and self.prefs.verbosePref:
                                warnings.warn("The size arg of {} ({}) conflicts with the length "
                                              "of its variable arg ({}) at element {}: variable takes precedence".
                                              format(self.name, size[i], variable[i], i))

        return variable

    def _handle_illegal_kwargs(self, **kwargs):
        illegal_args = [
            arg
            for arg in kwargs.keys()
            if arg not in (
                self.standard_constructor_args
                + self.parameters.names(show_all=True)
                # arguments to constructor
                + list(get_all_explicit_arguments(self.__class__, '__init__'))
            )
        ]

        if illegal_args:
            plural = ''
            if len(illegal_args) > 1:
                plural = 's'
            raise ComponentError(
                f"Unrecognized argument{plural} in constructor for {self.name} "
                f"(type: {self.__class__.__name__}): {repr(', '.join(illegal_args))}"
            )

    # breaking self convention here because when storing the args,
    # "self" is often among them. To avoid needing to preprocess to
    # avoid argument duplication, use "self_" in this method signature
    def _store_deferred_init_args(self_, **kwargs):
        self = self_

        try:
            del kwargs['self']
        except KeyError:
            pass

        # add unspecified kwargs
        kwargs_names = [
            k
            for k, v in inspect.signature(self.__init__).parameters.items()
            if v.kind is inspect.Parameter.VAR_KEYWORD
        ]

        self._init_args = {
            k: v
            for k, v in kwargs.items()
            if (
                k in get_all_explicit_arguments(self.__class__, '__init__')
                or k in kwargs_names
            )
        }
        try:
            self._init_args.update(self._init_args['kwargs'])
            del self._init_args['kwargs']
        except KeyError:
            pass

    @handle_external_context()
    def _deferred_init(self, context=None):
        """Use in subclasses that require deferred initialization
        """
        if self.initialization_status == ContextFlags.DEFERRED_INIT:

            # Flag that object is now being initialized
            #       (usually in _instantiate_function)
            self.initialization_status = ContextFlags.INITIALIZING

            self._init_args['context'] = context

            # Complete initialization
            # MODIFIED 10/27/18 OLD:
            super(self.__class__,self).__init__(**self._init_args)
            # MODIFIED 10/27/18 NEW:  FOLLOWING IS NEEDED TO HANDLE FUNCTION DEFERRED INIT (JDC)
            # try:
            #     super(self.__class__,self).__init__(**self._init_args)
            # except:
            #     self.__init__(**self._init_args)
            # MODIFIED 10/27/18 END

            # If name was assigned, "[DEFERRED INITIALIZATION]" was appended to it, so remove it
            if DEFERRED_INITIALIZATION in self.name:
                self.name = self.name.replace("[" + DEFERRED_INITIALIZATION + "]", "")
            # Otherwise, allow class to replace std default name with class-specific one if it has a method for doing so
            else:
                self._assign_default_name()

            del self._init_args

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

    def _set_parameter_value(self, param, val, context=None):
        getattr(self.parameters, param)._set(val, context)
        if hasattr(self, "parameter_ports"):
            if param in self.parameter_ports:
                new_port_value = self.parameter_ports[param].execute(
                    context=Context(execution_phase=ContextFlags.EXECUTING, execution_id=context.execution_id)
                )
                self.parameter_ports[param].parameters.value._set(new_port_value, context)
        elif hasattr(self, "owner"):
            if hasattr(self.owner, "parameter_ports"):
                if param in self.owner.parameter_ports:
                    new_port_value = self.owner.parameter_ports[param].execute(
                        context=Context(execution_phase=ContextFlags.EXECUTING, execution_id=context.execution_id)
                    )
                    self.owner.parameter_ports[param].parameters.value._set(new_port_value, context)

    def _check_args(self, variable=None, params=None, context=None, target_set=None):
        """validate variable and params, instantiate variable (if necessary) and assign any runtime params.

        Called by functions to validate variable and params
        Validation can be suppressed by turning parameter_validation attribute off
        target_set is a params dictionary to which params should be assigned;

        Does the following:
        - instantiate variable (if missing or callable)
        - validate variable if PARAM_VALIDATION is set
        - resets leftover runtime params back to original values (only if execute method was called directly)
        - sets runtime params
        - validate params if PARAM_VALIDATION is set

        :param variable: (anything but a dict) - variable to validate
        :param params: (dict) - params to validate
        :target_set: (dict) - set to which params should be assigned
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

        # If params have been passed, treat as runtime params
        self._validate_and_assign_runtime_params(params, context)

        self.parameters.variable._set(variable, context=context)
        return variable

    def _validate_and_assign_runtime_params(self, runtime_params, context):
        """Validate runtime_params, cache for reset, and assign values

        Check that all params belong either to Component or its function (raise error if any are found that don't)
        Cache params to reset in _runtime_params_reset
        """

        # # MODIFIED 5/8/20 OLD:
        # # reset any runtime params that were leftover from a direct call to .execute (atypical)
        # if context.execution_id in self._runtime_params_reset:
        #     for key in self._runtime_params_reset[context.execution_id]:
        #         self._set_parameter_value(key, self._runtime_params_reset[context.execution_id][key], context)
        # self._runtime_params_reset[context.execution_id] = {}
        # MODIFIED 5/8/20 END

        from psyneulink.core.components.functions.function import is_function_type, FunctionError
        def generate_error(param_name):
            owner_name = ""
            if hasattr(self, OWNER) and self.owner:
                owner_name = f" of {self.owner.name}"
                if hasattr(self.owner, OWNER) and self.owner.owner:
                    owner_name = f"{owner_name} of {self.owner.owner.name}"
            err_msg=f"Invalid specification in runtime_params arg for {self.name}{owner_name}: '{param_name}'."
            if is_function_type(self):
                raise FunctionError(err_msg)
            else:
                raise ComponentError(err_msg)

        if isinstance(runtime_params, dict):
            for param_name in runtime_params:
                if not isinstance(param_name, str):
                    generate_error(param_name)
                elif hasattr(self, param_name):
                    if param_name in {FUNCTION, INPUT_PORTS, OUTPUT_PORTS}:
                        generate_error(param_name)
                    if context.execution_id not in self._runtime_params_reset:
                        self._runtime_params_reset[context.execution_id] = {}
                    self._runtime_params_reset[context.execution_id][param_name] = getattr(self.parameters,
                                                                                           param_name)._get(context)
                    self._set_parameter_value(param_name, runtime_params[param_name], context)
                # Any remaining params should either belong to the Component's function
                #    or, if the Component is a Function, to it or its owner
                elif ( # If Component is not a function, and its function doesn't have the parameter or
                        (not is_function_type(self) and not hasattr(self.function, param_name))
                       # the Component is a standalone function:
                       or (is_function_type(self) and not self.owner)):
                    generate_error(param_name)

        elif runtime_params:    # not None
            raise ComponentError(f"Invalid specification of runtime parameters for {self.name}: {runtime_params}.")

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
          request_set must contain a dict of params to be assigned to target_set
          If assign_missing option is set, then any params defined for the class
              but not included in the requested set are assigned values from the default_set;
              if request_set is None, then all values in the target_set are assigned from the default_set
          Class defaults can not be passed as target_set
              IMPLEMENTATION NOTE:  for now, treating class defaults as hard coded;
                                    could be changed in the future simply by commenting out code below

          If not context:  instantiates function and any ports specified in request set
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

        # VALIDATE VARIABLE

        if not (context.source & (ContextFlags.COMMAND_LINE | ContextFlags.PROPERTY)):
            # if variable has been passed then validate and, if OK, assign as self.defaults.variable
            variable = self._validate_variable(variable, context=context)

        # If no params were passed, then done
        if request_set is None and target_set is None and default_set is None:
            return

        # VALIDATE PARAMS

        # if request_set has been passed or created then validate and, if OK, assign params to target_set
        if request_set:
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
        from psyneulink.core.components.shellclasses import (
            Composition_Base, Function, Mechanism, Port, Process_Base,
            Projection, System_Base
        )

        # excludes Function
        shared_types = (
            Mechanism,
            Port,
            Projection,
            System_Base,
            Process_Base,
            Composition_Base,
            ComponentsMeta,
            types.MethodType,
            functools.partial,
        )
        alias_names = {p.name for p in self.class_parameters if isinstance(p, ParameterAlias)}

        self.parameters = self.Parameters(owner=self, parent=self.class_parameters)

        # assign defaults based on pass in params and class defaults
        defaults = {
            k: v for (k, v) in self.class_defaults.values(show_all=True).items()
            if k not in alias_names
        }

        if param_defaults is not None:
            # Exclude any function_params from the items to set on this Component
            # because these should just be pointers to the parameters of the same
            # name on this Component's function
            # Exclude any pass parameters whose value is None (assume this means "use the normal default")
            d = {
                k: v for (k, v) in param_defaults.items()
                if (
                    (
                        k not in defaults
                        and k not in alias_names
                    )
                    or v is not None
                )
            }
            for p in d:
                try:
                    parameter_obj = getattr(self.parameters, p)
                except AttributeError:
                    # p in param_defaults does not correspond to a Parameter
                    continue

                if parameter_obj.structural:
                    parameter_obj.spec = d[p]

                if parameter_obj.modulable:
                    # later, validate this
                    try:
                        modulable_param_parser = self.parameters._get_prefixed_method(
                            parse=True,
                            modulable=True
                        )
                        parsed = modulable_param_parser(p, d[p])

                        if parsed is not d[p]:
                            # we have a modulable param spec
                            parameter_obj.spec = d[p]
                            d[p] = parsed
                            param_defaults[p] = parsed
                    except AttributeError:
                        pass

            defaults.update(d)

        for k in defaults:
            defaults[k] = copy_parameter_value(
                defaults[k],
                shared_types=shared_types
            )

        self.defaults = Defaults(owner=self, **defaults)

        def _is_user_specified(parameter):
            return (
                parameter.name in param_defaults
                and param_defaults[parameter.name] is not None
            )

        for p in filter(lambda x: isinstance(x, ParameterAlias), self.parameters):
            if _is_user_specified(p):
                if _is_user_specified(p.source):
                    if param_defaults[p.name] is not param_defaults[p.source.name]:
                        raise ComponentError(
                            f"Multiple values ({p.name}: {param_defaults[p.name]}"
                            f"\t{p.source.name}: {param_defaults[p.source.name]} "
                            f"assigned to identical Parameters. {p.name} is an alias "
                            f"of {p.source.name}",
                            component=self,
                        )
                else:
                    param_defaults[p.source.name] = param_defaults[p.name]

        for p in filter(lambda x: not isinstance(x, ParameterAlias), self.parameters):
            p._user_specified = _is_user_specified(p)

            # copy spec so it is not overwritten later
            # TODO: check if this is necessary
            p.spec = copy_parameter_value(p.spec)

            # set default to None context to ensure it exists
            if p.getter is None and p._get(context) is None:
                if p._user_specified:
                    val = param_defaults[p.name]

                    if isinstance(val, Function):
                        if val.owner is not None:
                            val = copy.deepcopy(val)

                        val.owner = self
                else:
                    val = copy_parameter_value(
                        p.default_value,
                        shared_types=shared_types
                    )

                    if isinstance(val, Function):
                        val.owner = self

                p.set(val, context=context, skip_history=True, override=True)

            if isinstance(p.default_value, Function):
                p.default_value.owner = p

    def _instantiate_parameter_classes(self, context=None):
        """
            An optional method that will take any Parameter values in
            **context** that are classes/types, and instantiate them.
        """
        from psyneulink.core.components.shellclasses import Function

        # (this originally occurred in _validate_params)
        for p in self.parameters:
            if p.getter is None:
                val = p._get(context)
                if (
                    p.name != FUNCTION
                    and not p.reference
                ):
                    if (
                        inspect.isclass(val)
                        and issubclass(val, Function)
                    ):
                        val = val()
                        val.owner = self
                        p._set(val, context)

        self._update_parameter_class_variables(context)

    def _update_parameter_class_variables(self, context=None):
        from psyneulink.core.components.shellclasses import Function
        for p in self.parameters:
            if p.getter is None:
                val = p._get(context)
                if (
                    p.name != FUNCTION
                    and not p.reference
                    and isinstance(val, Function)
                ):
                    try:
                        parse_variable_method = getattr(
                            self,
                            f'_parse_{p.name}_variable'
                        )
                        function_default_variable = copy.deepcopy(
                            parse_variable_method(self.defaults.variable)
                        )
                    except AttributeError:
                        # no parsing method, assume same shape as owner
                        function_default_variable = copy.deepcopy(
                            self.defaults.variable
                        )

                    incompatible = False

                    if function_default_variable.shape != val.defaults.variable.shape:
                        incompatible = True
                        if val._variable_shape_flexibility is DefaultsFlexibility.INCREASE_DIMENSION:
                            increased_dim = np.asarray([val.defaults.variable])

                            if increased_dim.shape == function_default_variable.shape:
                                function_default_variable = increased_dim
                                incompatible = False
                        elif val._variable_shape_flexibility is DefaultsFlexibility.FLEXIBLE:
                            incompatible = False

                    if not incompatible:
                        val._update_default_variable(
                            function_default_variable,
                            context
                        )

                        if isinstance(p.default_value, Function):
                            p.default_value._update_default_variable(
                                function_default_variable,
                                context
                            )

    @handle_external_context()
    def reset_params(self, mode=ResetMode.INSTANCE_TO_CLASS, context=None):
        """Reset current and/or instance defaults

        If called with:
            - CURRENT_TO_INSTANCE_DEFAULTS all current param settings are set to instance defaults
            - INSTANCE_TO_CLASS all instance defaults are set to class defaults
            - ALL_TO_CLASS_DEFAULTS all current and instance param settings are set to class defaults

        :param mode: (ResetMode) - determines which params are reset
        :return none:
        """

        if not isinstance(mode, ResetMode):
            warnings.warn("No ResetMode specified for reset_params; CURRENT_TO_INSTANCE_DEFAULTS will be used")

        for param in self.parameters:
            if mode == ResetMode.CURRENT_TO_INSTANCE_DEFAULTS:
                param._set(
                    copy_parameter_value(param.default_value),
                    context=context,
                    skip_history=True,
                    skip_log=True,
                )
            elif mode == ResetMode.INSTANCE_TO_CLASS:
                param.reset()
            elif mode == ResetMode.ALL_TO_CLASS_DEFAULTS:
                param.reset()
                param._set(
                    copy_parameter_value(param.default_value),
                    context=context,
                    skip_history=True,
                    skip_log=True,
                )

    def _initialize_from_context(self, context, base_context=Context(execution_id=None), override=True, visited=None):
        if context.execution_id is base_context.execution_id:
            return

        if visited is None:
            visited = set()

        for comp in self._dependent_components:
            if comp not in visited:
                visited.add(comp)
                comp._initialize_from_context(context, base_context, override, visited=visited)

        non_alias_params = [p for p in self.stateful_parameters if not isinstance(p, ParameterAlias)]
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
            for input ports is expected

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
        Multiple port variables, each with a single value variable:  [[0], [0]] -> [array[0], array[0]]

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

        This performs top-level type validation of params

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

            # Check that param is in self.defaults (if not, it is assumed to be invalid for this object)
            if param_name not in self.defaults.names(show_all=True):
                continue

            # The default value of the param is None: suppress type checking
            # IMPLEMENTATION NOTE: this can be used for params with multiple possible types,
            #                      until type lists are implemented (see below)
            if getattr(self.defaults, param_name) is None or getattr(self.defaults, param_name) is NotImplemented:
                if self.prefs.verbosePref:
                    warnings.warn(f"{param_name} is specified as None for {self.name} which suppresses type checking.")
                if target_set is not None:
                    target_set[param_name] = param_value
                continue

            # If the value in self.defaults is a type, check if param value is an instance of it
            if inspect.isclass(getattr(self.defaults, param_name)):
                if isinstance(param_value, getattr(self.defaults, param_name)):
                    target_set[param_name] = param_value
                    continue
                # If the value is a Function class, allow any instance of Function class
                from psyneulink.core.components.functions.function import Function_Base
                if issubclass(getattr(self.defaults, param_name), Function_Base):
                    # if isinstance(param_value, (function_type, Function_Base)):  <- would allow function of any kind
                    if isinstance(param_value, Function_Base):
                        target_set[param_name] = param_value
                        continue

            # If the value in self.defaults is an object, check if param value is the corresponding class
            # This occurs if the item specified by the param has not yet been implemented (e.g., a function)
            if inspect.isclass(param_value):
                if isinstance(getattr(self.defaults, param_name), param_value):
                    continue

            # If the value is a projection, projection class, or a keyword for one, for anything other than
            #    the FUNCTION param (which is not allowed to be specified as a projection)
            #    then simply assign value (implication of not specifying it explicitly);
            #    this also allows it to pass the test below and function execution to occur for initialization;
            from psyneulink.core.components.shellclasses import Projection
            if (((isinstance(param_value, str) and
                          param_value in {CONTROL_PROJECTION, LEARNING_PROJECTION, LEARNING}) or
                isinstance(param_value, Projection) or  # These should be just ControlProjection or LearningProjection
                inspect.isclass(param_value) and issubclass(param_value,(Projection)))
                and not param_name == FUNCTION):
                param_value = getattr(self.defaults, param_name)

            # If self is a Function and param is a class ref for function, instantiate it as the function
            from psyneulink.core.components.functions.function import Function_Base
            if (isinstance(self, Function_Base) and
                    inspect.isclass(param_value) and
                    inspect.isclass(getattr(self.defaults, param_name))
                    and issubclass(param_value, getattr(self.defaults, param_name))):
                    # Assign instance to target and move on
                    #  (compatiblity check no longer needed and can't handle function)
                    target_set[param_name] = param_value()
                    continue

            # Check if param value is of same type as one with the same name in defaults
            #    don't worry about length
            if iscompatible(param_value, getattr(self.defaults, param_name), **{kwCompatibilityLength:0}):
                if isinstance(param_value, dict):

                    # If assign_default_FUNCTION_PARAMS is False, it means that function's class is
                    #     compatible but different from the one in defaults;
                    #     therefore, FUNCTION_PARAMS will not match defaults;
                    #     instead, check that functionParams are compatible with the function's default params
                    if param_name == FUNCTION_PARAMS:
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
                                        getattr(function.defaults, entry_name)
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
                                # Make sure [entry_name] is in self.defaults
                                try:
                                    getattr(self.defaults, param_name)[entry_name]
                                except KeyError:
                                    raise ComponentError("{0} is not a valid entry in {1} for {2} ".
                                                        format(entry_name, param_name, self.name))
                                # TBI: (see above)
                                # if not iscompatible(entry_value,
                                #                     getattr(self.defaults, param_name)[entry_name],
                                #                     **{kwCompatibilityLength:0}):
                                #     raise ComponentError("{0} ({1}) in {2} of {3} must be a {4}".
                                #         format(entry_name, entry_value, param_name, self.name,
                                #                type(getattr(self.defaults, param_name)[entry_name]).__name__))
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
                    if not isinstance(param_value, Iterable) or isinstance(param_value, str):
                        target_set[param_name] = param_value
                    else:
                        # hack for validation until it's streamlined
                        # parse modulable parameter values
                        if getattr(self.parameters, param_name).modulable:
                            try:
                                target_set[param_name] = param_value.copy()
                            except AttributeError:
                                try:
                                    modulable_param_parser = self.parameters._get_prefixed_method(
                                        parse=True,
                                        modulable=True
                                    )
                                    param_value = modulable_param_parser(param_name, param_value)
                                    target_set[param_name] = param_value
                                except AttributeError:
                                    target_set[param_name] = param_value.copy()

                        else:
                            target_set[param_name] = copy.copy(param_value)

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
                    if is_matrix(getattr(self.defaults, param_name)):
                        # FIX:  ?? ASSIGN VALUE HERE, OR SIMPLY ALLOW AND ASSUME IT WILL BE PARSED ELSEWHERE
                        # param_value = getattr(self.defaults, param_name)
                        # target_set[param_name] = param_value
                        target_set[param_name] = param_value
                    else:
                        raise ComponentError("Value of {} param for {} ({}) must be a valid matrix specification".
                                             format(param_name, self.name, param_value))
                target_set[param_name] = param_value

            # Parameter is not a valid type
            else:
                if type(getattr(self.defaults, param_name)) is type:
                    type_name = 'the name of a subclass of ' + getattr(self.defaults, param_name).__base__.__name__
                raise ComponentError("Value of {} param for {} ({}) is not compatible with {}".
                                    format(param_name, self.name, param_value, type_name))

    def _validate_subfunctions(self):
        from psyneulink.core.components.shellclasses import Function

        for p in self.parameters:
            if (
                p.name != FUNCTION  # has specialized validation
                and isinstance(p.default_value, Function)
                and not p.reference
                and not p.function_parameter
            ):
                # TODO: assert it's not stateful?
                function_variable = p.default_value.defaults.variable
                expected_function_variable = self.defaults.variable

                try:
                    parse_variable_method = getattr(
                        self,
                        f'_parse_{p.name}_variable'
                    )
                    expected_function_variable = parse_variable_method(
                        expected_function_variable
                    )

                except AttributeError:
                    pass

                if not function_variable.shape == expected_function_variable.shape:
                    def _create_justified_line(k, v, error_line_len=110):
                        return f'{k}: {v.rjust(error_line_len - len(k))}'

                    raise ParameterError(
                        f'Variable shape incompatibility between {self} and its {p.name} Parameter'
                        + _create_justified_line(
                            f'\n{self}.variable',
                            f'{expected_function_variable}    (numpy.array shape: {np.asarray(expected_function_variable).shape})'
                        )
                        + _create_justified_line(
                            f'\n{self}.{p.name}.variable',
                            f'{function_variable}    (numpy.array shape: {np.asarray(function_variable).shape})'
                        )
                    )

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

        if param_spec not in MODULATORY_SPEC_KEYWORDS:
            return(param_value)

        try:
            param_default_value = getattr(self.defaults, param_name)
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
        from psyneulink.core.components.mechanisms.modulatory.modulatorymechanism import ModulatoryMechanism_Base
        from psyneulink.core.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base
        from psyneulink.core.components.ports.modulatorysignals.modulatorysignal import ModulatorySignal

        ALLOWABLE_TUPLE_SPEC_KEYWORDS = MODULATORY_SPEC_KEYWORDS
        ALLOWABLE_TUPLE_SPEC_CLASSES = (ModulatoryProjection_Base, ModulatorySignal, ModulatoryMechanism_Base)

        # If the 2nd item is a CONTROL or LEARNING SPEC, return the first item as the value
        if (isinstance(param_spec, tuple) and len(param_spec) == 2 and
                not isinstance(param_spec[1], (dict, list, np.ndarray)) and
                (param_spec[1] in ALLOWABLE_TUPLE_SPEC_KEYWORDS or
                 isinstance(param_spec[1], ALLOWABLE_TUPLE_SPEC_CLASSES) or
                 (inspect.isclass(param_spec[1]) and issubclass(param_spec[1], ALLOWABLE_TUPLE_SPEC_CLASSES)))
            ):
            value = param_spec[0]

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
        This checks for an execute method in function
        If a specification is not present or valid:
            - it checks self.execute and, if present, kwExecute is assigned to it
            - if self.execute is not present or valid, an exception is raised
        When completed, there is guaranteed to be a valid method in self.function and/or self.execute;
            otherwise, an exception is raised

        Notes:
            * no new assignments (to FUNCTION or self.execute) are made here, except:
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
                raise ComponentError("{0} must either implement a function method or specify one in {0}.Parameters".
                                    format(self.__class__.__name__))

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
            self.function = function
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
        """Instantiate function defined in <subclass>.function or <subclass>.function

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
            - self._function === self.function
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
        # JDC 3/6/19:  ?what about parameter ports for its parameters (see python function problem below)?
        if isinstance(function, types.MethodType):
            try:
                if isinstance(function.__self__, Function):
                    function = function.__self__
            except AttributeError:
                pass

        # Specification is a standard python function, so wrap as a UserDefnedFunction
        # Note:  parameter_ports for function's parameters will be created in_instantiate_attributes_after_function
        if isinstance(function, types.FunctionType):
            self.function = UserDefinedFunction(default_variable=function_variable,
                                                custom_function=function,
                                                owner=self,
                                                context=context)

        # Specification is an already implemented Function
        elif isinstance(function, Function):
            if function_variable.shape != function.defaults.variable.shape:
                owner_str = ''
                if hasattr(self, 'owner') and self.owner is not None:
                    owner_str = f' of {repr(self.owner.name)}'
                if function._variable_shape_flexibility is DefaultsFlexibility.RIGID:
                    raise ComponentError(f'Variable format ({function.defaults.variable}) of {function.name} '
                                         f'is not compatible with the variable format ({function_variable}) '
                                         f'of {repr(self.name)}{owner_str} to which it is being assigned.')
                                         # f'Make sure variable for {function.name} is 2d.')
                elif function._variable_shape_flexibility is DefaultsFlexibility.INCREASE_DIMENSION:
                    function_increased_dim = np.asarray([function.defaults.variable])
                    if function_variable.shape != function_increased_dim.shape:
                        raise ComponentError(f'Variable format ({function.defaults.variable}) of {function.name} '
                                             f'is not compatible with the variable format ({function_variable})'
                                             f' of {repr(self.name)}{owner_str} to which it is being assigned.')
                                             # f'Make sure variable for {function.name} is 2d.')

            # class default functions should always be copied, otherwise anything this component
            # does with its function will propagate to anything else that wants to use
            # the default
            if function.owner is None:
                self.function = function
            elif function.owner is self:
                try:
                    if function._is_pnl_inherent:
                        # This will most often occur if a Function instance is
                        # provided as a default argument in a constructor. These
                        # should instead be added as default values for the
                        # corresponding Parameter.
                        # Adding the function as a default constructor argument
                        # will lead to incorrect setting of
                        # Parameter._user_specified
                        warnings.warn(
                            f'{function} is generated once during import of'
                            ' psyneulink, and is now being reused. Please report'
                            ' this, including the script you were using, to the'
                            ' psyneulink developers at'
                            ' psyneulinkhelp@princeton.edu or'
                            ' https://github.com/PrincetonUniversity/PsyNeuLink/issues'
                        )
                        self.function = copy.deepcopy(function)
                except AttributeError:
                    self.function = function
            else:
                self.function = copy.deepcopy(function)

            # set owner first because needed for is_initializing calls
            self.function.owner = self
            self.function._update_default_variable(function_variable, context)

        # Specification is Function class
        # Note:  parameter_ports for function's parameters will be created in_instantiate_attributes_after_function
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

                # matrix is determined from ParameterPort based on string value in function_params
                # update it here if needed
                if MATRIX in kwargs_to_instantiate:
                    try:
                        kwargs_to_instantiate[MATRIX] = self.parameter_ports[MATRIX].defaults.value
                    except (AttributeError, KeyError, TypeError):
                        pass

            _, kwargs = prune_unused_args(function.__init__, args=[], kwargs=kwargs_to_instantiate)
            self.function = function(default_variable=function_variable, owner=self, **kwargs)

        else:
            raise ComponentError(f'Unsupported function type: {type(function)}, function={function}.')

        # KAM added 6/14/18 for functions that do not pass their has_initializers status up to their owner via property
        # FIX: need comprehensive solution for has_initializers; need to determine whether ports affect mechanism's
        # has_initializers status
        if self.function.has_initializers:
            self.has_initializers = True

        self._parse_param_port_sources()

    def _instantiate_attributes_after_function(self, context=None):
        if hasattr(self, "_parameter_ports"):
            for param_port in self._parameter_ports:
                setattr(self.__class__, "mod_" + param_port.name, make_property_mod(param_port.name))
                setattr(self.__class__, "get_mod_" + param_port.name, make_stateful_getter_mod(param_port.name))

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

    def _update_default_variable(self, new_default_variable, context=None):
        self.defaults.variable = copy.deepcopy(new_default_variable)

        # exclude value from validation because it isn't updated until
        # _instantiate_value is called
        call_with_pruned_args(
            self._validate_params,
            variable=new_default_variable,
            request_set={
                k: v.default_value
                for k, v in self.parameters.values(True).items()
                if k not in {'value'} and not isinstance(v, ParameterAlias)
            },
            target_set={},
            context=context
        )
        self._instantiate_value(context)

        function_variable = self._parse_function_variable(
            new_default_variable,
            context
        )
        try:
            self.function._update_default_variable(function_variable, context)
        except AttributeError:
            pass

        # TODO: is it necessary to call _validate_value here?

    def initialize(self, context=None):
        raise ComponentError("{} class does not support initialize() method".format(self.__class__.__name__))

    def _check_for_composition(self, context=None):
        """Allow Component to check whether it or its attributes are suitable for inclusion in a Composition
        Called by Composition.add_node.
        """
        pass

    @handle_external_context(execution_id=NotImplemented)
    def reset(self, *args, context=None):
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
            new_value = self.function.reset(*args, context=context)
            self.parameters.value.set(np.atleast_2d(new_value), context, override=True)
        else:
            raise ComponentError(f"Resetting {self.name} is not allowed because this Component is not stateful. "
                                 "(It does not have an accumulator to reset).")

    @handle_external_context()
    def execute(self, variable=None, context=None, runtime_params=None):
        if context is None:
            try:
                context = self.owner.most_recent_context
            except AttributeError:
                context = self.most_recent_context

        if context.source is ContextFlags.COMMAND_LINE:
            self._initialize_from_context(context, override=False)

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
                self._increment_num_executions(context,
                                               [TimeScale.TIME_STEP, TimeScale.PASS, TimeScale.TRIAL, TimeScale.RUN])
            self._update_current_execution_time(context=context)

        value = None

        # GET VALUE if specified in runtime_params
        if runtime_params and VALUE in runtime_params:
            # Get value and then pop from runtime_param, as no need to restore to previous value
            value = np.atleast_1d(runtime_params.pop(VALUE))
            # Eliminate any other params (including ones for function),
            #  since they will not be assigned and therefore should not be restored to previous value below
            #  (doing so would restore them to the previous previous value)
            runtime_params = {}

        # CALL FUNCTION if value is not specified
        if value is None:
            # IMPLEMENTATION NOTE:  **kwargs is included to accommodate required arguments
            #                     that are specific to particular class of Functions
            #                     (e.g., error_matrix for LearningMechanism and controller for EVCControlMechanism)
            function_variable = self._parse_function_variable(variable, context=context)
            # IMPLEMENTATION NOTE: Need to pass full runtime_params (and not just function's params) since
            #                      Mechanisms with secondary functions (e.g., IntegratorMechanism) seem them
            value = self.function(variable=function_variable, context=context, params=runtime_params, **kwargs)
            try:
                self.function.parameters.value._set(value, context)
            except AttributeError:
                pass

        self.most_recent_context = context

        # Restore runtime_params to previous value
        if runtime_params:
            for param in runtime_params:
                try:
                    prev_val = getattr(self.parameters, param).get_previous(context)
                    self._set_parameter_value(param, prev_val, context)
                except AttributeError:
                    try:
                        prev_val = getattr(self.function.parameters, param).get_previous(context)
                        self.function._set_parameter_value(param, prev_val, context)
                    except:
                        pass

        return value

    def is_finished(self, context=None):
        """
            set by a Component to signal completion of its `execution <Component_Execution>` in a `TRIAL
            <TimeScale.TRIAL>`; used by `Component-based Conditions <Conditions_Component_Based>` to predicate the
            execution of one or more other Components on a Component.
        """
        return self.parameters.is_finished_flag._get(context)

    def _parse_param_port_sources(self):
        try:
            for param_port in self._parameter_ports:
                if param_port.source == FUNCTION:
                    param_port.source = self.function
        except AttributeError:
            pass

    def _increment_execution_count(self, count=1):
        self.parameters.execution_count.set(self.execution_count + count, override=True)
        return self.execution_count

    def _increment_num_executions(self, context, time_scales:(list, TimeScale), count=1):
        # get relevant Time object:
        time_scales = list(time_scales)
        assert [isinstance(i, TimeScale) for i in time_scales], \
            'non-TimeScale value provided in time_scales argument of _increment_num_executions'
        curr_num_execs = self.parameters.num_executions._get(context)
        for time_scale in time_scales:
            new_val = curr_num_execs._get_by_time_scale(time_scale) + count
            curr_num_execs._set_by_time_scale(time_scale, new_val)
        self.parameters.num_executions.set(curr_num_execs, override=True)
        return curr_num_execs

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
                return context.composition.scheduler.get_clock(context).time
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
        """Check that a flag is one and only one status flag
        """
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
                                     "or appropriately configuring a Composition to which it belongs".
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

    def set_delivery_conditions(self, items, delivery_condition=LogCondition.EXECUTION):
        """
        _set_delivery_conditions(          \
            items                    \
            delivery_condition=EXECUTION  \
        )

        Specifies items to be delivered to external application via gRPC; these must be be `loggable_items <Component.loggable_items>`
        of the Component's `log <Component.log>`. This is a convenience method that calls the `_set_delivery_conditions <Log._set_delivery_conditions>`
        method of the Component's `log <Component.log>`.
        """
        self.log._set_delivery_conditions(items=items, delivery_condition=delivery_condition)

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

    @property
    def _dict_summary(self):
        from psyneulink.core.compositions.composition import Composition
        from psyneulink.core.components.ports.port import Port
        from psyneulink.core.components.ports.outputport import OutputPort
        from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection

        def parse_parameter_value(value):
            if isinstance(value, (list, tuple)):
                new_item = []
                for item in value:
                    new_item.append(parse_parameter_value(item))
                try:
                    value = type(value)(new_item)
                except TypeError:
                    value = type(value)(*new_item)
            elif isinstance(value, dict):
                value = {
                    parse_parameter_value(k): parse_parameter_value(v)
                    for k, v in value.items()
                }
            elif isinstance(value, Composition):
                value = value.name
            elif isinstance(value, Port):
                if isinstance(value, OutputPort):
                    state_port_name = MODEL_SPEC_ID_OUTPUT_PORTS
                else:
                    state_port_name = MODEL_SPEC_ID_INPUT_PORTS

                # assume we will use the identifier on reconstitution
                value = '{0}.{1}.{2}'.format(
                    value.owner.name,
                    state_port_name,
                    value.name
                )
            elif isinstance(value, Component):
                # could potentially create duplicates when it should
                # create a reference to an already existent Component like
                # with Compositions, but in a vacuum the full specification
                # is necessary.
                # in fact this would happen unless the parser specifically
                # handles it like ours does
                value = value._dict_summary
            elif isinstance(value, (types.FunctionType)):
                value = base64.encodebytes(dill.dumps(value)).decode('utf-8')

            return value

        # attributes (and their values) included in top-level dict
        basic_attributes = ['name']

        # attributes that aren't Parameters but are psyneulink-specific
        # and are stored in the PNL parameters section
        implicit_parameter_attributes = ['node_ordering', 'required_node_roles']

        parameters_dict = {}
        pnl_specific_parameters = {}
        deferred_init_values = {}

        if self.initialization_status is ContextFlags.DEFERRED_INIT:
            deferred_init_values = copy.copy(self._init_args)
            try:
                deferred_init_values.update(deferred_init_values['params'])
            except (KeyError, TypeError):
                pass

            # .parameters still refers to class parameters during deferred init
            assert self.parameters._owner is not self

        for p in self.parameters:
            if (
                p.name not in self._model_spec_parameter_blacklist
                and not isinstance(p, ParameterAlias)
            ):
                if self.initialization_status is ContextFlags.DEFERRED_INIT:
                    try:
                        val = deferred_init_values[p.name]
                    except KeyError:
                        # class default
                        val = p.default_value
                else:
                    # special handling because MappingProjection matrix just
                    # refers to its function's matrix but its default values are
                    # PNL-specific
                    if (
                        isinstance(self, MappingProjection)
                        and p.name == 'matrix'
                    ):
                        val = self.function.defaults.matrix
                    elif p.spec is not None:
                        val = p.spec
                    else:
                        val = p.default_value

                val = parse_parameter_value(val)

                try:
                    matching_parameter_port = self.owner.parameter_ports[p.name]

                    if matching_parameter_port.source is self:
                        val = {
                            MODEL_SPEC_ID_PARAMETER_SOURCE: '{0}.{1}.{2}'.format(
                                self.owner.name,
                                MODEL_SPEC_ID_INPUT_PORTS,
                                p.name
                            ),
                            MODEL_SPEC_ID_PARAMETER_VALUE: val,
                            MODEL_SPEC_ID_TYPE: type(val)
                        }
                # ContentAddressableList uses TypeError when key not found
                except (AttributeError, TypeError):
                    pass

                # split parameters designated as PsyNeuLink-specific and
                # parameters that are universal
                if p.pnl_internal:
                    pnl_specific_parameters[p.name] = val
                else:
                    parameters_dict[p.name] = val

        for attr in implicit_parameter_attributes:
            try:
                pnl_specific_parameters[attr] = getattr(self, attr)
            except AttributeError:
                pass

        if len(pnl_specific_parameters) > 0:
            parameters_dict[MODEL_SPEC_ID_PSYNEULINK] = pnl_specific_parameters

        function_dict = {}
        try:
            if isinstance(self.function, Component):
                function_dict['functions'] = [self.function._dict_summary]
        except AttributeError:
            pass

        type_dict = {}

        if self._model_spec_class_name_is_generic:
            type_dict[MODEL_SPEC_ID_GENERIC] = self.__class__.__name__
        else:
            if self._model_spec_generic_type_name is not NotImplemented:
                type_dict[MODEL_SPEC_ID_GENERIC] = self._model_spec_generic_type_name
            else:
                type_dict[MODEL_SPEC_ID_GENERIC] = None

            type_dict[MODEL_SPEC_ID_PSYNEULINK] = self.__class__.__name__

        return {
            **{attr: getattr(self, attr) for attr in basic_attributes},
            **{self._model_spec_id_parameters: parameters_dict},
            **function_dict,
            **{MODEL_SPEC_ID_TYPE: type_dict}
        }

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
    def _variable_shape_flexibility(self):
        try:
            return self.__variable_shape_flexibility
        except AttributeError:
            self.__variable_shape_flexibility = DefaultsFlexibility.FLEXIBLE
            return self.__variable_shape_flexibility

    @_variable_shape_flexibility.setter
    def _variable_shape_flexibility(self, value):
        self.__variable_shape_flexibility = value

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

    @handle_external_context()
    def _update_parameter_components(self, context=None):
        # store all Components in Parameters to be used in
        # _dependent_components for _initialize_from_context
        for p in self.parameters:
            try:
                param_value = p._get(context)
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

    @property
    def _model_spec_parameter_blacklist(self):
        """
            A set of Parameter names that should not be added to the generated
            constructor string
        """
        return {'function', 'value'}


COMPONENT_BASE_CLASS = Component


def make_property_mod(param_name):

    def getter(self):
        try:
            return self._parameter_ports[param_name].value
        except TypeError:
            raise ComponentError("{} does not have a '{}' ParameterPort."
                                 .format(self.name, param_name))

    def setter(self, value):
        raise ComponentError("Cannot set to {}'s mod_{} directly because it is computed by the ParameterPort."
                             .format(self.name, param_name))

    prop = property(getter).setter(setter)

    return prop


def make_stateful_getter_mod(param_name):

    def getter(self, context=None):
        try:
            return self._parameter_ports[param_name].parameters.value.get(context)
        except TypeError:
            raise ComponentError("{} does not have a '{}' ParameterPort."
                                 .format(self.name, param_name))

    return getter
