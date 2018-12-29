#
# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *******************************************  MEMORY FUNCTIONS ********************************************************
'''

Functions that store and can retrieve a record of their current input.

* `Buffer`
* `DND`

Overview
--------

Functions that store and can return a record of their input.

'''

from collections.__init__ import deque, OrderedDict

import numpy as np
import typecheck as tc

from psyneulink.core.components.functions.function import \
    Function_Base, FunctionError, is_function_type, MULTIPLICATIVE_PARAM, ADDITIVE_PARAM
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import StatefulFunction
from psyneulink.core.components.functions.selectionfunctions import OneHot
from psyneulink.core.components.functions.objectivefunctions import Distance
from psyneulink.core.globals.keywords import \
    BUFFER_FUNCTION, MEMORY_FUNCTION, COSINE, DND_FUNCTION, MIN_VAL, NOISE, RATE
from psyneulink.core.globals.utilities import all_within_range, parameter_spec
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set

__all__ = ['MemoryFunction', 'Buffer', 'DND']


class MemoryFunction(StatefulFunction):  # -----------------------------------------------------------------------------
    componentType = MEMORY_FUNCTION


class Buffer(MemoryFunction):  # ------------------------------------------------------------------------------
    """
    Buffer(                     \
        default_variable=None,  \
        rate=1.0,               \
        noise=0.0,              \
        history=None,           \
        initializer,            \
        params=None,            \
        owner=None,             \
        prefs=None,             \
        )

    .. _Buffer:

    Append `variable <Buffer.variable>` to the end of `previous_value <Buffer.previous_value>` (i.e., right-append
    to deque of previous inputs).

    .. note::
       Every appended item must have same shape as the first.

    If specified, `rate <Buffer.rate>` and/or `noise <Buffer.noise>` are applied to items already stored in the
    array, as follows:

    .. math::
        stored\\_item * rate + noise

    .. note::
       Because **rate** and **noise** are applied on every call, their effects accumulative exponentially over calls
       to `function <Buffer.function>`.

    If the length of the result exceeds `history <Buffer.history>`, delete the first item.
    Return `previous_value <Buffer.previous_value>` appended with `variable <Buffer.variable>`.

    Arguments
    ---------

    default_variable : number, list or array : default class_defaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float, list or 1d array : default 1.0
        specifies a value applied multiplicatively to each item already stored in the deque on each call to `function
        <Buffer.function>`;  must be in interval [0,1]

    noise : float or Function : default 0.0
        specifies a random value added to each item already in the deque on each call to `function <Buffer.function>`
        (see `noise <Buffer.noise>` for details).

    history : int : default None
        specifies the maxlen of the deque, and hence `value <Buffer.value>`.

    initializer float, list or ndarray : default []
        specifies a starting value for the deque;  if none is specified, the deque is initialized with an empty list.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
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

    variable : number or array
        current input value appended to the end of the deque.

    rate : float or 1d array with all elements in interval [0,1]
        multiplicatively applied to each item already in the deque on call to `function <Buffer.function>`;
        implements exponential decay of stored items.

    noise : float or Function
        random value added to each item of the deque in each call to `function <Buffer.function>`
        (see `noise <Stateful_Noise>` for additional details).

    history : int
        determines maxlen of the deque and the value returned by the `function <Buffer.function>`. If appending
        `variable <Buffer.variable>` to `previous_value <Buffer.previous_value>` exceeds history, the first item of
        `previous_value <Buffer.previous_value>` is deleted, and `variable <Buffer.variable>` is appended to it,
        so that `value <Buffer.previous_value>` maintains a constant length.  If history is not specified,
        the value returned continues to be extended indefinitely.

    initializer : float, list or ndarray
        value assigned as the first item of the deque when the Function is initialized, or reinitialized
        if the **new_previous_value** argument is not specified in the call to `reinitialize
        <StatefulFUnction.reinitialize>`.

    previous_value : 1d array : default class_defaults.variable
        state of the deque prior to appending `variable <Buffer.variable>` in the current call.

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = BUFFER_FUNCTION

    class Parameters(StatefulFunction.Parameters):
        rate = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        noise = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        history = None

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        NOISE: None,
        RATE: None
    })

    multiplicative_param = RATE
    # no additive_param?

    @tc.typecheck
    def __init__(self,
                 # FIX: 12/11/18 JDC - NOT SAFE TO SPECIFY A MUTABLE TYPE AS DEFAULT
                 default_variable=None,
                 # KAM 6/26/18 changed default param values because constructing a plain buffer function ("Buffer())
                 # was failing.
                 # For now, updated default_variable, noise, and Alternatively, we can change validation on
                 # default_variable=None,   # Changed to [] because None conflicts with initializer
                 rate=1.0,
                 noise=0.0,
                 # rate: parameter_spec=1.0,
                 # noise: parameter_spec=0.0,
                 # rate: tc.optional(tc.any(int, float)) = None,         # Changed to 1.0 because None fails validation
                 # noise: tc.optional(tc.any(int, float, callable)) = None,    # Changed to 0.0 - None fails validation
                 # rate: tc.optional(tc.any(int, float, list, np.ndarray)) = 1.0,
                 # noise: tc.optional(tc.any(int, float, list, np.ndarray, callable)) = 0.0,
                 history: tc.optional(int) = None,
                 initializer=[],
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None):

        if default_variable is None:
            default_variable = []

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  initializer=initializer,
                                                  noise=noise,
                                                  history=history,
                                                  params=params)

        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)

        self.has_initializers = True

    def _initialize_previous_value(self, initializer, execution_context=None):
        initializer = initializer or []
        previous_value = deque(initializer, maxlen=self.history)

        self.parameters.previous_value.set(previous_value, execution_context, override=True)

        return previous_value

    def _instantiate_attributes_before_function(self, function=None, context=None):

        self.has_initializers = True

    def reinitialize(self, *args, execution_context=None):
        """

        Clears the `previous_value <Buffer.previous_value>` deque.

        If an argument is passed into reinitialize or if the `initializer <Buffer.initializer>` attribute contains a
        value besides [], then that value is used to start the new `previous_value <Buffer.previous_value>` deque.
        Otherwise, the new `previous_value <Buffer.previous_value>` deque starts out empty.

        `value <Buffer.value>` takes on the same value as  `previous_value <Buffer.previous_value>`.

        """

        # no arguments were passed in -- use current values of initializer attributes
        if len(args) == 0 or args is None:
            reinitialization_value = self.get_current_function_param("initializer", execution_context)

        elif len(args) == 1:
            reinitialization_value = args[0]

        # arguments were passed in, but there was a mistake in their specification -- raise error!
        else:
            raise FunctionError("Invalid arguments ({}) specified for {}. Either one value must be passed to "
                                "reinitialize its stateful attribute (previous_value), or reinitialize must be called "
                                "without any arguments, in which case the current initializer value, will be used to "
                                "reinitialize previous_value".format(args,
                                                                     self.name))

        if reinitialization_value is None or reinitialization_value == []:
            self.get_previous_value(execution_context).clear()
            value = deque([], maxlen=self.history)

        else:
            value = self._initialize_previous_value(reinitialization_value, execution_context=execution_context)

        self.parameters.value.set(value, execution_context, override=True)
        return value

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """

        Arguments
        ---------

        variable : number, list or array : default class_defaults.variable
           a single value or array of values to be integrated.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        updated value of deque : deque

        """

        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        rate = np.array(self.get_current_function_param(RATE, execution_id)).astype(float)

        # execute noise if it is a function
        noise = self._try_execute_param(self.get_current_function_param(NOISE, execution_id), variable)

        # If this is an initialization run, leave deque empty (don't want to count it as an execution step);
        # Just return current input (for validation).
        if self.parameters.context.get(execution_id).initialization_status == ContextFlags.INITIALIZING:
            return variable

        previous_value = np.array(self.get_previous_value(execution_id))

        # Apply rate and/or noise, if they are specified, to all stored items
        if len(previous_value):
            if any(np.atleast_1d(rate) != 1.0):
                previous_value = previous_value * rate
            if any(np.atleast_1d(noise) != 0.0):
                previous_value = previous_value + noise

        previous_value = deque(previous_value, maxlen=self.history)

        previous_value.append(variable)

        self.parameters.previous_value.set(previous_value, execution_id)
        return self.convert_output_type(previous_value)


RETRIEVAL_PROB = 'retrieval_prob'
STORAGE_PROB = 'storage_prob'
DISTANCE_FUNCTION = 'distance_function'
SELECTION_FUNCTION = 'selection_function'


class DND(MemoryFunction):  # ------------------------------------------------------------------------------
    """
    DND(                                             \
        default_variable=None,                       \
        rate=None,                                   \
        noise=0.0,                                   \
        initializer=None,                            \
        distance_function=Distance(metric=COSINE),   \
        selection_function=OneHot(mode=MIN_VAL),     \
        max_entries=None,                            \
        params=None,                                 \
        owner=None,                                  \
        prefs=None,                                  \
        )

    Implements simple form of Differential Neural Dictionary described in `Ritter et al.
    <http://arxiv.org/abs/1805.09692>`_

    Based on implementation in `dlstm <https://github.com/qihongl/dlstm-demo>`_ by
    `Qihong Lu <https://github.com/qihongl>`_ (see also  `Kaiser et al. <http://arxiv.org/abs/1703.03129>`_
    and `Pritzel et al. <http://arxiv.org/abs/1703.01988>`_).

    .. _DND:

    * First, with probability `retrieval_prob <DND.retrieval_prob>`, retrieve vector from `dict <DND.dict>` using
      first item of `variable <DND.variable>` as key, and the matching algorithm specified by `distance_function
      <DND.distance_function>` and `selection_function <DND.selection_function>`; if no retrieval occures,
      an appropriately shaped zero-valued array is returned.
    ..
    * Then, with probability `storage_prob <DND.storage_prob>` add new entry using the first item of `variable
      <DND.variable>` as the key and its second item as the value. If specified, the values of the **rate** and
      **noise** arguments are applied to the key before storing:

    .. math::
        variable[1] * rate + noise

    .. note::
       * Keys in `dict <DND.dict>` are stored as tuples (since lists and arrays are not hashable);
         they are converted to arrays for evaluation during retrieval.
       ..
       * All keys must be the same length (for comparision during retrieval).

    Arguments
    ---------

    default_variable : list or 2d array : default class_defaults.variable
        specifies a template for the key and value entries of the dictionary;  list must have two entries, each
        of which is a list or array;  first item is used as key, and second as value entry of dictionary.

    retrieval_prob : float in interval [0,1] : default 1.0
        specifies probability of retrieiving a value from `dict <DND.dict>`.

    storage_prob : float in interval [0,1] : default 1.0
        specifies probability of adding `variable <DND.variable>` to `dict <DND.dict>`.

    rate : float, list, or array : default 1.0
        specifies a value used to multiply key (first item of `variable <DND.variable>`) before storing in `dict
        <DND.dict>` (see `rate <DND.noise> for details).

    noise : float, list, array, or Function : default 0.0
        specifies a random value added to key (first item of `variable <DND.variable>`) before storing in `dict
        <DND.dict>` (see `noise <DND.noise> for details).

    initializer dict : default {}
        specifies an initial set of entries for `dict <DND.dict>`;  each key must have the same shape as
        the first item of `variable <DND.variable>` and each value must have the same shape as its second item.

    distance_function : Distance or function : default Distance(metric=COSINE)
        specifies the function used during retrieval to compare the first item in `variable <DND.variable>`
        with keys in `dict <DND.dict>`.

    selection_function : OneHot or function : default OneHot(mode=MIN_VAL)
        specifies the function used during retrieval to evaluate the distances returned by `distance_function
        <DND.distance_function>` and select the item to return.

    max_entries : int : default None
        specifies the maximum number of entries allowed in `dict <DND.dict>` (see `max_entries <DND.max_entries for
        additional details>`).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
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

    variable : 2d array
        1st item (variable[0] is the key used to retrieve an enrtry from `dict <DND.dict>`, and 2nd item
        (variable[1]) is the value of the entry, paired with key and added to the `dict <DND.dict>`.

    key_size : int
        length of keys in `dict <DND.dict>`.

    retrieval_prob : float in interval [0,1]
        probability of retrieiving a value from `dict <DND.dict>`.

    storage_prob : float in interval [0,1]
        probability of adding `variable <DND.variable>` to `dict <DND.dict>`.

    rate : float or 1d array
        value applied multiplicatively to key (first item of `variable <DND.variable>`) before storing in `dict
        <DND.dict>` (see `rate <Stateful_Rate>` for additional details).

    noise : float, 1d array or Function
        value added to key (first item of `variable <DND.variable>`) before storing in `dict <DND.dict>`
        (see `noise <Stateful_Noise>` for additional details).

    initializer dict : default {}
        initial set of entries for `dict <DND.dict>`.

    dict : dict
        dictionary with current set of entries maintained by DND.

    distance_function : Distance or function : default Distance(metric=COSINE)
        function used during retrieval to compare the first item in `variable <DND.variable>`
        with keys in `dict <DND.dict>`.

    selection_function : OneHot or function : default OneHot(mode=MIN_VAL)
        function used during retrieval to evaluate the distances returned by `distance_function
        <DND.distance_function>` and select the item to return.

    previous_value : 1d array
        state of the `dict <DND.dict>` prior to storing `variable <DND.variable>` in the current call.

    max_entries : int
        maximum number of entries allowed in `dict <DND.dict>`;  if an attempt is made to add an additional entry
        an error is generated.

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    Returns
    -------

    value and key of entry that best matches first item of `variable <DND.variable>`  : 2d array
        if no retrieval occures, an appropriately shaped zero-valued array is returned.

    """

    componentName = DND_FUNCTION

    class Parameters(StatefulFunction.Parameters):
        variable = Parameter([[0],[0]])
        retrieval_prob = Parameter(1.0, modulable=True)
        storage_prob = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        key_size = Parameter(1, stateful=True)
        rate = Parameter(1.0, modulable=True)
        noise = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        max_entries = Parameter(1000)

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        NOISE: None,
        RATE: None,
        RETRIEVAL_PROB: 1.0,
        STORAGE_PROB: 1.0
    })

    multiplicative_param = RETRIEVAL_PROB
    # no additive_param?

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 retrieval_prob: tc.optional(tc.any(int, float))=1.0,
                 storage_prob: tc.optional(tc.any(int, float))=1.0,
                 noise: tc.optional(tc.any(int, float, list, np.ndarray, callable))=0.0,
                 rate: tc.optional(tc.any(int, float, list, np.ndarray))=1.0,
                 initializer: tc.optional(dict)=None,
                 distance_function:tc.any(Distance, is_function_type)=Distance(metric=COSINE),
                 selection_function:tc.any(OneHot, is_function_type)=OneHot(mode=MIN_VAL),
                 max_entries=1000,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None):

        initializer = initializer or []
        self.distance_function = distance_function
        self.selection_function = selection_function

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(retrieval_prob=retrieval_prob,
                                                  storage_prob=storage_prob,
                                                  initializer=initializer,
                                                  rate=rate,
                                                  noise=noise,
                                                  max_entries=max_entries,
                                                  params=params)

        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)

        self.has_initializers = True

    def _validate_params(self, request_set, target_set=None, context=None):
        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        if RETRIEVAL_PROB in request_set and request_set[RETRIEVAL_PROB] is not None:
            retrieval_prob = request_set[RETRIEVAL_PROB]
            if not all_within_range(retrieval_prob, 0, 1):
                raise FunctionError("{} arg of {} ({}) must be a float in the interval [0,1]".
                                    format(repr(RETRIEVAL_PROB), self.__class___.__name__, retrieval_prob))

        if STORAGE_PROB in request_set and request_set[STORAGE_PROB] is not None:
            storage_prob = request_set[STORAGE_PROB]
            if not all_within_range(storage_prob, 0, 1):
                raise FunctionError("{} arg of {} ({}) must be a float in the interval [0,1]".
                                    format(repr(STORAGE_PROB), self.__class___.__name__, storage_prob))

    def _validate(self):
        distance_function = self.distance_function
        if isinstance(distance_function, type):
            distance_function = distance_function()
            fct_msg = 'Function type'
        else:
            fct_msg = 'Function'
        try:
            test = [[0,0],[0,0]]
            result = distance_function(test)
            if not np.isscalar(result):
                raise FunctionError("Value returned by {} specified for {} ({}) must return a scalar".
                                    format(repr(DISTANCE_FUNCTION), self.__name__.__class__, result))
        except:
            raise FunctionError("{} specified for {} arg of {} ({}) "
                                "must accept a list with two 1d arrays or a 2d array as its argument".
                                format(fct_msg, repr(DISTANCE_FUNCTION), self.__name__.__class__,
                                       distance_function))

        selection_function = self.selection_function
        if isinstance(selection_function, type):
            selection_function = selection_function()
            fct_msg = 'Function type'
        else:
            fct_msg = 'Function'
        try:
            test = np.array([0,1,2,3])
            result = np.array(selection_function(test))
            if result.shape != test.shape or len(np.flatnonzero(result))>1:
                raise FunctionError("Value returned by {} specified for {} ({}) "
                                    "must return an array of the same length it receives with one nonzero value".
                                    format(repr(SELECTION_FUNCTION), self.__name__.__class__, result))
        except:
            raise FunctionError("{} specified for {} arg of {} ({}) must accept a 1d array "
                                "must accept a list with two 1d arrays or a 2d array as its argument".
                                format(fct_msg, repr(SELECTION_FUNCTION), self.__name__.__class__,
                                       selection_function))

    def _initialize_previous_value(self, initializer, execution_context=None):
        initializer = initializer or []
        previous_value = OrderedDict(initializer)

        self.parameters.previous_value.set(previous_value, execution_context, override=True)

        return previous_value

    def _instantiate_attributes_before_function(self, function=None, context=None):

        self.has_initializers = True

        if isinstance(self.distance_function, type):
            self.distance_function = self.distance_function()

        if isinstance(self.selection_function, type):
            self.selection_function = self.selection_function()

    def reinitialize(self, *args, execution_context=None):
        """
        reinitialize(<new_dictionary> default={})

        Clears the dict in `previous_value <DND.previous_value>`.

        If an argument is passed into reinitialize or if the `initializer <DND.initializer>` attribute contains a
        value besides [], then that value is used to start the new dict in `previous_value <DND.previous_value>`.
        Otherwise, the new `previous_value <DND.previous_value>` dict starts out empty.

        `value <DND.value>` takes on the same value as  `previous_value <DND.previous_value>`.
        """

        # no arguments were passed in -- use current values of initializer attributes
        if len(args) == 0 or args is None:
            reinitialization_value = self.get_current_function_param("initializer", execution_context)

        elif len(args) == 1:
            reinitialization_value = args[0]

        # arguments were passed in, but there was a mistake in their specification -- raise error!
        else:
            raise FunctionError("Invalid arguments ({}) specified for {}. Either one value must be passed to "
                                "reinitialize its stateful attribute (previous_value), or reinitialize must be called "
                                "without any arguments, in which case the current initializer value will be used to "
                                "reinitialize previous_value".format(args, self.name))

        if reinitialization_value is None or reinitialization_value == []:
            self.get_previous_value(execution_context).clear()
            value = dict()

        else:
            value = self._initialize_previous_value(reinitialization_value, execution_context=execution_context)

        self.parameters.value.set(value, execution_context, override=True)
        return value

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """
        Return value of entry in `dict <DND.dict>` best matched by first item of `variable <DND.variable>`, then add
        `variable <DND.variable>` to `dict <DND.dict>`.

        If the length of `dict <DND.dict>` exceeds `max_entries <DND.max_entries>`, generate an error.

        Arguments
        ---------

        variable : list or 2d array : default class_defaults.variable
           first item (variable[0]) is treated as the key for retrieval; second item (variable[1]), paired
           with key, is added to `dict <DND.dict>`.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        value of entry that best matches first item of `variable <DND.variable>`  : 1d array
        """

        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)
        key = variable[0]
        value = variable[1]

        retrieval_prob = np.array(self.get_current_function_param(RETRIEVAL_PROB, execution_id)).astype(float)
        storage_prob = np.array(self.get_current_function_param(STORAGE_PROB, execution_id)).astype(float)

        # execute noise if it is a function
        noise = self._try_execute_param(self.get_current_function_param(NOISE, execution_id), variable)

        # If this is an initialization run, leave dict empty (don't want to count it as an execution step),
        # and return current value (variable[1]) for validation.
        if self.parameters.context.get(execution_id).initialization_status == ContextFlags.INITIALIZING:
            return variable

        # Set key_size if this is the first entry
        if not self.get_previous_value(execution_id):
            self.parameters.key_size.set(len(key), execution_id)

        # Retrieve value from current dict with key that best matches key
        if retrieval_prob == 1.0 or (retrieval_prob > 0.0 and retrieval_prob > np.random.rand()):
            ret_val = self.get_memory(key, execution_id)
        else:
            # QUESTION: SHOULD IT RETURN ZERO VECTOR OR NOT RETRIEVE AT ALL (LEAVING VALUE AND OUTPUTSTATE FROM LAST TRIAL)?
            #           CURRENT PROBLEM WITH LATTER IS THAT IT CAUSES CRASH ON INIT, SINCE NOT OUTPUT_STATE
            #           SO, WOULD HAVE TO RETURN ZEROS ON INIT AND THEN SUPPRESS AFTERWARDS, AS MOCKED UP BELOW
            ret_val = np.zeros_like(self.defaults.variable)
            # if self.context.initialization_status == ContextFlags.INITIALIZING:
            #     ret_val = np.zeros_like(self.defaults.variable)
            # else:
            #     ret_val = None

        # Store variable to dict:
        if noise:
            key += noise
        if storage_prob == 1.0 or (storage_prob > 0.0 and storage_prob > np.random.rand()):
            self._store_memory(variable, execution_id)

        return self.convert_output_type(ret_val)

    @tc.typecheck
    def _validate_memory(self, memory:tc.any(list, np.ndarray), execution_id):

        # memory must be list or 2d array with 2 items
        if len(memory) != 2 and not all(np.array(i).ndim == 1 for i in memory):
            raise FunctionError("Attempt to store memory in {} ({}) that does not have 2 1d arrays".
                                format(self.__class__.__name__, memory))

        self._validate_key(memory[0], execution_id)

    @tc.typecheck
    def _validate_key(self, key:tc.any(list, np.ndarray), execution_id):
        # Length of key must be same as that of existing entries (so it can be matched on retrieval)
        if len(key) != self.parameters.key_size.get(execution_id):
            raise FunctionError("Length of {} to store in {} must be same as others in the dict ({})".
                                format(repr('key'), self.__class__.__name__, self._key_size))

    @tc.typecheck
    def get_memory(self, query_key:tc.any(list, np.ndarray), execution_id):
        """get_memory(query_key, execution_id=None)

        Retrieve memory from `dict <DND.dict>` based on `distance_function <DND.distance_function>` and
        `selection_function <DND.selection_function>`.

        Arguments
        ---------
        query_key : list or 1d array
            must be same length as key(s) of any existing entries in `dict <DND.dict>`.

        Returns
        -------
        value and key for item retrieved : 2d array
            if no retrieval occurs, returns appropriately shaped zero-valued array.

        """
        # QUESTION: SHOULD IT RETURN ZERO VECTOR OR NOT RETRIEVE AT ALL (LEAVING VALUE AND OUTPUTSTATE FROM LAST TRIAL)?
        #           ALSO, SHOULD PROBABILISTIC SUPPRESSION OF RETRIEVAL BE HANDLED HERE OR function (AS IT IS NOW).

        self._validate_key(query_key, execution_id)
        # if no memory, return the zero vector
        # if len(self.dict) == 0 or self.retrieval_prob == 0.0:
        if len(self.dict) == 0:
            return np.zeros_like(self.defaults.variable)
        # compute similarity(query_key, memory m ) for all m
        memory_dict = self.get_previous_value(execution_id)
        distances = [self.distance_function([query_key, list(m)]) for m in memory_dict.keys()]
        # get the best-match memory (one with the only non-zero value in the array)
        selection_array = self.selection_function(distances)
        index_of_selected_item = int(np.flatnonzero(selection_array))
        best_match_key = list(memory_dict.keys())[index_of_selected_item]
        best_match_val = list(memory_dict.values())[index_of_selected_item]

        return [best_match_val, best_match_key]

    @tc.typecheck
    def _store_memory(self, memory:tc.any(list, np.ndarray), execution_id):
        """Save an key-value pair to `dict <DND.dict>`

        Arguments
        ---------
        memory : list or 2d array
            must be two items, a key and a vaue, each of which must a list of numbers or 1d array;
            the key must be the same length as key(s) of any existing entries in `dict <DND.dict>`.
        """

        self._validate_memory(memory, execution_id)

        key = memory[0]
        val = memory[1]

        d = self.get_previous_value(execution_id) or {}

        if len(d) > self.max_entries:
            d.pop(list(d.keys())[len(d)-1])

        d.update({tuple(key):val})
        self.parameters.previous_value.set(d,execution_id)

    @tc.typecheck
    def insert_memories(self, memories:tc.any(list, np.ndarray), execution_id=None):
        """insert_memories(memories, execution_id=None)

        add key-value pairs to `dict <DND.dict>`.

        Arguments
        ---------
        memories : list or array
            a list or array of 2d arrays, each of which must be a valid "memory" consisting of two items,
            a key and a value, each of which is a list of numbers or 1d array;  the keys must all be the same
            length and equal to the length as key(s) of any existing entries in `dict <DND.dict>`.
        """
        memories = np.array(memories)
        if not 2 <= memories.ndim <= 3:
            raise FunctionError("{} arg for {} method of {} must be a list or ndarray made up of 2d arrays".
                                format(repr('memories'), repr('insert_memories'), self.__class__.__name ))
        for memory in memories:
            self.store_memory(memory[0], memory[1], execution_id)

    @property
    def dict(self):
        return self.previous_value
