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
"""

Functions that store and can retrieve a record of their current input.

* `Buffer`
* `ContentAddressableMemory`

Overview
--------

Functions that store and can return a record of their input.

"""

from collections import deque

import numpy as np
import typecheck as tc
import warnings

from psyneulink.core import llvm as pnlvm

from psyneulink.core.components.functions.function import \
    Function_Base, FunctionError, is_function_type, EPSILON
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import StatefulFunction
from psyneulink.core.components.functions.selectionfunctions import OneHot
from psyneulink.core.components.functions.objectivefunctions import Distance
from psyneulink.core.globals.keywords import \
    ADDITIVE_PARAM, BUFFER_FUNCTION, MEMORY_FUNCTION, COSINE, ContentAddressableMemory_FUNCTION, \
    MIN_INDICATOR, MULTIPLICATIVE_PARAM, NEWEST, NOISE, OLDEST, OVERWRITE, RATE, RANDOM
from psyneulink.core.globals.utilities import all_within_range, convert_to_np_array, parameter_spec, get_global_seed
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set

__all__ = ['MemoryFunction', 'Buffer', 'ContentAddressableMemory']


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
        a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
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
        value assigned as the first item of the deque when the Function is initialized, or reset
        if the **new_previous_value** argument is not specified in the call to `reset
        <StatefulFunction.reset>`.

    previous_value : 1d array : default class_defaults.variable
        state of the deque prior to appending `variable <Buffer.variable>` in the current call.

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a default is
        assigned by FunctionRegistry (see `Registry_Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see `Preferences`
        for details).
    """

    componentName = BUFFER_FUNCTION

    class Parameters(StatefulFunction.Parameters):
        """
            Attributes
            ----------

                history
                    see `history <Buffer.history>`

                    :default value: None
                    :type:

                initializer
                    see `initializer <Buffer.initializer>`

                    :default value: numpy.array([], dtype=float64)
                    :type: ``numpy.ndarray``

                noise
                    see `noise <Buffer.noise>`

                    :default value: 0.0
                    :type: ``float``

                rate
                    see `rate <Buffer.rate>`

                    :default value: 1.0
                    :type: ``float``
        """
        variable = Parameter([], pnl_internal=True, constructor_argument='default_variable')
        rate = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        noise = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        history = None
        initializer = Parameter(np.array([]), pnl_internal=True)


    @tc.typecheck
    def __init__(self,
                 # FIX: 12/11/18 JDC - NOT SAFE TO SPECIFY A MUTABLE TYPE AS DEFAULT
                 default_variable=None,
                 # KAM 6/26/18 changed default param values because constructing a plain buffer function ("Buffer())
                 # was failing.
                 # For now, updated default_variable, noise, and Alternatively, we can change validation on
                 # default_variable=None,   # Changed to [] because None conflicts with initializer
                 rate=None,
                 noise=None,
                 # rate: parameter_spec=1.0,
                 # noise: parameter_spec=0.0,
                 # rate: tc.optional(tc.optional(tc.any(int, float))) = None,         # Changed to 1.0 because None fails validation
                 # noise: tc.optional(tc.optional(tc.any(int, float, callable))) = None,    # Changed to 0.0 - None fails validation
                 # rate: tc.optional(tc.any(int, float, list, np.ndarray)) = 1.0,
                 # noise: tc.optional(tc.any(int, float, list, np.ndarray, callable)) = 0.0,
                 history: tc.optional(tc.optional(int)) = None,
                 initializer=None,
                 params: tc.optional(tc.optional(dict)) = None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

        super().__init__(
            default_variable=default_variable,
            rate=rate,
            initializer=initializer,
            noise=noise,
            history=history,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _initialize_previous_value(self, initializer, context=None):
        try:
            initializer = initializer.any() or []
        except AttributeError:
            initializer = initializer or []
        previous_value = deque(initializer, maxlen=self.parameters.history.get(context))

        self.parameters.previous_value.set(previous_value, context, override=True)

        return previous_value

    def _instantiate_attributes_before_function(self, function=None, context=None):
        self.parameters.previous_value._set(
            self._initialize_previous_value(
                self.parameters.initializer._get(context),
                context
            ),
            context
        )

    @handle_external_context(fallback_most_recent=True)
    def reset(self, previous_value=None, context=None):
        """

        Clears the `previous_value <Buffer.previous_value>` deque.

        If an argument is passed into reset or if the `initializer <Buffer.initializer>` attribute contains a
        value besides [], then that value is used to start the new `previous_value <Buffer.previous_value>` deque.
        Otherwise, the new `previous_value <Buffer.previous_value>` deque starts out empty.

        `value <Buffer.value>` takes on the same value as  `previous_value <Buffer.previous_value>`.

        """
        # no arguments were passed in -- use current values of initializer attributes
        if previous_value is None:
            previous_value = self._get_current_parameter_value("initializer", context)

        if previous_value is None or previous_value == []:
            self.parameters.previous_value._get(context).clear()
            value = deque([], maxlen=self.parameters.history.get(context))

        else:
            value = self._initialize_previous_value(previous_value, context=context)

        self.parameters.value.set(value, context, override=True)
        return value

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        variable : number, list or array : default class_defaults.variable
           a single value or array of values to be integrated.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        updated value of deque : deque

        """
        rate = np.array(self._get_current_parameter_value(RATE, context)).astype(float)

        # execute noise if it is a function
        noise = self._try_execute_param(self._get_current_parameter_value(NOISE, context), variable)

        # If this is an initialization run, leave deque empty (don't want to count it as an execution step);
        # Just return current input (for validation).
        if self.is_initializing:
            return variable

        previous_value = self.parameters.previous_value._get(context)

        # Apply rate and/or noise, if they are specified, to all stored items
        if len(previous_value):
            previous_value = convert_to_np_array(previous_value) * rate + noise

        previous_value = deque(previous_value, maxlen=self.parameters.history._get(context))

        previous_value.append(variable)

        self.parameters.previous_value._set(previous_value, context)
        return self.convert_output_type(previous_value)


RETRIEVAL_PROB = 'retrieval_prob'
STORAGE_PROB = 'storage_prob'
DISTANCE_FUNCTION = 'distance_function'
SELECTION_FUNCTION = 'selection_function'
KEYS = 0
VALS = 1


class ContentAddressableMemory(MemoryFunction):  # ---------------------------------------------------------------------
    """
    ContentAddressableMemory(                        \
        default_variable=None,                       \
        retrieval_prob=1.0                           \
        storage_prob=1.0                             \
        rate=None,                                   \
        noise=0.0,                                   \
        initializer=None,                            \
        distance_function=Distance(metric=COSINE),   \
        selection_function=OneHot(mode=MIN_VAL),     \
        equidistant_keys_select=RANDOM,              \
        duplicate_keys=False,                \
        max_entries=None,                            \
        params=None,                                 \
        owner=None,                                  \
        prefs=None,                                  \
        )

    .. _ContentAddressableMemory:

    Implement a configurable, dictionary-style storage and retrieval of key-value pairs, in which storage
    is determined by `storage_prob <ContentAddressableMemory.storage_prob>`, and retrieval of items is
    determined by `distance_function <ContentAddressableMemory.distance_function>`, `selection_function
    <ContentAddressableMemory.selection_function>`, and `retrieval_prob <ContentAddressableMemory.retrieval_prob>`.
    Keys and values may have different lengths, and values may vary in length from entry to entry, but all keys
    must be the same length. Duplicate keys can be allowed, disallowed, or overwritten using `duplicate_keys
    <ContentAddressableMemory.duplicate_keys>`), and how selection is made among duplicate keys or ones
    indistinguishable by the `distance_function <ContentAddressableMemory.distance_function>` can be specified
    using `equidistant_keys_select <ContentAddressableMemory.equidistant_keys_select>`.

    The class also provides methods for directly retrieving an entry (`get_memory
    <ContentAddressableMemory.get_memory>`), and adding (`add_to_memory <ContentAddressableMemory.add_to_memory>`)
    and deleting (`delete_from_memory <ContentAddressableMemory.delete_from_memory>`) one or more entries.

    .. _ContentAddressableMemory_Structure:

    Structure
    ---------

    An item is stored and retrieved as a 2d array containing a key-value pair ([[key][value]]).  A 3d array of such
    pairs can be used to initialize the contents of memory by providing it in the **initialzer** argument of the
    ContentAddressableMemory's constructor, or in a call to its `reset  <ContentAddressableMemory.reset>`
    method.  The current contents of the memory can be inspected using the `memory <ContentAddressableMemory.memory>`
    attribute, which returns a list containing the current entries, each as a 2 item list containing a key-value pair.

    .. _ContentAddressableMemory_Execution:

    Execution
    ---------

    When `function <ContentAddressableMemory.function>` is executed, it first retrieves the
    item in `memory <ContentAddressableMemory.memory>` with the key that most closely matches the key of the item
    (key-value pair) in the call, stores the latter in memory, and returns the retrieved item (key-value pair).
    If key of the pair in the call is an exact match of a key in memory and `duplicate_keys
    <ContentAddressableMemory.duplicate_keys>` is False, then the matching item is returned, but the
    pair in the call is not stored. These steps are described in more detail below:

    * First, with probability `retrieval_prob <ContentAddressableMemory.retrieval_prob>`, an entry is retrieved from
      `memory <ContentAddressableMemory.memory>` that has a key that is closest to the one in the call (first item of
      `variable <ContentAddressableMemory.variable>`), as determined by the `distance_function
      <ContentAddressableMemory.distance_function>` and `selection_function
      <ContentAddressableMemory.selection_function>`.  The `distance_function
      <ContentAddressableMemory.distance_function>` generates a list of distances of each key in memory from the
      one in the call;  the `selection_function <ContentAddressableMemory.selection_function>` then determines which
      to select ones for consideration.  If more than one entry from memory is identified, `equidistant_keys_select
      <ContentAddressableMemory.equidistant_keys_select>` is used to determine which to retrieve.  If no retrieval
      occurs, an appropriately shaped zero-valued array is assigned as the retrieved memory (and returned by the
      `function <ContentAddressableMemory.function>`.
    ..
    * After retrieval, the key-value pair in the call (`variable <ContentAddressableMemory.variable>`) is stored in
     `memory <ContentAddressableMemory.memory>` with probability `storage_prob <ContentAddressableMemory.storage_prob>`.
      If the key (`variable <ContentAddressableMemory.variable>`\\[0]) is identical to one already in `memory
      <ContentAddressableMemory.memory>` and `duplicate_keys <ContentAddressableMemory.duplicate_keys>`
      is set to False, storage is skipped; if it is set to *OVERWRITE*, the value of the key in memory is replaced
      with the one in the call.  If **rate** and/or **noise** arguments are specified in the
      construtor, it is applied to the key before storing, as follows:

    .. math::
        variable[1] * rate + noise

    If the number of entries exceeds `max_entries <ContentAddressableMemory.max_entries>, the first (oldest) item in
    memory is deleted.

    Arguments
    ---------

    default_variable : list or 2d array : default class_defaults.variable
        specifies a template for the key and value entries of the dictionary;  list must have two entries, each
        of which is a list or array;  first item is used as key, and second as value entry of dictionary.

    retrieval_prob : float in interval [0,1] : default 1.0
        specifies probability of retrieiving a key from `memory <ContentAddressableMemory.memory>`.

    storage_prob : float in interval [0,1] : default 1.0
        specifies probability of adding `variable <ContentAddressableMemory.variable>` to `memory
        <ContentAddressableMemory.memory>`.

    rate : float, list, or array : default 1.0
        specifies a value used to multiply key (first item of `variable <ContentAddressableMemory.variable>`) before
        storing in `memory <ContentAddressableMemory.memory>` (see `rate <ContentAddressableMemory.noise> for details).

    noise : float, list, array, or Function : default 0.0
        specifies a random value added to key (first item of `variable <ContentAddressableMemory.variable>`) before
        storing in `memory <ContentAddressableMemory.memory>` (see `noise <ContentAddressableMemory.noise> for details).

    initializer : 3d array or list : default None
        specifies an initial set of entries for `memory <ContentAddressableMemory.memory>`. It must be of the following
        form: [[[key],[value]], [[key],[value]], ...], such that each item in the outer dimension (axis 0)
        is a 2d array or list containing a key and a value pair for that entry. All of the keys must 1d arrays or
        lists of the same length.

    distance_function : Distance or function : default Distance(metric=COSINE)
        specifies the function used during retrieval to compare the first item in `variable
        <ContentAddressableMemory.variable>` with keys in `memory <ContentAddressableMemory.memory>`.

    selection_function : OneHot or function : default OneHot(mode=MIN_VAL)
        specifies the function used during retrieval to evaluate the distances returned by `distance_function
        <ContentAddressableMemory.distance_function>` and select the item to return.

    equidistant_keys_select:  RANDOM | OLDEST | NEWEST : default RANDOM
        specifies which item is chosen for retrieval if two or more keys have the same distance from the first item of
        `variable  <ContentAddressableMemory.variable>`.

    duplicate_keys : bool | OVERWRITE : default False
        specifies whether entries with duplicate keys are allowed in `memory <ContentAddressableMemory.memory>`
        (see `duplicate_keys <ContentAddressableMemory.duplicate_keys for additional details>`).

    max_entries : int : default None
        specifies the maximum number of entries allowed in `memory <ContentAddressableMemory.memory>`
        (see `max_entries <ContentAddressableMemory.max_entries for additional details>`).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
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
        1st item (variable[0] is the key used to retrieve an enrtry from `memory <ContentAddressableMemory.memory>`,
        and 2nd item (variable[1]) is the value of the entry, paired with key and added to the `memory
        <ContentAddressableMemory.memory>`.

    key_size : int
        length of keys in `memory <ContentAddressableMemory.memory>`.

    val_size : int
        length of values in `memory <ContentAddressableMemory.memory>`.

    retrieval_prob : float in interval [0,1]
        probability of retrieiving a value from `memory <ContentAddressableMemory.memory>`.

    storage_prob : float in interval [0,1]
        probability of adding `variable <ContentAddressableMemory.variable>` to `memory
        <ContentAddressableMemory.memory>`.

    rate : float or 1d array
        value applied multiplicatively to key (first item of `variable <ContentAddressableMemory.variable>`) before
        storing in `memory <ContentAddressableMemory.memory>` (see `rate <Stateful_Rate>` for additional details).

    noise : float, 1d array or Function
        value added to key (first item of `variable <ContentAddressableMemory.variable>`) before storing in
        `memory <ContentAddressableMemory.memory>` (see `noise <Stateful_Noise>` for additional details).

    initializer : 3d array
        initial set of entries for `memory <ContentAddressableMemory.memory>`; each is a 2d array with a key-value pair.

    memory : list
        list of key-value pairs containing entries in ContentAddressableMemory:
        [[[key 1], [value 1]], [[key 2], value 2]]...]

    distance_function : Distance or function : default Distance(metric=COSINE)
        function used during retrieval to compare the first item in `variable <ContentAddressableMemory.variable>`
        with keys in `memory <ContentAddressableMemory.memory>`.

    selection_function : OneHot or function : default OneHot(mode=MIN_VAL)
        function used during retrieval to evaluate the distances returned by `distance_function
        <ContentAddressableMemory.distance_function>` and select the item(s) to return.

    previous_value : 1d array
        state of the `memory <ContentAddressableMemory.memory>` prior to storing `variable
        <ContentAddressableMemory.variable>` in the current call.

    duplicate_keys : bool | OVERWRITE
        determines whether entries with duplicate keys are allowed in `memory <ContentAddressableMemory.memory>`.
        If True (the default), items with keys that are the same as ones in memory can be stored;  on retrieval, a
        single one is selected based on `equidistant_keys_select <ContentAddressableMemory.equidistant_keys_select>`.
        If False, then an attempt to store and item with a key that is already in `memory
        <ContentAddressableMemory.memory>` is ignored, and the entry already in memory with that key is retrieved.
        If a duplicate key is identified during retrieval (e.g., **duplicate_keys** is changed from True to
        False), a warning is issued and zeros are returned.  If *OVERWRITE*, then retrieval of a cue with an identical
        key causes the value at that entry to be overwritten with the new value.

    equidistant_keys_select:  RANDOM | OLDEST | NEWEST
        deterimines which entry is retrieved when duplicate keys are identified or are indistinguishable by the
        `distance_function <ContentAddressableMemory.distance_function>`.

    max_entries : int
        maximum number of entries allowed in `memory <ContentAddressableMemory.memory>`;  if storing a memory
        exceeds the number, the oldest memory is deleted.

    random_state : numpy.RandomState
        private pseudorandom number generator

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a default is
        assigned by FunctionRegistry (see `Registry_Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see `Preferences`
        for details).

    Returns
    -------

    value and key of entry that best matches first item of `variable <ContentAddressableMemory.variable>`  : 2d array
        if no retrieval occures, an appropriately shaped zero-valued array is returned.

    """

    componentName = ContentAddressableMemory_FUNCTION

    class Parameters(StatefulFunction.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <ContentAddressableMemory.variable>`

                    :default value: [[0], [0]]
                    :type: ``list``

                distance_function
                    see `distance_function <ContentAddressableMemory.distance_function>`

                    :default value: `Distance`(metric=cosine)
                    :type: `Function`

                duplicate_keys
                    see `duplicate_keys <ContentAddressableMemory.duplicate_keys>`

                    :default value: False
                    :type: ``bool``

                equidistant_keys_select
                    see `equidistant_keys_select <ContentAddressableMemory.equidistant_keys_select>`

                    :default value: `RANDOM`
                    :type: ``str``

                key_size
                    see `key_size <ContentAddressableMemory.key_size>`

                    :default value: 1
                    :type: ``int``

                max_entries
                    see `max_entries <ContentAddressableMemory.max_entries>`

                    :default value: 1000
                    :type: ``int``

                noise
                    see `noise <ContentAddressableMemory.noise>`

                    :default value: 0.0
                    :type: ``float``

                random_state
                    see `random_state <ContentAddressableMemory.random_state>`

                    :default value: None
                    :type: ``numpy.random.RandomState``

                rate
                    see `rate <ContentAddressableMemory.rate>`

                    :default value: 1.0
                    :type: ``float``

                retrieval_prob
                    see `retrieval_prob <ContentAddressableMemory.retrieval_prob>`

                    :default value: 1.0
                    :type: ``float``

                selection_function
                    see `selection_function <ContentAddressableMemory.selection_function>`

                    :default value: `OneHot`(mode=MIN_INDICATOR)
                    :type: `Function`

                storage_prob
                    see `storage_prob <ContentAddressableMemory.storage_prob>`

                    :default value: 1.0
                    :type: ``float``

                val_size
                    see `val_size <ContentAddressableMemory.val_size>`

                    :default value: 1
                    :type: ``int``
        """
        variable = Parameter([[0],[0]], pnl_internal=True, constructor_argument='default_variable')
        retrieval_prob = Parameter(1.0, modulable=True)
        storage_prob = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        key_size = Parameter(1, stateful=True)
        val_size = Parameter(1, stateful=True)
        duplicate_keys = Parameter(False)
        equidistant_keys_select = Parameter(RANDOM)
        rate = Parameter(1.0, modulable=True)
        noise = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        max_entries = Parameter(1000)
        random_state = Parameter(None, stateful=True, loggable=False)

        distance_function = Parameter(Distance(metric=COSINE), stateful=False, loggable=False)
        selection_function = Parameter(OneHot(mode=MIN_INDICATOR), stateful=False, loggable=False)


    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 retrieval_prob: tc.optional(tc.any(int, float))=None,
                 storage_prob: tc.optional(tc.any(int, float))=None,
                 noise: tc.optional(tc.any(int, float, list, np.ndarray, callable))=None,
                 rate: tc.optional(tc.any(int, float, list, np.ndarray))=None,
                 initializer=None,
                 distance_function:tc.optional(tc.any(Distance, is_function_type))=None,
                 selection_function:tc.optional(tc.any(OneHot, is_function_type))=None,
                 duplicate_keys:tc.optional(tc.any(bool, tc.enum(OVERWRITE)))=None,
                 equidistant_keys_select:tc.optional(tc.enum(RANDOM, OLDEST, NEWEST))=None,
                 max_entries=None,
                 seed=None,
                 params: tc.optional(tc.optional(tc.any(list, np.ndarray))) = None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

        if initializer is None:
            initializer = []

        if seed is None:
            seed = get_global_seed()
        random_state = np.random.RandomState([seed])

        self._memory = []

        super().__init__(
            default_variable=default_variable,
            retrieval_prob=retrieval_prob,
            storage_prob=storage_prob,
            initializer=initializer,
            duplicate_keys=duplicate_keys,
            equidistant_keys_select=equidistant_keys_select,
            rate=rate,
            noise=noise,
            max_entries=max_entries,
            random_state=random_state,
            params=params,
            owner=owner,
            prefs=prefs,
        )

        if self.previous_value.size != 0:
            self.parameters.key_size._set(len(self.previous_value[KEYS][0]), Context(execution_id=None))
            self.parameters.val_size._set(len(self.previous_value[VALS][0]), Context(execution_id=None))

    def _parse_distance_function_variable(self, variable):
        # actual used variable in execution (get_memory) checks distance
        # between key and key, not key and val as implied in _validate
        return convert_to_np_array([variable[KEYS], variable[KEYS]])

    def _parse_selection_function_variable(self, variable, context=None):
        # this should be replaced in the future with the variable
        # argument when function ordering (and so ordering of parsers)
        # is made explicit
        distance_result = self.distance_function.parameters.value._get(context)
        print(distance_result, self.distance_function.defaults.value)
        return np.asfarray([
            distance_result if i == 0 else np.zeros_like(distance_result)
            for i in range(self.defaults.max_entries)
        ])

    def _get_state_ids(self):
        return super()._get_state_ids() + ["ring_memory"]

    def _get_state_struct_type(self, ctx):
        # Construct a ring buffer
        max_entries = self.parameters.max_entries.get()
        key_type = ctx.convert_python_struct_to_llvm_ir(self.defaults.variable[0])
        keys_struct = pnlvm.ir.ArrayType(key_type, max_entries)
        val_type = ctx.convert_python_struct_to_llvm_ir(self.defaults.variable[1])
        vals_struct = pnlvm.ir.ArrayType(val_type, max_entries)
        ring_buffer_struct = pnlvm.ir.LiteralStructType((
            keys_struct, vals_struct, ctx.int32_ty, ctx.int32_ty))
        generic_struct = ctx.get_state_struct_type(super())
        return pnlvm.ir.LiteralStructType((*generic_struct,
                                           ring_buffer_struct))

    def _get_state_initializer(self, context):
        memory = self.parameters.previous_value._get(context)
        mem_init = pnlvm._tupleize([memory[0], memory[1], 0, 0])
        return (*super()._get_state_initializer(context), mem_init)

    def _gen_llvm_function_body(self, ctx, builder, params, state, arg_in, arg_out, *, tags:frozenset):
        # PRNG
        rand_struct = pnlvm.helpers.get_state_ptr(builder, self, state, "random_state")
        uniform_f = ctx.import_llvm_function("__pnl_builtin_mt_rand_double")

        # Ring buffer
        buffer_ptr = pnlvm.helpers.get_state_ptr(builder, self, state, "ring_memory")
        keys_ptr = builder.gep(buffer_ptr, [ctx.int32_ty(0), ctx.int32_ty(0)])
        vals_ptr = builder.gep(buffer_ptr, [ctx.int32_ty(0), ctx.int32_ty(1)])
        count_ptr = builder.gep(buffer_ptr, [ctx.int32_ty(0), ctx.int32_ty(2)])
        wr_ptr = builder.gep(buffer_ptr, [ctx.int32_ty(0), ctx.int32_ty(3)])

        # Input
        var_key_ptr = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(0)])
        var_val_ptr = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(1)])

        # Zero output
        builder.store(arg_out.type.pointee(None), arg_out)
        out_key_ptr = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(0)])
        out_val_ptr = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(1)])

        # Check retrieval probability
        retr_ptr = builder.alloca(pnlvm.ir.IntType(1))
        builder.store(retr_ptr.type.pointee(1), retr_ptr)
        retr_prob_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, RETRIEVAL_PROB)

        # Prob can be [x] if we are part of a mechanism
        retr_prob = pnlvm.helpers.load_extract_scalar_array_one(builder, retr_prob_ptr)
        retr_rand = builder.fcmp_ordered('<', retr_prob, retr_prob.type(1.0))

        max_entries = len(vals_ptr.type.pointee)
        entries = builder.load(count_ptr)
        entries = pnlvm.helpers.uint_min(builder, entries, max_entries)
        # The call to random function needs to be after check to match python
        with builder.if_then(retr_rand):
            rand_ptr = builder.alloca(ctx.float_ty)
            builder.call(uniform_f, [rand_struct, rand_ptr])
            rand = builder.load(rand_ptr)
            passed = builder.fcmp_ordered('<', rand, retr_prob)
            builder.store(passed, retr_ptr)

        # Retrieve
        retr = builder.load(retr_ptr)
        with builder.if_then(retr, likely=True):
            # Determine distances
            distance_f = ctx.import_llvm_function(self.distance_function)
            distance_params = pnlvm.helpers.get_param_ptr(builder, self, params, "distance_function")
            distance_state = pnlvm.helpers.get_state_ptr(builder, self, state, "distance_function")
            distance_arg_in = builder.alloca(distance_f.args[2].type.pointee)
            builder.store(builder.load(var_key_ptr),
                          builder.gep(distance_arg_in, [ctx.int32_ty(0),
                                                        ctx.int32_ty(0)]))
            selection_arg_in = builder.alloca(pnlvm.ir.ArrayType(distance_f.args[3].type.pointee, max_entries))
            with pnlvm.helpers.for_loop_zero_inc(builder, entries, "distance_loop") as (b, idx):
                compare_ptr = b.gep(keys_ptr, [ctx.int32_ty(0), idx])
                b.store(b.load(compare_ptr),
                        b.gep(distance_arg_in, [ctx.int32_ty(0), ctx.int32_ty(1)]))
                distance_arg_out = b.gep(selection_arg_in, [ctx.int32_ty(0), idx])
                b.call(distance_f, [distance_params, distance_state,
                                    distance_arg_in, distance_arg_out])

            selection_f = ctx.import_llvm_function(self.selection_function)
            selection_params = pnlvm.helpers.get_param_ptr(builder, self, params, "selection_function")
            selection_state = pnlvm.helpers.get_state_ptr(builder, self, state, "selection_function")
            selection_arg_out = builder.alloca(selection_f.args[3].type.pointee)
            builder.call(selection_f, [selection_params, selection_state,
                                       selection_arg_in, selection_arg_out])

            # Find the selected index
            selected_idx_ptr = builder.alloca(ctx.int32_ty)
            builder.store(ctx.int32_ty(0), selected_idx_ptr)
            with pnlvm.helpers.for_loop_zero_inc(builder, entries, "distance_loop") as (b,idx):
                selection_val = b.load(b.gep(selection_arg_out, [ctx.int32_ty(0), idx]))
                non_zero = b.fcmp_ordered('!=', selection_val, selection_val.type(0))
                with b.if_then(non_zero):
                    b.store(idx, selected_idx_ptr)

            selected_idx = builder.load(selected_idx_ptr)
            selected_key = builder.load(builder.gep(keys_ptr, [ctx.int32_ty(0),
                                                               selected_idx]))
            selected_val = builder.load(builder.gep(vals_ptr, [ctx.int32_ty(0),
                                                               selected_idx]))
            builder.store(selected_key, out_key_ptr)
            builder.store(selected_val, out_val_ptr)

        # Check storage probability
        store_ptr = builder.alloca(pnlvm.ir.IntType(1))
        builder.store(store_ptr.type.pointee(1), store_ptr)
        store_prob_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, STORAGE_PROB)

        # Prob can be [x] if we are part of a mechanism
        store_prob = pnlvm.helpers.load_extract_scalar_array_one(builder, store_prob_ptr)
        store_rand = builder.fcmp_ordered('<', store_prob, store_prob.type(1.0))

        # The call to random function needs to be behind jump to match python
        # code
        with builder.if_then(store_rand):
            rand_ptr = builder.alloca(ctx.float_ty)
            builder.call(uniform_f, [rand_struct, rand_ptr])
            rand = builder.load(rand_ptr)
            passed = builder.fcmp_ordered('<', rand, store_prob)
            builder.store(passed, store_ptr)

        # Store
        store = builder.load(store_ptr)
        with builder.if_then(store, likely=True):

            # Check if such key already exists
            is_new_key_ptr = builder.alloca(pnlvm.ir.IntType(1))
            builder.store(is_new_key_ptr.type.pointee(1), is_new_key_ptr)
            with pnlvm.helpers.for_loop_zero_inc(builder, entries, "distance_loop") as (b,idx):
                cmp_key_ptr = b.gep(keys_ptr, [ctx.int32_ty(0), idx])

                # Vector compare
                # TODO: move this to helpers
                key_differs_ptr = b.alloca(pnlvm.ir.IntType(1))
                b.store(key_differs_ptr.type.pointee(0), key_differs_ptr)
                with pnlvm.helpers.array_ptr_loop(b, cmp_key_ptr, "key_compare") as (b2, idx2):
                    var_key_element = b2.gep(var_key_ptr, [ctx.int32_ty(0), idx2])
                    cmp_key_element = b2.gep(cmp_key_ptr, [ctx.int32_ty(0), idx2])
                    element_differs = b.fcmp_unordered('!=',
                                                       b.load(var_key_element),
                                                       b.load(cmp_key_element))
                    key_differs = b2.load(key_differs_ptr)
                    key_differs = b2.or_(key_differs, element_differs)
                    b2.store(key_differs, key_differs_ptr)

                key_differs = b.load(key_differs_ptr)
                is_new_key = b.load(is_new_key_ptr)
                is_new_key = b.and_(is_new_key, key_differs)
                b.store(is_new_key, is_new_key_ptr)

            # Add new key + val if does not exist yet
            is_new_key = builder.load(is_new_key_ptr)
            with builder.if_then(is_new_key):
                write_idx = builder.load(wr_ptr)

                store_key_ptr = builder.gep(keys_ptr, [ctx.int32_ty(0), write_idx])
                store_val_ptr = builder.gep(vals_ptr, [ctx.int32_ty(0), write_idx])

                builder.store(builder.load(var_key_ptr), store_key_ptr)
                builder.store(builder.load(var_val_ptr), store_val_ptr)

                # Update counters
                write_idx = builder.add(write_idx, write_idx.type(1))
                write_idx = builder.urem(write_idx, write_idx.type(max_entries))
                builder.store(write_idx, wr_ptr)

                count = builder.load(count_ptr)
                count = builder.add(count, count.type(1))
                builder.store(count, count_ptr)

        return builder

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

    def _validate(self, context=None):
        """Validate distance_function, selection_function and memory store"""
        distance_function = self.distance_function
        test_var = [self.defaults.variable[KEYS], self.defaults.variable[VALS]]
        if isinstance(distance_function, type):
            distance_function = distance_function(default_variable=test_var)
            fct_msg = 'Function type'
        else:
            distance_function.defaults.variable = test_var
            distance_function._instantiate_value(context)
            fct_msg = 'Function'
        try:
            distance_result = distance_function(test_var, context=context)
            if not np.isscalar(distance_result):
                raise FunctionError("Value returned by {} specified for {} ({}) must return a scalar".
                                    format(repr(DISTANCE_FUNCTION), self.__name__.__class__, distance_result))
        except:
            raise FunctionError("{} specified for {} arg of {} ({}) "
                                "must accept a list with two 1d arrays or a 2d array as its argument".
                                format(fct_msg, repr(DISTANCE_FUNCTION), self.__class__,
                                       distance_function))

        # Default to full memory
        selection_function = self.selection_function
        test_var = np.asfarray([distance_result if i==0
                                else np.zeros_like(distance_result)
                                for i in range(self._get_current_parameter_value('max_entries', context))])
        if isinstance(selection_function, type):
            selection_function = selection_function(default_variable=test_var, context=context)
            fct_string = 'Function type'
        else:
            selection_function.defaults.variable = test_var
            selection_function._instantiate_value(context)
            fct_string = 'Function'
        try:
            result = np.asarray(selection_function(test_var, context=context))
        except e:
            raise FunctionError(f'{fct_string} specified for {repr(SELECTION_FUNCTION)} arg of {self.__class__} '
                                f'({selection_function}) must accept a 1d array as its argument')
        if result.shape != test_var.shape:
            raise FunctionError(f'Value returned by {repr(SELECTION_FUNCTION)} specified for {self.__class__} '
                                f'({result}) must return an array of the same length it receives')

    def _initialize_previous_value(self, initializer, context=None):
        """Ensure that initializer is appropriate for assignment as memory attribute and assign as previous_value

        - Validate, if initializer is specified, it is a 3d array
            (must be done here rather than in validate_params as it is needed to initialize previous_value
        - Insure that it has exactly 2 items in outer dimension (axis 0)
            and that all items in each of those two items are all arrays
        """
        # vals = [[k for k in initializer.keys()], [v for v in initializer.values()]]

        previous_value = np.ndarray(shape=(2, 0))
        if len(initializer) == 0:
            return previous_value
        else:
            # Set key_size and val_size if this is the first entry
            self.parameters.previous_value.set(previous_value, context, override=True)
            self.parameters.key_size.set(len(initializer[0][KEYS]), context)
            self.parameters.val_size.set(len(initializer[0][VALS]), context)
            for entry in initializer:
                if not self._store_memory(np.array(entry), context):
                    warnings.warn(f"Attempt to initialize memory of {self.__class__.__name__} with an entry ({entry}) "
                                  f"that has the same key as a previous one, while 'duplicate_keys'==False; "
                                  f"that entry has been skipped")
            return convert_to_np_array(self._memory)

    def _instantiate_attributes_before_function(self, function=None, context=None):
        self.parameters.previous_value._set(
            self._initialize_previous_value(
                self.parameters.initializer._get(context),
                context
            ),
            context
        )

        if isinstance(self.distance_function, type):
            self.distance_function = self.distance_function(context=context)

        if isinstance(self.selection_function, type):
            self.selection_function = self.selection_function(context=context)

    @handle_external_context(fallback_most_recent=True)
    def reset(self, previous_value=None, context=None):
        """
        reset(<new_dictionary> default={})

        Clears the memory in `previous_value <ContentAddressableMemory.previous_value>`.

        If an argument is passed into reset or if the `initializer <ContentAddressableMemory.initializer>`
        attribute contains a value besides [], then that value is used to start the new memory in `previous_value
        <ContentAddressableMemory.previous_value>`. Otherwise, the new `previous_value
        <ContentAddressableMemory.previous_value>` memory starts out empty.

        `value <ContentAddressableMemory.value>` takes on the same value as
        `previous_value <ContentAddressableMemory.previous_value>`.
        """
        # no arguments were passed in -- use current values of initializer attributes
        if previous_value is None:
            previous_value = self._get_current_parameter_value("initializer", context)

        if previous_value == []:
            self.parameters.previous_value._get(context).clear()
            value = np.ndarray(shape=(2, 0, len(self.defaults.variable[0])))

        else:
            value = self._initialize_previous_value(previous_value, context=context)

        self.parameters.value.set(value, context, override=True)
        return value

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """
        Return entry in `memory <ContentAddressableMemory.memory>` that key of which best matches first item of
        `variable <ContentAddressableMemory.variable>` (query key), then add `variable
        <ContentAddressableMemory.variable>` to `memory <ContentAddressableMemory.memory>` (see `above
        <ContentAddressableMemory_Execution>` for additional details).

        Arguments
        ---------

        variable : list or 2d array : default class_defaults.variable
           first item (variable[0]) is treated as the key for retrieval; second item (variable[1]), paired
           with key, is added to `memory <ContentAddressableMemory.memory>`.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        value of entry that best matches first item of `variable <ContentAddressableMemory.variable>`  : 1d array
        """

        key = variable[KEYS]
        # if len(variable)==2:
        val = variable[VALS]

        retrieval_prob = np.array(self._get_current_parameter_value(RETRIEVAL_PROB, context)).astype(float)
        storage_prob = np.array(self._get_current_parameter_value(STORAGE_PROB, context)).astype(float)

        # execute noise if it is a function
        noise = self._try_execute_param(self._get_current_parameter_value(NOISE, context), variable)

        # get random state
        random_state = self._get_current_parameter_value('random_state', context)

        # If this is an initialization run, leave memory empty (don't want to count it as an execution step),
        # and return current value (variable[1]) for validation.
        if self.is_initializing:
            return variable

        # Set key_size and val_size if this is the first entry
        if len(self.parameters.previous_value._get(context)[KEYS]) == 0:
            self.parameters.key_size._set(len(key), context)
            self.parameters.val_size._set(len(val), context)

        # Retrieve value from current dict with key that best matches key
        if retrieval_prob == 1.0 or (retrieval_prob > 0.0 and retrieval_prob > random_state.rand()):
            memory = self.get_memory(key, context)
        else:
            # QUESTION: SHOULD IT RETURN ZERO VECTOR OR NOT RETRIEVE AT ALL (LEAVING VALUE AND OutputPort FROM LAST TRIAL)?
            #           CURRENT PROBLEM WITH LATTER IS THAT IT CAUSES CRASH ON INIT, SINCE NOT OUTPUT_PORT
            #           SO, WOULD HAVE TO RETURN ZEROS ON INIT AND THEN SUPPRESS AFTERWARDS, AS MOCKED UP BELOW
            memory = [[0]* self.parameters.key_size._get(context), [0]* self.parameters.val_size._get(context)]
        # Store variable to dict:
        if noise:
            key += noise
        if storage_prob == 1.0 or (storage_prob > 0.0 and storage_prob > random_state.rand()):
            self._store_memory(variable, context)

        # Return 3d array with keys and vals as lists
        # IMPLEMENTATION NOTE:  if try to create np.ndarray directly, and keys and vals have same length
        #                       end up with array of arrays, rather than array of lists
        ret_val = convert_to_np_array([list(memory[0]),[]])
        ret_val[1] = list(memory[1])
        return ret_val

    @tc.typecheck
    def _validate_memory(self, memory:tc.any(list, np.ndarray), context):

        # memory must be list or 2d array with 2 items
        if len(memory) != 2 and not all(np.array(i).ndim == 1 for i in memory):
            raise FunctionError(f"Attempt to store memory in {self.__class__.__name__} ({memory}) "
                                f"that is not a 2d array with two items ([[key],[value]])")

        self._validate_key(memory[KEYS], context)

    @tc.typecheck
    def _validate_key(self, key:tc.any(list, np.ndarray), context):
        # Length of key must be same as that of existing entries (so it can be matched on retrieval)
        if len(key) != self.parameters.key_size._get(context):
            raise FunctionError(f"Length of 'key' ({key}) to store in {self.__class__.__name__} ({len(key)}) "
                                f"must be same as others in the dict ({self.parameters.key_size._get(context)})")

    @tc.typecheck
    @handle_external_context()
    def get_memory(self, query_key:tc.any(list, np.ndarray), context=None):
        """get_memory(query_key, context=None)

        Retrieve memory from `memory <ContentAddressableMemory.memory>` based on `distance_function
        <ContentAddressableMemory.distance_function>` and `selection_function
        <ContentAddressableMemory.selection_function>`.

        Arguments
        ---------
        query_key : list or 1d array
            must be same length as key(s) of any existing entries in `memory <ContentAddressableMemory.memory>`.

        Returns
        -------
        value and key for item retrieved : 2d array as list
            if no retrieval occurs, returns appropriately shaped zero-valued array.

        """
        # QUESTION: SHOULD IT RETURN ZERO VECTOR OR NOT RETRIEVE AT ALL (LEAVING VALUE AND OutputPort FROM LAST TRIAL)?
        #           ALSO, SHOULD PROBABILISTIC SUPPRESSION OF RETRIEVAL BE HANDLED HERE OR function (AS IT IS NOW).

        self._validate_key(query_key, context)
        _memory = self.parameters.previous_value._get(context)

        # if no memory, return the zero vector
        if len(_memory[KEYS]) == 0:
            zeros_key = [0] * self.key_size
            zeros_val = [0] * self.val_size
            return [zeros_key, zeros_val]

        # Get distances between query_key and all keys in memory
        distances = [self.distance_function([query_key, list(m)]) for m in _memory[KEYS]]

        # Get the best-match(es) in memory based on selection_function and return as non-zero value(s) in an array
        selection_array = self.selection_function(distances, context=context)
        indices_of_selected_items = np.flatnonzero(selection_array)

        # Single key identified
        if len(indices_of_selected_items)==1:
            index_of_selected_item = int(np.flatnonzero(selection_array))
        # More than one key identified
        else:
            selected_keys = _memory[KEYS]
            # Check for any duplicate keys in matches and, if they are not allowed, return zeros
            if (not self.duplicate_keys
                    and any(list(selected_keys[indices_of_selected_items[0]])==list(selected_keys[other])
                            for other in indices_of_selected_items[1:])):
                warnings.warn(f'More than one item matched key ({query_key}) in memory for {self.name} of '
                              f'{self.owner.name} even though {repr("duplicate_keys")} is False')
                return [[0]* self.parameters.key_size._get(context),
                        [0]* self.parameters.val_size._get(context)]
            if self.equidistant_keys_select == RANDOM:
                random_state = self._get_current_parameter_value('random_state', context)
                index_of_selected_item = random_state.choice(indices_of_selected_items)
            elif self.equidistant_keys_select == OLDEST:
                index_of_selected_item = indices_of_selected_items[0]
            elif self.equidistant_keys_select == NEWEST:
                index_of_selected_item = indices_of_selected_items[-1]
            else:
                assert False, f'PROGRAM ERROR:  bad specification ({self.equidistant_keys_select}) for  ' \
                    f'\'equidistant_keys_select parameter of {self.name} for {self.owner.name}'
        best_match_key = _memory[KEYS][index_of_selected_item]
        best_match_val = _memory[VALS][index_of_selected_item]

        # Return as list of lists
        return [list(best_match_key), list(best_match_val)]

    @tc.typecheck
    def _store_memory(self, memory:tc.any(list, np.ndarray), context):
        """Save an key-value pair to `memory <ContentAddressableMemory.memory>`

        Arguments
        ---------
        memory : list or 2d array
            must be two items, a key and a vaue, each of which must a list of numbers or 1d array;
            the key must be the same length as key(s) of any existing entries in `dict <ContentAddressableMemory.dict>`.
        """

        self._validate_memory(memory, context)

        key = list(memory[KEYS])
        val = list(memory[VALS])

        d = self.parameters.previous_value._get(context)

        matches = [k for k in d[KEYS] if key==list(k)]

        # If dupliciate keys are not allowed and key matches any existing keys, don't store
        if matches and self.duplicate_keys == False:
            storage_succeeded = False

        # If dupliciate_keys is specified as OVERWRITE, replace value for matching key:
        elif matches and self.duplicate_keys == OVERWRITE:
            if len(matches)>1:
                raise FunctionError(f"Attempt to store item ({memory}) in {self.name} "
                                    f"with 'duplicate_keys'='OVERWRITE' "
                                    f"when there is more than one matching key in its memory; "
                                    f"'duplicate_keys' may have previously been set to 'True'")
            try:
                index = d[KEYS].index(key)
            except AttributeError:
                index = d[KEYS].tolist().index(key)
            except ValueError:
                index = np.array(d[KEYS]).tolist().index(key)
            d[VALS][index] = val
            storage_succeeded = True

        else:
            # Append new key and value to their respective lists
            keys = list(d[KEYS])
            keys.append(key)
            values = list(d[VALS])
            values.append(val)

            # Return 3d array with keys and vals as lists
            d = [keys, values]
            storage_succeeded = True

        if len(d[KEYS]) > self.max_entries:
            d = np.delete(d, [KEYS], axis=1)

        self.parameters.previous_value._set(d,context)
        self._memory = d

        return storage_succeeded

    @tc.typecheck
    @handle_external_context()
    def add_to_memory(self, memories:tc.any(list, np.ndarray), context=None):
        """Add one or more key-value pairs into `memory <ContentAddressableMememory.memory>`

        Arguments
        ---------

        memories : list or array
            a single memory (list or 2d array) or list or array of memorys, each of which must be a valid entry
            consisting of two items (e.g., [[key],[value]] or [[[key1],[value1]],[[key2],[value2]]].
            The keys must all be the same length and equal to the length as key(s) of any existing entries in `dict
            <ContentAddressableMemory.dict>`.  Items are added to memory in the order listed.
        """
        memories = self._parse_memories(memories, 'add_to_memory', context)

        for memory in memories:
            self._store_memory(memory, context)

    @tc.typecheck
    @handle_external_context()
    def delete_from_memory(self, memories:tc.any(list, np.ndarray), key_only:bool= True, context=None):
        """Delete one or more key-value pairs from `memory <ContentAddressableMememory.memory>`

        Arguments
        ---------

        memories : list or array
            a single memory (list or 2d array) or list or array of memorys, each of which must be a valid entry
            consisting of two items (e.g., [[key],[value]] or [[[key1],[value1]],[[key2],[value2]]].

        key_only :  bool : default True
            if True, delete all memories with the same keys as those listed in **memories**;  if False,
            delete only memories that have the same key *and* value as those listed in **memories**.

        """
        memories = self._parse_memories(memories, 'add_to_memory', context)

        keys = [list(k) for k in memories[0]]
        vals = [list(k) for k in memories[0]]

        for i, key in enumerate(keys):
            for j, stored_key in enumerate(self._memory[KEYS]):
                if key == list(stored_key):
                    if key_only or vals[j] == list(self._memory[VALS][j]):
                        memory_keys = np.delete(self._memory[KEYS],j,axis=0)
                        memory_vals = np.delete(self._memory[VALS],j,axis=0)
                        self._memory = np.array([list(memory_keys), list(memory_vals)])
                        self.parameters.previous_value._set(self._memory, context)

    def _parse_memories(self, memories, method, context=None):
        """Parse passing of single vs. multiple memories, validate memories, and return ndarray"""
        memories = convert_to_np_array(memories)
        if not 1 <= memories.ndim <= 3:
            raise FunctionError(f"'memories' arg for {method} method of {self.__class__.__name__} "
                                f"must be a 2-item list or 2d array, or a list or 3d array containing those")

        if (memories.ndim == 2 and memories.dtype != object) or (memories.ndim == 1 and memories.dtype == object):
            memories = np.expand_dims(memories,axis=0)

        for memory in memories:
            self._validate_memory(memory, context)

        return memories

    @property
    def memory(self):
        try:
            # Return 3d array with keys and vals as lists
            # IMPLEMENTATION NOTE:  array is used for multi-line printout
            return np.array(list(zip(self._memory[KEYS],self._memory[VALS])))
        except:
            return np.array([])
