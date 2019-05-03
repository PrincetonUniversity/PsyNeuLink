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
from random import choice

import numpy as np
import typecheck as tc
import warnings

from psyneulink.core import llvm as pnlvm

from psyneulink.core.components.functions.function import \
    Function_Base, FunctionError, is_function_type, MULTIPLICATIVE_PARAM, ADDITIVE_PARAM
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import StatefulFunction
from psyneulink.core.components.functions.selectionfunctions import OneHot
from psyneulink.core.components.functions.objectivefunctions import Distance
from psyneulink.core.globals.keywords import \
    BUFFER_FUNCTION, MEMORY_FUNCTION, COSINE, DND_FUNCTION, MIN_INDICATOR, NOISE, RATE, RANDOM, OLDEST, NEWEST
from psyneulink.core.globals.utilities import all_within_range, parameter_spec, get_global_seed
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
        """
            Attributes
            ----------

                history
                    see `history <Buffer.history>`

                    :default value: None
                    :type:

                noise
                    see `noise <Buffer.noise>`

                    :default value: 0.0
                    :type: float

                rate
                    see `rate <Buffer.rate>`

                    :default value: 1.0
                    :type: float

        """
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

        # Assign args to params and functionParams dicts
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
        duplicate_keys_allowed=False,                \
        duplicate_keys_select=RANDOM,                \
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

    * First, with probability `retrieval_prob <DND.retrieval_prob>`, retrieve vector from `memory <DND.memory>` using
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
       * Keys in `memory <DND.memory>` are stored as tuples (since lists and arrays are not hashable);
         they are converted to arrays for evaluation during retrieval.
       ..
       * All keys must be the same length (for comparision during retrieval).

    Arguments
    ---------

    default_variable : list or 2d array : default class_defaults.variable
        specifies a template for the key and value entries of the dictionary;  list must have two entries, each
        of which is a list or array;  first item is used as key, and second as value entry of dictionary.

    retrieval_prob : float in interval [0,1] : default 1.0
        specifies probability of retrieiving a value from `memory <DND.memory>`.

    storage_prob : float in interval [0,1] : default 1.0
        specifies probability of adding `variable <DND.variable>` to `memory <DND.memory>`.

    rate : float, list, or array : default 1.0
        specifies a value used to multiply key (first item of `variable <DND.variable>`) before storing in `memory
        <DND.memory>` (see `rate <DND.noise> for details).

    noise : float, list, array, or Function : default 0.0
        specifies a random value added to key (first item of `variable <DND.variable>`) before storing in `memory
        <DND.memory>` (see `noise <DND.noise> for details).

    initializer : 3d array : default None
        specifies an initial set of entries for `memory <DND.memory>`.  The outer dimension (axis 0) must have two
        2d arrays (one for keys, the other for values);  the length of all the 1d arrays in keys and values must
        be the same.

    distance_function : Distance or function : default Distance(metric=COSINE)
        specifies the function used during retrieval to compare the first item in `variable <DND.variable>`
        with keys in `memory <DND.memory>`.

    selection_function : OneHot or function : default OneHot(mode=MIN_VAL)
        specifies the function used during retrieval to evaluate the distances returned by `distance_function
        <DND.distance_function>` and select the item to return.

    duplicate_keys_allowed : bool : default False
        specifies whether entries with duplicate keys are allowed in `memory <DND.memory>` (see `duplicate_keys_allowed
        <DND.duplicate_keys_allowed for additional details>`).

    duplicate_keys_select:  RANDOM | OLDEST | NEWEST : default RANDOM
        if duplicate_keys_allowed is True, specifies which entry is retrieved from a set with duplicate keys.

    max_entries : int : default None
        specifies the maximum number of entries allowed in `memory <DND.memory>` (see `max_entries <DND.max_entries for
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
        1st item (variable[0] is the key used to retrieve an enrtry from `memory <DND.memory>`, and 2nd item
        (variable[1]) is the value of the entry, paired with key and added to the `memory <DND.memory>`.

    key_size : int
        length of keys in `memory <DND.memory>`.

    retrieval_prob : float in interval [0,1]
        probability of retrieiving a value from `memory <DND.memory>`.

    storage_prob : float in interval [0,1]
        probability of adding `variable <DND.variable>` to `memory <DND.memory>`.

    rate : float or 1d array
        value applied multiplicatively to key (first item of `variable <DND.variable>`) before storing in `memory
        <DND.memory>` (see `rate <Stateful_Rate>` for additional details).

    noise : float, 1d array or Function
        value added to key (first item of `variable <DND.variable>`) before storing in `memory <DND.memory>`
        (see `noise <Stateful_Noise>` for additional details).

    initializer : 3d array
        initial set of entries for `memory <DND.memory>`.

    memory : 3d array
        array of key-value pairs containing entries in DND:  [[[key 1], [value 1]],[[key 2], value 2]]...]

    distance_function : Distance or function : default Distance(metric=COSINE)
        function used during retrieval to compare the first item in `variable <DND.variable>`
        with keys in `memory <DND.memory>`.

    selection_function : OneHot or function : default OneHot(mode=MIN_VAL)
        function used during retrieval to evaluate the distances returned by `distance_function
        <DND.distance_function>` and select the item to return.

    previous_value : 1d array
        state of the `memory <DND.memory>` prior to storing `variable <DND.variable>` in the current call.

    duplicate_keys_allowed : bool
        determines whether entries with duplicate keys are allowed in `memory <DND.memory>`.  If False,
        then an attempt to store and item with a key that is already in `memory <DND.memory>` is ignored.
        If True, such items can be stored, and on retrieval using that key, a random entry matching the key is selected.
        
    duplicate_keys_select:  RANDOM | OLDEST | NEWEST
        if duplicate_keys_allowed is True, deterimines which entry is retrieved from a set with duplicate keys.

    max_entries : int
        maximum number of entries allowed in `memory <DND.memory>`;  if an attempt is made to add an additional entry
        an error is generated.

    random_state: numpy.RandomState instance

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
        """
            Attributes
            ----------

                variable
                    see `variable <DND.variable>`

                    :default value: [[0], [0]]
                    :type: list

                key_size
                    see `key_size <DND.key_size>`

                    :default value: 1
                    :type: int

                duplicate_keys_allowed
                    see `duplicate_keys_allowed <DND.duplicate_keys_allowed>`

                    :default value: False
                    :type: bool

                duplicate_keys_select
                    see `duplicate_keys_select <DND.duplicate_keys_select>`

                    :default value: False
                    :type: bool

                max_entries
                    see `max_entries <DND.max_entries>`

                    :default value: 1000
                    :type: int

                noise
                    see `noise <DND.noise>`

                    :default value: 0.0
                    :type: float

                random_state
                    see `random_state <DND.random_state>`

                    :default value: None
                    :type:

                rate
                    see `rate <DND.rate>`

                    :default value: 1.0
                    :type: float

                retrieval_prob
                    see `retrieval_prob <DND.retrieval_prob>`

                    :default value: 1.0
                    :type: float

                storage_prob
                    see `storage_prob <DND.storage_prob>`

                    :default value: 1.0
                    :type: float

        """
        variable = Parameter([[0],[0]])
        retrieval_prob = Parameter(1.0, modulable=True)
        storage_prob = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        key_size = Parameter(1, stateful=True)
        duplicate_keys_allowed = Parameter(False)
        duplicate_keys_select = Parameter(RANDOM)
        rate = Parameter(1.0, modulable=True)
        noise = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        max_entries = Parameter(1000)
        random_state = Parameter(None, modulable=False, stateful=True)

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
                 initializer: tc.optional(tc.any(list, np.ndarray))=None,
                 distance_function:tc.optional(tc.any(Distance, is_function_type))=None,
                 selection_function:tc.optional(tc.any(OneHot, is_function_type))=None,
                 duplicate_keys_allowed:bool=False,
                 duplicate_keys_select:tc.enum(RANDOM, OLDEST, NEWEST)=RANDOM,
                 max_entries=1000,
                 seed=None,
                 params: tc.optional(tc.any(list, np.ndarray)) = None,
                 owner=None,
                 prefs: is_pref_set = None):

        initializer = np.array(initializer or [])

        # It is necessary to create custom instances. Otherwise python would
        # share the same default instance for all DND objects.
        distance_function = distance_function or Distance(metric=COSINE)
        selection_function = selection_function or OneHot(mode=MIN_INDICATOR)

        self.distance_function = distance_function
        self.selection_function = selection_function

        if seed is None:
            seed = get_global_seed()
        random_state = np.random.RandomState(np.asarray([seed]))

        self._memory = []

        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(retrieval_prob=retrieval_prob,
                                                  storage_prob=storage_prob,
                                                  initializer=initializer,
                                                  duplicate_keys_allowed=duplicate_keys_allowed,
                                                  duplicate_keys_select=duplicate_keys_select,
                                                  rate=rate,
                                                  noise=noise,
                                                  max_entries=max_entries,
                                                  random_state=random_state,
                                                  params=params)

        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)

        if len(initializer) != 0:
            # self.parameters.key_size.set(len(list(initializer.keys())[0]))
            self.parameters.key_size.set(initializer.shape[2])

        self.has_initializers = True
        self.stateful_attributes = ["previous_value", "random_state"]

    def _get_context_struct_type(self, ctx):
        distance_state = ctx.get_context_struct_type(self.distance_function)
        selection_state = ctx.get_context_struct_type(self.selection_function)
        # Get random state
        random_state_struct = ctx.convert_python_struct_to_llvm_ir(self.get_current_function_param("random_state"))
        # Construct a ring buffer
        keys_struct = pnlvm.ir.ArrayType(
            ctx.convert_python_struct_to_llvm_ir(self.defaults.variable[0]),
            self.get_current_function_param("max_entries"))
        vals_struct = pnlvm.ir.ArrayType(
            ctx.convert_python_struct_to_llvm_ir(self.defaults.variable[1]),
            self.get_current_function_param("max_entries"))
        ring_buffer_struct = pnlvm.ir.LiteralStructType([
            keys_struct, vals_struct, ctx.int32_ty, ctx.int32_ty])
        my_state = pnlvm.ir.LiteralStructType([random_state_struct, ring_buffer_struct])
        return pnlvm.ir.LiteralStructType([distance_state, selection_state, my_state])

    def _get_param_struct_type(self, ctx):
        distance_params = ctx.get_param_struct_type(self.distance_function)
        selection_params = ctx.get_param_struct_type(self.selection_function)

        my_param_struct = self._get_param_values()
        my_params = ctx.convert_python_struct_to_llvm_ir(my_param_struct)
        elements = [e for e in my_params.elements]
        elements.append(distance_params)
        elements.append(selection_params)
        return pnlvm.ir.LiteralStructType(elements)

    def _get_param_initializer(self, execution_id):
        distance_init = self.distance_function._get_param_initializer(execution_id)
        selection_init = self.selection_function._get_param_initializer(execution_id)
        my_init = super()._get_param_initializer(execution_id)
        return tuple([*my_init, distance_init, selection_init])

    def _get_context_initializer(self, execution_id):
        distance_init = self.distance_function._get_context_initializer(execution_id)
        selection_init = self.selection_function._get_context_initializer(execution_id)
        random_state = self.get_current_function_param("random_state", execution_id).get_state()[1:]
        memory = self.get_previous_value(execution_id)
        my_init = pnlvm._tupleize([random_state, [memory[0], memory[1], 0, 0]])
        return tuple([distance_init, selection_init, my_init])

    def _gen_llvm_function_body(self, ctx, builder, params, state, arg_in, arg_out):
        my_state = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(2)])
        # PRNG
        rand_struct = builder.gep(my_state, [ctx.int32_ty(0), ctx.int32_ty(0)])
        uniform_f = ctx.get_llvm_function("__pnl_builtin_mt_rand_double")

        # Ring buffer
        buffer_ptr = builder.gep(my_state, [ctx.int32_ty(0), ctx.int32_ty(1)])
        keys_ptr = builder.gep(buffer_ptr, [ctx.int32_ty(0), ctx.int32_ty(0)])
        vals_ptr = builder.gep(buffer_ptr, [ctx.int32_ty(0), ctx.int32_ty(1)])
        count_ptr = builder.gep(buffer_ptr, [ctx.int32_ty(0), ctx.int32_ty(2)])
        wr_ptr = builder.gep(buffer_ptr, [ctx.int32_ty(0), ctx.int32_ty(3)])
        max_entries = len(vals_ptr.type.pointee)

        # Input
        var_key_ptr = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(0)])
        var_val_ptr = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(1)])

        # Zero output
        out_key_ptr = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(0)])
        out_val_ptr = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(1)])
        with pnlvm.helpers.array_ptr_loop(builder, out_key_ptr, "zero_key") as (b, i):
            out_ptr = b.gep(out_key_ptr, [ctx.int32_ty(0), i])
            b.store(out_ptr.type.pointee(0), out_ptr)
        with pnlvm.helpers.array_ptr_loop(builder, out_val_ptr, "zero_val") as (b, i):
            out_ptr = b.gep(out_val_ptr, [ctx.int32_ty(0), i])
            b.store(out_ptr.type.pointee(0), out_ptr)

        # Check retrieval probability
        retr_ptr = builder.alloca(pnlvm.ir.IntType(1))
        builder.store(retr_ptr.type.pointee(1), retr_ptr)
        retr_prob_ptr = ctx.get_param_ptr(self, builder, params, RETRIEVAL_PROB)

        # Prob can be [x] if we are part of a mechanism
        retr_prob = pnlvm.helpers.load_extract_scalar_array_one(builder, retr_prob_ptr)
        retr_rand = builder.fcmp_ordered('<', retr_prob, retr_prob.type(1.0))

        entries = builder.load(count_ptr)
        entries = pnlvm.helpers.uint_min(builder, entries, max_entries)
        # The call to random function needs to be behind jump to match python
        # code
        with builder.if_then(retr_rand):
            rand_ptr = builder.alloca(ctx.float_ty)
            builder.call(uniform_f, [rand_struct, rand_ptr])
            rand = builder.load(rand_ptr)
            passed = builder.fcmp_ordered('<', rand, retr_prob)
            builder.store(passed, retr_ptr)

        param_count = len(list(self._get_compilation_params()))
        # Retrieve
        retr = builder.load(retr_ptr)
        with builder.if_then(retr, likely=True):
            # Determine distances
            distance_f = ctx.get_llvm_function(self.distance_function)
            distance_params = builder.gep(params, [ctx.int32_ty(0), ctx.int32_ty(param_count)])
            distance_state = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(0)])
            distance_arg_in = builder.alloca(distance_f.args[2].type.pointee)
            builder.store(builder.load(var_key_ptr),
                          builder.gep(distance_arg_in, [ctx.int32_ty(0),
                                                        ctx.int32_ty(0)]))
            selection_arg_in = builder.alloca(pnlvm.ir.ArrayType(distance_f.args[3].type.pointee, max_entries))
            with pnlvm.helpers.for_loop_zero_inc(builder, entries, "distance_loop") as (b,idx):
                compare_ptr = b.gep(keys_ptr, [ctx.int32_ty(0), idx])
                b.store(b.load(compare_ptr),
                        b.gep(distance_arg_in, [ctx.int32_ty(0), ctx.int32_ty(1)]))
                distance_arg_out = b.gep(selection_arg_in, [ctx.int32_ty(0), idx])
                b.call(distance_f, [distance_params, distance_state,
                                    distance_arg_in, distance_arg_out])

            selection_f = ctx.get_llvm_function(self.selection_function)
            selection_params = builder.gep(params, [ctx.int32_ty(0), ctx.int32_ty(param_count + 1)])
            selection_state = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(1)])
            selection_arg_out = builder.alloca(selection_f.args[3].type.pointee)
            builder.call(selection_f, [selection_params, selection_state,
                                       selection_arg_in, selection_arg_out])

            # Find the selected index
            selected_idx_ptr = builder.alloca(ctx.int32_ty)
            builder.store(ctx.int32_ty(0), selected_idx_ptr)
            with pnlvm.helpers.for_loop_zero_inc(builder, entries, "distance_loop") as (b,idx):
                selection_val = b.load(b.gep(selection_arg_out, [ctx.int32_ty(0), idx]))
                non_zero = b.fcmp_ordered('!=', selection_val, ctx.float_ty(0))
                with b.if_then(non_zero):
                    b.store(idx, selected_idx_ptr)

            selected_idx = builder.load(selected_idx_ptr)
            selected_key = builder.load(builder.gep(keys_ptr, [ctx.int32_ty(0),
                                                               selected_idx]))
            selected_val = builder.load(builder.gep(vals_ptr, [ctx.int32_ty(0),
                                                               selected_idx]))
            builder.store(selected_key, builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(0)]))
            builder.store(selected_val, builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(1)]))

        # Check storage probability
        store_ptr = builder.alloca(pnlvm.ir.IntType(1))
        builder.store(store_ptr.type.pointee(1), store_ptr)
        store_prob_ptr = ctx.get_param_ptr(self, builder, params, STORAGE_PROB)

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

    def _validate(self):
        distance_function = self.distance_function
        test_var = [self.defaults.variable[0], self.defaults.variable[0]]
        if isinstance(distance_function, type):
            distance_function = distance_function(default_variable=test_var)
            fct_msg = 'Function type'
        else:
            distance_function.defaults.variable = test_var
            distance_function._instantiate_value()
            fct_msg = 'Function'
        try:
            distance_result = distance_function(test_var)
            if not np.isscalar(distance_result):
                raise FunctionError("Value returned by {} specified for {} ({}) must return a scalar".
                                    format(repr(DISTANCE_FUNCTION), self.__name__.__class__, distance_result))
        except:
            raise FunctionError("{} specified for {} arg of {} ({}) "
                                "must accept a list with two 1d arrays or a 2d array as its argument".
                                format(fct_msg, repr(DISTANCE_FUNCTION), self.__class__,
                                       distance_function))

        # Default to full memory dictionary
        selection_function = self.selection_function
        test_var = np.asfarray([distance_result if i==0
                                else np.zeros_like(distance_result)
                                for i in range(self.get_current_function_param('max_entries'))])
        if isinstance(selection_function, type):
            selection_function = selection_function(default_variable=test_var)
            fct_msg = 'Function type'
        else:
            selection_function.defaults.variable = test_var
            selection_function._instantiate_value()
            fct_msg = 'Function'
        try:
            result = np.asarray(selection_function(test_var))
        except e:
            raise FunctionError("{} specified for {} arg of {} ({}) must accept a 1d array "
                                "must accept a list with two 1d arrays or a 2d array as its argument".
                                format(fct_msg, repr(SELECTION_FUNCTION), self.__class__,
                                       selection_function))
        if result.shape != test_var.shape:
            raise FunctionError("Value returned by {} specified for {} ({}) "
                                "must return an array of the same length it receives".
                                format(repr(SELECTION_FUNCTION), self.__class__, result))

    def _initialize_previous_value(self, initializer, execution_context=None):
        # vals = [[k for k in initializer.keys()], [v for v in initializer.values()]]
        vals = initializer
        previous_value = np.asfarray(vals) if len(initializer) != 0 else np.ndarray(shape=(2, 0, len(self.defaults.variable[0])))

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

        Clears the memory in `previous_value <DND.previous_value>`.

        If an argument is passed into reinitialize or if the `initializer <DND.initializer>` attribute contains a
        value besides [], then that value is used to start the new memory in `previous_value <DND.previous_value>`.
        Otherwise, the new `previous_value <DND.previous_value>` memory starts out empty.

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

        if reinitialization_value == []:
            self.get_previous_value(execution_context).clear()
            value = np.ndarray(shape=(2, 0, len(self.defaults.variable[0])))

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
        Return value of entry in `memory <DND.memory>` best matched by first item of `variable <DND.variable>`, then add
        `variable <DND.variable>` to `memory <DND.memory>`.

        If the length of `memory <DND.memory>` exceeds `max_entries <DND.max_entries>`, generate an error.

        Arguments
        ---------

        variable : list or 2d array : default class_defaults.variable
           first item (variable[0]) is treated as the key for retrieval; second item (variable[1]), paired
           with key, is added to `memory <DND.memory>`.

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

        # get random state
        random_state = self.get_current_function_param('random_state', execution_id)

        # If this is an initialization run, leave memory empty (don't want to count it as an execution step),
        # and return current value (variable[1]) for validation.
        if self.parameters.context.get(execution_id).initialization_status == ContextFlags.INITIALIZING:
            return variable

        # Set key_size if this is the first entry
        if len(self.get_previous_value(execution_id)[0]) == 0:
            self.parameters.key_size.set(len(key), execution_id)

        # Retrieve value from current dict with key that best matches key
        if retrieval_prob == 1.0 or (retrieval_prob > 0.0 and retrieval_prob > random_state.rand()):
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
        if storage_prob == 1.0 or (storage_prob > 0.0 and storage_prob > random_state.rand()):
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
            raise FunctionError("Length of 'key'({}) to store in {} must be same as others in the dict ({})".
                                format(len(key), self.__class__.__name__, self.parameters.key_size.get(execution_id)))

    @tc.typecheck
    def get_memory(self, query_key:tc.any(list, np.ndarray), execution_id=None):
        """get_memory(query_key, execution_id=None)

        Retrieve memory from `memory <DND.memory>` based on `distance_function <DND.distance_function>` and
        `selection_function <DND.selection_function>`.

        Arguments
        ---------
        query_key : list or 1d array
            must be same length as key(s) of any existing entries in `memory <DND.memory>`.

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
        # compute similarity(query_key, memory m ) for all m
        _memory = self.get_previous_value(execution_id)
        if len(_memory[0]) == 0:
            return np.zeros_like(self.defaults.variable)
        distances = [self.distance_function([query_key, list(m)]) for m in _memory[0]]
        # get the best-match memory (one with the only non-zero value in the array)
        selection_array = self.selection_function(distances)
        indices_of_selected_items = np.flatnonzero(selection_array)
        if self.duplicate_keys_allowed:
            # FIX ADD HANDLING OF DUPLICATE KEYS ARE BASE ON duplicate_keys_select
            if self.duplicate_keys_select == RANDOM:
                index_of_selected_item = choice(indices_of_selected_items)
            elif self.duplicate_keys_select == OLDEST:
                index_of_selected_item = indices_of_selected_items[0]
            elif self.duplicate_keys_select == NEWEST:
                index_of_selected_item = indices_of_selected_items[-1]
            else:
                assert False, f'PROGRAM ERROR:  bad specification ({self.duplicate_keys_select}) for  ' \
                    f'\'duplicate_keys_select parameter of {self.name} for {self.owner.name}'
        # else:
        #     assert len(indices_of_selected_items)==1, \
        #     f'PROGRAM ERROR:  More than one item matched key ({_memory[0]}) ' \
        #         f'in memory for {self.name} of {self.owner.name} even though \'duplicate_keys_allowed\' is False'
        #     index_of_selected_item = int(np.flatnonzero(selection_array))
        #     return np.array([])
        elif len(indices_of_selected_items)==1:
            index_of_selected_item = int(np.flatnonzero(selection_array))
        else:
            warnings.warn(f'More than one item matched key ({_memory[0]}) in memory for {self.name} of '
                          f'{self.owner.name} even though \'duplicate_keys_allowed\' is False')
            return np.array([])
        best_match_key = _memory[0][index_of_selected_item]
        best_match_val = _memory[1][index_of_selected_item]

        return np.array([best_match_key, best_match_val])

    @tc.typecheck
    def _store_memory(self, memory:tc.any(list, np.ndarray), execution_id):
        """Save an key-value pair to `memory <DND.memory>`

        Arguments
        ---------
        memory : list or 2d array
            must be two items, a key and a vaue, each of which must a list of numbers or 1d array;
            the key must be the same length as key(s) of any existing entries in `dict <DND.dict>`.
        """

        self._validate_memory(memory, execution_id)

        key = memory[0]
        val = memory[1]

        d = self.get_previous_value(execution_id)

        if len(d[0]) >= self.max_entries:
            d = np.delete(d, [0], axis=1)

        # If dupliciate keys are not allowed and key matches any existing keys then don't encode
        if not self.duplicate_keys_allowed and any(d==0 for d in [self.distance_function([key, list(m)]) for m in d[0]]):
            pass
        else:
            keys = np.append(d[0], key).reshape(len(d[0])+1, len(key))
            values = np.append(d[1], val).reshape(len(d[1])+1, len(val))
            d = np.asfarray([keys, values])

        self.parameters.previous_value.set(d,execution_id)
        self._memory = d

    @tc.typecheck
    def insert_memories(self, memories:tc.any(list, np.ndarray), execution_id=None):
        """insert_memories(memories, execution_id=None)

        add key-value pairs to `memory <DND.memory>`.

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
            # self._store_memory(memory[0], memory[1], execution_id)
            self._store_memory(memory, execution_id)

    @property
    def memory(self):
        try:
            return np.array([[k,v] for k,v in zip(self._memory[0],self._memory[1])])
        except:
            return np.array([])