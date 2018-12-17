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

* `MemoryFunction`
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
from psyneulink.core.globals.keywords import BUFFER_FUNCTION, COSINE, DND_FUNCTION, MIN_VAL, NOISE, RATE
from psyneulink.core.globals.utilities import all_within_range
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.parameters import Param
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set

__all__ = ['Buffer', 'DND']


class Buffer(StatefulFunction):  # ------------------------------------------------------------------------------
    """
    Buffer(                     \
        default_variable=None,  \
        rate=None,              \
        noise=0.0,              \
        history=None,           \
        initializer,            \
        params=None,            \
        owner=None,             \
        prefs=None,             \
        )

    .. _Buffer:

    Appends `variable <Buffer.variable>` to the end of `previous_value <Buffer.previous_value>` (i.e., right-appends)
    which makes it a deque of previous inputs.  If specified, the values of the **rate** and **noise** arguments are
    applied to each item in the deque (including the newly added one) on each call, as follows:

        :math: item * `rate <Buffer.rate>` + `noise <Buffer.noise>`

    .. note::
       Because **rate** and **noise** are applied on every call, their effects are cumulative over calls.

    Arguments
    ---------

    default_variable : number, list or array : default ClassDefaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float : default None
        specifies a value applied to each item in the deque on each call.

    noise : float or Function : default 0.0
        specifies a random value added to each item in the deque on each call.

    history : int : default None
        specifies the maxlen of the deque, and hence `value <Buffer.value>`.

    initializer float, list or ndarray : default []
        specifies a starting value for the deque;  if none is specified, the deque is initialized with an
        empty list.

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

    rate : float
        value added to each item of the deque on each call.

    noise : float or Function
        random value added to each item of the deque in each call.

        .. note::
            In order to generate random noise, a probability distribution function should be used (see
            `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value from
            its distribution on each execution. If noise is specified as a float or as a function with a fixed output,
            then the noise will simply be an offset that remains the same across all executions.

    history : int
        determines maxlen of the deque and the value returned by the `function <Buffer.function>`. If appending
        `variable <Buffer.variable>` to `previous_value <Buffer.previous_value>` exceeds history, the first item of
        `previous_value <Buffer.previous_value>` is deleted, and `variable <Buffer.variable>` is appended to it,
        so that `value <Buffer.previous_value>` maintains a constant length.  If history is not specified,
        the value returned continues to be extended indefinitely.

    initializer : float, list or ndarray
        the value assigned as the first item of the deque when the Function is initialized, or reinitialized
        if the **new_previous_value** argument is not specified in the call to `reinitialize
        <IntegratorFunction.reinitialize>`.

    previous_value : 1d array : default ClassDefaults.variable
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

    class Params(StatefulFunction.Params):
        rate = Param(0.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        noise = Param(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
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
                 # rate: parameter_spec=1.0,
                 # noise=0.0,
                 # rate: tc.optional(tc.any(int, float)) = None,         # Changed to 1.0 because None fails validation
                 # noise: tc.optional(tc.any(int, float, callable)) = None,    # Changed to 0.0 - None fails validation
                 rate: tc.optional(tc.any(int, float)) = 1.0,
                 noise: tc.optional(tc.any(int, float, callable)) = 0.0,
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
        Return: `previous_value <Buffer.previous_value>` appended with `variable
        <Buffer.variable>` * `rate <Buffer.rate>` + `noise <Buffer.noise>`;

        If the length of the result exceeds `history <Buffer.history>`, delete the first item.

        Arguments
        ---------

        variable : number, list or array : default ClassDefaults.variable
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

        previous_value = self.get_previous_value(execution_id)
        previous_value.append(variable)

        # Apply rate and/or noise if they are specified
        if rate != 1.0:
            previous_value *= rate
        if noise:
            previous_value += noise

        previous_value = deque(previous_value, maxlen=self.history)

        self.parameters.previous_value.set(previous_value, execution_id)
        return self.convert_output_type(previous_value)


RETRIEVAL_PROB = 'retrieval_prob'
STORAGE_PROB = 'storage_prob'
DISTANCE_FUNCTION = 'distance_function'
SELECTION_FUNCTION = 'selection_function'


class DND(StatefulFunction):  # ------------------------------------------------------------------------------
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

    Implements simple form of `Differential Neural Dictionary described in `Ritter et al.
    <http://arxiv.org/abs/1805.09692>`_

    Based on implementation in `dlstm <https://github.com/qihongl/dlstm-demo>`_ by
    `Qihong Lu <https://github.com/qihongl>`_.  See also  `Kaiser et al. <http://arxiv.org/abs/1703.03129>`_
    and `Pritzel et al. <http://arxiv.org/abs/1703.01988>`_.

    .. _DND:

    First, with probability `retrieval_prob <DND.retrieval.prob>`, retrieve vector from `dict <DND.dict>` using
    first item of `variable <DND.variable>` as key, and the matching algorithm specified in `metric <DND.metric>`;
    if not retrieval occures, an appropriately shaped zero-valued array is returned.

    Then, with probability `storage_prob <DND.storage_prob>` add new entry using the first item of `variable
    <DND.variable>` as the key and its second item as the value. If specified, the values of the **rate** and
    **noise** arguments are applied to the key before storing:

    .. math::
        variable[1] * rate + noise

    .. note::
       Keys in `dict <DND.dict>` are stored as tuples (since lists and arrays are not hashable);
       they are converted to arrays for evaluation during retrieval.

    Arguments
    ---------

    default_variable : list or 2d array : default ClassDefaults.variable
        specifies a template for the key and value entries of the dictionary;  list must have two entries, each
        of which is a list or array;  first item is used as key, and second as value entry of dictionary.

    retrieval_prob : float in interval [0,1] : default 1.0
        specifies probability of retrieiving a value from `dict <DND.dict>`.

    storage_prob : float in interval [0,1] : default 1.0
        specifies probability of adding `variable <DND.variable>` to `dict <DND.dict>`.

    noise : float, list, array, or Function : default 0.0
        specifies a value applied to key (first item of `variable <DND.variable>`) before storing in `dict <DND.dict>`.

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

    retrieval_prob : float in interval [0,1]
        probability of retrieiving a value from `dict <DND.dict>`.

    storage_prob : float in interval [0,1]
        probability of adding `variable <DND.variable>` to `dict <DND.dict>`.

    noise : float, list, array, or Function
        value added to key (first item of `variable <DND.variable>`) before storing in `dict <DND.dict>`.

        .. note::
            In order to generate random noise, a probability distribution function should be used (see
            `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value from
            its distribution on each execution. If noise is specified as a float or as a function with a fixed output,
            then the noise will simply be an offset that remains the same across all executions.

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

    class Params(StatefulFunction.Params):
        variable = Param([[0],[0]])
        retrieval_prob = Param(1.0, modulable=True)
        storage_prob = Param(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        noise = Param(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        max_entries = Param(1000)

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        NOISE: None,
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
                 noise: tc.optional(tc.any(int, float, callable))=0.0,
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
        self.distance_function = self.distance_function.function

        if isinstance(self.selection_function, type):
            self.selection_function = self.selection_function()
        self.selection_function = self.selection_function.function

    def reinitialize(self, *args, execution_context=None):
        """

        Clears the `previous_value <Buffer.previous_value>` deque.

        If an argument is passed into reinitialize or if the `initializer <DND.initializer>` attribute contains a
        value besides [], then that value is used to start the new `previous_value <DND.previous_value>` dict.
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

        variable : list or 2d array : default ClassDefaults.variable
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

        previous_value = self.get_previous_value(execution_id)

        # Retrieve value from current dict with key that best matches key
        if retrieval_prob == 1.0 or (retrieval_prob > 0.0 and retrieval_prob > np.random.rand()):
            ret_val = self.get_memory(key)
        else:
            # QUESTION: SHOULD IT RETURN ZERO VECTOR OR NOT RETRIEVE AT ALL (LEAVING VALUE AND OUTPUTSTATE FROM LAST TRIAL)?
            #           CURRENT PROBLEM WITH LATTER IS THAT IT CAUSES CRASH ON INIT, SINCE NOT OUTPUT_STATE
            #           SO, WOULD HAVE TO RETURN ZEROS ON INIT AND THEN SUPPRESS AFTERWARDS, AS MOCKED UP BELOW
            ret_val = np.zeros_like(self.instance_defaults.variable)
            # if self.context.initialization_status == ContextFlags.INITIALIZING:
            #     ret_val = np.zeros_like(self.instance_defaults.variable)
            # else:
            #     ret_val = None

        # Store variable to dict:
        if noise:
            key += noise
        if storage_prob == 1.0 or (storage_prob > 0.0 and storage_prob > np.random.rand()):
            self.store_memory(key, value)

        self.parameters.previous_value.set(previous_value, execution_id)

        return self.convert_output_type(ret_val)

    def get_memory(self, query_key):
        """Perform a 1-NN search over dnd

        Parameters
        ----------
        query_key : 1d array
            used to retrieve item with key that best matches query_key, based on `distance_function
            <DND.distance_function>` and `selection_function <DND.selection_function>`.

        Returns
        -------
        value and key for item retrieved : 2d array
            if no retrieval occurs, returns appropriately shaped zero-valued array.

        """
        # QUESTION: SHOULD IT RETURN ZERO VECTOR OR NOT RETRIEVE AT ALL (LEAVING VALUE AND OUTPUTSTATE FROM LAST TRIAL)?
        #           ALSO, SHOULD PROBABILISTIC SUPPRESSION OF RETRIEVAL BE HANDLED HERE OR function (AS IT IS NOW).
        # if no memory, return the zero vector
        # if len(self.dict) == 0 or self.retrieval_prob == 0.0:
        if len(self.dict) == 0:
            return np.zeros_like(self.instance_defaults.variable)
        # compute similarity(query_key, memory m ) for all m
        distances = [self.distance_function([query_key, list(m)]) for m in self.dict.keys()]
        # get the best-match memory (one with the only non-zero value in the array)
        selection_array = self.selection_function(distances)
        index_of_selected_item = int(np.flatnonzero(selection_array))
        best_match_key = list(self.dict.keys())[index_of_selected_item]
        best_match_val = list(self.dict.values())[index_of_selected_item]

        return [best_match_val, best_match_key]

    def store_memory(self, memory_key, memory_val):
        """Save an episodic memory to the dictionary

        Parameters
        ----------
        memory_key : a row vector
            a DND key, used to for memory search
        memory_val : a row vector
            a DND value, representing the memory content
        """
        # add new memory to the the dictionary
        # get data is necessary for gradient reason
        # QUESTION: WHY DO IT THIS WAY RATHER THAN JUST UPDATE DICT?
        #           WHAT IS data.view?  Using Pandas?
        # self.keys.append(memory_key.data.view(1, -1))
        # self.vals.append(memory_val.data.view(1, -1))
        # remove the oldest memory, if overflow
        d = self.dict
        if len(self.dict) > self.max_entries:
            d.pop(list(d.keys())[len(d)-1])
        d[tuple(memory_key)]=memory_val

    def add_memories(self, input_keys, input_vals):
        """Inject pre-defined keys and values

        Parameters
        ----------
        input_keys : list
            a list of memory keys
        input_vals : list
            a list of memory content
        """
        for k, v in zip(input_keys, input_vals):
            self.store_memory(k, v)

    @property
    def dict(self):
        return self.get_previous_value()