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
* `DictionaryMemory`

Overview
--------

Functions that store and can return a record of their input.

"""

import numbers
import warnings
from collections import deque
from itertools import combinations, product
# from typing import Optional, Union, Literal, Callable
from typing import Optional, Union

import numpy as np
import typecheck as tc

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.functions.function import (
    DEFAULT_SEED, FunctionError, _random_state_getter, _seed_setter, is_function_type, EPSILON, _noise_setter
)
from psyneulink.core.components.functions.nonstateful.objectivefunctions import Distance
from psyneulink.core.components.functions.nonstateful.selectionfunctions import OneHot
from psyneulink.core.components.functions.stateful.integratorfunctions import StatefulFunction
from psyneulink.core.globals.context import handle_external_context
from psyneulink.core.globals.keywords import \
    ADDITIVE_PARAM, BUFFER_FUNCTION, MEMORY_FUNCTION, COSINE, \
    ContentAddressableMemory_FUNCTION, DictionaryMemory_FUNCTION, \
    MIN_INDICATOR, MULTIPLICATIVE_PARAM, NEWEST, NOISE, OLDEST, OVERWRITE, RATE, RANDOM, VARIABLE
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.utilities import \
    all_within_range, convert_to_np_array, convert_to_list, convert_all_elements_to_np_array

__all__ = ['MemoryFunction', 'Buffer', 'DictionaryMemory', 'ContentAddressableMemory', 'RETRIEVAL_PROB', 'STORAGE_PROB']


class MemoryFunction(StatefulFunction):  # -----------------------------------------------------------------------------
    componentType = MEMORY_FUNCTION

    # TODO: refactor to avoid skip of direct super
    def _update_default_variable(self, new_default_variable, context=None):
        if not self.parameters.initializer._user_specified:
            # use * 0 instead of zeros_like to deal with ragged arrays
            self._initialize_previous_value([new_default_variable * 0], context)

        # bypass the additional _initialize_previous_value call used by
        # other stateful functions
        super(StatefulFunction, self)._update_default_variable(new_default_variable, context=context)


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

                changes_shape
                    see `changes_shape <Function_Base.changes_shape>`

                    :default value: True
                    :type: bool

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
        noise = Parameter(
            0.0, modulable=True, aliases=[ADDITIVE_PARAM], setter=_noise_setter
        )
        history = None
        initializer = Parameter(np.array([]), pnl_internal=True)
        changes_shape = Parameter(True, stateful=False, loggable=False, pnl_internal=True)


    @check_user_specified
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
                 # rate:Optional[Union[int, float, np.ndarray]]=None,
                 # noise:Optional[Union[int, float, np.ndarray]]=None,
                 # rate: parameter_spec=1.0,
                 # noise: parameter_spec=0.0,
                 # rate: Optional[Union(int, float]] = None,  # Changed to 1.0: None fails validation
                 # noise: Optional[Union[int, float, callable]] = None, # Changed to 0.0 - None fails validation
                 # rate: Optional[Union[int, float, list, np.ndarray]] = 1.0,
                 # noise: Optional[Union[int, float, list, np.ndarray, callable]] = 0.0,
                 history:tc.optional(int)=None,
                 # history: Optional[int] = None,
                 initializer=None,
                 params: tc.optional(dict) = None,
                 # params: Optional[dict] = None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None
                 ):

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
        noise = self._try_execute_param(self._get_current_parameter_value(NOISE, context), variable, context=context)

        # If this is an initialization run, leave deque empty (don't want to count it as an execution step);
        # Just return current input (for validation).
        if self.is_initializing:
            return variable

        previous_value = self.parameters.previous_value._get(context)

        # Apply rate and/or noise, if they are specified, to all stored items
        if len(previous_value):
            # TODO: remove this shape hack when buffer shapes made consistent
            noise = np.reshape(noise, np.asarray(previous_value[0]).shape)
            previous_value = convert_to_np_array(previous_value) * rate + noise

        previous_value = deque(previous_value, maxlen=self.parameters.history._get(context))

        previous_value.append(variable)

        self.parameters.previous_value._set(previous_value, context)
        return self.convert_output_type(previous_value)


RETRIEVAL_PROB = 'retrieval_prob'
STORAGE_PROB = 'storage_prob'
DISTANCE_FUNCTION = 'distance_function'
SELECTION_FUNCTION = 'selection_function'
DISTANCE_FIELD_WEIGHTS = 'distance_field_weights'
equidistant_entries_select_keywords = [RANDOM, OLDEST, NEWEST]


class ContentAddressableMemory(MemoryFunction): # ---------------------------------------------------------------------
    """
    ContentAddressableMemory(                        \
        default_variable=None,                       \
        retrieval_prob=1.0,                          \
        storage_prob=1.0,                            \
        rate=None,                                   \
        noise=0.0,                                   \
        initializer=None,                            \
        distance_field_weights=None,                 \
        distance_function=Distance(metric=COSINE),   \
        selection_function=OneHot(mode=MIN_VAL),     \
        duplicate_entries_allowed=False,             \
        duplicate_threshold=0,                       \
        equidistant_entries_select=RANDOM,           \
        max_entries=None,                            \
        params=None,                                 \
        owner=None,                                  \
        prefs=None,                                  \
        )

    .. _ContentAddressableMemory:

    **Sections**

    - `Overview <ContentAddressableMemory_Overview>` \n
        `Entries and Fields <ContentAddressableMemory_Entries_and_Fields>` \n
        `Content-based Retrieval <ContentAddressableMemory_Retrieval>` \n
        `Duplicate entries <ContentAddressableMemory_Duplicate_Entries>` \n
    - `Structure <ContentAddressableMemory_Structure>`
    - `Execution <ContentAddressableMemory_Execution>` \n
        `Retrieval <ContentAddressableMemory_Execution_Retrieval>` \n
        `Storage <ContentAddressableMemory_Execution_Storage>` \n
    - `Examples <ContentAddressableMemory_Examples>`
    - `Class Reference <ContentAddressableMemory_Class_Reference>`

    .. _ContentAddressableMemory_Overview:

    **Overview**

    The ContentAddressableMemory `Function` implements a configurable, content-addressable storage and retrieval of
    entries from `memory <ContentAddressableMemory.memory>`. Storage is determined by `storage_prob
    <ContentAddressableMemory.storage_prob>`, and retrieval of entries is determined by
    `distance_function <ContentAddressableMemory.distance_function>`,
    `selection_function <ContentAddressableMemory.selection_function>`, and `retrieval_prob
    <ContentAddressableMemory.retrieval_prob>`.

    .. _ContentAddressableMemory_Entries_and_Fields:

    **Entries and Fields**. The **default_variable** argument specifies the shape of an entry in `memory
    <ContentAddressableMemory.storage_prob>`, each of which is a list or array of fields that are themselves lists or
    1d arrays (see `EpisodicMemoryMechanism_Memory_Fields`). An entry can have an arbitrary number of fields, and
    each field can have an arbitrary length.  However, all entries must have the same number of fields, and the
    corresponding fields must all have the same length across entries.  Fields can be weighted to determine the
    influence they have on retrieval, using the `distance_field_weights <ContentAddressableMemory.memory>` parameter
    (see `retrieval <ContentAddressableMemory_Retrieval>` below).

    .. hint::
        Entries in `memory <ContentAddressableMemory.memory>` can be assigned "labels" -- i.e., values
        that are not used in the calculation of distance -- by assigning them a weight of 0 or None in
        `distance_field_weights <ContentAddressableMemory.memory>`); either can be used for labels that
        are numeric values; however, if non-numeric values are assigned to a field as labels, then None
        must be specified for that field in `distance_field_weights <ContentAddressableMemory.memory>`.

    .. _ContentAddressableMemory_Retrieval:

    **Retrieval**. Entries are retrieved from `memory <ContentAddressableMemory.memory>` based on their distance from
    `variable <ContentAddressableMemory.variable>`, used as the cue for retrieval. The distance is computed using the
    `distance_function <ContentAddressableMemory.distance_function>`, which compares `variable
    <ContentAddressableMemory.variable>` with each entry in `memory <ContentAddressableMemory.storage_prob>` as full
    vectors (i.e., with all fields of each concatenated into a single array), or by computing the distance of each
    field in `variable <ContentAddressableMemory.variable>` with the corresponding ones of each entry in `memory
    <ContentAddressableMemory.memory>`, and then averaging those distances, possibly weighted by coefficients specified
    in `distance_field_weights <ContentAddressableMemory.distance_field_weights>`. The distances computed between
    `variable `<ContentAddressableMemory.variable>` and each entry in `memory <ContentAddressableMemory.memory>` are
    used by `selection_function <ContentAddressableMemory.selection_function>` to determine which entry is retrieved.
    The distance used for the last retrieval (i.e., between `variable <ContentAddressableMemory.variable>` and the
    entry retrieved), the distances of each of their corresponding fields (weighted by `distance_field_weights
    <ContentAddressableMemory.distance_field_weights>`), and the distances to all other entries are stored in `distance
    <ContentAddressableMemory.distance>` and `distances_by_field <ContentAddressableMemory.distances_by_field>`, and
    `distances_to_entries <ContentAddressableMemory.distances_to_entries>` respectively.

    .. _ContentAddressableMemory_Duplicate_Entries:

    **Duplicate Entries**. These can be allowed, disallowed, or overwritten during storage using
    `duplicate_entries_allowed <ContentAddressableMemory.duplicate_entries_allowed>`),
    and how selection is made among duplicate entries or ones indistinguishable by the
    `distance_function <ContentAddressableMemory.distance_function>` can be specified
    using `equidistant_entries_select <ContentAddressableMemory.equidistant_entries_select>`.

    The class also provides methods for directly retrieving (`get_memory
    <ContentAddressableMemory.get_memory>`), adding (`add_to_memory <ContentAddressableMemory.add_to_memory>`)
    and deleting (`delete_from_memory <ContentAddressableMemory.delete_from_memory>`) one or more entries from
    `memory <ContentAddressableMemory.memory>`.

    .. _ContentAddressableMemory_Structure:

    **Structure**

    An entry is stored and retrieved as an array containing a set of `fields <EpisodicMemoryMechanism_Memory_Fields>`
    each of which is a 1d array.  An array containing such entries can be used to initialize the contents of `memory
    <ContentAddressableMemory.memory>` by providing it in the **initializer** argument of the ContentAddressableMemory's
    constructor, or in a call to its  `reset  <ContentAddressableMemory.reset>` method.  The current contents of `memory
    <ContentAddressableMemory.memory>` can be inspected using the `memory <ContentAddressableMemory.memory>` attribute,
    which returns a list containing the current entries, each as a list containing all fields for that entry.  The
    `memory_num_fields <ContentAddressableMemory.memory_num_fields>` contains the number of fields expected for each
    entry, `memory_field_shapes <ContentAddressableMemory.memory_field_shapes>` their shapes, and `memory_num_entries
    <ContentAddressableMemory.memory_num_entries>` the total number of entries in `memory
    <ContentAddressableMemory.memory>`.

    .. _ContentAddressableMemory_Shapes:

    .. technical_note::
       Both `memory <ContentAddressableMemory.memory>` and all entries are stored as np.ndarrays, the dimensionality of
       which is determined by the shape of the fields of an entry.  If all fields have the same length (regular), then
       they are 2d arrays and `memory <ContentAddressableMemory.memory>` is a 3d array.  However, if fields vary in
       length (`ragged <https://en.wikipedia.org/wiki/Jagged_array>`_) then, although each field is 1d, an entry is
       also 1d (with dtype='object'), and `memory <ContentAddressableMemory.memory>` is 2d (with dtype='object').

    .. _ContentAddressableMemory_Execution:

    **Execution**

    When the ContentAddressableMemory function is executed, it first retrieves the entry in `memory
    <ContentAddressableMemory.memory>` that most closely matches `variable
    <ContentAddressableMemory.variable>` in the call, stores the latter in `memory <ContentAddressableMemory.memory>`,
    and returns the retrieved entry.  If `variable <ContentAddressableMemory.variable>` is an exact match of an entry
    in `memory <ContentAddressableMemory.memory>`, and `duplicate_entries_allowed
    <ContentAddressableMemory.duplicate_entries_allowed>` is False, then the matching item is returned, but `variable
    <ContentAddressableMemory.variable>` is not stored. These steps are described in more detail below.

    .. _ContentAddressableMemory_Execution_Retrieval:

    * **Retrieval:** first, with probability `retrieval_prob <ContentAddressableMemory.retrieval_prob>`,
      the entry closest to `variable <ContentAddressableMemory.variable>` is retrieved from is retrieved from `memory
      <ContentAddressableMemory.memory>`.  The entry is chosen by calling, in order:

        * `distance_function <ContentAddressableMemory.distance_function>`\: generates a list of and compares
          `distances <ContentAddressableMemory.distances>` between `variable <ContentAddressableMemory.variable>`
          and each entry in `memory <ContentAddressableMemory.memory>`, possibly weighted by `distance_field_weights
          <ContentAddressableMemory.distance_field_weights>`, as follows:

          .. _ContentAddressableMemory_Distance_Field_Weights:

          * if `distance_field_weights <ContentAddressableMemory.distance_field_weights>` is either a scalar or an
            array of scalars that are all the same, then it is used simply to scale the distance computed between
            `variable <ContentAddressableMemory.variable>` and each entry in `memory <ContentAddressableMemory.memory>`,
            each of which is computed by concatenating all items of `variable <ContentAddressableMemory.variable>` into
            a 1d array, similarly concatenating all `memory_fields <EpisodicMemoryMechanism_Memory_Fields>` of an
            entry in `memory <ContentAddressableMemory.memory>`, and then using `distance_function
            <ContentAddressableMemory.distance_function>` to compute the distance betwen them.

          * if `distance_field_weights <ContentAddressableMemory.distance_field_weights>` is an array of scalars with
            different values, then `variable <ContentAddressableMemory.variable>` is compared with each entry in `memory
            <ContentAddressableMemory.memory>` by using `distance_function <ContentAddressableMemory.distance_function>`
            to compute the distance of each item in `variable <ContentAddressableMemory.variable>` with the
            corresponding field of the entry in memory, and then averaging those distances weighted by the
            corresponding element of `distance_field_weights<ContentAddressableMemory.distance_field_weights>`.

            .. note::
               Fields assigned a weight of *0* or *None* are ignored in the distance calculation; that is, the distances
               between `variable <ContentAddressableMemory.variable>` and entries for those fields are not included
               in the averaging of distances by field.

        * `selection_function <ContentAddressableMemory.selection_function>`\: called with the list of distances
          to determine which entries to select for consideration. If more than on entry from `memory
          <ContentAddressableMemory.memory>` is identified, `equidistant_entries_select
          <ContentAddressableMemory.equidistant_entries_select>` is used to determine which to retrieve.  If no
          retrieval occurs, an appropriately shaped zero-valued array is assigned as the retrieved memory, and
          returned by the function.

        The distance between `variable <ContentAddressableMemory.variable>` and the retrieved entry is assigned to
        `distance `<ContentAddressableMemory.distance>`, the distance between of each of their fields is assigned to
        `distances_by_field <ContentAddressableMemory.distances_by_field>`, and the distances of `variable
        <ContentAddressableMemory.variable>` to all entries in `memory <ContentAddressableMemory.memory>` is assigned
        to `distances_to_entries <ContentAddressableMemory.distances_to_entries>`.

    .. _ContentAddressableMemory_Execution_Storage:

    * **Storage:** after retrieval, an attempt is made to store `variable <ContentAddressableMemory.variable>`
      in `memory memory <ContentAddressableMemory.memory>` with probability `storage_prob
      <ContentAddressableMemory.storage_prob>`;  if the attempt is made:

      * if `variable <ContentAddressableMemory.variable>` is identical to an entry already in `memory
        <ContentAddressableMemory.memory>`, as evaluated by
        `distance_function <ContentAddressableMemory.distance_function>` and `duplicate_threshold
        <ContentAddressableMemory.duplicate_threshold>`, then `duplicate_entries_allowed
        <ContentAddressableMemory.duplicate_entries_allowed>` determines whether or not to store the entry;

        if `duplicate_entries_allowed <ContentAddressableMemory.duplicate_entries_allowed>` is:

            * False -- storage is skipped;

            * True -- `variable <ContentAddressableMemory.variable>` is stored as another duplicate;

            * *OVERWRITE* -- the duplicate entry in `memory <ContentAddressableMemory.memory>` is replaced with
            `variable <ContentAddressableMemory.variable>` (which may be slightly different than the item it
            replaces, within the tolerance of `duplicate_threshold <ContentAddressableMemory.duplicate_threshold>`),
            and the matching entry is returned;

            .. note::

               If `duplicate_entries_allowed <ContentAddressableMemory.duplicate_entries_allowed>` is OVERWRITE but
               a duplicate entry is nevertheless identified during retrieval (e.g., **duplicate_entries_allowed** was
               previously changed from True to False), a warning is issued, and duplicate entry is overwritten with
               `variable <ContentAddressableMemory.variable>`.

      * if storage **rate** and/or **noise** arguments are specified in the constructor, they are
        applied to `variable <ContentAddressableMemory.variable>` before storage as :math:`variable * rate + noise`;

      * finally, if the number of entries in `memory <ContentAddressableMemory.memory>` exceeds `max_entries
        <ContentAddressableMemory.max_entries>`, the first (oldest) entry is deleted.  The current number of entries
        in memory is contained in the `memory_num_entries <ContentAddressableMemory.memory_num_entries>` attribute.

    .. _ContentAddressableMemory_Examples:

    **Examples**

    *Initialize memory with **default_variable*

    The format for entries in `memory <ContentAddressableMemory.memory` can be specified using either the
    **default_variable** or **initializer** arguments of the Function's constructor.  **default_variable** specifies
    the shape of entries, without creating any entries::

        >>> c = ContentAddressableMemory(default_variable=[[0,0],[0,0,0]])
        >>> c([[1,2]])
        [array([0, 0])]

    Since `memory <ContentAddressableMemory.memory>` was not intialized, the first call to the Function returns an
    array of zeros, formatted as specified in **defaul_variable**.  However, the input in the call to the Function
    (``[[1,2]]``) is stored as an entry in `memory <EpisodicMemoryMechanism.memory>`::

        >>> c.memory
        array([[[1., 2.]]])

    and is returned on the next call::

        >>> c([[2,5]])
        array([[1., 2.]])

    Note that even though **default_variable** and the inputs to the Function are specified as lists, the entries
    returned are arrays; `memory <ContentAddressableMemory.memory>` and all of its entries are always formated as
    arrays.

    *Initialize memory with **initializer*

    The **initializer** argument of a ContentAddressableMemory's constructor can be used to initialize its `memory
    <ContentAddressableMemory.memory>`::

        >>> c = ContentAddressableMemory(initializer=[[[1,2],[3,4,5]],
        ...                                            [[10,9],[8,7,6]]])
        >>> c([[1,2],[3,4,6]])
        array([array([1., 2.]), array([3., 4., 5.])], dtype=object)
        >>> c([[1,2],[3,4,6]])
        array([array([1., 2.]), array([3., 4., 6.])], dtype=object)

    Note that there was no need to use **default_variable**, and in fact it would overidden if specified.

    .. _ContentAddressableMemory_Examples_Weighting_Fields:

    *Weighting fields*

    The **distance_field_weights** argument can be used to differentially weight memory fields to modify their
    influence on retrieval (see `distance_field_weights <ContentAddressableMemory_Distance_Field_Weights>`).  For
    example, this can be used to configure the Function as a dictionary, using the first field for keys (on which
    retrieval is based) and the second for values (that are retrieved with a matching key), as follows:

        >>> c = ContentAddressableMemory(initializer=[[[1,2],[3,4]],
        ...                                            [[1,5],[10,11]]],
        ...                              distance_field_weights=[1,0])
        >>> c([[1,2.5],[10,11]])
        array([[1., 2.],
               [3., 4.]])

    Note that the first entry ``[[1,2],[3,4]]`` in `memory <ContentAddressableMemory.memory>` was retrieved,
    even though the cue used in the call (``[[1,2.5],[10,11]]``) was an exact match to the second field of the
    second entry (``[[1,5],[10,11]]``).  However, since that field was assigned 0 in **distance_field_weights**,
    it was ignored and, using only the first entry, the cue was closer to the first entry. This is confirmed by
    repeating the example without specifying **distance_field_weights**::

        >>> c = ContentAddressableMemory(initializer=[[[1,2],[3,4]],
        ...                                            [[1,5],[10,11]]])
        >>> c([[1,2.5],[10,11]])
        array([[ 1.,  5.],
               [10., 11.]])

    COMMENT:
    # FIX: ADD EXAMPLES FOR ENTRIES WITH DIFFERENT SHAPES
    COMMENT

    *Duplicate entries*

    By default, duplicate entries are precluded from a ContentAddressableMemory's `memory
    <ContentAddressableMemory.memory>`.  So, for an initializer with identical entries, only one copy of
    the duplicates will be stored::

        >>> c = ContentAddressableMemory(initializer=[[[1,2],[3,4]],
        ...                                           [[1,2],[3,4]]])
        >>> c.memory
        array([[[1., 2.],
                [3., 4.]]])

    and using the same array as input to the function will retrieve that array but not store another copy::

        >>> c([[1,2],[3,4]])
        array([[1., 2.],
               [3., 4.]])
        >>> c.memory
        array([[[1., 2.],
                [3., 4.]]])

    Only fields with non-zero weights in `distance_field_weights <ContentAddressableMemory.distance_field_weights>`
    are considered when evaluating whether entries are duplicates. So, in the following example, where the weight
    for the second field is 0, the two entries are considered duplicates and only the first is stored::

        >>> c = ContentAddressableMemory(initializer=[[[1,2],[3,4]],
        ...                                           [[1,2],[5,6]]],
        ...                              distance_field_weights=[1,0])
        >>> c.memory
        array([[[1., 2.],
                [3., 4.]]])

    Duplicates can be allowed by setting the **duplicate_entries_allowed** argument to True or *OVERWRITE*.  Setting
    it to True allows duplicate entries to accumulate in `memory <ContentAddressableMemory.memory>`, as shown
    here::

        >>> c = ContentAddressableMemory(initializer=[[[1,2],[3,4]],
        ...                                            [[1,5],[10,11]]],
        ...                              duplicate_entries_allowed=True)
        >>> c([[1,2],[3,4]])
        array([[1., 2.],
               [3., 4.]])
        >>> c.memory
        array([[[ 1.,  2.],
                [ 3.,  4.]],
        <BLANKLINE>
               [[ 1.,  5.],
                [10., 11.]],
        <BLANKLINE>
               [[ 1.,  2.],
                [ 3.,  4.]]])

    Duplicates are determined by comparing entries using the functions `distance_function
    <ContentAddressableMemory.distance_function>`;  if the `distance <ContentAddressableMemory.distance>`
    is less than `duplicate_threshold <ContentAddressableMemory.duplicate_threshold>`, they are considered to be
    duplicates;  otherwise they are treated a distinct entries.  By default, `duplicate_threshold
    <ContentAddressableMemory.duplicate_threshold>` is 0.  In the folloiwng example it is increased, so that
    two very similar, but non-identical entries, are nonetheless treated as duplicates::

        >>> c = ContentAddressableMemory(initializer=[[[1, 2.0], [3, 4]],
        ...                                           [[1, 2.5], [3, 4]]],
        ...                              duplicate_entries_allowed=False,
        ...                              duplicate_threshold=0.2)

        >>> c.memory
        array([[[1., 2.],
                [3., 4.]]])

    Setting **duplicate_entries_allowed** argument to *OVERWRITE* allows an entry to replace one that is considered
    duplicate, even if it is not identical, as in the following example::

        >>> c.duplicate_entries_allowed=OVERWRITE
        >>> c([[1, 2.1], [3, 4]])
        array([[1., 2.],
               [3., 4.]])
        >>> c.memory
        array([[[1. , 2.1],
                [3. , 4. ]]])

    Note that the entry considered to be the duplicate (``[[1, 2.1], [3, 4]]``) is returned, and then replaced in
    `memory <ContentAddressableMemory.memory>`.  Finally, if `duplicate_entries_allowed
    <ContentAddressableMemory.duplicate_entries_allowed>` is True, and duplicates have accumulated, the
    `equidistant_entries_select <ContentAddressableMemory.equidistant_entries_select>` attribute can be used to
    specify how to select among them for retrieval, either by chosing randomly (*RANDOM*) or selecting either the
    first one (*OLDEST*) or last one (*NEWEST*) stored.

    .. _ContentAddressableMemory_Class_Reference:

    **Class Reference**

    Arguments
    ---------

    default_variable : list or 2d array : default class_defaults.variable
        specifies a template for an entry in the dictionary;  the list or array can have any number of items,
        each of which must be a list or array of any length;  however, at present entries are constrained to be
        at most 2d.

    retrieval_prob : float in interval [0,1] : default 1.0
        specifies probability of retrieving an entry from `memory <ContentAddressableMemory.memory>`.

    storage_prob : float in interval [0,1] : default 1.0
        specifies probability of adding `variable <ContentAddressableMemory.variable>` to `memory
        <ContentAddressableMemory.memory>`.

    rate : float, list, or array : default 1.0
        specifies a value used to multiply `variable <ContentAddressableMemory.variable>` before storing in
        `memory <ContentAddressableMemory.memory>` (see `rate <ContentAddressableMemory.rate>` for details).

    noise : float, list, 2d array, or Function : default 0.0
        specifies random value(s) added to `variable <ContentAddressableMemory.variable>` before storing in
        `memory <ContentAddressableMemory.memory>`\;  if a list or 2d array, it must be the same shape as `variable
         ContentAddressableMemory.variable>` (see `noise <ContentAddressableMemory.noise>` for details).

    initializer : 3d array or list : default None
        specifies an initial set of entries for `memory <ContentAddressableMemory.memory>` (see
        `initializer <ContentAddressableMemory.initializer>` for additional details).

    distance_field_weights : 1d array : default None
        specifies the weight to use in computing the distance between each item of `variable
        <ContentAddressableMemory.variable>` and the corresponding `memory_field
        <EpisodicMemoryMechanism_Memory_Fields>` of each item in `memory <ContentAddressableMemory.memory>` (see
        `distance_field_weights <ContentAddressableMemory.distance_field_weights>` for additional details).

    distance_function : Distance or function : default Distance(metric=COSINE)
        specifies the function used during retrieval to compare `variable <ContentAddressableMemory.variable>` with
        entries in `memory <ContentAddressableMemory.memory>`.

    selection_function : OneHot or function : default OneHot(mode=MIN_VAL)
        specifies the function used during retrieval to evaluate the distances returned by `distance_function
        <ContentAddressableMemory.distance_function>` and select the item to retrieve.

    duplicate_entries_allowed : bool : default False
        specifies whether duplicate entries are allowed in `memory <ContentAddressableMemory.memory>`
        (see `duplicate entries <ContentAddressableMemory_Duplicate_Entries>` for additional details).

    duplicate_threshold : float : default 0
        specifies how similar `variable <ContentAddressableMemory.variable>` must be to an entry in `memory
        <ContentAddressableMemory.memory>` based on `distance_function <ContentAddressableMemory.distance_function>` to
        be considered a duplicate (see `duplicate entries <ContentAddressableMemory_Duplicate_Entries>`
        for additional details).

    equidistant_entries_select :  RANDOM | OLDEST | NEWEST : default RANDOM
        specifies which entry in `memory <ContentAddressableMemory.memory>` is chosen for retrieval if two or more
        have the same distance from `variable <ContentAddressableMemory.variable>`.

    max_entries : int : default None
        specifies the maximum number of entries allowed in `memory <ContentAddressableMemory.memory>`
        (see `max_entries <ContentAddressableMemory.max_entries>` for additional details).

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
        used to retrieve an entry from `memory <ContentAddressableMemory.memory>`, and then stored there.

    retrieval_prob : float in interval [0,1]
        probability of retrieiving a value from `memory <ContentAddressableMemory.memory>`.

    storage_prob : float in interval [0,1]
        probability of adding `variable <ContentAddressableMemory.variable>` to `memory
        <ContentAddressableMemory.memory>`.

        .. note::
           storage_prob does not apply to `initializer <ContentAddressableMemory,initializer>`, the entries of
           which are added to `memory <ContentAddressableMemory.memory>` irrespective of storage_prob.

    rate : float or 1d array
        value applied multiplicatively to `variable <ContentAddressableMemory.variable>`) before storing
        in`memory <ContentAddressableMemory.memory>` (see `rate <Stateful_Rate>` for additional details).

    noise : float, 2d array or Function
        value added to `variable <ContentAddressableMemory.variable>`) before storing in
        `memory <ContentAddressableMemory.memory>` (see `noise <Stateful_Noise>` for additional details).
        If a 2d array (or `Function` that returns one), its shape must be the same as `variable
        <ContentAddressableMemory.variable>`; that is, each array in the outer dimension (Axis 0) must have the
        same length as the corresponding one in `variable <ContentAddressableMemory.variable>`, so that it
        can be added Hadamard style to `variable <ContentAddressableMemory.variable>` before storing it in
        `memory <ContentAddressableMemory.memory>`.

    initializer : ndarray
        initial set of entries for `memory <ContentAddressableMemory.memory>`.  It should be either a 3d regular
        array or a 2d ragged array (if the fields of an entry have different lengths), but it can be specified
        in the **initializer** argument of the constructor using some simpler forms for convenience.  Specifically,
        scalars, 1d and regular 2d arrays are allowed, which are interpreted as a single entry that is converted to
        a 3d array to initialize `memory <ContentAddressableMemory.memory>`.

    memory : list
        list of entries in ContentAddressableMemory, each of which is an array of fields containing stored items;
        the fields of an entry must be lists or arrays, each of which can be different shapes, but the corresponding
        fields of all entries must have the same shape;  for example, the following could be a pair of entries in
        memory:

        +-------------+------------------------------+--------------------------------------------+
        |                entry 1                     |                  entry 2                   |
        +-------------+--------------+---------------+-----------+--------------+-----------------+
        |    field1   |    field2    |     field3    |   field1  |    field2    |    field3       |
        +-------------+--------------+---------------+-----------+--------------+-----------------+
        |  [[ [a],    |  [b,c,d],    |  [[e],[f]] ], | [  [u],   |   [v,w,x],   |  [[y],[z]]  ]]  |
        +-------------+--------------+---------------+-----------+--------------+-----------------+

    distance_field_weights : 1d array : default None
        determines the weight used in computing the distance between each item of `variable
        <ContentAddressableMemory.variable>` and the corresponding `memory_field
        <EpisodicMemoryMechanism_Memory_Fields>` of each entry in `memory <ContentAddressableMemory.memory>`; if all
        elements are identical, it is treated as a scalar coefficient on `distance <ContentAddressableMemory.distance>`
        (see `ContentAddressableMemory_Distance_Field_Weights` for additional details).

    distance_function : Distance or function : default Distance(metric=COSINE)
        function used during retrieval to compare `variable <ContentAddressableMemory.variable>` with entries in
        `memory <ContentAddressableMemory.memory>`.

    distance : float : default 0
        contains distance used for retrieval last cue to last entry returned in a given `context <Context>`.

    distances_by_field : array : default [0]
        contains array of distances between each `memory field <ContentAddressableMemory_Memory_Fields>`
        of the last cue and the corresponding ones of the last entry returned in a given `context <Context>`.

    distances_to_entries : array : default [0]
        contains array of distances between last cue retrieved in a given `context <Context>` an all entries at that
        time.

    memory_num_entries : int
        contains the number of entries in `memory <ContentAddressableMemory.memory>`.

    memory_num_fields : int
        contains the number of `memory fields <EpisodicMemoryMechanism_Memory_Fields>` in each entry of `memory
        <ContentAddressableMemory.memory>`.

    memory_field_shapes : array
        contains the shapes of each `memory field <EpisodicMemoryMechanism_Memory_Fields>`  in each entry of `memory
        <ContentAddressableMemory.memory>`.

    selection_function : OneHot or function
        function used during retrieval to evaluate the distances returned by `distance_function
        <ContentAddressableMemory.distance_function>` and select the item(s) to return.

    duplicate_entries_allowed : bool | OVERWRITE
        determines whether duplicate entries are allowed in `memory <ContentAddressableMemory.memory>`,
        as evaluated by `distance_function <ContentAddressableMemory.distance_function>` and `duplicate_threshold
        <ContentAddressableMemory.duplicate_threshold>`. (see `duplicate entries
        <ContentAddressableMemory_Duplicate_Entries>` for additional details).

    duplicate_threshold : float
        determines how similar `variable <ContentAddressableMemory.variable>` must be to an entry in `memory
        `<ContentAddressableMemory.memory>` based on `distance_function <ContentAddressableMemory.distance_function>`
        to be considered a duplicate (see `duplicate entries <ContentAddressableMemory_Duplicate_Entries>` for
        additional details).

    equidistant_entries_select:  RANDOM | OLDEST | NEWEST
        determines which entry is retrieved when duplicate entries are identified or are indistinguishable by the
        `distance_function <ContentAddressableMemory.distance_function>`.

    max_entries : int
        maximum number of entries allowed in `memory <ContentAddressableMemory.memory>`;  if storing a memory
        exceeds the number, the oldest memory is deleted.

    previous_value : ndarray
        state of the `memory <ContentAddressableMemory.memory>` prior to storing `variable
        <ContentAddressableMemory.variable>` in the current call.

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

    entry from `memory <ContentAddressableMemory.memory>` that best matches `variable
    <ContentAddressableMemory.variable>` : 2d array
        if no retrieval occurs, an appropriately shaped zero-valued array is returned.

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

                distance
                    see `distance <ContentAddressableMemory.distance>`

                    :default value: 0
                    :type: ``float``

                distance_field_weights
                    see `distance_field_weights <ContentAddressableMemory.distance_field_weights>`

                    :default value: [1]
                    :type: ``numpy.ndarray``

                 distance_function
                    see `distance_function <ContentAddressableMemory.distance_function>`

                    :default value: Distance(metric=COSINE)
                    :type: ``Function``

                distances_by_field
                    see `distances_by_field <ContentAddressableMemory.distances_by_field>`

                    :default value: [0]
                    :type: ``numpy.ndarray``

                distances_to_entries
                    see `distances_to_entries <ContentAddressableMemory.distances_to_entries>`

                    :default value: [0]
                    :type: ``numpy.ndarray``

                duplicate_entries_allowed
                    see `duplicate_entries_allowed <ContentAddressableMemory.duplicate_entries_allowed>`

                    :default value: False
                    :type: ``bool or OVERWRITE``

                duplicate_threshold
                    see `duplicate_threshold <ContentAddressableMemory.duplicate_threshold>`

                    :default value: 0
                    :type: ``float``

                equidistant_entries_select
                    see `equidistant_entries_select <ContentAddressableMemory.equidistant_entries_select>`

                    :default value: `RANDOM`
                    :type: ``str``

                memory_num_fields
                    see `memory_num_fields <ContentAddressableMemory.memory_num_fields>`

                    :default value: 1
                    :type: ``int``

                memory_field_shapes
                    see `memory_field_shapes <ContentAddressableMemory.memory_field_shapes>`

                    :default value: [1]
                    :type: ``numpy.ndarray``

               initializer
                    see `initializer <ContentAddressableMemory.initializer>`

                    :default value: None
                    :type: ``numpy.ndarray``

                max_entries
                    see `max_entries <ContentAddressableMemory.max_entries>`

                    :default value: 1000
                    :type: ``int``

                noise
                    see `noise <ContentAddressableMemory.noise>`

                    :default value: 0.0
                    :type: ``float``

                previous_value
                    see `previous_value <ContentAddressableMemory.previous_value>`

                    :default value: None
                    :type: ``numpy.ndarray``

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
        initializer = Parameter(None, pnl_internal=True)
        previous_value = Parameter(None, initializer='initializer')
        retrieval_prob = Parameter(1.0, modulable=True)
        storage_prob = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        # FIX: MAKE THESE ATTRIBUTES RATHER THAN PARAMETERS:
        memory_num_fields = Parameter(None, stateful=False, read_only=True)
        memory_field_shapes = Parameter(None, stateful=False, read_only=True)
        # FIX: --------------------
        distance_field_weights = Parameter([1], stateful=True, modulable=True, dependencies='initializer')
        duplicate_entries_allowed = Parameter(False, stateful=True)
        duplicate_threshold = Parameter(EPSILON, stateful=False, modulable=True)
        equidistant_entries_select = Parameter(RANDOM)
        rate = Parameter(1.0, modulable=True)
        noise = Parameter(
            0.0, modulable=True, aliases=[ADDITIVE_PARAM], setter=_noise_setter
        )
        max_entries = Parameter(1000)
        random_state = Parameter(None, loggable=False, getter=_random_state_getter, dependencies='seed')
        seed = Parameter(DEFAULT_SEED, modulable=True, fallback_default=True, setter=_seed_setter)
        distance_function = Parameter(Distance(metric=COSINE), stateful=False, loggable=False)
        selection_function = Parameter(OneHot(mode=MIN_INDICATOR), stateful=False, loggable=False)
        distance = Parameter(0, stateful=True, read_only=True)
        distances_by_field = Parameter([0], stateful=True, read_only=True)
        distances_to_entries = Parameter([0], stateful=True, read_only=True)

        def _validate_retrieval_prob(self, retrieval_prob):
            retrieval_prob = float(retrieval_prob)
            if not all_within_range(retrieval_prob, 0, 1):
                return f"must be a float in the interval [0,1]."

        def _validate_storage_prob(self, storage_prob):
            storage_prob = float(storage_prob)
            if not all_within_range(storage_prob, 0, 1):
                return f"must be a float in the interval [0,1]."

        def _validate_distance_field_weights(self, field_weights):
            if self.distance_field_weights._user_specified is True and self.initializer.default_value is not None:
                field_weights = np.array(field_weights)
                if not np.isscalar(field_weights) and field_weights.ndim != 1:
                    return f"must be a scalar or list or 1d array of scalars"
                fw_len = len(field_weights)
                num_fields = convert_all_elements_to_np_array(self.initializer.default_value).shape[1]
                if len(field_weights) not in {1, num_fields}:
                    return f"length ({fw_len}) must be same as number of fields " \
                           f"in entries of initializer ({num_fields})."
                if not np.any(field_weights):
                    warnings.warn(f"All weights in the 'distance_fields_weights' Parameter of {self._owner.name} "
                                  f"are set to '0', so all entries of its memory will be treated as duplicates.")

        def _validate_equidistant_entries_select(self, equidistant_entries_select):
            if equidistant_entries_select not in equidistant_entries_select_keywords:
                return f"must be {' or '.join(equidistant_entries_select_keywords)}."

        def _validate_duplicate_entries_allowed(self, duplicate_entries_allowed):
            if not isinstance(duplicate_entries_allowed, bool) and duplicate_entries_allowed != OVERWRITE:
                return f"must be a bool or 'OVERWRITE'."

        def _validate_initializer(self, initializer):
            pass

        def _parse_initializer(self, initializer):
            if initializer is not None:
                initializer = ContentAddressableMemory._enforce_memory_shape(initializer)
            return initializer

    @check_user_specified
    @tc.typecheck
    def __init__(self,
                 # FIX: REINSTATE WHEN 3.6 IS RETIRED:
                 # default_variable=None,
                 # retrieval_prob: Optional[Union[int, float]]=None,
                 # storage_prob: Optional[Union[int, float]]=None,
                 # rate: Optional[Union[int, float, list, np.ndarray]]=None,
                 # noise: Optional[Union[int, float, list, np.ndarray, callable]]=None,
                 # initializer:Optional[Union[list, np.ndarray]]=None,
                 # distance_field_weights:Optional[Union[list, np.ndarray]]=None,
                 # distance_function:Optional[Union[Distance, is_function_type]]=None,
                 # selection_function:Optional[Union[OneHot, is_function_type]]=None,
                 # duplicate_entries_allowed:Optional[Union[(bool, Literal[OVERWRITE]]]=None,
                 # duplicate_threshold:Optional[int]=None,
                 # equidistant_entries_select:Optional[Literal[Union[RANDOM, OLDEST, NEWEST]]]=None,
                 # max_entries:Optional[int]=None,
                 # seed:Optional[int]=None,
                 # params:Optional[Union[list, np.ndarray]]=None,
                 # owner=None,
                 # prefs:tc.optional(is_pref_set)=None):
                 default_variable=None,
                 retrieval_prob=None,
                 storage_prob=None,
                 rate=None,
                 noise=None,
                 initializer=None,
                 distance_field_weights=None,
                 distance_function=None,
                 selection_function=None,
                 duplicate_entries_allowed=None,
                 duplicate_threshold=None,
                 equidistant_entries_select=None,
                 max_entries=None,
                 seed=None,
                 params=None,
                 owner=None,
                 prefs:tc.optional(is_pref_set)=None):

        self._memory = []

        super().__init__(
            default_variable=default_variable,
            retrieval_prob=retrieval_prob,
            storage_prob=storage_prob,
            initializer=initializer,
            duplicate_entries_allowed=duplicate_entries_allowed,
            duplicate_threshold=duplicate_threshold,
            equidistant_entries_select=equidistant_entries_select,
            distance_function=distance_function,
            distance_field_weights=distance_field_weights,
            rate=rate,
            noise=noise,
            max_entries=max_entries,
            seed=seed,
            params=params,
            owner=owner,
            prefs=prefs,
        )

        if self.previous_value is not None:
            self.parameters.memory_num_fields.set(self.previous_value.shape[1], override=True)
            self.parameters.memory_field_shapes.set([item.shape for item in self.previous_value[0]], override=True)

    def _parse_selection_function_variable(self, variable, context=None):
        # this should be replaced in the future with the variable
        # argument when function ordering (and so ordering of parsers)
        # is made explicit
        distance_result = self.distance_function.parameters.value._get(context)
        # TEST PRINT:
        # print(distance_result, self.distance_function.defaults.value)
        return np.asfarray([
            distance_result if i == 0 else np.zeros_like(distance_result)
            for i in range(self.defaults.max_entries)
        ])

    def _validate(self, context=None):
        """Validate distance_function, selection_function and memory store"""
        distance_function = self.distance_function

        if self.get_previous_value(context) is not None:
            test_var = self.get_previous_value(context)[0]
        else:
            test_var = self.defaults.variable
        if isinstance(distance_function, type):
            distance_function = distance_function(default_variable=test_var)
            fct_msg = 'Function type'
        else:
            distance_function.defaults.variable = [test_var,test_var]
            distance_function._instantiate_value(context)
            fct_msg = 'Function'

        if (isinstance(distance_function, Distance)
                and distance_function.metric == COSINE
                and any([len(v)==1 for v in test_var])):
            warnings.warn(f"{self.__class__.__name__} is using {distance_function} with metric=COSINE and has "
                          f"at least one memory field that is a scalar (i.e., size=1), which will always produce "
                          f"a distance of 0 (the angle of scalars is not defined).")

        field_wts_homog = np.full(len(test_var),1).tolist()
        field_wts_heterog = np.full(len(test_var),range(0,len(test_var))).tolist()

        for granularity, field_weights in product(['full_entry', 'per_field'],[field_wts_homog, field_wts_heterog]):
            try:
                distance_result = self._get_distance(test_var, test_var, field_weights, granularity, context=context)
            except:
                raise FunctionError(f"{fct_msg} specified for {repr(DISTANCE_FUNCTION)} arg of "
                                    f"{self.__class__.__name__} ({distance_function}) must accept an array "
                                    f"with two lists or 1d arrays, or a 2d array, as its argument.")
            if granularity == 'full_entry' and not np.isscalar(distance_result):
                raise FunctionError(f"Value returned by {repr(DISTANCE_FUNCTION)} "
                                    f"({distance_function.__class__.__name__}) specified for "
                                    f"{self.__class__.__name__} must return a scalar if "
                                    f"{repr(DISTANCE_FIELD_WEIGHTS)} is not specified or is homogenous "
                                    f"(i.e., all elements are the same.")
            if granularity == 'per_field' and not len(distance_result)==len(field_weights):
                raise FunctionError(f"Value returned by {repr(DISTANCE_FUNCTION)} "
                                    f"({distance_function.__class__.__name__}) specified for "
                                    f"{self.__class__.__name__} must return an array "
                                    f"if {repr(DISTANCE_FIELD_WEIGHTS)} is a non-homogenous list or array"
                                    f"(i.e., not all elements are the same.")

        # FIX: 4/5/21 SHOULD VALIDATE NOISE AND RATE HERE AS WELL?

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

    @handle_external_context()
    def _update_default_variable(self, new_default_variable, context=None):
        """Override method on parent (StatefulFunction) since it can't handle arbitrarily-shaped fields"""
        if not self.parameters.initializer._user_specified and self.parameters.variable._user_specified:
            new_default_variable = self.parameters.variable.default_value
        super(StatefulFunction, self)._update_default_variable(new_default_variable, context=context)

    def _initialize_previous_value(self, initializer, context=None):
        """Ensure that initializer is appropriate for assignment as memory attribute and assign as previous_value

        If specified and it is the first entry:
        - set memory_num_fields and memory_field_shapes based on initializer
        - use to set previous_value (and return previous_value)
            (must be done here rather than in validate_params as it is needed to initialize previous_value
        """

        if initializer is None or convert_all_elements_to_np_array(initializer).size == 0:
            return None

        # FIX: HOW DOES THIS RELATE TO WHAT IS DONE IN __init__()?
        # Set memory fields shapes if this is the first entry
        self.parameters.memory_num_fields.set(initializer.shape[1],
                                              context=context, override=True)
        self.parameters.memory_field_shapes.set([item.shape for item in initializer[0]],
                                               context=context, override=True)
        self.parameters.previous_value.set(None, context, override=True)

        for entry in initializer:
            # Store each item, which also validates it by call to _validate_entry()
            if not self._store_memory(entry, context):
                warnings.warn(f"Attempt to initialize memory of {self.__class__.__name__} with an entry ({entry}) "
                              f"that is identical to an existing one while 'duplicate_entries_allowed'==False; "
                              f"that entry has been skipped")
        previous_value = self._memory
        self.parameters.previous_value.set(previous_value, context, override=True)
        return previous_value

    def _instantiate_attributes_before_function(self, function=None, context=None):
        self._initialize_previous_value(self.parameters.initializer._get(context), context)

        if isinstance(self.distance_function, type):
            self.distance_function = self.distance_function(context=context)

        if isinstance(self.selection_function, type):
            self.selection_function = self.selection_function(context=context)

    @handle_external_context(fallback_most_recent=True)
    def reset(self, new_value=None, context=None):
        """
        reset(<new_dictionary> default={})

        Clears the memory in `previous_value <ContentAddressableMemory.previous_value>`.

        If **new_value** is passed into reset or if the `initializer <ContentAddressableMemory.initializer>`
        attribute contains a value besides [], then that value is used to start the new memory in `previous_value
        <ContentAddressableMemory.previous_value>`. Otherwise, the new `previous_value
        <ContentAddressableMemory.previous_value>` memory starts out as None.

        `value <ContentAddressableMemory.value>` takes on the same value as
        `previous_value <ContentAddressableMemory.previous_value>`.
        """

        if new_value is not None:
            value = self._initialize_previous_value(ContentAddressableMemory._enforce_memory_shape(new_value),
                                                    context=context)

        else:
            # no arguments were passed in -- use current values of initializer attributes
            initializer = self._get_current_parameter_value("initializer", context)
            if initializer is not None:
                # set previous_value to initializer and get value
                value = self._initialize_previous_value(initializer, context=context)
            else:
                # no initializer, so clear previous_value and set value to None
                self.parameters.previous_value._get(context).clear()
                value = None

        self.parameters.value.set(value, context, override=True)
        return value

    def _function(self,
                 variable:Optional[Union[list, np.array]]=None,
                 context=None,
                 params=None,
                 ) -> list:
        """
        Return entry in `memory <ContentAddressableMemory.memory>` that best matches `variable
        <ContentAddressableMemory.variable>`, then add `variable <ContentAddressableMemory.variable>` to `memory
        <ContentAddressableMemory.memory>` (see `above <ContentAddressableMemory_Execution>` for additional details).

        Arguments
        ---------

        variable : list or 2d array : default class_defaults.variable
           used to retrieve an entry from `memory <ContentAddressableMemory.memory>`, and then stored there.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        value of entry that best matches `variable <ContentAddressableMemory.variable>`  : 1d array
        """

        # Enforce variable to be shape of an entry (1d for ragged fields or 2d for regular ones)
        # - note: this allows entries with a single field to be specified as a 1d regular array
        #         (i.e., without enclosing it in an outer list or array), which are converted to a 2d array
        variable = convert_all_elements_to_np_array(variable)
        if variable.dtype != object and variable.ndim==1:
            variable = np.expand_dims(variable, axis=0)

        retrieval_prob = np.array(self._get_current_parameter_value(RETRIEVAL_PROB, context)).astype(float)
        storage_prob = np.array(self._get_current_parameter_value(STORAGE_PROB, context)).astype(float)

        # get random state
        random_state = self._get_current_parameter_value('random_state', context)

        # get memory field weights (which are modulable)
        distance_field_weights = self._get_current_parameter_value('distance_field_weights', context)

        # If this is an initialization run, leave memory empty (don't want to count it as an execution step),
        # but set entry size and then return current value (variable[1]) for validation.
        if self.is_initializing:
            return variable

        # Set memory fields sizes and total size if this is the first entry
        if self.parameters.previous_value._get(context) is None:
            self.parameters.memory_num_fields.set(len(variable), context=context, override=True)
            self.parameters.memory_field_shapes.set([item.shape for item in variable], context=context, override=True)

        # Retrieve entry from memory that best matches variable
        if retrieval_prob == 1.0 or (retrieval_prob > 0.0 and retrieval_prob > random_state.uniform()):
            entry = self.get_memory(variable, distance_field_weights, context).copy()
        else:
            # QUESTION: SHOULD IT RETURN ZERO VECTOR OR NOT RETRIEVE AT ALL (LEAVING VALUE AND OutputPort FROM LAST TRIAL)?
            #           CURRENT PROBLEM WITH LATTER IS THAT IT CAUSES CRASH ON INIT, SINCE NOT OUTPUT_PORT
            #           SO, WOULD HAVE TO RETURN ZEROS ON INIT AND THEN SUPPRESS AFTERWARDS, AS MOCKED UP BELOW
            entry = self.uniform_entry(0, context)

        # Store variable in memory
        if storage_prob == 1.0 or (storage_prob > 0.0 and storage_prob > random_state.uniform()):
            self._store_memory(variable, context)

        return entry

    def _validate_entry(self, entry:Union[list, np.ndarray], context) -> None:

        field_shapes = self.parameters.memory_field_shapes.get(context)
        num_fields = self.parameters.memory_num_fields.get(context)

        if not entry.ndim:
            # IMPLEMENTATION NOTE:  Remove this if/when >2d arrays are supported more generally in PsyNeuLink
            raise FunctionError(f"Attempt to store and/or retrieve an entry in {self.__class__.__name__} that has "
                                f"has dimensions ({entry}); must be a list or 1d or 2d array.")

        if entry.ndim >2:
            # IMPLEMENTATION NOTE:  Remove this if/when >2d arrays are supported more generally in PsyNeuLink
            raise FunctionError(f"Attempt to store and/or retrieve an entry in {self.__class__.__name__} ({entry}) "
                                f"that has more than 2 dimensions ({entry.ndim});  try flattening innermost ones.")

        if not len(entry) == num_fields:
            raise FunctionError(f"Attempt to store and/or retrieve entry in {self.__class__.__name__} ({entry}) "
                                f"that has an incorrect number of fields ({len(entry)}; should be {num_fields}).")

        owner_name = f'of {self.owner.name}' if self.owner else ''
        for i, field in enumerate(entry):
            field = np.array(field)
            # IMPLEMENTATION NOTE:  Remove requirement field.ndim==1  if/when >2d arrays are supported more generally
            if field.ndim != 1 or field.shape != field_shapes[i]:
                raise FunctionError(f"Field {i} of entry ({entry}) has incorrect shape ({field.shape}) "
                                    f"for memory of '{self.name}{owner_name}';  should be: {field_shapes[i]}.")

    def uniform_entry(self, value:Union[int, float], context) -> np.ndarray:
        return [np.full(i,value) for i in self.parameters.memory_field_shapes._get(context)]

    @handle_external_context()
    def get_memory(self, cue:Union[list, np.ndarray], field_weights=None, context=None) -> np.ndarray:
        """get_memory(query_key, context=None)

        Retrieve entry from `memory <ContentAddressableMemory.memory>` based on `distance_function
        <ContentAddressableMemory.distance_function>` and `selection_function
        <ContentAddressableMemory.selection_function>`.

        Arguments
        ---------
        cue : list or 2d array
          must have same number and shapes of fields as existing entries in `memory <ContentAddressableMemory.memory>`.

        Returns
        -------
        entry retrieved : 2d array
          if no retrieval occurs, returns appropriately shaped zero-valued array.

        """
        # QUESTION: SHOULD IT RETURN ZERO VECTOR OR NOT RETRIEVE AT ALL (LEAVING VALUE AND OutputPort FROM LAST TRIAL)?
        #           ALSO, SHOULD PROBABILISTIC SUPPRESSION OF RETRIEVAL BE HANDLED HERE OR function (AS IT IS NOW).

        # FIX: RETRIEVE BASED ON SIMILARITY WITHIN EACH FIELD WEIGHTE BY distance_field_weights

        _memory = self.parameters.previous_value._get(context)
        # if no entries in memory, return the zero vector
        if _memory is None:
            return self.uniform_entry(0, context)

        cue = convert_all_elements_to_np_array(cue)
        self._validate_entry(cue, context)

        # Get mean of field-wise distances between cue each entry in memory
        distances_to_entries = []
        for entry in _memory:
            distances_to_entries.append(self._get_distance(cue, entry, field_weights, 'full_entry', context))

        # Get the best-match(es) in memory based on selection_function and return as non-zero value(s) in an array
        selection_array = self.selection_function(distances_to_entries, context=context)
        indices_of_selected_items = np.flatnonzero(selection_array)

        # Single entry identified
        if len(indices_of_selected_items)==1:
            index_of_selected_item = int(np.flatnonzero(selection_array))

        # More than one entry identified
        else:
            # Check for any duplicate entries in matches and, if they are not allowed, return zeros
            if (not self.duplicate_entries_allowed
                    and any(self._is_duplicate(_memory[i],_memory[j], field_weights, context)
                            for i, j in combinations(indices_of_selected_items, 2))):
                warnings.warn(f"More than one entry matched cue ({cue}) in memory for {self.name}"
                              f"{'of ' + self.owner.name if self.owner else ''} even though "
                              f"{repr('duplicate_entries_allowed')} is False; zeros returned as retrieved item.")
                return self.uniform_entry(0, context)
            if self.equidistant_entries_select == RANDOM:
                random_state = self._get_current_parameter_value('random_state', context)
                index_of_selected_item = random_state.choice(indices_of_selected_items)
            elif self.equidistant_entries_select == OLDEST:
                index_of_selected_item = indices_of_selected_items[0]
            elif self.equidistant_entries_select == NEWEST:
                index_of_selected_item = indices_of_selected_items[-1]
            else:
                assert False, f"PROGRAM ERROR:  bad specification ({repr(self.equidistant_entries_select)}) for " \
                              f"'equidistant_entries_select' parameter of {self.name}" \
                              f"{'for ' + self.owner.name if self.owner else ''}"

        best_match = _memory[index_of_selected_item]
        best_match_distances = self._get_distance(cue,best_match,field_weights, 'per_field',context)
        self.parameters.distance.set(distances_to_entries[index_of_selected_item], context, override=True)
        self.parameters.distances_by_field.set(best_match_distances,override=True)
        self.parameters.distances_to_entries.set(distances_to_entries, context,override=True)

        # Return entry
        return best_match

    def _store_memory(self, entry:Union[list, np.ndarray], context) -> bool:
        """Add an entry to `memory <ContentAddressableMemory.memory>`

        Arguments
        ---------
        entry : list or 2d array
            should be a list or 2d array containing 1d arrays (fields) each of which should be list or at least a 1d
            array; scalars, 1d and simple 2d arrays are allowed, and are interpreted as a single entry with a single
            field, which is converted to a 3d array. If any entries already exist in `memory
            <ContentAddressableMemory.memory>`, then both the number of fields and their shapes must match existing
            entries (contained in the `memory_num_fields <ContentAddressableMemory.memory_num_fields>` and
            `memory_field_shapes <ContentAddressableMemory.memory_field_shapes>` attributes, respectively).  All
            elements of all entries are converted to np.arrays.

            .. technical_note::
               this method supports adding entries with items in each field that are greater than 1d for potential
               future use (see format_for_storage() below); however they are currently rejected in _validate_entry
               as currently they may produce unexpected results (by returning entries that are greater than 2d).
        """

        self._validate_entry(entry, context)
        # convert all fields and entry itself to arrays
        entry = convert_all_elements_to_np_array(entry)

        num_fields = self.parameters.memory_num_fields._get(context)
        field_weights = self.parameters.distance_field_weights._get(context)

        # execute noise if it is a function
        noise = self._try_execute_param(self._get_current_parameter_value(NOISE, context), entry, context=context)
        if noise is not None:
            try:
                entry = entry + noise
            except:
                raise FunctionError(f"'noise' for '{self.name}' of '{self.owner.name}' "
                                    f"not appropriate shape (single number or array of length {num_fields}.")
        existing_entries = self.parameters.previous_value._get(context)

        def format_for_storage(entry:np.ndarray) -> np.ndarray:
            """Format an entry to be added to memory
            Returns entry formatted to match the shape of `memory <EpisodicMemoryMechanism.memory>`,
            so that it can be appended (or, if it is the first, simply assigned) to memory:
            - if entry is a regular array (all fields [axis 0 items] have the same shape),
                returns object with ndim = entry.ndim + 1 (see `technical_note <ContentAddressableMemory_Shapes>` above)
            - if the entry is a ragged array (fields [axis 0 items] have differing shapes),
                returns 2d object with dtype=object.
            """
            # Ragged array (i.e., fields of different shapes)
            if entry.ndim == 1 and entry.dtype==object:
                shape = (1, num_fields)
            # Regular array (all fields have the same shapes)
            elif entry.ndim >= 2:
                # Note: if greater ndim>2, item in each field is >1d
                shape = (1, num_fields, entry.shape[1])
            else:
                raise ContentAddressableMemory(f"Unrecognized format for entry to be stored in {self.name}: {entry}.")
            return np.atleast_3d(entry).reshape(shape)

        if existing_entries is not None:
            # Check for matches of entry with existing entries
            matches = [m for m in existing_entries if len(m) and self._is_duplicate(entry, m, field_weights, context)]

            # If duplicate entries are not allowed and entry matches any existing entries, don't store
            if matches and self.duplicate_entries_allowed is False:
                storage_succeeded = False

            # If duplicate_entries_allowed is True or OVERWRITE, replace value for matching entry:
            # FIX: SHOULD BE OVERWRITE or False
            elif matches and self.duplicate_entries_allowed == OVERWRITE:
                if len(matches)>1:
                    # If there is already more than one duplicate, raise error as it is not clear what to overwrite
                    raise FunctionError(f"Attempt to store item ({entry}) in {self.name} "
                                        f"with 'duplicate_entries_allowed'='OVERWRITE' "
                                        f"when there is more than one matching entry in its memory; "
                                        f"'duplicate_entries_allowed' may have previously been set to 'True'")
                try:
                    index = existing_entries.index(entry)
                except AttributeError:
                    index = [i for i,e in enumerate(existing_entries) if np.all(e == matches[0])][0]
                except ValueError:
                    index = existing_entries.tolist().index(entry)
                existing_entries[index] = entry
                storage_succeeded = True
            else:
                # Add to existing entries
                existing_entries = np.append(existing_entries, format_for_storage(entry), axis=0)
                storage_succeeded = True

        else:
            # No entries yet, so add new one
            existing_entries = format_for_storage(entry)
            storage_succeeded = True

        if len(existing_entries) > self.max_entries:
            existing_entries = np.delete(existing_entries,0,axis=0)

        self.parameters.previous_value._set(existing_entries,context)
        self._memory = existing_entries

        return storage_succeeded

    def _get_distance(self, cue:Union[list, np.ndarray],
                      candidate:Union[list, np.ndarray],
                      field_weights:Union[list, np.ndarray],
                      granularity:str,
                      # granularity:Literal[Union['full_entry', 'per_field']],
                      context) -> Union[float, np.ndarray]:
        """Get distance of cue from candidate using `distance_function <ContentAddressableMemory.distance_function>`.

        - If **granularity**=='full_entry':
            returns *single scalar distance* computed over full **cue** and **candidate** entries if all elements of
                **fields_weights** are equal (i.e., it is a homogenous array);  otherwise it is used to weight the
                the distance computed between each field of **cue** and corresponding one of **candidate**,
                when computing their mean field-wise distances.
        - if **granularity**=='per_field':
            returns *array of distances* computed field-wise (hadamard) for **cue** and **candidate**,
            weighted by **field_weights**.

        .. note::
           granularity is only used for reporting field-wise distances in `distances_by_field
           <ContentAddressableMemory.distances_by_field>`, and not used to determine retrieval or storage

        :returns
            scalar if **granularity**=='full_entry';
            array if **granularity**=='per_fields'
        """

        # Get distance function and params
        distance_fct = self.parameters.distance_function._get(context)
        num_fields = self.parameters.memory_num_fields._get(context) or len(field_weights)
        if field_weights is None:
            # Could be from get_memory called from COMMAND LINE without field_weights
            field_weights = self._get_current_parameter_value('distance_field_weights', context)
        field_weights = np.atleast_1d(field_weights)

        if granularity == 'per_field':
            # Note: this is just used for reporting, and not determining storage or retrieval

            # Replace None's with 0 to allow multiplication
            distances_by_field = np.array([distance_fct([cue[i], candidate[i]])
                                           for i in range(num_fields)]
                                          ) * np.array([f if f is not None else 0 for f in field_weights])
            # If field_weights is scalar, splay out as array of length num_fields so can iterate through all of them
            if len(field_weights)==1:
                field_weights = np.full(num_fields, field_weights[0])
            # Replace 0's with None's for fields with None in field_weights
            distances_by_field = np.array([distances_by_field[i]
                                           if f is not None else None for i,f in enumerate(field_weights)])
            return distances_by_field

        elif granularity == 'full_entry':
            # Use first element as scalar if it is a homogenous array (i.e., all elements are the same)
            field_weights = field_weights[0] if np.all(field_weights[0]==field_weights) else field_weights
            distance_by_fields = not np.isscalar(field_weights)
            if distance_by_fields:
                num_non_zero_fields = len([fw for fw in field_weights if fw])
                # Get mean of field-wise distances between cue each entry in memory, weighted by field_weights
                distance = np.sum([distance_fct([cue[i], candidate[i]]) * field_weights[i]
                                   for i in range(num_fields) if field_weights[i]]) / num_non_zero_fields
            else:
                # Get distances between entire cue vector and all that for each entry in memory
                #    Note: in this case, field_weights is just a scalar coefficient
                distance = distance_fct([np.hstack(cue), np.hstack(candidate)]) * field_weights
            return distance

        else:
            assert False, f"PROGRAM ERROR: call to 'ContentAddressableMemory.get_distance()' method " \
                          f"with invalid 'granularity' argument ({granularity});  " \
                          f"should be 'full_entry' or 'per_field."

    def _parse_distance_function_variable(self, variable):
        return convert_to_np_array([variable[0], variable[0]])

    @classmethod
    def _enforce_memory_shape(cls, memory):
        # Enforce memory to be 2d for ragged fields or 3d for regular ones
        # - note: this also allows memory (e.g., via initializer or reset) to be specified with a single entry
        #         (i.e., without enclosing it in an outer list or array)
        memory = convert_all_elements_to_np_array(memory)
        memory = np.atleast_2d(memory)
        if memory.dtype != object and memory.ndim==2:
            memory = np.expand_dims(memory, axis=0)
        return memory

    def _is_duplicate(self, entry1:np.ndarray, entry2:np.ndarray, field_weights:np.ndarray, context) -> bool:
        """Determines whether two entries are duplicates
         Duplicates are treated as ones with a distance within the tolerance specified by duplicate_threshold.
         Distances are computed using distance_field_weights.
         """
        if (self._get_distance(entry1, entry2, field_weights, 'full_entry', context)
                <= self.parameters.duplicate_threshold.get(context)):
            return True
        return False

    @handle_external_context()
    def add_to_memory(self, entries:Union[list, np.ndarray], context=None):
        """Add one or more entries into `memory <ContentAddressableMememory.memory>`

        Arguments
        ---------

        entries : list or array
            a single entry (list or array) or list or array of entries, each of which must be a valid entry;
            each must have the same number of and shapes of corresponding fields;
            items are added to memory in the order listed.
        """
        entries = self._parse_memories(entries, 'add_to_memory', context)

        for entry in entries:
            self._store_memory(entry, context)

    @handle_external_context()
    def delete_from_memory(self,
                           entries:Union[list, np.ndarray],
                           fields:Optional[Union[int, list]]= None,
                           context=None):
        """Delete one or more entries from `memory <ContentAddressableMememory.memory>`

        Arguments
        ---------

        memories : list or array
            a single entry (list or 2d array) or list or array of entries,
            each of which must be a valid entry (i.e. same number of fields and shapes of each
            as entries already in `memory <ContentAddressableMemory.memory>`.

        fields :  int or list : default None
            if None, delete all entries in `memory <ContentAddressableMemory.memory>` that are identical
            to any of the **memories** specified; if int or list, delete all entries with the same values as those
            in the field(s) specified.
        """
        memories = self._parse_memories(entries, 'add_to_memory', context)
        # FIX: ??IS THIS NEEDED (IS IT JUST A HOLDOVER FROM KEYS OR NEEDED FOR LIST-T0-LIST COMPARISON BELOW?):
        entries = [list(m) for m in memories]
        fields = convert_to_list(fields)

        existing_memory = self.parameters.previous_value._get(context)
        pruned_memory = existing_memory.copy()
        for entry, memory in product(entries, existing_memory):
            if (np.all(entry == memory)
                    or fields and all(entry[f] == memory[f] for f in fields)):
                pruned_memory = np.delete(pruned_memory, pruned_memory.tolist().index(memory.tolist()), axis=0)
        self._memory = convert_all_elements_to_np_array(pruned_memory)
        self.parameters.previous_value._set(self._memory, context)

    def _parse_memories(self, entries, method, context=None):
        """Parse passing of single vs. multiple memories, validate memories, and return ndarray
        Used by add_to_memory and delete_from_memory
        """
        memories = convert_all_elements_to_np_array(entries)
        if not 1 <= memories.ndim <= 3:
            was_str = f'(was {memories.ndim}d)' if memories.ndim else '(was scalar)'
            raise FunctionError(f"The 'memories' arg for {method} method of "
                                f"must be a list or array containing 1d or 2d arrays {was_str}.")

        # if (memories.ndim == 2 and memories.dtype != object) or (memories.ndim == 1 and memories.dtype == object):
        if (memories.ndim == 2 and memories.dtype != object) or (memories.ndim == 1):
            memories = np.expand_dims(memories,axis=0)

        for entry in memories:
            self._validate_entry(entry, context)

        return memories

    @property
    def memory(self):
        """Return entries in self._memory as lists in an outer np.array;
           use np.array for multi-line printout
       """
        try:
            return self._memory
        except:
            return np.array([])

    @property
    def memory_num_entries(self):
        """Return number of entries in self._memory.
       """
        return len(self._memory)


KEYS = 0
VALS = 1


class DictionaryMemory(MemoryFunction):  # ---------------------------------------------------------------------
    """
    DictionaryMemory(                                \
        default_variable=None,                       \
        retrieval_prob=1.0                           \
        storage_prob=1.0                             \
        rate=None,                                   \
        noise=0.0,                                   \
        initializer=None,                            \
        distance_function=Distance(metric=COSINE),   \
        selection_function=OneHot(mode=MIN_VAL),     \
        equidistant_keys_select=RANDOM,              \
        duplicate_keys=False,                        \
        max_entries=None,                            \
        params=None,                                 \
        owner=None,                                  \
        prefs=None,                                  \
        )

    .. _DictionaryMemory:

    Implement a configurable, dictionary-style storage and retrieval of key-value pairs, in which storage
    is determined by `storage_prob <DictionaryMemory.storage_prob>`, and retrieval of items is
    determined by `distance_function <DictionaryMemory.distance_function>`, `selection_function
    <DictionaryMemory.selection_function>`, and `retrieval_prob <DictionaryMemory.retrieval_prob>`.
    Keys and values may have different lengths, and values may vary in length from entry to entry, but all keys
    must be the same length. Duplicate keys can be allowed, disallowed, or overwritten using `duplicate_keys
    <DictionaryMemory.duplicate_keys>`), and how selection is made among duplicate keys or ones
    indistinguishable by the `distance_function <DictionaryMemory.distance_function>` can be specified
    using `equidistant_keys_select <DictionaryMemory.equidistant_keys_select>`.

    The class also provides methods for directly retrieving an entry (`get_memory
    <DictionaryMemory.get_memory>`), and adding (`add_to_memory <DictionaryMemory.add_to_memory>`)
    and deleting (`delete_from_memory <DictionaryMemory.delete_from_memory>`) one or more entries.

    .. _DictionaryMemory_Structure:

    Structure
    ---------

    An item is stored and retrieved as a 2d array containing a key-value pair ([[key][value]]).  A 3d array of such
    pairs can be used to initialize the contents of memory by providing it in the **initialzer** argument of the
    DictionaryMemory's constructor, or in a call to its `reset  <DictionaryMemory.reset>`
    method.  The current contents of the memory can be inspected using the `memory <DictionaryMemory.memory>`
    attribute, which returns a list containing the current entries, each as a 2 item list containing a key-value pair.

    .. _DictionaryMemory_Execution:

    Execution
    ---------

    When `function <DictionaryMemory.function>` is executed, it first retrieves the
    item in `memory <DictionaryMemory.memory>` with the key that most closely matches the key of the item
    (key-value pair) in the call, stores the latter in memory, and returns the retrieved item (key-value pair).
    If the key of the pair in the call is an exact match of a key in memory and `duplicate_keys
    <DictionaryMemory.duplicate_keys>` is False, then the matching item is returned, but the
    pair in the call is not stored. These steps are described in more detail below:

    * First, with probability `retrieval_prob <DictionaryMemory.retrieval_prob>`, an entry is retrieved from
      `memory <DictionaryMemory.memory>` that has a key that is closest to the one in the call (first item of
      `variable <DictionaryMemory.variable>`), as determined by the `distance_function
      <DictionaryMemory.distance_function>` and `selection_function
      <DictionaryMemory.selection_function>`.  The `distance_function
      <DictionaryMemory.distance_function>` generates a list of distances of each key in memory from the
      one in the call;  the `selection_function <DictionaryMemory.selection_function>` then determines which
      to select ones for consideration.  If more than one entry from memory is identified, `equidistant_keys_select
      <DictionaryMemory.equidistant_keys_select>` is used to determine which to retrieve.  If no retrieval
      occurs, an appropriately shaped zero-valued array is assigned as the retrieved memory (and returned by the
      `function <DictionaryMemory.function>`.
    ..
    * After retrieval, the key-value pair in the call (`variable <DictionaryMemory.variable>`) is stored in
      `memory <DictionaryMemory.memory>` with probability `storage_prob <DictionaryMemory.storage_prob>`.
      If the key (`variable <DictionaryMemory.variable>`\\[0]) is identical to one already in `memory
      <DictionaryMemory.memory>` and `duplicate_keys <DictionaryMemory.duplicate_keys>`
      is set to False, storage is skipped; if it is set to *OVERWRITE*, the value of the key in memory is replaced
      with the one in the call.  If **rate** and/or **noise** arguments are specified in the
      constructor, it is applied to the key before storing, as follows:

    .. math::
        variable[1] * rate + noise

    If the number of entries exceeds `max_entries <DictionaryMemory.max_entries>, the first (oldest) item in
    memory is deleted.

    Arguments
    ---------

    default_variable : list or 2d array : default class_defaults.variable
        specifies a template for the key and value entries of the dictionary;  list must have two entries, each
        of which is a list or array;  first item is used as key, and second as value entry of dictionary.

    retrieval_prob : float in interval [0,1] : default 1.0
        specifies probability of retrieiving a key from `memory <DictionaryMemory.memory>`.

    storage_prob : float in interval [0,1] : default 1.0
        specifies probability of adding `variable <DictionaryMemory.variable>` to `memory
        <DictionaryMemory.memory>`.

    rate : float, list, or array : default 1.0
        specifies a value used to multiply key (first item of `variable <DictionaryMemory.variable>`) before
        storing in `memory <DictionaryMemory.memory>` (see `rate <DictionaryMemory.noise> for details).

    noise : float, list, array, or Function : default 0.0
        specifies a random value added to key (first item of `variable <DictionaryMemory.variable>`) before
        storing in `memory <DictionaryMemory.memory>` (see `noise <DictionaryMemory.noise> for details).

    initializer : 3d array or list : default None
        specifies an initial set of entries for `memory <DictionaryMemory.memory>`. It must be of the following
        form: [[[key],[value]], [[key],[value]], ...], such that each item in the outer dimension (axis 0)
        is a 2d array or list containing a key and a value pair for that entry. All of the keys must be 1d arrays or
        lists of the same length.

    distance_function : Distance or function : default Distance(metric=COSINE)
        specifies the function used during retrieval to compare the first item in `variable
        <DictionaryMemory.variable>` with keys in `memory <DictionaryMemory.memory>`.

    selection_function : OneHot or function : default OneHot(mode=MIN_VAL)
        specifies the function used during retrieval to evaluate the distances returned by `distance_function
        <DictionaryMemory.distance_function>` and select the item to return.

    equidistant_keys_select:  RANDOM | OLDEST | NEWEST : default RANDOM
        specifies which item is chosen for retrieval if two or more keys have the same distance from the first item of
        `variable  <DictionaryMemory.variable>`.

    duplicate_keys : bool | OVERWRITE : default False
        specifies whether entries with duplicate keys are allowed in `memory <DictionaryMemory.memory>`
        (see `duplicate_keys <DictionaryMemory.duplicate_keys for additional details>`).

    max_entries : int : default None
        specifies the maximum number of entries allowed in `memory <DictionaryMemory.memory>`
        (see `max_entries <DictionaryMemory.max_entries for additional details>`).

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
        1st item (variable[0] is the key used to retrieve an enrtry from `memory <DictionaryMemory.memory>`,
        and 2nd item (variable[1]) is the value of the entry, paired with key and added to the `memory
        <DictionaryMemory.memory>`.

    key_size : int
        length of keys in `memory <DictionaryMemory.memory>`.

    val_size : int
        length of values in `memory <DictionaryMemory.memory>`.

    retrieval_prob : float in interval [0,1]
        probability of retrieving a value from `memory <DictionaryMemory.memory>`.

    storage_prob : float in interval [0,1]
        probability of adding `variable <DictionaryMemory.variable>` to `memory
        <DictionaryMemory.memory>`;

    rate : float or 1d array
        value applied multiplicatively to key (first item of `variable <DictionaryMemory.variable>`) before
        storing in `memory <DictionaryMemory.memory>` (see `rate <Stateful_Rate>` for additional details).

    noise : float, 1d array or Function
        value added to key (first item of `variable <DictionaryMemory.variable>`) before storing in
        `memory <DictionaryMemory.memory>` (see `noise <Stateful_Noise>` for additional details).

    initializer : 3d array
        initial set of entries for `memory <DictionaryMemory.memory>`; each is a 2d array with a key-value pair.

    memory : list
        list of key-value pairs containing entries in DictionaryMemory:
        [[[key 1], [value 1]], [[key 2], value 2]]...]

    distance_function : Distance or function : default Distance(metric=COSINE)
        function used during retrieval to compare the first item in `variable <DictionaryMemory.variable>`
        with keys in `memory <DictionaryMemory.memory>`.

    selection_function : OneHot or function : default OneHot(mode=MIN_VAL)
        function used during retrieval to evaluate the distances returned by `distance_function
        <DictionaryMemory.distance_function>` and select the item(s) to return.

    previous_value : 1d array
        state of the `memory <DictionaryMemory.memory>` prior to storing `variable
        <DictionaryMemory.variable>` in the current call.

    duplicate_keys : bool | OVERWRITE
        determines whether entries with duplicate keys are allowed in `memory <DictionaryMemory.memory>`.
        If True (the default), items with keys that are the same as ones in memory can be stored;  on retrieval, a
        single one is selected based on `equidistant_keys_select <DictionaryMemory.equidistant_keys_select>`.
        If False, then an attempt to store and item with a key that is already in `memory
        <DictionaryMemory.memory>` is ignored, and the entry already in memory with that key is retrieved.
        If a duplicate key is identified during retrieval (e.g., **duplicate_keys** is changed from True to
        False), a warning is issued and zeros are returned.  If *OVERWRITE*, then retrieval of a cue with an identical
        key causes the value at that entry to be overwritten with the new value.

    equidistant_keys_select:  RANDOM | OLDEST | NEWEST
        deterimines which entry is retrieved when duplicate keys are identified or are indistinguishable by the
        `distance_function <DictionaryMemory.distance_function>`.

    max_entries : int
        maximum number of entries allowed in `memory <DictionaryMemory.memory>`;  if storing a memory
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

    value and key of entry that best matches first item of `variable <DictionaryMemory.variable>`  : 2d array
        if no retrieval occures, an appropriately shaped zero-valued array is returned.

    """

    componentName = DictionaryMemory_FUNCTION

    class Parameters(StatefulFunction.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <DictionaryMemory.variable>`

                    :default value: [[0], [0]]
                    :type: ``list``

                distance_function
                    see `distance_function <DictionaryMemory.distance_function>`

                    :default value: `Distance`(metric=cosine)
                    :type: `Function`

                duplicate_keys
                    see `duplicate_keys <DictionaryMemory.duplicate_keys>`

                    :default value: False
                    :type: ``bool``

                equidistant_keys_select
                    see `equidistant_keys_select <DictionaryMemory.equidistant_keys_select>`

                    :default value: `RANDOM`
                    :type: ``str``

                key_size
                    see `key_size <DictionaryMemory.key_size>`

                    :default value: 1
                    :type: ``int``

                max_entries
                    see `max_entries <DictionaryMemory.max_entries>`

                    :default value: 1000
                    :type: ``int``

                noise
                    see `noise <DictionaryMemory.noise>`

                    :default value: 0.0
                    :type: ``float``

                random_state
                    see `random_state <DictionaryMemory.random_state>`

                    :default value: None
                    :type: ``numpy.random.RandomState``

                rate
                    see `rate <DictionaryMemory.rate>`

                    :default value: 1.0
                    :type: ``float``

                retrieval_prob
                    see `retrieval_prob <DictionaryMemory.retrieval_prob>`

                    :default value: 1.0
                    :type: ``float``

                selection_function
                    see `selection_function <DictionaryMemory.selection_function>`

                    :default value: `OneHot`(mode=MIN_INDICATOR)
                    :type: `Function`

                storage_prob
                    see `storage_prob <DictionaryMemory.storage_prob>`

                    :default value: 1.0
                    :type: ``float``

                val_size
                    see `val_size <DictionaryMemory.val_size>`

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
        noise = Parameter(
            0.0, modulable=True, aliases=[ADDITIVE_PARAM], setter=_noise_setter
        )
        max_entries = Parameter(1000)
        random_state = Parameter(None, loggable=False, getter=_random_state_getter, dependencies='seed')
        seed = Parameter(DEFAULT_SEED, modulable=True, fallback_default=True, setter=_seed_setter)

        distance_function = Parameter(Distance(metric=COSINE), stateful=False, loggable=False)
        selection_function = Parameter(OneHot(mode=MIN_INDICATOR), stateful=False, loggable=False)


    @check_user_specified
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
            seed=seed,
            params=params,
            owner=owner,
            prefs=prefs,
        )

        if self.previous_value.size != 0:
            self.parameters.key_size.set(len(self.previous_value[KEYS][0]))
            self.parameters.val_size.set(len(self.previous_value[VALS][0]))

    def _parse_distance_function_variable(self, variable):
        # actual used variable in execution (get_memory) checks distance
        # between key and key, not key and val as implied in _validate
        return convert_to_np_array([variable[KEYS], variable[KEYS]])

    def _parse_selection_function_variable(self, variable, context=None):
        # this should be replaced in the future with the variable
        # argument when function ordering (and so ordering of parsers)
        # is made explicit
        distance_result = self.distance_function.parameters.value._get(context)
        # TEST PRINT:
        # print(distance_result, self.distance_function.defaults.value)
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
        rand_struct = ctx.get_random_state_ptr(builder, self, state, params)
        uniform_f = ctx.get_uniform_dist_function_by_state(rand_struct)

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
        retr_ptr = builder.alloca(ctx.bool_ty)
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
        store_ptr = builder.alloca(ctx.bool_ty)
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
            is_new_key_ptr = builder.alloca(ctx.bool_ty)
            builder.store(is_new_key_ptr.type.pointee(1), is_new_key_ptr)
            with pnlvm.helpers.for_loop_zero_inc(builder, entries, "distance_loop") as (b,idx):
                cmp_key_ptr = b.gep(keys_ptr, [ctx.int32_ty(0), idx])

                # Vector compare
                # TODO: move this to helpers
                key_differs_ptr = b.alloca(ctx.bool_ty)
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

        Clears the memory in `previous_value <DictionaryMemory.previous_value>`.

        If an argument is passed into reset or if the `initializer <DictionaryMemory.initializer>`
        attribute contains a value besides [], then that value is used to start the new memory in `previous_value
        <DictionaryMemory.previous_value>`. Otherwise, the new `previous_value
        <DictionaryMemory.previous_value>` memory starts out empty.

        `value <DictionaryMemory.value>` takes on the same value as
        `previous_value <DictionaryMemory.previous_value>`.
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
        Return entry in `memory <DictionaryMemory.memory>` that key of which best matches first item of
        `variable <DictionaryMemory.variable>` (query key), then add `variable
        <DictionaryMemory.variable>` to `memory <DictionaryMemory.memory>` (see `above
        <DictionaryMemory_Execution>` for additional details).

        Arguments
        ---------

        variable : list or 2d array : default class_defaults.variable
           first item (variable[0]) is treated as the key for retrieval; second item (variable[1]), paired
           with key, is added to `memory <DictionaryMemory.memory>`.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        value of entry that best matches first item of `variable <DictionaryMemory.variable>`  : 1d array
        """

        key = variable[KEYS]
        # if len(variable)==2:
        val = variable[VALS]

        retrieval_prob = np.array(self._get_current_parameter_value(RETRIEVAL_PROB, context)).astype(float)
        storage_prob = np.array(self._get_current_parameter_value(STORAGE_PROB, context)).astype(float)

        # execute noise if it is a function
        noise = self._try_execute_param(self._get_current_parameter_value(NOISE, context), variable, context=context)

        # get random state
        random_state = self._get_current_parameter_value('random_state', context)

        # If this is an initialization run, leave memory empty (don't want to count it as an execution step),
        # but set key and value size and then return current value (variable[1]) for validation.
        if self.is_initializing:
            return variable

        # Set key_size and val_size if this is the first entry
        if len(self.parameters.previous_value._get(context)[KEYS]) == 0:
            self.parameters.key_size._set(len(key), context)
            self.parameters.val_size._set(len(val), context)

        # Retrieve value from current dict with key that best matches key
        if retrieval_prob == 1.0 or (retrieval_prob > 0.0 and retrieval_prob > random_state.uniform()):
            memory = self.get_memory(key, context)
        else:
            # QUESTION: SHOULD IT RETURN 0's VECTOR OR NOT RETRIEVE AT ALL (LEAVING VALUE & OutputPort FROM LAST TRIAL)?
            #           CURRENT PROBLEM WITH LATTER IS THAT IT CAUSES CRASH ON INIT, SINCE NOT OUTPUT_PORT
            #           SO, WOULD HAVE TO RETURN ZEROS ON INIT AND THEN SUPPRESS AFTERWARDS, AS MOCKED UP BELOW
            memory = [[0]* self.parameters.key_size._get(context), [0]* self.parameters.val_size._get(context)]
        # Store variable to dict:
        if noise is not None:
            key = np.asarray(key, dtype=float)
            if isinstance(noise, numbers.Number):
                key += noise
            else:
                # assume array with same shape as variable
                # TODO: does val need noise?
                key += noise[KEYS]

        if storage_prob == 1.0 or (storage_prob > 0.0 and storage_prob > random_state.uniform()):
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

        Retrieve memory from `memory <DictionaryMemory.memory>` based on `distance_function
        <DictionaryMemory.distance_function>` and `selection_function
        <DictionaryMemory.selection_function>`.

        Arguments
        ---------
        query_key : list or 1d array
            must be same length as key(s) of any existing entries in `memory <DictionaryMemory.memory>`.

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
            # zeros_key = [0] * self.parameters.key_size.get(context)
            # zeros_val = [0] * self.parameters.val_size.get(context)
            zeros_key = [0] * self.parameters.key_size.get(context)
            zeros_val = [0] * self.parameters.val_size.get(context)
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
        """Save an key-value pair to `memory <DictionaryMemory.memory>`

        Arguments
        ---------
        memory : list or 2d array
            must be two items, a key and a vaue, each of which must a list of numbers or 1d array;
            the key must be the same length as key(s) of any existing entries in `dict <DictionaryMemory.dict>`.
        """

        self._validate_memory(memory, context)

        key = list(memory[KEYS])
        val = list(memory[VALS])

        d = self.parameters.previous_value._get(context)

        matches = [k for k in d[KEYS] if key==list(k)]

        # If dupliciate keys are not allowed and key matches any existing keys, don't store
        if matches and self.duplicate_keys is False:
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
        """Add one or more key-value pairs into `memory <ContentAddressableMemory.memory>`

        Arguments
        ---------

        memories : list or array
            a single memory (list or 2d array) or list or array of memorys, each of which must be a valid entry
            consisting of two items (e.g., [[key],[value]] or [[[key1],[value1]],[[key2],[value2]]].
            The keys must all be the same length and equal to the length as key(s) of any existing entries in `dict
            <DictionaryMemory.dict>`.  Items are added to memory in the order listed.
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
