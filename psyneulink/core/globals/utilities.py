# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *************************************************  Utilities *********************************************************

"""Utilities that must be accessible to all PsyNeuLink modules, but are not PsyNeuLink-specific

   That is:
       do not require any information about PsyNeuLink objects
       are not constrained to be used by PsyNeuLink objects

************************************************* UTILITIES ************************************************************


CONTENTS
--------

*TYPE CHECKING VALUE COMPARISON*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   PsyNeuLink-specific typechecking functions are in the `Component <Component>` module

* `parameter_spec`
* `optional_parameter_spec`
* `all_within_range`
* `is_matrix
* `is_matrix_spec`
* `is_numeric`
* `is_numeric_or_none`
* `is_iterable`
* `iscompatible`
* `is_value_spec`
* `is_unit_interval`
* `is_same_function_spec`
* `is_component`
* `is_comparison_operator`

*ENUM*
~~~~~~

* `Autonumber`
* `Modulation`
* `get_modulationOperation_name`

*KVO*
~~~~~

.. note::
   This is for potential future use;  not currently used by PsyNeuLink objects

* observe_value_at_keypath

*MATHEMATICAL*
~~~~~~~~~~~~~~

* norm
* sinusoid
* scalar_distance
* powerset
* tensor_power


*OTHER*
~~~~~~~

* `get_args`
* `recursive_update`
* `multi_getattr`
* `np_array_less_that_2d`
* `convert_to_np_array`
* `type_match`
* `get_value_from_array`
* `is_matrix`
* `underscore_to_camelCase`
* `append_type_to_name`
* `ReadOnlyOrderedDict`
* `ContentAddressableList`
* `make_readonly_property`
* `get_class_attributes`
* `insert_list`
* `flatten_list`
* `convert_to_list`
* `get_global_seed`
* `set_global_seed`

"""

import collections
import copy
import inspect
import logging
import numbers
import psyneulink
import re
import time
import warnings
import weakref
import types
import typing
import typecheck as tc

from enum import Enum, EnumMeta, IntEnum
from collections.abc import Mapping
from collections import UserDict, UserList
from itertools import chain, combinations

import numpy as np

from psyneulink.core.globals.keywords import \
    comparison_operators, DISTANCE_METRICS, EXPONENTIAL, GAUSSIAN, LINEAR, MATRIX_KEYWORD_VALUES, NAME, SINUSOID, VALUE

__all__ = [
    'append_type_to_name', 'AutoNumber', 'ContentAddressableList', 'convert_to_list', 'convert_to_np_array',
    'convert_all_elements_to_np_array', 'copy_iterable_with_shared', 'get_class_attributes', 'flatten_list',
    'get_all_explicit_arguments', 'get_modulationOperation_name', 'get_value_from_array',
    'insert_list', 'is_matrix_spec', 'all_within_range',
    'is_comparison_operator',  'iscompatible', 'is_component', 'is_distance_metric', 'is_iterable', 'is_matrix',
    'is_modulation_operation', 'is_numeric', 'is_numeric_or_none', 'is_same_function_spec', 'is_unit_interval',
    'is_value_spec',
    'kwCompatibilityLength', 'kwCompatibilityNumeric', 'kwCompatibilityType',
    'make_readonly_property',
    'Modulation', 'MODULATION_ADD', 'MODULATION_MULTIPLY','MODULATION_OVERRIDE',
    'multi_getattr', 'np_array_less_than_2d', 'object_has_single_value', 'optional_parameter_spec', 'normpdf',
    'parse_valid_identifier', 'parse_string_to_psyneulink_object_string', 'parameter_spec', 'powerset',
    'random_matrix', 'ReadOnlyOrderedDict', 'safe_equals', 'safe_len',
    'scalar_distance', 'sinusoid',
    'tensor_power', 'TEST_CONDTION', 'type_match',
    'underscore_to_camelCase', 'UtilitiesError', 'unproxy_weakproxy', 'create_union_set', 'merge_dictionaries',
    'contains_type'
]

logger = logging.getLogger(__name__)


class UtilitiesError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


MODULATION_OVERRIDE = 'Modulation.OVERRIDE'
MODULATION_MULTIPLY = 'Modulation.MULTIPLY'
MODULATION_ADD = 'Modulation.ADD'

class Modulation(Enum):
    MULTIPLY = lambda runtime, default : runtime * default
    ADD = lambda runtime, default : runtime + default
    OVERRIDE = lambda runtime, default : runtime
    DISABLE = 0

def is_modulation_operation(val):
    # try:
    #     val(0,0)
    #     return True
    # except:
    #     return False
    return get_modulationOperation_name(val)

def get_modulationOperation_name(operation):
        x = operation(1, 2)
        if x == 1:
            return MODULATION_OVERRIDE
        elif x == 2:
            return MODULATION_MULTIPLY
        elif x == 3:
            return MODULATION_ADD
        else:
            return False



class AutoNumber(IntEnum):
    """Autonumbers IntEnum type

    First item in list of declarations is numbered 0;
    Others incremented by 1

    Sample:

        >>> class NumberedList(AutoNumber):
        ...    FIRST_ITEM = ()
        ...    SECOND_ITEM = ()

        >>> NumberedList.FIRST_ITEM.value
        0
        >>> NumberedList.SECOND_ITEM.value
        1

    Adapted from AutoNumber example for Enum at https://docs.python.org/3/library/enum.html#enum.IntEnum:
    Notes:
    * Start of numbering changed to 0 (from 1 in example)
    * obj based on int rather than object
    """
    def __new__(component_type):
        # Original example:
        # value = len(component_type.__members__) + 1
        # obj = object.__new__(component_type)
        value = len(component_type.__members__)
        obj = int.__new__(component_type)
        obj._value_ = value
        return obj

# ******************************** GLOBAL STRUCTURES, CONSTANTS AND METHODS  *******************************************

TEST_CONDTION = False


def optional_parameter_spec(param):
    """Test whether param is a legal PsyNeuLink parameter specification or `None`

    Calls parameter_spec if param is not `None`
    Used with typecheck

    Returns
    -------
    `True` if it is a legal parameter or `None`.
    `False` if it is neither.


    """
    if not param:
        return True
    return parameter_spec(param)


def parameter_spec(param, numeric_only=None):
    """Test whether param is a legal PsyNeuLink parameter specification

    Used with typecheck

    Returns
    -------
    `True` if it is a legal parameter.
    `False` if it is not.
    """
    # if isinstance(param, property):
    #     param = ??
    # if is_numeric(param):
    from psyneulink.core.components.shellclasses import Projection
    from psyneulink.core.components.component import parameter_keywords
    from psyneulink.core.globals.keywords import MODULATORY_SPEC_KEYWORDS
    from psyneulink.core.components.component import Component

    if inspect.isclass(param):
        param = param.__name__
    elif isinstance(param, Component):
        param = param.__class__.__name__
    if (isinstance(param, (numbers.Number,
                           np.ndarray,
                           list,
                           tuple,
                           dict,
                           types.FunctionType,
                           Projection))
        or param in MODULATORY_SPEC_KEYWORDS
        or param in parameter_keywords):
        if numeric_only:
            if not is_numeric(param):
                return False
        return True
    return False


def all_within_range(x, min, max):
    x = np.array(x)
    try:
        if min is not None and (x<min).all():
            return False
        if max is not None and (x>max).all():
            return False
        return True
    except (ValueError, TypeError):
        for i in x:
            if not all_within_range(i, min, max):
                return False
        return True

def is_numeric_or_none(x):
    if x is None:
        return True
    return is_numeric(x)


def is_numeric(x):
    return iscompatible(x, **{kwCompatibilityNumeric:True, kwCompatibilityLength:0})


def is_matrix_spec(m):
    return isinstance(m, str) and m in MATRIX_KEYWORD_VALUES


def is_matrix(m):
    from psyneulink.core.components.component import Component

    if is_matrix_spec(m):
        return True
    if isinstance(m, (list, np.ndarray, np.matrix)):
        return True
    if m is None or isinstance(m, (Component, dict, set)) or (inspect.isclass(m) and issubclass(m, Component)):
        return False
    try:
        m2 = np.matrix(m)
        return is_matrix(m2)
    except:
        pass
    if callable(m):
        try:
            return is_matrix(m())
        except:
            return False
    return False


def is_distance_metric(s):
    if s in DISTANCE_METRICS:
        return True
    else:
        return False


def is_iterable(x):
    """
    Returns
    -------
        True - if **x** can be iterated on
        False - otherwise
    """
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return False
    else:
        return isinstance(x, collections.abc.Iterable)


kwCompatibilityType = "type"
kwCompatibilityLength = "length"
kwCompatibilityNumeric = "numeric"

# IMPLEMENT SUPPORT OF *LIST* OF TYPES IN kwCompatibilityType (see Function.__init__ FOR EXAMPLE)
# IMPLEMENT: IF REFERENCE IS np.ndarray, try converting candidate to array and comparing
def iscompatible(candidate, reference=None, **kargs):
    """Check if candidate matches reference or, if that is omitted, it matches type, length and/or numeric specification

    If kargs is omitted, candidate and reference must match in type and length
    If reference and kargs are omitted, candidate must be a list of numbers (of any length)
    If reference is omitted, candidate must match specs in kargs

    kargs is an optional dictionary with the following entries:
        kwCompatibilityType ("type"):<type> (default: list):  (spec local_variable: match_type)
            - if reference is provided, candidate's type must match or be subclass of reference,
                irrespective of whether kwCompatibilityType is specified or absent;  however, if:
                + kwCompatibilityType is absent, enums are treated as numbers (of which they are a subclass)
                + kwCompatibilityType = enum, then candidate must be an enum if reference is one
            - if reference is absent:
                if kwCompatibilityType is also absent:
                    if kwCompatibilityNumeric is `True`, all elements of candidate must be numbers
                    if kwCompatibilityNumeric is :keyword:`False`, candidate can contain any type
                if kwCompatibilityType is specified, candidate's type must match or be subclass of specified type
            - for iterables, if kwNumeric is :keyword:`False`, candidate can have multiple types but
                if a reference is provided, then the corresponding items must have the same type
        kwCompatibilityLength ("length"):<int>  (default: 0):    (spec local_variable: match_length)
            - if kwCompatibilityLength is absent:
                if reference is absent, candidate can be of any length
                if reference is provided, candidate must match reference length
            - if kwCompatibilityLength = 0:
                candidate can be any length, irrespective of the reference or its length
            - if kwCompatibilityLength > 0:
                if reference is provided, candidate must be same length as reference
                if reference is omitted, length of candidate must equal value of kwLength
            Note: kwCompatibility < 0 is illegal;  it will generate a warning and be set to 0
        kwCompatibilityNumeric ("number": <bool> (default: `True`)  (spec local_variable: number_only)
            If kwCompatibilityNumeric is `True`, candidate must be either numeric or a list or tuple of
                numeric types
            If kwCompatibilityNumberic is :keyword:`False`, candidate can be strings, lists or tuples of strings,
                or dicts
                Note: if the candidate is a dict, the number of entries (lengths) are compared, but not their contents

    :param candidate: (value)
    :param reference:  (value)
    :param args: (dict)
    :return:
    """

    # If the two are equal, can settle it right here
    # IMPLEMENTATION NOTE: remove the duck typing when numpy supports a direct comparison of iterables
    try:
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            if reference is not None and (candidate == reference):
                return True
    except ValueError:
        # raise UtilitiesError("Could not compare {0} and {1}".format(candidate, reference))
        # IMPLEMENTATION NOTE: np.array generates the following error:
        # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
        pass

    # If args not provided, assign to default values
    # if not specified in args, use these:
    #     args[kwCompatibilityType] = list
    #     args[kwCompatibilityLength] = 1
    #     args[kwCompatibilityNumeric] = True
    try:
        kargs[kwCompatibilityType]
    except KeyError:
        kargs[kwCompatibilityType] = list
    try:
        kargs[kwCompatibilityLength]
    except KeyError:
        kargs[kwCompatibilityLength] = 1
    try:
        # number_only = kargs[kwCompatibilityNumeric]
        kargs[kwCompatibilityNumeric]
    except KeyError:
        kargs[kwCompatibilityNumeric] = True
        # number_only = True


    # If reference is not provided, assign local_variables to arg values (provided or default)
    if reference is None:
        match_type = kargs[kwCompatibilityType]
        match_length = kargs[kwCompatibilityLength]
        number_only = kargs[kwCompatibilityNumeric]
    # If reference is provided, assign specification local_variables to reference-based values
    else:
        match_type = type(reference)
        # If length specification is non-zero (i.e., use length) and reference is an object for which len is defined:
        if (kargs[kwCompatibilityLength] and
                (isinstance(reference, (list, tuple, dict)) or
                         isinstance(reference, np.ndarray) and reference.ndim)
            ):
            match_length = len(reference)
        else:
            match_length = 0
        # If reference is not a number, then don't require the candidate to be one
        if not isinstance(reference, numbers.Number):
            number_only = False
        else:
            number_only = kargs[kwCompatibilityNumeric]

    if match_length < 0:
        # if settings & Settings.VERBOSE:
        print("\niscompatible({0}, {1}): length argument must be non-negative; it has been set to 0\n".
              format(candidate, kargs, match_length))
        match_length = 0

    # # FIX??
    # # Reference is a matrix or a keyword specification for one
    if is_matrix_spec(reference):
        return is_matrix(candidate)

    # MODIFIED 10/29/17 NEW:
    # IMPLEMENTATION NOTE: This allows a number in an ndarray to match a float or int
    # If both the candidate and reference are either a number or an ndarray of dim 0, consider it a match
    if ((isinstance(candidate, numbers.Number) or (isinstance(candidate, np.ndarray) and candidate.ndim == 0)) or
            (isinstance(reference, numbers.Number) or (isinstance(reference, np.ndarray) and reference.ndim == 0))):
        return True
    # MODIFIED 10/29/17 END

    # IMPLEMENTATION NOTE:
    #   modified to allow numeric type mismatches (e.g., int and float;
    #   should be added as option in future (i.e., to disallow it)
    if (isinstance(candidate, match_type) or
            (isinstance(candidate, (list, np.ndarray)) and (issubclass(match_type, (list, np.ndarray)))) or
            (isinstance(candidate, numbers.Number) and issubclass(match_type,numbers.Number)) or
            # IMPLEMENTATION NOTE: Allow UserDict types to match dict (make this an option in the future)
            (isinstance(candidate, UserDict) and match_type is dict) or
            # IMPLEMENTATION NOTE: Allow UserList types to match list (make this an option in the future)
            (isinstance(candidate, UserList) and match_type is list) or
            # IMPLEMENTATION NOTE: This is needed when kwCompatiblityType is not specified
            #                      and so match_type==list as default
            (isinstance(candidate, numbers.Number) and issubclass(match_type,list)) or
                (isinstance(candidate, np.ndarray) and issubclass(match_type,list))
        ):

        # Check compatibility of enum's
        # IMPLEMENTATION NOTE: THE FIRST VERSION BELOW SOUGHT TO CHECK COMPATIBILTY OF ENUM VALUE;  NEEDS WORK
        # if (kargs[kwCompatibilityType] is Enum and
        #         (isinstance(candidate, Enum) or isinstance(match_type, (Enum, IntEnum, EnumMeta)))):
        #     # If either the candidate enum's label is not in the reference's Enum class
        #     #    or its value is different, then return with fail
        #     try:
        #         match_type.__dict__['_member_names_']
        #     except:
        #         pass
        #     if not (candidate.name in match_type.__dict__['_member_names_'] and
        #                 candidate.value is match_type.__dict__['_member_map_'][candidate.name].value):
        #         return False
        # This version simply enforces the constraint that, if either is an enum of some sort, then both must be
        if (kargs[kwCompatibilityType] is Enum and
                (isinstance(candidate, Enum) != isinstance(match_type, (Enum, IntEnum, EnumMeta)))
            ):
            return False

        if isinstance(candidate, numbers.Number):
            return True
        if number_only:
            if isinstance(candidate, np.ndarray) and candidate.ndim ==0 and np.isreal(candidate):
                return True
            if not isinstance(candidate, (list, tuple, np.ndarray, np.matrix)):
                return False
            def recursively_check_elements_for_numeric(value):
                # Matrices can't be checked recursively, so convert to array
                if isinstance(value, np.matrix):
                    value = value.A
                if isinstance(value, (list, np.ndarray)):
                    for item in value:
                        if not recursively_check_elements_for_numeric(item):
                            return False
                        else:
                            return True
                else:
                    if not isinstance(value, numbers.Number):
                        try:
                            # True for autograd ArrayBox (and maybe other types?)
                            # if isinstance(value._value, numbers.Number):
                            from autograd.numpy.numpy_boxes import ArrayBox
                            if isinstance(value, ArrayBox):
                                return True
                        except:
                            return False
                    else:
                        return True
            # Test copy since may need to convert matrix to array (see above)
            if not recursively_check_elements_for_numeric(candidate.copy()):
                return False

        if isinstance(candidate, (list, tuple, dict, np.ndarray)):
            if not match_length:
                return True
            else:
                if len(candidate) == match_length:
                    # No reference, so item by item comparison is not relevant
                    if reference is None:
                        return True
                    # Otherwise, carry out recursive elementwise comparison
                    if isinstance(candidate, np.matrix):
                        candidate = np.asarray(candidate)
                    if isinstance(reference, np.matrix):
                        reference = np.asarray(reference)
                    cr = zip(candidate, reference)
                    if all(iscompatible(c, r, **kargs) for c, r in cr):
                        return True
                    # IMPLEMENTATION NOTE:  ??No longer needed given recursive call above
                    # Deal with ints in one and floats in the other
                    # # elif all((isinstance(c, numbers.Number) and isinstance(r, numbers.Number))
                    # #          for c, r in cr):
                    # #     return True
                else:
                    return False
        else:
            return True
    else:
        return False


# MATHEMATICAL  ********************************************************************************************************

def normpdf(x, mu=0, sigma=1):
    u = float((x - mu) / abs(sigma))
    y = np.exp(-u * u / 2) / (np.sqrt(2 * np.pi) * abs(sigma))
    return y

def sinusoid(x, amplitude=1, frequency=1, phase=0):
    return amplitude * np.sin(2 * np.pi * frequency * x + phase)

def scalar_distance(measure, value, scale=1, offset=0):
    if measure == GAUSSIAN:
        return normpdf(value, offset, scale)
    if measure == LINEAR:
        return scale * value + offset
    if measure == EXPONENTIAL:
        return np.exp(scale * value + offset)
    if measure == SINUSOID:
        return sinusoid(value, frequency=scale, phase=offset)


def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

@tc.typecheck
def tensor_power(items, levels:tc.optional(range)=None, flat=False):
    """return tensor product for all members of powerset of items

    levels specifies a range of set levels to return;  1=first order terms, 2=2nd order terms, etc.
    if None, all terms will be returned

    if flat=False, returns list of 1d arrays with tensor product for each member of the powerset
    if flat=True, returns 1d array of values
    """

    ps = list(powerset(items))
    max_levels = max([len(s) for s in ps])
    levels = levels or range(1,max_levels)
    max_spec = max(list(levels))
    min_spec = min(list(levels))
    if max_spec > max_levels:
        raise UtilitiesError("range ({},{}) specified for {} arg of tensor_power() "
                             "exceeds max for items specified ({})".
                             format(min_spec, max_spec + 1, repr('levels'), max_levels + 1))

    pp = []
    for s in ps:
        order = len(s)
        if order not in list(levels):
            continue
        if order==1:
            pp.append(np.array(s[0]))
        else:
            i = 0
            tp = np.tensordot(s[i],s[i + 1],axes=0)
            i += 2
            while i < order:
                tp = np.tensordot(tp, s[i], axes=0)
                i += 1
            if flat is True:
                pp.extend(tp.reshape(-1))
            else:
                pp.append(tp.reshape(-1))
    return pp



# OTHER ****************************************************************************************************************

def get_args(frame):
    """Gets dictionary of arguments and their values for a function
    Frame should be assigned as follows in the function itself:  frame = inspect.currentframe()
    """
    args, _, _, values = inspect.getargvalues(frame)
    return dict((key, value) for key, value in values.items() if key in args)


def recursive_update(d, u, non_destructive=False):
    """Recursively update entries of dictionary d with dictionary u
    From: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            r = recursive_update(d.get(k, {}), v)
            d[k] = r
        else:
            if non_destructive and k in d and d[k] is not None:
                continue
            d[k] = u[k]
    return d


def multi_getattr(obj, attr, default = None):
    """
    Get a named attribute from an object; multi_getattr(x, 'a.b.c.d') is
    equivalent to x.a.b.c.d. When a default argument is given, it is
    returned when any attribute in the chain doesn't exist; without
    it, an exception is raised when a missing attribute is encountered.

    """
    attributes = attr.split(".")
    for i in attributes:
        try:
            obj = getattr(obj, i)
        except AttributeError:
            if default:
                return default
            else:
                raise
    return obj

# # Example usage
# obj  = [1,2,3]
# attr = "append.__doc__.capitalize.__doc__"

# http://pythoncentral.io/how-to-get-an-attribute-from-an-object-in-python/
# getattr also accepts modules as arguments.

# # http://code.activestate.com/recipes/577346-getattr-with-arbitrary-depth/
# multi_getattr(obj, attr) #Will return the docstring for the
#                          #capitalize method of the builtin string
#                          #object


# based off the answer here https://stackoverflow.com/a/15774013/3131666
def get_deepcopy_with_shared(shared_keys=frozenset(), shared_types=()):
    """
        Arguments
        ---------
            shared_keys
                an Iterable containing strings that should be shallow copied

            shared_types
                an Iterable containing types that when objects of that type are encountered
                will be shallow copied

        Returns
        -------
            a __deepcopy__ function
    """
    shared_types = tuple(shared_types)
    shared_keys = frozenset(shared_keys)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k in shared_keys or isinstance(v, shared_types):
                res_val = v
            else:
                try:
                    res_val = copy_iterable_with_shared(v, shared_types, memo)
                except TypeError:
                    res_val = copy.deepcopy(v, memo)
            setattr(result, k, res_val)
        return result

    return __deepcopy__


def copy_iterable_with_shared(obj, shared_types=None, memo=None):
    try:
        shared_types = tuple(shared_types)
    except TypeError:
        shared_types = (shared_types, )

    dict_types = (dict, collections.UserDict)
    list_types = (list, collections.UserList, collections.deque)
    tuple_types = (tuple, )
    all_types_using_recursion = dict_types + list_types + tuple_types

    if isinstance(obj, dict_types):
        result = obj.__class__()
        for (k, v) in obj.items():
            # key can never be unhashable dict or list
            new_k = k if isinstance(k, shared_types) else copy.deepcopy(k, memo)

            if isinstance(v, all_types_using_recursion):
                new_v = copy_iterable_with_shared(v, shared_types, memo)
            elif isinstance(v, shared_types):
                new_v = v
            else:
                new_v = copy.deepcopy(v, memo)

            try:
                result[new_k] = new_v
            except UtilitiesError:
                # handle ReadOnlyOrderedDict
                result.__additem__(new_k, new_v)

    elif isinstance(obj, list_types + tuple_types):
        is_tuple = isinstance(obj, tuple_types)
        if is_tuple:
            result = list()

        # If this is a deque, make sure we copy the maxlen parameter as well
        elif isinstance(obj, collections.deque):
            # FIXME: Should have a better method for supporting properties like this in general
            # We could do something like result = copy(obj); result.clear() but that would be
            # wasteful copying I guess.
            result = obj.__class__(maxlen=obj.maxlen)
        else:
            result = obj.__class__()

        for item in obj:
            if isinstance(item, all_types_using_recursion):
                new_item = copy_iterable_with_shared(item, shared_types, memo)
            elif isinstance(item, shared_types):
                new_item = item
            else:
                new_item = copy.deepcopy(item, memo)
            result.append(new_item)

        if is_tuple:
            try:
                result = obj.__class__(result)
            except TypeError:
                # handle namedtuple
                result = obj.__class__(*result)
    else:
        raise TypeError

    return result


def get_alias_property_getter(name, attr=None):
    """
        Arguments
        ---------
            name : str

            attr : str : default None

        Returns
        -------
            a property getter method that

            if **attr** is None, returns the **name** attribute of an object

            if **attr** is not None, returns the **name** attribute of the
            **attr** attribute of an object
    """
    if attr is not None:
        def getter(obj):
            return getattr(getattr(obj, attr), name)
    else:
        def getter(obj):
            return getattr(obj, name)

    return getter


def get_alias_property_setter(name, attr=None):
    """
        Arguments
        ---------
            name : str

            attr : str : default None

        Returns
        -------
            a property setter method that

            if **attr** is None, sets the **name** attribute of an object

            if **attr** is not None, sets the **name** attribute of the
            **attr** attribute of an object
    """
    if attr is not None:
        def setter(obj, value):
            setattr(getattr(obj, attr), name, value)
    else:
        def setter(obj, value):
            setattr(obj, name, value)

    return setter


#region NUMPY ARRAY METHODS ******************************************************************************************

def np_array_less_than_2d(array):
    if not isinstance(array, np.ndarray):
        raise UtilitiesError("Argument to np_array_less_than_2d() ({0}) must be an np.ndarray".format(array))
    if array.ndim <= 1:
        return True
    else:
        return False

def convert_to_np_array(value, dimension=None):
    """
        Converts value to np.ndarray if it is not already. Handles
        creation of "ragged" arrays
        (https://numpy.org/neps/nep-0034-infer-dtype-is-object.html)

        Args:
            value
                item to convert to np.ndarray

            dimension : 1, 2, None
                minimum dimension of np.ndarray to convert to

        Returns:
            value : np.ndarray
    """
    def safe_create_np_array(value):
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
            # NOTE: this will raise a ValueError in the future.
            # See https://numpy.org/neps/nep-0034-infer-dtype-is-object.html
            try:
                try:
                    return np.asarray(value)
                except np.VisibleDeprecationWarning:
                    return np.asarray(value, dtype=object)
            except ValueError as e:
                msg = str(e)
                if 'cannot guess the desired dtype from the input' in msg:
                    return np.asarray(value, dtype=object)
                # KDM 6/29/20: this case handles a previously noted case
                # by KAM 6/28/18, #877:
                # [[0.0], [0.0], np.array([[0.0, 0.0]])]
                # but was only handled for dimension=1
                elif 'could not broadcast' in msg:
                    return convert_all_elements_to_np_array(value)
                else:
                    raise

    value = safe_create_np_array(value)

    if dimension == 1:
        value = np.atleast_1d(value)
    elif dimension == 2:
        # Array is made up of non-uniform elements, so treat as 2d array and pass
        if (
            value.ndim > 0
            and value.dtype == object
            and any([safe_len(x) > 1 for x in value])
        ):
            pass
        else:
            value = np.atleast_2d(value)
    elif dimension is not None:
        raise UtilitiesError("dimension param ({0}) must be None, 1, or 2".format(dimension))

    return value


def object_has_single_value(obj):
    """
        Returns
        -------
            True : if **obj** contains only one value, in any dimension
            False : otherwise

            **obj** will be cast to a numpy array if it is not already one
    """
    if not isinstance(obj, np.ndarray):
        obj = np.asarray(obj)

    for s in obj.shape:
        if s > 1:
            return False

    return True


def type_match(value, value_type):
    if isinstance(value, value_type):
        return value
    if value_type in {int, np.integer, np.int64, np.int32}:
        return int(value)
    if value_type in {float, np.float, np.float64, np.float32}:
        return float(value)
    if value_type is np.ndarray:
        return np.array(value)
    if value_type is list:
        return list(value)
    if value_type is None:
        return None
    if value_type is type(None):
        return value
    raise UtilitiesError("Type of {} not recognized".format(value_type))

def get_value_from_array(array):
    """Extract numeric value from array, preserving numeric type
    :param array:
    :return:
    """

def random_matrix(sender, receiver, clip=1, offset=0):
    """Generate a random matrix

    Calls np.random.rand to generate a 2d np.array with random values.

    Arguments
    ----------
    sender : int
        specifies number of rows.

    receiver : int
        spcifies number of columns.

    range : int
        specifies upper limit (lower limit = 0).

    offset : int
        specifies amount added to each entry of the matrix.

    Returns
    -------
    2d np.array
    """
    return (clip * np.random.rand(sender, receiver)) + offset

def underscore_to_camelCase(item):
    item = item[1:]
    item = ''.join(x.capitalize() or '_' for x in item.split('_'))
    return item[0].lower() + item[1:]

def append_type_to_name(object, type=None):
    name = object.name
    # type = type or object.componentType
    # type = type or object.__class__.__base__.__base__.__base__.__name__
    type = type or object.__class__.__base__.__name__
    if any(token in name for token in [type.lower(), type.upper(), type.capitalize()]):
        string = name
    else:
        string = "\'" + name + "\'" + ' ' + type.lower()
        # string = name + ' ' + type.lower()
    return string
#endregion


class ReadOnlyOrderedDict(UserDict):
    def __init__(self, dict=None, name=None, **kwargs):
        self.name = name or self.__class__.__name__
        UserDict.__init__(self, dict, **kwargs)
        self._ordered_keys = []
    def __setitem__(self, key, item):
        raise UtilitiesError("{} is read-only".format(self.name))
    def __delitem__(self, key):
        raise UtilitiesError("{} is read-only".format(self.name))
    def clear(self):
        raise UtilitiesError("{} is read-only".format(self.name))
    def pop(self, key, *args):
        raise UtilitiesError("{} is read-only".format(self.name))
    def popitem(self):
        raise UtilitiesError("{} is read-only".format(self.name))
    def __additem__(self, key, value):
        self.data[key] = value
        if key not in self._ordered_keys:
            self._ordered_keys.append(key)
    def __deleteitem__(self, key):
        del self.data[key]
    def keys(self):
        return self._ordered_keys
    def copy(self):
        return self.data.copy()

class ContentAddressableList(UserList):
    """
    ContentAddressableList( component_type, key=None, list=None)

    Implements dict-like list, that can be keyed by a specified attribute of the `Compoments <Component>` in its
    entries.  If called, returns list of items.

    Instance is callable (with no arguments): returns list of its items.

    The key with which it is created is also assigned as a property of the class, that returns a list
    with the keyed attribute of its entries.  For example, the `output_ports <Mechanism_Base.output_ports>` attribute
    of a `Mechanism <Mechanism>` is a ContentAddressableList of the Mechanism's `OutputPorts <OutputPort>`, keyed by
    their names.  Therefore, ``my_mech.output_ports.names`` returns the names of all of the Mechanism's OutputPorts::

        >>> import psyneulink as pnl
        >>> print(pnl.DDM().output_ports.names)
        ['DECISION_VARIABLE', 'RESPONSE_TIME']

    The keyed attribute can also be used to access an item of the list.  For examples::

        >>> print(pnl.DDM().output_ports['DECISION_VARIABLE'])
        (OutputPort DECISION_VARIABLE)

    Supports:
      * getting and setting entries in the list using keys (string), in addition to numeric indices.
        the key to use is specified by the **key** arg of the constructor, and must be a string attribute;
        * for getting an entry:
          the key must match the keyed attribute of a Component in the list; otherwise an exception is raised;
        * for setting an entry:
            - the key must match the key of the component being assigned;
            - if there is already a component in the list the keyed vaue of which matches the key, it is replaced;
            - if there is no component in the list the keyed attribute of which matches the key,
              the component is appended to the list;
        * for getting lists of the names, values of the keyed attributes, and values of the `value <Component.value>`
            attributes of components in the list.

    IMPLEMENTATION NOTE:
        This class allows Components to be maintained in lists, while providing ordered storage
        and the convenience of access and assignment by name (e.g., akin to a dict).
        Lists are used (instead of a dict or OrderedDict) since:
            - ordering is in many instances convenient, and in some critical (e.g., for consistent mapping from
                collections of ports to other variables, such as lists of their values);
            - they are most commonly accessed either exhaustively (e.g., in looping through them during execution),
                or by key (e.g., to get the first, "primary" one), which makes the efficiencies of a dict for
                accessing by key/name less critical;
            - the number of ports in a collection for a given Mechanism is likely to be small so that, even when
                accessed by key/name, the inefficiencies of searching a list are likely to be inconsequential.

    Arguments
    ---------

    component_type : Class
        specifies the class of the items in the list.

    name : str : 'ContentAddressableList'
        name to use for ContentAddressableList

    key : str : default `name`
        specifies the attribute of **component_type** used to key items in the list by content;
        **component_type** must have this attribute or, if it is not provided, an attribute with the name 'name'.

    list : List : default None
        specifies a list used to initialize the list;
        all of the items must be of type **component_type** and have the **key** attribute.

    Attributes
    ----------

    component_type : Class
        the class of the items in the list.

    name: str
        name if provided as arg, else name of of ContentAddressableList class

    key : str
        the attribute of `component_type <ContentAddressableList.component_type>` used to key items in the list by content;

    data : List (property)
        the actual list of items.

    names : List (property)
        values of the `name <Component>` attribute of components in the list.

    key_values : List (property)
        values of the keyed attribute of each component in the list.

    values : List (property)
        values of the `value <Component>` attribute of components in the list.

    """

    legal_key_type_strings = ['int', 'str', 'Port']

    def __init__(self, component_type, key=None, list=None, name=None, **kwargs):
        self.component_type = component_type
        self.key = key or 'name'
        self.component_type = component_type
        self.name = name or component_type.__name__
        if not isinstance(component_type, type):
            raise UtilitiesError("component_type arg for {} ({}) must be a class"
                                 .format(self.name, component_type))
        if not isinstance(self.key, str):
            raise UtilitiesError("key arg for {} ({}) must be a string".
                                 format(self.name, self.key))
        if not hasattr(component_type, self.key):
            raise UtilitiesError("key arg for {} (\'{}\') must be an attribute of {}".
                                 format(self.name,
                                        self.key,
                                        component_type.__name__))
        if list is not None:
            if not all(isinstance(obj, self.component_type) for obj in list):
                raise UtilitiesError("All of the items in the list arg for {} "
                                     "must be of the type specified in the component_type arg ({})"
                                     .format(self.name, self.component_type.__name__))
        UserList.__init__(self, list, **kwargs)

    # def __repr__(self):
    #     return '[\n\t{0}\n]'.format('\n\t'.join(['{0}\t{1}\t{2}'.format(i, self[i].name,
    #                                                                     repr(self[i].value))
    #                                              for i in range(len(self))]))

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.data = self.data.copy()
        return result

    def __getitem__(self, key):
        if key is None:
            raise KeyError("None is not a legal key for {}".format(self.name))
        try:
            return self.data[key]
        except TypeError:
            key_num = self._get_key_for_item(key)
            if key_num is None:
                # raise TypeError("\'{}\' is not a key in the {} being addressed".
                                # format(key, self.__class__.__name__))
                # raise KeyError("\'{}\' is not a key in {}".
                raise TypeError("\'{}\' is not a key in {}".
                                format(key, self.name))
            return self.data[key_num]

    def __setitem__(self, key, value):
        # For efficiency, first assume the key is numeric (duck typing in action!)
        try:
            self.data[key] = value
        # If it is not
        except TypeError:
            # It must be a string
            if not isinstance(key, str):
                raise UtilitiesError("Non-numeric key used for {} ({}) must be "
                                     "a string)".format(self.name, key))
            # The specified string must also match the value of the attribute of the class used for addressing
            if not key.startswith(value.name):
            # if not key == type(value).__name__:
                raise UtilitiesError("The key of the entry for {} {} ({}) "
                                     "must match the value of its {} attribute "
                                     "({})".format(self.name,
                                                   value.__class__.__name__,
                                                   key,
                                                   self.key,
                                                   getattr(value, self.key)))
            key_num = self._get_key_for_item(key)
            if key_num is not None:
                self.data[key_num] = value
            else:
                self.data.append(value)

    def __contains__(self, item):
        if super().__contains__(item):
            return True
        else:
            try:
                self.__getitem__(item)
                return True
            except (KeyError, TypeError, UtilitiesError, ValueError):
                return False

    def _get_key_for_item(self, key):
        if isinstance(key, str):
            obj = next((obj for obj in self.data if obj.name == key), None)
            if obj is None:
                return None
            else:
                return self.data.index(obj)
        elif isinstance(key, self.component_type):
            return self.data.index(key)
        else:
            raise UtilitiesError(
                "{} is not a legal key for {} (must be {})".format(
                    key,
                    self.key,
                    gen_friendly_comma_str(self.legal_key_type_strings)
                )
            )

    def __delitem__(self, key):
        if key is None:
            raise KeyError("None is not a legal key for {}".format(self.name))
        try:
            del self.data[key]
        except TypeError:
            key_num = self._get_key_for_item(key)
            del self.data[key_num]

    def __call__(self):
        return self.data

    def clear(self):
        super().clear()

    # def pop(self, key, *args):
    #     raise UtilitiesError("{} is read-only".format(self.name))
    #
    # def popitem(self):
    #     raise UtilitiesError("{} is read-only".format(self.name))

    def __additem__(self, key, value):
        if key >= len(self.data):
            self.data.append(value)
        else:
            self.data[key] = value

    def __add__(self, item):
        try:
            if self.component_type != item.component_type:
                raise TypeError(f'Type mismatch {self.component_type} and {item.component_type}')

            if self.key != item.key:
                raise TypeError(f'Key mismatch {self.key} and {item.key}')

        except AttributeError:
            raise TypeError('ContentAddressableList can only be added to ContentAddressableList')

        return ContentAddressableList(self.component_type, self.key, self.data + item.data, self.name)

    def copy(self):
        return self.data.copy()

    @property
    def names(self):
        """Return list of `values <Component.value>` of the name attribute of components in the list.
        Returns
        -------
        names :  list
            list of the values of the `name <Component.name>` attributes of components in the list.
        """
        return [getattr(item, NAME) for item in self.data]

    @property
    def key_values(self):
        """Return list of `values <Component.value>` of the keyed attribute of components in the list.
        Returns
        -------
        key_values :  list
            list of the values of the `keyed <Component.name>` attributes of components in the list.
        """
        return [getattr(item, self.key) for item in self.data]

    def get_key_values(self, context=None):
        """Return list of `values <Component.value>` of the keyed
        parameter of components in the list.
        Returns
        -------
        key_values :  list
            list of the values of the `keyed <Component.name>`
            parameter of components in the list  for **context**
        """
        return [getattr(item.parameters, self.key).get(context) for item in self.data]

    @property
    def values(self):
        """Return list of values of components in the list.
        Returns
        -------
        values :  list
            list of the values of the `value <Component.value>` attributes of components in the list.
        """
        return [getattr(item, VALUE) for item in self.data]

    def get_values(self, context=None):
        """Return list of values of components in the list.
        Returns
        -------
        values :  list
            list of the values of the `value <Component.value>`
            parameter of components in the list  for **context**
        """
        return [item.parameters.value.get(context) for item in self.data]

    @property
    def values_as_lists(self):
        """Return list of values of components in the list, each converted to a list.
        Returns
        -------
        values :  list
            list of list values of the `value <Component.value>` attributes of components in the list,
        """
        return [np.ndarray.tolist(getattr(item, VALUE)) for item in self.data]

    def get_values_as_lists(self, context=None):
        """Return list of values of components in the list, each converted to a list.
        Returns
        -------
        values :  list
            list of list values of the `value <Component.value>` attributes of components in the list,
        """
        return [np.ndarray.tolist(item.parameters.value.get(context)) for item in self.data]


def is_value_spec(spec):
    if isinstance(spec, (numbers.Number, np.ndarray)):
        return True
    elif isinstance(spec, list) and is_numeric(spec):
        return True
    else:
        return False


def is_unit_interval(spec):
    if isinstance(spec, (int, float)) and 0 <= spec <= 1:
        return True
    else:
        return False


def is_same_function_spec(fct_spec_1, fct_spec_2):
    """Compare two function specs and return True if both are for the same class of Function

    Arguments can be either a class or instance of a PsyNeuLink Function, or any other callable method or function
    Return True only if both are instances or classes of a PsyNeuLink Function;  otherwise, return False
    """
    from psyneulink.core.components.functions.function import Function

    def _convert_to_type(fct):
        if isinstance(fct, Function):
            return type(fct)
        elif inspect.isclass(fct) and issubclass(fct, Function):
            return fct
        else:
            return None

    fct_1 = _convert_to_type(fct_spec_1)
    fct_2 = _convert_to_type(fct_spec_2)

    if fct_1 and fct_2 and fct_1 == fct_2:
        return True
    else:
        return False


def is_component(val):
    """This allows type-checking for Component definitions where Component module can't be imported
    """
    from psyneulink.core.components.component import Component
    return isinstance(val, Component)

def is_instance_or_subclass(candidate, spec):
    """
    Returns
    -------

    True if **candidate** is a subclass of **spec** or an instance thereof, False otherwise
    """
    return isinstance(candidate, spec) or (inspect.isclass(candidate) and issubclass(candidate, spec))

def is_comparison_operator(o):
    """
    Returns
    -------

    True if **o** is an entry in comparison_operators dictionary (in keywords)
    """
    if o in comparison_operators.values():
        return True
    return False


def make_readonly_property(val, name=None):
    """Return property that provides read-only access to its value
    """
    if name is None:
        name = val

    def getter(self):
        return val

    def setter(self, val):
        raise UtilitiesError("{} is read-only property of {}".format(name, self.__class__.__name__))

    # Create the property
    prop = property(getter).setter(setter)
    return prop


def get_class_attributes(cls):
    boring = dir(type('dummy', (object,), {}))
    return [item
            for item in inspect.getmembers(cls)
            if item[0] not in boring]


def convert_all_elements_to_np_array(arr, cast_from=None, cast_to=None):
    """
        Recursively converts all items in **arr** to numpy arrays, optionally casting
        items of type/dtype **cast_from** to type/dtype **cast_to**

        Arguments
        ---------
            cast_from - type, numpy.dtype - type when encountered to cast to **cast_to**
            cast_to - type, numpy.dtype - type to cast **cast_from** to

        Returns
        -------
        a numpy array containing the converted **arr**
    """
    if isinstance(arr, np.ndarray) and arr.ndim == 0:
        if cast_from is not None and isinstance(arr.item(0), cast_from):
            return np.asarray(arr, dtype=cast_to)
        else:
            return arr

    if cast_from is not None and isinstance(arr, cast_from):
        return np.asarray(arr, dtype=cast_to)

    if not isinstance(arr, collections.abc.Iterable) or isinstance(arr, str):
        return np.array(arr)

    if isinstance(arr, np.matrix):
        if arr.dtype == object:
            return np.asarray([convert_all_elements_to_np_array(arr.item(i), cast_from, cast_to) for i in range(arr.size)])
        else:
            return arr

    subarr = [convert_all_elements_to_np_array(x, cast_from, cast_to) for x in arr]

    if all([subarr[i].shape == subarr[0].shape for i in range(1, len(subarr))]):
        # the elements are all uniform in shape, so we can use numpy's standard behavior
        return np.asarray(subarr)
    else:
        # the elements are nonuniform, so create an array that just wraps them individually
        # numpy cannot easily create arrays with subarrays of certain dimensions, workaround here
        # https://stackoverflow.com/q/26885508/3131666
        len_subarr = len(subarr)
        elementwise_subarr = np.empty(len_subarr, dtype=np.ndarray)
        for i in range(len_subarr):
            elementwise_subarr[i] = subarr[i]

        return elementwise_subarr


def insert_list(list1, position, list2):
    """Insert list2 into list1 at position"""
    return list1[:position] + list2 + list1[position:]


def convert_to_list(l):
    if l is None:
        return None
    elif isinstance(l, list):
        return l
    elif isinstance(l, ContentAddressableList):
        return list(l)
    elif isinstance(l, set):
        return list(l)
    else:
        return [l]

def flatten_list(l):
    return [item for sublist in l for item in sublist]


_seed = np.int32((time.time() * 1000) % 2**31)
def get_global_seed(offset=1):
    global _seed
    _seed += offset
    _seed %= 2**31
    return _seed - offset


def set_global_seed(new_seed):
    global _seed
    _seed = new_seed


def safe_len(arr, fallback=1):
    """
    Returns
    -------
        len(**arr**) if possible, otherwise **fallback**
    """
    try:
        return len(arr)
    except TypeError:
        return fallback


def safe_equals(x, y):
    """
        An == comparison that handles numpy's new behavior of returning
        an array of booleans instead of a single boolean for ==
    """
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        try:
            val = x == y
            if isinstance(val, bool):
                return val
            else:
                raise ValueError
        except (ValueError, DeprecationWarning, FutureWarning):
            return np.array_equal(x, y)


@tc.typecheck
def _get_arg_from_stack(arg_name:str):
    # Get arg from the stack

    curr_frame = inspect.currentframe()
    prev_frame = inspect.getouterframes(curr_frame, 2)
    i = 1
    # Search stack for first frame (most recent call) with a arg_val specification
    arg_val = None
    while arg_val is None:
        try:
            arg_val = inspect.getargvalues(prev_frame[i][0]).locals[arg_name]
        except KeyError:
            # Try earlier frame
            i += 1
        except IndexError:
            # Ran out of frames, so just set arg_val to empty string
            arg_val = ""
        else:
            break

    # No arg_val was specified in any frame
    if arg_val is None: # cxt-done
        raise UtilitiesError("PROGRAM ERROR: arg_name not found in any frame")

    return arg_val


_unused_args_sig_cache = weakref.WeakKeyDictionary()


def prune_unused_args(func, args=None, kwargs=None):
    """
        Arguments
        ---------
            func : function

            args : *args

            kwargs : **kwargs


        Returns
        -------
            a tuple such that the first item is the intersection of **args** and the
            positional arguments of **func**, and the second item is the intersection
            of **kwargs** and the keyword arguments of **func**

    """
    # use the func signature to filter out arguments that aren't compatible
    try:
        sig = _unused_args_sig_cache[func]
    except KeyError:
        sig = inspect.signature(func)
        _unused_args_sig_cache[func] = sig

    has_args_param = False
    has_kwargs_param = False
    count_positional = 0
    func_kwargs_names = set()

    for name, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            has_args_param = True
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            has_kwargs_param = True
        elif param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD or param.kind is inspect.Parameter.KEYWORD_ONLY:
            if param.default is inspect.Parameter.empty:
                count_positional += 1
            func_kwargs_names.add(name)

    if args is not None:
        try:
            args = list(args)
        except TypeError:
            args = [args]

        if not has_args_param:
            num_extra_args = len(args) - count_positional
            if num_extra_args > 0:
                logger.debug('{1} extra arguments specified to function {0}, will be ignored (values: {2})'.format(func, num_extra_args, args[-num_extra_args:]))
            args = args[:count_positional]
    else:
        args = []

    if kwargs is not None:
        kwargs = dict(kwargs)

        if not has_kwargs_param:
            filtered = set()
            for kw in kwargs:
                if kw not in func_kwargs_names:
                    filtered.add(kw)
            if len(filtered) > 0:
                logger.debug('{1} extra keyword arguments specified to function {0}, will be ignored (values: {2})'.format(func, len(filtered), filtered))
            for kw in filtered:
                del kwargs[kw]
    else:
        kwargs = {}

    return args, kwargs


def call_with_pruned_args(func, *args, **kwargs):
    """
        Calls **func** with only the **args** and **kwargs** that
        exist in its signature
    """
    args, kwargs = prune_unused_args(func, args, kwargs)
    return func(*args, **kwargs)


def unproxy_weakproxy(proxy):
    """
        Returns the actual object weak-referenced by a weakproxy.proxy object
        Much like
            >>> a = weakref.ref(b)
            >>> a() is b
            True
    """
    try:
        return proxy.__repr__.__self__
    except AttributeError:
        # handles the case where proxy references a class
        return proxy.__mro__[0]


def parse_valid_identifier(orig_identifier):
    """
        Returns
        -------
            A version of **orig_identifier** with characters replaced
            so that it is a valid python identifier
    """
    change_invalid_beginning = re.sub(r'^([^a-zA-Z_])', r'_\1', orig_identifier)
    return re.sub(r'[^a-zA-Z0-9_]', '_', change_invalid_beginning)


def parse_string_to_psyneulink_object_string(string):
    """
        Returns
        -------
            a string corresponding to **string** that is an attribute
            of psyneulink if it exists, otherwise None

            The output of this function will cause
            getattr(psyneulink, <output>) to return a psyneulink object
    """
    try:
        eval(f'psyneulink.{string}')
        return string
    except (AttributeError, SyntaxError, TypeError):
        pass

    # handle potential psyneulink keyword
    try:
        # insert space between camel case words
        keyword = re.sub('([a-z])([A-Z])', r'\1 \2', string)
        keyword = keyword.upper().replace(' ', '_')
        eval(f'psyneulink.{keyword}')
        return keyword
    except (AttributeError, SyntaxError, TypeError):
        pass

    return None


def get_all_explicit_arguments(cls_, func_str):
    """
        Returns
        -------
            all explicitly specified (named) arguments for the function
            **cls_**.**funct_str** including any arguments specified in
            functions of the same name on parent classes, if \\*args or
            \\*\\*kwargs are specified
    """
    all_arguments = set()

    for cls_ in cls_.__mro__:
        func = getattr(cls_, func_str)
        has_args_or_kwargs = False

        for arg_name, arg in inspect.signature(func).parameters.items():
            if (
                arg.kind is inspect.Parameter.VAR_POSITIONAL
                or arg.kind is inspect.Parameter.VAR_KEYWORD
            ):
                has_args_or_kwargs = True
            else:
                all_arguments.add(arg_name)

        if not has_args_or_kwargs:
            break

    return all_arguments


def create_union_set(*args) -> set:
    """
        Returns:
            a ``set`` containing all items in **args**, expanding
            iterables
    """
    result = set()
    for item in args:
        if hasattr(item, '__iter__') and not isinstance(item, str):
            result = result.union(item)
        else:
            result = result.union([item])

    return result


def merge_dictionaries(a: dict, b: dict) -> typing.Tuple[dict, bool]:
    """
        Returns: a tuple containing:
            - a ``dict`` containing each key-value pair in **a** and
            **b** where the values of shared keys are sets of their
            original values

            - a ``bool`` indicating if **a** and **b** have any shared
            keys
    """
    shared_keys = [x for x in a if x in b]

    new_dict = {k: v for k, v in a.items()}
    new_dict.update(b)
    new_dict.update({k: create_union_set(a[k], b[k]) for k in shared_keys})

    return new_dict, len(new_dict) < (len(a) + len(b))


def gen_friendly_comma_str(items):
    """
        Returns:
            a proper English comma-separated string of each item in
            **items**
    """
    if isinstance(items, str) or not is_iterable(items):
        return str(items)

    items = [str(x) for x in items]

    if len(items) < 2:
        return ''.join(items)
    else:
        divider = ' or '
        if len(items) > 2:
            divider = f',{divider}'

        return f"{', '.join(items[:-1])}{divider}{items[-1]}"


def contains_type(
    arr: collections.abc.Iterable,
    typ: typing.Union[type, typing.Tuple[type, ...]]
) -> bool:
    """
        Returns:
            True if **arr** is a possibly nested Iterable that contains
            an instance of **typ** (or one type in **typ** if tuple)

        Note: `isinstance(**arr**, **typ**)` should be used to check
        **arr** itself if needed
    """
    try:
        for a in arr:
            if isinstance(a, typ) or (a is not arr and contains_type(a, typ)):
                return True
    except TypeError:
        pass

    return False
