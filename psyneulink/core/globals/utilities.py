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

* `deprecation_warning`

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

*LIST MANAGEMENT*
~~~~~~~~~~~~~~~~~

* `insert_list`
* `convert_to_list`
* `flatten_list`
* `nesting_depth`


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
* `set_global_seed`

"""

import collections
import copy
import functools
import importlib
import inspect
import itertools
import logging
import os
import psyneulink
import re
import time
import warnings
import weakref
import toposort
import types
import typing
from beartype import beartype

from numbers import Number
from psyneulink._typing import Any, Callable, Optional, Union, Literal, Type, List, Tuple, Iterable

from enum import Enum, EnumMeta, IntEnum
from collections.abc import Mapping
from collections import UserDict, UserList
from itertools import chain, combinations

import numpy as np
from numpy.typing import DTypeLike

try:
    from numpy import exceptions as np_exceptions
except ImportError:
    # Numpy exceptions is only available in Numpy 1.25+
    np_exceptions = np

# Conditionally import torch
try:
    import torch
except ImportError:
    torch = None

from psyneulink.core.globals.keywords import (comparison_operators, DISTANCE_METRICS, EXPONENTIAL, GAUSSIAN, LINEAR,
                                              MATRIX_KEYWORD_VALUES, MPS, NAME, SINUSOID, VALUE)



__all__ = [
    'append_type_to_name', 'AutoNumber', 'ContentAddressableList', 'convert_to_list', 'convert_to_np_array',
    'convert_all_elements_to_np_array', 'copy_iterable_with_shared', 'get_class_attributes', 'extended_array_equal', 'flatten_list',
    'get_all_explicit_arguments', 'get_modulationOperation_name', 'get_value_from_array',
    'insert_list', 'is_matrix_keyword', 'all_within_range',
    'is_comparison_operator',  'iscompatible', 'is_component', 'is_distance_metric', 'is_iterable', 'is_matrix',
    'is_modulation_operation', 'is_numeric', 'is_numeric_or_none', 'is_number', 'is_same_function_spec', 'is_unit_interval',
    'is_value_spec',
    'kwCompatibilityLength', 'kwCompatibilityNumeric', 'kwCompatibilityType',
    'nesting_depth',
    'make_readonly_property',
    'Modulation', 'MODULATION_ADD', 'MODULATION_MULTIPLY','MODULATION_OVERRIDE',
    'multi_getattr', 'np_array_less_than_2d', 'object_has_single_value', 'optional_parameter_spec', 'normpdf',
    'parse_valid_identifier', 'parse_string_to_psyneulink_object_string', 'parameter_spec', 'powerset',
    'random_matrix', 'ReadOnlyOrderedDict', 'safe_equals', 'safe_len',
    'scalar_distance', 'sinusoid',
    'tensor_power', 'TEST_CONDTION', 'type_match',
    'underscore_to_camelCase', 'UtilitiesError', 'unproxy_weakproxy', 'create_union_set', 'merge_dictionaries',
    'contains_type', 'is_numeric_scalar', 'try_extract_0d_array_item', 'fill_array', 'update_array_in_place', 'array_from_matrix_string', 'get_module_file_prefix', 'get_stacklevel_skip_file_prefixes',
]

logger = logging.getLogger(__name__)
_signature_cache = weakref.WeakKeyDictionary()


class UtilitiesError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)



def deprecation_warning(component, kwargs:dict, deprecated_args:dict) -> dict:
    """Identify and warn about any deprecated args, and return their values for reassignment
    Format of deprecated_args dict:  {'deprecated_arg_name':'real_arg"name')}
    Format of returned dict:  {'real_arg_name':<value assigned to deprecated arg>)}
    """
    value_assignments = dict()
    for deprecated_arg in deprecated_args:
        if deprecated_arg in kwargs:
            real_arg = deprecated_args[deprecated_arg]
            arg_value = kwargs.pop(deprecated_arg)
            if arg_value:
                # Value for real arg was also specified:
                warnings.warn(f"Both '{deprecated_arg}' and '{real_arg}' "
                              f"were specified in the constructor for a(n) {component.__class__.__name__}; "
                              f"{deprecated_arg} ({arg_value}) will be used,"
                              f"but note that it is deprecated  and may be removed in the future.")
            else:
                # Only deprecated arg was specified:
                warnings.warn(f"'{deprecated_arg}' was specified in the constructor for a(n)"
                              f" {component.__class__.__name__}; note that this has been deprecated "
                              f"and may be removed in the future; '{real_arg}' "
                              f"should be used instead.")
            value_assignments.update({real_arg:arg_value})
        continue
    return value_assignments


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


#region ******************************** GLOBAL STRUCTURES, CONSTANTS AND METHODS  *************************************
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
    if (isinstance(param, (Number,
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


NumericCollections = Union[Number, List[List[Number]], List[Number],
                           Tuple[Number], Tuple[Tuple[Number]], np.ndarray]


# A set of all valid parameter specification types
ValidParamSpecType = Union[
    Number,
    np.ndarray,
    list,
    tuple,
    dict,
    types.FunctionType,
    'psyneulink.core.components.shellclasses.Projection',
    'psyneulink.core.components.mechanisms.ControlMechanism',
    Type['psyneulink.core.components.mechanisms.ControlMechanism'],
    'psyneulink.core.components.projections.ControlProjection',
    Type['psyneulink.core.components.projections.ControlProjection'],
    'psyneulink.core.components.ports.ControlSignal',
    Type['psyneulink.core.components.ports.ControlSignal'],
    'psyneulink.core.components.mechanisms.GatingMechanism',
    Type['psyneulink.core.components.mechanisms.GatingMechanism'],
    'psyneulink.core.components.projections.GatingProjection',
    Type['psyneulink.core.components.projections.GatingProjection'],
    'psyneulink.core.components.ports.GatingSignal',
    Type['psyneulink.core.components.ports.GatingSignal'],
    'psyneulink.core.components.mechanisms.LearningMechanism',
    Type['psyneulink.core.components.mechanisms.LearningMechanism'],
    'psyneulink.core.components.projections.LearningProjection',
    Type['psyneulink.core.components.projections.LearningProjection'],
    'psyneulink.core.components.ports.LearningSignal',
    Type['psyneulink.core.components.ports.LearningSignal'],
    'psyneulink.library.components.projections.AutoAssociativeProjection',
    Type['psyneulink.library.components.projections.AutoAssociativeProjection'],
    'psyneulink.core.components.projections.MappingProjection',
    Type['psyneulink.core.components.projections.MappingProjection'],
    'psyneulink.library.components.projections.MaskedMappingProjection',
    Type['psyneulink.library.components.projections.MaskedMappingProjection'],
    Literal['LEARNING', 'adaptive','bias', 'control', 'gain', 'gate', 'leak', 'offset',
    'ControlSignal', 'ControlProjection'],
]


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


def is_number(x):
    return (
        isinstance(x, Number)
        and not isinstance(x, (bool, Enum))
    )


def is_matrix_keyword(m):
    return isinstance(m, str) and m in MATRIX_KEYWORD_VALUES

def is_matrix(m):
    from psyneulink.core.components.component import Component

    if is_matrix_keyword(m):
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
            try:
                # random_matrix and RandomMatrix are allowable functions, but require num_rows and num_cols parameters
                return is_matrix(1,2)
            except:
                return False
    return False


def is_distance_metric(s):
    if s in DISTANCE_METRICS:
        return True
    else:
        return False


# Allowed distance metrics literals
DistanceMetricLiteral = Literal[
    'max_abs_diff',
    'difference',
    'dot_product',
    'normed_L0_similarity',
    'euclidean',
    'angle',
    'correlation',
    'cosine',
    'entropy',
    'cross-entropy',
    'energy'
]


def is_iterable(x: Any, exclude_str: bool = False) -> bool:
    """
    Args:
        x (Any)
        exclude_str (bool, optional): if True, **x** of type str will
            return False. Defaults to False.

    Returns
    -------
        True - if **x** can be iterated on
        False - otherwise
    """
    try:
        iter(x)
    except TypeError:
        return False
    else:
        return not exclude_str or not isinstance(x, str)


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
        if (reference is not None and np.array(candidate, dtype=object).size > 0
                and safe_equals(candidate, reference)):
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
        if not isinstance(reference, Number):
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
    if is_matrix_keyword(reference):
        return is_matrix(candidate)

    # IMPLEMENTATION NOTE: This allows a number in an ndarray to match a float or int
    # If both the candidate and reference are either a number or an ndarray of dim 0, consider it a match
    if ((is_number(candidate) or (isinstance(candidate, np.ndarray) and candidate.ndim == 0)) or
            (is_number(reference) or (isinstance(reference, np.ndarray) and reference.ndim == 0))):
        return True

    # IMPLEMENTATION NOTE:
    #   modified to allow numeric type mismatches (e.g., int and float;
    #   should be added as option in future (i.e., to disallow it)
    if (isinstance(candidate, match_type) or
            (isinstance(candidate, (list, np.ndarray)) and (issubclass(match_type, (list, np.ndarray)))) or
            (is_number(candidate) and issubclass(match_type, Number)) or
            # IMPLEMENTATION NOTE: Allow UserDict types to match dict (make this an option in the future)
            (isinstance(candidate, UserDict) and match_type is dict) or
            # IMPLEMENTATION NOTE: Allow UserList types to match list (make this an option in the future)
            (isinstance(candidate, UserList) and match_type is list) or
            # IMPLEMENTATION NOTE: This is needed when kwCompatiblityType is not specified
            #                      and so match_type==list as default
            (is_number(candidate) and issubclass(match_type,list)) or
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
        if kargs[kwCompatibilityType] is Enum:
            return isinstance(candidate, Enum) == isinstance(match_type, (Enum, IntEnum, EnumMeta))

        if is_number(candidate):
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
                if isinstance(value, (list, np.ndarray)) and not is_numeric_scalar(value):
                    try:
                        if value.ndim == 0:
                            return recursively_check_elements_for_numeric(value.item())
                    except AttributeError:
                        pass

                    for item in value:
                        if not recursively_check_elements_for_numeric(item):
                            return False
                        else:
                            return True
                else:
                    return is_number(value)
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
                    if (isinstance(candidate, np.ndarray) and isinstance(reference, np.ndarray)
                            and candidate.shape == reference.shape):
                        return True
                    cr = zip(candidate, reference)
                    if all(iscompatible(c, r, **kargs) for c, r in cr):
                        return True
                    # IMPLEMENTATION NOTE:  ??No longer needed given recursive call above
                    # Deal with ints in one and floats in the other
                    # # elif all((isinstance(c, Number) and isinstance(r, Number))
                    # #          for c, r in cr):
                    # #     return True
                else:
                    return False
        else:
            return True
    else:
        return False

#endregion

#region MATHEMATICAL ***************************************************************************************************

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

@beartype
def tensor_power(items, levels: Optional[range] = None, flat=False):
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
#endregion

#region LIST MANAGEMENT ************************************************************************************************

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
    elif isinstance(l, np.ndarray) and l.ndim > 0:
        return list(l)
    else:
        return [l]

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def nesting_depth(l):
    if isinstance(l, np.ndarray):
        l = l.tolist()
    return isinstance(l, list) and max(map(nesting_depth, l)) + 1
#endregion

#region OTHER **********************************************************************************************************
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
def get_deepcopy_with_shared(shared_keys=frozenset()):
    """
        Arguments
        ---------
            shared_keys
                an Iterable containing strings that should be shallow copied

        Returns
        -------
            a __deepcopy__ function
    """
    shared_keys = frozenset(shared_keys)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        try:
            # follow dependency order for Parameters to allow validation involving other parameters
            ordered_dict_keys = sorted(self.__dict__, key=self._dependency_order_key(names=True))
        except AttributeError:
            ordered_dict_keys = self.__dict__

        for k in copy.copy(ordered_dict_keys):
            v = self.__dict__[k]
            if k in shared_keys:
                res_val = v
            else:
                res_val = copy.deepcopy(v, memo)
            setattr(result, k, res_val)
        return result

    return __deepcopy__


def _copy_shared_iterable_elementwise_as_list(obj, shared_types, memo, result_obj=None):
    result = result_obj or list()

    for item in obj:
        try:
            new_item = copy_iterable_with_shared(item, shared_types, memo)
        except TypeError:
            if isinstance(item, shared_types):
                new_item = item
            else:
                new_item = copy.deepcopy(item, memo)
        result.append(new_item)

    return result


def copy_iterable_with_shared(obj, shared_types=type(None), memo=None):
    try:
        shared_types = tuple(shared_types)
    except TypeError:
        shared_types = (shared_types, )

    dict_types = (
        dict,
        collections.UserDict,
        weakref.WeakKeyDictionary,
        weakref.WeakValueDictionary
    )
    list_types = (list, collections.UserList, collections.deque)
    tuple_types = (tuple, set, weakref.WeakSet)
    all_types_using_recursion = dict_types + list_types + tuple_types

    # ContentAddressableList
    cal_component_type = getattr(obj, 'component_type', None)
    if cal_component_type and issubclass(cal_component_type, shared_types):
        return copy.copy(obj)

    if isinstance(obj, dict_types):
        result = copy.copy(obj)
        del_keys = set()
        for (k, v) in obj.items():
            # key can never be unhashable dict or list
            new_k = k if isinstance(k, shared_types) else copy.deepcopy(k, memo)

            if new_k is not k:
                del_keys.add(k)

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
        for k in del_keys:
            del result[k]

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

        result = _copy_shared_iterable_elementwise_as_list(obj, shared_types, memo, result)

        if is_tuple:
            try:
                result = obj.__class__(result)
            except TypeError:
                # handle namedtuple
                result = obj.__class__(*result)
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        if obj.ndim > 0:
            result = _copy_shared_iterable_elementwise_as_list(obj, shared_types, memo)
            result = safe_create_np_array(result)
        elif isinstance(obj, shared_types):
            result = np.array(obj)
        else:
            result = copy.deepcopy(obj)
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
#endregion

#region NUMPY ARRAY METHODS ******************************************************************************************

def are_all_entries_same(arr):
    """Return True if all entries in arr are the same, False otherwise"""
    unique_values = np.unique(arr)
    return len(unique_values) == 1

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
    value = safe_create_np_array(value)

    if dimension == 1:
        if torch and torch.is_tensor(value):
            value = torch.atleast_1d(value)
        else:
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
            if torch and torch.is_tensor(value):
                value = torch.atleast_2d(value)
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
    if value_type in {float, np.float64, np.float32}:
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

def random_matrix(num_rows, num_cols, offset=0.0, scale=1.0):
    """Generate a random matrix

    Calls np.random.rand to generate a 2d np.array with random values and shape (num_rows, num_cols):

        :math:`matrix = (random[0.0:1.0] + offset) * scale

    With the default values of **offset** and **scale**, values of matrix are floats between 0 and 1.
    However, **offset** can be used to center the range on other values (e.g., **offset**=-0.5 centers values on 0),
    and **scale** can be used to narrow or widen the range.  As a conveniuence the keyword 'ZERO_CENTER' can be used
    in place of -.05.

    Arguments
    ----------
    num_rows : int
        specifies number of rows.

    num_cols : int
        specifies number of columns.

    offset : float or 'zero_center'
        specifies amount added to each entry of the matrix before it is scaled.

    scale : float
        specifies amount by which random value + **offset** is multiplicatively scaled.

    Returns
    -------
    2d np.array
    """
    if isinstance(offset,str):
        if offset.upper() == 'ZERO_CENTER':
            offset = -0.5
        else:
            raise UtilitiesError(f"'offset' arg of random_matrix must be a number of 'zero_center'")
    return (np.random.rand(num_rows, num_cols) + offset) * scale

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
            raise KeyError(f"None is not a legal key for '{self.name}'.")
        try:
            return self.data[key]
        except TypeError:
            key_num = self._get_key_for_item(key)
            if key_num is None:
                # raise TypeError(f"'{key}' is not a key in {self.name}.")
                raise TypeError(f"'{key}' is not in {self.name}.")
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
    from psyneulink.core.components.component import Component

    if isinstance(spec, (Number, np.ndarray)):
        return True
    elif (
        isinstance(spec, list)
        and is_numeric(spec)
        and not contains_type(spec, (Component, types.FunctionType))
    ):
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
    def recurse(arr):
        if isinstance(arr, np.ndarray):
            if cast_from is not None and arr.dtype == cast_from:
                return np.asarray(arr, dtype=cast_to)
            elif arr.ndim == 0 or arr.dtype != object:
                return arr

        if isinstance(arr, np.number):
            return np.asarray(arr)

        if cast_from is not None and isinstance(arr, cast_from):
            return cast_to(arr)

        if not isinstance(arr, collections.abc.Iterable) or isinstance(arr, str):
            return arr

        if isinstance(arr, np.matrix):
            if arr.dtype == object:
                return np.asarray([recurse(arr.item(i)) for i in range(arr.size)])
            else:
                return arr

        subarr = [recurse(x) for x in arr]

        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='.*ragged.*', category=np_exceptions.VisibleDeprecationWarning)
            try:
                # the elements are all uniform in shape, so we can use numpy's standard behavior
                return np.asarray(subarr)
            except np_exceptions.VisibleDeprecationWarning:
                pass
            except ValueError as e:
                # Numpy 1.24+ switch jagged array from warning to ValueError.
                # Note that the below call can still raise other ValueErrors.
                if 'The requested array has an inhomogeneous shape' not in str(e):
                    raise

        # the elements are nonuniform, so create an array that just wraps them individually
        # numpy cannot easily create arrays with subarrays of certain dimensions, workaround here
        # https://stackoverflow.com/q/26885508/3131666
        len_subarr = len(subarr)
        elementwise_subarr = np.empty(len_subarr, dtype=np.ndarray)
        for i in range(len_subarr):
            elementwise_subarr[i] = subarr[i]

        return elementwise_subarr

    if not isinstance(arr, collections.abc.Iterable) or isinstance(arr, str):
        # only wrap a noniterable if it's the outermost value
        return np.asarray(arr)
    else:
        return recurse(arr)

# Seeds and randomness

class SeededRandomState(np.random.RandomState):
    def __init__(self, *args, **kwargs):
        # Extract seed
        self.used_seed = (kwargs.get('seed', None) or args[0])[:]
        super().__init__(*args, **kwargs)

    def __deepcopy__(self, memo):
        # There's no easy way to deepcopy parent first.
        # Create new instance and rewrite the state.
        dup = type(self)(seed=self.used_seed)
        dup.set_state(self.get_state())
        return dup

    def seed(self, seed):
        assert False, "Use 'seed' parameter instead of seeding the random state directly"


class _SeededPhilox(np.random.Generator):
    def __init__(self, *args, **kwargs):
        # Extract seed
        self.used_seed = (kwargs.get('seed', None) or args[0])[:]
        state = np.random.Philox([self.used_seed])
        super().__init__(state)

    def __deepcopy__(self, memo):
        # There's no easy way to deepcopy parent first.
        # Create new instance and rewrite the state.
        dup = type(self)(seed=self.used_seed)
        dup.bit_generator.state = self.bit_generator.state
        return dup

    def seed(self, seed):
        assert False, "Use 'seed' parameter instead of seeding the random state directly"


_seed = np.uint32((time.time() * 1000) % 2**31)
def _get_global_seed(offset=1):
    global _seed
    old_seed = _seed
    _seed = (_seed + offset) % 2**31
    return old_seed


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
            try:
                return np.array_equal(x, y)
            except (DeprecationWarning, FutureWarning):
                # both should have len because non-len objects would not
                # have triggered the warnings on == or array_equal
                len_x = len(x)
                if len_x != len(y):
                    return False

                if hasattr(x, 'keys') and hasattr(y, 'keys'):
                    # dictionary-like
                    if x.keys() != y.keys():
                        return False
                    subelements = x.keys()
                elif hasattr(x, 'keys') or hasattr(y, 'keys'):
                    return False
                else:
                    # list-like
                    subelements = range(len_x)

                return all([safe_equals(x[i], y[i]) for i in subelements])



@beartype
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
        sig = _signature_cache[func]
    except KeyError:
        sig = inspect.signature(func)
        _signature_cache[func] = sig

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
    def is_pnl_obj(string):
        try:
            # remove parens to get rid of class instantiations
            string = re.sub(r'\(.*?\)', '', string)
            attr_sequence = string.split('.')
            obj = getattr(psyneulink, attr_sequence[0])

            for item in attr_sequence[1:]:
                obj = getattr(obj, item)

            return True
        except (AttributeError, TypeError):
            return False

    if is_pnl_obj(string):
        return string

    # handle potential psyneulink keyword
    try:
        # insert space between camel case words
        keyword = re.sub('([a-z])([A-Z])', r'\1 \2', string)
        keyword = keyword.upper().replace(' ', '_')

        if is_pnl_obj(keyword):
            return keyword
    except TypeError:
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
        arr_items = iter(arr)
    except TypeError:
        return False

    recurse = not isinstance(arr, np.matrix)
    for a in arr_items:
        if isinstance(a, typ) or (a is not arr and recurse and contains_type(a, typ)):
            return True

    return False


def _is_module_class(class_: type, module: types.ModuleType) -> bool:
    """
    Returns:
        bool: True if **class_** is a member of **module**, False otherwise
    """
    if module.__name__ == class_.__module__:
        try:
            return class_ is getattr(module, class_.__name__)
        except AttributeError:
            pass

    return False


def get_function_sig_default_value(
    function: typing.Union[types.FunctionType, types.MethodType],
    parameter: str
):
    """
        Returns:
            the default value of the **parameter** argument of
            **function** if it exists, or inspect._empty
    """
    try:
        sig = _signature_cache[function]
    except KeyError:
        sig = inspect.signature(function)
        _signature_cache[function] = sig

    try:
        return sig.parameters[parameter].default
    except KeyError:
        return inspect._empty


def toposort_key(
    dependency_dict: typing.Dict[typing.Hashable, typing.Iterable[typing.Any]]
) -> typing.Callable[[typing.Any], int]:
    """
    Creates a key function for python sorting that causes all items in
    **dependency_dict** to be sorted after their dependencies

    Args:
        dependency_dict (typing.Dict[typing.Hashable, typing.Iterable[typing.Any]]):
        a dictionary where values are the dependencies of keys

    Returns:
        typing.Callable[[typing.Any], int]: a key function for python
        sorting
    """
    topo_ordering = list(toposort.toposort(dependency_dict))
    topo_ordering = list(itertools.chain.from_iterable(topo_ordering))

    def _generated_toposort_key(obj):
        try:
            return topo_ordering.index(obj)
        except ValueError:
            return -1

    return _generated_toposort_key


def fill_array(arr: np.ndarray, value: Any):
    """
    Fills all elements of **arr** with **value**, maintaining embedded
    shapes of object-dtype arrays

    Args:
        arr (np.ndarray)
        value (Any)
    """
    if arr.ndim != 0 and arr.dtype == object:
        for item in arr:
            fill_array(item, value)
    else:
        arr.fill(value)


# np.isscalar returns true on non-numeric items
def is_numeric_scalar(obj) -> bool:
    """
        Returns:
            True if **obj** is a numbers.Number or a numpy ndarray
                containing a single numeric value
            False otherwise
    """

    try:
        # getting .item() and checking type is significantly slower
        return obj.ndim == 0 and obj.dtype.kind in {'i', 'f'}
    except (AttributeError, ValueError):
        return isinstance(obj, Number)


def try_extract_0d_array_item(arr: np.ndarray):
    """
        Returns:
            the single item in **arr** if **arr** is a 0-dimensional
            numpy ndarray, otherwise **arr**
    """
    try:
        if arr.ndim == 0:
            return arr.item()
    except AttributeError:
        pass
    return arr


def _extended_array_compare(a, b, comparison_fct: Callable[[Any, Any], bool]) -> bool:
    """
    Recursively determine equality of **a** and **b** using
    **comparison_fct** as an equality function. Shape and size of nested
    arrays must be the same for equality.

    Args:
        a (np.ndarray-like)
        b (np.ndarray-like)
        comparison_fct (Callable[[Any, Any], bool]): a comparison
        function to be called on **a** and **b**. For example,
        numpy.array_equal

    Returns:
        bool: result of comparison_fct(**a**, **b**)
    """
    try:
        a_ndim = a.ndim
    except AttributeError:
        a_ndim = None

    try:
        b_ndim = b.ndim
    except AttributeError:
        b_ndim = None

    # a or b is not a numpy array
    if a_ndim is None or b_ndim is None:
        return comparison_fct(a, b)

    if a_ndim != b_ndim:
        return False

    # b_ndim is also 0
    if a_ndim == 0:
        return comparison_fct(a, b)

    if len(a) != len(b):
        return False

    if a.dtype != b.dtype:
        return False

    # safe to use standard numpy comparison here because not ragged
    if a.dtype != object:
        return comparison_fct(a, b)

    for i in range(len(a)):
        if not _extended_array_compare(a[i], b[i], comparison_fct):
            return False

    return True


def extended_array_equal(a, b, equal_nan: bool = False) -> bool:
    """
    Tests equality like numpy.array_equal, while recursively checking
    object-dtype arrays.

    Args:
        a (np.ndarray-like)
        b (np.ndarray-like)
        equal_nan (bool, optional): Whether to consider NaN as equal.
            See numpy.array_equal. Defaults to False.

    Returns:
        bool: **a** and **b** are equal.

    Example:
        `X = np.array([np.array([0]), np.array([0, 0])], dtype=object)`

        | a | b | np.array_equal | extended_array_equal |
        |---|---|----------------|----------------------|
        | X | X | False          | True                 |
    """
    a = convert_all_elements_to_np_array(a)
    b = convert_all_elements_to_np_array(b)

    return _extended_array_compare(
        a, b, functools.partial(np.array_equal, equal_nan=equal_nan)
    )


def _check_array_attr_equiv(a, b, attr):
    err_msg = '{0} is not a numpy.ndarray'

    try:
        a_val = getattr(a, attr)
    except AttributeError:
        raise ValueError(err_msg.format(a))

    try:
        b_val = getattr(b, attr)
    except AttributeError:
        raise ValueError(err_msg.format(b))

    if a_val != b_val:
        raise ValueError(f'{attr}s {a_val} and {b_val} differ')


def _update_array_in_place(
    target: np.ndarray,
    source: np.ndarray,
    casting: Literal['no', 'equiv', 'safe', 'same_kind', 'unsafe'],
    _dry_run: bool,
    _in_object_dtype: bool,
):
    # enforce dtype equivalence when recursing in an object-dtype target
    # array, because we won't know if np.copyto will succeed on every
    # element until we try
    if _in_object_dtype:
        _check_array_attr_equiv(target, source, 'dtype')

    # enforce shape equivalence so that we know when the python-side
    # values become incompatible with compiled structs
    _check_array_attr_equiv(target, source, 'shape')

    if target.dtype == object:
        len_target = len(target)
        len_source = len(source)

        if len_source != len_target:
            raise ValueError(f'lengths {len_target} and {len_source} differ')

        # check all elements before update to avoid partial update
        if not _dry_run:
            for i in range(len_target):
                _update_array_in_place(
                    target[i],
                    source[i],
                    casting=casting,
                    _dry_run=True,
                    _in_object_dtype=True,
                )

        for i in range(len_target):
            _update_array_in_place(
                target[i],
                source[i],
                casting=casting,
                _dry_run=_dry_run,
                _in_object_dtype=True,
            )
    else:
        np.broadcast(source, target)  # only here to throw error if broadcast fails
        if not _dry_run:
            np.copyto(target, source, casting=casting)


def update_array_in_place(
    target: np.ndarray,
    source: np.ndarray,
    casting: Literal['no', 'equiv', 'safe', 'same_kind', 'unsafe'] = 'same_kind',
):
    """
    Copies the values in **source** to **target**, supporting ragged
    object-dtype arrays.

    Args:
        target (numpy.ndarray): array receiving values
        source (numpy.ndarray): array providing values
        casting (Literal["no", "equiv", "safe", "same_kind", "unsafe"],
            optional): See `numpy.copyto`. Defaults to 'same_kind'.
    """
    _update_array_in_place(
        target=target,
        source=source,
        casting=casting,
        _dry_run=False,
        _in_object_dtype=False
    )


def array_from_matrix_string(
    s: str, row_sep: str = ';', col_sep: str = ' ', dtype: DTypeLike = float
) -> np.ndarray:
    """
    Constructs a numpy array from a string in forms like '1 2; 3 4'
    replicating the function of the numpy.matrix constructor.

    Args:
        s (str): matrix descriptor
        row_sep (str, optional): separator for matrix rows. Defaults to ';'.
        col_sep (str, optional): separator for matrix columns. Defaults to ' '.
        dtype (DTypeLike, optional): dtype of result array. Defaults to float.

    Returns:
        np.ndarray: array representation of **s**
    """
    rows = s.split(row_sep)
    arr = []
    for r in rows:
        # filter empty columns, commonly in form like '1 2; 3 4'
        arr.append([c for c in r.split(col_sep) if len(c)])

    return np.asarray(arr, dtype=dtype)

#endregion

#region PYTORCH TENSOR METHODS *****************************************************************************************

def get_torch_tensor(value, dtype, device):
    if device == MPS or device == torch.device(MPS):
        if isinstance(value, torch.Tensor):
            return torch.tensor(value, dtype=torch.float32, device=device)
        return torch.tensor(np.array(value, dtype=np.float32), device=device)
    else:
        if dtype in {np.float32, torch.float32}:
            return torch.tensor(value, device=device).float()
        elif dtype in {np.float64, torch.float64}:
            return torch.tensor(value, device=device).double()
        else:
            return torch.tensor(value, device=device)

def safe_create_np_array(value):
    with warnings.catch_warnings():

        # If we have a torch tensor, allow it to pass through unchanged
        if torch and torch.is_tensor(value):
            return value

        warnings.filterwarnings('error', category=np_exceptions.VisibleDeprecationWarning)
        try:
            try:
                return np.asarray(value)
            except np_exceptions.VisibleDeprecationWarning:
                return np.asarray(value, dtype=object)
            except ValueError as e:
                # Numpy 1.24+ switch jagged array from warning to ValueError.
                # Note that the below call can still raise other ValueErrors.
                if 'The requested array has an inhomogeneous shape' in str(e):
                    return np.asarray(value, dtype=object)
                raise

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


def get_module_file_prefix(module: Union[str, types.ModuleType]) -> str:
    """
    Gets the file prefix of **module**, which may be used with
    `get_stacklevel_skip_file_prefixes` or the `skip_file_prefixes`
    argument of :func:`warnings.warn` (python 3.12+)

    Args:
        module (Union[str, types.ModuleType]): a python module or a name
        of a module

    Returns:
        str: the file path of **module**, excluding __init__.py
    """
    try:
        module = importlib.import_module(module)
    except AttributeError:
        # ModuleType
        pass

    module = inspect.getfile(module)

    if module.endswith('__init__.py'):
        module = os.path.dirname(module)

    return module


def get_stacklevel_skip_file_prefixes(
    modules: Iterable[Union[str, types.ModuleType]]
) -> int:
    """
    Gets a value for the `stacklevel` argument of :func:`warnings.warn`
    or :func:`logging.log` that corresponds to the outermost stack frame
    that does not belong to any of **modules** as determined by their
    file paths. Functions similarly to the `skip_file_prefixes` argument
    of :func:`warnings.warn` (python 3.12+).

    Args:
        modules (Iterable[Union[str, types.ModuleType]]): python modules
        or names of modules to skip

    Returns:
        int: the outermost stacklevel that excludes **modules**
    """
    prefixes = [get_module_file_prefix(p) for p in modules]

    res = 1
    # skip this function
    for i, frame_info in enumerate(inspect.stack()[1:]):
        for p in prefixes:
            if frame_info.frame.f_code.co_filename.startswith(p):
                break
        else:
            return i + 1
    return res

#endregion
