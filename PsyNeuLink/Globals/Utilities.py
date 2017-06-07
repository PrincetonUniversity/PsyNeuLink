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

TYPE CHECKING VALUE COMPARISON
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   PsyNeuLink-specific typechecking functions are in the `Component` module

* `parameter_spec`
* `optional_parameter_spec`
* `is_matrix
* `is_matrix_spec`
* `is_numeric`
* `is_numeric_or_none`
* `iscompatible`
* `is_value_spec`
* `is_unit_interval`

ENUM
~~~~

* `Autonumber`
* `Modulation`
* `get_modulationOperation_name`

KVO
~~~

.. note::
   This is for potential future use;  not currently used by PsyNeuLink objects

* observe_value_at_keypath

OTHER
~~~~~
* `merge_param_dicts`
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

"""

import warnings
# THE FOLLOWING CAUSES ALL WARNINGS TO GENERATE AN EXCEPTION:
warnings.filterwarnings("error")

import numbers
import numpy as np
from enum import EnumMeta
from enum import IntEnum
import typecheck as tc
import inspect

from PsyNeuLink.Globals.Defaults import *
from PsyNeuLink.Globals.Keywords import *

from PsyNeuLink.Globals.TimeScale import *


class UtilitiesError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


MODULATION_OVERRIDE = 'Modulation.OVERRIDE'
MODULATION_MULTIPLY = 'Modulation.MULTIPLY'
MODULATION_ADD = 'Modulation.ADD'


class Modulation(Enum):
    DISABLED = 0
    ADD = lambda runtime, default : runtime + default
    MULTIPLY = lambda runtime, default : runtime * default
    OVERRIDE = lambda runtime, default : runtime

def is_modulation_operation(val):
    # try:
    #     val(0,0)
    #     return True
    # except:
    #     return False
    return get_modulationOperation_name(val)

def get_modulationOperation_name(operation):
        x = operation(1,2)
        if x is 1:
            return MODULATION_OVERRIDE
        elif x is 2:
            return MODULATION_MULTIPLY
        elif x is 3:
            return MODULATION_ADD
        else:
            return False



class AutoNumber(IntEnum):
    """Autonumbers IntEnum type

    First item in list of declarations is numbered 0;
    Others incremented by 1

    Sample:

        class NumberedList(AutoNumber):
            FIRST_ITEM = ()
            SECOND_ITEM = ()

        >>>NumberedList.FIRST_ITEM.value
         0
        >>>NumberedList.SECOND_ITEM.value
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


def parameter_spec(param):
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
    from PsyNeuLink.Components.Functions.Function import function_type
    from PsyNeuLink.Components.Projections.Projection import Projection
    from PsyNeuLink.Components.Component import parameter_keywords

    if (isinstance(param, (numbers.Number,
                           np.ndarray,
                           list,
                           tuple,
                           function_type,
                           Projection)) or
        (inspect.isclass(param) and issubclass(param, Projection)) or
        param in parameter_keywords):
        return True
    return False


def is_numeric_or_none(x):
    if x is None:
        return True
    return is_numeric(x)


def is_numeric(x):
    return iscompatible(x, **{kwCompatibilityNumeric:True, kwCompatibilityLength:0})


def is_matrix_spec(m):
    return isinstance(m, str) and m in MATRIX_KEYWORD_VALUES


def is_matrix(m):
    if is_matrix_spec(m):
        return True
    if isinstance(m, (list, np.ndarray, np.matrix)):
        return True
    if callable(m):
        try:
            return is_matrix(m())
        except:
            return False


def is_distance_metric(s):
    if s in DISTANCE_METRICS:
        return True
    else:
        return False


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
                    if kwCompatibilityNumeric is :keyword:`True`, all elements of candidate must be numbers
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
        kwCompatibilityNumeric ("number": <bool> (default: :keyword:`True`)  (spec local_variable: number_only)
            If kwCompatibilityNumeric is :keyword:`True`, candidate must be either numeric or a list or tuple of
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

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            if reference and (candidate == reference):
                return True
        except Warning:
            # IMPLEMENTATION NOTE: np.array generates the following warning:
            # FutureWarning: elementwise comparison failed; returning scalar instead,
            #     but in the future will perform elementwise comparison
            pass
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
        if kargs[kwCompatibilityLength] and isinstance(reference, (list, tuple, dict, np.ndarray)):
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
    # # from PsyNeuLink.Components.Functions.Function import matrix_spec
    if is_matrix_spec(reference):
        return is_matrix(candidate)

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
        # MODIFIED 6/7/16 TO ALLOW ndarray AND list TO MATCH;
            # if not isinstance(candidate, list) and not isinstance(candidate, tuple):
            if not isinstance(candidate, (list, tuple, np.ndarray)):
        # END MODIFIED 6/7/16
                return False
            if (not all(isinstance(elem, numbers.Number) for elem in candidate) or
                    not all(isinstance(elem, numbers.Number) for elem in candidate)):
                return False
        if isinstance(candidate, (list, tuple, dict, np.ndarray)):
            if not match_length:
                return True
            else:
                if len(candidate) == match_length:
                    # No reference,so item by item comparison is not relevant
                    if reference is None:
                        return True
                    # If reference was provided, compare element by element
                    elif all(isinstance(c, type(r)) for c, r in zip(candidate,reference)):
                        return True
                    # MODIFIED 10/28/16 NEW:
                    # Deal with ints in one and floats in the other
                    elif all((isinstance(c, numbers.Number) and isinstance(r, numbers.Number))
                             for c, r in zip(candidate,reference)):
                        return True
                    # MODIFIED 10/28/16 END
                    else:
                        return False
                else:
                    return False
        else:
            return True
    else:
        return False

def merge_param_dicts(source, specific, general):
    """Search source dict for specific and general dicts, merge specific with general, and return merged

    Description:
        - used to merge only a subset of dicts in param set (that may have several dicts
        - allows dicts to be referenced by name (e.g., paramName) rather than by object
        - searches source dict for specific and general dicts
        - if both are found, merges them, with entries from specific overwriting any duplicates in general
        - if only one is found, returns just that dict
        - if neither are found, returns empty dict

    Arguments:
        - source (dict): container dict (entries are dicts); search entries for specific and general dicts
        - specific (dict or str): if str, use as key to look for specific dict in source, and check that it is a dict
        - general (dict or str): if str, use as key to look for general dict in source, and check that it is a dict


    :param source: (dict)
    :param specific: (dict or str)
    :param general: (dict or str)
    :return merged: (dict)
    """

    # Validate source as dict
    if not source:
        return
    if not isinstance(source, dict):
        raise UtilitiesError("merge_param_dicts: source {0} must be a dict".format(source))

    # Get specific and make sure it is a dict
    if isinstance(specific, str):
        try:
            specific = source[specific]
        except (KeyError, TypeError):
            specific = {}
    if not isinstance(specific, dict):
        raise UtilitiesError("merge_param_dicts: specific {0} must be dict or the name of one in {1}".
                        format(specific, source))

    # Get general and make sure it is a dict
    if isinstance(general, str):
        try:
            general = source[general]
        except (KeyError, TypeError):
            general = {}
    if not isinstance(general, dict):
        raise UtilitiesError("merge_param_dicts: general {0} must be dict or the name of one in {1}".
                        format(general, source))

# FIX: SHOULD THIS BE specific, NOT source???
#     # MODIFIED 7/16/16 OLD:
#     return general.update(source)
    # MODIFIED 7/16/16 NEW:
    return general.update(specific)

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


#region NUMPY ARRAY METHODS ******************************************************************************************

def np_array_less_than_2d(array):
    if not isinstance(array, np.ndarray):
        raise UtilitiesError("Argument to np_array_less_than_2d() ({0}) must be an np.ndarray".format(array))
    if array.ndim <= 1:
        return True
    else:
        return False

def convert_to_np_array(value, dimension):
    """Convert value to np.array if it is not already

    Check whether value is already an np.ndarray and whether it has non-numeric elements

    Return np.array of specified dimension

    :param value:
    :return:
    """
    if value is None:
        return None

    if dimension is 1:
        value = np.atleast_1d(value)
    elif dimension is 2:
        from numpy import ndarray
        # if isinstance(value, ndarray) and value.dtype==object and len(value) == 2:
        value = np.array(value)
        # if value.dtype==object and len(value) == 2:
        # Array is made up of non-uniform elements, so treat as 2d array and pass
        if value.dtype==object:
            pass
        else:
            value = np.atleast_2d(value)
    else:
        raise UtilitiesError("dimensions param ({0}) must be 1 or 2".format(dimension))
    if 'U' in repr(value.dtype):
        raise UtilitiesError("{0} has non-numeric entries".format(value))
    return value

def type_match(value, value_type):
    if isinstance(value, value_type):
        return value
    if value_type in {int, np.int64}:
        return int(value)
    if value_type in {float, np.float64}:
        return float(value)
    if value_type is np.ndarray:
        return np.array(value)
    if value_type is list:
        return list(value)
    if value_type is None:
        return None
    if value_type is type(None):
        raise UtilitiesError("PROGRAM ERROR: template provided to type_match for {} is \'None\'".format(value))
    raise UtilitiesError("Type of {} not recognized".format(value_type))

def get_value_from_array(array):
    """Extract numeric value from array, preserving numeric type
    :param array:
    :return:
    """

def random_matrix(sender, receiver, range=1, offset=0):
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
    return (range * np.random.rand(sender, receiver)) + offset

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
        string = "\'" + name +  "\'" + ' ' + type.lower()
        # string = name + ' ' + type.lower()
    return string
#endregion


from collections import UserDict, OrderedDict
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
        if not key in self._ordered_keys:
            self._ordered_keys.append(key)
    def keys(self):
        return self._ordered_keys
    def copy(self):
        return self.data.copy()

from collections import UserList
class ContentAddressableList(UserList):
    """
    ContentAddressableList( component_type, key=None, list=None)
    
    Implements dict-like list, that can be keyed by the names of the `compoments <Component>` in its entries.

    Supports:
      * getting and setting entries in the list using keys (string), in addition to numeric indices.
        the key to use is specified by the **key** arg of the constructor, and must be a string attribute;
        * for getting an entry:
          the key must match the keyed attribute of a component in the list; otherwise an exception is raised;
        * for setting an entry:
            - the key must match the key of the component being assigned;
            - if there is already a component in the list the keyed vaue of which matches the key, it is replaced;
            - if there is no component in the list the keyed attribute of which matches the key,
              the component is appended to the list;
        * for getting lists of the names, values of the keyed attributes, and values of the `value <Component.value>`
            attributes of components in the list.

    IMPLEMENTATION NOTE:
        This class allows components to be maintained in lists, while providing ordered storage
        and the convenience access and assignment by name (e.g., akin to a dict).
        Lists are used (instead of a dict or OrderedDict) since:
            - ordering is in many instances convenient, and in some critical (e.g., for consistent mapping from
                collections of states to other variables, such as lists of their values);
            - they are most commonly accessed either exhaustively (e.g., in looping through them during execution),
                or by key (e.g., to get the first, "primary" one), which makes the efficiencies of a dict for
                accessing by key/name less critical;
            - the number of states in a collection for a given mechanism is likely to be small so that, even when
                accessed by key/name, the inefficiencies of searching a list are likely to be inconsequential.

    Arguments
    ---------

    component_type : Class
        specifies the class of the items in the list.

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

    def __init__(self, component_type, key=None, list=None, **kwargs):
        self.component_type = component_type
        self.key = key or 'name'
        if not isinstance(component_type, type):
            raise UtilitiesError("component_type arg for {} ({}) must be a class"
                                 .format(self.__class__.__name__, component_type))
        if not isinstance(self.key, str):
            raise UtilitiesError("key arg for {} ({}) must be a string".
                                 format(self.__class__.__name__, self.key))
        if not hasattr(component_type, self.key):
            raise UtilitiesError("key arg for {} (\'{}\') must be an attribute of {}".
                                 format(self.__class__.__name__,
                                        self.key,
                                        component_type.__name__))
        if list is not None:
            if not all(isinstance(obj, self.component_type) for obj in list):
                raise UtilitiesError("All of the items in the list arg for {} "
                                     "must be of the type specified in the component_type arg ({})"
                                     .format(self.__class__.__name__, self.component_type.__name__))
        UserList.__init__(self, list, **kwargs)

    def __repr__(self):
        return '[\n\t{0}\n]'.format('\n\t'.join(['{0}\t{1}\t{2}'.format(i, self[i].name, repr(self[i].value)) for i in range(len(self))]))

    def __getitem__(self, key):
        if key is None:
            raise KeyError("None is not a legal key for {}".format(self.__class__.__name__))
        try:
            return self.data[key]
        except TypeError:
            key_num = self._get_key_for_item(key)
            if key_num is None:
                raise TypeError("\'{}\' is not a key in the {} being being addressed".
                                format(key, self.__class__.__name__))
            return self.data[key_num]


    def __setitem__(self, key, value):
        # For efficiency, first assume the key is numeric (duck typing in action!)
        try:
            self.data[key] = value
        # If it is not
        except TypeError:
            # It must be a string
            if not isinstance(key, str):
                raise UtilitiesError("Non-numer key used for {} ({})must be a string)".
                                      format(self.__class__.__name__, key))
            # The specified string must also match the value of the attribute of the class used for addressing
            if not key == value.name:
                raise UtilitiesError("The key of the entry for {} {} ({}) "
                                     "must match the value of its {} attribute ({})".
                                      format(self.__class__.__name__,
                                             value.__class__.__name__,
                                             key,
                                             self.key,
                                             getattr(value, self.key)))
            #
            key_num = self._get_key_for_item(key)
            if key_num is not None:
                self.data[key_num] = value
            else:
                self.data.append(value)

    def __contains__(self, item):
        if super().__contains__(item):
            return True
        else:
            return any(item == obj.name for obj in self.data)

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
            raise UtilitiesError("{} is not a legal key for {} (must be number, string or State)".
                                  format(key, self.key))

    def __delitem__(self, key):
        if key is None:
            raise KeyError("None is not a legal key for {}".format(self.__class__.__name__))
        try:
            del self.data[key]
        except TypeError:
            key_num = self._get_key_for_item(key)
            del self.data[key_num]

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

    @property
    def values(self):
        """Return list of values of components in the list.
        Returns
        -------
        values :  list
            list of the values of the `value <Component.value>` attributes of components in the list.
        """
        return [getattr(item, VALUE) for item in self.data]

    @property
    def values_as_lists(self):
        """Return list of values of components in the list, each converted to a list.
        Returns
        -------
        values :  list
            list of list values of the `value <Component.value>` attributes of components in the list,
        """
        return [np.ndarray.tolist(getattr(item, VALUE)) for item in self.data]


def is_value_spec(spec):
    if isinstance(spec, (int, float, list, np.ndarray)):
        return True
    else:
        return False


def is_unit_interval(spec):
    if isinstance(spec, (int, float)) and 0 <= spec <= 1:
        return True
    else:
        return False


def make_readonly_property(val):
    """Return property that provides read-only access to its value
    """

    def getter(self):
        return val

    def setter(self, val):
        raise UtilitiesError("{} is read-only property of {}".format(val, self.__class__.__name__))

    # Create the property
    prop = property(getter).setter(setter)
    return prop


def get_class_attributes(cls):
    boring = dir(type('dummy', (object,), {}))
    return [item
            for item in inspect.getmembers(cls)
            if item[0] not in boring]