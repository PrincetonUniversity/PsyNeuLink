#
# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *****************************************  COMBINATION FUNCTIONS  ****************************************************

"""
* `Concatenate`
* `Rearrange`
* `Reduce`
* `LinearCombination`
* `CombineMeans`
* `MatrixTransform`
* `PredictionErrorDeltaFunction`

Overview
--------

Functions that combine multiple items with the same shape, yielding a result with a single item that has the same
shape as the individual items.

All Transformfunctions must have two attributes - **multiplicative_param** and **additive_param** -
each of which is assigned the name of one of the function's parameters;
this is for use by ModulatoryProjections (and, in particular, GatingProjections,
when the TransformFunction is used as the function of an InputPort or OutputPort).


"""

import numbers
import types
import warnings

import numpy as np

try:
    import torch
except ImportError:
    torch = None
from beartype import beartype

from psyneulink._typing import Optional, Union, Literal

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.functions import function
from psyneulink.core.components.functions.function import (
    Function_Base, FunctionError, FunctionOutputType, function_keywords, get_matrix)
from psyneulink.core.components.shellclasses import Projection
from psyneulink.core.globals.keywords import (
    ADDITIVE_PARAM, ARRANGEMENT, COMBINATION_FUNCTION_TYPE, COMBINE_MEANS_FUNCTION, CONCATENATE_FUNCTION,
     CROSS_ENTROPY, DEFAULT_VARIABLE, DOT_PRODUCT, EXPONENTS,
     HAS_INITIALIZERS, HOLLOW_MATRIX, IDENTITY_MATRIX, LINEAR_COMBINATION_FUNCTION, L0,
     MATRIX, MATRIX_KEYWORD_NAMES, MATRIX_TRANSFORM_FUNCTION,  MULTIPLICATIVE_PARAM, NORMALIZE,
     OFFSET, OPERATION, PREDICTION_ERROR_DELTA_FUNCTION, PRODUCT,
     REARRANGE_FUNCTION, RECEIVER, REDUCE_FUNCTION, SCALE, SUM, WEIGHTS, PREFERENCE_SET_NAME)
from psyneulink.core.globals.utilities import (
    convert_all_elements_to_np_array, convert_to_np_array, is_numeric, is_matrix_keyword, is_numeric_scalar,
    np_array_less_than_2d, ValidParamSpecType)
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.parameters import Parameter, check_user_specified, copy_parameter_value
from psyneulink.core.globals.preferences.basepreferenceset import \
    REPORT_OUTPUT_PREF, ValidPrefSet, PreferenceEntry, PreferenceLevel

__all__ = ['TransformFunction', 'Concatenate', 'CombineMeans', 'Rearrange', 'Reduce',
           'LinearCombination', 'MatrixTransform', 'PredictionErrorDeltaFunction']

class TransformFunction(Function_Base):
    """Function that combines multiple items, yielding a result with the same shape as its operands

    All Transformfunctions must have two attributes - multiplicative_param and additive_param -
        each of which is assigned the name of one of the function's parameters;
        this is for use by ModulatoryProjections (and, in particular, GatingProjections,
        when the TransformFunction is used as the function of an InputPort or OutputPort).

    """
    componentType = COMBINATION_FUNCTION_TYPE

    class Parameters(Function_Base.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <TransformFunction.variable>`

                    :default value: numpy.array([0])
                    :type: ``numpy.ndarray``
                    :read only: True
        """
        # variable = np.array([0, 0])
        variable = Parameter(np.array([0]), read_only=True, pnl_internal=True, constructor_argument='default_variable')

    def _gen_llvm_load_param(self, ctx, builder, params, param_name, index, default):
        param_ptr = ctx.get_param_or_state_ptr(builder, self, param_name, param_struct_ptr=params)
        param_type = param_ptr.type.pointee
        if isinstance(param_type, pnlvm.ir.LiteralStructType):
            assert len(param_type) == 0
            return ctx.float_ty(default)
        elif isinstance(param_type, pnlvm.ir.ArrayType):
            index = ctx.int32_ty(0) if len(param_type) == 1 else index
            param_ptr = builder.gep(param_ptr, [ctx.int32_ty(0), index])
        return builder.load(param_ptr)

    def _gen_llvm_function_body(self, ctx, builder, params, _, arg_in, arg_out, *, tags:frozenset):
        # Sometimes we arg_out to 2d array
        arg_out = pnlvm.helpers.unwrap_2d_array(builder, arg_out)

        with pnlvm.helpers.array_ptr_loop(builder, arg_out, "linear") as args:
            self._gen_llvm_combine(ctx=ctx, vi=arg_in, vo=arg_out, params=params, *args)
        return builder


class Concatenate(TransformFunction):  # ------------------------------------------------------------------------
    """
    Concatenate(                                   \
         default_variable=class_defaults.variable, \
         scale=1.0,                                \
         offset=0.0,                               \
         params=None,                              \
         owner=None,                               \
         prefs=None,                               \
    )

    .. _Concatenate:

    Concatenates items in outer dimension (axis 0) of `variable <Concatenate.variable>` into a single array,
    optionally scaling and/or adding an offset to the result after concatenating.

    `function <Concatenate.function>` returns a 1d array with length equal to the sum of the lengths of the items
    in `variable <Concatenate.variable>`.

    `derivative <Concatenate.derivative>` returns `scale <Concatenate.slope>`.


    Arguments
    ---------

    default_variable : list or np.array : default class_defaults.variable
        specifies a template for the value to be transformed and its default value;  all entries must be numeric.

    scale : float
        specifies a value by which to multiply each element of the output of `function <Concatenate.function>`
        (see `scale <Concatenate.scale>` for details)

    offset : float
        specifies a value to add to each element of the output of `function <Concatenate.function>`
        (see `offset <Concatenate.offset>` for details)

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

    default_variable : list or np.array
        contains template of array(s) to be concatenated.

    scale : float
        value is applied multiplicatively to each element of the concatenated, before  applying the `offset
        <Concatenate.offset>` (if it is specified).

    offset : float
        value is added to each element of the concatentated array, after `scale <Concatenate.scale>` has been
        applied (if it is specified).

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a default is
        assigned by FunctionRegistry (see `Registry_Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see `Preferences` for
        details).
    """
    componentName = CONCATENATE_FUNCTION


    class Parameters(TransformFunction.Parameters):
        """
            Attributes
            ----------

                changes_shape
                    see `changes_shape <Function_Base.changes_shape>`

                    :default value: True
                    :type: bool

                offset
                    see `offset <Concatenate.offset>`

                    :default value: 0.0
                    :type: ``float``

                scale
                    see `scale <Concatenate.scale>`

                    :default value: 1.0
                    :type: ``float``
        """
        scale = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        offset = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        changes_shape = Parameter(True, stateful=False, loggable=False, pnl_internal=True)

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 scale: Optional[ValidParamSpecType] = None,
                 offset: Optional[ValidParamSpecType] = None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):

        super().__init__(
            default_variable=default_variable,
            scale=scale,
            offset=offset,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _validate_variable(self, variable, context=None):
        """Insure that list or array is 1d and that all elements are numeric

        Args:
            variable:
            context:
        """
        variable = super()._validate_variable(variable=variable, context=context)
        if not is_numeric(variable):
            raise FunctionError("All elements of {} must be scalar values".
                                format(self.__class__.__name__))
        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate scale and offset parameters

        Check that SCALE and OFFSET are scalars.
        """

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if SCALE in target_set and target_set[SCALE] is not None:
            scale = target_set[SCALE]
            if not is_numeric_scalar(scale):
                raise FunctionError("{} param of {} ({}) must be a scalar".format(SCALE, self.name, scale))

        if OFFSET in target_set and target_set[OFFSET] is not None:
            offset = target_set[OFFSET]
            if not is_numeric_scalar(offset):
                raise FunctionError("{} param of {} ({}) must be a scalar".format(OFFSET, self.name, offset))

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """Use numpy hstack to concatenate items in outer dimension (axis 0) of variable.

        Arguments
        ---------

        variable : list or np.array : default class_defaults.variable
           a list or np.array of numeric values.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        Concatenated array of items in variable : array
            in an array that is one dimension less than `variable <Concatenate.variable>`.

        """
        scale = self._get_current_parameter_value(SCALE, context)
        offset = self._get_current_parameter_value(OFFSET, context)

        result = np.hstack(variable) * scale + offset

        return self.convert_output_type(result)

    @handle_external_context()
    def derivative(self, input=None, output=None, covariates=None, context=None):
        """
        derivative(input)

        Derivative of `function <Concatenate._function>` at **input**.

        Arguments
        ---------

        input : number
            value of the input to the function at which derivative is to be taken.

        covariates : 2d np.array : default class_defaults.variable[1:]
            the input(s) to the Concatenate function other than the one for which the derivative is being
            computed;  these are ignored and are accepted for consistency with other functions.

        Returns
        -------

        Scale of function :  number or array

        """

        return self._get_current_parameter_value(SCALE, context)

    def _gen_pytorch_fct(self, device, context=None):
        scale = self._get_pytorch_fct_param_value('scale', device, context)
        offset = self._get_pytorch_fct_param_value('offset', device, context)
        # return lambda x: torch.concatenate(tuple(x)) * scale + offset
        return lambda x: torch.hstack(tuple(x)) * scale + offset


class Rearrange(TransformFunction):  # ------------------------------------------------------------------------
    """
    Rearrange(                                     \
         default_variable=class_defaults.variable, \
         arrangement=None,                         \
         scale=1.0,                                \
         offset=0.0,                               \
         params=None,                              \
         owner=None,                               \
         prefs=None,                               \
    )

    .. _Rearrange:

    Rearranges items in outer dimension (axis 0) of `variable <Rearrange.variable>`, as specified by **arrangement**,
    optionally scaling and/or adding an offset to the result after concatenating.

    .. _Rearrange_Arrangement:

    The **arrangement** argument specifies how to rearrange the items of `variable <Rearrange.variable>`, possibly
    concatenating subsets of them into single 1d arrays.  The specification must be an integer, a tuple of integers,
    or a list containing either or both.  Each integer must be an index of an item in the outer dimension (axis 0) of
    `variable <Rearrange.variable>`.  Items referenced in a tuple are concatenated in the order specified into a single
    1d array, and that 1d array is included in the resulting 2d array in the order it appears in **arrangement**.
    If **arrangement** is specified, then only the items of `variable <Rearrange.variable>` referenced in the
    specification are included in the result; if **arrangement** is not specified, all of the items of `variable
    <Rearrange.variable>` are concatenated into a single 1d array (i.e., it functions identically to `Concatenate`).

    `function <Rearrange.function>` returns a 2d array with the items of `variable` rearranged
    (and possibly concatenated) as specified by **arrangement**.

    Examples
    --------

    >>> r = Rearrange(arrangement=[(1,2),(0)])
    >>> print(r(np.array([[0,0],[1,1],[2,2]])))
    [array([1., 1., 2., 2.]) array([0., 0.])]

    >>> r = Rearrange()
    >>> print(r(np.array([[0,0],[1,1],[2,2]])))
    [0. 0. 1. 1. 2. 2.]


    Arguments
    ---------

    default_variable : list or np.array : default class_defaults.variable
        specifies a template for the value to be transformed and its default value;  all entries must be numeric.

    arrangement : int, tuple, or list : default None
        specifies ordering of items in `variable <Rearrange.variable>` and/or ones to concatenate.
        (see `above <Rearrange_Arrangement>` for details).

    scale : float
        specifies a value by which to multiply each element of the output of `function <Rearrange.function>`
        (see `scale <Rearrange.scale>` for details).

    offset : float
        specifies a value to add to each element of the output of `function <Rearrange.function>`
        (see `offset <Rearrange.offset>` for details).

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

    default_variable : list or np.array
        contains template of array(s) to be concatenated.

    arrangement : list of one or more tuples
        determines ordering of items in `variable <Rearrange.variable>` and/or ones to concatenate
        (see `above <Rearrange_Arrangement>` for additional details).

    scale : float
        value is applied multiplicatively to each element of the concatenated, before  applying the `offset
        <Rearrange.offset>` (if it is specified).

    offset : float
        value is added to each element of the concatentated array, after `scale <Rearrange.scale>` has been
        applied (if it is specified).

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
    componentName = REARRANGE_FUNCTION

    class Parameters(TransformFunction.Parameters):
        """
            Attributes
            ----------

                arrangement
                    see `arrangement <Rearrange_Arrangement>`

                    :default value: None
                    :type:

                offset
                    see `offset <Rearrange.offset>`

                    :default value: 0.0
                    :type: ``float``

                scale
                    see `scale <Rearrange.scale>`

                    :default value: 1.0
                    :type: ``float``
        """
        arrangement = Parameter(None, modulable=False)
        scale = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        offset = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 scale: Optional[ValidParamSpecType] = None,
                 offset: Optional[ValidParamSpecType] = None,
                 arrangement:Optional[Union[int, tuple, list]]=None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):

        super().__init__(
            default_variable=default_variable,
            arrangement=arrangement,
            scale=scale,
            offset=offset,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _validate_variable(self, variable, context=None):
        """Insure that all elements are numeric and that list or array is at least 2d
        """
        variable = super()._validate_variable(variable=variable, context=context)
        if not is_numeric(variable):
            raise FunctionError(
                    f"All elements of {repr(DEFAULT_VARIABLE)} for {self.__class__.__name__} must be scalar values.")

        if self.parameters.variable._user_specified and np.array(variable).ndim<2:
            raise FunctionError(f"{repr(DEFAULT_VARIABLE)} for {self.__class__.__name__} must be at least 2d.")

        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate arrangement, scale and offset parameters"""

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if ARRANGEMENT in target_set and target_set[ARRANGEMENT] is not None:

            # If default_varilable was specified by user, validate indices in arrangement
            owner_str = ''
            if self.owner:
                owner_str = f' of {self.owner.name}'
            for i in self._indices:
                if not isinstance(i, int):
                    raise FunctionError(f"Index specified in {repr(ARRANGEMENT)} arg for "
                                        f"{self.name}{owner_str} ({repr(i)}) is not an int.")
                if self.parameters.variable._user_specified:
                    try:
                        self.parameters.variable.default_value[i]
                    except IndexError:
                        raise FunctionError(f"Index ({i}) specified in {repr(ARRANGEMENT)} arg for "
                                            f"{self.name}{owner_str} is out of bounds for its {repr(DEFAULT_VARIABLE)} "
                                            f"arg (max index = {len(self.parameters.variable.default_value) - 1}).")

        # Check that SCALE and OFFSET are scalars.
        if SCALE in target_set and target_set[SCALE] is not None:
            scale = target_set[SCALE]
            if not is_numeric_scalar(scale):
                raise FunctionError("{} param of {} ({}) must be a scalar".format(SCALE, self.name, scale))

        if OFFSET in target_set and target_set[OFFSET] is not None:
            offset = target_set[OFFSET]
            if not is_numeric_scalar(offset):
                raise FunctionError("{} param of {} ({}) must be a scalar".format(OFFSET, self.name, offset))

    def _instantiate_attributes_before_function(self, function=None, context=None):
        """Insure all items of arrangement are tuples and compatibility with default_variable

        If arrangement is specified, convert all items to tuples
        If default_variable is NOT specified, assign with length in outer dimension = max index in arragnement
        If default_variable IS _user_specified, compatiblility with arrangement is checked in _validate_params
        """

        arrangement = self.parameters.arrangement.get()

        if arrangement is not None:
            # Insure that all items are tuples
            self.parameters.arrangement.set([item if isinstance(item,tuple) else tuple([item]) for item in arrangement])

        if not self.parameters.variable._user_specified:
            # Reshape variable.default_value to match maximum index specified in arrangement
            self.parameters.variable.default_value = np.zeros((max(self._indices) + 1, 1))

        super()._instantiate_attributes_before_function(function, context)

    @property
    def _indices(self):
        arrangement = list(self.parameters.arrangement.get())
        items = [list(item) if isinstance(item, tuple) else [item] for item in arrangement]
        indices = []
        for item in items:
            indices.extend(item)
        return indices

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """Rearrange items in outer dimension (axis 0) of variable according to `arrangement <Rearrange.arrangement>`.

        Arguments
        ---------

        variable : list or np.array : default class_defaults.variable
           a list or np.array of numeric values.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        Rearranged items of outer dimension (axis 0) of **variable** : array
            in a 2d array.
        """
        variable = np.atleast_2d(variable)

        scale = self._get_current_parameter_value(SCALE, context)
        offset = self._get_current_parameter_value(OFFSET, context)
        arrangement = self.parameters.arrangement.get(context)

        if arrangement is None:
            result = np.hstack(variable) * scale + offset

        else:
            try:
                result = []
                for item in arrangement:
                    stack = []
                    for index in item:
                        stack.append(variable[index])
                    result.append(np.hstack(tuple(stack)))
                result = convert_to_np_array(result) * scale + offset
            except IndexError:
                assert False, f"PROGRAM ERROR: Bad index specified in {repr(ARRANGEMENT)} arg -- " \
                    f"should have been caught in _validate_params or _instantiate_attributes_before_function"

        return self.convert_output_type(result, FunctionOutputType.NP_2D_ARRAY)


class Reduce(TransformFunction):  # ------------------------------------------------------------------------
    # FIX: CONFIRM THAT 1D KWEIGHTS USES EACH ELEMENT TO SCALE CORRESPONDING VECTOR IN VARIABLE
    # FIX  CONFIRM THAT LINEAR TRANSFORMATION (OFFSET, SCALE) APPLY TO THE RESULTING ARRAY
    # FIX: CONFIRM RETURNS LIST IF GIVEN LIST, AND SIMLARLY FOR NP.ARRAY
    """
    Reduce(                                       \
         default_variable=class_defaults.variable, \
         weights=None,                            \
         exponents=None,                          \
         operation=SUM,                           \
         scale=1.0,                               \
         offset=0.0,                              \
         params=None,                             \
         owner=None,                              \
         prefs=None,                              \
    )

    .. _Reduce:

    Combines values in each of one or more arrays into a single value for each array, with optional weighting and/or
    exponentiation of each item within an array prior to combining, and scaling and/or offset of result after combining.

    `function <Reduce.function>` returns an array of scalar values, one for each array in `variable <Reduce.variable>`.

    COMMENT:
        IMPLEMENTATION NOTE: EXTEND TO MULTIDIMENSIONAL ARRAY ALONG ARBITRARY AXIS
    COMMENT

    Arguments
    ---------

    default_variable : list or np.array : default class_defaults.variable
        specifies a template for the value to be transformed and its default value;  all entries must be numeric.

    weights : 1d or 2d np.array : default None
        specifies values used to multiply the elements of each array in `variable  <LinearCombination.variable>`.
        If it is 1d, its length must equal the number of items in `variable <LinearCombination.variable>`;
        if it is 2d, the length of each item must be the same as those in `variable <LinearCombination.variable>`,
        and there must be the same number of items as there are in `variable <LinearCombination.variable>`
        (see `weights <LinearCombination.weights>` for details)

    exponents : 1d or 2d np.array : default None
        specifies values used to exponentiate the elements of each array in `variable  <LinearCombination.variable>`.
        If it is 1d, its length must equal the number of items in `variable <LinearCombination.variable>`;
        if it is 2d, the length of each item must be the same as those in `variable <LinearCombination.variable>`,
        and there must be the same number of items as there are in `variable <LinearCombination.variable>`
        (see `exponents <LinearCombination.exponents>` for details)

    operation : SUM or PRODUCT : default SUM
        specifies whether to sum or multiply the elements in `variable <Reduce.function.variable>` of
        `function <Reduce.function>`.

    scale : float
        specifies a value by which to multiply each element of the output of `function <Reduce.function>`
        (see `scale <Reduce.scale>` for details)

    offset : float
        specifies a value to add to each element of the output of `function <Reduce.function>`
        (see `offset <Reduce.offset>` for details)

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

    default_variable : list or np.array
        contains array(s) to be reduced.

    operation : SUM or PRODUCT
        determines whether elements of each array in `variable <Reduce.function.variable>` of
        `function <Reduce.function>` are summmed or multiplied.

    scale : float
        value is applied multiplicatively to each element of the array after applying the `operation <Reduce.operation>`
        (see `scale <Reduce.scale>` for details);  this done before applying the `offset <Reduce.offset>`
        (if it is specified).

    offset : float
        value is added to each element of the array after applying the `operation <Reduce.operation>`
        and `scale <Reduce.scale>` (if it is specified).

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
    componentName = REDUCE_FUNCTION


    class Parameters(TransformFunction.Parameters):
        """
            Attributes
            ----------

                exponents
                    see `exponents <Reduce.exponents>`

                    :default value: None
                    :type:

                changes_shape
                    see `changes_shape <Function_Base.changes_shape>`

                    :default value: True
                    :type: bool

                offset
                    see `offset <Reduce.offset>`

                    :default value: 0.0
                    :type: ``float``

                operation
                    see `operation <Reduce.operation>`

                    :default value: `SUM`
                    :type: ``str``

                scale
                    see `scale <Reduce.scale>`

                    :default value: 1.0
                    :type: ``float``

                weights
                    see `weights <Reduce.weights>`

                    :default value: None
                    :type:
        """
        weights = None
        exponents = None
        operation = SUM
        scale = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        offset = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        changes_shape = Parameter(True, stateful=False, loggable=False, pnl_internal=True)

        def _validate_scale(self, scale):
            if not is_numeric_scalar(scale):
                return "scale must be a scalar"

        def _validate_offset(self, offset):
            if not is_numeric_scalar(offset):
                return "vector offset is not supported"


    @check_user_specified
    @beartype
    def __init__(self,
                 # weights:  Optional[ValidParamSpecType] = None,
                 # exponents:  Optional[ValidParamSpecType] = None,
                 weights=None,
                 exponents=None,
                 default_variable=None,
                 operation: Optional[Literal['sum', 'product']] = None,
                 scale: Optional[ValidParamSpecType] = None,
                 offset: Optional[ValidParamSpecType] = None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):

        super().__init__(
            default_variable=default_variable,
            weights=weights,
            exponents=exponents,
            operation=operation,
            scale=scale,
            offset=offset,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _validate_variable(self, variable, context=None):
        """Insure that list or array is 1d and that all elements are numeric

        Args:
            variable:
            context:
        """
        variable = super()._validate_variable(variable=variable, context=context)
        if not is_numeric(variable):
            if self.owner:
                err_msg = f"{self.__class__.__name__} function of {repr(self.owner.name)} " \
                          f"passed variable ({variable}) with non-scalar element."
            else:
                err_msg = f"All elements of variable ({variable}) for {self.__class__.__name__} must be scalar values."
            raise FunctionError(err_msg)
        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate weghts, exponents, scale and offset parameters

        Check that WEIGHTS and EXPONENTS are lists or np.arrays of numbers with length equal to variable.
        Check that SCALE and OFFSET are scalars.

        Note: the checks of compatibility with variable are only performed for validation calls during execution
              (i.e., from check_args(), since during initialization or COMMAND_LINE assignment,
              a parameter may be re-assigned before variable assigned during is known
        """

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if WEIGHTS in target_set and target_set[WEIGHTS] is not None:
            self._validate_parameter_spec(target_set[WEIGHTS], WEIGHTS, numeric_only=True)
            target_set[WEIGHTS] = np.atleast_1d(target_set[WEIGHTS])
            if context.execution_phase & (ContextFlags.EXECUTING | ContextFlags.LEARNING):
                if len(target_set[WEIGHTS]) != len(self.defaults.variable):
                    raise FunctionError("Number of weights ({0}) is not equal to number of elements in variable ({1})".
                                        format(len(target_set[WEIGHTS]), len(self.defaults.variable)))

        if EXPONENTS in target_set and target_set[EXPONENTS] is not None:
            self._validate_parameter_spec(target_set[EXPONENTS], EXPONENTS, numeric_only=True)
            target_set[EXPONENTS] = np.atleast_1d(target_set[EXPONENTS])
            if context.execution_phase & (ContextFlags.EXECUTING | ContextFlags.LEARNING):
                if len(target_set[EXPONENTS]) != len(self.defaults.variable):
                    raise FunctionError("Number of exponents ({0}) does not equal number of elements in variable ({1})".
                                        format(len(target_set[EXPONENTS]), len(self.defaults.variable)))

        if SCALE in target_set and target_set[SCALE] is not None:
            scale = target_set[SCALE]
            if not is_numeric_scalar(scale):
                raise FunctionError("{} param of {} ({}) must be a scalar".format(SCALE, self.name, scale))

        if OFFSET in target_set and target_set[OFFSET] is not None:
            offset = target_set[OFFSET]
            if not is_numeric_scalar(offset):
                raise FunctionError("{} param of {} ({}) must be a scalar".format(OFFSET, self.name, offset))

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        variable : list or np.array : default class_defaults.variable
           a list or np.array of numeric values.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        Sum or product of arrays in variable : array
            in an array that is one dimension less than `variable <Reduce.variable>`.


        """
        weights = self._get_current_parameter_value(WEIGHTS, context)
        exponents = self._get_current_parameter_value(EXPONENTS, context)
        operation = self._get_current_parameter_value(OPERATION, context)
        scale = self._get_current_parameter_value(SCALE, context)
        offset = self._get_current_parameter_value(OFFSET, context)

        # FIX FOR EFFICIENCY: CHANGE THIS AND WEIGHTS TO TRY/EXCEPT // OR IS IT EVEN NECESSARY, GIVEN VALIDATION ABOVE??
        # Apply exponents if they were specified
        if exponents is not None:
            # Avoid divide by zero warning:
            #    make sure there are no zeros for an element that is assigned a negative exponent
            # Allow during initialization because 0s are common in default_variable argument
            if self.is_initializing:
                with np.errstate(divide='raise'):
                    try:
                        variable = variable ** exponents
                    except FloatingPointError:
                        variable = np.ones_like(variable)
            else:
                # if this fails with FloatingPointError it should not be caught outside of initialization
                variable = variable ** exponents

        # Apply weights if they were specified
        if weights is not None:
            variable = variable * weights

        # Calculate using relevant aggregation operation and return
        if operation == SUM:
            # result = np.sum(np.atleast_2d(variable), axis=0) * scale + offset
            result = np.sum(np.atleast_2d(variable), axis=1) * scale + offset
        elif operation == PRODUCT:
            result = np.prod(np.atleast_2d(variable), axis=1) * scale + offset
        else:
            raise FunctionError("Unrecognized operator ({0}) for Reduce function".
                                format(self._get_current_parameter_value(OPERATION, context)))

        return self.convert_output_type(result)

    def _get_input_struct_type(self, ctx):
        # FIXME: Workaround a special case of simple array.
        #        It should just pass through to modifiers, which matches what
        #        single element 2d array does
        default_var = np.atleast_2d(self.defaults.variable)
        return ctx.convert_python_struct_to_llvm_ir(default_var)

    def _gen_llvm_combine(self, builder, index, ctx, vi, vo, params):
        scale = self._gen_llvm_load_param(ctx, builder, params, SCALE, index, 1.0)
        offset = self._gen_llvm_load_param(ctx, builder, params, OFFSET, index, -0.0)

        # assume operation does not change dynamically
        operation = self.parameters.operation.get()
        if operation == SUM:
            val = ctx.float_ty(-0.0)
            comb_op = "fadd"
        elif operation == PRODUCT:
            val = ctx.float_ty(1.0)
            comb_op = "fmul"
        else:
            assert False, "Unknown operation: {}".format(operation)

        val_p = builder.alloca(val.type, name="reduced_result")
        builder.store(val, val_p)

        pow_f = ctx.get_builtin("pow", [ctx.float_ty])

        vi = builder.gep(vi, [ctx.int32_ty(0), index])
        with pnlvm.helpers.array_ptr_loop(builder, vi, "reduce") as (b, idx):
            ptri = b.gep(vi, [ctx.int32_ty(0), idx])
            in_val = b.load(ptri)

            exponent = self._gen_llvm_load_param(ctx, b, params, EXPONENTS,
                                                 index, 1.0)
            # Vector of vectors (even 1-element vectors)
            if isinstance(exponent.type, pnlvm.ir.ArrayType):
                assert len(exponent.type) == 1 # FIXME: Add support for matrix weights
                exponent = b.extract_value(exponent, [0])
            # FIXME: Remove this micro-optimization,
            #        it should be handled by the compiler
            if not isinstance(exponent, pnlvm.ir.Constant) or exponent.constant != 1.0:
                in_val = b.call(pow_f, [in_val, exponent])

            # Try per element weights first
            weight = self._gen_llvm_load_param(ctx, b, params, WEIGHTS,
                                               idx, 1.0)

            # Vector of vectors (even 1-element vectors)
            if isinstance(weight.type, pnlvm.ir.ArrayType):
                weight = self._gen_llvm_load_param(ctx, b, params, WEIGHTS,
                                                   index, 1.0)
                assert len(weight.type) == 1 # FIXME: Add support for matrix weights
                weight = b.extract_value(weight, [0])

            in_val = b.fmul(in_val, weight)

            val = b.load(val_p)
            val = getattr(b, comb_op)(val, in_val)
            b.store(val, val_p)

        val = b.load(val_p)
        val = builder.fmul(val, scale)
        val = builder.fadd(val, offset)

        ptro = builder.gep(vo, [ctx.int32_ty(0), index])
        builder.store(val, ptro)


class LinearCombination(
    TransformFunction):  # ------------------------------------------------------------------------
    """
    LinearCombination(     \
         default_variable, \
         weights=None,     \
         exponents=None,   \
         operation=SUM,    \
         scale=None,       \
         offset=None,      \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _LinearCombination:

    Linearly combine arrays of values, optionally weighting and/or exponentiating each array prior to combining,
    and scaling and/or offsetting after combining.

    `function <LinearCombination.function>` combines the arrays in the outermost dimension (axis 0) of `variable
    <LinearCombination.variable>` either additively or multiplicatively (as specified by `operation
    <LinearCombination.operation>`), applying `weights <LinearCombination.weights>` and/or `exponents
    <LinearCombination.exponents>` (if specified) to each array prior to combining them, and applying `scale
    <LinearCombination.scale>` and/or `offeset <LinearCombination.offset>` (if specified) to the result after
    combining, and returns an array of the same length as the operand arrays.

    COMMENT:
        Description:
            Combine corresponding elements of arrays in variable arg, using arithmetic operation determined by OPERATION
            Use optional SCALE and OFFSET parameters to linearly transform the resulting array
            Returns a list or 1D array of the same length as the individual ones in the variable

            Notes:
            * If variable contains only a single array, it is simply linearly transformed using SCALE and OFFSET
            * If there is more than one array in variable, they must all be of the same length
            * WEIGHTS and EXPONENTS can be:
                - 1D: each array in variable is scaled by the corresponding element of WEIGHTS or EXPONENTS
                - 2D: each array in variable is scaled by (Hadamard-wise) corresponding array of WEIGHTS or EXPONENTS
        Initialization arguments:
         - variable (value, np.ndarray or list): values to be combined;
             can be a list of lists, or a 1D or 2D np.array;  a 1D np.array is always returned
             if it is a list, it must be a list of numbers, lists, or np.arrays
             all items in the list or 2D np.array must be of equal length
             + WEIGHTS (list of numbers or 1D np.array): multiplies each item of variable before combining them
                  (default: [1,1])
             + EXPONENTS (list of numbers or 1D np.array): exponentiates each item of variable before combining them
                  (default: [1,1])
         - params (dict) can include:
             + WEIGHTS (list of numbers or 1D np.array): multiplies each variable before combining them (default: [1,1])
             + OFFSET (value): added to the result (after the arithmetic operation is applied; default is 0)
             + SCALE (value): multiples the result (after combining elements; default: 1)
             + OPERATION (Operation Enum) - method used to combine terms (default: SUM)
                  SUM: element-wise sum of the arrays in variable
                  PRODUCT: Hadamard Product of the arrays in variable

        LinearCombination.function returns combined values:
        - single number if variable was a single number
        - list of numbers if variable was list of numbers
        - 1D np.array if variable was a single np.variable or np.ndarray
    COMMENT

    Arguments
    ---------

    variable : 1d or 2d np.array : default class_defaults.variable
        specifies a template for the arrays to be combined.  If it is 2d, all items must have the same length.

    weights : scalar or 1d or 2d np.array : default None
        specifies values used to multiply the elements of each array in **variable**.
        If it is 1d, its length must equal the number of items in `variable <LinearCombination.variable>`;
        if it is 2d, the length of each item must be the same as those in `variable <LinearCombination.variable>`,
        and there must be the same number of items as there are in `variable <LinearCombination.variable>`
        (see `weights <LinearCombination.weights>` for details of how weights are applied).

    exponents : scalar or 1d or 2d np.array : default None
        specifies values used to exponentiate the elements of each array in `variable  <LinearCombination.variable>`.
        If it is 1d, its length must equal the number of items in `variable <LinearCombination.variable>`;
        if it is 2d, the length of each item must be the same as those in `variable <LinearCombination.variable>`,
        and there must be the same number of items as there are in `variable <LinearCombination.variable>`
        (see `exponents <LinearCombination.exponents>` for details of how exponents are applied).

    operation : SUM, PRODUCT or CROSS_ENTROPY : default SUM
        specifies whether the `function <LinearCombination.function>` takes the elementwise (Hadamarad)
        sum, product or cross entropy of the arrays in `variable  <LinearCombination.variable>`.

    scale : float or np.ndarray : default None
        specifies a value by which to multiply each element of the result of `function <LinearCombination.function>`
        (see `scale <LinearCombination.scale>` for details)

    offset : float or np.ndarray : default None
        specifies a value to add to each element of the result of `function <LinearCombination.function>`
        (see `offset <LinearCombination.offset>` for details)

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

    variable : 1d or 2d np.array
        contains the arrays to be combined by `function <LinearCombination>`.  If it is 1d, the array is simply
        linearly transformed by and `scale <LinearCombination.scale>` and `offset <LinearCombination.scale>`.
        If it is 2d, the arrays (all of which must be of equal length) are weighted and/or exponentiated as
        specified by `weights <LinearCombination.weights>` and/or `exponents <LinearCombination.exponents>`
        and then combined as specified by `operation <LinearCombination.operation>`.

    weights : scalar or 1d or 2d np.array
        if it is a scalar, the value is used to multiply all elements of all arrays in `variable
        <LinearCombination.variable>`; if it is a 1d array, each element is used to multiply all elements in the
        corresponding array of `variable <LinearCombination.variable>`;  if it is a 2d array, then each array is
        multiplied elementwise (i.e., the Hadamard Product is taken) with the corresponding array of `variable
        <LinearCombinations.variable>`. All `weights` are applied before any exponentiation (if it is specified).

    exponents : scalar or 1d or 2d np.array
        if it is a scalar, the value is used to exponentiate all elements of all arrays in `variable
        <LinearCombination.variable>`; if it is a 1d array, each element is used to exponentiate the elements of the
        corresponding array of `variable <LinearCombinations.variable>`;  if it is a 2d array, the element of each
        array is used to exponentiate the corresponding element of the corresponding array of `variable
        <LinearCombination.variable>`. In either case, all exponents are applied after application of the `weights
        <LinearCombination.weights>` (if any are specified).

    operation : SUM or PRODUCT
        determines whether the `function <LinearCombination.function>` takes the elementwise (Hadamard) sum,
        product, or cross entropy of the arrays in `variable  <LinearCombination.variable>`.

    scale : float or np.ndarray
        value is applied multiplicatively to each element of the array after applying the
        `operation <LinearCombination.operation>` (see `scale <LinearCombination.scale>` for details);
        this done before applying the `offset <LinearCombination.offset>` (if it is specified).

    offset : float or np.ndarray
        value is added to each element of the array after applying the `operation <LinearCombination.operation>`
        and `scale <LinearCombination.scale>` (if it is specified).

    COMMENT:
    function : function
        applies the `weights <LinearCombination.weights>` and/or `exponents <LinearCombinations.weights>` to the
        arrays in `variable <LinearCombination.variable>`, then takes their sum or product (as specified by
        `operation <LinearCombination.operation>`), and finally applies `scale <LinearCombination.scale>` and/or
        `offset <LinearCombination.offset>`.

    enable_output_type_conversion : Bool : False
        specifies whether `function output type conversion <Function_Output_Type_Conversion>` is enabled.

    output_type : FunctionOutputType : None
        used to specify the return type for the `function <Function_Base.function>`;  `functionOuputTypeConversion`
        must be enabled and implemented for the class (see `FunctionOutputType <Function_Output_Type_Conversion>`
        for details).
    COMMENT

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

    componentName = LINEAR_COMBINATION_FUNCTION

    classPreferences = {
        PREFERENCE_SET_NAME: 'LinearCombinationCustomClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    }

    class Parameters(TransformFunction.Parameters):
        """
            Attributes
            ----------

                exponents
                    see `exponents <LinearCombination.exponents>`

                    :default value: None
                    :type:

                offset
                    see `offset <LinearCombination.offset>`

                    :default value: 0.0
                    :type: ``float``

                operation
                    see `operation <LinearCombination.operation>`

                    :default value: `SUM`
                    :type: ``str``

                scale
                    see `scale <LinearCombination.scale>`

                    :default value: 1.0
                    :type: ``float``

                weights
                    see `weights <LinearCombination.weights>`

                    :default value: None
                    :type:
        """
        operation = SUM

        weights = Parameter(None, modulable=True)
        exponents = Parameter(None, modulable=True)
        scale = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        offset = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 # weights:  Optional[ValidParamSpecType] = None,
                 # exponents:  Optional[ValidParamSpecType] = None,
                 weights=None,
                 exponents=None,
                 operation: Optional[Literal['sum', 'product', 'cross-entropy']] = None,
                 scale=None,
                 offset=None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):

        super().__init__(
            default_variable=default_variable,
            weights=weights,
            exponents=exponents,
            operation=operation,
            scale=scale,
            offset=offset,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _validate_variable(self, variable, context=None):
        """Insure that all items of list or np.array in variable are of the same length

        Args:
            variable:
            context:
        """
        variable = super()._validate_variable(variable=variable, context=context)
        # FIX: CONVERT TO AT LEAST 1D NP ARRAY IN INIT AND EXECUTE, SO ALWAYS NP ARRAY
        # FIX: THEN TEST THAT SHAPES OF EVERY ELEMENT ALONG AXIS 0 ARE THE SAME
        # FIX; PUT THIS IN DOCUMENTATION
        if isinstance(variable, (list, np.ndarray)):
            if isinstance(variable, np.ndarray) and not variable.ndim:
                return variable
            length = 0
            for i in range(len(variable)):
                if i == 0:
                    continue
                if isinstance(variable[i - 1], numbers.Number):
                    old_length = 1
                else:
                    old_length = len(variable[i - 1])
                if variable[i] is None:
                    owner_str = f"'{self.owner.name}' " if self.owner else ''
                    raise FunctionError(f"One of the elements of variable for {self.__class__.__name__} function "
                                        f"of {owner_str}is None; variable: {variable}.")
                elif isinstance(variable[i], numbers.Number):
                    new_length = 1
                else:
                    new_length = len(variable[i])
                if old_length != new_length:
                    owner_str = f"'{self.owner.name }' " if self.owner else ''
                    raise FunctionError(f"Length of all arrays in variable for {self.__class__.__name__} function "
                                        f"of {owner_str}must be the same; variable: {variable}.")
        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate weghts, exponents, scale and offset parameters

        Check that WEIGHTS and EXPONENTS are lists or np.arrays of numbers with length equal to variable
        Check that SCALE and OFFSET are either scalars or np.arrays of numbers with length and shape equal to variable

        Note: the checks of compatibility with variable are only performed for validation calls during execution
              (i.e., from check_args(), since during initialization or COMMAND_LINE assignment,
              a parameter may be re-assigned before variable assigned during is known
        """

        # FIX: MAKE SURE THAT IF OPERATION IS SUBTRACT OR DIVIDE, THERE ARE ONLY TWO VECTORS

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if WEIGHTS in target_set and target_set[WEIGHTS] is not None:
            self._validate_parameter_spec(target_set[WEIGHTS], WEIGHTS, numeric_only=True)
            if context.execution_phase & (ContextFlags.EXECUTING | ContextFlags.LEARNING):
                if np.array(target_set[WEIGHTS]).shape != self.defaults.variable.shape:
                    raise FunctionError("Number of weights ({0}) is not equal to number of items in variable ({1})".
                                        format(len(target_set[WEIGHTS]), len(self.defaults.variable)))

        if EXPONENTS in target_set and target_set[EXPONENTS] is not None:
            self._validate_parameter_spec(target_set[EXPONENTS], EXPONENTS, numeric_only=True)
            if context.execution_phase & (ContextFlags.PROCESSING | ContextFlags.LEARNING):
                if np.array(target_set[EXPONENTS]).shape != self.defaults.variable.shape:
                    raise FunctionError("Number of exponents ({0}) does not equal number of items in variable ({1})".
                                        format(len(target_set[EXPONENTS]), len(self.defaults.variable)))

        if SCALE in target_set and target_set[SCALE] is not None:
            scale = target_set[SCALE]
            if isinstance(scale, numbers.Number):
                pass
            elif isinstance(scale, np.ndarray):
                target_set[SCALE] = np.array(scale)
            if context.execution_phase & (ContextFlags.PROCESSING | ContextFlags.LEARNING):
                if not is_numeric_scalar(scale):
                    err_msg = "Scale is using Hadamard modulation but its shape and/or size (scale shape: {}, size:{})" \
                              " do not match the variable being modulated (variable shape: {}, size: {})". \
                        format(scale.shape, scale.size, self.defaults.variable.shape,
                               self.defaults.variable.size)
                    if len(self.defaults.variable.shape) == 0:
                        raise FunctionError(err_msg)
                    if (scale.shape != self.defaults.variable.shape) and \
                            (scale.shape != self.defaults.variable.shape[1:]):
                        raise FunctionError(err_msg)

        if OFFSET in target_set and target_set[OFFSET] is not None:
            offset = target_set[OFFSET]
            if isinstance(offset, numbers.Number):
                pass
            elif isinstance(offset, np.ndarray):
                target_set[OFFSET] = np.array(offset)

            if context.execution_phase & (ContextFlags.PROCESSING | ContextFlags.LEARNING):
                if not is_numeric_scalar(offset):
                    err_msg = "Offset is using Hadamard modulation but its shape and/or size (offset shape: {}, size:{})" \
                              " do not match the variable being modulated (variable shape: {}, size: {})". \
                        format(offset.shape, offset.size, self.defaults.variable.shape,
                               self.defaults.variable.size)
                    if len(self.defaults.variable.shape) == 0:
                        raise FunctionError(err_msg)
                    if (offset.shape != self.defaults.variable.shape) and \
                            (offset.shape != self.defaults.variable.shape[1:]):
                        raise FunctionError(err_msg)

                        # if not operation:
                        #     raise FunctionError("Operation param missing")
                        # if not operation == self.Operation.SUM and not operation == self.Operation.PRODUCT:
                        #     raise FunctionError("Operation param ({0}) must be Operation.SUM or Operation.PRODUCT".
                        #     format(operation))

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        variable : 1d or 2d np.array : default class_defaults.variable
           a single numeric array, or multiple arrays to be combined; if it is 2d, all arrays must have the same length.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        combined array : 1d array
            the result of linearly combining the arrays in `variable <LinearCombination.variable>`.

        """
        weights = self._get_current_parameter_value(WEIGHTS, context)
        exponents = self._get_current_parameter_value(EXPONENTS, context)
        # if self.initialization_status == ContextFlags.INITIALIZED:
        #     if weights is not None and weights.shape != variable.shape:
        #         weights = weights.reshape(variable.shape)
        #     if exponents is not None and exponents.shape != variable.shape:
        #         exponents = exponents.reshape(variable.shape)
        operation = self._get_current_parameter_value(OPERATION, context)
        scale = self._get_current_parameter_value(SCALE, context)
        offset = self._get_current_parameter_value(OFFSET, context)

        # QUESTION:  WHICH IS LESS EFFICIENT:
        #                A) UNECESSARY ARITHMETIC OPERATIONS IF SCALE AND/OR OFFSET ARE 1.0 AND 0, RESPECTIVELY?
        #                   (DOES THE COMPILER KNOW NOT TO BOTHER WITH MULT BY 1 AND/OR ADD 0?)
        #                B) EVALUATION OF IF STATEMENTS TO DETERMINE THE ABOVE?
        # IMPLEMENTATION NOTE:  FOR NOW, ASSUME B) ABOVE, AND ASSIGN DEFAULT "NULL" VALUES TO offset AND scale
        if offset is None:
            offset = 0.0

        if scale is None:
            scale = 1.0

        # IMPLEMENTATION NOTE: CONFIRM: SHOULD NEVER OCCUR, AS _validate_variable NOW ENFORCES 2D np.ndarray
        # If variable is 0D or 1D:
        if np_array_less_than_2d(variable):
            return self.convert_output_type((variable * scale) + offset)

        # FIX FOR EFFICIENCY: CHANGE THIS AND WEIGHTS TO TRY/EXCEPT // OR IS IT EVEN NECESSARY, GIVEN VALIDATION ABOVE??
        # Apply exponents if they were specified
        if exponents is not None:
            # Avoid divide by zero warning:
            #    make sure there are no zeros for an element that is assigned a negative exponent
            # Allow during initialization because 0s are common in default_variable argument
            if self.is_initializing:
                with np.errstate(divide='raise'):
                    try:
                        variable = variable ** exponents
                    except FloatingPointError:
                        variable = np.ones_like(variable)
            else:
                # if this fails with FloatingPointError it should not be caught outside of initialization
                variable = variable ** exponents

        # Apply weights if they were specified
        if weights is not None:
            variable = variable * weights

        # CW 3/19/18: a total hack, e.g. to make scale=[4.] turn into scale=4. Used b/c the `scale` ParameterPort
        # changes scale's format: e.g. if you write c = pnl.LinearCombination(scale = 4), print(c.scale) returns [4.]
        # Don't use try_extract_0d_array_item because that will only
        # handle 0d arrays, not 1d.
        try:
            scale = scale.item()
        except (AttributeError, ValueError):
            pass
        try:
            offset = offset.item()
        except (AttributeError, ValueError):
            pass

        # CALCULATE RESULT USING RELEVANT COMBINATION OPERATION AND MODULATION
        if operation == SUM:
            combination = np.sum(variable, axis=0)
        elif operation == PRODUCT:
            combination = np.prod(variable, axis=0)
        elif operation == CROSS_ENTROPY:
            v1 = variable[0]
            v2 = variable[1]
            both_zero = np.logical_and(v1 == 0, v2 == 0)
            combination = v1 * np.where(both_zero, 0.0, np.log(v2, where=np.logical_not(both_zero)))
        else:
            raise FunctionError("Unrecognized operator ({0}) for LinearCombination function".
                                format(operation.self.Operation.SUM))
        if isinstance(scale, numbers.Number):
            # scalar scale
            product = combination * scale
        else:
            # Hadamard scale
            product = np.prod([combination, scale], axis=0)

        if isinstance(offset, numbers.Number):
            # scalar offset
            result = product + offset
        else:
            # Hadamard offset
            result = np.sum([product, offset], axis=0)

        return self.convert_output_type(result)

    @handle_external_context()
    def derivative(self, input=None, output=None, covariates=None, context=None):
        """
        derivative(input)

        Derivative of `function <LinearCombination._function>` at **input**.

        Arguments
        ---------

        output : 1d np.array : default class_defaults.variable[0]
            value of the input to the Linear transform at which derivative is to be taken.
           a single numeric array or multiple arrays being combined, and at which derivative is to be taken.

           .. technical_note::
              output arg is used for consistency with other derivatives used by BackPropagation, and is ignored.

        covariates : 2d np.array : default class_defaults.variable[1:]
            the input(s) to the LinearCombination function other than the one for which the derivative is being
            computed;  these are used to calculate the Jacobian of the LinearCombination function.

        Returns
        -------

        Scale :  number (if input is 1d) or array (if input is 2d)

        """
        if covariates is None or self.operation == SUM:
            jacobian = self._get_current_parameter_value(SCALE, context)
        else:
            jacobian = np.prod(np.vstack(covariates), axis=0)  * self._get_current_parameter_value(SCALE, context)

        return np.eye(len(output)) * jacobian

    def _get_input_struct_type(self, ctx):
        # FIXME: Workaround a special case of simple array.
        #        It should just pass through to modifiers, which matches what
        #        single element 2d array does
        default_var = np.atleast_2d(self.defaults.variable)
        return ctx.convert_python_struct_to_llvm_ir(default_var)

    def _gen_llvm_combine(self, builder, index, ctx, vi, vo, params):
        scale = self._gen_llvm_load_param(ctx, builder, params, SCALE, index, 1.0)
        offset = self._gen_llvm_load_param(ctx, builder, params, OFFSET, index, -0.0)

        # assume operation does not change dynamically
        operation = self.parameters.operation.get()
        if operation == SUM:
            val = ctx.float_ty(-0.0)
            comb_op = "fadd"
        elif operation == PRODUCT:
            val = ctx.float_ty(1.0)
            comb_op = "fmul"
        elif operation == CROSS_ENTROPY:
            raise FunctionError(f"LinearCombination Function does not (yet) support CROSS_ENTROPY operation.")
            # FIX: THIS NEEDS TO BE REPLACED TO GENERATE A VECTOR WITH HADAMARD CROSS-ENTROPY OF vi AND vo
            # ptr1 = builder.gep(vi, [index])
            # ptr2 = builder.gep(vo, [index])
            # val1 = builder.load(ptr1)
            # val2 = builder.load(ptr2)
            # log_f = ctx.get_builtin("log", [ctx.float_ty])
            # log = builder.call(log_f, [val2])
            # prod = builder.fmul(val1, log)
        else:
            assert False, "Unknown operation: {}".format(operation)

        val_p = builder.alloca(val.type, name="combined_result")
        builder.store(val, val_p)

        pow_f = ctx.get_builtin("pow", [ctx.float_ty])

        with pnlvm.helpers.array_ptr_loop(builder, vi, "combine") as (b, idx):
            ptri = b.gep(vi, [ctx.int32_ty(0), idx, index])
            in_val = b.load(ptri)

            exponent = self._gen_llvm_load_param(ctx, b, params, EXPONENTS,
                                                 idx, 1.0)
            # Vector of vectors (even 1-element vectors)
            if isinstance(exponent.type, pnlvm.ir.ArrayType):
                assert len(exponent.type) == 1 # FIXME: Add support for matrix weights
                exponent = b.extract_value(exponent, [0])
            # FIXME: Remove this micro-optimization,
            #        it should be handled by the compiler
            if not isinstance(exponent, pnlvm.ir.Constant) or exponent.constant != 1.0:
                in_val = b.call(pow_f, [in_val, exponent])

            weight = self._gen_llvm_load_param(ctx, b, params, WEIGHTS,
                                               idx, 1.0)
            # Vector of vectors (even 1-element vectors)
            if isinstance(weight.type, pnlvm.ir.ArrayType):
                assert len(weight.type) == 1 # FIXME: Add support for matrix weights
                weight = b.extract_value(weight, [0])

            in_val = b.fmul(in_val, weight)

            val = b.load(val_p)
            val = getattr(b, comb_op)(val, in_val)
            b.store(val, val_p)

        val = builder.load(val_p)
        val = builder.fmul(val, scale)
        val = builder.fadd(val, offset)

        ptro = builder.gep(vo, [ctx.int32_ty(0), index])
        builder.store(val, ptro)

    def _gen_pytorch_fct(self, device, context=None):
        weights = self._get_pytorch_fct_param_value('weights', device, context)
        if weights is not None:
            weights = torch.tensor(weights, device=device).double()
        if self.operation == SUM:
            if weights is not None:
                return lambda x: torch.sum(torch.stack(x) * weights, 0)
            else:
                return lambda x: torch.sum(torch.stack(x), 0)
        elif self.operation == PRODUCT:
            if weights is not None:
                return lambda x: torch.prod(torch.stack(x) * weights, 0)
            else:
                return lambda x: torch.prod(torch.stack(x), 0)
        else:
            from psyneulink.library.compositions.autodiffcomposition import AutodiffCompositionError
            raise AutodiffCompositionError(f"The 'operation' parameter of {function.componentName} is not supported "
                                           f"by AutodiffComposition; use 'SUM' or 'PRODUCT' if possible.")


# **********************************************************************************************************************
#                                                 MatrixTransform
# **********************************************************************************************************************

class MatrixTransform(TransformFunction):  # -------------------------------------------------------------------------------
    """
    MatrixTransform(            \
         default_variable,      \
         matrix=None,           \
         operation=DOT_PRODUCT, \
         normalize=False,       \
         params=None,           \
         owner=None,            \
         name=None,             \
         prefs=None             \
         )

    .. _MatrixTransform:

    Matrix transform of `variable <MatrixTransform.variable>`.

    `function <MatrixTransform._function>` returns a matrix transform of `variable <MatrixTransform.variable>`
     based on the **operation** argument.

    **operation** = *DOT_PRODUCT*:

        Returns the dot (inner) product of `variable <MatrixTransform.variable>` and `matrix <MatrixTransform.matrix>`:

        .. math::
            {variable} \\bullet |matrix|

        If **normalize** =True, the result is normalized by the product of the norms of the variable and matrix:

        .. math::
            \\frac{variable \\bullet matrix}{\\|variable\\| \\cdot \\|matrix\\|}

        .. note::
           For **normalize** =True, the result is the same as the cosine of the angle between pairs of vectors.

    **operation** = *L0*:

        Returns the absolute value of the difference between `variable <MatrixTransform.variable>` and `matrix
        <MatrixTransform.matrix>`:

        .. math::
            |variable - matrix|

        If **normalize** =True, the result is normalized by the norm of the sum of differences between the variable and
        matrix, which is then subtracted from 1:

        .. math::
            1 - \\frac{|variable - matrix|}{\\|variable - matrix\\|}

        .. note::
           For **normalize** =True, the result has the same effect as the normalized *DOT_PRODUCT* operation,
           with more similar pairs of vectors producing larger values (closer to 1).

        .. warning::
           For **normalize** =False, the result is smaller (closer to 0) for more similar pairs of vectors,
           which is **opposite** the effect of the *DOT_PRODUCT* and normalized *L0* operations.  If the desired
           result is that more similar pairs of vectors produce larger values, set **normalize** =True or
           use the *DOT_PRODUCT* operation.


    COMMENT:  [CONVERT TO FIGURE]
        ----------------------------------------------------------------------------------------------------------
        MATRIX FORMAT <shape: (3,5)>
                                         INDICES:
                                     Output elements:
                              0       1       2       3       4
                         0  [0,0]   [0,1]   [0,2]   [0,3]   [0,4]
        Input elements:  1  [1,0]   [1,1]   [1,2]   [1,3]   [1,4]
                         2  [2,0]   [2,1]   [2,2]   [2,3]   [2,4]

        matrix.shape => (input/rows, output/cols)

        ----------------------------------------------------------------------------------------------------------
        ARRAY FORMAT
                                                                            INDICES
                                          [ [      Input 0 (row0)       ], [       Input 1 (row1)      ]... ]
                                          [ [ out0,  out1,  out2,  out3 ], [ out0,  out1,  out2,  out3 ]... ]
        matrix[input/rows, output/cols]:  [ [ row0,  row0,  row0,  row0 ], [ row1,  row1,  row1,  row1 ]... ]
                                          [ [ col0,  col1,  col2,  col3 ], [ col0,  col1,  col2,  col3 ]... ]
                                          [ [[0,0], [0,1], [0,2], [0,3] ], [[1,0], [1,1], [1,2], [1,3] ]... ]

        ----------------------------------------------------------------------------------------------------------
    COMMENT


    Arguments
    ---------

    variable : list or 1d array : default class_defaults.variable
        specifies a template for the value to be transformed; length must equal the number of rows of `matrix
        <MatrixTransform.matrix>`.

    matrix : number, list, 1d or 2d np.ndarray, function, or matrix keyword : default IDENTITY_MATRIX
        specifies matrix used to transform `variable <MatrixTransform.variable>`
        (see `matrix <MatrixTransform.matrix>` for specification details).

        When MatrixTransform is the `function <Projection_Base.function>` of a projection:

            - the matrix specification must be compatible with the variables of the `sender <Projection_Base.sender>`
              and `receiver <Projection_Base.receiver>`

            - a matrix keyword specification generates a matrix based on the sender and receiver shapes

        When MatrixTransform is instantiated on its own, or as the function of a `Mechanism <Mechanism>` or `Port`:

            - the matrix specification must be compatible with the function's own `variable <MatrixTransform.variable>`

            - if matrix is not specified, a square identity matrix is generated based on the number of columns in
              `variable <MatrixTransform.variable>`

            - matrix keywords are not valid matrix specifications

    operation : DOT_PRODUCT or L0 : default DOT_PRODUCT
        specifies whether to take the dot product or difference of `variable <MatrixTransform.variable>`
        and `matrix <MatrixTransform.matrix>`.

    normalize : bool : default False
        specifies whether to normalize the result of `function <LinearCombination.function>` by dividing it by the
        norm of `variable <MatrixTransform.variable>` x the norm of `matrix <MatrixTransform.matrix>`;  this cannot
        be used if `variable <MatrixTransform.variable>` is a scalar (i.e., has only one element), and **operation**
        is set to *L0* (since it is not needed, and can produce a divide by zero error).

    bounds : None

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

    variable : 1d array
        contains value to be transformed.

    matrix : 2d array
        matrix used to transform `variable <MatrixTransform.variable>`.
        Can be specified as any of the following:
            * number - used as the filler value for all elements of the :keyword:`matrix` (call to np.fill);
            * list of arrays, 2d array - assigned as the value of :keyword:`matrix`;
            * matrix keyword - see `MatrixKeywords` for list of options.
        Rows correspond to elements of the input array (outer index), and
        columns correspond to elements of the output array (inner index).

    operation : DOT_PRODUCT or L0 : default DOT_PRODUCT
        determines whether dot product or difference of `variable <MatrixTransform.variable>` and `matrix
        <MatrixTransform.matrix>` is taken.  If the length of `variable <MatrixTransform.variable>` is greater
        than 1 and L0 is specified, the `variable <MatrixTransform.variable>` array is subtracted from each
        array of `matrix <MatrixTransform.matrix>` and the resulting array is summed, to produce the corresponding
        element of the array returned by the function.

    normalize : bool
        determines whether the result of `function <LinearCombination.function>` is normalized, by dividing it by the
        norm of `variable <MatrixTransform.variable>` x the norm of `matrix <MatrixTransform.matrix>`.

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a default is
        assigned by FunctionRegistry (see `Registry_Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see `PreferenceSet`
        for details).
    """

    componentName = MATRIX_TRANSFORM_FUNCTION

    DEFAULT_FILLER_VALUE = 0

    _model_spec_generic_type_name = 'onnx::MatMul'

    class Parameters(TransformFunction.Parameters):
        """
            Attributes
            ----------

                matrix
                    see `matrix <MatrixTransform.matrix>`

                    :default value: None
                    :type:

                operation
                    see `operation <MatrixTransform.operation>`

                    :default value: DOT_PRODUCT
                    :type: bool

                normalize
                    see `normalize <MatrixTransform.normalize>`

                    :default value: False
                    :type: bool
        """
        variable = Parameter(np.array([0]), read_only=True, pnl_internal=True, constructor_argument='default_variable', mdf_name='A')
        matrix = Parameter(None, modulable=True, mdf_name='B')
        operation = Parameter(DOT_PRODUCT, stateful=False)
        normalize = Parameter(False)
        bounds = None

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 matrix=None,
                 operation=None,
                 normalize=None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):

        # Note: this calls _validate_variable and _validate_params which are overridden below;
        #       the latter implements the matrix if required
        # super(MatrixTransform, self).__init__(default_variable=default_variable,
        super().__init__(
            default_variable=default_variable,
            matrix=matrix,
            operation=operation,
            normalize=normalize,
            params=params,
            owner=owner,
            prefs=prefs,
        )

        self.parameters.matrix.set(
            self.instantiate_matrix(self.parameters.matrix.get()),
            skip_log=True,
        )


    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate params and assign to targets

        This overrides the class method, to perform more detailed type checking (see explanation in class method).
        Note: this method (or the class version) is called only if the parameter_validation attribute is `True`

        :param request_set: (dict) - params to be validated
        :param target_set: (dict) - destination of validated params
        :param context: (str)
        :return none:
        """

        super()._validate_params(request_set, target_set, context)

        param_set = target_set
        # proxy for checking whether the owner is a projection
        if hasattr(self.owner, 'receiver'):
            sender = self.defaults.variable
            sender_len = np.size(np.atleast_2d(self.defaults.variable), 1)

            # Check for and validate receiver first, since it may be needed to validate and/or construct the matrix
            # First try to get receiver from specification in params
            if RECEIVER in param_set:
                self.receiver = param_set[RECEIVER]
                # Check that specification is a list of numbers or an array
                if ((isinstance(self.receiver, list) and all(
                        isinstance(elem, numbers.Number) for elem in self.receiver)) or
                        isinstance(self.receiver, np.ndarray)):
                    self.receiver = np.atleast_1d(self.receiver)
                else:
                    raise FunctionError("receiver param ({0}) for {1} must be a list of numbers or an np.array".
                                        format(self.receiver, self.name))
            # No receiver, so use sender as template (assuming square -- e.g., IDENTITY -- matrix)
            else:
                if (self.owner and self.owner.prefs.verbosePref) or self.prefs.verbosePref:
                    print("Identity matrix requested but 'receiver' not specified; sender length ({0}) will be used".
                          format(sender_len))
                self.receiver = param_set[RECEIVER] = sender

            receiver_len = len(self.receiver)

            # Check rest of params
            message = ""
            for param_name, param_value in param_set.items():

                # receiver param already checked above
                if param_name == RECEIVER:
                    continue

                # Not currently used here
                if param_name in function_keywords:
                    continue

                if param_name == HAS_INITIALIZERS:
                    continue

                # matrix specification param
                elif param_name == MATRIX:

                    # A number (to be used as a filler), so OK
                    if isinstance(param_value, numbers.Number):
                        continue

                    # np.matrix or np.ndarray provided, so validate that it is numeric and check dimensions
                    elif isinstance(param_value, (list, np.ndarray, np.matrix)):
                        # get dimensions specified by:
                        #   variable (sender): width/cols/outer index
                        #   kwReceiver param: height/rows/inner index

                        weight_matrix = np.atleast_2d(param_value)
                        if 'U' in repr(weight_matrix.dtype):
                            raise FunctionError("Non-numeric entry in MATRIX "
                                                "specification ({}) for the {} "
                                                "function of {}".format(param_value,
                                                                        self.name,
                                                                        self.owner_name))

                        if weight_matrix.ndim != 2:
                            raise FunctionError("The matrix provided for the {} function of {} must be 2d (it is {}d".
                                                format(weight_matrix.ndim, self.name, self.owner_name))

                        matrix_rows = weight_matrix.shape[0]
                        matrix_cols = weight_matrix.shape[1]

                        # Check that number of rows equals length of sender vector (variable)
                        if matrix_rows != sender_len:
                            raise FunctionError("The number of rows ({}) of the "
                                                "matrix provided for {} function "
                                                "of {} does not equal the length "
                                                "({}) of the sender vector "
                                                "(variable)".format(matrix_rows,
                                                                    self.name,
                                                                    self.owner_name,
                                                                    sender_len))

                    # Auto, full or random connectivity matrix requested (using keyword):
                    # Note:  assume that these will be properly processed by caller
                    #        (e.g., MappingProjection._instantiate_receiver)
                    elif is_matrix_keyword(param_value):
                        continue

                    # Identity matrix requested (using keyword), so check send_len == receiver_len
                    elif param_value in {IDENTITY_MATRIX, HOLLOW_MATRIX}:
                        # Receiver length doesn't equal sender length
                        if not (self.receiver.shape == sender.shape and self.receiver.size == sender.size):
                            # if self.owner.prefs.verbosePref:
                            #     print ("Identity matrix requested, but length of receiver ({0})"
                            #            " does not match length of sender ({1});  sender length will be used".
                            #            format(receiver_len, sender_len))
                            # # Set receiver to sender
                            # param_set[kwReceiver] = sender
                            raise FunctionError("{} requested for the {} function of {}, "
                                                "but length of receiver ({}) does not match length of sender ({})".
                                                format(param_value, self.name, self.owner_name, receiver_len,
                                                       sender_len))
                        continue

                    # list used to describe matrix, so convert to 2D array and pass to validation of matrix below
                    elif isinstance(param_value, list):
                        try:
                            param_value = np.atleast_2d(param_value)
                        except (ValueError, TypeError) as error_msg:
                            raise FunctionError(
                                "Error in list specification ({}) of matrix for the {} function of {}: {})".
                                    # format(param_value, self.__class__.__name__, error_msg))
                                    format(param_value, self.name, self.owner_name, error_msg))

                    # string used to describe matrix, so convert to np.array and pass to validation of matrix below
                    elif isinstance(param_value, str):
                        try:
                            param_value = np.atleast_2d(param_value)
                        except (ValueError, TypeError) as error_msg:
                            raise FunctionError("Error in string specification ({}) of the matrix "
                                                "for the {} function of {}: {})".
                                                # format(param_value, self.__class__.__name__, error_msg))
                                                format(param_value, self.name, self.owner_name, error_msg))

                    # function so:
                    # - assume it uses random.rand()
                    # - call with two args as place markers for cols and rows
                    # -  validate that it returns an array
                    elif isinstance(param_value, types.FunctionType):
                        test = param_value(1, 1)
                        if not isinstance(test, np.ndarray):
                            raise FunctionError("A function is specified for the matrix of the {} function of {}: {}) "
                                                "that returns a value ({}) that is not an array".
                                                # format(param_value, self.__class__.__name__, test))
                                                format(self.name, self.owner_name, param_value, test))

                    elif param_value is None:
                        raise FunctionError("TEMP ERROR: param value is None.")

                    else:
                        raise FunctionError("Value of {} param ({}) for the {} function of {} "
                                            "must be a matrix, a number (for filler), or a matrix keyword ({})".
                                            format(param_name,
                                                   param_value,
                                                   self.name,
                                                   self.owner_name,
                                                   MATRIX_KEYWORD_NAMES))
                else:
                    continue

        # owner is a mechanism, state
        # OR function was defined on its own (no owner)
        else:
            if MATRIX in param_set:
                param_value = param_set[MATRIX]

                # numeric value specified; verify that it is compatible with variable
                if isinstance(param_value, (float, list, np.ndarray, np.matrix)):
                    param_size = np.size(np.atleast_2d(param_value), 0)
                    param_shape = np.shape(np.atleast_2d(param_value))
                    variable_size = np.size(np.atleast_2d(self.defaults.variable),1)
                    variable_shape = np.shape(np.atleast_2d(self.defaults.variable))
                    if param_size != variable_size:
                        raise FunctionError("Specification of matrix and/or default_variable for {} is not valid. The "
                                            "shapes of variable {} and matrix {} are not compatible for multiplication".
                                            format(self.name, variable_shape, param_shape))

                # keyword matrix specified - not valid outside of a projection
                elif is_matrix_keyword(param_value):
                    raise FunctionError("{} is not a valid specification for the matrix parameter of {}. Keywords "
                                        "may only be used to specify the matrix parameter of a Projection's "
                                        "MatrixTransform function. When the MatrixTransform function is implemented in a "
                                        "mechanism, such as {}, the correct matrix cannot be determined from a "
                                        "keyword. Instead, the matrix must be fully specified as a float, list, "
                                        "np.ndarray".
                                        format(param_value, self.name, self.owner.name))

                # The only remaining valid option is matrix = None (sorted out in instantiate_attribs_before_fn)
                elif param_value is not None:
                    raise FunctionError("Value of the matrix param ({}) for the {} function of {} "
                                        "must be a matrix, a number (for filler), or a matrix keyword ({})".
                                        format(param_value,
                                               self.name,
                                               self.owner_name,
                                               MATRIX_KEYWORD_NAMES))

    def _instantiate_attributes_before_function(self, function=None, context=None):
        # replicates setting of receiver in _validate_params
        if isinstance(self.owner, Projection):
            self.receiver = copy_parameter_value(self.defaults.variable)

        matrix = self.parameters.matrix._get(context)

        if matrix is None and not hasattr(self.owner, "receiver"):
            variable_length = np.size(np.atleast_2d(self.defaults.variable), 1)
            matrix = np.identity(variable_length)
        self.parameters.matrix._set(self.instantiate_matrix(matrix), context)

    def instantiate_matrix(self, specification, context=None):
        """Implements matrix indicated by specification

         Specification is derived from MATRIX param (passed to self.__init__ or self._function)

         Specification (validated in _validate_params):
            + single number (used to fill self.matrix)
            + matrix keyword (see get_matrix)
            + 2D list or np.ndarray of numbers

        :return matrix: (2D list)
        """
        from psyneulink.core.components.projections.projection import Projection
        if isinstance(self.owner, Projection):
            # Matrix provided (and validated in _validate_params); convert to array
            if isinstance(specification, np.matrix):
                return np.array(specification)

            sender = copy_parameter_value(self.defaults.variable)
            sender_len = sender.shape[0]
            try:
                receiver = self.receiver
            except:
                raise FunctionError("Can't instantiate matrix specification ({}) for the {} function of {} "
                                    "since its receiver has not been specified".
                                    format(specification, self.name, self.owner_name))
                # receiver = sender
            receiver_len = receiver.shape[0]

            matrix = get_matrix(specification, rows=sender_len, cols=receiver_len, context=context)

            # This should never happen (should have been picked up in validate_param or above)
            if matrix is None:
                raise FunctionError("MATRIX param ({}) for the {} function of {} must be a matrix, a function "
                                    "that returns one, a matrix specification keyword ({}), or a number (filler)".
                                    format(specification, self.name, self.owner_name, MATRIX_KEYWORD_NAMES))
            else:
                return matrix
        else:
            return np.array(specification)


    def _gen_llvm_function_body(self, ctx, builder, params, state, arg_in, arg_out, *, tags:frozenset):
        # Restrict to 1d arrays
        if self.defaults.variable.ndim != 1:
            warnings.warn("Shape mismatch: {} (in {}) got 2D input: {}".format(
                          self, self.owner, self.defaults.variable),
                          pnlvm.PNLCompilerWarning)
            arg_in = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(0)])
        if self.defaults.value.ndim != 1:
            warnings.warn("Shape mismatch: {} (in {}) has 2D output: {}".format(
                          self, self.owner, self.defaults.value),
                          pnlvm.PNLCompilerWarning)
            arg_out = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(0)])

        matrix = ctx.get_param_or_state_ptr(builder, self, MATRIX, param_struct_ptr=params, state_struct_ptr=state)
        normalize = ctx.get_param_or_state_ptr(builder, self, NORMALIZE, param_struct_ptr=params)

        # Convert array pointer to pointer to the fist element
        matrix = builder.gep(matrix, [ctx.int32_ty(0), ctx.int32_ty(0)])
        vec_in = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(0)])
        vec_out = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(0)])

        input_length = ctx.int32_ty(arg_in.type.pointee.count)
        output_length = ctx.int32_ty(arg_out.type.pointee.count)

        # if normalize:
        #     if vec_in is not zeros:
        #     # FIX: NORMALIZE vec_in and matrix here
        #         vec_in_sum = fsum(builder, vec_in)
        #         vec_in = fdiv(builder, vec_in, vec_in_sum)
        #     if matrix is not zeros:
        #     # FIX: NORMALIZE matrix here

        builtin = ctx.import_llvm_function("__pnl_builtin_vxm")
        builder.call(builtin, [vec_in, matrix, input_length, output_length, vec_out])
        return builder

    def _gen_pytorch_fct(self, device, context=None):
        operation = self._get_pytorch_fct_param_value('operation', device, context)
        normalize = self._get_pytorch_fct_param_value('normalize', device, context)

        def dot_product_with_normalization(vector, matrix):
            if torch.any(vector):
                vector = vector / torch.norm(vector)
            if torch.any(matrix):
                matrix = matrix / torch.norm(matrix)
            return torch.matmul(vector, matrix)

        def diff_with_normalization(vector, matrix):
            normalize = torch.sum(torch.abs(vector - matrix))
            return torch.sum((1 - torch.abs(vector - matrix) / normalize), axis=0)

        if operation is DOT_PRODUCT:
            if normalize:
                return dot_product_with_normalization
            else:
                return lambda x, y : torch.matmul(x, y)

        elif operation is L0:
            if normalize:
                return diff_with_normalization
            else:
                return lambda x, y: torch.sum(torch.abs(x - y),axis=0)

        else:
            from psyneulink.library.compositions.autodiffcomposition import AutodiffCompositionError
            raise AutodiffCompositionError(f"The 'operation' parameter of {function.componentName} is not supported "
                                           f"by AutodiffComposition; use 'DOT_PRODUCT' or 'L0'.")

    def _function(self,
                 variable=None,
                 context=None,
                 params=None):
        """

        Arguments
        ---------
        variable : list or 1d array
            array to be transformed;  length must equal the number of rows of `matrix <MatrixTransform.matrix>`.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        ---------

        dot product of or difference between variable and matrix : 1d array
            length of the array returned equals the number of columns of `matrix <MatrixTransform.matrix>`.

        """
        vector = np.array(variable)
        matrix = self._get_current_parameter_value(MATRIX, context)
        operation = self._get_current_parameter_value(OPERATION, context)
        normalize = self._get_current_parameter_value(NORMALIZE, context)

        if operation == DOT_PRODUCT:
            if normalize:
                if np.any(vector):
                    vector = vector / np.linalg.norm(vector)
                if np.any(matrix):
                    # FIX: the axis along which norming is carried out should probably be a parameter
                    #      Also need to deal with column- (or row-) wise zeros which cause div by zero
                    #      Replace columns (if norming axis 0) or rows (if norming axis 1) of zeros with 1's
                    # matrix = matrix / np.linalg.norm(matrix,axis=-1,keepdims=True)
                    matrix = matrix / np.linalg.norm(matrix,axis=0,keepdims=True)
            result = np.dot(vector, matrix)

        elif operation == L0:
            if normalize:
                normalization = np.sum(np.abs(vector - matrix)) or 1
                result = np.sum((1 - (np.abs(vector - matrix)) / normalization),axis=0)
            else:
                result = np.sum((np.abs(vector - matrix)),axis=0)

        return self.convert_output_type(result)

    @staticmethod
    def keyword(obj, keyword):

        from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
        rows = None
        cols = None
        # use of variable attribute here should be ok because it's using it as a format/type
        if isinstance(obj, MappingProjection):
            if isinstance(obj.sender.defaults.value, numbers.Number):
                rows = 1
            else:
                rows = len(obj.sender.defaults.value)
            if isinstance(obj.receiver.defaults.variable, numbers.Number):
                cols = 1
            else:
                cols = obj.receiver.socket_width
        matrix = get_matrix(keyword, rows, cols)

        if matrix is None:
            raise FunctionError("Unrecognized keyword ({}) specified for the {} function of {}".
                                format(keyword, obj.name, obj.owner_name))
        else:
            return matrix

    def param_function(owner, function):
        sender_len = len(owner.sender.defaults.value)
        receiver_len = len(owner.receiver.defaults.variable)
        return function(sender_len, receiver_len)

    def _is_identity(self, context=None, defaults=False):
        if defaults:
            matrix = self.defaults.matrix
        else:
            matrix = self.parameters.matrix._get(context)

        # if matrix is not an np array with at least one dimension,
        # this isn't an identity matrix
        try:
            size = matrix.shape[0]
        except (AttributeError, IndexError):
            return False

        # check if the matrix is the same as the identity matrix
        # note that we can use the first dimension size to create the identity matrix
        # because if the matrix is not square, this comparison will fail anyway
        identity_matrix = np.identity(size)
        # numpy has deprecated == comparisons of arrays
        try:
            return np.array_equal(matrix, identity_matrix)
        except TypeError:
            return matrix == identity_matrix

# def is_matrix_spec(m):
#     if m is None:
#         return True
#     if isinstance(m, (list, np.ndarray, types.FunctionType)):
#         return True
#     if m in MATRIX_KEYWORD_VALUES:
#         return True
#     return False



class CombineMeans(TransformFunction):  # ------------------------------------------------------------------------
    # FIX: CONFIRM THAT 1D KWEIGHTS USES EACH ELEMENT TO SCALE CORRESPONDING VECTOR IN VARIABLE
    # FIX  CONFIRM THAT LINEAR TRANSFORMATION (OFFSET, SCALE) APPLY TO THE RESULTING ARRAY
    # FIX: CONFIRM RETURNS LIST IF GIVEN LIST, AND SIMLARLY FOR NP.ARRAY
    """
    CombineMeans(            \
         default_variable, \
         weights=None,     \
         exponents=None,   \
         operation=SUM,    \
         scale=None,       \
         offset=None,      \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _CombineMeans:

    Calculate and combine mean(s) for arrays of values, optionally weighting and/or exponentiating each mean prior to
    combining, and scaling and/or offsetting after combining.

    `function <CombineMeans.function>` takes the mean of each array in the outermost dimension (axis 0) of `variable
    <CombineMeans.variable>`, and combines them either additively or multiplicatively (as specified by `operation
    <CombineMeans.operation>`), applying `weights <LinearCombination.weights>` and/or `exponents
    <LinearCombination.exponents>` (if specified) to each mean prior to combining them, and applying `scale
    <LinearCombination.scale>` and/or `offeset <LinearCombination.offset>` (if specified) to the result after combining,
    and returns a scalar value.

    COMMENT:
        Description:
            Take means of elements of each array in variable arg,
                and combine using arithmetic operation determined by OPERATION
            Use optional SCALE and OFFSET parameters to linearly transform the resulting array
            Returns a scalar

            Notes:
            * WEIGHTS and EXPONENTS can be:
                - 1D: each array in variable is scaled by the corresponding element of WEIGHTS or EXPONENTS
                - 2D: each array in variable is scaled by (Hadamard-wise) corresponding array of WEIGHTS or EXPONENTS
        Initialization arguments:
         - variable (value, np.ndarray or list): values to be combined;
             can be a list of lists, or a 1D or 2D np.array;  a scalar is always returned
             if it is a list, it must be a list of numbers, lists, or np.arrays
             if WEIGHTS or EXPONENTS are specified, their length along the outermost dimension (axis 0)
                 must equal the number of items in the variable
         - params (dict) can include:
             + WEIGHTS (list of numbers or 1D np.array): multiplies each item of variable before combining them
                  (default: [1,1])
             + EXPONENTS (list of numbers or 1D np.array): exponentiates each item of variable before combining them
                  (default: [1,1])
             + OFFSET (value): added to the result (after the arithmetic operation is applied; default is 0)
             + SCALE (value): multiples the result (after combining elements; default: 1)
             + OPERATION (Operation Enum) - method used to combine the means of the arrays in variable (default: SUM)
                  SUM: sum of the means of the arrays in variable
                  PRODUCT: product of the means of the arrays in variable

        CombineMeans.function returns a scalar value
    COMMENT

    Arguments
    ---------

    variable : 1d or 2d np.array : default class_defaults.variable
        specifies a template for the arrays to be combined.  If it is 2d, all items must have the same length.

    weights : 1d or 2d np.array : default None
        specifies values used to multiply the elements of each array in `variable  <CombineMeans.variable>`.
        If it is 1d, its length must equal the number of items in `variable <CombineMeans.variable>`;
        if it is 2d, the length of each item must be the same as those in `variable <CombineMeans.variable>`,
        and there must be the same number of items as there are in `variable <CombineMeans.variable>`
        (see `weights <CombineMeans.weights>` for details)

    exponents : 1d or 2d np.array : default None
        specifies values used to exponentiate the elements of each array in `variable  <CombineMeans.variable>`.
        If it is 1d, its length must equal the number of items in `variable <CombineMeans.variable>`;
        if it is 2d, the length of each item must be the same as those in `variable <CombineMeans.variable>`,
        and there must be the same number of items as there are in `variable <CombineMeans.variable>`
        (see `exponents <CombineMeans.exponents>` for details)

    operation : SUM or PRODUCT : default SUM
        specifies whether the `function <CombineMeans.function>` takes the sum or product of the means of the arrays in
        `variable  <CombineMeans.variable>`.

    scale : float or np.ndarray : default None
        specifies a value by which to multiply the result of `function <CombineMeans.function>`
        (see `scale <CombineMeans.scale>` for details)

    offset : float or np.ndarray : default None
        specifies a value to add to the result of `function <CombineMeans.function>`
        (see `offset <CombineMeans.offset>` for details)

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

    variable : 1d or 2d np.array
        contains the arrays to be combined by `function <CombineMeans>`.  If it is 1d, the array is simply
        linearly transformed by and `scale <CombineMeans.scale>` and `offset <CombineMeans.scale>`.
        If it is 2d, the arrays (all of which must be of equal length) are weighted and/or exponentiated as
        specified by `weights <CombineMeans.weights>` and/or `exponents <CombineMeans.exponents>`
        and then combined as specified by `operation <CombineMeans.operation>`.

    weights : 1d or 2d np.array : default NOne
        if it is 1d, each element is used to multiply all elements in the corresponding array of
        `variable <CombineMeans.variable>`;    if it is 2d, then each array is multiplied elementwise
        (i.e., the Hadamard Product is taken) with the corresponding array of `variable <CombineMeanss.variable>`.
        All :keyword:`weights` are applied before any exponentiation (if it is specified).

    exponents : 1d or 2d np.array : default None
        if it is 1d, each element is used to exponentiate the elements of the corresponding array of
        `variable <CombineMeans.variable>`;  if it is 2d, the element of each array is used to exponentiate
        the corresponding element of the corresponding array of `variable <CombineMeans.variable>`.
        In either case, exponentiating is applied after application of the `weights <CombineMeans.weights>`
        (if any are specified).

    operation : SUM or PRODUCT : default SUM
        determines whether the `function <CombineMeans.function>` takes the elementwise (Hadamard) sum or
        product of the arrays in `variable  <CombineMeans.variable>`.

    scale : float or np.ndarray : default None
        value is applied multiplicatively to each element of the array after applying the
        `operation <CombineMeans.operation>` (see `scale <CombineMeans.scale>` for details);
        this done before applying the `offset <CombineMeans.offset>` (if it is specified).

    offset : float or np.ndarray : default None
        value is added to each element of the array after applying the `operation <CombineMeans.operation>`
        and `scale <CombineMeans.scale>` (if it is specified).

    COMMENT:
    function : function
        applies the `weights <CombineMeans.weights>` and/or `exponents <CombineMeanss.weights>` to the
        arrays in `variable <CombineMeans.variable>`, then takes their sum or product (as specified by
        `operation <CombineMeans.operation>`), and finally applies `scale <CombineMeans.scale>` and/or
        `offset <CombineMeans.offset>`.

    enable_output_type_conversion : Bool : False
        specifies whether `function output type conversion <Function_Output_Type_Conversion>` is enabled.

    output_type : FunctionOutputType : None
        used to specify the return type for the `function <Function_Base.function>`;  `functionOuputTypeConversion`
        must be enabled and implemented for the class (see `FunctionOutputType <Function_Output_Type_Conversion>`
        for details).
    COMMENT

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

    componentName = COMBINE_MEANS_FUNCTION

    classPreferences = {
        PREFERENCE_SET_NAME: 'CombineMeansCustomClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    }

    class Parameters(TransformFunction.Parameters):
        """
            Attributes
            ----------

                exponents
                    see `exponents <CombineMeans.exponents>`

                    :default value: None
                    :type:

                offset
                    see `offset <CombineMeans.offset>`

                    :default value: 0.0
                    :type: ``float``

                operation
                    see `operation <CombineMeans.operation>`

                    :default value: `SUM`
                    :type: ``str``

                scale
                    see `scale <CombineMeans.scale>`

                    :default value: 1.0
                    :type: ``float``

                weights
                    see `weights <CombineMeans.weights>`

                    :default value: None
                    :type:
        """
        weights = None
        exponents = None
        operation = SUM
        scale = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        offset = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 # weights: Optional[ValidParamSpecType] = None,
                 # exponents: Optional[ValidParamSpecType] = None,
                 weights=None,
                 exponents=None,
                 operation: Optional[Literal['sum', 'product']] = None,
                 scale=None,
                 offset=None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):

        super().__init__(
            default_variable=default_variable,
            weights=weights,
            exponents=exponents,
            operation=operation,
            scale=scale,
            offset=offset,
            params=params,
            owner=owner,
            prefs=prefs,
        )

        if self.weights is not None:
            self.weights = np.atleast_2d(self.weights).reshape(-1, 1)
        if self.exponents is not None:
            self.exponents = np.atleast_2d(self.exponents).reshape(-1, 1)

    def _validate_variable(self, variable, context=None):
        """Insure that all items of variable are numeric
        """
        variable = super()._validate_variable(variable=variable, context=context)
        # if any(not is_numeric(item) for item in variable):
        #     raise FunctionError("All items of the variable for {} must be numeric".format(self.componentName))
        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate weights, exponents, scale and offset parameters

        Check that WEIGHTS and EXPONENTS are lists or np.arrays of numbers with length equal to variable
        Check that SCALE and OFFSET are either scalars or np.arrays of numbers with length and shape equal to variable

        Note: the checks of compatibility with variable are only performed for validation calls during execution
              (i.e., from check_args(), since during initialization or COMMAND_LINE assignment,
              a parameter may be re-assigned before variable assigned during is known
        """

        # FIX: MAKE SURE THAT IF OPERATION IS SUBTRACT OR DIVIDE, THERE ARE ONLY TWO VECTORS

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if WEIGHTS in target_set and target_set[WEIGHTS] is not None:
            target_set[WEIGHTS] = np.atleast_2d(target_set[WEIGHTS]).reshape(-1, 1)
            if context.execution_phase & (ContextFlags.PROCESSING | ContextFlags.LEARNING):
                if len(target_set[WEIGHTS]) != len(self.defaults.variable):
                    raise FunctionError("Number of weights ({0}) is not equal to number of items in variable ({1})".
                                        format(len(target_set[WEIGHTS]), len(self.defaults.variable.shape)))

        if EXPONENTS in target_set and target_set[EXPONENTS] is not None:
            target_set[EXPONENTS] = np.atleast_2d(target_set[EXPONENTS]).reshape(-1, 1)
            if context.execution_phase & (ContextFlags.PROCESSING | ContextFlags.LEARNING):
                if len(target_set[EXPONENTS]) != len(self.defaults.variable):
                    raise FunctionError("Number of exponents ({0}) does not equal number of items in variable ({1})".
                                        format(len(target_set[EXPONENTS]), len(self.defaults.variable.shape)))

        if SCALE in target_set and target_set[SCALE] is not None:
            scale = target_set[SCALE]
            if isinstance(scale, numbers.Number):
                pass
            elif isinstance(scale, np.ndarray):
                target_set[SCALE] = np.array(scale)
            else:
                raise FunctionError("{} param of {} ({}) must be a scalar or an np.ndarray".
                                    format(SCALE, self.name, scale))
            if context.execution_phase & (ContextFlags.PROCESSING | ContextFlags.LEARNING):
                if (isinstance(scale, np.ndarray) and
                        (scale.size != self.defaults.variable.size or
                                 scale.shape != self.defaults.variable.shape)):
                    raise FunctionError("Scale is using Hadamard modulation "
                                        "but its shape and/or size (shape: {}, size:{}) "
                                        "do not match the variable being modulated (shape: {}, size: {})".
                                        format(scale.shape, scale.size, self.defaults.variable.shape,
                                               self.defaults.variable.size))

        if OFFSET in target_set and target_set[OFFSET] is not None:
            offset = target_set[OFFSET]
            if isinstance(offset, numbers.Number):
                pass
            elif isinstance(offset, np.ndarray):
                target_set[OFFSET] = np.array(offset)
            else:
                raise FunctionError("{} param of {} ({}) must be a scalar or an np.ndarray".
                                    format(OFFSET, self.name, offset))
            if context.execution_phase & (ContextFlags.PROCESSING | ContextFlags.LEARNING):
                if (isinstance(offset, np.ndarray) and
                        (offset.size != self.defaults.variable.size or
                                 offset.shape != self.defaults.variable.shape)):
                    raise FunctionError("Offset is using Hadamard modulation "
                                        "but its shape and/or size (shape: {}, size:{}) "
                                        "do not match the variable being modulated (shape: {}, size: {})".
                                        format(offset.shape, offset.size, self.defaults.variable.shape,
                                               self.defaults.variable.size))

                    # if not operation:
                    #     raise FunctionError("Operation param missing")
                    # if not operation == self.Operation.SUM and not operation == self.Operation.PRODUCT:
                    #     raise FunctionError("Operation param ({0}) must be Operation.SUM or Operation.PRODUCT".
                    #     format(operation))

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        variable : 1d or 2d np.array : default class_defaults.variable
           a single numeric array, or multiple arrays to be combined; if it is 2d, all arrays must have the same length.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        combined means : number
            the result of taking the means of each array in `variable <CombineMeans.variable>` and combining them.

        """
        exponents = self._get_current_parameter_value(EXPONENTS, context)
        weights = self._get_current_parameter_value(WEIGHTS, context)
        operation = self._get_current_parameter_value(OPERATION, context)
        offset = self._get_current_parameter_value(OFFSET, context)
        scale = self._get_current_parameter_value(SCALE, context)

        # QUESTION:  WHICH IS LESS EFFICIENT:
        #                A) UNECESSARY ARITHMETIC OPERATIONS IF SCALE AND/OR OFFSET ARE 1.0 AND 0, RESPECTIVELY?
        #                   (DOES THE COMPILER KNOW NOT TO BOTHER WITH MULT BY 1 AND/OR ADD 0?)
        #                B) EVALUATION OF IF STATEMENTS TO DETERMINE THE ABOVE?
        # IMPLEMENTATION NOTE:  FOR NOW, ASSUME B) ABOVE, AND ASSIGN DEFAULT "NULL" VALUES TO offset AND scale
        if offset is None:
            offset = 0.0

        if scale is None:
            scale = 1.0

        # IMPLEMENTATION NOTE: CONFIRM: SHOULD NEVER OCCUR, AS _validate_variable NOW ENFORCES 2D np.ndarray
        # If variable is 0D or 1D:
        # if np_array_less_than_2d(variable):
        #     return (variable * scale) + offset

        means = convert_all_elements_to_np_array([np.mean(item) for item in variable])

        # FIX FOR EFFICIENCY: CHANGE THIS AND WEIGHTS TO TRY/EXCEPT // OR IS IT EVEN NECESSARY, GIVEN VALIDATION ABOVE??
        # Apply exponents if they were specified
        if exponents is not None:
            # Avoid divide by zero warning:
            #    make sure there are no zeros for an element that is assigned a negative exponent
            if (self.is_initializing and
                    any(not any(i) and j < 0 for i, j in zip(variable, exponents))):
                means = np.ones_like(means)
            else:
                means = means ** exponents

        # Apply weights if they were specified
        if weights is not None:
            means = means * weights

        # CALCULATE RESULT USING RELEVANT COMBINATION OPERATION AND MODULATION

        if operation == SUM:
            result = np.sum(means, axis=0) * scale + offset

        elif operation == PRODUCT:
            result = np.prod(means, axis=0) * scale + offset

        else:
            raise FunctionError("Unrecognized operator ({0}) for CombineMeans function".
                                format(self._get_current_parameter_value(OPERATION, context)))

        return self.convert_output_type(result)

    @property
    def offset(self):
        if not hasattr(self, '_offset'):
            return None
        else:
            return self._offset

    @offset.setter
    def offset(self, val):
        self._offset = val

    @property
    def scale(self):
        if not hasattr(self, '_scale'):
            return None
        else:
            return self._scale

    @scale.setter
    def scale(self, val):
        self._scale = val


GAMMA = 'gamma'


class PredictionErrorDeltaFunction(TransformFunction):
    """
    Calculate temporal difference prediction error.

    `function <PredictionErrorDeltaFunction.function>` returns the prediction error using arrays in `variable
    <PredictionErrorDeltaFunction.variable>`:

    .. math::
        \\delta(t) = r(t) + \\gamma sample(t) - sample(t - 1)

    """
    componentName = PREDICTION_ERROR_DELTA_FUNCTION

    classPreferences = {
        PREFERENCE_SET_NAME: 'PredictionErrorDeltaCustomClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    }

    class Parameters(TransformFunction.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <PredictionErrorDeltaFunction.variable>`

                    :default value: numpy.array([[1], [1]])
                    :type: ``numpy.ndarray``

                gamma
                    see `gamma <PredictionErrorDeltaFunction.gamma>`

                    :default value: 1.0
                    :type: ``float``
        """
        variable = Parameter(np.array([[1], [1]]), pnl_internal=True, constructor_argument='default_variable')
        gamma = Parameter(1.0, modulable=True)

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 gamma: Optional[float] = None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):

        super().__init__(
            default_variable=default_variable,
            gamma=gamma,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _validate_variable(self, variable, context=None):
        """
        Insure that all items of variable are numeric

        Parameters
        ----------
        variable
        context

        Returns
        -------
        variable if all items are numeric
        """
        variable = super()._validate_variable(variable=variable, context=context)

        if isinstance(variable, (list, np.ndarray)):
            if isinstance(variable, np.ndarray) and not variable.ndim:
                return variable
            length = 0
            for i in range(1, len(variable)):
                if i == 0:
                    continue
                if isinstance(variable[i - 1], numbers.Number):
                    old_length = 1
                else:
                    old_length = len(variable[i - 1])
                if isinstance(variable[i], numbers.Number):
                    new_length = 1
                else:
                    new_length = len(variable[i])
                if old_length != new_length:
                    raise FunctionError("Length of all arrays in variable {} "
                                        "for {} must be the same".format(variable,
                                                                         self.__class__.__name__))
        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """
        Checks that WEIGHTS is a list or np.array of numbers with length equal
        to variable.

        Note: the checks of compatibility with variable are only performed for
        validation calls during execution (i.e. from `check_args()`), since
        during initialization or COMMAND_LINE assignment, a parameter may be
        re-assigned before variable assigned during is known

        Parameters
        ----------
        request_set
        target_set
        context

        Returns
        -------
        None
        """
        super()._validate_params(request_set,
                                 target_set=target_set,
                                 context=context)

        if GAMMA in target_set and target_set[GAMMA] is not None:
            self._validate_parameter_spec(target_set[GAMMA], GAMMA, numeric_only=True)

        if WEIGHTS in target_set and target_set[WEIGHTS] is not None:
            self._validate_parameter_spec(target_set[WEIGHTS], WEIGHTS, numeric_only=True)
            target_set[WEIGHTS] = np.atleast_2d(target_set[WEIGHTS]).reshape(-1, 1)
            if context.execution_phase & (ContextFlags.EXECUTING):
                if len(target_set[WEIGHTS]) != len(
                        self.defaults.variable):
                    raise FunctionError("Number of weights {} is not equal to "
                                        "number of items in variable {}".format(
                        len(target_set[WEIGHTS]),
                        len(self.defaults.variable.shape)))

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ----------
        variable : 2d np.array : default class_defaults.variable
            a 2d array representing the sample and target values to be used to
            calculate the temporal difference delta values. Both arrays must
            have the same length

        params : Dict[param keyword, param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that
            specifies the parameters for the function. Values specified for
            parameters in the dictionary override any assigned to those
            parameters in arguments of the constructor.


        Returns
        -------
        delta values : 1d np.array

        """
        gamma = self._get_current_parameter_value(GAMMA, context).item()
        sample = variable[0]
        reward = variable[1]
        delta = np.zeros(sample.shape)

        for t in range(1, len(sample)):
            delta[t] = reward[t] + gamma * sample[t] - sample[t - 1]

        return self.convert_output_type(delta)
