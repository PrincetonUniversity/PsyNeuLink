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
* `PredictionErrorDeltaFunction`

Overview
--------

Functions that combine multiple items with the same shape, yielding a result with a single item that has the same
shape as the individual items.

All CombinationFunctions must have two attributes - **multiplicative_param** and **additive_param** -
each of which is assigned the name of one of the function's parameters;
this is for use by ModulatoryProjections (and, in particular, GatingProjections,
when the CombinationFunction is used as the function of an InputPort or OutputPort).


"""

import numbers

import numpy as np
import typecheck as tc

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.functions.function import Function_Base, FunctionError, FunctionOutputType
from psyneulink.core.globals.keywords import \
    ADDITIVE_PARAM, ARRANGEMENT, COMBINATION_FUNCTION_TYPE, COMBINE_MEANS_FUNCTION, CONCATENATE_FUNCTION, \
    DEFAULT_VARIABLE, EXPONENTS, LINEAR_COMBINATION_FUNCTION, MULTIPLICATIVE_PARAM, OFFSET, OPERATION, \
    PREDICTION_ERROR_DELTA_FUNCTION, PRODUCT, REARRANGE_FUNCTION, REDUCE_FUNCTION, SCALE, SUM, WEIGHTS, \
    PREFERENCE_SET_NAME
from psyneulink.core.globals.utilities import convert_to_np_array, is_numeric, np_array_less_than_2d, parameter_spec
from psyneulink.core.globals.context import Context, ContextFlags
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import \
    REPORT_OUTPUT_PREF, is_pref_set, PreferenceEntry, PreferenceLevel

__all__ = ['CombinationFunction', 'Concatenate', 'CombineMeans', 'Rearrange', 'Reduce', 'LinearCombination',
           'PredictionErrorDeltaFunction']

class CombinationFunction(Function_Base):
    """Function that combines multiple items, yielding a result with the same shape as its operands

    All CombinationFunctions must have two attributes - multiplicative_param and additive_param -
        each of which is assigned the name of one of the function's parameters;
        this is for use by ModulatoryProjections (and, in particular, GatingProjections,
        when the CombinationFunction is used as the function of an InputPort or OutputPort).

    """
    componentType = COMBINATION_FUNCTION_TYPE

    class Parameters(Function_Base.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <CombinationFunction.variable>`

                    :default value: numpy.array([0])
                    :type: ``numpy.ndarray``
                    :read only: True
        """
        # variable = np.array([0, 0])
        variable = Parameter(np.array([0]), read_only=True, pnl_internal=True, constructor_argument='default_variable')

    def _gen_llvm_load_param(self, ctx, builder, params, param_name, index, default):
        param_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, param_name)
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


class Concatenate(CombinationFunction):  # ------------------------------------------------------------------------
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

    Concatenates items in outer dimension (axis 0) of of `variable <Concatenate.variable>` into a single array,
    optionally scaling and/or adding an offset to the result after concatenating.

    `function <Concatenate.function>` returns a 1d array with lenght equal to the sum of the lengths of the items
    in `variable <Concatenate.variable>`.

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


    class Parameters(CombinationFunction.Parameters):
        """
            Attributes
            ----------

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

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 scale: tc.optional(parameter_spec) = None,
                 offset: tc.optional(parameter_spec) = None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

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
            if not isinstance(scale, numbers.Number):
                raise FunctionError("{} param of {} ({}) must be a scalar".format(SCALE, self.name, scale))

        if OFFSET in target_set and target_set[OFFSET] is not None:
            offset = target_set[OFFSET]
            if not isinstance(offset, numbers.Number):
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


class Rearrange(CombinationFunction):  # ------------------------------------------------------------------------
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

    class Parameters(CombinationFunction.Parameters):
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

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 scale: tc.optional(parameter_spec) = None,
                 offset: tc.optional(parameter_spec) = None,
                 arrangement:tc.optional(tc.any(int, tuple, list))=None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

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
                                            f"arg (max index = {len(self.parameters.variable.default_value)-1}).")

        # Check that SCALE and OFFSET are scalars.
        if SCALE in target_set and target_set[SCALE] is not None:
            scale = target_set[SCALE]
            if not isinstance(scale, numbers.Number):
                raise FunctionError("{} param of {} ({}) must be a scalar".format(SCALE, self.name, scale))

        if OFFSET in target_set and target_set[OFFSET] is not None:
            offset = target_set[OFFSET]
            if not isinstance(offset, numbers.Number):
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


class Reduce(CombinationFunction):  # ------------------------------------------------------------------------
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


    class Parameters(CombinationFunction.Parameters):
        """
            Attributes
            ----------

                exponents
                    see `exponents <Reduce.exponents>`

                    :default value: None
                    :type:

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

    @tc.typecheck
    def __init__(self,
                 # weights: tc.optional(parameter_spec)=None,
                 # exponents: tc.optional(parameter_spec)=None,
                 weights=None,
                 exponents=None,
                 default_variable=None,
                 operation: tc.optional(tc.enum(SUM, PRODUCT)) = None,
                 scale: tc.optional(parameter_spec) = None,
                 offset: tc.optional(parameter_spec) = None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

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
            if not isinstance(scale, numbers.Number):
                raise FunctionError("{} param of {} ({}) must be a scalar".format(SCALE, self.name, scale))

        if OFFSET in target_set and target_set[OFFSET] is not None:
            offset = target_set[OFFSET]
            if not isinstance(offset, numbers.Number):
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
            result = np.product(np.atleast_2d(variable), axis=1) * scale + offset
        else:
            raise FunctionError("Unrecognized operator ({0}) for Reduce function".
                                format(self._get_current_parameter_value(OPERATION, context)))

        return self.convert_output_type(result)

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

        val_p = builder.alloca(val.type)
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
    CombinationFunction):  # ------------------------------------------------------------------------
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

    operation : SUM or PRODUCT : default SUM
        specifies whether the `function <LinearCombination.function>` takes the elementwise (Hadamarad)
        sum or product of the arrays in `variable  <LinearCombination.variable>`.

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
        determines whether the `function <LinearCombination.function>` takes the elementwise (Hadamard) sum or
        product of the arrays in `variable  <LinearCombination.variable>`.

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

    class Parameters(CombinationFunction.Parameters):
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

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 # weights: tc.optional(parameter_spec)=None,
                 # exponents: tc.optional(parameter_spec)=None,
                 weights=None,
                 exponents=None,
                 operation: tc.optional(tc.enum(SUM, PRODUCT)) = None,
                 scale=None,
                 offset=None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

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
        """Insure that all items of list or np.ndarray in variable are of the same length

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
                if isinstance(variable[i], numbers.Number):
                    new_length = 1
                else:
                    new_length = len(variable[i])
                if old_length != new_length:
                    raise FunctionError("Length of all arrays in variable for {0} must be the same; variable: {1}".
                                        format(self.__class__.__name__, variable))
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
            scale_is_a_scalar = isinstance(scale, numbers.Number) or (len(scale) == 1) and isinstance(scale[0],
                                                                                                      numbers.Number)
            if context.execution_phase & (ContextFlags.PROCESSING | ContextFlags.LEARNING):
                if not scale_is_a_scalar:
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

            offset_is_a_scalar = isinstance(offset, numbers.Number) or (len(offset) == 1) and isinstance(offset[0],
                                                                                                         numbers.Number)
            if context.execution_phase & (ContextFlags.PROCESSING | ContextFlags.LEARNING):
                if not offset_is_a_scalar:
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
        if isinstance(scale, (list, np.ndarray)):
            if len(scale) == 1 and isinstance(scale[0], numbers.Number):
                scale = scale[0]
        if isinstance(offset, (list, np.ndarray)):
            if len(offset) == 1 and isinstance(offset[0], numbers.Number):
                offset = offset[0]

        # CALCULATE RESULT USING RELEVANT COMBINATION OPERATION AND MODULATION
        if operation == SUM:
            combination = np.sum(variable, axis=0)
        elif operation == PRODUCT:
            combination = np.product(variable, axis=0)
        else:
            raise FunctionError("Unrecognized operator ({0}) for LinearCombination function".
                                format(operation.self.Operation.SUM))
        if isinstance(scale, numbers.Number):
            # scalar scale
            product = combination * scale
        else:
            # Hadamard scale
            product = np.product([combination, scale], axis=0)

        if isinstance(offset, numbers.Number):
            # scalar offset
            result = product + offset
        else:
            # Hadamard offset
            result = np.sum([product, offset], axis=0)

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

        val_p = builder.alloca(val.type)
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


class CombineMeans(CombinationFunction):  # ------------------------------------------------------------------------
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

    class Parameters(CombinationFunction.Parameters):
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

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 # weights:tc.optional(parameter_spec)=None,
                 # exponents:tc.optional(parameter_spec)=None,
                 weights=None,
                 exponents=None,
                 operation: tc.optional(tc.enum(SUM, PRODUCT)) = None,
                 scale=None,
                 offset=None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

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

        means = np.array([[None]] * len(variable))
        for i, item in enumerate(variable):
            means[i] = np.mean(item)

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
            result = np.product(means, axis=0) * scale + offset

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


class PredictionErrorDeltaFunction(CombinationFunction):
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

    class Parameters(CombinationFunction.Parameters):
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

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 gamma: tc.optional(tc.optional(float)) = None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

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
        gamma = self._get_current_parameter_value(GAMMA, context)
        sample = variable[0]
        reward = variable[1]
        delta = np.zeros(sample.shape)

        for t in range(1, len(sample)):
            delta[t] = reward[t] + gamma * sample[t] - sample[t - 1]

        return self.convert_output_type(delta)
