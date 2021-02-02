#
# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *******************************************  TRANSFER FUNCTIONS  *****************************************************
"""

* `Identity`
* `Linear`
* `Exponential`
* `Logistic`
* `Tanh`
* `ReLU`
* `Gaussian`
* `GaussianDistort`
* `SoftMax`
* `LinearMatrix`
* `TransferWithCosts`

Overview
--------

Functions that transform their variable but maintain its shape.

All TransferFunctions have the following attributes:

* **bounds**:  specifies the lower and upper limits of the result;  if there are none, the attribute is set to
  `None`;  if it has at least one bound, the attribute is set to a tuple specifying the lower and upper bounds,
  respectively, with `None` as the entry for no bound.
..
* **multiplicative_param** and **additive_param**:
  each of these is assigned the name of one of the function's
  parameters and used by `ModulatoryProjections <ModulatoryProjection>` to modulate the output of the
  TransferFunction's function (see `Function_Modulatory_Params`).

"""

import numbers
import numpy as np
import typecheck as tc
import types
import warnings

from enum import IntEnum
from math import e, pi, sqrt

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.component import parameter_keywords
from psyneulink.core.components.functions.function import (
    Function, Function_Base, FunctionError, function_keywords, get_matrix, is_function_type,
)
from psyneulink.core.components.functions.combinationfunctions import LinearCombination
from psyneulink.core.components.functions.selectionfunctions import OneHot
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import SimpleIntegrator
from psyneulink.core.components.shellclasses import Projection
from psyneulink.core.globals.keywords import \
    ADDITIVE_PARAM, ALL, BIAS, EXPONENTIAL_FUNCTION, \
    GAIN, GAUSSIAN_DISTORT_FUNCTION, GAUSSIAN_FUNCTION, HAS_INITIALIZERS, HOLLOW_MATRIX, \
    IDENTITY_FUNCTION, IDENTITY_MATRIX, INTERCEPT, LEAK, LINEAR_FUNCTION, LINEAR_MATRIX_FUNCTION, LOGISTIC_FUNCTION, \
    TANH_FUNCTION, MATRIX_KEYWORD_NAMES, MATRIX, MATRIX_KEYWORD_VALUES, MAX_INDICATOR, MAX_VAL, MULTIPLICATIVE_PARAM, \
    OFF, OFFSET, ON, PER_ITEM, PROB, PRODUCT, OUTPUT_TYPE, PROB_INDICATOR, \
    RATE, RECEIVER, RELU_FUNCTION, SCALE, SLOPE, SOFTMAX_FUNCTION, STANDARD_DEVIATION, SUM,\
    TRANSFER_FUNCTION_TYPE, TRANSFER_WITH_COSTS_FUNCTION, VARIANCE, VARIABLE, X_0, PREFERENCE_SET_NAME
from psyneulink.core.globals.parameters import \
    FunctionParameter, Parameter, get_validator_by_function
from psyneulink.core.globals.utilities import parameter_spec, get_global_seed, safe_len, convert_to_np_array
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.preferences.basepreferenceset import \
    REPORT_OUTPUT_PREF, PreferenceEntry, PreferenceLevel, is_pref_set

__all__ = ['Exponential', 'Gaussian', 'GaussianDistort', 'Identity', 'Linear', 'LinearMatrix',
           'Logistic', 'ReLU', 'SoftMax', 'Tanh', 'TransferFunction', 'TransferWithCosts'
           ]

class TransferFunction(Function_Base):
    """Function that transforms variable but maintains its shape.

    All TransferFunctions MUST have the following attributes:

    `bounds` -- specifies the lower and upper limits of the result;  if there are none, the attribute is set to
    `None`;  if it has at least one bound, the attribute is set to a tuple specifying the lower and upper bounds,
    respectively, with `None` as the entry for no bound.

    `multiplicative_param <Function_Modulatory_Params>` and `additive_param <Function_Modulatory_Params>` -- each
    of these is assigned the name of one of the function's parameters and used by `ModulatoryProjections
    <ModulatoryProjection>` to modulate the output of the TransferFunction's `function <TransferFunction.function>`
    (see  `Function_Modulatory_Params`).

    """
    componentType = TRANSFER_FUNCTION_TYPE

    class Parameters(Function_Base.Parameters):
        """
            Attributes
            ----------

                bounds
                    see `bounds <TransferFunction.bounds>`

                    :default value: None
                    :type:
        """
        bounds = None


    def _gen_llvm_function_body(self, ctx, builder, params, state, arg_in, arg_out, *, tags:frozenset):
        # Pretend we have one huge array to work on
        # TODO: should this be invoked in parts?
        assert isinstance(arg_in.type.pointee, pnlvm.ir.ArrayType)
        assert arg_in.type == arg_out.type

        is_2d = isinstance(arg_in.type.pointee.element, pnlvm.ir.ArrayType)

        assert arg_in.type == arg_out.type
        with pnlvm.helpers.array_ptr_loop(builder, arg_in, "transfer_loop") as (b, idx):
            if is_2d:
                vi = b.gep(arg_in, [ctx.int32_ty(0), idx])
                vo = b.gep(arg_out, [ctx.int32_ty(0), idx])
                with pnlvm.helpers.array_ptr_loop(b, vi, "nested_transfer_loop") as args:
                    self._gen_llvm_transfer(ctx=ctx, vi=vi, vo=vo,
                                            params=params, state=state, *args, tags=tags)
            else:
                self._gen_llvm_transfer(b, idx, ctx=ctx, vi=arg_in, vo=arg_out,
                                        params=params, state=state, tags=tags)

        return builder


# **********************************************************************************************************************
#                                                 Identity
# **********************************************************************************************************************

class Identity(TransferFunction):  # -----------------------------------------------------------------------------------
    """
    Identity(                  \
             default_variable, \
             params=None,      \
             owner=None,       \
             name=None,        \
             prefs=None        \
            )

    .. _Identity:

    Returns variable.

    Arguments
    ---------

    variable : number or np.array : default class_defaults.variable
        specifies a template for the value to be returned.

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

    variable : number or np.array
        contains value to be returned.

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

    componentName = IDENTITY_FUNCTION

    classPreferences = {
        PREFERENCE_SET_NAME: 'IdentityClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    }

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):
        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         )

        # self.functionOutputType = None

    def _function(
        self,
        variable=None,
        context=None,
        params=None,

    ):
        """
        Return: `variable <Identity.variable>`

        Arguments
        ---------

        variable : number or np.array : default class_defaults.variable
           a single value or array to be returned.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        variable : number or np.array

        """
        # outputType = self.functionOutputType

        return variable

    def _gen_llvm_function_body(self, ctx, builder, _1, _2, arg_in, arg_out, *, tags:frozenset):
        val = builder.load(arg_in)
        builder.store(val, arg_out)
        return builder


# **********************************************************************************************************************
#                                                    Linear
# **********************************************************************************************************************

class Linear(TransferFunction):  # -------------------------------------------------------------------------------------
    """
    Linear(                \
         default_variable, \
         slope=1.0,        \
         intercept=0.0,    \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _Linear:

    `function <Logistic.function>` returns linear transform of `variable <Linear.variable>`:

    .. math::

        slope * variable + intercept

    Note: default values for `slope <Linear.slope>` and `intercept <Linear.intercept>` implement the
    *IDENTITY_FUNCTION*.

    `derivative <Linear.derivative>` returns `slope <Linear.slope>`.

    Arguments
    ---------

    default_variable : number or array : default class_defaults.variable
        specifies a template for the value to be transformed.

    slope : float : default 1.0
        specifies a value by which to multiply `variable <Linear.variable>`.

    intercept : float : default 0.0
        specifies a value to add to each element of `variable <Linear.variable>` after applying `slope <Linear.slope>`.

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
        contains value to be transformed.

    slope : float
        value by which each element of `variable <Linear.variable>` is multiplied before applying the
        `intercept <Linear.intercept>` (if it is specified).

    intercept : float
        value added to each element of `variable <Linear.variable>` after applying the `slope <Linear.slope>`
        (if it is specified).

    bounds : Tuple or None
        determines the lower and upper limits of the result;  if at least one bound is specified, the attribute is
        a tuple specifying the lower and upper bounds, respectively, with `None` as the entry for no bound.

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

    componentName = LINEAR_FUNCTION

    classPreferences = {
        PREFERENCE_SET_NAME: 'LinearClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    }

    _model_spec_class_name_is_generic = True

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                intercept
                    see `intercept <Linear.intercept>`

                    :default value: 0.0
                    :type: ``float``

                slope
                    see `slope <Linear.slope>`

                    :default value: 1.0
                    :type: ``float``
        """
        slope = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        intercept = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 slope: tc.optional(tc.optional(parameter_spec)) = None,
                 intercept: tc.optional(tc.optional(parameter_spec)) = None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

        super().__init__(
            default_variable=default_variable,
            slope=slope,
            intercept=intercept,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])
        slope_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, SLOPE)
        intercept_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, INTERCEPT)

        slope = pnlvm.helpers.load_extract_scalar_array_one(builder, slope_ptr)
        intercept = pnlvm.helpers.load_extract_scalar_array_one(builder, intercept_ptr)


        if "derivative" in tags:
            # f'(x) = m
            val = slope
        else:
            # f(x) = mx + b
            val = builder.load(ptri)
            val = builder.fmul(val, slope)
            val = builder.fadd(val, intercept)

        builder.store(val, ptro)

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        variable : number or array : default class_defaults.variable
           a single value or array to be transformed.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        linear transformation of variable : number or array

        """
        slope = self._get_current_parameter_value(SLOPE, context)
        intercept = self._get_current_parameter_value(INTERCEPT, context)

        # MODIFIED 11/9/17 NEW:
        try:
            # By default, result should be returned as np.ndarray with same dimensionality as input
            result = variable * slope + intercept
        except TypeError:
            if hasattr(variable, "dtype"):
                # If variable is an array with mixed sizes or types, try item-by-item operation
                if variable.dtype == object:
                    result = np.zeros_like(variable)
                    for i, item in enumerate(variable):
                        result[i] = variable[i] * slope + intercept
                else:
                    raise FunctionError("Unrecognized type for {} of {} ({})".format(VARIABLE, self.name, variable))
            # KAM 6/28/18: If the variable does not have a "dtype" attr but made it to this line, then it must be of a
            # type that even np does not recognize -- typically a custom OutputPort variable with items of different
            # shapes (e.g. variable = [[0.0], [0.0], array([[0.0, 0.0]])] )
            elif isinstance(variable, list):
                result = []
                for variable_item in variable:
                    result.append(np.multiply(variable_item, slope) + intercept)
            else:
                raise FunctionError("Unrecognized type for {} of {} ({})".format(VARIABLE, self.name, variable))

        return self.convert_output_type(result)

    @handle_external_context()
    def derivative(self, input=None, output=None, context=None):
        """
        derivative(input)

        Derivative of `function <Linear.function>` at **input**.

        Arguments
        ---------

        input : number
            value of the input to the Linear transform at which derivative is to be taken.

        Returns
        -------

        Slope of function :  number or array

        """

        return self._get_current_parameter_value(SLOPE, context)

    def _is_identity(self, context=None):
        return (
            self.parameters.slope._get(context) == 1
            and self.parameters.intercept._get(context) == 0
        )


# **********************************************************************************************************************
#                                                    Exponential
# **********************************************************************************************************************

class Exponential(TransferFunction):  # --------------------------------------------------------------------------------
    """
    Exponential(           \
         default_variable, \
         rate=1.0,         \
         bias=0.0,         \
         scale=1.0,        \
         offset=0.0,       \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _Exponential:

    `function <Exponential.function>` returns exponential transform of `variable <Exponential.variable>`:

    .. math::
         scale * e^{rate*variable+bias} + offset

    `derivative <Exponential.derivative>` returns the derivative of the Exponential:

    .. math::
        rate*input+bias


    Arguments
    ---------

    default_variable : number or array : default class_defaults.variable
        specifies a template for the value to be transformed.

    rate : float : default 1.0
        specifies a value by which to multiply `variable <Exponential.variable>` before exponentiation.

    bias : float : default 0.0
        specifies a value to add to `variable <Exponential.variable>` after multplying by `rate <Exponential.rate>`
        and before exponentiation.

    scale : float : default 1.0
        specifies a value by which to multiply the exponentiated value of `variable <Exponential.variable>`.

    offset : float : default 0.0
        specifies value to add to the exponentiated value of `variable <Exponential.variable>`
        after multiplying by `scale <Exponentinal.scale>`.

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
        contains value to be transformed.

    rate : float
        value by which `variable <Exponential.variable>` is multiplied before exponentiation;
        assigned as *MULTILICATIVE_PARAM* of the Exponential Function.

    bias : float
        value added to `variable <Exponential.variable>` after multiplying by `rate <Exponential.rate>`
        and before exponentiation;  assigned as *ADDITIVE_PARAM* of the Exponential Function.

    scale : float
        value by which the exponentiated value is multiplied.

    offset : float
        value added to exponentiated value after multiplying by `scale <Exponentinal.scale>`.

    bounds : (0, None)

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

    componentName = EXPONENTIAL_FUNCTION

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                bias
                    see `bias <Exponential.bias>`

                    :default value: 0.0
                    :type: ``float``

                offset
                    see `offset <Exponential.offset>`

                    :default value: 0.0
                    :type: ``float``

                rate
                    see `rate <Exponential.rate>`

                    :default value: 1.0
                    :type: ``float``

                scale
                    see `scale <Exponential.scale>`

                    :default value: 1.0
                    :type: ``float``
        """
        rate = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        bias = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        scale = Parameter(1.0, modulable=True)
        offset = Parameter(0.0, modulable=True)
        bounds = (0, None)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate: tc.optional(parameter_spec) = None,
                 scale: tc.optional(parameter_spec) = None,
                 bias: tc.optional(parameter_spec) = None,
                 offset: tc.optional(parameter_spec) = None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):
        super().__init__(
            default_variable=default_variable,
            rate=rate,
            bias=bias,
            scale=scale,
            offset=offset,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        rate_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, RATE)
        bias_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, BIAS)
        scale_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, SCALE)
        offset_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, OFFSET)

        rate = pnlvm.helpers.load_extract_scalar_array_one(builder, rate_ptr)
        bias = pnlvm.helpers.load_extract_scalar_array_one(builder, bias_ptr)
        scale = pnlvm.helpers.load_extract_scalar_array_one(builder, scale_ptr)
        offset = pnlvm.helpers.load_extract_scalar_array_one(builder, offset_ptr)

        exp_f = ctx.get_builtin("exp", [ctx.float_ty])
        val = builder.load(ptri)
        val = builder.fmul(val, rate)
        val = builder.fadd(val, bias)
        val = builder.call(exp_f, [val])

        if "derivative" in tags:
            # f'(x) = s*r*e^(r*x + b)
            val = builder.fmul(val, scale)
            val = builder.fmul(val, rate)
        else:
            # f(x) = s*e^(r*x + b) + o
            val = builder.fmul(val, scale)
            val = builder.fadd(val, offset)

        builder.store(val, ptro)

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        variable : number or array : default class_defaults.variable
           a single value or array to be exponentiated.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        Exponential transformation of variable : number or array

        """
        rate = self._get_current_parameter_value(RATE, context)
        bias = self._get_current_parameter_value(BIAS, context)
        scale = self._get_current_parameter_value(SCALE, context)
        offset = self._get_current_parameter_value(OFFSET, context)

        # The following doesn't work with autograd (https://github.com/HIPS/autograd/issues/416)
        # result = scale * np.exp(rate * variable + bias) + offset
        result = scale * e**(rate * variable + bias) + offset
        return self.convert_output_type(result)

    @handle_external_context()
    def derivative(self, input, output=None, context=None):
        """
        derivative(input)

        Arguments
        ---------

        input : number
            value of the input to the Exponential transform at which derivative is to be taken.

        Derivative of `function <Exponential.function>` at **input**.

        Returns
        -------

        derivative :  number or array


        """
        rate = self._get_current_parameter_value(RATE, context)
        scale = self._get_current_parameter_value(SCALE, context)
        bias = self._get_current_parameter_value(BIAS, context)

        return rate * scale * e**(rate * input + bias)

# **********************************************************************************************************************
#                                                   Logistic
# **********************************************************************************************************************

class Logistic(TransferFunction):  # ------------------------------------------------------------------------------------
    """
    Logistic(              \
         default_variable, \
         gain=1.0,         \
         bias=0.0,         \
         x_0=0.0,          \
         offset=0.0,       \
         scale=1.0,        \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _Logistic_Function:

    `function <Logistic.function>` returns logistic transform of `variable <Logistic.variable>`:

    .. math::
        \\frac{1}{1 + e^{ - gain ( variable + bias  - x_{0}) + offset}}

    (this is an offset and scaled version of the `Tanh`, which is centered on origin).

    .. note::
        The **bias** and **x_0** arguments are identical, apart from opposite signs: **bias** is included to
        accomodate the convention in the machine learning community; **x_0** is included to match the `standard
        form of the Logistic Function <https://en.wikipedia.org/wiki/Logistic_function>`_ (in which **gain**
        corresponds to the *k* parameter and **scale** corresponds to the *L* parameter).

    `derivative <Logistic.derivative>` returns the derivative of the Logistic using its **output**:

    .. math::
        gain * scale * output * (1-output)

    Arguments
    ---------

    default_variable : number or array : default class_defaults.variable
        specifies a template for the value to be transformed.

    gain : float : default 1.0
        specifies value by which to multiply `variable <Logistic.variable>` before logistic transformation

    bias : float : default 0.0
        specifies value to add to each element of `variable <Logistic.variable>` before applying `gain <Logistic.gain>`
        and before logistic transformation. This argument is identical to x_0, with the opposite sign.

    x_0 : float : default 0.0
        specifies value to subtract from each element of `variable <Logistic.variable>` before applying `gain <Logistic.gain>`
        and before logistic transformation. This argument is identical to bias, with the opposite sign.

    offset : float : default 0.0
        specifies value to add to each element of `variable <Logistic.variable>` after applying `gain <Logistic.gain>`
        but before logistic transformation.

    scale : float : default 0.0
        specifies value by which each element is multiplied after applying the logistic transformation.

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
        contains value to be transformed.

    gain : float : default 1.0
        value by which each element of `variable <Logistic.variable>` is multiplied before applying the
        `bias <Logistic.bias>` (if it is specified).

    bias : float : default 0.0
        value added to each element of `variable <Logistic.variable>` before applying the `gain <Logistic.gain>`
        (if it is specified). This attribute is identical to x_0, with the opposite sign.

    x_0 : float : default 0.0
        value subtracted from each element of `variable <Logistic.variable>` before applying the `gain <Logistic.gain>`
        (if it is specified). This attribute is identical to bias, with the opposite sign.

    offset : float : default 0.0
        value to added to each element of `variable <Logistic.variable>` after applying `gain <Logistic.gain>`
        but before logistic transformation.

    scale : float : default 0.0
        value by which each element is multiplied after applying the Logistic transform.

    bounds : (0,1)

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

    componentName = LOGISTIC_FUNCTION
    parameter_keywords.update({GAIN, BIAS, OFFSET})
    _model_spec_class_name_is_generic = True

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                bias
                    see `bias <Logistic.bias>`

                    :default value: 0.0
                    :type: ``float``

                gain
                    see `gain <Logistic.gain>`

                    :default value: 1.0
                    :type: ``float``

                offset
                    see `offset <Logistic.offset>`

                    :default value: 0.0
                    :type: ``float``

                scale
                    see `scale <Logistic.scale>`

                    :default value: 1.0
                    :type: ``float``

                x_0
                    see `x_0 <Logistic.x_0>`

                    :default value: 0.0
                    :type: ``float``
        """
        gain = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        x_0 = Parameter(0.0, modulable=True)
        bias = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        offset = Parameter(0.0, modulable=True)
        scale = Parameter(1.0, modulable=True)
        bounds = (0, 1)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 gain: tc.optional(parameter_spec) = None,
                 x_0=None,
                 bias=None,
                 offset: tc.optional(parameter_spec) = None,
                 scale: tc.optional(parameter_spec) = None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):
        super().__init__(
            default_variable=default_variable,
            gain=gain,
            x_0=x_0,
            bias=bias,
            offset=offset,
            scale=scale,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        gain_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, GAIN)
        bias_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, BIAS)
        x_0_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, X_0)
        scale_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, SCALE)
        offset_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, OFFSET)

        gain = pnlvm.helpers.load_extract_scalar_array_one(builder, gain_ptr)
        bias = pnlvm.helpers.load_extract_scalar_array_one(builder, bias_ptr)
        x_0 = pnlvm.helpers.load_extract_scalar_array_one(builder, x_0_ptr)
        offset = pnlvm.helpers.load_extract_scalar_array_one(builder, offset_ptr)
        scale = pnlvm.helpers.load_extract_scalar_array_one(builder, scale_ptr)

        exp_f = ctx.get_builtin("exp", [ctx.float_ty])
        val = builder.load(ptri)

        val = builder.fadd(val, bias)
        val = builder.fsub(val, x_0)
        val = builder.fmul(val, gain)
        val = builder.fsub(offset, val)
        val = builder.call(exp_f, [val])
        val = builder.fadd(ctx.float_ty(1), val)
        val = builder.fdiv(ctx.float_ty(1), val)
        val = builder.fmul(val, scale)

        if "derivative" in tags:
            # f(x) = g * s * o * (1-o)
            function_val = val
            val = builder.fsub(ctx.float_ty(1), function_val)
            val = builder.fmul(function_val, val)
            val = builder.fmul(gain, val)
            val = builder.fmul(scale, val)

        builder.store(val, ptro)

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        variable : number or array : default class_defaults.variable
           a single value or array to be transformed.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        Logistic transformation of variable : number or array

        """
        gain = self._get_current_parameter_value(GAIN, context)
        bias = self._get_current_parameter_value(BIAS, context)
        x_0 = self._get_current_parameter_value(X_0, context)
        offset = self._get_current_parameter_value(OFFSET, context)
        scale = self._get_current_parameter_value(SCALE, context)

        # The following doesn't work with autograd (https://github.com/HIPS/autograd/issues/416)
        # result = 1. / (1 + np.exp(-gain * (variable - bias) + offset))
        result = scale * (1. / (1 + e**(-gain * (variable + bias - x_0) + offset)))

        return self.convert_output_type(result)

    @handle_external_context()
    def derivative(self, input=None, output=None, context=None):
        """
        derivative(input=None, output=None)

        Derivative of `function <Exponential.function>` at either **input** or **output**.

        Either **input** or **ouput** must be specified.  If **output** is not specified, it is computed from **input**.
        If both are specified, **input** is ignored unless paramValidationPref is set, in which case
        an error is generated if **output** does not correspond to `function <Logistic.function>`\\(**input**).

        Arguments
        ---------

        input : number
            value of the input to the Logistic transform at which derivative is to be taken.

        output : number
            value of the output of the Logistic transform at which derivative is to be taken.

        Returns
        -------

        Deriviative of logistic transform at output:  number or array

        """
        if output is not None and input is not None and self.prefs.paramValidationPref:
            if isinstance(input, numbers.Number):
                valid = output == self.function(input, context=context)
            else:
                valid = all(output[i] == self.function(input, context=context)[i] for i in range(len(input)))
            if not valid:
                raise FunctionError("Value of {} arg passed to {} ({}) "
                                    "does not match the value expected for specified {} ({})".
                                    format(repr('output'), self.__class__.__name__ + '.' + 'derivative', output,
                                           repr('input'), input))

        gain = self._get_current_parameter_value(GAIN, context)
        scale = self._get_current_parameter_value(SCALE, context)

        if output is None:
            output = self.function(input, context=context)

        return gain * scale * output * (1 - output)


# **********************************************************************************************************************
#                                                    Tanh
# **********************************************************************************************************************

class Tanh(TransferFunction):  # ------------------------------------------------------------------------------------
    """
    Tanh(                  \
         default_variable, \
         gain=1.0,         \
         bias=0.0,         \
         x_0=0.0,          \
         offset=0.0,       \
         scale=1.0,        \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _Tanh_Function:

    `function <Logistic.function>` returns hyperbolic tangent of `variable <Logistic.variable>`:

    .. math::

        \\scale*frac{1 - e^{-2(gain*(variable+bias-x\\_0)+offset)}}{1 + e^{-2(gain*(variable+bias-x\\_0)+offset)}}

    .. note::

       The `Logistic` function is an offset and scaled version of this function.
       The parameters used here have the same meaning as those used for the `Logistic` Function.

    `derivative <Tanh.derivative>` returns the derivative of the hyperbolic tangent at its **input**:

    .. math::
        \\frac{gain*scale}{(\\frac{1+e^{-2(gain*(variable+bias-x\\_0)+offset)}}{2e^{-(gain*(
       variable+bias-x\\_0)+offset)}})^2}

    Arguments
    ---------

    default_variable : number or array : default class_defaults.variable
        specifies template for the value to be transformed.

    gain : float : default 1.0
        specifies value by which to multiply `variable <Tanh.variable>` before logistic transformation

    bias : float : default 0.0
        specifies value to add to each element of `variable <Tanh.variable>` before applying `gain <Tanh.gain>`
        and before logistic transformation. This argument is identical to x_0, with the opposite sign.

    x_0 : float : default 0.0
        specifies value to subtract from each element of `variable <Tanh.variable>` before applying `gain <Tanh.gain>`
        and before logistic transformation. This argument is identical to bias, with the opposite sign.

    offset : float : default 0.0
        specifies value to add to each element of `variable <Tanh.variable>` after applying `gain <Tanh.gain>`
        but before logistic transformation.

    scale : float : default 1.0
        specifies value by which to multiply each element after applying Tanh transform.

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
        contains value to be transformed.

    gain : float : default 1.0
        value by which each element of `variable <Tanh.variable>` is multiplied before applying the
        `bias <Tanh.bias>` (if it is specified).

    bias : float : default 0.0
        value added to each element of `variable <Tanh.variable>` before applying the `gain <Tanh.gain>`
        (if it is specified). This attribute is identical to x_0, with the opposite sign.

    x_0 : float : default 0.0
        value subtracted from each element of `variable <Tanh.variable>` before applying the `gain <Tanh.gain>`
        (if it is specified). This attribute is identical to bias, with the opposite sign.

    offset : float : default 0.0
        value to added to each element of `variable <Tanh.variable>` after applying `gain <Tanh.gain>`
        but before logistic transformation.

    scale : float : default 1.0
        value by which element is multiplied after applying Tanh transform.

    bounds : (0,1)

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

    componentName = TANH_FUNCTION
    parameter_keywords.update({GAIN, BIAS, OFFSET})

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                bias
                    see `bias <Tanh.bias>`

                    :default value: 0.0
                    :type: ``float``

                gain
                    see `gain <Tanh.gain>`

                    :default value: 1.0
                    :type: ``float``

                offset
                    see `offset <Tanh.offset>`

                    :default value: 0.0
                    :type: ``float``

                scale
                    see `scale <Tanh.scale>`

                    :default value: 1.0
                    :type: ``float``

                x_0
                    see `x_0 <Tanh.x_0>`

                    :default value: 0.0
                    :type: ``float``
        """
        gain = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        x_0 = Parameter(0.0, modulable=True)
        bias = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        offset = Parameter(0.0, modulable=True)
        scale = Parameter(1.0, modulable=True)
        bounds = (0, 1)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 gain: tc.optional(parameter_spec) = None,
                 x_0=None,
                 bias=None,
                 offset: tc.optional(parameter_spec) = None,
                 scale: tc.optional(parameter_spec) = None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):
        super().__init__(
            default_variable=default_variable,
            gain=gain,
            x_0=x_0,
            bias=bias,
            offset=offset,
            scale=scale,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        gain_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, GAIN)
        bias_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, BIAS)
        x_0_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, X_0)
        offset_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, OFFSET)
        scale_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, SCALE)

        gain = pnlvm.helpers.load_extract_scalar_array_one(builder, gain_ptr)
        bias = pnlvm.helpers.load_extract_scalar_array_one(builder, bias_ptr)
        x_0 = pnlvm.helpers.load_extract_scalar_array_one(builder, x_0_ptr)
        offset = pnlvm.helpers.load_extract_scalar_array_one(builder, offset_ptr)
        scale = pnlvm.helpers.load_extract_scalar_array_one(builder, scale_ptr)

        variable = builder.load(ptri)
        exp_f = ctx.get_builtin("exp", [ctx.float_ty])

        if "derivative" in tags:
            exponent = builder.fadd(variable, bias)
            exponent = builder.fsub(exponent, x_0)
            exponent = builder.fmul(gain, exponent)
            exponent = builder.fadd(exponent, offset)
            exponent = builder.fmul(exponent.type(-2), exponent)

            mult = builder.fmul(gain, scale)
            mult = builder.fmul(mult.type(-2), mult)

            exp_val = builder.call(exp_f, [exponent])
            numerator = builder.fmul(exp_val.type(-2), exp_val)

            denominator = builder.fadd(exp_val.type(1), exp_val)
            denominator = builder.fmul(denominator, denominator)

            val = builder.fdiv(numerator, denominator)
            val = builder.fmul(val, mult)
        else:
            exp_val = builder.fadd(variable, bias)
            exp_val = builder.fsub(exp_val, x_0)
            exp_val = builder.fmul(exp_val, gain)
            exp_val = builder.fadd(exp_val, offset)
            exp_val = builder.fmul(exp_val.type(-2), exp_val)

            val = builder.call(exp_f, [exp_val])
            val1 = builder.fsub(val.type(1), val)
            val2 = builder.fadd(val.type(1), val)
            val = builder.fdiv(val1, val2)
            val = builder.fmul(val, scale)

        builder.store(val, ptro)

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        variable : number or array : default class_defaults.variable
           a single value or array to be transformed.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        hyperbolic tangent of variable : number or array

        """
        gain = self._get_current_parameter_value(GAIN, context)
        bias = self._get_current_parameter_value(BIAS, context)
        x_0 = self._get_current_parameter_value(X_0, context)
        offset = self._get_current_parameter_value(OFFSET, context)
        scale = self._get_current_parameter_value(SCALE, context)

        # The following probably doesn't work with autograd (https://github.com/HIPS/autograd/issues/416)
        #   (since np.exp doesn't work)
        # result = 1. / (1 + np.tanh(-gain * (variable - bias) + offset))
        exponent = -2 * (gain * (variable + bias - x_0) + offset)
        result = scale * (1 - e**exponent)/ (1 + e**exponent)

        return self.convert_output_type(result)


    @handle_external_context()
    def derivative(self, input, output=None, context=None):
        """
        derivative(input)

        Derivative of `function <Tanh.function>` at **input**.

        Arguments
        ---------

        input : number
            value of the input to the Tanh transform at which derivative is to be taken.

        Returns
        -------
        derivative :  number or array

        """
        gain = self._get_current_parameter_value(GAIN, context)
        bias = self._get_current_parameter_value(BIAS, context)
        x_0 = self._get_current_parameter_value(X_0, context)
        offset = self._get_current_parameter_value(OFFSET, context)
        scale = self._get_current_parameter_value(SCALE, context)

        exponent = -2 * (gain * (input + bias - x_0) + offset)
        mult = -2 * gain * scale
        numerator = -2 * e**(exponent)
        denominator = (1 + e**(exponent))**2

        return mult * (numerator / denominator)


# **********************************************************************************************************************
#                                                    ReLU
# **********************************************************************************************************************

class ReLU(TransferFunction):  # ------------------------------------------------------------------------------------
    """
    ReLU(                  \
         default_variable, \
         gain=1.0,         \
         bias=0.0,         \
         leak=0.0,         \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )
    .. _Relu:

    `function <ReLU.function>` returns rectified linear tranform of `variable <ReLU.variable>`:

    .. math::
        x = gain*(variable - bias)

    .. math::
        max(x, leak * x)

    Commonly used by `ReLU <https://en.wikipedia.org/wiki/Rectifier_(neural_networks>`_ units in neural networks.

    `derivative <ReLU.derivative>` returns the derivative of of the rectified linear tranform at its **input**:

    .. math::
        gain\\ if\\ input > 0,\\ gain*leak\\ otherwise

    Arguments
    ---------
    default_variable : number or array : default class_defaults.variable
        specifies a template for the value to be transformed.
    gain : float : default 1.0
        specifies a value by which to multiply `variable <ReLU.variable>` after `bias <ReLU.bias>` is subtracted
        from it.
    bias : float : default 0.0
        specifies a value to subtract from each element of `variable <ReLU.variable>`.
    leak : float : default 0.0
        specifies a scaling factor between 0 and 1 when (variable - bias) is less than or equal to 0.
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
        contains value to be transformed.

    gain : float : default 1.0
        value by which to multiply `variable <ReLU.variable>` after `bias <ReLU.bias>` is subtracted
        from it.

    bias : float : default 0.0
        value to subtract from each element of `variable <ReLU.variable>`.

    leak : float : default 0.0
        scaling factor between 0 and 1 when (variable - bias) is less than or equal to 0.

    bounds : (None,None)

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

    componentName = RELU_FUNCTION
    parameter_keywords.update({GAIN, BIAS, LEAK})

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                bias
                    see `bias <ReLU.bias>`

                    :default value: 0.0
                    :type: ``float``

                gain
                    see `gain <ReLU.gain>`

                    :default value: 1.0
                    :type: ``float``

                leak
                    see `leak <ReLU.leak>`

                    :default value: 0.0
                    :type: ``float``
        """
        gain = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        bias = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        leak = Parameter(0.0, modulable=True)
        bounds = (None, None)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 gain: tc.optional(parameter_spec) = None,
                 bias: tc.optional(parameter_spec) = None,
                 leak: tc.optional(parameter_spec) = None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):
        super().__init__(
            default_variable=default_variable,
            gain=gain,
            bias=bias,
            leak=leak,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        variable : number or array : default class_defaults.variable
           a single value or array to be transformed.
        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        ReLU transformation of variable : number or array
        """
        gain = self._get_current_parameter_value(GAIN, context)
        bias = self._get_current_parameter_value(BIAS, context)
        leak = self._get_current_parameter_value(LEAK, context)

        # KAM modified 2/15/19 to match https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Leaky_ReLUs
        x = gain * (variable - bias)
        result = np.maximum(x, leak * x)

        return self.convert_output_type(result)

    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        gain_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, GAIN)
        bias_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, BIAS)
        leak_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, LEAK)

        gain = pnlvm.helpers.load_extract_scalar_array_one(builder, gain_ptr)
        bias = pnlvm.helpers.load_extract_scalar_array_one(builder, bias_ptr)
        leak = pnlvm.helpers.load_extract_scalar_array_one(builder, leak_ptr)

        # Maxnum for some reason needs full function prototype
        max_f = ctx.get_builtin("maxnum", [ctx.float_ty])
        var = builder.load(ptri)

        if "derivative" in tags:
            predicate = builder.fcmp_ordered('>', var, var.type(0))
            val = builder.select(predicate, gain, builder.fmul(gain, leak))
        else:
            val = builder.fsub(var, bias)
            val1 = builder.fmul(val, gain)
            val2 = builder.fmul(val1, leak)

            val = builder.call(max_f, [val1, val2])

        builder.store(val, ptro)

    @handle_external_context()
    def derivative(self, input, output=None, context=None):
        """
        derivative(input)

        Derivative of `function <ReLU.function>` at **input**.

        Arguments
        ---------

        input : number
            value of the input to the ReLU transform at which derivative is to be taken.

        Returns
        -------

        derivative :  number or array

        """
        gain = self._get_current_parameter_value(GAIN, context)
        leak = self._get_current_parameter_value(LEAK, context)

        input = np.asarray(input).copy()
        input[input>0] = gain
        input[input<=0] = gain * leak

        return input


# **********************************************************************************************************************
#                                                    Gaussian
# **********************************************************************************************************************

class Gaussian(TransferFunction):  # -----------------------------------------------------------------------------------
    """
    Gaussian(                    \
         default_variable,       \
         standard_deviation=1.0, \
         bias=0.0,               \
         scale=1.0,              \
         offset=0.0,             \
         params=None,            \
         owner=None,             \
         name=None,              \
         prefs=None              \
         )

    .. _Gaussian_Function:

    `function <Gaussian.function>` returns Gaussian transform of `variable <Gaussian.variable>`:

    .. math::
      scale*\\frac{e^{-\\frac{(varible-bias)^{2}}{2\\sigma^{2}}}}{\\sqrt{2\\pi}\\sigma}+offset

    where :math:`\\sigma` = `standard_deviation <Gaussian.standard_deviation>`

    .. note::
        the value returned is deterministic (i.e., the value of the probability density function at variable),
        not a randomly chosen sample from the Gaussian distribution; for the latter, use `GaussianDistort`.

    `derivative <Gaussian.derivative>` returns derivative of the Gaussian transform of `variable <Gaussian.variable>`:

    .. math::

       \\frac{-(variable-bias)*e^{-\\frac{(variable-bias)^{2}}{2\\sigma^{2}}}}{\\sqrt{2\\pi}\\sigma^{3}}

    Arguments
    ---------

    default_variable : number or array : default class_defaults.variable
        specifies a template for the value used as the mean for the Guassian transform.

    standard_deviation : float : default 1.0
        specifies "width" of the Gaussian transform applied to each element of `variable <Gaussian.variable>`.

    bias : float : default 0.0
        value to add to each element of `variable <Gaussian.variable>` before applying Gaussian transform.

    offset : float : default 0.0
        value to add to each element after applying Gaussian transform and `scale <Gaussian.scale>`.

    scale : float : default 1.0
        value by which to multiply each element after applying Gaussian transform.

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
        value used as the mean of the Gaussian transform.

    standard_deviation : float : default 1.0
        standard_deviation used for Gaussian transform.

    bias : float : default 0.0
        value added to each element of `variable <Gaussian.variable>` before applying the Gaussian transform.

    scale : float : default 0.0
        value by which each element is multiplied after applying the Gaussian transform.

    offset : float : default 0.0
        value added to each element after applying the Gaussian transform and scale.

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

    componentName = GAUSSIAN_FUNCTION
    # parameter_keywords.update({STANDARD_DEVIATION, BIAS, SCALE, OFFSET})

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                bias
                    see `bias <Gaussian.bias>`

                    :default value: 0.0
                    :type: ``float``

                offset
                    see `offset <Gaussian.offset>`

                    :default value: 0.0
                    :type: ``float``

                scale
                    see `scale <Gaussian.scale>`

                    :default value: 1.0
                    :type: ``float``

                standard_deviation
                    see `standard_deviation <Gaussian.standard_deviation>`

                    :default value: 1.0
                    :type: ``float``
        """
        standard_deviation = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        bias = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        scale = Parameter(1.0, modulable=True)
        offset = Parameter(0.0, modulable=True)
        bounds = (None, None)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 standard_deviation: tc.optional(parameter_spec) = None,
                 bias: tc.optional(parameter_spec) = None,
                 scale: tc.optional(parameter_spec) = None,
                 offset: tc.optional(parameter_spec) = None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):
        super().__init__(
            default_variable=default_variable,
            standard_deviation=standard_deviation,
            bias=bias,
            scale=scale,
            offset=offset,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        standard_deviation_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, STANDARD_DEVIATION)
        bias_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, BIAS)
        scale_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, SCALE)
        offset_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, OFFSET)

        standard_deviation = pnlvm.helpers.load_extract_scalar_array_one(builder, standard_deviation_ptr)
        bias = pnlvm.helpers.load_extract_scalar_array_one(builder, bias_ptr)
        scale = pnlvm.helpers.load_extract_scalar_array_one(builder, scale_ptr)
        offset = pnlvm.helpers.load_extract_scalar_array_one(builder, offset_ptr)

        exp_f = ctx.get_builtin("exp", [ctx.float_ty])
        sqrt_f = ctx.get_builtin("sqrt", [ctx.float_ty])

        var = builder.load(ptri)
        exp_num = builder.fsub(var, bias)
        exp_num = builder.fmul(exp_num, exp_num)
        exp_num = pnlvm.helpers.fneg(builder, exp_num)

        exp_denom = builder.fmul(standard_deviation, standard_deviation)
        exp_denom = builder.fmul(exp_denom.type(2), exp_denom)
        exp = builder.fdiv(exp_num, exp_denom)
        numerator = builder.call(exp_f, [exp])

        denom = builder.fmul(standard_deviation.type(2 * pi), standard_deviation)
        denom = builder.call(sqrt_f, [denom])
        val = builder.fdiv(numerator, denom)

        val = builder.fmul(scale, val)
        val = builder.fadd(offset, val)

        builder.store(val, ptro)

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        variable : number or array : default class_defaults.variable
           a single value or array to be transformed.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        Gaussian transformation of variable : number or array

        """
        standard_deviation = self._get_current_parameter_value(STANDARD_DEVIATION, context)
        bias = self._get_current_parameter_value(BIAS, context)
        scale = self._get_current_parameter_value(SCALE, context)
        offset = self._get_current_parameter_value(OFFSET, context)

        gaussian = e**(-(variable - bias)**2 / (2 * standard_deviation**2)) / sqrt(2 * pi * standard_deviation)
        result = scale * gaussian + offset

        return self.convert_output_type(result)

    @handle_external_context()
    def derivative(self, input, output=None, context=None):
        """
        derivative(input)

        Derivative of `function <Gaussian.function>` at **input**.


        Arguments
        ---------

        input : number
            value of the input of the Gaussian transform at which derivative is to be taken.


        Returns
        -------

        Derivative of Guassian of variable :  number or array

        """
        sigma = self._get_current_parameter_value(STANDARD_DEVIATION, context)
        bias = self._get_current_parameter_value(BIAS, context)

        adjusted_input = input - bias
        result = (-adjusted_input * e**(-(adjusted_input**2 / (2 * sigma**2)))) / sqrt(2 * pi * sigma**3)

        return self.convert_output_type(result)


# **********************************************************************************************************************
#                                               GaussianDistort
# **********************************************************************************************************************

class GaussianDistort(TransferFunction):  #-----------------------------------------------------------------------------
    """
    GaussianDistort(       \
         default_variable, \
         variance=1.0,     \
         bias=0.0,         \
         scale=1.0,        \
         offset=0.0,       \
         seed=None,        \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _GaussianDistort_Function:

    `function <GaussianDistort.function>` returns random value from a Gaussian distribution with
     mean = `variable <GaussianDistort.variable>` and variance = `variance <GaussianDistort.variance>`

    .. note::
        if the Gaussian transform of `variable <GaussianDistort.variable>` is desired (i.e., the value of the
        probability density function at `variable <GaussianDistort.variable>`, not a randomly chosen sample from the
        Gaussian distribution, then use `Gaussian`.

    COMMENT:
    `derivative <Gaussian.derivative>` returns derivative of the Gaussian transform of `variable <Logistic.variable>`:

    .. math::

       \\frac{-(variable-bias)*e^{-\\frac{(variable-bias)^{2}}{2\\sigma^{2}}}}{\\sqrt{2\\pi}\\sigma^{3}}
    COMMENT

    Arguments
    ---------

    default_variable : number or array : default class_defaults.variable
        specifies a template for the value(s) used as the mean of the Guassian distribution from which each sample is
        drawn.

    variance : float : default 1.0
        specifies "width" of the Gaussian distribution around each element of `variable <GaussianDistort.variable>`
        from which sample is drawn.

    bias : float : default 0.0
        specifies value to add to each element of `variable <GaussianDistort.variable>` before drawing sample.

    scale : float : default 1.0
        specifies value by which to multiply each sample.

    offset : float : default 0.0
        specifies value to add to each sample after it is drawn and `scale <GaussianDistort.scale>` is applied

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
        each element determines mean of the Gaussian distribution from which each sample is drawn.

    variance : float
        determines variance of Gaussian distribution from which each sample is drawn.

    bias : float
        determines value added to each element of `variable <GaussianDistort.variable>` before drawing sample.

    scale : float
        determines value by which each sample is multiplied after it is drawn.

    offset : float
        determines value added to each sample after it is drawn and `scale <GaussianDistort.scale>` is applied

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
    """

    componentName = GAUSSIAN_DISTORT_FUNCTION
    # parameter_keywords.update({VARIANCE, BIAS, SCALE, OFFSET})

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                bias
                    see `bias <GaussianDistort.bias>`

                    :default value: 0.0
                    :type: ``float``

                offset
                    see `offset <GaussianDistort.offset>`

                    :default value: 0.0
                    :type: ``float``

                random_state
                    see `random_state <GaussianDistort.random_state>`

                    :default value: None
                    :type: ``numpy.random.RandomState``

                scale
                    see `scale <GaussianDistort.scale>`

                    :default value: 1.0
                    :type: ``float``

                variance
                    see `variance <GaussianDistort.variance>`

                    :default value: 1.0
                    :type: ``float``
        """
        variance = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        bias = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        scale = Parameter(1.0, modulable=True)
        offset = Parameter(0.0, modulable=True)
        random_state = Parameter(None, stateful=True, loggable=False)
        bounds = (None, None)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 variance: tc.optional(parameter_spec) = None,
                 bias: tc.optional(parameter_spec) = None,
                 scale: tc.optional(parameter_spec) = None,
                 offset: tc.optional(parameter_spec) = None,
                 seed=None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

        if seed is None:
            seed = get_global_seed()

        random_state = np.random.RandomState([seed])

        super().__init__(
            default_variable=default_variable,
            variance=variance,
            bias=bias,
            scale=scale,
            offset=offset,
            random_state=random_state,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        variance_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, VARIANCE)
        bias_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, BIAS)
        scale_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, SCALE)
        offset_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, OFFSET)

        variance = pnlvm.helpers.load_extract_scalar_array_one(builder, variance_ptr)
        bias = pnlvm.helpers.load_extract_scalar_array_one(builder, bias_ptr)
        scale = pnlvm.helpers.load_extract_scalar_array_one(builder, scale_ptr)
        offset = pnlvm.helpers.load_extract_scalar_array_one(builder, offset_ptr)

        rvalp = builder.alloca(ptri.type.pointee)
        rand_state_ptr = pnlvm.helpers.get_state_ptr(builder, self, state, "random_state")
        normal_f = ctx.import_llvm_function("__pnl_builtin_mt_rand_normal")
        builder.call(normal_f, [rand_state_ptr, rvalp])

        rval = builder.load(rvalp)
        rval = builder.fmul(rval, variance)
        val = builder.load(ptri)
        val = builder.fadd(val, bias)
        val = builder.fadd(rval, val)
        val = builder.fmul(val, scale)
        val = builder.fadd(offset, val)

        builder.store(val, ptro)

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        variable : number or array : default class_defaults.variable
           a single value or array to be transformed.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        Sample from Gaussian distribution for each element of variable : number or array

        """
        variance = self._get_current_parameter_value(VARIANCE, context)
        bias = self._get_current_parameter_value(BIAS, context)
        scale = self._get_current_parameter_value(SCALE, context)
        offset = self._get_current_parameter_value(OFFSET, context)
        random_state = self._get_current_parameter_value('random_state', context)

        # The following doesn't work with autograd (https://github.com/HIPS/autograd/issues/416)
        result = scale * random_state.normal(variable + bias, variance) + offset

        return self.convert_output_type(result)

    # def derivative(self, output, input=None, context=None):
    #     """
    #     derivative(output, input):
    #
    #     Derivative of `function <Logistic.function>`:
    #
    #         -input/:math:`{variance^3}*\\sqrt{2\\pi}`
    #
    #
    #     Returns
    #     -------
    #
    #     Derivative of Guassian of variable :  number or array
    #
    #     """
    #     variance = self._get_current_parameter_value(VARIANCE, context)
    #     bias = self._get_current_parameter_value(BIAS, context)
    #     scale = self._get_current_parameter_value(SCALE, context)
    #     offset = self._get_current_parameter_value(OFFSET, context)
    #
    #     # The following doesn't work with autograd (https://github.com/HIPS/autograd/issues/416)
    #     f = scale * np.random.normal(input+bias, variance) + offset
    #
    #     # FIX: SHOULD THIS BE variance**1.5 (since variance = sd**2 and term below is supposed to be sd**3)??
    #     df = -input(variance**3 * np.sqrt(2 * np.pi))
    #
    #     return self.convert_output_type(df*f)


# **********************************************************************************************************************
#                                                   SoftMax
# **********************************************************************************************************************

class SoftMax(TransferFunction):
    """
    SoftMax(               \
         default_variable, \
         gain=1.0,         \
         output=ALL,       \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _SoftMax:

    SoftMax transform of `variable <Softmax.variable>`

    `function <SoftMax.function>` returns SoftMax transform of `variable <Softmax.variable>`:

    .. math::

        \\frac{e^{gain * variable_i}}{\\sum\\limits^{len(variable)}e^{gain * variable}}

    filtered by `ouptput <SoftMax.output>` specification (see `The Softmax function and its derivative
    <http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/>`_ for a nice discussion).

    `derivative <SoftMax.derivative>` returns the derivative of the SoftMax.  If *OUTPUT_TYPE* for the SoftMax
    is *ALL*, returns Jacobian matrix (derivative for each element of the output array with respect to each of the
    others):

    .. math::
        D_jS_i = S_i(\\delta_{i,j} - S_j),\\ where\\ \\delta_{i,j}=1\\ if\\ i=j\\ and\\ \\delta_{i,j}=0\\ if\\ i≠j.

    If *OUTPUT_TYPE* is *MAX_VAL* or *MAX_INDICATOR*, returns 1d array of the derivatives of the maximum
    value with respect to the others (calculated as above). If *OUTPUT_TYPE* is *PROB*, raises an exception
    (since it is ambiguous as to which element would have been chosen by the SoftMax function)

    Arguments
    ---------

    default_variable : 1d array : default class_defaults.variable
        specifies a template for the value to be transformed.

    gain : float : default 1.0
        specifies a value by which to multiply `variable <Linear.variable>` before SoftMax transformation.

    output : ALL, MAX_VAL, MAX_INDICATOR, or PROB : default ALL
        specifies the format of array returned by `function <SoftMax.function>`
        (see `output <SoftMax.output>` for details).

    per_item : boolean : default True
        for 2d variables, determines whether the SoftMax function will be applied to the entire variable (per_item =
        False), or applied to each item in the variable separately (per_item = True).

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

    gain : float
        value by which `variable <Logistic.variable>` is multiplied before the SoftMax transformation;  determines
        the "sharpness" of the distribution.

    output : ALL, MAX_VAL, MAX_INDICATOR, or PROB
        determines how the SoftMax-transformed values of the elements in `variable <SoftMax.variable>` are reported
        in the array returned by `function <SoftMax.function>`:
            * **ALL**: array of all SoftMax-transformed values (the default);
            * **MAX_VAL**: SoftMax-transformed value for the element with the maximum such value, 0 for all others;
            * **MAX_INDICATOR**: 1 for the element with the maximum SoftMax-transformed value, 0 for all others;
            * **PROB**: probabilistically chosen element based on SoftMax-transformed values after setting the
              sum of values to 1 (i.e., their `Luce Ratio <https://en.wikipedia.org/wiki/Luce%27s_choice_axiom>`_),
              0 for all others.

    per_item : boolean : default True
        for 2d variables, determines whether the SoftMax function will be applied to the entire variable (per_item =
        False), or applied to each item in the variable separately (per_item = True).

    bounds : None if `output <SoftMax.output>` == MAX_VAL, else (0,1) : default (0,1)

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

    componentName = SOFTMAX_FUNCTION

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <SoftMax.variable>`

                    :default value: numpy.array(0.)
                    :type: ``numpy.ndarray``
                    :read only: True

                bounds
                    see `bounds <SoftMax.bounds>`

                    :default value: (0, 1)
                    :type: <class 'tuple'>

                gain
                    see `gain <SoftMax.gain>`

                    :default value: 1.0
                    :type: ``float``

                output
                    see `output <SoftMax.output>`

                    :default value: `ALL`
                    :type: ``str``

                per_item
                    see `per_item <SoftMax.per_item>`

                    :default value: True
                    :type: ``bool``
        """
        variable = Parameter(np.array([[0.0]]), read_only=True, pnl_internal=True, constructor_argument='default_variable')
        gain = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        bounds = (0, 1)
        output = ALL
        per_item = Parameter(True, pnl_internal=True)
        one_hot_function = Parameter(OneHot, stateful=False, loggable=False)

        def _validate_output(self, output):
            options = {ALL, MAX_VAL, MAX_INDICATOR, PROB}
            if output in options:
                return None
            else:
                return 'not one of {0}'.format(options)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 gain: tc.optional(parameter_spec) = None,
                 output=None,
                 per_item=None,
                 params: tc.optional(tc.optional(dict)) = None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

        try:
            # needed because one_hot_function is initialized here based
            # on output argument, which may also be passed in params
            output = params['output']
        except (TypeError, KeyError):
            pass

        if output not in {None, ALL}:
            one_hot_function = OneHot(mode=output)
        else:
            one_hot_function = None

        super().__init__(
            default_variable=default_variable,
            gain=gain,
            per_item=per_item,
            output=output,
            one_hot_function=one_hot_function,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _parse_one_hot_function_variable(self, variable):
        if self.defaults.per_item and len(np.shape(variable)) > 1:
            variable = variable[0]

        if self.defaults.output in {PROB, PROB_INDICATOR}:
            prob_dist = np.asarray(variable)
            # creates probability distribution in shape of variable
            prob_dist = np.ones(variable.shape) / safe_len(prob_dist)

            variable = np.asarray([variable, prob_dist])

        return variable

    def _validate_variable(self, variable, context=None):
        if variable is None:
            try:
                return self.defaults.variable
            except AttributeError:
                return self.class_defaults.variable

        return np.asarray(variable)

    def __gen_llvm_exp_sum_max(self, builder, index, ctx, vi, gain, max_ptr, exp_sum_ptr, max_ind_ptr):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])

        exp_f = ctx.get_builtin("exp", [ctx.float_ty])
        orig_val = builder.load(ptri)
        val = builder.fmul(orig_val, gain)
        exp_val = builder.call(exp_f, [val])

        exp_sum = builder.load(exp_sum_ptr)
        new_exp_sum = builder.fadd(exp_sum, exp_val)
        builder.store(new_exp_sum, exp_sum_ptr)

        old_max = builder.load(max_ptr)
        gt = builder.fcmp_ordered(">", exp_val, old_max)
        new_max = builder.select(gt, exp_val, old_max)
        builder.store(new_max, max_ptr)

        old_index = builder.load(max_ind_ptr)
        new_index = builder.select(gt, index, old_index)
        builder.store(new_index, max_ind_ptr)

    def __gen_llvm_exp_div(self, builder, index, ctx, vi, vo, gain, exp_sum):
        assert self.output == ALL
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        exp_f = ctx.get_builtin("exp", [ctx.float_ty])
        orig_val = builder.load(ptri)
        val = builder.fmul(orig_val, gain)
        val = builder.call(exp_f, [val])
        val = builder.fdiv(val, exp_sum)

        builder.store(val, ptro)

    def __gen_llvm_apply(self, ctx, builder, params, _, arg_in, arg_out):
        exp_sum_ptr = builder.alloca(ctx.float_ty)
        builder.store(exp_sum_ptr.type.pointee(0), exp_sum_ptr)

        max_ptr = builder.alloca(ctx.float_ty)
        builder.store(max_ptr.type.pointee(float('-inf')), max_ptr)

        max_ind_ptr = builder.alloca(ctx.int32_ty)
        builder.store(max_ind_ptr.type.pointee(-1), max_ind_ptr)

        gain_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, GAIN)
        gain = pnlvm.helpers.load_extract_scalar_array_one(builder, gain_ptr)

        with pnlvm.helpers.array_ptr_loop(builder, arg_in, "exp_sum_max") as args:
            self.__gen_llvm_exp_sum_max(*args, ctx=ctx, vi=arg_in,
                                        max_ptr=max_ptr, gain=gain,
                                        max_ind_ptr=max_ind_ptr,
                                        exp_sum_ptr=exp_sum_ptr)

        output_type = self.output
        exp_sum = builder.load(exp_sum_ptr)
        index = builder.load(max_ind_ptr)
        ptro = builder.gep(arg_out, [ctx.int32_ty(0), index])

        if output_type == ALL:
            with pnlvm.helpers.array_ptr_loop(builder, arg_in, "exp_div") as args:
                self.__gen_llvm_exp_div(ctx=ctx, vi=arg_in, vo=arg_out,
                                        gain=gain, exp_sum=exp_sum, *args)
        elif output_type == MAX_VAL:
            # zero out the output array
            with pnlvm.helpers.array_ptr_loop(builder, arg_in, "zero_output") as (b,i):
                b.store(ctx.float_ty(0), b.gep(arg_out, [ctx.int32_ty(0), i]))

            ptri = builder.gep(arg_in, [ctx.int32_ty(0), index])
            exp_f = ctx.get_builtin("exp", [ctx.float_ty])
            orig_val = builder.load(ptri)
            val = builder.fmul(orig_val, gain)
            val = builder.call(exp_f, [val])
            val = builder.fdiv(val, exp_sum)
            builder.store(val, ptro)
        elif output_type == MAX_INDICATOR:
            # zero out the output array
            with pnlvm.helpers.array_ptr_loop(builder, arg_in, "zero_output") as (b,i):
                b.store(ctx.float_ty(0), b.gep(arg_out, [ctx.int32_ty(0), i]))
            builder.store(ctx.float_ty(1), ptro)

        return builder

    def _gen_llvm_function_body(self, ctx, builder, params, _, arg_in, arg_out, *, tags:frozenset):
        if self.parameters.per_item.get():
            assert isinstance(arg_in.type.pointee.element, pnlvm.ir.ArrayType)
            assert isinstance(arg_out.type.pointee.element, pnlvm.ir.ArrayType)
            for i in range(arg_in.type.pointee.count):
                inner_in = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(i)])
                inner_out = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(i)])
                builder = self.__gen_llvm_apply(ctx, builder, params, _, inner_in, inner_out)
            return builder
        else:
            return self.__gen_llvm_apply(ctx, builder, params, _, arg_in, arg_out)

    def apply_softmax(self, input_value, gain, output_type):
        # Modulate input_value by gain
        v = gain * input_value
        # Shift by max to avoid extreme values:
        v = v - np.max(v)
        # Exponentiate
        v = np.exp(v)
        # Normalize (to sum to 1)
        sm = v / np.sum(v, axis=0)

        # Generate one-hot encoding based on selected output_type

        if output_type in {MAX_VAL, MAX_INDICATOR}:
            return self.one_hot_function(sm)
        elif output_type in {PROB, PROB_INDICATOR}:
            return self.one_hot_function([input_value, sm])
        else:
            return sm

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        variable : 1d array : default class_defaults.variable
           an array to be transformed.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        SoftMax transformation of variable : number or array

        """
        # Assign the params and return the result
        output_type = self._get_current_parameter_value(OUTPUT_TYPE, context)
        gain = self._get_current_parameter_value(GAIN, context)
        per_item = self._get_current_parameter_value(PER_ITEM, context)
        # Compute softmax and assign to sm

        if per_item and len(np.shape(variable)) > 1:
            output = []
            for item in variable:
                output.append(self.apply_softmax(item, gain, output_type))
        else:
            output = self.apply_softmax(variable, gain, output_type)

        return self.convert_output_type(output)

    @handle_external_context()
    def derivative(self, output, input=None, context=None):
        """
        derivative(output)

        Returns
        -------

        derivative of values returned by SoftMax :  1d or 2d array (depending on *OUTPUT_TYPE* of SoftMax)
        """

        output_type = self.output_type
        size = len(output)
        sm = self.function(output, params={OUTPUT_TYPE: ALL}, context=context)

        if output_type == ALL:
            # Return full Jacobian matrix of derivatives
            derivative = np.empty([size, size])
            for j in range(size):
                for i, val in zip(range(size), output):
                    if i == j:
                        d = 1
                    else:
                        d = 0
                    derivative[j, i] = sm[i] * (d - sm[j])

        elif output_type in {MAX_VAL, MAX_INDICATOR}:
            # Return 1d array of derivatives for max element (i.e., the one chosen by SoftMax)
            derivative = np.empty(size)
            # Get the element of output returned as non-zero when output_type is not ALL
            index_of_max = int(np.where(output == np.max(output))[0])
            max_val = sm[index_of_max]
            for i in range(size):
                if i == index_of_max:
                    d = 1
                else:
                    d = 0
                derivative[i] = sm[i] * (d - max_val)

        else:
            raise FunctionError("Can't assign derivative for SoftMax function{} since OUTPUT_TYPE is PROB "
                                "(and therefore the relevant element is ambiguous)".format(self.owner_name))

        return derivative


# **********************************************************************************************************************
#                                                 LinearMatrix
# **********************************************************************************************************************

class LinearMatrix(TransferFunction):  # -------------------------------------------------------------------------------
    """
    LinearMatrix(          \
         default_variable, \
         matrix=None,      \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _LinearMatrix:

    Matrix transform of `variable <LinearMatrix.variable>`.

    `function <LinearMatrix.function>` returns dot product of variable with matrix:

    .. math::
        variable \\bullet matrix

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
        <LinearMatrix.matrix>`.

    matrix : number, list, 1d or 2d np.ndarray, np.matrix, function, or matrix keyword : default IDENTITY_MATRIX
        specifies matrix used to transform `variable <LinearMatrix.variable>`
        (see `matrix <LinearMatrix.matrix>` for specification details).

        When LinearMatrix is the `function <Projection_Base.function>` of a projection:

            - the matrix specification must be compatible with the variables of the `sender <Projection_Base.sender>`
              and `receiver <Projection_Base.receiver>`

            - a matrix keyword specification generates a matrix based on the sender and receiver shapes

        When LinearMatrix is instantiated on its own, or as the function of a `Mechanism <Mechanism>` or `Port`:

            - the matrix specification must be compatible with the function's own `variable <LinearMatrix.variable>`

            - if matrix is not specified, a square identity matrix is generated based on the number of columns in
              `variable <LinearMatrix.variable>`

            - matrix keywords are not valid matrix specifications

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
        matrix used to transform `variable <LinearMatrix.variable>`.
        Can be specified as any of the following:
            * number - used as the filler value for all elements of the :keyword:`matrix` (call to np.fill);
            * list of arrays, 2d array or np.matrix - assigned as the value of :keyword:`matrix`;
            * matrix keyword - see `MatrixKeywords` for list of options.
        Rows correspond to elements of the input array (outer index), and
        columns correspond to elements of the output array (inner index).

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

    componentName = LINEAR_MATRIX_FUNCTION

    DEFAULT_FILLER_VALUE = 0

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                matrix
                    see `matrix <LinearMatrix.matrix>`

                    :default value: None
                    :type:
        """
        matrix = Parameter(None, modulable=True)
        bounds = None

    # def is_matrix_spec(m):
    #     if m is None:
    #         return True
    #     if m in MATRIX_KEYWORD_VALUES:
    #         return True
    #     if isinstance(m, (list, np.ndarray, np.matrix, types.FunctionType)):
    #         return True
    #     return False

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 matrix=None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

        # Note: this calls _validate_variable and _validate_params which are overridden below;
        #       the latter implements the matrix if required
        # super(LinearMatrix, self).__init__(default_variable=default_variable,
        super().__init__(
            default_variable=default_variable,
            matrix=matrix,
            params=params,
            owner=owner,
            prefs=prefs,
        )

        self.parameters.matrix.set(
            self.instantiate_matrix(self.parameters.matrix.get()),
            skip_log=True,
        )

    # def _validate_variable(self, variable, context=None):
    #     """Insure that variable passed to LinearMatrix is a max 2D array
    #
    #     :param variable: (max 2D array)
    #     :param context:
    #     :return:
    #     """
    #     variable = super()._validate_variable(variable, context)
    #
    #     # Check that variable <= 2D
    #     try:
    #         if not variable.ndim <= 2:
    #             raise FunctionError("variable ({0}) for {1} must be a numpy.ndarray of dimension at most 2".format(variable, self.__class__.__name__))
    #     except AttributeError:
    #         raise FunctionError("PROGRAM ERROR: variable ({0}) for {1} should be a numpy.ndarray".
    #                                 format(variable, self.__class__.__name__))
    #
    #     return variable


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
        if hasattr(self.owner, "receiver"):
            sender = self.defaults.variable
            sender_len = np.size(np.atleast_2d(self.defaults.variable), 1)

            # FIX: RELABEL sender -> input AND receiver -> output
            # FIX: THIS NEEDS TO BE CLEANED UP:
            #      - AT LEAST CHANGE THE NAME FROM kwReceiver TO output_template OR SOMETHING LIKE THAT
            #      - MAKE ARG?  OR ADD OTHER PARAMS:  E.G., FILLER?
            #      - OR REFACTOR TO INCLUDE AS MATRIX SPEC:
            #          IF MATRIX IS 1D, USE AS OUTPUT TEMPLATE
            #          IF ALL ITS VALUES ARE 1'S => FULL CONNECTIVITY MATRIX
            #          IF ALL ITS VALUES ARE 0'S => RANDOM CONNECTIVITY MATRIX
            #          NOTE:  NO NEED FOR IDENTITY MATRIX, AS THAT WOULD BE SQUARE SO NO NEED FOR OUTPUT TEMPLATE
            #      - DOCUMENT WHEN DONE
            # MODIFIED 3/26/17 OLD:
            # Check for and validate kwReceiver first, since it may be needed to validate and/or construct the matrix
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
            # No receiver, so use sender as template (assuming square -- e.g., identity -- matrix)
            else:
                if (self.owner and self.owner.prefs.verbosePref) or self.prefs.verbosePref:
                    print("Identity matrix requested but kwReceiver not specified; sender length ({0}) will be used".
                          format(sender_len))
                self.receiver = param_set[RECEIVER] = sender

            receiver_len = len(self.receiver)

            # Check rest of params
            message = ""
            for param_name, param_value in param_set.items():

                # Receiver param already checked above
                if param_name == RECEIVER:
                    continue

                # Not currently used here
                if param_name in function_keywords:
                    continue

                if param_name == HAS_INITIALIZERS:
                    continue

                # Matrix specification param
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
                    elif param_value in MATRIX_KEYWORD_VALUES:
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

                    # string used to describe matrix, so convert to np.matrix and pass to validation of matrix below
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
                    # -  validate that it returns an array or np.matrix
                    elif isinstance(param_value, types.FunctionType):
                        test = param_value(1, 1)
                        if not isinstance(test, (np.ndarray, np.matrix)):
                            raise FunctionError("A function is specified for the matrix of the {} function of {}: {}) "
                                                "that returns a value ({}) that is neither a matrix nor an array".
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
            if message:
                raise FunctionError(message)

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
                elif param_value in MATRIX_KEYWORD_VALUES:
                    raise FunctionError("{} is not a valid specification for the matrix parameter of {}. Keywords "
                                        "may only be used to specify the matrix parameter of a Projection's "
                                        "LinearMatrix function. When the LinearMatrix function is implemented in a "
                                        "mechanism, such as {}, the correct matrix cannot be determined from a "
                                        "keyword. Instead, the matrix must be fully specified as a float, list, "
                                        "np.ndarray, or np.matrix".
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
            self.receiver = self.defaults.variable

        matrix = self.parameters.matrix._get(context)

        if matrix is None and not hasattr(self.owner, "receiver"):
            variable_length = np.size(np.atleast_2d(self.defaults.variable), 1)
            matrix = np.identity(variable_length)
        self.parameters.matrix._set(self.instantiate_matrix(matrix), context)

    def instantiate_matrix(self, specification, context=None):
        """Implements matrix indicated by specification

         Specification is derived from MATRIX param (passed to self.__init__ or self.function)

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

            sender = self.defaults.variable
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


    def _gen_llvm_function_body(self, ctx, builder, params, _, arg_in, arg_out, *, tags:frozenset):
        # Restrict to 1d arrays
        if self.defaults.variable.ndim != 1:
            warnings.warn("Shape mismatch: {} (in {}) got 2D input: {}".format(self, self.owner, self.defaults.variable))
            arg_in = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(0)])
        if self.defaults.value.ndim != 1:
            warnings.warn("Shape mismatch: {} (in {}) has 2D output: {}".format(self, self.owner, self.defaults.value))
            arg_out = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(0)])

        matrix = pnlvm.helpers.get_param_ptr(builder, self, params, MATRIX)

        # Convert array pointer to pointer to the fist element
        matrix = builder.gep(matrix, [ctx.int32_ty(0), ctx.int32_ty(0)])
        vec_in = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(0)])
        vec_out = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(0)])

        input_length = ctx.int32_ty(arg_in.type.pointee.count)
        output_length = ctx.int32_ty(arg_out.type.pointee.count)
        builtin = ctx.import_llvm_function("__pnl_builtin_vxm")
        builder.call(builtin, [vec_in, matrix, input_length, output_length, vec_out])
        return builder

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------
        variable : list or 1d array
            array to be transformed;  length must equal the number of rows of `matrix <LinearMatrix.matrix>`.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        ---------

        dot product of variable and matrix : 1d array
            length of the array returned equals the number of columns of `matrix <LinearMatrix.matrix>`.

        """
        matrix = self._get_current_parameter_value(MATRIX, context)
        result = np.dot(variable, matrix)
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

    def _is_identity(self, context=None):
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
#     if isinstance(m, (list, np.ndarray, np.matrix, types.FunctionType)):
#         return True
#     if m in MATRIX_KEYWORD_VALUES:
#         return True
#     return False


# **********************************************************************************************************************
#                                             TransferWithCosts
# **********************************************************************************************************************

# Keywords for TransferWithCosts arguments, cost functions and their parameters ----------------------------------------

# Make accessible externally
__all__.extend(['ENABLED_COST_FUNCTIONS',
                'INTENSITY_COST',
                'INTENSITY_COST_FUNCTION',
                'INTENSITY_COST_FCT_MULTIPLICATIVE_PARAM',
                'INTENSITY_COST_FCT_ADDITIVE_PARAM',
                'ADJUSTMENT_COST',
                'ADJUSTMENT_COST_FUNCTION',
                'ADJUSTMENT_COST_FCT_MULTIPLICATIVE_PARAM',
                'ADJUSTMENT_COST_FCT_ADDITIVE_PARAM',
                'DURATION_COST',
                'DURATION_COST_FUNCTION',
                'DURATION_COST_FCT_MULTIPLICATIVE_PARAM',
                'DURATION_COST_FCT_ADDITIVE_PARAM',
                'COMBINED_COSTS',
                'COMBINE_COSTS_FUNCTION',
                'COMBINE_COSTS_FCT_MULTIPLICATIVE_PARAM',
                'COMBINE_COSTS_FCT_ADDITIVE_PARAM',
                'costFunctionNames', 'CostFunctions'
                ])

ENABLED_COST_FUNCTIONS = 'enabled_cost_functions'

# These are assigned to TransferWithCosts Function to make them accesible for modulation
INTENSITY_COST = 'intensity_cost'
INTENSITY_COST_FUNCTION = 'intensity_cost_fct'
INTENSITY_COST_FCT_MULTIPLICATIVE_PARAM = 'intensity_cost_fct_mult_param'
INTENSITY_COST_FCT_ADDITIVE_PARAM = 'intensity_cost_fct_add_param'

ADJUSTMENT_COST = 'adjustment_cost'
ADJUSTMENT_COST_FUNCTION = 'adjustment_cost_fct'
ADJUSTMENT_COST_FCT_MULTIPLICATIVE_PARAM = 'adjustment_cost_fct_mult_param'
ADJUSTMENT_COST_FCT_ADDITIVE_PARAM = 'adjustment_cost_fct_add_param'

DURATION_COST = 'duration_cost'
DURATION_COST_FUNCTION = 'duration_cost_fct'
DURATION_COST_FCT_MULTIPLICATIVE_PARAM = 'duration_cost_fct_mult_param'
DURATION_COST_FCT_ADDITIVE_PARAM = 'duration_cost_fct_add_param'

COMBINED_COSTS = 'combined_costs'
COMBINE_COSTS_FUNCTION = 'combine_costs_fct'
COMBINE_COSTS_FCT_MULTIPLICATIVE_PARAM = 'combine_costs_fct_mult_param'
COMBINE_COSTS_FCT_ADDITIVE_PARAM = 'combine_costs_fct_add_param'

costFunctionNames = [INTENSITY_COST_FUNCTION,
                     ADJUSTMENT_COST_FUNCTION,
                     DURATION_COST_FUNCTION,
                     COMBINE_COSTS_FUNCTION]


class CostFunctions(IntEnum):
    """Options for selecting constituent cost functions to be used by a `TransferWithCosts` Function.

    These can be used alone or in combination with one another, by enabling or disabling each using the
    `TransferWithCosts` Function's `enable_costs <TransferWithCosts.enable_costs>`,
    `disable_costs <TransferWithCosts.disable_costs>`, `toggle_cost <TransferWithCosts.toggle_cost>` and
    `assign_costs <TransferWithCosts.assign_costs>` methods.

    Attributes
    ----------

    NONE
        `cost <TransferWithCosts.cost>` is not computed.

    INTENSITY
        `duration_cost_fct` is used to calculate a contribution to the `cost <TransferWithCosts.cost>`
        based its current `intensity <TransferWithCosts.intensity>` value.

    ADJUSTMENT
        `adjustment_cost_fct` is used to calculate a contribution to the `cost <TransferWithCosts.cost>`
        based on the change in its `intensity <TransferWithCosts.intensity>` from its last value.

    DURATION
        `duration_cost_fct` is used to calculate a contribitution to the `cost <TransferWithCosts.cost>`
        based on its integral (i.e., it accumulated value over multiple executions).

    ALL
        all of the cost functions are used to calculate `cost <TransferWithCosts.cost>`.

    DEFAULTS
        assign default set of cost functions as `INTENSITY`).

    """
    NONE          = 0
    INTENSITY     = 1 << 1
    ADJUSTMENT    = 1 << 2
    DURATION      = 1 << 3
    ALL           = INTENSITY | ADJUSTMENT | DURATION
    DEFAULTS      = INTENSITY


TRANSFER_FCT = 'transfer_fct'
INTENSITY_COST_FCT = 'intensity_cost_fct'
ADJUSTMENT_COST_FCT = 'adjustment_cost_fct'
DURATION_COST_FCT = 'duration_cost_fct'
COMBINE_COSTS_FCT = 'combine_costs_fct'

class TransferWithCosts(TransferFunction):
    """
    TransferWithCosts(                      \
        default_variable=None,              \
        size=None,                          \
        transfer_fct=Line                   \
        enabled_cost_functions=None,        \
        intensity_fct=Exponential           \
        adjustment_fct=Linear               \
        duration_fct=SimpleIntegrator       \
        combine_costs_fct=LinearCombination \
        params=None,                        \
        owner=None,                         \
        prefs=None                          \
        )

    .. _TransferWithCosts:

    returns value of `variable <TransferWithCosts.variable>` transformed by `transfer_fct
    <TransferWithCosts.transfer_fct>`, after calling any cost functions that are enabled and assigning
    the result(s) to the corresponding parameter(s), as described below.

    .. _TransferWithCosts_Cost_Functions:

    **Cost Functions**

    The TransferWithCosts function has three individual cost functions that it can execute when its `function
    <TransferWithCosts.function>` is executed, which assign their results to the attributes indicated below:

    * `intensity_cost_fct <TransferWithCosts.intensity_cost_fct>` -> `intensity_cost <TransferWithCosts.intensity_cost>`;
    * `adjustment_cost_fct <TransferWithCosts.adjustment_cost_fct>` -> `adjustment_cost <TransferWithCosts.adjustment_cost>`;
    * `duration_cost_fct <TransferWithCosts.duration_cost_fct>` -> `duration_cost <TransferWithCosts.duration_cost>`;

    Which functions are called is determined by the settings in `enabled_cost_functions
    <TransferWithCosts.enabled_cost_functions>`, that can be initialized in the constructor using the
    **enabled_cost_functions** argument, and later modified using the `enable_costs <TransferWithCosts.enable_costs>`,
    `disable_costs <TransferWithCosts.disable_costs>`, `toggle_cost <TransferWithCosts.toggle_cost>` and
    `assign_costs <TransferWithCosts.assign_costs>` methods.  The value of any cost for which its function has
    *never* been enabled is None;  otherwise, it is the value assigned when it was last enabled and executed
    (see `duration_cost_fct <TransferWithCosts.duration_cost_fct>` for additional details concerning that function).

    If any cost functions are enabled, then the `combine_costs_fct <TransferWithCosts.combine_costs_fct>` function
    is executed, which sums the results of those that are enabled (Hadamard style, if the costs are arrays), and
    stores the result in the `combined_costs <TransferWithCosts.combined_costs>` attribute.  Its value is None if no
    cost functions have ever been enabled;  otherwise it is the value assigned the last time one or more cost functions
    were enabled.

    .. _TransferWithCosts_Modulation_of_Cost_Params:

    **Modulation of Cost Function Parameters**

    The `multiplicative_param <Function_Modulatory_Params>` and `additive_param <Function_Modulatory_Params>` of each
    `cost function <TransferWithCosts_Cost_Functions>` is assigned as a parameter of the TransferWithCost `Function`.
    This makes them accessible for `modulation <ModulatorySignal_Modulation>` when the Function is assigned to a
    `Port` (e.g., as the default `function <ControlSignal.function>` of a `ControlSignal`), or a `Mechanism
    <Mechanism>`.  They can be referred to in the **modulation** argument of a `ModulatorySignal`\\'s constructor
    (see `ModulatorySignal_Types`) using the following keywords:

        *INTENSITY_COST_FCT_MULTIPLICATIVE_PARAM*
        *INTENSITY_COST_FCT_ADDITIVE_PARAM*
        *ADJUSTMENT_COST_FCT_MULTIPLICATIVE_PARAM*
        *ADJUSTMENT_COST_FCT_ADDITIVE_PARAM*
        *DURATION_COST_FCT_MULTIPLICATIVE_PARAM*
        *DURATION_COST_FCT_ADDITIVE_PARAM*
        *COMBINE_COSTS_FCT_MULTIPLICATIVE_PARAM*
        *COMBINE_COSTS_FCT_ADDITIVE_PARAM*
    |
    See `example <ControlSignal_Example_Modulate_Costs>` of how these keywords can be used to
    modulate the parameters of the cost functions of a TransferMechanism assigned to a ControlSignal.

    Arguments
    ---------

    variable : list or 1d array of numbers: Default class_defaults.variable
        specifies shape and default value of the array for variable used by `transfer_fct
        <TransferWithCosts.transfer_fct>`
        on which costs are calculated.

    size : int : None
        specifies length of the array for `variable <TransferWithCosts.variable>` used by `function
        <TransferWithCosts.function>` and on which costs are calculated;  can be used in place of
        default_value, in which case zeros are assigned as the value(s). An error is generated if both are
        specified but size != len(default_value).

    transfer_fct : TransferFunction : Linear
        specifies the primary function, used to generate the value it returns.

    enabled_cost_functions : CostFunctions or List[CostFunctions] : None
        specifies the costs to execute when `function <TransferWithCosts.function>` is called, and
        include in the computation of `combined_costs <TransferWithCosts.combined_costs>`.

    intensity_cost_fct : Optional[`TransferFunction`] : default `Exponential`
        specifies the function used to compute the `intensity_cost <TransferWithCosts.intensity_cost>`.

    adjustment_cost_fct : Optional[`TransferFunction`] : default `Linear`
        specifies the function used to compute the `adjustment_cost <TransferWithCosts.adjustment_cost>`.

    duration_cost_fct : `IntegratorFunction` : default `IntegratorFunction`
        specifies the function used to compute the `duration_cost <TransferWithCosts.duration_cost>`.

    combine_costs_fct : function : default `LinearCombination`
        specifies the function used to compute `combined_costs <TransferWithCosts.combined_costs>`.

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
        value used by `function <TransferWithCosts.function>`, and on which `intensity <TransferWithCosts.intensity>`
        and associated costs are calculated.

    size : int
        length of array for `variable <TransferWithCosts.variable>`.

    intensity : 1 array
        the result of the transfer_fct <TransferWithCosts.transfer_fct>`, and the value returned by
        `function <TransferWithCosts.function>`.

    function : TransferFunction
        primary function, specified by **transfer_fct** argument of constructor, and also stored in
        `transfer_fct <TransferWithCosts.transfer_fct>`.

    transfer_fct : TransferMechanism
        the TransferWithCosts Function's primary function, used to generate the value it returns;
        same as `function <TransferWithCosts.function>`.

    enabled_cost_functions : CostFunctions or None
        boolean combination of currently enabled CostFunctions;  determines which `cost functions
        <TransferWithCosts_Cost_Functions>` are calculated when `function <TransferWithCosts.function>` is called, and
        are included in the computation of `combined_costs <TransferWithCosts.combined_costs>` (see
        `Cost Functions <TransferWithCosts_Cost_Functions>` for additional details).

    intensity_cost : float or None
        cost computed by `intensity_cost_fct <TransferWithCosts.intensity_cost_fct>` for current `intensity
        <TransferWithCosts.intensity>`.  Value is None if `intensity_cost_fct <TransferWithCosts.intensity_cost_fct>`
        has not been enabled (see `Cost Functions <TransferWithCosts_Cost_Functions>` for additional details).

    intensity_cost_fct : TransferFunction
        calculates `intensity_cost` from the current value of `intensity <TransferWithCosts.intensity>`.
        It can be any `TransferFunction`, or any other function that takes and returns a scalar value.
        The default is `Exponential`.

    intensity_cost_fct_mult_param : value
        references value of the `multiplicative_param <Function_Modulatory_Params>` of `intensity_cost_fct
        <TransferWithCosts.intensity_cost_fct>`.

    intensity_cost_fct_add_param : value
        references value of the `additive_param <Function_Modulatory_Params>` of `intensity_cost_fct
        <TransferWithCosts.intensity_cost_fct>`.

    adjustment_cost : float or None
        cost of change in `intensity <TransferWithCosts.intensity>` from the last time `function
        <TransferWithCosts.function>` was executed.  Value is None if `adjustment_cost_fct
        <TransferWithCosts.adjustment_cost_fct>` has not been enabled (see `Cost Functions
        <TransferWithCosts_Cost_Functions>` for additional details).

    adjustment_cost_fct : TransferFunction
        calculates `adjustment_cost <TransferWithCosts.adjustment_cost>` based on the change in `intensity
        <TransferWithCosts.intensity>` from its value the last time `function <TransferWithCosts.function>` was
        executed. It can be any `TransferFunction`, or any other function that takes and returns a scalar value.

    adjustment_cost_fct_mult_param : value
        references value of the `multiplicative_param <Function_Modulatory_Params>` of `adjustment_cost_fct
        <TransferWithCosts.adjustment_cost_fct>`.

    adjustment_cost_fct_add_param : value
        references value of the `additive_param <Function_Modulatory_Params>` of `adjustment_cost_fct
        <TransferWithCosts.adjustment_cost_fct>`.

    duration_cost : float or None
        integral of `intensity <intensity <TransferWithCosts.intensity>`,  computed by `duration_cost_fct
        <TransferWithCosts.duration_cost_fct>`.  Value is None if `duration_cost_fct
        <TransferWithCosts.duration_cost_fct>` has not been enabled; othewise, the integral of
        `intensity <intensity <TransferWithCosts.intensity>` is only for those executions of `function
        <TransferWithCosts.function>` in which `function <TransferWithCosts.duration_cost_fct>` was enabled.

    duration_cost_fct : IntegratorFunction
        calculates an integral of `intensity <TransferWithCosts.intensity>`.  It can be any `IntegratorFunction`,
        or any other function that takes a list or array of two values and returns a scalar value.

    duration_cost_fct_mult_param : value
        references value of the `multiplicative_param <Function_Modulatory_Params>` of `duration_cost_fct
        <TransferWithCosts.duration_cost_fct>`.

    duration_cost_fct_add_param : value
        references value of the `additive_param <Function_Modulatory_Params>` of `duration_cost_fct
        <TransferWithCosts.duration_cost_fct>`.

    combined_costs : float or None
        combined result of all `cost functions <TransferWithCostss_Cost_Functions>` that are enabled;
        computed by `combined_costs_fct <TransferWithCosts.combined_costs_fct>` for current `intensity
        <TransferWithCosts.intensity>`.  Value is None if no costs have been enabled (see
        `Cost Functions <TransferWithCosts_Cost_Functions>` for additional details).

    combine_costs_fct : function
        combines the results of all `cost functions <TransferWithCostss_Cost_Functions>` that are enabled, and assigns
        the result to `cost <TransferWithCosts.cost>`. It can be any function that takes an array and returns a scalar
        value.

    combined_costs_fct_mult_param : value
        references value of the `multiplicative_param <Function_Modulatory_Params>` of `combined_costs_fct
        <TransferWithCosts.combined_costs_fct>`.

    combined_costs_fct_add_param : value
        references value of the `additive_param <Function_Modulatory_Params>` of `combined_costs_fct
        <TransferWithCosts.combined_costs_fct>`.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    name : str
        name of the Function.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        determines the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).
    """

    componentName = TRANSFER_WITH_COSTS_FUNCTION

    classPreferences = {
        PREFERENCE_SET_NAME: 'TransferWithCostssClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    }

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <TransferWithCosts.variable>`

                    :default value: numpy.array([0])
                    :type: ``numpy.ndarray``

                LinearCombination
                    see `LinearCombination <TransferWithCosts.LinearCombination>`

                    :default value: `LinearCombination`
                    :type: `Function`

                SimpleIntegrator
                    see `SimpleIntegrator <TransferWithCosts.SimpleIntegrator>`

                    :default value: `SimpleIntegrator`
                    :type: `Function`

                adjustment_cost
                    see `adjustment_cost <TransferWithCosts.adjustment_cost>`

                    :default value: None
                    :type:

                adjustment_cost_fct
                    see `adjustment_cost_fct <TransferWithCosts.adjustment_cost_fct>`

                    :default value: `Linear`
                    :type: `Function`

                adjustment_cost_fct_add_param
                    see `adjustment_cost_fct_add_param <TransferWithCosts.adjustment_cost_fct_add_param>`

                    :default value: None
                    :type:

                adjustment_cost_fct_mult_param
                    see `adjustment_cost_fct_mult_param <TransferWithCosts.adjustment_cost_fct_mult_param>`

                    :default value: None
                    :type:

                combine_costs_fct
                    see `combine_costs_fct <TransferWithCosts.combine_costs_fct>`

                    :default value: `LinearCombination`
                    :type: `Function`

                combine_costs_fct_add_param
                    see `combine_costs_fct_add_param <TransferWithCosts.combine_costs_fct_add_param>`

                    :default value: None
                    :type:

                combine_costs_fct_mult_param
                    see `combine_costs_fct_mult_param <TransferWithCosts.combine_costs_fct_mult_param>`

                    :default value: None
                    :type:

                combined_costs
                    see `combined_costs <TransferWithCosts.combined_costs>`

                    :default value: None
                    :type:

                duration_cost
                    see `duration_cost <TransferWithCosts.duration_cost>`

                    :default value: None
                    :type:

                duration_cost_fct
                    see `duration_cost_fct <TransferWithCosts.duration_cost_fct>`

                    :default value: `SimpleIntegrator`
                    :type: `Function`

                duration_cost_fct_add_param
                    see `duration_cost_fct_add_param <TransferWithCosts.duration_cost_fct_add_param>`

                    :default value: None
                    :type:

                duration_cost_fct_mult_param
                    see `duration_cost_fct_mult_param <TransferWithCosts.duration_cost_fct_mult_param>`

                    :default value: None
                    :type:

                enabled_cost_functions
                    see `enabled_cost_functions <TransferWithCosts.enabled_cost_functions>`

                    :default value: CostFunctions.INTENSITY
                    :type: `CostFunctions`

                intensity
                    see `intensity <TransferWithCosts.intensity>`

                    :default value: numpy.array([0])
                    :type: ``numpy.ndarray``

                intensity_cost
                    see `intensity_cost <TransferWithCosts.intensity_cost>`

                    :default value: None
                    :type:

                intensity_cost_fct
                    see `intensity_cost_fct <TransferWithCosts.intensity_cost_fct>`

                    :default value: `Exponential`
                    :type: `Function`

                intensity_cost_fct_add_param
                    see `intensity_cost_fct_add_param <TransferWithCosts.intensity_cost_fct_add_param>`

                    :default value: None
                    :type:

                intensity_cost_fct_mult_param
                    see `intensity_cost_fct_mult_param <TransferWithCosts.intensity_cost_fct_mult_param>`

                    :default value: None
                    :type:

                transfer_fct
                    see `transfer_fct <TransferWithCosts.transfer_fct>`

                    :default value: `Linear`
                    :type: `Function`

                transfer_fct_add_param
                    see `transfer_fct_add_param <TransferWithCosts.transfer_fct_add_param>`

                    :default value: None
                    :type:

                transfer_fct_mult_param
                    see `transfer_fct_mult_param <TransferWithCosts.transfer_fct_mult_param>`

                    :default value: None
                    :type:
        """
        variable = Parameter(np.array([0]),
                             history_min_length=1)

        intensity = Parameter(np.zeros_like(variable.default_value),
                              history_min_length=1)

        # Create primary functions' modulation params for TransferWithCosts
        transfer_fct = Parameter(Linear, stateful=False)
        _validate_transfer_fct = get_validator_by_function(is_function_type)
        transfer_fct_mult_param = FunctionParameter(
            aliases=MULTIPLICATIVE_PARAM,
            modulation_combination_function=PRODUCT,
            function_name='transfer_fct',
            function_parameter_name=MULTIPLICATIVE_PARAM,
        )
        transfer_fct_add_param = FunctionParameter(
            aliases=ADDITIVE_PARAM,
            modulation_combination_function=SUM,
            function_name='transfer_fct',
            function_parameter_name=ADDITIVE_PARAM,
        )

        enabled_cost_functions = Parameter(
            CostFunctions.DEFAULTS,
            valid_types=(CostFunctions, list)
        )

        # Create versions of cost functions' modulation params for TransferWithCosts

        intensity_cost = None
        intensity_cost_fct = Parameter(Exponential, stateful=False)
        _validate_intensity_cost_fct = get_validator_by_function(is_function_type)
        intensity_cost_fct_mult_param = FunctionParameter(
            modulation_combination_function=PRODUCT,
            function_name='intensity_cost_fct',
            function_parameter_name=MULTIPLICATIVE_PARAM,
        )
        intensity_cost_fct_add_param = FunctionParameter(
            modulation_combination_function=SUM,
            function_name='intensity_cost_fct',
            function_parameter_name=ADDITIVE_PARAM,
        )

        adjustment_cost = None
        adjustment_cost_fct = Parameter(Linear, stateful=False)
        _validate_adjustment_cost_fct = get_validator_by_function(is_function_type)
        adjustment_cost_fct_mult_param = FunctionParameter(
            modulation_combination_function=PRODUCT,
            function_name='adjustment_cost_fct',
            function_parameter_name=MULTIPLICATIVE_PARAM,
        )
        adjustment_cost_fct_add_param = FunctionParameter(
            modulation_combination_function=SUM,
            function_name='adjustment_cost_fct',
            function_parameter_name=ADDITIVE_PARAM,
        )

        duration_cost = None
        duration_cost_fct = Parameter(SimpleIntegrator, stateful=False)
        _validate_duration_cost_fct = get_validator_by_function(is_function_type)
        duration_cost_fct_mult_param = FunctionParameter(
            modulation_combination_function=PRODUCT,
            function_name='duration_cost_fct',
            function_parameter_name=MULTIPLICATIVE_PARAM,
        )
        duration_cost_fct_add_param = FunctionParameter(
            modulation_combination_function=SUM,
            function_name='duration_cost_fct',
            function_parameter_name=ADDITIVE_PARAM,
        )

        combined_costs = None
        combine_costs_fct = Parameter(LinearCombination, stateful=False)
        _validate_combine_costs_fct = get_validator_by_function(is_function_type)
        combine_costs_fct_mult_param = FunctionParameter(
            modulation_combination_function=PRODUCT,
            function_name='combine_costs_fct',
            function_parameter_name=MULTIPLICATIVE_PARAM,
        )
        combine_costs_fct_add_param = FunctionParameter(
            modulation_combination_function=SUM,
            function_name='combine_costs_fct',
            function_parameter_name=ADDITIVE_PARAM,
        )

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 transfer_fct:tc.optional(is_function_type)=None,
                 enabled_cost_functions:tc.optional(tc.any(CostFunctions, list))=None,
                 intensity_cost_fct:tc.optional(is_function_type)=None,
                 adjustment_cost_fct:tc.optional(is_function_type)=None,
                 duration_cost_fct:tc.optional(is_function_type)=None,
                 combine_costs_fct:tc.optional(is_function_type)=None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

        # if size:
        #     if default_variable is None:
        #         default_variable = np.zeros(size)
        #     elif size != len(default_variable):
        #         raise FunctionError(f"Both {repr(DEFAULT_VARIABLE)} ({default_variable}) and {repr(SIZE)} ({size}) "
        #                             f"are specified for {self.name} but are {SIZE}!=len({DEFAULT_VARIABLE}).")

        super().__init__(
            default_variable=default_variable,
            transfer_fct=transfer_fct,
            enabled_cost_functions=enabled_cost_functions,
            intensity_cost_fct=intensity_cost_fct,
            adjustment_cost_fct=adjustment_cost_fct,
            duration_cost_fct=duration_cost_fct,
            combine_costs_fct=combine_costs_fct,
            params=params,
            owner=owner,
            prefs=prefs,
        )

        # # MODIFIED 6/12/19 NEW: [JDC]
        # self._variable_shape_flexibility = DefaultsFlexibility.FLEXIBLE
        # # MODIFIED 6/12/19 END

    def _instantiate_attributes_before_function(self, function=None, context=None):
        """Instantiate `cost functions <TransferWithCosts_Cost_Functions>` specified in `enabled_cost_functions
        <TransferWithCostss.enabled_cost_functions>`.
        """
        super()._instantiate_attributes_before_function(function=function, context=None)
        self._instantiate_cost_functions(context=context)

    def _instantiate_cost_functions(self, context):
        """Instantiate cost functions and the multiplicative and additive modulatory parameters for them.

        Parse specification of cost functions to enable
        Instantiate cost functions specified in construtor arguments, and enable ones in enabled_cost_functions
        Assign default value for multipicative and additive parameters for each, from the values of those parameters
            on the respective cost functions just instantiated.
        Initialize intensity_cost
        """

        if self.enabled_cost_functions:
            self.assign_costs(self.enabled_cost_functions)

        def instantiate_fct(fct_name, fct):
            if not fct:
                self.toggle_cost(fct_name, OFF)
                return None
            # # MODIFIED 3/10/20 OLD:
            # if isinstance(fct, (Function, types.FunctionType, types.MethodType)):
            # MODIFIED 3/10/20 NEW: [JDC]
            elif isinstance(fct, Function):
                return fct
            elif isinstance(fct, (types.FunctionType, types.MethodType)):
                from psyneulink.core.components.functions.userdefinedfunction import UserDefinedFunction
                return UserDefinedFunction(#default_variable=function_variable,
                        custom_function=fct,
                        owner=self,
                        context=context)
                # MODIFIED 3/10/20 END
            elif issubclass(fct, Function):
                return fct()
            else:
                raise FunctionError(f"{fct} is not a valid cost function for {fct_name}.")

        self.intensity_cost_fct = instantiate_fct(INTENSITY_COST_FUNCTION, self.intensity_cost_fct)
        # Initialize default_value for TransferWithCosts' modulation params from intensity_cost_fct's values
        self.parameters.intensity_cost_fct_mult_param.default_value = \
            self.parameters.intensity_cost_fct_mult_param.get()
        self.parameters.intensity_cost_fct_add_param.default_value = \
            self.parameters.intensity_cost_fct_add_param.get()

        self.adjustment_cost_fct = instantiate_fct(ADJUSTMENT_COST_FUNCTION, self.adjustment_cost_fct)
        # Initialize default_value for TransferWithCosts' modulation params from adjustment_cost_fct's values
        self.parameters.adjustment_cost_fct_mult_param.default_value = \
            self.parameters.adjustment_cost_fct_mult_param.get()
        self.parameters.adjustment_cost_fct_add_param.default_value = \
            self.parameters.adjustment_cost_fct_add_param.get()

        self.duration_cost_fct = instantiate_fct(DURATION_COST_FUNCTION, self.duration_cost_fct)
        # Initialize default_value for TransferWithCosts' modulation params from duration_cost_fct's values
        self.parameters.duration_cost_fct_mult_param.default_value = \
            self.parameters.duration_cost_fct_add_param.get()
        self.parameters.duration_cost_fct_add_param.default_value = \
            self.parameters.duration_cost_fct_add_param.get()

        self.combine_costs_fct = instantiate_fct(COMBINE_COSTS_FUNCTION, self.combine_costs_fct)
        # Initialize default_value for TransferWithCosts' modulation params from combined_costs_fct's values
        self.parameters.combine_costs_fct_mult_param.default_value = \
            self.parameters.combine_costs_fct_mult_param.get()
        self.parameters.combine_costs_fct_add_param.default_value = \
            self.parameters.combine_costs_fct_add_param.get()

        # Initialize intensity attributes
        if self.enabled_cost_functions:
            # Default cost params
            if self.owner:
                if self.owner.context.initialization_status != ContextFlags.DEFERRED_INIT:
                    self.intensity_cost = self.intensity_cost_fct(self.owner.defaults.variable)
                else:
                    self.intensity_cost = self.intensity_cost_fct(self.owner.class_defaults.variable)
            else:
                self.intensity_cost = self.intensity_cost_fct(self.defaults.variable)
                self.defaults.intensity_cost = self.intensity_cost

    def _function(self,
                 variable=None,
                 params=None,
                 context=None):
        """

        Arguments
        ---------

        variable : number or array : default class_defaults.variable
           a single value or array to be transformed.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the function.
            Values specified for parameters in the dictionary override any assigned to those parameters in arguments
            of the constructor.

        Returns
        -------

        transformation of variable using `transfer_fct <TransferWithCostss.transfer_fct>` : number or array

        """

        self._check_args(variable=variable, params=params, context=context)

        # FIRST, DEAL WITH CURRENT INTENSITY

        # Compute current intensity
        intensity = self.parameters.transfer_fct._get(context)(variable, context=context)

        # THEN, DEAL WITH COSTS
        # Note: only compute costs that are enabled;  others are left as None, or with their value when last enabled.

        # Get costs for each cost function that is enabled in enabled_cost_functions
        enabled_cost_functions = self.parameters.enabled_cost_functions._get(context)
        enabled_costs = [] # Used to aggregate costs that are enabled and submit to combine_costs_fct
        if enabled_cost_functions:

            # For each cost function that is enabled:
            # - get params for the cost functon using _get_current_parameter_value:
            #   - if TransferWithControl is owned by a Mechanism, get value from ParameterPort for param
            #   - otherwise, get from TransferWithControl modulation parameter (which is also subject to modulation)

            # Compute intensity_cost
            if enabled_cost_functions & CostFunctions.INTENSITY:
                # Execute intensity_cost function
                intensity_cost = self.intensity_cost_fct(intensity, context=context)
                self.parameters.intensity_cost._set(intensity_cost, context)
                enabled_costs.append(intensity_cost)

            # Compute adjustment_cost
            if enabled_cost_functions & CostFunctions.ADJUSTMENT:
                # Compute intensity change
                try:
                    intensity_change = np.abs(intensity - self.parameters.intensity._get(context))
                except TypeError:
                    intensity_change = np.zeros_like(self.parameters_intensity._get(context))
                # Execute adjustment_cost function
                adjustment_cost = self.adjustment_cost_fct(intensity_change, context=context)
                self.parameters.adjustment_cost._set(adjustment_cost, context)
                enabled_costs.append(adjustment_cost)

            # Compute duration_cost
            if enabled_cost_functions & CostFunctions.DURATION:
                # Execute duration_cost function
                duration_cost = self.duration_cost_fct(intensity, context=context)
                self.parameters.duration_cost._set(duration_cost, context)
                enabled_costs.append(duration_cost)

            # Alwasy execute combined_costs_fct if *any* costs are enabled
            # Execute combine_costs function
            combined_costs = self.combine_costs_fct(enabled_costs,
                                                    context=context)
            self.parameters.combined_costs._set(combined_costs, context)

        # Store current intensity
        self.parameters.intensity._set(intensity, context)

        return intensity

    def _is_identity(self, context=None):
        return (self.parameters.transfer_fct.get()._is_identity(context) and
                self.parameters.enabled_cost_functions.get(context) == CostFunctions.NONE)

    @tc.typecheck
    def assign_costs(self, cost_functions: tc.any(CostFunctions, list), execution_context=None):
        """Assigns specified functions; all others are disabled.

        Arguments
        ---------
        cost_functions: CostFunctions or List[CostFunctions]
            `cost function <TransferWithCosts_Cost_Functions>` or list of ones to be used;  all other will be disabled.
        Returns
        -------
        enabled_cost_functions :  boolean combination of CostFunctions
            current value of `enabled_cost_functions <TransferWithCosts.enabled_cost_functions>`.

        """
        if isinstance(cost_functions, CostFunctions):
            cost_functions = [cost_functions]
        self.parameters.enabled_cost_functions.set(CostFunctions.NONE, execution_context)
        return self.enable_costs(cost_functions, execution_context)

    @tc.typecheck
    def enable_costs(self, cost_functions: tc.any(CostFunctions, list), execution_context=None):
        """Enable specified `cost functions <TransferWithCosts_Cost_Functions>`;
        settings for all other cost functions are left intact.

        Arguments
        ---------
        cost_functions: CostFunctions or List[CostFunctions]
            `cost function <TransferWithCosts_Cost_Functions>` or list of ones to be enabled,
            in addition to any that are already enabled.
        Returns
        -------
        enabled_cost_functions :  boolean combination of CostFunctions
            current value of `enabled_cost_functions <TransferWithCosts.enabled_cost_functions>`.
        """
        if isinstance(cost_functions, CostFunctions):
            cost_functions = [cost_functions]
        enabled_cost_functions = self.parameters.enabled_cost_functions.get(execution_context)
        for cost_function in cost_functions:
            enabled_cost_functions |= cost_function

        self.parameters.enabled_cost_functions.set(enabled_cost_functions, execution_context)
        return enabled_cost_functions

    @tc.typecheck
    def disable_costs(self, cost_functions: tc.any(CostFunctions, list), execution_context=None):
        """Disable specified `cost functions <TransferWithCosts_Cost_Functions>`;
        settings for all other cost functions are left intact.

        Arguments
        ---------
        cost_functions: CostFunction or List[CostFunctions]
            `cost function <TransferWithCosts_Cost_Functions>` or list of ones to be disabled.
        Returns
        -------
        enabled_cost_functions :  boolean combination of CostFunctions
            current value of `enabled_cost_functions <TransferWithCosts.enabled_cost_functions>`.
        """
        if isinstance(cost_functions, CostFunctions):
            cost_functions = [cost_functions]
        enabled_cost_functions = self.parameters.enabled_cost_functions.get(execution_context)
        for cost_function in cost_functions:
            enabled_cost_functions &= ~cost_function

        self.parameters.enabled_cost_functions.set(enabled_cost_functions, execution_context)
        return enabled_cost_functions

    def toggle_cost(self, cost_function_name:tc.any(str, CostFunctions),
                             assignment:bool=ON,
                             execution_context=None):
        """Enable/disable a `cost functions <TransferWithCosts_Cost_Functions>`.

        Arguments
        ---------
        cost_function_name : str or CostFunction
            Must be the name of a `cost function <TransferWithCosts_Cost_Functions>` or a value of CostFunction enum.

        Returns
        -------
        enabled_cost_functions :  boolean combination of CostFunctions
            current value of `enabled_cost_functions <TransferWithCosts.enabled_cost_functions>`.

        """
        if cost_function_name in {INTENSITY_COST_FUNCTION, CostFunctions.INTENSITY}:
            cost_function = CostFunctions.INTENSITY
            cost_function_name = INTENSITY_COST_FUNCTION
        elif cost_function_name in {ADJUSTMENT_COST_FUNCTION, CostFunctions.ADJUSTMENT}:
            cost_function = CostFunctions.ADJUSTMENT
            cost_function_name = ADJUSTMENT_COST_FUNCTION
        elif cost_function_name in {DURATION_COST_FUNCTION, CostFunctions.DURATION}:
            cost_function = CostFunctions.DURATION
            cost_function_name = DURATION_COST_FUNCTION
        elif cost_function_name == COMBINE_COSTS_FUNCTION:
            raise FunctionError("{} cannot be disabled".format(COMBINE_COSTS_FUNCTION))
        else:
            raise FunctionError("toggle_cost: unrecognized cost function: {}".format(cost_function_name))

        enabled_cost_functions = self.parameters.enabled_cost_functions.get(execution_context)
        if assignment:
            if cost_function_name not in self.parameters.names():
                raise FunctionError("Unable to toggle {} ON as function assignment is \'None\'".
                                         format(cost_function_name))
            if not enabled_cost_functions:
                enabled_cost_functions = cost_function
            else:
                enabled_cost_functions |= cost_function
        else:
            enabled_cost_functions &= ~cost_function

        self.parameters.enabled_cost_functions.set(enabled_cost_functions, execution_context)
        return enabled_cost_functions

    def _gen_llvm_function_body(self, ctx, builder, params, state, arg_in, arg_out, *, tags:frozenset):
        # Run transfer function first
        transfer_f = self.parameters.transfer_fct
        trans_f = ctx.import_llvm_function(transfer_f.get())
        trans_p = pnlvm.helpers.get_param_ptr(builder, self, params, transfer_f.name)
        trans_s = pnlvm.helpers.get_state_ptr(builder, self, state, transfer_f.name)
        trans_in = arg_in
        if trans_in.type != trans_f.args[2].type:
            warnings.warn("Shape mismatch: {} input does not match the transfer function ({}): {} vs. {}".format(self, transfer_f.get(), self.defaults.variable, transfer_f.get().defaults.variable))
            trans_in = builder.gep(trans_in, [ctx.int32_ty(0), ctx.int32_ty(0)])
        trans_out = arg_out
        if trans_out.type != trans_f.args[3].type:
            warnings.warn("Shape mismatch: {} output does not match the transfer function ({}): {} vs. {}".format(self, transfer_f.get(), self.defaults.value, transfer_f.get().defaults.value))
            trans_out = builder.gep(trans_out, [ctx.int32_ty(0), ctx.int32_ty(0)])
        builder.call(trans_f, [trans_p, trans_s, trans_in, trans_out])

        # TODO: Implement cost calculations
        return builder
