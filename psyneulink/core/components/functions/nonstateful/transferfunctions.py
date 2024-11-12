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
* `Angle`
* `Gaussian`
* `GaussianDistort`
* `BinomialDistort`
* `Dropout`
* `SoftMax`
* `TransferWithCosts`

Overview
--------

Functions that transform their variable but maintain its shape.

.. _TransferFunction_StandardAttributes:

Standard Attributes
~~~~~~~~~~~~~~~~~~~

All TransferFunctions have the following attributes:

* **bounds**:  specifies the lower and upper limits of the result;  if there are none, the attribute is set to
  `None`;  if it has at least one bound, the attribute is set to a tuple specifying the lower and upper bounds,
  respectively, with `None` as the entry for no bound.
..
* **multiplicative_param** and **additive_param**:
  each of these is assigned the name of one of the function's
  parameters and used by `ModulatoryProjections <ModulatoryProjection>` to modulate the output of the
  TransferFunction's function (see `Function_Modulatory_Params`).

.. _TransferFunction_Derivative:

Derivatives
~~~~~~~~~~~

Most TransferFunctions have a derivative method.  These take both an **input** and **output** argument.  In general,
the **input** is used to compute the derivative of the function at that value. If that is not provided, some
Functions can compute the derivative using the function's output, either directly (such as `Logistic.derivative`) or by
inferring the input from the **output** and then computing the derivative for that value (such as `ReLU.derivative`)


TranferFunction Class References
--------------------------------


"""

import numbers
import types
import warnings
from enum import Flag, auto
from math import e, pi, sqrt

import numpy as np
try:
    import torch
except ImportError:
    torch = None
from beartype import beartype

from psyneulink._typing import Callable, Mapping, Optional, Union

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.component import parameter_keywords
from psyneulink.core.components.functions.function import (
    DEFAULT_SEED, Function, Function_Base, FunctionError, _random_state_getter, _seed_setter, function_keywords,
    get_matrix, is_function_type,
)
from psyneulink.core.components.functions.nonstateful.transformfunctions import LinearCombination
from psyneulink.core.components.functions.nonstateful.selectionfunctions import OneHot, ARG_MAX, ARG_MAX_INDICATOR
from psyneulink.core.components.functions.stateful.integratorfunctions import SimpleIntegrator
from psyneulink.core.components.shellclasses import Projection
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.utilities import is_numeric_scalar
from psyneulink.core.globals.keywords import \
    (ADAPTIVE, ADDITIVE_PARAM, ALL, ANGLE_FUNCTION, BIAS, BINOMIAL_DISTORT_FUNCTION, DROPOUT_FUNCTION,
     EXPONENTIAL_FUNCTION, GAIN, GAUSSIAN_DISTORT_FUNCTION, GAUSSIAN_FUNCTION,
     IDENTITY_FUNCTION, INTERCEPT, LEAK, LINEAR_FUNCTION, LOGISTIC_FUNCTION,
     TANH_FUNCTION, MAX_INDICATOR, MAX_VAL, MULTIPLICATIVE_PARAM,
     OFF, OFFSET, ON, OUTPUT_TYPE, PER_ITEM, PROB, PRODUCT, PROB_INDICATOR,
     RATE, RELU_FUNCTION, SCALE, SLOPE, SOFTMAX_FUNCTION, STANDARD_DEVIATION, SUM,
     TRANSFER_FUNCTION_TYPE, TRANSFER_WITH_COSTS_FUNCTION, VARIANCE, VARIABLE, X_0, PREFERENCE_SET_NAME)
from psyneulink.core.globals.parameters import \
    FunctionParameter, Parameter, get_validator_by_function, check_user_specified, copy_parameter_value
from psyneulink.core.globals.preferences.basepreferenceset import \
    REPORT_OUTPUT_PREF, PreferenceEntry, PreferenceLevel, ValidPrefSet
from psyneulink.core.globals.utilities import ValidParamSpecType, convert_all_elements_to_np_array, safe_len, is_matrix_keyword

__all__ = ['Angle', 'BinomialDistort', 'Dropout', 'Exponential', 'Gaussian', 'GaussianDistort', 'Identity',
           'Linear', 'Logistic', 'ReLU', 'SoftMax', 'Tanh', 'TransferFunction', 'TransferWithCosts'
           ]

class TransferFunction(Function_Base):
    """Function that transforms variable but maintains its shape.

    All TransferFunctions MUST have the following attributes:

    `bounds` -- specifies the lower and upper limits of the result;  if there are none, the attribute is set to
    `None`;  if it has at least one bound, the attribute is set to a tuple specifying the lower and upper bounds,
    respectively, with `None` as the entry for no bound.

    `multiplicative_param <Function_Modulatory_Params>` and `additive_param <Function_Modulatory_Params>` -- each
    of these is assigned the name of one of the function's parameters and used by `ModulatoryProjections
    <ModulatoryProjection>` to modulate the output of the TransferFunction's `function <TransferFunction._function>`
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

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):
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

    def _gen_pytorch_fct(self, device, context=None):
        return lambda x: x


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

    `function <Linear._function>` returns linear transform of `variable <Linear.variable>`:

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

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 slope: Optional[ValidParamSpecType] = None,
                 intercept: Optional[ValidParamSpecType] = None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):

        super().__init__(
            default_variable=default_variable,
            slope=slope,
            intercept=intercept,
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

        linear transformation of variable : number or array

        """
        slope = self._get_current_parameter_value(SLOPE, context)
        intercept = self._get_current_parameter_value(INTERCEPT, context)

        try:
            # By default, result should be returned as np.ndarray with same dimensionality as input
            result = variable * slope + intercept
        except TypeError:
            if hasattr(variable, "dtype"):
                # If variable is an array with mixed sizes or types, try item-by-item operation
                if variable.dtype == object:
                    result = np.zeros_like(variable)
                    for i, item in enumerate(variable):
                        try:
                            result[i] = variable[i] * slope + intercept
                        except TypeError:
                            owner_str = f" of '{self.owner.name}'" if self.owner else ""
                            if variable[i] is None:
                                err_msg = (f"Item {i} of {VARIABLE} passed to {self.name}{owner_str} is 'None'; "
                                           f"may be due to missing afferent projection to input_ports[{i}]")
                            else:
                                err_msg = (f"Unrecognized type for item {i} of {VARIABLE} (variable[i]) "
                                           f"passed to {self.name}{owner_str}.")
                            raise FunctionError(err_msg)
                else:
                    owner_str = f"'{self.owner.name}'" if self.owner else ""
                    raise FunctionError(f"Unrecognized type for {VARIABLE} ({variable}) "
                                        f"passed to {self.name}{owner_str}.")
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

        Derivative of `function <Linear._function>` at **input**.

        Arguments
        ---------

        input : number
            value of the input to the Linear transform at which derivative is to be taken.

        Returns
        -------

        Slope of function :  number or array

        """
        return self._get_current_parameter_value(SLOPE, context)

    def _is_identity(self, context=None, defaults=False):
        if defaults:
            slope = self.defaults.slope
            intercept = self.defaults.intercept
        else:
            slope = self.parameters.slope._get(context)
            intercept = self.parameters.intercept._get(context)

        return slope == 1 and intercept == 0

    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])
        slope_ptr = ctx.get_param_or_state_ptr(builder, self, SLOPE, param_struct_ptr=params)
        intercept_ptr = ctx.get_param_or_state_ptr(builder, self, INTERCEPT, param_struct_ptr=params)

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

    def _gen_pytorch_fct(self, device, context=None):
        slope = self._get_pytorch_fct_param_value('slope', device, context)
        intercept = self._get_pytorch_fct_param_value('intercept', device, context)
        return lambda x: x * slope + intercept


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

    `function <Exponential._function>` returns exponential transform of `variable <Exponential.variable>`:

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

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 rate: Optional[ValidParamSpecType] = None,
                 scale: Optional[ValidParamSpecType] = None,
                 bias: Optional[ValidParamSpecType] = None,
                 offset: Optional[ValidParamSpecType] = None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):
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

        Derivative of `function <Exponential._function>` at **input**.

        Returns
        -------
        derivative :  number or array
        """

        rate = self._get_current_parameter_value(RATE, context)
        scale = self._get_current_parameter_value(SCALE, context)
        bias = self._get_current_parameter_value(BIAS, context)

        return rate * scale * e**(rate * input + bias)

    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        rate_ptr = ctx.get_param_or_state_ptr(builder, self, RATE, param_struct_ptr=params)
        bias_ptr = ctx.get_param_or_state_ptr(builder, self, BIAS, param_struct_ptr=params)
        scale_ptr = ctx.get_param_or_state_ptr(builder, self, SCALE, param_struct_ptr=params)
        offset_ptr = ctx.get_param_or_state_ptr(builder, self, OFFSET, param_struct_ptr=params)

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

    def _gen_pytorch_fct(self, device, context=None):
        rate = self._get_pytorch_fct_param_value('rate', device, context)
        scale = self._get_pytorch_fct_param_value('scale', device, context)
        bias = self._get_pytorch_fct_param_value('bias', device, context)

        return rate * scale * torch.exp(rate * input + bias)


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

    `function <Logistic._function>` returns logistic transform of `variable <Logistic.variable>`:

    .. math::
         scale * \\frac{1}{1 + e^{ - gain ( variable + bias - x_{0} ) + offset}}

    (this is a vertically offset and scaled version of `Tanh`, which is centered on origin).

    .. _Logistic_Note:

    .. note::
        The **bias** and **x_0** arguments are identical, apart from having opposite signs: **bias** is included to
        accommodate the convention in the machine learning community; **x_0** is included to match the `standard
        form of the Logistic Function <https://en.wikipedia.org/wiki/Logistic_function>`_ (in which **gain**
        corresponds to the *k* parameter and **scale** corresponds to the *L* parameter); **offset** implements a
        form of bias that is not modulated by gain (i.e., it produces an offset of the function along the horizontal
        axis).

    `derivative <Logistic.derivative>` returns the derivative of the Logistic using its **output**:

    .. math::
        gain * scale * output * (1-output)

    Arguments
    ---------

    default_variable : number or array : default class_defaults.variable
        specifies a template for the value to be transformed.

    gain : float : default 1.0
        specifies value by which to multiply each element of `variable <Logistic.variable>` after it is adjusted by
        `bias <Logistic.bias>` and/or `x_0 <Logistic.x_0>`, but before adjustment by `offset <Logistic.offset>` and
        logistic transformation (see `note <Logistic_Note>` above).

    bias : float : default 0.0
        specifies value to add to each element of `variable <Logistic.variable>` before applying `gain <Logistic.gain>`;
        this argument has an effect identical to x_0, but with the opposite sign (see `note <Logistic_Note>` above).

    x_0 : float : default 0.0
        specifies value to add to each element of `variable <Logistic.variable>` before applying `gain <Logistic.gain>`;
        this argument has an effect identical to bias, but with the opposite sign (see `note <Logistic_Note>` above).

    offset : float : default 0.0
        specifies value to add to each element of `variable <Logistic.variable>` after adjusting by `bias
        <Logistic.bias>` and/or `x_0 <Logistic.x_0>` and applying `gain <Logistic.gain>`, but before logistic
        transformation (see `note <Logistic_Note>` above).

    scale : float : default 0.0
        specifies value by which to multiply each element of `variable <Logistic.variable>` after all other parameters
        and logistic transformation have been applied (see `note <Logistic_Note>` above).

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

    gain : float
        value by which to multiply each element of `variable <Logistic.variable>` after it is adjusted by `bias
        <Logistic.bias>` and/or `x_0 <Logistic.x_0>`, but before adjustment by `offset <Logistic.offset>` and
        logistic transformation (see `note <Logistic_Note>` above).

    bias : float
        value to add to each element of `variable <Logistic.variable>` before applying `gain <Logistic.gain>`;
        this argument has an effect identical to x_0, but with the opposite sign (see `note <Logistic_Note>` above).

    x_0 : float
        value to add to each element of `variable <Logistic.variable>` before applying `gain <Logistic.gain>`;
        this argument has an effect identical to bias, but with the opposite sign (see `note <Logistic_Note>` above).

    offset : float
        value to add to each element of `variable <Logistic.variable>` after adjusting by `bias <Logistic.bias>`
        and/or `x_0 <Logistic.x_0>` and applying `gain <Logistic.gain>`, but before logistic transformation
        (see `note <Logistic_Note>` above).

    scale : float
        value by which to multiply each element of `variable <Logistic.variable>` after all other parameters and
        logistic transformation have been applied (see `note <Logistic_Note>` above).

    bounds : (0,1)
        COMMENT:
        the lower and upper limits of the result which, in the case of the `Logistic`, is determined by the function
        itself.
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

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 gain: Optional[ValidParamSpecType] = None,
                 x_0=None,
                 bias=None,
                 offset: Optional[ValidParamSpecType] = None,
                 scale: Optional[ValidParamSpecType] = None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):
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

        result = scale * (1. / (1 + e**(-gain * (variable + bias - x_0) + offset)))

        return self.convert_output_type(result)

    @handle_external_context()
    def derivative(self, input=None, output=None, context=None):
        """
        derivative(input=None, output=None)

        Derivative of `function <Exponential._function>` at either **input** or **output**.

        COMMENT:  RESTORE WHEN TEST IN DERIVATIVE IS RESTORED
        Either **input** or **output** must be specified.
        If **output** is not specified, it is computed from  **input**.
        If both are specified, **input** is ignored unless paramValidationPref is set, in which case
        an error is generated if **output** does not correspond to `function <Logistic._function>`\\(**input**).
        COMMENT
        Either **input** or **output** must be specified.
        If **output** is not specified, derivative is computed from **input**.
        If both are specified, **input** is ignored and derivative is computed from **output**
        .. technical_note::
           allowing both to be specified is supported for consistency with `BackPropagation` `LearningFunction`
           which uses output to compute Logistic

        Arguments
        ---------

        input : number
            value of the input to the Logistic transform at which derivative is to be taken.

        output : number
            value of the output of the Logistic transform at which derivative is to be taken.

        Returns
        -------
        derivative  of logistic transform at output :  number or array
        """

        gain = self._get_current_parameter_value(GAIN, context)
        scale = self._get_current_parameter_value(SCALE, context)

        # Favor use of output: compute it from input if it is not provided
        if output is None:
            output = self.function(input, context=context)

        return gain * scale * output * (1 - output)

    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        gain_ptr = ctx.get_param_or_state_ptr(builder, self, GAIN, param_struct_ptr=params)
        bias_ptr = ctx.get_param_or_state_ptr(builder, self, BIAS, param_struct_ptr=params)
        x_0_ptr = ctx.get_param_or_state_ptr(builder, self, X_0, param_struct_ptr=params)
        scale_ptr = ctx.get_param_or_state_ptr(builder, self, SCALE, param_struct_ptr=params)
        offset_ptr = ctx.get_param_or_state_ptr(builder, self, OFFSET, param_struct_ptr=params)

        gain = pnlvm.helpers.load_extract_scalar_array_one(builder, gain_ptr)
        bias = pnlvm.helpers.load_extract_scalar_array_one(builder, bias_ptr)
        x_0 = pnlvm.helpers.load_extract_scalar_array_one(builder, x_0_ptr)
        offset = pnlvm.helpers.load_extract_scalar_array_one(builder, offset_ptr)
        scale = pnlvm.helpers.load_extract_scalar_array_one(builder, scale_ptr)

        exp_f = ctx.get_builtin("exp", [ctx.float_ty])
        val = builder.load(ptri)

        if "derivative_out" not in tags:
            val = builder.fadd(val, bias)
            val = builder.fsub(val, x_0)
            val = builder.fmul(val, gain)
            val = builder.fsub(offset, val)
            val = builder.call(exp_f, [val])
            val = builder.fadd(ctx.float_ty(1), val)
            val = builder.fdiv(ctx.float_ty(1), val)
            val = builder.fmul(val, scale)

        if "derivative" in tags or "derivative_out" in tags:
            # f(x) = g * s * o * (1-o)
            function_val = val
            val = builder.fsub(ctx.float_ty(1), function_val)
            val = builder.fmul(function_val, val)
            val = builder.fmul(gain, val)
            val = builder.fmul(scale, val)

        builder.store(val, ptro)

    def _gen_pytorch_fct(self, device, context=None):
        gain = self._get_pytorch_fct_param_value('gain', device, context)
        bias = self._get_pytorch_fct_param_value('bias', device, context)
        offset = self._get_pytorch_fct_param_value('offset', device, context)
        return lambda x: 1 / (1 + torch.exp(-gain * (x + bias) + offset))

    def as_mdf_model(self):
        model = super().as_mdf_model()

        # x_0 is included in bias in MDF logistic
        self._set_mdf_arg(model, 'bias', np.array(model.args['bias'] - model.args['x_0']))
        self._set_mdf_arg(model, 'x_0', np.array(0))

        if model.args['scale'] != 1.0:
            warnings.warn(
                f"Scale (set to {model.args['scale']} is not a supported"
                ' parameter for MDF logistic'
            )
        return model


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

    `function <Logistic._function>` returns hyperbolic tangent of `variable <Logistic.variable>`:

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

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 gain: Optional[ValidParamSpecType] = None,
                 x_0=None,
                 bias=None,
                 offset: Optional[ValidParamSpecType] = None,
                 scale: Optional[ValidParamSpecType] = None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):
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

        exponent = -2 * (gain * (variable + bias - x_0) + offset)
        result = scale * (1 - e**exponent)/ (1 + e**exponent)

        return self.convert_output_type(result)


    @handle_external_context()
    def derivative(self, input, output=None, context=None):
        """
        derivative(input)

        Derivative of `function <Tanh._function>` at **input**.

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

    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        gain_ptr = ctx.get_param_or_state_ptr(builder, self, GAIN, param_struct_ptr=params)
        bias_ptr = ctx.get_param_or_state_ptr(builder, self, BIAS, param_struct_ptr=params)
        x_0_ptr = ctx.get_param_or_state_ptr(builder, self, X_0, param_struct_ptr=params)
        offset_ptr = ctx.get_param_or_state_ptr(builder, self, OFFSET, param_struct_ptr=params)
        scale_ptr = ctx.get_param_or_state_ptr(builder, self, SCALE, param_struct_ptr=params)

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

    def _gen_pytorch_fct(self, device, context=None):
        gain = self._get_pytorch_fct_param_value('gain', device, context)
        bias = self._get_pytorch_fct_param_value('bias', device, context)
        offset = self._get_pytorch_fct_param_value('offset', device, context)
        # return lambda x: 1 / (1 + torch.exp(-gain * (x + bias) + offset))
        return lambda x: ((torch.exp(-gain * (x + bias) + offset) - torch.exp(-gain * (-x + bias) + offset))
                          / (torch.exp(-gain * (x + bias) + offset) + torch.exp(-gain * (-x + bias) + offset)))

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

    .. _RelU_Function:

    `function <ReLU._function>` returns rectified linear tranform of `variable <ReLU.variable>`:

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
        specifies a value to subtract from each element of `variable <ReLU.variable>`; functions as threshold.
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
        value to subtract from each element of `variable <ReLU.variable>`; functions as threshold.

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

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 gain: Optional[ValidParamSpecType] = None,
                 bias: Optional[ValidParamSpecType] = None,
                 leak: Optional[ValidParamSpecType] = None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):
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

    @handle_external_context()
    def derivative(self, input=None, output=None, context=None):
        """
        derivative(input or else output)

        Derivative of `function <ReLU._function>` at **input** or **output**.  If **input** is specified, that
        is used to compute the derivative;  if **input** is not specified, it is inferred from the **output**
        and then used to compute the derivative.

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
        bias = self._get_current_parameter_value(BIAS, context)

        if input is not None:
            # Use input if provided
            variable = np.array(input) - bias
        else:
            # Infer input from output
            variable = np.array(output) / gain

        value = np.where(variable > 0, gain, gain * leak)
        return value

    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        gain_ptr = ctx.get_param_or_state_ptr(builder, self, GAIN, param_struct_ptr=params)
        bias_ptr = ctx.get_param_or_state_ptr(builder, self, BIAS, param_struct_ptr=params)
        leak_ptr = ctx.get_param_or_state_ptr(builder, self, LEAK, param_struct_ptr=params)

        gain = pnlvm.helpers.load_extract_scalar_array_one(builder, gain_ptr)
        bias = pnlvm.helpers.load_extract_scalar_array_one(builder, bias_ptr)
        leak = pnlvm.helpers.load_extract_scalar_array_one(builder, leak_ptr)

        # Maxnum for some reason needs full function prototype
        max_f = ctx.get_builtin("maxnum", [ctx.float_ty])
        var = builder.load(ptri)
        if "derivative_out" in tags:
            val = builder.fdiv(var, gain)
        else:
            val = builder.fsub(var, bias)

        if "derivative" in tags or "derivative_out" in tags:
            predicate = builder.fcmp_ordered('>', val, val.type(0))
            val = builder.select(predicate, gain, builder.fmul(gain, leak))
        else:
            val1 = builder.fmul(val, gain)
            val2 = builder.fmul(val1, leak)

            val = builder.call(max_f, [val1, val2])

        builder.store(val, ptro)

    def _gen_pytorch_fct(self, device, context=None):
        gain = self._get_pytorch_fct_param_value('gain', device, context)
        bias = self._get_pytorch_fct_param_value('bias', device, context)
        leak = self._get_pytorch_fct_param_value('leak', device, context)
        return lambda x: (torch.max(input=(x - bias), other=torch.tensor([0], device=device).double()) * gain +
                            torch.min(input=(x - bias), other=torch.tensor([0], device=device).double()) * leak)


# **********************************************************************************************************************
#                                                    Angle
# **********************************************************************************************************************

# FIX: VALIDATE LEN(VARIABLE)>=2

class Angle(TransferFunction):  # -------------------------------------------------------------------------------------
    """
    Angle(                 \
         default_variable, \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _Angle_Function:

    `function <angle._function>` returns Angle transform of vector in `variable <Angle.variable>`:

    COMMENT:
    FIX: WITH PROPER MATHEMATICAL DEFN
    .. math::

        slope * variable + intercept

    `derivative <Angle.derivative>` returns `slope <Angle.slope>`.
    COMMENT

    Arguments
    ---------

    default_variable : 1array : default class_defaults.variable
        specifies a template for the value to be transformed;  length must be at least 2.

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

    componentName = ANGLE_FUNCTION

    classPreferences = {
        PREFERENCE_SET_NAME: 'AngleClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    }

    _model_spec_class_name_is_generic = True

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <Angle.variable>`

                    :default value: numpy.array([0.,0,])
                    :type: ``numpy.ndarray``
                    :read only: True

        """
        variable = Parameter(np.array([1,1]),
                             read_only=True,
                             pnl_internal=True,
                             constructor_argument='default_variable')

        def _validate_variable(self, variable):
            variable = np.squeeze(variable)
            if variable.ndim != 1 or len(variable) < 2:
                return f"must be list or 1d array of length 2 or greater."

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):

        super().__init__(
            default_variable=default_variable,
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

        variable : ndarray : default class_defaults.variable
           an array of coordinates on a sphere to be transformed to n+1d angular coordinates;  must be at least 2d.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        Angle transformation of variable : ndarray of variable.ndim+1

        """
        try:
            # By default, result should be returned as np.ndarray with same dimensionality as input
            result = self._angle(variable)
        except TypeError:
            if hasattr(variable, "dtype"):
                # If variable is an array with mixed sizes or types, try item-by-item operation
                if variable.dtype == object:
                    result = np.zeros_like(variable)
                    for i, item in enumerate(variable):
                        result[i] = self._angle(variable[i])
                else:
                    raise FunctionError("Unrecognized type for {} of {} ({})".format(VARIABLE, self.name, variable))
            # KAM 6/28/18: If the variable does not have a "dtype" attr but made it to this line, then it must be of a
            # type that even np does not recognize -- typically a custom OutputPort variable with items of different
            # shapes (e.g. variable = [[0.0], [0.0], array([[0.0, 0.0]])] )
            elif isinstance(variable, list):
                result = []
                for variable_item in variable:
                    result.append(self._angle(variable_item))
            else:
                raise FunctionError("Unrecognized type for {} of {} ({})".format(VARIABLE, self.name, variable))

        return self.convert_output_type(result)

    def _angle(self, value):
        """Take nd value and return n+1d coordinates for angle on a sphere"""
        value = np.squeeze(value)
        dim = len(value) + 1
        angle = np.zeros(dim)
        sin_value = np.sin(value)
        cos_value = np.cos(value)
        angle[0] = cos_value[0]
        prod_a = np.cumprod(np.flip(sin_value))[:-1]
        angle[dim - 1] = prod_a[-1]
        prod_a[-1] = 1.

        # going down from the top of cumprod we skip: 2 edge values +1 extra for output size
        for j in range(1, dim - 1):
            angle[j] = prod_a[dim -3 -j] * cos_value[j]
        return angle

    def _gen_llvm_function_body(self, ctx, builder, params, state, arg_in, arg_out, *, tags:frozenset):
        assert isinstance(arg_in.type.pointee, pnlvm.ir.ArrayType)
        assert isinstance(arg_out.type.pointee, pnlvm.ir.ArrayType)
        assert len(arg_in.type.pointee) + 1 == len(arg_out.type.pointee)

        # The first cos
        res0_ptr = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(0)])
        val0_ptr = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(0)])
        val0 = builder.load(val0_ptr)
        cos_f = ctx.get_builtin("cos", [val0.type])
        cos_val0 = builder.call(cos_f, [val0])
        builder.store(cos_val0, res0_ptr)

        # calculate suffix product
        sin_f = ctx.get_builtin("sin", [val0.type])
        prod_ptr = builder.alloca(val0.type)
        builder.store(prod_ptr.type.pointee(1.0), prod_ptr)

        dim_m1 = ctx.int32_ty(len(arg_out.type.pointee) - 1)
        with pnlvm.helpers.for_loop(builder, dim_m1.type(1), dim_m1, dim_m1.type(1), id="suff_prod") as (b, idx):
            #revert the index to go from the end
            idx = b.sub(dim_m1, idx)

            prod = b.load(prod_ptr)
            val_ptr = b.gep(arg_in, [ctx.int32_ty(0), idx])
            val = b.load(val_ptr)

            # calculate suffix product of sin(input)
            val_sin = b.call(sin_f, [val])
            new_prod = b.fmul(prod, val_sin)
            b.store(new_prod, prod_ptr)

            # output value is suffix product * cos(val)
            val_cos = b.call(cos_f, [val])
            res = b.fmul(prod, val_cos)
            res_ptr = b.gep(arg_out, [ctx.int32_ty(0), idx])
            b.store(res, res_ptr)

        # The last element is just the suffix product * 1
        last_ptr = builder.gep(arg_out, [ctx.int32_ty(0), dim_m1])
        builder.store(builder.load(prod_ptr), last_ptr)

        return builder

    # @handle_external_context()
    # def derivative(self, input=None, output=None, context=None):
    #     """
    #     derivative(input)
    #
    #     Derivative of `function <Angle._function>` at **input**.
    #
    #     Arguments
    #     ---------
    #
    #     input : number
    #         value of the input to the Angle transform at which derivative is to be taken.
    #
    #     Returns
    #     -------
    #
    #     Slope of function :  number or array
    #
    #     """
    #
    #     return self._get_current_parameter_value(SLOPE, context)
    #
    # def _is_identity(self, context=None):
    #     return (
    #         self.parameters.slope._get(context) == 1
    #         and self.parameters.intercept._get(context) == 0
    #     )


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

    `function <Gaussian._function>` returns Gaussian transform of `variable <Gaussian.variable>`:

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

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 standard_deviation: Optional[ValidParamSpecType] = None,
                 bias: Optional[ValidParamSpecType] = None,
                 scale: Optional[ValidParamSpecType] = None,
                 offset: Optional[ValidParamSpecType] = None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):
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

        standard_deviation_ptr = ctx.get_param_or_state_ptr(builder, self, STANDARD_DEVIATION, param_struct_ptr=params)
        bias_ptr = ctx.get_param_or_state_ptr(builder, self, BIAS, param_struct_ptr=params)
        scale_ptr = ctx.get_param_or_state_ptr(builder, self, SCALE, param_struct_ptr=params)
        offset_ptr = ctx.get_param_or_state_ptr(builder, self, OFFSET, param_struct_ptr=params)

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
           a single value or array to be distorted by Guassian distribution.

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

        Derivative of `function <Gaussian._function>` at **input**.


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

    `function <GaussianDistort._function>` returns random value from a Gaussian distribution with
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
        random_state = Parameter(None, loggable=False, getter=_random_state_getter, dependencies='seed')
        seed = Parameter(DEFAULT_SEED(), modulable=True, fallback_default=True, setter=_seed_setter)
        bounds = (None, None)

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 variance: Optional[ValidParamSpecType] = None,
                 bias: Optional[ValidParamSpecType] = None,
                 scale: Optional[ValidParamSpecType] = None,
                 offset: Optional[ValidParamSpecType] = None,
                 seed=None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):

        super().__init__(
            default_variable=default_variable,
            variance=variance,
            bias=bias,
            scale=scale,
            offset=offset,
            seed=seed,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        variance_ptr = ctx.get_param_or_state_ptr(builder, self, VARIANCE, param_struct_ptr=params)
        bias_ptr = ctx.get_param_or_state_ptr(builder, self, BIAS, param_struct_ptr=params)
        scale_ptr = ctx.get_param_or_state_ptr(builder, self, SCALE, param_struct_ptr=params)
        offset_ptr = ctx.get_param_or_state_ptr(builder, self, OFFSET, param_struct_ptr=params)

        variance = pnlvm.helpers.load_extract_scalar_array_one(builder, variance_ptr)
        bias = pnlvm.helpers.load_extract_scalar_array_one(builder, bias_ptr)
        scale = pnlvm.helpers.load_extract_scalar_array_one(builder, scale_ptr)
        offset = pnlvm.helpers.load_extract_scalar_array_one(builder, offset_ptr)

        rvalp = builder.alloca(ptri.type.pointee, name="random_out")
        rand_state_ptr = ctx.get_random_state_ptr(builder, self, state, params)
        normal_f = ctx.get_normal_dist_function_by_state(rand_state_ptr)
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
#                                               BinomialDistort
# **********************************************************************************************************************

class BinomialDistort(TransferFunction):  #-----------------------------------------------------------------------------
    """
    BinomialDistort(          \
         default_variable,    \
         p=0.05,              \
         seed=None,           \
         params=None,         \
         owner=None,          \
         name=None,           \
         prefs=None           \
         )

    .. _BinomialDistort:

    `function <BinomialDistort._function>` returns `variable <BinomialDistort.variable>` with elements randomly
    zeroed with probability **p**:

    .. math::

       if \\ \\ rand[0,1] > p: output_i=0 \\\\
       else: \\ output_i = variable_i

    `derivative <Binomial.derivative>` returns `variable`

    Arguments
    ---------

    default_variable : number or array : default class_defaults.variable
        specifies a template for the value(s) used as the mean of the Guassian distribution from which each sample is
        drawn.

    p : float : default 0.5
        specifies the probability with which each element of `variable` is replaced with zero.

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

    p : float
        the probability with which each element of `variable` is replaced with zero.

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

    componentName = BINOMIAL_DISTORT_FUNCTION

    classPreferences = {
        PREFERENCE_SET_NAME: 'BinomialClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    }

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------
                p
                    see `p <BinomialDistort.p>`

                    :default value: 0.5
                    :type: ``float``

                random_state
                    see `random_state <BinomialDistort.random_state>`

                    :default value: None
                    :type: ``numpy.random.RandomState``

        """
        p = Parameter(0.5, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        random_state = Parameter(None, loggable=False, getter=_random_state_getter, dependencies='seed')
        seed = Parameter(DEFAULT_SEED(), modulable=True, fallback_default=True, setter=_seed_setter)
        bounds = (None, None)

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 p: Optional[ValidParamSpecType] = None,
                 seed=None,
                 params=None,
                 owner=None,
                 prefs: Optional[ValidPrefSet] = None):

        super().__init__(
            default_variable=default_variable,
            p=p,
            seed=seed,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        p_ptr = ctx.get_param_or_state_ptr(builder, self, 'p', param_struct_ptr=params)
        p = builder.load(p_ptr)
        mod_p = builder.fsub(p.type(1), p)
        p_mod_ptr = builder.alloca(mod_p.type)
        builder.store(mod_p, p_mod_ptr)

        n_ptr = builder.alloca(ctx.int32_ty)
        builder.store(n_ptr.type.pointee(1), n_ptr)

        rand_state_ptr = ctx.get_random_state_ptr(builder, self, state, params)
        binomial_f = ctx.get_binomial_dist_function_by_state(rand_state_ptr)

        rvalp = builder.alloca(binomial_f.args[-1].type.pointee, name="random_out")
        builder.call(binomial_f, [rand_state_ptr, n_ptr, p_mod_ptr, rvalp])

        val = builder.load(ptri)
        rval = builder.load(rvalp)
        rval = builder.uitofp(rval, val.type)
        val = builder.fmul(val, rval)

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
           a single value or array to be randomly zeroed.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        variable with elements zeroed with probability p : number or array

        """
        p = self._get_current_parameter_value('p', context)
        random_state = self._get_current_parameter_value('random_state', context)
        result = variable * random_state.binomial(size=len(variable), n=1, p=(1 - p))
        return self.convert_output_type(result)

    def _is_identity(self, context=None, defaults=False):
        if defaults:
            p = self.defaults.p
        else:
            p = self.parameters.p._get(context)
        return p == 0.0

    def derivative(self, output, input=None, context=None):
        raise FunctionError(f"Derivative of BinomialDistort not yet supported.")
    #     """
    #     derivative(input, output):
    #
    #     Derivative of `function <BinomialDistort.function>`:
    #
    #         -input/:math:`{variance^3}*\\sqrt{2\\pi}`
    #
    #
    #     Returns
    #     -------
    #
    #     Derivative of Binomial of variable :  number or array
    #
    #     """
    #     bias = self._get_current_parameter_value(BIAS, context)
    #     scale = self._get_current_parameter_value(SCALE, context)
    #     offset = self._get_current_parameter_value(OFFSET, context)
    #
    #     # The following doesn't work with autograd (https://github.com/HIPS/autograd/issues/416)
    #     f = scale * np.random.normal(input+bias, variance) + offset
    #
    # # FIX: ?WHICH IF EITHER IS CORRECT?:
    # return self._get_current_parameter_value(VARIABLE, context)
    # # return 1.0


# **********************************************************************************************************************
#                                                    Dropout
# **********************************************************************************************************************

class Dropout(TransferFunction):  #
    # -------------------------------------------------------------------------------------
    """
    Dropout(               \
         default_variable, \
         p=0.5,            \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _Dropout:

    `function <Dropout._function>` returns `variable <Dropout.variable>` with elements randomly zeroed with
    probability **p** during learning; otherwise functions as `Identity` Function.  During learning, the output
    of the function is scaled by :math:`\\frac{1}{(1-p)}`, which implements the inverse scaling form of `dropout
    <https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html?highlight=dropout>`_ used by by PyTorch.

    .. math::

       if \\ (context.runmode == ContextFlags.LEARNING\\_MODE) \\ and \\ (rand[0,1] > p):  output_i = 0 \\\\
       else: \\ output_i = \\frac{1}{(1-p)}variable_i

    .. _technical_note::
       **learning_only** uses ``context.runmode`` == `ContextFlags.LEARNING_MODE`
       to determine when learning is in effect

    `derivative <Dropout.derivative>` returns `variable`

    Arguments
    ---------

    default_variable : number or array : default class_defaults.variable
        specifies a template for the value to be transformed.

    p : float : default 0.5
        specifies the probability with which each element of `variable` is replaced with zero.

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

    p : float
        the probability with which each element of `variable` is replaced with zero.

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

    componentName = DROPOUT_FUNCTION

    classPreferences = {
        PREFERENCE_SET_NAME: 'DropoutClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    }

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                p
                    see `p <Dropout.p>`

                    :default value: 0.5
                    :type: ``float``

                random_state
                    see `random_state <GaussianDistort.random_state>`

                    :default value: None
                    :type: ``numpy.random.RandomState``
        """
        p = Parameter(0.5, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        random_state = Parameter(None, loggable=False, getter=_random_state_getter, dependencies='seed')
        seed = Parameter(DEFAULT_SEED(), modulable=True, fallback_default=True, setter=_seed_setter)

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 p: Optional[ValidParamSpecType] = None,
                 params=None,
                 owner=None,
                 prefs: Optional[ValidPrefSet]  = None):
        self.binomial_distort = BinomialDistort(default_variable=default_variable, p=p)

        super().__init__(
            default_variable=default_variable,
            p=p,
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
           a single value or array to be randomly zeroed.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        During learning, variable with elements zeroed with probability p, else scaled by :math:`frac{1}{(1-p)}`;
        otherwise returns variable : number or array

        """
        p = self._get_current_parameter_value('p', context)

        if context.runmode != ContextFlags.LEARNING_MODE:
            result = variable

        else:
            p = p or self.defaults.p
            self.binomial_distort.parameters.p.set(p, context)
            result = self.binomial_distort(variable) * (1 / (1 - p))

        return self.convert_output_type(result)

    @handle_external_context()
    def derivative(self, input=None, output=None, context=None):
        # raise FunctionError(f"Derivative of Dropout not yet supported.")
        """
        derivative(input)

        Derivative of `function <Dropout._function>` at **input**.

        Arguments
        ---------

        input : number or array
            value of the input to the Dropouput function at which derivative is to be taken.

        Returns
        -------

        variable :  number or array

        """
        # FIX: ?WHICH IS CORRECT:
        # return self._get_current_parameter_value(VARIABLE, context)
        return 1.0

    def _is_identity(self, context=None, defaults=False):
        if defaults:
            p = self.defaults.p
        else:
            p = self.parameters.p._get(context)

        return (context.run_mode != ContextFlags.LEARNING_MODE) or (p == 0.0)

    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        val = builder.load(ptri)
        builder.store(val, ptro)

    def _gen_pytorch_fct(self, device, context=None):
        prob = self._get_pytorch_fct_param_value('p')
        return lambda x: (torch.dropout(input=x, p=prob, train=False))


# **********************************************************************************************************************
#                                                   SoftMax
# **********************************************************************************************************************

softmax_modes = {ALL, ARG_MAX, ARG_MAX_INDICATOR, MAX_VAL, MAX_INDICATOR, PROB, PROB_INDICATOR}


class SoftMax(TransferFunction):
    """
    SoftMax(                        \
         default_variable,          \
         gain=1.0,                  \
         mask_threshold=None,       \
         adapt_scale=1,             \
         adapt_base=1,              \
         adapt_entropy_weighting=.1 \
         output=ALL,                \
         params=None,               \
         owner=None,                \
         name=None,                 \
         prefs=None                 \
         )

    .. _SoftMax:

    SoftMax transform of `variable <Softmax.variable>`

    `function <SoftMax._function>` returns SoftMax transform of `variable <Softmax.variable>`:

    .. math::

        \\frac{e^{gain * variable_i}}{\\sum\\limits^{len(variable)}e^{gain * variable}}

    filtered by `output <SoftMax.output>` specification (see `The Softmax function and its derivative
    <http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/>`_ for a nice discussion).

        .. note::
           If `variable <SoftMax.variable>` is all zeros, the SoftMax transform returns all zeros.

    .. _SoftMax_AdaptGain:

    *Thresholding and Adaptive Gain*

    For cases in which SoftMax is used with sparse vectors (e.g., one-hots), the value(s) of the most significant
    entries (e.g., the 1s in a one-hot) can be sensitive to (diminished by) the number of other values in the vector
    (i.e., its length). For example, whereas for ``[1 0]`` the SoftMax is ``[0.73105858 0.26894142]``, for ``[1 0 0 0]``
    it is ``[0.47536689 0.1748777  0.1748777  0.1748777]``. This can be addressed in one of two ways: either by
    thresholding `variable <SoftMax.variable>` before applying the SoftMax function, or by adapting the `gain
    <SoftMax.gain>` parametrically based on the `variable <SoftMax.variable>`:

    - *mask_threshold* -- setting the **mask_threshold** argument to a scalar value causes the `variable
      <SoftMax.variable>` to be thresholded by that value before applying the SoftMax function; any elements of
      `variable <SoftMax.variable>` with an absolute value below the threshold are set to 0; all others are scaled
      by the specified `gain <SoftMax.gain>` and then passed through the SoftMax function.  This only applies if the
      **gain** argument is specified as a scalar; if it is specified as *ADAPTIVE*, then the **mask_threshold**
      argument is ignored.

    - *ADAPTIVE* -- setting **gain** argument to *ADAPTIVE* causes it to be dynamically adjusted,
      based on the entropy and length of the variable, to keep the mass of the distribution around the highest values
      as consistent as possible over different sized vectors. If *ADAPTIVE* is specified, then the `mask_threshold
      <SoftMax.mask_threshold>` argument is ignored. The gain is adapted by calling the SoftMax function's `adapt_gain
      <SoftMax.adapt_gain>` method. This can be finicky, and may need to be further tuned to the length of `variable
      <SoftMax.variable>`, which can be done using the SoftMax Function's **adapt_scale**, **adapt_base**, and
      **adapt_entropy_weighting** arguments.

    .. _SoftMax_Derivative:

    *Derivative*

    `derivative <SoftMax.derivative>` returns the derivative of the SoftMax.  If *OUTPUT_TYPE* for the SoftMax
    is *ALL*, returns Jacobian matrix (derivative for each element of the output array with respect to each of the
    others):

    .. math::
        D_jS_i = S_i(\\delta_{i,j} - S_j),\\ where\\ \\delta_{i,j}=1\\ if\\ i=j\\ and\\ \\delta_{i,j}=0\\ if\\ i≠j.

    If *OUTPUT_TYPE* is *ARG_MAX*, *ARG_MAX_INDICATOR*, *MAX_VAL*, *MAX_INDICATOR*, returns 1d array of the
    derivatives of the maximum value(s) with respect to the others (calculated as above). If *OUTPUT_TYPE* is *PROB*,
    raises an exception (since it is ambiguous as to which element would have been chosen by the SoftMax function)

    Arguments
    ---------

    default_variable : 1d array : default class_defaults.variable
        specifies a template for the value to be transformed.

    gain : scalar or ADAPTIVE : default 1.0
        specifies the value by which to multiply `variable <Linear.variable>` before SoftMax transformation,
        which functions as the inverse "temperature" of the function.  If it is a scalar, it must be greater
        than zero.  If *ADAPTIVE* is specified, the value is determined dynamically based on the `variable
        <SoftMax.variable>`; see `Thresholding and Adaptive Gain <SoftMax_AdaptGain>` for details).

    mask_threshold : scalar : default None
        specifies whether to mask_threshold the `variable <SoftMax.variable>` before applying the SoftMax function;
        this only applies if `gain <SoftMax.gain>` is specified as a scalar;  otherwise it is ignored
        (see `Thresholding and Adaptive Gain <SoftMax_AdaptGain>` for details).

    adapt_scale : scalar : default 1
        specifies the *scale* parameter using by the `adapt_gain <SoftMax.adapt_gain>` method (see method for details).

    adapt_base : scalar : default 1
        specifies the *base* parameter using by the `adapt_gain <SoftMax.adapt_gain>` method (see method for details).

    adapt_entropy_weighting : default .1
        specifies the *entropy_weighting* parameter using by the `adapt_gain <SoftMax.adapt_gain>` method
        (see method for details).

    output : ALL, ARG_MAX, ARG_MAX_INDICATOR, MAX_VAL, MAX_INDICATOR, or PROB : default ALL
        specifies the format of array returned by `function <SoftMax._function>`
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

    gain : scalar or ADAPTIVE
        determines how `variable <Logistic.variable>` is scaled before the SoftMax transformation, determining the
        "sharpness" of the distribution (it is equivalent to the inverse of the temperature of the SoftMax function);
        if it is 'ADAPTIVE', it is determined dynamically adjusted using the `adapt_gain <SoftMax.adapt_gain>` method
        (see `Thresholding and Adaptive Gain <SoftMax_AdaptGain>` for additional details).

    mask_threshold : scalar or None
        determines whether the `variable <SoftMax.variable>` is thresholded before applying the SoftMax function;
        if it is a scalar, only elements of `variable <SoftMax.variable>` with an absolute value greater than that
        value are considered when applying the SoftMax function (which are then scaled by the `gain <SoftMax.gain>`
        parameter; all other elements are assigned 0.  This only applies if `gain <SoftMax.gain>` is specified as a
        scalar;  otherwise it is ignored (see `Thresholding and Adaptive Gain <SoftMax_AdaptGain>` for details).

    adapt_scale : scalar
        determined the *scale* parameter using by the `adapt_gain <SoftMax.adapt_gain>` method (see method for details).

    adapt_base : scalar
        determines the *base* parameter using by the `adapt_gain <SoftMax.adapt_gain>` method (see method for details).

    adapt_entropy_weighting : scalar
        determines the *entropy_weighting* parameter using by the `adapt_gain <SoftMax.adapt_gain>` method
        (see method for details).

    output : ALL, ARG_MAX, ARG_MAX_INDICATOR, MAX_VAL, MAX_INDICATOR, or PROB
        determines how the SoftMax-transformed values of the elements in `variable <SoftMax.variable>` are reported
        in the array returned by `function <SoftMax._function>`:
            * *ALL*: array of all SoftMax-transformed values (the default);
            * *ARG_MAX*: 1 for single element with the maximum SoftMax-transformed value, 0 for all others;
              (one with lowest index of there are multiple maximum values);
            * *ARG_MAX_INDICATOR*: 1 for a single element with the maximum SoftMax-transformed value, 0 for all others;
              (one with lowest index of there are multiple maximum values);
            * *MAX_VAL*: SoftMax-transformed value for the element(s) with the maximum such value, 0 for all others;
            * *MAX_INDICATOR*: 1 for the element(s) with the maximum SoftMax-transformed value, 0 for all others;
            * *PROB*: probabilistically chosen element based on SoftMax-transformed values after setting the
              sum of values to 1 (i.e., their `Luce Ratio <https://en.wikipedia.org/wiki/Luce%27s_choice_axiom>`_),
              0 for all others.

    per_item : boolean : default True
        for 2d variables, determines whether the SoftMax function is applied to the entire variable (per_item =
        False), or applied to each item in the variable separately (per_item = True).

    bounds : None if `output <SoftMax.output>` in {ARG_MAX, MAX_VAL}, else (0,1) : default (0,1)

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

                adapt_scale
                    see `adapt_scale <SoftMax.adapt_scale>`

                    :default value: 1.0
                    :type: ``float``

                adapt_base
                    see `adapt_base <SoftMax.adapt_base>`

                    :default value: 1.0
                    :type: ``float``

                adapt_entropy_weighting
                    see `adapt_entropy_weighting <SoftMax.adapt_entropy_weighting>`

                    :default value: 0.1
                    :type: ``float``

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

                mask_threshold
                    see `mask_threshold <SoftMax.mask_threshold>`

                    :default value: None
                    :type: ``float``
        """
        variable = Parameter(np.array([[0.0]]), read_only=True, pnl_internal=True, constructor_argument='default_variable')
        gain = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        mask_threshold = Parameter(None, modulable=True)
        adapt_scale = Parameter(1.0, modulable=True)
        adapt_base = Parameter(1.0, modulable=True)
        adapt_entropy_weighting = Parameter(0.95, modulable=True)
        bounds = (0, 1)
        output = ALL
        per_item = Parameter(True, pnl_internal=True)
        one_hot_function = Parameter(None, stateful=False, loggable=False)

        def _validate_gain(self, gain):
            if is_numeric_scalar(gain):
                if gain <= 0:
                    return 'must be a scalar greater than 0'
            elif isinstance(gain, str):
                if gain != ADAPTIVE:
                    return f'the keyword for adaptive gain is {ADAPTIVE}'
            else:
                return f'must be a scalar greater than 0 or the keyword {ADAPTIVE}'

        def _validate_mask_threshold(self, mask_threshold):
            if mask_threshold is not None:
                if is_numeric_scalar(mask_threshold):
                    if mask_threshold <= 0:
                        return 'must be a scalar greater than 0'
                    return None
                return f'must be a scalar greater than 0'

        def _validate_adapt_scale(self, adapt_scale):
            if is_numeric_scalar(adapt_scale):
                if adapt_scale <= 0:
                    return 'must be a scalar greater than 0'
                return None
            return f'must be a scalar greater than 0'

        def _validate_adapt_base(self, adapt_base):
            if is_numeric_scalar(adapt_base):
                if adapt_base <= 0:
                    return 'must be a scalar greater than 0'
                return None
            return f'must be a scalar greater than 0'

        def _validate_adapt_entropy_weighting(self, adapt_entropy_weighting):
            if is_numeric_scalar(adapt_entropy_weighting):
                if adapt_entropy_weighting <= 0:
                    return 'must be a scalar greater than 0'
                return None
            return f'must be a scalar greater than 0'

        def _validate_output(self, output):
            if output not in softmax_modes:
                return 'not one of {0}'.format(softmax_modes)

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 gain: Optional[ValidParamSpecType] = None,
                 mask_threshold: Optional[ValidParamSpecType] = None,
                 adapt_scale: Optional[ValidParamSpecType] = None,
                 adapt_base: Optional[ValidParamSpecType] = None,
                 adapt_entropy_weighting: Optional[ValidParamSpecType] = None,
                 output=None,
                 per_item=None,
                 params: Optional[Mapping] = None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):

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
            mask_threshold=mask_threshold,
            adapt_scale=adapt_scale,
            adapt_base=adapt_base,
            adapt_entropy_weighting=adapt_entropy_weighting,
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

    def apply_softmax(self, input_value, gain, mask_threshold, output_type):

        # Modulate input_value by gain
        v = gain * input_value
        # Shift by max to avoid extreme values:
        v = v - np.max(v)
        # Exponentiate
        v = np.exp(v)
        # Threshold if specified:
        if mask_threshold:
            v = v * np.where(input_value > mask_threshold, v, 0)
        # Normalize (to sum to 1)
        if not any(v):
            # If v is all zeros, avoid divide by zero in normalize and return all zeros for softmax
            sm = v
        else:
            sm = v / np.sum(v, axis=0)

        # Generate one-hot encoding based on selected output_type
        if output_type in {ARG_MAX, ARG_MAX_INDICATOR, MAX_VAL, MAX_INDICATOR}:
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
        mask_threshold = self._get_current_parameter_value('mask_threshold', context)
        if isinstance(gain, str) and gain == ADAPTIVE:
            gain = self.adapt_gain(variable, context)
        per_item = self._get_current_parameter_value(PER_ITEM, context)

        # Compute softmax and assign to sm
        if per_item and len(np.shape(variable)) > 1:
            output = []
            for item in variable:
                output.append(self.apply_softmax(item, gain, mask_threshold, output_type))
            output = convert_all_elements_to_np_array(output)
        else:
            output = self.apply_softmax(variable, gain, mask_threshold, output_type)

        return self.convert_output_type(output)

    def adapt_gain(self, v, context)->float:
        """Compute the softmax gain (inverse temperature) based on the entropy of the distribution of values.
        Uses base, scale, and entropy_weighting parameters of SoftMax function to compute gain:

        .. math:: gain = scale * (base + (entropy\\_weighting * log(entropy(logistic(v)))))
        """
        scale = self._get_current_parameter_value('adapt_scale', context)
        base = self._get_current_parameter_value('adapt_base', context)
        entropy_weighting = self._get_current_parameter_value('adapt_entropy_weighting', context)
        entropy_weighting = np.log(len(v)) * entropy_weighting

        v = np.squeeze(v)
        gain = scale * (base +
                        (entropy_weighting *
                         np.log(
                             -1 * np.sum((1 / (1 + np.exp(-1 * v))) * np.log(1 / (1 + np.exp(-1 * v)))))))
        return gain


    @handle_external_context()
    def derivative(self, input=None, output=None, context=None):
        """
        derivative(output)

        .. technical note::
           If ARG_MAX or MAX_VAL is specified for the `output <SoftMax.output>` parameter, and there is more than one
           equivalent maximum value, the element with the lowest index is used to compute the derivative (see
           IMPLEMENTATION NOTE below).

        Returns
        -------
        derivative of values returned by SoftMax :  1d or 2d array (depending on *OUTPUT_TYPE* of SoftMax)
        """

        if output is None:
            output = self.function(input, params={OUTPUT_TYPE: ALL}, context=context)
        elif np.any(np.equal(0, output)) and context.source == ContextFlags.CONSTRUCTOR:
            # Allow derivative to be computed when output is 0 during initialization
            output = np.where(output, output==0, 1)
        else:
            assert not np.any(np.equal(0, output)), \
                f"Derivative of SoftMax function for '{self.owner.name}' is not defined when output is 0."

        per_item = self._get_current_parameter_value(PER_ITEM, context)
        if not per_item:
            output = [output]

        if np.array(output).ndim == 1:
            output = np.atleast_2d(output)

        result = []
        for sm in output:
            size = len(sm)

            output_type = self._get_current_parameter_value(OUTPUT_TYPE, context)
            if output_type == ALL:
                # Return full Jacobian matrix of derivatives using Kronecker's delta method:
                derivative = np.empty([size, size])
                for i, j in np.ndindex(size, size):
                    if i == j:
                        d = 1
                    else:
                        d = 0
                    derivative[j, i] = sm[i] * (d - sm[j])
            elif output_type in {ARG_MAX, ARG_MAX_INDICATOR, MAX_VAL, MAX_INDICATOR}:
                # Return 1d array of derivatives for max element (i.e., the one chosen by SoftMax)
                derivative = np.empty(size)
                # Get the element of output returned as non-zero (max val) when output_type is not ALL
                # IMPLEMENTATION NOTE:
                #    if there is a tie for max, this chooses the item in sm with the lowest index in sm:
                index_of_max = int(np.where(sm == np.max(sm))[-1][0])
                #    the following would randomly choose a value in case of a tie,
                #    but may cause problems with compilation:
                # index_of_max = np.where(sm == np.max(sm))[0]
                # if len(index_of_max)>1:
                #     index_of_max = int(np.random.choice(index_of_max))
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

            result.append(derivative)

        assert per_item or len(result) == 1
        return result[0] if not per_item or np.array(result).ndim == 3 else result

    def __gen_llvm_exp_sum(self, builder, index, ctx, vi, gain, exp_sum_ptr):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])

        exp_f = ctx.get_builtin("exp", [ctx.float_ty])
        orig_val = builder.load(ptri)
        val = builder.fmul(orig_val, gain)
        exp_val = builder.call(exp_f, [val])

        exp_sum = builder.load(exp_sum_ptr)
        new_exp_sum = builder.fadd(exp_sum, exp_val)
        builder.store(new_exp_sum, exp_sum_ptr)

    def __gen_llvm_exp_div(self, builder, index, ctx, vi, vo, gain, exp_sum):
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        exp_f = ctx.get_builtin("exp", [ctx.float_ty])
        orig_val = builder.load(ptri)
        val = builder.fmul(orig_val, gain)
        val = builder.call(exp_f, [val])
        val = builder.fdiv(val, exp_sum)

        builder.store(val, ptro)

    def __gen_llvm_apply(self, ctx, builder, params, state, arg_in, arg_out, output_type, tags:frozenset):
        exp_sum_ptr = builder.alloca(ctx.float_ty)
        builder.store(exp_sum_ptr.type.pointee(0), exp_sum_ptr)

        gain_ptr = ctx.get_param_or_state_ptr(builder, self, GAIN, param_struct_ptr=params)
        gain = pnlvm.helpers.load_extract_scalar_array_one(builder, gain_ptr)

        with pnlvm.helpers.array_ptr_loop(builder, arg_in, "exp_sum_max") as args:
            self.__gen_llvm_exp_sum(*args, ctx=ctx, vi=arg_in, gain=gain,
                                    exp_sum_ptr=exp_sum_ptr)

        exp_sum = builder.load(exp_sum_ptr)

        if output_type == ALL:
            one_hot_p = ctx.get_param_or_state_ptr(builder, self, 'one_hot_function', param_struct_ptr=params, state_struct_ptr=state)

            # Derivative first gets the output_type == ALL result even if the selected output type is different.
            assert self.output != output_type or one_hot_p.type.pointee.elements == (), \
                "OneHot parameter should be empty for output_type == ALL: {}".format(one_hot_p)
            with pnlvm.helpers.array_ptr_loop(builder, arg_in, "exp_div") as args:
                self.__gen_llvm_exp_div(ctx=ctx, vi=arg_in, vo=arg_out,
                                        gain=gain, exp_sum=exp_sum, *args)
            return builder

        one_hot_p, one_hot_s = ctx.get_param_or_state_ptr(builder, self, 'one_hot_function', param_struct_ptr=params, state_struct_ptr=state)
        one_hot_f = ctx.import_llvm_function(self.one_hot_function, tags=tags)

        assert one_hot_f.args[3].type == arg_out.type
        one_hot_out = arg_out
        one_hot_in = builder.alloca(one_hot_f.args[2].type.pointee)

        if output_type in {ARG_MAX, ARG_MAX_INDICATOR}:
            with pnlvm.helpers.array_ptr_loop(builder, arg_in, "exp_div") as (b, i):
                self.__gen_llvm_exp_div(ctx=ctx, vi=arg_in, vo=one_hot_in,
                                        gain=gain, exp_sum=exp_sum, builder=b, index=i)

            builder.call(one_hot_f, [one_hot_p, one_hot_s, one_hot_in, one_hot_out])

        elif output_type in PROB:
            one_hot_in_data = builder.gep(one_hot_in, [ctx.int32_ty(0), ctx.int32_ty(0)])
            one_hot_in_dist = builder.gep(one_hot_in, [ctx.int32_ty(0), ctx.int32_ty(1)])

            with pnlvm.helpers.array_ptr_loop(builder, arg_in, "exp_div") as (b, i):
                self.__gen_llvm_exp_div(ctx=ctx, vi=arg_in, vo=one_hot_in_dist,
                                        gain=gain, exp_sum=exp_sum, builder=b, index=i)

                dist_in = b.gep(arg_in, [ctx.int32_ty(0), i])
                dist_out = b.gep(one_hot_in_data, [ctx.int32_ty(0), i])
                b.store(b.load(dist_in), dist_out)


            builder.call(one_hot_f, [one_hot_p, one_hot_s, one_hot_in, one_hot_out])
        else:
            assert False, "Unsupported output in {} for LLVM execution mode: {}".format(self, output_type)

        return builder

    def _gen_llvm_function_derivative_body(self, ctx, builder, params, state, arg_in, arg_out, *, tags:frozenset):
        assert "derivative" in tags or "derivative_out" in tags
        assert arg_in.type == arg_out.type
        forward_tags = tags.difference({"derivative", "derivative_out"})

        # SoftMax derivative is calculated from the "ALL" results.
        if "derivative_out" in tags:
            all_out = arg_in
        else:
            all_out = builder.alloca(arg_out.type.pointee)
            builder = self._gen_llvm_function_body(ctx, builder, params, state, arg_in, all_out, output_type=ALL, tags=forward_tags)

        if self.parameters.per_item.get():
            assert isinstance(arg_in.type.pointee.element, pnlvm.ir.ArrayType)
            assert isinstance(arg_out.type.pointee.element, pnlvm.ir.ArrayType)
            for i in range(arg_in.type.pointee.count):
                inner_all_out = builder.gep(all_out, [ctx.int32_ty(0), ctx.int32_ty(i)])
                inner_out = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(i)])
                builder = self.__gen_llvm_apply_derivative(ctx, builder, params, state, inner_all_out, inner_out, tags=tags)
            return builder
        else:
            return self.__gen_llvm_apply_derivative(ctx, builder, params, state, all_out, arg_out, tags=tags)

    def __gen_llvm_apply_derivative(self, ctx, builder, params, state, all_out, arg_out, *, tags:frozenset):

        assert self.output in {ARG_MAX, ARG_MAX_INDICATOR, MAX_VAL, MAX_INDICATOR}, (
            "Derivative of SoftMax is only implemented for ARG_MAX and ARG_MAX_INDICATOR "
            "in LLVM execution mode ({})".format(self.output))

        max_pos_ptr = builder.alloca(ctx.int32_ty)
        builder.store(max_pos_ptr.type.pointee(-1), max_pos_ptr)
        max_val_ptr = builder.alloca(arg_out.type.pointee.element)
        builder.store(max_val_ptr.type.pointee(float("NaN")), max_val_ptr)

        with pnlvm.helpers.array_ptr_loop(builder, all_out, id="max") as (b, idx):
            val_ptr = b.gep(all_out, [ctx.int32_ty(0), idx])
            val = b.load(val_ptr)
            max_val = b.load(max_val_ptr)
            new_max = b.fcmp_unordered(">", val, max_val)
            with b.if_then(new_max):
                b.store(val, max_val_ptr)
                b.store(idx, max_pos_ptr)

        max_val = builder.load(max_val_ptr)
        max_pos = builder.load(max_pos_ptr)

        with pnlvm.helpers.array_ptr_loop(builder, all_out, id="derivative") as (b, idx):
            val_ptr = b.gep(all_out, [ctx.int32_ty(0), idx])
            val = b.load(val_ptr)
            is_max_pos = b.icmp_unsigned("==", idx, max_pos)

            d = b.select(is_max_pos, val.type(1), val.type(0))
            dv = b.fsub(d, max_val)
            val = b.fmul(val, dv)

            out_ptr = b.gep(arg_out, [ctx.int32_ty(0), idx])
            b.store(val, out_ptr)

        return builder

    def _gen_llvm_function_body(self, ctx, builder, params, state, arg_in, arg_out, output_type=None, *, tags:frozenset):
        output_type = self.output if output_type is None else output_type
        if "derivative" in tags or "derivative_out" in tags:
            return self._gen_llvm_function_derivative_body(ctx, builder, params, state, arg_in, arg_out, tags=tags)

        if self.parameters.per_item.get():
            assert isinstance(arg_in.type.pointee.element, pnlvm.ir.ArrayType)
            assert isinstance(arg_out.type.pointee.element, pnlvm.ir.ArrayType)
            for i in range(arg_in.type.pointee.count):
                inner_in = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(i)])
                inner_out = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(i)])
                builder = self.__gen_llvm_apply(ctx, builder, params, state, inner_in, inner_out, output_type, tags=tags)
            return builder
        else:
            return self.__gen_llvm_apply(ctx, builder, params, state, arg_in, arg_out, output_type, tags=tags)

    def _gen_pytorch_fct(self, device, context=None):
        gain = self._get_pytorch_fct_param_value('gain', device, context)
        mask_threshold = self._get_pytorch_fct_param_value('mask_threshold', device, context)

        if isinstance(gain, str) and gain == ADAPTIVE:
            return lambda x: (torch.softmax(self._gen_pytorch_adapt_gain_fct(device, context)(x) * x, 0))

        elif mask_threshold:
            def pytorch_thresholded_softmax(_input: torch.Tensor) -> torch.Tensor:
                # Mask elements of input below threshold
                _mask = (torch.abs(_input) > mask_threshold)
                # Subtract off the max value in the input to eliminate extreme values, exponentiate, and apply mask
                masked_exp = _mask * torch.exp(gain * (_input - torch.max(_input, 0, keepdim=True)[0]))
                if not any(masked_exp):
                    return masked_exp
                return masked_exp / torch.sum(masked_exp, 0, keepdim=True)
            # Return the function
            return pytorch_thresholded_softmax

        else:
            return lambda x: (torch.softmax(gain * x, 0))

    def _gen_pytorch_adapt_gain_fct(self, device, context=None):
        scale = self._get_pytorch_fct_param_value('adapt_scale', device, context)
        base = self._get_pytorch_fct_param_value('adapt_base', device, context)
        entropy_weighting = self._get_pytorch_fct_param_value('adapt_entropy_weighting', device, context)
        # v = torch.squeeze(v)
        return lambda x : scale * (base +
                                   (entropy_weighting * len(x) *
                                    torch.log(-1 * torch.sum((1 / (1 + torch.exp(-1 * x)))
                                                             * torch.log(1 / (1 + torch.exp(-1 * x)))))))


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


class CostFunctions(Flag):
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
    INTENSITY     = auto()
    ADJUSTMENT    = auto()
    DURATION      = auto()
    ALL           = INTENSITY | ADJUSTMENT | DURATION
    DEFAULTS      = NONE


TRANSFER_FCT = 'transfer_fct'
INTENSITY_COST_FCT = 'intensity_cost_fct'
ADJUSTMENT_COST_FCT = 'adjustment_cost_fct'
DURATION_COST_FCT = 'duration_cost_fct'
COMBINE_COSTS_FCT = 'combine_costs_fct'

class TransferWithCosts(TransferFunction):
    """
    TransferWithCosts(                      \
        default_variable=None,              \
        input_shapes=None,                          \
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
    <TransferWithCosts._function>` is executed, which assign their results to the attributes indicated below:

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
    `Port` (e.g., as the default `function <ControlSignal._function>` of a `ControlSignal`), or a `Mechanism
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

    input_shapes : int : None
        specifies length of the array for `variable <TransferWithCosts.variable>` used by `function
        <TransferWithCosts._function>` and on which costs are calculated;  can be used in place of
        default_value, in which case zeros are assigned as the value(s). An error is generated if both are
        specified but input_shapes != len(default_value).

    transfer_fct : TransferFunction : Linear
        specifies the primary function, used to generate the value it returns.

    enabled_cost_functions : CostFunctions or List[CostFunctions] : None
        specifies the costs to execute when `function <TransferWithCosts._function>` is called, and
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
        value used by `function <TransferWithCosts._function>`, and on which `intensity <TransferWithCosts.intensity>`
        and associated costs are calculated.

    input_shapes : int
        length of array for `variable <TransferWithCosts.variable>`.

    intensity : 1 array
        the result of the transfer_fct <TransferWithCosts.transfer_fct>`, and the value returned by
        `function <TransferWithCosts._function>`.

    function : TransferFunction
        primary function, specified by **transfer_fct** argument of constructor, and also stored in
        `transfer_fct <TransferWithCosts.transfer_fct>`.

    transfer_fct : TransferMechanism
        the TransferWithCosts Function's primary function, used to generate the value it returns;
        same as `function <TransferWithCosts._function>`.

    enabled_cost_functions : CostFunctions or None
        boolean combination of currently enabled CostFunctions;  determines which `cost functions
        <TransferWithCosts_Cost_Functions>` are calculated when `function <TransferWithCosts._function>`
        is called, and are included in the computation of `combined_costs <TransferWithCosts.combined_costs>`
        (see `Cost Functions <TransferWithCosts_Cost_Functions>` for additional details).

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
        <TransferWithCosts._function>` was executed.  Value is None if `adjustment_cost_fct
        <TransferWithCosts.adjustment_cost_fct>` has not been enabled (see `Cost Functions
        <TransferWithCosts_Cost_Functions>` for additional details).

    adjustment_cost_fct : TransferFunction
        calculates `adjustment_cost <TransferWithCosts.adjustment_cost>` based on the change in `intensity
        <TransferWithCosts.intensity>` from its value the last time `function <TransferWithCosts._function>` was
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
        <TransferWithCosts._function>` in which `function <TransferWithCosts.duration_cost_fct>` was enabled.

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
        variable = Parameter(np.array([0]), history_min_length=1, constructor_argument='default_variable')

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

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 input_shapes=None,
                 transfer_fct: Optional[Callable] = None,
                 enabled_cost_functions: Optional[Union[CostFunctions, list]] = None,
                 intensity_cost_fct: Optional[Callable] = None,
                 adjustment_cost_fct: Optional[Callable] = None,
                 duration_cost_fct: Optional[Callable] = None,
                 combine_costs_fct: Optional[Callable] = None,
                 params=None,
                 owner=None,
                 prefs: Optional[ValidPrefSet] = None):

        # if input_shapes:
        #     if default_variable is None:
        #         default_variable = np.zeros(input_shapes)
        #     elif input_shapes != len(default_variable):
        #         raise FunctionError(f"Both {repr(DEFAULT_VARIABLE)} ({default_variable}) and {repr(SIZE)} ({input_shapes}) "
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
        self.adjustment_cost_fct = instantiate_fct(ADJUSTMENT_COST_FUNCTION, self.adjustment_cost_fct)
        self.duration_cost_fct = instantiate_fct(DURATION_COST_FUNCTION, self.duration_cost_fct)
        self.combine_costs_fct = instantiate_fct(COMBINE_COSTS_FUNCTION, self.combine_costs_fct)

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
        self.parameters.intensity._set(copy_parameter_value(intensity), context)

        return intensity

    def _is_identity(self, context=None, defaults=False):
        transfer_fct = self.parameters.transfer_fct.get()

        if defaults:
            enabled_cost_functions = self.defaults.enabled_cost_functions
        else:
            enabled_cost_functions = self.parameters.enabled_cost_functions.get(context)

        return (
            transfer_fct._is_identity(context, defaults=defaults)
            and enabled_cost_functions == CostFunctions.NONE
        )

    @beartype
    def assign_costs(self, cost_functions: Union[CostFunctions, list], execution_context=None):
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

    @beartype
    def enable_costs(self, cost_functions: Union[CostFunctions, list], execution_context=None):
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

    @beartype
    def disable_costs(self, cost_functions: Union[CostFunctions, list], execution_context=None):
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

    def toggle_cost(self, cost_function_name: Union[str, CostFunctions],
                    assignment: bool = ON,
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
        trans_p, trans_s = ctx.get_param_or_state_ptr(builder,
                                                      self,
                                                      transfer_f.name,
                                                      param_struct_ptr=params,
                                                      state_struct_ptr=state)
        trans_in = arg_in
        trans_out = arg_out
        builder.call(trans_f, [trans_p, trans_s, trans_in, trans_out])
        intensity_ptr = ctx.get_state_space(builder, self, state, self.parameters.intensity)

        costs = [(self.parameters.intensity_cost_fct, CostFunctions.INTENSITY, self.parameters.intensity_cost),
                 (self.parameters.adjustment_cost_fct, CostFunctions.ADJUSTMENT, self.parameters.adjustment_cost),
                 (self.parameters.duration_cost_fct, CostFunctions.DURATION, self.parameters.duration_cost)]

        for (func, flag, res_param) in costs:

            cost_in = trans_out
            cost_out = ctx.get_state_space(builder, self, state, res_param)

            # The check for enablement is structural and has to be done in Python.
            # If a cost function is not enabled the cost parameter is None
            if flag in self.parameters.enabled_cost_functions.get():
                cost_f = ctx.import_llvm_function(func.get())
                cost_p, cost_s = ctx.get_param_or_state_ptr(builder,
                                                            self,
                                                            func,
                                                            param_struct_ptr=params,
                                                            state_struct_ptr=state)

                if flag == CostFunctions.ADJUSTMENT:
                    old_intensity = pnlvm.helpers.load_extract_scalar_array_one(builder, intensity_ptr)
                    new_intensity = pnlvm.helpers.load_extract_scalar_array_one(builder, trans_out)
                    adjustment = builder.fsub(new_intensity, old_intensity)

                    fabs_f = ctx.get_builtin("fabs", [adjustment.type])
                    adjustment = builder.call(fabs_f, [adjustment])

                    cost_in = builder.alloca(cost_in.type.pointee)
                    builder.store(adjustment, builder.gep(cost_in, [ctx.int32_ty(0), ctx.int32_ty(0)]))

                builder.call(cost_f, [cost_p, cost_s, cost_in, cost_out])
            else:
                # Intensity is [1] when the cost function is disabled but other cost functions are enabled
                # https://github.com/PrincetonUniversity/PsyNeuLink/issues/2711
                exp_out_len = 0 if self.parameters.enabled_cost_functions.get() == CostFunctions.NONE or flag != CostFunctions.INTENSITY else 1
                assert len(cost_out.type.pointee) == exp_out_len, "Unexpected out sturct for {}: {}".format(flag, cost_out.type.pointee)


        # TODO: combine above costs via a call to combine_costs_fct
        # depends on: https://github.com/PrincetonUniversity/PsyNeuLink/issues/2712
        # This function is still used in OCM so track both state and parameters
        combine_p, combine_s = ctx.get_param_or_state_ptr(builder,
                                                          self,
                                                          self.parameters.combine_costs_fct,
                                                          param_struct_ptr=params,
                                                          state_struct_ptr=state)

        builder.store(builder.load(trans_out), intensity_ptr)

        return builder
