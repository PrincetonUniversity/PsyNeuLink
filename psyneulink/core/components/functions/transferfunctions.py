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

* `Linear`
* `Exponential`
* `Logistic`
* `Tanh`
* `ReLU`
* `Gaussian`
* `SoftMax`
* `LinearMatrix`

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

import functools
import numbers

import numpy as np
import typecheck as tc

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.component import parameter_keywords
from psyneulink.core.components.functions.function import \
    Function_Base, FunctionError, function_keywords, MULTIPLICATIVE_PARAM, ADDITIVE_PARAM
from psyneulink.core.components.component import function_type
from psyneulink.core.globals.keywords import \
    PER_ITEM, TRANSFER_FUNCTION_TYPE, \
    LINEAR_FUNCTION, SLOPE, INTERCEPT, PARAMETER_STATE_PARAMS, \
    VARIABLE, EXPONENTIAL_FUNCTION, RATE, BIAS, SCALE, OFFSET, \
    LOGISTIC_FUNCTION, GAIN, X_0, RELU_FUNCTION, LEAK, NORMAL_FUNCTION, VARIANCE, \
    SOFTMAX_FUNCTION, ALL, MAX_VAL, MAX_INDICATOR, PROB, OUTPUT_TYPE, PROB_INDICATOR, LINEAR_MATRIX_FUNCTION, MATRIX, \
    RECEIVER, HAS_INITIALIZERS, MATRIX_KEYWORD_VALUES, IDENTITY_MATRIX, HOLLOW_MATRIX, \
    MATRIX_KEYWORD_NAMES, AUTO_ASSIGN_MATRIX, FULL_CONNECTIVITY_MATRIX, RANDOM_CONNECTIVITY_MATRIX, kwPreferenceSetName, \
    GAUSSIAN_FUNCTION, STANDARD_DEVIATION

from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.utilities import parameter_spec
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.preferences.componentpreferenceset import \
    kpReportOutputPref, PreferenceEntry, PreferenceLevel, is_pref_set

__all__ = ['TransferFunction', 'Linear', 'LinearMatrix', 'Exponential', 'Logistic', 'Tanh', 'ReLU',
           'Gaussian', 'SoftMax', 'get_matrix', 'BOUNDS', 'MODE']

BOUNDS = 'bounds'
MODE = 'mode'


class TransferFunction(Function_Base):
    """Function that transforms variable but maintains its shape

    All TransferFunctions MUST have the following attributes:

    `bounds` -- specifies the lower and upper limits of the result;  if there are none, the attribute is set to
    `None`;  if it has at least one bound, the attribute is set to a tuple specifying the lower and upper bounds,
    respectively, with `None` as the entry for no bound.

    `multiplicative_param` and `additive_param` -- each of these is assigned the name of one of the function's
    parameters and used by `ModulatoryProjections <ModulatoryProjection>` to modulate the output of the
    TransferFunction's function (see `Function_Modulatory_Params`).

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

    # IMPLEMENTATION NOTE: THESE SHOULD SHOULD BE REPLACED WITH ABC WHEN IMPLEMENTED
    def __init__(self, default_variable,
                 params,
                 owner,
                 prefs,
                 context):

        if not hasattr(self, BOUNDS):
            raise FunctionError("PROGRAM ERROR: {} must implement a {} attribute".
                                format(self.__class__.__name__, BOUNDS))

        if not hasattr(self, MULTIPLICATIVE_PARAM):
            raise FunctionError("PROGRAM ERROR: {} must implement a {} attribute".
                                format(self.__class__.__name__, MULTIPLICATIVE_PARAM))

        if not hasattr(self, ADDITIVE_PARAM):
            raise FunctionError("PROGRAM ERROR: {} must implement an {} attribute".
                                format(self.__class__.__name__, ADDITIVE_PARAM))

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

    @property
    def multiplicative(self):
        return getattr(self, self.multiplicative_param)

    @multiplicative.setter
    def multiplicative(self, val):
        setattr(self, self.multiplicative_param, val)

    @property
    def additive(self):
        return getattr(self, self.additive_param)

    @additive.setter
    def additive(self, val):
        setattr(self, self.additive_param, val)

    def _gen_llvm_function_body(self, ctx, builder, params, _, arg_in, arg_out):
        # Pretend we have one huge array to work on
        # TODO: should this be invoked in parts?
        assert isinstance(arg_in.type.pointee, pnlvm.ir.ArrayType)
        if isinstance(arg_in.type.pointee.element, pnlvm.ir.ArrayType):
            assert arg_in.type == arg_out.type
            # Array elements need all to be of the same size
            length = arg_in.type.pointee.count * arg_in.type.pointee.element.count
            arg_in = builder.bitcast(arg_in, pnlvm.ir.ArrayType(ctx.float_ty, length).as_pointer())
            arg_out = builder.bitcast(arg_out, pnlvm.ir.ArrayType(ctx.float_ty, length).as_pointer())

        kwargs = {"ctx": ctx, "vi": arg_in, "vo": arg_out, "params": params}
        inner = functools.partial(self._gen_llvm_transfer, **kwargs)

        assert arg_in.type.pointee.count == arg_out.type.pointee.count
        with pnlvm.helpers.array_ptr_loop(builder, arg_in, "transfer_loop") as args:
            inner(*args)

        return builder


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
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = LINEAR_FUNCTION

    bounds = None
    multiplicative_param = SLOPE
    additive_param = INTERCEPT

    classPreferences = {
        kwPreferenceSetName: 'LinearClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    }

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                intercept
                    see `intercept <Linear.intercept>`

                    :default value: 0.0
                    :type: float

                slope
                    see `slope <Linear.slope>`

                    :default value: 1.0
                    :type: float

        """
        slope = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        intercept = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        PARAMETER_STATE_PARAMS: None
    })

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 slope: parameter_spec = 1.0,
                 intercept: parameter_spec = 0.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(slope=slope,
                                                  intercept=intercept,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])
        slope_ptr, builder = ctx.get_param_ptr(self, builder, params, SLOPE)
        intercept_ptr, builder = ctx.get_param_ptr(self, builder, params, INTERCEPT)

        slope = pnlvm.helpers.load_extract_scalar_array_one(builder, slope_ptr)
        intercept = pnlvm.helpers.load_extract_scalar_array_one(builder, intercept_ptr)

        val = builder.load(ptri)
        val = builder.fmul(val, slope)
        val = builder.fadd(val, intercept)

        builder.store(val, ptro)

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """

        Arguments
        ---------

        variable : number or array : default class_defaults.variable
           a single value or array to be transformed.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        linear transformation of variable : number or array

        """

        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)
        slope = self.get_current_function_param(SLOPE, execution_id)
        intercept = self.get_current_function_param(INTERCEPT, execution_id)

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
            # type that even np does not recognize -- typically a custom output state variable with items of different
            # shapes (e.g. variable = [[0.0], [0.0], array([[0.0, 0.0]])] )
            elif isinstance(variable, list):
                result = []
                for variable_item in variable:
                    result.append(np.multiply(variable_item, slope) + intercept)
            else:
                raise FunctionError("Unrecognized type for {} of {} ({})".format(VARIABLE, self.name, variable))

        return self.convert_output_type(result)

    def derivative(self, input=None, output=None, execution_id=None):
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

        return self.get_current_function_param(SLOPE, execution_id)


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
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = EXPONENTIAL_FUNCTION

    bounds = (0, None)
    multiplicative_param = RATE
    additive_param = BIAS

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                bias
                    see `bias <Exponential.bias>`

                    :default value: 0.0
                    :type: float

                offset
                    see `offset <Exponential.offset>`

                    :default value: 0.0
                    :type: float

                rate
                    see `rate <Exponential.rate>`

                    :default value: 1.0
                    :type: float

                scale
                    see `scale <Exponential.scale>`

                    :default value: 1.0
                    :type: float

        """
        rate = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        bias = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        scale = Parameter(1.0, modulable=True)
        offset = Parameter(0.0, modulable=True)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate: parameter_spec = 1.0,
                 scale: parameter_spec = 1.0,
                 bias: parameter_spec = 0.0,
                 offset: parameter_spec = 0.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  bias=bias,
                                                  scale=scale,
                                                  offset=offset,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        rate_ptr, builder = ctx.get_param_ptr(self, builder, params, RATE)
        bias_ptr, builder = ctx.get_param_ptr(self, builder, params, BIAS)
        scale_ptr, builder = ctx.get_param_ptr(self, builder, params, SCALE)
        offset_ptr, builder = ctx.get_param_ptr(self, builder, params, OFFSET)

        rate = pnlvm.helpers.load_extract_scalar_array_one(builder, rate_ptr)
        bias = pnlvm.helpers.load_extract_scalar_array_one(builder, bias_ptr)
        scale = pnlvm.helpers.load_extract_scalar_array_one(builder, scale_ptr)
        offset = pnlvm.helpers.load_extract_scalar_array_one(builder, offset_ptr)

        exp_f = ctx.get_builtin("exp", [ctx.float_ty])
        val = builder.load(ptri)
        val = builder.fmul(val, rate)
        val = builder.fadd(val, bias)
        val = builder.call(exp_f, [val])
        val = builder.fmul(val, scale)
        val = builder.fadd(val, offset)

        builder.store(val, ptro)

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """

        Arguments
        ---------

        variable : number or array : default class_defaults.variable
           a single value or array to be exponentiated.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        Exponential transformation of variable : number or array

        """

        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)
        rate = self.get_current_function_param(RATE, execution_id)
        bias = self.get_current_function_param(BIAS, execution_id)
        scale = self.get_current_function_param(SCALE, execution_id)
        offset = self.get_current_function_param(OFFSET, execution_id)

        # The following doesn't work with autograd (https://github.com/HIPS/autograd/issues/416)
        # result = scale * np.exp(rate * variable + bias) + offset
        from math import e
        result = scale * e**(rate * variable + bias) + offset
        return self.convert_output_type(result)

    def derivative(self, input, output=None, execution_id=None):
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
        return self.get_current_function_param(RATE, execution_id) * input + self.get_current_function_param(BIAS, execution_id)


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
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = LOGISTIC_FUNCTION
    parameter_keywords.update({GAIN, BIAS, OFFSET})

    bounds = (0, 1)
    multiplicative_param = GAIN
    additive_param = BIAS

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                bias
                    see `bias <Logistic.bias>`

                    :default value: 0.0
                    :type: float

                gain
                    see `gain <Logistic.gain>`

                    :default value: 1.0
                    :type: float

                offset
                    see `offset <Logistic.offset>`

                    :default value: 0.0
                    :type: float

                scale
                    see `scale <Logistic.scale>`

                    :default value: 1.0
                    :type: float

                x_0
                    see `x_0 <Logistic.x_0>`

                    :default value: 0.0
                    :type: float

        """
        gain = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        x_0 = Parameter(0.0, modulable=True)
        bias = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        offset = Parameter(0.0, modulable=True)
        scale = Parameter(1.0, modulable=True)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 gain: parameter_spec = 1.0,
                 x_0=0.0,
                 bias=0.0,
                 offset: parameter_spec = 0.0,
                 scale: parameter_spec = 1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(gain=gain,
                                                  x_0=x_0,
                                                  bias=bias,
                                                  offset=offset,
                                                  scale=scale,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        gain_ptr, builder = ctx.get_param_ptr(self, builder, params, GAIN)
        bias_ptr, builder = ctx.get_param_ptr(self, builder, params, BIAS)
        x_0_ptr, builder = ctx.get_param_ptr(self, builder, params, X_0)
        scale_ptr, builder = ctx.get_param_ptr(self, builder, params, SCALE)
        offset_ptr, builder = ctx.get_param_ptr(self, builder, params, OFFSET)

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
        val = builder.fmul(val, scale)
        val = builder.fadd(ctx.float_ty(1), val)
        val = builder.fdiv(ctx.float_ty(1), val)

        builder.store(val, ptro)

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """

        Arguments
        ---------

        variable : number or array : default class_defaults.variable
           a single value or array to be transformed.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        Logistic transformation of variable : number or array

        """

        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)
        gain = self.get_current_function_param(GAIN, execution_id)
        bias = self.get_current_function_param(BIAS, execution_id)
        x_0 = self.get_current_function_param(X_0, execution_id)
        offset = self.get_current_function_param(OFFSET, execution_id)
        scale = self.get_current_function_param(SCALE, execution_id)

        # The following doesn't work with autograd (https://github.com/HIPS/autograd/issues/416)
        # result = 1. / (1 + np.exp(-gain * (variable - bias) + offset))
        from math import e
        result = scale * (1. / (1 + e**(-gain * (variable + bias - x_0) + offset)))

        return self.convert_output_type(result)

    def derivative(self, input=None, output=None, execution_id=None):
        """
        derivative(input=None, output=None)

        Derivative of `function <Exponential.function>` at either **input** or **output**.

        Either **input** or **ouput** must be specified.  If **output** is not specified, it is computed from **input**.
        If both are specified, **input** is ignored unless paramValidationPref is set, in which case
        an error is generated if **output** does not correspond to `function <Logistic.function>`\(**input**).

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
                valid = output == self.function(input, execution_id=execution_id)
            else:
                valid = all(output[i] == self.function(input, execution_id=execution_id)[i] for i in range(len(input)))
            if not valid:
                raise FunctionError("Value of {} arg passed to {} ({}) "
                                    "does not match the value expected for specified {} ({})".
                                    format(repr('output'), self.__class__.__name__+'.'+'derivative', output,
                                           repr('input'), input))

        gain = self.get_current_function_param(GAIN, execution_id)
        scale = self.get_current_function_param(SCALE, execution_id)

        if output is None:
            output = self.function(input, execution_id=execution_id)

        return gain * scale * output * (1 - output)


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

        \\frac{1 - e^{-2(gain*(variable+bias-x\_0)+offset)}}{1 + e^{-2(gain*(variable+bias-x\_0)+offset)}}

    .. note::

       The `Logistic` function is an offset and scaled version of this function.
       The parameters used here have the same meaning as those used for the `Logistic` Function.

    `derivative <Tanh.derivative>` returns the derivative of the hyperbolic tangent at its **input**:

    .. math::
        \\frac{gain*scale}{(\\frac{1+e^{-2(gain*(variable+bias-x\_0)+offset)}}{2e^{-(gain*(
       variable+bias-x\_0)+offset)}})^2}

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
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = LOGISTIC_FUNCTION
    parameter_keywords.update({GAIN, BIAS, OFFSET})

    bounds = (0, 1)
    multiplicative_param = GAIN
    additive_param = BIAS

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                bias
                    see `bias <Tanh.bias>`

                    :default value: 0.0
                    :type: float

                gain
                    see `gain <Tanh.gain>`

                    :default value: 1.0
                    :type: float

                offset
                    see `offset <Tanh.offset>`

                    :default value: 0.0
                    :type: float

                scale
                    see `scale <Tanh.scale>`

                    :default value: 1.0
                    :type: float

                x_0
                    see `x_0 <Tanh.x_0>`

                    :default value: 0.0
                    :type: float

        """
        gain = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        x_0 = Parameter(0.0, modulable=True)
        bias = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        offset = Parameter(0.0, modulable=True)
        scale = Parameter(1.0, modulable=True)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 gain: parameter_spec = 1.0,
                 x_0=0.0,
                 bias=0.0,
                 offset: parameter_spec = 0.0,
                 scale: parameter_spec = 1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(gain=gain,
                                                  x_0=x_0,
                                                  bias=bias,
                                                  offset=offset,
                                                  scale=scale,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        gain_ptr, builder = ctx.get_param_ptr(self, builder, params, GAIN)
        bias_ptr, builder = ctx.get_param_ptr(self, builder, params, BIAS)
        x_0_ptr, builder = ctx.get_param_ptr(self, builder, params, X_0)
        offset_ptr, builder = ctx.get_param_ptr(self, builder, params, OFFSET)
        scale_ptr, builder = ctx.get_param_ptr(self, builder, params, SCALE)

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
        val = builder.fmul(val, scale)
        val = builder.fadd(ctx.float_ty(1), val)
        val = builder.fdiv(ctx.float_ty(1), val)

        builder.store(val, ptro)

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """

        Arguments
        ---------

        variable : number or array : default class_defaults.variable
           a single value or array to be transformed.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        hyperbolic tangent of variable : number or array

        """

        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)
        gain = self.get_current_function_param(GAIN, execution_id)
        bias = self.get_current_function_param(BIAS, execution_id)
        x_0 = self.get_current_function_param(X_0, execution_id)
        offset = self.get_current_function_param(OFFSET, execution_id)

        # The following probably doesn't work with autograd (https://github.com/HIPS/autograd/issues/416)
        #   (since np.exp doesn't work)
        # result = 1. / (1 + np.tanh(-gain * (variable - bias) + offset))
        from math import e
        exponent = -2*(gain * (variable + bias - x_0) + offset)
        result = (1 - e**exponent)/ (1 + e**exponent)

        return self.convert_output_type(result)


    def derivative(self, input, output=None, execution_id=None):
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
        gain = self.get_current_function_param(GAIN, execution_id)
        bias = self.get_current_function_param(BIAS, execution_id)
        x_0 = self.get_current_function_param(X_0, execution_id)
        offset = self.get_current_function_param(OFFSET, execution_id)
        scale = self.get_current_function_param(SCALE, execution_id)

        from math import e
        return gain*scale / ((1 + e**(-2*(gain*(input+bias-x_0)+offset))) / (2 * e**(-gain*(input+bias-x_0)+offset)))**2


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
        gain*(variable - bias)\ if\ (variable - bias) > 0,\ leak*(variable - bias)\ otherwise

    Commonly used by `ReLU <https://en.wikipedia.org/wiki/Rectifier_(neural_networks>`_ units in neural networks.

    `derivative <ReLU.derivative>` returns the derivative of of the rectified linear tranform at its **input**:

    .. math::
        gain\ if\ input > 0,\ leak\ otherwise

    Arguments
    ---------
    default_variable : number or array : default class_defaults.variable
        specifies a template for the value to be transformed.
    gain : float : default 1.0
        specifies a value by which to multiply `variable <ReLU.variable>` after `bias <ReLU.bias>` is subtracted
        from it, if (variable - bias) is greater than 0.
    bias : float : default 0.0
        specifies a value to subtract from each element of `variable <ReLU.variable>` before checking if the
        result is greater than 0 and multiplying by either gain or leak based on the result.
    leak : float : default 0.0
        specifies a value by which to multiply `variable <ReLU.variable>` after `bias <ReLU.bias>` is subtracted
        from it if (variable - bias) is lesser than or equal to 0.
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
        contains value to be transformed.
    gain : float : default 1.0
        value multiplied with `variable <ReLU.variable>` after `bias <ReLU.bias>` is subtracted from it if
        (variable - bias) is greater than 0.
    bias : float : default 0.0
        value subtracted from each element of `variable <ReLU.variable>` before checking if the result is
        greater than 0 and multiplying by either gain or leak based on the result.
    leak : float : default 0.0
        value multiplied with `variable <ReLU.variable>` after `bias <ReLU.bias>` is subtracted from it if
        (variable - bias) is lesser than or equal to 0.
    bounds : (None,None)
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

    componentName = RELU_FUNCTION
    parameter_keywords.update({GAIN, BIAS, LEAK})

    bounds = (None,None)
    multiplicative_param = GAIN
    additive_param = BIAS

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                bias
                    see `bias <ReLU.bias>`

                    :default value: 0.0
                    :type: float

                gain
                    see `gain <ReLU.gain>`

                    :default value: 1.0
                    :type: float

                leak
                    see `leak <ReLU.leak>`

                    :default value: 0.0
                    :type: float

        """
        gain = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        bias = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        leak = Parameter(0.0, modulable=True)
    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 gain: parameter_spec = 1.0,
                 bias: parameter_spec = 0.0,
                 leak: parameter_spec = 0.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(gain=gain,
                                                  bias=bias,
                                                  leak=leak,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """

        Arguments
        ---------

        variable : number or array : default class_defaults.variable
           a single value or array to be transformed.
        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        ReLU transformation of variable : number or array
        """

        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        gain = self.get_current_function_param(GAIN, execution_id)
        bias = self.get_current_function_param(BIAS, execution_id)
        leak = self.get_current_function_param(LEAK, execution_id)

        result = np.maximum(gain * (variable - bias), bias, leak * (variable - bias))
        return self.convert_output_type(result)

    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        gain_ptr, builder = ctx.get_param_ptr(self, builder, params, GAIN)
        bias_ptr, builder = ctx.get_param_ptr(self, builder, params, BIAS)
        leak_ptr, builder = ctx.get_param_ptr(self, builder, params, LEAK)

        gain = pnlvm.helpers.load_extract_scalar_array_one(builder, gain_ptr)
        bias = pnlvm.helpers.load_extract_scalar_array_one(builder, bias_ptr)
        leak = pnlvm.helpers.load_extract_scalar_array_one(builder, leak_ptr)

        # Maxnum for some reason needs full function prototype
        max_f = ctx.get_builtin("maxnum", [ctx.float_ty],
            pnlvm.ir.FunctionType(ctx.float_ty, [ctx.float_ty, ctx.float_ty]))
        var = builder.load(ptri)
        val = builder.fsub(var, bias)
        val1 = builder.fmul(val, gain)
        val2 = builder.fmul(val, leak)

        val = builder.call(max_f, [val1, bias])
        # TODO: WHat is the third param to np.maximum
        # val = builder.call(max_f, [val, val2])

        builder.store(val, ptro)

    def derivative(self, input, output=None, execution_id=None):
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
        gain = self.get_current_function_param(GAIN, execution_id)
        leak = self.get_current_function_param(LEAK, execution_id)

        if (input > 0): return gain
        else: return leak


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

    `function <Gaussian.function>` returns Gaussian transform of `variable <Logistic.variable>`:

    .. math::
      scale*\\frac{e^{-\\frac{(varible-bias)^{2}}{2\\sigma^{2}}}}{\\sqrt{2\\pi}\\sigma}+offset

    where :math:`\\sigma` = `standard_deviation <Gaussian.standard_deviation>`

    .. note::
        the value returned is deterministic (i.e., the value of the probability density function at variable),
        not a randomly chosen sample from the Gaussian distribution; for the latter, use `NormalDist` and set
        `mean <NormalDist.mean>` equal to variable.

    `derivative <Gaussian.derivative>` returns derivative of the Gaussian transform of `variable <Logistic.variable>`:

    .. math::

       \\frac{-(variable-bias)*e^{-\\frac{(variable-bias)^{2}}{2\\sigma^{2}}}}{\\sqrt{2\\pi}\\sigma^{3}}

    Arguments
    ---------

    default_variable : number or array : default class_defaults.variable
        specifies a template for the value used as the mean for the Guassian transform.

    standard_deviation : float : default 1.0
        specifies "width" of the Gaussian transform applied to each element of `variable <Gaussian.variable>`.

    bias : float : default 0.0
        value to add to each element after applying height and before applying Gaussian transform.

    offset : float : default 0.0
        value to add to each element after applying Gaussian transform and `scale <Gaussian.scale>`.

    scale : float : default 1.0
        value by which to multiply each element after applying Gaussian transform.

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
        value used as the mean of the Gaussian transform.

    standard_deviation : float : default 1.0
        standard_deviation used for Gaussian transform.

    bias : float : default 0.0
        value added to each element after applying height and before applying the Gaussian transform.

    scale : float : default 0.0
        value by which each element is multiplied after applying the Gaussian transform.

    offset : float : default 0.0
        value added to each element after applying the Gaussian transform and scale.

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

    componentName = GAUSSIAN_FUNCTION
    # parameter_keywords.update({STANDARD_DEVIATION, BIAS, SCALE, OFFSET})

    bounds = (None,None)
    multiplicative_param = STANDARD_DEVIATION
    additive_param = BIAS

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                bias
                    see `bias <Gaussian.bias>`

                    :default value: 0.0
                    :type: float

                offset
                    see `offset <Gaussian.offset>`

                    :default value: 0.0
                    :type: float

                scale
                    see `scale <Gaussian.scale>`

                    :default value: 0.0
                    :type: float

                standard_deviation
                    see `standard_deviation <Gaussian.standard_deviation>`

                    :default value: 1.0
                    :type: float

        """
        standard_deviation = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        bias = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        scale = Parameter(0.0, modulable=True)
        offset = Parameter(0.0, modulable=True)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 standard_deviation: parameter_spec = 1.0,
                 bias: parameter_spec = 0.0,
                 scale: parameter_spec = 1.0,
                 offset: parameter_spec = 0.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(standard_deviation=standard_deviation,
                                                  bias=bias,
                                                  scale=scale,
                                                  offset=offset,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        standard_deviation_ptr, builder = ctx.get_param_ptr(self, builder, params, STANDARD_DEVIATION)
        bias_ptr, builder = ctx.get_param_ptr(self, builder, params, BIAS)
        scale_ptr, builder = ctx.get_param_ptr(self, builder, params, SCALE)
        offset_ptr, builder = ctx.get_param_ptr(self, builder, params, OFFSET)

        standard_deviation = pnlvm.helpers.load_extract_scalar_array_one(builder, standard_deviation_ptr)
        bias = pnlvm.helpers.load_extract_scalar_array_one(builder, bias_ptr)
        scale = pnlvm.helpers.load_extract_scalar_array_one(builder, scale_ptr)
        offset = pnlvm.helpers.load_extract_scalar_array_one(builder, offset_ptr)

        exp_f = ctx.module.declare_intrinsic("llvm.exp", [ctx.float_ty])

        numerator = builder.load(ptri)
        numerator = builder.fsub(bias, numerator)
        numerator = builder.fmul(numerator, numerator)
        numerator = builder.fneg(numerator)

        denom = builder.fmul(standard_deviation, standard_deviation)
        denom = builder.fmul(2, denom)
        numerator = builder.fdiv(denom, numerator)
        numerator = builder.call(exp_f, [numerator])

        from math import pi
        denom = ctx.float_ty(2 * pi)
        denom = builder.fmul(standard_deviation, denom)
        denom = builder.sqrtpd(denom)
        val = builder.fdiv(denom,numerator)

        val = builder.fmul(scale, val)
        val = builder.fadd(offset, val)

        val = builder.fadd(ctx.float_ty(1), val)
        val = builder.fdiv(ctx.float_ty(1), val)

        builder.store(val, ptro)

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """

        Arguments
        ---------

        variable : number or array : default class_defaults.variable
           a single value or array to be transformed.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        Gaussian transformation of variable : number or array

        """

        variable = self._check_args(variable=variable, params=params, context=context)
        standard_deviation = self.get_current_function_param(STANDARD_DEVIATION, execution_id)
        bias = self.get_current_function_param(BIAS, execution_id)
        scale = self.get_current_function_param(SCALE, execution_id)
        offset = self.get_current_function_param(OFFSET, execution_id)

        from math import e, pi, sqrt
        gaussian = e**(-(variable-bias)**2/(2*standard_deviation**2)) / sqrt(2*pi*standard_deviation)
        result = scale * gaussian + offset

        return self.convert_output_type(result)

    def derivative(self, input, output=None, execution_id=None):
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
        sigma = self.get_current_function_param(STANDARD_DEVIATION, execution_id)
        bias = self.get_current_function_param(BIAS, execution_id)

        from math import e, pi, sqrt
        adjusted_input = input-bias
        result = (-adjusted_input * e**(-(adjusted_input**2/(2*sigma**2)))) / sqrt(2*pi*sigma**3)

        return self.convert_output_type(result)


# Another TransferFunction (e.g. Linear or Logistic) with noise=NormalDist should be used in place of this:
# class Normal(TransferFunction):  # -----------------------------------------------------------------------------------
#     """
#     Normal(              \
#          default_variable, \
#          variance=1.0,     \
#          bias=0.0,         \
#          scale=1.0,        \
#          offset=0.0,       \
#          params=None,      \
#          owner=None,       \
#          name=None,        \
#          prefs=None        \
#          )
#
#     .. _Normal_Function:
#
#     Sample from the normal distribution for each element of `variable <Normal.variable>`, centered on each
#     element's value.
#
#     Arguments
#     ---------
#
#     default_variable : number or array : default class_defaults.variable
#         specifies a template for the value used as the mean for the Guassian transform.
#
#     variance : float : default 1.0
#         specifies "width" of the Normal transform applied to each element of `variable <Normal.variable>`.
#
#     bias : float : default 0.0
#         value to add to each element after applying height and before applying Normal transform.
#
#     scale : float : default 1.0
#         value by which to multiply each element after applying Normal transform.
#
#     offset : float : default 0.0
#         value to add to each element after applying Normal transform and `scale <Normal.scale>`.
#
#     params : Dict[param keyword: param value] : default None
#         a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
#         function.  Values specified for parameters in the dictionary override any assigned to those parameters in
#         arguments of the constructor.
#
#     owner : Component
#         `component <Component>` to which to assign the Function.
#
#     name : str : default see `name <Function.name>`
#         specifies the name of the Function.
#
#     prefs : PreferenceSet or specification dict : default Function.classPreferences
#         specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).
#
#     Attributes
#     ----------
#
#     variable : number or array
#         value used as the mean of the Normal transform.
#
#     variance : float : default 1.0
#         variance used for Normal transform.
#
#     bias : float : default 0.0
#         value added to each element after applying height and before applying the Normal transform.
#
#     scale : float : default 0.0
#         value by which each element is multiplied after applying the Normal transform.
#
#     offset : float : default 0.0
#         value added to each element after applying the Normal transform and scale.
#
#     owner : Component
#         `component <Component>` to which the Function has been assigned.
#
#     name : str
#         the name of the Function; if it is not specified in the **name** argument of the constructor, a
#         default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).
#
#     prefs : PreferenceSet or specification dict : Function.classPreferences
#         the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
#         constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
#         <LINK>` for details).
#     """
#
#     componentName = NORMAL_FUNCTION
#     # parameter_keywords.update({VARIANCE, BIAS, SCALE, OFFSET})
#
#     bounds = (None,None)
#     multiplicative_param = VARIANCE
#     additive_param = BIAS
#
#     paramClassDefaults = Function_Base.paramClassDefaults.copy()
#
#     class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <SoftMax.variable>`

                    :default value: numpy.array(0.)
                    :type: numpy.ndarray
                    :read only: True

                bounds
                    see `bounds <SoftMax.bounds>`

                    :default value: (0, 1)
                    :type: <class 'tuple'>

                gain
                    see `gain <SoftMax.gain>`

                    :default value: 1.0
                    :type: float

                output
                    see `output <SoftMax.output>`

                    :default value: `ALL`
                    :type: str

                per_item
                    see `per_item <SoftMax.per_item>`

                    :default value: True
                    :type: bool

        """
#         variance = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
#         bias = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
#         scale = Parameter(0.0, modulable=True)
#         offset = Parameter(0.0, modulable=True)
#
#     @tc.typecheck
#     def __init__(self,
#                  default_variable=None,
#                  variance: parameter_spec = 1.0,
#                  bias: parameter_spec = 0.0,
#                  scale: parameter_spec = 1.0,
#                  offset: parameter_spec = 0.0,
#                  params=None,
#                  owner=None,
#                  prefs: is_pref_set = None):
#         # Assign args to params and functionParams dicts (kwConstants must == arg names)
#         params = self._assign_args_to_param_dicts(variance=variance,
#                                                   bias=bias,
#                                                   scale=scale,
#                                                   offset=offset,
#                                                   params=params)
#
#         super().__init__(default_variable=default_variable,
#                          params=params,
#                          owner=owner,
#                          prefs=prefs,
#                          context=ContextFlags.CONSTRUCTOR)

    # def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params):
    #     ptri = builder.gep(vi, [ctx.int32_ty(0), index])
    #     ptro = builder.gep(vo, [ctx.int32_ty(0), index])
    #
    #     variance_ptr, builder = ctx.get_param_ptr(self, builder, params, VARIANCE)
    #     bias_ptr, builder = ctx.get_param_ptr(self, builder, params, BIAS)
    #     scale_ptr, builder = ctx.get_param_ptr(self, builder, params, SCALE)
    #     offset_ptr, builder = ctx.get_param_ptr(self, builder, params, OFFSET)
    #
    #     variance = pnlvm.helpers.load_extract_scalar_array_one(builder, variance_ptr)
    #     bias = pnlvm.helpers.load_extract_scalar_array_one(builder, bias_ptr)
    #     scale = pnlvm.helpers.load_extract_scalar_array_one(builder, scale_ptr)
    #     offset = pnlvm.helpers.load_extract_scalar_array_one(builder, offset_ptr)
    #
    #     exp_f = ctx.module.declare_intrinsic("llvm.exp", [ctx.float_ty])
    #     val = builder.load(ptri)
    #     val = builder.fadd(val, bias)
    #     val = builder.fmul(val, variance)
    #     val = builder.fsub(offset, val)
    #     val = builder.call(exp_f, [val])
    #     val = builder.fadd(ctx.float_ty(1), val)
    #     val = builder.fdiv(ctx.float_ty(1), val)
    #
    #     builder.store(val, ptro)

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """

        Arguments
        ---------

        variable : number or array : default class_defaults.variable
           a single value or array to be transformed.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        Samples from normal distribution for each element of variable : number or array

        """

        variable = self._check_args(variable=variable, params=params, context=context)
        variance = self.get_current_function_param(VARIANCE, execution_id)
        bias = self.get_current_function_param(BIAS, execution_id)
        scale = self.get_current_function_param(SCALE, execution_id)
        offset = self.get_current_function_param(OFFSET, execution_id)

        # The following doesn't work with autograd (https://github.com/HIPS/autograd/issues/416)
        result = scale * np.random.normal(variable+bias, variance) + offset

        return self.convert_output_type(result)

    # def derivative(self, output, input=None, execution_id=None):
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
    #     variance = self.get_current_function_param(VARIANCE, execution_id)
    #     bias = self.get_current_function_param(BIAS, execution_id)
    #     scale = self.get_current_function_param(SCALE, execution_id)
    #     offset = self.get_current_function_param(OFFSET, execution_id)
    #
    #     # The following doesn't work with autograd (https://github.com/HIPS/autograd/issues/416)
    #     f = scale * np.random.normal(input+bias, variance) + offset
    #
    #     # FIX: SHOULD THIS BE variance**1.5 (since variance = sd**2 and term below is supposed to be sd**3)??
    #     df = -input(variance**3 * np.sqrt(2 * np.pi))
    #
    #     return self.convert_output_type(df*f)


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
        D_jS_i = S_i(\\delta_{i,j} - S_j),\ where\ \\delta_{i,j}=1\ if\ i=j\ and\ \\delta_{i,j}=0\ if\ i≠j.

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
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).
    """

    componentName = SOFTMAX_FUNCTION

    bounds = (0, 1)
    multiplicative_param = GAIN
    additive_param = None

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <SoftMax.variable>`

                    :default value: numpy.array(0.)
                    :type: numpy.ndarray
                    :read only: True

                bounds
                    see `bounds <SoftMax.bounds>`

                    :default value: (0, 1)
                    :type: <class 'tuple'>

                gain
                    see `gain <SoftMax.gain>`

                    :default value: 1.0
                    :type: float

                output
                    see `output <SoftMax.output>`

                    :default value: `ALL`
                    :type: str

                per_item
                    see `per_item <SoftMax.per_item>`

                    :default value: True
                    :type: bool

        """
        variable = Parameter(np.array(0.0), read_only=True)
        gain = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        bounds = (0, 1)
        output = ALL
        per_item = True

        def _validate_output(self, output):
            options = {ALL, MAX_VAL, MAX_INDICATOR, PROB}
            if output in options:
                return None
            else:
                return 'not one of {0}'.format(options)

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 gain: parameter_spec = 1.0,
                 output: tc.enum(ALL, MAX_VAL, MAX_INDICATOR, PROB) = ALL,
                 per_item=True,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(gain=gain,
                                                  per_item=per_item,
                                                  output=output,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

    def _validate_variable(self, variable, context=None):
        if variable is None:
            try:
                return self.defaults.variable
            except AttributeError:
                return self.class_defaults.variable

        return np.asarray(variable)

    def _instantiate_function(self, function, function_params=None, context=None):

        self.one_hot_function = None
        output_type = self.get_current_function_param(OUTPUT_TYPE)
        bounds = None

        if not output_type is ALL:
            from psyneulink.core.components.functions.selectionfunctions import OneHot
            self.one_hot_function = OneHot(mode=output_type).function

        super()._instantiate_function(function, function_params=function_params, context=context)

    def __gen_llvm_exp_sum_max(self, builder, index, ctx, vi, vo, gain, max_ptr, exp_sum_ptr, max_ind_ptr):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

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
        assert self.get_current_function_param(OUTPUT_TYPE) == ALL
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
        builder.store(ctx.float_ty(0), exp_sum_ptr)

        max_ptr = builder.alloca(ctx.float_ty)
        builder.store(ctx.float_ty(float('-inf')), max_ptr)

        max_ind_ptr = builder.alloca(ctx.int32_ty)
        gain_ptr, builder = ctx.get_param_ptr(self, builder, params, GAIN)

        gain = pnlvm.helpers.load_extract_scalar_array_one(builder, gain_ptr)

        kwargs = {"ctx": ctx, "vi": arg_in, "vo": arg_out, "max_ptr": max_ptr, "gain": gain, "max_ind_ptr": max_ind_ptr, "exp_sum_ptr": exp_sum_ptr}
        inner = functools.partial(self.__gen_llvm_exp_sum_max, **kwargs)

        with pnlvm.helpers.array_ptr_loop(builder, arg_in, "exp_sum_max") as args:
            inner(*args)

        output_type = self.get_current_function_param(OUTPUT_TYPE)
        exp_sum = builder.load(exp_sum_ptr)
        index = builder.load(max_ind_ptr)
        ptro = builder.gep(arg_out, [ctx.int32_ty(0), index])

        if output_type == ALL:
            kwargs = {"ctx": ctx, "vi": arg_in, "vo": arg_out, "gain": gain, "exp_sum": exp_sum}
            inner = functools.partial(self.__gen_llvm_exp_div, **kwargs)
            with pnlvm.helpers.array_ptr_loop(builder, arg_in, "exp_div") as args:
                inner(*args)
        elif output_type == MAX_VAL:
            ptri = builder.gep(arg_in, [ctx.int32_ty(0), index])
            exp_f = ctx.get_builtin("exp", [ctx.float_ty])
            orig_val = builder.load(ptri)
            val = builder.fmul(orig_val, gain)
            val = builder.call(exp_f, [val])
            val = builder.fdiv(val, exp_sum)
            builder.store(val, ptro)
        elif output_type == MAX_INDICATOR:
            builder.store(ctx.float_ty(1), ptro)

        return builder

    def _gen_llvm_function_body(self, ctx, builder, params, _, arg_in, arg_out):
        if self.get_current_function_param(PER_ITEM):
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

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """

        Arguments
        ---------

        variable : 1d array : default class_defaults.variable
           an array to be transformed.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        SoftMax transformation of variable : number or array

        """

        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        # Assign the params and return the result
        output_type = self.get_current_function_param(OUTPUT_TYPE, execution_id)
        gain = self.get_current_function_param(GAIN, execution_id)
        per_item = self.get_current_function_param(PER_ITEM, execution_id)
        # Compute softmax and assign to sm

        if per_item and len(np.shape(variable)) > 1:
            output = []
            for item in variable:
                output.append(self.apply_softmax(item, gain, output_type))
        else:
            output = self.apply_softmax(variable, gain, output_type)

        return self.convert_output_type(output)

    def derivative(self, output, input=None, execution_id=None):
        """
        derivative(output)

        Returns
        -------

        derivative of values returned by SoftMax :  1d or 2d array (depending on *OUTPUT_TYPE* of SoftMax)
        """

        output_type = self.params[OUTPUT_TYPE]
        size = len(output)
        sm = self.function(output, params={OUTPUT_TYPE: ALL}, execution_id=execution_id)

        if output_type is ALL:
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

        When LinearMatrix is the `function <Projection.function>` of a projection:

            - the matrix specification must be compatible with the variables of the `sender <Projection.sender>` and
              `receiver <Projection.receiver>`

            - a matrix keyword specification generates a matrix based on the sender and receiver shapes

        When LinearMatrix is instantiated on its own, or as the function of `Mechanism` or `State`:

            - the matrix specification must be compatible with the function's own `variable <LinearMatrix.variable>`

            - if matrix is not specified, a square identity matrix is generated based on the number of columns in
              `variable <LinearMatrix.variable>`

            - matrix keywords are not valid matrix specifications

    bounds : None

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
        the name of the Function; if it is not specified in the **name** argument of the constructor, a
        default is assigned by FunctionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see `PreferenceSet`
        for details).
    """

    componentName = LINEAR_MATRIX_FUNCTION

    bounds = None
    multiplicative_param = None
    additive_param = None

    DEFAULT_FILLER_VALUE = 0

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

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
    #     if isinstance(m, (list, np.ndarray, np.matrix, function_type)):
    #         return True
    #     return False

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 matrix=None,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(matrix=matrix,
                                                  params=params)

        # Note: this calls _validate_variable and _validate_params which are overridden below;
        #       the latter implements the matrix if required
        # super(LinearMatrix, self).__init__(default_variable=default_variable,
        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

        self.matrix = self.instantiate_matrix(self.paramsCurrent[MATRIX])

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
            # Note: this assumes variable is a 1D array, as enforced by _validate_variable
            sender_len = sender.size

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
                if param_name is RECEIVER:
                    continue

                # Not currently used here
                if param_name in function_keywords:
                    continue

                if param_name is HAS_INITIALIZERS:
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

                        weight_matrix = np.matrix(param_value)
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
                            param_value = np.matrix(param_value)
                        except (ValueError, TypeError) as error_msg:
                            raise FunctionError("Error in string specification ({}) of the matrix "
                                                "for the {} function of {}: {})".
                                                # format(param_value, self.__class__.__name__, error_msg))
                                                format(param_value, self.name, self.owner_name, error_msg))

                    # function so:
                    # - assume it uses random.rand()
                    # - call with two args as place markers for cols and rows
                    # -  validate that it returns an array or np.matrix
                    elif isinstance(param_value, function_type):
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
                    message += "Unrecognized param ({}) specified for the {} function of {}\n". \
                        format(param_name, self.componentName, self.owner_name)
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
        if self.matrix is None and not hasattr(self.owner, "receiver"):
            variable_length = np.size(np.atleast_2d(self.defaults.variable), 1)
            self.matrix = np.identity(variable_length)
        self.matrix = self.instantiate_matrix(self.matrix)

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

    def _gen_llvm_function_body(self, ctx, builder, params, _, arg_in, arg_out):
        # Restrict to 1d arrays
        assert self.defaults.variable.ndim == 1

        matrix, builder = ctx.get_param_ptr(self, builder, params, MATRIX)

        # Convert array pointer to pointer to the fist element
        matrix = builder.gep(matrix, [ctx.int32_ty(0), ctx.int32_ty(0)])
        vec_in = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(0)])
        vec_out = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(0)])

        input_length = ctx.int32_ty(arg_in.type.pointee.count)
        output_length = ctx.int32_ty(arg_out.type.pointee.count)
        builtin = ctx.get_llvm_function('__pnl_builtin_vxm')
        builder.call(builtin, [vec_in, matrix, input_length, output_length, vec_out])
        return builder

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """

        Arguments
        ---------
        variable : list or 1d array
            array to be transformed;  length must equal the number of rows of `matrix <LinearMatrix.matrix>`.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        ---------

        dot product of variable and matrix : 1d array
            length of the array returned equals the number of columns of `matrix <LinearMatrix.matrix>`.

        """

        # Note: this calls _validate_variable and _validate_params which are overridden above;
        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)
        matrix = self.get_current_function_param(MATRIX, execution_id)
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


# def is_matrix_spec(m):
#     if m is None:
#         return True
#     if isinstance(m, (list, np.ndarray, np.matrix, function_type)):
#         return True
#     if m in MATRIX_KEYWORD_VALUES:
#         return True
#     return False


def get_matrix(specification, rows=1, cols=1, context=None):
    """Returns matrix conforming to specification with dimensions = rows x cols or None

     Specification can be a matrix keyword, filler value or np.ndarray

     Specification (validated in _validate_params):
        + single number (used to fill self.matrix)
        + matrix keyword:
            + AUTO_ASSIGN_MATRIX: IDENTITY_MATRIX if it is square, othwerwise FULL_CONNECTIVITY_MATRIX
            + IDENTITY_MATRIX: 1's on diagonal, 0's elsewhere (must be square matrix), otherwise generates error
            + HOLLOW_MATRIX: 0's on diagonal, 1's elsewhere (must be square matrix), otherwise generates error
            + FULL_CONNECTIVITY_MATRIX: all 1's
            + RANDOM_CONNECTIVITY_MATRIX (random floats uniformly distributed between 0 and 1)
        + 2D list or np.ndarray of numbers

     Returns 2D array with length=rows in dim 0 and length=cols in dim 1, or none if specification is not recognized
    """

    # Matrix provided (and validated in _validate_params); convert to array
    if isinstance(specification, (list, np.matrix)):
        specification = np.array(specification)

    if isinstance(specification, np.ndarray):
        if specification.ndim == 2:
            return specification
        # FIX: MAKE THIS AN np.array WITH THE SAME DIMENSIONS??
        elif specification.ndim < 2:
            return np.atleast_2d(specification)
        else:
            raise FunctionError("Specification of np.array for matrix ({}) is more than 2d".
                                format(specification))

    if specification == AUTO_ASSIGN_MATRIX:
        if rows == cols:
            specification = IDENTITY_MATRIX
        else:
            specification = FULL_CONNECTIVITY_MATRIX

    if specification == FULL_CONNECTIVITY_MATRIX:
        return np.full((rows, cols), 1.0)

    if specification == IDENTITY_MATRIX:
        if rows != cols:
            raise FunctionError("Sender length ({}) must equal receiver length ({}) to use {}".
                                format(rows, cols, specification))
        return np.identity(rows)

    if specification == HOLLOW_MATRIX:
        if rows != cols:
            raise FunctionError("Sender length ({}) must equal receiver length ({}) to use {}".
                                format(rows, cols, specification))
        return 1-np.identity(rows)

    if specification == RANDOM_CONNECTIVITY_MATRIX:
        return np.random.rand(rows, cols)

    # Function is specified, so assume it uses random.rand() and call with sender_len and receiver_len
    if isinstance(specification, function_type):
        return specification(rows, cols)

    # (7/12/17 CW) this is a PATCH (like the one in MappingProjection) to allow users to
    # specify 'matrix' as a string (e.g. r = RecurrentTransferMechanism(matrix='1 2; 3 4'))
    if type(specification) == str:
        try:
            return np.array(np.matrix(specification))
        except (ValueError, NameError, TypeError):
            # np.matrix(specification) will give ValueError if specification is a bad value (e.g. 'abc', '1; 1 2')
            #                          [JDC] actually gives NameError if specification is a string (e.g., 'abc')
            pass

    # Specification not recognized
    return None


