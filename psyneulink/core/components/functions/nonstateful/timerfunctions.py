#
# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *******************************************  TIMER FUNCTIONS  *****************************************************
"""

* `LinearRise`
* `LinearDecay`
* `ExponentialRise`
* `ExponentialDecay`
* `LogarithmicRise`
* `LogarithmicDecay`

Overview
--------

Functions for which a `start <TimerFunction.start>` and `end <TimerFunction.end>` value can be specified, for
use with a `TimerMechanism`.

.. _TimerFunction_StandardAttributes:

Standard Attributes
~~~~~~~~~~~~~~~~~~~

All TimerFunctions have the following attributes:

* **start**: specifies the `value <Function_Base.value>` that the function should have when its `variable
  <Function_Base.variable>` is 0.

* **end**: specifies the value of the `variable <Function_Base.variable>` at which the`value <Function_Base.value>` of
    the function should be equal to 0.

TimerFunction Class References
------------------------------

"""

from math import e

import numpy as np
try:
    import torch
except ImportError:
    torch = None
from beartype import beartype

from psyneulink._typing import Optional, Union

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.functions.function import FunctionError
from psyneulink.core.components.functions.nonstateful.transferfunctions import TransferFunction
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.parameters import \
    FunctionParameter, Parameter, get_validator_by_function, check_user_specified, copy_parameter_value
from psyneulink.core.globals.utilities import (
    ValidParamSpecType, convert_all_elements_to_np_array, safe_len, is_matrix_keyword)
from psyneulink.core.globals.preferences.basepreferenceset import \
    REPORT_OUTPUT_PREF, PreferenceEntry, PreferenceLevel, ValidPrefSet
from psyneulink.core.globals.keywords import \
    (ADDITIVE_PARAM, END, EXPONENTIAL_DECAY_FUNCTION, EXPONENTIAL_RISE_FUNCTION,
     LINEAR_RISE_FUNCTION, LINEAR_DECAY_FUNCTION, LOGARITHMIC_DECAY_FUNCTION, MULTIPLICATIVE_PARAM,
     PREFERENCE_SET_NAME, SCALE, START, TIMER_FUNCTION_TYPE, TOLERANCE)

__all__ = ['ExponentialDecay', 'ExponentialRise', 'LinearDecay', 'LinearRise', 'LogarithmicDecay', 'LogarithmicRise']


class TimerFunction(TransferFunction):  # --------------------------------------------------------------------------------
    """Subclass of TransferFunction that allows a start and end value to be specified.

    In addition to the required attributes of a `TransferFunction `,
    all TimerFunctions MUST have the following attributes:

    `start` -- specifies the `value <Function_Base.value>` that the function should have when its `variable
    <Function_Base.variable>` is 0.

    `end` -- specifies the value of the `variable <Function_Base.variable>` at which the`value <Function_Base.value>`
    of the function should be equal to 0.

    """
    componentType = TIMER_FUNCTION_TYPE

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                end
                    see `end <TimerFunction.start>`

                    :default value: None
                    :type: 'float'

                start
                    see `start <TimerFunction.start>`

                    :default value: None
                    :type: 'float'
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


class ExponentialDecay(TransferFunction):  # ---------------------------------------------------------------------------
    """
    ExponentialDecay(      \
         default_variable, \
         start=1.0,        \
         offset=0.0,       \
         end=1.0,          \
         tolerance=0.01,   \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _ExponentialDecay:
    |
    `function <ExponentialDecay._function>` returns exponentially decaying transform of `variable
    <ExponentialDecay.variable>`

    .. math::
       offset + start*e^{-\\frac{variable\ *\ \\ln\\left(\\frac{1}{tolerance}\\right)}{end}}

    such that:

    .. math::
        value = start + offset\ for\ variable=0

        value = (start * tolerance) + offset\ for\ variable=end

    where:

        **start**, together with `offset <ExponentialDecay.offset>`, determines the value of the function when
        `variable <ExponentialDecay.variable>` = 0, and is used together with `tolerance <ExponentialDecay.tolerance>`
        to determine the value of the function when `variable <ExponentialDecay.variable>` = `end
        <ExponentialDecay.end>`.

        **offset**, together with `start <ExponentialDecay.start>`, determines the value of the function
        when `variable <ExponentialDecay.variable>` = 0, and its linear offset from 0 for all other values;

        **end** determines the value of `variable <ExponentialDecay.variable>` at which
        the value of the function should equal :math:`start * tolerance + offset`.

        **tolerance** is the fraction of `start <ExponentialDecay.start>` when, added to `offset
        <ExponentialDecay.offset>`, is used to determine the value of the function when `variable
        <ExponentialDecay.variable>` should equal `end <ExponentialDecay.end>`.

    `derivative <ExponentialDecay.derivative>` returns the derivative of the ExponentialDecay Function:

      .. math::
        \\frac{start * \\ln\\left(\\frac{1}{tolerance}\\right) *
        e^{-\\frac{variable * \\ln\\left(\\frac{1}{tolerance}\\right)}{end}}}{end}

    COMMENT:
    FOR TIMER VERSION:
    `function <ExponentialDecay._function>` returns exponentially decaying transform of `variable
    <ExponentialDecay.variable>`, that has a value of `start <ExponentialDecay.start>` + `offset
    <ExponentialDecay.offset>` at `variable <ExponentialDecay.variable>` = 0, and a value of `threshold
    <ExponentialDecay.end>` * `start <ExponentialDecay.start>` + `offset <ExponentialDecay.offset>` at
    `variable at `variable <ExponentialDecay.variable>` = `end <ExponentialDecay.end>`:
    COMMENT

    Arguments
    ---------

    default_variable : number or array : default class_defaults.variable
        specifies a template for the value to be transformed.

    start : float : default 1.0
        specifies, together with `offset <ExponentialDecay.offset>`, the value of the function when `variable
        <ExponentialDecay.variable>` = 0; must be greater than 0.

    offset : float : default 0.0
        specifies, together with `start <ExponentialDecay.start>`, the value of the function when `variable
        <ExponentialDecay.variable>` = 0, and its linear offset for all other values.

    end : float : default 1.0
        specifies the value of `variable <ExponentialDecay.variable>` at which the `value of the function
        should equal `start <ExponentialDecay.start>` * `tolerance <ExponentialDecay.tolerance>` + `offset
        <ExponentialDecay.offset>`; must be greater than 0.

    tolerance : float : default 0.01
        specifies the fraction of `start <ExponentialDecay.start>` when added to `offset <ExponentialDecay.offset>`,
        that determines the value of the function when `variable <ExponentialDecay.variable>` = `end
        <ExponentialDecay.end>`; must be between 0 and 1.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters
        in arguments of the constructor.

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

    start : float (>0)
        determines, together with `offset <ExponentialDecay.offset>`, the value of the function when `variable
        <ExponentialDecay.variable>` = 0.

    offset : float
        determines, together with `start <ExponentialDecay.start>`, the value of the function when `variable
        <ExponentialDecay.variable>` = 0, and its linear offset for all other values.

    end : float (>0)
        determines the value of `variable <ExponentialDecay.variable>` at which the value of the function should
        equal `start <ExponentialDecay.start>` * `tolerance <ExponentialDecay.tolerance>` + `offset <ExponentialDecay.offset>`.

    tolerance : float (0,1)
        determines the fraction of `start <ExponentialDecay.start>` when added to `offset <ExponentialDecay.offset>`,
        that determines the value of the function when `variable <ExponentialDecay.variable>` = `end
        <ExponentialDecay.end>`.

    bounds : (None, None)

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

    componentName = EXPONENTIAL_DECAY_FUNCTION

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                start
                    see `start <ExponentialDecay.start>`

                    :default value: 1.0
                    :type: ``float``

                offset
                    see `offset <ExponentialDecay.offset>`

                    :default value: 0.0
                    :type: ``float``

                end
                    see `end <ExponentialDecay.end>`

                    :default value: 1.0
                    :type: ``float``

                tolerance
                    see `tolerance <ExponentialDecay.tolerance>`

                    :default value: 0.01
                    :type: ``float``
        """
        start = Parameter(1.0, modulable=True)
        offset = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        end = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        tolerance = Parameter(0.01, modulable=True, aliases=[SCALE])
        bounds = (None, None)

        def _validate_start(self, start):
            if start < 0:
                return f"must be greater than 0."

        def _validate_end(self, end):
            if end < 0:
                return f"must be greater than 0."

        def _validate_tolerance(self, tolerance):
            if tolerance < 0:
                return f"must be between 0 and 1."

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 start: Optional[ValidParamSpecType] = None,
                 offset: Optional[ValidParamSpecType] = None,
                 end: Optional[ValidParamSpecType] = None,
                 tolerance: Optional[ValidParamSpecType] = None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):
        super().__init__(
            default_variable=default_variable,
            start=start,
            offset=offset,
            end=end,
            tolerance=tolerance,
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
           amount by which to increment timer on current execution;  if this is not specified, the timer is incremented
           by the value of `increment <ExponentialDecay.increment>`.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        Exponentially decayed value of variable : number or array

        """
        start = self._get_current_parameter_value(START, context)
        offset = self._get_current_parameter_value(OFFSET, context)
        end = self._get_current_parameter_value(END, context)
        tolerance = self._get_current_parameter_value(TOLERANCE, context)

        result = offset + start * np.exp(-variable * np.log(1 / tolerance) / end)

        return self.convert_output_type(result)

    @handle_external_context()
    def derivative(self, input, output=None, context=None):
        """
        derivative(input)
        .. math::
            \frac{start\ln\left(\frac{1}{tolerance}\right)e^{-\left(\frac{variable\ln
            \left(\frac{1}{tolerance}\right)}{end}\right)}}{end}

        Arguments
        ---------

        input : number
            value of the input to the ExponentialDecay transform at which derivative is to be taken.

        Derivative of `function <ExponentialDecay._function>` at **input**.

        Returns
        -------
        derivative :  number or array
        """

        start = self._get_current_parameter_value(START, context)
        end = self._get_current_parameter_value(END, context)
        tolerance = self._get_current_parameter_value(TOLERANCE, context)

        return (start * np.log(1/tolerance) / end) * np.exp(-input * np.log(tolerance) / end)

    # FIX:
    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        end_ptr = ctx.get_param_or_state_ptr(builder, self, END, param_struct_ptr=params)
        start_ptr = ctx.get_param_or_state_ptr(builder, self, START, param_struct_ptr=params)
        tolerance_ptr = ctx.get_param_or_state_ptr(builder, self, TOLERANCE, param_struct_ptr=params)
        offset_ptr = ctx.get_param_or_state_ptr(builder, self, OFFSET, param_struct_ptr=params)

        end = pnlvm.helpers.load_extract_scalar_array_one(builder, end_ptr)
        start = pnlvm.helpers.load_extract_scalar_array_one(builder, start_ptr)
        tolerance = pnlvm.helpers.load_extract_scalar_array_one(builder, tolerance_ptr)
        offset = pnlvm.helpers.load_extract_scalar_array_one(builder, offset_ptr)

        exp_f = ctx.get_builtin("exp", [ctx.float_ty])
        val = builder.load(ptri)
        val = builder.fmul(val, end)
        val = builder.fadd(val, start)
        val = builder.call(exp_f, [val])

        if "derivative" in tags:
            # f'(x) = s*r*e^(r*x + b)
            val = builder.fmul(val, tolerance)
            val = builder.fmul(val, end)
        else:
            # f(x) = s*e^(r*x + b) + o
            val = builder.fmul(val, tolerance)
            val = builder.fadd(val, offset)

        builder.store(val, ptro)

    def _gen_pytorch_fct(self, device, context=None):
        offset = self._get_pytorch_fct_param_value(OFFSET, device, context)
        end = self._get_pytorch_fct_param_value(END, device, context)
        tolerance = self._get_pytorch_fct_param_value(TOLERANCE, device, context)
        start = self._get_pytorch_fct_param_value(START, device, context)

        return lambda x : offset + start * torch.exp(-x * torch.log(1 / tolerance) / end)


class LogarithmicDecay(TransferFunction): # ---------------------------------------------------------------------------
    """
    LogarithmicDecay(     \
         default_variable, \
         start=1.0,        \
         end=1.0,          \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _LogarithmicDecay:
    |
    `function <LogarithmicDecay._function>` returns exponentially decaying transform of `variable
    <LogarithmicDecay.variable>`

    .. math::
       start + start \\frac{e^{-e}}{e^{\\left(end-e-k^{end}\\right)}} \\left(1-e^{variable}\\right)

    such that:

    .. math::
        value = start \ for\ variable=0

        value = 0\ for\ variable=end

    where:

        **start** determines the value of the function when `variable <LogarithmicDecay.variable>` = 0.

        **end** determines the value of `variable <LogarithmicDecay.variable>` at which the value of the function = 0.

        **k** is a constant used to enforce that the value of the function when `variable
        <LogarithmicDecay.variable>` = `end <LogarithmicDecay.end>` is as close to 0 as possible.

    `derivative <LogarithmicDecay.derivative>` returns the derivative of the LogarithmicDecay Function:

      .. math::
       - start \\frac{e^{\\left(variable-e\\right)}}{e^{\\left(end-e-k^{end}\\right)}}

    Arguments
    ---------

    default_variable : number or array : default class_defaults.variable
        specifies a template for the value to be transformed.

    start : float : default 1.0
        specifies the value function should have when `variable <LogarithmicDecay.variable>` = 0;
        must be greater than 0.

    end : float : default 1.0
        specifies the value of `variable <LogarithmicDecay.variable>` at which the value of the function
        should equal 0; must be greater than 0.

    k : float : default 0.4
        specifies the constant used to ensure that value of the function when `variable
        <LogarithmicDecay.variable>` = `end <LogarithmicDecay.end>` is as close to 0 as possible;
        must be beetween 0 and 1.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters
        in arguments of the constructor.

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

    start : float (>0)
        determines the value the function should have when `variable <LogarithmicDecay.variable>` = 0.

    end : float (>0)
        determines the value of `variable <LogarithmicDecay.variable>` at which the value of the function equals 0.

    k : float (0,1)
        determines the constant used to ensure that value of the function when `variable
        <LogarithmicDecay.variable>` = `end <LogarithmicDecay.end>` is as close to 0 as possible.

    bounds : (None, None)

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

    componentName = LOGARITHMIC_DECAY_FUNCTION

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                start
                    see `start <LogarithmicDecay.start>`

                    :default value: 1.0
                    :type: ``float``

                end
                    see `end <LogarithmicDecay.end>`

                    :default value: 1.0
                    :type: ``float``

                k
                    see `end <LogarithmicDecay.k>`

                    :default value: 0.4
                    :type: ``float``

        """
        start = Parameter(1.0, modulable=True, aliases=[ADDITIVE_PARAM])
        end = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        k = Parameter(0.4, modulable=True)
        bounds = (None, None)

        def _validate_start(self, start):
            if start < 0:
                return f"must be greater than 0."

        def _validate_end(self, end):
            if end < 0:
                return f"must be greater than 0."
        def _validate_k(self, k):
            if k <= 0 or k >= 1:
                return f"must be greater than 0 and less than 1."

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 start: Optional[ValidParamSpecType] = None,
                 end: Optional[ValidParamSpecType] = None,
                 k: Optional[ValidParamSpecType] = None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):
        super().__init__(
            default_variable=default_variable,
            start=start,
            end=end,
            k=k,
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

        Exponentially decayed transformation of variable : number or array

        """
        start = self._get_current_parameter_value(START, context)
        end = self._get_current_parameter_value(END, context)
        k = self._get_current_parameter_value('k', context)

        result = start + start * np.exp(-e)/np.exp(end - e - k**end) * (1 - np.exp(variable))

        return self.convert_output_type(result)

    @handle_external_context()
    def derivative(self, input, output=None, context=None):
        """
        derivative(input)
        .. math::
         - start \\frac{e^{\\left(variable-e\\right)}}{e^{\\left(end-e-k^{end}\\right)}}

        Arguments
        ---------

        input : number
            value of the input to the LogarithmicDecay transform at which derivative is to be taken.

        Derivative of `function <LogarithmicDecay._function>` at **input**.

        Returns
        -------
        derivative :  number or array
        """

        start = self._get_current_parameter_value(START, context)
        end = self._get_current_parameter_value(END, context)
        k = self._get_current_parameter_value('k', context)

        return -start * np.exp(input - e) / np.exp(end - e - k**end)

    # FIX:
    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        end_ptr = ctx.get_param_or_state_ptr(builder, self, END, param_struct_ptr=params)
        start_ptr = ctx.get_param_or_state_ptr(builder, self, START, param_struct_ptr=params)
        tolerance_ptr = ctx.get_param_or_state_ptr(builder, self, TOLERANCE, param_struct_ptr=params)
        offset_ptr = ctx.get_param_or_state_ptr(builder, self, OFFSET, param_struct_ptr=params)

        end = pnlvm.helpers.load_extract_scalar_array_one(builder, end_ptr)
        start = pnlvm.helpers.load_extract_scalar_array_one(builder, start_ptr)
        tolerance = pnlvm.helpers.load_extract_scalar_array_one(builder, tolerance_ptr)
        offset = pnlvm.helpers.load_extract_scalar_array_one(builder, offset_ptr)

        exp_f = ctx.get_builtin("exp", [ctx.float_ty])
        val = builder.load(ptri)
        val = builder.fmul(val, end)
        val = builder.fadd(val, start)
        val = builder.call(exp_f, [val])

        if "derivative" in tags:
            # f'(x) = s*r*e^(r*x + b)
            val = builder.fmul(val, tolerance)
            val = builder.fmul(val, end)
        else:
            # f(x) = s*e^(r*x + b) + o
            val = builder.fmul(val, tolerance)
            val = builder.fadd(val, offset)

        builder.store(val, ptro)

    def _gen_pytorch_fct(self, device, context=None):
        start = self._get_pytorch_fct_param_value(START, device, context)
        end = self._get_pytorch_fct_param_value(END, device, context)
        k = self._get_pytorch_fct_param_value('k', device, context)

        return lambda x : start + start * torch.exp(-e)/torch.exp(end - e - k**end) * (1 - np.exp(x))


class ExponentialRise(TransferFunction):
    pass


class LogarithmicRise(TransferFunction):
    pass


class LinearRise(TransferFunction):
    pass


class LinearDecay(TransferFunction):
    pass
