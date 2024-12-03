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
* `AcceleratingRise`
* `AcceleratingDecay`
* `DeceleratingRise`
* `DeceleratingDecay`

Overview
--------

Functions for which a `start <TimerFunction.start>`, `threshold <TimerFunction.threshold>`, and `end
<TimerFunction.end>` value can be specified, for use with a `TimerMechanism`.

.. _TimerFunction_StandardAttributes:

Standard Attributes
~~~~~~~~~~~~~~~~~~~

All TimerFunctions have the following attributes:

* **start**: specifies the `value <Function_Base.value>` that the function should have when its `variable
  <Function_Base.variable>` is 0.

* **threshold**: specifies the `value <Function_Base.value>` that the function should have when its `variable
  <Function_Base.variable>` is equal to `end <TimerFunction.end>`.

* **end**: specifies the value of the `variable <Function_Base.variable>` at which the`value <Function_Base.value>` of
    the function should be equal to `threshold <TimerFunction.threshold>`.

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
    (ADDITIVE_PARAM, ACCELERATING_DECAY_FUNCTION, ACCELERATING_RISE_FUNCTION, DECELERATING_DECAY_FUNCTION,
     DECELERATING_RISE_FUNCTION, END, LINEAR_DECAY_FUNCTION, LINEAR_RISE_FUNCTION,
     MULTIPLICATIVE_PARAM, OFFSET, PREFERENCE_SET_NAME, SCALE, START, THRESHOLD, TIMER_FUNCTION_TYPE)

__all__ = ['LinearRise','LinearDecay','AcceleratingRise','AcceleratingDecay','DeceleratingRise','DeceleratingDecay']


class TimerFunction(TransferFunction):  # --------------------------------------------------------------------------------
    """Subclass of TransferFunction that allows a start and end value to be specified.

    In addition to the required attributes of a `TransferFunction `,
    all TimerFunctions MUST have the following attributes:

    `start` -- specifies the `value <Function_Base.value>` that the function should have when its `variable
    <Function_Base.variable>` is 0.

    `threshold` -- specifies the `value <Function_Base.value>` that the function should have when its `variable
    <Function_Base.variable>` equals its `threshold <TimerFunction.threshold>`.

    `end` -- specifies the value of the `variable <Function_Base.variable>` at which the `value <Function_Base.value>`
    of the function should be equal to its `threshold <TimerFunction.threshold>`.

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

                threshold
                    see `threshold <TimerFunction.threshold>`

                    :default value: None
                    :type: 'float'
        """
        start = Parameter(1.0, modulable=True)
        threshold = Parameter(0.0, modulable=True)
        end = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])

        def _validate_start(self, start):
            if start <= 0:
                return f"must be greater than 0."

        def _validate_threshold(self, threshold):
            if threshold < 0:
                return f"must be greater than or equal to 0."

        def _validate_end(self, end):
            if end <= 0:
                return f"must be greater than 0."


class LinearRise(TimerFunction):
    pass


class LinearDecay(TimerFunction):
    pass


class AcceleratingRise(TimerFunction):
    """
    AcceleratingRise(     \
         default_variable, \
         start=0.0,        \
         threshold=1.0,    \
         end=1.0,          \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _AcceleratingRise:
    |
    `function <AcceleratingRise._function>` returns accelerating rise transform of `variable
    <AcceleratingRise.variable>`; this is the inverse of the `AcceleratingDecay` Function.

    .. math::
       start + (threshold - start) \\left(\\frac{variable + (end * e^{variable}) - end}{end*e^{end}}\\right)

    such that:

    .. math::
        value=start \ for\ variable=0

        value=threshold\ for\ variable=end

    where:

        **start** determines the value of the function when `variable <AcceleratingRise.variable>` = 0.

        **threshold** determines the value of the function when `variable <AcceleratingRise.variable>` = end.

        **end** determines the value of `variable <AcceleratingRise.variable>` at which the value of the function =
        threshold.

    `derivative <AcceleratingRise.derivative>` returns the derivative of the AcceleratingRise Function:

      .. math::
         (threshold - start) * \\left(\\frac{(1 + end * e^{variable})}{end * e^{end}}\\right)

    # FIX:
    See `graph <https://www.desmos.com/calculator/keo5d328gn>`_ for interactive plot of the function using `Desmos
    <https://www.desmos.com>`_.

    Arguments
    ---------

    default_variable : number or array : default class_defaults.variable
        specifies a template for the value to be transformed.

    start : float : default 1.0
        specifies the value the function should have when `variable <AcceleratingRise.variable>` = 0;
        must be greater than or equal to 0.

    threshold : float : default 1.0
        specifies the value the function should have when `variable <AcceleratingRise.variable>` = `end
        <TimerFunction.end>`; must be greater than `start <AcceleratingRise.start>`.

    end : float : default 1.0
        specifies the value of `variable <AcceleratingRise.variable>` at which the value of the function
        should equal `threshold <AcceleratingRise.threshold>`; must be greater than 0.

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
        determines the value of the function when `variable <AcceleratingRise.variable>` = 0.

    threshold : float
        determines the value of the function when `variable <AcceleratingRise.variable>` = `end <TimerFunction.end>`.

    end : float (>0)
        determines the value of `variable <AcceleratingRise.variable>` at which the value of the function is equal
        to `threshold <AcceleratingRise.threshold>`.

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

    componentName = ACCELERATING_RISE_FUNCTION

    classPreferences = {
        PREFERENCE_SET_NAME: 'AcceleratingRiseClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    }

    _model_spec_class_name_is_generic = True

    # FIX: REINSTATE Parameters AND VALIDATE START, THRESHOLD AND END

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 start: Optional[ValidParamSpecType] = None,
                 threshold: Optional[ValidParamSpecType] = None,
                 end: Optional[ValidParamSpecType] = None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):
        super().__init__(
            default_variable=default_variable,
            start=start,
            threshold=threshold,
            end=end,
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

        Accelerating rise transform of variable : number or array

        """
        start = self._get_current_parameter_value(START, context)
        threshold = self._get_current_parameter_value(THRESHOLD, context)
        end = self._get_current_parameter_value(END, context)

        result = start + (threshold - start) * ((variable + (end * np.exp(variable)) - end) / (end * np.exp(end)))

        return self.convert_output_type(result)

    @handle_external_context()
    def derivative(self, input, output=None, context=None):
        """Derivative of `function <AcceleratingRise._function>` at **input**:

        .. math::
           (threshold - start) * \\left(\\frac{(1 + end * e^{variable})}{end * e^{end}}\\right)

        Arguments
        ---------

        input : number
            value of the input to the AcceleratingRise transform at which derivative is to be taken.

        Returns
        -------
        derivative :  number or array
        """
        start = self._get_current_parameter_value(START, context)
        threshold = self._get_current_parameter_value(THRESHOLD, context)
        end = self._get_current_parameter_value(END, context)

        return (threshold - start) * (1 + end * np.exp(input) / end * np.exp(end))

    # FIX:
    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        end_ptr = ctx.get_param_or_state_ptr(builder, self, END, param_struct_ptr=params)
        start_ptr = ctx.get_param_or_state_ptr(builder, self, START, param_struct_ptr=params)
        threshold_ptr = ctx.get_param_or_state_ptr(builder, self, THRESHOLD, param_struct_ptr=params)
        offset_ptr = ctx.get_param_or_state_ptr(builder, self, OFFSET, param_struct_ptr=params)

        end = pnlvm.helpers.load_extract_scalar_array_one(builder, end_ptr)
        start = pnlvm.helpers.load_extract_scalar_array_one(builder, start_ptr)
        threshold = pnlvm.helpers.load_extract_scalar_array_one(builder, threshold_ptr)
        offset = pnlvm.helpers.load_extract_scalar_array_one(builder, offset_ptr)

        exp_f = ctx.get_builtin("exp", [ctx.float_ty])
        val = builder.load(ptri)
        val = builder.fmul(val, end)
        val = builder.fadd(val, start)
        val = builder.call(exp_f, [val])

        if "derivative" in tags:
            # f'(x) = s*r*e^(r*x + b)
            val = builder.fmul(val, threshold)
            val = builder.fmul(val, end)
        else:
            # f(x) = s*e^(r*x + b) + o
            val = builder.fmul(val, threshold)
            val = builder.fadd(val, offset)

        builder.store(val, ptro)

    # FIX:
    def _gen_pytorch_fct(self, device, context=None):
        start = self._get_pytorch_fct_param_value(START, device, context)
        end = self._get_pytorch_fct_param_value(END, device, context)
        k = self._get_pytorch_fct_param_value('k', device, context)

        return lambda x : start + start * torch.exp(-e)/torch.exp(end - e - k**end) * (1 - np.exp(x))


class AcceleratingDecay(TimerFunction): # ---------------------------------------------------------------------------
    """
    AcceleratingDecay(     \
         default_variable, \
         start=1.0,        \
         threshold=0.0,    \
         end=1.0,          \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _AcceleratingDecay:
    |
    `function <AcceleratingDecay._function>` returns accelerating decay transform of `variable
    <AcceleratingDecay.variable>`; this is the inverse of the `AcceleratingRise` Function.

    .. math::
       threshold + (start - threshold) \\left(1-\\frac{variable + (end * e^{variable}) - end}{end*e^{end}}\\right)

    such that:

    .. math::
        value=start \ for\ variable=0

        value=threshold\ for\ variable=end

    where:

        **start** determines the value of the function when `variable <AcceleratingDecay.variable>` = 0.

        **threshold** determines the value of the function when `variable <AcceleratingDecay.variable>` = end.

        **end** determines the value of `variable <AcceleratingDecay.variable>` at which the value of the function =
        threshold.

    `derivative <AcceleratingDecay.derivative>` returns the derivative of the AcceleratingDecay Function:

      .. math::
       (start - threshold) * \\left(1-\\frac{end*e^{variable}}{end*e^{end}}\\right)

    See `graph <https://www.desmos.com/calculator/keo5d328gn>`_ for interactive plot of the function using `Desmos
    <https://www.desmos.com>`_.

    Arguments
    ---------

    default_variable : number or array : default class_defaults.variable
        specifies a template for the value to be transformed.

    start : float : default 1.0
        specifies the value function should have when `variable <AcceleratingDecay.variable>` = 0;
        must be greater than 0.

    threshold : float : default 1.0
        specifies the value the function should have when `variable <AcceleratingDecay.variable>` = `end
        <TimerFunction.end>`; must be greater than or equal to 0.

    end : float : default 1.0
        specifies the value of `variable <AcceleratingDecay.variable>` at which the value of the function
        should equal `threshold <AcceleratingDecay.threshold>`; must be greater than 0.

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
        determines the value of the function when `variable <AcceleratingDecay.variable>` = 0.

    threshold : float
        determines the value of the function when `variable <AcceleratingDecay.variable>` = `end <TimerFunction.end>`.

    end : float (>0)
        determines the value of `variable <AcceleratingDecay.variable>` at which the value of the function is equal
        to `threshold <AcceleratingDecay.threshold>`.

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

    componentName = ACCELERATING_DECAY_FUNCTION

    classPreferences = {
        PREFERENCE_SET_NAME: 'AcceleratingDecayClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    }

    _model_spec_class_name_is_generic = True

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 start: Optional[ValidParamSpecType] = None,
                 threshold: Optional[ValidParamSpecType] = None,
                 end: Optional[ValidParamSpecType] = None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):
        super().__init__(
            default_variable=default_variable,
            start=start,
            threshold=threshold,
            end=end,
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
        threshold = self._get_current_parameter_value(THRESHOLD, context)
        end = self._get_current_parameter_value(END, context)

        # result = start * (1 - ((np.exp(variable) - 1) / np.exp(end)) - (variable / end * np.exp(end)))
        result = (threshold + (start - threshold) *
                  (1 - (variable + (end * np.exp(variable)) - end) / (end * np.exp(end))))

        return self.convert_output_type(result)

    @handle_external_context()
    def derivative(self, input, output=None, context=None):
        """Derivative of `function <AcceleratingDecay._function>` at **input**:

        .. math::
           (start - threshold) * \\left(1-\\frac{end*e^{variable}}{end*e^{end}}\\right)

        Arguments
        ---------

        input : number
            value of the input to the AcceleratingDecay transform at which derivative is to be taken.

        Returns
        -------
        derivative :  number or array
        """
        start = self._get_current_parameter_value(START, context)
        threshold = self._get_current_parameter_value(THRESHOLD, context)
        end = self._get_current_parameter_value(END, context)

        return (start - threshold) * -(np.exp(input) / np.exp(end)) - (1 / end * np.exp(end))

    # FIX:
    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        end_ptr = ctx.get_param_or_state_ptr(builder, self, END, param_struct_ptr=params)
        start_ptr = ctx.get_param_or_state_ptr(builder, self, START, param_struct_ptr=params)
        threshold_ptr = ctx.get_param_or_state_ptr(builder, self, THRESHOLD, param_struct_ptr=params)
        offset_ptr = ctx.get_param_or_state_ptr(builder, self, OFFSET, param_struct_ptr=params)

        end = pnlvm.helpers.load_extract_scalar_array_one(builder, end_ptr)
        start = pnlvm.helpers.load_extract_scalar_array_one(builder, start_ptr)
        threshold = pnlvm.helpers.load_extract_scalar_array_one(builder, threshold_ptr)
        offset = pnlvm.helpers.load_extract_scalar_array_one(builder, offset_ptr)

        exp_f = ctx.get_builtin("exp", [ctx.float_ty])
        val = builder.load(ptri)
        val = builder.fmul(val, end)
        val = builder.fadd(val, start)
        val = builder.call(exp_f, [val])

        if "derivative" in tags:
            # f'(x) = s*r*e^(r*x + b)
            val = builder.fmul(val, threshold)
            val = builder.fmul(val, end)
        else:
            # f(x) = s*e^(r*x + b) + o
            val = builder.fmul(val, threshold)
            val = builder.fadd(val, offset)

        builder.store(val, ptro)

    # FIX:
    def _gen_pytorch_fct(self, device, context=None):
        start = self._get_pytorch_fct_param_value(START, device, context)
        end = self._get_pytorch_fct_param_value(END, device, context)
        k = self._get_pytorch_fct_param_value('k', device, context)

        return lambda x : start + start * torch.exp(-e)/torch.exp(end - e - k**end) * (1 - np.exp(x))


class DeceleratingRise(TimerFunction):
    pass


class DeceleratingDecay(TimerFunction):  # ---------------------------------------------------------------------------
    """
    DeceleratingDecay(      \
         default_variable, \
         start=1.0,        \
         offset=0.0,       \
         end=1.0,          \
         threshold=0.01,   \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _DeceleratingDecay:
    |
    `function <DeceleratingDecay._function>` returns exponentially decaying transform of `variable
    <DeceleratingDecay.variable>`

    .. math::
       offset + start*e^{-\\frac{variable\ *\ \\ln\\left(\\frac{1}{threshold}\\right)}{end}}

    such that:

    .. math::
        value = start + offset\ for\ variable=0

        value = (start * threshold) + offset\ for\ variable=end

    where:

        **start**, together with `offset <DeceleratingDecay.offset>`, determines the value of the function when
        `variable <DeceleratingDecay.variable>` = 0, and is used together with `threshold <DeceleratingDecay.threshold>`
        to determine the value of the function when `variable <DeceleratingDecay.variable>` = `end
        <DeceleratingDecay.end>`.

        **offset**, together with `start <DeceleratingDecay.start>`, determines the value of the function
        when `variable <DeceleratingDecay.variable>` = 0, and its linear offset from 0 for all other values;

        **end** determines the value of `variable <DeceleratingDecay.variable>` at which
        the value of the function should equal :math:`start * threshold + offset`.

        **threshold** is the fraction of `start <DeceleratingDecay.start>` when, added to `offset
        <DeceleratingDecay.offset>`, is used to determine the value of the function when `variable
        <DeceleratingDecay.variable>` should equal `end <DeceleratingDecay.end>`.

    `derivative <DeceleratingDecay.derivative>` returns the derivative of the DeceleratingDecay Function:

      .. math::
        \\frac{start * \\ln\\left(\\frac{1}{threshold}\\right) *
        e^{-\\frac{variable * \\ln\\left(\\frac{1}{threshold}\\right)}{end}}}{end}

    COMMENT:
    FOR TIMER VERSION:
    `function <DeceleratingDecay._function>` returns exponentially decaying transform of `variable
    <DeceleratingDecay.variable>`, that has a value of `start <DeceleratingDecay.start>` + `offset
    <DeceleratingDecay.offset>` at `variable <DeceleratingDecay.variable>` = 0, and a value of `threshold
    <DeceleratingDecay.end>` * `start <DeceleratingDecay.start>` + `offset <DeceleratingDecay.offset>` at
    `variable at `variable <DeceleratingDecay.variable>` = `end <DeceleratingDecay.end>`:
    COMMENT

    Arguments
    ---------

    default_variable : number or array : default class_defaults.variable
        specifies a template for the value to be transformed.

    start : float : default 1.0
        specifies, together with `offset <DeceleratingDecay.offset>`, the value of the function when `variable
        <DeceleratingDecay.variable>` = 0; must be greater than 0.

    offset : float : default 0.0
        specifies, together with `start <DeceleratingDecay.start>`, the value of the function when `variable
        <DeceleratingDecay.variable>` = 0, and its linear offset for all other values.

    end : float : default 1.0
        specifies the value of `variable <DeceleratingDecay.variable>` at which the `value of the function
        should equal `start <DeceleratingDecay.start>` * `threshold <DeceleratingDecay.threshold>` + `offset
        <DeceleratingDecay.offset>`; must be greater than 0.

    threshold : float : default 0.01
        specifies the fraction of `start <DeceleratingDecay.start>` when added to `offset <DeceleratingDecay.offset>`,
        that determines the value of the function when `variable <DeceleratingDecay.variable>` = `end
        <DeceleratingDecay.end>`; must be between 0 and 1.

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
        determines, together with `offset <DeceleratingDecay.offset>`, the value of the function when `variable
        <DeceleratingDecay.variable>` = 0.

    offset : float
        determines, together with `start <DeceleratingDecay.start>`, the value of the function when `variable
        <DeceleratingDecay.variable>` = 0, and its linear offset for all other values.

    end : float (>0)
        determines the value of `variable <DeceleratingDecay.variable>` at which the value of the function should
        equal `start <DeceleratingDecay.start>` * `threshold <DeceleratingDecay.threshold>` + `offset <DeceleratingDecay.offset>`.

    threshold : float (0,1)
        determines the fraction of `start <DeceleratingDecay.start>` when added to `offset <DeceleratingDecay.offset>`,
        that determines the value of the function when `variable <DeceleratingDecay.variable>` = `end
        <DeceleratingDecay.end>`.

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

    componentName = DECELERATING_DECAY_FUNCTION

    classPreferences = {
        PREFERENCE_SET_NAME: 'DeceleratingDecayClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    }

    _model_spec_class_name_is_generic = True

    class Parameters(TimerFunction.Parameters):
        """
            Attributes
            ----------

                offset
                    see `offset <DeceleratingDecay.offset>`

                    :default value: 0.0
                    :type: ``float``

                threshold
                    see `threshold <DeceleratingDecay.threshold>`

                    :default value: 0.01
                    :type: ``float``
        """
        offset = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        threshold = Parameter(0.01, modulable=True, aliases=[SCALE])

        def _validate_threshold(self, threshold):
            if threshold < 0:
                return f"must be between 0 and 1."

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 start: Optional[ValidParamSpecType] = None,
                 offset: Optional[ValidParamSpecType] = None,
                 end: Optional[ValidParamSpecType] = None,
                 threshold: Optional[ValidParamSpecType] = None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):
        super().__init__(
            default_variable=default_variable,
            start=start,
            offset=offset,
            end=end,
            threshold=threshold,
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
           by the value of `increment <DeceleratingDecay.increment>`.

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
        threshold = self._get_current_parameter_value(THRESHOLD, context)

        result = offset + start * np.exp(-variable * np.log(1 / threshold) / end)

        return self.convert_output_type(result)

    @handle_external_context()
    def derivative(self, input, output=None, context=None):
        """Derivative of `function <DeceleratingDecay._function>` at **input**:

        .. math::
           \\frac{start * \\ln\\left(\\frac{1}{threshold}\\right) * e^{-\\left(\\frac{variable\\ln\\left(\\frac{1}{
           threshold}\\right)}{end}\\right)}}{end}

        Arguments
        ---------

        input : number
            value of the input to the DeceleratingDecay transform at which derivative is to be taken.

        Derivative of `function <DeceleratingDecay._function>` at **input**.

        Returns
        -------
        derivative :  number or array
        """

        start = self._get_current_parameter_value(START, context)
        end = self._get_current_parameter_value(END, context)
        threshold = self._get_current_parameter_value(THRESHOLD, context)

        return (start * np.log(1/threshold) / end) * np.exp(-input * np.log(threshold) / end)

    # FIX:
    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        end_ptr = ctx.get_param_or_state_ptr(builder, self, END, param_struct_ptr=params)
        start_ptr = ctx.get_param_or_state_ptr(builder, self, START, param_struct_ptr=params)
        threshold_ptr = ctx.get_param_or_state_ptr(builder, self, THRESHOLD, param_struct_ptr=params)
        offset_ptr = ctx.get_param_or_state_ptr(builder, self, OFFSET, param_struct_ptr=params)

        end = pnlvm.helpers.load_extract_scalar_array_one(builder, end_ptr)
        start = pnlvm.helpers.load_extract_scalar_array_one(builder, start_ptr)
        threshold = pnlvm.helpers.load_extract_scalar_array_one(builder, threshold_ptr)
        offset = pnlvm.helpers.load_extract_scalar_array_one(builder, offset_ptr)

        exp_f = ctx.get_builtin("exp", [ctx.float_ty])
        val = builder.load(ptri)
        val = builder.fmul(val, end)
        val = builder.fadd(val, start)
        val = builder.call(exp_f, [val])

        if "derivative" in tags:
            # f'(x) = s*r*e^(r*x + b)
            val = builder.fmul(val, threshold)
            val = builder.fmul(val, end)
        else:
            # f(x) = s*e^(r*x + b) + o
            val = builder.fmul(val, threshold)
            val = builder.fadd(val, offset)

        builder.store(val, ptro)

    def _gen_pytorch_fct(self, device, context=None):
        offset = self._get_pytorch_fct_param_value(OFFSET, device, context)
        end = self._get_pytorch_fct_param_value(END, device, context)
        threshold = self._get_pytorch_fct_param_value(THRESHOLD, device, context)
        start = self._get_pytorch_fct_param_value(START, device, context)

        return lambda x : offset + start * torch.exp(-x * torch.log(1 / threshold) / end)


