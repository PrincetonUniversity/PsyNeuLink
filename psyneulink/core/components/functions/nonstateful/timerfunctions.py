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

* `TimerFunction`
* `LinearTimer`
* `AcceleratingTimer`
* `DeceleratingTimer`
* `AsymptoticTimer`

Overview
--------

Functions for which `initial <TimerFunction.initial>` and `final <TimerFunction.final>` values and a `duration
<TimerFunction.duration>` can be specified, for use with a `TimerMechanism`.

.. _TimerFunction_Types:

Types
~~~~~

There are four types that implement different functional forms, each of which is rising if `initial
<TimerFunction.initial>` is less than `final <TimerFunction.final>` and declining for the reverse:

* **LinearTimer** - progresses linearly from `initial <TimerFunction.initial>` to `final <TimerFunction.final>` value.
  (see `interactive graph <https://www.desmos.com/calculator/i0knnnozcs>`_).

* **AcceleratingTimer** - advances from initial <TimerFunction.initial>` to `final <TimerFunction.final>` value
  by progressively larger amounts at an adjustable exponential `rate <AcceleratingTimerRise.rate>`
  (see `interactive graph <https://www.desmos.com/calculator/rms6z2ji8g>`_).

* **DeceleratingTimer** - advances from initial <TimerFunction.initial>` to `final <TimerFunction.final>` value
  by progressively smaller amounts at an adjustable exponential `rate <DeceleratingTimer.rate>`
  (see `interactive graph <https://www.desmos.com/calculator/cshkzip0ai>`_).

* **AsymptoticTimer** - progresses at a fixed exponential `rate <AsymptoticTimer.rate>` from `initial
  <TimerFunction.initial>` to within `tolerance <AsymptoticTimer.tolerance>` of `final <TimerFunction.final>`
    (see `interactive graph <https://www.desmos.com/calculator/tmfs4ps9cp>`_).


.. _TimerFunction_StandardAttributes:

Standard Attributes
~~~~~~~~~~~~~~~~~~~

TimerFunctions have the following Parameters:

.. _TimerFunction_Initial:
* **initial**: specifies the `value <Function_Base.value>` that the function has when its `variable
  <Function_Base.variable>` is 0.

.. _TimerFunction_Final:
* **final**: specifies the `value <Function_Base.value>` that the function has when its `variable
  <Function_Base.variable>` is equal to `duration <TimerFunction.duration>`.

.. _TimerFunction_Duration:
* **duration**: specifies the value of the `variable <Function_Base.variable>` at which the`value
  <Function_Base.value>` of the function is equal to `final <TimerFunction.final>`.

.. _TimerFunction_Rate:
* **rate**: specifies the rate at which the progression of the `value <Function_Base.value>` of the function changes.

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

from psyneulink._typing import Optional

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.functions.nonstateful.transferfunctions import TransferFunction
from psyneulink.core.globals.context import handle_external_context
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.utilities import (ValidParamSpecType)
from psyneulink.core.globals.preferences.basepreferenceset import \
    REPORT_OUTPUT_PREF, PreferenceEntry, PreferenceLevel, ValidPrefSet
from psyneulink.core.globals.keywords import \
    (ADDITIVE_PARAM, ACCELERATING_TIMER_FUNCTION, ASYMPTOTIC_TIMER_FUNCTION, DECELERATING_TIMER_FUNCTION,
     DURATION, FINAL, INITIAL, LINEAR_TIMER_FUNCTION, MULTIPLICATIVE_PARAM, OFFSET, PREFERENCE_SET_NAME,
     RATE, TIMER_FUNCTION, TIMER_FUNCTION_TYPE, TOLERANCE)

__all__ = ['LinearTimer','AcceleratingTimer','DeceleratingTimer','AsymptoticTimer']


class TimerFunction(TransferFunction):  # --------------------------------------------------------------------------------
    """Subclass of TransferFunction that allows a initial, final and duration value to be specified;
    for use with a `TimerMechanism`.

    Attributes
    ----------

    variable : number or array
        contains value to be transformed.

    initial : float (>0)
        determines the value of the function when `variable <TimerFunction.variable>` = 0.

    final : float
        determines the value of the function when `variable <TimerFunction.variable>` = `duration <TimerFunction.duration>`.

    duration : float (>0)
        determines the value of `variable <TimerFunction.variable>` at which the value of the function is equal
        to `final <TimerFunction.final>`.

    rate : float (>1.0)
        determines the rate at which the value of the function accelerates.

    """
    componentType = TIMER_FUNCTION_TYPE
    componentName = TIMER_FUNCTION

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                duration
                    see `duration <TimerFunction.initial>`

                    :default value: None
                    :type: 'float'

                final
                    see `final <TimerFunction.final>`

                    :default value: None
                    :type: 'float'

                initial
                    see `initial <TimerFunction.initial>`

                    :default value: None
                    :type: 'float'

                rate
                    see `rate <TimerFunction.rate>`

                    :default value: None
                    :type: 'float'
        """
        initial = Parameter(1.0, modulable=True)
        final = Parameter(0.0, modulable=True)
        duration = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        rate = Parameter(1.0, modulable=True)

        def _validate_duration(self, duration):
            if duration <= 0:
                return f"must be greater than 0."

        def _validate_rate(self, rate):
            if rate < 1:
                return f"must be greater than or equal to 1.0."


class LinearTimer(TimerFunction):
    """LinearTimer(        \
         default_variable, \
         initial=0.0,      \
         final=1.0,        \
         duration=1.0,     \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _LinearTimer:
    |
    `function <LinearTimer._function>` returns linear transform of `variable <LinearTimer.variable>`.

    .. math::
       \\left(\\frac{final-initial}{duration}\\right) \\cdot variable + initial

    such that:

    .. math::
        value=initial \\ for\\ variable=0

        value=final\\ for\\ variable=duration

    where:

        **initial** determines the `value <Function_Base.value>` of the function
        when its `variable <LinearTimer.variable>` = 0.

        **final** determines the `value <Function_Base.value>` of the function
        when its `variable <LinearTimer.variable>` = duration.

        **duration** determines the value of `variable <LinearTimer.variable>`
        at which the value of the function = final.

    `derivative <LinearTimer.derivative>` returns the derivative of the LinearTimer Function:

      .. math::
         \\frac{final-initial}{duration}

    See `graph <https://www.desmos.com/calculator/i0knnnozcs>`_ for interactive plot of the function using `Desmos
    <https://www.desmos.com>`_.

    Arguments
    ---------

    default_variable : number or array : default class_defaults.variable
        specifies a template for the value to be transformed.

    initial : float : default 1.0
        specifies the value the function should have when `variable <LinearTimer.variable>` = 0;
        must be greater than or equal to 0.

    final : float : default 1.0
        specifies the value the function should have when `variable <LinearTimer.variable>` = `duration
        <TimerFunction.duration>`; must be greater than `initial <LinearTimer.initial>`.

    duration : float : default 1.0
        specifies the value of `variable <LinearTimer.variable>` at which the value of the function
        should equal `final <LinearTimer.final>`; must be greater than 0.

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

    initial : float (>0)
        determines the value of the function when `variable <LinearTimer.variable>` = 0.

    final : float
        determines the value of the function when `variable <LinearTimer.variable>` = `duration <TimerFunction.duration>`.

    duration : float (>0)
        determines the value of `variable <LinearTimer.variable>` at which the value of the function is equal
        to `final <LinearTimer.final>`.

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

    componentName = LINEAR_TIMER_FUNCTION

    classPreferences = {
        PREFERENCE_SET_NAME: 'LinearTimerClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    }

    _model_spec_class_name_is_generic = True

    # FIX: REINSTATE Parameters AND VALIDATE INITIAL, FINAL AND DURATION

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 initial: Optional[ValidParamSpecType] = None,
                 final: Optional[ValidParamSpecType] = None,
                 duration: Optional[ValidParamSpecType] = None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):
        super().__init__(
            default_variable=default_variable,
            initial=initial,
            final=final,
            duration=duration,
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

        LinearTimer rise transform of variable : number or array

        """
        initial = self._get_current_parameter_value(INITIAL, context)
        final = self._get_current_parameter_value(FINAL, context)
        duration = self._get_current_parameter_value(DURATION, context)
        result = ((final - initial) / duration) * variable + initial
        return self.convert_output_type(result)

    @handle_external_context()
    def derivative(self, input, output=None, context=None):
        """Derivative of `function <LinearTimer._function>` at **input**:

        .. math::
           (final - initial) * \\left(\\frac{(1 + duration * e^{variable})}{duration * e^{duration}}\\right)

        Arguments
        ---------

        input : number
            value of the input to the LinearTimer transform at which derivative is to be taken.

        Returns
        -------
        derivative :  number or array
        """
        initial = self._get_current_parameter_value(INITIAL, context)
        final = self._get_current_parameter_value(FINAL, context)
        duration = self._get_current_parameter_value(DURATION, context)
        return (final - initial) / duration

    # FIX:
    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        duration_ptr = ctx.get_param_or_state_ptr(builder, self, DURATION, param_struct_ptr=params)
        initial_ptr = ctx.get_param_or_state_ptr(builder, self, INITIAL, param_struct_ptr=params)
        final_ptr = ctx.get_param_or_state_ptr(builder, self, FINAL, param_struct_ptr=params)
        offset_ptr = ctx.get_param_or_state_ptr(builder, self, OFFSET, param_struct_ptr=params)

        initial = pnlvm.helpers.load_extract_scalar_array_one(builder, initial_ptr)
        final = pnlvm.helpers.load_extract_scalar_array_one(builder, final_ptr)
        duration = pnlvm.helpers.load_extract_scalar_array_one(builder, duration_ptr)
        offset = pnlvm.helpers.load_extract_scalar_array_one(builder, offset_ptr)

        exp_f = ctx.get_builtin("exp", [ctx.float_ty])
        val = builder.load(ptri)
        val = builder.fmul(val, duration)
        val = builder.fadd(val, initial)
        val = builder.call(exp_f, [val])

        if "derivative" in tags:
            # f'(x) = s*r*e^(r*x + b)
            val = builder.fmul(val, final)
            val = builder.fmul(val, duration)
        else:
            # f(x) = s*e^(r*x + b) + o
            val = builder.fmul(val, final)
            val = builder.fadd(val, offset)

        builder.store(val, ptro)

    # FIX:
    def _gen_pytorch_fct(self, device, context=None):
        final = self._get_pytorch_fct_param_value(FINAL, device, context)
        initial = self._get_pytorch_fct_param_value(INITIAL, device, context)
        duration = self._get_pytorch_fct_param_value(DURATION, device, context)
        return lambda x : ((final - initial) / duration) * x + initial


class AcceleratingTimer(TimerFunction):
    """
    AcceleratingTimer(     \
         default_variable, \
         initial=0.0,      \
         final=1.0,        \
         duration=1.0,     \
         rate=1.0,         \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _AcceleratingTimer:
    |
    `function <AcceleratingTimer._function>` returns acceleratingTimer rise transform of `variable
    <AcceleratingTimer.variable>`; this is the inverse of the `AcceleratingTimer` Function.

    .. math::
       initial+\\left(final-initial\\right)\\left(\\frac{variable}{duration}\\right)^{rate}e^{\\left(\\left(
       \\frac{variable}{duration}\\right)^{rate}-1\\right)}

    such that:

    .. math::
        value=initial \\ for\\ variable=0

        value=final\\ for\\ variable=duration

    where:

        **initial** determines the `value <Function_Base.value>` of the function
        when its `variable <AcceleratingTimer.variable>` = 0.

        **final** determines the `value <Function_Base.value>` of the function
        when its `variable <AcceleratingTimer.variable>` = duration.

        **duration** determines the value of `variable <AcceleratingTimer.variable>`
        at which the value of the function = final.

        **rate** determines the `rate <AcceleratingTimer.rate>` of acceleration of the function.

    `derivative <AcceleratingTimer.derivative>` returns the derivative of the AcceleratingTimer Function:

      .. math::
         (final-initial) \\cdot \\left[ rate \\cdot \\left(\\frac{variable}{duration}\\right)^{rate-1}
         \\cdot \\frac{1}{duration} \\cdot e^{\\left(\\left(\\frac{variable}{duration}\\right)^{rate}-1\\right)} +
         \\left(\\frac{variable}{duration}\\right)^{rate} \\cdot e^{\\left(\\left(\\frac{variable}{duration}\\right)^{
         rate}-1\\right)} \\cdot rate \\cdot \\frac{1}{duration}\\right]

    See `graph <https://www.desmos.com/calculator/rms6z2ji8g>`_ for interactive plot of the function using `Desmos
    <https://www.desmos.com>`_.

    Arguments
    ---------

    default_variable : number or array : default class_defaults.variable
        specifies a template for the value to be transformed.

    initial : float : default 1.0
        specifies the value the function should have when `variable <AcceleratingTimer.variable>` = 0;
        must be greater than or equal to 0.

    final : float : default 1.0
        specifies the value the function should have when `variable <AcceleratingTimer.variable>` = `duration
        <TimerFunction.duration>`; must be greater than `initial <AcceleratingTimer.initial>`.

    duration : float : default 1.0
        specifies the value of `variable <AcceleratingTimer.variable>` at which the value of the function
        should equal `final <AcceleratingTimer.final>`; must be greater than 0.

    rate : float : default 1.0
        specifies the rate at which the value of the function accelerates; must be greater than 0.

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

    initial : float (>0)
        determines the value of the function when `variable <AcceleratingTimer.variable>` = 0.

    final : float
        determines the value of the function when `variable <AcceleratingTimer.variable>` = `duration <TimerFunction.duration>`.

    duration : float (>0)
        determines the value of `variable <AcceleratingTimer.variable>` at which the value of the function is equal
        to `final <AcceleratingTimer.final>`.

    rate : float (>1.0)
        determines the rate at which the value of the function accelerates.

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

    componentName = ACCELERATING_TIMER_FUNCTION

    classPreferences = {
        PREFERENCE_SET_NAME: 'AcceleratingTimerClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    }

    _model_spec_class_name_is_generic = True

    # FIX: REINSTATE Parameters AND VALIDATE INITIAL, FINAL AND DURATION

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 initial: Optional[ValidParamSpecType] = None,
                 final: Optional[ValidParamSpecType] = None,
                 duration: Optional[ValidParamSpecType] = None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):
        super().__init__(
            default_variable=default_variable,
            initial=initial,
            final=final,
            duration=duration,
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

        AcceleratingTimer rise transform of variable : number or array

        """
        initial = self._get_current_parameter_value(INITIAL, context)
        final = self._get_current_parameter_value(FINAL, context)
        duration = self._get_current_parameter_value(DURATION, context)
        rate = self._get_current_parameter_value(RATE, context)

        result = (initial + (final - initial) * np.power((variable / duration),rate)
                  * np.exp(np.power((variable / duration),rate) - 1))

        return self.convert_output_type(result)

    @handle_external_context()
    def derivative(self, input, output=None, context=None):
        """Derivative of `function <AcceleratingTimer._function>` at **input**:

        .. math::
           (final - initial) * \\left(\\frac{(1 + duration * e^{variable})}{duration * e^{duration}}\\right)

        Arguments
        ---------

        input : number
            value of the input to the AcceleratingTimer transform at which derivative is to be taken.

        Returns
        -------
        derivative :  number or array
        """
        initial = self._get_current_parameter_value(INITIAL, context)
        final = self._get_current_parameter_value(FINAL, context)
        duration = self._get_current_parameter_value(DURATION, context)
        rate = self._get_current_parameter_value(RATE, context)

        return ((final - initial) *
                (rate * np.power((input / duration), (rate - 1))
                 * ((1 / duration)
                    * (np.exp(np.power((input / duration), rate) - 1)
                       + (np.power((input / duration), rate))
                       * np.exp(np.power((input / duration), rate) - 1) * rate * 1 / duration))))

    # FIX:
    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        duration_ptr = ctx.get_param_or_state_ptr(builder, self, DURATION, param_struct_ptr=params)
        initial_ptr = ctx.get_param_or_state_ptr(builder, self, INITIAL, param_struct_ptr=params)
        final_ptr = ctx.get_param_or_state_ptr(builder, self, FINAL, param_struct_ptr=params)
        offset_ptr = ctx.get_param_or_state_ptr(builder, self, OFFSET, param_struct_ptr=params)

        duration = pnlvm.helpers.load_extract_scalar_array_one(builder, duration_ptr)
        initial = pnlvm.helpers.load_extract_scalar_array_one(builder, initial_ptr)
        final = pnlvm.helpers.load_extract_scalar_array_one(builder, final_ptr)
        offset = pnlvm.helpers.load_extract_scalar_array_one(builder, offset_ptr)

        exp_f = ctx.get_builtin("exp", [ctx.float_ty])
        val = builder.load(ptri)
        val = builder.fmul(val, duration)
        val = builder.fadd(val, initial)
        val = builder.call(exp_f, [val])

        if "derivative" in tags:
            # f'(x) = s*r*e^(r*x + b)
            val = builder.fmul(val, final)
            val = builder.fmul(val, duration)
        else:
            # f(x) = s*e^(r*x + b) + o
            val = builder.fmul(val, final)
            val = builder.fadd(val, offset)

        builder.store(val, ptro)

    # FIX:
    def _gen_pytorch_fct(self, device, context=None):
        final = self._get_pytorch_fct_param_value(FINAL, device, context)
        initial = self._get_pytorch_fct_param_value(INITIAL, device, context)
        duration = self._get_pytorch_fct_param_value(DURATION, device, context)
        rate = self._get_pytorch_fct_param_value(RATE, device, context)

        return lambda x : (initial + (final - initial) * torch.power((x / duration),rate)
                           * torch.exp(torch.power((x / duration),rate) - 1))


class DeceleratingTimer(TimerFunction):  # ---------------------------------------------------------------------------
    """
    DeceleratingTimer(     \
         default_variable, \
         initial=1.0,      \
         duration=1.0,     \
         final=0.01,       \
         rate=1.0,         \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. _DeceleratingTimer:
    |
    `function <DeceleratingTimer._function>` returns exponentially decaying transform of `variable
    <DeceleratingTimer.variable>`:

    .. math::
       \\frac{\\left(initial-final-direction\\right)}{e^{\\ln\\left(-direction\\left(initial\\ -\\
       final-direction\\right)\\right)\\left(\\frac{variable}{duration}\\right)^{
       rate}}}+final+direction

    such that:

    .. math::
        value = initial + offset\\ for\\ variable=0

        value = (initial * final) + offset\\ for\\ variable=duration

    where:

        **initial**, together with `offset <DeceleratingTimer.offset>`, determines the value of the function when
        `variable <DeceleratingTimer.variable>` = 0, and is used together with `final <DeceleratingTimer.final>`
        to determine the value of the function when `variable <DeceleratingTimer.variable>` = `duration
        <DeceleratingTimer.duration>`.

        **duration** determines the value of `variable <DeceleratingTimer.variable>` at which
        the value of the function should equal :math:`initial * final + offset`.

        **final** is the fraction of `initial <DeceleratingTimer.initial>` when, added to `offset
        <DeceleratingTimer.offset>`, is used to determine the value of the function when `variable
        <DeceleratingTimer.variable>` should equal `duration <DeceleratingTimer.duration>`.

        **rate** determines the `rate <DeceleratingTimer.rate>` of deceleration of the function.

        **direction** is +1 if final > initial, otherwise -1, and is used to determine the direction of the
        progression (rising or decaying) of the TimerFunction.

    `derivative <DeceleratingTimer.derivative>` returns the derivative of the DeceleratingTimer Function:

      .. math::
         \\frac{direction \\cdot rate \\cdot(initial-final-direction)\\cdot\\ln(direction(final-initial+direction)) \\cdot \\left(\\frac{
         variable}{duration}\\right)^{rate-1}}{duration\\cdot e^{\\ln(direction(final-initial+direction))\\left(\\frac{variable}{
         duration}\\right)^{rate}}}

    See `graph <https://www.desmos.com/calculator/cshkzip0ai>`_ for interactive plot of the function using `Desmos
    <https://www.desmos.com>`_.


    Arguments
    ---------

    default_variable : number or array : default class_defaults.variable
        specifies a template for the value to be transformed.

    initial : float : default 1.0
        specifies, together with `offset <DeceleratingTimer.offset>`, the value of the function when `variable
        <DeceleratingTimer.variable>` = 0; must be greater than 0.

    final : float : default 0.01
        specifies the fraction of `initial <DeceleratingTimer.initial>` when added to `offset <DeceleratingTimer.offset>`,
        that determines the value of the function when `variable <DeceleratingTimer.variable>` = `duration
        <DeceleratingTimer.duration>`; must be between 0 and 1.

    duration : float : default 1.0
        specifies the value of `variable <DeceleratingTimer.variable>` at which the `value of the function
        should equal `initial <DeceleratingTimer.initial>` * `final <DeceleratingTimer.final>` + `offset
        <DeceleratingTimer.offset>`; must be greater than 0.

    rate : float : default 1.0
        specifies the rate at which the value of the function decelerates; must be greater than 0.

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

    initial : float
        determines, together with `offset <DeceleratingTimer.offset>`, the value of the function when `variable
        <DeceleratingTimer.variable>` = 0.

    final : float
        determines the fraction of `initial <DeceleratingTimer.initial>` when added to `offset <DeceleratingTimer.offset>`,
        that determines the value of the function when `variable <DeceleratingTimer.variable>` = `duration
        <DeceleratingTimer.duration>`.

    duration : float (>0)
        determines the value of `variable <DeceleratingTimer.variable>` at which the value of the function should
        equal `initial <DeceleratingTimer.initial>` * `final <DeceleratingTimer.final>` + `offset <DeceleratingTimer.offset>`.

    rate : float (>1.0)
        determines the rate at which the value of the function decelerates.

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

    componentName = DECELERATING_TIMER_FUNCTION

    classPreferences = {
        PREFERENCE_SET_NAME: 'DeceleratingTimerClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    }

    _model_spec_class_name_is_generic = True

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 initial: Optional[ValidParamSpecType] = None,
                 final: Optional[ValidParamSpecType] = None,
                 duration: Optional[ValidParamSpecType] = None,
                 rate: Optional[ValidParamSpecType] = None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):
        super().__init__(
            default_variable=default_variable,
            initial=initial,
            final=final,
            duration=duration,
            rate=rate,
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
           by the value of `increment <DeceleratingTimer.increment>`.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        Exponentially decayed value of variable : number or array

        """
        initial = self._get_current_parameter_value(INITIAL, context)
        final = self._get_current_parameter_value(FINAL, context)
        duration = self._get_current_parameter_value(DURATION, context)
        rate = self._get_current_parameter_value(RATE, context)

        direction = 1 if final > initial else -1

        result = ((initial - final - direction) /
                  np.exp((np.log(-direction * (initial - final - direction)) * np.power((variable / duration),rate)))
                  + final + direction)

        return self.convert_output_type(result)

    @handle_external_context()
    def derivative(self, input, output=None, context=None):
        """Derivative of `function <DeceleratingTimer._function>` at **input**:

      .. math::
         \\frac{direction \\cdot rate \\cdot(initial-final-direction)\\cdot\\ln(direction(
         final-initial+direction))\\cdot \\left(\\frac{variable}{duration}\\right)^{rate-1}}{duration\\cdot e^{\\ln(
         direction(final-initial+direction))\\left(\\frac{variable}{duration}\\right)^{rate}}}


        Arguments
        ---------

        input : number
            value of the input to the DeceleratingTimer transform at which derivative is to be taken.

        Derivative of `function <DeceleratingTimer._function>` at **input**.

        Returns
        -------
        derivative :  number or array
        """

        initial = self._get_current_parameter_value(INITIAL, context)
        final = self._get_current_parameter_value(FINAL, context)
        duration = self._get_current_parameter_value(DURATION, context)
        rate = self._get_current_parameter_value(RATE, context)
        direction = 1 if final > initial else -1

        return (direction * rate * (initial - final - direction) * np.log(direction * (final - initial + direction)) *
                np.power((input / duration), (rate - 1))
                / (duration * np.exp(np.log(direction * (final - initial + direction)) *
                                     np.power((input / duration), rate))))

    # FIX:
    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        duration_ptr = ctx.get_param_or_state_ptr(builder, self, DURATION, param_struct_ptr=params)
        initial_ptr = ctx.get_param_or_state_ptr(builder, self, INITIAL, param_struct_ptr=params)
        final_ptr = ctx.get_param_or_state_ptr(builder, self, FINAL, param_struct_ptr=params)
        offset_ptr = ctx.get_param_or_state_ptr(builder, self, OFFSET, param_struct_ptr=params)

        duration = pnlvm.helpers.load_extract_scalar_array_one(builder, duration_ptr)
        initial = pnlvm.helpers.load_extract_scalar_array_one(builder, initial_ptr)
        final = pnlvm.helpers.load_extract_scalar_array_one(builder, final_ptr)
        offset = pnlvm.helpers.load_extract_scalar_array_one(builder, offset_ptr)

        exp_f = ctx.get_builtin("exp", [ctx.float_ty])
        val = builder.load(ptri)
        val = builder.fmul(val, duration)
        val = builder.fadd(val, initial)
        val = builder.call(exp_f, [val])

        if "derivative" in tags:
            # f'(x) = s*r*e^(r*x + b)
            val = builder.fmul(val, final)
            val = builder.fmul(val, duration)
        else:
            # f(x) = s*e^(r*x + b) + o
            val = builder.fmul(val, final)
            val = builder.fadd(val, offset)

        builder.store(val, ptro)

    def _gen_pytorch_fct(self, device, context=None):
        final = self._get_pytorch_fct_param_value(FINAL, device, context)
        initial = self._get_pytorch_fct_param_value(INITIAL, device, context)
        duration = self._get_pytorch_fct_param_value(DURATION, device, context)
        rate = self._get_pytorch_fct_param_value(RATE, device, context)
        direction = 1 if final > initial else -1

        return lambda x : ((initial - final - direction) /
                           (torch.log(-direction(initial - final - direction)) * torch.power((x / duration),rate))
                           + final + direction)


class AsymptoticTimer(TimerFunction):  # ---------------------------------------------------------------------------
    """
    AsymptoticTimer(       \
         default_variable, \
         initial=1.0,      \
         final=0,          \
         duration=1.0,     \
         tolerance=0.01,   \
         params=None,      \
         owner=None,       \
         name=None,        \
         prefs=None        \
         )

    .. AsymptoticTimer:
    |
    `function <AsymptoticTimer._function>` returns exponentially progressing transform of `variable
    <AsymptoticTimer.variable>` toward an asymptoticTimer value that reaches `duration <AsymptoticTimer.duration>`
    when it falls within the specified `tolerance <AsymptoticTimer.tolerance>` of `final
    <AsymptoticTimer.final>`:

    .. math::
       (initial - final) * \\frac{\\ln(tolerance)}{duration} *e^{\\left(\\frac{variable * \\ln(tolerance)}
       {duration}\\right)} + final

    such that:

    .. math::
        value = initial for\\ variable=0

        value = ((initial - final) \\cdot tolerance) for\\ variable=duration

    where:

        **initial**, determines the value of the function when `variable <AsymptoticTimer.variable>` = 0,
        and is used together with `final <AsymptoticTimer.final>` and `tolerance
        <AsymptoticTimer.tolerance>` to determine the value of the function at which `variable
        <AsymptoticTimer.variable>` = `duration <AsymptoticTimer.duration>`.

        **final** is the asymptoticTimer value toward which the function decays.

        **tolerance** is the fraction of `initial <AsymptoticTimer.initial>` - `final
        <AsymptoticTimer.final>` used to determine the value of the function when `variable
        <AsymptoticTimer.variable>` is equal to `duration <AsymptoticTimer.duration>`.

        **duration** determines the value of `variable <AsymptoticTimer.variable>` at which
        the value of the function is equal to :math:`initial \\cdot final`.

    .. _note::
       The function rises if `final <AsymptoticTimer.final>` > `initial <AsymptoticTimer.initial>` >,
       and decays if `final <AsymptoticTimer.final>` < `initial <AsymptoticTimer.initial>`.

    `derivative <AsymptoticTimer.derivative>` returns the derivative of the AsymptoticTimer Function:

      .. math::
         \\frac{initial\\cdot\\ln(tolerance)}{duration}\\cdot e^{\\frac{variable\\cdot\\ln(tolerance)}{duration}}

    See `graph <https://www.desmos.com/calculator/tmfs4ps9cp>`_ for interactive plot of the function using `Desmos
    <https://www.desmos.com>`_.

    Arguments
    ---------

    default_variable : number or array : default class_defaults.variable
        specifies a template for the value to be transformed.

    initial : float : default 1.0
        specifies the value of the function when `variable<AsymptoticTimer.variable>`=0; must be greater than 0.

    final : float : default 0.0
        specifies the asymptoticTimer value toward which the function decays.

    tolerance : float : default 0.01
        specifies the fraction of `initial <AsymptoticTimer.initial>`-`final <AsymptoticTimer.final>`
        that determines the value of the function when `variable <AsymptoticTimer.variable>` = `duration; must be
        between 0 and 1.

    duration : float : default 1.0
        specifies the value of `variable <AsymptoticTimer.variable>` at which the `value of the function
        should equal `initial <AsymptoticTimer.initial>` * `final <AsymptoticTimer.final>`;
        must be greater than 0.

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

    initial : float (>0)
        determines the value of the function when `variable <AsymptoticTimer.variable>` = 0.

    final : float
        determines the asymptoticTimer value toward which the function decays.

    tolerance : float (0,1)
        determines the fraction of `initial <AsymptoticTimer.initial>` - final <AsymptoticTimer.final>
        that determines the value of the function when `variable <AsymptoticTimer.variable>` = `duration
        <AsymptoticTimer.duration>`.

    duration : float (>0)
        determines the value of `variable <AsymptoticTimer.variable>` at which the value of the function should
        equal `initial <AsymptoticTimer.initial>` * `final <AsymptoticTimer.final>`.

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

    componentName = ASYMPTOTIC_TIMER_FUNCTION

    classPreferences = {
        PREFERENCE_SET_NAME: 'AsymptoticTimerClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE),
    }

    _model_spec_class_name_is_generic = True

    class Parameters(TimerFunction.Parameters):
        """
            Attributes
            ----------
                rate
                    see `tolerance <AsymptoticTimer.tolerance>`

                    :default value: None
                    :type: ``float``

                tolerance
                    see `tolerance <AsymptoticTimer.tolerance>`

                    :default value: 0.01
                    :type: ``float``
        """
        rate = Parameter(None)
        tolerance = Parameter(0.01, modulable=True)

        def _validate_rate(self, rate):
            if rate is not None:
                return f"is not used and should be left as None."

        def _validate_tolerance(self, tolerance):
            if tolerance <= 0 or tolerance >= 1:
                return f"must be between 0 and 1."

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 initial: Optional[ValidParamSpecType] = None,
                 final: Optional[ValidParamSpecType] = None,
                 tolerance: Optional[ValidParamSpecType] = None,
                 duration: Optional[ValidParamSpecType] = None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):
        super().__init__(
            default_variable=default_variable,
            initial=initial,
            duration=duration,
            final=final,
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
           by the value of `increment <AsymptoticTimer.increment>`.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        Exponentially decayed value of variable : number or array

        """
        initial = self._get_current_parameter_value(INITIAL, context)
        duration = self._get_current_parameter_value(DURATION, context)
        tolerance = self._get_current_parameter_value(TOLERANCE, context)
        final = self._get_current_parameter_value(FINAL, context)

        result = (initial - final) * np.exp(variable * np.log(tolerance) / duration) + final

        return self.convert_output_type(result)

    @handle_external_context()
    def derivative(self, input, output=None, context=None):
        """Derivative of `function <AsymptoticTimer._function>` at **input**:

        .. math::
           \\frac{initial\\cdot\\ln(tolerance)}{duration}\\cdot e^{\\frac{variable\\cdot\\ln(tolerance)}{duration}}

        Arguments
        ---------

        input : number
            value of the input to the AsymptoticTimer transform at which derivative is to be taken.

        Derivative of `function <AsymptoticTimer._function>` at **input**.

        Returns
        -------
        derivative :  number or array
        """

        initial = self._get_current_parameter_value(INITIAL, context)
        tolerance = self._get_current_parameter_value(TOLERANCE, context)
        final = self._get_current_parameter_value(FINAL, context)
        duration = self._get_current_parameter_value(DURATION, context)

        return (initial * np.log(tolerance) / duration) * np.exp(input * np.log(tolerance) / duration)

    # FIX:
    def _gen_llvm_transfer(self, builder, index, ctx, vi, vo, params, state, *, tags:frozenset):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])

        duration_ptr = ctx.get_param_or_state_ptr(builder, self, DURATION, param_struct_ptr=params)
        initial_ptr = ctx.get_param_or_state_ptr(builder, self, INITIAL, param_struct_ptr=params)
        final_ptr = ctx.get_param_or_state_ptr(builder, self, FINAL, param_struct_ptr=params)

        duration = pnlvm.helpers.load_extract_scalar_array_one(builder, duration_ptr)
        initial = pnlvm.helpers.load_extract_scalar_array_one(builder, initial_ptr)
        final = pnlvm.helpers.load_extract_scalar_array_one(builder, final_ptr)

        exp_f = ctx.get_builtin("exp", [ctx.float_ty])
        val = builder.load(ptri)
        val = builder.fmul(val, duration)
        val = builder.fadd(val, initial)
        val = builder.call(exp_f, [val])

        if "derivative" in tags:
            # f'(x) = s*r*e^(r*x + b)
            val = builder.fmul(val, final)
            val = builder.fmul(val, duration)
        else:
            # f(x) = s*e^(r*x + b) + o
            val = builder.fmul(val, final)

        builder.store(val, ptro)

    def _gen_pytorch_fct(self, device, context=None):
        initial = self._get_pytorch_fct_param_value(INITIAL, device, context)
        tolerance = self._get_pytorch_fct_param_value(TOLERANCE, device, context)
        final = self._get_pytorch_fct_param_value(FINAL, device, context)
        duration = self._get_pytorch_fct_param_value(DURATION, device, context)

        return lambda x : (initial - final) * torch.exp(x * torch.log(tolerance) / duration) + final
