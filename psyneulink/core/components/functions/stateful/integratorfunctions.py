#
# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *****************************************  INTEGRATOR FUNCTIONS ******************************************************
"""

Functions that integrate current value of input with previous value.

* `IntegratorFunction`
* `AccumulatorIntegrator`
* `SimpleIntegrator`
* `AdaptiveIntegrator`
* `DualAdaptiveIntegrator`
* `DriftDiffusionIntegrator`
* `DriftOnASphereIntegrator`
* `OrnsteinUhlenbeckIntegrator`
* `InteractiveActivationIntegrator`
* `LeakyCompetingIntegrator`
* `FitzHughNagumoIntegrator`

"""

import warnings

import numpy as np
import typecheck as tc

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.component import DefaultsFlexibility
from psyneulink.core.components.functions.nonstateful.distributionfunctions import DistributionFunction
from psyneulink.core.components.functions.function import (
    DEFAULT_SEED, FunctionError, _random_state_getter,
    _seed_setter, _noise_setter
)
from psyneulink.core.components.functions.stateful.statefulfunction import StatefulFunction
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import \
    ACCUMULATOR_INTEGRATOR_FUNCTION, ADAPTIVE_INTEGRATOR_FUNCTION, ADDITIVE_PARAM, \
    DECAY, DEFAULT_VARIABLE, DRIFT_DIFFUSION_INTEGRATOR_FUNCTION, DRIFT_ON_A_SPHERE_INTEGRATOR_FUNCTION, \
    DUAL_ADAPTIVE_INTEGRATOR_FUNCTION, FITZHUGHNAGUMO_INTEGRATOR_FUNCTION, FUNCTION, \
    INCREMENT, INITIALIZER, INPUT_PORTS, INTEGRATOR_FUNCTION, INTEGRATOR_FUNCTION_TYPE, \
    INTERACTIVE_ACTIVATION_INTEGRATOR_FUNCTION, LEAKY_COMPETING_INTEGRATOR_FUNCTION, \
    MULTIPLICATIVE_PARAM, NOISE, OFFSET, OPERATION, ORNSTEIN_UHLENBECK_INTEGRATOR_FUNCTION, OUTPUT_PORTS, PRODUCT, \
    RATE, REST, SIMPLE_INTEGRATOR_FUNCTION, SUM, TIME_STEP_SIZE, THRESHOLD, VARIABLE, MODEL_SPEC_ID_MDF_VARIABLE
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.utilities import parameter_spec, all_within_range, \
    convert_all_elements_to_np_array

__all__ = ['SimpleIntegrator', 'AdaptiveIntegrator', 'DriftDiffusionIntegrator', 'DriftOnASphereIntegrator',
           'OrnsteinUhlenbeckIntegrator', 'FitzHughNagumoIntegrator', 'AccumulatorIntegrator',
           'LeakyCompetingIntegrator', 'DualAdaptiveIntegrator', 'InteractiveActivationIntegrator',
           'S_MINUS_L', 'L_MINUS_S', 'IntegratorFunction'
           ]


# • why does integrator return a 2d array?
# • are rate and noise converted to 1d np.array?  If not, correct docstring
# • can noise and initializer be an array?  If so, validated in validate_param?

class IntegratorFunction(StatefulFunction):  # -------------------------------------------------------------------------
    """
    IntegratorFunction(         \
        default_variable=None,  \
        initializer=None,       \
        rate=1.0,               \
        noise=0.0,              \
        time_step_size=1.0,     \
        params=None,            \
        owner=None,             \
        prefs=None,             \
        )

    .. _Integrator:

    Base class for Functions that integrate current value of `variable <IntegratorFunction.variable>` with its prior
    value.  For most subclasses, `variable <IntegratorFunction.variable>` can be a single float or an array.  If it is
    an array, each element is integrated independently of the others.

    .. _IntegratorFunction_Parameter_Spec:

    .. note::
        If `variable <IntegratorFunction.variable>` is an array, for any parameter that is specified as a float its
        value is applied uniformly to all elements of the relevant term of the integral (e.g., `variable
        <IntegratorFunction.variable>` or `previous_value <IntegratorFunction.previous_value>`, depending on the
        subclass);  for any parameter specified as an array, it must be the same length as `variable
        <IntegratorFunction.variable>`, and it is applied elementwise (Hadarmard) to the relevant term of the integral.
        If, on initialization, the default_variable is not specified, any parameters specified as an array must be
        the same length, and the default_variable is assumed to have the same length as those parameters.

    Arguments
    ---------

    default_variable : number, list or array : default class_defaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    initializer : float, list or 1d array : default 0.0
        specifies starting value(s) for integration.  If it is a list or array, it must be the same length as
        `variable <IntegratorFunction.variable>` (see `initializer <IntegratorFunction.initializer>`
        for details).

    rate : float, list or 1d array : default 1.0
        specifies the rate of integration.  If it is a list or array, it must be the same length as
        `variable <IntegratorFunction.variable>` (see `rate <IntegratorFunction.rate>` for details).

    noise : float, list, array or function : default 0.0
        specifies value added to integral in each call to `function <IntegratorFunction._function>`;
        if it is a list or array, it must be the same length as `variable <IntegratorFunction.variable>`
        (see `noise <IntegratorFunction.noise>` for additional details).

    time_step_size : float : default 0.0
        determines the timing precision of the integration process

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
        current input value some portion of which (determined by `rate <IntegratorFunction.rate>`) that will be
        added to the prior value;  if it is an array, each element is independently integrated.

    rate : float or 1d array
        determines the rate of integration. If it is a float or has a single value, it is applied to all elements
        of `variable <IntegratorFunction.variable>` and/or `previous_value <IntegratorFunction.previous_value>`
        (depending on the subclass);  if it has more than one element, each element is applied to the corresponding
        element of `variable <IntegratorFunction.variable>` and/or `previous_value <IntegratorFunction.previous_value>`.

    noise : float, array or Function
        value is added to integral in each call to `function <IntegratorFunction._function>`. If noise is a
        float,  it is applied to all elements of `variable <IntegratorFunction.variable>`; if it is an array,
        it is applied Hadamard (elementwise) to each element of `variable <IntegratorFunction.variable>`. If it is a
        function, it is executed separately and applied independently to each element of `variable
        <IntegratorFunction.variable>`.

        .. hint::
            To generate random noise that varies for every execution and across all elements of an array, a
            `DistributionFunction` should be used, that generates a new value on each execution. If noise is
            specified as a float, a function with a fixed output, or an array of either of these, then noise
            is simply an offset that is the same across all elements and executions.

    initializer : float or 1d array
        determines the starting value(s) for integration (i.e., the value(s) to which `previous_value
        <IntegratorFunction.previous_value>` is set.  If `variable <IntegratorFunction.variable>` is a list or array,
        and initializer is a float or has a single element, it is applied to each element of `previous_value
        <IntegratorFunction.previous_value>`. If initializer is a list or array, each element is applied to the
        corresponding element of `previous_value <IntegratorFunction.previous_value>`.

    previous_value : 1d array
        stores previous value with which `variable <IntegratorFunction.variable>` is integrated.

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

    componentType = INTEGRATOR_FUNCTION_TYPE
    componentName = INTEGRATOR_FUNCTION

    class Parameters(StatefulFunction.Parameters):
        """
            Attributes
            ----------

                initializer
                    see `initializer <IntegratorFunction.initializer>`

                    :default value: numpy.array([0])
                    :type: ``numpy.ndarray``

                noise
                    see `noise <IntegratorFunction.noise>`

                    :default value: 0.0
                    :type: ``float``

                previous_value
                    see `previous_value <IntegratorFunction.previous_value>`

                    :default value: numpy.array([0])
                    :type: ``numpy.ndarray``

                rate
                    see `rate <IntegratorFunction.rate>`

                    :default value: 1.0
                    :type: ``float``
        """
        rate = Parameter(1.0, modulable=True, function_arg=True)
        noise = Parameter(
            0.0, modulable=True, function_arg=True, setter=_noise_setter
        )
        previous_value = Parameter(np.array([0]), initializer='initializer')
        initializer = Parameter(np.array([0]), pnl_internal=True)

    @check_user_specified
    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate=None,
                 noise=None,
                 initializer=None,
                 params: tc.optional(tc.optional(dict)) = None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None,
                 context=None,
                 **kwargs):

        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            rate=rate,
            noise=noise,
            params=params,
            owner=owner,
            prefs=prefs,
            context=context,
            **kwargs
        )

    # FIX CONSIDER MOVING THIS TO THE LEVEL OF Function_Base OR EVEN Component
    def _validate_params(self, request_set, target_set=None, context=None):
        """Check inner dimension (length) of all parameters used for the function

        Insure that for any parameters that are in the Parameters class, designated as function_arg, and
            specified by the user with length>1:
            1) they all have the same length;
            2) if default_variable:
               - was specified by the user, the parameters all have the same length as that
               - was NOT specified by the user, they all have the same length as each other;
                 in this case, default_variable will be set to the length of those parameters in
                 _instantiate_attributes_before_function below
        """


        super()._validate_params(
            request_set=request_set,
            target_set=target_set,
            context=context
        )

        # Use dict to be able to report names of params that are in violating set
        params_to_check = {}

        for param in request_set:
            value = request_set[param]
            # If param is in Parameter class for function and it is a function_arg:
            if (param in self.parameters.names() and getattr(self.parameters, param).function_arg
                    and getattr(self.parameters, param)._user_specified):
                if value is not None and isinstance(value, (list, np.ndarray)) and len(value)>1:
                    # Store ones with length > 1 in dict for evaluation below
                    params_to_check.update({param:value})

        values = list(params_to_check.values())

        # If default_variable was specified by user, check that all function_arg params have same length
        #    as the length of items in the inner-most dimension (axis) of default_variable
        if self.parameters.variable._user_specified:
            default_variable_len = self.parameters.variable.default_value.shape[-1]
            violators = [k for k,v in params_to_check.items() if np.array(v).shape[-1]!=default_variable_len]
            if violators:
                raise FunctionError(f"The following parameters with len>1 specified for {self.name} "
                                    f"don't have the same length as its {repr(DEFAULT_VARIABLE)} "
                                    f"({default_variable_len}): {violators}.", component=self)

        # Check that all function_arg params with length > 1 have the same length
        elif any(len(v)!=len(values[0]) for v in values):
            raise FunctionError(f"The parameters with len>1 specified for {self.name} "
                                f"({sorted(params_to_check.keys())}) don't all have the same length")

    def _instantiate_attributes_before_function(self, function=None, context=None):
        """Insure inner dimension of default_variable matches the length of any parameters that have len>1"""

        # Note:  if default_variable was user specified, equal length of parameters was validated in _validate_params
        if not self.parameters.variable._user_specified:
            values_with_a_len = [param.default_value for param in self.parameters if
                                 param.function_arg and
                                 isinstance(param.default_value, (list, np.ndarray)) and
                                 len(param.default_value)>1]
            # One or more parameters are specified with length > 1 in the inner dimension
            if values_with_a_len:
                # If shape already matches,
                #    leave alone in case default_variable was specified by class with values other than zero
                #    (since reshaping below is done with zeros)
                variable_shape = list(self.parameters.variable.default_value.shape)
                # IMPLEMENTATION NOTE:
                #    Don't want to just test here with np.broadcast_to since default_variable could be len=1
                #    in which case a parameter with len>1 applied to it will generate a variable of len>1;
                #    need default_variable to be cast to have same length as parameters
                if variable_shape[-1] != np.array(values_with_a_len[0]).shape[-1]:
                    variable_shape[-1] = np.array(values_with_a_len[0]).shape[-1]
                    self.parameters.variable.default_value = np.zeros(tuple(variable_shape))
                    # Since default_variable is being determined by user specification of parameter:
                    self.parameters.variable._user_specified = True

        super()._instantiate_attributes_before_function(function=function, context=context)

    def _EWMA_filter(self, previous_value, rate, variable):
        """Return `exponentially weighted moving average (EWMA)
        <https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average>`_ of a variable.
        """
        return (1 - rate) * previous_value + rate * variable

    def _logistic(self, variable, gain, bias):
        """Return logistic transform of variable"""
        return 1 / (1 + np.exp(-(gain * variable) + bias))

    def _euler(self, previous_value, previous_time, slope, time_step_size):

        if callable(slope):
            slope = slope(previous_time, previous_value)

        return previous_value + slope * time_step_size

    def _runge_kutta_4(self, previous_value, previous_time, slope, time_step_size):

        if callable(slope):
            slope_approx_1 = slope(previous_time,
                                   previous_value)

            slope_approx_2 = slope(previous_time + time_step_size / 2,
                                   previous_value + (0.5 * time_step_size * slope_approx_1))

            slope_approx_3 = slope(previous_time + time_step_size / 2,
                                   previous_value + (0.5 * time_step_size * slope_approx_2))

            slope_approx_4 = slope(previous_time + time_step_size,
                                   previous_value + (time_step_size * slope_approx_3))

            value = previous_value \
                    + (time_step_size / 6) * (slope_approx_1 + 2 * (slope_approx_2 + slope_approx_3) + slope_approx_4)

        else:
            value = previous_value + time_step_size * slope

        return value

    def _function(self, *args, **kwargs):
        raise FunctionError("IntegratorFunction is not meant to be called explicitly")

    def _gen_llvm_function_body(self, ctx, builder, params, state, arg_in, arg_out, *, tags:frozenset):
        # Get rid of 2d array.
        # When part of a Mechanism, the input and output are 2d arrays.
        arg_in = pnlvm.helpers.unwrap_2d_array(builder, arg_in)

        # output may be 2d with multiple items (e.g. DriftDiffusionIntegrator,
        # FitzHughNagumoIntegrator)
        if arg_out.type.pointee.count == 1:
            arg_out = pnlvm.helpers.unwrap_2d_array(builder, arg_out)

        with pnlvm.helpers.array_ptr_loop(builder, arg_in, "integrate") as args:
            self._gen_llvm_integrate(*args, ctx, arg_in, arg_out, params, state)

        return builder

    def _gen_llvm_load_param(self, ctx, builder, params, index, param, *,
                             state=None):
        param_p = pnlvm.helpers.get_param_ptr(builder, self, params, param)
        if param == NOISE and isinstance(param_p.type.pointee, pnlvm.ir.LiteralStructType):
            # This is a noise function so call it to get value
            assert state is not None
            state_p = pnlvm.helpers.get_state_ptr(builder, self, state, NOISE)
            noise_f = ctx.import_llvm_function(self.parameters.noise.get())
            noise_in = builder.alloca(noise_f.args[2].type.pointee)
            noise_out = builder.alloca(noise_f.args[3].type.pointee)
            builder.call(noise_f, [param_p, state_p, noise_in, noise_out])
            value_p = noise_out

        elif isinstance(param_p.type.pointee, pnlvm.ir.ArrayType) and param_p.type.pointee.count > 1:
            value_p = builder.gep(param_p, [ctx.int32_ty(0), index])
        else:
            value_p = param_p
        return pnlvm.helpers.load_extract_scalar_array_one(builder, value_p)



# *********************************************** INTEGRATOR FUNCTIONS *************************************************


class AccumulatorIntegrator(IntegratorFunction):  # --------------------------------------------------------------------
    """
    AccumulatorIntegrator(              \
        default_variable=None,          \
        rate=1.0,                       \
        increment=0.0,                  \
        noise=0.0,                      \
        initializer=None,               \
        params=None,                    \
        owner=None,                     \
        prefs=None,                     \
        )

    .. _AccumulatorIntegrator:

    Accumulates at a constant rate, that is either linear or exponential, depending on `rate
    <AccumulatorIntegrator.rate>`.  `function <AccumulatorIntegrator._function>` ignores `variable
    <AccumulatorIntegrator.variable>` and returns:

    .. math::
        previous\\_value \\cdot rate + increment  + noise

    so that, with each call to `function <AccumulatorIntegrator._function>`, the accumulated value increases by:

    .. math::
        increment \\cdot rate^{time\\ step}.

    Thus, accumulation increases lineary in steps of `increment <AccumulatorIntegrator.increment>`
    if `rate <AccumulatorIntegrator.rate>`\\=1.0, and exponentially otherwise.

    *Modulatory Parameters:*

    | *MULTIPLICATIVE_PARAM:* `rate <AccumulatorIntegrator.rate>`
    | *ADDITIVE_PARAM:* `increment <AccumulatorIntegrator.increment>`
    |

    Arguments
    ---------

    default_variable : number, list or array : default class_defaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float, list or 1d array : default 1.0
        specifies the rate of decay;  if it is a list or array, it must be the same length as `variable
        <AccumulatorIntegrator.variable>` (see `rate <AccumulatorIntegrator.rate>` for additional details.

    increment : float, list or 1d array : default 0.0
        specifies an amount to be added to `previous_value <AccumulatorIntegrator.previous_value>` in each call to
        `function <AccumulatorIntegrator._function>`; if it is a list or array, it must be the same length as
        `variable <AccumulatorIntegrator.variable>` (see `increment <AccumulatorIntegrator.increment>` for details).

    noise : float, Function, list or 1d array : default 0.0
        specifies random value added to `prevous_value <AccumulatorIntegrator.previous_value>` in each call to
        `function <AccumulatorIntegrator._function>`; if it is a list or array, it must be the same length as
        `variable <AccumulatorIntegrator.variable>` (see `noise <Integrator.noise>` for additional details).

    initializer : float, list or 1d array : default 0.0
        specifies starting value(s) for integration.  If it is a list or array, it must be the same length as
        `variable <AccumulatorIntegrator.variable>` (see `initializer <AccumulatorIntegrator.initializer>` for details).

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
        **Ignored** by the AccumulatorIntegrator function. Use `LeakyCompetingIntegrator` or `AdaptiveIntegrator` for
        integrator functions that depend on both a prior value and a new value (variable).

    rate : float or 1d array
        determines the rate of exponential decay of `previous_value <AccumulatorIntegrator.previous_value>` in each
        call to `function <AccumulatorIntegrator._function>`. If it is a float or has a single element, its value is
        applied to all the elements of `previous_value <AccumulatorIntegrator.previous_value>`; if it is an array, each
        element is applied to the corresponding element of `previous_value <AccumulatorIntegrator.previous_value>`.
        Serves as *MULTIPLICATIVE_PARAM* for `modulation <ModulatorySignal_Modulation>` of `function
        <AccumulatorIntegrator._function>`.

    increment : float, function, or 1d array
        determines the amount added to `previous_value <AccumulatorIntegrator.previous_value>` in each call to
        `function <AccumulatorIntegrator._function>`.  If it is a list or array, it must be the same length as
        `variable <AccumulatorIntegrator.variable>` and each element is added to the corresponding element of
        `previous_value <AccumulatorIntegrator.previous_value>` (i.e., it is used for Hadamard addition).  If it is a
        scalar or has a single element, its value is added to all the elements of `previous_value
        <AccumulatorIntegrator.previous_value>`.  Serves as *ADDITIVE_PARAM* for
        `modulation <ModulatorySignal_Modulation>` of `function <AccumulatorIntegrator._function>`.

    noise : float, Function or 1d array
        random value added in each call to `function <AccumulatorIntegrator._function>`
        (see `noise <Integrator.noise>` for details).

    initializer : float or 1d array
        determines the starting value(s) for integration (i.e., the value(s) to which `previous_value
        <AccumulatorIntegrator.previous_value>` is set (see `initializer <AccumulatorIntegrator.initializer>`
        for details).

    previous_value : 1d array : default class_defaults.variable
        stores previous value to which `rate <AccumulatorIntegrator.rate>` and `noise <AccumulatorIntegrator.noise>`
        will be added.

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

    componentName = ACCUMULATOR_INTEGRATOR_FUNCTION

    class Parameters(IntegratorFunction.Parameters):
        """
            Attributes
            ----------

                increment
                    see `increment <AccumulatorIntegrator.increment>`

                    :default value: 0.0
                    :type: ``float``

                rate
                    see `rate <AccumulatorIntegrator.rate>`

                    :default value: 1.0
                    :type: ``float``
        """
        rate = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM], function_arg=True)
        increment = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM], function_arg=True)

    @check_user_specified
    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate=None,
                 increment=None,
                 noise=None,
                 initializer=None,
                 params: tc.optional(tc.optional(dict)) = None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

        super().__init__(
            default_variable=default_variable,
            rate=rate,
            increment=increment,
            noise=noise,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _accumulator_check_args(self, variable=None, context=None, params=None, target_set=None):
        """validate params and assign any runtime params.

        Called by AccumulatorIntegrator to validate params
        Validation can be suppressed by turning parameter_validation attribute off
        target_set is a params dictionary to which params should be assigned;

        Does the following:
        - assign runtime params to context
        - validate params if PARAM_VALIDATION is set

        :param params: (dict) - params to validate
        :target_set: (dict) - set to which params should be assigned
        :return:
        """

        # PARAMS ------------------------------------------------------------

        # # MODIFIED 11/27/16 OLD:
        # # If parameter_validation is set, the function was called with params,
        # #   and they have changed, then validate requested values and assign to target_set
        # if self.prefs.paramValidationPref and params and not params is None and not params is target_set:
        #     # self._validate_params(params, target_set, context=FUNCTION_CHECK_ARGS)
        #     self._validate_params(request_set=params, target_set=target_set, context=context)

        # If params have been passed, treat as runtime params
        #   (relabel params as runtime_params for clarity)
        if context.execution_id in self._runtime_params_reset:
            for key in self._runtime_params_reset[context.execution_id]:
                self._set_parameter_value(key, self._runtime_params_reset[context.execution_id][key], context)
        self._runtime_params_reset[context.execution_id] = {}

        runtime_params = params
        if runtime_params:
            for param_name in runtime_params:
                if param_name in self.parameters:
                    if param_name in {FUNCTION, INPUT_PORTS, OUTPUT_PORTS}:
                        continue
                    if context.execution_id not in self._runtime_params_reset:
                        self._runtime_params_reset[context.execution_id] = {}
                    self._runtime_params_reset[context.execution_id][param_name] = getattr(self.parameters, param_name)._get(context)
                    self._set_parameter_value(param_name, runtime_params[param_name], context)

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        updated value of integral : 2d array

        """
        self._accumulator_check_args(variable, context=context, params=params)

        # Warn if being called as a standalone function and variable is passed
        # Don't warn if it belongs to a Component, ans that Component's function may pass in a value for variable
        # (such as a MappingProjection that uses AccumulatorFunction in its matrix ParameterPort for learning)
        if (not self.owner
                and self.initialization_status != ContextFlags.INITIALIZING
                and variable is not None
                and variable is not self.defaults.variable):
            warnings.warn("{} does not use its variable;  value passed ({}) will be ignored".
                          format(self.__class__.__name__, variable))

        rate = self._get_current_parameter_value(RATE, context)
        increment = self._get_current_parameter_value(INCREMENT, context)
        noise = self._try_execute_param(self._get_current_parameter_value(NOISE, context), variable, context=context)

        previous_value = np.atleast_2d(self.parameters.previous_value._get(context))

        value = previous_value * rate + noise + increment

        # If this NOT an initialization run, update the old value
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if not self.is_initializing:
            self.parameters.previous_value._set(value, context)

        return self.convert_output_type(value)

    def _gen_llvm_integrate(self, builder, index, ctx, vi, vo, params, state):
        rate = self._gen_llvm_load_param(ctx, builder, params, index, RATE)
        increment = self._gen_llvm_load_param(ctx, builder, params, index, INCREMENT)
        noise = self._gen_llvm_load_param(ctx, builder, params, index, NOISE,
                                          state=state)

        # Get the only context member -- previous value
        prev_ptr = pnlvm.helpers.get_state_ptr(builder, self, state, "previous_value")
        # Get rid of 2d array. When part of a Mechanism the input,
        # (and output, and context) are 2d arrays.
        prev_ptr = pnlvm.helpers.unwrap_2d_array(builder, prev_ptr)
        assert len(prev_ptr.type.pointee) == len(vi.type.pointee)

        prev_ptr = builder.gep(prev_ptr, [ctx.int32_ty(0), index])
        prev_val = builder.load(prev_ptr)

        res = builder.fmul(prev_val, rate)
        res = builder.fadd(res, noise)
        res = builder.fadd(res, increment)

        vo_ptr = builder.gep(vo, [ctx.int32_ty(0), index])
        builder.store(res, vo_ptr)
        builder.store(res, prev_ptr)

    def as_expression(self):
        return 'previous_value * rate + noise + increment'


class SimpleIntegrator(IntegratorFunction):  # -------------------------------------------------------------------------
    """
    SimpleIntegrator(           \
        default_variable=None,  \
        rate=1.0,               \
        noise=0.0,              \
        offset=0.0,             \
        initializer=None,       \
        params=None,            \
        owner=None,             \
        prefs=None,             \
        )

    .. _SimpleIntegrator:

    `function <SimpleIntegrator._function>` returns:

    .. math::

        previous_value + rate * variable + noise + offset

    *Modulatory Parameters:*

    | *MULTIPLICATIVE_PARAM:* `rate <SimpleIntegrator.rate>`
    | *ADDITIVE_PARAM:* `offset <SimpleIntegrator.offset>`
    |

    Arguments
    ---------

    default_variable : number, list or array : default class_defaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float, list or 1d array : default 1.0
        specifies the rate of integration;  if it is a list or array, it must be the same length as
        `variable <SimpleIntegrator.variable>` (see `rate <SimpleIntegrator.rate>` for details).

    noise : float, function, list or 1d array : default 0.0
        specifies random value added to integral in each call to `function <SimpleIntegrator._function>`;
        if it is a list or array, it must be the same length as `variable <SimpleIntegrator.variable>`
        (see `noise <Integrator.noise>` for details).

    offset : float, list or 1d array : default 0.0
        specifies constant value added to integral in each call to `function <SimpleIntegrator._function>`;
        if it is a list or array, it must be the same length as `variable <SimpleIntegrator.variable>`
        (see `offset <SimpleIntegrator.offset>` for details).

    initializer : float, list or 1d array : default 0.0
        specifies starting value(s) for integration;  if it is a list or array, it must be the same length as
        `variable <SimpleIntegrator.variable>` (see `initializer <IntegratorFunction.initializer>` for details).

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
        current input value some portion of which (determined by `rate <SimpleIntegrator.rate>`) will be
        added to the prior value;  if it is an array, each element is independently integrated.

    rate : float or 1d array
        determines the rate of integration. If it is a float or has a single element, it is applied to all elements
        `variable <SimpleIntegrator.variable>`;  if it has more than one element, each element is applied to the
        corresponding element of `variable <SimpleIntegrator.variable>`.  Serves as *MULTIPLICATIVE_PARAM* for
        `modulation <ModulatorySignal_Modulation>` of `function <SimpleIntegrator._function>`.

    noise : float, Function or 1d array
        random value added to integral in each call to `function <SimpleIntegrator._function>`
        (see `noise <Integrator.noise>` for details).

    offset : float or 1d array
        constant value added to integral in each call to `function <SimpleIntegrator._function>`. If `variable
        <SimpleIntegrator.variable>` is an array and offset is a float, offset is applied to each element of the
        integral;  if offset is a list or array, each of its elements is applied to each of the corresponding
        elements of the integral (i.e., Hadamard addition). Serves as *ADDITIVE_PARAM* for `modulation
        <ModulatorySignal_Modulation>` of `function <SimpleIntegrator._function>`.

    initializer : float or 1d array
        determines the starting value(s) for integration (i.e., the value to which `previous_value
        <SimpleIntegrator.previous_value>` is set (see `initializer <IntegratorFunction.initializer>` for details).

    previous_value : 1d array : default class_defaults.variable
        stores previous value with which `variable <SimpleIntegrator.variable>` is integrated.

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

    componentName = SIMPLE_INTEGRATOR_FUNCTION


    class Parameters(IntegratorFunction.Parameters):
        """
            Attributes
            ----------

                offset
                    see `offset <SimpleIntegrator.offset>`

                    :default value: 0.0
                    :type: ``float``

                rate
                    see `rate <SimpleIntegrator.rate>`

                    :default value: 1.0
                    :type: ``float``
        """
        rate = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM], function_arg=True)
        offset = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM], function_arg=True)

    @check_user_specified
    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate: tc.optional(parameter_spec) = None,
                 noise=None,
                 offset=None,
                 initializer=None,
                 params: tc.optional(tc.optional(dict)) = None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):
        super().__init__(
            default_variable=default_variable,
            rate=rate,
            noise=noise,
            offset=offset,
            initializer=initializer,
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

        variable : number, list or array : default class_defaults.variable
           a single value or array of values to be integrated.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        updated value of integral : 2d array

        """
        rate = np.array(self._get_current_parameter_value(RATE, context)).astype(float)

        offset = self._get_current_parameter_value(OFFSET, context)

        # execute noise if it is a function
        noise = self._try_execute_param(self._get_current_parameter_value(NOISE, context), variable, context=context)
        previous_value = self.parameters.previous_value._get(context)
        new_value = variable

        value = previous_value + (new_value * rate) + noise

        adjusted_value = value + offset

        # If this NOT an initialization run, update the old value
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if not self.is_initializing:
            self.parameters.previous_value._set(adjusted_value, context)

        return self.convert_output_type(adjusted_value)

    def _gen_llvm_integrate(self, builder, index, ctx, vi, vo, params, state):
        rate = self._gen_llvm_load_param(ctx, builder, params, index, RATE)
        offset = self._gen_llvm_load_param(ctx, builder, params, index, OFFSET)
        noise = self._gen_llvm_load_param(ctx, builder, params, index, NOISE,
                                          state=state)

        # Get the only context member -- previous value
        prev_ptr = pnlvm.helpers.get_state_ptr(builder, self, state, "previous_value")
        # Get rid of 2d array. When part of a Mechanism the input,
        # (and output, and context) are 2d arrays.
        prev_ptr = pnlvm.helpers.unwrap_2d_array(builder, prev_ptr)
        assert len(prev_ptr.type.pointee) == len(vi.type.pointee)

        prev_ptr = builder.gep(prev_ptr, [ctx.int32_ty(0), index])
        prev_val = builder.load(prev_ptr)

        vi_ptr = builder.gep(vi, [ctx.int32_ty(0), index])
        vi_val = builder.load(vi_ptr)

        new_val = builder.fmul(vi_val, rate)

        ret = builder.fadd(prev_val, new_val)
        ret = builder.fadd(ret, noise)
        res = builder.fadd(ret, offset)

        vo_ptr = builder.gep(vo, [ctx.int32_ty(0), index])
        builder.store(res, vo_ptr)
        builder.store(res, prev_ptr)

    def as_expression(self):
        return f'previous_value + ({MODEL_SPEC_ID_MDF_VARIABLE} * rate) + noise + offset'


class AdaptiveIntegrator(IntegratorFunction):  # -----------------------------------------------------------------------
    """
    AdaptiveIntegrator(         \
        default_variable=None,  \
        rate=1.0,               \
        noise=0.0,              \
        offset=0.0,             \
        initializer=None,       \
        params=None,            \
        owner=None,             \
        prefs=None,             \
        )

    .. _AdaptiveIntegrator:

    `function <AdaptiveIntegrator._function>` returns `exponentially weighted moving average (EWMA)
    <https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average>`_ of input:

    .. math::
        ((1-rate) * previous_value) + (rate * variable)  + noise + offset

    *Modulatory Parameters:*

    | *MULTIPLICATIVE_PARAM:* `rate <AdaptiveIntegrator.rate>`
    | *ADDITIVE_PARAM:* `offset <AdaptiveIntegrator.offset>`
    |

    Arguments
    ---------

    default_variable : number, list or array : default class_defaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float, list or 1d array : default 1.0
        specifies the smoothing factor of the `EWMA <AdaptiveIntegrator>`.  If it is a list or array, it must be the
        same length as `variable <AdaptiveIntegrator.variable>` (see `rate <AdaptiveIntegrator.rate>` for
        details).

    noise : float, function, list or 1d array : default 0.0
        specifies random value added to integral in each call to `function <AdaptiveIntegrator._function>`;
        if it is a list or array, it must be the same length as `variable <AdaptiveIntegrator.variable>`
        (see `noise <Integrator.noise>` for details).

    offset : float, list or 1d array : default 0.0
        specifies constant value added to integral in each call to `function <AdaptiveIntegrator._function>`;
        if it is a list or array, it must be the same length as `variable <AdaptiveIntegrator.variable>`
        (see `offset <AdaptiveIntegrator.offset>` for details).

    initializer : float, list or 1d array : default 0.0
        specifies starting value(s) for integration.  If it is a list or array, it must be the same length as
        `variable <AdaptiveIntegrator.variable>` (see `initializer <IntegratorFunction.initializer>` for details).

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
        current input value some portion of which (determined by `rate <AdaptiveIntegrator.rate>`) will be
        added to the prior value;  if it is an array, each element is independently integrated.

    rate : float or 1d array
        determines the smoothing factor of the `EWMA <AdaptiveIntegrator>`. All rate elements must be between 0 and 1
        (rate = 0 --> no change, `variable <AdaptiveIntegrator.variable>` is ignored; rate = 1 -->
        `previous_value <AdaptiveIntegrator.previous_value>` is ignored).  If rate is a float or has a single element,
        its value is applied to all elements of `variable <AdaptiveIntegrator.variable>` and `previous_value
        <AdaptiveIntegrator.previous_value>`; if it is an array, each element is applied to the corresponding element
        of `variable <AdaptiveIntegrator.variable>` and `previous_value <AdaptiveIntegrator.previous_value>`).
        Serves as *MULTIPLICATIVE_PARAM*  for `modulation <ModulatorySignal_Modulation>` of `function
        <AdaptiveIntegrator._function>`.

    noise : float, Function or 1d array
        random value added to integral in each call to `function <AdaptiveIntegrator._function>`
        (see `noise <Integrator.noise>` for details).

    offset : float or 1d array
        constant value added to integral in each call to `function <AdaptiveIntegrator._function>`.
        If `variable <AdaptiveIntegrator.variable>` is a list or array and offset is a float, offset is applied
        to each element of the integral;  if offset is a list or array, each of its elements is applied to each of
        the corresponding elements of the integral (i.e., Hadamard addition). Serves as *ADDITIVE_PARAM* for
        `modulation <ModulatorySignal_Modulation>` of `function <AdaptiveIntegrator._function>`.

    initializer : float or 1d array
        determines the starting value(s) for integration (i.e., the value(s) to which `previous_value
        <AdaptiveIntegrator.previous_value>` is set (see `initializer <IntegratorFunction.initializer>` for details).

    previous_value : 1d array : default class_defaults.variable
        stores previous value with which `variable <AdaptiveIntegrator.variable>` is integrated.

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

    componentName = ADAPTIVE_INTEGRATOR_FUNCTION

    class Parameters(IntegratorFunction.Parameters):
        """
            Attributes
            ----------

                offset
                    see `offset <AdaptiveIntegrator.offset>`

                    :default value: 0.0
                    :type: ``float``

                rate
                    see `rate <AdaptiveIntegrator.rate>`

                    :default value: 1.0
                    :type: ``float``
        """
        rate = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM], function_arg=True)
        offset = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM], function_arg=True)

    @check_user_specified
    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate=None,
                 noise=None,
                 offset=None,
                 initializer=None,
                 params: tc.optional(tc.optional(dict)) = None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

        super().__init__(
            default_variable=default_variable,
            rate=rate,
            noise=noise,
            offset=offset,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _validate_params(self, request_set, target_set=None, context=None):
        super()._validate_params(
            request_set=request_set,
            target_set=target_set,
            context=context
        )

        # Handle list or array for rate specification
        if RATE in request_set:
            rate = request_set[RATE]
            if isinstance(rate, (list, np.ndarray)):
                if len(rate) != 1 and len(rate) != np.array(self.defaults.variable).size:
                    # If the variable was not specified, then reformat it to match rate specification
                    #    and assign class_defaults.variable accordingly
                    # Note: this situation can arise when the rate is parametrized (e.g., as an array) in the
                    #       AdaptiveIntegrator's constructor, where that is used as a specification for a function
                    #       parameter (e.g., for an IntegratorMechanism), whereas the input is specified as part of the
                    #       object to which the function parameter belongs (e.g., the IntegratorMechanism);
                    #       in that case, the IntegratorFunction gets instantiated using its class_defaults.variable ([[0]])
                    #       before the object itself, thus does not see the array specification for the input.
                    if self._variable_shape_flexibility is DefaultsFlexibility.FLEXIBLE:
                        self._instantiate_defaults(variable=np.zeros_like(np.array(rate)), context=context)
                        if self.verbosePref:
                            warnings.warn(
                                "The length ({}) of the array specified for the {} parameter ({}) of {} "
                                "must match the length ({}) of the default input ({});  "
                                "the default input has been updated to match".
                                    format(len(rate), repr(RATE), rate, self.name,
                                    np.array(self.defaults.variable).size, self.defaults.variable))
                    else:
                        raise FunctionError(
                            f"The length ({len(rate)}) of the array specified for the rate parameter ({rate}) "
                            f"of {self.name} must match the length ({np.array(self.defaults.variable).size}) "
                            f"of the default input ({self.defaults.variable}).")

        # FIX: 12/9/18 [JDC] REPLACE WITH USE OF all_within_range
        if self.parameters.rate._user_specified:
            # cannot use _validate_rate here because it assumes it's being run after instantiation of the object
            rate_value_msg = "The rate parameter ({}) (or all of its elements) of {} " \
                             "must be between 0.0 and 1.0 because it is an AdaptiveIntegrator"

            rate = self.defaults.rate

            if isinstance(rate, np.ndarray) and rate.ndim > 0:
                for r in rate:
                    if r < 0.0 or r > 1.0:
                        raise FunctionError(rate_value_msg.format(rate, self.name))
            else:
                if rate < 0.0 or rate > 1.0:
                    raise FunctionError(rate_value_msg.format(rate, self.name))

        if NOISE in target_set:
            noise = target_set[NOISE]
            if isinstance(noise, DistributionFunction):
                noise.owner = self
                target_set[NOISE] = noise.execute
            self._validate_noise(target_set[NOISE])
            # if INITIALIZER in target_set:
            #     self._validate_initializer(target_set[INITIALIZER])

    def _validate_rate(self, rate):
        super()._validate_rate(rate)

        if isinstance(rate, list):
            rate = np.asarray(rate)

        rate_value_msg = "The rate parameter ({}) (or all of its elements) of {} " \
                         "must be between 0.0 and 1.0 because it is an AdaptiveIntegrator"

        if not all_within_range(rate, 0, 1):
            raise FunctionError(rate_value_msg.format(rate, self.name))

    def _gen_llvm_integrate(self, builder, index, ctx, vi, vo, params, state):
        rate = self._gen_llvm_load_param(ctx, builder, params, index, RATE)
        offset = self._gen_llvm_load_param(ctx, builder, params, index, OFFSET)
        noise = self._gen_llvm_load_param(ctx, builder, params, index, NOISE,
                                          state=state)

        # Get the only context member -- previous value
        prev_ptr = pnlvm.helpers.get_state_ptr(builder, self, state, "previous_value")
        # Get rid of 2d array. When part of a Mechanism the input,
        # (and output, and context) are 2d arrays.
        prev_ptr = pnlvm.helpers.unwrap_2d_array(builder, prev_ptr)
        assert len(prev_ptr.type.pointee) == len(vi.type.pointee)

        prev_ptr = builder.gep(prev_ptr, [ctx.int32_ty(0), index])
        prev_val = builder.load(prev_ptr)

        vi_ptr = builder.gep(vi, [ctx.int32_ty(0), index])
        vi_val = builder.load(vi_ptr)

        rev_rate = builder.fsub(ctx.float_ty(1), rate)
        old_val = builder.fmul(prev_val, rev_rate)
        new_val = builder.fmul(vi_val, rate)

        ret = builder.fadd(old_val, new_val)
        ret = builder.fadd(ret, noise)
        res = builder.fadd(ret, offset)

        vo_ptr = builder.gep(vo, [ctx.int32_ty(0), index])
        builder.store(res, vo_ptr)
        builder.store(res, prev_ptr)

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        variable : number, list or array : default class_defaults.variable
           a single value or array of values to be integrated.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        updated value of integral : ndarray (dimension equal to variable)

        """
        rate = np.array(self._get_current_parameter_value(RATE, context)).astype(float)
        offset = self._get_current_parameter_value(OFFSET, context)
        # execute noise if it is a function
        noise = self._try_execute_param(self._get_current_parameter_value(NOISE, context), variable, context=context)

        # # MODIFIED 6/14/19 OLD:
        # previous_value = np.atleast_2d(self.parameters.previous_value._get(context))
        # # MODIFIED 6/14/19 NEW: [JDC]
        previous_value = self.parameters.previous_value._get(context)
        # MODIFIED 6/14/19 END

        try:
            value = self._EWMA_filter(previous_value, rate, variable) + noise
        except TypeError:
            # TODO: this should be standardized along with the other instances
            # of this error
            raise FunctionError("Unrecognized type for {} of {} ({})".format(VARIABLE, self.name, variable))

        adjusted_value = value + offset

        # If this NOT an initialization run, update the old value
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if not self.is_initializing:
            self.parameters.previous_value._set(adjusted_value, context)

        # # MODIFIED 6/21/19 OLD:
        # return self.convert_output_type(adjusted_value)
        # MODIFIED 6/21/19 NEW: [JDC]
        return self.convert_output_type(adjusted_value, variable)
        # MODIFIED 6/21/19 END

    def as_expression(self):
        return f'(1 - rate) * previous_value + rate * {MODEL_SPEC_ID_MDF_VARIABLE} + noise + offset'


S_MINUS_L = 's-l'
L_MINUS_S = 'l-s'
OPERATIONS = {PRODUCT, SUM, S_MINUS_L, L_MINUS_S}


class DualAdaptiveIntegrator(IntegratorFunction):  # ------------------------------------------------------------------
    """
    DualAdaptiveIntegrator(         \
        default_variable=None,       \
        initializer=None,            \
        initial_short_term_avg=0.0,  \
        initial_long_term_avg=0.0,   \
        short_term_gain=1.0,         \
        long_term_gain=1.0,          \
        short_term_bias=0.0,         \
        long_term_bias=0.0,          \
        short_term_rate=1.0,         \
        long_term_rate=1.0,          \
        operation=PRODUCT,           \
        offset=0.0,                  \
        params=None,                 \
        owner=None,                  \
        prefs=None,                  \
        )

    .. _DualAdaptiveIntegrator:

    Combines two `exponentially weighted moving averages (EWMA)
    <https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average>`_ of its input, each with a different
    rate, as implemented in `Aston-Jones & Cohen (2005)
    <https://www.annualreviews.org/doi/abs/10.1146/annurev.neuro.28.061604.135709>`_ to integrate utility over two
    time scales.

    `function <DualAdaptiveIntegrator._function>` computes the EWMA of `variable <DualAdaptiveIntegrator.variable>`
    using two integration rates (`short_term_rate <DualAdaptiveIntegrator.short_term_rate>` and `long_term_rate
    <DualAdaptiveIntegrator.long_term_rate>`), transforms each using a logistic function, and then combines them,
    as follows:

    * **short time scale integral**:

      .. math::
         short\\_term\\_avg = short\\_term\\_rate \\cdot variable + (1 - short\\_term\\_rate) \\cdot
         previous\\_short\\_term\\_avg

    * **long time scale integral**:

      .. math::
         long\\_term\\_avg = long\\_term\\_rate \\cdot variable + (1 - long\\_term\\_rate) \\cdot
         previous\\_long\\_term\\_avg

    .. _DualAdaptive_Combined:

    * **combined integral**:

      .. math::
         value = operation(1-\\frac{1}{1+e^{short\\_term\\_gain\\ \\cdot\\ short\\_term\\_avg\\ +\\
         short\\_term\\_bias}},\\
         \\frac{1}{1+e^{long\\_term\\_gain\\ \\cdot\\ long\\_term\\_avg + long\\_term\\_bias}})\\ +\\ offset

      where *operation* is the arithmetic `operation <DualAdaptiveIntegrator.operation>` used to combine the terms.


    *Modulatory Parameters:*

    | *ADDITIVE_PARAM:* `offset <AdaptiveIntegrator.offset>`
    |

    Arguments
    ---------

    COMMENT:
    noise : float, function, list or 1d array : default 0.0
        TBI?
    COMMENT

    initial_short_term_avg : float : default 0.0
        specifies starting value for integration of short_term_avg

    initial_long_term_avg : float : default 0.0
        specifies starting value for integration of long_term_avg

    short_term_gain : float : default 1.0
        specifies gain for logistic function applied to short_term_avg

    long_term_gain : float : default 1.0
        specifies gain for logistic function applied to long_term_avg

    short_term_bias : float : default 0.0
        specifies bias for logistic function applied to short_term_avg

    long_term_bias : float : default 0.0
        specifies bias for logistic function applied to long_term_avg

    short_term_rate : float : default 1.0
        specifies smoothing factor of `EWMA <DualAdaptiveIntegrator>` filter applied to short_term_avg

    long_term_rate : float : default 1.0
        specifies smoothing factor of `EWMA <DualAdaptiveIntegrator>` filter applied to long_term_avg

    COMMENT:
    rate : float or 1d array
        determines weight assigned to `short_term_logistic and long_term_logistic <DualAdaptive_Combined>` when combined
        by `operation <DualAdaptiveIntegrator.operation>` (see `rate <DualAdaptiveIntegrator.rate>` for details.
    COMMENT

    operation : PRODUCT, SUM, S_MINUS_L or L_MINUS_S : default PRODUCT
        specifies the arithmetic operation used to combine the logistics of the short_term_avg and long_term_avg
        (see `operation <DualAdaptiveIntegrator.operation>` for details).

    offset : float or 1d array
        constant value added to integral in each call to `function <DualAdaptiveIntegrator._function>` after logistics
        of short_term_avg and long_term_avg are combined; if it is a list or array, it must be the same length as
        `variable <DualAdaptiveIntegrator.variable>` (see `offset <DualAdaptiveIntegrator.offset>` for details.

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
        current input value used to compute both the short term and long term `EWMA <DualAdaptiveIntegrator>` averages.

    COMMENT:
    noise : float, Function or 1d array : default 0.0
        TBI?
    COMMENT

    initial_short_term_avg : float
        determines starting value for integration of short_term_avg

    initial_long_term_avg : float
        determines starting value for integration of long_term_avg

    short_term_gain : float
        determines gain for logistic function applied to short_term_avg

    long_term_gain : float
        determines gain for logistic function applied to long_term_avg

    short_term_bias : float
        determines bias for logistic function applied to short_term_avg

    long_term_bias : float
        determines bias for logistic function applied to long_term_avg

    short_term_rate : float
        determines smoothing factor of `EWMA <DualAdaptiveIntegrator>` filter applied to short_term_avg

    long_term_rate : float
        determines smoothing factor of `EWMA <DualAdaptiveIntegrator>` filter applied to long_term_avg

    operation : str
        determines the arithmetic operation used to combine `short_term_logistic and long_term_logistic
        <DualAdaptive_Combined>`:

        * *PRODUCT* = (1 - short_term_logistic) * long_term_logistic
        * *SUM* = (1 - short_term_logistic) + long_term_logistic
        * *S_MINUS_L* = (1 - short_term_logistic) - long_term_logistic
        * *L_MINUS_S* = long_term_logistic - (1 - short_term_logistic)

    COMMENT:
    rate : float or 1d array with element(s) in interval [0,1]: default 0.5
        determines the linearly-weighted contribution of `short_term_logistic and long_term_logistic
        <DualAdaptive_Combined>` to the integral.  For rate=0.5, each receives and equal weight of 1;  for rate<0.5,
        short_term_avg diminishes linearly to 0 while long_term_avg remains at 1;  for rate>0.5, long_term_avg
        diminishes linearly to 0 wile short_term_avg remains at 1.  If it is a float or has a single element,
        its value is applied to all the elements of short_term_logistic and long_term_logistic; if it is an array,
        each element is applied to the corresponding elements of each logistic. Serves as *MULTIPLICATIVE_PARAM*
        for `modulation <ModulatorySignal_Modulation>` of `function <DualAdaptiveIntegrator._function>`.
    COMMENT

    offset : float or 1d array
        constant value added to integral in each call to `function <DualAdaptiveIntegrator._function>` after logistics
        of short_term_avg and long_term_avg are combined. If `variable <DualAdaptiveIntegrator.variable>` is an array
        and offset is a float, offset is applied to each element of the integral;  if offset is a list or array, each
        of its elements is applied to each of the corresponding elements of the integral (i.e., Hadamard addition).
        Serves as *ADDITIVE_PARAM* for `modulation <ModulatorySignal_Modulation>` of `function
        <DualAdaptiveIntegrator._function>`.

    previous_short_term_avg : 1d array
        stores previous value with which `variable <DualAdaptiveIntegrator.variable>` is integrated using the
        `EWMA <DualAdaptiveIntegrator>` filter and short term parameters

    previous_long_term_avg : 1d array
        stores previous value with which `variable <DualAdaptiveIntegrator.variable>` is integrated using the
        `EWMA <DualAdaptiveIntegrator>` filter and long term parameters

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

    componentName = DUAL_ADAPTIVE_INTEGRATOR_FUNCTION

    class Parameters(IntegratorFunction.Parameters):
        """
            Attributes
            ----------

                initial_long_term_avg
                    see `initial_long_term_avg <DualAdaptiveIntegrator.initial_long_term_avg>`

                    :default value: 0.0
                    :type: ``float``

                initial_short_term_avg
                    see `initial_short_term_avg <DualAdaptiveIntegrator.initial_short_term_avg>`

                    :default value: 0.0
                    :type: ``float``

                long_term_bias
                    see `long_term_bias <DualAdaptiveIntegrator.long_term_bias>`

                    :default value: 0.0
                    :type: ``float``

                long_term_gain
                    see `long_term_gain <DualAdaptiveIntegrator.long_term_gain>`

                    :default value: 1.0
                    :type: ``float``

                long_term_logistic
                    see `long_term_logistic <DualAdaptiveIntegrator.long_term_logistic>`

                    :default value: None
                    :type:

                long_term_rate
                    see `long_term_rate <DualAdaptiveIntegrator.long_term_rate>`

                    :default value: 0.1
                    :type: ``float``

                offset
                    see `offset <DualAdaptiveIntegrator.offset>`

                    :default value: 0.0
                    :type: ``float``

                operation
                    see `operation <DualAdaptiveIntegrator.operation>`

                    :default value: `PRODUCT`
                    :type: ``str``

                previous_long_term_avg
                    see `previous_long_term_avg <DualAdaptiveIntegrator.previous_long_term_avg>`

                    :default value: None
                    :type:

                previous_short_term_avg
                    see `previous_short_term_avg <DualAdaptiveIntegrator.previous_short_term_avg>`

                    :default value: None
                    :type:

                rate
                    see `rate <DualAdaptiveIntegrator.rate>`

                    :default value: 0.5
                    :type: ``float``

                short_term_bias
                    see `short_term_bias <DualAdaptiveIntegrator.short_term_bias>`

                    :default value: 0.0
                    :type: ``float``

                short_term_gain
                    see `short_term_gain <DualAdaptiveIntegrator.short_term_gain>`

                    :default value: 1.0
                    :type: ``float``

                short_term_logistic
                    see `short_term_logistic <DualAdaptiveIntegrator.short_term_logistic>`

                    :default value: None
                    :type:

                short_term_rate
                    see `short_term_rate <DualAdaptiveIntegrator.short_term_rate>`

                    :default value: 0.9
                    :type: ``float``
        """
        rate = Parameter(0.5, modulable=True, aliases=[MULTIPLICATIVE_PARAM], function_arg=True)
        initial_short_term_avg = 0.0
        initial_long_term_avg = 0.0
        short_term_gain = Parameter(1.0, modulable=True, function_arg=True)
        long_term_gain = Parameter(1.0, modulable=True, function_arg=True)
        short_term_bias = Parameter(0.0, modulable=True, function_arg=True)
        long_term_bias = Parameter(0.0, modulable=True, function_arg=True)
        short_term_rate = Parameter(0.9, modulable=True, function_arg=True)
        long_term_rate = Parameter(0.1, modulable=True, function_arg=True)
        operation = PRODUCT
        offset = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM], function_arg=True)
        previous_short_term_avg = Parameter(None, initializer='initial_short_term_avg', pnl_internal=True)
        previous_long_term_avg = Parameter(None, initializer='initial_long_term_avg', pnl_internal=True)
        short_term_logistic = None
        long_term_logistic = None


    @check_user_specified
    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 # rate: parameter_spec = 0.5,
                 # noise=0.0,
                 initializer=None,
                 initial_short_term_avg=None,
                 initial_long_term_avg=None,
                 short_term_gain=None,
                 long_term_gain=None,
                 short_term_bias=None,
                 long_term_bias=None,
                 short_term_rate=None,
                 long_term_rate=None,
                 operation=None,
                 offset=None,
                 params: tc.optional(tc.optional(dict)) = None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            offset=offset,
            previous_long_term_avg=initial_long_term_avg,
            previous_short_term_avg=initial_short_term_avg,
            initial_short_term_avg=initial_short_term_avg,
            initial_long_term_avg=initial_long_term_avg,
            short_term_gain=short_term_gain,
            long_term_gain=long_term_gain,
            short_term_bias=short_term_bias,
            long_term_bias=long_term_bias,
            short_term_rate=short_term_rate,
            long_term_rate=long_term_rate,
            operation=operation,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _validate_params(self, request_set, target_set=None, context=None):

        # Handle list or array for rate specification
        if RATE in request_set:
            rate = request_set[RATE]
            if isinstance(rate, (list, np.ndarray)):
                if len(rate) != 1 and len(rate) != np.array(self.defaults.variable).size:
                    # If the variable was not specified, then reformat it to match rate specification
                    #    and assign class_defaults.variable accordingly
                    # Note: this situation can arise when the rate is parametrized (e.g., as an array) in the
                    #       DualAdaptiveIntegrator's constructor, where that is used as a specification for a function parameter
                    #       (e.g., for an IntegratorMechanism), whereas the input is specified as part of the
                    #       object to which the function parameter belongs (e.g., the IntegratorMechanism);
                    #       in that case, the IntegratorFunction gets instantiated using its class_defaults.variable ([[0]]) before
                    #       the object itself, thus does not see the array specification for the input.
                    if self._variable_shape_flexibility is DefaultsFlexibility.FLEXIBLE:
                        self._instantiate_defaults(variable=np.zeros_like(np.array(rate)), context=context)
                        if self.verbosePref:
                            warnings.warn(
                                "The length ({}) of the array specified for the rate parameter ({}) of {} "
                                "must match the length ({}) of the default input ({});  "
                                "the default input has been updated to match".format(
                                    len(rate),
                                    rate,
                                    self.name,
                                    np.array(self.defaults.variable).size
                                ),
                                self.defaults.variable
                            )
                    else:
                        raise FunctionError(
                            "The length ({}) of the array specified for the rate parameter ({}) of {} "
                            "must match the length ({}) of the default input ({})".format(
                                len(rate),
                                rate,
                                self.name,
                                np.array(self.defaults.variable).size,
                                self.defaults.variable,
                            )
                        )

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if RATE in target_set:
            # if isinstance(target_set[RATE], (list, np.ndarray)):
            #     for r in target_set[RATE]:
            #         if r < 0.0 or r > 1.0:
            #             raise FunctionError("The rate parameter ({}) (or all of its elements) of {} must be "
            #                                 "between 0.0 and 1.0 when integration_type is set to ADAPTIVE.".
            #                                 format(target_set[RATE], self.name))
            # else:
            #     if target_set[RATE] < 0.0 or target_set[RATE] > 1.0:
            #         raise FunctionError(
            #             "The rate parameter ({}) (or all of its elements) of {} must be between 0.0 and "
            #             "1.0 when integration_type is set to ADAPTIVE.".format(target_set[RATE], self.name))
            if target_set[RATE] is not None and not all_within_range(target_set[RATE], 0, 1):
                raise FunctionError("The rate parameter ({}) (or all of its elements) of {} "
                                    "must be in the interval [0,1]".format(target_set[RATE], self.name))

        if NOISE in target_set:
            noise = target_set[NOISE]
            if isinstance(noise, DistributionFunction):
                noise.owner = self
                target_set[NOISE] = noise.execute
            self._validate_noise(target_set[NOISE])
            # if INITIALIZER in target_set:
            #     self._validate_initializer(target_set[INITIALIZER])

        if OPERATION in target_set:
            if target_set[OPERATION] is not None and not target_set[OPERATION] in OPERATIONS:
                raise FunctionError("\'{}\' arg for {} must be one of the following: {}".
                                    format(OPERATION, self.name, OPERATIONS))

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        variable : number, list or array : default class_defaults.variable
           a single value or array of values to be integrated.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        updated value of integral : 2d array

        """
        # rate = np.array(self._get_current_parameter_value(RATE, context)).astype(float)
        # execute noise if it is a function
        # noise = self._try_execute_param(self._get_current_parameter_value(NOISE, context), variable, context=context)
        short_term_rate = self._get_current_parameter_value("short_term_rate", context)
        long_term_rate = self._get_current_parameter_value("long_term_rate", context)

        # Integrate Short Term Utility:
        short_term_avg = self._EWMA_filter(short_term_rate,
                                           self.previous_short_term_avg,
                                           variable)
        # Integrate Long Term Utility:
        long_term_avg = self._EWMA_filter(long_term_rate,
                                          self.previous_long_term_avg,
                                          variable)

        value = self._combine_terms(short_term_avg, long_term_avg, context=context)

        if not self.is_initializing:
            self.parameters.previous_short_term_avg._set(short_term_avg, context)
            self.parameters.previous_long_term_avg._set(long_term_avg, context)

        return self.convert_output_type(value)

    def _combine_terms(self, short_term_avg, long_term_avg, context=None):

        short_term_gain = self._get_current_parameter_value("short_term_gain", context)
        short_term_bias = self._get_current_parameter_value("short_term_bias", context)
        long_term_gain = self._get_current_parameter_value("long_term_gain", context)
        long_term_bias = self._get_current_parameter_value("long_term_bias", context)
        rate = self._get_current_parameter_value(RATE, context)
        operation = self._get_current_parameter_value(OPERATION, context)
        offset = self._get_current_parameter_value(OFFSET, context)

        # s = 2*rate if rate <= 0.5 else 1
        # l = 2-(2*rate) if rate >= 0.5 else 1

        short_term_logistic = self._logistic(variable=short_term_avg,
                                             gain=short_term_gain,
                                             bias=short_term_bias,
                                             )
        self.parameters.short_term_logistic._set(short_term_logistic, context)

        long_term_logistic = self._logistic(variable=long_term_avg,
                                            gain=long_term_gain,
                                            bias=long_term_bias,
                                            )
        self.parameters.long_term_logistic._set(long_term_logistic, context)

        if operation == PRODUCT:
            value = (1 - short_term_logistic) * long_term_logistic
        elif operation == SUM:
            value = (1 - short_term_logistic) + long_term_logistic
        elif operation == S_MINUS_L:
            value = (1 - short_term_logistic) - long_term_logistic
        elif operation == L_MINUS_S:
            value = long_term_logistic - (1 - short_term_logistic)

        return value + offset

    @handle_external_context(fallback_most_recent=True)
    def reset(self, short=None, long=None, context=NotImplemented):
        """
        Effectively begins accumulation over again at the specified utilities.

        Sets `previous_short_term_avg <DualAdaptiveIntegrator.previous_short_term_avg>` to the quantity specified
        in the first argument and `previous_long_term_avg <DualAdaptiveIntegrator.previous_long_term_avg>` to the
        quantity specified in the second argument.

        Sets `value <DualAdaptiveIntegrator.value>` by computing it based on the newly updated values for
        `previous_short_term_avg <DualAdaptiveIntegrator.previous_short_term_avg>` and
        `previous_long_term_avg <DualAdaptiveIntegrator.previous_long_term_avg>`.

        If no arguments are specified, then the current values of `initial_short_term_avg
        <DualAdaptiveIntegrator.initial_short_term_avg>` and `initial_long_term_avg
        <DualAdaptiveIntegrator.initial_long_term_avg>` are used.
        """
        if short is None:
            short = self._get_current_parameter_value("initial_short_term_avg", context)
        if long is None:
            long = self._get_current_parameter_value("initial_long_term_avg", context)

        self.parameters.previous_short_term_avg.set(short, context)
        self.parameters.previous_long_term_avg.set(long, context)
        value = self._combine_terms(short, long, context)

        self.parameters.value.set(value, context, override=True)
        return value


class InteractiveActivationIntegrator(IntegratorFunction):  # ----------------------------------------------------------
    """
    InteractiveActivationIntegrator(      \
        default_variable=None,  \
        rate=1.0,               \
        decay=1.0,              \
        rest=0.0,               \
        max_val=1.0,            \
        min_val=-1.0,           \
        noise=0.0,              \
        initializer=None,       \
        params=None,            \
        owner=None,             \
        prefs=None,             \
        )

    .. _InteractiveActivationIntegrator:

    Implements a generalized version of the interactive activation from `McClelland and Rumelhart (1981)
    <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.298.4480&rep=rep1&type=pdf>`_ that integrates current
    value of `variable <InteractiveActivationIntegrator.variable>` toward an asymptotic maximum value `max_val
    <InteractiveActivationIntegrator.max_val>` for positive inputs and toward an asymptotic mininum value (`min_val
    <InteractiveActivationIntegrator.min_val>`) for negative inputs, and decays asymptotically towards an intermediate
    resting value (`rest <InteractiveActivationIntegrator.rest>`).

    `function <InteractiveActivationIntegrator._function>` returns:

    .. math::
        previous\\_value + (rate * (variable + noise) * distance\\_from\\_asymptote) - (decay * distance\\_from\\_rest)

    where:

    .. math::
        if\\ variable > 0,\\ distance\\_from\\_asymptote = max\\_val - previous\\_value

    .. math::
        if\\ variable < 0,\\ distance\\_from\\_asymptote = previous\\_value - min\\_val

    .. math::
        if\\ variable = 0,\\ distance\\_from\\_asymptote = 0


    Arguments
    ---------

    default_variable : number, list or array : default class_defaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float, list or 1d array : default 1.0
        specifies the rate of change in activity; its value(s) must be in the interval [0,1].  If it is a list or
        array, it must be the same length as `variable <InteractiveActivationIntegrator.variable>`.

    decay : float, list or 1d array : default 1.0
        specifies the rate of at which activity decays toward `rest <InteractiveActivationIntegrator.rest>`.
        If it is a list or array, it must be the same length as `variable <InteractiveActivationIntegrator.variable>`;
        its value(s) must be in the interval [0,1].

    rest : float, list or 1d array : default 0.0
        specifies the initial value and one toward which value `decays <InteractiveActivationIntegrator.decay>`.
        If it is a list or array, it must be the same length as `variable <InteractiveActivationIntegrator.variable>`.
        COMMENT:
        its value(s) must be between `max_val <InteractiveActivationIntegrator.max_val>` and `min_val
        <InteractiveActivationIntegrator.min_val>`.
        COMMENT

    max_val : float, list or 1d array : default 1.0
        specifies the maximum asymptotic value toward which integration occurs for positive values of `variable
        <InteractiveActivationIntegrator.variable>`.  If it is a list or array, it must be the same length as `variable
        <InteractiveActivationIntegrator.variable>`; all values must be greater than the corresponding values of
        `min_val <InteractiveActivationIntegrator.min_val>` (see `max_val <InteractiveActivationIntegrator.max_val>`
        for details).

    min_val : float, list or 1d array : default 1.0
        specifies the minimum asymptotic value toward which integration occurs for negative values of `variable
        <InteractiveActivationIntegrator.variable>`.  If it is a list or array, it must be the same length as `variable
        <InteractiveActivationIntegrator.variable>`; all values must be greater than the corresponding values of
        `max_val <InteractiveActivationIntegrator.min_val>` (see `max_val <InteractiveActivationIntegrator.min_val>`
        for details).

    noise : float, function, list or 1d array : default 0.0
        specifies random value added to `variable <InteractiveActivationIntegrator.variable>` in each call to `function
        <InteractiveActivationIntegrator._function>`; if it is a list or array, it must be the same length as `variable
        <IntegratorFunction.variable>` (see `noise <Integrator.noise>` for details).

    initializer : float, list or 1d array : default 0.0
        specifies starting value(s) for integration.  If it is a list or array, it must be the same length as `variable
        <InteractiveActivationIntegrator.variable>` (see `initializer <IntegratorFunction.initializer>` for details).

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
        current input value some portion of which (determined by `rate <InteractiveActivationIntegrator.rate>`) will be
        added to the prior value;  if it is an array, each element is independently integrated.

    rate : float or 1d array in interval [0,1]
        determines the rate at which activity increments toward either `max_val
        <InteractiveActivationIntegrator.max_val>` (`variable <InteractiveActivationIntegrator.variable>` is
        positive) or `min_val <InteractiveActivationIntegrator.min_val>` (if `variable
        <InteractiveActivationIntegrator.variable>` is negative).  If it is a float or has a single element, it is
        applied to all elements of `variable <InteractiveActivationIntegrator.variable>`; if it has more than one
        element, each element is applied to the corresponding element of `variable
        <InteractiveActivationIntegrator.variable>`. Serves as *MULTIPLICATIVE_PARAM* for `modulation
        <ModulatorySignal_Modulation>` of `function <InteractiveActivationIntegrator._function>`.

    decay : float or 1d array
        determines the rate of at which activity decays toward `rest <InteractiveActivationIntegrator.rest>` (similary
        to *rate* in other IntegratorFuncgtions).  If it is a float or has a single element, it applies to all elements
        of `variable <InteractiveActivationIntegrator.variable>`;  if it has more than one element, each element applies
        to the corresponding element of `variable <InteractiveActivationIntegrator.variable>`.

    rest : float or 1d array
        determines the initial value and one toward which value `decays <InteractiveActivationIntegrator.decay>`
        (similar to *bias* in other IntegratorFunctions).  If it is a float or has a single element, it applies to
        all elements of `variable <InteractiveActivationIntegrator.variable>`;  if it has more than one element,
        each element applies to the corresponding element of `variable <InteractiveActivationIntegrator.variable>`.

    max_val : float or 1d array
        determines the maximum asymptotic value toward which integration occurs for positive values of `variable
        <InteractiveActivationIntegrator.variable>`.  If it is a float or has a single element, it applies to all
        elements of `variable <InteractiveActivationIntegrator.variable>`;  if it has more than one element,
        each element applies to the corresponding element of `variable <InteractiveActivationIntegrator.variable>`.

    min_val : float or 1d array
        determines the minimum asymptotic value toward which integration occurs for negative values of `variable
        <InteractiveActivationIntegrator.variable>`.  If it is a float or has a single element, it applies to all
        elements of `variable <InteractiveActivationIntegrator.variable>`;  if it has more than one element,
        each element applies to the corresponding element of `variable <InteractiveActivationIntegrator.variable>`.

    noise : float, Function or 1d array
        random value added to `variable <InteractiveActivationIntegrator.noise>` in each call to `function
        <InteractiveActivationIntegrator._function>` (see `noise <Integrator.noise>` for details).

    initializer : float or 1d array
        determines the starting value(s) for integration (i.e., the value(s) to which `previous_value
        <InteractiveActivationIntegrator.previous_value>` is set (see `initializer <IntegratorFunction.initializer>`
        for details).

    previous_value : 1d array : default class_defaults.variable
        stores previous value with which `variable <InteractiveActivationIntegrator.variable>` is integrated.

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

    componentName = INTERACTIVE_ACTIVATION_INTEGRATOR_FUNCTION


    class Parameters(IntegratorFunction.Parameters):
        """
            Attributes
            ----------

                decay
                    see `decay <InteractiveActivationIntegrator.decay>`

                    :default value: 0.0
                    :type: ``float``

                max_val
                    see `max_val <InteractiveActivationIntegrator.max_val>`

                    :default value: 1.0
                    :type: ``float``

                min_val
                    see `min_val <InteractiveActivationIntegrator.min_val>`

                    :default value: -1.0
                    :type: ``float``

                rate
                    see `rate <InteractiveActivationIntegrator.rate>`

                    :default value: 1.0
                    :type: ``float``

                rest
                    see `rest <InteractiveActivationIntegrator.rest>`

                    :default value: 0.0
                    :type: ``float``
        """
        rate = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM], function_arg=True)
        decay = Parameter(0.0, modulable=True, function_arg=True)
        rest = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM], function_arg=True)
        max_val = Parameter(1.0, function_arg=True)
        min_val = Parameter(-1.0, function_arg=True)

    @check_user_specified
    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate: tc.optional(parameter_spec) = None,
                 decay: tc.optional(parameter_spec) = None,
                 rest: tc.optional(parameter_spec) = None,
                 max_val: tc.optional(parameter_spec) = None,
                 min_val: tc.optional(parameter_spec) = None,
                 noise=None,
                 initializer=None,
                 params: tc.optional(tc.optional(dict)) = None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None,
                 # **kwargs
                 ):

        # # This may be passed (as standard IntegratorFunction arg) but is not used by IA
        # unsupported_args = {OFFSET}
        # if any(k in unsupported_args for k in kwargs):
        #     s = 's' if len(kwargs)>1 else ''
        #     warnings.warn("{} arg{} not supported in {}".
        #                   format(repr(", ".join(list(kwargs.keys()))),s, self.__class__.__name__))
        #     for k in unsupported_args:
        #         if k in kwargs:
        #             del kwargs[k]


        if initializer is None:
            initializer = rest
        if default_variable is None:
            default_variable = initializer

        super().__init__(
            default_variable=default_variable,
            rate=rate,
            decay=decay,
            rest=rest,
            max_val=max_val,
            min_val=min_val,
            initializer=initializer,
            noise=noise,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _validate_params(self, request_set, target_set=None, context=None):

        super()._validate_params(request_set=request_set, target_set=target_set,context=context)

        if RATE in request_set and request_set[RATE] is not None:
            rate = request_set[RATE]
            if np.isscalar(rate):
                rate = [rate]
            if not all_within_range(rate, 0, 1):
                raise FunctionError("Value(s) specified for {} argument of {} ({}) must be in interval [0,1]".
                                    format(repr(RATE), self.__class__.__name__, rate))

        if DECAY in request_set and request_set[DECAY] is not None:
            decay = request_set[DECAY]
            if np.isscalar(decay):
                decay = [decay]
            if not all(0.0 <= d <= 1.0 for d in decay):
                raise FunctionError("Value(s) specified for {} argument of {} ({}) must be in interval [0,1]".
                                    format(repr(DECAY), self.__class__.__name__, decay))

    def _function(self, variable=None, context=None, params=None):
        """

        Arguments
        ---------

        variable : number, list or array : default class_defaults.variable
           a single value or array of values to be integrated.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        updated value of integral : 2d array

        """
        rate = np.array(self._get_current_parameter_value(RATE, context)).astype(float)
        decay = np.array(self._get_current_parameter_value(DECAY, context)).astype(float)
        rest = np.array(self._get_current_parameter_value(REST, context)).astype(float)
        # FIX: only works with "max_val". Keyword MAX_VAL = "MAX_VAL", not max_val
        max_val = np.array(self._get_current_parameter_value("max_val", context)).astype(float)
        min_val = np.array(self._get_current_parameter_value("min_val", context)).astype(float)

        # execute noise if it is a function
        noise = self._try_execute_param(self._get_current_parameter_value(NOISE, context), variable, context=context)

        current_input = variable

        # FIX: ?CLEAN THIS UP BY SETTING initializer IN __init__ OR OTHER RELEVANT PLACE?
        if self.is_initializing:
            if rest.ndim == 0 or len(rest)==1:
                # self.parameters.previous_value._set(np.full_like(current_input, rest), context)
                self._initialize_previous_value(np.full_like(current_input, rest), context)
            elif np.atleast_2d(rest).shape == current_input.shape:
                # self.parameters.previous_value._set(rest, context)
                self._initialize_previous_value(rest, context)
            else:
                raise FunctionError("The {} argument of {} ({}) must be an int or float, "
                                    "or a list or array of the same length as its variable ({})".
                                    format(repr(REST), self.__class__.__name__, rest, len(variable)))
        previous_value = self.parameters.previous_value._get(context)

        current_input = np.atleast_2d(variable)
        prev_val = np.atleast_2d(previous_value)

        dist_from_asymptote = np.zeros_like(current_input, dtype=float)
        for i in range(len(current_input)):
            for j in range(len(current_input[i])):
                if current_input[i][j] > 0:
                    d = max_val - prev_val[i][j]
                elif current_input[i][j] < 0:
                    d = prev_val[i][j] - min_val
                else:
                    d = 0
                dist_from_asymptote[i][j] = d

        dist_from_rest = prev_val - rest

        new_value = previous_value + (rate * (current_input + noise) * dist_from_asymptote) - (decay * dist_from_rest)

        if not self.is_initializing:
            self.parameters.previous_value._set(new_value, context)

        return self.convert_output_type(new_value)


class DriftDiffusionIntegrator(IntegratorFunction):  # -----------------------------------------------------------------
    """
    DriftDiffusionIntegrator(           \
        default_variable=None,          \
        rate=1.0,                       \
        noise=0.0,                      \
        offset= 0.0,                    \
        non_decision_time=0.0,          \
        threshold=1.0                   \
        time_step_size=0.01,            \
        initializer=None,               \
        params=None,                    \
        owner=None,                     \
        prefs=None,                     \
        )

    .. _DriftDiffusionIntegrator:

    Accumulate "evidence" to a bound.  `function <DriftDiffusionIntegrator._function>` returns one
    time step of integration:

    ..  math::
        previous\\_value + rate \\cdot variable \\cdot time\\_step\\_size + \\mathcal{N}(\\sigma^2)

    where

    ..  math::
        \\sigma^2 =\\sqrt{time\\_step\\_size \\cdot noise}

    *Modulatory Parameters:*

    | *MULTIPLICATIVE_PARAM:* `rate <AdaptiveIntegrator.rate>`
    | *ADDITIVE_PARAM:* `offset <AdaptiveIntegrator.offset>`
    |

    Arguments
    ---------

    default_variable : number, list or array : default class_defaults.variable
        specifies the stimulus component of drift rate -- the drift rate is the product of variable and rate

    rate : float, list or 1d array : default 1.0
        applied multiplicatively to `variable <DriftDiffusionIntegrator.variable>`;  If it is a list or array, it must
        be the same length as `variable <DriftDiffusionIntegrator.variable>` (see `rate <DriftDiffusionIntegrator.rate>`
        for details).

    noise : float : default 0.0
        specifies a value by which to scale the normally distributed random value added to the integral in each call to
        `function <DriftDiffusionIntegrator._function>` (see `noise <DriftDiffusionIntegrator.noise>` for details).

    COMMENT:
    FIX: REPLACE ABOVE WITH THIS ONCE LIST/ARRAY SPECIFICATION OF NOISE IS FULLY IMPLEMENTED
    noise : float, list or 1d array : default 0.0
        specifies a value by which to scale the normally distributed random value added to the integral in each call to
        `function <DriftDiffusionIntegrator._function>`; if it is a list or array, it must be the same length as
        `variable <DriftDiffusionIntegrator.variable>` (see `noise <DriftDiffusionIntegrator.noise>` for details).
    COMMENT

    offset : float, list or 1d array : default 0.0
        specifies constant value added to integral in each call to `function <DriftDiffusionIntegrator._function>`
        if it's absolute value is below `threshold <DriftDiffusionIntegrator.threshold>`\;
        if it is a list or array, it must be the same length as `variable <DriftDiffusionIntegrator.variable>`
        (see `offset <DriftDiffusionIntegrator.offset>` for details).

    starting_value : float, list or 1d array:  default 0.0
        specifies the starting value for the integration process; if it is a list or array, it must be the
        same length as `variable <DriftDiffusionIntegrator.variable>` (see `starting_value
        <DriftDiffusionIntegrator.starting_value>` for details).

    threshold : float : default 0.0
        specifies the threshold (boundaries) of the drift diffusion process -- i.e., at which the
        integration process terminates (see `threshold <DriftDiffusionIntegrator.threshold>` for details).

    time_step_size : float : default 0.0
        specifies the timing precision of the integration process (see `time_step_size
        <DriftDiffusionIntegrator.time_step_size>` for details.

    initializer : float, list or 1d array : default 0.0
        specifies starting value(s) for integration.  If it is a list or array, it must be the same length as `variable
        <DriftDiffusionIntegrator.variable>` (see `initializer <IntegratorFunction.initializer>` for details).

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

    variable : float or array
        current input value (can be thought of as implementing the stimulus component of the drift rate);  if it is an
        array, each element represents an independently integrated decision variable.

    rate : float or 1d array
        applied multiplicatively to `variable <DriftDiffusionIntegrator.variable>` (can be thought of as implementing
        the attentional component of the drift rate).  If it is a float or has a single element, its value is applied
        to all the elements of `variable <DriftDiffusionIntegrator.variable>`; if it is an array, each element is
        applied to the corresponding element of `variable <DriftDiffusionIntegrator.variable>`. Serves as
        *MULTIPLICATIVE_PARAM* for `modulation <ModulatorySignal_Modulation>` of `function
        <DriftDiffusionIntegrator._function>`.

    random_state : numpy.RandomState
        private pseudorandom number generator

    noise : float or 1d array
        scales the normally distributed random value added to integral in each call to `function
        <DriftDiffusionIntegrator._function>`. If `variable <DriftDiffusionIntegrator.variable>` is a list or array,
        and noise is a float, a single random term is generated and applied for each element of `variable
        <DriftDiffusionIntegrator.variable>`.  If noise is a list or array, it must be the same length as `variable
        <DriftDiffusionIntegrator.variable>`, and a separate random term scaled by noise is applied for each of the
        corresponding elements of `variable <DriftDiffusionIntegrator.variable>`.

    offset : float or 1d array
        constant value added to integral in each call to `function <DriftDiffusionIntegrator._function>`
        if it's absolute value is below `threshold <DriftDiffusionIntegrator.threshold>`.
        If `variable <DriftDiffusionIntegrator.variable>` is an array and offset is a float, offset is applied
        to each element of the integral;  if offset is a list or array, each of its elements is applied to each of
        the corresponding elements of the integral (i.e., Hadamard addition). Serves as *ADDITIVE_PARAM* for
        `modulation <ModulatorySignal_Modulation>` of `function <DriftDiffusionIntegrator._function>`.

    starting_value : float or 1d array
        determines the starting value for the integration process; if it is a list or array, it must be the
        same length as `variable <DriftDiffusionIntegrator.variable>`. If `variable <DriftDiffusionIntegrator.variable>`
        is an array and starting_value is a float, starting_value is used for each element of the integral;  if
        starting_value is a list or array, each of its elements is used as the starting point for each element of the
        integral.

    non_decision_time : float : default 0.0
        specifies the starting time of the model and is used to compute `previous_time
        <DriftDiffusionIntegrator.previous_time>`

    threshold : float
        determines the boundaries of the drift diffusion process:  the integration process can be scheduled to
        terminate when the result of `function <DriftDiffusionIntegrator._function>` equals or exceeds either the
        positive or negative value of threshold (see hint).
        NOTE: Vector version of this parameter acts as a saturation barrier.
        While it is possible to subtract from value == threshold, any movement
        in the threshold direction will be capped at the threshold value.

        .. hint::
           To terminate execution of the `Mechanism <Mechanism>` to which the `function
           <DriftDiffusionIntegrator._function>` is assigned, a `WhenFinished` `Condition` should be assigned for that
           Mechanism to `scheduler <Composition.scheduler>` of the `Composition` to which the Mechanism belongs.

    time_step_size : float
        determines the timing precision of the integration process and is used to scale the `noise
        <DriftDiffusionIntegrator.noise>` parameter according to the standard DDM probability distribution.

    initializer : float or 1d array
        determines the starting value(s) for integration (i.e., the value(s) to which `previous_value
        <DriftDiffusionIntegrator.previous_value>` is set (see `initializer <IntegratorFunction.initializer>`
        for details).

    previous_time : float
        stores previous time at which the function was executed and accumulates with each execution according to
        `time_step_size <DriftDiffusionIntegrator.default_time_step_size>`.

    previous_value : 1d array : default class_defaults.variable
        stores previous value with which `variable <DriftDiffusionIntegrator.variable>` is integrated.

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

    componentName = DRIFT_DIFFUSION_INTEGRATOR_FUNCTION

    class Parameters(IntegratorFunction.Parameters):
        """
            Attributes
            ----------

                enable_output_type_conversion
                    see `enable_output_type_conversion <DriftDiffusionIntegrator.enable_output_type_conversion>`

                    :default value: False
                    :type: ``bool``
                    :read only: True

                offset
                    see `offset <DriftDiffusionIntegrator.offset>`

                    :default value: 0.0
                    :type: ``float``

                previous_time
                    see `previous_time <DriftDiffusionIntegrator.previous_time>`

                    :default value: None
                    :type:

                random_state
                    see `random_state <DriftDiffusionIntegrator.random_state>`

                    :default value: None
                    :type: ``numpy.random.RandomState``

                rate
                    see `rate <DriftDiffusionIntegrator.rate>`

                    :default value: 1.0
                    :type: ``float``

                seed
                    see `seed <DriftDiffusionIntegrator.seed>`

                    :default value: None
                    :type:
                    :read only: True

                non_decision_time
                    see `non_decision_time <DriftDiffusionIntegrator.non_decision_time>`

                    :default value: 0.0
                    :type: ``float``

                threshold
                    see `threshold <DriftDiffusionIntegrator.threshold>`

                    :default value: 100.0
                    :type: ``float``

                time_step_size
                    see `time_step_size <DriftDiffusionIntegrator.time_step_size>`

                    :default value: 1.0
                    :type: ``float``
        """
        rate = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        offset = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        initializer = Parameter(np.array([0]), aliases=['starting_value'])
        non_decision_time = Parameter(0.0, modulable=True)
        threshold = Parameter(100.0, modulable=True)
        time_step_size = Parameter(1.0, modulable=True)
        previous_time = Parameter(None, initializer='non_decision_time', pnl_internal=True)
        random_state = Parameter(None, loggable=False, getter=_random_state_getter, dependencies='seed')
        seed = Parameter(DEFAULT_SEED, modulable=True, fallback_default=True, setter=_seed_setter)
        enable_output_type_conversion = Parameter(
            False,
            stateful=False,
            loggable=False,
            pnl_internal=True,
            read_only=True
        )

        def _parse_initializer(self, initializer):
            if initializer.ndim > 1:
                return np.atleast_1d(initializer.squeeze())
            else:
                return initializer

    @check_user_specified
    @tc.typecheck
    def __init__(
        self,
        default_variable=None,
        rate: tc.optional(parameter_spec) = None,
        noise=None,
        offset: tc.optional(parameter_spec) = None,
        starting_value=None,
        non_decision_time=None,
        threshold=None,
        time_step_size=None,
        seed=None,
        params: tc.optional(tc.optional(dict)) = None,
        owner=None,
        prefs: tc.optional(is_pref_set) = None,
        **kwargs
    ):

        # Assign here as default, for use in initialization of function
        super().__init__(
            default_variable=default_variable,
            rate=rate,
            time_step_size=time_step_size,
            starting_value=starting_value,
            non_decision_time=non_decision_time,
            threshold=threshold,
            noise=noise,
            offset=offset,
            seed=seed,
            params=params,
            owner=owner,
            prefs=prefs,
            **kwargs
        )

    def _validate_noise(self, noise):
        if noise is not None and not isinstance(noise, float) and not (isinstance(noise, np.ndarray) and np.issubdtype(noise.dtype, np.floating)):
            raise FunctionError(
                "Invalid noise parameter for {}: {}. DriftDiffusionIntegrator requires noise parameter to be a float or float array."
                " Noise parameter is used to construct the standard DDM noise distribution".format(self.name, type(noise)))

    def _initialize_previous_value(self, initializer, context=None):
        return super()._initialize_previous_value(self.parameters._parse_initializer(initializer), context)

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        variable : number, list or array : default class_defaults.variable
           a single value or array of values to be integrated (can be thought of as the stimulus component of
           the drift rate).

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        updated value of integral : 2d array

        """
        rate = np.array(self._get_current_parameter_value(RATE, context)).astype(float)
        noise = self._get_current_parameter_value(NOISE, context)
        offset = self._get_current_parameter_value(OFFSET, context)
        threshold = self._get_current_parameter_value(THRESHOLD, context)
        time_step_size = self._get_current_parameter_value(TIME_STEP_SIZE, context)
        random_state = self._get_current_parameter_value("random_state", context)

        variable = self.parameters._parse_initializer(variable)
        previous_value = self.parameters.previous_value._get(context)

        random_draw = np.array([random_state.normal() for _ in list(variable)])
        value = previous_value + rate * variable * time_step_size \
                + noise * np.sqrt(time_step_size) * random_draw

        adjusted_value = np.clip(value + offset, -threshold, threshold)

        # If this NOT an initialization run, update the old value and time
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        previous_time = self._get_current_parameter_value('previous_time', context)
        if not self.is_initializing:
            previous_value = adjusted_value
            previous_time = previous_time + time_step_size

            self.parameters.previous_time._set(previous_time, context)

        self.parameters.previous_value._set(previous_value, context)
        return convert_all_elements_to_np_array([previous_value, previous_time])

    def _gen_llvm_integrate(self, builder, index, ctx, vi, vo, params, state):
        # Get parameter pointers
        rate = self._gen_llvm_load_param(ctx, builder, params, index, RATE)
        noise = self._gen_llvm_load_param(ctx, builder, params, index, NOISE,
                                          state=state)
        offset = self._gen_llvm_load_param(ctx, builder, params, index, OFFSET)
        threshold = self._gen_llvm_load_param(ctx, builder, params, index, THRESHOLD)
        time_step_size = self._gen_llvm_load_param(ctx, builder, params, index, TIME_STEP_SIZE)

        random_state = ctx.get_random_state_ptr(builder, self, state, params)
        rand_val_ptr = builder.alloca(ctx.float_ty, name="random_out")
        rand_f = ctx.get_normal_dist_function_by_state(random_state)
        builder.call(rand_f, [random_state, rand_val_ptr])
        rand_val = builder.load(rand_val_ptr)

        # Get state pointers
        prev_ptr = pnlvm.helpers.get_state_ptr(builder, self, state, "previous_value")
        prev_time_ptr = pnlvm.helpers.get_state_ptr(builder, self, state, "previous_time")

        # value = previous_value + rate * variable * time_step_size \
        #       + np.sqrt(time_step_size * noise) * random_state.normal()
        prev_val_ptr = builder.gep(prev_ptr, [ctx.int32_ty(0), index])
        prev_val = builder.load(prev_val_ptr)

        val = builder.load(builder.gep(vi, [ctx.int32_ty(0), index]))
        val = builder.fmul(val, rate)
        val = builder.fmul(val, time_step_size)
        val = builder.fadd(val, prev_val)

        sqrt_f = ctx.get_builtin("sqrt", [ctx.float_ty])
        factor = builder.call(sqrt_f, [time_step_size])
        factor = builder.fmul(noise, factor)
        factor = builder.fmul(rand_val, factor)

        val = builder.fadd(val, factor)

        val = builder.fadd(val, offset)
        neg_threshold = pnlvm.helpers.fneg(builder, threshold)
        val = pnlvm.helpers.fclamp(builder, val, neg_threshold, threshold)

        # Store value result
        data_vo_ptr = builder.gep(vo, [ctx.int32_ty(0),
                                       ctx.int32_ty(0), index])
        builder.store(val, data_vo_ptr)
        builder.store(val, prev_val_ptr)

        # Update timestep
        prev_time_ptr = builder.gep(prev_time_ptr, [ctx.int32_ty(0), index])
        prev_time = builder.load(prev_time_ptr)
        curr_time = builder.fadd(prev_time, time_step_size)
        builder.store(curr_time, prev_time_ptr)

        time_vo_ptr = builder.gep(vo, [ctx.int32_ty(0), ctx.int32_ty(1), index])
        builder.store(curr_time, time_vo_ptr)

    def reset(self, previous_value=None, previous_time=None, context=None):
        return super().reset(
            previous_value=previous_value,
            previous_time=previous_time,
            context=context
        )


class DriftOnASphereIntegrator(IntegratorFunction):  # -----------------------------------------------------------------
    """
    DriftOnASphereIntegrator(                \
        default_variable=None,               \
        rate=1.0,                            \
        noise=0.0,                           \
        offset= 0.0,                         \
        starting_point=0.0,                  \
        threshold=1.0                        \
        time_step_size=1.0,                  \
        initializer=None,                    \
        dimension=2,                         \
        params=None,                         \
        owner=None,                          \
        prefs=None,                          \
        )
        COMMENT: REMOVED FROM ABOVE
        threshold=1.0
        COMMENT

    .. _DriftOnASphereIntegrator:

    Drift and diffuse on a sphere.  `function <DriftOnASphereIntegrator._function>` integrates previous coordinates
    with drift and/or noise that is applied either equally to all coordinates or dimension by dimension:

    ..  math::
        previous\\_value + rate \\cdot variable \\cdot time\\_step\\_size + \\mathcal{N}(\\sigma^2)

    where

    ..  math::
        \\sigma^2 =\\sqrt{time\\_step\\_size \\cdot noise}

    *Modulatory Parameters:*

    | *MULTIPLICATIVE_PARAM:* `rate <AdaptiveIntegrator.rate>`
    | *ADDITIVE_PARAM:* `offset <AdaptiveIntegrator.offset>`
    |

    Arguments
    ---------

    default_variable : list or 1d array : default class_defaults.variable
        specifies template for drift:  if specified, its length must be 1 or the value specified for *dimension*
        minus 1 (see `variable <DriftOnASphereIntegrator._function.variable>` for additional details).

    rate : float, list or 1d array : default 1.0
        applied multiplicatively to `variable <DriftOnASphereIntegrator.variable>`;  If it is a list or array, it must
        be the same length as `variable <DriftOnASphereIntegrator.variable>` (see `rate <DriftOnASphereIntegrator.rate>`
        for details).

    noise : float : default 0.0
        specifies a value by which to scale the normally distributed random value added to the integral in each call to
        `function <DriftOnASphereIntegrator._function>` (see `noise <DriftOnASphereIntegrator.noise>` for details).

    COMMENT:
    FIX: REPLACE ABOVE WITH THIS ONCE LIST/ARRAY SPECIFICATION OF NOISE IS FULLY IMPLEMENTED
    noise : float, list or 1d array : default 0.0
        specifies a value by which to scale the normally distributed random value added to the integral in each call to
        `function <DriftOnASphereIntegrator._function>`; if it is a list or array, it must be the same length as
        `variable <DriftOnASphereIntegrator.variable>` (see `noise <DriftOnASphereIntegrator.noise>` for details).
    COMMENT

    offset : float, list or 1d array : default 0.0
        specifies constant value added to integral in each call to `function <DriftOnASphereIntegrator._function>`;
        if it is a list or array, it must be the same length as `variable <DriftOnASphereIntegrator.variable>`
        (see `offset <DriftOnASphereIntegrator.offset>` for details).
        COMMENT:
        specifies constant value added to integral in each call to `function <DriftOnASphereIntegrator._function>`
        if it's absolute value is below `threshold <DriftOnASphereIntegrator.threshold>`;
        if it is a list or array, it must be the same length as `variable <DriftOnASphereIntegrator.variable>`
        (see `offset <DriftOnASphereIntegrator.offset>` for details).
        COMMENT

    starting_point : float, list or 1d array:  default 0.0
        specifies the starting value for the integration process; if it is a list or array, it must be the
        same length as `variable <DriftOnASphereIntegrator.variable>` (see `starting_point
        <DriftOnASphereIntegrator.starting_point>` for details).

    COMMENT:
    threshold : float : default 0.0
        specifies the threshold (boundaries) of the drift diffusion process -- i.e., at which the
        integration process terminates (see `threshold <DriftOnASphereIntegrator.threshold>` for details).
    COMMENT

    time_step_size : float : default 0.0
        specifies the timing precision of the integration process (see `time_step_size
        <DriftOnASphereIntegrator.time_step_size>` for details.

    dimension : int : default 2
        specifies dimensionality of the sphere over which drift occurs.

    initializer : 1d array : default [0]
        specifies the starting point on the sphere from which angle is derived;  its length must be equal to
        `dimension <DriftOnASphereIntegrator.dimension>` (see `initializer <DriftOnASphereIntegrator.initializer>`
        for additional details).

    angle_function : TransferFunction : default Angle
        specifies function used to compute angle from position on sphere specified by coordinates in
        `previous_value <DriftOnASphereIntegrator.previous_value>`.

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

    variable : float or array
        current input value that determines rate of drift on the sphere.  If it is a scalar, it is applied equally
        to all coordinates (subject to `noise <DriftOnASphereIntegrator.noise>`; if it is an array, each element
        determines rate of drift over a given dimension;  coordinate values are integrated over each execution, with
        the previous set stored in `previous_value <DriftOnASphereIntegrator.previous_value>`.

    rate : float or 1d array
        applied multiplicatively to `variable <DriftOnASphereIntegrator.variable>` (can be thought of as implementing
        the attentional component of the drift rate).  If it is a float or has a single element, its value is applied
        to all the elements of `variable <DriftOnASphereIntegrator.variable>`; if it is an array, each element is
        applied to the corresponding element of `variable <DriftOnASphereIntegrator.variable>`. Serves as
        *MULTIPLICATIVE_PARAM* for `modulation <ModulatorySignal_Modulation>` of `function
        <DriftOnASphereIntegrator._function>`.

    random_state : numpy.RandomState
        private pseudorandom number generator

    noise : float or 1d array
        scales the normally distributed random value added to integral in each call to `function
        <DriftOnASphereIntegrator._function>`. If `variable <DriftOnASphereIntegrator.variable>` is a list or array,
        and noise is a float, a single random term is generated and applied for each element of `variable
        <DriftOnASphereIntegrator.variable>`.  If noise is a list or array, it must be the same length as `variable
        <DriftOnASphereIntegrator.variable>`, and a separate random term scaled by noise is applied for each of the
        corresponding elements of `variable <DriftOnASphereIntegrator.variable>`.

    offset : float or 1d array
        constant value added to integral in each call to `function <DriftOnASphereIntegrator._function>`.
        If `variable <DriftOnASphereIntegrator.variable>` is an array and offset is a float, offset is applied
        to each element of the integral;  if offset is a list or array, each of its elements is applied to each of
        the corresponding elements of the integral (i.e., Hadamard addition). Serves as *ADDITIVE_PARAM* for
        `modulation <ModulatorySignal_Modulation>` of `function <DriftOnASphereIntegrator._function>`.
        COMMENT:
        constant value added to integral in each call to `function <DriftOnASphereIntegrator._function>`
        if it's absolute value is below `threshold <DriftOnASphereIntegrator.threshold>`.
        If `variable <DriftOnASphereIntegrator.variable>` is an array and offset is a float, offset is applied
        to each element of the integral;  if offset is a list or array, each of its elements is applied to each of
        the corresponding elements of the integral (i.e., Hadamard addition). Serves as *ADDITIVE_PARAM* for
        `modulation <ModulatorySignal_Modulation>` of `function <DriftOnASphereIntegrator._function>`.
        COMMENT

    COMMENT:
    FIX: starting_point MAY NEED TO BE REDEFINED (HERE AND FOR DriftDiffusionIntegratorFunction?)
    COMMENT

    starting_point : float or 1d array
        determines the starting value for the integration process; if it is a list or array, it must be the
        same length as `variable <DriftOnASphereIntegrator.variable>`. If `variable <DriftOnASphereIntegrator.variable>`
        is an array and starting_point is a float, starting_point is used for each element of the integral;  if
        starting_point is a list or array, each of its elements is used as the starting point for each element of the
        integral.

    COMMENT:
    threshold : float
        determines the boundaries of the drift diffusion process:  the integration process can be scheduled to
        terminate when the result of `function <DriftOnASphereIntegrator._function>` equals or exceeds either the
        positive or negative value of threshold (see hint).
        NOTE: Vector version of this parameter acts as a saturation barrier.
        While it is possible to subtract from value == threshold, any movement
        in the threshold direction will be capped at the threshold value.

        .. hint::
           To terminate execution of the `Mechanism <Mechanism>` to which the `function
           <DriftOnASphereIntegrator._function>` is assigned, a `WhenFinished` `Condition` should be assigned for that
           Mechanism to `scheduler <Composition.scheduler>` of the `Composition` to which the Mechanism belongs.
    COMMENT

    time_step_size : float
        determines the timing precision of the integration process and is used to scale the `noise
        <DriftOnASphereIntegrator.noise>` parameter.

    dimension : int
        determines dimensionality of sphere on which drift occurs.

    initializer : 1d array
        determines the starting point on the sphere from which angle is derived;  its length must be equal to
        `dimension <DriftOnASphereIntegrator.dimension>`.

    angle_function : TransferFunction
        determines the function used to compute angle (reproted as result) from coordinates on sphere specified by
        coordinates in `previous_value <DriftOnASphereIntegrator.previous_value>` displace by `variable
        <DriftOnASphereIntegrator.variable>` and possibly `noise <DriftOnASphereIntegrator.noise>`.

    previous_time : float
        stores previous time at which the function was executed and accumulates with each execution according to
        `time_step_size <DriftOnASphereIntegrator.default_time_step_size>`.

    previous_value : 1d array : default class_defaults.variable
        stores previous set of coordinates on sphere over which drift is occuring.

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

    componentName = DRIFT_ON_A_SPHERE_INTEGRATOR_FUNCTION

    class Parameters(IntegratorFunction.Parameters):
        """
            Attributes
            ----------

                angle_function
                    see `angle_function <DriftOnASphereIntegrator.angle_function>`

                    :default value: Angle
                    :type: ``Function``

                dimension
                    see `dimension <DriftOnASphereIntegrator.dimension>`

                    :default value: 2
                    :type: ``int``

                enable_output_type_conversion
                    see `enable_output_type_conversion <DriftOnASphereIntegrator.enable_output_type_conversion>`

                    :default value: False
                    :type: ``bool``
                    :read only: True

                initializer
                    see `initializer <DriftOnASphereIntegrator.initializer>`

                    :default value: np.zeros(dimension)
                    :type: ``numpy.ndarray``

                offset
                    see `offset <DriftOnASphereIntegrator.offset>`

                    :default value: 0.0
                    :type: ``float``

                previous_time
                    see `previous_time <DriftOnASphereIntegrator.previous_time>`

                    :default value: None
                    :type:

                random_state
                    see `random_state <DriftOnASphereIntegrator.random_state>`

                    :default value: None
                    :type: ``numpy.random.RandomState``

                rate
                    see `rate <DriftOnASphereIntegrator.rate>`

                    :default value: 1.0
                    :type: ``float``

                seed
                    see `seed <DriftOnASphereIntegrator.seed>`

                    :default value: None
                    :type:
                    :read only: True

                starting_point
                    see `starting_point <DriftOnASphereIntegrator.starting_point>`

                    :default value: 0.0
                    :type: ``float``

                COMMENT:
                threshold
                    see `threshold <DriftOnASphereIntegrator.threshold>`

                    :default value: 100.0
                    :type: ``float``
                COMMENT

                time_step_size
                    see `time_step_size <DriftOnASphereIntegrator.time_step_size>`

                    :default value: 1.0
                    :type: ``float``
        """
        rate = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        offset = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        starting_point = 0.0
        # threshold = Parameter(100.0, modulable=True)
        time_step_size = Parameter(1.0, modulable=True)
        previous_time = Parameter(None, initializer='starting_point', pnl_internal=True)
        dimension = Parameter(3, stateful=False, read_only=True)
        initializer = Parameter([0], initalizer='variable', stateful=True)
        angle_function = Parameter(None, stateful=False, loggable=False)
        random_state = Parameter(None, loggable=False, getter=_random_state_getter, dependencies='seed')
        seed = Parameter(DEFAULT_SEED, modulable=True, fallback_default=True, setter=_seed_setter)
        enable_output_type_conversion = Parameter(
            False,
            stateful=False,
            loggable=False,
            pnl_internal=True,
            read_only=True
        )

        def _validate_dimension(self, dimension):
            if not isinstance(dimension, int) or dimension < 2:
                return 'dimension must be an integer >= 2'

        # FIX: THIS SEEMS DUPLICATIVE OF DriftOnASphereIntegrator._validate_params() (THOUGH THAT GETS CAUGHT EARLIER)
        def _validate_initializer(self, initializer):
            initializer_len = self.dimension.default_value - 1
            if (self.initializer._user_specified
                    and (initializer.ndim != 1 or len(initializer) != initializer_len)):
                return f"'initializer' must be a list or 1d array of length {initializer_len} " \
                       f"(the value of the \'dimension\' parameter minus 1)"

        def _parse_initializer(self, initializer):
            """Assign initial value as array of random values of length dimension-1"""
            initializer_dim = self.dimension.default_value - 1
            if initializer.ndim != 1 or len(initializer) != initializer_dim:
                initializer = np.random.random(initializer_dim)
                self.initializer._set_default_value(initializer)
            return initializer

        def _parse_noise(self, noise):
            """Assign initial value as array of random values of length dimension-1"""
            if isinstance(noise, list):
                noise = np.array(noise)
            return noise

    @check_user_specified
    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate: tc.optional(parameter_spec) = None,
                 noise=None,
                 offset: tc.optional(parameter_spec) = None,
                 starting_point=None,
                 # threshold=None,
                 time_step_size=None,
                 dimension=None,
                 initializer=None,
                 angle_function=None,
                 seed=None,
                 params: tc.optional(tc.optional(dict)) = None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None,
                 **kwargs):

        # Assign here as default, for use in initialization of function
        super().__init__(
            default_variable=default_variable,
            rate=rate,
            time_step_size=time_step_size,
            starting_point=starting_point,
            initializer=initializer,
            angle_function=angle_function,
            # threshold=threshold,
            noise=noise,
            offset=offset,
            dimension=dimension,
            seed=seed,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _validate_params(self, request_set, target_set=None, context=None):

        # FIX: THIS SEEMS DUPLICATIVE OF Parameters._validate_initializer (THOUGHT THIS GETS CAUGHT EARLIER)
        if INITIALIZER in request_set and request_set[INITIALIZER] is not None:
            initializer = np.array(request_set[INITIALIZER])
            initializer_len = self.parameters.dimension.default_value - 1
            if (self.parameters.initializer._user_specified
                    and (initializer.ndim != 1 or len(initializer) != initializer_len)):
                raise FunctionError(f"'initializer' must be a list or 1d array of length {initializer_len} "
                                    f"(the value of the \'dimension\' parameter minus 1)")

        super()._validate_params(request_set=request_set, target_set=target_set,context=context)

    def _validate_noise(self, noise):
        if isinstance(noise, list):
            noise = np.array(noise)
        if noise is not None:
            if (not isinstance(noise, float)
                    and not (isinstance(noise, np.ndarray) and np.issubdtype(noise.dtype, np.floating))):
                raise FunctionError(
                    f"Invalid noise parameter for {self.name}: {type(noise)}. "
                    f"DriftOnASphereIntegrator requires noise parameter to be a float or float array.")
            if isinstance(noise, np.ndarray):
                initializer_len = self.parameters.dimension.default_value - 1
                if noise.ndim !=1 or len(noise) != initializer_len:
                    owner_str = f"'of '{self.owner.name}" if self.owner else ""
                    raise FunctionError(f"'noise' parameter for {self.name}{owner_str} must be a list or 1d array of "
                                        f"length {initializer_len} (the value of the \'dimension\' parameter minus 1)")

    def _validate_initializers(self, default_variable, context=None):
        """Need to override this to manage mismatch in dimensionality of initializer vs. variable"""
        pass

    def _parse_angle_function_variable(self, variable):
        return np.ones(self.parameters.dimension.default_value - 1)

    def _instantiate_attributes_before_function(self, function=None, context=None):
        """Need to override this to manage mismatch in dimensionality of initializer vs. variable"""

        if not self.parameters.initializer._user_specified:
            self._initialize_previous_value(self.parameters.initializer.get(context), context)

        # Remove initializer from self.initializers to manage mismatch in dimensionality of initializer vs. variable
        initializers = list(self.initializers)
        initializers.remove('initializer')
        self._instantiate_stateful_attributes(self.stateful_attributes, initializers, context)

        from psyneulink.core.components.functions.nonstateful.transferfunctions import Angle
        angle_function = self.parameters.angle_function.default_value or Angle
        dimension = self.parameters.dimension.default_value

        if isinstance(angle_function, type):
            self.parameters.angle_function._set_default_value(angle_function(np.ones(dimension - 1)))
        else:
            angle_function.defaults.variable = np.ones(dimension - 1)
            angle_function._instantiate_value(context)

    # FIX: IS THIS STILL NECESSARY?
    # FIX: FROM MemoryFunctions -- USE AS TEMPLATE TO ABSORB MUCH OF THE ABOVE?
    #                              (e.g., CAN BE USED TO OVERRIDE VALIDATION OF INITIALIZER??)
    def _validate(self, context=None):
        """Validate angle_function"""

        angle_function = self.parameters.angle_function.default_value
        dimension = self.parameters.dimension.default_value

        if self.get_previous_value(context) is not None:
            test_var = self.get_previous_value(context)
        else:
            test_var = self.defaults.variable

        if isinstance(angle_function, type):
            fct_msg = 'Function type'
        else:
            fct_msg = 'Function'

        try:
            angle_result = angle_function(test_var)
            if angle_result.ndim != 1 and len(angle_result) != dimension:
                raise FunctionError(f"{fct_msg} specified for 'angle_function' arg of "
                                    f"{self.__class__.__name__} ({angle_function}) must accept a list or 1d array "
                                    f"of length {dimension-1} and return a 1d array of length {dimension}.")
        except:
            raise FunctionError(f"Problem with {fct_msg} specified for 'angle_function' arg of "
                                f"{self.__class__.__name__} ({angle_function}).")

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        variable : number, list or 1d array : default class_defaults.variable
           value used as drift;  if it is a number, then used for all coordinates;  if it is  list or array its
           length must equal `dimension <DriftOnASphereIntegrator.dimension>` - 1, and is applied Hadamard to
           each coordinate.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        updated coordinates : 1d array
        """

        rate = np.array(self._get_current_parameter_value(RATE, context)).astype(float)
        noise = self._get_current_parameter_value(NOISE, context)
        offset = self._get_current_parameter_value(OFFSET, context)
        # threshold = self._get_current_parameter_value(THRESHOLD, context)
        time_step_size = self._get_current_parameter_value(TIME_STEP_SIZE, context)
        random_state = self._get_current_parameter_value("random_state", context)

        angle_function = self.parameters.angle_function.default_value
        dimension = self.parameters.dimension.get()

        previous_value = self.parameters.previous_value._get(context)

        try:
            drift = np.full(dimension - 1, variable)
        except ValueError:
            owner_str = f"'of '{self.owner.name}" if self.owner else ""
            raise FunctionError(f"Length of 'variable' for {self.name}{owner_str} ({len(variable)}) must be "
                                # f"1 or one less than its 'dimension' parameter ({dimension}-1={dimension-1}).")
                                f"1 or {dimension-1} (one less than its 'dimension' parameter: {dimension}).")

        random_draw = np.array([random_state.normal() for i in range(dimension - 1)])
        value = previous_value + rate * drift * time_step_size \
                + np.sqrt(time_step_size * noise) * random_draw

        # adjusted_value = np.clip(value + offset, -threshold, threshold)
        adjusted_value = value + offset

        # If this NOT an initialization run, update the old value and time
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        previous_time = self._get_current_parameter_value('previous_time', context)
        if not self.is_initializing:
            value = adjusted_value
            previous_time = previous_time + time_step_size
            self.parameters.previous_time._set(previous_time, context)

        self.parameters.previous_value._set(value, context)

        return angle_function(value)

    # def _gen_llvm_integrate(self, builder, index, ctx, vi, vo, params, state):
    #     # Get parameter pointers
    #     rate = self._gen_llvm_load_param(ctx, builder, params, index, RATE)
    #     noise = self._gen_llvm_load_param(ctx, builder, params, index, NOISE,
    #                                       state=state)
    #     offset = self._gen_llvm_load_param(ctx, builder, params, index, OFFSET)
    #     threshold = self._gen_llvm_load_param(ctx, builder, params, index, THRESHOLD)
    #     time_step_size = self._gen_llvm_load_param(ctx, builder, params, index, TIME_STEP_SIZE)
    #
    #     random_state = pnlvm.helpers.get_state_ptr(builder, self, state, "random_state")
    #     rand_val_ptr = builder.alloca(ctx.float_ty)
    #     rand_f = ctx.import_llvm_function("__pnl_builtin_mt_rand_normal")
    #     builder.call(rand_f, [random_state, rand_val_ptr])
    #     rand_val = builder.load(rand_val_ptr)
    #
    #     if isinstance(rate.type, pnlvm.ir.ArrayType):
    #         assert len(rate.type) == 1
    #         rate = builder.extract_value(rate, 0)
    #
    #     # Get state pointers
    #     prev_ptr = pnlvm.helpers.get_state_ptr(builder, self, state, "previous_value")
    #     prev_time_ptr = pnlvm.helpers.get_state_ptr(builder, self, state, "previous_time")
    #
    #     # value = previous_value + rate * variable * time_step_size \
    #     #       + np.sqrt(time_step_size * noise) * random_state.normal()
    #     prev_val_ptr = builder.gep(prev_ptr, [ctx.int32_ty(0), index])
    #     prev_val = builder.load(prev_val_ptr)
    #     val = builder.load(builder.gep(vi, [ctx.int32_ty(0), index]))
    #     if isinstance(val.type, pnlvm.ir.ArrayType):
    #         assert len(val.type) == 1
    #         val = builder.extract_value(val, 0)
    #     val = builder.fmul(val, rate)
    #     val = builder.fmul(val, time_step_size)
    #     val = builder.fadd(val, prev_val)
    #
    #     factor = builder.fmul(noise, time_step_size)
    #     sqrt_f = ctx.get_builtin("sqrt", [ctx.float_ty])
    #     factor = builder.call(sqrt_f, [factor])
    #
    #     factor = builder.fmul(rand_val, factor)
    #
    #     val = builder.fadd(val, factor)
    #
    #     val = builder.fadd(val, offset)
    #     neg_threshold = pnlvm.helpers.fneg(builder, threshold)
    #     val = pnlvm.helpers.fclamp(builder, val, neg_threshold, threshold)
    #
    #     # Store value result
    #     data_vo_ptr = builder.gep(vo, [ctx.int32_ty(0),
    #                                    ctx.int32_ty(0), index])
    #     builder.store(val, data_vo_ptr)
    #     builder.store(val, prev_val_ptr)
    #
    #     # Update timestep
    #     prev_time_ptr = builder.gep(prev_time_ptr, [ctx.int32_ty(0), index])
    #     prev_time = builder.load(prev_time_ptr)
    #     curr_time = builder.fadd(prev_time, time_step_size)
    #     builder.store(curr_time, prev_time_ptr)
    #
    #     time_vo_ptr = builder.gep(vo, [ctx.int32_ty(0), ctx.int32_ty(1), index])
    #     builder.store(curr_time, time_vo_ptr)

    def reset(self, previous_value=None, previous_time=None, context=None):
        return super().reset(
            previous_value=previous_value,
            previous_time=previous_time,
            context=context
        )


class OrnsteinUhlenbeckIntegrator(IntegratorFunction):  # --------------------------------------------------------------
    """
    OrnsteinUhlenbeckIntegrator(         \
        default_variable=None,           \
        rate=1.0,                        \
        decay=1.0,                       \
        noise=0.0,                       \
        offset= 0.0,                     \
        non_decision_time=0.0,           \
        time_step_size=1.0,              \
        initializer=0.0,                 \
        params=None,                     \
        owner=None,                      \
        prefs=None,                      \
        )

    .. _OrnsteinUhlenbeckIntegrator:

    `function <_OrnsteinUhlenbeckIntegrator._function>` returns one time step of integration according to an
    `Ornstein Uhlenbeck process <https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process>`_:

    .. math::
       previous\\_value + (decay \\cdot  previous\\_value) - (rate \\cdot variable) + \\mathcal{N}(\\sigma^2)
    where
    ..  math::
        \\sigma^2 =\\sqrt{time\\_step\\_size \\cdot noise}

    *Modulatory Parameters:*

    | *MULTIPLICATIVE_PARAM:* `rate <AdaptiveIntegrator.rate>`
    | *ADDITIVE_PARAM:* `offset <AdaptiveIntegrator.offset>`
    |

    Arguments
    ---------

    default_variable : number, list or array : default class_defaults.variable
        specifies a template for  the stimulus component of drift rate -- the drift rate is the product of variable and
        rate

    rate : float, list or 1d array : default 1.0
        specifies value applied multiplicatively to `variable <OrnsteinUhlenbeckIntegrator.variable>`;  If it is a
        list or array, it must be the same length as `variable <OrnsteinUhlenbeckIntegrator.variable>` (see `rate
        <OrnsteinUhlenbeckIntegrator.rate>` for details).

    decay : float, list or 1d array : default 1.0
        specifies value applied multiplicatively to `previous_value <OrnsteinUhlenbeckIntegrator.previous_value>`;
        If it is a list or array, it must be the same length as `variable <OrnsteinUhlenbeckIntegrator.variable>` (
        see `decay <OrnsteinUhlenbeckIntegrator.rate>` for details).

    noise : float : default 0.0
        specifies a value by which to scale the normally distributed random value added to the integral in each
        call to `function <OrnsteinUhlenbeckIntegrator._function>` (see `noise <OrnsteinUhlenbeckIntegrator.noise>`
        for details).

    COMMENT:
    FIX: REPLACE ABOVE WITH THIS ONCE LIST/ARRAY SPECIFICATION OF NOISE IS FULLY IMPLEMENTED
    noise : float, list or 1d array : default 0.0
        specifies a value by which to scale the normally distributed random value added to the integral in each call to
        `function <OrnsteinUhlenbeckIntegrator._function>`; if it is a list or array, it must be the same length as
        `variable <OrnsteinUhlenbeckIntegrator.variable>` (see `noise <OrnsteinUhlenbeckIntegrator.noise>` for details).
    COMMENT

    offset : float, list or 1d array : default 0.0
        specifies a constant value added to integral in each call to `function <OrnsteinUhlenbeckIntegrator._function>`;
        if it is a list or array, it must be the same length as `variable <OrnsteinUhlenbeckIntegrator.variable>`
        (see `offset <OrnsteinUhlenbeckIntegrator.offset>` for details)

    non_decision_time : float : default 0.0
        specifies the starting time of the model and is used to compute `previous_time
        <OrnsteinUhlenbeckIntegrator.previous_time>`

    time_step_size : float : default 0.0
        determines the timing precision of the integration process (see `time_step_size
        <OrnsteinUhlenbeckIntegrator.time_step_size>` for details.

    initializer : float, list or 1d array : default 0.0
        specifies starting value(s) for integration.  If it is a list or array, it must be the same length as `variable
        <OrnsteinUhlenbeckIntegrator.variable>` (see `initializer <IntegratorFunction.initializer>` for details).

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
        represents the stimulus component of drift. The product of
        `variable <OrnsteinUhlenbeckIntegrator.variable>` and `rate <OrnsteinUhlenbeckIntegrator.rate>` is multiplied
        by `time_step_size <OrnsteinUhlenbeckIntegrator.time_step_size>` to model the accumulation of evidence during
        one step.

    rate : float or 1d array
        applied multiplicatively to `variable <OrnsteinUhlenbeckIntegrator.variable>`.  If it is a float or has a
        single element, its value is applied to all the elements of `variable <OrnsteinUhlenbeckIntegrator.variable>`;
        if it is an array, each element is applied to the corresponding element of `variable
        <OrnsteinUhlenbeckIntegrator.variable>`.  Serves as *MULTIPLICATIVE_PARAM* for `modulation
        <ModulatorySignal_Modulation>` of `function <OrnsteinUhlenbeckIntegrator._function>`.

    decay : float or 1d array
        applied multiplicatively to `previous_value <OrnsteinUhlenbeckIntegrator.previous_value>`; If it is a float or
        has a single element, its value is applied to all the elements of `previous_value
        <OrnsteinUhlenbeckIntegrator.previous_value>`; if it is an array, each element is applied to the corresponding
        element of `previous_value <OrnsteinUhlenbeckIntegrator.previous_value>`.

    noise : float
        scales the normally distributed random value added to integral in each call to `function
        <OrnsteinUhlenbeckIntegrator._function>`.  A single random term is generated each execution, and applied to all
        elements of `variable <OrnsteinUhlenbeckIntegrator.variable>` if that is an array with more than one element.

    COMMENT:
    FIX: REPLACE ABOVE WITH THIS ONCE LIST/ARRAY SPECIFICATION OF NOISE IS FULLY IMPLEMENTED
    noise : float or 1d array
        scales the normally distributed random value added to integral in each call to `function
        <OrnsteinUhlenbeckIntegrator._function>`. If `variable <OrnsteinUhlenbeckIntegrator.variable>` is a list or
        array, and noise is a float, a single random term is generated and applied for each element of `variable
        <OrnsteinUhlenbeckIntegrator.variable>`.  If noise is a list or array, it must be the same length as `variable
        <OrnsteinUhlenbeckIntegrator.variable>`, and a separate random term scaled by noise is applied for each of the
        corresponding elements of `variable <OrnsteinUhlenbeckIntegrator.variable>`.
    COMMENT

    offset : float or 1d array
        constant value added to integral in each call to `function <OrnsteinUhlenbeckIntegrator._function>`.
        If `variable <OrnsteinUhlenbeckIntegrator.variable>` is an array and offset is a float, offset is applied
        to each element of the integral;  if offset is a list or array, each of its elements is applied to each of
        the corresponding elements of the integral (i.e., Hadamard addition). Serves as *ADDITIVE_PARAM* for
        `modulation <ModulatorySignal_Modulation>` of `function <OrnsteinUhlenbeckIntegrator._function>`.

    non_decision_time : float
        determines the start time of the integration process.

    time_step_size : float
        determines the timing precision of the integration process and is used to scale the `noise
        <OrnsteinUhlenbeckIntegrator.noise>` parameter appropriately.

    initializer : float or 1d array
        determines the starting value(s) for integration (i.e., the value(s) to which `previous_value
        <OrnsteinUhlenbeckIntegrator.previous_value>` is set (see `initializer <IntegratorFunction.initializer>`
        for details).

    previous_value : 1d array : default class_defaults.variable
        stores previous value with which `variable <OrnsteinUhlenbeckIntegrator.variable>` is integrated.

    previous_time : float
        stores previous time at which the function was executed and accumulates with each execution according to
        `time_step_size <OrnsteinUhlenbeckIntegrator.default_time_step_size>`.

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

    componentName = ORNSTEIN_UHLENBECK_INTEGRATOR_FUNCTION

    class Parameters(IntegratorFunction.Parameters):
        """
            Attributes
            ----------

                decay
                    see `decay <OrnsteinUhlenbeckIntegrator.decay>`

                    :default value: 1.0
                    :type: ``float``

                enable_output_type_conversion
                    see `enable_output_type_conversion <OrnsteinUhlenbeckIntegrator.enable_output_type_conversion>`

                    :default value: False
                    :type: ``bool``
                    :read only: True

                offset
                    see `offset <OrnsteinUhlenbeckIntegrator.offset>`

                    :default value: 0.0
                    :type: ``float``

                previous_time
                    see `previous_time <OrnsteinUhlenbeckIntegrator.previous_time>`

                    :default value: 0.0
                    :type: ``float``

                random_state
                    see `random_state <OrnsteinUhlenbeckIntegrator.random_state>`

                    :default value: None
                    :type: ``numpy.random.RandomState``

                rate
                    see `rate <OrnsteinUhlenbeckIntegrator.rate>`

                    :default value: 1.0
                    :type: ``float``

                non_decision_time
                    see `non_decision_time <OrnsteinUhlenbeckIntegrator.non_decision_time>`

                    :default value: 0.0
                    :type: ``float``

                time_step_size
                    see `time_step_size <OrnsteinUhlenbeckIntegrator.time_step_size>`

                    :default value: 1.0
                    :type: ``float``
        """
        # FIX 6/21/19 [JDC]: MAKE ALL OF THESE PARAMETERS AND ADD function_arg TO THEM TO "PARALLELIZE" INTEGRATION
        rate = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        decay = Parameter(1.0, modulable=True)
        offset = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        time_step_size = Parameter(1.0, modulable=True)
        initializer = Parameter(np.array([0]), aliases=['starting_value'])
        non_decision_time = Parameter(0.0, modulable=True)
        previous_time = Parameter(0.0, initializer='non_decision_time', pnl_internal=True)
        random_state = Parameter(None, loggable=False, getter=_random_state_getter, dependencies='seed')
        seed = Parameter(DEFAULT_SEED, modulable=True, fallback_default=True, setter=_seed_setter)
        enable_output_type_conversion = Parameter(
            False,
            stateful=False,
            loggable=False,
            pnl_internal=True,
            read_only=True
        )

    @check_user_specified
    @tc.typecheck
    def __init__(
        self,
        default_variable=None,
        rate: tc.optional(parameter_spec) = None,
        decay=None,
        noise=None,
        offset: tc.optional(parameter_spec) = None,
        non_decision_time=None,
        time_step_size=None,
        starting_value=None,
        params: tc.optional(tc.optional(dict)) = None,
        seed=None,
        owner=None,
        prefs: tc.optional(is_pref_set) = None,
        **kwargs
    ):

        super().__init__(
            default_variable=default_variable,
            rate=rate,
            decay=decay,
            noise=noise,
            offset=offset,
            non_decision_time=non_decision_time,
            time_step_size=time_step_size,
            starting_value=starting_value,
            previous_value=starting_value,
            previous_time=non_decision_time,
            params=params,
            seed=seed,
            owner=owner,
            prefs=prefs,
            **kwargs
        )

    def _validate_noise(self, noise):
        if noise is not None and not isinstance(noise, float):
            raise FunctionError(
                "Invalid noise parameter for {}. OrnsteinUhlenbeckIntegrator requires noise parameter to be a float. "
                "Noise parameter is used to construct the standard DDM noise distribution".format(self.name))

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        variable : number, list or array : default class_defaults.variable
           a single value or array of values to be integrated.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        updated value of integral : 2d array

        """
        rate = np.array(self._get_current_parameter_value(RATE, context)).astype(float)
        decay = self._get_current_parameter_value(DECAY, context)
        noise = self._get_current_parameter_value(NOISE, context)
        offset = self._get_current_parameter_value(OFFSET, context)
        time_step_size = self._get_current_parameter_value(TIME_STEP_SIZE, context)
        random_state = self._get_current_parameter_value('random_state', context)

        previous_value = np.atleast_2d(self.parameters.previous_value._get(context))

        random_normal = random_state.normal()

        # dx = (lambda*x + A)dt + c*dW
        value = previous_value + (decay * previous_value - rate * variable) * time_step_size + np.sqrt(
            time_step_size * noise) * random_normal

        # If this NOT an initialization run, update the old value and time
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        adjusted_value = value + offset

        previous_time = self._get_current_parameter_value('previous_time', context)
        if not self.is_initializing:
            previous_value = adjusted_value
            previous_time = previous_time + time_step_size
            if not np.isscalar(variable):
                previous_time = np.broadcast_to(
                    previous_time,
                    variable.shape
                ).copy()
            self.parameters.previous_time._set(previous_time, context)

        self.parameters.previous_value._set(previous_value, context)
        return previous_value, previous_time

    def reset(self, previous_value=None, previous_time=None, context=None):
        return super().reset(
            previous_value=previous_value,
            previous_time=previous_time,
            context=context
        )


class LeakyCompetingIntegrator(IntegratorFunction):  # -----------------------------------------------------------------
    """
    LeakyCompetingIntegrator(                  \
        default_variable=None,      \
        leak=1.0,                   \
        noise=0.0,                  \
        offset=None,                \
        time_step_size=0.1,         \
        initializer=0.0,            \
        params=None,                \
        owner=None,                 \
        prefs=None,                 \
        )

    .. _LeakyCompetingIntegrator:

    Implements Leaky Competitive Accumulator (LCA) described in `Usher & McClelland (2001)
    <https://www.ncbi.nlm.nih.gov/pubmed/11488378>`_.  `function <LeakyCompetingIntegrator._function>` returns:

    .. math::

        previous\\_value + (variable - leak \\cdot previous\\_value) \\cdot time\\_step\\_size +
        noise \\sqrt{time\\_step\\_size}

    where `variable <LeakyCompetingIntegrator.variable>` corresponds to
    :math:`\\rho_i` + :math:`\\beta`:math:`\\Sigma f(x_{\\neq i})` (the net input to a unit),
    `leak <LeakyCompetingIntegrator.leak>` corresponds to :math:`k`, and `time_step_size
    <LeakyCompetingIntegrator.time_step_size>` corresponds to :math:`\\frac{dt}{\\tau}`
    in Equation 4 of `Usher & McClelland (2001) <https://www.ncbi.nlm.nih.gov/pubmed/11488378>`_.

    .. note::
        When used as the `function <Mechanism._function>` of an `LCAMechanism`, the value passed to `variable
        <LeakyCompetingIntegrator.variable>` is the sum of the external and recurrent inputs to the Mechanism
        (see `here <RecurrentTransferMechanism_Structure>` for how the external and recurrent inputs can be
        configured in a `RecurrentTransferMechanism`, of which LCAMechanism is subclass).

    .. note::
        the value of the **leak** argument is assigned to the `rate <LeakyCompetingIntegrator.rate>` parameter (and
        the `leak <LeakyCompetingIntegrator.leak>` parameter as an alias of the `rate <LeakyCompetingIntegrator.rate>`
        parameter); this is to be consistent with the parent class, `IntegratorFunction`.  However, note that
        in contrast to a standard IntegratorFunction, where :math:`rate \\cdot previous\\_value` is added to
        `variable <LeakyCompetingIntegrator.variable>`, here it is subtracted from `variable
        <LeakyCompetingIntegrator.variable>` in order to implement decay. Thus, the value returned by the function can
        increase in a given time step only if `rate <LeakyCompetingIntegrator.rate>` (aka `leak
        <LeakyCompetingIntegrator.leak>`) is negative or `variable <LeakyCompetingIntegrator.variable>` is
        sufficiently positive.

    *Modulatory Parameters:*

    | *MULTIPLICATIVE_PARAM:* `rate <AdaptiveIntegrator.rate>`
    | *ADDITIVE_PARAM:* `offset <AdaptiveIntegrator.offset>`
    |

    Arguments
    ---------

    default_variable : number, list or array : default class_defaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    leak : float, list or 1d array : default 1.0
        specifies the value used to scale the rate of decay of the integral on each time step.
        If it is a list or array, it must be the same length as `variable <LeakyCompetingIntegrator.variable>` (see
        `leak <LeakyCompetingIntegrator.leak>` for details).

    noise : float, function, list or 1d array : default 0.0
        specifies random value added to integral in each call to `function <LeakyCompetingIntegrator._function>`;
        if it is a list or array, it must be the same length as `variable <LeakyCompetingIntegrator.variable>`
        (see `noise <Integrator.noise>` for additonal details).

    offset : float, list or 1d array : default 0.0
        specifies a constant value added to integral in each call to `function <LeakyCompetingIntegrator._function>`;
        if it is a list or array, it must be the same length as `variable <LeakyCompetingIntegrator.variable>`
        (see `offset <LeakyCompetingIntegrator.offset>` for details).

    time_step_size : float : default 0.0
        specifies the timing precision of the integration process (see `time_step_size
        <LeakyCompetingIntegrator.time_step_size>` for details.

    initializer : float, list or 1d array : default 0.0
        specifies starting value(s) for integration.  If it is a list or array, it must be the same length as `variable
        <LeakyCompetingIntegrator.variable>` (see `initializer <IntegratorFunction.initializer>` for details).

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
        current input value some portion of which (determined by `rate <LeakyCompetingIntegrator.rate>`) will be
        added to the prior value;  if it is an array, each element is independently integrated.

    rate : float, list or 1d array
        scales the contribution of `previous_value <LeakyCompetingIntegrator.previous_value>` to the decay of
        the `value <LeakyCompetingIntegrator.value>` on each time step (corresponding to the ``leak`` term of the
        function described in Equation 4 of `Usher & McClelland, 2001) <https://www.ncbi.nlm.nih.gov/pubmed/11488378>`_.
        If it is a float or has a single element, its value is applied to all the elements of `previous_value
        <LeakyCompetingIntegrator.previous_value>`; if it is an array, each element is applied to the corresponding
        element of `previous_value <LeakyCompetingIntegrator.previous_value>`.  Serves as *MULTIPLICATIVE_PARAM*  for
        `modulation <ModulatorySignal_Modulation>` of `function <LeakyCompetingIntegrator._function>`.

        .. note::
          aliased by the `leak <LeakyCompetingIntegrator.leak>` parameter.

    leak : float, list or 1d array
        alias of `rate <LeakyCompetingIntegrator.rate>` (to be consistent with the standard format of an
        `IntegratorFunction`).

    noise : float, Function, or 1d array
        random value added to integral in each call to `function <LeakyCompetingIntegrator._function>`.
        (see `noise <Integrator.noise>` for details).

    offset : float or 1d array
        constant value added to integral in each call to `function <LeakyCompetingIntegrator._function>`. If `variable
        <LeakyCompetingIntegrator.variable>` is an array and offset is a float, offset is applied to each element  of
        the integral;  if offset is a list or array, each of its elements is applied to each of the corresponding
        elements of the integral (i.e., Hadamard addition). Serves as *ADDITIVE_PARAM* for `modulation
        <ModulatorySignal_Modulation>` of `function <LeakyCompetingIntegrator._function>`.

    time_step_size : float
        determines the timing precision of the integration process and is used to scale the `noise
        <LeakyCompetingIntegrator.noise>` parameter appropriately.

    initializer : float or 1d array
        determines the starting value(s) for integration (i.e., the value(s) to which `previous_value
        <LeakyCompetingIntegrator.previous_value>` is set (see `initializer <IntegratorFunction.initializer>`
        for details).

    previous_value : 1d array : default class_defaults.variable
        stores previous value with which `variable <LeakyCompetingIntegrator.variable>` is integrated.

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

    componentName = LEAKY_COMPETING_INTEGRATOR_FUNCTION

    class Parameters(IntegratorFunction.Parameters):
        """
            Attributes
            ----------

                offset
                    see `offset <LeakyCompetingIntegrator.offset>`

                    :default value: 0.0
                    :type: ``float``

                rate
                    see `rate <LeakyCompetingIntegrator.rate>`

                    :default value: 1.0
                    :type: ``float``

                time_step_size
                    see `time_step_size <LeakyCompetingIntegrator.time_step_size>`

                    :default value: 0.1
                    :type: ``float``
        """
        rate = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM, 'leak'], function_arg=True)
        offset = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM], function_arg=True)
        time_step_size = Parameter(0.1, modulable=True, function_arg=True)

    @check_user_specified
    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 leak: tc.optional(parameter_spec) = None,
                 noise=None,
                 offset=None,
                 time_step_size=None,
                 initializer=None,
                 params: tc.optional(tc.optional(dict)) = None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None,
                 **kwargs):

        # IMPLEMENTATION NOTE:  For backward compatibility of LeakyFun in tests/functions/test_integrator.py
        if RATE in kwargs:
            leak = kwargs[RATE]

        super().__init__(
            default_variable=default_variable,
            rate=leak,
            noise=noise,
            offset=offset,
            time_step_size=time_step_size,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs
        )

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        variable : number, list or array : default class_defaults.variable
           a single value or array of values to be integrated.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        updated value of integral : 2d array

        """
        rate = np.atleast_1d(self._get_current_parameter_value(RATE, context))
        initializer = self._get_current_parameter_value(INITIALIZER, context)  # unnecessary?
        time_step_size = self._get_current_parameter_value(TIME_STEP_SIZE, context)
        offset = self._get_current_parameter_value(OFFSET, context)

        # execute noise if it is a function
        noise = self._try_execute_param(self._get_current_parameter_value(NOISE, context), variable, context=context)
        previous_value = self.parameters.previous_value._get(context)
        new_value = variable

        # Gilzenrat: previous_value + (-previous_value + variable)*self.time_step_size + noise --> rate = -1
        value = previous_value + (-rate * previous_value + new_value) * time_step_size + noise * (time_step_size ** 0.5)

        adjusted_value = value + offset

        # If this NOT an initialization run, update the old value
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if not self.is_initializing:
            self.parameters.previous_value._set(adjusted_value, context)

        return self.convert_output_type(adjusted_value)

    def _gen_llvm_integrate(self, builder, index, ctx, vi, vo, params, state):
        rate = self._gen_llvm_load_param(ctx, builder, params, index, RATE)
        noise = self._gen_llvm_load_param(ctx, builder, params, index, NOISE,
                                          state=state)
        offset = self._gen_llvm_load_param(ctx, builder, params, index, OFFSET)
        time_step = self._gen_llvm_load_param(ctx, builder, params, index, TIME_STEP_SIZE)

        # Get the only context member -- previous value
        prev_ptr = pnlvm.helpers.get_state_ptr(builder, self, state, "previous_value")
        # Get rid of 2d array. When part of a Mechanism the input,
        # (and output, and context) are 2d arrays.
        prev_ptr = pnlvm.helpers.unwrap_2d_array(builder, prev_ptr)
        assert len(prev_ptr.type.pointee) == len(vi.type.pointee)

        prev_ptr = builder.gep(prev_ptr, [ctx.int32_ty(0), index])
        prev_val = builder.load(prev_ptr)

        in_ptr = builder.gep(vi, [ctx.int32_ty(0), index])
        in_val = builder.load(in_ptr)

        ret = builder.fmul(prev_val, rate)
        # ret = builder.fadd(ret, in_val)
        ret = builder.fsub(in_val, ret)
        ret = builder.fmul(ret, time_step)

        sqrt_f = ctx.get_builtin("sqrt", [ctx.float_ty])
        mod_step = builder.call(sqrt_f, [time_step])
        mod_noise = builder.fmul(noise, mod_step)

        ret = builder.fadd(ret, prev_val)
        ret = builder.fadd(ret, mod_noise)
        ret = builder.fadd(ret, offset)

        out_ptr = builder.gep(vo, [ctx.int32_ty(0), index])
        builder.store(ret, out_ptr)
        builder.store(ret, prev_ptr)

    def as_expression(self):
        return f'previous_value + (-rate * previous_value + {MODEL_SPEC_ID_MDF_VARIABLE}) * time_step_size + noise * (time_step_size ** 0.5)'


class FitzHughNagumoIntegrator(IntegratorFunction):  # ----------------------------------------------------------------------------
    """
    FitzHughNagumoIntegrator(                      \
        default_variable=1.0,           \
        initial_w=0.0,                  \
        initial_v=0.0,                  \
        time_step_size=0.05,            \
        t_0=0.0,                        \
        a_v=-1/3,                       \
        b_v=0.0,                        \
        c_v=1.0,                        \
        d_v=0.0,                        \
        e_v=-1.0,                       \
        f_v=1.0,                        \
        threshold=-1.0                  \
        time_constant_v=1.0,            \
        a_w=1.0,                        \
        b_w=-0.8,                       \
        c_w=0.7,                        \
        mode=1.0,                       \
        uncorrelated_activity=0.0       \
        time_constant_w = 12.5,         \
        integration_method="RK4"        \
        params=None,                    \
        owner=None,                     \
        prefs=None,                     \
        )

    .. _FitzHughNagumoIntegrator:

    `function <FitzHughNagumoIntegrator._function>` returns one time step of integration of the `Fitzhugh-Nagumo model
    https://en.wikipedia.org/wiki/FitzHugh–Nagumo_model>`_ of an excitable oscillator:

    .. math::
            time\\_constant_v \\frac{dv}{dt} = a_v * v^3 + (1 + threshold) * b_v * v^2 + (- threshold) * c_v * v^2 +
            d_v + e_v * w + f_v * I_{ext}

    where
    .. math::
            time\\_constant_w * \\frac{dw}{dt} =` mode * a_w * v + b_w * w +c_w + (1 - mode) * uncorrelated\\_activity

    Either `Euler <https://en.wikipedia.org/wiki/Euler_method>`_ or `Dormand–Prince (4th Order Runge-Kutta)
    <https://en.wikipedia.org/wiki/Dormand–Prince_method>`_ methods of numerical integration can be used.

    The FitzHughNagumoIntegrator implements all of the parameters of the FitzHughNagumo model; however, not all combinations of
    these are sensible. Typically, they are combined into two sets.  These are described below, followed by
    a describption of how they are used to implement three common variants of the model with the FitzHughNagumoIntegrator.

    Parameter Sets
    ^^^^^^^^^^^^^^

    **Fast, Excitatory Variable:**

    .. math::
       \\frac{dv}{dt} = \\frac{a_v v^{3} + b_v v^{2} (1+threshold) - c_v v\\, threshold + d_v + e_v\\, previous_w +
       f_v\\, variable)}{time\\, constant_v}


    **Slow, Inactivating Variable:**

    .. math::
       \\frac{dw}{dt} = \\frac{a_w\\, mode\\, previous_v + b_w w + c_w + uncorrelated\\,activity\\,(1-mode)}{time\\,
       constant_w}

    Three Common Variants
    ^^^^^^^^^^^^^^^^^^^^^

    (1) **Fitzhugh-Nagumo Model**

        **Fast, Excitatory Variable:**

        .. math::
            \\frac{dv}{dt} = v - \\frac{v^3}{3} - w + I_{ext}

        **Slow, Inactivating Variable:**

        .. math::
            \\frac{dw}{dt} = \\frac{v + a - bw}{T}

        :math:`\\frac{dw}{dt}` often has the following parameter values:

        .. math::
            \\frac{dw}{dt} = 0.08\\,(v + 0.7 - 0.8 w)

        **Implementation in FitzHughNagumoIntegrator**

        The default values implement the above equations.


    (2) **Modified FitzHughNagumo Model**

        **Fast, Excitatory Variable:**

        .. math::
            \\frac{dv}{dt} = v(a-v)(v-1) - w + I_{ext}

        **Slow, Inactivating Variable:**

        .. math::
            \\frac{dw}{dt} = bv - cw

        `Mahbub Khan (2013) <http://pcwww.liv.ac.uk/~bnvasiev/Past%20students/Mahbub_549.pdf>`_ provides a nice summary
        of why this formulation is useful.

        **Implementation in FitzHughNagumoIntegrator**

            The following parameter values must be specified in the equation for :math:`\\frac{dv}{dt}`:

            +--------------------------------------+-----+-----+-----+-----+-----+-----+---------------+
            |**FitzHughNagumoIntegrator Parameter**| a_v | b_v | c_v | d_v | e_v | f_v |time_constant_v|
            +--------------------------------------+-----+-----+-----+-----+-----+-----+---------------+
            |**Value**                             |-1.0 |1.0  |1.0  |0.0  |-1.0 |1.0  |1.0            |
            +--------------------------------------+-----+-----+-----+-----+-----+-----+---------------+

            When the parameters above are set to the listed values, the FitzHughNagumoIntegrator equation for :math:`\\frac{dv}{dt}`
            reduces to the Modified FitzHughNagumo formulation, and the remaining parameters in the :math:`\\frac{dv}{dt}` equation
            correspond as follows:

            +----------------------------------------+--------------------------------------------------+-----------------------------------------------+
            |**FitzHughNagumoIntegrator Parameter**  |`threshold <FitzHughNagumoIntegrator.threshold>`  |`variable <FitzHughNagumoIntegrator.variable>` |
            +----------------------------------------+--------------------------------------------------+-----------------------------------------------+
            |**Modified FitzHughNagumo Parameter**   |a                                                 |:math:`I_{ext}`                                |
            +----------------------------------------+--------------------------------------------------+-----------------------------------------------+

            The following parameter values must be set in the equation for :math:`\\frac{dw}{dt}`:

            +---------------------------------------+-----+------+-----------------+-----------------------+
            |**FitzHughNagumoIntegrator Parameter** |c_w  | mode | time_constant_w | uncorrelated_activity |
            +---------------------------------------+-----+------+-----------------+-----------------------+
            |**Value**                              | 0.0 | 1.0  | 1.0             |  0.0                  |
            +---------------------------------------+-----+------+-----------------+-----------------------+

            When the parameters above are set to the listed values, the FitzHughNagumoIntegrator equation for :math:`\\frac{dw}{dt}`
            reduces to the Modified FitzHughNagumo formulation, and the remaining parameters in the :math:`\\frac{dw}{dt}` equation
            correspond as follows:

            +-----------------------------------------+---------------------------------------+------------------------------------------------+
            |**FitzHughNagumoIntegrator Parameter**   |`a_w <FitzHughNagumoIntegrator.a_w>`   |*NEGATIVE* `b_w <FitzHughNagumoIntegrator.b_w>` |
            +-----------------------------------------+---------------------------------------+------------------------------------------------+
            |**Modified FitzHughNagumo Parameter**    |b                                      |c                                               |
            +-----------------------------------------+---------------------------------------+------------------------------------------------+

    (3) **Modified FitzHughNagumo Model as implemented in** `Gilzenrat (2002) <http://www.sciencedirect.com/science/article/pii/S0893608002000552?via%3Dihub>`_

        **Fast, Excitatory Variable:**

        [Eq. (6) in `Gilzenrat (2002) <http://www.sciencedirect.com/science/article/pii/S0893608002000552?via%3Dihub>`_ ]

        .. math::

            \\tau_v \\frac{dv}{dt} = v(a-v)(v-1) - u + w_{vX_1}\\, f(X_1)

        **Slow, Inactivating Variable:**

        [Eq. (7) & Eq. (8) in `Gilzenrat (2002) <http://www.sciencedirect.com/science/article/pii/S0893608002000552?via%3Dihub>`_ ]

        .. math::

            \\tau_u \\frac{du}{dt} = Cv + (1-C)\\, d - u

        **Implementation in FitzHughNagumoIntegrator**

            The following FitzHughNagumoIntegrator parameter values must be set in the equation for :math:`\\frac{dv}{dt}`:

            +--------------------------------------+-----+-----+-----+-----+-----+
            |**FitzHughNagumoIntegrator Parameter**| a_v | b_v | c_v | d_v | e_v |
            +--------------------------------------+-----+-----+-----+-----+-----+
            |**Value**                             |-1.0 |1.0  |1.0  |0.0  |-1.0 |
            +--------------------------------------+-----+-----+-----+-----+-----+

            When the parameters above are set to the listed values, the FitzHughNagumoIntegrator equation for :math:`\\frac{dv}{dt}`
            reduces to the Gilzenrat formulation, and the remaining parameters in the :math:`\\frac{dv}{dt}` equation
            correspond as follows:

            +---------------------------------------+------------------------------------------------+----------------------------------------------+------------------------------------+---------------------------------------------------------------+
            |**FitzHughNagumoIntegrator Parameter** |`threshold <FitzHughNagumoIntegrator.threshold>`|`variable <FitzHughNagumoIntegrator.variable>`|`f_v <FitzHughNagumoIntegrator.f_v>`|`time_constant_v <FitzHughNagumoIntegrator.time_constant_v>`   |
            +---------------------------------------+------------------------------------------------+----------------------------------------------+-------------------------+--------------------------------------------------------------------------+
            |**Gilzenrat Parameter**                |a                                               |:math:`f(X_1)`                                |:math:`w_{vX_1}`                    |:math:`T_{v}`                                                  |
            +---------------------------------------+------------------------------------------------+----------------------------------------------+------------------------------------+---------------------------------------------------------------+

            The following FitzHughNagumoIntegrator parameter values must be set in the equation for :math:`\\frac{dw}{dt}`:

            +---------------------------------------+-----+-----+-----+
            |**FitzHughNagumoIntegrator Parameter** | a_w | b_w | c_w |
            +---------------------------------------+-----+-----+-----+
            |**Value**                              | 1.0 |-1.0 |0.0  |
            +---------------------------------------+-----+-----+-----+

            When the parameters above are set to the listed values, the FitzHughNagumoIntegrator equation for
            :math:`\\frac{dw}{dt}` reduces to the Gilzenrat formulation, and the remaining parameters in the
            :math:`\\frac{dw}{dt}` equation correspond as follows:

            +----------------------------------------+----------------------------------------+------------------------------------------------------------------------+---------------------------------------------------------------+
            |**FitzHughNagumoIntegrator Parameter**  |`mode <FitzHughNagumoIntegrator.mode>`  |`uncorrelated_activity <FitzHughNagumoIntegrator.uncorrelated_activity>`|`time_constant_v <FitzHughNagumoIntegrator.time_constant_w>`   |
            +----------------------------------------+----------------------------------------+------------------------------------------------------------------------+---------------------------------------------------------------+
            |**Gilzenrat Parameter**                 |C                                       |d                                                                       |:math:`T_{u}`                                                  |
            +----------------------------------------+----------------------------------------+------------------------------------------------------------------------+---------------------------------------------------------------+

    Arguments
    ---------

    default_variable : number, list or array : default class_defaults.variable
        specifies a template for the external stimulus

    initial_w : float, list or 1d array : default 0.0
        specifies starting value for integration of dw/dt.  If it is a list or array, it must be the same length as
        `variable <FitzHughNagumoIntegrator.variable>`.

    initial_v : float, list or 1d array : default 0.0
        specifies starting value for integration of dv/dt.  If it is a list or array, it must be the same length as
        `variable <FitzHughNagumoIntegrator.variable>`

    time_step_size : float : default 0.1
        specifies the time step size of numerical integration

    t_0 : float : default 0.0
        specifies starting value for time

    a_v : float : default -1/3
        coefficient on the v^3 term of the dv/dt equation

    b_v : float : default 0.0
        coefficient on the v^2 term of the dv/dt equation

    c_v : float : default 1.0
        coefficient on the v term of the dv/dt equation

    d_v : float : default 0.0
        constant term in the dv/dt equation

    e_v : float : default -1.0
        coefficient on the w term in the dv/dt equation

    f_v : float : default  1.0
        coefficient on the external stimulus (`variable <FitzHughNagumoIntegrator.variable>`) term in the dv/dt equation

    time_constant_v : float : default 1.0
        scaling factor on the dv/dt equation

    a_w : float : default 1.0,
        coefficient on the v term of the dw/dt equation

    b_w : float : default -0.8,
        coefficient on the w term of the dv/dt equation

    c_w : float : default 0.7,
        constant term in the dw/dt equation

    threshold : float : default -1.0
        specifies a value of the input below which the LC will tend not to respond and above which it will

    mode : float : default 1.0
        coefficient which simulates electrotonic coupling by scaling the values of dw/dt such that the v term
        (representing the input from the LC) increases when the uncorrelated_activity term (representing baseline
        activity) decreases

    uncorrelated_activity : float : default 0.0
        constant term in the dw/dt equation

    time_constant_w : float : default 12.5
        scaling factor on the dv/dt equation

    integration_method: str : default "RK4"
        selects the numerical integration method. Currently, the choices are: "RK4" (4th Order Runge-Kutta) or "EULER"
        (Forward Euler)

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
        External stimulus

    previous_v : 1d array : default class_defaults.variable
        stores accumulated value of v during integration

    previous_w : 1d array : default class_defaults.variable
        stores accumulated value of w during integration

    previous_t : float
        stores accumulated value of time, which is incremented by time_step_size on each execution of the function

    owner : Component
        `component <Component>` to which the Function has been assigned.

    initial_w : float, list or 1d array : default 0.0
        specifies starting value for integration of dw/dt.  If it is a list or array, it must be the same length as
        `variable <FitzHughNagumoIntegrator.variable>`

    initial_v : float, list or 1d array : default 0.0
        specifies starting value for integration of dv/dt.  If it is a list or array, it must be the same length as
        `variable <FitzHughNagumoIntegrator.variable>`

    time_step_size : float : default 0.1
        specifies the time step size of numerical integration

    t_0 : float : default 0.0
        specifies starting value for time

    a_v : float : default -1/3
        coefficient on the v^3 term of the dv/dt equation

    b_v : float : default 0.0
        coefficient on the v^2 term of the dv/dt equation

    c_v : float : default 1.0
        coefficient on the v term of the dv/dt equation

    d_v : float : default 0.0
        constant term in the dv/dt equation

    e_v : float : default -1.0
        coefficient on the w term in the dv/dt equation

    f_v : float : default  1.0
        coefficient on the external stimulus (`variable <FitzHughNagumoIntegrator.variable>`) term in the dv/dt equation

    time_constant_v : float : default 1.0
        scaling factor on the dv/dt equation

    a_w : float : default 1.0
        coefficient on the v term of the dw/dt equation

    b_w : float : default -0.8
        coefficient on the w term of the dv/dt equation

    c_w : float : default 0.7
        constant term in the dw/dt equation

    threshold : float : default -1.0
        coefficient that scales both the v^2 [ (1+threshold)*v^2 ] and v [ (-threshold)*v ] terms in the dv/dt equation
        under a specific formulation of the FitzHughNagumo equations, the threshold parameter behaves as a "threshold of
        excitation", and has the following relationship with variable (the external stimulus):

            - when the external stimulus is below the threshold of excitation, the system is either in a stable state,
              or will emit a single excitation spike, then reach a stable state. The behavior varies depending on the
              magnitude of the difference between the threshold and the stimulus.

            - when the external stimulus is equal to or above the threshold of excitation, the system is
              unstable, and will emit many excitation spikes

            - when the external stimulus is too far above the threshold of excitation, the system will emit some
              excitation spikes before reaching a stable state.

    mode : float : default 1.0
        coefficient which simulates electrotonic coupling by scaling the values of dw/dt such that the v term
        (representing the input from the LC) increases when the uncorrelated_activity term (representing baseline
        activity) decreases

    uncorrelated_activity : float : default 0.0
        constant term in the dw/dt equation

    time_constant_w : float : default 12.5
        scaling factor on the dv/dt equation

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).
    """

    componentName = FITZHUGHNAGUMO_INTEGRATOR_FUNCTION

    class Parameters(IntegratorFunction.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <FitzHughNagumoIntegrator.variable>`

                    :default value: numpy.array([1.])
                    :type: ``numpy.ndarray``
                    :read only: True

                a_v
                    see `a_v <FitzHughNagumoIntegrator.a_v>`

                    :default value: -0.3333333333333333
                    :type: ``float``

                a_w
                    see `a_w <FitzHughNagumoIntegrator.a_w>`

                    :default value: 1.0
                    :type: ``float``

                b_v
                    see `b_v <FitzHughNagumoIntegrator.b_v>`

                    :default value: 0.0
                    :type: ``float``

                b_w
                    see `b_w <FitzHughNagumoIntegrator.b_w>`

                    :default value: -0.8
                    :type: ``float``

                c_v
                    see `c_v <FitzHughNagumoIntegrator.c_v>`

                    :default value: 1.0
                    :type: ``float``

                c_w
                    see `c_w <FitzHughNagumoIntegrator.c_w>`

                    :default value: 0.7
                    :type: ``float``

                d_v
                    see `d_v <FitzHughNagumoIntegrator.d_v>`

                    :default value: 0.0
                    :type: ``float``

                e_v
                    see `e_v <FitzHughNagumoIntegrator.e_v>`

                    :default value: -1.0
                    :type: ``float``

                enable_output_type_conversion
                    see `enable_output_type_conversion <FitzHughNagumoIntegrator.enable_output_type_conversion>`

                    :default value: False
                    :type: ``bool``
                    :read only: True

                f_v
                    see `f_v <FitzHughNagumoIntegrator.f_v>`

                    :default value: 1.0
                    :type: ``float``

                initial_v
                    see `initial_v <FitzHughNagumoIntegrator.initial_v>`

                    :default value: 0.0
                    :type: ``float``

                initial_w
                    see `initial_w <FitzHughNagumoIntegrator.initial_w>`

                    :default value: 0.0
                    :type: ``float``

                integration_method
                    see `integration_method <FitzHughNagumoIntegrator.integration_method>`

                    :default value: `RK4`
                    :type: ``str``

                mode
                    see `mode <FitzHughNagumoIntegrator.mode>`

                    :default value: 1.0
                    :type: ``float``

                previous_time
                    see `previous_time <FitzHughNagumoIntegrator.previous_time>`

                    :default value: 0.0
                    :type: ``float``

                previous_v
                    see `previous_v <FitzHughNagumoIntegrator.previous_v>`

                    :default value: numpy.array([1.])
                    :type: ``numpy.ndarray``

                previous_w
                    see `previous_w <FitzHughNagumoIntegrator.previous_w>`

                    :default value: numpy.array([1.])
                    :type: ``numpy.ndarray``

                t_0
                    see `t_0 <FitzHughNagumoIntegrator.t_0>`

                    :default value: 0.0
                    :type: ``float``

                threshold
                    see `threshold <FitzHughNagumoIntegrator.threshold>`

                    :default value: -1.0
                    :type: ``float``

                time_constant_v
                    see `time_constant_v <FitzHughNagumoIntegrator.time_constant_v>`

                    :default value: 1.0
                    :type: ``float``

                time_constant_w
                    see `time_constant_w <FitzHughNagumoIntegrator.time_constant_w>`

                    :default value: 12.5
                    :type: ``float``

                time_step_size
                    see `time_step_size <FitzHughNagumoIntegrator.time_step_size>`

                    :default value: 0.05
                    :type: ``float``

                uncorrelated_activity
                    see `uncorrelated_activity <FitzHughNagumoIntegrator.uncorrelated_activity>`

                    :default value: 0.0
                    :type: ``float``
        """
        variable = Parameter(np.array([1.0]), read_only=True, pnl_internal=True, constructor_argument='default_variable')
        time_step_size = Parameter(0.05, modulable=True)
        a_v = Parameter(-1.0 / 3, modulable=True)
        b_v = Parameter(0.0, modulable=True)
        c_v = Parameter(1.0, modulable=True)
        d_v = Parameter(0.0, modulable=True)
        e_v = Parameter(-1.0, modulable=True)
        f_v = Parameter(1.0, modulable=True)
        time_constant_v = Parameter(1.0, modulable=True)
        a_w = Parameter(1.0, modulable=True)
        b_w = Parameter(-0.8, modulable=True)
        c_w = Parameter(0.7, modulable=True)
        threshold = Parameter(-1.0, modulable=True)
        time_constant_w = Parameter(12.5, modulable=True)
        mode = Parameter(1.0, modulable=True)
        uncorrelated_activity = Parameter(0.0, modulable=True)

        # FIX: make an integration_method enum class for RK4/EULER
        integration_method = Parameter("RK4", stateful=False)

        initial_w = 0.0
        initial_v = 0.0
        t_0 = 0.0
        previous_w = Parameter(np.array([1.0]), initializer='initial_w', pnl_internal=True)
        previous_v = Parameter(np.array([1.0]), initializer='initial_v', pnl_internal=True)
        previous_time = Parameter(0.0, initializer='t_0', pnl_internal=True)

        # this should be removed because it's unused, but this will
        # require a larger refactoring on previous_value/value
        previous_value = Parameter(None, initializer='initializer')

        enable_output_type_conversion = Parameter(
            False,
            stateful=False,
            loggable=False,
            pnl_internal=True,
            read_only=True
        )

    @check_user_specified
    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 # scale=1.0,
                 # offset=0.0,
                 initial_w=None,
                 initial_v=None,
                 a_v=None,
                 b_v=None,
                 c_v=None,
                 d_v=None,
                 e_v=None,
                 f_v=None,
                 time_constant_v=None,
                 a_w=None,
                 b_w=None,
                 c_w=None,
                 time_constant_w=None,
                 t_0=None,
                 threshold=None,
                 time_step_size=None,
                 mode=None,
                 uncorrelated_activity=None,
                 integration_method=None,
                 params: tc.optional(tc.optional(dict)) = None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None,
                 **kwargs):

        # These may be passed (as standard IntegratorFunction args) but are not used by FitzHughNagumo
        unsupported_args = {NOISE, INITIALIZER, RATE, OFFSET}
        if any(k in unsupported_args for k in kwargs):
            s = 's' if len(kwargs)>1 else ''
            warnings.warn("{} arg{} not supported in {}".
                          format(repr(", ".join(list(kwargs.keys()))),s, self.__class__.__name__))
            for k in unsupported_args:
                if k in kwargs:
                    del kwargs[k]

        super().__init__(
            default_variable=default_variable,
            initial_v=initial_v,
            initial_w=initial_w,
            time_step_size=time_step_size,
            t_0=t_0,
            a_v=a_v,
            b_v=b_v,
            c_v=c_v,
            d_v=d_v,
            e_v=e_v,
            f_v=f_v,
            time_constant_v=time_constant_v,
            a_w=a_w,
            b_w=b_w,
            c_w=c_w,
            threshold=threshold,
            mode=mode,
            uncorrelated_activity=uncorrelated_activity,
            integration_method=integration_method,
            time_constant_w=time_constant_w,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _validate_params(self, request_set, target_set=None, context=None):
        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)
        if self.integration_method not in {"RK4", "EULER"}:
            raise FunctionError("Invalid integration method ({}) selected for {}. Choose 'RK4' or 'EULER'".
                                format(self.integration_method, self.name))

    def _euler_FitzHughNagumo(
        self, variable, previous_value_v, previous_value_w, previous_time, slope_v, slope_w, time_step_size,
        a_v,
        threshold, b_v, c_v, d_v, e_v, f_v, time_constant_v, mode, a_w, b_w, c_w, uncorrelated_activity,
        time_constant_w, context=None
    ):

        slope_v_approx = slope_v(
            variable,
            previous_time,
            previous_value_v,
            previous_value_w,
            a_v,
            threshold,
            b_v,
            c_v,
            d_v,
            e_v,
            f_v,
            time_constant_v,
            context=context
        )

        slope_w_approx = slope_w(
            variable,
            previous_time,
            previous_value_w,
            previous_value_v,
            mode,
            a_w,
            b_w,
            c_w,
            uncorrelated_activity,
            time_constant_w,
            context=context
        )

        new_v = previous_value_v + time_step_size * slope_v_approx
        new_w = previous_value_w + time_step_size * slope_w_approx

        return new_v, new_w

    def _runge_kutta_4_FitzHughNagumo(
        self, variable, previous_value_v, previous_value_w, previous_time, slope_v, slope_w,
        time_step_size,
        a_v, threshold, b_v, c_v, d_v, e_v, f_v, time_constant_v, mode, a_w, b_w, c_w,
        uncorrelated_activity, time_constant_w, context=None
    ):

        # First approximation
        # v is approximately previous_value_v
        # w is approximately previous_value_w

        slope_v_approx_1 = slope_v(
            variable,
            previous_time,
            previous_value_v,
            previous_value_w,
            a_v,
            threshold,
            b_v,
            c_v,
            d_v,
            e_v,
            f_v,
            time_constant_v,
            context=context
        )

        slope_w_approx_1 = slope_w(
            variable,
            previous_time,
            previous_value_w,
            previous_value_v,
            mode,
            a_w,
            b_w,
            c_w,
            uncorrelated_activity,
            time_constant_w,
            context=context
        )
        # Second approximation
        # v is approximately previous_value_v + 0.5 * time_step_size * slope_w_approx_1
        # w is approximately previous_value_w + 0.5 * time_step_size * slope_w_approx_1

        slope_v_approx_2 = slope_v(
            variable,
            previous_time + time_step_size / 2,
            previous_value_v + (0.5 * time_step_size * slope_v_approx_1),
            previous_value_w + (0.5 * time_step_size * slope_w_approx_1),
            a_v,
            threshold,
            b_v,
            c_v,
            d_v,
            e_v,
            f_v,
            time_constant_v,
            context=context
        )

        slope_w_approx_2 = slope_w(
            variable,
            previous_time + time_step_size / 2,
            previous_value_w + (0.5 * time_step_size * slope_w_approx_1),
            previous_value_v + (0.5 * time_step_size * slope_v_approx_1),
            mode,
            a_w,
            b_w,
            c_w,
            uncorrelated_activity,
            time_constant_w,
            context=context
        )

        # Third approximation
        # v is approximately previous_value_v + 0.5 * time_step_size * slope_v_approx_2
        # w is approximately previous_value_w + 0.5 * time_step_size * slope_w_approx_2

        slope_v_approx_3 = slope_v(
            variable,
            previous_time + time_step_size / 2,
            previous_value_v + (0.5 * time_step_size * slope_v_approx_2),
            previous_value_w + (0.5 * time_step_size * slope_w_approx_2),
            a_v,
            threshold,
            b_v,
            c_v,
            d_v,
            e_v,
            f_v,
            time_constant_v,
            context=context
        )

        slope_w_approx_3 = slope_w(
            variable,
            previous_time + time_step_size / 2,
            previous_value_w + (0.5 * time_step_size * slope_w_approx_2),
            previous_value_v + (0.5 * time_step_size * slope_v_approx_2),
            mode,
            a_w,
            b_w,
            c_w,
            uncorrelated_activity,
            time_constant_w,
            context=context
        )
        # Fourth approximation
        # v is approximately previous_value_v + time_step_size * slope_v_approx_3
        # w is approximately previous_value_w + time_step_size * slope_w_approx_3

        slope_v_approx_4 = slope_v(
            variable,
            previous_time + time_step_size,
            previous_value_v + (time_step_size * slope_v_approx_3),
            previous_value_w + (time_step_size * slope_w_approx_3),
            a_v,
            threshold,
            b_v,
            c_v,
            d_v,
            e_v,
            f_v,
            time_constant_v,
            context=context
        )

        slope_w_approx_4 = slope_w(
            variable,
            previous_time + time_step_size,
            previous_value_w + (time_step_size * slope_w_approx_3),
            previous_value_v + (time_step_size * slope_v_approx_3),
            mode,
            a_w,
            b_w,
            c_w,
            uncorrelated_activity,
            time_constant_w,
            context=context
        )

        new_v = previous_value_v \
                + (time_step_size / 6) * (
        slope_v_approx_1 + 2 * (slope_v_approx_2 + slope_v_approx_3) + slope_v_approx_4)
        new_w = previous_value_w \
                + (time_step_size / 6) * (
        slope_w_approx_1 + 2 * (slope_w_approx_2 + slope_w_approx_3) + slope_w_approx_4)

        return new_v, new_w

    def dv_dt(self, variable, time, v, w, a_v, threshold, b_v, c_v, d_v, e_v, f_v, time_constant_v, context=None):
        previous_w = self._get_current_parameter_value('previous_w', context)

        val = (a_v * (v ** 3) + (1 + threshold) * b_v * (v ** 2) + (-threshold) * c_v * v + d_v
               + e_v * previous_w + f_v * variable) / time_constant_v

        # Standard coefficients - hardcoded for testing
        # val = v - (v**3)/3 - w + variable
        # Gilzenrat paper - hardcoded for testing
        # val = (v*(v-0.5)*(1-v) - w + variable)/0.01
        return val

    def dw_dt(self, variable, time, w, v, mode, a_w, b_w, c_w, uncorrelated_activity, time_constant_w, context=None):
        previous_v = self._get_current_parameter_value('previous_v', context)

        # val = np.ones_like(variable)*(mode*a_w*self.previous_v + b_w*w + c_w + (1-mode)*uncorrelated_activity)/time_constant_w
        val = (mode * a_w * previous_v + b_w * w + c_w + (1 - mode) * uncorrelated_activity) / time_constant_w

        # Standard coefficients - hardcoded for testing
        # val = (v + 0.7 - 0.8*w)/12.5
        # Gilzenrat paper - hardcoded for testing

        # val = (v - 0.5*w)
        if not np.isscalar(variable):
            val = np.broadcast_to(val, variable.shape)

        return val

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        current value of v , current value of w : float, list, or array

        """

        # FIX: SHOULDN'T THERE BE A CALL TO _get_current_parameter_value('variable', context) HERE??

        # # FIX: TEMPORARY CHECK UNTIL ARRAY IS SUPPORTED
        # if variable is not None and not np.isscalar(variable) and len(variable)>1:
        #     raise FunctionError("{} presently supports only a scalar variable".format(self.__class__.__name__))

        a_v = self._get_current_parameter_value("a_v", context)
        b_v = self._get_current_parameter_value("b_v", context)
        c_v = self._get_current_parameter_value("c_v", context)
        d_v = self._get_current_parameter_value("d_v", context)
        e_v = self._get_current_parameter_value("e_v", context)
        f_v = self._get_current_parameter_value("f_v", context)
        time_constant_v = self._get_current_parameter_value("time_constant_v", context)
        threshold = self._get_current_parameter_value("threshold", context)
        a_w = self._get_current_parameter_value("a_w", context)
        b_w = self._get_current_parameter_value("b_w", context)
        c_w = self._get_current_parameter_value("c_w", context)
        uncorrelated_activity = self._get_current_parameter_value("uncorrelated_activity", context)
        time_constant_w = self._get_current_parameter_value("time_constant_w", context)
        mode = self._get_current_parameter_value("mode", context)
        time_step_size = self._get_current_parameter_value(TIME_STEP_SIZE, context)
        previous_v = self._get_current_parameter_value("previous_v", context)
        previous_w = self._get_current_parameter_value("previous_w", context)
        previous_time = self._get_current_parameter_value("previous_time", context)

        # integration_method is a compile time parameter
        integration_method = self.parameters.integration_method.get()
        if integration_method == "RK4":
            approximate_values = self._runge_kutta_4_FitzHughNagumo(
                variable,
                previous_v,
                previous_w,
                previous_time,
                self.dv_dt,
                self.dw_dt,
                time_step_size,
                a_v,
                threshold,
                b_v,
                c_v,
                d_v,
                e_v,
                f_v,
                time_constant_v,
                mode,
                a_w,
                b_w,
                c_w,
                uncorrelated_activity,
                time_constant_w,
                context=context
            )

        elif integration_method == "EULER":
            approximate_values = self._euler_FitzHughNagumo(
                variable,
                previous_v,
                previous_w,
                previous_time,
                self.dv_dt,
                self.dw_dt,
                time_step_size,
                a_v,
                threshold,
                b_v,
                c_v,
                d_v,
                e_v,
                f_v,
                time_constant_v,
                mode,
                a_w,
                b_w,
                c_w,
                uncorrelated_activity,
                time_constant_w,
                context=context
            )
        else:
            raise FunctionError("Invalid integration method ({}) selected for {}".
                                format(integration_method, self.name))

        if not self.is_initializing:
            previous_v = approximate_values[0]
            previous_w = approximate_values[1]
            previous_time = previous_time + time_step_size
            if not np.isscalar(variable):
                previous_time = np.broadcast_to(previous_time, variable.shape).copy()

            self.parameters.previous_v._set(previous_v, context)
            self.parameters.previous_w._set(previous_w, context)
            self.parameters.previous_time._set(previous_time, context)

        return previous_v, previous_w, previous_time

    def reset(self, previous_v=None, previous_w=None, previous_time=None, context=None):
        return super().reset(
            previous_v=previous_v,
            previous_w=previous_w,
            previous_time=previous_time,
            context=context
        )

    def _gen_llvm_function_body(self, ctx, builder, params, state, arg_in, arg_out, *, tags:frozenset):
        zero_i32 = ctx.int32_ty(0)

        # Get rid of 2d array. When part of a Mechanism the input,
        # (and output, and state) are 2d arrays.
        arg_in = pnlvm.helpers.unwrap_2d_array(builder, arg_in)

        # Get state pointers
        def _get_state_ptr(x):
            ptr = pnlvm.helpers.get_state_ptr(builder, self, state, x)
            return pnlvm.helpers.unwrap_2d_array(builder, ptr)
        prev = {s: _get_state_ptr(s) for s in self.llvm_state_ids}

        # Output locations
        def _get_out_ptr(i):
            ptr = builder.gep(arg_out, [zero_i32, ctx.int32_ty(i)])
            return pnlvm.helpers.unwrap_2d_array(builder, ptr)
        out = {l: _get_out_ptr(i) for i, l in enumerate(('v', 'w', 'time'))}

        # Load parameters
        def _get_param_val(x):
            ptr = pnlvm.helpers.get_param_ptr(builder, self, params, x)
            return pnlvm.helpers.load_extract_scalar_array_one(builder, ptr)
        param_vals = {p: _get_param_val(p) for p in self.llvm_param_ids}

        inner_args = {"ctx": ctx, "var_ptr": arg_in, "param_vals": param_vals,
                      "out_v": out['v'], "out_w": out['w'],
                      "out_time": out['time'],
                      "previous_v_ptr": prev['previous_v'],
                      "previous_w_ptr": prev['previous_w'],
                      "previous_time_ptr": prev['previous_time']}

        method = self.parameters.integration_method.get()

        with pnlvm.helpers.array_ptr_loop(builder, arg_in, method + "_body") as args:
            if method == "RK4":
                self.__gen_llvm_rk4_body(*args, **inner_args)
            elif method == "EULER":
                self.__gen_llvm_euler_body(*args, **inner_args)
            else:
                raise FunctionError("Invalid integration method ({}) selected for {}".
                                    format(method, self.name))

        # Save state
        for n, sptr in out.items():
            dptr = prev["previous_" + n]
            builder.store(builder.load(sptr), dptr)
        return builder

    def __gen_llvm_rk4_body(self, builder, index, ctx, var_ptr, out_v, out_w, out_time, param_vals, previous_v_ptr, previous_w_ptr, previous_time_ptr):
        var = builder.load(builder.gep(var_ptr, [ctx.int32_ty(0), index]))

        previous_v = builder.load(builder.gep(previous_v_ptr, [ctx.int32_ty(0), index]))
        previous_w = builder.load(builder.gep(previous_w_ptr, [ctx.int32_ty(0), index]))
        previous_time = builder.load(builder.gep(previous_time_ptr, [ctx.int32_ty(0), index]))

        out_v_ptr = builder.gep(out_v, [ctx.int32_ty(0), index])
        out_w_ptr = builder.gep(out_w, [ctx.int32_ty(0), index])
        out_time_ptr = builder.gep(out_time, [ctx.int32_ty(0), index])

        time_step_size = param_vals[TIME_STEP_SIZE]

        # Save output time
        time = builder.fadd(previous_time, time_step_size)
        builder.store(time, out_time_ptr)

        # First approximation uses previous_v
        input_v = previous_v
        slope_v_approx_1 = self.__gen_llvm_dv_dt(builder, ctx, var, input_v, previous_w, param_vals)

        # First approximation uses previous_w
        input_w = previous_w
        slope_w_approx_1 = self.__gen_llvm_dw_dt(builder, ctx, input_w, previous_v, param_vals)

        # Second approximation
        # v is approximately previous_value_v + 0.5 * time_step_size * slope_v_approx_1
        input_v = builder.fmul(ctx.float_ty(0.5), time_step_size)
        input_v = builder.fmul(input_v, slope_v_approx_1)
        input_v = builder.fadd(input_v, previous_v)
        slope_v_approx_2 = self.__gen_llvm_dv_dt(builder, ctx, var, input_v, previous_w, param_vals)

        # w is approximately previous_value_w + 0.5 * time_step_size * slope_w_approx_1
        input_w = builder.fmul(ctx.float_ty(0.5), time_step_size)
        input_w = builder.fmul(input_w, slope_w_approx_1)
        input_w = builder.fadd(input_w, previous_w)
        slope_w_approx_2 = self.__gen_llvm_dw_dt(builder, ctx, input_w, previous_v, param_vals)

        # Third approximation
        # v is approximately previous_value_v + 0.5 * time_step_size * slope_v_approx_2
        input_v = builder.fmul(ctx.float_ty(0.5), time_step_size)
        input_v = builder.fmul(input_v, slope_v_approx_2)
        input_v = builder.fadd(input_v, previous_v)
        slope_v_approx_3 = self.__gen_llvm_dv_dt(builder, ctx, var, input_v, previous_w, param_vals)

        # w is approximately previous_value_w + 0.5 * time_step_size * slope_w_approx_2
        input_w = builder.fmul(ctx.float_ty(0.5), time_step_size)
        input_w = builder.fmul(input_w, slope_w_approx_2)
        input_w = builder.fadd(input_w, previous_w)
        slope_w_approx_3 = self.__gen_llvm_dw_dt(builder, ctx, input_w, previous_v, param_vals)

        # Fourth approximation
        # v is approximately previous_value_v + time_step_size * slope_v_approx_3
        input_v = builder.fmul(time_step_size, slope_v_approx_3)
        input_v = builder.fadd(input_v, previous_v)
        slope_v_approx_4 = self.__gen_llvm_dv_dt(builder, ctx, var, input_v, previous_w, param_vals)

        # w is approximately previous_value_w + time_step_size * slope_w_approx_3
        input_w = builder.fmul(time_step_size, slope_w_approx_3)
        input_w = builder.fadd(input_w, previous_w)
        slope_w_approx_4 = self.__gen_llvm_dw_dt(builder, ctx, input_w, previous_v, param_vals)

        ts = builder.fdiv(time_step_size, ctx.float_ty(6.0))
        # new_v = previous_value_v \
        #    + (time_step_size/6) * (slope_v_approx_1
        #    + 2 * (slope_v_approx_2 + slope_v_approx_3) + slope_v_approx_4)
        new_v = builder.fadd(slope_v_approx_2, slope_v_approx_3)
        new_v = builder.fmul(new_v, ctx.float_ty(2.0))
        new_v = builder.fadd(new_v, slope_v_approx_1)
        new_v = builder.fadd(new_v, slope_v_approx_4)
        new_v = builder.fmul(new_v, ts)
        new_v = builder.fadd(new_v, previous_v)
        builder.store(new_v, out_v_ptr)

        # new_w = previous_walue_w \
        #    + (time_step_size/6) * (slope_w_approx_1
        #    + 2 * (slope_w_approx_2 + slope_w_approx_3) + slope_w_approx_4)
        new_w = builder.fadd(slope_w_approx_2, slope_w_approx_3)
        new_w = builder.fmul(new_w, ctx.float_ty(2.0))
        new_w = builder.fadd(new_w, slope_w_approx_1)
        new_w = builder.fadd(new_w, slope_w_approx_4)
        new_w = builder.fmul(new_w, ts)
        new_w = builder.fadd(new_w, previous_w)
        builder.store(new_w, out_w_ptr)

    def __gen_llvm_euler_body(self, builder, index, ctx, var_ptr, out_v, out_w, out_time, param_vals, previous_v_ptr, previous_w_ptr, previous_time_ptr):

        var = builder.load(builder.gep(var_ptr, [ctx.int32_ty(0), index]))
        previous_v = builder.load(builder.gep(previous_v_ptr, [ctx.int32_ty(0), index]))
        previous_w = builder.load(builder.gep(previous_w_ptr, [ctx.int32_ty(0), index]))
        previous_time = builder.load(builder.gep(previous_time_ptr, [ctx.int32_ty(0), index]))
        out_v_ptr = builder.gep(out_v, [ctx.int32_ty(0), index])
        out_w_ptr = builder.gep(out_w, [ctx.int32_ty(0), index])
        out_time_ptr = builder.gep(out_time, [ctx.int32_ty(0), index])

        time_step_size = param_vals[TIME_STEP_SIZE]

        # Save output time
        time = builder.fadd(previous_time, time_step_size)
        builder.store(time, out_time_ptr)

        # First approximation uses previous_v
        slope_v_approx = self.__gen_llvm_dv_dt(builder, ctx, var, previous_v, previous_w, param_vals)

        # First approximation uses previous_w
        slope_w_approx = self.__gen_llvm_dw_dt(builder, ctx, previous_w, previous_v, param_vals)

        # new_v = previous_value_v + time_step_size*slope_v_approx
        new_v = builder.fmul(time_step_size, slope_v_approx)
        new_v = builder.fadd(previous_v, new_v)
        builder.store(new_v, out_v_ptr)
        # new_w = previous_value_w + time_step_size*slope_w_approx
        new_w = builder.fmul(time_step_size, slope_w_approx)
        new_w = builder.fadd(previous_w, new_w)
        builder.store(new_w, out_w_ptr)

    def __gen_llvm_dv_dt(self, builder, ctx, var, v, previous_w, param_vals):
        # val = (a_v*(v**3) + (1+threshold)*b_v*(v**2) + (-threshold)*c_v*v +
        #       d_v + e_v*self.previous_w + f_v*variable)/time_constant_v
        pow_f = ctx.get_builtin("pow", [ctx.float_ty])

        v_3 = builder.call(pow_f, [v, ctx.float_ty(3.0)])
        tmp1 = builder.fmul(param_vals["a_v"], v_3)

        thr_p1 = builder.fadd(ctx.float_ty(1.0), param_vals["threshold"])
        tmp2 = builder.fmul(thr_p1, param_vals["b_v"])
        v_2 = builder.call(pow_f, [v, ctx.float_ty(2.0)])
        tmp2 = builder.fmul(tmp2, v_2)

        thr_neg = pnlvm.helpers.fneg(builder, param_vals["threshold"])
        tmp3 = builder.fmul(thr_neg, param_vals["c_v"])
        tmp3 = builder.fmul(tmp3, v)

        tmp4 = param_vals["d_v"]

        tmp5 = builder.fmul(param_vals["e_v"], previous_w)

        tmp6 = builder.fmul(param_vals["f_v"], var)

        sum = ctx.float_ty(-0.0)
        sum = builder.fadd(sum, tmp1)
        sum = builder.fadd(sum, tmp2)
        sum = builder.fadd(sum, tmp3)
        sum = builder.fadd(sum, tmp4)
        sum = builder.fadd(sum, tmp5)
        sum = builder.fadd(sum, tmp6)

        res = builder.fdiv(sum, param_vals["time_constant_v"])

        return res

    def __gen_llvm_dw_dt(self, builder, ctx, w, previous_v, param_vals):
        # val = (mode*a_w*self.previous_v + b_w*w + c_w +
        #       (1-mode)*uncorrelated_activity)/time_constant_w

        tmp1 = builder.fmul(param_vals["mode"], previous_v)
        tmp1 = builder.fmul(tmp1, param_vals["a_w"])

        tmp2 = builder.fmul(param_vals["b_w"], w)

        tmp3 = param_vals["c_w"]

        mod_1 = builder.fsub(ctx.float_ty(1.0), param_vals["mode"])
        tmp4 = builder.fmul(mod_1, param_vals["uncorrelated_activity"])

        sum = ctx.float_ty(-0.0)
        sum = builder.fadd(sum, tmp1)
        sum = builder.fadd(sum, tmp2)
        sum = builder.fadd(sum, tmp3)
        sum = builder.fadd(sum, tmp4)

        res = builder.fdiv(sum, param_vals["time_constant_w"])
        return res

    # TODO: remove with changes to previous_value/value as stated in
    # FitzHughNagumoIntegrator.Parameters
    @property
    def stateful_attributes(self):
        res = super().stateful_attributes
        res.remove('previous_value')
        return res
