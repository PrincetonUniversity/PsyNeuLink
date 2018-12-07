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
'''

* `Integrator`
* `SimpleIntegrator`
* `InteractiveActivation`
* `ConstantIntegrator`
* `Buffer`
* `AdaptiveIntegrator`
* `DriftDiffusionIntegrator`
* `OrnsteinUhlenbeckIntegrator`
* `AccumulatorIntegrator`
* `LCAIntegrator`
* `FHNIntegrator`
* `AGTUtilityIntegrator`
* `BogaczEtAl`
* `NavarroAndFuss`

Overview
--------

Functions that integrate their input.

'''

import functools
import itertools
import numbers
import warnings
from collections.__init__ import deque
from enum import IntEnum

import numpy as np
import typecheck as tc
from llvmlite import ir

from psyneulink.core.components.component import DefaultsFlexibility
from psyneulink.core.components.functions.function import Function_Base, FunctionError,  MULTIPLICATIVE_PARAM, ADDITIVE_PARAM
from psyneulink.core.components.functions.distributionfunctions import DistributionFunction
from psyneulink.core.globals.keywords import \
    ACCUMULATOR_INTEGRATOR_FUNCTION, ADAPTIVE_INTEGRATOR_FUNCTION, BUFFER_FUNCTION, CONSTANT_INTEGRATOR_FUNCTION, \
    DECAY, DRIFT_DIFFUSION_INTEGRATOR_FUNCTION, FHN_INTEGRATOR_FUNCTION, FUNCTION, INCREMENT, INITIALIZER, \
    INPUT_STATES, INTEGRATOR_FUNCTION, INTEGRATOR_FUNCTION_TYPE, INTERACTIVE_ACTIVATION_INTEGRATOR_FUNCTION, \
    LCAMechanism_INTEGRATOR_FUNCTION, NOISE, OFFSET, OPERATION, ORNSTEIN_UHLENBECK_INTEGRATOR_FUNCTION, OUTPUT_STATES, \
    RATE, REST, SCALE, SIMPLE_INTEGRATOR_FUNCTION, TIME_STEP_SIZE, UTILITY_INTEGRATOR_FUNCTION
from psyneulink.core.globals.parameters import Param
from psyneulink.core.globals.utilities import iscompatible, parameter_spec
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core import llvm as pnlvm
from psyneulink.core.llvm import helpers


__all__ = ['Integrator', 'IntegratorFunction', 'SimpleIntegrator', 'ConstantIntegrator', 'Buffer',
           'AdaptiveIntegrator', 'DriftDiffusionIntegrator', 'OrnsteinUhlenbeckIntegrator', 'FHNIntegrator',
           'AccumulatorIntegrator', 'LCAIntegrator', 'AGTUtilityIntegrator',
           'DRIFT_RATE', 'DRIFT_RATE_VARIABILITY', 'THRESHOLD', 'THRESHOLD_VARIABILITY', 'STARTING_POINT',
           'STARTING_POINT_VARIABILITY', 'NON_DECISION_TIME',
           'kwBogaczEtAl', 'kwNavarrosAndFuss', 'BogaczEtAl', 'NF_Results', 'NavarroAndFuss', 'InteractiveActivation']


class IntegratorFunction(Function_Base):
    componentType = INTEGRATOR_FUNCTION_TYPE


# • why does integrator return a 2d array?
# • are rate and noise converted to 1d np.array?  If not, correct docstring
# • can noise and initializer be an array?  If so, validated in validate_param?


class Integrator(IntegratorFunction):  # -------------------------------------------------------------------------------
    """
    Integrator(                 \
        default_variable=None,  \
        rate=1.0,               \
        noise=0.0,              \
        time_step_size=1.0,     \
        initializer,            \
        params=None,            \
        owner=None,             \
        prefs=None,             \
        )

    .. _Integrator:

    Integrate current value of `variable <Integrator.variable>` with its prior value.

    Arguments
    ---------

    default_variable : number, list or array : default ClassDefaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float, list or 1d array : default 1.0
        specifies the rate of integration.  If it is a list or array, it must be the same length as
        `variable <Integrator.default_variable>` (see `rate <Integrator.rate>` for details).

    noise : float, PsyNeuLink Function, list or 1d array : default 0.0
        specifies random value to be added in each call to `function <Integrator.function>`. (see
        `noise <Integrator.noise>` for details).

    time_step_size : float : default 0.0
        determines the timing precision of the integration process

    initializer float, list or 1d array : default 0.0
        specifies starting value for integration.  If it is a list or array, it must be the same length as
        `default_variable <Integrator.default_variable>` (see `initializer <Integrator.initializer>` for details).

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
        current input value some portion of which (determined by `rate <Integrator.rate>`) that will be
        added to the prior value;  if it is an array, each element is independently integrated.

    rate : float or 1d array
        determines the rate of integration based on current and prior values.  If integration_type is set to ADAPTIVE,
        all elements must be between 0 and 1 (0 = no change; 1 = instantaneous change). If it has a single element, it
        applies to all elements of `variable <Integrator.variable>`;  if it has more than one element, each element
        applies to the corresponding element of `variable <Integrator.variable>`.

    noise : float, function, list, or 1d array
        specifies random value to be added in each call to `function <Integrator.function>`.

        If noise is a list or array, it must be the same length as `variable <Integrator.default_variable>`. If noise is
        specified as a single float or function, while `variable <Integrator.variable>` is a list or array,
        noise will be applied to each variable element. In the case of a noise function, this means that the function
        will be executed separately for each variable element.

        Note that in the case of DIFFUSION, noise must be specified as a float (or list or array of floats) because this
        value will be used to construct the standard DDM probability distribution. For all other types of integration,
        in order to generate random noise, we recommend that you instead select a probability distribution function
        (see `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value from
        its distribution on each execution. If noise is specified as a float or as a function with a fixed output (or a
        list or array of these), then the noise will simply be an offset that remains the same across all executions.

    initializer : 1d array or list
        determines the starting value for integration (i.e., the value to which
        `previous_value <Integrator.previous_value>` is set.

        If initializer is a list or array, it must be the same length as `variable <Integrator.default_variable>`. If
        initializer is specified as a single float or function, while `variable <Integrator.variable>` is a list or
        array, initializer will be applied to each variable element. In the case of an initializer function, this means
        that the function will be executed separately for each variable element.

    previous_value : 1d array
        stores previous value with which `variable <Integrator.variable>` is integrated.

    initializers : list
        stores the names of the initialization attributes for each of the stateful attributes of the function. The
        index i item in initializers provides the initialization value for the index i item in `stateful_attributes
        <Integrator.stateful_attributes>`.

    stateful_attributes : list
        stores the names of each of the stateful attributes of the function. The index i item in stateful_attributes is
        initialized by the value of the initialization attribute whose name is stored in index i of `initializers
        <Integrator.initializers>`. In most cases, the stateful_attributes, in that order, are the return values of the
        function.

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

    componentName = INTEGRATOR_FUNCTION

    class Params(IntegratorFunction.Params):
        noise = Param(0.0, modulable=True)
        rate = Param(1.0, modulable=True)
        previous_value = np.array([0])
        initializer = np.array([0])

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        NOISE: None,
        RATE: None
    })

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate: parameter_spec = 1.0,
                 noise=0.0,
                 initializer=None,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None,
                 context=None):

        if not hasattr(self, "initializers"):
            self.initializers = ["initializer"]

        if not hasattr(self, "stateful_attributes"):
            self.stateful_attributes = ["previous_value"]

        if initializer is None:
            if params is not None and INITIALIZER in params and params[INITIALIZER] is not None:
                # This is only needed as long as a new copy of a function is created
                # whenever assigning the function to a mechanism.
                # The old values are compiled and passed in through params argument.
                initializer = params[INITIALIZER]

            else:
                initializer = self.ClassDefaults.variable

        previous_value = self._initialize_previous_value(initializer)

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  initializer=initializer,
                                                  previous_value=previous_value,
                                                  noise=noise,
                                                  params=params)

        # does not actually get set in _assign_args_to_param_dicts but we need it as an instance_default
        params[INITIALIZER] = initializer

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=context)

        self.has_initializers = True

    def _validate(self):
        self._validate_rate(self.instance_defaults.rate)
        self._validate_initializers(self.instance_defaults.variable)
        super()._validate()

    def _validate_params(self, request_set, target_set=None, context=None):

        # Handle list or array for rate specification
        if RATE in request_set:
            rate = request_set[RATE]

            if isinstance(rate, (list, np.ndarray)) and not iscompatible(rate, self.instance_defaults.variable):
                if len(rate) != 1 and len(rate) != np.array(self.instance_defaults.variable).size:
                    # If the variable was not specified, then reformat it to match rate specification
                    #    and assign ClassDefaults.variable accordingly
                    # Note: this situation can arise when the rate is parametrized (e.g., as an array) in the
                    #       Integrator's constructor, where that is used as a specification for a function parameter
                    #       (e.g., for an IntegratorMechanism), whereas the input is specified as part of the
                    #       object to which the function parameter belongs (e.g., the IntegratorMechanism); in that
                    #       case, the Integrator gets instantiated using its ClassDefaults.variable ([[0]]) before
                    #       the object itself, thus does not see the array specification for the input.
                    if self._default_variable_flexibility is DefaultsFlexibility.FLEXIBLE:
                        self._instantiate_defaults(variable=np.zeros_like(np.array(rate)), context=context)
                        if self.verbosePref:
                            warnings.warn(
                                "The length ({}) of the array specified for the rate parameter ({}) of {} "
                                "must match the length ({}) of the default input ({});  "
                                "the default input has been updated to match".format(
                                    len(rate),
                                    rate,
                                    self.name,
                                    np.array(self.instance_defaults.variable).size
                                ),
                                self.instance_defaults.variable,
                            )
                    else:
                        raise FunctionError(
                            "The length of the array specified for the rate parameter of {} ({})"
                            "must match the length of the default input ({}).".format(
                                len(rate),
                                # rate,
                                self.name,
                                np.array(self.instance_defaults.variable).size,
                                # self.instance_defaults.variable,
                            )
                        )
                        # OLD:
                        # self.paramClassDefaults[RATE] = np.zeros_like(np.array(rate))

                        # KAM changed 5/15 b/c paramClassDefaults were being updated and *requiring* future integrator functions
                        # to have a rate parameter of type ndarray/list

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if NOISE in target_set:
            noise = target_set[NOISE]
            if isinstance(noise, DistributionFunction):
                noise.owner = self
                target_set[NOISE] = noise._execute
            self._validate_noise(target_set[NOISE])

    def _validate_rate(self, rate):
        # kmantel: this duplicates much code in _validate_params above, but that calls _instantiate_defaults
        # which I don't think is the right thing to do here, but if you don't call it in _validate_params
        # then a lot of things don't get instantiated properly
        if rate is not None:
            if isinstance(rate, list):
                rate = np.asarray(rate)

            rate_type_msg = 'The rate parameter of {0} must be a number or an array/list of at most 1d (you gave: {1})'
            if isinstance(rate, np.ndarray):
                # kmantel: current test_gating test depends on 2d rate
                #   this should be looked at but for now this restriction is removed
                # if rate.ndim > 1:
                #     raise FunctionError(rate_type_msg.format(self.name, rate))
                pass
            elif not isinstance(rate, numbers.Number):
                raise FunctionError(rate_type_msg.format(self.name, rate))

            if isinstance(rate, np.ndarray) and not iscompatible(rate, self.instance_defaults.variable):
                if len(rate) != 1 and len(rate) != np.array(self.instance_defaults.variable).size:
                    if self._default_variable_flexibility is DefaultsFlexibility.FLEXIBLE:
                        self.instance_defaults.variable = np.zeros_like(np.array(rate))
                        if self.verbosePref:
                            warnings.warn(
                                "The length ({}) of the array specified for the rate parameter ({}) of {} "
                                "must match the length ({}) of the default input ({});  "
                                "the default input has been updated to match".format(
                                    len(rate),
                                    rate,
                                    self.name,
                                    np.array(self.instance_defaults.variable).size
                                ),
                                self.instance_defaults.variable,
                            )
                        self._instantiate_value()
                        self._default_variable_flexibility = DefaultsFlexibility.INCREASE_DIMENSION
                    else:
                        raise FunctionError(
                            "The length of the array specified for the rate parameter of {} ({})"
                            "must match the length of the default input ({}).".format(
                                len(rate),
                                # rate,
                                self.name,
                                np.array(self.instance_defaults.variable).size,
                                # self.instance_defaults.variable,
                            )
                        )

    def _instantiate_attributes_before_function(self, function=None, context=None):

        # use np.broadcast_to to guarantee that all initializer type attributes take on the same shape as variable
        if not np.isscalar(self.instance_defaults.variable):
            for attr in self.initializers:
                setattr(self, attr, np.broadcast_to(getattr(self, attr), self.instance_defaults.variable.shape).copy())

        # create all stateful attributes and initialize their values to the current values of their
        # corresponding initializer attributes
        for i in range(len(self.stateful_attributes)):
            attr_name = self.stateful_attributes[i]
            initializer_value = getattr(self, self.initializers[i]).copy()
            setattr(self, attr_name, initializer_value)

        self.has_initializers = True

        super()._instantiate_attributes_before_function(function=function, context=context)

    # Ensure that the noise parameter makes sense with the input type and shape; flag any noise functions that will
    # need to be executed
    def _validate_noise(self, noise):
        # Noise is a list or array
        if isinstance(noise, (np.ndarray, list)):
            if len(noise) == 1:
                pass
            # Variable is a list/array
            elif (not iscompatible(np.atleast_2d(noise), self.instance_defaults.variable)
                  and not iscompatible(np.atleast_1d(noise), self.instance_defaults.variable) and len(noise) > 1):
                raise FunctionError(
                    "Noise parameter ({}) does not match default variable ({}). Noise parameter of {} "
                    "must be specified as a float, a function, or an array of the appropriate shape ({})."
                        .format(noise, self.instance_defaults.variable, self.name,
                                np.shape(np.array(self.instance_defaults.variable))))
            else:
                for i in range(len(noise)):
                    if isinstance(noise[i], DistributionFunction):
                        noise[i] = noise[i]._execute
                    if not isinstance(noise[i], (float, int)) and not callable(noise[i]):
                        raise FunctionError("The elements of a noise list or array must be floats or functions. "
                                            "{} is not a valid noise element for {}".format(noise[i], self.name))

        # Otherwise, must be a float, int or function
        elif not isinstance(noise, (float, int)) and not callable(noise):
            raise FunctionError(
                "Noise parameter ({}) for {} must be a float, function, or array/list of these."
                    .format(noise, self.name))

    def _validate_initializers(self, default_variable):
        for initial_value_name in self.initializers:

            initial_value = self.get_current_function_param(initial_value_name)

            if isinstance(initial_value, (list, np.ndarray)):
                if len(initial_value) != 1:
                    # np.atleast_2d may not be necessary here?
                    if np.shape(np.atleast_2d(initial_value)) != np.shape(np.atleast_2d(default_variable)):
                        raise FunctionError("{}'s {} ({}) is incompatible with its default_variable ({}) ."
                                            .format(self.name, initial_value_name, initial_value, default_variable))
            elif not isinstance(initial_value, (float, int)):
                raise FunctionError("{}'s {} ({}) must be a number or a list/array of numbers."
                                    .format(self.name, initial_value_name, initial_value))

    def _initialize_previous_value(self, initializer, execution_context=None):
        if execution_context is None:
            # if this is run during initialization, self.parameters will refer to self.class_parameters
            # because self.parameters has not been created yet
            self.previous_value = np.atleast_1d(initializer)
        else:
            self.parameters.previous_value.set(np.atleast_1d(initializer), execution_context)

    def _try_execute_param(self, param, var):

        # param is a list; if any element is callable, execute it
        if isinstance(param, (np.ndarray, list)):
            # NOTE: np.atleast_2d will cause problems if the param has "rows" of different lengths
            param = np.atleast_2d(param)
            for i in range(len(param)):
                for j in range(len(param[i])):
                    if callable(param[i][j]):
                        param[i][j] = param[i][j]()
        # param is one function
        elif callable(param):
            # NOTE: np.atleast_2d will cause problems if the param has "rows" of different lengths
            new_param = []
            for row in np.atleast_2d(var):
                new_row = []
                for item in row:
                    new_row.append(param())
                new_param.append(new_row)
            param = new_param

        return param

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

    def reinitialize(self, *args, execution_context=None):
        """
            Effectively begins accumulation over again at the specified value(s).

            If arguments are passed into the reinitialize method, then reinitialize sets each of the attributes in
            `stateful_attributes <Integrator.stateful_attributes>` to the value of the corresponding argument. Next, it
            sets the `value <Integrator.value>` to a list containing each of the argument values.

            If reinitialize is called without arguments, then it sets each of the attributes in `stateful_attributes
            <Integrator.stateful_attributes>` to the value of the corresponding attribute in `initializers
            <Integrator.initializers>`. Next, it sets the `value <Integrator.value>` to a list containing the values of
            each of the attributes in `initializers <Integrator.initializers>`.

            Often, the only attribute in `stateful_attributes <Integrator.stateful_attributes>` is
            `previous_value <Integrator.previous_value>` and the only attribute in `initializers
            <Integrator.initializers>` is `initializer <Integrator.initializer>`, in which case the reinitialize method
            sets `previous_value <Integrator.previous_value>` and `value <Integrator.value>` to either the value of the
            argument (if an argument was passed into reinitialize) or the current value of `initializer
            <Integrator.initializer>`.

            For specific types of Integrator functions, the reinitialize method may carry out other reinitialization
            steps.

        """

        reinitialization_values = []

        # no arguments were passed in -- use current values of initializer attributes
        if len(args) == 0 or args is None or all(arg is None for arg in args):
            for i in range(len(self.initializers)):
                initializer_name = self.initializers[i]
                reinitialization_values.append(self.get_current_function_param(initializer_name, execution_context))

        elif len(args) == len(self.initializers):
            for i in range(len(self.initializers)):
                initializer_name = self.initializers[i]
                if args[i] is None:
                    reinitialization_values.append(self.get_current_function_param(initializer_name, execution_context))
                else:
                    # Not sure if np.atleast_1d is necessary here:
                    reinitialization_values.append(np.atleast_1d(args[i]))

        # arguments were passed in, but there was a mistake in their specification -- raise error!
        else:
            stateful_attributes_string = self.stateful_attributes[0]
            if len(self.stateful_attributes) > 1:
                for i in range(1, len(self.stateful_attributes) - 1):
                    stateful_attributes_string += ", "
                    stateful_attributes_string += self.stateful_attributes[i]
                stateful_attributes_string += " and "
                stateful_attributes_string += self.stateful_attributes[len(self.stateful_attributes) - 1]

            initializers_string = self.initializers[0]
            if len(self.initializers) > 1:
                for i in range(1, len(self.initializers) - 1):
                    initializers_string += ", "
                    initializers_string += self.initializers[i]
                initializers_string += " and "
                initializers_string += self.initializers[len(self.initializers) - 1]

            raise FunctionError("Invalid arguments ({}) specified for {}. If arguments are specified for the "
                                "reinitialize method of {}, then a value must be passed to reinitialize each of its "
                                "stateful_attributes: {}, in that order. Alternatively, reinitialize may be called "
                                "without any arguments, in which case the current values of {}'s initializers: {}, will"
                                " be used to reinitialize their corresponding stateful_attributes."
                                .format(args,
                                        self.name,
                                        self.name,
                                        stateful_attributes_string,
                                        self.name,
                                        initializers_string))

        # rebuilding self.value rather than simply returning reinitialization_values in case any of the stateful
        # attrs are modified during assignment
        value = []
        for i in range(len(self.stateful_attributes)):
            setattr(self, self.stateful_attributes[i], reinitialization_values[i])
            getattr(self.parameters, self.stateful_attributes[i]).set(reinitialization_values[i], execution_context, override=True)
            value.append(getattr(self, self.stateful_attributes[i]))

        self.parameters.value.set(value, execution_context, override=True)
        return value

    def function(self, *args, **kwargs):
        raise FunctionError("Integrator is not meant to be called explicitly")

    @property
    def _dependent_components(self):
        return list(itertools.chain(
            super()._dependent_components,
            [self.noise] if isinstance(self.noise, DistributionFunction) else []
        ))


class SimpleIntegrator(Integrator):  # --------------------------------------------------------------------------------
    """
    SimpleIntegrator(                 \
        default_variable=None,  \
        rate=1.0,               \
        noise=0.0,              \
        initializer,            \
        params=None,            \
        owner=None,             \
        prefs=None,             \
        )

    .. _SimpleIntegrator:

    Integrate current value of `variable <SimpleIntegrator.variable>` with its prior value:

    `previous_value <SimpleIntegrator.previous_value>` + \
    `rate <SimpleIntegrator.rate>` *`variable <variable.SimpleIntegrator.variable>` + \
    `noise <SimpleIntegrator.noise>`;

    Arguments
    ---------

    default_variable : number, list or array : default ClassDefaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float, list or 1d array : default 1.0
        specifies the rate of integration.  If it is a list or array, it must be the same length as
        `variable <SimpleIntegrator.default_variable>` (see `rate <SimpleIntegrator.rate>` for details).

    noise : float, PsyNeuLink Function, list or 1d array : default 0.0
        specifies random value to be added in each call to `function <SimpleIntegrator.function>`. (see
        `noise <SimpleIntegrator.noise>` for details).

    initializer float, list or 1d array : default 0.0
        specifies starting value for integration.  If it is a list or array, it must be the same length as
        `default_variable <SimpleIntegrator.default_variable>` (see `initializer <SimpleIntegrator.initializer>`
        for details).

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
        current input value some portion of which (determined by `rate <SimpleIntegrator.rate>`) will be
        added to the prior value;  if it is an array, each element is independently integrated.

    rate : float or 1d array
        determines the rate of integration based on current and prior values. If it has a single element, it applies
        to all elements of `variable <SimpleIntegrator.variable>`;  if it has more than one element, each element
        applies to the corresponding element of `variable <SimpleIntegrator.variable>`.

    noise : float, function, list, or 1d array
        specifies random value to be added in each call to `function <SimpleIntegrator.function>`.

        If noise is a list or array, it must be the same length as `variable <SimpleIntegrator.default_variable>`.

        If noise is specified as a single float or function, while `variable <SimpleIntegrator.variable>` is a list or
        array, noise will be applied to each variable element. In the case of a noise function, this means that the
        function will be executed separately for each variable element.


        .. note::
            In order to generate random noise, we recommend selecting a probability distribution function (see
            `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value from
            its distribution on each execution. If noise is specified as a float or as a function with a fixed output,
            then the noise will simply be an offset that remains the same across all executions.

    initializer : float, 1d array or list
        determines the starting value for integration (i.e., the value to which
        `previous_value <SimpleIntegrator.previous_value>` is set.

        If initializer is a list or array, it must be the same length as `variable <SimpleIntegrator.default_variable>`.

    previous_value : 1d array : default ClassDefaults.variable
        stores previous value with which `variable <SimpleIntegrator.variable>` is integrated.

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

    componentName = SIMPLE_INTEGRATOR_FUNCTION

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        NOISE: None,
        RATE: None
    })

    multiplicative_param = RATE
    additive_param = OFFSET

    class Params(Integrator.Params):
        rate = Param(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        offset = Param(0.0, modulable=True, aliases=[ADDITIVE_PARAM])

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate: parameter_spec = 1.0,
                 noise=0.0,
                 offset=None,
                 initializer=None,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  initializer=initializer,
                                                  noise=noise,
                                                  offset=offset,
                                                  params=params)

        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)

        self.has_initializers = True

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """
        Return: `variable <Linear.slope>` combined with `previous_value <SimpleIntegrator.previous_value>`
        according to `previous_value <SimpleIntegrator.previous_value>` + `rate <SimpleIntegrator.rate>` *`variable
        <variable.SimpleIntegrator.variable>` + `noise <SimpleIntegrator.noise>`;

        Arguments
        ---------

        variable : number, list or array : default ClassDefaults.variable
           a single value or array of values to be integrated.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        updated value of integral : 2d array

        """

        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        rate = np.array(self.get_current_function_param(RATE, execution_id)).astype(float)

        offset = self.get_current_function_param(OFFSET, execution_id)
        if offset is None:
            offset = 0.0

        # execute noise if it is a function
        noise = self._try_execute_param(self.get_current_function_param(NOISE, execution_id), variable)
        previous_value = self.get_previous_value(execution_id)
        new_value = variable

        value = previous_value + (new_value * rate) + noise

        adjusted_value = value + offset

        # If this NOT an initialization run, update the old value
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if self.parameters.context.get(execution_id).initialization_status != ContextFlags.INITIALIZING:
            self.parameters.previous_value.set(adjusted_value, execution_id)

        return self.convert_output_type(adjusted_value)


class InteractiveActivation(Integrator):  # ----------------------------------------------------------------------------
    """
    InteractiveActivation(      \
        default_variable=None,  \
        decay=1.0,              \
        rest=0.0,               \
        max_val=1.0,            \
        min_val=-1.0,           \
        noise=0.0,              \
        initializer,            \
        params=None,            \
        owner=None,             \
        prefs=None,             \
        )

    .. _InteractiveActivation:

    Integrate current value of `variable <InteractiveActivation.variable>` toward an asymptotic maximum
    value for positive inputs and toward an asymptotic mininum value for negative inputs.

    Implements a generalized version of the interactive activation function used to update unit activites in
    `McClelland and Rumelhart (1981)
    <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.298.4480&rep=rep1&type=pdf>`_.


    `function <InteractiveActivation.function>` returns:

    .. math::
        previous\_value + (variable * distance\_from\_asymptote) - (decay * distance\_from\_rest) + noise

    where:

    .. math::
        if\ variable > 0,\ distance\_from\_asymptote = max\_val - previous\_value

    .. math::
        if\ variable < 0,\ distance\_from\_asymptote = previous\_value - min\_val

    .. math::
        if\ variable = 0,\ distance\_from\_asymptote = 0


    Arguments
    ---------

    default_variable : number, list or array : default ClassDefaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float, list or 1d array : default 1.0
        specifies the rate of at which activity increments toward either `max_val <InteractiveActivation.max_val>` or
        `min_val <InteractiveActivation.min_val>`, depending on the sign of `variable <InteractiveActivation.variable>`.
        If it is a list or array, it must be the same length as `variable <InteractiveActivation.default_variable>`;
        its value(s) must be in the interval [0,1].

    rest : float, list or 1d array : default 0.0
        specifies the initial value and one toward which value `decays <InteractiveActivation.decay>`.
        If it is a list or array, it must be the same length as `variable <InteractiveActivation.default_variable>`.
        COMMENT:
        its value(s) must be between `max_val <InteractiveActivation.max_val>` and `min_val
        <InteractiveActivation.min_val>`.
        COMMENT

    decay : float, list or 1d array : default 1.0
        specifies the rate of at which activity decays toward `rest <InteractiveActivation.rest>`.
        If it is a list or array, it must be the same length as `variable <InteractiveActivation.default_variable>`;
        its value(s) must be in the interval [0,1].

    max_val : float, list or 1d array : default 1.0
        specifies the maximum asymptotic value toward which integration occurs for positive values of `variable
        <InteractiveActivation.variable>`.  If it is a list or array, it must be the same length as `variable
        <InteractiveActivation.default_variable>`; all values must be greater than the corresponding values of
        `min_val <InteractiveActivation.min_val>` (see `max_val <InteractiveActivation.max_val>` for details).

    min_val : float, list or 1d array : default 1.0
        specifies the minimum asymptotic value toward which integration occurs for negative values of `variable
        <InteractiveActivation.variable>`.  If it is a list or array, it must be the same length as `variable
        <InteractiveActivation.default_variable>`; all values must be greater than the corresponding values of
        `max_val <InteractiveActivation.min_val>` (see `max_val <InteractiveActivation.min_val>` for details).

    noise : float, PsyNeuLink Function, list or 1d array : default 0.0
        specifies random value to be added in each call to `function <InteractiveActivation.function>`
        (see `noise <InteractiveActivation.noise>` for details).

    initializer float, list or 1d array : default 0.0
        specifies starting value for integration.  If it is a list or array, it must be the same length as
        `default_variable <InteractiveActivation.default_variable>`
        (see `initializer <InteractiveActivation.initializer>` for details).

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
        current input value some portion of which (determined by `rate <InteractiveActivation.rate>`) will be
        added to the prior value;  if it is an array, each element is independently integrated.

    rate : float or 1d array in interval [0,1]
        determines the rate at which activity increments toward either `max_val <InteractiveActivation.max_val>`
        (`variable <InteractiveActivation.variable>` is positive) or `min_val <InteractiveActivation.min_val>`
        (if `variable <InteractiveActivation.variable>` is negative).  If it has more than one element, each element
        applies to the corresponding element of `variable <InteractiveActivation.variable>`.

    rest : float, list or 1d array
        determines the initial value and one toward which value `decays <InteractiveActivation.decay>` (similar
        to *bias* in other IntegratorFunctions).

    decay : float, list or 1d array
        determines the rate of at which activity decays toward `rest <InteractiveActivation.rest>` (similary to
        *rate* in other IntegratorFuncgtions).  If it is a list or array, it must be the same length as `variable
        <InteractiveActivation.default_variable>`.

    max_val : float or 1d array
        determines the maximum asymptotic value toward which integration occurs for positive values of `variable
        <InteractiveActivation.variable>`.  If it has a single element, it applies to all elements of `variable
        <InteractiveActivation.variable>`;  if it has more than one element, each element
        applies to the corresponding element of `variable <InteractiveActivation.variable>`.

    min_val : float or 1d array
        determines the minimum asymptotic value toward which integration occurs for negative values of `variable
        <InteractiveActivation.variable>`.  If it has a single element, it applies to all elements of `variable
        <InteractiveActivation.variable>`;  if it has more than one element, each element
        applies to the corresponding element of `variable <InteractiveActivation.variable>`.

    noise : float, function, list, or 1d array
        specifies random value to be added in each call to `function <InteractiveActivation.function>`.

        If noise is a list or array, it must be the same length as `variable <InteractiveActivation.default_variable>`.

        If noise is specified as a single float or function, while `variable <InteractiveActivation.variable>` is a list or
        array, noise will be applied to each variable element. In the case of a noise function, this means that the
        function will be executed separately for each variable element.

        .. note::
            In order to generate random noise, we recommend selecting a probability distribution function (see
            `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value from
            its distribution on each execution. If noise is specified as a float or as a function with a fixed output,
            then the noise will simply be an offset that remains the same across all executions.

    initializer : float, 1d array or list
        determines the starting value for integration (i.e., the value to which
        `previous_value <InteractiveActivation.previous_value>` is set.

        If initializer is a list or array, it must be the same length as `variable <InteractiveActivation.default_variable>`.

    previous_value : 1d array : default ClassDefaults.variable
        stores previous value with which `variable <InteractiveActivation.variable>` is integrated.

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

    componentName = INTERACTIVE_ACTIVATION_INTEGRATOR_FUNCTION

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        RATE: None,
        DECAY: None,
        REST: None,
        NOISE: None,
    })

    multiplicative_param = RATE
    additive_param = OFFSET

    class Params(Integrator.Params):
        rate = Param(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        decay = Param(1.0, modulable=True)
        rest = Param(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        max_val = Param(1.0)
        min_val = Param(1.0)
        offset = Param(0.0)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate: parameter_spec = 1.0,
                 decay: parameter_spec = 0.0,
                 rest: parameter_spec = 0.0,
                 max_val: parameter_spec = 1.0,
                 min_val: parameter_spec = -1.0,
                 noise=0.0,
                 offset=None,
                 initializer=None,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None):

        if initializer is None:
            initializer = rest
        if default_variable is None:
            default_variable = initializer

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  decay=decay,
                                                  rest=rest,
                                                  max_val=max_val,
                                                  min_val=min_val,
                                                  initializer=initializer,
                                                  noise=noise,
                                                  offset=offset,
                                                  params=params)

        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)

        self.has_initializers = True

    def _validate_params(self, request_set, target_set=None, context=None):

        super()._validate_params(request_set=request_set, target_set=target_set,context=context)

        if RATE in request_set and request_set[RATE] is not None:
            rate = request_set[RATE]
            if np.isscalar(rate):
                rate = [rate]
            if not all(0.0 <= d <= 1.0 for d in rate):
                raise FunctionError("Value(s) specified for {} argument of {} ({}) must be in interval [0,1]".
                                    format(repr(RATE), self.__class__.__name__, rate))

        if DECAY in request_set and request_set[DECAY] is not None:
            decay = request_set[DECAY]
            if np.isscalar(decay):
                decay = [decay]
            if not all(0.0 <= d <= 1.0 for d in decay):
                raise FunctionError("Value(s) specified for {} argument of {} ({}) must be in interval [0,1]".
                                    format(repr(DECAY), self.__class__.__name__, decay))

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """

        Arguments
        ---------

        variable : number, list or array : default ClassDefaults.variable
           a single value or array of values to be integrated.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        updated value of integral : 2d array

        """

        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        rate = np.array(self.get_current_function_param(RATE, execution_id)).astype(float)
        decay = np.array(self.get_current_function_param(DECAY, execution_id)).astype(float)
        rest = np.array(self.get_current_function_param(REST, execution_id)).astype(float)
        # only works with "max_val". Keyword MAX_VAL = "MAX_VAL", not max_val
        max_val = np.array(self.get_current_function_param("max_val", execution_id)).astype(float)
        min_val = np.array(self.get_current_function_param("min_val", execution_id)).astype(float)

        # execute noise if it is a function
        noise = self._try_execute_param(self.get_current_function_param(NOISE, execution_id), variable)

        current_input = variable

        # FIX: ?CLEAN THIS UP BY SETTING initializer IN __init__ OR OTHER RELEVANT PLACE?
        if self.context.initialization_status == ContextFlags.INITIALIZING:
            if rest.ndim == 0 or len(rest)==1:
                # self.parameters.previous_value.set(np.full_like(current_input, rest), execution_id)
                self._initialize_previous_value(np.full_like(current_input, rest), execution_id)
            elif np.atleast_2d(rest).shape == current_input.shape:
                # self.parameters.previous_value.set(rest, execution_id)
                self._initialize_previous_value(rest, execution_id)
            else:
                raise FunctionError("The {} argument of {} ({}) must be an int or float, "
                                    "or a list or array of the same length as its variable ({})".
                                    format(repr(REST), self.__class__.__name__, rest, len(variable)))
        previous_value = self.get_previous_value(execution_id)

        current_input = np.atleast_2d(variable)
        prev_val = np.atleast_2d(previous_value)

        # dist_from_asymptote = np.zeros_like(current_input)
        dist_from_asymptote = []
        for i in range(len(current_input)):
            l_temp = []
            for j in range(len(current_input[i])):
                if current_input[i][j] > 0:
                    # FIX: 12/7/18 [JDC] FOLLOWING IS NOT GETTING ASSIGNED ON SECOND PASS THROUGH EXECUTE
                    d = max_val - prev_val[i][j]
                elif current_input[i][j] < 0:
                    d = prev_val[i][j] - min_val
                else:
                    d = 0
                # dist_from_asymptote[i][j] = d
                l_temp.append(d)
            # l.append(l_temp)
            dist_from_asymptote.append(l_temp)
            # TEST PRINT:
            if self.context.initialization_status != ContextFlags.INITIALIZING:
                print(d, dist_from_asymptote[i][j])
        # TEST PRINT:
        if self.context.initialization_status != ContextFlags.INITIALIZING:
            print(dist_from_asymptote)

        dist_from_rest = prev_val - rest

        new_value = previous_value + (rate * current_input * dist_from_asymptote) - (decay * dist_from_rest) + noise

        if self.parameters.context.get(execution_id).initialization_status != ContextFlags.INITIALIZING:
            self.parameters.previous_value.set(new_value, execution_id)

        return self.convert_output_type(new_value)


class ConstantIntegrator(Integrator):  # -------------------------------------------------------------------------------
    """
    ConstantIntegrator(                 \
        default_variable=None,          \
        rate=1.0,                       \
        noise=0.0,                      \
        scale: parameter_spec = 1.0,    \
        offset: parameter_spec = 0.0,   \
        initializer,                    \
        params=None,                    \
        owner=None,                     \
        prefs=None,                     \
        )

    .. _ConstantIntegrator:

    Integrates prior value by adding `rate <Integrator.rate>` and `noise <Integrator.noise>`. (Ignores
    `variable <Integrator.variable>`).

    `previous_value <ConstantIntegrator.previous_value>` + `rate <ConstantIntegrator.rate>` +
    `noise <ConstantIntegrator.noise>`

    Arguments
    ---------

    default_variable : number, list or array : default ClassDefaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float, list or 1d array : default 1.0
        specifies the rate of integration.  If it is a list or array, it must be the same length as
        `variable <ConstantIntegrator.default_variable>` (see `rate <ConstantIntegrator.rate>` for details).

    noise : float, PsyNeuLink Function, list or 1d array : default 0.0
        specifies random value to be added in each call to `function <ConstantIntegrator.function>`. (see
        `noise <ConstantIntegrator.noise>` for details).

    initializer float, list or 1d array : default 0.0
        specifies starting value for integration.  If it is a list or array, it must be the same length as
        `default_variable <ConstantIntegrator.default_variable>` (see `initializer <ConstantIntegrator.initializer>`
        for details).

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
        **Ignored** by the ConstantIntegrator function. Refer to LCAIntegrator or AdaptiveIntegrator for integrator
         functions that depend on both a prior value and a new value (variable).

    rate : float or 1d array
        determines the rate of integration.

        If it has a single element, that element is added to each element of
        `previous_value <ConstantIntegrator.previous_value>`.

        If it has more than one element, each element is added to the corresponding element of
        `previous_value <ConstantIntegrator.previous_value>`.

    noise : float, function, list, or 1d array
        specifies random value to be added in each call to `function <ConstantIntegrator.function>`.

        If noise is a list or array, it must be the same length as `variable <ConstantIntegrator.default_variable>`.

        If noise is specified as a single float or function, while `variable <ConstantIntegrator.variable>` is a list
        or array, noise will be applied to each variable element. In the case of a noise function, this means that
        the function will be executed separately for each variable element.

        .. note::
            In order to generate random noise, we recommend selecting a probability distribution function (see
            `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value from
            its distribution on each execution. If noise is specified as a float or as a function with a fixed output,
            then the noise will simply be an offset that remains the same across all executions.

    initializer : float, 1d array or list
        determines the starting value for integration (i.e., the value to which
        `previous_value <ConstantIntegrator.previous_value>` is set.

        If initializer is a list or array, it must be the same length as
        `variable <ConstantIntegrator.default_variable>`.

    previous_value : 1d array : default ClassDefaults.variable
        stores previous value to which `rate <ConstantIntegrator.rate>` and `noise <ConstantIntegrator.noise>` will be
        added.

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

    componentName = CONSTANT_INTEGRATOR_FUNCTION

    class Params(Integrator.Params):
        scale = Param(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        rate = Param(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        offset = Param(0.0, modulable=True)
        noise = Param(0.0, modulable=True)

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        NOISE: None,
        RATE: None,
        OFFSET: None,
        SCALE: None,
    })

    multiplicative_param = SCALE
    additive_param = RATE

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 # rate: parameter_spec = 1.0,
                 rate=0.0,
                 noise=0.0,
                 offset=0.0,
                 scale=1.0,
                 initializer=None,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  initializer=initializer,
                                                  noise=noise,
                                                  scale=scale,
                                                  offset=offset,
                                                  params=params)

        # Assign here as default, for use in initialization of function
        self.previous_value = initializer

        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)

        # Reassign to initializer in case default value was overridden

        self.has_initializers = True

    def _validate_rate(self, rate):
        # unlike other Integrators, variable does not need to match rate

        if isinstance(rate, list):
            rate = np.asarray(rate)

        rate_type_msg = 'The rate parameter of {0} must be a number or an array/list of at most 1d (you gave: {1})'
        if isinstance(rate, np.ndarray):
            # kmantel: current test_gating test depends on 2d rate
            #   this should be looked at but for now this restriction is removed
            # if rate.ndim > 1:
            #     raise FunctionError(rate_type_msg.format(self.name, rate))
            pass
        elif not isinstance(rate, numbers.Number):
            raise FunctionError(rate_type_msg.format(self.name, rate))

        if self._default_variable_flexibility is DefaultsFlexibility.FLEXIBLE:
            self.instance_defaults.variable = np.zeros_like(np.array(rate))
            self._instantiate_value()
            self._default_variable_flexibility = DefaultsFlexibility.INCREASE_DIMENSION

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """
        Return: the sum of `previous_value <ConstantIntegrator.previous_value>`, `rate <ConstantIntegrator.rate>`, and
        `noise <ConstantIntegrator.noise>`.

        Arguments
        ---------

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        updated value of integral : 2d array

        """
        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        rate = np.array(self.rate).astype(float)
        offset = self.get_current_function_param(OFFSET, execution_id)
        scale = self.get_current_function_param(SCALE, execution_id)
        noise = self._try_execute_param(self.noise, variable)

        previous_value = np.atleast_2d(self.get_previous_value(execution_id))

        value = previous_value + rate + noise

        adjusted_value = value * scale + offset

        # If this NOT an initialization run, update the old value
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if self.parameters.context.get(execution_id).initialization_status != ContextFlags.INITIALIZING:
            self.parameters.previous_value.set(adjusted_value, execution_id)
        return self.convert_output_type(adjusted_value)


class Buffer(Integrator):  # ------------------------------------------------------------------------------
    """
    Buffer(                     \
        default_variable=None,  \
        rate=None,              \
        noise=0.0,              \
        history=None,           \
        initializer,            \
        params=None,            \
        owner=None,             \
        prefs=None,             \
        )

    .. _Buffer:

    Appends `variable <Buffer.variable>` to the end of `previous_value <Buffer.previous_value>` (i.e., right-appends)
    which is a deque of previous inputs.  If specified, the values of the **rate** and **noise** arguments are
    applied to each item in the deque (including the newly added one) on each call, as follows:

        :math: item * `rate <Buffer.rate>` + `noise <Buffer.noise>`

    .. note::
       Because **rate** and **noise** are applied on every call, their effects are cumulative over calls.

    Arguments
    ---------

    default_variable : number, list or array : default ClassDefaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float : default None
        specifies a value applied to each item in the deque on each call.

    noise : float or Function : default 0.0
        specifies a random value added to each item in the deque on each call.

    history : int : default None
        specifies the maxlen of the deque, and hence `value <Buffer.value>`.

    initializer float, list or ndarray : default []
        specifies a starting value for the deque;  if none is specified, the deque is initialized with an
        empty list.

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
        current input value appended to the end of the deque.

    rate : float
        value added to each item of the deque on each call.

    noise : float or Function
        random value added to each item of the deque in each call.

        .. note::
            In order to generate random noise, a probability distribution function should be used (see
            `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value from
            its distribution on each execution. If noise is specified as a float or as a function with a fixed output,
            then the noise will simply be an offset that remains the same across all executions.

    history : int
        determines maxlen of the deque and the value returned by the `function <Buffer.function>`. If appending
        `variable <Buffer.variable>` to `previous_value <Buffer.previous_value>` exceeds history, the first item of
        `previous_value <Buffer.previous_value>` is deleted, and `variable <Buffer.variable>` is appended to it,
        so that `value <Buffer.previous_value>` maintains a constant length.  If history is not specified,
        the value returned continues to be extended indefinitely.

    initializer : float, list or ndarray
        the value assigned as the first item of the deque when the Function is initialized, or reinitialized
        if the **new_previous_value** argument is not specified in the call to `reinitialize
        <IntegratorFunction.reinitialize>`.

    previous_value : 1d array : default ClassDefaults.variable
        state of the deque prior to appending `variable <Buffer.variable>` in the current call.

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

    componentName = BUFFER_FUNCTION

    class Params(Integrator.Params):
        rate = Param(0.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        noise = Param(0.0, modulable=True)
        history = None

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        NOISE: None,
        RATE: None
    })

    multiplicative_param = RATE
    # no additive_param?

    @tc.typecheck
    def __init__(self,
                 default_variable=[],
                 # KAM 6/26/18 changed default param values because constructing a plain buffer function ("Buffer())
                 # was failing.
                 # For now, updated default_variable, noise, and Alternatively, we can change validation on
                 # default_variable=None,   # Changed to [] because None conflicts with initializer
                 # rate: parameter_spec=1.0,
                 # noise=0.0,
                 # rate: tc.optional(tc.any(int, float)) = None,         # Changed to 1.0 because None fails validation
                 # noise: tc.optional(tc.any(int, float, callable)) = None,    # Changed to 0.0 - None fails validation
                 rate: tc.optional(tc.any(int, float)) = 1.0,
                 noise: tc.optional(tc.any(int, float, callable)) = 0.0,
                 history: tc.optional(int) = None,
                 initializer=[],
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  initializer=initializer,
                                                  noise=noise,
                                                  history=history,
                                                  params=params)

        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)

        self.has_initializers = True

    def _initialize_previous_value(self, initializer, execution_context=None):
        initializer = initializer or []
        previous_value = deque(initializer, maxlen=self.history)

        self.parameters.previous_value.set(previous_value, execution_context, override=True)

        return previous_value

    def _instantiate_attributes_before_function(self, function=None, context=None):

        self.has_initializers = True

    def reinitialize(self, *args, execution_context=None):
        """

        Clears the `previous_value <Buffer.previous_value>` deque.

        If an argument is passed into reinitialize or if the `initializer <Buffer.initializer>` attribute contains a
        value besides [], then that value is used to start the new `previous_value <Buffer.previous_value>` deque.
        Otherwise, the new `previous_value <Buffer.previous_value>` deque starts out empty.

        `value <Buffer.value>` takes on the same value as  `previous_value <Buffer.previous_value>`.

        """

        # no arguments were passed in -- use current values of initializer attributes
        if len(args) == 0 or args is None:
            reinitialization_value = self.get_current_function_param("initializer", execution_context)

        elif len(args) == 1:
            reinitialization_value = args[0]

        # arguments were passed in, but there was a mistake in their specification -- raise error!
        else:
            raise FunctionError("Invalid arguments ({}) specified for {}. Either one value must be passed to "
                                "reinitialize its stateful attribute (previous_value), or reinitialize must be called "
                                "without any arguments, in which case the current initializer value, will be used to "
                                "reinitialize previous_value".format(args,
                                                                     self.name))

        if reinitialization_value is None or reinitialization_value == []:
            self.get_previous_value(execution_context).clear()
            value = deque([], maxlen=self.history)

        else:
            value = self._initialize_previous_value(reinitialization_value, execution_context=execution_context)

        self.parameters.value.set(value, execution_context, override=True)
        return value

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """
        Return: `previous_value <Buffer.previous_value>` appended with `variable
        <Buffer.variable>` * `rate <Buffer.rate>` + `noise <Buffer.noise>`;

        If the length of the result exceeds `history <Buffer.history>`, delete the first item.

        Arguments
        ---------

        variable : number, list or array : default ClassDefaults.variable
           a single value or array of values to be integrated.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        updated value of deque : deque

        """

        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        rate = np.array(self.get_current_function_param(RATE, execution_id)).astype(float)

        # execute noise if it is a function
        noise = self._try_execute_param(self.get_current_function_param(NOISE, execution_id), variable)

        # If this is an initialization run, leave deque empty (don't want to count it as an execution step);
        # Just return current input (for validation).
        if self.parameters.context.get(execution_id).initialization_status == ContextFlags.INITIALIZING:
            return variable

        previous_value = self.get_previous_value(execution_id)
        previous_value.append(variable)

        # Apply rate and/or noise if they are specified
        if rate != 1.0:
            previous_value *= rate
        if noise:
            previous_value += noise

        previous_value = deque(previous_value, maxlen=self.history)

        self.parameters.previous_value.set(previous_value, execution_id)
        return self.convert_output_type(previous_value)


class AdaptiveIntegrator(Integrator):  # -------------------------------------------------------------------------------
    """
    AdaptiveIntegrator(                 \
        default_variable=None,          \
        rate=1.0,                       \
        noise=0.0,                      \
        scale: parameter_spec = 1.0,    \
        offset: parameter_spec = 0.0,   \
        initializer,                    \
        params=None,                    \
        owner=None,                     \
        prefs=None,                     \
        )

    .. _AdaptiveIntegrator:

    Computes an exponentially weighted moving average.

    (1 - `rate <AdaptiveIntegrator.rate>`) * `previous_value <AdaptiveIntegrator.previous_value>` + `rate
    <AdaptiveIntegrator.rate>` * `variable <AdaptiveIntegrator.variable>` + `noise <AdaptiveIntegrator.noise>`


    Arguments
    ---------

    default_variable : number, list or array : default ClassDefaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float, list or 1d array : default 1.0
        specifies the smoothing factor of the EWMA.  If it is a list or array, it must be the same length as
        `variable <AdaptiveIntegrator.default_variable>` (see `rate <AdaptiveIntegrator.rate>` for details).

    noise : float, PsyNeuLink Function, list or 1d array : default 0.0
        specifies random value to be added in each call to `function <AdaptiveIntegrator.function>`. (see
        `noise <AdaptiveIntegrator.noise>` for details).

    initializer float, list or 1d array : default 0.0
        specifies starting value for integration.  If it is a list or array, it must be the same length as
        `default_variable <AdaptiveIntegrator.default_variable>` (see `initializer <AdaptiveIntegrator.initializer>`
        for details).

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
        current input value some portion of which (determined by `rate <AdaptiveIntegrator.rate>`) will be
        added to the prior value;  if it is an array, each element is independently integrated.

    rate : float or 1d array
        determines the smoothing factor of the EWMA. All rate elements must be between 0 and 1 (rate = 0 --> no change,
        `variable <AdaptiveAdaptiveIntegrator.variable>` is ignored; rate = 1 -->
        `previous_value <AdaptiveIntegrator.previous_value>` is ignored).

        If rate is a float, it is applied to all elements of `variable <AdaptiveAdaptiveIntegrator.variable>` (and
        `previous_value <AdaptiveIntegrator.previous_value>`); if it has more than one element, each element is applied
        to the corresponding element of `variable <AdaptiveAdaptiveIntegrator.variable>` (and
        `previous_value <AdaptiveIntegrator.previous_value>`).

    noise : float, function, list, or 1d array
        specifies random value to be added in each call to `function <AdaptiveIntegrator.function>`.

        If noise is a list or array, it must be the same length as `variable <AdaptiveIntegrator.default_variable>`.

        If noise is specified as a single float or function, while `variable <AdaptiveIntegrator.variable>` is a list
        or array, noise will be applied to each variable element. In the case of a noise function, this means that
        the function will be executed separately for each variable element.

        .. note::
            In order to generate random noise, we recommend selecting a probability distribution function
            (see `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value from
            its distribution on each execution. If noise is specified as a float or as a function with a fixed output, then
            the noise will simply be an offset that remains the same across all executions.

    initializer : float, 1d array or list
        determines the starting value for time-averaging (i.e., the value to which
        `previous_value <AdaptiveIntegrator.previous_value>` is originally set).

        If initializer is a list or array, it must be the same length as
        `variable <AdaptiveIntegrator.default_variable>`.

    previous_value : 1d array : default ClassDefaults.variable
        stores previous value with which `variable <AdaptiveIntegrator.variable>` is integrated.

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

    componentName = ADAPTIVE_INTEGRATOR_FUNCTION

    multiplicative_param = RATE
    additive_param = OFFSET

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        NOISE: None,
        RATE: None
    })

    class Params(Integrator.Params):
        rate = Param(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        offset = Param(0.0, modulable=True, aliases=[ADDITIVE_PARAM])

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate: parameter_spec = 1.0,
                 noise=0.0,
                 offset=0.0,
                 initializer=None,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  initializer=initializer,
                                                  noise=noise,
                                                  offset=offset,
                                                  params=params)

        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)

        self.has_initializers = True

    def _validate_params(self, request_set, target_set=None, context=None):

        # Handle list or array for rate specification
        if RATE in request_set:
            rate = request_set[RATE]
            if isinstance(rate, (list, np.ndarray)):
                if len(rate) != 1 and len(rate) != np.array(self.instance_defaults.variable).size:
                    # If the variable was not specified, then reformat it to match rate specification
                    #    and assign ClassDefaults.variable accordingly
                    # Note: this situation can arise when the rate is parametrized (e.g., as an array) in the
                    #       AdaptiveIntegrator's constructor, where that is used as a specification for a function
                    #       parameter (e.g., for an IntegratorMechanism), whereas the input is specified as part of the
                    #       object to which the function parameter belongs (e.g., the IntegratorMechanism);
                    #       in that case, the Integrator gets instantiated using its ClassDefaults.variable ([[0]])
                    #       before the object itself, thus does not see the array specification for the input.
                    if self._default_variable_flexibility is DefaultsFlexibility.FLEXIBLE:
                        self._instantiate_defaults(variable=np.zeros_like(np.array(rate)), context=context)
                        if self.verbosePref:
                            warnings.warn(
                                "The length ({}) of the array specified for the rate parameter ({}) of {} "
                                "must match the length ({}) of the default input ({});  "
                                "the default input has been updated to match".format(
                                    len(rate),
                                    rate,
                                    self.name,
                                    np.array(self.instance_defaults.variable).size
                                ),
                                self.instance_defaults.variable
                            )
                    else:
                        raise FunctionError(
                            "The length ({}) of the array specified for the rate parameter ({}) of {} "
                            "must match the length ({}) of the default input ({})".format(
                                len(rate),
                                rate,
                                self.name,
                                np.array(self.instance_defaults.variable).size,
                                self.instance_defaults.variable,
                            )
                        )
                        # OLD:
                        # self.paramClassDefaults[RATE] = np.zeros_like(np.array(rate))

                        # KAM changed 5/15 b/c paramClassDefaults were being updated and *requiring* future integrator
                        # function to have a rate parameter of type ndarray/list

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if RATE in target_set:
            # cannot use _validate_rate here because it assumes it's being run after instantiation of the object
            rate_value_msg = "The rate parameter ({}) (or all of its elements) of {} must be between 0.0 and 1.0 because it is an AdaptiveIntegrator"
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
                target_set[NOISE] = noise._execute
            self._validate_noise(target_set[NOISE])
            # if INITIALIZER in target_set:
            #     self._validate_initializer(target_set[INITIALIZER])

    def _validate_rate(self, rate):
        super()._validate_rate(rate)

        if isinstance(rate, list):
            rate = np.asarray(rate)

        rate_value_msg = "The rate parameter ({}) (or all of its elements) of {} must be between 0.0 and 1.0 because it is an AdaptiveIntegrator"
        if isinstance(rate, np.ndarray) and rate.ndim > 0:
            for r in rate:
                if r < 0.0 or r > 1.0:
                    raise FunctionError(rate_value_msg.format(rate, self.name))
        else:
            if rate < 0.0 or rate > 1.0:
                raise FunctionError(rate_value_msg.format(rate, self.name))

    def _get_context_struct_type(self, ctx):
        return ctx.get_output_struct_type(self)

    def _get_context_initializer(self, execution_id):
        data = np.asfarray(self.parameters.previous_value.get(execution_id)).flatten().tolist()
        if self.instance_defaults.value.ndim > 1:
            return (tuple(data),)
        return tuple(data)

    def __gen_llvm_integrate(self, builder, index, ctx, vi, vo, params, state):
        rate_p, builder = ctx.get_param_ptr(self, builder, params, RATE)
        offset_p, builder = ctx.get_param_ptr(self, builder, params, OFFSET)

        rate = pnlvm.helpers.load_extract_scalar_array_one(builder, rate_p)
        offset = pnlvm.helpers.load_extract_scalar_array_one(builder, offset_p)

        noise_p, builder = ctx.get_param_ptr(self, builder, params, NOISE)
        if isinstance(noise_p.type.pointee, ir.ArrayType) and noise_p.type.pointee.count > 1:
            noise_p = builder.gep(noise_p, [ctx.int32_ty(0), index])

        noise = pnlvm.helpers.load_extract_scalar_array_one(builder, noise_p)

        # FIXME: Standalone function produces 2d array value
        if isinstance(state.type.pointee.element, ir.ArrayType):
            assert state.type.pointee.count == 1
            prev_ptr = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(0), index])
        else:
            prev_ptr = builder.gep(state, [ctx.int32_ty(0), index])
        prev_val = builder.load(prev_ptr)

        vi_ptr = builder.gep(vi, [ctx.int32_ty(0), index])
        vi_val = builder.load(vi_ptr)

        rev_rate = builder.fsub(ctx.float_ty(1), rate)
        old_val = builder.fmul(prev_val, rev_rate)
        new_val = builder.fmul(vi_val, rate)

        ret = builder.fadd(old_val, new_val)
        ret = builder.fadd(ret, noise)
        res = builder.fadd(ret, offset)

        # FIXME: Standalone function produces 2d array value
        if isinstance(vo.type.pointee.element, ir.ArrayType):
            assert state.type.pointee.count == 1
            vo_ptr = builder.gep(vo, [ctx.int32_ty(0), ctx.int32_ty(0), index])
        else:
            vo_ptr = builder.gep(vo, [ctx.int32_ty(0), index])
        builder.store(res, vo_ptr)
        builder.store(res, prev_ptr)

    def _gen_llvm_function_body(self, ctx, builder, params, context, arg_in, arg_out):
        # Eliminate one dimension for 2d variable
        if self.instance_defaults.variable.ndim > 1:
            assert self.instance_defaults.variable.shape[0] == 1
            arg_in = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(0)])
            arg_out = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(0)])
            context = builder.gep(context, [ctx.int32_ty(0), ctx.int32_ty(0)])

        kwargs = {"ctx": ctx, "vi": arg_in, "vo": arg_out, "params": params, "state": context}
        inner = functools.partial(self.__gen_llvm_integrate, **kwargs)
        with helpers.array_ptr_loop(builder, arg_in, "integrate") as args:
            inner(*args)

        return builder

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """
        Return: some fraction of `variable <AdaptiveIntegrator.variable>` combined with some fraction of `previous_value
        <AdaptiveIntegrator.previous_value>`.

        Arguments
        ---------

        variable : number, list or array : default ClassDefaults.variable
           a single value or array of values to be integrated.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        updated value of integral : 2d array

        """
        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        rate = np.array(self.get_current_function_param(RATE, execution_id)).astype(float)
        offset = self.get_current_function_param(OFFSET, execution_id)
        # execute noise if it is a function
        noise = self._try_execute_param(self.get_current_function_param(NOISE, execution_id), variable)

        previous_value = np.atleast_2d(self.get_previous_value(execution_id))

        value = (1 - rate) * previous_value + rate * variable + noise
        adjusted_value = value + offset

        # If this NOT an initialization run, update the old value
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if self.parameters.context.get(execution_id).initialization_status != ContextFlags.INITIALIZING:
            self.parameters.previous_value.set(adjusted_value, execution_id)

        return self.convert_output_type(adjusted_value)


class DriftDiffusionIntegrator(Integrator):  # -------------------------------------------------------------------------
    """
    DriftDiffusionIntegrator(           \
        default_variable=None,          \
        rate=1.0,                       \
        noise=0.0,                      \
        scale= 1.0,                     \
        offset= 0.0,                    \
        time_step_size=1.0,             \
        t0=0.0,                         \
        decay=0.0,                      \
        threshold=1.0                   \
        initializer,                    \
        params=None,                    \
        owner=None,                     \
        prefs=None,                     \
        )

    .. _DriftDiffusionIntegrator:

    Accumulates evidence over time based on a stimulus, rate, previous position, and noise. Stops accumulating at a
    threshold.

    Arguments
    ---------

    default_variable : number, list or array : default ClassDefaults.variable
        specifies the stimulus component of drift rate -- the drift rate is the product of variable and rate

    rate : float, list or 1d array : default 1.0
        specifies the attentional component of drift rate -- the drift rate is the product of variable and rate

    noise : float, PsyNeuLink Function, list or 1d array : default 0.0
        scales the random value to be added in each call to `function <DriftDiffusionIntegrator.function>`. (see
        `noise <DriftDiffusionIntegrator.noise>` for details).

    time_step_size : float : default 0.0
        determines the timing precision of the integration process (see `time_step_size
        <DriftDiffusionIntegrator.time_step_size>` for details.

    t0 : float
        determines the start time of the integration process and is used to compute the RESPONSE_TIME output state of
        the DDM Mechanism.

    initializer : float, list or 1d array : default 0.0
        specifies starting value for integration.  If it is a list or array, it must be the same length as
        `default_variable <DriftDiffusionIntegrator.default_variable>` (see `initializer
        <DriftDiffusionIntegrator.initializer>` for details).

    threshold : float : default 0.0
        specifies the threshold (boundaries) of the drift diffusion process (i.e., at which the
        integration process is assumed to terminate).

        Once the magnitude of the decision variable has exceeded the threshold, the function will simply return the
        threshold magnitude (with the appropriate sign) for that execution and any future executions.

        If the function is in a `DDM mechanism <DDM>`, crossing the threshold will also cause the return value of `is_finished`
        from False to True. This value may be important for the `Scheduler <Scheduler>` when using
         `Conditions <Condition>` such as `WhenFinished <WhenFinished>`.

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
        current input value, which represents the stimulus component of drift.

    rate : float or 1d array
        specifies the attentional component of drift rate -- the drift rate is the product of variable and rate

    noise : float, function, list, or 1d array
        scales the random value to be added in each call to `function <DriftDiffusionIntegrator.function> according to
        the standard DDM probability distribution.

        On each call to `function <DriftDiffusionIntegrator.function>, :math:`\\sqrt{time\\_step\\_size \\cdot noise}
        \\cdot Sample\\,From\\,Normal\\,distribution` is added to the accumulated evidence.

        Noise must be specified as a float (or list or array of floats).

    time_step_size : float
        determines the timing precision of the integration process and is used to scale the `noise
        <DriftDiffusionIntegrator.noise>` parameter according to the standard DDM probability distribution.

    t0 : float
        determines the start time of the integration process and is used to compute the RESPONSE_TIME output state of
        the DDM Mechanism.

    initializer : float, 1d array or list
        determines the starting value for integration (i.e., the value to which
        `previous_value <DriftDiffusionIntegrator.previous_value>` is set.

        If initializer is a list or array, it must be the same length as
        `variable <DriftDiffusionIntegrator.default_variable>`.

    previous_time : float
        stores previous time at which the function was executed and accumulates with each execution according to
        `time_step_size <DriftDiffusionIntegrator.default_time_step_size>`.

    previous_value : 1d array : default ClassDefaults.variable
        stores previous value with which `variable <DriftDiffusionIntegrator.variable>` is integrated.

    threshold : float : default 0.0
        when used properly determines the threshold (boundaries) of the drift diffusion process (i.e., at which the
        integration process is assumed to terminate).

        If the system is assembled as follows, then the DriftDiffusionIntegrator function stops accumulating when its
        value reaches +threshold or -threshold

            (1) the function is used in the `DDM mechanism <DDM>`

            (2) the mechanism is part of a `System <System>` with a `Scheduler <Scheduler>` which applies the
            `WhenFinished <WhenFinished>` `Condition <Condition>` to the mechanism

        Otherwise, `threshold <DriftDiffusionIntegrator.threshold>` does not influence the function at all.

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

    componentName = DRIFT_DIFFUSION_INTEGRATOR_FUNCTION

    multiplicative_param = RATE
    additive_param = OFFSET

    class Params(Integrator.Params):
        rate = Param(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        offset = Param(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        threshold = Param(100.0, modulable=True)
        time_step_size = Param(1.0, modulable=True)
        previous_time = None
        t0 = 0.0

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        NOISE: None,
        RATE: None
    })

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate: parameter_spec = 1.0,
                 noise=0.0,
                 offset: parameter_spec = 0.0,
                 threshold=100.0,
                 time_step_size=1.0,
                 t0=0.0,
                 initializer=None,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None):

        if not hasattr(self, "initializers"):
            self.initializers = ["initializer", "t0"]

        if not hasattr(self, "stateful_attributes"):
            self.stateful_attributes = ["previous_value", "previous_time"]

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  time_step_size=time_step_size,
                                                  t0=t0,
                                                  initializer=initializer,
                                                  threshold=threshold,
                                                  noise=noise,
                                                  offset=offset,
                                                  params=params)

        # Assign here as default, for use in initialization of function
        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)

        self.has_initializers = True

    @property
    def output_type(self):
        return self._output_type

    @output_type.setter
    def output_type(self, value):
        # disabled because it happens during normal execution, may be confusing
        # warnings.warn('output_type conversion disabled for {0}'.format(self.__class__.__name__))
        self._output_type = None

    def _validate_noise(self, noise):
        if not isinstance(noise, float):
            raise FunctionError(
                "Invalid noise parameter for {}. DriftDiffusionIntegrator requires noise parameter to be a float. Noise"
                " parameter is used to construct the standard DDM noise distribution".format(self.name))

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """
        Return: One time step of evidence accumulation according to the Drift Diffusion Model

        ..  math::

            previous\\_value + rate \\cdot variable \\cdot time\\_step\\_size + \\sqrt{time\\_step\\_size \\cdot noise}
            \\cdot Sample\\,from\\,Normal\\,Distribution

        Arguments
        ---------

        variable : number, list or array : default ClassDefaults.variable
            specifies the stimulus component of drift rate -- the drift rate is the product of variable and rate

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        updated value of integral : 2d array

        """
        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        rate = np.array(self.get_current_function_param(RATE, execution_id)).astype(float)
        offset = self.get_current_function_param(OFFSET, execution_id)
        noise = self.get_current_function_param(NOISE, execution_id)
        threshold = self.get_current_function_param(THRESHOLD, execution_id)
        time_step_size = self.get_current_function_param(TIME_STEP_SIZE, execution_id)

        previous_value = np.atleast_2d(self.get_previous_value(execution_id))

        value = previous_value + rate * variable * time_step_size \
                + np.sqrt(time_step_size * noise) * np.random.normal()

        if np.all(abs(value) < threshold):
            adjusted_value = value + offset
        elif np.all(value >= threshold):
            adjusted_value = np.atleast_2d(threshold)
        elif np.all(value <= -threshold):
            adjusted_value = np.atleast_2d(-threshold)

        # If this NOT an initialization run, update the old value and time
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        previous_time = self.get_current_function_param('previous_time', execution_id)
        if self.parameters.context.get(execution_id).initialization_status != ContextFlags.INITIALIZING:
            previous_value = adjusted_value
            previous_time = previous_time + time_step_size
            if not np.isscalar(variable):
                previous_time = np.broadcast_to(
                    previous_time,
                    variable.shape
                ).copy()

            self.parameters.previous_time.set(previous_time, execution_id)

        self.parameters.previous_value.set(previous_value, execution_id)
        return previous_value, previous_time


class OrnsteinUhlenbeckIntegrator(Integrator):  # ----------------------------------------------------------------------
    """
    OrnsteinUhlenbeckIntegrator(        \
        default_variable=None,          \
        rate=1.0,                       \
        noise=0.0,                      \
        offset= 0.0,                    \
        time_step_size=1.0,             \
        t0=0.0,                         \
        decay=1.0,                      \
        initializer=0.0,                \
        params=None,                    \
        owner=None,                     \
        prefs=None,                     \
        )

    .. _OrnsteinUhlenbeckIntegrator:

    Accumulate evidence overtime based on a stimulus, noise, decay, and previous position.

    Arguments
    ---------

    default_variable : number, list or array : default ClassDefaults.variable
        specifies a template for  the stimulus component of drift rate -- the drift rate is the product of variable and
        rate

    rate : float, list or 1d array : default 1.0
        specifies  the attentional component of drift rate -- the drift rate is the product of variable and rate

    noise : float, PsyNeuLink Function, list or 1d array : default 0.0
        scales random value to be added in each call to `function <OrnsteinUhlenbeckIntegrator.function>`. (see
        `noise <OrnsteinUhlenbeckIntegrator.noise>` for details).

    time_step_size : float : default 0.0
        determines the timing precision of the integration process (see `time_step_size
        <OrnsteinUhlenbeckIntegrator.time_step_size>` for details.

    t0 : float : default 0.0
        represents the starting time of the model and is used to compute
        `previous_time <OrnsteinUhlenbeckIntegrator.previous_time>`

    initializer float, list or 1d array : default 0.0
        specifies starting value for integration.  If it is a list or array, it must be the same length as
        `default_variable <OrnsteinUhlenbeckIntegrator.default_variable>` (see `initializer
        <OrnsteinUhlenbeckIntegrator.initializer>` for details).

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
        represents the stimulus component of drift. The product of
        `variable <OrnsteinUhlenbeckIntegrator.variable>` and `rate <OrnsteinUhlenbeckIntegrator.rate>` is multiplied
        by `time_step_size <OrnsteinUhlenbeckIntegrator.time_step_size>` to model the accumulation of evidence during
        one step.

    rate : float or 1d array
        represents the attentional component of drift. The product of `rate <OrnsteinUhlenbeckIntegrator.rate>` and
        `variable <OrnsteinUhlenbeckIntegrator.variable>` is multiplied by
        `time_step_size <OrnsteinUhlenbeckIntegrator.time_step_size>` to model the accumulation of evidence during
        one step.

    noise : float, function, list, or 1d array
        scales the random value to be added in each call to `function <OrnsteinUhlenbeckIntegrator.function>`

        Noise must be specified as a float (or list or array of floats) because this
        value will be used to construct the standard DDM probability distribution.

    time_step_size : float
        determines the timing precision of the integration process and is used to scale the `noise
        <OrnsteinUhlenbeckIntegrator.noise>` parameter appropriately.

    initializer : float, 1d array or list
        determines the starting value for integration (i.e., the value to which
        `previous_value <OrnsteinUhlenbeckIntegrator.previous_value>` is originally set.)

        If initializer is a list or array, it must be the same length as `variable
        <OrnsteinUhlenbeckIntegrator.default_variable>`.

    previous_value : 1d array : default ClassDefaults.variable
        stores previous value with which `variable <OrnsteinUhlenbeckIntegrator.variable>` is integrated.

    previous_time : float
        stores previous time at which the function was executed and accumulates with each execution according to
        `time_step_size <OrnsteinUhlenbeckIntegrator.default_time_step_size>`.

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

    componentName = ORNSTEIN_UHLENBECK_INTEGRATOR_FUNCTION

    multiplicative_param = RATE
    additive_param = OFFSET

    class Params(Integrator.Params):
        rate = Param(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        offset = Param(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        time_step_size = Param(1.0, modulable=True)
        decay = Param(1.0, modulable=True)
        t0 = 0.0
        previous_time = 0.0

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        NOISE: None,
        RATE: None
    })

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate: parameter_spec = 1.0,
                 noise=0.0,
                 offset: parameter_spec = 0.0,
                 time_step_size=1.0,
                 t0=0.0,
                 decay=1.0,
                 initializer=None,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None):

        if not hasattr(self, "initializers"):
            self.initializers = ["initializer", "t0"]

        if not hasattr(self, "stateful_attributes"):
            self.stateful_attributes = ["previous_value", "previous_time"]

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  time_step_size=time_step_size,
                                                  decay=decay,
                                                  initializer=initializer,
                                                  t0=t0,
                                                  noise=noise,
                                                  offset=offset,
                                                  params=params)

        # Assign here as default, for use in initialization of function
        self.previous_value = initializer
        self.previous_time = t0

        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)

        self.previous_time = self.t0
        self.has_initializers = True

    def _validate_noise(self, noise):
        if not isinstance(noise, float):
            raise FunctionError(
                "Invalid noise parameter for {}. OrnsteinUhlenbeckIntegrator requires noise parameter to be a float. "
                "Noise parameter is used to construct the standard DDM noise distribution".format(self.name))

    @property
    def output_type(self):
        return self._output_type

    @output_type.setter
    def output_type(self, value):
        # disabled because it happens during normal execution, may be confusing
        # warnings.warn('output_type conversion disabled for {0}'.format(self.__class__.__name__))
        self._output_type = None

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """
        Return: One time step of evidence accumulation according to the Ornstein Uhlenbeck Model

        previous_value + decay * (previous_value -  rate * variable) + :math:`\\sqrt{time_step_size * noise}` * random
        sample from Normal distribution


        Arguments
        ---------

        variable : number, list or array : default ClassDefaults.variable
           the stimulus component of drift rate in the Drift Diffusion Model.


        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        updated value of integral : 2d array

        """

        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)
        rate = np.array(self.get_current_function_param(RATE, execution_id)).astype(float)
        offset = self.get_current_function_param(OFFSET, execution_id)
        time_step_size = self.get_current_function_param(TIME_STEP_SIZE, execution_id)
        decay = self.get_current_function_param(DECAY, execution_id)
        noise = self.get_current_function_param(NOISE, execution_id)

        previous_value = np.atleast_2d(self.get_previous_value(execution_id))

        # dx = (lambda*x + A)dt + c*dW
        value = previous_value + (decay * previous_value - rate * variable) * time_step_size + np.sqrt(
            time_step_size * noise) * np.random.normal()

        # If this NOT an initialization run, update the old value and time
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        adjusted_value = value + offset

        previous_time = self.get_current_function_param('previous_time', execution_id)
        if self.parameters.context.get(execution_id).initialization_status != ContextFlags.INITIALIZING:
            previous_value = adjusted_value
            previous_time = previous_time + time_step_size
            if not np.isscalar(variable):
                previous_time = np.broadcast_to(
                    previous_time,
                    variable.shape
                ).copy()
            self.parameters.previous_time.set(previous_time, execution_id)

        self.parameters.previous_value.set(previous_value, execution_id)
        return previous_value, previous_time


class FHNIntegrator(Integrator):  # --------------------------------------------------------------------------------
    """
    FHNIntegrator(                      \
        default_variable=1.0,           \
        scale: parameter_spec = 1.0,    \
        offset: parameter_spec = 0.0,   \
        initial_w=0.0,                  \
        initial_v=0.0,                  \
        time_step_size=0.05,          \
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

    .. _FHNIntegrator:

    The FHN Integrator function in PsyNeuLink implements the Fitzhugh-Nagumo model using a choice of Euler or 4th Order
    Runge-Kutta numerical integration.

    In order to support several common representations of the model, the FHNIntegrator includes many parameters, some of
    which would not be sensible to use in combination. The equations of the Fitzhugh-Nagumo model are expressed below in
    terms of all of the parameters exposed in PsyNeuLink:

    **Fast, Excitatory Variable:**


    .. math::

        \\frac{dv}{dt} = \\frac{a_v v^{3} + b_v v^{2} (1+threshold) - c_v v\\, threshold + d_v + e_v\\, previous_w + f_v\\, variable)}{time\\, constant_v}


    **Slow, Inactivating Variable:**


    .. math::

        \\frac{dw}{dt} = \\frac{a_w\\, mode\\, previous_v + b_w w + c_w +
                    uncorrelated\\,activity\\,(1-mode)}{time\\, constant_w}

    *The three formulations that the FHNIntegrator was designed to allow are:*

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

        *How to implement this model in PsyNeuLink:*

            In PsyNeuLink, the default parameter values of the FHNIntegrator function implement the above equations.


    (2) **Modified FHN Model**

        **Fast, Excitatory Variable:**

        .. math::

            \\frac{dv}{dt} = v(a-v)(v-1) - w + I_{ext}

        **Slow, Inactivating Variable:**

        .. math::

            \\frac{dw}{dt} = bv - cw

        `Mahbub Khan (2013) <http://pcwww.liv.ac.uk/~bnvasiev/Past%20students/Mahbub_549.pdf>`_ provides a nice summary
        of why this formulation is useful.

        *How to implement this model in PsyNeuLink:*

            In order to implement the modified FHN model, the following PsyNeuLink parameter values must be set in the
            equation for :math:`\\frac{dv}{dt}`:

            +-----------------+-----+-----+-----+-----+-----+-----+---------------+
            |**PNL Parameter**| a_v | b_v | c_v | d_v | e_v | f_v |time_constant_v|
            +-----------------+-----+-----+-----+-----+-----+-----+---------------+
            |**Value**        |-1.0 |1.0  |1.0  |0.0  |-1.0 |1.0  |1.0            |
            +-----------------+-----+-----+-----+-----+-----+-----+---------------+

            When the parameters above are set to the listed values, the PsyNeuLink equation for :math:`\\frac{dv}{dt}`
            reduces to the Modified FHN formulation, and the remaining parameters in the :math:`\\frac{dv}{dt}` equation
            correspond as follows:

            +--------------------------+---------------------------------------+---------------------------------------+
            |**PNL Parameter**         |`threshold <FHNIntegrator.threshold>`  |`variable <FHNIntegrator.variable>`    |
            +--------------------------+---------------------------------------+---------------------------------------+
            |**Modified FHN Parameter**|a                                      |:math:`I_{ext}`                        |
            +--------------------------+---------------------------------------+---------------------------------------+

            In order to implement the modified FHN model, the following PsyNeuLink parameter values must be set in the
            equation for :math:`\\frac{dw}{dt}`:

            +-----------------+-----+------+---------------+----------------------+
            |**PNL Parameter**|c_w  | mode |time_constant_w|uncorrelated_activity |
            +-----------------+-----+------+---------------+----------------------+
            |**Value**        | 0.0 | 1.0  |1.0            | 0.0                  |
            +-----------------+-----+------+---------------+----------------------+

            When the parameters above are set to the listed values, the PsyNeuLink equation for :math:`\\frac{dw}{dt}`
            reduces to the Modified FHN formulation, and the remaining parameters in the :math:`\\frac{dw}{dt}` equation
            correspond as follows:

            +--------------------------+---------------------------------------+---------------------------------------+
            |**PNL Parameter**         |`a_w <FHNIntegrator.a_w>`              |*NEGATIVE* `b_w <FHNIntegrator.b_w>`   |
            +--------------------------+---------------------------------------+---------------------------------------+
            |**Modified FHN Parameter**|b                                      |c                                      |
            +--------------------------+---------------------------------------+---------------------------------------+

    (3) **Modified FHN Model as implemented in** `Gilzenrat (2002) <http://www.sciencedirect.com/science/article/pii/S0893608002000552?via%3Dihub>`_

        **Fast, Excitatory Variable:**

        [Eq. (6) in `Gilzenrat (2002) <http://www.sciencedirect.com/science/article/pii/S0893608002000552?via%3Dihub>`_ ]

        .. math::

            \\tau_v \\frac{dv}{dt} = v(a-v)(v-1) - u + w_{vX_1}\\, f(X_1)

        **Slow, Inactivating Variable:**

        [Eq. (7) & Eq. (8) in `Gilzenrat (2002) <http://www.sciencedirect.com/science/article/pii/S0893608002000552?via%3Dihub>`_ ]

        .. math::

            \\tau_u \\frac{du}{dt} = Cv + (1-C)\\, d - u

        *How to implement this model in PsyNeuLink:*

            In order to implement the Gilzenrat 2002 model, the following PsyNeuLink parameter values must be set in the
            equation for :math:`\\frac{dv}{dt}`:

            +-----------------+-----+-----+-----+-----+-----+
            |**PNL Parameter**| a_v | b_v | c_v | d_v | e_v |
            +-----------------+-----+-----+-----+-----+-----+
            |**Value**        |-1.0 |1.0  |1.0  |0.0  |-1.0 |
            +-----------------+-----+-----+-----+-----+-----+

            When the parameters above are set to the listed values, the PsyNeuLink equation for :math:`\\frac{dv}{dt}`
            reduces to the Gilzenrat formulation, and the remaining parameters in the :math:`\\frac{dv}{dt}` equation
            correspond as follows:

            +-----------------------+-------------------------------------+-----------------------------------+-------------------------+----------------------------------------------------+
            |**PNL Parameter**      |`threshold <FHNIntegrator.threshold>`|`variable <FHNIntegrator.variable>`|`f_v <FHNIntegrator.f_v>`|`time_constant_v <FHNIntegrator.time_constant_v>`   |
            +-----------------------+-------------------------------------+-----------------------------------+-------------------------+----------------------------------------------------+
            |**Gilzenrat Parameter**|a                                    |:math:`f(X_1)`                     |:math:`w_{vX_1}`         |:math:`T_{v}`                                       |
            +-----------------------+-------------------------------------+-----------------------------------+-------------------------+----------------------------------------------------+

            In order to implement the Gilzenrat 2002 model, the following PsyNeuLink parameter values must be set in the
            equation for :math:`\\frac{dw}{dt}`:

            +-----------------+-----+-----+-----+
            |**PNL Parameter**| a_w | b_w | c_w |
            +-----------------+-----+-----+-----+
            |**Value**        | 1.0 |-1.0 |0.0  |
            +-----------------+-----+-----+-----+

            When the parameters above are set to the listed values, the PsyNeuLink equation for :math:`\\frac{dw}{dt}`
            reduces to the Gilzenrat formulation, and the remaining parameters in the :math:`\\frac{dw}{dt}` equation
            correspond as follows:

            +--------------------------+---------------------------------------+-------------------------------------------------------------+----------------------------------------------------+
            |**PNL Parameter**         |`mode <FHNIntegrator.mode>`            |`uncorrelated_activity <FHNIntegrator.uncorrelated_activity>`|`time_constant_v <FHNIntegrator.time_constant_w>`   |
            +--------------------------+---------------------------------------+-------------------------------------------------------------+----------------------------------------------------+
            |**Gilzenrat Parameter**   |C                                      |d                                                            |:math:`T_{u}`                                       |
            +--------------------------+---------------------------------------+-------------------------------------------------------------+----------------------------------------------------+

    Arguments
    ---------

    default_variable : number, list or array : default ClassDefaults.variable
        specifies a template for the external stimulus

    initial_w : float, list or 1d array : default 0.0
        specifies starting value for integration of dw/dt.  If it is a list or array, it must be the same length as
        `default_variable <FHNIntegrator.default_variable>`

    initial_v : float, list or 1d array : default 0.0
        specifies starting value for integration of dv/dt.  If it is a list or array, it must be the same length as
        `default_variable <FHNIntegrator.default_variable>`

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
        coefficient on the external stimulus (`variable <FHNIntegrator.variable>`) term in the dv/dt equation

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
        External stimulus

    previous_v : 1d array : default ClassDefaults.variable
        stores accumulated value of v during integration

    previous_w : 1d array : default ClassDefaults.variable
        stores accumulated value of w during integration

    previous_t : float
        stores accumulated value of time, which is incremented by time_step_size on each execution of the function

    owner : Component
        `component <Component>` to which the Function has been assigned.

    initial_w : float, list or 1d array : default 0.0
        specifies starting value for integration of dw/dt.  If it is a list or array, it must be the same length as
        `default_variable <FHNIntegrator.default_variable>`

    initial_v : float, list or 1d array : default 0.0
        specifies starting value for integration of dv/dt.  If it is a list or array, it must be the same length as
        `default_variable <FHNIntegrator.default_variable>`

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
        coefficient on the external stimulus (`variable <FHNIntegrator.variable>`) term in the dv/dt equation

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
        under a specific formulation of the FHN equations, the threshold parameter behaves as a "threshold of
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

    componentName = FHN_INTEGRATOR_FUNCTION

    class Params(Integrator.Params):
        variable = Param(np.array([1.0]), read_only=True)
        scale = Param(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        offset = Param(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        time_step_size = Param(0.05, modulable=True)
        a_v = Param(1.0 / 3, modulable=True)
        b_v = Param(0.0, modulable=True)
        c_v = Param(1.0, modulable=True)
        d_v = Param(0.0, modulable=True)
        e_v = Param(-1.0, modulable=True)
        f_v = Param(1.0, modulable=True)
        time_constant_v = Param(1.0, modulable=True)
        a_w = Param(1.0, modulable=True)
        b_w = Param(-0.8, modulable=True)
        c_w = Param(0.7, modulable=True)
        threshold = Param(-1.0, modulable=True)
        time_constant_w = Param(12.5, modulable=True)
        mode = Param(1.0, modulable=True)
        uncorrelated_activity = Param(0.0, modulable=True)

        # FIX: make an integration_method enum class for RK4/EULER
        integration_method = Param("RK4", stateful=False)

        initial_w = np.array([1.0])
        initial_v = np.array([1.0])
        t_0 = 0.0
        previous_w = np.array([1.0])
        previous_v = np.array([1.0])
        previous_time = 0.0

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        NOISE: None,
        INCREMENT: None,
    })

    multiplicative_param = SCALE
    additive_param = OFFSET

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 offset=0.0,
                 scale=1.0,
                 initial_w=0.0,
                 initial_v=0.0,
                 time_step_size=0.05,
                 t_0=0.0,
                 a_v=-1 / 3,
                 b_v=0.0,
                 c_v=1.0,
                 d_v=0.0,
                 e_v=-1.0,
                 f_v=1.0,
                 time_constant_v=1.0,
                 a_w=1.0,
                 b_w=-0.8,
                 c_w=0.7,
                 threshold=-1.0,
                 time_constant_w=12.5,
                 mode=1.0,
                 uncorrelated_activity=0.0,
                 integration_method="RK4",
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None):

        if not hasattr(self, "initializers"):
            self.initializers = ["initial_v", "initial_w", "t_0"]

        if not hasattr(self, "stateful_attributes"):
            self.stateful_attributes = ["previous_v", "previous_w", "previous_time"]

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(default_variable=default_variable,
                                                  offset=offset,
                                                  scale=scale,
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
                                                  )

        super().__init__(
            default_variable=default_variable,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)

    @property
    def output_type(self):
        return self._output_type

    @output_type.setter
    def output_type(self, value):
        # disabled because it happens during normal execution, may be confusing
        # warnings.warn('output_type conversion disabled for {0}'.format(self.__class__.__name__))
        self._output_type = None

    def _validate_params(self, request_set, target_set=None, context=None):
        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)
        if self.integration_method not in {"RK4", "EULER"}:
            raise FunctionError("Invalid integration method ({}) selected for {}. Choose 'RK4' or 'EULER'".
                                format(self.integration_method, self.name))

    def _euler_FHN(
        self, variable, previous_value_v, previous_value_w, previous_time, slope_v, slope_w, time_step_size,
        a_v,
        threshold, b_v, c_v, d_v, e_v, f_v, time_constant_v, mode, a_w, b_w, c_w, uncorrelated_activity,
        time_constant_w, execution_id=None
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
            execution_id=execution_id
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
            execution_id=execution_id
        )

        new_v = previous_value_v + time_step_size * slope_v_approx
        new_w = previous_value_w + time_step_size * slope_w_approx

        return new_v, new_w

    def _runge_kutta_4_FHN(
        self, variable, previous_value_v, previous_value_w, previous_time, slope_v, slope_w,
        time_step_size,
        a_v, threshold, b_v, c_v, d_v, e_v, f_v, time_constant_v, mode, a_w, b_w, c_w,
        uncorrelated_activity, time_constant_w, execution_id=None
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
            execution_id=execution_id
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
            execution_id=execution_id
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
            execution_id=execution_id
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
            execution_id=execution_id
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
            execution_id=execution_id
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
            execution_id=execution_id
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
            execution_id=execution_id
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
            execution_id=execution_id
        )

        new_v = previous_value_v \
                + (time_step_size / 6) * (
        slope_v_approx_1 + 2 * (slope_v_approx_2 + slope_v_approx_3) + slope_v_approx_4)
        new_w = previous_value_w \
                + (time_step_size / 6) * (
        slope_w_approx_1 + 2 * (slope_w_approx_2 + slope_w_approx_3) + slope_w_approx_4)

        return new_v, new_w

    def dv_dt(self, variable, time, v, w, a_v, threshold, b_v, c_v, d_v, e_v, f_v, time_constant_v, execution_id=None):
        previous_w = self.get_current_function_param('previous_w', execution_id)

        val = (a_v * (v ** 3) + (1 + threshold) * b_v * (v ** 2) + (-threshold) * c_v * v + d_v
               + e_v * previous_w + f_v * variable) / time_constant_v

        # Standard coefficients - hardcoded for testing
        # val = v - (v**3)/3 - w + variable
        # Gilzenrat paper - hardcoded for testing
        # val = (v*(v-0.5)*(1-v) - w + variable)/0.01
        return val

    def dw_dt(self, variable, time, w, v, mode, a_w, b_w, c_w, uncorrelated_activity, time_constant_w, execution_id=None):
        previous_v = self.get_current_function_param('previous_v', execution_id)

        # val = np.ones_like(variable)*(mode*a_w*self.previous_v + b_w*w + c_w + (1-mode)*uncorrelated_activity)/time_constant_w
        val = (mode * a_w * previous_v + b_w * w + c_w + (1 - mode) * uncorrelated_activity) / time_constant_w

        # Standard coefficients - hardcoded for testing
        # val = (v + 0.7 - 0.8*w)/12.5
        # Gilzenrat paper - hardcoded for testing

        # val = (v - 0.5*w)
        if not np.isscalar(variable):
            val = np.broadcast_to(val, variable.shape)

        return val

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """
        Return: current v, current w

        The model is defined by the following system of differential equations:

            *time_constant_v* :math:`* \\frac{dv}{dt} =`

                *a_v* :math:`* v^3 + (1 + threshold) *` *b_v* :math:`* v^2 + (- threshold) *` *c_v*
                :math:`* v^2 +` *d_v* :math:`+` *e_v* :math:`* w +` *f_v* :math:`* I_{ext}`

            *time_constant_w* :math:`* dw/dt =`

                :math:`mode *` *a_w* :math:`* v +` *b_w* :math:`* w +` *c_w*
                :math:`+ (1 - self.mode) *` *self.uncorrelated_activity*


        Arguments
        ---------

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        current value of v , current value of w : float, list, or array

        """

        a_v = self.get_current_function_param("a_v", execution_id)
        b_v = self.get_current_function_param("b_v", execution_id)
        c_v = self.get_current_function_param("c_v", execution_id)
        d_v = self.get_current_function_param("d_v", execution_id)
        e_v = self.get_current_function_param("e_v", execution_id)
        f_v = self.get_current_function_param("f_v", execution_id)
        time_constant_v = self.get_current_function_param("time_constant_v", execution_id)
        threshold = self.get_current_function_param("threshold", execution_id)
        a_w = self.get_current_function_param("a_w", execution_id)
        b_w = self.get_current_function_param("b_w", execution_id)
        c_w = self.get_current_function_param("c_w", execution_id)
        uncorrelated_activity = self.get_current_function_param("uncorrelated_activity", execution_id)
        time_constant_w = self.get_current_function_param("time_constant_w", execution_id)
        mode = self.get_current_function_param("mode", execution_id)
        time_step_size = self.get_current_function_param(TIME_STEP_SIZE, execution_id)
        previous_v = self.get_current_function_param("previous_v", execution_id)
        previous_w = self.get_current_function_param("previous_w", execution_id)
        previous_time = self.get_current_function_param("previous_time", execution_id)

        # integration_method is a compile time parameter
        integration_method = self.get_current_function_param("integration_method", execution_id)
        if integration_method == "RK4":
            approximate_values = self._runge_kutta_4_FHN(
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
                execution_id=execution_id
            )

        elif integration_method == "EULER":
            approximate_values = self._euler_FHN(
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
                execution_id=execution_id
            )
        else:
            raise FunctionError("Invalid integration method ({}) selected for {}".
                                format(integration_method, self.name))

        if self.parameters.context.get(execution_id).initialization_status != ContextFlags.INITIALIZING:
            previous_v = approximate_values[0]
            previous_w = approximate_values[1]
            previous_time = previous_time + time_step_size
            if not np.isscalar(variable):
                previous_time = np.broadcast_to(previous_time, variable.shape).copy()

            self.parameters.previous_v.set(previous_v, execution_id)
            self.parameters.previous_w.set(previous_w, execution_id)
            self.parameters.previous_time.set(previous_time, execution_id)

        return previous_v, previous_w, previous_time

    def _get_context_struct_type(self, ctx):
        context = (self.previous_v, self.previous_w, self.previous_time)
        context_type = ctx.convert_python_struct_to_llvm_ir(context)
        return context_type

    def _get_context_initializer(self, execution_id):
        previous_v = self.parameters.previous_v.get(execution_id)
        previous_w = self.parameters.previous_w.get(execution_id)
        previous_time = self.parameters.previous_time.get(execution_id)

        v = previous_v if np.isscalar(previous_v) else tuple(previous_v)
        w = previous_w if np.isscalar(previous_w) else tuple(previous_w)
        time = previous_time if np.isscalar(previous_time) else tuple(previous_time)
        return (v, w, time)

    def _gen_llvm_function_body(self, ctx, builder, params, context, arg_in, arg_out):
        zero_i32 = ctx.int32_ty(0)

        # Get rid of 2d array
        assert isinstance(arg_in.type.pointee, ir.ArrayType)
        if isinstance(arg_in.type.pointee.element, ir.ArrayType):
            assert(arg_in.type.pointee.count == 1)
            arg_in = builder.gep(arg_in, [zero_i32, zero_i32])

        # Load context values
        previous_v_ptr = builder.gep(context, [zero_i32, ctx.int32_ty(0)])
        previous_w_ptr = builder.gep(context, [zero_i32, ctx.int32_ty(1)])
        previous_time_ptr = builder.gep(context, [zero_i32, ctx.int32_ty(2)])

        # Output locations
        out_v_ptr = builder.gep(arg_out, [zero_i32, ctx.int32_ty(0)])
        out_w_ptr = builder.gep(arg_out, [zero_i32, ctx.int32_ty(1)])
        out_time_ptr = builder.gep(arg_out, [zero_i32, ctx.int32_ty(2)])

        # Load parameters
        param_vals = {}
        for p in self._get_param_ids():
            param_ptr, builder = ctx.get_param_ptr(self, builder, params, p)
            param_vals[p] = pnlvm.helpers.load_extract_scalar_array_one(
                                            builder, param_ptr)

        inner_args = {"ctx": ctx, "var_ptr": arg_in, "param_vals": param_vals,
                      "out_v": out_v_ptr, "out_w": out_w_ptr,
                      "out_time": out_time_ptr,
                      "previous_v_ptr": previous_v_ptr,
                      "previous_w_ptr": previous_w_ptr,
                      "previous_time_ptr": previous_time_ptr}

        # KDM 11/7/18: since we're compiling with this set, I'm assuming it should be
        # stateless and considered an inherent feature of the function. Changing parameter
        # to stateful=False accordingly. If it should be stateful, need to pass an execution_id here
        method = self.get_current_function_param("integration_method")
        if method == "RK4":
            func = functools.partial(self.__gen_llvm_rk4_body, **inner_args)
        elif method == "EULER":
            func = functools.partial(self.__gen_llvm_euler_body, **inner_args)
        else:
            raise FunctionError("Invalid integration method ({}) selected for {}".
                                format(method, self.name))

        with helpers.array_ptr_loop(builder, arg_in, method + "_body") as args:
            func(*args)

        # Save context
        result = builder.load(arg_out)
        builder.store(result, context)
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

        thr_neg = builder.fsub(ctx.float_ty(0.0), param_vals["threshold"])
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


class AccumulatorIntegrator(Integrator):  # ----------------------------------------------------------------------------
    """
    AccumulatorIntegrator(              \
        default_variable=None,          \
        rate=1.0,                       \
        noise=0.0,                      \
        scale: parameter_spec = 1.0,    \
        offset: parameter_spec = 0.0,   \
        initializer,                    \
        params=None,                    \
        owner=None,                     \
        prefs=None,                     \
        )

    .. _AccumulatorIntegrator:

    Integrates prior value by multiplying `previous_value <AccumulatorIntegrator.previous_value>` by `rate
    <Integrator.rate>` and adding `increment <AccumulatorIntegrator.increment>` and  `noise
    <AccumulatorIntegrator.noise>`. Ignores `variable <Integrator.variable>`).

    Arguments
    ---------

    default_variable : number, list or array : default ClassDefaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float, list or 1d array : default 1.0
        specifies the multiplicative decrement of `previous_value <AccumulatorIntegrator.previous_value>` (i.e.,
        the rate of exponential decay).  If it is a list or array, it must be the same length as
        `variable <AccumulatorIntegrator.default_variable>`.

    increment : float, list or 1d array : default 0.0
        specifies an amount to be added to `previous_value <AccumulatorIntegrator.previous_value>` in each call to
        `function <AccumulatorIntegrator.function>` (see `increment <AccumulatorIntegrator.increment>` for details).
        If it is a list or array, it must be the same length as `variable <AccumulatorIntegrator.default_variable>`
        (see `increment <AccumulatorIntegrator.increment>` for details).

    noise : float, PsyNeuLink Function, list or 1d array : default 0.0
        specifies random value to be added to `prevous_value <AccumulatorIntegrator.previous_value>` in each call to
        `function <AccumulatorIntegrator.function>`. If it is a list or array, it must be the same length as
        `variable <AccumulatorIntegrator.default_variable>` (see `noise <AccumulatorIntegrator.noise>` for details).

    initializer float, list or 1d array : default 0.0
        specifies starting value for integration.  If it is a list or array, it must be the same length as
        `default_variable <AccumulatorIntegrator.default_variable>` (see `initializer
        <AccumulatorIntegrator.initializer>` for details).

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
        **Ignored** by the AccumulatorIntegrator function. Refer to LCAIntegrator or AdaptiveIntegrator for
        integrator functions that depend on both a prior value and a new value (variable).

    rate : float or 1d array
        determines the multiplicative decrement of `previous_value <AccumulatorIntegrator.previous_value>` (i.e., the
        rate of exponential decay) in each call to `function <AccumulatorIntegrator.function>`.  If it is a list or
        array, it must be the same length as `variable <AccumulatorIntegrator.default_variable>` and each element is
        used to multiply the corresponding element of `previous_value <AccumulatorIntegrator.previous_value>` (i.e.,
        it is used for Hadamard multiplication).  If it is a scalar or has a single element, its value is used to
        multiply all the elements of `previous_value <AccumulatorIntegrator.previous_value>`.

    increment : float, function, list, or 1d array
        determines the amount added to `previous_value <AccumulatorIntegrator.previous_value>` in each call to
        `function <AccumulatorIntegrator.function>`.  If it is a list or array, it must be the same length as
        `variable <AccumulatorIntegrator.default_variable>` and each element is added to the corresponding element of
        `previous_value <AccumulatorIntegrator.previous_value>` (i.e., it is used for Hadamard addition).  If it is a
        scalar or has a single element, its value is added to all the elements of `previous_value
        <AccumulatorIntegrator.previous_value>`.

    noise : float, function, list, or 1d array
        determines a random value to be added in each call to `function <AccumulatorIntegrator.function>`.
        If it is a list or array, it must be the same length as `variable <AccumulatorIntegrator.default_variable>` and
        each element is added to the corresponding element of `previous_value <AccumulatorIntegrator.previous_value>`
        (i.e., it is used for Hadamard addition).  If it is a scalar or has a single element, its value is added to all
        the elements of `previous_value <AccumulatorIntegrator.previous_value>`.  If it is a function, it will be
        executed separately and added to each element.

        .. note::

            In order to generate random noise, a probability distribution function should be selected (see
            `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value from
            its distribution on each execution. If noise is specified as a float or as a function with a fixed output,
            then the noise will simply be an offset that remains the same across all executions.

    initializer : float, 1d array or list
        determines the starting value for integration (i.e., the value to which `previous_value
        <AccumulatorIntegrator.previous_value>` is set. If initializer is a list or array, it must be the same length
        as `variable <AccumulatorIntegrator.default_variable>`.

    previous_value : 1d array : default ClassDefaults.variable
        stores previous value to which `rate <AccumulatorIntegrator.rate>` and `noise <AccumulatorIntegrator.noise>`
        will be added.

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

    componentName = ACCUMULATOR_INTEGRATOR_FUNCTION

    class Params(Integrator.Params):
        rate = Param(None, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        increment = Param(None, modulable=True, aliases=[ADDITIVE_PARAM])

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        NOISE: None,
        RATE: None,
        INCREMENT: None,
    })

    # multiplicative param does not make sense in this case
    multiplicative_param = RATE
    additive_param = INCREMENT

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 # rate: parameter_spec = 1.0,
                 rate=None,
                 noise=0.0,
                 increment=None,
                 initializer=None,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  initializer=initializer,
                                                  noise=noise,
                                                  increment=increment,
                                                  params=params)

        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)

        self.has_initializers = True

    def _accumulator_check_args(self, variable=None, execution_id=None, params=None, target_set=None, context=None):
        """validate params and assign any runtime params.

        Called by AccumulatorIntegrator to validate params
        Validation can be suppressed by turning parameter_validation attribute off
        target_set is a params dictionary to which params should be assigned;
           otherwise, they are assigned to paramsCurrent;

        Does the following:
        - assign runtime params to paramsCurrent
        - validate params if PARAM_VALIDATION is set

        :param params: (dict) - params to validate
        :target_set: (dict) - set to which params should be assigned (default: self.paramsCurrent)
        :return:
        """

        # PARAMS ------------------------------------------------------------

        # If target_set is not specified, use paramsCurrent
        if target_set is None:
            target_set = self.paramsCurrent

        # # MODIFIED 11/27/16 OLD:
        # # If parameter_validation is set, the function was called with params,
        # #   and they have changed, then validate requested values and assign to target_set
        # if self.prefs.paramValidationPref and params and not params is None and not params is target_set:
        #     # self._validate_params(params, target_set, context=FUNCTION_CHECK_ARGS)
        #     self._validate_params(request_set=params, target_set=target_set, context=context)

        # If params have been passed, treat as runtime params and assign to paramsCurrent
        #   (relabel params as runtime_params for clarity)
        if execution_id in self._runtime_params_reset:
            for key in self._runtime_params_reset[execution_id]:
                self._set_parameter_value(key, self._runtime_params_reset[execution_id][key], execution_id)
        self._runtime_params_reset[execution_id] = {}

        runtime_params = params
        if runtime_params:
            for param_name in runtime_params:
                if hasattr(self, param_name):
                    if param_name in {FUNCTION, INPUT_STATES, OUTPUT_STATES}:
                        continue
                    if execution_id not in self._runtime_params_reset:
                        self._runtime_params_reset[execution_id] = {}
                    self._runtime_params_reset[execution_id][param_name] = getattr(self.parameters, param_name).get(execution_id)
                    self._set_parameter_value(param_name, runtime_params[param_name], execution_id)

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """
        Return: `previous_value <ConstantIntegrator.previous_value>` combined with `rate <ConstantIntegrator.rate>` and
        `noise <ConstantIntegrator.noise>`.

        Arguments
        ---------

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------

        updated value of integral : 2d array

        """
        self._accumulator_check_args(variable, execution_id=execution_id, params=params, context=context)

        rate = self.get_current_function_param(RATE, execution_id)
        increment = self.get_current_function_param(INCREMENT, execution_id)
        noise = self._try_execute_param(self.get_current_function_param(NOISE, execution_id), variable)

        if rate is None:
            rate = 1.0

        if increment is None:
            increment = 0.0

        previous_value = np.atleast_2d(self.get_previous_value(execution_id))

        value = previous_value * rate + noise + increment

        # If this NOT an initialization run, update the old value
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if self.parameters.context.get(execution_id).initialization_status != ContextFlags.INITIALIZING:
            self.parameters.previous_value.set(value, execution_id, override=True)

        return self.convert_output_type(value)


class LCAIntegrator(Integrator):  # ------------------------------------------------------------------------------------
    """
    LCAIntegrator(                  \
        default_variable=None,      \
        noise=0.0,                  \
        initializer=0.0,            \
        rate=1.0,                   \
        offset=None,                \
        time_step_size=0.1,         \
        params=None,                \
        owner=None,                 \
        prefs=None,                 \
        )

    .. _LCAIntegrator:

    Integrate current value of `variable <LCAIntegrator.variable>` with its prior value:

    .. math::

        rate \\cdot previous\\_value + variable + noise \\sqrt{time\\_step\\_size}

    COMMENT:
    `rate <LCAIntegrator.rate>` * `previous_value <LCAIntegrator.previous_value>` + \
    `variable <variable.LCAIntegrator.variable>` + \
    `noise <LCAIntegrator.noise>`;
    COMMENT

    Arguments
    ---------

    default_variable : number, list or array : default ClassDefaults.variable
        specifies a template for the value to be integrated;  if it is a list or array, each element is independently
        integrated.

    rate : float, list or 1d array : default 1.0
        scales the contribution of `previous_value <LCAIntegrator.previous_value>` to the accumulation of the
        `value <LCAIntegrator.value>` on each time step

    noise : float, PsyNeuLink Function, list or 1d array : default 0.0
        specifies random value to be added in each call to `function <LCAIntegrator.function>`. (see
        `noise <LCAIntegrator.noise>` for details).

    initializer : float, list or 1d array : default 0.0
        specifies starting value for integration.  If it is a list or array, it must be the same length as
        `default_variable <LCAIntegrator.default_variable>` (see `initializer <LCAIntegrator.initializer>` for details).

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
        current input value some portion of which (determined by `rate <LCAIntegrator.rate>`) will be
        added to the prior value;  if it is an array, each element is independently integrated.

    rate : float or 1d array
        scales the contribution of `previous_value <LCAIntegrator.previous_value>` to the
        accumulation of the `value <LCAIntegrator.value>` on each time step. If rate has a single element, it
        applies to all elements of `variable <LCAIntegrator.variable>`;  if rate has more than one element, each element
        applies to the corresponding element of `variable <LCAIntegrator.variable>`.

    noise : float, function, list, or 1d array
        specifies a value to be added in each call to `function <LCAIntegrator.function>`.

        If noise is a list or array, it must be the same length as `variable <LCAIntegrator.default_variable>`.

        If noise is specified as a single float or function, while `variable <LCAIntegrator.variable>` is a list or
        array, noise will be applied to each variable element. In the case of a noise function, this means that the
        function will be executed separately for each variable element.

        .. note::
            In order to generate random noise, we recommend selecting a probability distribution function (see
            `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value from
            its distribution on each execution. If noise is specified as a float or as a function with a fixed output,
            then the noise will simply be an offset that remains the same across all executions.

    initializer : float, 1d array or list
        determines the starting value for integration (i.e., the value to which
        `previous_value <LCAIntegrator.previous_value>` is set.

        If initializer is a list or array, it must be the same length as `variable <LCAIntegrator.default_variable>`.

    previous_value : 1d array : default ClassDefaults.variable
        stores previous value with which `variable <LCAIntegrator.variable>` is integrated.

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

    componentName = LCAMechanism_INTEGRATOR_FUNCTION

    class Params(Integrator.Params):
        rate = Param(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        offset = Param(None, modulable=True, aliases=[ADDITIVE_PARAM])
        time_step_size = Param(0.1, modulable=True)

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        NOISE: None,
        RATE: None
    })

    multiplicative_param = RATE
    additive_param = OFFSET

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate: parameter_spec = 1.0,
                 noise=0.0,
                 offset=None,
                 initializer=None,
                 time_step_size=0.1,
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  initializer=initializer,
                                                  noise=noise,
                                                  time_step_size=time_step_size,
                                                  offset=offset,
                                                  params=params)

        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)

        self.has_initializers = True

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """
        Return:

        .. math::

            rate \\cdot previous\\_value + variable + noise \\sqrt{time\\_step\\_size}

        Arguments
        ---------

        variable : number, list or array : default ClassDefaults.variable
           a single value or array of values to be integrated.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        updated value of integral : 2d array

        """

        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        rate = np.atleast_1d(self.get_current_function_param(RATE, execution_id))
        initializer = self.get_current_function_param(INITIALIZER, execution_id)  # unnecessary?
        time_step_size = self.get_current_function_param(TIME_STEP_SIZE, execution_id)
        offset = self.get_current_function_param(OFFSET, execution_id)

        if offset is None:
            offset = 0.0

        # execute noise if it is a function
        noise = self._try_execute_param(self.get_current_function_param(NOISE, execution_id), variable)
        previous_value = self.get_previous_value(execution_id)
        new_value = variable

        # Gilzenrat: previous_value + (-previous_value + variable)*self.time_step_size + noise --> rate = -1
        value = previous_value + (rate * previous_value + new_value) * time_step_size + noise * (time_step_size ** 0.5)

        adjusted_value = value + offset

        # If this NOT an initialization run, update the old value
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if self.parameters.context.get(execution_id).initialization_status != ContextFlags.INITIALIZING:
            self.parameters.previous_value.set(adjusted_value, execution_id)

        return self.convert_output_type(adjusted_value)


class AGTUtilityIntegrator(Integrator):  # -----------------------------------------------------------------------------
    """
    AGTUtilityIntegrator(                    \
        default_variable=None,            \
        rate=1.0,                         \
        noise=0.0,                        \
        scale: parameter_spec = 1.0,      \
        offset: parameter_spec = 0.0,     \
        initializer,                      \
        initial_short_term_utility = 0.0, \
        initial_long_term_utility = 0.0,  \
        short_term_gain = 1.0,            \
        long_term_gain =1.0,              \
        short_term_bias = 0.0,            \
        long_term_bias=0.0,               \
        short_term_rate=1.0,              \
        long_term_rate=1.0,               \
        params=None,                      \
        owner=None,                       \
        prefs=None,                       \
        )

    .. _AGTUtilityIntegrator:

    Computes an exponentially weighted moving average on the variable using two sets of parameters:

    short_term_utility =

       (1 - `short_term_rate <AGTUtilityIntegrator.short_term_rate>`) :math:`*` `previous_short_term_utility
       <AGTUtilityIntegrator.previous_short_term_utility>` + `short_term_rate <AGTUtilityIntegrator.short_term_rate>`
       :math:`*` `variable <AGTUtilityIntegrator.variable>`

    long_term_utility =

       (1 - `long_term_rate <AGTUtilityIntegrator.long_term_rate>`) :math:`*` `previous_long_term_utility
       <AGTUtilityIntegrator.previous_long_term_utility>` + `long_term_rate <AGTUtilityIntegrator.long_term_rate>`
       :math:`*` `variable <AGTUtilityIntegrator.variable>`

    then takes the logistic of each utility value, using the corresponding (short term and long term) gain and bias.

    Finally, computes a single value which combines the two values according to:

    value = [1-short_term_utility_logistic]*long_term_utility_logistic

    Arguments
    ---------

    rate : float, list or 1d array : default 1.0
        specifies the overall smoothing factor of the EWMA used to combine the long term and short term utility values

    noise : float, PsyNeuLink Function, list or 1d array : default 0.0
        TBI?

    initial_short_term_utility : float : default 0.0
        specifies starting value for integration of short_term_utility

    initial_long_term_utility : float : default 0.0
        specifies starting value for integration of long_term_utility

    short_term_gain : float : default 1.0
        specifies gain for logistic function applied to short_term_utility

    long_term_gain : float : default 1.0
        specifies gain for logistic function applied to long_term_utility

    short_term_bias : float : default 0.0
        specifies bias for logistic function applied to short_term_utility

    long_term_bias : float : default 0.0
        specifies bias for logistic function applied to long_term_utility

    short_term_rate : float : default 1.0
        specifies smoothing factor of EWMA filter applied to short_term_utility

    long_term_rate : float : default 1.0
        specifies smoothing factor of EWMA filter applied to long_term_utility

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
        current input value used in both the short term and long term EWMA computations

    noise : float, PsyNeuLink Function, list or 1d array : default 0.0
        TBI?

    initial_short_term_utility : float : default 0.0
        specifies starting value for integration of short_term_utility

    initial_long_term_utility : float : default 0.0
        specifies starting value for integration of long_term_utility

    short_term_gain : float : default 1.0
        specifies gain for logistic function applied to short_term_utility

    long_term_gain : float : default 1.0
        specifies gain for logistic function applied to long_term_utility

    short_term_bias : float : default 0.0
        specifies bias for logistic function applied to short_term_utility

    long_term_bias : float : default 0.0
        specifies bias for logistic function applied to long_term_utility

    short_term_rate : float : default 1.0
        specifies smoothing factor of EWMA filter applied to short_term_utility

    long_term_rate : float : default 1.0
        specifies smoothing factor of EWMA filter applied to long_term_utility

    previous_short_term_utility : 1d array
        stores previous value with which `variable <AGTUtilityIntegrator.variable>` is integrated using the EWMA filter and
        short term parameters

    previous_long_term_utility : 1d array
        stores previous value with which `variable <AGTUtilityIntegrator.variable>` is integrated using the EWMA filter and
        long term parameters

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

    componentName = UTILITY_INTEGRATOR_FUNCTION

    multiplicative_param = RATE
    additive_param = OFFSET

    class Params(Integrator.Params):
        rate = Param(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        offset = Param(0.0, modulable=True, aliases=[ADDITIVE_PARAM])
        short_term_gain = Param(1.0, modulable=True)
        long_term_gain = Param(1.0, modulable=True)
        short_term_bias = Param(0.0, modulable=True)
        long_term_bias = Param(0.0, modulable=True)
        short_term_rate = Param(0.9, modulable=True)
        long_term_rate = Param(0.1, modulable=True)

        operation = "s*l"
        initial_short_term_utility = 0.0
        initial_long_term_utility = 0.0

        previous_short_term_utility = None
        previous_long_term_utility = None

        short_term_utility_logistic = None
        long_term_utility_logistic = None

    paramClassDefaults = Function_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        NOISE: None,
        RATE: None
    })

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate: parameter_spec = 1.0,
                 noise=0.0,
                 offset=0.0,
                 initializer=None,
                 initial_short_term_utility=0.0,
                 initial_long_term_utility=0.0,
                 short_term_gain=1.0,
                 long_term_gain=1.0,
                 short_term_bias=0.0,
                 long_term_bias=0.0,
                 short_term_rate=0.9,
                 long_term_rate=0.1,
                 operation="s*l",
                 params: tc.optional(dict) = None,
                 owner=None,
                 prefs: is_pref_set = None):

        if not hasattr(self, "initializers"):
            self.initializers = ["initial_long_term_utility", "initial_short_term_utility"]

        if not hasattr(self, "stateful_attributes"):
            self.stateful_attributes = ["previous_short_term_utility", "previous_long_term_utility"]

        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(rate=rate,
                                                  initializer=initializer,
                                                  noise=noise,
                                                  offset=offset,
                                                  initial_short_term_utility=initial_short_term_utility,
                                                  initial_long_term_utility=initial_long_term_utility,
                                                  short_term_gain=short_term_gain,
                                                  long_term_gain=long_term_gain,
                                                  short_term_bias=short_term_bias,
                                                  long_term_bias=long_term_bias,
                                                  short_term_rate=short_term_rate,
                                                  long_term_rate=long_term_rate,
                                                  operation=operation,
                                                  params=params)

        self.previous_long_term_utility = self.initial_long_term_utility
        self.previous_short_term_utility = self.initial_short_term_utility

        super().__init__(
            default_variable=default_variable,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR)

        self.has_initializers = True

    def _validate_params(self, request_set, target_set=None, context=None):

        # Handle list or array for rate specification
        if RATE in request_set:
            rate = request_set[RATE]
            if isinstance(rate, (list, np.ndarray)):
                if len(rate) != 1 and len(rate) != np.array(self.instance_defaults.variable).size:
                    # If the variable was not specified, then reformat it to match rate specification
                    #    and assign ClassDefaults.variable accordingly
                    # Note: this situation can arise when the rate is parametrized (e.g., as an array) in the
                    #       AGTUtilityIntegrator's constructor, where that is used as a specification for a function parameter
                    #       (e.g., for an IntegratorMechanism), whereas the input is specified as part of the
                    #       object to which the function parameter belongs (e.g., the IntegratorMechanism);
                    #       in that case, the Integrator gets instantiated using its ClassDefaults.variable ([[0]]) before
                    #       the object itself, thus does not see the array specification for the input.
                    if self._default_variable_flexibility is DefaultsFlexibility.FLEXIBLE:
                        self._instantiate_defaults(variable=np.zeros_like(np.array(rate)), context=context)
                        if self.verbosePref:
                            warnings.warn(
                                "The length ({}) of the array specified for the rate parameter ({}) of {} "
                                "must match the length ({}) of the default input ({});  "
                                "the default input has been updated to match".format(
                                    len(rate),
                                    rate,
                                    self.name,
                                    np.array(self.instance_defaults.variable).size
                                ),
                                self.instance_defaults.variable
                            )
                    else:
                        raise FunctionError(
                            "The length ({}) of the array specified for the rate parameter ({}) of {} "
                            "must match the length ({}) of the default input ({})".format(
                                len(rate),
                                rate,
                                self.name,
                                np.array(self.instance_defaults.variable).size,
                                self.instance_defaults.variable,
                            )
                        )
                        # OLD:
                        # self.paramClassDefaults[RATE] = np.zeros_like(np.array(rate))

                        # KAM changed 5/15 b/c paramClassDefaults were being updated and *requiring* future integrator functions
                        # to have a rate parameter of type ndarray/list

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if RATE in target_set:
            if isinstance(target_set[RATE], (list, np.ndarray)):
                for r in target_set[RATE]:
                    if r < 0.0 or r > 1.0:
                        raise FunctionError("The rate parameter ({}) (or all of its elements) of {} must be "
                                            "between 0.0 and 1.0 when integration_type is set to ADAPTIVE.".
                                            format(target_set[RATE], self.name))
            else:
                if target_set[RATE] < 0.0 or target_set[RATE] > 1.0:
                    raise FunctionError(
                        "The rate parameter ({}) (or all of its elements) of {} must be between 0.0 and "
                        "1.0 when integration_type is set to ADAPTIVE.".format(target_set[RATE], self.name))

        if NOISE in target_set:
            noise = target_set[NOISE]
            if isinstance(noise, DistributionFunction):
                noise.owner = self
                target_set[NOISE] = noise._execute
            self._validate_noise(target_set[NOISE])
            # if INITIALIZER in target_set:
            #     self._validate_initializer(target_set[INITIALIZER])

        if OPERATION in target_set:
            if not target_set[OPERATION] in {'s*l', 's+l', 's-l', 'l-s'}:
                raise FunctionError("\'{}\' arg for {} must be one of the following: {}".
                                    format(OPERATION, self.name, {'s*l', 's+l', 's-l', 'l-s'}))

    def _EWMA_filter(self, a, rate, b):

        return (1 - rate) * a + rate * b

    def _logistic(self, variable, gain, bias):

        return 1 / (1 + np.exp(-(gain * variable) + bias))

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """
        Return: some fraction of `variable <AGTUtilityIntegrator.variable>` combined with some fraction of `previous_value
        <AGTUtilityIntegrator.previous_value>`.

        Arguments
        ---------

        variable : number, list or array : default ClassDefaults.variable
           a single value or array of values to be integrated.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        updated value of integral : 2d array

        """
        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)
        rate = np.array(self.get_current_function_param(RATE, execution_id)).astype(float)
        # execute noise if it is a function
        noise = self._try_execute_param(self.get_current_function_param(NOISE, execution_id), variable)
        short_term_rate = self.get_current_function_param("short_term_rate", execution_id)
        long_term_rate = self.get_current_function_param("long_term_rate", execution_id)

        # Integrate Short Term Utility:
        short_term_utility = self._EWMA_filter(self.previous_short_term_utility,
                                               short_term_rate,
                                               variable)
        # Integrate Long Term Utility:
        long_term_utility = self._EWMA_filter(self.previous_long_term_utility,
                                              long_term_rate,
                                              variable)

        value = self.combine_utilities(short_term_utility, long_term_utility, execution_id=execution_id)

        if self.parameters.context.get(execution_id).initialization_status != ContextFlags.INITIALIZING:
            self.parameters.previous_short_term_utility.set(short_term_utility, execution_id)
            self.parameters.previous_long_term_utility.set(long_term_utility, execution_id)

        return self.convert_output_type(value)

    def combine_utilities(self, short_term_utility, long_term_utility, execution_id=None):
        short_term_gain = self.get_current_function_param("short_term_gain", execution_id)
        short_term_bias = self.get_current_function_param("short_term_bias", execution_id)
        long_term_gain = self.get_current_function_param("long_term_gain", execution_id)
        long_term_bias = self.get_current_function_param("long_term_bias", execution_id)
        operation = self.get_current_function_param(OPERATION, execution_id)
        offset = self.get_current_function_param(OFFSET, execution_id)

        short_term_utility_logistic = self._logistic(
            variable=short_term_utility,
            gain=short_term_gain,
            bias=short_term_bias,
        )
        self.parameters.short_term_utility_logistic.set(short_term_utility_logistic, execution_id)

        long_term_utility_logistic = self._logistic(
            variable=long_term_utility,
            gain=long_term_gain,
            bias=long_term_bias,
        )
        self.parameters.long_term_utility_logistic.set(long_term_utility_logistic, execution_id)

        if operation == "s*l":
            # Engagement in current task = [1—logistic(short term utility)]*[logistic{long - term utility}]
            value = (1 - short_term_utility_logistic) * long_term_utility_logistic
        elif operation == "s-l":
            # Engagement in current task = [1—logistic(short term utility)] - [logistic{long - term utility}]
            value = (1 - short_term_utility_logistic) - long_term_utility_logistic
        elif operation == "s+l":
            # Engagement in current task = [1—logistic(short term utility)] + [logistic{long - term utility}]
            value = (1 - short_term_utility_logistic) + long_term_utility_logistic
        elif operation == "l-s":
            # Engagement in current task = [logistic{long - term utility}] - [1—logistic(short term utility)]
            value = long_term_utility_logistic - (1 - short_term_utility_logistic)

        return value + offset

    def reinitialize(self, short=None, long=None, execution_context=None):

        """
        Effectively begins accumulation over again at the specified utilities.

        Sets `previous_short_term_utility <AGTUtilityIntegrator.previous_short_term_utility>` to the quantity specified
        in the first argument and `previous_long_term_utility <AGTUtilityIntegrator.previous_long_term_utility>` to the
        quantity specified in the second argument.

        Sets `value <AGTUtilityIntegrator.value>` by computing it based on the newly updated values for
        `previous_short_term_utility <AGTUtilityIntegrator.previous_short_term_utility>` and
        `previous_long_term_utility <AGTUtilityIntegrator.previous_long_term_utility>`.

        If no arguments are specified, then the current values of `initial_short_term_utility
        <AGTUtilityIntegrator.initial_short_term_utility>` and `initial_long_term_utility
        <AGTUtilityIntegrator.initial_long_term_utility>` are used.
        """

        if short is None:
            short = self.get_current_function_param("initial_short_term_utility", execution_context)
        if long is None:
            long = self.get_current_function_param("initial_long_term_utility", execution_context)

        self.parameters.previous_short_term_utility.set(short, execution_context)
        self.parameters.previous_long_term_utility.set(long, execution_context)
        value = self.combine_utilities(short, long)

        self.parameters.value.set(value, execution_context, override=True)
        return value


# Note:  For any of these that correspond to args, value must match the name of the corresponding arg in __init__()
DRIFT_RATE = 'drift_rate'
DRIFT_RATE_VARIABILITY = 'DDM_DriftRateVariability'
THRESHOLD = 'threshold'
THRESHOLD_VARIABILITY = 'DDM_ThresholdRateVariability'
STARTING_POINT = 'starting_point'
STARTING_POINT_VARIABILITY = "DDM_StartingPointVariability"
# NOISE = 'noise' -- Defined in Keywords
NON_DECISION_TIME = 't0'
# DDM solution options:
kwBogaczEtAl = "BogaczEtAl"
kwNavarrosAndFuss = "NavarroAndFuss"


def _BogaczEtAl_bias_getter(owning_component=None, execution_id=None):
    starting_point = owning_component.parameters.starting_point.get(execution_id)
    threshold = owning_component.parameters.threshold.get(execution_id)
    return (starting_point + threshold) / (2 * threshold)


# QUESTION: IF VARIABLE IS AN ARRAY, DOES IT RETURN AN ARRAY FOR EACH RETURN VALUE (RT, ER, ETC.)
class BogaczEtAl(IntegratorFunction):  # -------------------------------------------------------------------------------
    """
    BogaczEtAl(                 \
        default_variable=None,  \
        drift_rate=1.0,         \
        threshold=1.0,          \
        starting_point=0.0,     \
        t0=0.2                  \
        noise=0.5,              \
        params=None,            \
        owner=None,             \
        prefs=None              \
        )

    .. _BogaczEtAl:

    Return terminal value of decision variable, mean accuracy, and mean response time computed analytically for the
    drift diffusion process as described in `Bogacz et al (2006) <https://www.ncbi.nlm.nih.gov/pubmed/17014301>`_.

    Arguments
    ---------

    default_variable : number, list or array : default ClassDefaults.variable
        specifies a template for decision variable(s);  if it is list or array, a separate solution is computed
        independently for each element.

    drift_rate : float, list or 1d array : default 1.0
        specifies the drift_rate of the drift diffusion process.  If it is a list or array,
        it must be the same length as `default_variable <BogaczEtAl.default_variable>`.

    threshold : float, list or 1d array : default 1.0
        specifies the threshold (boundary) of the drift diffusion process.  If it is a list or array,
        it must be the same length as `default_variable <BogaczEtAl.default_variable>`.

    starting_point : float, list or 1d array : default 1.0
        specifies the initial value of the decision variable for the drift diffusion process.  If it is a list or
        array, it must be the same length as `default_variable <BogaczEtAl.default_variable>`.

    noise : float, list or 1d array : default 0.0
        specifies the noise term (corresponding to the diffusion component) of the drift diffusion process.
        If it is a float, it must be a number from 0 to 1.  If it is a list or array, it must be the same length as
        `default_variable <BogaczEtAl.default_variable>` and all elements must be floats from 0 to 1.

    t0 : float, list or 1d array : default 0.2
        specifies the non-decision time for solution. If it is a float, it must be a number from 0 to 1.  If it is a
        list or array, it must be the same length as  `default_variable <BogaczEtAl.default_variable>` and all
        elements must be floats from 0 to 1.

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

    variable : number or 1d array
        holds initial value assigned to :keyword:`default_variable` argument;
        ignored by `function <BogaczEtal.function>`.

    drift_rate : float or 1d array
        determines the drift component of the drift diffusion process.

    threshold : float or 1d array
        determines the threshold (boundary) of the drift diffusion process (i.e., at which the integration
        process is assumed to terminate).

    starting_point : float or 1d array
        determines the initial value of the decision variable for the drift diffusion process.

    noise : float or 1d array
        determines the diffusion component of the drift diffusion process (used to specify the variance of a
        Gaussian random process).

    t0 : float or 1d array
        determines the assumed non-decision time to determine the response time returned by the solution.

    bias : float or 1d array
        normalized starting point:
        (`starting_point <BogaczEtAl.starting_point>` + `threshold <BogaczEtAl.threshold>`) /
        (2 * `threshold <BogaczEtAl.threshold>`)

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

    componentName = kwBogaczEtAl

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    class Params(IntegratorFunction.Params):
        drift_rate = Param(1.0, modulable=True)
        starting_point = Param(0.0, modulable=True)
        threshold = Param(1.0, modulable=True)
        noise = Param(0.5, modulable=True)
        t0 = .200
        bias = Param(0.5, read_only=True, getter=_BogaczEtAl_bias_getter)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 drift_rate: parameter_spec = 1.0,
                 starting_point: parameter_spec = 0.0,
                 threshold: parameter_spec = 1.0,
                 noise: parameter_spec = 0.5,
                 t0: parameter_spec = .200,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(drift_rate=drift_rate,
                                                  starting_point=starting_point,
                                                  threshold=threshold,
                                                  noise=noise,
                                                  t0=t0,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

    @property
    def output_type(self):
        return self._output_type

    @output_type.setter
    def output_type(self, value):
        # disabled because it happens during normal execution, may be confusing
        # warnings.warn('output_type conversion disabled for {0}'.format(self.__class__.__name__))
        self._output_type = None

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """
        Return: terminal value of decision variable (equal to threshold), mean accuracy (error rate; ER) and mean
        response time (RT)

        Arguments
        ---------

        variable : 2d array
            ignored.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------
        Decision variable, mean ER, mean RT : (float, float, float)

        """

        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        attentional_drift_rate = float(self.get_current_function_param(DRIFT_RATE, execution_id))
        stimulus_drift_rate = float(variable)
        drift_rate = attentional_drift_rate * stimulus_drift_rate
        threshold = float(self.get_current_function_param(THRESHOLD, execution_id))
        starting_point = float(self.get_current_function_param(STARTING_POINT, execution_id))
        noise = float(self.get_current_function_param(NOISE, execution_id))
        t0 = float(self.get_current_function_param(NON_DECISION_TIME, execution_id))

        # drift_rate = float(self.drift_rate) * float(variable)
        # threshold = float(self.threshold)
        # starting_point = float(self.starting_point)
        # noise = float(self.noise)
        # t0 = float(self.t0)

        bias = (starting_point + threshold) / (2 * threshold)

        # Prevents div by 0 issue below:
        if bias <= 0:
            bias = 1e-8
        if bias >= 1:
            bias = 1 - 1e-8

        # drift_rate close to or at 0 (avoid float comparison)
        if np.abs(drift_rate) < 1e-8:
            # back to absolute bias in order to apply limit
            bias_abs = bias * 2 * threshold - threshold
            # use expression for limit a->0 from Srivastava et al. 2016
            rt = t0 + (threshold ** 2 - bias_abs ** 2) / (noise ** 2)
            er = (threshold - bias_abs) / (2 * threshold)
        else:
            drift_rate_normed = np.abs(drift_rate)
            ztilde = threshold / drift_rate_normed
            atilde = (drift_rate_normed / noise) ** 2

            is_neg_drift = drift_rate < 0
            bias_adj = (is_neg_drift == 1) * (1 - bias) + (is_neg_drift == 0) * bias
            y0tilde = ((noise ** 2) / 2) * np.log(bias_adj / (1 - bias_adj))
            if np.abs(y0tilde) > threshold:
                y0tilde = -1 * (is_neg_drift == 1) * threshold + (is_neg_drift == 0) * threshold
            x0tilde = y0tilde / drift_rate_normed

            with np.errstate(over='raise', under='raise'):
                try:
                    rt = ztilde * np.tanh(ztilde * atilde) + \
                         ((2 * ztilde * (1 - np.exp(-2 * x0tilde * atilde))) / (
                             np.exp(2 * ztilde * atilde) - np.exp(-2 * ztilde * atilde)) - x0tilde) + t0
                    er = 1 / (1 + np.exp(2 * ztilde * atilde)) - \
                         ((1 - np.exp(-2 * x0tilde * atilde)) / (
                         np.exp(2 * ztilde * atilde) - np.exp(-2 * ztilde * atilde)))

                except FloatingPointError:
                    # Per Mike Shvartsman:
                    # If ±2*ztilde*atilde (~ 2*z*a/(c^2) gets very large, the diffusion vanishes relative to drift
                    # and the problem is near-deterministic. Without diffusion, error rate goes to 0 or 1
                    # depending on the sign of the drift, and so decision time goes to a point mass on z/a – x0, and
                    # generates a "RuntimeWarning: overflow encountered in exp"
                    er = 0
                    rt = ztilde / atilde - x0tilde + t0

            # This last line makes it report back in terms of a fixed reference point
            #    (i.e., closer to 1 always means higher p(upper boundary))
            # If you comment this out it will report errors in the reference frame of the drift rate
            #    (i.e., reports p(upper) if drift is positive, and p(lower if drift is negative)
            er = (is_neg_drift == 1) * (1 - er) + (is_neg_drift == 0) * (er)

        return rt, er

    def derivative(self, output=None, input=None, execution_id=None):
        """
        derivative(output, input)

        Calculate the derivative of :math:`\\frac{1}{reward rate}` with respect to the threshold (**output** arg)
        and drift_rate (**input** arg).  Reward rate (:math:`RR`) is assumed to be:

            :math:`RR = delay_{ITI} + \\frac{Z}{A} + ED`;

        the derivative of :math:`\\frac{1}{RR}` with respect to the `threshold <BogaczEtAl.threshold>` is:

            :math:`\\frac{1}{A} - \\frac{E}{A} - 2\\frac{A}{c^2}ED`;

        and the derivative of 1/RR with respect to the `drift_rate <BogaczEtAl.drift_rate>` is:

            :math:`-\\frac{Z}{A^2} + \\frac{Z}{A^2}E - \\frac{2Z}{c^2}ED`

        where:

            *A* = `drift_rate <BogaczEtAl.drift_rate>`,

            *Z* = `threshold <BogaczEtAl.threshold>`,

            *c* = `noise <BogaczEtAl.noise>`,

            *E* = :math:`e^{-2\\frac{ZA}{c^2}}`,

            *D* = :math:`delay_{ITI} + delay_{penalty} - \\frac{Z}{A}`,

            :math:`delay_{ITI}` is the intertrial interval and :math:`delay_{penalty}` is a penalty delay.


        Returns
        -------

        derivatives :  List[float, float)
            of :math:`\\frac{1}{RR}` with respect to `threshold <BogaczEtAl.threshold>` and `drift_rate
            <BogaczEtAl.drift_rate>`.

        """
        Z = output or self.get_current_function_param(THRESHOLD, execution_id)
        A = input or self.get_current_function_param(DRIFT_RATE, execution_id)
        c = self.get_current_function_param(NOISE, execution_id)
        c_sq = c ** 2
        E = np.exp(-2 * Z * A / c_sq)
        D_iti = 0
        D_pen = 0
        D = D_iti + D_pen
        # RR =  1/(D_iti + Z/A + (E*D))

        dRR_dZ = 1 / A + E / A + (2 * A / c_sq) * E * D
        dRR_dA = -Z / A ** 2 + (Z / A ** 2) * E - (2 * Z / c_sq) * E * D

        return [dRR_dZ, dRR_dA]


# Results from Navarro and Fuss DDM solution (indices for return value tuple)
class NF_Results(IntEnum):
    MEAN_ER = 0
    MEAN_RT = 1
    MEAN_DT = 2
    COND_RTS = 3
    COND_VAR_RTS = 4
    COND_SKEW_RTS = 5


class NavarroAndFuss(IntegratorFunction):  # ----------------------------------------------------------------------------
    """
    NavarroAndFuss(                             \
        default_variable=None,                  \
        drift_rate=1.0,                         \
        threshold=1.0,                          \
        starting_point=0.0,                     \
        t0=0.2                                  \
        noise=0.5,                              \
        params=None,                            \
        owner=None,                             \
        prefs=None                              \
        )

    .. _NavarroAndFuss:

    Return terminal value of decision variable, mean accuracy, mean response time (RT), correct RT mean, correct RT
    variance and correct RT skew computed analytically for the drift diffusion process (Wiener diffusion model)
    as described in `Navarro and Fuss (2009) <http://www.sciencedirect.com/science/article/pii/S0022249609000200>`_.

    .. note::
       Use of this Function requires that the MatLab engine is installed.

    Arguments
    ---------

    default_variable : number, list or array : default ClassDefaults.variable
        specifies a template for decision variable(s);  if it is list or array, a separate solution is computed
        independently for each element.

    drift_rate : float, list or 1d array : default 1.0
        specifies the drift_rate of the drift diffusion process.  If it is a list or array,
        it must be the same length as `default_variable <BogaczEtAl.default_variable>`.

    threshold : float, list or 1d array : default 1.0
        specifies the threshold (boundary) of the drift diffusion process.  If it is a list or array,
        it must be the same length as `default_variable <BogaczEtAl.default_variable>`.

    starting_point : float, list or 1d array : default 1.0
        specifies the initial value of the decision variable for the drift diffusion process.  If it is a list or
        array, it must be the same length as `default_variable <BogaczEtAl.default_variable>`.

    noise : float, list or 1d array : default 0.0
        specifies the noise term (corresponding to the diffusion component) of the drift diffusion process.
        If it is a float, it must be a number from 0 to 1.  If it is a list or array, it must be the same length as
        `default_variable <BogaczEtAl.default_variable>` and all elements must be floats from 0 to 1.

    t0 : float, list or 1d array : default 0.2
        specifies the non-decision time for solution. If it is a float, it must be a number from 0 to 1.  If it is a
        list or array, it must be the same length as  `default_variable <BogaczEtAl.default_variable>` and all
        elements must be floats from 0 to 1.

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

    variable : number or 1d array
        holds initial value assigned to :keyword:`default_variable` argument;
        ignored by `function <NovarroAndFuss.function>`.

    drift_rate : float or 1d array
        determines the drift component of the drift diffusion process.

    threshold : float or 1d array
        determines the threshold (bound) of the drift diffusion process (i.e., at which the integration
        process is assumed to terminate).

    starting_point : float or 1d array
        determines the initial value of the decision variable for the drift diffusion process.

    noise : float or 1d array
        determines the diffusion component of the drift diffusion process (used to specify the variance of a
        Gaussian random process).

    t0 : float or 1d array
        determines the assumed non-decision time to determine the response time returned by the solution.

    bias : float or 1d array
        normalized starting point:
        (`starting_point <BogaczEtAl.starting_point>` + `threshold <BogaczEtAl.threshold>`) /
        (2 * `threshold <BogaczEtAl.threshold>`)

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

    componentName = kwNavarrosAndFuss

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    class Params(IntegratorFunction.Params):
        drift_rate = Param(1.0, modulable=True)
        starting_point = Param(0.0, modulable=True)
        threshold = Param(1.0, modulable=True)
        noise = Param(0.5, modulable=True)
        t0 = .200

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 drift_rate: parameter_spec = 1.0,
                 starting_point: parameter_spec = 0.0,
                 threshold: parameter_spec = 1.0,
                 noise: parameter_spec = 0.5,
                 t0: parameter_spec = .200,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(drift_rate=drift_rate,
                                                  starting_point=starting_point,
                                                  threshold=threshold,
                                                  noise=noise,
                                                  t0=t0,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

    def _instantiate_function(self, function, function_params=None, context=None):
        import os
        import sys
        try:
            import matlab.engine
        except ImportError as e:
            raise ImportError(
                'python failed to import matlab. Ensure that MATLAB and the python API is installed. See'
                ' https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html'
                ' for more info'
            )

        ddm_functions_path = os.path.abspath(sys.modules['psyneulink'].__path__[0] + '/../Matlab/DDMFunctions')

        # must add the package-included MATLAB files to the engine path to run when not running from the path
        # MATLAB is very finnicky about the formatting here to actually add the path so be careful if you modify
        self.eng1 = matlab.engine.start_matlab("-r 'addpath(char(\"{0}\"))' -nojvm".format(ddm_functions_path))

        super()._instantiate_function(function=function, function_params=function_params, context=context)

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        """
        Return: terminal value of decision variable, mean accuracy (error rate; ER), mean response time (RT),
        correct RT mean, correct RT variance and correct RT skew.  **Requires that the MatLab engine is installed.**

        Arguments
        ---------

        variable : 2d array
            ignored.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.


        Returns
        -------
        Decision variable, mean ER, mean RT, correct RT mean, correct RT variance, correct RT skew : \
        (float, float, float, float, float, float)

        """

        self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        drift_rate = float(self.get_current_function_param(DRIFT_RATE, execution_id))
        threshold = float(self.get_current_function_param(THRESHOLD, execution_id))
        starting_point = float(self.get_current_function_param(STARTING_POINT, execution_id))
        noise = float(self.get_current_function_param(NOISE, execution_id))
        t0 = float(self.get_current_function_param(NON_DECISION_TIME, execution_id))

        # used to pass values in a way that the matlab script can handle
        ddm_struct = {
            'z': threshold,
            'c': noise,
            'T0': t0
        }

        results = self.eng1.ddmSimFRG(drift_rate, starting_point, ddm_struct, 1, nargout=6)

        return self.convert_output_type(results)
