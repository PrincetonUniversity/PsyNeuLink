#
# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *****************************************  STATEFUL FUNCTION *********************************************************
'''

* `StatefulFunction`
'''

import numpy as np
import typecheck as tc
import itertools

from psyneulink.core.components.functions.function import Function_Base, FunctionError
from psyneulink.core.components.functions.distributionfunctions import DistributionFunction
from psyneulink.core.globals.keywords import INTEGRATOR_FUNCTION_TYPE, INTEGRATOR_FUNCTION, INITIALIZER
from psyneulink.core.globals.utilities import parameter_spec, iscompatible
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set

__all__ = ['IntegratorFunction']


# FIX: √ RENAME AS StatefulFunction
#      THEN RENAME Integrator AS IntegratorFunction
#           BUT EXCLUDE COMMENTS (so that docstring refs can be left as IntegratorFunction)
#      THEN CHECK COMMENTS FOR APPROPRIATENESS OF IntegratorFunction REFERENCES
#      THEN EDIT StatefulFunction AND IntegratorFunction TO BE COMPLEMENTARY
#      THEN MOVE Buffer AND DND TO THEIR OWN MemoryFunctions MODULE (WITH MemoryFunctions SUBCLASS OF StatefulFunctions)


class StatefulFunction(Function_Base):
    # -------------------------------------------------------------------------------
    """
    StatefulFunction(           \
        default_variable=None,  \
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

    componentType = INTEGRATOR_FUNCTION_TYPE
    componentName = INTEGRATOR_FUNCTION

    class Params(Function_Base.Params):
        """
            Attributes
            ----------

        """
        previous_value = np.array([0])
        initializer = np.array([0])

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

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


