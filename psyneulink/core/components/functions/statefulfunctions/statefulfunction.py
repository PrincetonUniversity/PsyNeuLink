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
"""

* `StatefulFunction`
* `IntegratorFunctions`
* `MemoryFunctions`

"""

import abc
import typecheck as tc
import warnings
import numbers

import numpy as np

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.component import DefaultsFlexibility
from psyneulink.core.components.functions.function import Function_Base, FunctionError
from psyneulink.core.components.functions.distributionfunctions import DistributionFunction
from psyneulink.core.globals.keywords import INITIALIZER, STATEFUL_FUNCTION_TYPE, STATEFUL_FUNCTION, NOISE, RATE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.utilities import parameter_spec, iscompatible, object_has_single_value
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.context import ContextFlags, handle_external_context

__all__ = ['StatefulFunction']


class StatefulFunction(Function_Base): #  ---------------------------------------------------------------------
    """
    StatefulFunction(           \
        default_variable=None,  \
        initializer,            \
        rate=1.0,               \
        noise=0.0,              \
        params=None,            \
        owner=None,             \
        prefs=None,             \
        )

    .. _StatefulFunction:

    Abstract base class for Functions the result of which depend on their `previous_value
    <StatefulFunction.previous_value>` attribute.

    COMMENT:
    NARRATIVE HERE THAT EXPLAINS:
    A) initializers and stateful_attributes
    B) initializer (note singular) is a prespecified member of initializers
       that contains the value with which to initiailzer previous_value
    COMMENT


    Arguments
    ---------

    default_variable : number, list or array : default class_defaults.variable
        specifies a template for `variable <StatefulFunction.variable>`.

    initializer : float, list or 1d array : default 0.0
        specifies initial value for `previous_value <StatefulFunction.previous_value>`.  If it is a list or array,
        it must be the same length as `variable <StatefulFunction.variable>` (see `initializer
        <StatefulFunction.initializer>` for details).

    rate : float, list or 1d array : default 1.0
        specifies value used as a scaling parameter in a subclass-dependent way (see `rate <StatefulFunction.rate>` for
        details); if it is a list or array, it must be the same length as `variable <StatefulFunction.default_variable>`.

    noise : float, function, list or 1d array : default 0.0
        specifies random value added in each call to `function <StatefulFunction.function>`; if it is a list or
        array, it must be the same length as `variable <StatefulFunction.default_variable>` (see `noise
        <StatefulFunction.noise>` for details).

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
        current input value.

    initializer : float or 1d array
        determines initial value assigned to `previous_value <StatefulFunction.previous_value>`. If `variable
        <StatefulFunction.variable>` is a list or array, and initializer is a float or has a single element, it is
        applied to each element of `previous_value <StatefulFunction.previous_value>`. If initializer is a list or
        array,each element is applied to the corresponding element of `previous_value <Integrator.previous_value>`.

    previous_value : 1d array
        last value returned (i.e., for which state is being maintained).

    initializers : list
        stores the names of the initialization attributes for each of the stateful attributes of the function. The
        index i item in initializers provides the initialization value for the index i item in `stateful_attributes
        <StatefulFunction.stateful_attributes>`.

    stateful_attributes : list
        stores the names of each of the stateful attributes of the function. The index i item in stateful_attributes is
        initialized by the value of the initialization attribute whose name is stored in index i of `initializers
        <StatefulFunction.initializers>`. In most cases, the stateful_attributes, in that order, are the return values
        of the function.

    .. _Stateful_Rate:

    rate : float or 1d array
        on each call to `function <StatefulFunction.function>`, applied to `variable <StatefulFunction.variable>`,
        `previous_value <StatefulFunction.previous_value>`, neither, or both, depending on implementation by
        subclass.  If it is a float or has a single value, it is applied to all elements of its target(s);  if it has
        more than one element, each element is applied to the corresponding element of its target(s).

    .. _Stateful_Noise:

    noise : float, function, list, or 1d array
        random value added on each call to `function <StatefulFunction.function>`. If `variable
        <StatefulFunction.variable>` is a list or array, and noise is a float or function, it is applied
        for each element of `variable <StatefulFunction.variable>`. If noise is a function, it is executed and applied
        separately for each element of `variable <StatefulFunction.variable>`.  If noise is a list or array,
        it is applied elementwise (i.e., in Hadamard form).

        .. hint::
            To generate random noise that varies for every execution, a probability distribution function should be
            used (see `Distribution Functions <DistributionFunction>` for details), that generates a new noise value
            from its distribution on each execution. If noise is specified as a float, a function with a fixed
            output, or a list or array of either of these, then noise is simply an offset that remains the same
            across all executions.

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a default is
        assigned by FunctionRegistry (see `Registry_Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the Function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see `Preferences`
        for details).
    """

    componentType = STATEFUL_FUNCTION_TYPE
    componentName = STATEFUL_FUNCTION

    class Parameters(Function_Base.Parameters):
        """
            Attributes
            ----------

                initializer
                    see `initializer <StatefulFunction.initializer>`

                    :default value: numpy.array([0])
                    :type: ``numpy.ndarray``

                noise
                    see `noise <StatefulFunction.noise>`

                    :default value: 0.0
                    :type: ``float``

                previous_value
                    see `previous_value <StatefulFunction.previous_value>`

                    :default value: numpy.array([0])
                    :type: ``numpy.ndarray``

                rate
                    see `rate <StatefulFunction.rate>`

                    :default value: 1.0
                    :type: ``float``
        """
        noise = Parameter(0.0, modulable=True)
        rate = Parameter(1.0, modulable=True)
        previous_value = Parameter(np.array([0]), pnl_internal=True)
        initializer = Parameter(np.array([0]), pnl_internal=True)


    @handle_external_context()
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
                 **kwargs
                 ):

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
                initializer = self.class_defaults.variable

        super().__init__(
            default_variable=default_variable,
            rate=rate,
            initializer=initializer,
            noise=noise,
            params=params,
            owner=owner,
            prefs=prefs,
            context=context,
            **kwargs
        )

        self.has_initializers = True

    def _validate(self, context=None):
        self._validate_rate(self.defaults.rate)
        self._validate_initializers(self.defaults.variable, context=context)
        super()._validate(context=context)

    def _validate_params(self, request_set, target_set=None, context=None):

        # Handle list or array for rate specification
        if RATE in request_set:
            rate = request_set[RATE]

            if isinstance(rate, (list, np.ndarray)) and not iscompatible(rate, self.defaults.variable):
                if len(rate) != 1 and len(rate) != np.array(self.defaults.variable).size:
                    # If the variable was not specified, then reformat it to match rate specification
                    #    and assign class_defaults.variable accordingly
                    # Note: this situation can arise when the rate is parametrized (e.g., as an array) in the
                    #       StatefulFunction's constructor, where that is used as a specification for a function parameter
                    #       (e.g., for an IntegratorMechanism), whereas the input is specified as part of the
                    #       object to which the function parameter belongs (e.g., the IntegratorMechanism); in that
                    #       case, the StatefulFunction gets instantiated using its class_defaults.variable ([[0]]) before
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
                                self.defaults.variable,
                            )
                    else:
                        raise FunctionError(
                            "The length of the array specified for the rate parameter of {} ({}) "
                            "must match the length of the default input ({}).".format(
                                self.name,
                                # rate,
                                len(rate),
                                np.array(self.defaults.variable).size,
                                # self.defaults.variable,
                            )
                        )

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if NOISE in target_set:
            noise = target_set[NOISE]
            if isinstance(noise, DistributionFunction):
                noise.owner = self
                target_set[NOISE] = noise.execute
            self._validate_noise(target_set[NOISE])

    def _validate_initializers(self, default_variable, context=None):
        for initial_value_name in self.initializers:

            initial_value = self._get_current_function_param(initial_value_name, context=context)

            if isinstance(initial_value, (list, np.ndarray)):
                if len(initial_value) != 1:
                    # np.atleast_2d may not be necessary here?
                    if np.shape(np.atleast_2d(initial_value)) != np.shape(np.atleast_2d(default_variable)):
                        raise FunctionError("{}'s {} ({}) is incompatible with its default_variable ({}) ."
                                            .format(self.name, initial_value_name, initial_value, default_variable))
            elif not isinstance(initial_value, (float, int)):
                raise FunctionError("{}'s {} ({}) must be a number or a list/array of numbers."
                                    .format(self.name, initial_value_name, initial_value))

    def _validate_rate(self, rate):
        # FIX: CAN WE JUST GET RID OF THIS?
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

            if isinstance(rate, np.ndarray) and not iscompatible(rate, self.defaults.variable):
                if len(rate) != 1 and len(rate) != np.array(self.defaults.variable).size:
                    if self._variable_shape_flexibility is DefaultsFlexibility.FLEXIBLE:
                        self.defaults.variable = np.zeros_like(np.array(rate))
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
                                self.defaults.variable,
                            )
                        self._instantiate_value()
                        self._variable_shape_flexibility = DefaultsFlexibility.INCREASE_DIMENSION
                    else:
                        raise FunctionError(
                            "The length of the array specified for the rate parameter of {} ({})"
                            "must match the length of the default input ({}).".format(
                                len(rate),
                                # rate,
                                self.name,
                                np.array(self.defaults.variable).size,
                                # self.defaults.variable,
                            )
                        )

    # Ensure that the noise parameter makes sense with the input type and shape; flag any noise functions that will
    # need to be executed
    def _validate_noise(self, noise):
        # Noise is a list or array
        if isinstance(noise, (np.ndarray, list)):
            if len(noise) == 1:
                pass
            # Variable is a list/array
            elif (not iscompatible(np.atleast_2d(noise), self.defaults.variable)
                  and not iscompatible(np.atleast_1d(noise), self.defaults.variable) and len(noise) > 1):
                raise FunctionError(
                    "Noise parameter ({}) does not match default variable ({}). Noise parameter of {} "
                    "must be specified as a float, a function, or an array of the appropriate shape ({}).".format(
                        noise, self.defaults.variable, self.name,
                        np.shape(np.array(self.defaults.variable))
                    ),
                    component=self
                )
            else:
                for i in range(len(noise)):
                    if isinstance(noise[i], DistributionFunction):
                        noise[i] = noise[i].execute
                    # if not isinstance(noise[i], (float, int)) and not callable(noise[i]):
                    if not np.isscalar(noise[i]) and not callable(noise[i]):
                        raise FunctionError("The elements of a noise list or array must be scalars or functions. "
                                            "{} is not a valid noise element for {}".format(noise[i], self.name))

        # Otherwise, must be a float, int or function
        elif noise is not None and not isinstance(noise, (float, int)) and not callable(noise):
            raise FunctionError(
                "Noise parameter ({}) for {} must be a float, function, or array/list of these."
                    .format(noise, self.name))

    def _try_execute_param(self, param, var, context=None):

        # FIX: [JDC 12/18/18 - HACK TO DEAL WITH ENFORCEMENT OF 2D BELOW]
        param_shape = np.array(param).shape
        if not len(param_shape):
            param_shape = np.array(var).shape
        # param is a list; if any element is callable, execute it
        if isinstance(param, (np.ndarray, list)):
            # NOTE: np.atleast_2d will cause problems if the param has "rows" of different lengths
            # FIX: WHY FORCE 2d??
            param = np.atleast_2d(param)
            for i in range(len(param)):
                for j in range(len(param[i])):
                    if callable(param[i][j]):
                        param[i][j] = param[i][j]()
            try:
                param = param.reshape(param_shape)
            except ValueError:
                if object_has_single_value(param):
                    param = np.full(param_shape, float(param))

        # param is one function
        elif callable(param):
            # NOTE: np.atleast_2d will cause problems if the param has "rows" of different lengths
            new_param = []
            # FIX: WHY FORCE 2d??
            for row in np.atleast_2d(var):
            # for row in np.atleast_1d(var):
            # for row in var:
                new_row = []
                for item in row:
                    new_row.append(param())
                new_param.append(new_row)
            param = new_param
            # FIX: [JDC 12/18/18 - HACK TO DEAL WITH ENFORCEMENT OF 2D ABOVE]
            try:
                if len(np.squeeze(np.array(param))):
                    param = np.array(param).reshape(param_shape)
            except TypeError:
                pass

        return param

    def _instantiate_attributes_before_function(self, function=None, context=None):
        self.parameters.previous_value._set(
            self._initialize_previous_value(
                self.parameters.initializer._get(context),
                context
            ),
            context
        )

        # use np.broadcast_to to guarantee that all initializer type attributes take on the same shape as variable
        if not np.isscalar(self.defaults.variable):
            for attr in self.initializers:
                setattr(self, attr, np.broadcast_to(getattr(self, attr), self.defaults.variable.shape).copy())

        # create all stateful attributes and initialize their values to the current values of their
        # corresponding initializer attributes
        for i, attr_name in enumerate(self.stateful_attributes):
            initializer_value = getattr(self, self.initializers[i]).copy()
            setattr(self, attr_name, initializer_value)

        self.has_initializers = True

        super()._instantiate_attributes_before_function(function=function, context=context)

    def _initialize_previous_value(self, initializer, context=None):
        val = np.atleast_1d(initializer)
        if context is None:
            # Since this is run during initialization, self.parameters will refer to self.class_parameters
            # because self.parameters has not been created yet
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                self.previous_value = val
        else:
            self.parameters.previous_value.set(val, context)

        return val

    @handle_external_context(execution_id=NotImplemented)
    def reset(self, *args, context=None):
        """
            Resets `value <StatefulFunction.previous_value>`  and `previous_value <StatefulFunction.previous_value>`
            to the specified value(s).

            If arguments are passed into the reset method, then reset sets each of the attributes in
            `stateful_attributes <StatefulFunction.stateful_attributes>` to the value of the corresponding argument.
            Next, it sets the `value <StatefulFunction.value>` to a list containing each of the argument values.

            If reset is called without arguments, then it sets each of the attributes in `stateful_attributes
            <StatefulFunction.stateful_attributes>` to the value of the corresponding attribute in `initializers
            <StatefulFunction.initializers>`. Next, it sets the `value <StatefulFunction.value>` to a list containing
            the values of each of the attributes in `initializers <StatefulFunction.initializers>`.

            Often, the only attribute in `stateful_attributes <StatefulFunction.stateful_attributes>` is
            `previous_value <StatefulFunction.previous_value>` and the only attribute in `initializers
            <StatefulFunction.initializers>` is `initializer <StatefulFunction.initializer>`, in which case
            the reset method sets `previous_value <StatefulFunction.previous_value>` and `value
            <StatefulFunction.value>` to either the value of the argument (if an argument was passed into
            reset) or the current value of `initializer <StatefulFunction.initializer>`.

            For specific types of StatefulFunction functions, the reset method may carry out other
            reinitialization steps.

        """

        if context.execution_id is NotImplemented:
            context.execution_id = self.most_recent_context.execution_id

        reinitialization_values = []

        # no arguments were passed in -- use current values of initializer attributes
        if len(args) == 0 or args is None or all(arg is None for arg in args):
            for i in range(len(self.initializers)):
                initializer_name = self.initializers[i]
                reinitialization_values.append(self._get_current_function_param(initializer_name, context))

        elif len(args) == len(self.initializers):
            for i in range(len(self.initializers)):
                initializer_name = self.initializers[i]
                if args[i] is None:
                    reinitialization_values.append(self._get_current_function_param(initializer_name, context))
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
                                "reset method of {}, then a value must be passed to reset each of its "
                                "stateful_attributes: {}, in that order. Alternatively, reset may be called "
                                "without any arguments, in which case the current values of {}'s initializers: {}, will"
                                " be used to reset their corresponding stateful_attributes."
                                .format(args,
                                        self.name,
                                        self.name,
                                        stateful_attributes_string,
                                        self.name,
                                        initializers_string))

        # rebuilding value rather than simply returning reinitialization_values in case any of the stateful
        # attrs are modified during assignment
        value = []
        for i, attr in enumerate(self.stateful_attributes):
            # FIXME: HACK: Do not reinitialize random_state
            if attr != "random_state":
                setattr(self, attr, reinitialization_values[i])
                getattr(self.parameters, attr).set(reinitialization_values[i],
                                                   context, override=True)
                value.append(getattr(self, self.stateful_attributes[i]))

        self.parameters.value.set(value, context, override=True)
        return value

    def _gen_llvm_function_reset(self, ctx, builder, params, state, arg_in, arg_out, *, tags:frozenset):
        assert "reset" in tags
        for i, a in enumerate(self.stateful_attributes):
            source_ptr = pnlvm.helpers.get_param_ptr(builder, self, params, self.initializers[i])
            dest_ptr = pnlvm.helpers.get_state_ptr(builder, self, state, a)
            if source_ptr.type != dest_ptr.type:
                warnings.warn("Shape mismatch: stateful param does not match the initializer: {}({}) vs. {}({})".format(self.initializers[i], source_ptr.type, a, dest_ptr.type))
                # Take a guess that dest just has an extra dimension
                assert len(dest_ptr.type.pointee) == 1
                dest_ptr = builder.gep(dest_ptr, [ctx.int32_ty(0),
                                                  ctx.int32_ty(0)])
            builder.store(builder.load(source_ptr), dest_ptr)

        return builder

    @abc.abstractmethod
    def _function(self, *args, **kwargs):
        raise FunctionError("StatefulFunction is not meant to be called explicitly")
