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
import collections
import numbers
import warnings

import numpy as np
from beartype import beartype

from psyneulink._typing import Optional

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.component import DefaultsFlexibility, _has_initializers_setter, ComponentsMeta
from psyneulink.core.components.functions.nonstateful.distributionfunctions import DistributionFunction
from psyneulink.core.components.functions.function import Function_Base, FunctionError, _noise_setter
from psyneulink.core.globals.context import handle_external_context
from psyneulink.core.globals.keywords import STATEFUL_FUNCTION_TYPE, STATEFUL_FUNCTION, NOISE, RATE
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.preferences.basepreferenceset import ValidPrefSet
from psyneulink.core.globals.utilities import iscompatible, convert_to_np_array, contains_type

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
    B) initializer (note singular) is a pre-specified member of initializers
       that contains the value with which to initialize previous_value
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

        .. note::
            A ParameterPort for noise will only be generated, and the
            noise Parameter itself will only be stateful, if the value
            of noise is entirely numeric (contains no functions) at the
            time of Mechanism construction.

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

    # TODO: consider moving this to a Parameter attribute
    _mdf_stateful_parameter_indices = {
        'previous_value': None
    }

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
        noise = Parameter(0.0, modulable=True, setter=_noise_setter)
        rate = Parameter(1.0, modulable=True)
        previous_value = Parameter(np.array([0]), initializer='initializer')
        initializer = Parameter(np.array([0]), pnl_internal=True)
        has_initializers = Parameter(True, setter=_has_initializers_setter, pnl_internal=True)

        def _validate_noise(self, noise):
            if (
                isinstance(noise, collections.abc.Iterable)
                # assume ComponentsMeta are functions
                and contains_type(noise, ComponentsMeta)
            ):
                # TODO: make this validation unnecessary by handling automatically?
                return 'functions in a list must be instantiated and have the desired noise variable shape'

    @handle_external_context()
    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 rate=None,
                 noise=None,
                 initializer=None,
                 params: Optional[dict] = None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None,
                 context=None,
                 **kwargs
                 ):

        if not hasattr(self, "initializers"):
            self.initializers = ["initializer"]

        if not hasattr(self, "stateful_attributes"):
            self.stateful_attributes = ["previous_value"]

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
                    #       StatefulFunction's constructor, where that is used as specification for a function parameter
                    #       (e.g., for an IntegratorMechanism), whereas the input is specified as part of the
                    #       object to which the function parameter belongs (e.g., the IntegratorMechanism); in that
                    #       case, the StatefulFunction gets instantiated using its class_defaults.variable ([[0]])
                    #       before the object itself, thus does not see the array specification for the input.
                    if self._variable_shape_flexibility is DefaultsFlexibility.FLEXIBLE:
                        self._instantiate_defaults(variable=np.zeros_like(np.array(rate)), context=context)
                        if self.verbosePref:
                            warnings.warn(f"The length ({len(rate)}) of the array specified for "
                                          f"the rate parameter ({rate}) of {self.name} must match the length "
                                          f"({np.array(self.defaults.variable).size}) of the default input "
                                          f"({self.defaults.variable}); the default input has been updated to match.")
                    else:
                        raise FunctionError(f"The length of the array specified for the rate parameter of {self.name}"
                                            f"({len(rate)}) must match the length of the default input "
                                            f"({np.array(self.defaults.variable).size}).")

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

            initial_value = self._get_current_parameter_value(initial_value_name, context=context)

            if isinstance(initial_value, (list, np.ndarray)):
                if len(initial_value) != 1:
                    # np.atleast_2d may not be necessary here?
                    if np.shape(np.atleast_2d(initial_value)) != np.shape(np.atleast_2d(default_variable)):
                        raise FunctionError(f"{self.name}'s {initial_value_name} ({initial_value}) is incompatible "
                                            f"with its default_variable ({default_variable}).")
            elif not isinstance(initial_value, (float, int)):
                raise FunctionError(f"{self.name}'s {initial_value_name} ({initial_value}) "
                                    f"must be a number or a list/array of numbers.")

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
                            warnings.warn(f"The length ({len(rate)}) of the array specified for the rate parameter "
                                          f"({rate}) of {self.name} must match the length "
                                          f"({np.array(self.defaults.variable).size}) of the default input "
                                          f"({self.defaults.variable}); the default input has been updated to match.")
                        self._instantiate_value()
                        self._variable_shape_flexibility = DefaultsFlexibility.INCREASE_DIMENSION
                    else:
                        raise FunctionError(f"The length of the array specified for the rate parameter of "
                                            f"{len(rate)} ({self.name}) must match the length of the default input "
                                            f"({np.array(self.defaults.variable).size}).")

    # Ensure that the noise parameter makes sense with the input type and shape; flag any noise functions that will
    # need to be executed
    def _validate_noise(self, noise):
        # Noise must be a scalar, list, array or Distribution Function

        if isinstance(noise, DistributionFunction):
            noise = noise.execute

        if isinstance(noise, (np.ndarray, list)):
            if len(noise) == 1:
                pass
            # Variable is a list/array
            elif (not iscompatible(np.atleast_2d(noise), self.defaults.variable)
                  and not iscompatible(np.atleast_1d(noise), self.defaults.variable) and len(noise) > 1):
                raise FunctionError(f"Noise parameter ({noise})  for '{self.name}' does not match default variable "
                                    f"({self.defaults.variable}); it must be specified as a float, a function, "
                                    f"or an array of the appropriate shape "
                                    f"({np.shape(np.array(self.defaults.variable))}).",
                    component=self)
            else:
                for i in range(len(noise)):
                    if isinstance(noise[i], DistributionFunction):
                        noise[i] = noise[i].execute
                    if (not np.isscalar(noise[i]) and not callable(noise[i])
                            and not iscompatible(np.atleast_2d(noise[i]), self.defaults.variable[i])
                            and not iscompatible(np.atleast_1d(noise[i]), self.defaults.variable[i])):
                        raise FunctionError(f"The element '{noise[i]}' specified in 'noise' for {self.name} "
                                             f"is not valid; noise must be list or array must be floats or functions.")

    def _instantiate_attributes_before_function(self, function=None, context=None):
        if not self.parameters.initializer._user_specified:
            self._initialize_previous_value(np.zeros_like(self.defaults.variable), context)
        self._instantiate_stateful_attributes(self.stateful_attributes, self.initializers, context)
        super()._instantiate_attributes_before_function(function=function, context=context)

    def _instantiate_stateful_attributes(self, stateful_attributes:list, initializers:list, context) -> None:
        # use np.broadcast_to to guarantee that all initializer type attributes take on the same shape as variable
        if not np.isscalar(self.defaults.variable):
            for attr in initializers:
                param = getattr(self.parameters, attr)
                param._set(
                    np.broadcast_to(
                        param._get(context),
                        self.defaults.variable.shape
                    ).copy(),
                    context
                )

        # create all stateful attributes and initialize their values to the current values of their
        # corresponding initializer attributes
        for attr_name in stateful_attributes:
            initializer_value = getattr(self.parameters, getattr(self.parameters, attr_name).initializer)._get(context).copy()
            getattr(self.parameters, attr_name)._set(initializer_value, context)

    def _initialize_previous_value(self, initializer, context=None):
        initializer = convert_to_np_array(initializer, dimension=1)

        self.defaults.initializer = initializer.copy()
        self.parameters.initializer._set(initializer.copy(), context)

        self.defaults.previous_value = initializer.copy()
        self.parameters.previous_value.set(initializer.copy(), context)

        return initializer

    @handle_external_context()
    def _update_default_variable(self, new_default_variable, context=None):
        if not self.parameters.initializer._user_specified:
            self._initialize_previous_value(np.zeros_like(new_default_variable), context)

        super()._update_default_variable(new_default_variable, context=context)

    def _parse_value_order(self, **kwargs):
        """
            Returns:
                tuple: the values of the keyword arguments in the order
                in which they appear in this Component's `value
                <Component.value>`
        """
        return tuple(v for k, v in kwargs.items())

    @handle_external_context(fallback_most_recent=True)
    def reset(self, *args, context=None, **kwargs):
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
        num_stateful_attrs = len(self.stateful_attributes)
        if num_stateful_attrs >= 2:
            # old args specification can be supported only in subclasses
            # that explicitly define an order by overriding reset
            if len(args) > 0:
                raise FunctionError(f'{self}.reset has more than one stateful attribute'
                                    f' ({self.stateful_attributes}). You must specify reset values by keyword.')
            if len(kwargs) != num_stateful_attrs:
                type_name = type(self).__name__
                raise FunctionError(f'StatefulFunction.reset must receive a keyword argument for'
                                    f' each item in {type_name}.stateful_attributes in the order in'
                                    f' which they appear in {type_name}.value.')

        if num_stateful_attrs == 1:
            try:
                kwargs[self.stateful_attributes[0]]
            except KeyError:
                try:
                    kwargs[self.stateful_attributes[0]] = args[0]
                except IndexError:
                    kwargs[self.stateful_attributes[0]] = None

        invalid_args = []

        # iterates in order arguments are sent in function call, so it
        # will match their order in value as long as they are listed
        # properly in subclass reset method signatures
        for attr in kwargs:
            try:
                kwargs[attr]
            except KeyError:
                kwargs[attr] = None

            if kwargs[attr] is not None:
                # from before: unsure if conversion to 1d necessary
                kwargs[attr] = np.atleast_1d(kwargs[attr])
            else:
                try:
                    initializer_ref = getattr(self.parameters, attr).initializer
                    if initializer_ref:
                        initializer = getattr(self.parameters, initializer_ref)
                    # FIX: ?NEED TO HANDLE initializer IF IT IS A NUMBER?
                    if initializer is not None and initializer.port and initializer.port.mod_afferents:
                        # If the initializer is subject to control, get its control_allocation
                        initializer_mod_proj = initializer.port.mod_afferents[0]
                        mod_parameter_source = initializer_mod_proj.sender.owner
                        from psyneulink.core.compositions.composition import CompositionInterfaceMechanism
                        from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism \
                            import ControlMechanism
                        if isinstance(mod_parameter_source, CompositionInterfaceMechanism):
                            ctl_sig,_,_  = mod_parameter_source._get_source_of_modulation_for_parameter_CIM(
                                initializer_mod_proj.sender)
                        elif isinstance(mod_parameter_source, ControlMechanism):
                            ctl_sig = mod_parameter_source.control_signals[0]
                        else:
                            assert False, f"Cannot reset {self.name} because " \
                                          f"the source of modulation is not of correct type."
                        kwargs[attr] = ctl_sig.parameters.value.get(context)
                    else:
                        # Otherwise, just use the default (or user-assigned) initializer
                        kwargs[attr] = self._get_current_parameter_value(initializer, context=context)

                except AttributeError:
                    invalid_args.append(attr)

        if len(invalid_args) > 0:
            raise FunctionError(f'Arguments {invalid_args} to reset are invalid because they do'
                                f" not correspond to any of {self}'s stateful_attributes.")

        # rebuilding value rather than simply returning reinitialization_values in case any of the stateful
        # attrs are modified during assignment
        value = []
        for attr, v in kwargs.items():
            # FIXME: HACK: Do not reinitialize random_state
            if attr != "random_state":
                getattr(self.parameters, attr).set(kwargs[attr],
                                                   context, override=True)
                value.append(getattr(self.parameters, attr)._get(context))

        self.parameters.value.set(value, context, override=True)
        return value

    def _gen_llvm_function_reset(self, ctx, builder, params, state, arg_in, arg_out, *, tags:frozenset):
        assert "reset" in tags
        for a in self.stateful_attributes:
            initializer = getattr(self.parameters, a).initializer
            source_ptr = ctx.get_param_or_state_ptr(builder, self, initializer, param_struct_ptr=params)
            dest_ptr = ctx.get_param_or_state_ptr(builder, self, a, state_struct_ptr=state)
            builder.store(builder.load(source_ptr), dest_ptr)

        return builder

    @abc.abstractmethod
    def _function(self, *args, **kwargs):
        raise FunctionError("StatefulFunction is not meant to be called explicitly")
