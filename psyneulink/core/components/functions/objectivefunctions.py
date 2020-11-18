 #
# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *****************************************  OBJECTIVE FUNCTIONS *******************************************************
"""

* `Stability`
* `Energy`
* `Entropy`
* `Distance`

Functions that return a scalar evaluation of their input.

"""

import functools

import numpy as np
import typecheck as tc
import types

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.component import DefaultsFlexibility
from psyneulink.core.components.functions.function import EPSILON, FunctionError, Function_Base, get_matrix
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import \
    CORRELATION, COSINE, CROSS_ENTROPY, \
    DEFAULT_VARIABLE, DIFFERENCE, DISTANCE_FUNCTION, DISTANCE_METRICS, DistanceMetrics, \
    ENERGY, ENTROPY, EUCLIDEAN, HOLLOW_MATRIX, MATRIX, MAX_ABS_DIFF, METRIC, \
    NORMED_L0_SIMILARITY, OBJECTIVE_FUNCTION_TYPE, SIZE, STABILITY_FUNCTION
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.utilities import is_distance_metric, safe_len, convert_to_np_array
from psyneulink.core.globals.utilities import is_iterable


__all__ = ['ObjectiveFunction', 'Stability', 'Distance', 'Energy', 'Entropy']


class ObjectiveFunction(Function_Base):
    """Abstract class of `Function` used for evaluating ports.
    """

    componentType = OBJECTIVE_FUNCTION_TYPE

    class Parameters(Function_Base.Parameters):
        """
            Attributes
            ----------

                metric
                    see `metric <ObjectiveFunction.metric>`

                    :default value: None
                    :type:

                normalize
                    see `normalize <ObjectiveFunction.normalize>`

                    :default value: False
                    :type: ``bool``
        """
        normalize = Parameter(False, stateful=False)
        metric = Parameter(None, stateful=False)


class Stability(ObjectiveFunction):
    """
    Stability(                                  \
        default_variable=None,                  \
        size=None,                              \
        matrix=HOLLOW_MATRIX,                   \
        metric=ENERGY                           \
        transfer_fct=None                       \
        normalize=False,                        \
        params=None,                            \
        owner=None,                             \
        prefs=None                              \
        )

    .. _StabilityFunction:

    Return the stability of `variable <Stability.variable>` based on a state transformation matrix.

    The value of `variable <Stability.variable>` is passed through the `matrix <Stability.matrix>`,
    transformed using the `transfer_fct <Stability.transfer_fct>` (if specified), and then compared with its initial
    value using the `distance metric <DistanceMetric>` specified by `metric <Stability.metric>`.  If `normalize
    <Stability.normalize>` is `True`, the result is normalized by the length of (number of elements in) `variable
    <Stability.variable>`.

    Arguments
    ---------

    variable : list or 1d array of numbers: Default class_defaults.variable
        specifies shape and default value of the array for which stability is calculated.

    size : int : None
        specifies length of the array over which stability is calculated;  can be used in place of default_value,
        in which case zeros are assigned as the value(s). An error is generated if both are specified but
        size != len(default_value).

    matrix : list, np.ndarray, np.matrix, or matrix keyword : default HOLLOW_MATRIX
        specifies the matrix of recurrent weights;  must be a square matrix with the same width as the
        length of `variable <Stability.variable>`.

    metric : keyword in DistanceMetrics : Default ENERGY
        specifies a `metric <DistanceMetrics>` from `DistanceMetrics` used to compute stability.

    transfer_fct : function or method : Default None
        specifies the function used to transform output of weight `matrix <Stability.matrix>`.

    normalize : bool : Default False
        specifies whether to normalize the stability value by the length of `variable <Stability.variable>`.

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
        array for which stability is calculated.

    size : int
        length of array for which stability is calculated.

    matrix : list, np.ndarray, np.matrix, function keyword, or MappingProjection : default HOLLOW_MATRIX
        weight matrix from each element of `variable <Stability.variablity>` to each other;  if a matrix other
        than HOLLOW_MATRIX is assigned, it is convolved with HOLLOW_MATRIX to eliminate self-connections from the
        stability calculation.

    metric : keyword in DistanceMetrics
        metric used to compute stability; must be a `DistanceMetrics` keyword. The `Distance` Function is used to
        compute the stability of `variable <Stability.variable>` with respect to its value after its transformation
        by `matrix <Stability.matrix>` and `transfer_fct <Stability.transfer_fct>`.

    transfer_fct : function or method
        function used to transform output of weight `matrix <Stability.matrix>` prior to computing stability.

    normalize : bool
        if `True`, result of stability calculation is normalized by the length of `variable <Stability.variable>`.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).
    """

    componentName = STABILITY_FUNCTION

    class Parameters(ObjectiveFunction.Parameters):
        """
            Attributes
            ----------

                matrix
                    see `matrix <Stability.matrix>`

                    :default value: `HOLLOW_MATRIX`
                    :type: ``str``

                metric
                    see `metric <Stability.metric>`

                    :default value: `ENERGY`
                    :type: ``str``

                metric_fct
                    see `metric_fct <Stability.metric_fct>`

                    :default value: None
                    :type:

                transfer_fct
                    see `transfer_fct <Stability.transfer_fct>`

                    :default value: None
                    :type:
        """
        matrix = HOLLOW_MATRIX
        metric = Parameter(ENERGY, stateful=False)
        metric_fct = Parameter(None, stateful=False, loggable=False)
        transfer_fct = Parameter(None, stateful=False, loggable=False)
        normalize = Parameter(False, stateful=False)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 matrix=None,
                 # metric:is_distance_metric=None,
                 metric: tc.optional(tc.any(tc.enum(ENERGY, ENTROPY), is_distance_metric)) = None,
                 transfer_fct: tc.optional(tc.optional(tc.any(types.FunctionType, types.MethodType))) = None,
                 normalize: tc.optional(bool) = None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

        if size:
            if default_variable is None:
                default_variable = np.zeros(size)
            elif size != len(default_variable):
                raise FunctionError(f"Both {repr(DEFAULT_VARIABLE)} ({default_variable}) and {repr(SIZE)} ({size}) "
                                    f"are specified for {self.name} but are {SIZE}!=len({DEFAULT_VARIABLE}).")

        super().__init__(
            default_variable=default_variable,
            matrix=matrix,
            metric=metric,
            transfer_fct=transfer_fct,
            normalize=normalize,
            params=params,
            owner=owner,
            prefs=prefs,
        )

        # MODIFIED 6/12/19 NEW: [JDC]
        self._variable_shape_flexibility = DefaultsFlexibility.FLEXIBLE
        # MODIFIED 6/12/19 END

    def _validate_variable(self, variable, context=None):
        """Validates that variable is 1d array
        """
        if len(np.atleast_2d(variable)) != 1:
            raise FunctionError("Variable for {} must contain a single array or list of numbers".format(self.name))
        return variable

    def _validate_params(self, variable, request_set, target_set=None, context=None):
        """Validate matrix param

        `matrix <Stability.matrix>` argument must be one of the following
            - 2d list, np.ndarray or np.matrix
            - ParameterPort for one of the above
            - MappingProjection with a parameterPorts[MATRIX] for one of the above

        Parse matrix specification to insure it resolves to a square matrix
        (but leave in the form in which it was specified so that, if it is a ParameterPort or MappingProjection,
         its current value can be accessed at runtime (i.e., it can be used as a "pointer")
        """

        # Validate matrix specification
        # (str can be automatically transformed to variable shape)
        if MATRIX in target_set and not isinstance(target_set[MATRIX], str):

            from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
            from psyneulink.core.components.ports.parameterport import ParameterPort

            matrix = target_set[MATRIX]

            if isinstance(matrix, MappingProjection):
                try:
                    matrix = matrix._parameter_ports[MATRIX].value
                    param_type_string = "MappingProjection's ParameterPort"
                except KeyError:
                    raise FunctionError("The MappingProjection specified for the {} arg of {} ({}) must have a {} "
                                        "ParameterPort that has been assigned a 2d array or matrix".
                                        format(MATRIX, self.name, matrix.shape, MATRIX))

            elif isinstance(matrix, ParameterPort):
                try:
                    matrix = matrix.value
                    param_type_string = "ParameterPort"
                except KeyError:
                    raise FunctionError("The value of the {} parameterPort specified for the {} arg of {} ({}) "
                                        "must be a 2d array or matrix".
                                        format(MATRIX, MATRIX, self.name, matrix.shape))

            else:
                param_type_string = "array or matrix"

            matrix = np.array(matrix)
            if matrix.ndim != 2:
                raise FunctionError("The value of the {} specified for the {} arg of {} ({}) "
                                    "must be a 2d array or matrix".
                                    format(param_type_string, MATRIX, self.name, matrix))
            rows = matrix.shape[0]
            cols = matrix.shape[1]

            # this mirrors the transformation in _function
            # it is a hack, and a general solution should be found
            squeezed = np.array(self.defaults.variable)
            if squeezed.ndim > 1:
                squeezed = np.squeeze(squeezed)

            size = safe_len(squeezed)

            if rows != size:
                raise FunctionError("The value of the {} specified for the {} arg of {} is the wrong size;"
                                    "it is {}x{}, but must be square matrix of size {}".
                                    format(param_type_string, MATRIX, self.name, rows, cols, size))

            if rows != cols:
                raise FunctionError("The value of the {} specified for the {} arg of {} ({}) "
                                    "must be a square matrix".
                                    format(param_type_string, MATRIX, self.name, matrix))

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

    def _parse_metric_fct_variable(self, variable):
        return convert_to_np_array([variable, variable])

    def _instantiate_attributes_before_function(self, function=None, context=None):
        """Instantiate matrix

        Specified matrix is convolved with HOLLOW_MATRIX
            to eliminate the diagonal (self-connections) from the calculation.
        The `Distance` Function is used for all calculations except ENERGY (which is not really a distance metric).
        If ENTROPY is specified as the metric, convert to CROSS_ENTROPY for use with the Distance Function.
        :param function:

        """
        from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
        from psyneulink.core.components.ports.parameterport import ParameterPort

        # this mirrors the transformation in _function
        # it is a hack, and a general solution should be found
        squeezed = np.array(self.defaults.variable)
        if squeezed.ndim > 1:
            squeezed = np.squeeze(squeezed)

        size = safe_len(squeezed)

        matrix = self.parameters.matrix._get(context)

        if isinstance(matrix, MappingProjection):
            matrix = matrix._parameter_ports[MATRIX]
        elif isinstance(matrix, ParameterPort):
            pass
        else:
            matrix = get_matrix(matrix, size, size)

        self.parameters.matrix._set(matrix, context)

        self._hollow_matrix = get_matrix(HOLLOW_MATRIX, size, size)

        default_variable = [self.defaults.variable,
                            self.defaults.variable]

        if self.metric == ENTROPY:
            self.metric_fct = Distance(default_variable=default_variable, metric=CROSS_ENTROPY, normalize=self.normalize)
        elif self.metric in DISTANCE_METRICS._set():
            self.metric_fct = Distance(default_variable=default_variable, metric=self.metric, normalize=self.normalize)
        else:
            assert False, "Unknown metric"
        #FIXME: This is a hack to make sure metric-fct param is set
        self.parameters.metric_fct.set(self.metric_fct)

    def _update_default_variable(self, new_default_variable, context):
        from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
        from psyneulink.core.components.ports.parameterport import ParameterPort

        # this mirrors the transformation in _function
        # it is a hack, and a general solution should be found
        squeezed = np.array(new_default_variable)
        if squeezed.ndim > 1:
            squeezed = np.squeeze(squeezed)

        size = safe_len(squeezed)
        matrix = self.parameters.matrix._get(context)

        if isinstance(matrix, MappingProjection):
            matrix = matrix._parameter_ports[MATRIX]
        elif isinstance(matrix, ParameterPort):
            pass
        else:
            matrix = get_matrix(self.defaults.matrix, size, size)

        self.parameters.matrix._set(matrix, context)

        self._hollow_matrix = get_matrix(HOLLOW_MATRIX, size, size)

        super()._update_default_variable(new_default_variable, context)

    def _gen_llvm_function_body(self, ctx, builder, params, state, arg_in, arg_out, *, tags:frozenset):
        # Dot product
        dot_out = builder.alloca(arg_in.type.pointee)
        matrix = pnlvm.helpers.get_param_ptr(builder, self, params, MATRIX)

        # Convert array pointer to pointer to the fist element
        matrix = builder.gep(matrix, [ctx.int32_ty(0), ctx.int32_ty(0)])
        vec_in = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(0)])
        vec_out = builder.gep(dot_out, [ctx.int32_ty(0), ctx.int32_ty(0)])

        input_length = ctx.int32_ty(arg_in.type.pointee.count)
        output_length = ctx.int32_ty(arg_in.type.pointee.count)
        builtin = ctx.import_llvm_function("__pnl_builtin_vxm")
        builder.call(builtin, [vec_in, matrix, input_length, output_length, vec_out])

        # Prepare metric function
        metric_fun = ctx.import_llvm_function(self.metric_fct)
        metric_in = builder.alloca(metric_fun.args[2].type.pointee)

        # Transfer Function if configured
        if self.transfer_fct is not None:
            #FIXME: implement this
            assert False, "Support for transfer functions is not implemented"
        else:
            trans_out = builder.gep(metric_in, [ctx.int32_ty(0), ctx.int32_ty(1)])
            builder.store(builder.load(dot_out), trans_out)

        # Copy original variable
        builder.store(builder.load(arg_in), builder.gep(metric_in, [ctx.int32_ty(0), ctx.int32_ty(0)]))

        # Distance Function
        metric_params = pnlvm.helpers.get_param_ptr(builder, self, params, "metric_fct")
        metric_state = pnlvm.helpers.get_state_ptr(builder, self, state, "metric_fct")
        metric_out = arg_out
        builder.call(metric_fun, [metric_params, metric_state, metric_in, metric_out])
        return builder

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """Calculate the stability of `variable <Stability.variable>`.

        Compare the value of `variable <Stability.variable>` with its value after transformation by
        `matrix <Stability.matrix>` and `transfer_fct <Stability.transfer_fct>` (if specified), using the specified
        `metric <Stability.metric>`.  If `normalize <Stability.normalize>` is `True`, the result is divided
        by the length of `variable <Stability.variable>`.

        Returns
        -------

        stability : scalar

        """

        # MODIFIED 6/12/19 NEW: [JDC]
        variable = np.array(variable)
        if variable.ndim > 1:
            variable = np.squeeze(variable)
        # MODIFIED 6/12/19 END

        matrix = self._get_current_parameter_value(MATRIX, context)

        current = variable

        transformed = np.dot(matrix * self._hollow_matrix, variable)
        if self.transfer_fct is not None:
            transformed = self.transfer_fct(transformed)

        result = self.metric_fct(variable=[current, transformed], context=context)

        return self.convert_output_type(result)


class Energy(Stability):
    """
    Energy(                           \
        default_variable=None,        \
        size=None,                    \
        matrix=INVERSE_HOLLOW_MATRIX, \
        transfer_fct=None             \
        normalize=False,              \
        params=None,                  \
        owner=None,                   \
        prefs=None                    \
        )

    .. _EnergyFunction:

    Subclass of `Stability` Function that returns the energy of an array.

    Arguments
    ---------

    variable : list or 1d array of numbers: Default class_defaults.variable
        specifies shape and default value of the array for which energy is calculated.

    size : int : None
        specifies length of the array over which energy is calculated;  can be used in place of default_value,
        in which case zeros are assigned as the value(s). An error is generated if both are specified but
        size != len(default_value).

    matrix : list, np.ndarray, np.matrix, or matrix keyword : default INVERSE_HOLLOW_MATRIX
        specifies the matrix of recurrent weights;  must be a square matrix with the same width as the
        length of `variable <Stability.variable>`.

    transfer_fct : function or method : Default None
        specifies the function used to transform output of `matrix <Stability.matrix>` prior to the energy calculation
        (see `Stablility <Stability>` for explanation).

    normalize : bool : Default False
        specifies whether to normalize the energy value by the length of `variable <Stability.variable>`.

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
        array for which energy is calculated.

    size : int
        length of array for which energy is calculated.

    matrix : list, np.ndarray, np.matrix, or matrix keyword
        weight matrix from each element of `variable <Energy.variablity>` to each other;  if a matrix other
        than INVERSE_HOLLOW_MATRIX is assigned, it is convolved with HOLLOW_MATRIX to eliminate self-connections from
        the energy calculation.

    transfer_fct : function or method
        function used to transform output of `matrix <Stability.matrix>` prior to computing energy
        (see `Stability` for explanation).

    normalize : bool
        if `True`, result of energy calculation is normalized by the length of `variable <Energy.variable>`.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).
    """

    def __init__(self,
                 default_variable=None,
                 size=None,
                 normalize:bool=None,
                 # transfer_fct=None,
                 matrix=None,
                 params=None,
                 owner=None,
                 prefs=None):

        super().__init__(
            default_variable=default_variable,
            size=size,
                         metric=ENERGY,
                         matrix=matrix,
                         # transfer_fct=transfer_fct,
                         normalize=normalize,
                         params=params,
                         owner=owner,
                         prefs=prefs)


class Entropy(Stability):
    """
    Entropy(                          \
        default_variable=None,        \
        size=None,                    \
        matrix=INVERSE_HOLLOW_MATRIX, \
        transfer_fct=None             \
        normalize=False,              \
        params=None,                  \
        owner=None,                   \
        prefs=None                    \
        )

    .. _EntropyFunction:

    Subclass of `Stability` Function that returns the entropy of an array.

    Arguments
    ---------

    variable : list or 1d array of numbers: Default class_defaults.variable
        specifies shape and default value of the array for which entropy is calculated.

    size : int : None
        specifies length of the array over which entropy is calculated;  can be used in place of default_value,
        in which case zeros are assigned as the value(s). An error is generated if both are specified but
        size != len(default_value).

    matrix : list, np.ndarray, np.matrix, or matrix keyword : default INVERSE_HOLLOW_MATRIX
        specifies the matrix of recurrent weights;  must be a square matrix with the same width as the
        length of `variable <Stability.variable>`.

    transfer_fct : function or method : Default None
        specifies the function used to transform output of `matrix <Stability.matrix>` prior to the entropy calculation
        (see `Stability` for explanation).

    normalize : bool : Default False
        specifies whether to normalize the entropy value by the length of `variable <Stability.variable>`.

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
        array for which entropy is calculated.

    size : int
        length of array for which energy is calculated.

    matrix : list, np.ndarray, np.matrix, or matrix keyword
        weight matrix from each element of `variable <Entropy.variablity>` to each other;  if a matrix other
        than INVERSE_HOLLOW_MATRIX is assigned, it is convolved with HOLLOW_MATRIX to eliminate self-connections from
        the entropy calculation.

    transfer_fct : function or method
        function used to transform output of `matrix <Stability.matrix>` prior to computing entropy
        (see `Stability` for explanation).

    normalize : bool
        if `True`, result of entropy calculation is normalized by the length of `variable <Entropy.variable>`.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).
    """

    def __init__(self,
                 default_variable=None,
                 normalize:bool=None,
                 transfer_fct=None,
                 params=None,
                 owner=None,
                 prefs=None):

        super().__init__(
            default_variable=default_variable,
            # matrix=matrix,
                         metric=ENTROPY,
                         transfer_fct=transfer_fct,
                         normalize=normalize,
                         params=params,
                         owner=owner,
                         prefs=prefs)


class Distance(ObjectiveFunction):
    """
    Distance(                                    \
       default_variable=None,                    \
       metric=EUCLIDEAN                          \
       normalize=False,                          \
       params=None,                              \
       owner=None,                               \
       prefs=None                                \
       )

    .. _Distance:

    Return the distance between the vectors in the two items of `variable <Distance.variable>` using the `distance
    metric <DistanceMetrics>` specified in the `metric <Stability.metric>` attribute.  If `normalize
    <Distance.normalize>` is `True`, the result is normalized by the length of (number of elements in) `variable
    <Stability.variable>`.

    Arguments
    ---------

    variable : 2d array with two items : Default class_defaults.variable
        the arrays between which the distance is calculated.

    metric : keyword in DistancesMetrics : Default EUCLIDEAN
        specifies a `distance metric <DistanceMetrics>` used to compute the distance between the two items in `variable
        <Distance.variable>`.

    normalize : bool : Default False
        specifies whether to normalize the distance by the length of `variable <Distance.variable>`.

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

    variable : 2d array with two items
        contains the arrays between which the distance is calculated.

    metric : keyword in DistanceMetrics
        determines the `metric <DistanceMetrics>` used to compute the distance between the two items in `variable
        <Distance.variable>`.

    normalize : bool
        determines whether the distance is normalized by the length of `variable <Distance.variable>`.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).
    """

    componentName = DISTANCE_FUNCTION

    class Parameters(ObjectiveFunction.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <Distance.variable>`

                    :default value: numpy.array([[0], [0]])
                    :type: ``numpy.ndarray``
                    :read only: True

                metric
                    see `metric <Distance.metric>`

                    :default value: `DIFFERENCE`
                    :type: ``str``
        """
        variable = Parameter(np.array([[0], [0]]), read_only=True, pnl_internal=True, constructor_argument='default_variable')
        metric = Parameter(DIFFERENCE, stateful=False)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 metric: tc.optional(DistanceMetrics._is_metric) = None,
                 normalize: tc.optional(bool) = None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):
        super().__init__(
            default_variable=default_variable,
            metric=metric,
            normalize=normalize,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _validate_params(self, request_set, target_set=None, variable=None, context=None):
        """Validate that variable had two items of equal length

        """
        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        err_two_items = FunctionError("variable for {} ({}) must have two items".format(self.name, variable))

        try:
            if len(variable) != 2:
                raise err_two_items
        except TypeError:
            raise err_two_items

        try:
            if len(variable[0]) != len(variable[1]):
                raise FunctionError(
                    "The lengths of the items in the variable for {0} ({1},{2}) must be equal".format(
                        self.name,
                        variable[0],
                        variable[1]
                    )
                )
        except TypeError:
            if is_iterable(variable[0]) ^ is_iterable(variable[1]):
                raise FunctionError(
                    "The lengths of the items in the variable for {0} ({1},{2}) must be equal".format(
                        self.name,
                        variable[0],
                        variable[1]
                    )
                )

    def cosine(v1, v2):
        numer = np.sum(v1 * v2)
        denom = np.sqrt(np.sum(v1 ** 2)) * np.sqrt(np.sum(v2 ** 2)) or EPSILON
        return numer / denom

    def correlation(v1, v2):
        v1_norm = v1 - np.mean(v1)
        v2_norm = v2 - np.mean(v2)
        denom = np.sqrt(np.sum(v1_norm ** 2) * np.sum(v2_norm ** 2)) or EPSILON
        return np.sum(v1_norm * v2_norm) / denom

    def __gen_llvm_sum_difference(self, builder, index, ctx, v1, v2, acc):
        ptr1 = builder.gep(v1, [index])
        ptr2 = builder.gep(v2, [index])
        val1 = builder.load(ptr1)
        val2 = builder.load(ptr2)

        sub = builder.fsub(val1, val2)
        ltz = builder.fcmp_ordered("<", sub, ctx.float_ty(0))
        abs_val = builder.select(ltz, pnlvm.helpers.fneg(builder, sub), sub)
        acc_val = builder.load(acc)
        new_acc = builder.fadd(acc_val, abs_val)
        builder.store(new_acc, acc)

    def __gen_llvm_sum_diff_squares(self, builder, index, ctx, v1, v2, acc):
        ptr1 = builder.gep(v1, [index])
        ptr2 = builder.gep(v2, [index])
        val1 = builder.load(ptr1)
        val2 = builder.load(ptr2)

        sub = builder.fsub(val1, val2)
        sqr = builder.fmul(sub, sub)
        acc_val = builder.load(acc)
        new_acc = builder.fadd(acc_val, sqr)
        builder.store(new_acc, acc)

    def __gen_llvm_cross_entropy(self, builder, index, ctx, v1, v2, acc):
        ptr1 = builder.gep(v1, [index])
        ptr2 = builder.gep(v2, [index])
        val1 = builder.load(ptr1)
        val2 = builder.load(ptr2)

        log_f = ctx.get_builtin("log", [ctx.float_ty])
        log = builder.call(log_f, [val2])
        prod = builder.fmul(val1, log)

        acc_val = builder.load(acc)
        new_acc = builder.fsub(acc_val, prod)
        builder.store(new_acc, acc)

    def __gen_llvm_sum_product(self, builder, index, ctx, v1, v2, acc):
        ptr1 = builder.gep(v1, [index])
        ptr2 = builder.gep(v2, [index])
        val1 = builder.load(ptr1)
        val2 = builder.load(ptr2)

        prod = builder.fmul(val1, val2)

        acc_val = builder.load(acc)
        new_acc = builder.fadd(acc_val, prod)
        builder.store(new_acc, acc)

    def __gen_llvm_cosine(self, builder, index, ctx, v1, v2, numer_acc, denom1_acc, denom2_acc):
        ptr1 = builder.gep(v1, [index])
        ptr2 = builder.gep(v2, [index])
        val1 = builder.load(ptr1)
        val2 = builder.load(ptr2)

        # Numerator
        numer = builder.load(numer_acc)
        val = builder.fmul(val1, val2)
        numer = builder.fadd(numer, val)
        builder.store(numer, numer_acc)

        # Denominator1
        denom = builder.load(denom1_acc)
        val = builder.fmul(val1, val1)
        denom = builder.fadd(denom, val)
        builder.store(denom, denom1_acc)
        # Denominator2
        denom = builder.load(denom2_acc)
        val = builder.fmul(val2, val2)
        denom = builder.fadd(denom, val)
        builder.store(denom, denom2_acc)

    def __gen_llvm_max_diff(self, builder, index, ctx, v1, v2, max_diff_ptr):
        ptr1 = builder.gep(v1, [index])
        ptr2 = builder.gep(v2, [index])
        val1 = builder.load(ptr1)
        val2 = builder.load(ptr2)

        # Get the difference
        diff = builder.fsub(val1, val2)

        # Get absolute value
        fabs = ctx.get_builtin("fabs", [ctx.float_ty])
        diff = builder.call(fabs, [diff])

        # Get maximum
        old_max = builder.load(max_diff_ptr)
        fmax = ctx.get_builtin("maxnum", [ctx.float_ty])
        max_diff = builder.call(fmax, [diff, old_max])

        builder.store(max_diff, max_diff_ptr)

    def __gen_llvm_pearson(self, builder, index, ctx, v1, v2, acc_x, acc_y, acc_xy, acc_x2, acc_y2):
        ptr1 = builder.gep(v1, [index])
        ptr2 = builder.gep(v2, [index])
        val1 = builder.load(ptr1)
        val2 = builder.load(ptr2)

        # Sum X
        acc_x_val = builder.load(acc_x)
        acc_x_val = builder.fadd(acc_x_val, val1)
        builder.store(acc_x_val, acc_x)

        # Sum Y
        acc_y_val = builder.load(acc_y)
        acc_y_val = builder.fadd(acc_y_val, val2)
        builder.store(acc_y_val, acc_y)

        # Sum XY
        acc_xy_val = builder.load(acc_xy)
        xy = builder.fmul(val1, val2)
        acc_xy_val = builder.fadd(acc_xy_val, xy)
        builder.store(acc_xy_val, acc_xy)

        # Sum X2
        acc_x2_val = builder.load(acc_x2)
        x2 = builder.fmul(val1, val1)
        acc_x2_val = builder.fadd(acc_x2_val, x2)
        builder.store(acc_x2_val, acc_x2)

        # Sum Y2
        acc_y2_val = builder.load(acc_y2)
        y2 = builder.fmul(val2, val2)
        acc_y2_val = builder.fadd(acc_y2_val, y2)
        builder.store(acc_y2_val, acc_y2)

    def _gen_llvm_function_body(self, ctx, builder, params, _, arg_in, arg_out, *, tags:frozenset):
        assert isinstance(arg_in.type.pointee, pnlvm.ir.ArrayType)
        assert isinstance(arg_in.type.pointee.element, pnlvm.ir.ArrayType)
        assert arg_in.type.pointee.count == 2

        v1 = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(0)])
        v2 = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(1), ctx.int32_ty(0)])

        acc_ptr = builder.alloca(ctx.float_ty)
        builder.store(ctx.float_ty(-0.0), acc_ptr)

        kwargs = {"ctx": ctx, "v1": v1, "v2": v2, "acc": acc_ptr}
        if self.metric == DIFFERENCE or self.metric == NORMED_L0_SIMILARITY:
            inner = functools.partial(self.__gen_llvm_sum_difference, **kwargs)
        elif self.metric == EUCLIDEAN:
            inner = functools.partial(self.__gen_llvm_sum_diff_squares, **kwargs)
        elif self.metric == ENERGY:
            inner = functools.partial(self.__gen_llvm_sum_product, **kwargs)
        elif self.metric == CROSS_ENTROPY:
            inner = functools.partial(self.__gen_llvm_cross_entropy, **kwargs)
        elif self.metric == COSINE:
            del kwargs['acc']
            numer_acc = builder.alloca(ctx.float_ty)
            denom1_acc = builder.alloca(ctx.float_ty)
            denom2_acc = builder.alloca(ctx.float_ty)
            for loc in numer_acc, denom1_acc, denom2_acc:
                builder.store(ctx.float_ty(-0.0), loc)
            kwargs['numer_acc'] = numer_acc
            kwargs['denom1_acc'] = denom1_acc
            kwargs['denom2_acc'] = denom2_acc
            inner = functools.partial(self.__gen_llvm_cosine, **kwargs)
        elif self.metric == MAX_ABS_DIFF:
            del kwargs['acc']
            max_diff_ptr = builder.alloca(ctx.float_ty)
            builder.store(ctx.float_ty(float("NaN")), max_diff_ptr)
            kwargs['max_diff_ptr'] = max_diff_ptr
            inner = functools.partial(self.__gen_llvm_max_diff, **kwargs)
        elif self.metric == CORRELATION:
            acc_x_ptr = builder.alloca(ctx.float_ty)
            acc_y_ptr = builder.alloca(ctx.float_ty)
            acc_xy_ptr = builder.alloca(ctx.float_ty)
            acc_x2_ptr = builder.alloca(ctx.float_ty)
            acc_y2_ptr = builder.alloca(ctx.float_ty)
            for loc in [acc_x_ptr, acc_y_ptr, acc_xy_ptr, acc_x2_ptr, acc_y2_ptr]:
                builder.store(ctx.float_ty(-0.0), loc)
            del kwargs['acc']
            kwargs['acc_x'] = acc_x_ptr
            kwargs['acc_y'] = acc_y_ptr
            kwargs['acc_xy'] = acc_xy_ptr
            kwargs['acc_x2'] = acc_x2_ptr
            kwargs['acc_y2'] = acc_y2_ptr
            inner = functools.partial(self.__gen_llvm_pearson, **kwargs)
        else:
            raise RuntimeError('Unsupported metric')

        input_length = arg_in.type.pointee.element.count
        vector_length = ctx.int32_ty(input_length)
        with pnlvm.helpers.for_loop_zero_inc(builder, vector_length, self.metric) as args:
            inner(*args)

        sqrt = ctx.get_builtin("sqrt", [ctx.float_ty])
        fabs = ctx.get_builtin("fabs", [ctx.float_ty])
        ret = builder.load(acc_ptr)
        if self.metric == NORMED_L0_SIMILARITY:
            ret = builder.fdiv(ret, ret.type(4))
            ret = builder.fsub(ret.type(1), ret)
        elif self.metric == ENERGY:
            ret = builder.fmul(ret, ret.type(-0.5))
        elif self.metric == EUCLIDEAN:
            ret = builder.call(sqrt, [ret])
        elif self.metric == MAX_ABS_DIFF:
            ret = builder.load(max_diff_ptr)
        elif self.metric == COSINE:
            numer = builder.load(numer_acc)
            denom1 = builder.load(denom1_acc)
            denom1 = builder.call(sqrt, [denom1])
            denom2 = builder.load(denom2_acc)
            denom2 = builder.call(sqrt, [denom2])
            denom = builder.fmul(denom1, denom2)

            ret = builder.fdiv(numer, denom)
            ret = builder.call(fabs, [ret])
            ret = builder.fsub(ret.type(1), ret)

        elif self.metric == CORRELATION:
            n = ctx.float_ty(input_length)
            acc_xy = builder.load(acc_xy_ptr)
            acc_x = builder.load(acc_x_ptr)
            acc_y = builder.load(acc_y_ptr)
            acc_x2 = builder.load(acc_x2_ptr)
            acc_y2 = builder.load(acc_y2_ptr)

            # We'll need meanx,y below
            mean_x = builder.fdiv(acc_x, n)
            mean_y = builder.fdiv(acc_y, n)

            # Numerator: sum((x - mean(x))*(y - mean(y)) =
            # sum(x*y - x*mean(y) - y*mean(x) + mean(x)*mean(y)) =
            # sum(x*y) - sum(x)*mean(y) - sum(y)*mean(x) + mean(x)*mean(y)*n
            b = builder.fmul(acc_x, mean_y)
            c = builder.fmul(acc_y, mean_x)
            d = builder.fmul(mean_x, mean_y)
            d = builder.fmul(d, n)

            numerator = builder.fsub(acc_xy, b)
            numerator = builder.fsub(numerator, c)
            numerator = builder.fadd(numerator, d)

            # Denominator: sqrt(D_X * D_Y)
            # D_X = sum((x - mean(x))^2) = sum(x^2 - 2*x*mean(x) + mean(x)^2) =
            # sum(x^2) - 2 * sum(x) * mean(x) + n * mean(x)^2
            dxb = builder.fmul(acc_x, mean_x)
            dxb = builder.fadd(dxb, dxb)        # *2
            dxc = builder.fmul(mean_x, mean_x)  # ^2
            dxc = builder.fmul(dxc, n)

            dx = builder.fsub(acc_x2, dxb)
            dx = builder.fadd(dx, dxc)

            # Similarly for y
            dyb = builder.fmul(acc_y, mean_y)
            dyb = builder.fadd(dyb, dyb)        # *2
            dyc = builder.fmul(mean_y, mean_y)  # ^2
            dyc = builder.fmul(dyc, n)

            dy = builder.fsub(acc_y2, dyb)
            dy = builder.fadd(dy, dyc)

            # Denominator: sqrt(D_X * D_Y)
            denominator = builder.fmul(dx, dy)
            denominator = builder.call(sqrt, [denominator])

            corr = builder.fdiv(numerator, denominator)

            # ret =  1 - abs(corr)
            ret = builder.call(fabs, [corr])
            ret = builder.fsub(ctx.float_ty(1), ret)

        # MAX_ABS_DIFF, CORRELATION, and COSINE ignore normalization
        ignores = frozenset((MAX_ABS_DIFF, CORRELATION, COSINE))
        if self.normalize and self.metric not in ignores:
            norm_factor = input_length
            if self.metric == ENERGY:
                norm_factor = norm_factor ** 2
            ret = builder.fdiv(ret, ctx.float_ty(norm_factor), name="normalized")
        if arg_out.type.pointee != ret.type:
            # Some instance use 2d output values
            arg_out = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(0),
                                            ctx.int32_ty(0)])
        builder.store(ret, arg_out)

        return builder

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """Calculate the distance between the two vectors in `variable <Stability.variable>`.

        Use the `distance metric <DistanceMetrics>` specified in `metric <Distance.metric>` to calculate the distance.
        If `normalize <Distance.normalize>` is `True`, the result is divided by the length of `variable
        <Distance.variable>`.

        Returns
        -------

        distance : scalar

        """

        try:
            v1 = np.hstack(variable[0])
        except TypeError:
            v1 = variable[0]

        try:
            v2 = np.hstack(variable[1])
        except TypeError:
            v2 = variable[1]

        # Maximum of Hadamard (elementwise) difference of v1 and v2
        if self.metric == MAX_ABS_DIFF:
            result = np.max(np.fabs(v1 - v2))

        # Simple Hadamard (elementwise) difference of v1 and v2
        elif self.metric == DIFFERENCE:
            result = np.sum(np.fabs(v1 - v2))

        # Similarity (used specifically for testing Compilation of Predator-Prey Model)
        elif self.metric == NORMED_L0_SIMILARITY:
            result = 1.0 - np.sum(np.abs(v1 - v2)) / 4.0

        # Euclidean distance between v1 and v2
        elif self.metric == EUCLIDEAN:
            result = np.linalg.norm(v2 - v1)

        # Cosine similarity of v1 and v2
        elif self.metric == COSINE:
            # result = np.correlate(v1, v2)
            result = 1.0 - np.fabs(Distance.cosine(v1, v2))
            return self.convert_output_type(result)

        # Correlation of v1 and v2
        elif self.metric == CORRELATION:
            # result = np.correlate(v1, v2)
            result = 1.0 - np.fabs(Distance.correlation(v1, v2))
            return self.convert_output_type(result)

        # Cross-entropy of v1 and v2
        elif self.metric == CROSS_ENTROPY:
            # FIX: VALIDATE THAT ALL ELEMENTS OF V1 AND V2 ARE 0 TO 1
            if not self.is_initializing:
                v1 = np.where(v1 == 0, EPSILON, v1)
                v2 = np.where(v2 == 0, EPSILON, v2)
            # MODIFIED CW 3/20/18: avoid divide by zero error by plugging in two zeros
            # FIX: unsure about desired behavior when v2 = 0 and v1 != 0
            # JDC: returns [inf]; leave, and let it generate a warning or error message for user
            result = -np.sum(np.where(np.logical_and(v1 == 0, v2 == 0), 0.0, v1 * np.log(v2)))

        # Energy
        elif self.metric == ENERGY:
            result = -np.sum(v1 * v2) / 2.0

        else:
            assert False, '{} not a recognized metric in {}'.format(self.metric, self.__class__.__name__)

        if self.normalize and self.metric not in {MAX_ABS_DIFF, CORRELATION}:
            if self.metric == ENERGY:
                result /= len(v1) ** 2.0
            else:
                result /= len(v1)

        return self.convert_output_type(result)
