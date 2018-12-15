#
# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ****************************************   DISTRIBUTION FUNCTIONS   **************************************************
'''

* `NormalDist`
* `UniformToNormalDist`
* `ExponentialDist`
* `UniformDist`
* `GammaDist`
* `WaldDist`

Overview
--------

Functions that return one or more samples from a distribution.

'''

import numpy as np
import typecheck as tc

from psyneulink.core.components.functions.function import Function_Base, FunctionError
from psyneulink.core.globals.keywords import \
    DIST_FUNCTION_TYPE, NORMAL_DIST_FUNCTION, STANDARD_DEVIATION, DIST_MEAN, EXPONENTIAL_DIST_FUNCTION, \
    BETA, UNIFORM_DIST_FUNCTION, LOW, HIGH, GAMMA_DIST_FUNCTION, SCALE, DIST_SHAPE, WALD_DIST_FUNCTION
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set

from psyneulink.core.globals.parameters import Param

__all__ = ['DistributionFunction', 'NormalDist', 'UniformDist', 'UniformToNormalDist', 'ExponentialDist',
           'GammaDist', 'WaldDist']


class DistributionFunction(Function_Base):
    componentType = DIST_FUNCTION_TYPE


class NormalDist(DistributionFunction):
    """
    NormalDist(                      \
             mean=0.0,               \
             standard_deviation=1.0, \
             params=None,            \
             owner=None,             \
             prefs=None              \
             )

    .. _NormalDist:

    Return a random sample from a normal distribution using numpy.random.normal;

    Arguments
    ---------

    mean : float : default 0.0
        The mean or center of the normal distribution

    standard_deviation : float : default 1.0
        Standard deviation of the normal distribution. Must be > 0.0

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

    mean : float : default 0.0
        The mean or center of the normal distribution.

    standard_deviation : float : default 1.0
        Standard deviation of the normal distribution; if it is 0.0, returns `mean <NormalDist.mean>`.

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

    """

    componentName = NORMAL_DIST_FUNCTION

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    class Params(DistributionFunction.Params):
        """
            Attributes
            ----------

                mean
                    see `mean <WaldDist.mean>`

                    :default value: 1.0
                    :type: float

                scale
                    see `scale <WaldDist.scale>`

                    :default value: 1.0
                    :type: float

        """
        mean = Param(0.0, modulable=True)
        standard_deviation = Param(1.0, modulable=True)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 mean=0.0,
                 standard_deviation=1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(mean=mean,
                                                  standard_deviation=standard_deviation,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         params=params,
                         owner=owner,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

    def _validate_params(self, request_set, target_set=None, context=None):
        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        if STANDARD_DEVIATION in target_set:
            if target_set[STANDARD_DEVIATION] < 0.0:
                raise FunctionError("The standard_deviation parameter ({}) of {} must be greater than zero.".
                                    format(target_set[STANDARD_DEVIATION], self.name))

    def function(self,
                 variable=None,
                 execution_id=None,
                 params=None,
                 context=None):
        # Validate variable and validate params
        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        mean = self.get_current_function_param(DIST_MEAN, execution_id)
        standard_deviation = self.get_current_function_param(STANDARD_DEVIATION, execution_id)

        result = np.random.normal(mean, standard_deviation)

        return self.convert_output_type(result)


class UniformToNormalDist(DistributionFunction):
    """
    UniformToNormalDist(             \
             mean=0.0,               \
             standard_deviation=1.0, \
             params=None,            \
             owner=None,             \
             prefs=None              \
             )

    .. _UniformToNormalDist:

    Return a random sample from a normal distribution using first np.random.rand(1) to generate a sample from a uniform
    distribution, and then converting that sample to a sample from a normal distribution with the following equation:

    .. math::

        normal\\_sample = \\sqrt{2} \\cdot standard\\_dev \\cdot scipy.special.erfinv(2 \\cdot uniform\\_sample - 1)  + mean

    The uniform --> normal conversion allows for a more direct comparison with MATLAB scripts.

    .. note::

        This function requires `SciPy <https://pypi.python.org/pypi/scipy>`_.

    (https://github.com/jonasrauber/randn-matlab-python)

    Arguments
    ---------

    mean : float : default 0.0
        The mean or center of the normal distribution

    standard_deviation : float : default 1.0
        Standard deviation of the normal distribution

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

    mean : float : default 0.0
        The mean or center of the normal distribution

    standard_deviation : float : default 1.0
        Standard deviation of the normal distribution

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

    """

    componentName = NORMAL_DIST_FUNCTION

    class Params(DistributionFunction.Params):
        """
            Attributes
            ----------

                mean
                    see `mean <WaldDist.mean>`

                    :default value: 1.0
                    :type: float

                scale
                    see `scale <WaldDist.scale>`

                    :default value: 1.0
                    :type: float

        """
        variable = Param(np.array([0]), read_only=True)
        mean = Param(0.0, modulable=True)
        standard_deviation = Param(1.0, modulable=True)

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 mean=0.0,
                 standard_deviation=1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(mean=mean,
                                                  standard_deviation=standard_deviation,
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

        try:
            from scipy.special import erfinv
        except:
            raise FunctionError("The UniformToNormalDist function requires the SciPy package.")

        # Validate variable and validate params
        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        mean = self.get_current_function_param(DIST_MEAN, execution_id)
        standard_deviation = self.get_current_function_param(STANDARD_DEVIATION, execution_id)

        sample = np.random.rand(1)[0]
        result = ((np.sqrt(2) * erfinv(2 * sample - 1)) * standard_deviation) + mean

        return self.convert_output_type(result)


class ExponentialDist(DistributionFunction):
    """
    ExponentialDist(                \
             beta=1.0,              \
             params=None,           \
             owner=None,            \
             prefs=None             \
             )

    .. _ExponentialDist:

    Return a random sample from a exponential distribution using numpy.random.exponential

    Arguments
    ---------

    beta : float : default 1.0
        The scale parameter of the exponential distribution

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

    beta : float : default 1.0
        The scale parameter of the exponential distribution

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

    """
    componentName = EXPONENTIAL_DIST_FUNCTION

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    class Params(DistributionFunction.Params):
        """
            Attributes
            ----------

                mean
                    see `mean <WaldDist.mean>`

                    :default value: 1.0
                    :type: float

                scale
                    see `scale <WaldDist.scale>`

                    :default value: 1.0
                    :type: float

        """
        beta = Param(1.0, modulable=True)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 beta=1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(beta=beta,
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
        # Validate variable and validate params
        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        beta = self.get_current_function_param(BETA, execution_id)
        result = np.random.exponential(beta)

        return self.convert_output_type(result)


class UniformDist(DistributionFunction):
    """
    UniformDist(                      \
             low=0.0,             \
             high=1.0,             \
             params=None,           \
             owner=None,            \
             prefs=None             \
             )

    .. _UniformDist:

    Return a random sample from a uniform distribution using numpy.random.uniform

    Arguments
    ---------

    low : float : default 0.0
        Lower bound of the uniform distribution

    high : float : default 1.0
        Upper bound of the uniform distribution

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

    low : float : default 0.0
        Lower bound of the uniform distribution

    high : float : default 1.0
        Upper bound of the uniform distribution

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

    """
    componentName = UNIFORM_DIST_FUNCTION

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    class Params(DistributionFunction.Params):
        """
            Attributes
            ----------

                mean
                    see `mean <WaldDist.mean>`

                    :default value: 1.0
                    :type: float

                scale
                    see `scale <WaldDist.scale>`

                    :default value: 1.0
                    :type: float

        """
        low = Param(0.0, modulable=True)
        high = Param(1.0, modulable=True)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 low=0.0,
                 high=1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(low=low,
                                                  high=high,
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
        # Validate variable and validate params
        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        low = self.get_current_function_param(LOW, execution_id)
        high = self.get_current_function_param(HIGH, execution_id)
        result = np.random.uniform(low, high)

        return self.convert_output_type(result)


class GammaDist(DistributionFunction):
    """
    GammaDist(\
             scale=1.0,\
             dist_shape=1.0,\
             params=None,\
             owner=None,\
             prefs=None\
             )

    .. _GammaDist:

    Return a random sample from a gamma distribution using numpy.random.gamma

    Arguments
    ---------

    scale : float : default 1.0
        The scale of the gamma distribution. Should be greater than zero.

    dist_shape : float : default 1.0
        The shape of the gamma distribution. Should be greater than zero.

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

    scale : float : default 1.0
        The dist_shape of the gamma distribution. Should be greater than zero.

    dist_shape : float : default 1.0
        The scale of the gamma distribution. Should be greater than zero.

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

    """

    componentName = GAMMA_DIST_FUNCTION

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    class Params(DistributionFunction.Params):
        """
            Attributes
            ----------

                mean
                    see `mean <WaldDist.mean>`

                    :default value: 1.0
                    :type: float

                scale
                    see `scale <WaldDist.scale>`

                    :default value: 1.0
                    :type: float

        """
        scale = Param(1.0, modulable=True)
        dist_shape = Param(1.0, modulable=True)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 scale=1.0,
                 dist_shape=1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(scale=scale,
                                                  dist_shape=dist_shape,
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
        # Validate variable and validate params
        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        scale = self.get_current_function_param(SCALE, execution_id)
        dist_shape = self.get_current_function_param(DIST_SHAPE, execution_id)

        result = np.random.gamma(dist_shape, scale)

        return self.convert_output_type(result)


class WaldDist(DistributionFunction):
    """
     WaldDist(             \
              scale=1.0,\
              mean=1.0,\
              params=None,\
              owner=None,\
              prefs=None\
              )

     .. _WaldDist:

     Return a random sample from a Wald distribution using numpy.random.wald

     Arguments
     ---------

     scale : float : default 1.0
         Scale parameter of the Wald distribution. Should be greater than zero.

     mean : float : default 1.0
         Mean of the Wald distribution. Should be greater than or equal to zero.

     params : Dict[param keyword: param value] : default None
         a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
         function.  Values specified for parameters in the dictionary override any assigned to those parameters in
         arguments of the constructor.

     owner : Component
         `component <Component>` to which to assign the Function.

     prefs : PreferenceSet or specification dict : default Function.classPreferences
         the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
         defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


     Attributes
     ----------

     scale : float : default 1.0
         Scale parameter of the Wald distribution. Should be greater than zero.

     mean : float : default 1.0
         Mean of the Wald distribution. Should be greater than or equal to zero.

     params : Dict[param keyword: param value] : default None
         a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
         function.  Values specified for parameters in the dictionary override any assigned to those parameters in
         arguments of the constructor.

     owner : Component
         `component <Component>` to which to assign the Function.

     prefs : PreferenceSet or specification dict : default Function.classPreferences
         the `PreferenceSet` for the Function. If it is not specified, a default is assigned using `classPreferences`
         defined in __init__.py (see :doc:`PreferenceSet <LINK>` for details).


     """

    componentName = WALD_DIST_FUNCTION

    paramClassDefaults = Function_Base.paramClassDefaults.copy()

    class Params(DistributionFunction.Params):
        """
            Attributes
            ----------

                mean
                    see `mean <WaldDist.mean>`

                    :default value: 1.0
                    :type: float

                scale
                    see `scale <WaldDist.scale>`

                    :default value: 1.0
                    :type: float

        """
        scale = Param(1.0, modulable=True)
        mean = Param(1.0, modulable=True)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 scale=1.0,
                 mean=1.0,
                 params=None,
                 owner=None,
                 prefs: is_pref_set = None):
        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(scale=scale,
                                                  mean=mean,
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
        # Validate variable and validate params
        variable = self._check_args(variable=variable, execution_id=execution_id, params=params, context=context)

        scale = self.get_current_function_param(SCALE, execution_id)
        mean = self.get_current_function_param(DIST_MEAN, execution_id)

        result = np.random.wald(mean, scale)

        return self.convert_output_type(result)