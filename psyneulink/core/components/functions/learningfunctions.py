#
# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *******************************************   LEARNING FUNCTIONS *****************************************************
"""

* `BayesGLM`
* `Kohonen`
* `Hebbian`
* `ContrastiveHebbian`
* `Reinforcement`
* `BackPropagation`
* `TDLearning`

Functions that parameterize a function.

"""

from collections import namedtuple

import numpy as np
import typecheck as tc
import types

from psyneulink.core.components.functions.function import Function_Base, FunctionError, is_function_type
from psyneulink.core.components.functions.transferfunctions import Logistic
from psyneulink.core.components.component import ComponentError
from psyneulink.core.globals.keywords import \
    CONTRASTIVE_HEBBIAN_FUNCTION, TDLEARNING_FUNCTION, LEARNING_FUNCTION_TYPE, LEARNING_RATE, \
    KOHONEN_FUNCTION, GAUSSIAN, LINEAR, EXPONENTIAL, HEBBIAN_FUNCTION, RL_FUNCTION, BACKPROPAGATION_FUNCTION, MATRIX, \
    MSE, SSE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.utilities import is_numeric, scalar_distance, get_global_seed, convert_to_np_array
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set

__all__ = ['LearningFunction', 'Kohonen', 'Hebbian', 'ContrastiveHebbian',
           'Reinforcement', 'BayesGLM', 'BackPropagation', 'TDLearning',
           'LEARNING_ACTIVATION_FUNCTION','LEARNING_ACTIVATION_INPUT','LEARNING_ACTIVATION_OUTPUT',
           'LEARNING_ERROR_OUTPUT','AUTOASSOCIATIVE']

AUTOASSOCIATIVE = 'AUTOASSOCIATIVE'

# Inidices of terms in variable:
LEARNING_ACTIVATION_FUNCTION = 'activation_function'
LEARNING_ACTIVATION_INPUT = 0  # a(j)
LEARNING_ACTIVATION_OUTPUT = 1  # a(i)
LEARNING_ERROR_OUTPUT = 2

# Other indices
WT_MATRIX_SENDERS_DIM = 0
WT_MATRIX_RECEIVERS_DIM = 1

# Argument and attribute names:
ACTIVATION_INPUT = 'activation_input'
ACTIVATION_OUTPUT = 'activation_output'
ERROR_SIGNAL = 'error_signal'
ERROR_MATRIX = 'error_matrix'


ReturnVal = namedtuple('ReturnVal', 'learning_signal, error_signal')


class LearningFunction(Function_Base):
    """Abstract class of `Function <Function>` used for learning.

    COMMENT:
    IMPLEMENTATION NOTE:
       The function method of a LearningFunction *must* include a **kwargs argument, which accomodates
       Function-specific parameters;  this is to accommodate the ability of LearningMechanisms to call
       the function of a LearningFunction with arguments that may not be implemented for all LearningFunctions
    COMMENT

    Attributes
    ----------

    variable : list or array
        most LearningFunctions take a list or 2d array that must contain three items:

        * the input to the parameter being modified (variable[LEARNING_ACTIVATION_INPUT]);
        * the output of the parameter being modified (variable[LEARNING_ACTIVATION_OUTPUT]);
        * the error associated with the output (variable[LEARNING_ERROR_OUTPUT]).

        However, the exact specification depends on the funtion's type.

    default_learning_rate : numeric
        the value used for the function's `learning_rate <LearningFunction.learning_rate>` parameter if none of the
        following are specified:  the `learning_rate <LearningMechanism.learning_rate>` for the `LearningMechanism` to
        which the function has been assigned, the `learning_rate <Process.learning_rate>` for any `Process` or
        the `learning_rate <System.learning_rate>` for any `System` to which that LearningMechanism belongs.
        The exact form of the value (i.e., whether it is a scalar or array) depends on the function's type.

    learning_rate : numeric
        generally used to multiply the result of the function before it is returned;  however, both the form of the
        value (i.e., whether it is a scalar or array) and how it is used depend on the function's type.

    Returns
    -------

    The number of items returned and their format depend on the function's type.

    Most return an array (used as the `learning_signal <LearningMechanism.learning_signal>` by a \
    `LearningMechanism`), and some also return a similarly formatted array containing either the \
    error received (in the third item of the `variable <LearningFunction.variable>`) or a \
    version of it modified by the function.

    """

    componentType = LEARNING_FUNCTION_TYPE

    class Parameters(Function_Base.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <LearningFunction.variable>`

                    :default value: numpy.array([0, 0, 0])
                    :type: ``numpy.ndarray``
                    :read only: True

                learning_rate
                    see `learning_rate <LearningFunction.learning_rate>`

                    :default value: 0.05
                    :type: ``float``
        """
        variable = Parameter(np.array([0, 0, 0]), read_only=True, pnl_internal=True, constructor_argument='default_variable')
        learning_rate = Parameter(0.05, modulable=True)

    def _validate_learning_rate(self, learning_rate, type=None):

        learning_rate = np.array(learning_rate).copy()
        learning_rate_dim = learning_rate.ndim

        self._validate_parameter_spec(learning_rate, LEARNING_RATE)

        if type == AUTOASSOCIATIVE:

            if learning_rate_dim == 1 and len(learning_rate) != len(self.defaults.variable):
                raise FunctionError("Length of {} arg for {} ({}) must be the same as its variable ({})".
                                    format(LEARNING_RATE, self.name, len(learning_rate),
                                           len(self.defaults.variable)))

            if learning_rate_dim == 2:
                shape = learning_rate.shape
                if shape[0] != shape[1] or shape[0] != len(self.defaults.variable):
                    raise FunctionError("Shape of {} arg for {} ({}) must be square and "
                                        "of the same width as the length of its variable ({})".
                                        format(LEARNING_RATE, self.name, shape, len(self.defaults.variable)))

            if learning_rate_dim > 2:
                raise FunctionError("{} arg for {} ({}) must be a single value of a 1d or 2d array".
                                    format(LEARNING_RATE, self.name, learning_rate))

        else:
            if learning_rate_dim:
                raise FunctionError("{} arg for {} ({}) must be a single value".
                                    format(LEARNING_RATE, self.name, learning_rate))


class BayesGLM(LearningFunction):
    """
    BayesGLM(                   \
        default_variable=None,  \
        mu_0=0,                 \
        sigma_0=1,              \
        gamma_shape_0=1,        \
        gamma_size_0=1,         \
        params=None,            \
        prefs=None)

    Use Bayesian linear regression to find means and distributions of weights that predict dependent variable(s).

    `function <BayesGLM.function>` uses a normal linear model:

     .. math::
        dependent\\ variable(s) = predictor(s)\\ \\Theta + \\epsilon,

     with predictor(s) in `variable <BayesGLM.variable>`\\[0] and dependent variable(s) in `variable
     <BayesGLM.variable>`\\[1], and a normal-gamma prior distribution of weights (:math:`\\Theta`), to update
     the weight distribution parameters `mu_n <BayesGLM.mu_n>`, `Lambda_n <BayesGLM.Lambda_n>`, `gamma_shape_n
     <BayesGLM.gamma_shape_n>`, and `gamma_size_n <BayesGLM.gamma_size_n>`, and returns an array of prediction
     weights sampled from the multivariate normal-gamma distribution [based on Falk Lieder's BayesianGLM.m,
     adapted for Python by Yotam Sagiv and for PsyNeuLink by Jon Cohen; useful reference:
    `Bayesian Inference <http://www2.stat.duke.edu/~sayan/Sta613/2017/read/chapter_9.pdf>`_.]

    .. hint::
       The **mu_0** or **sigma_0** arguments of the consructor can be used in place of **default_variable** to define
       the size of the predictors array and, correspondingly, the weights array returned by the function (see
       **Parameters** below).

    Arguments
    ---------

    default_variable : 3d array : default None
        first item of axis 0 must be a 2d array with one or more 1d arrays to use as predictor vectors, one for
        each sample to be fit;  second item must be a 2d array of equal length to the first item, with a 1d array
        containing a scalar that is the dependent (to-be-predicted) value for the corresponding sample in the first
        item.  If `None` is specified, but either **mu_0** or **sigma_0 is specified, then the they are used to
        determine the shape of `variable <BayesGLM.variable>`.  If neither **mu_0** nor **sigma_0** are specified,
        then the shape of `variable <BayesGLM.variable>` is determined by the first call to its `function
        <BayesGLM.function>`, as are `mu_prior <BayesGLM.mu_prior>`, `sigma_prior <BayesGLM.mu_prior>`,
        `gamma_shape_prior <BayesGLM.gamma_shape_prior>` and `gamma_size_prior <BayesGLM.gamma_size_prior>`.

    mu_0 : int, float or 1d array : default 0
        specifies initial value of `mu_prior <BayesGLM.mu_prior>` (the prior for the mean of the distribution for
        the prediction weights returned by the function).  If a scalar is specified, the same value will be used
        for all elements of `mu_prior <BayesGLM.mu_prior>`;  if it is an array, it must be the same length as
        the predictor array(s) in axis 0 of **default_variable**.  If **default_variable** is not specified, the
        specification for **mu_0** is used to determine the shape of `variable <BayesGLM.variable>` and
        `sigma_prior <BayesGLM.sigma_prior>`.

    sigma_0 : int, float or 1d array : default 0
        specifies initial value of `sigma_prior <BayesGLM.Lambda_prior>` (the prior for the variance of the distribution
        for the prediction weights returned by the function).  If a scalar is specified, the same value will be used for
        all elements of `Lambda_prior <BayesGLM.Lambda_prior>`;  if it is an array, it must be the same length as the
        predictor array(s) in axis 0 of **default_variable**.  If neither **default_variable** nor **mu_0** is
        specified, the specification for **sigma_0** is used to determine their shapes.

    gamma_shape_0 : int or float : default 1
        specifies the shape of the gamma distribution from which samples of the weights are drawn (see documentation
        for `numpy.random.gamma <https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.gamma.html>`_.

    gamma_size_0 : int or float : default 1
        specifies the size of the gamma distribution from which samples of the weights are drawn (see documentation for
        `numpy.random.gamma <https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.gamma.html>`_.

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

    variable : 3d array
        samples used to update parameters of prediction weight distributions.
        variable[0] is a 2d array of predictor vectors, all of the same length;
        variable[1] is a 2d array of scalar dependent variables, one for each predictor vector.

    mu_0 : int, float or 2d array
        determines the initial prior(s) for the means of the distributions of the prediction weights;
        if it is a scalar, that value is assigned as the priors for all means.

    mu_prior : 2d array
        current priors for the means of the distributions of the predictions weights.

    mu_n : 2d array
        current means for the distributions of the prediction weights.

    sigma_0 : int, float or 2d array
        value used to determine the initial prior(s) for the variances of the distributions of the prediction
        weights; if it is a scalar, that value is assigned as the priors for all variances.

    Lambda_prior :  2d array
        current priors for the variances of the distributions of the predictions weights.

    Lambda_n :  2d array
        current variances for the distributions of the prediction weights.

    gamma_shape_0 : int or float
        determines the initial value used for the shape parameter of the gamma distribution used to sample the
        prediction weights.

    gamma_shape_prior : int or float
        current prior for the shape parameter of the gamma distribution used to sample the prediction weights.

    gamma_shape_n : int or float
        current value of the shape parameter of the gamma distribution used to sample the prediction weights.

    gamma_size_0 : int or float
        determines the initial value used for the size parameter of the gamma distribution used to sample the
        prediction weights.

    gamma_size_prior : int or float
        current prior for the size parameter of the gamma distribution used to sample the prediction weights.

    gamma_size_n : 2d array with single scalar value
        current value of the size parameter of the gamma distribution used to sample the prediction weights.

    random_state : numpy.RandomState
      private pseudorandom number generator

    weights_sample : 1d array
        last sample of prediction weights drawn in call to `sample_weights <BayesGLM.sample_weights>` and returned by
        `function <BayesGLM.function>`.

    owner : Component
        `Mechanism <Mechanism>` to which the Function belongs.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).
    """
    class Parameters(LearningFunction.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <BayesGLM.variable>`

                    :default value: [numpy.array([0, 0, 0]), numpy.array([0])]
                    :type: ``list``
                    :read only: True

                value
                    see `value <BayesGLM.value>`

                    :default value: numpy.array([0])
                    :type: ``numpy.ndarray``
                    :read only: True

                Lambda_0
                    see `Lambda_0 <BayesGLM.Lambda_0>`

                    :default value: 0
                    :type: ``int``

                Lambda_n
                    see `Lambda_n <BayesGLM.Lambda_n>`

                    :default value: 0
                    :type: ``int``

                Lambda_prior
                    see `Lambda_prior <BayesGLM.Lambda_prior>`

                    :default value: 0
                    :type: ``int``

                gamma_shape_0
                    see `gamma_shape_0 <BayesGLM.gamma_shape_0>`

                    :default value: 1
                    :type: ``int``

                gamma_shape_n
                    see `gamma_shape_n <BayesGLM.gamma_shape_n>`

                    :default value: 1
                    :type: ``int``

                gamma_shape_prior
                    see `gamma_shape_prior <BayesGLM.gamma_shape_prior>`

                    :default value: 1
                    :type: ``int``

                gamma_size_0
                    see `gamma_size_0 <BayesGLM.gamma_size_0>`

                    :default value: 1
                    :type: ``int``

                gamma_size_n
                    see `gamma_size_n <BayesGLM.gamma_size_n>`

                    :default value: 1
                    :type: ``int``

                gamma_size_prior
                    see `gamma_size_prior <BayesGLM.gamma_size_prior>`

                    :default value: 1
                    :type: ``int``

                mu_0
                    see `mu_0 <BayesGLM.mu_0>`

                    :default value: 0
                    :type: ``int``

                mu_n
                    see `mu_n <BayesGLM.mu_n>`

                    :default value: 0
                    :type: ``int``

                mu_prior
                    see `mu_prior <BayesGLM.mu_prior>`

                    :default value: 0
                    :type: ``int``

                random_state
                    see `random_state <BayesGLM.random_state>`

                    :default value: None
                    :type: ``numpy.random.RandomState``

                sigma_0
                    see `sigma_0 <BayesGLM.sigma_0>`

                    :default value: 1
                    :type: ``int``
        """
        random_state = Parameter(None, stateful=True, loggable=False)
        variable = Parameter([np.array([0, 0, 0]), np.array([0])], read_only=True, pnl_internal=True, constructor_argument='default_variable')
        value = Parameter(np.array([0]), read_only=True, aliases=['sample_weights'], pnl_internal=True)

        Lambda_0 = 0
        Lambda_prior = 0
        Lambda_n = 0

        mu_0 = 0
        mu_prior = 0
        mu_n = 0

        sigma_0 = 1

        gamma_shape_0 = 1
        gamma_shape_n = 1
        gamma_shape_prior = 1

        gamma_size_0 = 1
        gamma_size_n = 1
        gamma_size_prior = 1

    def __init__(self,
                 default_variable=None,
                 mu_0=None,
                 sigma_0=None,
                 gamma_shape_0=None,
                 gamma_size_0=None,
                 params=None,
                 owner=None,
                 seed=None,
                 prefs: tc.optional(is_pref_set) = None):

        self.user_specified_default_variable = default_variable

        if seed is None:
            seed = get_global_seed()

        random_state = np.random.RandomState([seed])

        super().__init__(
            default_variable=default_variable,
            mu_0=mu_0,
            sigma_0=sigma_0,
            gamma_shape_0=gamma_shape_0,
            gamma_size_0=gamma_size_0,
            random_state=random_state,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _handle_default_variable(self, default_variable=None, size=None):

        # If default_variable was not specified by user...
        if default_variable is None and size in {None, NotImplemented}:
            #  but mu_0 and/or sigma_0 was specified as an array...
            if isinstance(self.mu_0, (list, np.ndarray)) or isinstance(self.sigma_0, (list, np.ndarray)):
                # if both are specified, make sure they are the same size
                if (isinstance(self.mu_0, (list, np.ndarray))
                        and isinstance(self.sigma_0, (list, np.ndarray))
                        and len(self.mu_0) != len(self.self.sigma_0)):
                    raise FunctionError("Length of {} ({}) does not match length of {} ({}) for {}".
                                        format(repr('mu_0'), len(self.mu_0),
                                                    repr('sigma_0'), len(self.self.sigma_0),
                                                         self.__class.__.__name__))
                # allow their size to determine the size of variable
                if isinstance(self.mu_0, (list, np.ndarray)):
                    default_variable = [np.zeros_like(self.mu_0), np.zeros((1,1))]
                else:
                    default_variable = [np.zeros_like(self.sigma_0), np.zeros((1,1))]

        return super()._handle_default_variable(default_variable=default_variable, size=size)

    def initialize_priors(self):
        """Set the prior parameters (`mu_prior <BayesGLM.mu_prior>`, `Lamba_prior <BayesGLM.Lambda_prior>`,
        `gamma_shape_prior <BayesGLM.gamma_shape_prior>`, and `gamma_size_prior <BayesGLM.gamma_size_prior>`)
        to their initial (_0) values, and assign current (_n) values to the priors
        """

        variable = np.array(self.defaults.variable)
        variable = self.defaults.variable
        if np.array(variable).dtype != object:
            variable = np.atleast_2d(variable)

        n = len(variable[0])

        if isinstance(self.mu_0, (int, float)):
            self.mu_prior = np.full((n, 1),self.mu_0)
        else:
            if len(self.mu_0) != n:
                raise FunctionError("Length of mu_0 ({}) does not match number of predictors ({})".
                                    format(len(self.mu_0), n))
            self.mu_prior = np.array(self.mu_0).reshape(len(self._mu_0),1)

        if isinstance(self.sigma_0, (int, float)):
            Lambda_0 = (1 / (self.sigma_0 ** 2)) * np.eye(n)
        else:
            if len(self.sigma_0) != n:
                raise FunctionError("Length of sigma_0 ({}) does not match number of predictors ({})".
                                    format(len(self.sigma_0), n))
            Lambda_0 = (1 / (np.array(self.sigma_0) ** 2)) * np.eye(n)
        self.Lambda_prior = Lambda_0

        # before we see any data, the posterior is the prior
        self.mu_n = self.mu_prior
        self.Lambda_n = self.Lambda_prior
        self.gamma_shape_n = self.gamma_shape_0
        self.gamma_size_n = self.gamma_size_0

    @handle_external_context(fallback_most_recent=True)
    def reset(self, default_variable=None, context=None):
        # If variable passed during execution does not match default assigned during initialization,
        #    reassign default and re-initialize priors
        if default_variable is not None:
            self._update_default_variable(
                np.array([np.zeros_like(default_variable), np.zeros_like(default_variable)]),
                context=context
            )
            self.initialize_priors()

    def _function(
        self,
        variable=None,
        context=None,
        params=None,
        ):
        """

        Arguments
        ---------

        variable : 2d or 3d array : default class_defaults.variable
           If it is a 2d array, the first item must be a 1d array of scalar predictors,
               and the second must be a 1d array containing the dependent variable to be predicted by the predictors.
           If it is a 3d array, the first item in the outermost dimension must be a 2d array containing one or more
               1d arrays of scalar predictors, and the second item must be a 2d array containing 1d arrays
               each of which contains a scalar dependent variable for the corresponding predictor vector.

        params : Dict[param keyword: param value] : default None
           a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
           function.  Values specified for parameters in the dictionary override any assigned to those parameters in
           arguments of the constructor.

        Returns
        -------

        sample weights : 1d array
            array of weights drawn from updated weight distributions.

        """

        if self.is_initializing:
            self.initialize_priors()

        # # MODIFIED 10/26/18 OLD:
        # # If variable passed during execution does not match default assigned during initialization,
        # #    reassign default and re-initialize priors
        # elif np.array(variable).shape != self.defaults.variable.shape:
        #     self.defaults.variable = np.array([np.zeros_like(variable[0]),np.zeros_like(variable[1])])
        #     self.initialize_priors()
        # MODIFIED 10/26/18 END

        # Today's prior is yesterday's posterior
        Lambda_prior = self._get_current_parameter_value('Lambda_n', context)
        mu_prior = self._get_current_parameter_value('mu_n', context)
        # # MODIFIED 6/3/19 OLD: [JDC]: THE FOLLOWING ARE YOTAM'S ADDITION (NOT in FALK's CODE)
        # gamma_shape_prior = self._get_current_parameter_value('gamma_shape_n', context)
        # gamma_size_prior = self._get_current_parameter_value('gamma_size_n', context)
        # MODIFIED 6/3/19 NEW:
        gamma_shape_prior = self.parameters.gamma_shape_n.default_value
        gamma_size_prior = self.parameters.gamma_size_n.default_value
        # MODIFIED 6/3/19 END


        variable = self._check_args(
            [convert_to_np_array(variable[0], dimension=2), convert_to_np_array(variable[1], dimension=2)],
            params,
            context,
        )
        predictors = variable[0]
        dependent_vars = variable[1].astype(float)

        # online update rules as per the given reference
        Lambda_n = (predictors.T @ predictors) + Lambda_prior
        mu_n = np.linalg.inv(Lambda_n) @ ((predictors.T @ dependent_vars) + (Lambda_prior @ mu_prior))
        gamma_shape_n = gamma_shape_prior + dependent_vars.shape[1]
        gamma_size_n = gamma_size_prior + (dependent_vars.T @ dependent_vars) \
            + (mu_prior.T @ Lambda_prior @ mu_prior) \
            - (mu_n.T @ Lambda_n @ mu_n)

        self.parameters.Lambda_prior._set(Lambda_prior, context)
        self.parameters.mu_prior._set(mu_prior, context)
        self.parameters.gamma_shape_prior._set(gamma_shape_prior, context)
        self.parameters.gamma_size_prior._set(gamma_size_prior, context)

        self.parameters.Lambda_n._set(Lambda_n, context)
        self.parameters.mu_n._set(mu_n, context)
        self.parameters.gamma_shape_n._set(gamma_shape_n, context)
        self.parameters.gamma_size_n._set(gamma_size_n, context)

        return self.sample_weights(gamma_shape_n, gamma_size_n, mu_n, Lambda_n, context)

    def sample_weights(self, gamma_shape_n, gamma_size_n, mu_n, Lambda_n, context):
        """Draw a sample of prediction weights from the distributions parameterized by `mu_n <BayesGLM.mu_n>`,
        `Lambda_n <BayesGLM.Lambda_n>`, `gamma_shape_n <BayesGLM.gamma_shape_n>`, and `gamma_size_n
        <BayesGLM.gamma_size_n>`.
        """
        random_state = self._get_current_parameter_value('random_state', context)

        phi = random_state.gamma(gamma_shape_n / 2, gamma_size_n / 2)
        return random_state.multivariate_normal(mu_n.reshape(-1,), phi * np.linalg.inv(Lambda_n))


class Kohonen(LearningFunction):  # -------------------------------------------------------------------------------
    """
    Kohonen(                       \
        default_variable=None,     \
        learning_rate=None,        \
        distance_measure=GAUSSIAN, \
        params=None,               \
        name=None,                 \
        prefs=None)

    Calculates a matrix of weight changes using the Kohenen learning rule.

    The Kohonen learning rule is used to implement a `Kohonen netowrk
    <http://scholarpedia.org/article/Kohonen_network>`_, which is an instance of the more general category of
    `self organizing map (SOM) <https://en.wikipedia.org/wiki/Self-organizing_map>`_, that modifies the weights to
    each element in the network in proportion to its difference from the current input pattern and the distance of
    that element from the one with the weights most similar to the current input pattern.

    `function <Kohonen.function>` calculates and returns a matrix of weight changes from an array of activity values
    (in `variable <Kohonen.variable>`\\[1]) and a weight matrix that generated them (in `variable
    <Kohonen.variable>`\\[2]) using the Kohonen learning rule:

    .. math::
        learning\\_rate * distance_j * variable[0]-w_j

    where :math:`distance_j` is the distance of the jth element of `variable <Kohonen.variable>`\\[1] from the
    element with the weights most similar to activity array in `variable <Kohonen.variable>`\\[1],
    and :math:`w_j` is the column of the matrix in `variable <Kohonen.variable>`\\[2] that corresponds to
    the jth element of the activity array in `variable <Kohonen.variable>`\\[1].

    .. _note::
       the array of activities in `variable <Kohonen.variable>`\\[1] is assumed to have been generated by the
       dot product of the input pattern in `variable <Kohonen.variable>`\\[0] and the matrix in `variable
       <Kohonen.variable>`\\[2], and thus the element with the greatest value in `variable <Kohonen.variable>`\\[1]
       can be assumed to be the one with weights most similar to the input pattern.


    Arguments
    ---------

    variable: List[array(float64), array(float64), 2d array[[float64]]] : default class_defaults.variable
        input pattern, array of activation values, and matrix used to calculate the weights changes.

    learning_rate : scalar or list, 1d or 2d array, or np.matrix of numeric values: default default_learning_rate
        specifies the learning rate used by the `function <Kohonen.function>`; supersedes any specification  for the
        `Process` and/or `System` to which the function's `owner <Function.owner>` belongs (see `learning_rate
        <Kohonen.learning_rate>` for details).

    distance_measure : GAUSSIAN, LINEAR, EXPONENTIAL, SINUSOID or function
        specifies the method used to calculate the distance of each element in `variable <Kohonen.variable>`\\[2]
        from the one with the greatest value.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the function.
        Values specified for parameters in the dictionary override any assigned to those parameters in arguments
        of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable: List[array(float64), array(float64), 2d array[[float64]]]
        input pattern, array of activation values, and weight matrix  used to generate the weight change matrix
        returned by `function <Kohonen.function>`.

    learning_rate : float, 1d or 2d array
        used by the `function <Kohonen.function>` to scale the weight change matrix returned by the `function
        <Kohonen.function>`.  If specified, it supersedes any learning_rate specified for the `Process
        <Process_Base_Learning>` and/or `System <System_Learning>` to which the function's `owner <Kohonen.owner>`
        belongs.  If it is a scalar, it is multiplied by the weight change matrix;  if it is a 1d array, it is
        multiplied Hadamard (elementwise) by the `variable` <Kohonen.variable>` before calculating the weight change
        matrix;  if it is a 2d array, it is multiplied Hadamard (elementwise) by the weight change matrix; if it is
        `None`, then the `learning_rate <Process.learning_rate>` specified for the Process to which the `owner
        <Kohonen.owner>` belongs is used;  and, if that is `None`, then the `learning_rate <System.learning_rate>`
        for the System to which it belongs is used. If all are `None`, then the `default_learning_rate
        <Kohonen.default_learning_rate>` is used.

    default_learning_rate : float
        the value used for the `learning_rate <Kohonen.learning_rate>` if it is not otherwise specified.

    function : function
         calculates a matrix of weight changes from: i) the difference between an input pattern (variable
         <Kohonen.variable>`\\[0]) and the weights in a weigh matrix (`variable <Kohonen.variable>`\\[2]) to each
         element of an activity array (`variable <Kohonen.variable>`\\[1]); and ii) the distance of each element of
         the activity array (variable <Kohonen.variable>`\\[1])) from the one with the weights most similar to the
         input array (variable <Kohonen.variable>`\\[0])) using `distance_measure <Kohonen.distance_measure>`.

    owner : Component
        `Mechanism <Mechanism>` to which the Function belongs.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).
    """

    componentName = KOHONEN_FUNCTION

    class Parameters(LearningFunction.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <Kohonen.variable>`

                    :default value: [[0, 0], [0, 0], numpy.array([[0, 0], [0, 0]])]
                    :type: ``list``
                    :read only: True

                distance_function
                    see `distance_function <Kohonen.distance_function>`

                    :default value: `GAUSSIAN`
                    :type: ``str``
        """
        variable = Parameter([[0, 0], [0, 0], np.array([[0, 0], [0, 0]])], read_only=True, pnl_internal=True, constructor_argument='default_variable')
        distance_function = Parameter(GAUSSIAN, stateful=False)

        def _validate_distance_function(self, distance_function):
            options = {GAUSSIAN, LINEAR, EXPONENTIAL}
            if distance_function in options:
                # returns None indicating no error message (this is a valid assignment)
                return None
            else:
                # returns error message
                return 'not one of {0}'.format(options)

    default_learning_rate = 0.05

    def __init__(self,
                 default_variable=None,
                 # learning_rate: tc.optional(tc.optional(parameter_spec)) = None,
                 learning_rate=None,
                 distance_function:tc.any(tc.enum(GAUSSIAN, LINEAR, EXPONENTIAL), is_function_type)=None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

        super().__init__(
            default_variable=default_variable,
            distance_function=distance_function,
            learning_rate=learning_rate,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _validate_variable(self, variable, context=None):
        variable = super()._validate_variable(variable, context)

        # variable = np.squeeze(np.array(variable))

        name = self.name
        if self.owner and self.owner.name:
            name = name + " for {}".format(self.owner.name)

        if not is_numeric(variable):
            raise ComponentError("Variable for {} ({}) contains non-numeric entries".
                                 format(name, variable))

        if len(variable)!=3:
            raise FunctionError("variable for {} has {} items ({}) but must have three:  "
                                "input pattern (1d array), activity array (1d array) and matrix (2d array)"
                                "".format(name, len(variable), variable))

        input = np.array(variable[0])
        activity = np.array(variable[1])
        matrix = np.array(variable[2])

        if input.ndim != 1:
            raise FunctionError("First item of variable ({}) for {} must be a 1d array".
                                format(input, name))

        if activity.ndim != 1:
            raise FunctionError("Second item of variable ({}) for {} must be a 1d array".
                                format(activity, name))

        if matrix.ndim != 2:
            raise FunctionError("Third item of variable ({}) for {} must be a 2d array or matrix".
                                format(matrix, name))

        if len(input) != len(activity):
            raise FunctionError("Length of first ({}) and second ({}) items of variable for {} must be the same".
                                format(len(input), len(activity), name))

        #     VALIDATE THAT len(variable[0])==len(variable[1])==len(variable[2].shape)
        if (len(input) != matrix.shape[0]) or (matrix.shape[0] != matrix.shape[1]):
            raise FunctionError("Third item of variable for {} ({}) must be a square matrix the dimension of which "
                                "must be the same as the length ({}) of the first and second items of the variable".
                                format(name, matrix, len(input)))

        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate learning_rate
        """
        super()._validate_params(request_set=request_set, target_set=target_set, context=context)
        if LEARNING_RATE in target_set and target_set[LEARNING_RATE] is not None:
            self._validate_learning_rate(target_set[LEARNING_RATE], AUTOASSOCIATIVE)

    def _instantiate_attributes_before_function(self, function=None, context=None):
        super()._instantiate_attributes_before_function(function, context)

        if isinstance(self.distance_function, str):
            self.measure=self.distance_function
            self.distance_function = scalar_distance

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        variable : array or List[1d array, 1d array, 2d array] : default class_defaults.variable
           input pattern, array of activation values, and matrix used to calculate the weights changes.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the function.
            Values specified for parameters in the dictionary override any assigned to those parameters in arguments
            of the constructor.

        Returns
        -------

        weight change matrix : 2d array
            matrix of weight changes scaled by difference of the current weights from the input pattern in
            `variable <Kohonen.variable>`\\[0] and the distance of each element from the one with the weights most
            similar to that input pattern.

        """

        # IMPLEMENTATION NOTE: have to do this here, rather than in validate_params for the following reasons:
        #                      1) if no learning_rate is specified for the Mechanism, need to assign None
        #                          so that the process or system can see it is free to be assigned
        #                      2) if neither the system nor the process assigns a value to the learning_rate,
        #                          then need to assign it to the default value
        # If learning_rate was not specified for instance or composition, use default value
        learning_rate = self._get_current_parameter_value(LEARNING_RATE, context)
        if learning_rate is None:
            learning_rate = self.defaults.learning_rate

        # FIX: SHOULD PUT THIS ON SUPER (THERE, BUT NEEDS TO BE DEBUGGED)
        learning_rate_dim = None
        if learning_rate is not None:
            learning_rate_dim = np.array(learning_rate).ndim

        # KDM 9/3/20: if learning_rate comes from a parameter port, it
        # will be 1d and will be multiplied twice (variable ->
        # activities -> distances)
        # If learning_rate is a 1d array, multiply it by variable
        if learning_rate_dim == 1:
            variable = variable * learning_rate

        input_pattern = np.atleast_2d(variable[0]).transpose()
        activities = np.atleast_2d(variable[1]).transpose()
        matrix = variable[2]
        measure = self.distance_function

        # Calculate I-w[j]
        input_cols = np.repeat(input_pattern,len(input_pattern),1)
        differences = matrix - input_cols

        # Calculate distances
        index_of_max = list(activities).index(max(activities))
        distances = np.zeros_like(activities)
        for i, item in enumerate(activities):
            distances[i] = self.distance_function(self.measure, abs(i - index_of_max))
        distances = 1 - np.atleast_2d(distances).transpose()

        # Multiply distances by differences and learning_rate
        weight_change_matrix = distances * differences * learning_rate

        return self.convert_output_type(weight_change_matrix)


class Hebbian(LearningFunction):  # -------------------------------------------------------------------------------
    """
    Hebbian(                    \
        default_variable=None,  \
        learning_rate=None,     \
        params=None,            \
        name=None,              \
        prefs=None)

    .. _Hebbian_Learning_Rule:

    Calculate a matrix of weight changes using the Hebbian (correlational) learning rule.

    `function <Hebbian.function>` calculates a matrix of weight changes from a 1d array of activity values in `variable
    <Hebbian.variable>` using the `Hebbian learning rule <https://en.wikipedia.org/wiki/Hebbian_theory#Principles>`_:

    .. math::

        \\Delta w_{ij} = learning\\_rate * a_ia_j\\ if\\ i \\neq j,\\ else\\ 0

    where :math:`a_i` and :math:`a_j` are elements of `variable <Hebbian.variable>`.


    Arguments
    ---------

    variable : List[number] or 1d array : default class_defaults.variable
       specifies the activation values, the pair-wise products of which are used to generate the a weight change matrix.

    COMMENT:
    activation_function : Function or function : SoftMax
        specifies the `function <Mechanism_Base.function>` of the `Mechanism <Mechanism>` that generated the array of
        activations in `variable <Hebbian.variable>`.
    COMMENT

    learning_rate : scalar or list, 1d or 2d array, or np.matrix of numeric values: default default_learning_rate
        specifies the learning rate used by the `function <Hebbian.function>`; supersedes any specification  for the
        `Process` and/or `System` to which the function's `owner <Function.owner>` belongs (see `learning_rate
        <Hebbian.learning_rate>` for details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the function.
        Values specified for parameters in the dictionary override any assigned to those parameters in arguments
        of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).
    Attributes
    ----------

    variable: 1d array
        activation values, the pair-wise products of which are used to generate the weight change matrix returned by
        the `function <Hebbian.function>`.

    COMMENT:
    activation_function : Function or function : SoftMax
        the `function <Mechanism_Base.function>` of the `Mechanism <Mechanism>` that generated the array of activations
        in `variable <Hebbian.variable>`.
    COMMENT

    learning_rate : float, 1d or 2d array
        used by the `function <Hebbian.function>` to scale the weight change matrix returned by the `function
        <Hebbian.function>`.  If specified, it supersedes any learning_rate specified for the `Process
        <Process_Base_Learning>` and/or `System <System_Learning>` to which the function's `owner <Hebbian.owner>`
        belongs.  If it is a scalar, it is multiplied by the weight change matrix;  if it is a 1d array, it is
        multiplied Hadamard (elementwise) by the `variable` <Hebbian.variable>` before calculating the weight change
        matrix;  if it is a 2d array, it is multiplied Hadamard (elementwise) by the weight change matrix; if it is
        `None`, then the `learning_rate <Process.learning_rate>` specified for the Process to which the `owner
        <Hebbian.owner>` belongs is used;  and, if that is `None`, then the `learning_rate <System.learning_rate>`
        for the System to which it belongs is used. If all are `None`, then the `default_learning_rate
        <Hebbian.default_learning_rate>` is used.

    default_learning_rate : float
        the value used for the `learning_rate <Hebbian.learning_rate>` if it is not otherwise specified.

    function : function
         calculates the pairwise product of all elements in the `variable <Hebbian.variable>`, and then
         scales that by the `learning_rate <Hebbian.learning_rate>` to generate the weight change matrix
         returned by the function.

    owner : Component
        `Mechanism <Mechanism>` to which the Function belongs.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).
    """

    componentName = HEBBIAN_FUNCTION

    class Parameters(LearningFunction.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <Hebbian.variable>`

                    :default value: numpy.array([0, 0])
                    :type: ``numpy.ndarray``
                    :read only: True

                learning_rate
                    see `learning_rate <Hebbian.learning_rate>`

                    :default value: 0.05
                    :type: ``float``
        """
        variable = Parameter(np.array([0, 0]), read_only=True, pnl_internal=True, constructor_argument='default_variable')
        learning_rate = Parameter(0.05, modulable=True)
    default_learning_rate = 0.05

    def __init__(self,
                 default_variable=None,
                 learning_rate=None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

        super().__init__(
            default_variable=default_variable,
            learning_rate=learning_rate,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _validate_variable(self, variable, context=None):
        variable = super()._validate_variable(variable, context)

        variable = np.squeeze(np.array(variable))

        if not is_numeric(variable):
            raise ComponentError("Variable for {} ({}) contains non-numeric entries".
                                 format(self.name, variable))
        if variable.ndim == 0:
            raise ComponentError("Variable for {} is a single number ({}) "
                                 "which doesn't make much sense for associative learning".
                                 format(self.name, variable))
        if variable.ndim > 1:
            raise ComponentError("Variable for {} ({}) must be a list or 1d np.array of numbers".
                                 format(self.name, variable))
        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate learning_rate
        """
        super()._validate_params(request_set=request_set, target_set=target_set, context=context)
        if LEARNING_RATE in target_set and target_set[LEARNING_RATE] is not None:
            self._validate_learning_rate(target_set[LEARNING_RATE], AUTOASSOCIATIVE)

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        variable : List[number] or 1d array : default class_defaults.variable
            array of activity values, the pairwise products of which are used to generate a weight change matrix.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the function.
            Values specified for parameters in the dictionary override any assigned to those parameters in arguments
            of the constructor.

        Returns
        -------

        weight change matrix : 2d array
            matrix of weight changes generated by the `Hebbian Learning rule <Hebbian_Learning_Rule>`,
            with all diagonal elements = 0 (i.e., hollow matix).

        """

        self._check_args(variable=variable, context=context, params=params)

        # IMPLEMENTATION NOTE: have to do this here, rather than in validate_params for the following reasons:
        #                      1) if no learning_rate is specified for the Mechanism, need to assign None
        #                          so that the process or system can see it is free to be assigned
        #                      2) if neither the system nor the process assigns a value to the learning_rate,
        #                          then need to assign it to the default value
        # If learning_rate was not specified for instance or composition, use default value
        learning_rate = self._get_current_parameter_value(LEARNING_RATE, context)
        # learning_rate = self.learning_rate
        if learning_rate is None:
            learning_rate = self.defaults.learning_rate
        #
        # # FIX: SHOULD PUT THIS ON SUPER (THERE, BUT NEEDS TO BE DEBUGGED)
        learning_rate_dim = None
        if learning_rate is not None:
            learning_rate_dim = np.array(learning_rate).ndim

        # MODIFIED 9/21/17 NEW:
        # FIX: SHOULDN'T BE NECESSARY TO DO THIS;  WHY IS IT GETTING A 2D ARRAY AT THIS POINT?
        if not isinstance(variable, np.ndarray):
            variable = np.array(variable)
        if variable.ndim > 1:
            variable = np.squeeze(variable)
        # MODIFIED 9/21/17 END

        # Generate the column array from the variable
        # col = variable.reshape(len(variable),1)
        col = np.atleast_2d(variable).transpose()

        # If learning_rate is a 1d array, multiply it by variable
        # KDM 11/21/19: if learning_rate comes from a parameter_port, it will
        # be 1 dimensional even if it "should" be a float. This causes test
        # failures
        # KDM 8/17/20: fix by determining col first. learning_rate otherwise
        # would be multiplied twice
        if learning_rate_dim == 1:
            variable = variable * learning_rate

        # Calculate weight chhange matrix
        weight_change_matrix = variable * col
        # Zero diagonals (i.e., don't allow correlation of a unit with itself to be included)
        weight_change_matrix = weight_change_matrix * (1 - np.identity(len(variable)))

        # If learning_rate is scalar or 2d, multiply it by the weight change matrix
        if learning_rate_dim in {0, 2}:
            weight_change_matrix = weight_change_matrix * learning_rate

        return self.convert_output_type(weight_change_matrix)


class ContrastiveHebbian(LearningFunction):  # -------------------------------------------------------------------------
    """
    ContrastiveHebbian(         \
        default_variable=None,  \
        learning_rate=None,     \
        params=None,            \
        name=None,              \
        prefs=None)

    .. _ContrastiveHebbian_Learning_Rule:

    Calculate a matrix of weight changes using the Contrastive Hebbian learning rule.

    `function <ContrastiveHebbian.function>` calculates a matrix of weight changes from a 1d array of activity values
    in `variable <ContrastiveHebbian.variable>` using the `ContrastiveHebbian learning rule
    <https://www.sciencedirect.com/science/article/pii/B978148321448150007X>`_:


    .. math::
       \\Delta w_{ij} = learning\\_rate * (a_i^+a_j^+ - a_i^-a_j^-) \\ if\\ i \\neq j,\\ else\\ 0

    where :math:`a_i^+` and :math:`a_j^+` are the activites of elements of `variable <ContrastiveHebbian.variable>`
    in the `plus_phase <ContrastiveHebbian_Plus_Phase>` of execution, and :math:`a_i^-` and :math:`a_j^-` are the
    activities of those elements in the `minus phase <ContrastiveHebbian_Minus_Phase>` of execution.

    Arguments
    ---------

    variable : List[number] or 1d array : default class_defaults.variable
       specifies the activation values, the pair-wise products of which are used to generate the a weight change matrix.

    COMMENT:
    activation_function : Function or function : SoftMax
        specifies the `function <Mechanism_Base.function>` of the `Mechanism <Mechanism>` that generated the array of
        activations in `variable <ContrastiveHebbian.variable>`.
    COMMENT

    learning_rate : scalar or list, 1d or 2d array, or np.matrix of numeric values: default default_learning_rate
        specifies the learning rate used by the `function <ContrastiveHebbian.function>`; supersedes any specification
        for the `Process` and/or `System` to which the function's `owner <ContrastiveHebbian.owner>` belongs (see
        `learning_rate <ContrastiveHebbian.learning_rate>` for details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the function.
        Values specified for parameters in the dictionary override any assigned to those parameters in arguments
        of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).
    Attributes
    ----------

    variable: 1d array
        activation values, the pair-wise products of which are used to generate the weight change matrix returned by
        the `function <ContrastiveHebbian.function>`.

    COMMENT:
    activation_function : Function or function : SoftMax
        the `function <Mechanism_Base.function>` of the `Mechanism <Mechanism>` that generated the array of activations
        in `variable <ContrastiveHebbian.variable>`.
    COMMENT

    learning_rate : float, 1d or 2d array
        used by the `function <ContrastiveHebbian.function>` to scale the weight change matrix returned by the `function
        <ContrastiveHebbian.function>`.  If specified, it supersedes any learning_rate specified for the `Process
        <Process_Base_Learning>` and/or `System <System_Learning>` to which the function's `owner
        <ContrastiveHebbian.owner>` belongs.  If it is a scalar, it is multiplied by the weight change matrix;  if it
        is a 1d array, it is multiplied Hadamard (elementwise) by the `variable` <ContrastiveHebbian.variable>`
        before calculating the weight change matrix;  if it is a 2d array, it is multiplied Hadamard (elementwise) by
        the weight change matrix; if it is `None`, then the `learning_rate <Process.learning_rate>` specified for the
        Process to which the `owner <ContrastiveHebbian.owner>` belongs is used;  and, if that is `None`, then the
        `learning_rate <System.learning_rate>` for the System to which it belongs is used. If all are `None`, then the
        `default_learning_rate <ContrastiveHebbian.default_learning_rate>` is used.

    default_learning_rate : float
        the value used for the `learning_rate <ContrastiveHebbian.learning_rate>` if it is not otherwise specified.

    function : function
         calculates the pairwise product of all elements in the `variable <ContrastiveHebbian.variable>`, and then
         scales that by the `learning_rate <ContrastiveHebbian.learning_rate>` to generate the weight change matrix
         returned by the function.

    owner : Component
        `Mechanism <Mechanism>` to which the Function belongs.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).
    """

    componentName = CONTRASTIVE_HEBBIAN_FUNCTION

    class Parameters(LearningFunction.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <ContrastiveHebbian.variable>`

                    :default value: numpy.array([0, 0])
                    :type: ``numpy.ndarray``
                    :read only: True
        """
        variable = Parameter(np.array([0, 0]), read_only=True, pnl_internal=True, constructor_argument='default_variable')

    default_learning_rate = 0.05

    def __init__(self,
                 default_variable=None,
                 # learning_rate: tc.optional(tc.optional(parameter_spec)) = None,
                 learning_rate=None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

        super().__init__(
            default_variable=default_variable,
            learning_rate=learning_rate,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _validate_variable(self, variable, context=None):
        variable = super()._validate_variable(variable, context)

        variable = np.squeeze(np.array(variable))

        if not is_numeric(variable):
            raise ComponentError("Variable for {} ({}) contains non-numeric entries".
                                 format(self.name, variable))
        if variable.ndim == 0:
            raise ComponentError("Variable for {} is a single number ({}) "
                                 "which doesn't make much sense for associative learning".
                                 format(self.name, variable))
        if variable.ndim > 1:
            raise ComponentError("Variable for {} ({}) must be a list or 1d np.array of numbers".
                                 format(self.name, variable))
        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate learning_rate
        """
        super()._validate_params(request_set=request_set, target_set=target_set, context=context)
        if LEARNING_RATE in target_set and target_set[LEARNING_RATE] is not None:
            self._validate_learning_rate(target_set[LEARNING_RATE], AUTOASSOCIATIVE)

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):
        """

        Arguments
        ---------

        variable : List[number] or 1d np.array : default class_defaults.variable
            array of activity values, the pairwise products of which are used to generate a weight change matrix.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the function.
            Values specified for parameters in the dictionary override any assigned to those parameters in arguments
            of the constructor.

        Returns
        -------

        weight change matrix : 2d array
            matrix of weight changes generated by the `ContrastiveHeabbian learning rule
            <ContrastiveHebbian_Learning_Rule>`, with all diagonal elements = 0 (i.e., hollow matix).

        """

        self._check_args(variable=variable, context=context, params=params)

        # IMPLEMENTATION NOTE: have to do this here, rather than in validate_params for the following reasons:
        #                      1) if no learning_rate is specified for the Mechanism, need to assign None
        #                          so that the process or system can see it is free to be assigned
        #                      2) if neither the system nor the process assigns a value to the learning_rate,
        #                          then need to assign it to the default value
        # If learning_rate was not specified for instance or composition, use default value
        learning_rate = self._get_current_parameter_value(LEARNING_RATE, context)
        if learning_rate is None:
            learning_rate = self.defaults.learning_rate

        # FIX: SHOULD PUT THIS ON SUPER (THERE, BUT NEEDS TO BE DEBUGGED)
        learning_rate_dim = None
        if learning_rate is not None:
            learning_rate_dim = np.array(learning_rate).ndim

        # MODIFIED 9/21/17 NEW:
        # FIX: SHOULDN'T BE NECESSARY TO DO THIS;  WHY IS IT GETTING A 2D ARRAY AT THIS POINT?
        if not isinstance(variable, np.ndarray):
            variable = np.array(variable)
        if variable.ndim > 1:
            variable = np.squeeze(variable)
        # MODIFIED 9/21/17 END

        # IMPLEMENTATION NOTE:  THE FOLLOWING NEEDS TO BE REPLACED BY THE CONTRASTIVE HEBBIAN LEARNING RULE:

        # Generate the column array from the variable
        # col = variable.reshape(len(variable),1)
        col = convert_to_np_array(variable, 2).transpose()

        # If learning_rate is a 1d array, multiply it by variable
        if learning_rate_dim == 1:
            variable = variable * learning_rate

        # Calculate weight chhange matrix
        weight_change_matrix = variable * col
        # Zero diagonals (i.e., don't allow correlation of a unit with itself to be included)
        weight_change_matrix = weight_change_matrix * (1 - np.identity(len(variable)))

        # If learning_rate is scalar or 2d, multiply it by the weight change matrix
        if learning_rate_dim in {0, 2}:
            weight_change_matrix = weight_change_matrix * learning_rate

        return self.convert_output_type(weight_change_matrix)


def _activation_input_getter(owning_component=None, context=None):
    return owning_component.parameters.variable._get(context)[LEARNING_ACTIVATION_INPUT]


def _activation_output_getter(owning_component=None, context=None):
    return owning_component.parameters.variable._get(context)[LEARNING_ACTIVATION_OUTPUT]


def _error_signal_getter(owning_component=None, context=None):
    return owning_component.parameters.variable._get(context)[LEARNING_ERROR_OUTPUT]



class Reinforcement(LearningFunction):  # -----------------------------------------------------------------------------
    """
    Reinforcement(                     \
        default_variable=None,         \
        learning_rate=None,            \
        params=None,                   \
        name=None,                     \
        prefs=None)

    Calculate error term for a single item in an input array, scaled by the learning_rate.

    `function <Reinforcement.function>` takes an array (`activation_output <Reinforcement.activation_output>`) with
    only one non-zero value, and returns an array of the same length with the one non-zero value replaced by
    :math:`\\Delta w` -- the `error_signal <Reinforcement.error_signal>` scaled by the `learning_rate
    <Reinforcement.learning_rate>`:

    .. math::
        \\Delta w_i = learning\\_rate * error\\_signal_i\\ if\\ activation\\_output_i \\neq 0,\\ otherwise\\ 0

    The non-zero item in `activation_output <Reinforcement.activation_output>` can be thought of as the predicted
    likelihood of a stimulus or value of an action, and the `error_signal <Reinforcement.error_signal>` as the error in
    the prediction for that value.

    .. _Reinforcement_Note:

    .. note::
       To preserve compatibility with other LearningFunctions:

       * the **variable** argument of both the constructor and calls to `function <Reinforcement.function>`
         must have three items, although only the 2nd and 3rd items are used; these are referenced by the
         `activation_output <Reinforcement.activation_output>` and `error_signal <Reinforcement.error_signal>`
         attributes, respectively (the first item is used by other LearningFunctions as their `activation_input
         <LearningFunction.activation_input>` attribute).
       ..
       * `function <Reinforcement.function>` returns two copies of the error array
         (the first is a "place-marker", where a matrix of weights changes is often returned).

    Arguments
    ---------

    default_variable : List or 2d array : default class_defaults.variable
       template for the three items provided as the variable in the call to the `function <Reinforcement.function>`
       (in order):

           * `activation_input <Reinforcement.activation_input>` (1d array) (not used);

           * `activation_output <Reinforcement.activation_output>` (1d array with only one non-zero value);

           * `error_signal <Reinforcement.error_signal>`  (1d array with a single scalar element).

    learning_rate : float : default default_learning_rate
        supersedes any specification for the `Process` and/or `System` to which the function's
        `owner <Function.owner>` belongs (see `learning_rate <Reinforcement.learning_rate>` for details).

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

    variable: 2d array
        specifies three values used as input to the `function <Reinforcement.function>`:

            * `activation_input <Reinforcement.activation_input>`,

            * `activation_output <Reinforcement.activation_output>`, and

            * `error_signal <Reinforcement.error_signal>`.

    activation_input : 1d array
        first item of `variable <Reinforcement.variable>`;  this is not used (it is implemented for compatibility
        with other LearningFunctions).

    activation_output : 1d array
        second item of `variable <Reinforcement.variable>`;  contains a single "prediction" or "action" value as one
        of its elements, the others of which are zero.

    error_signal : 1d array
        third item of `variable <Reinforcement.variable>`; contains a single scalar value, specifying the error
        associated with the non-zero item in `activation_output <Reinforcement.activation_output>`.

    learning_rate : float
        the learning rate used by the function.  If specified, it supersedes any learning_rate specified for the
        `Process <Process_Base_Learning>` and/or `System <System_Learning>` to which the function's
        `owner <Reinforcement.owner>` belongs.  If it is `None`, then the `learning_rate <Process.learning_rate>`
        specified for the Process to which the `owner <Reinforcement.owner>` belongs is used;  and, if that is `None`,
        then the `learning_rate <System.learning_rate>` for the System to which it belongs is used. If all are
        `None`, then the `default_learning_rate <Reinforcement.default_learning_rate>` is used.

    default_learning_rate : float
        the value used for the `learning_rate <Reinforcement.learning_rate>` if it is not otherwise specified.

    function : function
         the function that computes the weight change matrix, and returns that along with the
         `error_signal <Reinforcement.error_signal>` received.

    owner : Component
        `Mechanism <Mechanism>` to which the Function belongs.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).
    """

    componentName = RL_FUNCTION

    class Parameters(LearningFunction.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <Reinforcement.variable>`

                    :default value: numpy.array([[0], [0], [0]])
                    :type: ``numpy.ndarray``
                    :read only: True

                activation_input
                    see `activation_input <Reinforcement.activation_input>`

                    :default value: [0]
                    :type: ``list``
                    :read only: True

                activation_output
                    see `activation_output <Reinforcement.activation_output>`

                    :default value: [0]
                    :type: ``list``
                    :read only: True

                enable_output_type_conversion
                    see `enable_output_type_conversion <Reinforcement.enable_output_type_conversion>`

                    :default value: False
                    :type: ``bool``
                    :read only: True

                error_signal
                    see `error_signal <Reinforcement.error_signal>`

                    :default value: [0]
                    :type: ``list``
                    :read only: True
        """
        variable = Parameter(np.array([[0], [0], [0]]), read_only=True, pnl_internal=True, constructor_argument='default_variable')
        activation_input = Parameter([0], read_only=True, getter=_activation_input_getter)
        activation_output = Parameter([0], read_only=True, getter=_activation_output_getter)
        error_signal = Parameter([0], read_only=True, getter=_error_signal_getter)
        enable_output_type_conversion = Parameter(
            False,
            stateful=False,
            loggable=False,
            pnl_internal=True,
            read_only=True
        )

    def __init__(self,
                 default_variable=None,
                 # learning_rate: tc.optional(tc.optional(parameter_spec)) = None,
                 learning_rate=None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

        super().__init__(
            default_variable=default_variable,
            learning_rate=learning_rate,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _validate_variable(self, variable, context=None):
        variable = super()._validate_variable(variable, context)
        if len(variable) != 3:
            raise ComponentError("Variable for {} ({}) must have three items (input, output and error arrays)".
                                 format(self.name, variable))

        if len(variable[LEARNING_ERROR_OUTPUT]) != 1:
            raise ComponentError("Error term for {} (the third item of its variable arg) must be an array with a "
                                 "single element for {}".
                                 format(self.name, variable[LEARNING_ERROR_OUTPUT]))

        # Allow initialization with zero but not during a run (i.e., when called from check_args())
        if not self.is_initializing:
            if np.count_nonzero(variable[LEARNING_ACTIVATION_OUTPUT]) != 1:
                raise ComponentError(
                    "Second item ({}) of variable for {} must be an array with only one non-zero value "
                    "(if output Mechanism being trained uses softmax,"
                    " its \'output\' arg may need to be set to to PROB)".
                    format(variable[LEARNING_ACTIVATION_OUTPUT], self.componentName))

        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate learning_rate
        """
        super()._validate_params(request_set=request_set, target_set=target_set, context=context)
        if LEARNING_RATE in target_set and target_set[LEARNING_RATE] is not None:
            self._validate_learning_rate(target_set[LEARNING_RATE], AUTOASSOCIATIVE)

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 **kwargs):
        """

        Arguments
        ---------

        variable : List or 2d np.array [length 3 in axis 0] : default class_defaults.variable
           must have three items that are (in order):

               * `activation_input <Reinforcement.activation_input>` (not used);

               * `activation_output <Reinforcement.activation_output>` (1d array with only one non-zero value);

               * `error_signal <Reinforcement.error_signal>` (1d array with a single scalar element);

           (see `note <Reinforcement_Note>` above).

        params : Dict[param keyword: param value] : default None
           a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
           function.  Values specified for parameters in the dictionary override any assigned to those parameters in
           arguments of the constructor.

        Returns
        -------

        error array : List[1d array, 1d array]
            Both 1d arrays are the same, with a single non-zero error term (see `note <Reinforcement_Note>` above).

        """

        self._check_args(variable=variable, context=context, params=params)

        output = self._get_current_parameter_value(ACTIVATION_OUTPUT, context)
        error = self._get_current_parameter_value(ERROR_SIGNAL, context)
        learning_rate = self._get_current_parameter_value(LEARNING_RATE, context)

        # IMPLEMENTATION NOTE: have to do this here, rather than in validate_params for the following reasons:
        #                      1) if no learning_rate is specified for the Mechanism, need to assign None
        #                          so that the process or system can see it is free to be assigned
        #                      2) if neither the system nor the process assigns a value to the learning_rate,
        #                          then need to assign it to the default value
        # If learning_rate was not specified for instance or composition, use default value
        if learning_rate is None:
            learning_rate = self.defaults.learning_rate

        # Assign error term to chosen item of output array
        error_array = (np.where(output, learning_rate * error, 0))

        # Construct weight change matrix with error term in proper element
        weight_change_matrix = np.diag(error_array)
        return [error_array, error_array]


class BackPropagation(LearningFunction):
    """
    BackPropagation(                                     \
        default_variable=None,                           \
        activation_derivative_fct=Logistic().derivative, \
        learning_rate=None,                              \
        params=None,                                     \
        name=None,                                       \
        prefs=None)

    Calculate and return a matrix of weight changes and weighted error signal from arrays of inputs, outputs and error
    terms.

    `function <BackPropagation.function>` calculates a matrix of weight changes using the
    `backpropagation <https://en.wikipedia.org/wiki/Backpropagation>`_ (`Generalized Delta Rule
    <http://www.nature.com/nature/journal/v323/n6088/abs/323533a0.html>`_) learning algorithm, computed as:

    COMMENT:
        *weight_change_matrix* = `learning_rate <BackPropagation.learning_rate>` * `activation_input
        <BackPropagation.activation_input>` * :math:`\\frac{\\delta E}{\\delta W}`

            where:

               :math:`\\frac{\\delta E}{\\delta W}` = :math:`\\frac{\\delta E}{\\delta A} * \\frac{\\delta A}{\\delta W}`

                 is the derivative of the `error_signal <BackPropagation.error_signal>` with respect to the weights;

               :math:`\\frac{\\delta E}{\\delta A}` = `error_matrix <BackPropagation.error_matrix>` :math:`\\bullet`
               `error_signal <BackPropagation.error_signal>`

                 is the derivative of the error with respect to `activation_output
                 <BackPropagation.activation_output>` (i.e., the weighted contribution to the `error_signal
                 <BackPropagation.error_signal>` of each unit that receives activity from the weight matrix being
                 learned); and

               :math:`\\frac{\\delta A}{\\delta W}` =
               `activation_derivative_fct <BackPropagation.activation_derivative_fct>`
               (*input =* `activation_input <BackPropagation.activation_input>`,
               *output =* `activation_output <BackPropagation.activation_output>`\\)

                 is the derivative of the activation function responsible for generating `activation_output
                 <BackPropagation.activation_output>` at the point that generates each of its entries.
    COMMENT

        .. math::
            weight\\_change\\_matrix = learning\\_rate * activation\\_input * \\frac{\\delta E}{\\delta W}

        where

        .. math::
            \\frac{\\delta E}{\\delta W} = \\frac{\\delta E}{\\delta A} * \\frac{\\delta A}{\\delta W}

        is the derivative of the error_signal with respect to the weights, in which:

          :math:`\\frac{\\delta E}{\\delta A} = error\\_matrix \\bullet error\\_signal`

          is the derivative of the error with respect to activation_output (i.e., the weighted contribution to the
          error_signal of each unit that receives activity from the weight matrix being learned), and

          :math:`\\frac{\\delta A}{\\delta W} = activation\\_derivative\\_fct(input=activation\\_input,output=activation\\_output)`

          is the derivative of the activation function responsible for generating activation_output
          at the point that generates each of its entries.

    The values of `activation_input <BackPropagation.activation_input>`, `activation_output
    <BackPropagation.activation_output>` and  `error_signal <BackPropagation.error_signal>` are specified as
    items of the `variable <BackPropgation.variable>` both in the constructor for the BackPropagation Function,
    and in calls to its `function <BackPropagation.function>`.  Although `error_matrix <BackPropagation.error_matrix>`
    is not specified in the constructor, it is required as an argument of the `function <BackPropagation.function>`;
    it is assumed that it's value is determined in context at the time of execution (e.g., by a `LearningMechanism`
    that uses the BackPropagation LearningFunction).

    The BackPropagation `function <BackPropagation.function>` returns the *weight_change_matrix* as well as the
    `error_signal <BackPropagation.error_signal>` it receives weighted by the contribution made by each
    element of `activation_output <BackPropagation.activation_output>` as a function of the
    `error_matrix <BackPropagation.error_matrix>` :math:`\\frac{\\delta E}{\\delta W}`.

    Arguments
    ---------

    variable : List or 2d array [length 3 in axis 0] : default class_defaults.variable
       specifies a template for the three items provided as the variable in the call to the
       `function <BackPropagation.function>` (in order):
       `activation_input <BackPropagation.activation_input>` (1d array),
       `activation_output <BackPropagation.activation_output>` (1d array),
       `error_signal <BackPropagation.error_signal>` (1d array).

    activation_derivative_fct : Function or function
        specifies the derivative for the function of the Mechanism that generates
        `activation_output <BackPropagation.activation_output>`.

    COMMENT:
    error_derivative : Function or function
        specifies the derivative for the function of the Mechanism that is the receiver of the
        `error_matrix <BackPropagation.error_matrix>`.
    COMMENT

    COMMENT:
    error_matrix : List, 2d array, np.matrix, ParameterPort, or MappingProjection
        matrix, the output of which is used to calculate the `error_signal <BackPropagation.error_signal>`.
        If it is specified as a ParameterPort it must be one for the `matrix <MappingProjection.matrix>`
        parameter of a `MappingProjection`;  if it is a MappingProjection, it must be one with a
        MATRIX parameterPort.
    COMMENT

    learning_rate : float : default default_learning_rate
        supersedes any specification for the `Process` and/or `System` to which the function's
        `owner <Function.owner>` belongs (see `learning_rate <BackPropagation.learning_rate>` for details).

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

    variable: 2d array
        contains the three values used as input to the `function <BackPropagation.function>`:
       `activation_input <BackPropagation.activation_input>`,
       `activation_output <BackPropagation.activation_output>`, and
       `error_signal <BackPropagation.error_signal>`.

    activation_input : 1d array
        the input to the matrix being modified; same as 1st item of `variable <BackPropagation.variable>`.

    activation_output : 1d array
        the output of the function for which the matrix being modified provides the input;
        same as 2nd item of `variable <BackPropagation.variable>`.

    activation_derivative_fct : Function or function
        the derivative for the function of the Mechanism that generates
        `activation_output <BackPropagation.activation_output>`.

    error_signal : 1d array
        the error signal for the next matrix (layer above) in the learning pathway, or the error computed from the
        target (training signal) and the output of the last Mechanism in the sequence;
        same as 3rd item of `variable <BackPropagation.variable>`.

    error_matrix : 2d array or ParameterPort
        matrix, the input of which is `activation_output <BackPropagation.activation_output>` and the output of which
        is used to calculate the `error_signal <BackPropagation.error_signal>`; if it is a `ParameterPort`,
        it refers to the MATRIX parameterPort of the `MappingProjection` being learned.

    learning_rate : float
        the learning rate used by the function.  If specified, it supersedes any learning_rate specified for the
        `process <Process.learning_Rate>` and/or `system <System.learning_rate>` to which the function's  `owner
        <BackPropagation.owner>` belongs.  If it is `None`, then the learning_rate specified for the process to
        which the `owner <BackPropagation.owner>` belongs is used;  and, if that is `None`, then the learning_rate for
        the system to which it belongs is used. If all are `None`, then the
        `default_learning_rate <BackPropagation.default_learning_rate>` is used.

    default_learning_rate : float
        the value used for the `learning_rate <BackPropagation.learning_rate>` if it is not otherwise specified.

    loss_function : string : default 'MSE'
        the operation to apply to the error signal before computing weight changes.

    owner : Component
        `Mechanism <Mechanism>` to which the Function belongs.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).
    """

    componentName = BACKPROPAGATION_FUNCTION

    class Parameters(LearningFunction.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <BackPropagation.variable>`

                    :default value: numpy.array([[0], [0], [0]])
                    :type: ``numpy.ndarray``
                    :read only: True

                activation_derivative_fct
                    see `activation_derivative_fct <BackPropagation.activation_derivative_fct>`

                    :default value: `Logistic`.derivative
                    :type: ``types.FunctionType``

                activation_input
                    see `activation_input <BackPropagation.activation_input>`

                    :default value: [0]
                    :type: ``list``
                    :read only: True

                activation_output
                    see `activation_output <BackPropagation.activation_output>`

                    :default value: [0]
                    :type: ``list``
                    :read only: True

                error_matrix
                    see `error_matrix <BackPropagation.error_matrix>`

                    :default value: None
                    :type:
                    :read only: True

                error_signal
                    see `error_signal <BackPropagation.error_signal>`

                    :default value: [0]
                    :type: ``list``
                    :read only: True

                learning_rate
                    see `learning_rate <BackPropagation.learning_rate>`

                    :default value: 1.0
                    :type: ``float``

                loss_function
                    see `loss_function <BackPropagation.loss_function>`

                    :default value: None
                    :type:
                    :read only: True
        """
        variable = Parameter(np.array([[0], [0], [0]]), read_only=True, pnl_internal=True, constructor_argument='default_variable')
        learning_rate = Parameter(1.0, modulable=True)
        loss_function = Parameter(None, read_only=True)

        activation_input = Parameter([0], read_only=True, getter=_activation_input_getter)
        activation_output = Parameter([0], read_only=True, getter=_activation_output_getter)
        error_signal = Parameter([0], read_only=True, getter=_error_signal_getter)

        error_matrix = Parameter(None, read_only=True)

        activation_derivative_fct = Parameter(Logistic.derivative, stateful=False, loggable=False)

    default_learning_rate = 1.0

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 activation_derivative_fct: tc.optional(tc.optional(tc.any(types.FunctionType, types.MethodType))) = None,
                 # learning_rate: tc.optional(tc.optional(parameter_spec)) = None,
                 learning_rate=None,
                 loss_function=None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):

        error_matrix = np.zeros((len(default_variable[LEARNING_ACTIVATION_OUTPUT]),
                                 len(default_variable[LEARNING_ERROR_OUTPUT])))

        # self.return_val = ReturnVal(None, None)

        super().__init__(
            default_variable=default_variable,
            activation_derivative_fct=activation_derivative_fct,
            error_matrix=error_matrix,
            learning_rate=learning_rate,
            loss_function=loss_function,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    @property
    def output_type(self):
        return self._output_type

    @output_type.setter
    def output_type(self, value):
        # disabled because it happens during normal execution, may be confusing
        # warnings.warn('output_type conversion disabled for {0}'.format(self.__class__.__name__))
        self._output_type = None

    def _validate_variable(self, variable, context=None):
        variable = super()._validate_variable(variable, context)

        if len(variable) != 3:
            raise ComponentError("Variable for {} ({}) must have three items: {}, {}, and {})".
                                 format(self.name, variable, ACTIVATION_INPUT, ACTIVATION_OUTPUT, ERROR_SIGNAL))

        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate learning_rate and error_matrix params

        `error_matrix` argument must be one of the following
            - 2d list, np.ndarray or np.matrix
            - ParameterPort for one of the above
            - MappingProjection with a parameterPorts[MATRIX] for one of the above

        Parse error_matrix specification and insure it is compatible with error_signal and activation_output

        Insure that the length of the error_signal matches the number of cols (receiver elements) of error_matrix
            (since it will be dot-producted to generate the weighted error signal)

        Insure that length of activation_output matches the number of rows (sender elements) of error_matrix
           (since it will be compared against the *result* of the dot product of the error_matrix and error_signal

        Note: error_matrix is left in the form in which it was specified so that, if it is a ParameterPort
              or MappingProjection, its current value can be accessed at runtime (i.e., it can be used as a "pointer")
        """

        # # MODIFIED 3/22/17 OLD:
        # # This allows callers to specify None as learning_rate (e.g., _instantiate_learning_components)
        # if request_set[LEARNING_RATE] is None:
        #     request_set[LEARNING_RATE] = 1.0
        # # request_set[LEARNING_RATE] = request_set[LEARNING_RATE] or 1.0
        # # MODIFIED 3/22/17 END

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        if LEARNING_RATE in target_set and target_set[LEARNING_RATE] is not None:
            self._validate_learning_rate(target_set[LEARNING_RATE], AUTOASSOCIATIVE)

        # Validate error_matrix specification
        if ERROR_MATRIX in target_set:

            error_matrix = target_set[ERROR_MATRIX]

            from psyneulink.core.components.ports.parameterport import ParameterPort
            from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
            if not isinstance(error_matrix, (list, np.ndarray, np.matrix, ParameterPort, MappingProjection)):
                raise FunctionError("The {} arg for {} ({}) must be a list, 2d np.array, ParamaterState or "
                                    "MappingProjection".format(ERROR_MATRIX, self.__class__.__name__, error_matrix))

            if isinstance(error_matrix, MappingProjection):
                try:
                    error_matrix = error_matrix._parameter_ports[MATRIX].value
                    param_type_string = "MappingProjection's ParameterPort"
                except KeyError:
                    raise FunctionError("The MappingProjection specified for the {} arg of {} ({}) must have a {} "
                                        "paramaterState that has been assigned a 2d array or matrix".
                                        format(ERROR_MATRIX, self.__class__.__name__, error_matrix.shape, MATRIX))

            elif isinstance(error_matrix, ParameterPort):
                try:
                    error_matrix = error_matrix.value
                    param_type_string = "ParameterPort"
                except KeyError:
                    raise FunctionError("The value of the {} parameterPort specified for the {} arg of {} ({}) "
                                        "must be a 2d array or matrix".
                                        format(MATRIX, ERROR_MATRIX, self.__class__.__name__, error_matrix.shape))

            else:
                param_type_string = "array or matrix"

            error_matrix = np.array(error_matrix)
            rows = error_matrix.shape[WT_MATRIX_SENDERS_DIM]
            cols = error_matrix.shape[WT_MATRIX_RECEIVERS_DIM]
            activity_output_len = len(self.defaults.variable[LEARNING_ACTIVATION_OUTPUT])
            error_signal_len = len(self.defaults.variable[LEARNING_ERROR_OUTPUT])

            if error_matrix.ndim != 2:
                raise FunctionError("The value of the {} specified for the {} arg of {} ({}) "
                                    "must be a 2d array or matrix".
                                    format(param_type_string, ERROR_MATRIX, self.name, error_matrix))

            # The length of the sender outputPort.value (the error signal) must be the
            #     same as the width (# columns) of the MappingProjection's weight matrix (# of receivers)

            # Validate that columns (number of receiver elements) of error_matrix equals length of error_signal
            if cols != error_signal_len:
                raise FunctionError("The width (number of columns, {}) of the \'{}\' arg ({}) specified for {} "
                                    "must match the length of the error signal ({}) it receives".
                                    format(cols, MATRIX, error_matrix.shape, self.name, error_signal_len))

            # Validate that rows (number of sender elements) of error_matrix equals length of activity_output,
            if rows != activity_output_len:
                raise FunctionError("The height (number of rows, {}) of \'{}\' arg specified for {} must match the "
                                    "length of the output {} of the activity vector being monitored ({})".
                                    format(rows, MATRIX, self.name, activity_output_len))

    def _function(self,
                 variable=None,
                 context=None,
                 error_matrix=None,
                 params=None,
                 **kwargs):
        """
        .. note::
           Both variable and error_matrix must be specified for the function to execute.

        Arguments
        ---------

        variable : List or 2d array [length 3 in axis 0]
           must have three items that are the values for (in order):
           `activation_input <BackPropagation.activation_input>` (1d array),
           `activation_output <BackPropagation.activation_output>` (1d array),
           `error_signal <BackPropagation.error_signal>` (1d array).

        error_matrix : List, 2d array, np.matrix, ParameterPort, or MappingProjection
            matrix of weights that were used to generate the `error_signal <BackPropagation.error_signal>` (3rd item
            of `variable <BackPropagation.variable>` from `activation_output <BackPropagation.activation_output>`;
            its dimensions must be the length of `activation_output <BackPropagation.activation_output>` (rows) x
            length of `error_signal <BackPropagation.error_signal>` (cols).

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        weight change matrix, weighted error signal : List[2d array, 1d array]
            the modifications to make to the matrix, `error_signal <BackPropagation.error_signal>` weighted by the
            contribution made by each element of `activation_output <BackPropagation.activation_output>` as a
            function of `error_matrix <BackPropagation.error_matrix>`.
        """

        self._check_args(variable=variable, context=context, params=params)

        # Manage error_matrix param
        # During init, function is called directly from Component (i.e., not from LearningMechanism execute() method),
        #     so need "placemarker" error_matrix for validation
        if error_matrix is None:
            if self.is_initializing:
                error_matrix = np.zeros(
                    (len(variable[LEARNING_ACTIVATION_OUTPUT]), len(variable[LEARNING_ERROR_OUTPUT]))
                )
            # Raise exception if error_matrix is not specified
            else:
                owner_string = ""
                if self.owner:
                    owner_string = " of " + self.owner.name
                raise FunctionError("Call to {} function{} must include \'ERROR_MATRIX\' in params arg".
                                    format(self.__class__.__name__, owner_string))

        self.parameters.error_matrix._set(error_matrix, context)
        # self._check_args(variable=variable, context=context, params=params, context=context)

        # Manage learning_rate
        # IMPLEMENTATION NOTE: have to do this here, rather than in validate_params for the following reasons:
        #                      1) if no learning_rate is specified for the Mechanism, need to assign None
        #                          so that the process or system can see it is free to be assigned
        #                      2) if neither the system nor the process assigns a value to the learning_rate,
        #                          then need to assign it to the default value
        # If learning_rate was not specified for instance or composition, use default value
        learning_rate = self._get_current_parameter_value(LEARNING_RATE, context)
        if learning_rate is None:
            learning_rate = self.defaults.learning_rate

        # make activation_input a 1D row array
        activation_input = self._get_current_parameter_value(ACTIVATION_INPUT, context)
        activation_input = np.array(activation_input).reshape(len(activation_input), 1)

        # Derivative of error with respect to output activity (contribution of each output unit to the error above)
        loss_function = self.parameters.loss_function.get(context)
        if loss_function == MSE:
            num_output_units = self._get_current_parameter_value(ERROR_SIGNAL, context).shape[0]
            dE_dA = np.dot(error_matrix, self._get_current_parameter_value(ERROR_SIGNAL, context)) / num_output_units * 2
        elif loss_function == SSE:
            dE_dA = np.dot(error_matrix, self._get_current_parameter_value(ERROR_SIGNAL, context)) * 2
        else:
            dE_dA = np.dot(error_matrix, self._get_current_parameter_value(ERROR_SIGNAL, context))

        # Derivative of the output activity
        activation_output = self._get_current_parameter_value(ACTIVATION_OUTPUT, context)
        # FIX: THIS ASSUMES DERIVATIVE CAN BE COMPUTED FROM output OF FUNCTION (AS IT CAN FOR THE Logistic)
        dA_dW = self.activation_derivative_fct(input=None, output=activation_output, context=context)

        # Chain rule to get the derivative of the error with respect to the weights
        dE_dW = dE_dA * dA_dW

        # Weight changes = delta rule (learning rate * activity * error)
        weight_change_matrix = learning_rate * activation_input * dE_dW

        return [weight_change_matrix, dE_dW]


class TDLearning(Reinforcement):
    """Implement temporal difference learning using the `Reinforcement` Function
    (see `Reinforcement` for class details).
    """
    componentName = TDLEARNING_FUNCTION

    def __init__(self,
                 default_variable=None,
                 learning_rate=None,
                 params=None,
                 owner=None,
                 prefs=None):
        """
        Dummy function used to implement TD Learning via Reinforcement Learning

        Parameters
        ----------
        default_variable
        learning_rate: float: default 0.05
        params
        owner
        prefs
        context
        """

        super().__init__(
            default_variable=default_variable,
            learning_rate=learning_rate,
            params=params,
            owner=owner,
            prefs=prefs
        )

    def _validate_variable(self, variable, context=None):
        variable = super(Reinforcement, self)._validate_variable(variable, context)

        if len(variable) != 3:
            raise ComponentError("Variable for {} ({}) must have three items (input, output, and error arrays)".
                                 format(self.name, variable))

        # if len(variable[LEARNING_ERROR_OUTPUT]) != len(variable[LEARNING_ACTIVATION_OUTPUT]):
        #     raise ComponentError("Error term does not match the length of the sample sequence")

        return variable
