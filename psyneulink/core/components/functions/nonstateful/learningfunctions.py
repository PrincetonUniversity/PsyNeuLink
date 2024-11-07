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

* `EMstorage`
* `BayesGLM`
* `Kohonen`
* `Hebbian`
* `ContrastiveHebbian`
* `Reinforcement`
* `BackPropagation`
* `TDLearning`

Functions that parameterize a function, usually of the matrix used by a `Projection's <Projection>` `Function`.

"""

import types
from collections import namedtuple

import numpy as np
try:
    import torch
except ImportError:
    torch = None
from beartype import beartype

from psyneulink._typing import Optional, Union, Literal, Callable

from psyneulink.core.components.component import ComponentError
from psyneulink.core.components.functions.function import (
    DEFAULT_SEED, Function_Base, FunctionError, _random_state_getter, _seed_setter,
)
from psyneulink.core.components.functions.nonstateful.transferfunctions import Logistic, SoftMax
from psyneulink.core.globals.context import handle_external_context
from psyneulink.core.globals.keywords import \
    CONTRASTIVE_HEBBIAN_FUNCTION, EM_STORAGE_FUNCTION, TDLEARNING_FUNCTION, LEARNING_FUNCTION_TYPE, LEARNING_RATE, \
    KOHONEN_FUNCTION, GAUSSIAN, LINEAR, EXPONENTIAL, HEBBIAN_FUNCTION, RL_FUNCTION, BACKPROPAGATION_FUNCTION, \
    MATRIX, Loss
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.preferences.basepreferenceset import ValidPrefSet
from psyneulink.core.globals.utilities import convert_all_elements_to_np_array, is_numeric, scalar_distance, convert_to_np_array, all_within_range, safe_len, is_numeric_scalar

__all__ = ['LearningFunction', 'Kohonen', 'Hebbian', 'ContrastiveHebbian',
           'Reinforcement', 'BayesGLM', 'BackPropagation', 'TDLearning', 'EMStorage',
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
COVARIATES = 'covariates'
ERROR_SIGNAL = 'error_signal'
ERROR_MATRIX = 'error_matrix'
MEMORY_MATRIX = 'memory_matrix'


ReturnVal = namedtuple('ReturnVal', 'learning_signal, error_signal')


class LearningFunction(Function_Base):
    """Abstract class of `Function <Function>` used for learning.

    .. technical_note::
       The ``_function`` method of a LearningFunction *must* include a **kwargs argument, which accomodates
       Function-specific parameters;  this is to accommodate the ability of LearningMechanisms to accomodate
       all LearningFunctions, by calling their ``_function`` method with arguments that may not be implemented
       for all LearningFunctions. Such arguments placed in the **params** argument of the ``_function`` method
       of a LearningFunction by Component._execute, and passed to the LearningFunction's ``_function`` method.

    Attributes
    ----------

    variable : list or array
        most LearningFunctions take a list or 2d array that must contain three items:

        * the input to the parameter being modified (variable[LEARNING_ACTIVATION_INPUT]);
        * the output of the parameter being modified (variable[LEARNING_ACTIVATION_OUTPUT]);
        * the error associated with the output (variable[LEARNING_ERROR_OUTPUT]).

        However, the exact specification depends on the function's type.

    learning_rate : array, float or int : function.defaults.parameter
        the value used for the function's `learning_rate <LearningFunction.learning_rate>` parameter, generally
        used to multiply the result of the function before it is returned;  however, both the form of the value (i.e.,
        whether it is a scalar or array) and how it is used depend on the function's type.  The parameter's default
        value is used if none of the following is specified:  the `learning_rate <LearningMechanism.learning_rate>`
        for the `LearningMechanism` to which the function has been assigned, the `learning_rate
        <Composition.learning_rate>` for any `Composition` to which that LearningMechanism belongs, or
        a **learning_rate** argument specified in either a Composition's `learning construction method
        <Composition_Learning_Methods>` or a call to the Composition's `learn <Composition.learn>` method at
        runtime (see description of learning_rate for `LearningMechanisms <LearningMechanism_Learning_Rate>`
        and for `Compositions <Composition_Learning_Rate>` for additional details).

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
        variable = Parameter(np.array([0, 0, 0]),
                             read_only=True,
                             pnl_internal=True,
                             constructor_argument='default_variable')
        learning_rate = Parameter(0.05,
                                  modulable=True)

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


class EMStorage(LearningFunction):
    """
    EMStorage(                 \
        default_variable=None, \
        axis=0,                \
        storage_location=None  \
        storage_prob=1.0,      \
        decay_rate=0.0,        \
        params=None,           \
        name=None,             \
        prefs=None)

    Assign an entry to a matrix with a specified probability specified by `storage_prob <EMStorage.storage_prob>.
    Used by `EMStorageMechanism` in an `EMComposition` to
    COMMENT:
    FROM CoPilot:
    implement the `Ebbinghaus illusion <https://en.wikipedia.org/wiki/Ebbinghaus_illusion>`_.
    COMMENT
    store an entry in `memory_matrix <EMStorage.memory_matrix>`.

    Its function takes the `entry <EMStorage.entry>` to be stored and `memory_matrix <EMStorage.memory_matrix>` in
    which to store it, and assigns the the entry to the corresponding location in the matrix with a probability
    specified by `storage_prob <EMStorage.storage_prob>`.

    Arguments
    ---------

    variable : List or 1d array : default class_defaults.variable
        specifies shape of `entry <EMStorage.entry>` passed in the call to the `function <EMStorage.function>`.

    axis : int : default 0
        specifies the axis of `memory_matrix <EMStorage.memory_matrix>` to which `entry <EMStorage.entry>` is assigned.

    storage_location : int : default None
        specifies the location (row or col determined by `axis <EMStorage.axis>`) of `memory_matrix
        <EMStorage.memory_matrix>` at which the new entry is stored (replacing the existing one);
        if None, the weeakest entry (one with the lowest norm) along `axis <EMStorage.axis>` of
        `memory_matrix <EMStorage.memory_matrix>` is used.

    storage_prob : float or int : default 1.0
        specifies the probability with which `entry <EMStorage.entry>` is assigned to `memory_matrix
        <EMStorage.memory_matrix>`.

    decay_rate : float or int : default 0.0
        specifies the rate at which pre-existing entries in `memory_matrix <EMStorage.memory_matrix>` are decayed.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those
        parameters in arguments of the constructor.d

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable: 2d array
        contains the value (`entry <EMStorage.entry>`) used as input to the `function <EMStorage.function>`.

    entry : 1d array
        value to be stored in `memory_matrix <EMStorage.memory_matrix>`.

    memory_matrix : 2d array or ParameterPort
        matrix to which the entry is assigned along `axis <EMstorage.axis>`.

    axis : int
        determines axis of `memory_matrix <EMStorage.memory_matrix>` to which `entry <EMStorage.entry>` is assigned.

    storage_location : int
        specifies the location (row or col determined by `axis <EMStorage.axis>`) of `memory_matrix
        <EMStorage.memory_matrix>` at which the new entry is stored.

    storage_prob : float
        determines the probability with which `entry <EMStorage.entry>` is stored in `memory_matrix
        <EMStorage.memory_matrix>`.

    decay_rate : float
        determines the rate at which pre-existing entries in `memory_matrix <EMStorage.memory_matrix>` are decayed.

    random_state : numpy.RandomState
        private pseudorandom number generator

    owner : Component
        `Mechanism <Mechanism>` to which the Function belongs.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).
    """

    componentName = EM_STORAGE_FUNCTION

    class Parameters(LearningFunction.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <EMStorage.variable>`

                    :default value: numpy.array([[0], [0], [0]])
                    :type: ``numpy.ndarray``
                    :read only: True

                axis
                    see `axis <EMStorage.axis>`

                    :default value: 0
                    :type: int
                    :read only: True

                decay_rate
                    see `decay_rate <EMStorage.axis>`

                    :default value: 0.0
                    :type: float

                entry
                    see `entry <EMStorage.error_signal>`

                    :default value: [0]
                    :type: ``list``
                    :read only: True

                memory_matrix
                    see `memory_matrix <EMStorage.memory_matrix>`

                    :default value: [[0][0]]
                    :type: ``np.ndarray``
                    :read only: True

                random_state
                    see `random_state <UniformDist.random_state>`

                    :default value: None
                    :type: ``numpy.random.RandomState``

                storage_location
                    see `storage_location <EMStorage.storage_location>`

                    :default value: None
                    :type: int

                storage_prob
                    see `storage_prob <EMStorage.storage_prob>`

                    :default value: 1.0
                    :type: ``float``

        """
        variable = Parameter(np.array([0]),
                             read_only=True,
                             pnl_internal=True,
                             constructor_argument='default_variable')
        entry = Parameter([0], read_only=True)
        memory_matrix = Parameter([[0],[0]], read_only=True)
        axis = Parameter(0, read_only=True, structural=True)
        storage_location = Parameter(None, read_only=True)
        storage_prob = Parameter(1.0, modulable=True)
        decay_rate = Parameter(0.0, modulable=True)
        random_state = Parameter(None, loggable=False, getter=_random_state_getter, dependencies='seed')
        seed = Parameter(DEFAULT_SEED(), modulable=True, fallback_default=True, setter=_seed_setter)

    def _validate_storage_prob(self, storage_prob):
        storage_prob = float(storage_prob)
        if not all_within_range(storage_prob, 0, 1):
            return f"must be a float in the interval [0,1]."


    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 axis=0,
                 storage_location=None,
                 storage_prob=1.0,
                 decay_rate=0.0,
                 seed=None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):

        super().__init__(
            default_variable=default_variable,
            axis=axis,
            storage_location=storage_location,
            storage_prob=storage_prob,
            decay_rate=decay_rate,
            seed=seed,
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

        if np.array(variable).ndim != 1:
            raise ComponentError(f"The number of items in the outer dimension of variable for '{self.name}' "
                                 f"(({len(variable)}) should be just one")
        return variable

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 **kwargs)->np.ndarray:
        """
        .. note::
           Both variable and error_matrix must be specified for the function to execute.

        Arguments
        ---------

        variable : List or 1d array
           array containing `entry <EMStorage.entry>` to be added to `memory_matrix <EMStorage.memory_matrix>`
           along `axis <EMStorage.axis>`.

        memory_matrix : List, 2d array, ParameterPort, or MappingProjection
            matrix to which `variable <EMStorage.variable>` is stored.

            .. technical_note::
               ``memory_matrix`` is listed here as an argument since it must be passed to the EMStorage Function;
               however it does not show in the signature for the function since it is passed through the `params
               <EMStorage.params>` argument, placed there by Component._execute.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------

        new memory_matrix : 2d array
            the new matrix contains `entry <EMStorage.entry>` stored in `memory_matrix <EMStorage.memory_matrix>`
            in slot with lowest norm along axis specified by `axis <EMStorage.axis>`.
        """

        self._check_args(variable=variable, context=context, params=params)

        entry = variable
        axis = self.parameters.axis._get(context)
        storage_location = self.parameters.storage_location._get(context)
        storage_prob = self.parameters.storage_prob._get(context)
        decay_rate = self.parameters.decay_rate._get(context)
        random_state = self.parameters.random_state._get(context)

        # FIX: IMPLEMENT decay_rate CALCULATION

        # IMPLEMENTATION NOTE: if memory_matrix is an arg, it must in params (put there by Component.function()
        # Manage memory_matrix param
        memory_matrix = None
        if params:
            memory_matrix = params.pop(MEMORY_MATRIX, None)
            axis = params.pop('axis', axis)
            storage_location = params.pop('storage_location', storage_location)
            storage_prob = params.pop('storage_prob', storage_prob)
            decay_rate = params.pop('decay_rate', decay_rate)
        # During init, function is called directly from Component (i.e., not from LearningMechanism execute() method),
        #     so need "placemarker" error_matrix for validation
        if memory_matrix is None:
            if self.is_initializing:
                # Can use variable, since it is 2d with length of an entry
                #   (so essentially a memory_matrix with one entry)
                memory_matrix = np.zeros((1,1,len(variable)))
            # Raise exception if memory_matrix is not specified
            else:
                owner_string = ""
                if self.owner:
                    owner_string = " of " + self.owner.name
                raise FunctionError(f"Call to {self.__class__.__name__} function {owner_string} "
                                    f"must include '{MEMORY_MATRIX}' in params arg.")

        if self.is_initializing:
            # Don't store entry during initialization to avoid contaminating memory_matrix
            pass
        elif random_state.uniform(0, 1) < storage_prob:
            if decay_rate:
                memory_matrix *= decay_rate
            if storage_location is not None:
                idx_of_min = storage_location
            else:
                # Find weakest entry (i.e., with lowest norm) along specified axis of matrix
                idx_of_min = np.argmin(np.linalg.norm(memory_matrix, axis=axis))
            if axis == 0:
                memory_matrix[:,idx_of_min] = np.array(entry)
            elif axis == 1:
                memory_matrix[idx_of_min,:] = np.array(entry)
            else:
                raise FunctionError(f"PROGRAM ERROR: axis ({axis}) is not 0 or 1")

            # Store entry in memory:
            self.parameters.memory_matrix._set(memory_matrix, context)
            # Record entry updated:
            self.parameters.entry._set(entry, context)

        return self.convert_output_type(memory_matrix)

    def _gen_pytorch_fct(self, device, context=None):
        def func(entry_to_store,
                 memory_matrix,
                 axis,
                 storage_location,
                 storage_prob,
                 decay_rate,
                 random_state)->torch.tensor:
            """Decay existing memories and replace weakest entry with entry_to_store (parallel EMStorage._function)"""
            if random_state.uniform(0, 1) < storage_prob:
                if decay_rate:
                    memory_matrix *= torch.tensor(decay_rate)
                if storage_location is not None:
                    idx_of_min = storage_location
                else:
                    # Find weakest entry (i.e., with lowest norm) along specified axis of matrix
                    idx_of_min = torch.argmin(torch.linalg.norm(memory_matrix, axis=axis))
                if axis == 0:
                    memory_matrix[:,idx_of_min] = entry_to_store
                elif axis == 1:
                    memory_matrix[idx_of_min,:] = entry_to_store
            return memory_matrix
        return func

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
        random_state = Parameter(None, loggable=False, getter=_random_state_getter, dependencies='seed')
        seed = Parameter(DEFAULT_SEED(), modulable=True, fallback_default=True, setter=_seed_setter)
        variable = Parameter([np.array([0, 0, 0]),
                              np.array([0])],
                             read_only=True,
                             pnl_internal=True,
                             constructor_argument='default_variable')
        value = Parameter(np.array([0]),
                          read_only=True,
                          aliases=['sample_weights'],
                          pnl_internal=True)

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

    @check_user_specified
    def __init__(self,
                 default_variable=None,
                 mu_0=None,
                 sigma_0=None,
                 gamma_shape_0=None,
                 gamma_size_0=None,
                 params=None,
                 owner=None,
                 seed=None,
                 prefs:  Optional[ValidPrefSet] = None):

        self.user_specified_default_variable = default_variable

        super().__init__(
            default_variable=default_variable,
            mu_0=mu_0,
            sigma_0=sigma_0,
            gamma_shape_0=gamma_shape_0,
            gamma_size_0=gamma_size_0,
            seed=seed,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _handle_default_variable(self, default_variable=None, input_shapes=None):

        # If default_variable was not specified by user...
        if default_variable is None and input_shapes in {None, NotImplemented}:
            #  but mu_0 and/or sigma_0 was specified as an array...
            if isinstance(self.mu_0, (list, np.ndarray)) or isinstance(self.sigma_0, (list, np.ndarray)):
                # if both are specified, make sure they are the same size
                if (isinstance(self.mu_0, (list, np.ndarray))
                        and isinstance(self.sigma_0, (list, np.ndarray))
                        and safe_len(self.mu_0) != safe_len(self.sigma_0)):
                    raise FunctionError("Length of {} ({}) does not match length of {} ({}) for {}".
                                        format(repr('mu_0'), safe_len(self.mu_0),
                                               repr('sigma_0'), safe_len(self.sigma_0),
                                                         self.__class.__.__name__))
                # allow their size to determine the size of variable
                if isinstance(self.mu_0, (list, np.ndarray)):
                    default_variable = [np.zeros_like(self.mu_0), np.zeros((1,1))]
                else:
                    default_variable = [np.zeros_like(self.sigma_0), np.zeros((1,1))]

        return super()._handle_default_variable(default_variable=default_variable, input_shapes=input_shapes)

    def initialize_priors(self):
        """Set the prior parameters (`mu_prior <BayesGLM.mu_prior>`, `Lamba_prior <BayesGLM.Lambda_prior>`,
        `gamma_shape_prior <BayesGLM.gamma_shape_prior>`, and `gamma_size_prior <BayesGLM.gamma_size_prior>`)
        to their initial (_0) values, and assign current (_n) values to the priors
        """

        variable = np.array(self.defaults.variable)
        variable = self.defaults.variable
        if np.array(variable).dtype != object:
            variable = np.atleast_2d(variable)

        n = safe_len(variable[0])

        if is_numeric_scalar(self.mu_0):
            self.mu_prior = np.full((n, 1),self.mu_0)
        else:
            if safe_len(self.mu_0) != n:
                raise FunctionError("Length of mu_0 ({}) does not match number of predictors ({})".
                                    format(safe_len(self.mu_0), n))
            self.mu_prior = np.array(self.mu_0).reshape(safe_len(self._mu_0), 1)

        if is_numeric_scalar(self.sigma_0):
            Lambda_0 = (1 / (self.sigma_0 ** 2)) * np.eye(n)
        else:
            if safe_len(self.sigma_0) != n:
                raise FunctionError("Length of sigma_0 ({}) does not match number of predictors ({})".
                                    format(safe_len(self.sigma_0), n))
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
        gamma_shape_n = np.array(gamma_shape_prior + dependent_vars.shape[1])
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

    .. note::
       the array of activities in `variable <Kohonen.variable>`\\[1] is assumed to have been generated by the
       dot product of the input pattern in `variable <Kohonen.variable>`\\[0] and the matrix in `variable
       <Kohonen.variable>`\\[2], and thus the element with the greatest value in `variable <Kohonen.variable>`\\[1]
       can be assumed to be the one with weights most similar to the input pattern.


    Arguments
    ---------

    variable: List[array(float64), array(float64), 2d array[[float64]]] : default class_defaults.variable
        input pattern, array of activation values, and matrix used to calculate the weights changes.

    learning_rate : scalar or list, 1d or 2d array of numeric values: default .05
        specifies the learning rate used by the `function <Kohonen.function>` (see `learning_rate
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
        <Kohonen.function>`.  If it is a scalar, it is multiplied by the weight change matrix;  if it is a 1d array,
        it is multiplied Hadamard (elementwise) by the `variable` <Kohonen.variable>` before calculating the weight
        change matrix;  if it is a 2d array, it is multiplied Hadamard (elementwise) by the weight change matrix. If
        learning_rate is not specified explicitly in the constructor for the function or otherwise (see `learning_rate
        <LearningMechanism.learning_rate>`) then the function's default learning_rate is used.

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
        variable = Parameter([[0, 0], [0, 0], np.array([[0, 0], [0, 0]])],
                             read_only=True,
                             pnl_internal=True,
                             constructor_argument='default_variable')
        distance_function = Parameter(GAUSSIAN, stateful=False)

        def _validate_distance_function(self, distance_function):
            options = {GAUSSIAN, LINEAR, EXPONENTIAL}
            if distance_function in options:
                # returns None indicating no error message (this is a valid assignment)
                return None
            else:
                # returns error message
                return 'not one of {0}'.format(options)

    @check_user_specified
    def __init__(self,
                 default_variable=None,
                 # learning_rate: Optional[ValidParamSpecType] = None,
                 learning_rate=None,
                 distance_function: Union[Literal['gaussian', 'linear', 'exponential'], Callable] = None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):

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
       specifies the activation values, the pair-wise products of which are used to generate the weight change matrix.

    COMMENT:
    activation_function : Function or function : SoftMax
        specifies the `function <Mechanism_Base.function>` of the `Mechanism <Mechanism>` that generated the array of
        activations in `variable <Hebbian.variable>`.
    COMMENT

    learning_rate : scalar or list, 1d or 2d array of numeric values: default .05
        specifies the learning rate used by the `function <Hebbian.function>`; (see `learning_rate
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
        <Hebbian.function>`.  If it is a scalar, it is multiplied by the weight change matrix;  if it is a 1d array,
        it is multiplied Hadamard (elementwise) by the `variable` <Hebbian.variable>` before calculating the weight
        change matrix;  if it is a 2d array, it is multiplied Hadamard (elementwise) by the weight change matrix. If
        learning_rate is not specified explicitly in the constructor for the function or otherwise (see `learning_rate
        <LearningMechanism.learning_rate>`) then the function's default learning_rate is used.

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

        """
        variable = Parameter(np.array([0, 0]),
                             read_only=True,
                             pnl_internal=True,
                             constructor_argument='default_variable')

    @check_user_specified
    def __init__(self,
                 default_variable=None,
                 learning_rate=None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):

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

        # If learning_rate was not specified for instance or composition, use default value
        learning_rate = self._get_current_parameter_value(LEARNING_RATE, context)
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

    learning_rate : scalar or list, 1d or 2d array of numeric values: default .05
        specifies the learning rate used by the `function <ContrastiveHebbian.function>`. (see `learning_rate
        <ContrastiveHebbian.learning_rate>` for details).

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
        <ContrastiveHebbian.function>`.  If it is a scalar, it is multiplied by the weight change matrix;  if it
        is a 1d array, it is multiplied Hadamard (elementwise) by the `variable` <ContrastiveHebbian.variable>`
        before calculating the weight change matrix;  if it is a 2d array, it is multiplied Hadamard (elementwise) by
        the weight change matrix.  If learning_rate is not specified explicitly in the constructor for the function
        or otherwise (see `learning_rate <LearningMechanism.learning_rate>`) then the function's defaultlearning_rate
        is used.

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
        variable = Parameter(np.array([0, 0]),
                             read_only=True,
                             pnl_internal=True,
                             constructor_argument='default_variable')

    @check_user_specified
    def __init__(self,
                 default_variable=None,
                 # learning_rate: Optional[ValidParamSpecType] = None,
                 learning_rate:Optional[Union[int,float]]=None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):

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
    try:
        return owning_component.parameters.variable._get(context)[LEARNING_ACTIVATION_INPUT]
    except (AttributeError, TypeError):
        return None


def _activation_output_getter(owning_component=None, context=None):
    try:
        return owning_component.parameters.variable._get(context)[LEARNING_ACTIVATION_OUTPUT]
    except (AttributeError, TypeError):
        return None


def _error_signal_getter(owning_component=None, context=None):
    try:
        return owning_component.parameters.variable._get(context)[LEARNING_ERROR_OUTPUT]
    except (AttributeError, TypeError):
        return None



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

    .. technical_note::
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

    learning_rate : float : default 0.05
        specifies the learning_rate used by the function (see `learning_rate <Reinforcement.learning_rate>` for
        details).

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
        the learning rate used by the function. If learning_rate is not specified explicitly in the constructor for
        the function or otherwise (see `learning_rate <LearningMechanism.learning_rate>`) then the function's default
        learning_rate is used.

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
        variable = Parameter(np.array([[0], [0], [0]]),
                             read_only=True,
                             pnl_internal=True,
                             constructor_argument='default_variable')
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

    @check_user_specified
    def __init__(self,
                 default_variable=None,
                 # learning_rate: Optional[ValidParamSpecType] = None,
                 learning_rate=None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):

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

        # Must have only one (or no) non-zero entries in LEARNING_ACTIVATION_OUTPUT
        if np.count_nonzero(variable[LEARNING_ACTIVATION_OUTPUT]) > 1:
            owner_str = f" of {self.owner.name}" if self.owner else ""
            raise ComponentError(f"Second item ({variable[LEARNING_ACTIVATION_OUTPUT]}) of variable for "
                                 f"{self.componentName}{owner_str} must be an array with no more than one non-zero "
                                 f"value; if output Mechanism being trained uses {SoftMax.componentName},"
                                 f" that function's \'output\' arg may need to be set to to 'PROB').")
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

        # If learning_rate was not specified for instance or composition, use default value
        if learning_rate is None:
            learning_rate = self.defaults.learning_rate

        # Assign error term to chosen item of output array
        error_array = (np.where(output, learning_rate * error, 0))

        # Construct weight change matrix with error term in proper element
        weight_change_matrix = np.diag(error_array)
        return convert_all_elements_to_np_array([error_array, error_array])


class TDLearning(Reinforcement):
    """Implement temporal difference learning using the `Reinforcement` Function
    (see `Reinforcement` for class details).
    """
    componentName = TDLEARNING_FUNCTION

    @check_user_specified
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


class BackPropagation(LearningFunction):
    """
    BackPropagation(                                     \
        default_variable=None,                           \
        activation_derivative_fct=Logistic().derivative, \
        learning_rate=None,                              \
        loss_spec=None,                                  \
        params=None,                                     \
        name=None,                                       \
        prefs=None)

    Calculate and return a matrix of weight changes and weighted error signal from arrays of inputs, outputs and error
    terms.

    This implements the standard form of the `backpropagation learning algorithm
    <https://en.wikipedia.org/wiki/Backpropagation>`_, using a form of loss determined by the `error_signal
    <LearningMechanism_Input_Error_Signal>` of the `LearningMechanism` to which it is assigned.

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
    <BackPropagation.activation_output>` and `error_signal <BackPropagation.error_signal>` are specified as
    items of the `variable <BackPropgation.variable>` both in the constructor for the BackPropagation Function,
    and in calls to its `function <BackPropagation.function>`.  If the activation function (i.e., the `function
    of that generates `activation_output <BackPropagation.activation_output>`) takes more than one argument that
    influence its `activation_function_derivative <BackPropagation.activation_function_derivative>`, then a
    template for these (exhibiting their shape) must be passed in the **covariates** argument of the constructor
    for the BackPropagation Function, and the values of these must be specified in the **covariates** argument of
    a call to its `function <BackPropagation.function>`, which is passed in the **params** argument of to the
    _function method.

    Although `error_matrix <BackPropagation.error_matrix>` is not specified in the constructor, it must be provided
    in the **error_matrix** argument of a call to the `function <BackPropagation.function>`, which is passed in the
    **params** argument to the _function method.

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

    covariates : List[1d array]
        specifies a template for values of arguments used by the `activation_derivative_fct
        <BackPropagation.activation_derivative_fct>` other than `activation_input <BackPropagation.activation_input>`
        and `activation_output <BackPropagation.activation_output>`, to compute the derivative of the activation
        function with respect to activation_output.

    COMMENT:
    error_derivative : Function or function
        specifies the derivative for the function of the Mechanism that is the receiver of the
        `error_matrix <BackPropagation.error_matrix>`.
    COMMENT

    COMMENT:
    error_matrix : List, 2d array, ParameterPort, or MappingProjection
        matrix, the output of which is used to calculate the `error_signal <BackPropagation.error_signal>`.
        If it is specified as a ParameterPort it must be one for the `matrix <MappingProjection.matrix>`
        parameter of a `MappingProjection`;  if it is a MappingProjection, it must be one with a
        MATRIX parameterPort.
    COMMENT

    learning_rate : float : default default_learning_rate
        specifies the learning_rate used by the function (see `learning_rate <BackPropagation.learning_rate>` for
        details).

    loss_spec : Loss : default None
        specifies the operation to apply to the error signal (i.e., method of calculating the derivative of the errror
        with respect to activation) before computing weight changes.

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
       `activation_output <BackPropagation.activation_output>`,
       `error_signal <BackPropagation.error_signal>`, and
       `covariates <BackPropagation.covariates>`.

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

    covariates : List[1d array]
        a template for the values of arguments used by the `activation_derivative_fct
        <BackPropagation.activation_derivative_fct>` other than activation_input and activation_output, to compute
        the derivative of the activation function
        with respect to activation_output.

    error_matrix : 2d array or ParameterPort
        matrix, the input of which is `activation_output <BackPropagation.activation_output>` and the output of which
        is used to calculate the `error_signal <BackPropagation.error_signal>`; if it is a `ParameterPort`,
        it refers to the MATRIX parameterPort of the `MappingProjection` being learned.

    learning_rate : float
        the learning rate used by the function.   If learning_rate is not specified explicitly in the constructor for
        the function or otherwise (see `learning_rate <LearningMechanism.learning_rate>`) then the function's default
        learning_rate is used.

    loss_spec : Loss or None
        the operation to apply to the error signal (i.e., method of calculating the derivative of the errror
        with respect to activation) before computing weight changes.

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

                covariates
                    see `covariates <BackPropagation.covariates>`

                    :default value: []
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

                loss_spec
                    see `loss_spec <BackPropagation.loss_spec>`

                    :default value: None
                    :type:
                    :read only: True
        """
        variable = Parameter(np.array([[0], [0], [0]]),
                             read_only=True,
                             pnl_internal=True,
                             constructor_argument='default_variable')
        loss_spec = Parameter(None, read_only=True)
        activation_input = Parameter([0], read_only=True, getter=_activation_input_getter)
        activation_output = Parameter([0], read_only=True, getter=_activation_output_getter)
        error_signal = Parameter([0], read_only=True, getter=_error_signal_getter)
        covariates = Parameter([], read_only=True)
        error_matrix = Parameter(None, read_only=True)
        activation_derivative_fct = Parameter(Logistic.derivative, stateful=False, loggable=False)

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 activation_derivative_fct: Optional[Union[types.FunctionType, types.MethodType]]=None,
                 covariates=None,
                 learning_rate:Optional[Union[int,float]]=None,
                 loss_spec=None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None):

        default_variable = default_variable if default_variable is not None else [[0], [0], [0]]
        try:
            error_matrix = np.zeros((len(default_variable[LEARNING_ACTIVATION_OUTPUT]),
                                     len(default_variable[LEARNING_ERROR_OUTPUT])))
        except IndexError:
            error_matrix = None

        super().__init__(
            default_variable=default_variable,
            activation_derivative_fct=activation_derivative_fct,
            covariates=covariates,
            error_matrix=error_matrix,
            learning_rate=learning_rate,
            loss_spec=loss_spec,
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
            raise ComponentError(f"Variable for '{self.name}' ({variable}) must have three items: "
                                 f"{ACTIVATION_INPUT}, {ACTIVATION_OUTPUT}, and {ERROR_SIGNAL}).")

        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate learning_rate and error_matrix params

        `error_matrix` argument must be one of the following
            - 2d list, np.ndarray
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

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        if LEARNING_RATE in target_set and target_set[LEARNING_RATE] is not None:
            self._validate_learning_rate(target_set[LEARNING_RATE], AUTOASSOCIATIVE)

        # FIX: ADD VALIDATION OF COVARIATES, INSURING THAT # OF THEM MATCHES NUMBER OF ARGS TAKEN BY ACTIVATION FUNCTION
        #      AND THAT DERIVATIVE EXISTS AND TAKES COVARIATES AS ARGS

        # Validate error_matrix specification
        if ERROR_MATRIX in target_set:

            error_matrix = target_set[ERROR_MATRIX]

            from psyneulink.core.components.ports.parameterport import ParameterPort
            from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
            if not isinstance(error_matrix, (list, np.ndarray, np.matrix, ParameterPort, MappingProjection)):
                raise FunctionError(f"The '{ERROR_MATRIX}' arg for {self.__class__.__name__} ({error_matrix}) "
                                    f"must be a list, 2d np.array, ParamaterPort or MappingProjection.")

            if isinstance(error_matrix, MappingProjection):
                try:
                    error_matrix = error_matrix._parameter_ports[MATRIX].value
                    param_type_string = "MappingProjection's ParameterPort"
                except KeyError:
                    raise FunctionError(f"The MappingProjection specified for the '{ERROR_MATRIX}' arg of "
                                        f"of {self.__class__.__name__} ({error_matrix.shape}) must have a "
                                        f"{MATRIX} ParamaterPort that has been assigned a 2d array or matrix.")

            elif isinstance(error_matrix, ParameterPort):
                try:
                    error_matrix = error_matrix.value
                    param_type_string = "ParameterPort"
                except KeyError:
                    raise FunctionError(f"The value of the {MATRIX} ParameterPort specified for the '{ERROR_MATRIX}' "
                                        f"arg of {self.__class__.__name__} ({error_matrix.shape}).")

            else:
                param_type_string = "array or matrix"

            error_matrix = np.array(error_matrix)
            rows = error_matrix.shape[WT_MATRIX_SENDERS_DIM]
            cols = error_matrix.shape[WT_MATRIX_RECEIVERS_DIM]
            activity_output_len = len(self.defaults.variable[LEARNING_ACTIVATION_OUTPUT])
            error_signal_len = len(self.defaults.variable[LEARNING_ERROR_OUTPUT])

            if error_matrix.ndim != 2:
                raise FunctionError(f"The value of the {param_type_string} specified for the '{ERROR_MATRIX}' arg "
                                    f"of '{self.name}' ({error_matrix}) must be a 2d array or matrix.")

            # The length of the sender outputPort.value (the error signal) must be the
            #     same as the width (# columns) of the MappingProjection's weight matrix (# of receivers)

            # Validate that columns (number of receiver elements) of error_matrix equals length of error_signal
            if cols != error_signal_len:
                raise FunctionError(f"The width (number of columns, {cols}) of the '{MATRIX}' arg "
                                    f"({error_matrix.shape}) specified for '{self.name}' must match "
                                    f"the length of the error signal ({error_signal_len}) it receives.")

            # Validate that rows (number of sender elements) of error_matrix equals length of activity_output,
            if rows != activity_output_len:
                activation_input = self._get_current_parameter_value(ACTIVATION_INPUT, context)
                raise FunctionError(f"The height (number of rows, {rows}) of '{MATRIX}' arg specified for "
                                    f"'{self.name}' must match the length of the output {activity_output_len} "
                                    f"of the activity vector being monitored ({activation_input}).")

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 **kwargs):
        """
        .. note::
           Both variable and error_matrix must be specified for the function to execute, with
           error_matrix passed in the `params` argument.

        Arguments
        ---------

        variable : List or 2d array [length 3 in axis 0]
           must have three items that are the values for (in order):
           `activation_input <BackPropagation.activation_input>` (1d array),
           `activation_output <BackPropagation.activation_output>` (1d array),
           `error_signal <BackPropagation.error_signal>` (1d array).

        covariates : List[1d array]
            values of arguments used by the `activation_derivative_fct <BackPropagation.activation_derivative_fct>`
            other than activation_input and activation_output, to compute the derivative of the activation function
            with respect to `activation_output <BackPropagation.activation_output>`.

        error_matrix : List, 2d array, ParameterPort, or MappingProjection
            matrix of weights that were used to generate the `error_signal <BackPropagation.error_signal>` (3rd item
            of `variable <BackPropagation.variable>` from `activation_output <BackPropagation.activation_output>`;
            its dimensions must be the length of `activation_output <BackPropagation.activation_output>` (rows) x
            length of `error_signal <BackPropagation.error_signal>` (cols).

            .. technical_note::
               ``error_matrix`` is listed here as an argument since it must be passed to the BackPropagation Function;
               however it does not show in the signature for the function since it is passed through the `params
               <BackPropagation.params>` argument, placed there by Component._execute.

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

        # IMPLEMENTATION NOTE: if error_matrix is an arg, it must in params (put there by Component._execute)
        error_matrix = None
        if params:
            error_matrix = params.pop(ERROR_MATRIX, None)

        covariates = None
        if params:
            covariates = params.pop(COVARIATES, None)

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
                raise FunctionError(f"Call to {self.__class__.__name__} function {owner_string} "
                                    f"must include '{ERROR_MATRIX}' in params arg.")

        self.parameters.error_matrix._set(error_matrix, context)
        # self._check_args(variable=variable, context=context, params=params, context=context)

        # If learning_rate was not specified for instance or composition or in params, use default value
        if params and LEARNING_RATE in params and params[LEARNING_RATE] is not None:
            learning_rate = params[LEARNING_RATE]
        else:
            learning_rate = self._get_current_parameter_value(LEARNING_RATE, context)
        if learning_rate is None:
            learning_rate = self.defaults.learning_rate

        # IMPLEMENTATION NOTE: FOR DEBUGGING
        # if not self.is_initializing:
        #     print(f"{self.owner.name}: executed learning_rate: {learning_rate}")

        # make activation_input a 1D row array
        activation_input = self._get_current_parameter_value(ACTIVATION_INPUT, context)
        activation_input = np.array(activation_input).reshape(len(activation_input), 1)

        # Derivative of error with respect to output activity (contribution of each output unit to the error above)
        loss_spec = self.parameters.loss_spec.get(context)
        if loss_spec == Loss.MSE:
            num_output_units = self._get_current_parameter_value(ERROR_SIGNAL, context).shape[0]
            dE_dA = np.dot(error_matrix, self._get_current_parameter_value(ERROR_SIGNAL, context)) / num_output_units * 2
        elif loss_spec == Loss.SSE:
            dE_dA = np.dot(error_matrix, self._get_current_parameter_value(ERROR_SIGNAL, context)) * 2
        else:
            # Use L0 (this applies to hidden layers) (Jacobian vector product)
            dE_dA = np.dot(error_matrix, self._get_current_parameter_value(ERROR_SIGNAL, context))

        # Derivative of the output activity
        activation_output = self._get_current_parameter_value(ACTIVATION_OUTPUT, context)
        if covariates is None:
            dA_dW = self.activation_derivative_fct(input=None, output=activation_output, context=context)
        else:
            dA_dW = self.activation_derivative_fct(input=None, output=activation_output,
                                                   covariates=covariates, context=context)

        # Chain rule to get the derivative of the error with respect to the weights
        if np.array(dA_dW).ndim <= 1:
            dE_dW = dE_dA * dA_dW
        elif dA_dW.ndim == 2:
            dE_dW = np.matmul(dE_dA, dA_dW)
        else:
            owner_str = f" of {self.owner.name}" if self.owner else ""
            raise FunctionError(f"Dimensionality of dA_dW ({dA_dW.ndim}) for {self.name}{owner_str} is not 1 or 2.")

        # Weight changes = delta rule (learning rate * activity * error)
        weight_change_matrix = learning_rate * activation_input * dE_dW

        return convert_all_elements_to_np_array([weight_change_matrix, dE_dW])
