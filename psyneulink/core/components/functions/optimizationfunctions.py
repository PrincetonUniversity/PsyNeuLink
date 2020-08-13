#
# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ******************************************   OPTIMIZATION FUNCTIONS **************************************************
"""
Contents
--------

* `OptimizationFunction`
* `GradientOptimization`
* `GridSearch`
* `GaussianProcess`
COMMENT:
uncomment this when ParamEstimationFunction is ready for users
* `ParamEstimationFunction`
COMMENT

Overview
--------

Functions that return the sample of a variable yielding the optimized value of an objective_function.

"""

import warnings
import sys
# from fractions import Fraction
import itertools
import numpy as np
import typecheck as tc
from numbers import Number

from typing import Iterator

from psyneulink.core.components.functions.function import Function_Base, is_function_type
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.defaults import MPI_IMPLEMENTATION
from psyneulink.core.globals.keywords import \
    DEFAULT_VARIABLE, GRADIENT_OPTIMIZATION_FUNCTION, GRID_SEARCH_FUNCTION, GAUSSIAN_PROCESS_FUNCTION, \
    OPTIMIZATION_FUNCTION_TYPE, OWNER, VALUE, VARIABLE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.utilities import call_with_pruned_args, get_global_seed
from psyneulink.core.globals.sampleiterator import SampleIterator

from psyneulink.core import llvm as pnlvm
import contextlib

__all__ = ['OptimizationFunction', 'GradientOptimization', 'GridSearch', 'GaussianProcess',
           'ParamEstimationFunction',
           'OBJECTIVE_FUNCTION', 'SEARCH_FUNCTION', 'SEARCH_SPACE', 'SEARCH_TERMINATION_FUNCTION',
           'DIRECTION', 'ASCENT', 'DESCENT', 'MAXIMIZE', 'MINIMIZE']

OBJECTIVE_FUNCTION = 'objective_function'
SEARCH_FUNCTION = 'search_function'
SEARCH_SPACE = 'search_space'
SEARCH_TERMINATION_FUNCTION = 'search_termination_function'
DIRECTION = 'direction'

class OptimizationFunctionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class OptimizationFunction(Function_Base):
    """
    OptimizationFunction(                            \
    default_variable=None,                           \
    objective_function=lambda x:0,                   \
    search_function=lambda x:x,                      \
    search_space=[0],                                \
    search_termination_function=lambda x,y,z:True,   \
    save_samples=False,                              \
    save_values=False,                               \
    max_iterations=None,                             \
    params=Nonse,                                    \
    owner=Nonse,                                     \
    prefs=None)

    Provides an interface to subclasses and external optimization functions. The default `function
    <OptimizationFunction.function>` executes iteratively, generating samples from `search_space
    <OptimizationFunction.search_space>` using `search_function <OptimizationFunction.search_function>`,
    evaluating them using `objective_function <OptimizationFunction.objective_function>`, and reporting the
    value of each using `report_value <OptimizationFunction.report_value>` until terminated by
    `search_termination_function <OptimizationFunction.search_termination_function>`. Subclasses can override
    `function <OptimizationFunction.function>` to implement their own optimization function or call an external one.

    Samples in `search_space <OptimizationFunction.search_space>` are assumed to be a list of one or more
    `SampleIterator` objects.

    .. _OptimizationFunction_Procedure:

    **Default Optimization Procedure**

    When `function <OptimizationFunction.function>` is executed, it iterates over the following steps:

        - get sample from `search_space <OptimizationFunction.search_space>` by calling `search_function
          <OptimizationFunction.search_function>`
        ..
        - compute value of `objective_function <OptimizationFunction.objective_function>` for the sample
          by calling `objective_function <OptimizationFunction.objective_function>`;
        ..
        - report value returned by `objective_function <OptimizationFunction.objective_function>` for the sample
          by calling `report_value <OptimizationFunction.report_value>`;
        ..
        - evaluate `search_termination_function <OptimizationFunction.search_termination_function>`.

    The current iteration numberris contained in `iteration <OptimizationFunction.iteration>`. Iteration continues until
    all values of `search_space <OptimizationFunction.search_space>` have been evaluated and/or
    `search_termination_function <OptimizationFunction.search_termination_function>` returns `True`.  The `function
    <OptimizationFunction.function>` returns:

    - the last sample evaluated (which may or may not be the optimal value, depending on the `objective_function
      <OptimizationFunction.objective_function>`);

    - the value of `objective_function <OptimizationFunction.objective_function>` associated with the last sample;

    - two lists, that may contain all of the samples evaluated and their values, depending on whether `save_samples
      <OptimizationFunction.save_samples>` and/or `save_vales <OptimizationFunction.save_values>` are `True`,
      respectively.

    .. _OptimizationFunction_Defaults:

    .. note::

        An OptimizationFunction or any of its subclasses can be created by calling its constructor.  This provides
        runnable defaults for all of its arguments (see below). However these do not yield useful results, and are
        meant simply to allow the  constructor of the OptimziationFunction to be used to specify some but not all of
        its parameters when specifying the OptimizationFunction in the constructor for another Component. For
        example, an OptimizationFunction may use for its `objective_function <OptimizationFunction.objective_function>`
        or `search_function <OptimizationFunction.search_function>` a method of the Component to which it is being
        assigned;  however, those methods will not yet be available, as the Component itself has not yet been
        constructed. This can be handled by calling the OptimizationFunction's `reset
        <OptimizationFunction.reset>` method after the Component has been instantiated, with a parameter
        specification dictionary with a key for each entry that is the name of a parameter and its value the value to
        be assigned to the parameter.  This is done automatically for Mechanisms that take an ObjectiveFunction as
        their `function <Mechanism_Base.function>` (such as the `OptimizationControlMechanism`), but will require it be
        done explicitly for Components for which that is not the case. A warning is issued if defaults are used for
        the arguments of an OptimizationFunction or its subclasses;  this can be suppressed by specifying the
        relevant argument(s) as `NotImplemnted`.

    COMMENT:
    NOTES TO DEVELOPERS:
    - Constructors of subclasses should include **kwargs in their constructor method, to accomodate arguments required
      by some subclasses but not others (e.g., search_space needed by `GridSearch` but not `GradientOptimization`) so
      that subclasses can be used interchangeably by OptimizationMechanisms.

    - Subclasses with attributes that depend on one of the OptimizationFunction's parameters should implement the
      `reset <OptimizationFunction.reset>` method, that calls super().reset(*args) and then
      reassigns the values of the dependent attributes accordingly.  If an argument is not needed for the subclass,
      `NotImplemented` should be passed as the argument's value in the call to super (i.e., the OptimizationFunction's
      constructor).
    COMMENT


    Arguments
    ---------

    default_variable : list or ndarray : default None
        specifies a template for (i.e., an example of the shape of) the samples used to evaluate the
        `objective_function <OptimizationFunction.objective_function>`.

    objective_function : function or method : default None
        specifies function used to evaluate sample in each iteration of the `optimization process
        <OptimizationFunction_Procedure>`; if it is not specified, a default function is used that simply returns
        the value passed as its `variable <OptimizationFunction.variable>` parameter (see `note
        <OptimizationFunction_Defaults>`).

    search_function : function or method : default None
        specifies function used to select a sample for `objective_function <OptimizationFunction.objective_function>`
        in each iteration of the `optimization process <OptimizationFunction_Procedure>`.  It **must be specified**
        if the `objective_function <OptimizationFunction.objective_function>` does not generate samples on its own
        (e.g., as does `GradientOptimization`).  If it is required and not specified, the optimization process
        executes exactly once using the value passed as its `variable <OptimizationFunction.variable>` parameter
        (see `note <OptimizationFunction_Defaults>`).

    search_space : list or array of SampleIterators : default None
        specifies iterators used by `search_function <OptimizationFunction.search_function>` to generate samples
        evaluated `objective_function <OptimizationFunction.objective_function>` in each iteration of the
        `optimization process <OptimizationFunction_Procedure>`. It **must be specified**
        if the `objective_function <OptimizationFunction.objective_function>` does not generate samples on its own
        (e.g., as does `GradientOptimization`). If it is required and not specified, the optimization process
        executes exactly once using the value passed as its `variable <OptimizationFunction.variable>` parameter
        (see `note <OptimizationFunction_Defaults>`).

    search_termination_function : function or method : None
        specifies function used to terminate iterations of the `optimization process <OptimizationFunction_Procedure>`.
        It must return a boolean value, and it  **must be specified** if the
        `objective_function <OptimizationFunction.objective_function>` is not overridden.  If it is required and not
        specified, the optimization process executes exactly once (see `note <OptimizationFunction_Defaults>`).

    save_samples : bool
        specifies whether or not to save and return the values of the samples used to evalute `objective_function
        <OptimizationFunction.objective_function>` over all iterations of the `optimization process
        <OptimizationFunction_Procedure>`.

    save_values : bool
        specifies whether or not to save and return the values of `objective_function
        <OptimizationFunction.objective_function>` for samples evaluated in all iterations of the
        `optimization process <OptimizationFunction_Procedure>`.

    max_iterations : int : default 1000
        specifies the maximum number of times the `optimization process <OptimizationFunction_Procedure>` is allowed
        to iterate; if exceeded, a warning is issued and the function returns the last sample evaluated.


    Attributes
    ----------

    variable : ndarray
        first sample evaluated by `objective_function <OptimizationFunction.objective_function>` (i.e., one used to
        evaluate it in the first iteration of the `optimization process <OptimizationFunction_Procedure>`).

    objective_function : function or method
        used to evaluate the sample in each iteration of the `optimization process <OptimizationFunction_Procedure>`.

    search_function : function, method or None
        used to select a sample evaluated by `objective_function <OptimizationFunction.objective_function>`
        in each iteration of the `optimization process <OptimizationFunction_Procedure>`.  `NotImplemented` if
        the `objective_function <OptimizationFunction.objective_function>` generates its own samples.

    search_space : list or array of `SampleIterators <SampleIterator>`
        used by `search_function <OptimizationFunction.search_function>` to generate samples evaluated by
        `objective_function <OptimizationFunction.objective_function>` in each iteration of the `optimization process
        <OptimizationFunction_Procedure>`.  The number of SampleIterators in the list determines the dimensionality
        of each sample:  in each iteration of the `optimization process <OptimizationFunction_Procedure>`, each
        SampleIterator is called upon to provide the value for one of the dimensions of the sample.m`NotImplemented`
        if the `objective_function <OptimizationFunction.objective_function>` generates its own samples.  If it is
        required and not specified, the optimization process executes exactly once using the value passed as its
        `variable <OptimizationFunction.variable>` parameter (see `note <OptimizationFunction_Defaults>`).

    search_termination_function : function or method that returns a boolean value
        used to terminate iterations of the `optimization process <OptimizationFunction_Procedure>`; if it is required
        and not specified, the optimization process executes exactly once (see `note <OptimizationFunction_Defaults>`).

    iteration : int
        the current iteration of the `optimization process <OptimizationFunction_Procedure>`.

    max_iterations : int : default 1000
        specifies the maximum number of times the `optimization process <OptimizationFunction_Procedure>` is allowed
        to iterate; if exceeded, a warning is issued and the function returns the last sample evaluated.

    save_samples : bool
        determines whether or not to save the values of the samples used to evalute `objective_function
        <OptimizationFunction.objective_function>` over all iterations of the `optimization process
        <OptimizationFunction_Procedure>`.

    save_values : bool
        determines whether or not to save and return the values of `objective_function
        <OptimizationFunction.objective_function>` for samples evaluated in all iterations of the
        `optimization process <OptimizationFunction_Procedure>`.
    """

    componentType = OPTIMIZATION_FUNCTION_TYPE

    class Parameters(Function_Base.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <OptimizationFunction.variable>`

                    :default value: numpy.array([0, 0, 0])
                    :type: ``numpy.ndarray``
                    :read only: True

                max_iterations
                    see `max_iterations <OptimizationFunction.max_iterations>`

                    :default value: None
                    :type:

                objective_function
                    see `objective_function <OptimizationFunction.objective_function>`

                    :default value: lambda x: 0
                    :type: ``types.FunctionType``

                save_samples
                    see `save_samples <OptimizationFunction.save_samples>`

                    :default value: False
                    :type: ``bool``

                save_values
                    see `save_values <OptimizationFunction.save_values>`

                    :default value: False
                    :type: ``bool``

                saved_samples
                    see `saved_samples <OptimizationFunction.saved_samples>`

                    :default value: []
                    :type: ``list``
                    :read only: True

                saved_values
                    see `saved_values <OptimizationFunction.saved_values>`

                    :default value: []
                    :type: ``list``
                    :read only: True

                search_function
                    see `search_function <OptimizationFunction.search_function>`

                    :default value: lambda x: x
                    :type: ``types.FunctionType``

                search_space
                    see `search_space <OptimizationFunction.search_space>`

                    :default value: [`SampleIterator`]
                    :type: ``list``

                search_termination_function
                    see `search_termination_function <OptimizationFunction.search_termination_function>`

                    :default value: lambda x, y, z: True
                    :type: ``types.FunctionType``
        """
        variable = Parameter(np.array([0, 0, 0]), read_only=True, pnl_internal=True, constructor_argument='default_variable')

        objective_function = Parameter(lambda x: 0, stateful=False, loggable=False)
        search_function = Parameter(lambda x: x, stateful=False, loggable=False)
        search_termination_function = Parameter(lambda x, y, z: True, stateful=False, loggable=False)
        search_space = Parameter([SampleIterator([0])], stateful=False, loggable=False)

        save_samples = Parameter(False, pnl_internal=True)
        save_values = Parameter(False, pnl_internal=True)

        # these are created as parameter ports, but should they be?
        max_iterations = Parameter(None, modulable=True)

        saved_samples = Parameter([], read_only=True, pnl_internal=True)
        saved_values = Parameter([], read_only=True, pnl_internal=True)

    @tc.typecheck
    def __init__(
        self,
        default_variable=None,
        objective_function:tc.optional(is_function_type)=None,
        search_function:tc.optional(is_function_type)=None,
        search_space=None,
        search_termination_function:tc.optional(is_function_type)=None,
        save_samples:tc.optional(bool)=None,
        save_values:tc.optional(bool)=None,
        max_iterations:tc.optional(int)=None,
        params=None,
        owner=None,
        prefs=None,
        context=None,
        **kwargs
    ):

        self._unspecified_args = []

        if objective_function is None:
            self._unspecified_args.append(OBJECTIVE_FUNCTION)

        if search_function is None:
            self._unspecified_args.append(SEARCH_FUNCTION)

        if search_termination_function is None:
            self._unspecified_args.append(SEARCH_TERMINATION_FUNCTION)

        super().__init__(
            default_variable=default_variable,
            save_samples=save_samples,
            save_values=save_values,
            max_iterations=max_iterations,
            search_space=search_space,
            objective_function=objective_function,
            search_function=search_function,
            search_termination_function=search_termination_function,
            params=params,
            owner=owner,
            prefs=prefs,
            context=context,
            **kwargs
        )

    def _validate_params(self, request_set, target_set=None, context=None):

        # super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        if OBJECTIVE_FUNCTION in request_set and request_set[OBJECTIVE_FUNCTION] is not None:
            if not is_function_type(request_set[OBJECTIVE_FUNCTION]):
                raise OptimizationFunctionError("Specification of {} arg for {} ({}) must be a function or method".
                                                format(repr(OBJECTIVE_FUNCTION), self.__class__.__name__,
                                                       request_set[OBJECTIVE_FUNCTION].__name__))

        if SEARCH_FUNCTION in request_set and request_set[SEARCH_FUNCTION] is not None:
            if not is_function_type(request_set[SEARCH_FUNCTION]):
                raise OptimizationFunctionError("Specification of {} arg for {} ({}) must be a function or method".
                                                format(repr(SEARCH_FUNCTION), self.__class__.__name__,
                                                       request_set[SEARCH_FUNCTION].__name__))

        if SEARCH_SPACE in request_set and request_set[SEARCH_SPACE] is not None:
            search_space = request_set[SEARCH_SPACE]
            if not all(isinstance(s, (SampleIterator, type(None))) for s in search_space):
                raise OptimizationFunctionError("All entries in list specified for {} arg of {} must be a {}".
                                                format(repr(SEARCH_SPACE),
                                                       self.__class__.__name__,
                                                       SampleIterator.__name__))

        if SEARCH_TERMINATION_FUNCTION in request_set and request_set[SEARCH_TERMINATION_FUNCTION] is not None:
            if not is_function_type(request_set[SEARCH_TERMINATION_FUNCTION]):
                raise OptimizationFunctionError("Specification of {} arg for {} ({}) must be a function or method".
                                                format(repr(SEARCH_TERMINATION_FUNCTION), self.__class__.__name__,
                                                       request_set[SEARCH_TERMINATION_FUNCTION].__name__))

            try:
                b = request_set[SEARCH_TERMINATION_FUNCTION]()
                if not isinstance(b, bool):
                    raise OptimizationFunctionError("Function ({}) specified for {} arg of {} must return a boolean value".
                                                    format(request_set[SEARCH_TERMINATION_FUNCTION].__name__,
                                                           repr(SEARCH_TERMINATION_FUNCTION),
                                                           self.__class__.__name__))
            except TypeError as e:
                # we cannot validate arbitrary functions here if they
                # require arguments
                if 'required positional arguments' not in str(e):
                    raise

    @handle_external_context(execution_id=NotImplemented)
    def reset(self, *args, context=None):
        """Reset parameters of the OptimizationFunction

        Parameters to be reset should be specified in a parameter specification dictionary, in which they key
        for each entry is the name of one of the following parameters, and its value is the value to be assigned to the
        parameter.  The following parameters can be reset:

            * `default_variable <OptimizationFunction.default_variable>`
            * `objective_function <OptimizationFunction.objective_function>`
            * `search_function <OptimizationFunction.search_function>`
            * `search_termination_function <OptimizationFunction.search_termination_function>`
        """
        if context.execution_id is NotImplemented:
            context.execution_id = self.most_recent_context.execution_id
        self._validate_params(request_set=args[0])

        if DEFAULT_VARIABLE in args[0]:
            self.defaults.variable = args[0][DEFAULT_VARIABLE]
        if OBJECTIVE_FUNCTION in args[0] and args[0][OBJECTIVE_FUNCTION] is not None:
            self.objective_function = args[0][OBJECTIVE_FUNCTION]
            if OBJECTIVE_FUNCTION in self._unspecified_args:
                del self._unspecified_args[self._unspecified_args.index(OBJECTIVE_FUNCTION)]
        if SEARCH_FUNCTION in args[0] and args[0][SEARCH_FUNCTION] is not None:
            self.search_function = args[0][SEARCH_FUNCTION]
            if SEARCH_FUNCTION in self._unspecified_args:
                del self._unspecified_args[self._unspecified_args.index(SEARCH_FUNCTION)]
        if SEARCH_TERMINATION_FUNCTION in args[0] and args[0][SEARCH_TERMINATION_FUNCTION] is not None:
            self.search_termination_function = args[0][SEARCH_TERMINATION_FUNCTION]
            if SEARCH_TERMINATION_FUNCTION in self._unspecified_args:
                del self._unspecified_args[self._unspecified_args.index(SEARCH_TERMINATION_FUNCTION)]
        if SEARCH_SPACE in args[0] and args[0][SEARCH_SPACE] is not None:
            self.parameters.search_space._set(args[0][SEARCH_SPACE], context)
            if SEARCH_SPACE in self._unspecified_args:
                del self._unspecified_args[self._unspecified_args.index(SEARCH_SPACE)]

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 **kwargs):
        """Find the sample that yields the optimal value of `objective_function
        <OptimizationFunction.objective_function>`.

        See `optimization process <OptimizationFunction_Procedure>` for details.

        Returns
        -------

        optimal sample, optimal value, saved_samples, saved_values : array, array, list, list
            first array contains sample that yields the optimal value of the `optimization process
            <OptimizationFunction_Procedure>`, and second array contains the value of `objective_function
            <OptimizationFunction.objective_function>` for that sample.  If `save_samples
            <OptimizationFunction.save_samples>` is `True`, first list contains all the values sampled in the order
            they were evaluated; otherwise it is empty.  If `save_values <OptimizationFunction.save_values>` is `True`,
            second list contains the values returned by `objective_function <OptimizationFunction.objective_function>`
            for all the samples in the order they were evaluated; otherwise it is empty.
        """

        if self._unspecified_args and self.initialization_status == ContextFlags.INITIALIZED:
            warnings.warn("The following arg(s) were not specified for {}: {} -- using default(s)".
                          format(self.name, ', '.join(self._unspecified_args)))
            self._unspecified_args = []

        current_sample = self._check_args(variable=variable, context=context, params=params)
        current_value = self.owner.objective_mechanism.parameters.value._get(context) if self.owner else 0.

        samples = []
        values = []

        # Initialize variables used in while loop
        iteration = 0

        # Set up progress bar
        _show_progress = False
        if hasattr(self, OWNER) and self.owner and self.owner.prefs.reportOutputPref:
            _show_progress = True
            _progress_bar_char = '.'
            _progress_bar_rate_str = ""
            _search_space_size = len(self.search_space)
            _progress_bar_rate = int(10**(np.log10(_search_space_size) - 2))
            if _progress_bar_rate > 1:
                _progress_bar_rate_str = str(_progress_bar_rate) + " "
            print("\n{} executing optimization process (one {} for each {}of {} samples): ".
                  format(self.owner.name, repr(_progress_bar_char), _progress_bar_rate_str, _search_space_size))
            _progress_bar_count = 0
        # Iterate optimization process
        while not call_with_pruned_args(self.search_termination_function,
                                        current_sample,
                                        current_value, iteration,
                                        context=context):

            if _show_progress:
                increment_progress_bar = (_progress_bar_rate < 1) or not (_progress_bar_count % _progress_bar_rate)
                if increment_progress_bar:
                    print(_progress_bar_char, end='', flush=True)
                _progress_bar_count +=1

            # Get next sample of sample
            new_sample = call_with_pruned_args(self.search_function, current_sample, iteration, context=context)
            # Compute new value based on new sample
            new_value = call_with_pruned_args(self.objective_function, new_sample, context=context)
            self._report_value(new_value)
            iteration += 1
            max_iterations = self.parameters.max_iterations._get(context)
            if max_iterations and iteration > max_iterations:
                warnings.warn("{} failed to converge after {} iterations".format(self.name, max_iterations))
                break

            current_sample = new_sample
            current_value = new_value

            if self.parameters.save_samples._get(context):
                samples.append(new_sample)
                self.parameters.saved_samples._set(samples, context)
            if self.parameters.save_values._get(context):
                values.append(current_value)
                self.parameters.saved_values._set(values, context)

        return new_sample, new_value, samples, values

    def _report_value(self, new_value):
        """Report value returned by `objective_function <OptimizationFunction.objective_function>` for sample."""
        pass


ASCENT = 'ascent'
DESCENT = 'descent'


class GradientOptimization(OptimizationFunction):
    """
    GradientOptimization(            \
        default_variable=None,       \
        objective_function=None,     \
        gradient_function=None,      \
        direction=ASCENT,            \
        search_space=None,           \
        step_size=1.0,               \
        annealing_function=None,     \
        convergence_criterion=VALUE, \
        convergence_threshold=.001,  \
        max_iterations=1000,         \
        save_samples=False,          \
        save_values=False,           \
        params=None,                 \
        owner=None,                  \
        prefs=None                   \
        )

    Sample variable by following gradient with respect to the value of `objective_function
    <GradientOptimization.objective_function>` it generates, and return the sample that generates either the
    highest (**direction=*ASCENT*) or lowest (**direction=*DESCENT*) value.

    .. _GradientOptimization_Procedure:

    **Optimization Procedure**

    When `function <GradientOptimization.function>` is executed, it iterates over the folowing steps:

        - `compute gradient <GradientOptimization_Gradient_Calculation>` using the `gradient_function
          <GradientOptimization.gradient_function>`;
        ..
        - adjust `variable <GradientOptimization.variable>` based on the gradient, in the specified
          `direction <GradientOptimization.direction>` and by an amount specified by `step_size
          <GradientOptimization.step_size>` and possibly `annealing_function
          <GradientOptimization.annealing_function>`;
        ..
        - compute value of `objective_function <GradientOptimization.objective_function>` using the adjusted value of
          `variable <GradientOptimization.variable>`;
        ..
        - adjust `step_size <GradientOptimization.udpate_rate>` using `annealing_function
          <GradientOptimization.annealing_function>`, if specified, for use in the next iteration;
        ..
        - evaluate `convergence_criterion <GradientOptimization.convergence_criterion>` and test whether it is below
          the `convergence_threshold <GradientOptimization.convergence_threshold>`.

    The current iteration is contained in `iteration <GradientOptimization.iteration>`. Iteration continues until
    `convergence_criterion <GradientOptimization.convergence_criterion>` falls below `convergence_threshold
    <GradientOptimization.convergence_threshold>` or the number of iterations exceeds `max_iterations
    <GradientOptimization.max_iterations>`.  The `function <GradientOptimization.function>` returns the last sample
    evaluated by `objective_function <GradientOptimization.objective_function>` (presumed to be the optimal one),
    the value of the function, as well as lists that may contain all of the samples evaluated and their values,
    depending on whether `save_samples <OptimizationFunction.save_samples>` and/or `save_vales
    <OptimizationFunction.save_values>` are `True`, respectively.

    .. _GradientOptimization_Gradient_Calculation:

    **Gradient Calculation**

    The gradient is evaluated by `gradient_function <GradientOptimization.gradient_function>`,
    which should be the derivative of the `objective_function <GradientOptimization.objective_function>`
    with respect to `variable <GradientOptimization.variable>` at its current value:
    :math:`\\frac{d(objective\\_function(variable))}{d(variable)}`.  If the **gradient_function* argument of the
    constructor is not specified, then an attempt is made to use `Autograd's <https://github.com/HIPS/autograd>`_ `grad
    <autograd.grad>` method to generate `gradient_function <GradientOptimization.gradient_function>`.  If that fails,
    an erorr occurs.  The **search_space** argument can be used to specify lower and/or upper bounds for each dimension
    of the sample; if the gradient causes a value of the sample to exceed a bound along a dimenson, the value of the
    bound is used for that dimension, unless/until the gradient shifts and causes it to return back within the bound.

    Arguments
    ---------

    default_variable : list or ndarray : default None
        specifies a template for (i.e., an example of the shape of) the samples used to evaluate the
        `objective_function <GradientOptimization.objective_function>`.

    objective_function : function or method
        specifies function used to evaluate `variable <GradientOptimization.variable>`
        in each iteration of the `optimization process  <GradientOptimization_Procedure>`;
        it must be specified and it must return a scalar value.

    gradient_function : function
        specifies function used to compute the gradient in each iteration of the `optimization process
        <GradientOptimization_Procedure>`;  if it is not specified, an attempt is made to compute it using
        `autograd.grad <https://github.com/HIPS/autograd>`_.

    direction : ASCENT or DESCENT : default ASCENT
        specifies the direction of gradient optimization: if *ASCENT*, movement is attempted in the positive direction
        (i.e., "up" the gradient);  if *DESCENT*, movement is attempted in the negative direction (i.e. "down"
        the gradient).

    step_size : int or float : default 1.0
        specifies the rate at which the `variable <GradientOptimization.variable>` is updated in each
        iteration of the `optimization process <GradientOptimization_Procedure>`;  if `annealing_function
        <GradientOptimization.annealing_function>` is specified, **step_size** specifies the intial value of
        `step_size <GradientOptimization.step_size>`.

    search_space : list or array : default None
        specifies bounds of the samples used to evaluate `objective_function <GaussianProcess.objective_function>`
        along each dimension of `variable <GaussianProcess.variable>`;  each item must be a list or tuple,
        or a `SampleIterator` that resolves to one.  If the item has two elements, they are used as the lower and
        upper bounds respectively, and the lower must be less than the upper;  None can be used in either place,
        in which case that bound is ignored.  If an item has more than two elements, the min is used as the lower
        bound and the max is used as the upper bound; none of the elements can be None.

    annealing_function : function or method : default None
        specifies function used to adapt `step_size <GradientOptimization.step_size>` in each
        iteration of the `optimization process <GradientOptimization_Procedure>`;  must take accept two parameters —
        `step_size <GradientOptimization.step_size>` and `iteration <GradientOptimization_Procedure>`, in that
        order — and return a scalar value, that is used for the next iteration of optimization.

    convergence_criterion : *VARIABLE* or *VALUE* : default *VALUE*
        specifies the parameter used to terminate the `optimization process <GradientOptimization_Procedure>`.
        *VARIABLE*: process terminates when the most recent sample differs from the previous one by less than
        `convergence_threshold <GradientOptimization.convergence_threshold>`;  *VALUE*: process terminates when the
        last value returned by `objective_function <GradientOptimization.objective_function>` differs from the
        previous one by less than `convergence_threshold <GradientOptimization.convergence_threshold>`.

    convergence_threshold : int or float : default 0.001
        specifies the change in value of `convergence_criterion` below which the optimization process is terminated.

    max_iterations : int : default 1000
        specifies the maximum number of times the `optimization process<GradientOptimization_Procedure>` is allowed to
        iterate; if exceeded, a warning is issued and the function returns the last sample evaluated.

    save_samples : bool
        specifies whether or not to save and return all of the samples used to evaluate `objective_function
        <GradientOptimization.objective_function>` in the `optimization process<GradientOptimization_Procedure>`.

    save_values : bool
        specifies whether or not to save and return the values of `objective_function
        <GradientOptimization.objective_function>` for all samples evaluated in the `optimization
        process<GradientOptimization_Procedure>`

    Attributes
    ----------

    variable : ndarray
        sample used as the starting point for the `optimization process <GradientOptimization_Procedure>` (i.e., one
        used to evaluate `objective_function <GradientOptimization.objective_function>` in the first iteration).

    objective_function : function or method
        function used to evaluate `variable <GradientOptimization.variable>`
        in each iteration of the `optimization process <GradientOptimization_Procedure>`;
        it must be specified and it must return a scalar value.

    gradient_function : function
        function used to compute the gradient in each iteration of the `optimization process
        <GradientOptimization_Procedure>` (see `Gradient Calculation <GradientOptimization_Gradient_Calculation>` for
        details).

    direction : ASCENT or DESCENT
        direction of gradient optimization:  if *ASCENT*, movement is attempted in the positive direction
        (i.e., "up" the gradient);  if *DESCENT*, movement is attempted in the negative direction (i.e. "down"
        the gradient).

    step_size : int or float
        determines the rate at which the `variable <GradientOptimization.variable>` is updated in each
        iteration of the `optimization process <GradientOptimization_Procedure>`;  if `annealing_function
        <GradientOptimization.annealing_function>` is specified, `step_size <GradientOptimization.step_size>`
        determines the initial value.

    search_space : list or array
        contains tuples specifying bounds within which each dimension of `variable <GaussianProcess.variable>` is
        sampled, and used to evaluate `objective_function <GaussianProcess.objective_function>` in iterations of the
        `optimization process <GaussianProcess_Procedure>`.

    bounds : tuple
        contains two 2d arrays; the 1st contains the lower bounds for each dimension of the sample (`variable
        <GradientOptimization.variable>`), and the 2nd the upper bound of each.

    annealing_function : function or method
        function used to adapt `step_size <GradientOptimization.step_size>` in each iteration of the `optimization
        process <GradientOptimization_Procedure>`;  if `None`, no call is made and the same `step_size
        <GradientOptimization.step_size>` is used in each iteration.

    iteration : int
        the currention iteration of the `optimization process <GradientOptimization_Procedure>`.

    convergence_criterion : VARIABLE or VALUE
        determines parameter used to terminate the `optimization process<GradientOptimization_Procedure>`.
        *VARIABLE*: process terminates when the most recent sample differs from the previous one by less than
        `convergence_threshold <GradientOptimization.convergence_threshold>`;  *VALUE*: process terminates when the
        last value returned by `objective_function <GradientOptimization.objective_function>` differs from the
        previous one by less than `convergence_threshold <GradientOptimization.convergence_threshold>`.

    convergence_threshold : int or float
        determines the change in value of `convergence_criterion` below which the `optimization process
        <GradientOptimization_Procedure>` is terminated.

    max_iterations : int
        determines the maximum number of times the `optimization process<GradientOptimization_Procedure>` is allowed to
        iterate; if exceeded, a warning is issued and the function returns the last sample evaluated.

    save_samples : bool
        determines whether or not to save and return all of the samples used to evaluate `objective_function
        <GradientOptimization.objective_function>` in the `optimization process<GradientOptimization_Procedure>`.

    save_values : bool
        determines whether or not to save and return the values of `objective_function
        <GradientOptimization.objective_function>` for all samples evaluated in the `optimization
        process<GradientOptimization_Procedure>`
    """

    componentName = GRADIENT_OPTIMIZATION_FUNCTION

    class Parameters(OptimizationFunction.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <GradientOptimization.variable>`

                    :default value: [[0], [0]]
                    :type: ``list``
                    :read only: True

                annealing_function
                    see `annealing_function <GradientOptimization.annealing_function>`

                    :default value: None
                    :type:

                convergence_criterion
                    see `convergence_criterion <GradientOptimization.convergence_criterion>`

                    :default value: `VALUE`
                    :type: ``str``

                convergence_threshold
                    see `convergence_threshold <GradientOptimization.convergence_threshold>`

                    :default value: 0.001
                    :type: ``float``

                direction
                    see `direction <GradientOptimization.direction>`

                    :default value: `ASCENT`
                    :type: ``str``

                gradient_function
                    see `gradient_function <GradientOptimization.gradient_function>`

                    :default value: None
                    :type:

                max_iterations
                    see `max_iterations <GradientOptimization.max_iterations>`

                    :default value: 1000
                    :type: ``int``

                previous_value
                    see `previous_value <GradientOptimization.previous_value>`

                    :default value: [[0], [0]]
                    :type: ``list``
                    :read only: True

                previous_variable
                    see `previous_variable <GradientOptimization.previous_variable>`

                    :default value: [[0], [0]]
                    :type: ``list``
                    :read only: True

                step_size
                    see `step_size <GradientOptimization.step_size>`

                    :default value: 1.0
                    :type: ``float``
        """
        variable = Parameter([[0], [0]], read_only=True, pnl_internal=True, constructor_argument='default_variable')

        # these should be removed and use switched to .get_previous()
        previous_variable = Parameter([[0], [0]], read_only=True, pnl_internal=True, constructor_argument='default_variable')
        previous_value = Parameter([[0], [0]], read_only=True, pnl_internal=True)

        gradient_function = Parameter(None, stateful=False, loggable=False)
        step_size = Parameter(1.0, modulable=True)
        annealing_function = Parameter(None, stateful=False, loggable=False)
        convergence_threshold = Parameter(.001, modulable=True)
        max_iterations = Parameter(1000, modulable=True)
        search_space = Parameter([SampleIterator([0, 0])], stateful=False, loggable=False)

        direction = ASCENT
        convergence_criterion = Parameter(VALUE, pnl_internal=True)

        def _parse_direction(self, direction):
            if direction == ASCENT:
                return 1
            else:
                return -1

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 objective_function:tc.optional(is_function_type)=None,
                 gradient_function:tc.optional(is_function_type)=None,
                 direction:tc.optional(tc.enum(ASCENT, DESCENT))=None,
                 search_space=None,
                 step_size:tc.optional(tc.any(int, float))=None,
                 annealing_function:tc.optional(is_function_type)=None,
                 convergence_criterion:tc.optional(tc.enum(VARIABLE, VALUE))=None,
                 convergence_threshold:tc.optional(tc.any(int, float))=None,
                 max_iterations:tc.optional(int)=None,
                 save_samples:tc.optional(bool)=None,
                 save_values:tc.optional(bool)=None,
                 params=None,
                 owner=None,
                 prefs=None):

        search_function = self._follow_gradient
        search_termination_function = self._convergence_condition

        super().__init__(
            default_variable=default_variable,
            objective_function=objective_function,
            search_function=search_function,
            search_space=search_space,
            search_termination_function=search_termination_function,
            max_iterations=max_iterations,
            save_samples=save_samples,
            save_values=save_values,
            step_size=step_size,
            convergence_criterion=convergence_criterion,
            convergence_threshold=convergence_threshold,
            gradient_function=gradient_function,
            annealing_function=annealing_function,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _validate_params(self, request_set, target_set=None, context=None):

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        if SEARCH_SPACE in request_set and request_set[SEARCH_SPACE] is not None:
            search_space = request_set[SEARCH_SPACE]
            if all(s is None for s in search_space):
                return
            # If search space is a single 2-item list or tuple with numbers (i.e., bounds),
            #     wrap in list for handling below
            if len(search_space)==2 and all(isinstance(i, Number) for i in search_space):
                search_space = [search_space]
            for s in search_space:
                if isinstance(s, SampleIterator):
                    s = s()
                if len(s) != 2:
                    owner_str = ''
                    if self.owner:
                        owner_str = f' of {self.owner.name}'
                    raise OptimizationFunctionError(f"All items in {repr(SEARCH_SPACE)} arg for {self.name}{owner_str} "
                                                    f"must be or resolve to a 2-item list or tuple; this doesn't: {s}.")

    @handle_external_context(execution_id=NotImplemented)
    def reset(self, *args, context=None):
        super().reset(*args)

        # Differentiate objective_function using autograd.grad()
        if OBJECTIVE_FUNCTION in args[0]:
            try:
                from autograd import grad
                self.gradient_function = grad(self.objective_function)
            except:
                raise OptimizationFunctionError("Unable to use autograd with {} specified for {} Function: {}.".
                                                format(repr(OBJECTIVE_FUNCTION), self.__class__.__name__,
                                                       args[0][OBJECTIVE_FUNCTION].__name__))
        search_space = self.search_space
        bounds = None

        if self.owner:
            owner_str = ' of {self.owner.name}'

        # Get bounds from search_space if it has any non-None entries
        if any(i is not None for i in self.search_space):
            # Get min and max of each dimension of search space
            #    and assign to corresponding elements of lower and upper items of bounds
            lower = []
            upper = []
            bounds = (lower, upper)
            for i in search_space:
                if i is None:
                    lower.append(None)
                    upper.append(None)
                else:
                    if isinstance(i, SampleIterator):
                        i = i()
                    # Spec is bound (tuple or list with two values: lower and uppper)
                    if len(i)==2:
                        lower.append(i[0])
                        upper.append(i[1])
                    else:
                        lower.append(min(i))
                        upper.append(max(i))

        # Validate bounds and reformat into arrays for lower and upper bounds, for use in _follow_gradient
        #     (each should be same length as sample), and replace any None's with + or - inf)
        if bounds:
            if bounds[0] is None and bounds[1] is None:
                bounds = None
            else:
                sample_len = len(args[0][DEFAULT_VARIABLE])
                lower = np.atleast_1d(bounds[0])
                if len(lower)==1:
                    # Single value specified for lower bound, so distribute over array with length = sample_len
                    lower = np.full(sample_len, lower).reshape(sample_len,1)
                elif len(lower)!=sample_len:
                    raise OptimizationFunctionError(f"Array used for lower value of {repr(BOUNDS)} arg ({lower}) in "
                                                    f"{self.name}{owner_str} must have the same number of elements "
                                                    f"({sample_len}) as the sample over which optimization is being "
                                                    f"performed.")
                # Array specified for lower bound, so replace any None's with -inf
                lower = np.array([[-float('inf')] if n[0] is None else n for n in lower.reshape(sample_len,1)])

                upper = np.atleast_1d(bounds[1])
                if len(upper)==1:
                    # Single value specified for upper bound, so distribute over array with length = sample_len
                    upper = np.full(sample_len, upper).reshape(sample_len,1)
                elif len(upper)!=sample_len:
                    raise OptimizationFunctionError(f"Array used for upper value of {repr(BOUNDS)} arg ({upper}) in "
                                                    f"{self.name}{owner_str} must have the same number of elements "
                                                    f"({sample_len}) as the sample over which optimization is being "
                                                    f"performed.")
                # Array specified for upper bound, so replace any None's with +inf
                upper = np.array([[float('inf')] if n[0] is None else n for n in upper.reshape(sample_len,1)])

                if not all(lower<upper):
                    raise OptimizationFunctionError(f"Specification of {repr(BOUNDS)} arg ({bounds}) for {self.name}"
                                                    f"{owner_str} resulted in lower >= corresponding upper for one or "
                                                    f"more elements (lower: {lower.tolist()}; uuper: {upper.tolist()}).")

                bounds = (lower,upper)

        self.bounds = bounds

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 **kwargs):
        """Return the sample that yields the optimal value of `objective_function
        <GradientOptimization.objective_function>`, and possibly all samples evaluated and their corresponding values.

        Optimal value is defined by `direction <GradientOptimization.direction>`:
        - if *ASCENT*, returns greatest value
        - if *DESCENT*, returns least value

        Returns
        -------

        optimal sample, optimal value, saved_samples, saved_values : ndarray, list, list
            first array contains sample that yields the highest or lowest value of `objective_function
            <GradientOptimization.objective_function>`, depending on `direction <GradientOptimization.direction>`,
            and the second array contains the value of the function for that sample.
            If `save_samples <GradientOptimization.save_samples>` is `True`, first list contains all the values
            sampled in the order they were evaluated; otherwise it is empty.  If `save_values
            <GradientOptimization.save_values>` is `True`, second list contains the values returned by
            `objective_function <GradientOptimization.objective_function>` for all the samples in the order they were
            evaluated; otherwise it is empty.
        """

        optimal_sample, optimal_value, all_samples, all_values = super()._function(variable=variable,
                                                                                  context=context,
                                                                                  params=params,
                                                                                  )
        return_all_samples = return_all_values = []
        if self.parameters.save_samples._get(context):
            return_all_samples = all_samples
        if self.parameters.save_values._get(context):
            return_all_values = all_values
        # return last_variable
        return optimal_sample, optimal_value, return_all_samples, return_all_values

    def _follow_gradient(self, sample, sample_num, context=None):

        if self.gradient_function is None:
            return sample

        # Index from 1 rather than 0
        # Update step_size
        step_size = self.parameters.step_size._get(context)
        if sample_num == 0:
            # Start from initial value (sepcified by user in step_size arg)
            step_size = self.parameters.step_size.default_value
            self.parameters.step_size._set(step_size, context)
        if self.annealing_function:
            step_size = call_with_pruned_args(self.annealing_function, step_size, sample_num, context=context)
            self.parameters.step_size._set(step_size, context)

        # Compute gradients with respect to current sample
        _gradients = call_with_pruned_args(self.gradient_function, sample, context=context)

        # Get new sample based on new gradients
        new_sample = sample + self.parameters.direction._get(context) * step_size * np.array(_gradients)

        # Constrain new sample to be within bounds
        if self.bounds:
            new_sample = np.array(np.maximum(self.bounds[0],
                                             np.minimum(self.bounds[1], new_sample))).reshape(sample.shape)

        return new_sample

    def _convergence_condition(self, variable, value, iteration, context=None):
        previous_variable = self.parameters.previous_variable._get(context)
        previous_value = self.parameters.previous_value._get(context)

        if iteration == 0:
            # self._convergence_metric = self.convergence_threshold + EPSILON
            self.parameters.previous_variable._set(variable, context)
            self.parameters.previous_value._set(value, context)
            return False

        # Evaluate for convergence
        if self.convergence_criterion == VALUE:
            convergence_metric = np.abs(value - previous_value)
        else:
            convergence_metric = np.max(np.abs(np.array(variable) -
                                               np.array(previous_variable)))

        self.parameters.previous_variable._set(variable, context)
        self.parameters.previous_value._set(value, context)

        return convergence_metric <= self.parameters.convergence_threshold._get(context)


MAXIMIZE = 'maximize'
MINIMIZE = 'minimize'


class GridSearch(OptimizationFunction):
    """
    GridSearch(                      \
        default_variable=None,       \
        objective_function=None,     \
        direction=MAXIMIZE,          \
        max_iterations=1000,         \
        save_samples=False,          \
        save_values=False,           \
        params=None,                 \
        owner=None,                  \
        prefs=None                   \
        )

    Search over all samples generated by `search_space <GridSearch.search_space>` for the one that optimizes the
    value of `objective_function <GridSearch.objective_function>`.

    .. _GridSearch_Procedure:

    **Grid Search Procedure**

    When `function <GridSearch.function>` is executed, it iterates over the folowing steps:

        - get next sample from `search_space <GridSearch.search_space>`;
        ..
        - compute value of `objective_function <GridSearch.objective_function>` for that sample;

    The current iteration is contained in `iteration <GridSearch.iteration>` and the total number comprising the
    `search_space <GridSearch.search_space>2` is contained in `num_iterations <GridSearch.num_iterations>`).
    Iteration continues until all values in `search_space <GridSearch.search_space>` have been evaluated (i.e.,
    `num_iterations <GridSearch.num_iterations>` is reached), or `max_iterations <GridSearch.max_iterations>` is
    execeeded.  The function returns the sample that yielded either the highest (if `direction <GridSearch.direction>`
    is *MAXIMIZE*) or lowest (if `direction <GridSearch.direction>` is *MINIMIZE*) value of the `objective_function
    <GridSearch.objective_function>`, along with the value for that sample, as well as lists containing all of the
    samples evaluated and their values if either `save_samples <GridSearch.save_samples>` or `save_values
    <GridSearch.save_values>` is `True`, respectively.

    Arguments
    ---------

    default_variable : list or ndarray : default None
        specifies a template for (i.e., an example of the shape of) the samples used to evaluate the
        `objective_function <GridSearch.objective_function>`.

    objective_function : function or method
        specifies function used to evaluate sample in each iteration of the `optimization process <GridSearch_Procedure>`;
        it must be specified and must return a scalar value.

    search_space : list or array of SampleIterators
        specifies `SampleIterators <SampleIterator>` used to generate samples evaluated by `objective_function
        <GridSearch.objective_function>`;  all of the iterators be finite (i.e., must have a `num <SampleIterator>`
        attribute;  see `SampleSpec` for additional details).

    direction : MAXIMIZE or MINIMIZE : default MAXIMIZE
        specifies the direction of optimization:  if *MAXIMIZE*, the highest value of `objective_function
        <GridSearch.objective_function>` is sought;  if *MINIMIZE*, the lowest value is sought.

    max_iterations : int : default 1000
        specifies the maximum number of times the `optimization process<GridSearch_Procedure>` is allowed to iterate;
        if exceeded, a warning is issued and the function returns the optimal sample of those evaluated.

    save_samples : bool
        specifies whether or not to return all of the samples used to evaluate `objective_function
        <GridSearch.objective_function>` in the `optimization process <GridSearch_Procedure>`
        (i.e., a copy of the samples generated from the `search_space <GridSearch.search_space>`.

    save_values : bool
        specifies whether or not to save and return the values of `objective_function <GridSearch.objective_function>`
        for all samples evaluated in the `optimization process <GridSearch_Procedure>`.

    Attributes
    ----------

    variable : ndarray
        first sample evaluated by `objective_function <GridSearch.objective_function>` (i.e., one used to evaluate it
        in the first iteration of the `optimization process <GridSearch_Procedure>`).

    objective_function : function or method
        function used to evaluate sample in each iteration of the `optimization process <GridSearch_Procedure>`.

    search_space : list or array of Sampleiterators
        contains `SampleIterators <SampleIterator>` for generating samples evaluated by `objective_function
        <GridSearch.objective_function>` in iterations of the `optimization process <GridSearch_Procedure>`;

    grid : iterator
        generates samples from the Cartesian product of `SampleIterators in `search_space <GridSearch.search_sapce>`.

    direction : MAXIMIZE or MINIMIZE : default MAXIMIZE
        determines the direction of optimization:  if *MAXIMIZE*, the greatest value of `objective_function
        <GridSearch.objective_function>` is sought;  if *MINIMIZE*, the least value is sought.

    iteration : int
        the currention iteration of the `optimization process <GridSearch_Procedure>`.

    num_iterations : int
        number of iterations required to complete the entire grid search;  equal to the produce of all the `num
        <SampleIterator.num>` attributes of the `SampleIterators <SampleIterator>` in the `search_space
        <GridSearch.search_space>`.

    max_iterations : int
        determines the maximum number of times the `optimization process<GridSearch_Procedure>` is allowed to iterate;
        if exceeded, a warning is issued and the function returns the optimal sample of those evaluated.

    save_samples : True
        determines whether or not to save and return all samples generated from `search_space <GridSearch.search_space>`
        and evaluated by the  `objective_function <GridSearch.objective_function>` in the `optimization process
        <GridSearch_Procedure>`.

    save_values : bool
        determines whether or not to save and return the value of `objective_function
        <GridSearch.objective_function>` for all samples evaluated in the `optimization process <GridSearch_Procedure>`.
    """

    componentName = GRID_SEARCH_FUNCTION

    class Parameters(OptimizationFunction.Parameters):
        """
            Attributes
            ----------

                direction
                    see `direction <GridSearch.direction>`

                    :default value: `MAXIMIZE`
                    :type: ``str``

                grid
                    see `grid <GridSearch.grid>`

                    :default value: None
                    :type:

                random_state
                    see `random_state <GridSearch.random_state>`

                    :default value: None
                    :type: ``numpy.random.RandomState``

                save_samples
                    see `save_samples <GridSearch.save_samples>`

                    :default value: True
                    :type: ``bool``

                save_values
                    see `save_values <GridSearch.save_values>`

                    :default value: True
                    :type: ``bool``
        """
        grid = Parameter(None)
        save_samples = Parameter(True, pnl_internal=True)
        save_values = Parameter(True, pnl_internal=True)
        random_state = Parameter(None, stateful=True, loggable=False)
        select_randomly_from_optimal_values = Parameter(False)

        direction = MAXIMIZE

    # TODO: should save_values be in the constructor if it's ignored?
    # is False or True the correct value?
    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 objective_function:tc.optional(is_function_type)=None,
                 search_space=None,
                 direction:tc.optional(tc.enum(MAXIMIZE, MINIMIZE))=None,
                 save_values:tc.optional(bool)=None,
                 # tolerance=0.,
                 select_randomly_from_optimal_values=None,
                 seed=None,
                 params=None,
                 owner=None,
                 prefs=None,
                 **kwargs):

        search_function = self._traverse_grid
        search_termination_function = self._grid_complete
        self._return_values = save_values
        self._return_samples = save_values
        try:
            search_space = [x if isinstance(x, SampleIterator) else SampleIterator(x) for x in search_space]
        except TypeError:
            pass

        self.num_iterations = 1 if search_space is None else np.product([i.num for i in search_space])
        # self.tolerance = tolerance

        if seed is None:
            seed = get_global_seed()
        random_state = np.random.RandomState([seed])

        super().__init__(
            default_variable=default_variable,
            objective_function=objective_function,
            search_function=search_function,
            search_termination_function=search_termination_function,
            search_space=search_space,
            select_randomly_from_optimal_values=select_randomly_from_optimal_values,
            save_samples=True,
            save_values=save_values,
            random_state=random_state,
            direction=direction,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _validate_params(self, request_set, target_set=None, context=None):

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)
        if SEARCH_SPACE in request_set and request_set[SEARCH_SPACE] is not None:
            search_space = request_set[SEARCH_SPACE]

            # Check that all iterators are finite (i.e., with num!=None)
            if not all(s.num is not None for s in search_space if (s is not None and s.num)):
                raise OptimizationFunctionError("All {}s in {} arg of {} must be finite (i.e., SampleIteror.num!=None)".
                                                format(SampleIterator.__name__,
                                                       repr(SEARCH_SPACE),
                                                       self.__class__.__name__))

            # # Check that all finite iterators (i.e., with num!=None) are of the same length:
            # finite_iterators = [s.num for s in search_space if s.num is not None]
            # if not all(l==finite_iterators[0] for l in finite_iterators):
            #     raise OptimizationFunctionError("All finite {}s in {} arg of {} must have the same number of steps".
            #                                     format(SampleIterator.__name__,
            #                                            repr(SEARCH_SPACE),
            #                                            self.__class__.__name__,
            #                                            ))

    @handle_external_context(execution_id=NotImplemented)
    def reset(self, *args, context=None):
        """Assign size of `search_space <GridSearch.search_space>"""
        if context.execution_id is NotImplemented:
            context.execution_id = self.most_recent_context.execution_id
        super(GridSearch, self).reset(*args, context=context)
        sample_iterators = args[0]['search_space']
        owner_str = ''
        if self.owner:
            owner_str = f' of {self.owner.name}'
        for i in sample_iterators:
            if i is None:
                raise OptimizationFunctionError(f"Invalid {repr(SEARCH_SPACE)} arg for {self.name}{owner_str}; "
                                                f"every dimension must be assigned a {SampleIterator.__name__}.")
            if i.num is None:
                raise OptimizationFunctionError(f"Invalid {repr(SEARCH_SPACE)} arg for {self.name}{owner_str}; each "
                                                f"{SampleIterator.__name__} must have a value for its 'num' attribute.")

        self.num_iterations = np.product([i.num for i in sample_iterators])

    def reset_grid(self):
        """Reset iterators in `search_space <GridSearch.search_space>"""
        for s in self.search_space:
            s.reset()
        self.grid = itertools.product(*[s for s in self.search_space])

    def _get_optimized_composition(self):
        # self.objective_function may be a bound method of
        # OptimizationControlMechanism
        return getattr(self.objective_function, '__self__', None)

    def _gen_llvm_function(self, *, ctx:pnlvm.LLVMBuilderContext, tags:frozenset):
        if "select_min" in tags:
            return self._gen_llvm_select_min_function(ctx=ctx, tags=tags)
        ocm = self._get_optimized_composition()
        if ocm is not None:
            # self.objective_function may be a bound method of
            # OptimizationControlMechanism
            extra_args = [ctx.get_param_struct_type(ocm.agent_rep).as_pointer(),
                          ctx.get_state_struct_type(ocm.agent_rep).as_pointer(),
                          ctx.get_data_struct_type(ocm.agent_rep).as_pointer()]
        else:
            extra_args = []

        f = super()._gen_llvm_function(ctx=ctx, extra_args=extra_args, tags=tags)
        if len(extra_args) > 0:
            for a in f.args[-len(extra_args):]:
                a.attributes.add('nonnull')

        return f

    def _get_input_struct_type(self, ctx):
        if self.owner is not None:
            variable = [port.defaults.value for port in self.owner.input_ports]
            # Python list does not care about ndarrays of different lengths
            # we do care, so convert to tuple to create struct
            if all(type(x) == np.ndarray for x in variable) and not all(len(x) == len(variable[0]) for x in variable):
                variable = tuple(variable)

            warnings.warn("Shape mismatch: {} variable expected: {} vs. got: {}".format(self, variable, self.defaults.variable))

        else:
            variable = self.defaults.variable

        return ctx.convert_python_struct_to_llvm_ir(variable)

    def _get_output_struct_type(self, ctx):
        val = self.defaults.value
        # compiled version should never return 'all values'
        if len(val[0]) != len(self.search_space):
            val = list(val)
            val[0] = [0.0] * len(self.search_space)
        return ctx.convert_python_struct_to_llvm_ir((val[0], val[1]))

    def _gen_llvm_select_min_function(self, *, ctx:pnlvm.LLVMBuilderContext, tags:frozenset):
        assert "select_min" in tags
        ocm = self._get_optimized_composition()
        if ocm is not None:
            assert ocm.function is self
            sample_t = ocm._get_evaluate_alloc_struct_type(ctx)
            value_t = ocm._get_evaluate_output_struct_type(ctx)
        else:
            obj_func = ctx.import_llvm_function(self.objective_function)
            sample_t = obj_func.args[2].type.pointee
            value_t = obj_func.args[3].type.pointee

        args = [ctx.get_param_struct_type(self).as_pointer(),
                ctx.get_state_struct_type(self).as_pointer(),
                sample_t.as_pointer(),
                sample_t.as_pointer(),
                value_t.as_pointer(),
                value_t.as_pointer(),
                ctx.float_ty.as_pointer(),
                ctx.int32_ty]
        builder = ctx.create_llvm_function(args, self, tags=tags)

        params, state, min_sample_ptr, samples_ptr, min_value_ptr, values_ptr, opt_count_ptr, count = builder.function.args
        for p in builder.function.args[:-1]:
            p.attributes.add('noalias')
            p.attributes.add('nonnull')

        random_state = pnlvm.helpers.get_state_ptr(builder, self, state,
                                                   self.parameters.random_state.name)
        select_random_ptr = pnlvm.helpers.get_param_ptr(builder, self, params,
                                                        self.parameters.select_randomly_from_optimal_values.name)

        select_random_val = builder.load(select_random_ptr)
        select_random = builder.fcmp_ordered("!=", select_random_val,
                                             select_random_val.type(0))

        rand_out_ptr = builder.alloca(ctx.float_ty)

        # KDM 8/22/19: nonstateful direction here - OK?
        direction = "<" if self.direction == MINIMIZE else ">"
        replace_ptr = builder.alloca(pnlvm.ir.IntType(1))

        # Check the value against current min
        with pnlvm.helpers.for_loop_zero_inc(builder, count, "compare_loop") as (b, idx):
            value_ptr = b.gep(values_ptr, [idx])
            sample_ptr = b.gep(samples_ptr, [idx])
            value = b.load(value_ptr)
            min_value = b.load(min_value_ptr)

            replace = b.fcmp_unordered(direction, value, min_value)
            b.store(replace, replace_ptr)

            # Python does "is_close" check first.
            # This implements reservoir sampling
            with b.if_then(select_random):
                close = pnlvm.helpers.is_close(b, value, min_value)
                with b.if_else(close) as (tb, eb):
                    with tb:
                        opt_count = b.load(opt_count_ptr)
                        opt_count = b.fadd(opt_count, opt_count.type(1))
                        b.store(opt_count, opt_count_ptr)

                        # Roll a dice to see if we should replace the current min
                        prob = b.fdiv(opt_count.type(1), opt_count)
                        rand_f = ctx.import_llvm_function("__pnl_builtin_mt_rand_double")
                        b.call(rand_f, [random_state, rand_out_ptr])
                        rand_out = b.load(rand_out_ptr)
                        replace = b.fcmp_ordered("<", rand_out, prob)
                        b.store(replace, replace_ptr)
                    with eb:
                        # Reset the counter if we are replacing with new best value
                        with b.if_then(b.load(replace_ptr)):
                            b.store(opt_count_ptr.type.pointee(1), opt_count_ptr)

            with b.if_then(b.load(replace_ptr)):
                b.store(b.load(value_ptr), min_value_ptr)
                b.store(b.load(sample_ptr), min_sample_ptr)

        builder.ret_void()
        return builder.function

    def _gen_llvm_function_body(self, ctx, builder, params, state, arg_in, arg_out, *, tags:frozenset):
        ocm = self._get_optimized_composition()
        if ocm is not None:
            assert ocm.function is self
            obj_func = ctx.import_llvm_function(ocm, tags=tags.union({"evaluate"}))
            comp_args = builder.function.args[-3:]
            obj_param_ptr = comp_args[0]
            obj_state_ptr = comp_args[1]
            extra_args = [arg_in, comp_args[2]]
        else:
            obj_func = ctx.import_llvm_function(self.objective_function)
            obj_state_ptr = pnlvm.helpers.get_state_ptr(builder, self, state,
                                                        "objective_function")
            obj_param_ptr = pnlvm.helpers.get_param_ptr(builder, self, params,
                                                        "objective_function")
            extra_args = []

        sample_t = obj_func.args[2].type.pointee
        value_t = obj_func.args[3].type.pointee
        min_sample_ptr = builder.alloca(sample_t)
        min_value_ptr = builder.alloca(value_t)
        sample_ptr = builder.alloca(sample_t)
        value_ptr = builder.alloca(value_t)

        search_space_ptr = pnlvm.helpers.get_param_ptr(builder, self, params,
                                                       self.parameters.search_space.name)

        opt_count_ptr = builder.alloca(ctx.float_ty)
        builder.store(opt_count_ptr.type.pointee(0), opt_count_ptr)

        # Use NaN here. fcmp_unordered below returns true if one of the
        # operands is a NaN. This makes sure we always set min_*
        # in the first iteration
        builder.store(min_value_ptr.type.pointee("NaN"), min_value_ptr)

        b = builder
        with contextlib.ExitStack() as stack:
            for i in range(len(search_space_ptr.type.pointee)):
                dimension = b.gep(search_space_ptr, [ctx.int32_ty(0), ctx.int32_ty(i)])
                arg_elem = b.gep(sample_ptr, [ctx.int32_ty(0), ctx.int32_ty(i)])
                if isinstance(dimension.type.pointee,  pnlvm.ir.ArrayType):
                    b, idx = stack.enter_context(pnlvm.helpers.array_ptr_loop(b, dimension, "loop_" + str(i)))
                    alloc_elem = b.gep(dimension, [ctx.int32_ty(0), idx])
                    b.store(b.load(alloc_elem), arg_elem)
                elif isinstance(dimension.type.pointee, pnlvm.ir.LiteralStructType):
                    assert len(dimension.type.pointee) == 3
                    start_ptr = b.gep(dimension, [ctx.int32_ty(0), ctx.int32_ty(0)])
                    step_ptr = b.gep(dimension, [ctx.int32_ty(0), ctx.int32_ty(1)])
                    num_ptr = b.gep(dimension, [ctx.int32_ty(0), ctx.int32_ty(2)])
                    start = b.load(start_ptr)
                    step = b.load(step_ptr)
                    num = b.load(num_ptr)
                    b, idx = stack.enter_context(pnlvm.helpers.for_loop_zero_inc(b, num, "loop_" + str(i)))
                    val = b.uitofp(idx, start.type)
                    val = b.fmul(val, step)
                    val = b.fadd(val, start)
                    b.store(val, arg_elem)
                else:
                    assert False, "Unknown dimension type: {}".format(dimension.type)

            # We are in the inner most loop now with sample_ptr setup for execution
            b.call(obj_func, [obj_param_ptr, obj_state_ptr, sample_ptr,
                              value_ptr] + extra_args)

            # Check if smaller than current best.
            select_min_f = ctx.import_llvm_function(self, tags=tags.union({"select_min"}))
            b.call(select_min_f, [params, state, min_sample_ptr, sample_ptr,
                                  min_value_ptr, value_ptr, opt_count_ptr,
                                  ctx.int32_ty(1)])

            builder = b

        # Produce output
        out_sample_ptr = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(0)])
        out_value_ptr = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(1)])
        builder.store(builder.load(min_sample_ptr), out_sample_ptr)
        builder.store(builder.load(min_value_ptr), out_value_ptr)
        return builder

    def _run_cuda_grid(self, ocm, variable, context):
        assert ocm is ocm.agent_rep.controller
        # Compiled evaluate expects the same variable as mech function
        new_variable = [np.asfarray(ip.parameters.value.get(context))
                        for ip in ocm.input_ports]
        new_variable = np.array(new_variable, dtype=np.object)
        # Map allocations to values
        comp_exec = pnlvm.execution.CompExecution(ocm.agent_rep, [context.execution_id])
        ct_alloc, ct_values = comp_exec.cuda_evaluate(new_variable,
                                                      self.search_space)

        # Reduce array of values to min/max
        # select_min params are:
        # params, state, min_sample_ptr, sample_ptr, min_value_ptr, value_ptr, opt_count_ptr, count
        bin_func = pnlvm.LLVMBinaryFunction.from_obj(self, tags=frozenset({"select_min"}))
        ct_param = bin_func.byref_arg_types[0](*self._get_param_initializer(context))
        ct_state = bin_func.byref_arg_types[1](*self._get_state_initializer(context))
        ct_opt_sample = bin_func.byref_arg_types[2](float("NaN"))
        ct_opt_value = bin_func.byref_arg_types[4]()
        ct_opt_count = bin_func.byref_arg_types[6](0)
        assert len(ct_values) == len(ct_alloc)
        ct_count = bin_func.c_func.argtypes[7](len(ct_alloc))

        bin_func(ct_param, ct_state, ct_opt_sample, ct_alloc, ct_opt_value,
                 ct_values, ct_opt_count, ct_count)

        return ct_opt_sample, ct_opt_value, ct_alloc, ct_values

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 **kwargs):
        """Return the sample that yields the optimal value of `objective_function <GridSearch.objective_function>`,
        and possibly all samples evaluated and their corresponding values.

        Optimal value is defined by `direction <GridSearch.direction>`:
        - if *MAXIMIZE*, returns greatest value
        - if *MINIMIZE*, returns least value

        Returns
        -------

        optimal sample, optimal value, saved_samples, saved_values : ndarray, list, list
            first array contains sample that yields the highest or lowest value of `objective_function
            <GridSearch.objective_function>`, depending on `direction <GridSearch.direction>`, and the
            second array contains the value of the function for that sample. If `save_samples
            <GridSearch.save_samples>` is `True`, first list contains all the values sampled in the order they were
            evaluated; otherwise it is empty.  If `save_values <GridSearch.save_values>` is `True`, second list
            contains the values returned by `objective_function <GridSearch.objective_function>` for all the samples
            in the order they were evaluated; otherwise it is empty.
        """

        self.reset_grid()
        return_all_samples = return_all_values = []

        direction = self.parameters.direction._get(context)
        if MPI_IMPLEMENTATION:

            from mpi4py import MPI

            Comm = MPI.COMM_WORLD
            rank = Comm.Get_rank()
            size = Comm.Get_size()

            self.search_space = np.atleast_2d(self.search_space)

            chunk_size = (len(self.search_space) + (size - 1)) // size
            start = chunk_size * rank
            stop = chunk_size * (rank + 1)
            if start > len(self.search_space):
                start = len(self.search_space)
            if stop > len(self.search_space):
                stop = len(self.search_space)

            # FIX:  INITIALIZE TO FULL LENGTH AND ASSIGN DEFAULT VALUES (MORE EFFICIENT):
            samples = np.array([[]])
            sample_optimal = np.empty_like(self.search_space[0])
            values = np.array([])
            value_optimal = float('-Infinity')
            sample_value_max_tuple = (sample_optimal, value_optimal)

            # Set up progress bar
            _show_progress = False
            if hasattr(self, OWNER) and self.owner and self.owner.prefs.reportOutputPref:
                _show_progress = True
                _progress_bar_char = '.'
                _progress_bar_rate_str = ""
                _search_space_size = len(self.search_space)
                _progress_bar_rate = int(10**(np.log10(_search_space_size) - 2))
                if _progress_bar_rate > 1:
                    _progress_bar_rate_str = str(_progress_bar_rate) + " "
                print("\n{} executing optimization process (one {} for each {}of {} samples): ".
                      format(self.owner.name, repr(_progress_bar_char), _progress_bar_rate_str, _search_space_size))
                _progress_bar_count = 0

            for sample in self.search_space[start:stop,:]:

                if _show_progress:
                    increment_progress_bar = (_progress_bar_rate < 1) or not (_progress_bar_count % _progress_bar_rate)
                    if increment_progress_bar:
                        print(_progress_bar_char, end='', flush=True)
                    _progress_bar_count +=1

                # Evaluate objective_function for current sample
                value = self.objective_function(sample, context=context)

                # Evaluate for optimal value
                if direction == MAXIMIZE:
                    value_optimal = max(value, value_optimal)
                elif direction == MINIMIZE:
                    value_optimal = min(value, value_optimal)
                else:
                    assert False, "PROGRAM ERROR: bad value for {} arg of {}: {}".\
                        format(repr(DIRECTION),self.name,direction)

                # FIX: PUT ERROR HERE IF value AND/OR value_max ARE EMPTY (E.G., WHEN EXECUTION_ID IS WRONG)
                # If value is optimal, store corresponing sample
                if value == value_optimal:
                    # Keep track of port values and allocation policy associated with EVC max
                    sample_optimal = sample
                    sample_value_max_tuple = (sample_optimal, value_optimal)

                # Save samples and/or values if specified
                if self.save_values:
                    # FIX:  ASSIGN BY INDEX (MORE EFFICIENT)
                    values = np.append(values, np.atleast_1d(value), axis=0)
                if self.save_samples:
                    if len(samples[0])==0:
                        samples = np.atleast_2d(sample)
                    else:
                        samples = np.append(samples, np.atleast_2d(sample), axis=0)

            # Aggregate, reduce and assign global results
            # combine max result tuples from all processes and distribute to all processes
            max_tuples = Comm.allgather(sample_value_max_tuple)
            # get tuple with "value_max of maxes"
            max_value_of_max_tuples = max(max_tuples, key=lambda max_tuple: max_tuple[1])
            # get value_optimal, port values and allocation policy associated with "max of maxes"
            return_optimal_sample = max_value_of_max_tuples[0]
            return_optimal_value = max_value_of_max_tuples[1]

            if self._return_samples:
                return_all_samples = np.concatenate(Comm.allgather(samples), axis=0)
            if self._return_values:
                return_all_values = np.concatenate(Comm.allgather(values), axis=0)

        else:
            assert direction == MAXIMIZE or direction == MINIMIZE, \
                "PROGRAM ERROR: bad value for {} arg of {}: {}, {}". \
                    format(repr(DIRECTION), self.name, direction)


            ocm = self._get_optimized_composition()
            if ocm is not None and \
               ocm.parameters.comp_execution_mode._get(context).startswith("PTX"):
                    opt_sample, opt_value, all_samples, all_values = self._run_cuda_grid(ocm, variable, context)
                    value_optimal = opt_value
                    sample_optimal = opt_sample
            else:
                last_sample, last_value, all_samples, all_values = super()._function(
                    variable=variable,
                    context=context,
                    params=params,
                )

                optimal_value_count = 1
                value_sample_pairs = zip(all_values, all_samples)
                value_optimal, sample_optimal = next(value_sample_pairs)

                select_randomly = self.parameters.select_randomly_from_optimal_values._get(context)
                for value, sample in value_sample_pairs:
                    if select_randomly and np.allclose(value, value_optimal):
                        optimal_value_count += 1

                        # swap with probability = 1/optimal_value_count in order to achieve
                        # uniformly random selection from identical outcomes
                        probability = 1 / optimal_value_count
                        random_state = self._get_current_function_param("random_state", context)
                        random_value = random_state.rand()

                        if random_value < probability:
                            value_optimal, sample_optimal = value, sample

                    elif (value > value_optimal and direction == MAXIMIZE) or \
                            (value < value_optimal and direction == MINIMIZE):
                        value_optimal, sample_optimal = value, sample
                        optimal_value_count = 1

            if self._return_samples:
                return_all_samples = all_samples
            if self._return_values:
                return_all_values = all_values

        return sample_optimal, value_optimal, return_all_samples, return_all_values

    def _traverse_grid(self, variable, sample_num, context=None):
        """Get next sample from grid.
        This is assigned as the `search_function <OptimizationFunction.search_function>` of the `OptimizationFunction`.
        """
        if self.is_initializing:
            return [signal.start for signal in self.search_space]
        try:
            sample = next(self.grid)
        except StopIteration:
            raise OptimizationFunctionError("Expired grid in {} run from {} "
                                            "(execution_count: {}; num_iterations: {})".
                format(self.__class__.__name__, self.owner.name,
                       self.owner.parameters.execution_count.get(), self.num_iterations))
        return sample

    def _grid_complete(self, variable, value, iteration, context=None):
        """Return False when search of grid is complete
        This is assigned as the `search_termination_function <OptimizationFunction.search_termination_function>`
        of the `OptimizationFunction`.
        """
        try:
            return iteration == self.num_iterations
        except AttributeError:
            return True


class GaussianProcess(OptimizationFunction):
    """
    GaussianProcess(                 \
        default_variable=None,       \
        objective_function=None,     \
        search_space=None,           \
        direction=MAXIMIZE,          \
        max_iterations=1000,         \
        save_samples=False,          \
        save_values=False,           \
        params=None,                 \
        owner=None,                  \
        prefs=None                   \
        )

    Draw samples with dimensionality and bounds specified by `search_space <GaussianProcess.search_space>` and
    return one that optimizes the value of `objective_function <GaussianProcess.objective_function>`.

    .. _GaussianProcess_Procedure:

    **Gaussian Process Procedure**

    The number of items (`SampleIterators <SampleIteartor>` in `search_space <GaussianProcess.search_space>` determines
    the dimensionality of each sample to evaluate by `objective_function <GaussianProcess.objective_function>`,
    with the `start <SampleIterator.start>` and `stop <SampleIterator.stop>` attributes of each `SampleIterator`
    specifying the bounds for sampling along the corresponding dimension.

    When `function <GaussianProcess.function>` is executed, it iterates over the folowing steps:

        - draw sample along each dimension of `search_space <GaussianProcess.search_space>`, within bounds
          specified by `start <SampleIterator.start>` and `stop <SampleIterator.stop>` attributes of each
          `SampleIterator` in the `search_space <GaussianProcess.search_space>` list.
        ..
        - compute value of `objective_function <GaussianProcess.objective_function>` for that sample;

    The current iteration is contained in `iteration <GaussianProcess.iteration>`. Iteration continues until [
    FRED: FILL IN THE BLANK], or `max_iterations <GaussianProcess.max_iterations>` is execeeded.  The function
    returns the sample that yielded either the highest (if `direction <GaussianProcess.direction>`
    is *MAXIMIZE*) or lowest (if `direction <GaussianProcess.direction>` is *MINIMIZE*) value of the `objective_function
    <GaussianProcess.objective_function>`, along with the value for that sample, as well as lists containing all of the
    samples evaluated and their values if either `save_samples <GaussianProcess.save_samples>` or `save_values
    <GaussianProcess.save_values>` is `True`, respectively.

    Arguments
    ---------

    default_variable : list or ndarray : default None
        specifies a template for (i.e., an example of the shape of) the samples used to evaluate the
        `objective_function <GaussianProcess.objective_function>`.

    objective_function : function or method
        specifies function used to evaluate sample in each iteration of the `optimization process
        <GaussianProcess_Procedure>`; it must be specified and must return a scalar value.

    search_space : list or array
        specifies bounds of the samples used to evaluate `objective_function <GaussianProcess.objective_function>`
        along each dimension of `variable <GaussianProcess.variable>`;  each item must be a tuple the first element
        of which specifies the lower bound and the second of which specifies the upper bound.

    direction : MAXIMIZE or MINIMIZE : default MAXIMIZE
        specifies the direction of optimization:  if *MAXIMIZE*, the highest value of `objective_function
        <GaussianProcess.objective_function>` is sought;  if *MINIMIZE*, the lowest value is sought.

    max_iterations : int : default 1000
        specifies the maximum number of times the `optimization process<GaussianProcess_Procedure>` is allowed to
        iterate; if exceeded, a warning is issued and the function returns the optimal sample of those evaluated.

    save_samples : bool
        specifies whether or not to return all of the samples used to evaluate `objective_function
        <GaussianProcess.objective_function>` in the `optimization process <GaussianProcess_Procedure>`
        (i.e., a copy of the `search_space <GaussianProcess.search_space>`.

    save_values : bool
        specifies whether or not to save and return the values of `objective_function <GaussianProcess.objective_function>`
        for all samples evaluated in the `optimization process <GaussianProcess_Procedure>`.

    Attributes
    ----------

    variable : ndarray
        template for sample evaluated by `objective_function <GaussianProcess.objective_function>`.

    objective_function : function or method
        function used to evaluate sample in each iteration of the `optimization process <GaussianProcess_Procedure>`.

    search_space : list or array
        contains tuples specifying bounds within which each dimension of `variable <GaussianProcess.variable>` is
        sampled, and used to evaluate `objective_function <GaussianProcess.objective_function>` in iterations of the
        `optimization process <GaussianProcess_Procedure>`.

    direction : MAXIMIZE or MINIMIZE : default MAXIMIZE
        determines the direction of optimization:  if *MAXIMIZE*, the greatest value of `objective_function
        <GaussianProcess.objective_function>` is sought;  if *MINIMIZE*, the least value is sought.

    iteration : int
        the currention iteration of the `optimization process <GaussianProcess_Procedure>`.

    max_iterations : int
        determines the maximum number of times the `optimization process<GaussianProcess_Procedure>` is allowed to iterate;
        if exceeded, a warning is issued and the function returns the optimal sample of those evaluated.

    save_samples : True
        determines whether or not to save and return all samples evaluated by the `objective_function
        <GaussianProcess.objective_function>` in the `optimization process <GaussianProcess_Procedure>` (if the process
        completes, this should be identical to `search_space <GaussianProcess.search_space>`.

    save_values : bool
        determines whether or not to save and return the value of `objective_function
        <GaussianProcess.objective_function>` for all samples evaluated in the `optimization process <GaussianProcess_Procedure>`.
    """

    componentName = GAUSSIAN_PROCESS_FUNCTION

    class Parameters(OptimizationFunction.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <GaussianProcess.variable>`

                    :default value: [[0], [0]]
                    :type: ``list``
                    :read only: True

                direction
                    see `direction <GaussianProcess.direction>`

                    :default value: `MAXIMIZE`
                    :type: ``str``

                save_samples
                    see `save_samples <GaussianProcess.save_samples>`

                    :default value: True
                    :type: ``bool``

                save_values
                    see `save_values <GaussianProcess.save_values>`

                    :default value: True
                    :type: ``bool``
        """
        variable = Parameter([[0], [0]], read_only=True, pnl_internal=True, constructor_argument='default_variable')

        save_samples = True
        save_values = True

        direction = MAXIMIZE

    # TODO: should save_values be in the constructor if it's ignored?
    # is False or True the correct value?
    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 objective_function:tc.optional(is_function_type)=None,
                 search_space=None,
                 direction:tc.optional(tc.enum(MAXIMIZE, MINIMIZE))=None,
                 save_values:tc.optional(bool)=None,
                 params=None,
                 owner=None,
                 prefs=None,
                 **kwargs):

        search_function = self._gaussian_process_sample
        search_termination_function = self._gaussian_process_satisfied
        self._return_values = save_values
        self._return_samples = save_values
        self.direction = direction

        super().__init__(
            default_variable=default_variable,
            objective_function=objective_function,
            search_function=search_function,
            search_space=search_space,
            search_termination_function=search_termination_function,
            save_samples=True,
            save_values=save_values,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _validate_params(self, request_set, target_set=None, context=None):
        super()._validate_params(request_set=request_set, target_set=target_set,context=context)
        # if SEARCH_SPACE in request_set:
        #     search_space = request_set[SEARCH_SPACE]
        #     # search_space must be specified
        #     if search_space is None:
        #         raise OptimizationFunctionError("The {} arg must be specified for a {}".
        #                                         format(repr(SEARCH_SPACE), self.__class__.__name__))
        #     # must be a list or array
        #     if not isinstance(search_space, (list, np.ndarray)):
        #         raise OptimizationFunctionError("The specification for the {} arg of {} must be a list or array".
        #                                         format(repr(SEARCH_SPACE), self.__class__.__name__))
        #     # must have same number of items as variable
        #     if len(search_space) != len(self.defaults.variable):
        #         raise OptimizationFunctionError("The number of items in {} for {} ([]) must equal that of its {} ({})".
        #                                         format(repr(SEARCH_SPACE), self.__class__.__name__, len(search_space),
        #                                                repr(VARIABLE), len(self.defaults.variable)))
        #     # every item must be a tuple with two elements, both of which are scalars, and first must be <= second
        #     for i in search_space:
        #         if not isinstance(i, tuple) or len(i) != 2:
        #             raise OptimizationFunctionError("Item specified for {} of {} ({}) is not a tuple with two items".
        #                                             format(repr(SEARCH_SPACE), self.__class__.__name__, i))
        #         if not all([np.isscalar(j) for j in i]):
        #             raise OptimizationFunctionError("Both elements of item specified for {} of {} ({}) must be scalars".
        #                                             format(repr(SEARCH_SPACE), self.__class__.__name__, i))
        #         if not i[0] < i[1]:
        #             raise OptimizationFunctionError("First element of item in {} specified for {} ({}) "
        #                                             "must be less than or equal to its second element".
        #                                             format(repr(SEARCH_SPACE), self.__class__.__name__, i))

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 **kwargs):
        """Return the sample that yields the optimal value of `objective_function <GaussianProcess.objective_function>`,
        and possibly all samples evaluated and their corresponding values.

        Optimal value is defined by `direction <GaussianProcess.direction>`:
        - if *MAXIMIZE*, returns greatest value
        - if *MINIMIZE*, returns least value

        Returns
        -------

        optimal sample, optimal value, saved_samples, saved_values : ndarray, list, list
            first array contains sample that yields the highest or lowest value of `objective_function
            <GaussianProcess.objective_function>`, depending on `direction <GaussianProcess.direction>`, and the
            second array contains the value of the function for that sample. If `save_samples
            <GaussianProcess.save_samples>` is `True`, first list contains all the values sampled in the order they were
            evaluated; otherwise it is empty.  If `save_values <GaussianProcess.save_values>` is `True`, second list
            contains the values returned by `objective_function <GaussianProcess.objective_function>` for all the
            samples in the order they were evaluated; otherwise it is empty.
        """

        return_all_samples = return_all_values = []

        # Enforce no MPI for now
        MPI_IMPLEMENTATION = False
        if MPI_IMPLEMENTATION:
            # FIX: WORRY ABOUT THIS LATER
            pass

        else:
            last_sample, last_value, all_samples, all_values = super()._function(
                    variable=variable,
                    context=context,
                    params=params,

            )

            return_optimal_value = max(all_values)
            return_optimal_sample = all_samples[all_values.index(return_optimal_value)]
            if self._return_samples:
                return_all_samples = all_samples
            if self._return_values:
                return_all_values = all_values

        return return_optimal_sample, return_optimal_value, return_all_samples, return_all_values

    # FRED: THESE ARE THE SHELLS FOR THE METHODS I BELIEVE YOU NEED:
    def _gaussian_process_sample(self, variable, sample_num, context=None):
        """Draw and return sample from search_space."""
        # FRED: YOUR CODE HERE;  THIS IS THE search_function METHOD OF OptimizationControlMechanism (i.e., PARENT)
        # NOTES:
        #   This method is assigned as the search function of GaussianProcess,
        #     and should return a sample that will be evaluated in the call to GaussianProcess' `objective_function`
        #     (in the context of use with an OptimizationControlMechanism, a sample is a control_allocation,
        #     and the objective_function is the evaluate method of the agent_rep).
        #   You have accessible:
        #     variable arg:  the last sample evaluated
        #     sample_num:  number of current iteration in the search/sampling process
        #     self.search_space:  self.parameters.search_space._get(context), which you can assume will be a
        #                         list of tuples, each of which contains the sampling bounds for each dimension;
        #                         so its length = length of a sample
        #     (the extra stuff in getting the search space is to support statefulness in parallelization of sims)
        # return self._opt.ask() # [SAMPLE:  VECTOR SAME SHAPE AS VARIABLE]
        return variable

    def _gaussian_process_satisfied(self, variable, value, iteration, context=None):
        """Determine whether search should be terminated;  return `True` if so, `False` if not."""
        # FRED: YOUR CODE HERE;    THIS IS THE search_termination_function METHOD OF OptimizationControlMechanism (
        # i.e., PARENT)
        return iteration==2# [BOOLEAN, SPECIFIYING WHETHER TO END THE SEARCH/SAMPLING PROCESS]


class ParamEstimationFunction(OptimizationFunction):
    """
    ParamEstimationFunction(                 \
        default_variable=None,       \
        objective_function=None,     \
        direction=MAXIMIZE,          \
        max_iterations=1000,         \
        save_samples=False,          \
        save_values=False,           \
        params=None,                 \
        owner=None,                  \
        prefs=None                   \
        )

    Use likelihood free inference to estimate values of parameters for a composition
    so that it best matches some provided ground truth data.

    Arguments
    ---------

    default_variable : list or ndarray : default None
        specifies a template for (i.e., an example of the shape of) the samples used to evaluate the
        `objective_function <ParamEstimationFunction.objective_function>`.

    objective_function : function or method
        specifies function used to evaluate sample in each iteration of the `optimization process
        <ParamEstimationFunction_Procedure>`; it must be specified and must return a scalar value.

    search_space : list or array
        specifies bounds of the samples used to evaluate `objective_function <ParamEstimationFunction.objective_function>`
        along each dimension of `variable <ParamEstimationFunction.variable>`;  each item must be a tuple the first element
        of which specifies the lower bound and the second of which specifies the upper bound.

    direction : MAXIMIZE or MINIMIZE : default MAXIMIZE
        specifies the direction of optimization:  if *MAXIMIZE*, the highest value of `objective_function
        <ParamEstimationFunction.objective_function>` is sought;  if *MINIMIZE*, the lowest value is sought.

    max_iterations : int : default 1000
        specifies the maximum number of times the `optimization process<ParamEstimationFunction_Procedure>` is allowed to
        iterate; if exceeded, a warning is issued and the function returns the optimal sample of those evaluated.

    save_samples : bool
        specifies whether or not to return all of the samples used to evaluate `objective_function
        <ParamEstimationFunction.objective_function>` in the `optimization process <ParamEstimationFunction_Procedure>`
        (i.e., a copy of the `search_space <ParamEstimationFunction.search_space>`.

    save_values : bool
        specifies whether or not to save and return the values of `objective_function <ParamEstimationFunction.objective_function>`
        for all samples evaluated in the `optimization process <ParamEstimationFunction_Procedure>`.

    Attributes
    ----------

    variable : ndarray
        template for sample evaluated by `objective_function <ParamEstimationFunction.objective_function>`.

    objective_function : function or method
        function used to evaluate sample in each iteration of the `optimization process <ParamEstimationFunction_Procedure>`.

    search_space : list or array
        contains tuples specifying bounds within which each dimension of `variable <ParamEstimationFunction.variable>` is
        sampled, and used to evaluate `objective_function <ParamEstimationFunction.objective_function>` in iterations of the
        `optimization process <ParamEstimationFunction_Procedure>`.

    direction : MAXIMIZE or MINIMIZE : default MAXIMIZE
        determines the direction of optimization:  if *MAXIMIZE*, the greatest value of `objective_function
        <ParamEstimationFunction.objective_function>` is sought;  if *MINIMIZE*, the least value is sought.

    iteration : int
        the currention iteration of the `optimization process <ParamEstimationFunction_Procedure>`.

    max_iterations : int
        determines the maximum number of times the `optimization process<ParamEstimationFunction_Procedure>` is allowed to iterate;
        if exceeded, a warning is issued and the function returns the optimal sample of those evaluated.

    save_samples : True
        determines whether or not to save and return all samples evaluated by the `objective_function
        <ParamEstimationFunction.objective_function>` in the `optimization process <ParamEstimationFunction_Procedure>` (if the process
        completes, this should be identical to `search_space <ParamEstimationFunction.search_space>`.

    save_values : bool
        determines whether or not to save and return the value of `objective_function
        <ParamEstimationFunction.objective_function>` for all samples evaluated in the `optimization process <ParamEstimationFunction_Procedure>`.
    """

    componentName = "ParamEstimationFunction"

    class Parameters(OptimizationFunction.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <ParamEstimationFunction.variable>`

                    :default value: [[0], [0]]
                    :type: ``list``
                    :read only: True

                random_state
                    see `random_state <ParamEstimationFunction.random_state>`

                    :default value: None
                    :type: ``numpy.random.RandomState``

                save_samples
                    see `save_samples <ParamEstimationFunction.save_samples>`

                    :default value: True
                    :type: ``bool``

                save_values
                    see `save_values <ParamEstimationFunction.save_values>`

                    :default value: True
                    :type: ``bool``
        """
        variable = Parameter([[0], [0]], read_only=True)
        random_state = Parameter(None, stateful=True, loggable=False)
        save_samples = True
        save_values = True

    @tc.typecheck
    def __init__(self,
                 priors,
                 observed,
                 summary,
                 discrepancy,
                 n_samples,
                 threshold=None,
                 quantile=None,
                 n_sim=None,
                 seed=None,
                 default_variable=None,
                 objective_function:tc.optional(is_function_type)=None,
                 search_space=None,
                 params=None,
                 owner=None,
                 prefs=None,
                 **kwargs):

        # Setup all the arguments we will need to feed to
        # ELFI later.
        self._priors = priors
        self._observed = observed
        self._summary = summary
        self._discrepancy = discrepancy
        self._n_samples = n_samples
        self._threshold = threshold
        self._n_sim = n_sim
        self._quantile = quantile
        self._sampler_args = {'n_samples': self._n_samples,
                              'threshold': self._threshold,
                              'quantile': self._quantile,
                              'n_sim': self._n_sim,
                              'bar': False}

        # If no seed is specified, generate a different random one for
        # each instance
        if seed is None:
            self._seed = get_global_seed()
        else:
            self._seed = seed

        # Setup a RNG for our stuff, we will also pass the seed to ELFI for
        # its crap.
        random_state = np.random.RandomState([seed])

        # Our instance of elfi model
        self._elfi_model = None

        # The simulator function we will pass to ELFI, this is really the objective_function
        # with some stuff wrapped around it to massage its return values and arguments.
        self._sim_func = None

        super().__init__(
            default_variable=default_variable,
            objective_function=objective_function,
            search_function=None,
            search_space=search_space,
            search_termination_function=None,
            save_samples=True,
            save_values=True,
            random_state=random_state,
            params=params,
            owner=owner,
            prefs=prefs,
            context=ContextFlags.CONSTRUCTOR
        )

    @staticmethod
    def _import_elfi():
        """Import any need libraries needed for this classes opertation. Mainly ELFI"""

        # ELFI imports matplotlib at the top level for some dumb reason. Matplotlib
        # doesn't always play nice MacOS X backend. See example:
        # https://github.com/scikit-optimize/scikit-optimize/issues/637
        # Lets import and set the backend to PS to be safe. We aren't plotting anyway
        # I guess. Only do this on Mac OS X
        if sys.platform == "darwin":
            import matplotlib
            matplotlib.use('PS')

        import elfi

        return elfi

    def _validate_params(self, request_set, target_set=None, context=None):
        super()._validate_params(request_set=request_set, target_set=target_set,context=context)

    def _make_simulator_function(self, context):

        # If the objective function hasn't been setup yet, we can simulate anything.
        if self.objective_function is None:
            return None

        # FIXME: All of the checks below are for initializing state, must be a better way
        # Just because the objective_function is setup doesn't mean things are
        # ready to go yet, we could be in initialization or something. Try to
        # invoke it, if we get a TypeError, that means it is not the right
        # function yet.
        try:
            # Call the objective_function and check its return type
            zero_input = np.zeros(len(self.search_space))
            ret = self.objective_function(zero_input,
                                          context=context,
                                          return_results=True)
        except TypeError as ex:
            return None

        # This check is because the default return value for the default objective function
        # is an int.
        if type(ret) is int:
            return None

        # Things are ready, create a function for the simulator that invokes
        # the objective function.
        def simulator(*args, **kwargs):

            # If kwargs None then the simulator has been called without keyword argumets.
            # This means something has gone wrong because our arguments are the parameters
            # (or control signals in PsyNeuLink lingo) we are trying to estimate.
            if kwargs is None:
                raise ValueError("No arguments passed to simulator!")

            # Get the batch size and random state. ELFI passes these arguments around
            # for controlling the number of simulation samples to generate (batch_size)
            # and the numpy random state (random_state) to use during generation.
            batch_size = kwargs.pop('batch_size', 1)
            random_state = kwargs.pop('batch_size', None)

            # Make sure we still have some arguments after popping ELFI's crap (batch_size, randon_state)
            if not kwargs:
                raise ValueError("No arguments passed to simulator!")

            # The control signals (parameter values) of the composition are passed
            # in as arguments. We must set these parameter values before running the
            # simulation\composition. The order of the arguments is the same as the
            # order for the control signals. So the simulator function will have the
            # same argument order as the control signals as well. Note: this will not
            # work in Python 3.5 because dict's have pseudo-random order.
            control_allocation = args

            # FIXME: This doesn't work at the moment. Need to use for loop below.
            # The batch_size is the number of estimates/simulations, set it on the
            # optimization control mechanism.
            # self.owner.parameters.num_estimates.set(batch_size, execution_id)

            # Run batch_size simulations of the PsyNeuLink composition
            results = []
            for i in range(batch_size):
                net_outcome, result = self.objective_function(control_allocation,
                                                           context=context,
                                                           return_results=True)
                results.append(result[0])

            # Turn the list of simulation results into a 2D array of (batch_size, N)
            results = np.stack(results, axis=0)

            return results

        return simulator

    def _initialize_model(self, context):
        """
        Setup the ELFI model for sampling.

        :param context: The current context, we need to pass this to
        the objective function so it must be passed to our simulator function
        implicitly.
        :return: None
        """

        # If the model has not been initialized, try to do it.
        if self._elfi_model is None:

            elfi = ParamEstimationFunction._import_elfi()

            # Try to make the simulator function we will pass to ELFI, this will fail
            # when we are in psyneulink intializaztion phases.
            self._sim_func = self._make_simulator_function(context=context)

            # If it did fail, we return early without initializing, hopefully next time.
            if self._sim_func is None:
                return

            old_model = elfi.get_default_model()

            my_elfi_model = elfi.new_model(self.name, True)

            # FIXME: A lot of checking needs to happen, here. Correct order, valid elfi prior, etc.
            # Construct the ELFI priors from the list of prior specifcations
            elfi_priors = [elfi.Prior(*args, name=param_name) for param_name, args in self._priors.items()]

            # Create the simulator, specifying the priors in proper order as arguments
            Y = elfi.Simulator(self._sim_func, *elfi_priors, observed=self._observed)

            agent_rep_node = elfi.Constant(self.owner.agent_rep)

            # FIXME: This is a hack, we need to figure out a way to elegantly pass these
            # Create the summary nodes
            summary_nodes = [elfi.Summary(args[0], agent_rep_node, Y, *args[1:])
                             for args in self._summary]

            # Create the discrepancy node.
            d = elfi.Distance('euclidean', *summary_nodes)

            self._sampler = elfi.Rejection(d, batch_size=1, seed=self._seed)

            # Store our new model
            self._elfi_model = my_elfi_model

            # Restore the previous default
            elfi.set_default_model(old_model)

    def function(self,
                 variable=None,
                 params=None,
                 context=None,
                 **kwargs):
        """Return the sample that yields the optimal value of `objective_function <ParamEstimationFunction.objective_function>`,
        and possibly all samples evaluated and their corresponding values.

        Optimal value is defined by `direction <ParamEstimationFunction.direction>`:
        - if *MAXIMIZE*, returns greatest value
        - if *MINIMIZE*, returns least value

        Returns
        -------

        optimal sample, optimal value, saved_samples, saved_values : ndarray, list, list
            first array contains sample that yields the highest or lowest value of `objective_function
            <ParamEstimationFunction.objective_function>`, depending on `direction <ParamEstimationFunction.direction>`, and the
            second array contains the value of the function for that sample. If `save_samples
            <ParamEstimationFunction.save_samples>` is `True`, first list contains all the values sampled in the order they were
            evaluated; otherwise it is empty.  If `save_values <ParamEstimationFunction.save_values>` is `True`, second list
            contains the values returned by `objective_function <ParamEstimationFunction.objective_function>` for all the
            samples in the order they were evaluated; otherwise it is empty.
        """

        # Initialize the list of all samples and values
        return_all_samples = return_all_values = []

        # Intialize the optimial control allocation sample and value to zero.
        return_optimal_sample = np.zeros(len(self.search_space))
        return_optimal_value= 0.0

        # Try to initialize the model if it hasn't been.
        if self._elfi_model is None:
            self._initialize_model(context)

        # Intialization can fail for reasons silenty, mainly that PsyNeuLink needs to
        # invoke these functions multiple times during initialization. We only want
        # to proceed if this is the real deal.
        if self._elfi_model is None:
            return return_optimal_sample, return_optimal_value, return_all_samples, return_all_values

        elfi = ParamEstimationFunction._import_elfi()

        old_model = elfi.get_default_model()
        elfi.set_default_model(self._elfi_model)
        # Run the sampler
        result = self._sampler.sample(**self._sampler_args)

        # We now have a set of N samples, where N should be n_samples. This
        # is the N samples that represent the self._quantile from the total
        # number of simulations. N is not the total number of simulation
        # samples. We will return a random sample from this set for the
        # "optimal" control allocation.
        random_state = self._get_current_function_param("random_state", context)
        sample_idx = random_state.randint(low=0, high=result.n_samples)
        return_optimal_sample = np.array(result.samples_array[sample_idx])
        return_optimal_value = result.discrepancies[sample_idx]
        return_all_samples = np.array(result.samples_array)
        return_all_values = np.array(result.discrepancies)

        # Restore the old default
        elfi.set_default_model(old_model)

        print(result)
        return return_optimal_sample, return_optimal_value, return_all_samples, return_all_values
