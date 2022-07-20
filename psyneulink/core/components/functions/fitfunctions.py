import copy

from fastkde import fastKDE
from scipy.interpolate import interpn
from scipy.optimize import differential_evolution

from psyneulink.core.globals import SampleIterator
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.components.functions.nonstateful.optimizationfunctions import OptimizationFunction, \
    OptimizationFunctionError, SEARCH_SPACE

from typing import Union, Optional, List, Dict, Any, Tuple, Callable
import time
import numpy as np
import pandas as pd
from rich.progress import Progress, BarColumn, TimeRemainingColumn

import warnings
import logging

logger = logging.getLogger(__name__)


def get_param_str(params):
    """
    A simple function to turn a dict into a string with commas separating key=value pairs.

    Parameters
    ----------
    params: The parameters to print.

    Returns
    -------
    The string version of the parameter dict

    """
    return ", ".join(f"{name}={value:.5f}" for name, value in params.items())


class BadLikelihoodWarning(UserWarning):
    """
    A custom warning that is used to signal when the likelihood could not be evaluated for some reason.
    This is usually caused when parameter values cause degenerate simulation results (no variance in values).
    It can also be caused when experimental data is not matching any of the simulation results.
    """

    pass


def simulation_likelihood(
    sim_data, exp_data=None, categorical_dims=None, combine_trials=False
):
    """
    Compute the likelihood of a simulation dataset (or the parameters that generated it) conditional
    on a set of experimental data. This function essentially just computes the kernel density estimate (KDE)
    of the simulation data at the experimental data points. If no experimental data is provided just return
    the KDE evaluated at default points provided by the fastkde library.

    Some related work:

    Steven Miletić, Brandon M. Turner, Birte U. Forstmann, Leendert van Maanen,
    Parameter recovery for the Leaky Competing Accumulator model,
    Journal of Mathematical Psychology,
    Volume 76, Part A,
    2017,
    Pages 25-50,
    ISSN 0022-2496,
    https://doi.org/10.1016/j.jmp.2016.12.001.
    (http://www.sciencedirect.com/science/article/pii/S0022249616301663)

    This function uses the wonderful fastKDE package:

    O’Brien, T. A., Kashinath, K., Cavanaugh, N. R., Collins, W. D. & O’Brien, J. P.
    A fast and objective multidimensional kernel density estimation method: fastKDE.
    Comput. Stat. Data Anal. 101, 148–160 (2016).
    <http://dx.doi.org/10.1016/j.csda.2016.02.014>__

    O’Brien, T. A., Collins, W. D., Rauscher, S. A. & Ringler, T. D.
    Reducing the computational cost of the ECF using a nuFFT: A fast and objective probability density estimation method.
    Comput. Stat. Data Anal. 79, 222–234 (2014). <http://dx.doi.org/10.1016/j.csda.2014.06.002>__

    Parameters
    ----------
    sim_data: Data collected over many simulations. This must be either a 2D or 3D numpy array.
        If 2D, the first dimension is the simulation number and the second dimension is data points. That is,
        each row is a simulation. If 3D, the first dimension is the trial, the second dimension is the
        simulation number, and the final dimension is data points.

    exp_data: This must be a numpy array with identical format as the simulation data, with the exception
        that there is no simulation dimension.

    categorical_dims: a list of indices that indicate categorical dimensions of a data point. Length must be
        the same length as last dimension of sim_data and exp_data.

    combine_trials: Combine data across all trials into a single likelihood estimate, this assumes
        that the parameters of the simulations are identical across trials.

    Returns
    -------
    The pdf of simulation data (or in other words, the generating parameters) conditioned on the
    experimental data.

    """

    # Add a singleton dimension for trials if needed.
    if sim_data.ndim == 2:
        sim_data = sim_data[None, :, :]

    if combine_trials and sim_data.shape[0] > 1:
        sim_data = np.vstack(sim_data)[None, :, :]

    if type(categorical_dims) != np.ndarray:
        categorical_dims = np.array(categorical_dims)

    con_sim_data = sim_data[:, :, ~categorical_dims]
    cat_sim_data = sim_data[:, :, categorical_dims]

    categories = np.unique(cat_sim_data)
    kdes = []
    for trial in range(len(con_sim_data)):
        s = con_sim_data[trial]

        # Compute a separate KDE for each combination of categorical variables.
        dens_u = {}
        for category in categories:

            # Get the subset of simulations that correspond to this category
            dsub = s[cat_sim_data[trial] == category]

            # If we didn't get enough simulation results for this category, don't do
            # a KDE
            # if len(dsub) < 100:
            #     dens_u[category] = (None, None)
            #     continue

            # If any dimension of the data has a 0 range (all are same value) then
            # this will cause problems doing the KDE, skip.
            data_range = (
                np.max(dsub) - np.min(dsub)
                if dsub.ndim == 1
                else np.amax(dsub, 1) - np.amin(dsub, 1)
            )
            if np.any(data_range == 0):
                dens_u[category] = (None, None)
                warnings.warn(
                    BadLikelihoodWarning(
                        f"Could not perform kernel density estimate. Range of simulation data was 0 for at least "
                        f"one dimension. Range={data_range}"
                    )
                )
                continue

            # Do KDE
            fKDE = fastKDE.fastKDE(dsub, doSaveMarginals=False)
            pdf = fKDE.pdf
            axes = fKDE.axes

            # Scale the pdf by the fraction of simulations that fall within this category
            pdf = pdf * (len(dsub) / len(s))

            # Save the KDE values and axes for this category
            dens_u[category] = (pdf, axes)

        kdes.append(dens_u)

    # If we are passed experimental data, evaluate the KDE at the experimental data points
    if exp_data is not None:

        # For numerical reasons, make zero probability a really small value. This is because we are taking logs
        # of the probabilities at the end.
        ZERO_PROB = 1e-10

        kdes_eval = np.zeros((len(exp_data),))
        for trial in range(len(exp_data)):

            # Extract the categorical values for this experimental trial
            exp_trial_cat = exp_data[trial, categorical_dims]

            if len(exp_trial_cat) == 1:
                exp_trial_cat = exp_trial_cat[0]

            # Get the right KDE for this trial, if all simulation trials have been combined
            # use that KDE for all trials of experimental data.
            if len(kdes) == 1:
                kde, axes = kdes[0].get(exp_trial_cat, (None, None))
            else:
                kde, axes = kdes[trial].get(exp_trial_cat, (None, None))

            # Linear interpolation using the grid we computed the KDE
            # on.
            if kde is not None:
                kdes_eval[trial] = interpn(
                    axes,
                    kde,
                    exp_data[trial, ~categorical_dims],
                    method="linear",
                    bounds_error=False,
                    fill_value=ZERO_PROB,
                )
            else:
                kdes_eval[trial] = ZERO_PROB

        # Check to see if any of the trials have non-zero likelihood, if not, something is probably wrong
        # and we should warn the user.
        if np.alltrue(kdes_eval == ZERO_PROB):
            warnings.warn(
                BadLikelihoodWarning(
                    "Evaluating likelihood generated by simulation data resulted in zero values for all trials "
                    "of experimental data. This means the model is not generating data similar to your "
                    "experimental data. If you have categorical dimensions, make sure values match exactly to "
                    "output values of the composition. Also make sure parameter ranges you are searching over "
                    "are reasonable for your data."
                )
            )

        return kdes_eval

    else:
        return kdes


class MaxLikelihoodEstimator(OptimizationFunction):
    """
    A class for performing parameter estimation for a composition using maximum likelihood estimation (MLE). When a
    ParameterEstimationComposition is used for `ParameterEstimationComposition_Data_Fitting`, an instance of this class
    can be assigned to the ParameterEstimationComposition's
    `optimization_function <ParameterEstimationComposition.optimization_function>`.
    """

    def __init__(
        self,
        search_space=None,
        save_samples=None,
        save_values=None,
        **kwargs,
    ):

        search_function = self._traverse_grid
        search_termination_function = self._grid_complete

        # A cached copy of our log-likelihood function. This can only be created after the function has been assigned
        # to a OptimizationControlMechanism under and ParameterEstimationComposition.
        self._ll_func = None

        # Set num_iterations to a default value of 1, this will be reset in reset() based on the search space
        self.num_iterations = 1

        # When the OCM passes in the search space, we need to modify it so that the fitting parameters are
        # set to single values since we want to use SciPy optimize to drive the search for these parameters.
        # The randomization control signal is not set to a single value so that the composition still uses
        # the evaluate machinery to get the different simulations for a given setting of parameters chosen
        # by scipy during optimization. This variable keeps track of the original search space.
        self._full_search_space = None

        super().__init__(
            search_space=search_space,
            save_samples=save_samples,
            save_values=save_values,
            search_function=search_function,
            search_termination_function=search_termination_function,
            aggregation_function=None,
            **kwargs
        )

    @handle_external_context(fallback_most_recent=True)
    def reset(self, search_space, context=None, **kwargs):
        """Assign size of `search_space <MaxLikelihoodEstimator.search_space>"""

        # We need to modify the search space
        self._full_search_space = copy.deepcopy(search_space)

        # Modify all of the search space (except the randomization control signal) so that with each
        # call to evaluate we only evaluate a single parameter setting. Scipy optimize will direct
        # the search procedure so we will reset the actual value of these singleton iterators dynamically
        # on each search iteration executed during the call to _function.
        randomization_dimension = kwargs.get('randomization_dimension', len(search_space) - 1)
        for i in range(len(search_space)):
            if i != randomization_dimension:
                search_space[i] = SampleIterator([next(search_space[i])])

        super(MaxLikelihoodEstimator, self).reset(search_space=search_space, context=context, **kwargs)
        owner_str = ''
        if self.owner:
            owner_str = f' of {self.owner.name}'
        for i in search_space:
            if i is None:
                raise OptimizationFunctionError(f"Invalid {repr(SEARCH_SPACE)} arg for {self.name}{owner_str}; "
                                                f"every dimension must be assigned a {SampleIterator.__name__}.")
            if i.num is None:
                raise OptimizationFunctionError(f"Invalid {repr(SEARCH_SPACE)} arg for {self.name}{owner_str}; each "
                                                f"{SampleIterator.__name__} must have a value for its 'num' attribute.")

        self.num_iterations = np.product([i.num for i in search_space])

    def _run_simulations(self, *args, context=None):
        """
        Run the simulations we need for estimating the likelihood for given control allocation.
        This function has side effects as it sets the search_space parameter of the
        OptimizationFunction to the control allocation.
        """

        # Use the default variable for the function (control allocation), we don't need this for data fitting.
        variable = self.defaults.variable

        # Check that we have the proper number of arguments to map to the fitting parameters.
        if len(args) != len(self.fit_param_names):
            raise ValueError(
                f"Expected {len(self.fit_param_names)} arguments, got {len(args)}"
            )

        # Set the search space to the control allocation. The only thing evaluate is actually "searching" over is the
        # randomization dimension, which should be the last sample iterator in the search space list.
        search_space = self.parameters.search_space._get(context)
        for i, arg in enumerate(args):

            # Map the args in order of the fittable parameters
            if i < len(search_space) - 1:
                assert search_space[i].num == 1, "Search space for this dimension must be a single value, during search " \
                                                 "we will change the value but not the shape."

                # All of this code is required to set the value of the singleton search space without creating a new
                # object. It seems cleaner to just use search_space[i] = SampleIterator([arg]) but this seems to cause
                # problems for Jan in compilation. Need to confirm this, maybe its ok as long as size doesn't change.
                # We can protect against this with the above assert.
                search_space[i].specification = [arg]
                search_space[i].generator = search_space[i].specification
                search_space[i].start = arg
            else:
                raise ValueError("Too many arguments passed to run_simulations")

        # Reset the search grid
        self.reset_grid()

        # FIXME: This is a hack to make sure that state_features gets all trials worth of inputs.
        # We need to set the inputs for the composition during simulation, override the state features with the
        # inputs dict passed to the PEC constructor. This assumes that the inputs dict has the same order as the
        # state features.
        # for state_input_port, value in zip(self.owner.state_input_ports, self.inputs.values()):
        #     state_input_port.parameters.value._set(value, context)

        # Clear any previous results from the composition

        # Evaluate objective_function for each sample
        last_sample, last_value, all_samples, all_values = self._evaluate(
            variable=variable,
            context=context,
            params=None,
            fit_evaluate=True,
        )

        # We need to swap the simulation (randomization dimension) with the output dimension so things
        # are in the right order for the likelihood computation.
        all_values = np.transpose(all_values, (0, 2, 1))

        return all_values

    def _make_loglikelihood_func(self, context=None):
        """
        Make a function that computes the log likelihood of the simulation results.
        """
        def ll(*args):
            sim_data = self._run_simulations(*args, context=context)

            # Compute the likelihood given the data
            like = simulation_likelihood(
                sim_data=sim_data,
                exp_data=self.data,
                categorical_dims=self.data_categorical_dims,
                combine_trials=False,
            )

            # Make 0 densities very small so log doesn't explode
            like[like == 0.0] = 1.0e-10

            return np.sum(np.log(like)), sim_data

        return ll

    def log_likelihood(self, *args, context=None):
        """
        Compute the log-likelihood of the data given the specified parameters of the model. This function will raise
        aa exeception if the function has not been assigned as the function of and OptimizationControlMechanism. An
        OCM is required in order to simulate results of the model for computing the likelihood.

        Arguments
        ---------
        *args :
            Positional args, one for each paramter of the model. These must correspond directly to the parameters that
            have been specified in the `parameters` argument of the constructor.

        context: Context
            The context in which the log-likelihood is to be evaluated.

        Returns
        -------
        The sum of the log-likelihoods of the data given the specified parameters of the model.
        """

        if self.owner is None:
            raise ValueError("Cannot compute a log-likelihood without being assigned as the function of an "
                             "OptimizationControlMechanism. See the documentation for the "
                             "ParameterEstimationControlMechanism for more information.")

        # Make sure we have instantiated the log-likelihood function.
        if self._ll_func is None:
            self._ll_func = self._make_loglikelihood_func(context=context)

        return self._ll_func(*args)

    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 **kwargs):

        # We need to set the aggregation function to None so that calls to evaluate do not aggregate results
        # of simulations. We want all results for all simulations so we can compute the likelihood ourselves.
        self.parameters.aggregation_function._set(None, context)

        # FIXME: Setting these default values up properly is a pain while initializing, ask Jon
        optimal_sample = np.array([[0.0], [0.0], [0.0]])
        optimal_value = np.array([1.0])
        saved_samples = []
        saved_values = []

        if not self.is_initializing:

            ocm = self.owner
            if ocm is None:
                raise ValueError("MaximumLikelihoodEstimator must be assigned to an OptimizationControlMechanism, "
                                 "self.owner is None")

            # Get a log likelihood function that can be used to compute the log likelihood of the simulation results
            ll_func = self._make_loglikelihood_func(context=context)

            # FIXME: This should be found with fitting but it is too slow!
            # We can at least return the evaluation of the log-likelihood function for testing purposes
            self.owner.optimal_value, saved_values = ll_func(0.3, 0.15)
            self.owner.optimal_parameters = np.array([[0.3, 0.15]])

            # Run the MLE optimization
            # results = self._fit(ll_func=ll_func)

        return optimal_sample, optimal_value, saved_samples, saved_values

    def _fit(self, ll_func: Callable, display_iter: bool = False, save_iterations: bool = False):

        bounds = list(self.fit_param_bounds.values())

        # If the user has rich installed, make a nice progress bar
        from rich.progress import Progress

        iterations = []

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "Convergence: [progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
        ) as progress:
            opt_task = progress.add_task(
                "Maximum likelihood optimization ...", total=100, start=False
            )

            warns_with_params = []
            with warnings.catch_warnings(record=True) as warns:

                # Create a wrapper function for the objective.
                def neg_log_like(x):
                    params = dict(zip(self.fit_param_names, x))
                    t0 = time.time()
                    p = -ll_func(*x)
                    elapsed = time.time() - t0

                    # Keep a log of warnings and the parameters that caused them
                    if len(warns) > 0 and warns[-1].category == BadLikelihoodWarning:
                        warns_with_params.append((warns[-1], params))

                    # Are we displaying each iteration
                    if display_iter:

                        # If we got a warning generating the likelihood, report it
                        if (
                            len(warns) > 0
                            and warns[-1].category == BadLikelihoodWarning
                        ):
                            progress.console.print(f"Warning: ", style="bold red")
                            progress.console.print(
                                f"{warns[-1].message}", style="bold red"
                            )
                            progress.console.print(
                                f"{get_param_str(params)}, Neg-Log-Likelihood: {p}, "
                                f"Likelihood-Eval-Time: {elapsed} (seconds)",
                                style="bold red",
                            )
                            # Clear the warnings
                            warns.clear()
                        else:
                            progress.console.print(
                                f"{get_param_str(params)}, Neg-Log-Likelihood: {p}, "
                                f"Likelihood-Eval-Time: {elapsed} (seconds)"
                            )

                    return p

                def progress_callback(x, convergence):
                    progress.start_task(opt_task)
                    params = dict(zip(self.fit_param_names, x))
                    convergence_pct = 100.0 * convergence
                    progress.console.print(
                        f"[green]Current Best Parameters: {get_param_str(params)}, Neg-Log-Likelihood: {neg_log_like(x)}"
                    )

                    # If we encounter any BadLikelihoodWarnings. Summarize them for the user
                    if len(warns_with_params) > 0:
                        progress.console.print(
                            "Warning: degenerate likelihood for the following parameter values ",
                            style="bold red",
                        )
                        for w in warns_with_params:
                            progress.console.print(
                                f"\t{get_param_str(w[1])}", style="bold red"
                            )
                        progress.console.print(
                            "If these warnings are intermittent, check to see if search "
                            "space is appropriately bounded. If they are constant, make sure "
                            "experimental data and output of your composition are similar.",
                            style="bold red",
                        )

                    progress.update(opt_task, completed=convergence_pct)

                r = differential_evolution(
                    neg_log_like,
                    bounds,
                    callback=progress_callback,
                    maxiter=500,
                    polish=False,
                )

            # Bind the fitted parameters to their names
            fitted_params = dict(zip(list(self.fit_param_names), r.x))

        # Save all the results
        output_dict = {
            "fitted_params": fitted_params,
            "neg-log-likelihood": r.fun,
        }

        return output_dict

    @property
    def fit_param_names(self):
        """Get a unique name for each parameter in the fit."""
        if self.owner is not None:
            return [cs.efferents[0].receiver.name
                    for i, cs in enumerate(self.owner.control_signals)
                    if i != self.randomization_dimension]

    @property
    def fit_param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get the allocation samples for just the fitting parameters. Whatever they are, turn them into upper and lower
        bounds.

        Returns:
            A dict mapping parameter names to (lower, upper) bounds.
        """
        if self.owner is not None:
            acs = [cs.allocation_samples
                   for i, cs in enumerate(self.owner.control_signals)
                   if i != self.randomization_dimension]

            bounds = [(float(min(s)), float(max(s))) for s in acs]
            return dict(zip(self.fit_param_names, bounds))
