import copy

import optuna.samplers
from fastkde import fastKDE
from scipy.interpolate import interpn
from scipy.optimize import differential_evolution
from beartype import beartype

from psyneulink.core.globals import SampleIterator
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.components.functions.nonstateful.optimizationfunctions import (
    OptimizationFunction,
    OptimizationFunctionError,
    SEARCH_SPACE,
)
from psyneulink.core.globals.parameters import check_user_specified

from psyneulink._typing import (
    Dict,
    Tuple,
    Callable,
    List,
    Optional,
    Union,
    Type,
    Literal,
)


import time
import numpy as np

from rich.progress import Progress, BarColumn, TimeRemainingColumn

import warnings
import logging

logger = logging.getLogger(__name__)

__all__ = ["PECOptimizationFunction", "BadLikelihoodWarning", "PECObjectiveFuncWarning"]


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
    return ", ".join(
        f"{name.replace('PARAMETER_CIM_', '')}={value:.5f}"
        for name, value in params.items()
    )


class PECObjectiveFuncWarning(UserWarning):
    """
    A custom warning that is used to signal when the objective function could not be evaluated for some reason.
    This is usually caused when parameter values cause degenerate simulation results (no variance in values).
    """

    pass


class BadLikelihoodWarning(PECObjectiveFuncWarning):
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

    if len(categories) > 10:
        raise ValueError("Too many unique values present for a categorical dimension.")

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
            if len(dsub) < 10:
                dens_u[category] = (None, None)
                continue

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

        # Make 0 densities very small so log doesn't explode later
        kdes_eval[kdes_eval == 0.0] = ZERO_PROB

        return kdes_eval

    else:
        return kdes


class PECOptimizationFunction(OptimizationFunction):
    """
    A subclass of OptimizationFunction that is used to interface with the PEC. This class is used to specify the
    search method to utilize for optimization or data fitting. It is not to be confused with the `objective_function`
    that defines the optimization problem to be solved.

    Arguments
    ---------

    method :
        The search method to use for optimization. The following methods are currently supported:

            - 'differential_evolution' : Differential evolution as implemented by scipy.optimize.differential_evolution
            - optuna.samplers: Pass any instance of an optuna sampler to use optuna for optimization.

    objective_function :
        The objective function to use for optimization. This is the function that defines the optimization problem the
        PEC is trying to solve. The function is used to evaluate the `values <Mechanism_Base.value>` of the
        `outcome_variables <ParameterEstimationComposition.outcome_variables>`, according to which combinations of
        `parameters <ParameterEstimationComposition.parameters>` are assessed; this must be an `Callable`
        that takes a 3D array as its only argument, the shape of which must be (**num_estimates**, **num_trials**,
        number of **outcome_variables**).  The function should specify how to aggregate the value of each
        **outcome_variable** over **num_estimates** and/or **num_trials** if either is greater than 1.

    max_iterations :
        The maximum number of iterations to run the optimization for. In differential evolution, this is the number of
        generations. In optuna, this is the number of trials.

    direction :
        Whether to maximize or minimize the objective function. If 'maximize', the objective function is maximized. If
        'minimize', the objective function is minimized.


    """

    @check_user_specified
    @beartype
    def __init__(
        self,
        method: Union[Literal["differential_evolution"], optuna.samplers.BaseSampler],
        objective_function: Optional[Callable] = None,
        search_space=None,
        save_samples: Optional[bool] = None,
        save_values: Optional[bool] = None,
        max_iterations: int = 500,
        direction: Literal["maximize", "minimize"] = "maximize",
        **kwargs,
    ):
        self.method = method
        self.direction = direction

        # The outcome variables to select from the composition's output need to be specified. These will be
        # set automatically by the PEC when PECOptimizationFunction is passed to it.
        self.outcome_variable_indices = None

        # The objective function to use for optimization. We can't set objective_function directly
        # because that will be set to agent_rep.evaluate when the PECOptimizationFunction is passed to
        # the OCM. Instead, we set self._pec_objective_function to the objective function, self.objective_function
        # will be used to compute just the simulation results, the these will then be passed to the
        # _pec_objective_function. Very confusing!
        self._pec_objective_function = objective_function

        # Are we in data fitting mode, or generic optimization. This is set automatically by the PEC when
        # PECOptimizationFunction is passed to it. It only really determines whether some cosmetic
        # things.
        self.data_fitting_mode = False

        # This is a bit confusing but PECOptimizationFunction utilizes the OCM search machinery only to run
        # simulations of the composition under different randomization. Thus, regardless of the method passed
        # to PECOptimize, we always set the search_function and search_termination_function for GridSearch.
        # The grid in our case is only over the randomization control signal.
        search_function = self._traverse_grid
        search_termination_function = self._grid_complete

        # When the OCM passes in the search space, we need to modify it so that the fitting parameters are
        # set to single values since we want to use SciPy optimize to drive the search for these parameters.
        # The randomization control signal is not set to a single value so that the composition still uses
        # the evaluate machinery to get the different simulations for a given setting of parameters chosen
        # by scipy during optimization. This variable keeps track of the original search space.
        self._full_search_space = None

        # Set num_iterations to a default value of 1, this will be reset in reset() based on the search space
        self.num_iterations = 1

        # Store max_iterations, this should be a common parameter for all optimization methods
        self.max_iterations = max_iterations

        # A cached copy of our log-likelihood function. This can only be created after the function has been assigned
        # to a OptimizationControlMechanism under and ParameterEstimationComposition.
        self._ll_func = None

        # This is the generation number we are on in the search, this corresponds to iterations in
        # differential_evolution
        self.gen_count = 1

        # Keeps track of the number of objective function evaluations during search
        self.num_evals = 0

        # Keep track of the best parameters
        self._best_params = {}

        super().__init__(
            search_space=search_space,
            save_samples=save_samples,
            save_values=save_values,
            search_function=search_function,
            search_termination_function=search_termination_function,
            aggregation_function=None,
            **kwargs,
        )

    def set_pec_objective_function(self, objective_function: Callable):
        """
        Set the PEC objective function, this is the function that will be called by the OCM to evaluate
        the simulation results generated by the composition when it is simulated by the PEC.
        """
        self._pec_objective_function = objective_function

    @handle_external_context(fallback_most_recent=True)
    def reset(self, search_space, context=None, **kwargs):
        """Assign size of `search_space <MaxLikelihoodEstimator.search_space>"""

        # We need to modify the search space
        self._full_search_space = copy.deepcopy(search_space)

        # Modify all of the search space (except the randomization control signal) so that with each
        # call to evaluate we only evaluate a single parameter setting. Scipy optimize will direct
        # the search procedure so we will reset the actual value of these singleton iterators dynamically
        # on each search iteration executed during the call to _function.
        randomization_dimension = kwargs.get(
            "randomization_dimension", len(search_space) - 1
        )
        for i in range(len(search_space)):
            if i != randomization_dimension:
                search_space[i] = SampleIterator([next(search_space[i])])

        super().reset(search_space=search_space, context=context, **kwargs)
        owner_str = ""
        if self.owner:
            owner_str = f" of {self.owner.name}"
        for i in search_space:
            if i is None:
                raise OptimizationFunctionError(
                    f"Invalid {repr(SEARCH_SPACE)} arg for {self.name}{owner_str}; "
                    f"every dimension must be assigned a {SampleIterator.__name__}."
                )
            if i.num is None:
                raise OptimizationFunctionError(
                    f"Invalid {repr(SEARCH_SPACE)} arg for {self.name}{owner_str}; each "
                    f"{SampleIterator.__name__} must have a value for its 'num' attribute."
                )

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
            len_search_space = (
                len(search_space)
                if self.owner.num_estimates is None
                else len(search_space) - 1
            )
            if i < len_search_space:
                assert search_space[i].num == 1, (
                    "Search space for this dimension must be a single value, during search "
                    "we will change the value but not the shape."
                )

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

        # Evaluate objective_function for each sample
        last_sample, last_value, all_samples, all_values = self._evaluate(
            variable=variable,
            context=context,
            params=None,
            fit_evaluate=True,
        )

        # We need to swap the simulation (randomization dimension) with the output dimension so things
        # are in the right order passing to the objective_function call signature.
        all_values = np.transpose(all_values, (0, 2, 1))

        return all_values

    def _make_objective_func(self, context=None):
        """
        Make an objective function to pass to an optimization algorithm. Creates a function that runs simulations and
        then feeds the results self._pec_objective_func. This cannot be invoked until the PECOptimizationFunction
        (self) has been assigned to an OptimizationControlMechanism.
        """

        def objfunc(*args):
            sim_data = self._run_simulations(*args, context=context)

            # The composition might have more outputs than outcome variables, we need to subset the ones we need.
            sim_data = sim_data[:, :, self.outcome_variable_indices]

            return self._pec_objective_function(sim_data)

        return objfunc

    def _function(self, variable=None, context=None, params=None, **kwargs):
        """
        Run the optimization algorithm to find the optimal control allocation.
        """

        optimal_sample = self.variable
        optimal_value = np.array([1.0])
        saved_samples = []
        saved_values = []

        if not self.is_initializing:
            ocm = self.owner
            if ocm is None:
                raise ValueError(
                    "PECOptimizationFunction must be assigned to an OptimizationControlMechanism, "
                    "self.owner is None"
                )

            # Get the objective function that we are trying to minimize
            f = self._make_objective_func(context=context)

            # Run the MLE optimization
            results = self._fit(obj_func=f)

            # Get the optimal function value and sample
            optimal_value = results["optimal_value"]

            optimal_sample = list(results["fitted_params"].values())

            # Replace randomization dimension to match expected dimension of output_values of OCM. This is ugly but
            # necessary.
            if self.owner.num_estimates is not None:
                optimal_sample = optimal_sample + [0.0]

        return optimal_sample, optimal_value, saved_samples, saved_values

    @property
    def obj_func_desc_str(self):
        return "Log-Likelihood" if self.data_fitting_mode else "Obj-Func-Value"

    @property
    def opt_task_desc_str(self):
        direction_str = "Maximizing" if self.direction == "maximize" else "Minimizing"
        task_disp = (
            "Maximum Likelihood Estimation"
            if self.data_fitting_mode
            else f"{direction_str} Objective Function"
        )
        return f"{task_disp} (num_estimates={self.num_estimates}) ..."

    def _fit(
        self,
        obj_func: Callable,
        display_iter: bool = True,
    ):
        if self.method == "differential_evolution":
            return self._fit_differential_evolution(obj_func, display_iter)
        elif isinstance(self.method, optuna.samplers.BaseSampler):
            return self._fit_optuna(
                obj_func=obj_func, opt_func=self.method, display_iter=display_iter
            )
        else:
            raise ValueError(f"Invalid optimization_function method: {self.method}")

    def _make_obj_func_wrapper(
        self,
        progress,
        display_iter,
        warns,
        warns_with_params,
        obj_func,
        like_eval_task=None,
        evals_per_iteration=None,
        ignore_direction=False,
    ):
        """
        Create a wrapper function for the objective function that keeps track of progress and warnings.
        """
        direction = 1 if self.direction == "minimize" or ignore_direction else -1

        # This is the number of evaluations we need per search iteration.
        self.num_evals = 0

        # Create a wrapper function for the objective. This lets us keep track of progress and such
        def objfunc_wrapper(x):
            params = dict(zip(self.fit_param_names, x))
            t0 = time.time()
            obj_val = obj_func(*x)
            p = direction * obj_val
            elapsed = time.time() - t0
            self.num_evals = self.num_evals + 1

            # Keep a log of warnings and the parameters that caused them
            if len(warns) > 0 and warns[-1].category == PECObjectiveFuncWarning:
                warns_with_params.append((warns[-1], params))

            # Are we displaying each iteration
            if display_iter:
                # If we got a warning generating the objective function value, report it
                if len(warns) > 0 and warns[-1].category == PECObjectiveFuncWarning:
                    progress.console.print(f"Warning: ", style="bold red")
                    progress.console.print(f"{warns[-1].message}", style="bold red")
                    progress.console.print(
                        f"{get_param_str(params)}, {self.obj_func_desc_str}: {obj_val}, "
                        f"Eval-Time: {elapsed} (seconds)",
                        style="bold red",
                    )
                    # Clear the warnings
                    warns.clear()
                else:
                    progress.console.print(
                        f"{get_param_str(params)}, {self.obj_func_desc_str}: {obj_val}, "
                        f"Eval-Time: {elapsed} (seconds)"
                    )

                # Certain algorithms like differential evolution evaluate the objective function multiple times per
                # iteration. We need to update the progress bar accordingly.
                if evals_per_iteration is not None:
                    # We need to update the progress bar differently depending on whether we are doing
                    # the first generation (which is twice as long) or not.
                    if self.num_evals < 2 * evals_per_iteration:
                        max_evals = 2 * evals_per_iteration
                        progress.tasks[like_eval_task].total = max_evals
                        eval_task_str = f"|-- Iteration {self.gen_count} ..."
                        progress.tasks[like_eval_task].description = eval_task_str
                        progress.update(
                            like_eval_task, completed=self.num_evals % max_evals
                        )
                    else:
                        max_evals = evals_per_iteration
                        progress.tasks[like_eval_task].total = max_evals
                        eval_task_str = f"|-- Iteration {self.gen_count} ..."
                        progress.tasks[like_eval_task].description = eval_task_str
                        progress.update(
                            like_eval_task,
                            completed=(self.num_evals - (2 * evals_per_iteration))
                            % max_evals,
                        )

            return p

        return objfunc_wrapper

    def _fit_differential_evolution(
        self,
        obj_func: Callable,
        display_iter: bool = True,
    ):
        """
        Implementation of search using scipy's differential_evolution algorithm.
        """

        bounds = self.fit_param_bounds

        # We just need the upper and lower bounds for the differential evolution algorithm. The step size is not used.
        bounds = list([(lb, ub) for name, (lb, ub, step) in bounds.items()])

        # Get a seed to pass to scipy for its search. Make this dependent on the seed of the
        # OCM
        seed_for_scipy = self.owner.initial_seed

        direction = 1 if self.direction == "minimize" else -1

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "Completed: [progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
        ) as progress:
            opt_task = progress.add_task(self.opt_task_desc_str, total=100)

            # This is the number of evaluations we need per search iteration.
            evals_per_iteration = 15 * len(self.fit_param_names)
            self.gen_count = 1

            if display_iter:
                eval_task_str = f"|-- Iteration 1 ..."
                like_eval_task = progress.add_task(
                    eval_task_str, total=evals_per_iteration
                )

            progress.update(opt_task, completed=0)

            warns_with_params = []
            with warnings.catch_warnings(record=True) as warns:
                objfunc_wrapper = self._make_obj_func_wrapper(
                    progress,
                    display_iter,
                    warns,
                    warns_with_params,
                    obj_func,
                    like_eval_task,
                    evals_per_iteration,
                )

                def progress_callback(x, convergence):
                    params = dict(zip(self.fit_param_names, x))
                    convergence_pct = 100.0 * convergence
                    progress.console.print(
                        f"[green]Current Best Parameters: {get_param_str(params)}, "
                        f"{self.obj_func_desc_str}: {obj_func(*x)}, "
                        f"Convergence: {convergence_pct}"
                    )

                    # If we encounter any PECObjectiveFuncWarnings. Summarize them for the user
                    if len(warns_with_params) > 0:
                        progress.console.print(
                            f"Warning: degenerate {self.obj_func_desc_str} values for the following parameter values ",
                            style="bold red",
                        )
                        for w in warns_with_params:
                            progress.console.print(
                                f"\t{get_param_str(w[1])}", style="bold red"
                            )
                        progress.console.print(
                            "If these warnings are intermittent, check to see if search "
                            "space is appropriately bounded. If they are constant, and you are fitting to"
                            "data, make sure experimental data and output of your composition are similar.",
                            style="bold red",
                        )

                    progress.update(opt_task, completed=convergence_pct)
                    self.gen_count = self.gen_count + 1

                r = differential_evolution(
                    objfunc_wrapper,
                    bounds,
                    callback=progress_callback,
                    maxiter=self.parameters.max_iterations.get() - 1,
                    seed=seed_for_scipy,
                    popsize=15,
                    polish=False,
                )

            # Bind the fitted parameters to their names
            fitted_params = dict(zip(list(self.fit_param_names), r.x))

        # Save all the results
        output_dict = {
            "fitted_params": fitted_params,
            "optimal_value": direction * r.fun,
        }

        return output_dict

    def _fit_optuna(
        self,
        obj_func: Callable,
        opt_func: Union[optuna.samplers.BaseSampler, Type[optuna.samplers.BaseSampler]],
        display_iter: bool = True,
    ):
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "Completed: [progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
        ) as progress:
            max_iterations = self.parameters.max_iterations.get()

            opt_task = progress.add_task(self.opt_task_desc_str, total=max_iterations)
            progress.update(opt_task, completed=0)

            warns_with_params = []
            with warnings.catch_warnings(record=True) as warns:
                # Create a wrapper for the objective function that will let us catch warnings and record progress.
                # For optuna, we can ignore the direction of search because it is handled by the Optuna API when
                # setting up the optimization.
                objfunc_wrapper = self._make_obj_func_wrapper(
                    progress=progress,
                    display_iter=display_iter,
                    warns=warns,
                    warns_with_params=warns_with_params,
                    obj_func=obj_func,
                    ignore_direction=True,
                )

                # Optuna has an interface where the objective function calls the API to get
                # the current values for the parameter rather than them being passed
                # directly. So we need to wrap the wrapper
                def objfunc_wrapper_wrapper(trial):
                    for name, (lower, upper, step) in self.fit_param_bounds.items():
                        trial.suggest_float(name, lower, upper, step=step)

                    return objfunc_wrapper(list(trial.params.values()))

                self._best_params = {}

                def progress_callback(study, trial):
                    if self._best_params != study.best_params:
                        self._best_params = study.best_params
                        progress.console.print(
                            f"[green]Current Best Parameters: {get_param_str(self._best_params)}, "
                            f"{self.obj_func_desc_str}: {study.best_value}, "
                        )

                    # If we encounter any PECObjectiveFuncWarnings. Summarize them for the user
                    if len(warns_with_params) > 0:
                        progress.console.print(
                            f"Warning: degenerate {self.obj_func_desc_str} values for the following parameter values ",
                            style="bold red",
                        )
                        for w in warns_with_params:
                            progress.console.print(
                                f"\t{get_param_str(w[1])}", style="bold red"
                            )
                        progress.console.print(
                            "If these warnings are intermittent, check to see if search "
                            "space is appropriately bounded. If they are constant, and you are fitting to"
                            "data, make sure experimental data and output of your composition are similar.",
                            style="bold red",
                        )

                    progress.update(opt_task, advance=1)

                # We need to hook into Optuna's random number generator here so that we can allow PsyNeuLink's RNS to
                # determine the seed for Optuna's RNG. Pretty hacky unfortunately.
                opt_func._rng = np.random.RandomState(self.owner.initial_seed)

                # Turn off optuna logging except for errors or warnings, it doesn't work well with our PNL progress bar
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                study = optuna.create_study(
                    sampler=self.method, direction=self.direction
                )
                study.optimize(
                    objfunc_wrapper_wrapper,
                    n_trials=max_iterations,
                    callbacks=[progress_callback],
                )

            # Bind the fitted parameters to their names
            fitted_params = dict(
                zip(list(self.fit_param_names), study.best_params.values())
            )

        # Save all the results
        output_dict = {
            "fitted_params": fitted_params,
            "optimal_value": study.best_value,
        }

        return output_dict

    @property
    def fit_param_names(self) -> List[str]:
        """Get a unique name for each parameter in the fit."""
        if self.owner is not None:
            return [
                cs.efferents[0].receiver.name
                for i, cs in enumerate(self.owner.control_signals)
                if i != self.randomization_dimension
            ]
        else:
            return None

    @property
    def fit_param_bounds(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Get the allocation samples for just the fitting parameters. Whatever they are, turn them into upper and lower
        bounds, with a step size as well.

        Returns:
            A dict mapping parameter names to (lower, upper) bounds.
            A dict mapping parameter names to step sizes.
        """

        if self.owner is not None:
            acs = [
                cs.specification
                for i, cs in enumerate(self._full_search_space)
                if i != self.randomization_dimension
            ]

            bounds = [(float(min(s)), float(max(s))) for s in acs]

            # Get the step size for each parameter.
            steps = [np.unique(np.diff(s).round(decimals=5)) for s in acs]

            # We also check if step size is constant, if not we raise an error
            for s in steps:
                if len(s) > 1:
                    raise ValueError("Step size for each parameter must be constant")

            steps = [float(s[0]) for s in steps]

            return dict(
                zip(
                    self.fit_param_names,
                    ((l, u, s) for (l, u), s in zip(bounds, steps)),
                )
            )
        else:
            return None

    @handle_external_context(fallback_most_recent=True)
    def log_likelihood(self, *args, context=None):
        """
        Compute the log-likelihood of the data given the specified parameters of the model. This function will raise
        aa exception if the function has not been assigned as the function of and OptimizationControlMechanism. An
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
            raise ValueError(
                "Cannot compute a log-likelihood without being assigned as the function of an "
                "OptimizationControlMechanism. See the documentation for the "
                "ParameterEstimationControlMechanism for more information."
            )

        # Make sure we have instantiated the log-likelihood function.
        if self._ll_func is None:
            self._ll_func = self._make_objective_func(context=context)

        context.execution_phase = ContextFlags.PROCESSING
        ll, sim_data = self._ll_func(*args)
        context.remove_flag(ContextFlags.PROCESSING)

        return ll, sim_data
