from fastkde import fastKDE
from scipy.interpolate import interpn
from scipy.optimize import differential_evolution

from psyneulink.core.globals.context import Context
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.scheduling.condition import AtTrialStart
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.llvm import ExecutionMode
from psyneulink.core.components.functions.nonstateful.optimizationfunctions import OptimizationFunction


import typing
import time
import numpy as np
import collections
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
            if len(dsub) < 100:
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

        return kdes_eval

    else:
        return kdes


def make_likelihood_function(
    composition: "psyneulink.core.composition.Composition",
    fit_params: typing.List[Parameter],
    inputs: typing.Union[np.ndarray, typing.List],
    data_to_fit: typing.Union[np.ndarray, pd.DataFrame],
    categorical_dims: typing.Union[np.ndarray, None] = None,
    num_sims_per_trial: int = 1000,
    fixed_params: typing.Optional[typing.Dict[Parameter, typing.Any]] = None,
    combine_trials=True,
):
    """
    Construct and return a Python function that returns the log likelihood of experimental
    data being generated by a PsyNeuLink composition. The likelihood function is parameterized
    by fit_params

    Parameters
    ----------
    composition: A PsyNeuLink composition. This function returns a function that runs
        many simulations of this composition to generate a kernel density estimate of the likelihood
        of a dataset under different parameter settings. The output (composition.results) should match
        the columns of data_to_fit exactly.
    fit_params: A list of PsyNeuLink parameters to fit. Each on of these parameters will map to
        an argument of the likelihood function that is returned. Values passed via these arguments
        will be assigned to the composition before simulation.
    fixed_params: A dict of PsyNeuLink parameters and their corresponding fixed values. These
        parameters will be applied to the composition before simulation but will not be exposed as
        arguments to the likelihood function.
    inputs: A set of inputs to pass to the composition on each simulation of the likelihood. These
        inputs are passed directly to the composition run method as is.
    categorical_dims: If data_to_fit is a pandas DataFrame, this parameter is ignored and any Categorical column
        is considered categorical. If data_to_fit is a ndarray, categorical_dims should be a 1D logical array, where
        each element corresponds to a column of data_to_fit. If the element is True, the dimension should be considered
        categorical, if False, it should be treated as continuous. Categorical is suitable for outputs that will only
        take on a handful of unique values, such as the decision value of a DDM or LCA.
    data_to_fit: Either 2D numpy array or Pandas DataFrame, where the rows are trials and the columns are
        in the same format as outputs of the PsyNeuLink composition. This data essentially describes at
        what values the KDE of the likelihood should be evaluated.
    num_sims_per_trial: The number of simulations per trial to run to construct the KDE likelihood.
    combine_trials: Whether we can combine simulations across trials for one estimate of the likelihood.
        This can dramatically increase the speed of fitting by allowing a smaller number
        of total simulations to run per trial. However, this cannot be done if the likelihood will change across
        trials.

    Returns
    -------
    A tuple containing:
        - the likelihood function
        - A dict which maps elements of fit_params to their keyword argument names in the likelihood function.
    """

    # We need to parse the inputs like composition does to get the number of trials
    _, num_inputs_sets = composition._parse_run_inputs(inputs)
    num_trials = num_inputs_sets

    # Check to see if any parameters (fittable or fixed) have the same name,
    # this will cause problems, if so, we need to create a unique numbering.
    # If we have multiple parameters with this name already, assign it the
    # next available number
    all_param_names = [p.name for p in fit_params]
    dupe_counts = [
        all_param_names[:i].count(all_param_names[i]) + 1
        for i in range(len(all_param_names))
    ]
    all_param_names = [
        name if count == 1 else f"{name}_{count}"
        for name, count in zip(all_param_names, dupe_counts)
    ]
    param_name_map = dict(zip(fit_params, all_param_names))

    if type(data_to_fit) == np.ndarray:
        if data_to_fit.ndim != 2:
            raise ValueError("data_to_fit must be a 2D")

        # Assume all dimensions are continuous if this wasn't specified by the user and their data is a numpy array
        if categorical_dims is None:
            categorical_dims = [False for i in range(data_to_fit.shape[1])]

    elif type(data_to_fit) == pd.DataFrame:
        categorical_dims = [
            data_to_fit[c].dtype.name == "category" for c in data_to_fit.columns
        ]

    else:
        raise ValueError("data_to_fit must be a 2D numpy array or a Pandas DataFrame")

    def log_likelihood(**kwargs):
        context = Context()

        # Assign parameter values to composition, eventually this should be done
        # via the OptimizationControlMechanism and its control allocations stuff.
        # However, currently the Python code for that creates a fresh context
        # per simulation with considerable overhead. Instead, we create one context
        # and use reset_stateful_functions_when=AtTrialStart(). Additionally, all simulations
        # are computed in one call to compiled run via setting num_trials to
        # num_simulations * len(input).
        for param in fit_params:
            try:
                value = kwargs[param_name_map[param]]
            except KeyError:
                raise ValueError(
                    f"No argument {param_name_map[param]} passed to likelihood function for Parameter: \n{param}"
                )

            # FIXME: DDM requires that starting_value is an array in compiled mode for some reason. Ugly hack!
            if param.name == "starting_value" and type(value) != np.ndarray:
                value = np.array([value])

            # Apply the parameter value under a fresh context
            param.set(value, context)

        # Also apply the fixed parameters
        if fixed_params is not None:
            for param, value in fixed_params.items():
                param.set(value, context)

        # FIXME: Multiple calls to run retain results, we need to clear them. Is this OK?
        composition.results.clear()

        # Run the composition for all simulations, this corresponds to looping over the input
        # num_simulations times.
        composition.run(
            inputs=inputs,
            num_trials=num_sims_per_trial * num_trials,
            execution_mode=ExecutionMode.LLVMRun,
            context=context,
        )

        results = np.squeeze(np.array(composition.results))

        # Split results into (trials, simulations, data)
        sim_data = np.array(np.vsplit(results, num_trials))

        # Compute the likelihood given the data
        like = simulation_likelihood(
            sim_data=sim_data,
            exp_data=data_to_fit.to_numpy().astype(float),
            categorical_dims=categorical_dims,
            combine_trials=combine_trials,
        )

        # Make 0 densities very small so log doesn't explode
        like[like == 0.0] = 1.0e-10

        return np.sum(np.log(like))

    return log_likelihood, param_name_map


class MaxLikelihoodEstimatorFunction(OptimizationFunction):
    pass

class MaxLikelihoodEstimator:
    """
    Implements a maximum likelihood estimation given a likelihood function
    """

    def __init__(
        self,
        log_likelihood_function: typing.Callable,
        fit_params_bounds: typing.Dict[str, typing.Tuple],
    ):
        self.log_likelihood_function = log_likelihood_function
        self.fit_params_bounds = fit_params_bounds

        # Setup a named tuple to store records for each iteration if the user requests it
        self.IterRecord = collections.namedtuple(
            "IterRecord",
            f"{' '.join(self.fit_params_bounds.keys())} neg_log_likelihood likelihood_eval_time",
        )

    def fit(self, display_iter: bool = False, save_iterations: bool = False):

        bounds = list(self.fit_params_bounds.values())

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
                    params = dict(zip(self.fit_params_bounds.keys(), x))
                    t0 = time.time()
                    p = -self.log_likelihood_function(**params)
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

                    # Are we saving each iteration
                    if save_iterations:
                        iterations.append(
                            self.IterRecord(
                                **params,
                                neg_log_likelihood=p,
                                likelihood_eval_time=elapsed,
                            )
                        )

                    return p

                def progress_callback(x, convergence):
                    progress.start_task(opt_task)
                    params = dict(zip(self.fit_params_bounds.keys(), x))
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
            fitted_params = dict(zip(list(self.fit_params_bounds.keys()), r.x))

        # Save all the results
        output_dict = {
            "fitted_params": fitted_params,
            "neg-log-likelihood": r.fun,
        }

        # Return a record for each iteration if we were supposed to.
        if save_iterations:
            output_dict["iterations"] = pd.DataFrame.from_records(
                iterations, columns=self.IterRecord._fields
            )

        return output_dict
