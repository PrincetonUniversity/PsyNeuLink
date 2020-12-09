import typing
import time
import numpy as np

from psyneulink.core.globals.context import Context
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.scheduling.condition import AtTrialStart
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.components.functions.optimizationfunctions import OBJECTIVE_FUNCTION, SEARCH_SPACE, \
    OptimizationFunction

from fastkde import fastKDE
from scipy.interpolate import interpn
from scipy.optimize import differential_evolution


def simulation_likelihood(sim_data,
                          exp_data=None,
                          categorical_dims=None,
                          combine_trials=False):
    """
    Compute the likelihood of a simulation dataset (or the parameters that generated it) conditional
    on a set of experimental data. This function essentially just computes the kernel density estimate (KDE)
    of the simulation data at the experimental data points.
    If no experimental data is provided just return the KDE evaluated at default points provided by the fastkde
    library.

    Reference:

    Steven Miletić, Brandon M. Turner, Birte U. Forstmann, Leendert van Maanen,
    Parameter recovery for the Leaky Competing Accumulator model,
    Journal of Mathematical Psychology,
    Volume 76, Part A,
    2017,
    Pages 25-50,
    ISSN 0022-2496,
    https://doi.org/10.1016/j.jmp.2016.12.001.
    (http://www.sciencedirect.com/science/article/pii/S0022249616301663)

    O’Brien, T. A., Kashinath, K., Cavanaugh, N. R., Collins, W. D. & O’Brien, J. P.
    A fast and objective multidimensional kernel density estimation method: fastKDE.
    Comput. Stat. Data Anal. 101, 148–160 (2016).
    <http://dx.doi.org/10.1016/j.csda.2016.02.014>__

    O’Brien, T. A., Collins, W. D., Rauscher, S. A. & Ringler, T. D.
    Reducing the computational cost of the ECF using a nuFFT: A fast and objective probability density estimation method.
    Comput. Stat. Data Anal. 79, 222–234 (2014). <http://dx.doi.org/10.1016/j.csda.2014.06.002>__

    Parameters
    ----------
    sim_data: This must be a 3D numpy array where the first dimension is the trial, the
    second dimension is the simulation number, and the final dimension is data points.

    exp_data: This must be a numpy array with identical format as the simulation data, with the exception
    that there is no simulation dimension.

    categorical_dims: a list of indices that indicate categorical dimensions of a data point.

    combine_trials: Combine data across all trials into a single likelihood estimate, this assumes
    that the parameters of the simulations are identical across trials.

    Returns
    -------
    The pdf of simulation data (or in other words, the generating parameters) conditioned on the
    experimental data.

    """

    if combine_trials and sim_data.shape[0] > 1:
        sim_data = np.vstack(sim_data)[None, :, :]

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

            # Do KDE
            fKDE = fastKDE.fastKDE(dsub, doSaveMarginals=False)
            pdf = fKDE.pdf
            axes = fKDE.axes

            # Scale the pdf by the fraction of simulations that fall within this category
            pdf = pdf * (len(dsub)/len(s))

            # Save the KDE values and axes for this category
            dens_u[category] = (pdf, axes)

        kdes.append(dens_u)

    # If we are passed experimental data, evaluate the KDE at the experimental data points
    if exp_data is not None:

        kdes_eval = np.zeros((len(exp_data),))
        for trial in range(len(exp_data)):

            # Extract the categorical values for this experimental trial
            exp_trial_cat = exp_data[trial, categorical_dims]

            if len(exp_trial_cat) == 1:
                exp_trial_cat = exp_trial_cat[0]

            # Get the right KDE for this trial, if all simulation trials have been combined
            # use that KDE for all trials of experimental data.
            if len(kdes) == 1:
                kde, axes = kdes[0][exp_trial_cat]
            else:
                kde, axes = kdes[trial][exp_trial_cat]

            # Linear interpolation using the grid we computed the KDE
            # on.
            if kde is not None:
                kdes_eval[trial] = interpn(axes, kde, exp_data[trial, ~categorical_dims],
                                           method='linear', bounds_error=False, fill_value=1e-10)
            else:
                kdes_eval[trial] = 1e-10

        return kdes_eval

    else:
        return kdes


def make_likelihood_function(composition: 'psyneulink.core.composition.Composition',
                             fit_params: typing.List[Parameter],
                             inputs: typing.Union[np.ndarray, typing.List],
                             categorical_dims: np.ndarray,
                             data_to_fit: np.ndarray,
                             num_simulations: int = 1000,
                             fixed_params: typing.Optional[typing.Dict[Parameter, typing.Any]] = None,
                             combine_trials=True):
    """
    Construct and return a Python function that returns the log likelihood of experimental
    data being generated by a PsyNeuLink composition. The likelihood function is parameterized
    by fit_params

    Parameters
    ----------
    composition: A PsyNeuLink composition. This function returns a function that runs
    many simulations of this composition to generate a kernel density estimate of the likelihood
    of a dataset under different parameter settings. The output (composition.results) should match
    the format in data_to_fit.
    fit_params: A list of PsyNeuLink parameters to fit. Each on of these parameters will map to
    an argument of the likelihood function that is returned. Values passed via these arguments
    will be assigned to the composition before simulation.
    fixed_params: A dict of PsyNeuLink parameters and their corresponding fixed values. These
    parameters will be applied to the composition before simulation but will not be exposed as
    arguments to the likelihood function.
    inputs: A set of inputs to pass to the composition on each simulation of the likelihood. These
    inputs are passed directly to the composition run method as is.
    categorical_dims: A 1D logical array, where each dimension corresponds to an output dimension
    of the PsyNeuLink composition. If True, the dimension should be considered categorical, if False,
    it should be treated as continuous. Categorical is suitable for outputs that will only take on
    a handful of unique values, such as the decision value of a DDM.
    data_to_fit: A 2D numpy array where the first dimension is the trial number and the columns are
    in the same format as outputs of the PsyNeuLink composition. This data essentially describes at
    what values the KDE of the likelihood should be evaluated.
    num_simulations: The number of simulations (per trial) to run to construct the KDE likelihood.
    combine_trials: Whether we can combine simulations across trials for one estimate of the likelihood.
    This can dramatically increase the speed of the likelihood function by allowing a smaller number
    of total simulations to run per trial. However, this cannot be done if the trial by trial state
    of the composition is maintained.

    Returns
    -------
    A tuple containing:
        - the likelihood function
        - A dict which maps elements of fit_params to their string function argument names.
    """

    # Get the number of trials in the input data
    num_trials = len(next(iter(inputs)))

    # Check to see if any parameters (fittable or fixed) have the same name,
    # this will cause problems, if so, we need to create a unique numbering.
    # If we have multiple parameters with this name already, assign it the
    # next available number
    all_param_names = [p.name for p in fit_params]
    dupe_counts = [all_param_names[:i].count(all_param_names[i])+1 for i in range(len(all_param_names))]
    all_param_names = [name if count == 1 else f"{name}_{count}" for name, count in zip(all_param_names, dupe_counts)]
    param_name_map = dict(zip(fit_params, all_param_names))

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
                raise ValueError(f"No argument {param_name_map[param]} passed to likelihood function for Parameter: \n{param}")

            # Apply the parameter value under a fresh context
            param.set(value, context)

        # Also apply the fixed parameters
        if fixed_params:
            for param, value in fixed_params:
                param.set(value, context)

        # Run the composition for all simulations, this corresponds to looping over the input
        # num_simulations times.
        composition.run(inputs=inputs,
                        reset_stateful_functions_when=AtTrialStart(),
                        num_trials=num_simulations * num_trials,
                        bin_execute=True,
                        context=context)

        results = np.squeeze(np.array(composition.results))

        # Split results into (trials, simulations, data)
        sim_data = np.array(np.vsplit(results, len(input)))

        # Compute the likelihood given the data
        like = simulation_likelihood(sim_data=sim_data, exp_data=data_to_fit,
                                     categorical_dims=categorical_dims,
                                     combine_trials=combine_trials)

        # Make 0 densities very small so log doesn't explode
        like[like == 0.0] = 1.0e-10

        return np.sum(np.log(like))

    return log_likelihood, param_name_map


class MaxLikelihoodEstimator:
    """
    Implements a maximum likelihood estimation given a likelihood function
    """

    def __init__(self,
                 log_likelihood_function: typing.Callable,
                 fit_params_bounds: typing.Dict[str, typing.Tuple],
                 fixed_params: typing.Optional[typing.Dict[Parameter, typing.Any]]):
        self.log_likelihood_function = log_likelihood_function
        self.fit_params_bounds = fit_params_bounds

        if fixed_params is not None:
            self.fixed_params = fixed_params
        else:
            self.fixed_params = {}

    def fit(self):

        bounds = list(self.fit_params_bounds.values())

        # Check if any of are fixed params are in parameters to fit, this is a mistake
        for fixed_p, val in self.fixed_params.items():
            if fixed_p in self.fit_params_bounds:
                raise ValueError(f"Fixed parameter ({fixed_p}) is also present in the parameters to fit.")

        def print_param_vec(p, end="\n"):
            print(', '.join(f'{name}={value:.5f}' for name, value in p.items()), end=end)

        # Create a wrapper function for the objective.
        def neg_log_like(x):
            params = dict(zip(self.fit_params_bounds.keys(), x))
            # print_param_vec(params, end="")

            p = -self.log_likelihood_function(**self.fixed_params, **params)
            # print(f" neg_log_like={p:.5f}")
            return p

        t0 = time.time()

        def print_callback(x, convergence):
            global t0
            t1 = time.time()
            params = dict(zip(self.fit_params_bounds.keys(), x))
            print_param_vec(params, end="")
            print(f", convergence={convergence:.5f}, iter_time={t1 - t0} secs")
            t0 = t1

        t0 = time.time()

        # with Progress() as progress:
        #     opt_task = progress.add_task("Maximum likelihood optimization ...", total=100, start=False)
        #
        #     def progress_callback(x, convergence):
        #         convergence = 100.0 * convergence
        #         progress.update(opt_task, completed=convergence)

        r = differential_evolution(neg_log_like, bounds, callback=print_callback, maxiter=500, workers=6)

        print(f"Search Time: {(time.time() - t0) / 60.0} minutes")

        # Bind the fitted parameters to their names
        fitted_params = dict(zip(list(self.fit_params_bounds.keys()), r.x))
        print(f"Fitted Params: \n\t{fitted_params}\nLikelihood: {r.fun}")

        # Save all the results
        output_dict = {
            'fixed_params': self.fixed_params,
            'fitted_params': fitted_params,
            'likelihood': r.fun,
        }

        return output_dict

