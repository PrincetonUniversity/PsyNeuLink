import copy

import optuna.samplers
from fastkde import fastKDE
from scipy.interpolate import interpn
from scipy.optimize import differential_evolution
from beartype import beartype

from psyneulink.core.globals import SampleIterator
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.components.functions.nonstateful.optimizationfunctions import (
    OptimizationFunction,
    OptimizationFunctionError,
    SEARCH_SPACE,
)
from psyneulink.core.globals.parameters import SharedParameter, check_user_specified
from psyneulink.core.globals.utilities import try_extract_0d_array_item

from psyneulink._typing import (
    Dict,
    Tuple,
    Callable,
    List,
    Optional,
    Union,
    Type,
    Literal,
    Mapping,
)


import time
import numpy as np

from rich.progress import Progress, BarColumn, TimeRemainingColumn

import warnings
import logging

from rich.markup import escape

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
        f"[dodger_blue1]{escape(name.replace('PARAMETER_CIM_', ''))}[/dodger_blue1]=[spring_green1]{value:.5f}[/spring_green1]"
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


def _categorical_dims_to_mask(categorical_dims, num_dims):
    if categorical_dims is None:
        return np.zeros(num_dims, dtype=bool)

    categorical_dims = np.array(categorical_dims)

    if categorical_dims.dtype == bool:
        if len(categorical_dims) != num_dims:
            raise ValueError(
                f"categorical_dims mask must have length {num_dims}; got {len(categorical_dims)}."
            )
        return categorical_dims

    mask = np.zeros(num_dims, dtype=bool)
    if categorical_dims.size:
        mask[categorical_dims.astype(int)] = True
    return mask


def _histogram_categories(cat_sim_data):
    if cat_sim_data.shape[-1] == 0:
        return [None]

    if cat_sim_data.shape[-1] == 1:
        return np.unique(cat_sim_data[..., 0])

    categories = np.unique(
        cat_sim_data.reshape((-1, cat_sim_data.shape[-1])),
        axis=0,
    )
    return [tuple(category) for category in categories]


def _category_mask(cat_values, category):
    if cat_values.shape[-1] == 0:
        return np.ones(cat_values.shape[0], dtype=bool)

    if cat_values.shape[-1] == 1:
        return cat_values[:, 0] == category

    return np.all(cat_values == np.array(category), axis=1)


def _experimental_category(exp_trial_cat):
    if len(exp_trial_cat) == 0:
        return None
    if len(exp_trial_cat) == 1:
        return exp_trial_cat[0]
    return tuple(exp_trial_cat)


def _histogram_category_key(category):
    if category is None:
        return None

    if isinstance(category, tuple):
        return category

    category = np.asarray(category)
    if category.ndim == 0:
        return category.item()

    return tuple(category.tolist())


def _histogram_category_ids(cat_sim_data):
    if cat_sim_data.shape[-1] == 0:
        category_ids = np.zeros(cat_sim_data.shape[:2], dtype=int)
        return [None], {None: 0}, category_ids

    if cat_sim_data.shape[-1] == 1:
        categories, inverse = np.unique(cat_sim_data[..., 0].reshape(-1), return_inverse=True)
        categories = [_histogram_category_key(category) for category in categories]
    else:
        categories_arr, inverse = np.unique(
            cat_sim_data.reshape((-1, cat_sim_data.shape[-1])),
            axis=0,
            return_inverse=True,
        )
        categories = [_histogram_category_key(category) for category in categories_arr]

    category_ids = inverse.reshape(cat_sim_data.shape[:2])
    return categories, {category: i for i, category in enumerate(categories)}, category_ids


def _regular_histogram_edges(dsub, bins, range_pad):
    if dsub.ndim == 1:
        dsub = dsub[:, None]

    num_dims = dsub.shape[1]
    bins = [int(bins)] * num_dims if np.isscalar(bins) else [int(b) for b in bins]
    if len(bins) != num_dims:
        raise ValueError(f"bins must be scalar or have length {num_dims}; got {bins}.")

    ranges = []
    for dim in range(num_dims):
        low = float(np.min(dsub[:, dim]))
        high = float(np.max(dsub[:, dim]))
        span = high - low
        if span <= 0:
            return None, None
        pad = max(float(range_pad), span * float(range_pad))
        ranges.append((low - pad, high + pad))

    edges = [np.linspace(low, high, bins[dim] + 1) for dim, (low, high) in enumerate(ranges)]
    return bins, edges


def _histogram_counts_numpy(dsub, bins, edges):
    ranges = [(edge[0], edge[-1]) for edge in edges]
    counts, edges = np.histogramdd(dsub, bins=bins, range=ranges)
    return counts, edges


def _histogram_counts_boost(dsub, bins, edges, threads):
    try:
        import boost_histogram as bh
    except ImportError:
        return _histogram_counts_numpy(dsub, bins, edges), "numpy"

    axes = [
        bh.axis.Regular(bins[dim], edges[dim][0], edges[dim][-1], underflow=False, overflow=False)
        for dim in range(len(bins))
    ]
    hist = bh.Histogram(*axes, storage=bh.storage.Double())
    hist.fill(*(dsub[:, dim] for dim in range(dsub.shape[1])), threads=int(threads))
    return (hist.values(), [np.asarray(axis.edges) for axis in hist.axes]), "boost"


def _histogram_counts(dsub, bins, edges, backend, threads):
    if backend not in {"auto", "numpy", "boost"}:
        raise ValueError(
            f"Unknown histogram_backend {backend!r}; expected 'auto', 'numpy', or 'boost'."
        )

    if backend in {"auto", "boost"}:
        return _histogram_counts_boost(dsub, bins, edges, threads)[0]

    return _histogram_counts_numpy(dsub, bins, edges)


def _histogram_counts_vectorized_numpy(
    trial_ids,
    category_ids,
    con_data,
    num_trials,
    num_categories,
    bins,
    edges,
):
    samples = [trial_ids, category_ids]
    samples.extend(con_data[:, dim] for dim in range(con_data.shape[1]))
    samples = np.column_stack(samples)

    hist_bins = [num_trials, num_categories] + list(bins)
    hist_ranges = [(0, num_trials), (0, num_categories)]
    hist_ranges.extend((edge[0], edge[-1]) for edge in edges)
    counts, _ = np.histogramdd(samples, bins=hist_bins, range=hist_ranges)
    return counts


def _histogram_counts_vectorized_boost(
    trial_ids,
    category_ids,
    con_data,
    num_trials,
    num_categories,
    bins,
    edges,
    threads,
):
    try:
        import boost_histogram as bh
    except ImportError:
        return _histogram_counts_vectorized_numpy(
            trial_ids,
            category_ids,
            con_data,
            num_trials,
            num_categories,
            bins,
            edges,
        ), "numpy"

    axes = [
        bh.axis.Integer(0, num_trials, underflow=False, overflow=False),
        bh.axis.Integer(0, num_categories, underflow=False, overflow=False),
    ]
    axes.extend(
        bh.axis.Regular(bins[dim], edges[dim][0], edges[dim][-1], underflow=False, overflow=False)
        for dim in range(len(bins))
    )
    hist = bh.Histogram(*axes, storage=bh.storage.Double())
    hist.fill(
        trial_ids,
        category_ids,
        *(con_data[:, dim] for dim in range(con_data.shape[1])),
        threads=int(threads),
    )
    return hist.values(), "boost"


def _histogram_counts_vectorized(
    trial_ids,
    category_ids,
    con_data,
    num_trials,
    num_categories,
    bins,
    edges,
    backend,
    threads,
):
    if backend not in {"auto", "numpy", "boost"}:
        raise ValueError(
            f"Unknown histogram_backend {backend!r}; expected 'auto', 'numpy', or 'boost'."
        )

    if backend in {"auto", "boost"}:
        return _histogram_counts_vectorized_boost(
            trial_ids,
            category_ids,
            con_data,
            num_trials,
            num_categories,
            bins,
            edges,
            threads,
        )[0]

    return _histogram_counts_vectorized_numpy(
        trial_ids,
        category_ids,
        con_data,
        num_trials,
        num_categories,
        bins,
        edges,
    )


def _simulation_likelihood_histogram_vectorized(
    sim_data,
    exp_data=None,
    categorical_dims=None,
    combine_trials=False,
    bins=32,
    pseudocount=0.0,
    zero_prob=1e-10,
    range_pad=1e-9,
    histogram_backend="auto",
    threads=1,
):
    if exp_data is None:
        raise ValueError("The histogram estimator requires exp_data.")

    if sim_data.ndim == 2:
        sim_data = sim_data[None, :, :]

    if combine_trials and sim_data.shape[0] > 1:
        sim_data = np.vstack(sim_data)[None, :, :]

    categorical_dims = _categorical_dims_to_mask(categorical_dims, sim_data.shape[-1])
    con_sim_data = sim_data[:, :, ~categorical_dims]
    cat_sim_data = sim_data[:, :, categorical_dims]
    categories, category_to_index, category_ids = _histogram_category_ids(cat_sim_data)

    num_trials, num_estimates, num_con_dims = con_sim_data.shape
    kdes_eval = np.zeros((len(exp_data),))
    con_flat = con_sim_data.reshape((-1, num_con_dims))

    if num_con_dims:
        bins_for_dims, edges = _regular_histogram_edges(con_flat, bins, range_pad)
        if edges is None:
            warnings.warn(
                BadLikelihoodWarning(
                    "Could not perform histogram likelihood estimate. Range of simulation data was 0 "
                    "for at least one dimension."
                )
            )
            kdes_eval[:] = zero_prob
            return kdes_eval
    else:
        bins_for_dims = []
        edges = []

    trial_ids = np.repeat(np.arange(num_trials), num_estimates)
    counts = _histogram_counts_vectorized(
        trial_ids,
        category_ids.reshape(-1),
        con_flat,
        num_trials,
        len(categories),
        bins_for_dims,
        edges,
        histogram_backend,
        threads,
    )

    for trial in range(len(exp_data)):
        hist_trial = 0 if num_trials == 1 else trial
        exp_trial_cat = _histogram_category_key(_experimental_category(exp_data[trial, categorical_dims]))
        exp_trial_con = exp_data[trial, ~categorical_dims]
        category_index = category_to_index.get(exp_trial_cat)

        if category_index is None or hist_trial >= num_trials:
            kdes_eval[trial] = zero_prob
            continue

        category_counts = counts[hist_trial, category_index]
        category_count = np.sum(category_counts)
        category_prob = category_count / num_estimates

        if num_con_dims == 0:
            kdes_eval[trial] = max(zero_prob, category_prob)
            continue

        idx = []
        widths = []
        in_range = True
        for dim, edge in enumerate(edges):
            bin_idx = np.searchsorted(edge, exp_trial_con[dim], side="right") - 1
            if bin_idx == len(edge) - 1 and exp_trial_con[dim] == edge[-1]:
                bin_idx -= 1
            if bin_idx < 0 or bin_idx >= len(edge) - 1:
                in_range = False
                break
            idx.append(bin_idx)
            widths.append(edge[bin_idx + 1] - edge[bin_idx])

        if not in_range:
            kdes_eval[trial] = zero_prob
            continue

        num_cells = category_counts.size
        denom = (category_count + pseudocount * num_cells) * float(np.prod(widths))
        if denom <= 0:
            kdes_eval[trial] = zero_prob
            continue

        density = (category_counts[tuple(idx)] + pseudocount) / denom
        kdes_eval[trial] = max(zero_prob, category_prob * density)

    if all(kdes_eval == zero_prob):
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


def _simulation_likelihood_histogram(
    sim_data,
    exp_data=None,
    categorical_dims=None,
    combine_trials=False,
    bins=32,
    pseudocount=0.0,
    zero_prob=1e-10,
    range_pad=1e-9,
    histogram_backend="auto",
    threads=1,
    vectorized=False,
):
    if exp_data is None:
        raise ValueError("The histogram estimator requires exp_data.")

    if vectorized:
        return _simulation_likelihood_histogram_vectorized(
            sim_data=sim_data,
            exp_data=exp_data,
            categorical_dims=categorical_dims,
            combine_trials=combine_trials,
            bins=bins,
            pseudocount=pseudocount,
            zero_prob=zero_prob,
            range_pad=range_pad,
            histogram_backend=histogram_backend,
            threads=threads,
        )

    if sim_data.ndim == 2:
        sim_data = sim_data[None, :, :]

    if combine_trials and sim_data.shape[0] > 1:
        sim_data = np.vstack(sim_data)[None, :, :]

    categorical_dims = _categorical_dims_to_mask(categorical_dims, sim_data.shape[-1])
    con_sim_data = sim_data[:, :, ~categorical_dims]
    cat_sim_data = sim_data[:, :, categorical_dims]
    categories = _histogram_categories(cat_sim_data)

    kdes_eval = np.zeros((len(exp_data),))
    histogram_cache = []
    for trial in range(len(con_sim_data)):
        s = con_sim_data[trial]
        dens_u = {}
        for category in categories:
            category_member = _category_mask(cat_sim_data[trial], category)
            dsub = s[category_member]

            if len(dsub) < 10:
                dens_u[category] = (None, None, 0.0)
                continue

            dsub = np.atleast_2d(dsub)
            if dsub.shape[0] == 1 and dsub.shape[1] != s.shape[1]:
                dsub = dsub.T

            if dsub.shape[1] == 0:
                dens_u[category] = (None, None, len(dsub) / len(s))
                continue

            bins_for_dims, edges = _regular_histogram_edges(dsub, bins, range_pad)
            if edges is None:
                dens_u[category] = (None, None, 0.0)
                warnings.warn(
                    BadLikelihoodWarning(
                        "Could not perform histogram likelihood estimate. Range of simulation data was 0 "
                        "for at least one dimension."
                    )
                )
                continue

            counts, edges = _histogram_counts(
                dsub,
                bins_for_dims,
                edges,
                histogram_backend,
                threads,
            )
            dens_u[category] = (counts, edges, len(dsub) / len(s))
        histogram_cache.append(dens_u)

    for trial in range(len(exp_data)):
        exp_trial_cat = _experimental_category(exp_data[trial, categorical_dims])
        exp_trial_con = exp_data[trial, ~categorical_dims]

        if len(histogram_cache) == 1:
            counts, edges, category_prob = histogram_cache[0].get(exp_trial_cat, (None, None, 0.0))
        else:
            counts, edges, category_prob = histogram_cache[trial].get(exp_trial_cat, (None, None, 0.0))

        if counts is None and edges is None:
            kdes_eval[trial] = max(zero_prob, category_prob)
            continue

        idx = []
        widths = []
        in_range = True
        for dim, edge in enumerate(edges):
            bin_idx = np.searchsorted(edge, exp_trial_con[dim], side="right") - 1
            if bin_idx == len(edge) - 1 and exp_trial_con[dim] == edge[-1]:
                bin_idx -= 1
            if bin_idx < 0 or bin_idx >= len(edge) - 1:
                in_range = False
                break
            idx.append(bin_idx)
            widths.append(edge[bin_idx + 1] - edge[bin_idx])

        if not in_range:
            kdes_eval[trial] = zero_prob
            continue

        num_cells = counts.size
        bin_volume = float(np.prod(widths))
        density = (
            (counts[tuple(idx)] + pseudocount)
            / ((np.sum(counts) + pseudocount * num_cells) * bin_volume)
        )
        kdes_eval[trial] = max(zero_prob, category_prob * density)

    if all(kdes_eval == zero_prob):
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


def simulation_likelihood(
    sim_data,
    exp_data=None,
    categorical_dims=None,
    combine_trials=False,
    estimator="kde",
    estimator_kwargs=None,
    histogram_backend="auto",
):
    """
    Compute the likelihood of a simulation dataset (or the parameters that generated it) conditional
    on a set of experimental data. By default, this function computes the kernel density estimate (KDE)
    of the simulation data at the experimental data points. It can also compute an approximate per-trial
    histogram likelihood. If no experimental data is provided for KDE, return the KDE evaluated at default
    points provided by the fastkde library.

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
    sim_data: Data collected over many simulations. This must be either a 2d or 3d numpy array.
        If 2D, the first dimension is the simulation number and the second dimension is data points. That is,
        each row is a simulation. If 3d, the first dimension is the trial, the second dimension is the
        simulation number, and the final dimension is data points.

    exp_data: This must be a numpy array with identical format as the simulation data, with the exception
        that there is no simulation dimension.

    categorical_dims: a list of indices that indicate categorical dimensions of a data point. Length must be
        the same length as last dimension of sim_data and exp_data.

    combine_trials: Combine data across all trials into a single likelihood estimate, this assumes
        that the parameters of the simulations are identical across trials.

    estimator: Estimator used to compute likelihoods. Supported values are "kde" and "histogram".

    estimator_kwargs: Optional keyword arguments passed to the selected estimator.

    histogram_backend: Histogram backend used when estimator is "histogram". Supported values are
        "auto", "numpy", and "boost".

    For the "histogram" estimator, passing ``vectorized=True`` in ``estimator_kwargs`` uses shared continuous
    bin edges plus explicit trial and category axes to fill all simulation data in one histogram call.

    Returns
    -------
    The pdf of simulation data (or in other words, the generating parameters) conditioned on the
    experimental data.

    """

    estimator = estimator.lower()
    estimator_kwargs = {} if estimator_kwargs is None else dict(estimator_kwargs)

    if estimator == "histogram":
        estimator_kwargs.setdefault("histogram_backend", histogram_backend)
        return _simulation_likelihood_histogram(
            sim_data=sim_data,
            exp_data=exp_data,
            categorical_dims=categorical_dims,
            combine_trials=combine_trials,
            **estimator_kwargs,
        )

    if estimator != "kde":
        raise ValueError(f"Unknown likelihood estimator {estimator!r}; expected 'kde' or 'histogram'.")

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
            fKDE = fastKDE.fastKDE(dsub, do_save_marginals=False)

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

        # Check to see if all of the trials have non-zero likelihood, if so, something is probably wrong
        # and we should warn the user.
        if all(kdes_eval == ZERO_PROB):
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
            - Type[optuna.samplers.BaseSampler]: Pass a class of type optuna.samplers.BaseSampler to use optuna
            for optimization. In this case, the random seed used for the sampler will be the same as the seed used
            as the intial_seed passed to PEC at contruction. Additional desired keyword arguments can be passed to the
            sampler via the optuna_kwargs argument.

    optuna_kwargs :
        A dictionary of keyword arguments to pass to the optuna sampler. This is only used if method is an class of
        type optuna.samplers.BaseSampler. Note: this argument is ignored if method is an already instantiated instance
        of an optuna sampler.

    objective_function :
        The objective function to use for optimization. This is the function that defines the optimization problem the
        PEC is trying to solve. The function is used to evaluate the `values <Mechanism_Base.value>` of the
        `outcome_variables <ParameterEstimationComposition.outcome_variables>`, according to which combinations of
        `parameters <ParameterEstimationComposition.parameters>` are assessed; this must be an `Callable`
        that takes a 3d array as its only argument, the shape of which must be (**num_estimates**, **num_trials**,
        number of **outcome_variables**).  The function should specify how to aggregate the value of each
        **outcome_variable** over **num_estimates** and/or **num_trials** if either is greater than 1.

    max_iterations :
        The maximum number of iterations to run the optimization for. In differential evolution, this is the number of
        generations. In optuna, this is the number of trials.

    direction :
        Whether to maximize or minimize the objective function. If 'maximize', the objective function is maximized. If
        'minimize', the objective function is minimized.


    """
    class Parameters(OptimizationFunction.Parameters):
        initial_seed = SharedParameter(attribute_name='owner')

    @check_user_specified
    @beartype
    def __init__(
        self,
        method: Union[Literal["differential_evolution"], optuna.samplers.BaseSampler, Type[optuna.samplers.BaseSampler]],
        optuna_kwargs: Optional[Mapping] = None,
        objective_function: Optional[Callable] = None,
        search_space=None,
        save_samples: Optional[bool] = None,
        save_values: Optional[bool] = None,
        max_iterations: int = 500,
        direction: Literal["maximize", "minimize"] = "maximize",
        **kwargs,
    ):
        self.method = method
        self._optuna_kwargs = {} if optuna_kwargs is None else {**optuna_kwargs}

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

        self._method_kwargs = kwargs if kwargs else {}

        super().__init__(
            search_space=search_space,
            save_samples=save_samples,
            save_values=save_values,
            search_function=search_function,
            search_termination_function=search_termination_function,
            aggregation_function=None,
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

        self.num_iterations = np.prod([i.num for i in search_space])

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

        # If the model is in the inputs, then inputs are passed as list of lists and we need to add the fitting
        # parameters to each trial as a concatenated list
        inputs = self.owner.composition.controller._pec_input_values
        self.owner.composition.controller.set_parameters_in_inputs(parameters=args, inputs=inputs)

        # Reset the search grid
        self.reset_grid(context)

        # Evaluate objective_function for each sample
        last_sample, last_value, all_samples, all_values = self._evaluate(
            variable=variable,
            context=context,
            params=None,
            fit_evaluate=True,
        )

        # Change randomization for next sample if specified (relies on randomization being last dimension)
        if (self.owner and not self.owner.parameters.same_seed_for_all_allocations._get(context) and
                self.parameters.randomization_dimension._get(context) is not None):
            rand_idx = self.parameters.randomization_dimension._get(context)
            self.search_space[rand_idx] = SampleIterator(specification=self.owner.gen_new_seed_sequence(context))

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
            obj_val, _ = self._evaluate_objective_and_sim_data(*args, context=context)
            return obj_val

        return objfunc

    def _evaluate_objective_and_sim_data(self, *args, context=None):
        """
        Run simulations for a parameter setting and return both the PEC objective value and simulated data.
        """
        sim_data = self._run_simulations(*args, context=context)

        # The composition might have more outputs than outcome variables, we need to subset the ones we need.
        sim_data = sim_data[:, :, self.outcome_variable_indices]

        return self._pec_objective_function(sim_data), sim_data

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
            results = self._fit(obj_func=f, context=context)

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
        context: Context = None,
    ):
        if self.method == "differential_evolution":
            return self._fit_differential_evolution(obj_func, display_iter, context)
        elif isinstance(self.method, optuna.samplers.BaseSampler):

            if self.owner.initial_seed is not None:
                warnings.warn("initial_seed on PEC is not None, but instantiated optuna sampler is being used. If you "
                              "want deterministic behavior, make sure to specify seed on optuna sampler as well")

            return self._fit_optuna(
                obj_func=obj_func, opt_func=self.method, display_iter=display_iter
            )
        # If this is a class of type base sampler, instantiate it and pass it to _fit_optuna
        elif isinstance(self.method, type) and issubclass(self.method, optuna.samplers.BaseSampler):

            if self.owner.initial_seed is not None:
                if "seed" in self._optuna_kwargs:
                    warnings.warn(
                        f"Overriding seed passed to optuna sampler with seed passed to PEC. "
                        f"Optuna sampler seed: {self._optuna_kwargs['seed']}, PEC.initial_seed: {self.owner.initial_seed}"
                    )

                self._optuna_kwargs["seed"] = self.owner.initial_seed

            return self._fit_optuna(
                obj_func=obj_func, opt_func=self.method(**self._optuna_kwargs), display_iter=display_iter
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
            if len(warns) > 0 and issubclass(warns[-1].category, PECObjectiveFuncWarning):
                warns_with_params.append((warns[-1], params))

            # Are we displaying each iteration
            if display_iter:
                # If we got a warning generating the objective function value, report it
                if len(warns) > 0 and issubclass(warns[-1].category, PECObjectiveFuncWarning):
                    progress.console.print(f"Warning: ", style="bold red")
                    progress.console.print(f"{warns[-1].message}", style="bold red")
                    progress.console.print(
                        f"{get_param_str(params)}, {self.obj_func_desc_str}: {obj_val}, "
                        f"Eval-Time: {elapsed} (seconds)",
                        style="bold red",
                        highlight=False,
                    )
                    # Clear the warnings
                    warns.clear()
                else:
                    progress.console.print(
                        f"{get_param_str(params)}, {self.obj_func_desc_str}: {obj_val}, "
                        f"Eval-Time: {elapsed} (seconds)",
                        highlight=False,
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
        context: Context = None,
    ):
        """
        Implementation of search using scipy's differential_evolution algorithm.
        """

        bounds = self.fit_param_bounds

        # We just need the upper and lower bounds for the differential evolution algorithm. The step size is not used.
        bounds = list([(lb, ub) for name, (lb, ub, step) in bounds.items()])

        # Get a seed to pass to scipy for its search. Make this dependent on the seed of the
        # OCM
        seed_for_scipy = self._get_current_parameter_value('initial_seed', context)
        seed_for_scipy = try_extract_0d_array_item(seed_for_scipy)

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
                warnings.simplefilter("always", PECObjectiveFuncWarning)
                warnings.simplefilter("always", BadLikelihoodWarning)

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
                    **self._method_kwargs
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

                # Turn off optuna logging except for errors or warnings, it doesn't work well with our PNL progress bar
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                study = optuna.create_study(
                    sampler=opt_func, direction=self.direction
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
            # Go through each parameter and create a unique name for it
            if not self.owner.depends_on:
                return [f"{mech.name}.{param_name}"
                        for param_name, mech in self.owner.fit_parameters.keys()]
            else:
                names = []
                for param_name, mech in self.owner.fit_parameters.keys():
                    if (param_name, mech) in self.owner.cond_levels:
                        for level in self.owner.cond_levels[(param_name, mech)]:
                            names.append(f"{mech.name}.{param_name}[{level}]")
                    else:
                        names.append(f"{mech.name}.{param_name}")

                return names

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

            if not self.owner.depends_on:
                bounds = [(float(min(s)), float(max(s))) for s in self.owner.fit_parameters.values()]
                steps = [np.unique(np.diff(s).round(decimals=5)) for s in self.owner.fit_parameters.values()]
            else:
                bounds = []
                steps = []
                for param_name, mech in self.owner.fit_parameters.keys():
                    s = self.owner.fit_parameters[(param_name, mech)]
                    if (param_name, mech) in self.owner.cond_levels:
                        for _ in self.owner.cond_levels[(param_name, mech)]:
                            bounds.append((float(min(s)), float(max(s))))
                            steps.append(np.unique(np.diff(s).round(decimals=5)))
                    else:
                        bounds.append((float(min(s)), float(max(s))))
                        steps.append(np.unique(np.diff(s).round(decimals=5)))

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

        execution_phase_at_entry = context.execution_phase
        context.execution_phase = ContextFlags.PROCESSING
        try:
            ll, sim_data = self._evaluate_objective_and_sim_data(*args, context=context)
        finally:
            context.execution_phase = execution_phase_at_entry

        return ll, sim_data
