#%%
import argparse
import importlib.util
import os
import time

import numpy as np
import psyneulink as pnl
import pandas as pd

from psyneulink.core.globals.utilities import set_global_seed


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--num-estimates", type=int, default=10000)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--time-step-size", type=float, default=0.01)
    parser.add_argument("--likelihood-estimator", choices=["kde", "histogram"], default="kde")
    parser.add_argument("--histogram-backend", choices=["auto", "numpy", "boost"], default="auto")
    parser.add_argument("--histogram-bins", type=int, default=32)
    parser.add_argument("--histogram-pseudocount", type=float, default=0.0)
    parser.add_argument("--histogram-threads", type=int, default=1)
    return parser.parse_args()


def _set_threads(cores):
    if cores is None:
        return

    for env_var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[env_var] = str(cores)
    pnl.set_num_threads(cores)


def _histogram_backend_used(args):
    if args.likelihood_estimator != "histogram":
        return None
    if args.histogram_backend == "numpy":
        return "numpy"
    if args.histogram_backend == "boost" and importlib.util.find_spec("boost_histogram") is not None:
        return "boost"
    if args.histogram_backend == "auto" and importlib.util.find_spec("boost_histogram") is not None:
        return "boost"
    return "numpy"


def _likelihood_estimator_kwargs(args):
    if args.likelihood_estimator != "histogram":
        return None

    return {
        "histogram_backend": args.histogram_backend,
        "bins": args.histogram_bins,
        "pseudocount": args.histogram_pseudocount,
        "threads": args.histogram_threads,
    }


args = _parse_args()
_set_threads(args.cores)

# # Let's make things reproducible
set_global_seed(0)

# High-level parameters the impact performance of the test
num_trials = args.num_trials
time_step_size = args.time_step_size
num_estimates = args.num_estimates

print(
    f"DDM PEC config: cores={args.cores}, num_trials={num_trials}, "
    f"time_step_size={time_step_size}, num_estimates={num_estimates}, "
    f"max_iterations={args.max_iterations}, likelihood_estimator={args.likelihood_estimator}, "
    f"histogram_backend={args.histogram_backend}, histogram_backend_used={_histogram_backend_used(args)}, "
    f"histogram_bins={args.histogram_bins}, histogram_pseudocount={args.histogram_pseudocount}, "
    f"histogram_threads={args.histogram_threads}",
    flush=True,
)

ddm_params = dict(
    starting_value=0.0,
    rate=0.3,
    noise=1.0,
    threshold=0.6,
    non_decision_time=0.15,
    time_step_size=time_step_size,
)

# Create a simple one mechanism composition containing a DDM in integrator mode.
decision = pnl.DDM(
    function=pnl.DriftDiffusionIntegrator(**ddm_params),
    output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
    name="DDM",
)

comp = pnl.Composition(pathways=decision)

# Let's generate an "experimental" dataset to fit. This is a parameter recovery test
# Lets make 10% of the trials have a positive stimulus drift rate, and the other 90%
# have a negative stimulus drift rate.
# trial_inputs = np.ones((num_trials, 1))
rng = np.random.default_rng(12345)
trial_inputs = rng.choice(
    [5.0, -5.0], size=(num_trials, 1), p=[0.10, 0.9], replace=True
)

# Make the first and last input positive for sure. This helps make sure inputs are really getting
# passed to the composition correctly during parameter fitting, and we aren't just getting a single
# trials worth of a cached input.
trial_inputs[0] = np.abs(trial_inputs[0])
trial_inputs[-1] = np.abs(trial_inputs[-1])

inputs_dict = {decision: trial_inputs}

# Store the results of this "experiment" as a numpy array. This should be a
# 2D array of shape (len(input), 2). The first column being a discrete variable
# specifying whether the upper or lower decision boundary is reached and the second column is the
# reaction time. We will put the data into a pandas DataFrame, this makes it
# easier to specify which columns in the data are categorical or not.

# Run the composition to generate some data to fit
comp.run(inputs=inputs_dict)
results = comp.results

data_to_fit = pd.DataFrame(
    np.squeeze(np.array(results)), columns=["decision", "response_time"]
)
data_to_fit["decision"] = data_to_fit["decision"].astype("category")

# Create a parameter estimation composition to fit the data we just generated and hopefully recover the
# parameters of the DDM.

fit_parameters = {
    ("rate", decision): np.linspace(-0.5, 0.5, 1000),
    ("threshold", decision): np.linspace(0.5, 1.0, 1000),
    ('non_decision_time', decision): np.linspace(0.0, 1.0, 1000),
}

#%%
import optuna

pec = pnl.ParameterEstimationComposition(
    name="pec",
    nodes=[comp],
    parameters=fit_parameters,
    outcome_variables=[
        decision.output_ports[pnl.DECISION_OUTCOME],
        decision.output_ports[pnl.RESPONSE_TIME],
    ],
    data=data_to_fit,
    likelihood_estimator=args.likelihood_estimator,
    likelihood_estimator_kwargs=_likelihood_estimator_kwargs(args),
    optimization_function=pnl.PECOptimizationFunction(
        method=optuna.samplers.CmaEsSampler(seed=0),
        max_iterations=args.max_iterations,
    ),
    num_estimates=num_estimates,
    initial_seed=42,
)

pec.controller.parameters.comp_execution_mode.set("LLVM")
pec.controller.function.parameters.save_values.set(True)
eval_times = []
_objective_eval = pec.controller.function._evaluate_objective_and_sim_data


def _timed_objective_eval(*objective_args, **objective_kwargs):
    start = time.perf_counter()
    try:
        return _objective_eval(*objective_args, **objective_kwargs)
    finally:
        eval_times.append(time.perf_counter() - start)


pec.controller.function._evaluate_objective_and_sim_data = _timed_objective_eval
print("Running PEC", flush=True)
run_start = time.perf_counter()
ret = pec.run(inputs={comp: trial_inputs})
total_execution_time = time.perf_counter() - run_start
print(f"Total execution time: {total_execution_time:.6f} seconds", flush=True)
if eval_times:
    print(f"Average Eval-Time: {np.mean(eval_times):.6f} seconds", flush=True)
optimal_parameters = list(pec.optimized_parameter_values.values())

# Check that the parameters are recovered and that the log-likelihood is correct, set the tolerance pretty high,
# things are noisy because of the low number of trials and estimates.
print(
    "Recovered within atol=0.1:",
    np.allclose(
        optimal_parameters,
        [ddm_params["rate"], ddm_params["threshold"], ddm_params["non_decision_time"]],
        atol=0.1,
    ),
)

records = []
for (name, mech), recovered_param in zip(fit_parameters.keys(), optimal_parameters):
    percent_error = 100.0 * (abs(ddm_params[name] - recovered_param) / ddm_params[name])
    records.append((name, mech.name, ddm_params[name], recovered_param, percent_error))
df = pd.DataFrame(records, columns=['Parameter', 'Component', 'Value', 'Recovered Value', 'Percent Error'])
print(df)
