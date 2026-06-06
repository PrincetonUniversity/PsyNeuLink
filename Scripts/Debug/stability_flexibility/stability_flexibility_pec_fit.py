#%%
import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import psyneulink as pnl
import pandas as pd

from psyneulink.core.globals.utilities import set_global_seed

sys.path.append(".")

from stability_flexibility import make_stab_flex, generate_trial_sequence


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--num-estimates", type=int, default=10000)
    parser.add_argument("--max-iterations", type=int, default=300)
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

# Let's make things reproducible
pnl_seed = 0
set_global_seed(pnl_seed)
trial_seq_seed = 0

# High-level parameters the impact performance of the test
num_trials = args.num_trials
time_step_size = args.time_step_size
num_estimates = args.num_estimates

print(
    f"Stability-flexibility PEC config: cores={args.cores}, num_trials={num_trials}, "
    f"time_step_size={time_step_size}, num_estimates={num_estimates}, "
    f"max_iterations={args.max_iterations}, likelihood_estimator={args.likelihood_estimator}, "
    f"histogram_backend={args.histogram_backend}, histogram_backend_used={_histogram_backend_used(args)}, "
    f"histogram_bins={args.histogram_bins}, histogram_pseudocount={args.histogram_pseudocount}, "
    f"histogram_threads={args.histogram_threads}",
    flush=True,
)

sf_params = dict(
    gain=3.0,
    leak=3.0,
    competition=2.0,
    lca_time_step_size=time_step_size,
    non_decision_time=0.2,
    automaticity=0.01,
    starting_value=0.0,
    threshold=0.1,
    ddm_noise=0.1,
    lca_noise=0.0,
    scale=0.2,
    ddm_time_step_size=time_step_size,
)

# Generate some sample data to run the model on
taskTrain, stimulusTrain, cueTrain, correctResponse = generate_trial_sequence(240, 0.5, seed=trial_seq_seed)
taskTrain = taskTrain[0:num_trials]
stimulusTrain = stimulusTrain[0:num_trials]
cueTrain = cueTrain[0:num_trials]
correctResponse = correctResponse[0:num_trials]

# CSI is in terms of time steps, we need to scale by ten because original code
# was set to run with timestep size of 0.001
cueTrain = [c / 10.0 for c in cueTrain]

# Make a stability flexibility composition
comp = make_stab_flex(**sf_params)

# Let's run the model with some sample data
taskLayer = comp.nodes["Task Input [I1, I2]"]
stimulusInfo = comp.nodes["Stimulus Input [S1, S2]"]
cueInterval = comp.nodes["Cue-Stimulus Interval"]
correctInfo = comp.nodes["Correct Response Info"]

inputs = {
    taskLayer: [[np.array(taskTrain[i])] for i in range(num_trials)],
    stimulusInfo: [[np.array(stimulusTrain[i])] for i in range(num_trials)],
    cueInterval: [[np.array([cueTrain[i]])] for i in range(num_trials)],
    correctInfo: [[np.array([correctResponse[i]])] for i in range(num_trials)]
}

print("Running inner composition to generate data to fit for parameter recovery test.")
comp.run(inputs, execution_mode=pnl.ExecutionMode.LLVMRun)
results = comp.results

print("Setting up PEC")

data_to_fit = pd.DataFrame(
    np.squeeze(np.array(results))[:, 1:], columns=["decision", "response_time"]
)
data_to_fit["decision"] = data_to_fit["decision"].astype("category")

#%%

# Create a parameter estimation composition to fit the data we just generated and hopefully recover the
# parameters of the composition.

controlModule = comp.nodes["Task Activations [Act1, Act2]"]
congruenceWeighting = comp.nodes["Automaticity-weighted Stimulus Input [w*S1, w*S2]"]
decisionMaker = comp.nodes["DDM"]
decisionGate = comp.nodes["DECISION_GATE"]
responseGate = comp.nodes["RESPONSE_GATE"]

fit_parameters = {
    ("gain", controlModule): np.linspace(1.0, 10.0, 1000),  # Gain
    ("slope", congruenceWeighting): np.linspace(0.0, 0.1, 1000),  # Automaticity
    ("threshold", decisionMaker): np.linspace(0.01, 0.5, 1000),  # Threshold
    ("non_decision_time", decisionMaker): np.linspace(0.1, 0.4, 1000),  # Threshold
}

import optuna
pec = pnl.ParameterEstimationComposition(
    name="pec",
    nodes=comp,
    parameters=fit_parameters,
    outcome_variables=[
        decisionGate.output_ports[0],
        responseGate.output_ports[0],
    ],
    data=data_to_fit,
    likelihood_estimator=args.likelihood_estimator,
    likelihood_estimator_kwargs=_likelihood_estimator_kwargs(args),
    optimization_function=pnl.PECOptimizationFunction(
        method=optuna.samplers.CmaEsSampler(seed=0),
        max_iterations=args.max_iterations,
    ),
    num_estimates=num_estimates,
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

print("Running the PEC")
run_start = time.perf_counter()
ret = pec.run(inputs=inputs)
total_execution_time = time.perf_counter() - run_start
print(f"Total execution time: {total_execution_time:.6f} seconds", flush=True)
if eval_times:
    print(f"Average Eval-Time: {np.mean(eval_times):.6f} seconds", flush=True)
optimal_parameters = list(pec.optimized_parameter_values.values())

# Print the recovered parameters.
records = []
for (name, mech), recovered_param in zip(fit_parameters.keys(), optimal_parameters):

    if name == "slope":
        true_param = sf_params['automaticity']
    else:
        true_param = sf_params[name]

    percent_error = 100.0 * (abs(true_param - recovered_param) / true_param)
    records.append((name, mech.name, true_param, recovered_param, percent_error))
df = pd.DataFrame(records, columns=['Parameter', 'Component', 'Value', 'Recovered Value', 'Percent Error'])
print(df)
