#%%
import sys
import numpy as np
import psyneulink as pnl
import pandas as pd
import argparse
import optuna
import re
import copy
from datetime import datetime
from pathlib import Path
from psyneulink.core.batched import batched_node_op

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = SCRIPT_DIR / "data_to_fit_study3.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--subject_id", help="The subject ID to fit model.", default=1, type=int)
parser.add_argument("--cpu_count", help="The number of CPUs requested for this task.", default=None, type=int)
parser.add_argument(
    "--backend",
    choices=("llvm", "triton", "triton_cpu"),
    default="llvm",
    help="Likelihood simulator backend; use 'triton' for a CUDA GPU.",
)
parser.add_argument("--num_estimates", default=10000, type=int)
parser.add_argument("--max_iterations", default=5000, type=int)
parser.add_argument("--max_steps", default=1200, type=int)
parser.add_argument("--bins", default=100, type=int)
parser.add_argument("--seed", default=1, type=int)
parser.add_argument(
    "--data_file",
    "--data-file",
    dest="data_file",
    default=DEFAULT_DATA_FILE,
    type=Path,
    help=(
        "Behavioral CSV to fit (default: data_to_fit_study3.csv beside this "
        "script). The dataset is not stored in git."
    ),
)
parser.add_argument(
    "--parameter_batch_size",
    default=11,
    type=int,
    help=(
        "CMA-ES candidates evaluated in one GPU call; use 0 to evaluate the "
        "same 11-member population one candidate at a time."
    ),
)
parser.add_argument("--triton_block_size", "--triton-block-size", default=128, type=int)
parser.add_argument("--triton_num_warps", "--triton-num-warps", default=4, type=int)
parser.add_argument("--triton_maxnreg", "--triton-maxnreg", default=None, type=int)
parser.add_argument(
    "--skip_posterior_predictive",
    "--skip-posterior-predictive",
    action="store_true",
)
parser.add_argument("--skip_fit_output", "--skip-fit-output", action="store_true")

args = parser.parse_args()
data_file = args.data_file.expanduser()
if not data_file.is_file():
    parser.error(
        f"behavioral data file not found: {data_file}. "
        "Supply it with --data_file /path/to/data_to_fit_study3.csv."
    )
data_file = data_file.resolve()

sys.path.insert(0, str(SCRIPT_DIR))
from expectation_model_study2_study3 import make_stab_flex  # noqa: E402


@batched_node_op("Drift Rate Value")
def _batched_drift_rate(x0, x1, x2, x3, x4, x5, x6):
    """Triton implementation of the model's seven-input drift-rate UDF."""

    a = 1.0 / (1.0 + tl.exp(-((x0 - x1) + 4.0 * x4 - 4.0)))
    b = 1.0 / (1.0 + tl.exp(-((x1 - x0) + 4.0 * x4 - 4.0)))
    c = 1.0 / (1.0 + tl.exp(-((x2 - x3) + 4.0 * x5 - 4.0)))
    d = 1.0 / (1.0 + tl.exp(-((x3 - x2) + 4.0 * x5 - 4.0)))
    positive = 1.0 / (1.0 + tl.exp(-(a - b + c - d)))
    negative = 1.0 / (1.0 + tl.exp(-(-a + b - c + d)))
    return (positive - negative) * x6


def get_node(comp, name):
    """Return the first node whose name starts with `name`.

    Handles PsyNeuLink's automatic numeric suffixes (e.g. 'DDM-1') so the
    same helper works for both the primary and posterior-predictive compositions.
    """
    for node in comp.nodes:
        if node.name.startswith(name):
            return node
    return None


def make_input_dict(comp, taskSequence, stimulusSequence, correctResponseSequence):
    return {
        get_node(comp, "Task Input"):       [[np.array(v)] for v in taskSequence],
        get_node(comp, "Stimulus Input"):   [[np.array(v)] for v in stimulusSequence],
        get_node(comp, "Correct Response"): [[np.array(v)] for v in correctResponseSequence],
        get_node(comp, "Cue Stimulus Interval"): [[np.array([0])] if (taskSequence[i] == taskSequence[i - 1]).all() else [np.array([1])] for i in range(len(taskSequence))],
        get_node(comp, "Threshold Mechanism"): [[np.array([0])] for _ in taskSequence],
    }


# -- Optimization settings -----------------------------------------------------
num_estimates = args.num_estimates
max_iterations = args.max_iterations

sf_params = dict(
    gain=10.0,
    leak=12.0,
    competition=3.0,
    iti=100,
    csi_switch=0,
    threshold=0.06,
    threshold_collapse=0.0,
    non_decision_time=0.3,
    lca_time_step_size=0.01,
    ddm_time_step_size=0.01,
    lca_noise=0.0,
    ddm_noise=0.1,
)

# -- Load and prepare behavioral data -----------------------------------------
data = pd.read_csv(data_file)
data["decision"] = data["decision"].astype("category")
data["sequence"] = data["sequence"].astype("category")

actual_subject_id = data.subject_nr.unique()[args.subject_id - 1]

data_to_fit = data[
    (data.subject_nr == actual_subject_id) &
    data.sequence.isin(["RealRare", "RealFrequent", "NoInstruction"])
].reset_index(drop=True)

# -- Extract arrays for model inputs and PEC ----------------------------------
taskSequence            = data_to_fit[["T1", "T2"]].to_numpy()
stimulusSequence        = data_to_fit[["S1", "S2", "S3", "S4"]].to_numpy()
correctResponseSequence = data_to_fit["correct_response"].to_numpy()
likelihood_include_mask = data_to_fit["likelihood_include_mask"].to_numpy(dtype="bool")

data_to_fit = data_to_fit[["decision", "response_time", "sequence"]]

# -- Build composition and define fit parameters -------------------------------
comp = make_stab_flex(**sf_params)

cueStimulusInterval = get_node(comp, "Cue Stimulus Interval")
controlExecution    = get_node(comp, "Task Activations [C1, C2]")
thresholdMechanism  = get_node(comp, "Threshold Mechanism")
decisionMaker       = get_node(comp, "DDM")
decisionGate        = get_node(comp, "DECISION_GATE")
responseGate        = get_node(comp, "RESPONSE_GATE")

fit_parameters = {
    ("gain",                       controlExecution):    np.linspace(5.0,   35.0,  301),
    ("slope",                      cueStimulusInterval): np.linspace(0,     30,     31),
    ("intercept",                  thresholdMechanism):  np.linspace(0.05,  0.25,  401),
    ("offset-integrator_function", thresholdMechanism):  np.linspace(-0.003, 0.0,  301),
    ("non_decision_time",          decisionMaker):       np.linspace(0.1,   0.4,   301),
}

batched_options = {}
if args.backend != "llvm":
    batched_options = {
        "batched_backend": args.backend,
        "batched_max_steps": args.max_steps,
        "batched_bins": args.bins,
        "batched_seed": args.seed,
    }
    if args.parameter_batch_size:
        batched_options["batched_parameter_batch_size"] = args.parameter_batch_size
    if args.backend == "triton":
        batched_options["batched_triton_launch_options"] = {
            "block_size": args.triton_block_size,
            "num_warps": args.triton_num_warps,
            "maxnreg": args.triton_maxnreg,
        }

cma_population_size = args.parameter_batch_size or 11

pec = pnl.ParameterEstimationComposition(
    name="pec",
    nodes=comp,
    parameters=fit_parameters,
    depends_on={
        ("gain",                       controlExecution):    "sequence",
        ("intercept",                  thresholdMechanism):  "sequence",
        ("offset-integrator_function", thresholdMechanism):  "sequence",
        ("non_decision_time",          decisionMaker):       "sequence",
    },
    outcome_variables=[
        decisionGate.output_ports[0],
        responseGate.output_ports[0],
    ],
    data=data_to_fit,
    likelihood_include_mask=likelihood_include_mask,
    optimization_function=pnl.PECOptimizationFunction(
        method=optuna.samplers.CmaEsSampler(
            sigma0=0.2,
            lr_adapt=True,
            popsize=cma_population_size,
            seed=args.seed,
        ),
        max_iterations=max_iterations,
        **batched_options,
    ),
    num_estimates=num_estimates,
)

pec.controller.parameters.comp_execution_mode.set("LLVM")
pec.controller.function.parameters.save_values.set(True)

inputs = make_input_dict(comp, taskSequence, stimulusSequence, correctResponseSequence)

print(f"Parameters used to initialize the composition: ")
print(sf_params)

print(
    f"Fit Expectation Study 3.2 Real Sequences, No Warm Start, Single Surrogate CSI, "
    f"{num_estimates} Num Estimates, Sigma 0.2, LR Adapt = True, Leak = 12, "
    f"Participant {actual_subject_id}, Slurm Array {args.subject_id}, "
    f"Backend {args.backend}, Parameter Batch Size {args.parameter_batch_size}"
)
if args.backend == "triton":
    print(
        "Triton launch configuration: "
        f"block_size={args.triton_block_size}, "
        f"num_warps={args.triton_num_warps}, "
        f"maxnreg={args.triton_maxnreg}"
    )
print("Running the PEC")

start_time = datetime.now()
pec.run(inputs=inputs)
end_time = datetime.now()

optimal_parameters = pec.optimized_parameter_values
print(optimal_parameters)
print(f"Optimal Log-Likelihood: {pec.optimal_value}")
print("Fit Complete!")

# -- Save fit results ----------------------------------------------------------
df = pd.DataFrame({k: [v] for k, v in optimal_parameters.items()})

n_params = len(optimal_parameters.items())
n_trials = likelihood_include_mask.sum()
bic = n_params * np.log(n_trials) - 2 * pec.optimal_value
aic = 2 * n_params - 2 * pec.optimal_value
df["log_likelihood"] = pec.optimal_value
df["n_params"]        = n_params
df["n_trials"]         = n_trials
df["bic"]             = bic
df["aic"]             = aic
df["subject_nr"]     = actual_subject_id
df["fit_type"]       = "study3_real_sequences_single_csi"
df["num_estimates"]  = num_estimates
df["max_iterations"] = max_iterations
df["fit_duration"] = end_time - start_time
df["cpu_count"] = args.cpu_count
df["backend"] = args.backend
df = pd.concat([df, pd.DataFrame(sf_params, index=[0])], axis=1)

output_suffix = "" if args.backend == "llvm" else f"_{args.backend}"
output_dir = SCRIPT_DIR / "fits"
output_dir.mkdir(exist_ok=True)
output_stem = (
    "expectation_3.2_real_sequences_single_csi_leak12"
    f"_sub{actual_subject_id}{output_suffix}"
)
if not args.skip_fit_output:
    df.to_csv(
        output_dir / f"{output_stem}.csv",
        index=False,
    )

# -- Posterior predictive checks -----------------------------------------------
sim_posterior_predict = not args.skip_posterior_predictive
n_sim = 100

if sim_posterior_predict:
    comp_pp = make_stab_flex(**sf_params)

    cueStimulusInterval_pp = get_node(comp_pp, "Cue Stimulus Interval")
    controlExecution_pp    = get_node(comp_pp, "Task Activations [C1, C2]")
    thresholdMechanism_pp  = get_node(comp_pp, "Threshold Mechanism")
    decisionMaker_pp       = get_node(comp_pp, "DDM")

    # Reload the full trial sequence for this subject (no mask filtering)
    # and replicate n_sim times for a stable posterior predictive distribution.
    pp_data = data[
        (data.subject_nr == actual_subject_id) &
        data.sequence.isin(["RealRare", "RealFrequent", "NoInstruction"])
    ].reset_index(drop=True)
    pp_data = pd.concat([pp_data] * n_sim, ignore_index=True)

    taskSequence_pp            = pp_data[["T1", "T2"]].to_numpy()
    stimulusSequence_pp        = pp_data[["S1", "S2", "S3", "S4"]].to_numpy()
    correctResponseSequence_pp = pp_data["correct_response"].to_numpy()

    inputs_pp = make_input_dict(
        comp_pp, taskSequence_pp, stimulusSequence_pp, correctResponseSequence_pp
    )

    # Infer conditions present in this dataset.
    conditions = pp_data["sequence"].unique()

    def _select(param_base):
        """Map each trial to its fitted parameter value.

        For condition-specific parameters (depends_on by sequence), selects the
        appropriate condition column. For shared parameters (no depends_on),
        broadcasts the single fitted value across all trials.
        """
        col_with_cond = f"{param_base}[{conditions[0]}]"
        if col_with_cond in df.columns:
            return np.select(
                [pp_data["sequence"] == cond for cond in conditions],
                [df[f"{param_base}[{cond}]"] for cond in conditions],
            )
        else:
            return np.full(len(pp_data), df[param_base].iloc[0])

    gain_array = _select("Task Activations [C1, C2].gain")
    prep_time_array = _select("Cue Stimulus Interval.slope")
    threshold_array = _select("Threshold Mechanism.intercept")
    collapse_array = _select("Threshold Mechanism.offset-integrator_function")
    ndt_array = _select("DDM.non_decision_time")

    gain_mech      = pnl.ControlMechanism(name="Gain Mech",      control_signals=("gain",                       controlExecution_pp),   modulation=pnl.OVERRIDE)
    prep_time_mech = pnl.ControlMechanism(name="Prep Time Mech", control_signals=("slope",                      cueStimulusInterval_pp), modulation=pnl.OVERRIDE)
    threshold_mech = pnl.ControlMechanism(name="Threshold Mech", control_signals=("intercept",                  thresholdMechanism_pp),  modulation=pnl.OVERRIDE)
    collapse_mech  = pnl.ControlMechanism(name="Collapse Mech",  control_signals=("offset-integrator_function", thresholdMechanism_pp),  modulation=pnl.OVERRIDE)
    ndt_mech       = pnl.ControlMechanism(name="NDT Mech",       control_signals=("non_decision_time",          decisionMaker_pp),       modulation=pnl.OVERRIDE)

    comp_pp.add_nodes([gain_mech, prep_time_mech, threshold_mech, collapse_mech, ndt_mech])

    inputs_pp["Gain Mech"]      = [[np.array([v])] for v in gain_array]
    inputs_pp["Prep Time Mech"] = [[np.array([v])] for v in prep_time_array]
    inputs_pp["Threshold Mech"] = [[np.array([v])] for v in threshold_array]
    inputs_pp["Collapse Mech"]  = [[np.array([v])] for v in collapse_array]
    inputs_pp["NDT Mech"]       = [[np.array([v])] for v in ndt_array]

    comp_pp.run(inputs_pp, execution_mode=pnl.ExecutionMode.LLVMRun)

    sim_data = pd.DataFrame(
        np.squeeze(np.array(comp_pp.results))[:, 1:],
        columns=["decision", "response_time"],
    ).assign(
        sequence=pp_data["sequence"].to_numpy(),
        task_transition=pp_data["task_transition"].to_numpy(),
        congruence=pp_data["congruence"].to_numpy(),
        subject_nr=actual_subject_id,
    )

    posterior_output = output_dir / (
        "expectation_3.2_real_sequences_single_csi_leak12"
        f"_posterior_predict_sub{actual_subject_id}{output_suffix}.csv"
    )
    sim_data.to_csv(posterior_output, index=False)
    print("Simulation Complete!")
