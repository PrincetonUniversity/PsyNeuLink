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
parser.add_argument("--subject_id", "--subject-id", help="The subject ID to fit model.", default=1, type=int)
parser.add_argument("--cpu_count", "--cpu-count", help="The number of CPUs requested for this task.", default=None, type=int)
parser.add_argument(
    "--backend",
    choices=("llvm", "triton", "triton_cpu"),
    default="llvm",
    help="Likelihood simulator backend; use 'triton' for a CUDA GPU.",
)
parser.add_argument("--num_estimates", "--num-estimates", default=10000, type=int)
parser.add_argument("--max_iterations", "--max-iterations", default=5000, type=int)
parser.add_argument("--max_steps", "--max-steps", default=1200, type=int)
parser.add_argument("--bins", default=100, type=int)
parser.add_argument(
    "--smoothing_sigma",
    "--smoothing-sigma",
    default=0.0,
    type=float,
    help="Gaussian histogram smoothing bandwidth in RT-bin units (0 disables).",
)
parser.add_argument(
    "--pseudocount",
    default=0.0,
    type=float,
    help="Symmetric pseudocount per joint decision/RT histogram cell (0 disables).",
)
parser.add_argument(
    "--condition-observed-history",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Condition persistent CSI state on each observed choice/RT before "
        "simulating the next trial (default: enabled)."
    ),
)
parser.add_argument(
    "--deterministic-observed-history",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "For the zero-noise CSI LCA, compute its observed endpoint history once "
        "per parameter candidate and simulate only the DDM on the GPU. This "
        "replaces particle filtering and requires --backend triton."
    ),
)
parser.add_argument(
    "--seed",
    default=1,
    type=int,
    help="Legacy default used for optimizer and simulation seeds unless either is set explicitly.",
)
parser.add_argument("--optimizer_seed", "--optimizer-seed", default=None, type=int)
parser.add_argument("--simulation_seed", "--simulation-seed", default=None, type=int)
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
    "--parameter-batch-size",
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
parser.add_argument(
    "--posterior_predictive_simulations",
    "--posterior-predictive-simulations",
    default=100,
    type=int,
)
parser.add_argument(
    "--posterior_predictive_seed",
    "--posterior-predictive-seed",
    default=None,
    type=int,
    help="Random seed for posterior prediction (default: simulation seed + 1).",
)
parser.add_argument("--skip_fit_output", "--skip-fit-output", action="store_true")
parser.add_argument(
    "--output_dir",
    "--output-dir",
    default=SCRIPT_DIR / "fits",
    type=Path,
    help="Directory for fit, predictive, and rescoring CSV files.",
)
parser.add_argument(
    "--run_label",
    "--run-label",
    default=None,
    help="Optional experiment label recorded in output files.",
)
parser.add_argument(
    "--rescore_parameter_file",
    "--rescore-parameter-file",
    default=None,
    type=Path,
    help="Skip optimization and rescore the one-row fit CSV at fixed parameters.",
)
parser.add_argument(
    "--rescore_simulation_seeds",
    "--rescore-simulation-seeds",
    nargs="+",
    type=int,
    default=None,
    help="Simulation seeds used with --rescore-parameter-file.",
)

args = parser.parse_args()
optimizer_seed = args.seed if args.optimizer_seed is None else args.optimizer_seed
simulation_seed = args.seed if args.simulation_seed is None else args.simulation_seed
if args.posterior_predictive_simulations < 1:
    parser.error("--posterior-predictive-simulations must be at least 1.")
if args.smoothing_sigma < 0:
    parser.error("--smoothing-sigma must be nonnegative.")
if args.pseudocount < 0:
    parser.error("--pseudocount must be nonnegative.")
if args.backend == "llvm" and (args.smoothing_sigma != 0 or args.pseudocount != 0):
    parser.error("--smoothing-sigma and --pseudocount require a batched backend.")
if args.deterministic_observed_history and args.backend != "triton":
    parser.error("--deterministic-observed-history requires --backend triton.")
if args.deterministic_observed_history and not args.condition_observed_history:
    parser.error(
        "--deterministic-observed-history conflicts with "
        "--no-condition-observed-history."
    )
history_mode = (
    "deterministic"
    if args.deterministic_observed_history
    else ("particle" if args.condition_observed_history else "unconditional")
)
if args.rescore_simulation_seeds is not None and args.rescore_parameter_file is None:
    parser.error("--rescore-simulation-seeds requires --rescore-parameter-file.")
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
conditioned_options = {
    "conditioned_likelihood": (
        args.condition_observed_history
        and not args.deterministic_observed_history
    ),
    "deterministic_history_likelihood": args.deterministic_observed_history,
}
if args.condition_observed_history:
    conditioned_options.update(
        {
            "batched_bins": args.bins,
            "batched_smoothing_sigma": args.smoothing_sigma,
            "batched_pseudocount": args.pseudocount,
            "batched_categorical_cardinalities": [2],
            "batched_seed": simulation_seed,
        }
    )
if args.backend != "llvm":
    batched_options = {
        "batched_backend": args.backend,
        "batched_max_steps": args.max_steps,
    }
    if not args.condition_observed_history:
        batched_options.update(
            {
                "batched_bins": args.bins,
                "batched_smoothing_sigma": args.smoothing_sigma,
                "batched_pseudocount": args.pseudocount,
                # The DDM decision has two possible outcomes. Supplying this
                # explicitly keeps pseudocount normalization correct even when
                # one participant contains only one observed outcome.
                "batched_categorical_cardinalities": [2],
                "batched_seed": simulation_seed,
            }
        )
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
            seed=optimizer_seed,
        ),
        max_iterations=max_iterations,
        **conditioned_options,
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
    f"Backend {args.backend}, Parameter Batch Size {args.parameter_batch_size}, "
    f"Optimizer Seed {optimizer_seed}, Simulation Seed {simulation_seed}, "
    f"Smoothing Sigma {args.smoothing_sigma}, Pseudocount {args.pseudocount}, "
    f"History Mode {history_mode}"
)
if args.backend == "triton":
    print(
        "Triton launch configuration: "
        f"block_size={args.triton_block_size}, "
        f"num_warps={args.triton_num_warps}, "
        f"maxnreg={args.triton_maxnreg}"
    )

output_dir = args.output_dir.expanduser()
if not output_dir.is_absolute():
    output_dir = SCRIPT_DIR / output_dir
output_dir.mkdir(parents=True, exist_ok=True)
output_suffix = "" if args.backend == "llvm" else f"_{args.backend}"
output_stem = (
    "expectation_3.2_real_sequences_single_csi_leak12"
    f"_sub{actual_subject_id}{output_suffix}"
)

if args.rescore_parameter_file is not None:
    parameter_file = args.rescore_parameter_file.expanduser().resolve()
    if not parameter_file.is_file():
        parser.error(f"fit parameters to rescore were not found: {parameter_file}")
    candidate = pd.read_csv(parameter_file)
    if len(candidate) != 1:
        parser.error(
            f"--rescore-parameter-file must contain exactly one row; got {len(candidate)}"
        )
    if (
        "subject_nr" in candidate
        and int(candidate.loc[0, "subject_nr"]) != actual_subject_id
    ):
        parser.error(
            f"parameter file is for subject {candidate.loc[0, 'subject_nr']}, "
            f"but --subject-id selects {actual_subject_id}"
        )

    fit_function = pec.controller.function
    missing = [name for name in fit_function.fit_param_names if name not in candidate]
    if missing:
        parser.error(f"parameter file is missing fitted columns: {missing}")
    fit_values = tuple(
        float(candidate.loc[0, name]) for name in fit_function.fit_param_names
    )
    validation_seeds = args.rescore_simulation_seeds or [simulation_seed]
    rows = []
    print(
        f"Rescoring {parameter_file} with {num_estimates} estimates and seeds "
        f"{validation_seeds}"
    )
    for validation_seed in validation_seeds:
        fit_function.batched_seed = validation_seed
        score_start = datetime.now()
        score = pec.log_likelihood(*fit_values, inputs=inputs)
        score_duration = datetime.now() - score_start
        row = {
            name: value
            for name, value in zip(fit_function.fit_param_names, fit_values)
        }
        row.update(
            subject_nr=actual_subject_id,
            validation_log_likelihood=score,
            validation_simulation_seed=validation_seed,
            validation_num_estimates=num_estimates,
            validation_duration=score_duration,
            source_parameter_file=str(parameter_file),
            source_log_likelihood=(
                float(candidate.loc[0, "log_likelihood"])
                if "log_likelihood" in candidate
                else np.nan
            ),
            optimizer_seed=(
                candidate.loc[0, "optimizer_seed"]
                if "optimizer_seed" in candidate
                else optimizer_seed
            ),
            fitting_simulation_seed=(
                candidate.loc[0, "simulation_seed"]
                if "simulation_seed" in candidate
                else simulation_seed
            ),
            source_run_label=(
                candidate.loc[0, "run_label"]
                if "run_label" in candidate
                else args.run_label
            ),
            run_label=args.run_label,
            backend=args.backend,
            bins=args.bins,
            smoothing_sigma=args.smoothing_sigma,
            pseudocount=args.pseudocount,
        )
        rows.append(row)
        print(
            f"Validation seed {validation_seed}: log-likelihood={score}, "
            f"duration={score_duration}"
        )
    rescore_output = output_dir / f"{output_stem}_rescore.csv"
    pd.DataFrame(rows).to_csv(rescore_output, index=False)
    print(f"Rescore complete: {rescore_output}")
    raise SystemExit(0)

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
df["history_mode"] = history_mode
df["optimizer_seed"] = optimizer_seed
df["simulation_seed"] = simulation_seed
df["bins"] = args.bins
df["smoothing_sigma"] = args.smoothing_sigma
df["pseudocount"] = args.pseudocount
df["parameter_batch_size"] = args.parameter_batch_size
df["triton_block_size"] = args.triton_block_size
df["triton_num_warps"] = args.triton_num_warps
df["triton_maxnreg"] = args.triton_maxnreg
df["run_label"] = args.run_label
df = pd.concat([df, pd.DataFrame(sf_params, index=[0])], axis=1)

if not args.skip_fit_output:
    df.to_csv(
        output_dir / f"{output_stem}.csv",
        index=False,
    )

# -- Posterior predictive checks -----------------------------------------------
sim_posterior_predict = not args.skip_posterior_predictive
n_sim = args.posterior_predictive_simulations
posterior_predictive_seed = args.posterior_predictive_seed
if posterior_predictive_seed is None:
    posterior_predictive_seed = simulation_seed + 1

if sim_posterior_predict:
    posterior_predictive_start = datetime.now()
    # Keep the full trial sequence: the likelihood mask controls fitting only.
    pp_data = data[
        (data.subject_nr == actual_subject_id) &
        data.sequence.isin(["RealRare", "RealFrequent", "NoInstruction"])
    ].reset_index(drop=True)

    if args.backend == "triton":
        # Reuse the fitted model's cached plan. The fitting helper performs the
        # authoritative mapping from optimizer coordinates to scalar and
        # condition-dependent per-trial parameters.
        fit_function = pec.controller.function
        fit_values = tuple(
            optimal_parameters[name] for name in fit_function.fit_param_names
        )
        pp_parameter_set = fit_function._batched_parameter_set(fit_values)
        pp_plan = fit_function._compile_batched_plan()
        pp_result = pp_plan.run(
            fit_function._batched_stimulus_inputs(),
            [pp_parameter_set],
            num_estimates=n_sim,
            seed=posterior_predictive_seed,
            # With one parameter set, common-random-number indexing still gives
            # every estimate an independent stream and reuses the fit's kernel
            # specialization.
            common_random_numbers=True,
            triton_launch_options=fit_function.batched_triton_launch_options,
        )
        outcome_indices = fit_function._batched_outcome_indices(pp_plan)
        selected = np.take(pp_result.values, outcome_indices, axis=-1)
        # Runtime layout is [parameter, subject, trial, estimate, outcome].
        # Put independent simulations first so each CSV replicate contains one
        # complete, stateful trial sequence.
        simulated_outcomes = np.transpose(selected[0, 0], (1, 0, 2))
        if simulated_outcomes.shape != (n_sim, len(pp_data), 2):
            raise RuntimeError(
                "Unexpected posterior-predictive result shape: "
                f"{simulated_outcomes.shape}; expected {(n_sim, len(pp_data), 2)}."
            )
        simulated_outcomes = simulated_outcomes.reshape(-1, 2)
        repeated_pp_data = pd.concat([pp_data] * n_sim, ignore_index=True)
        print(
            f"GPU posterior prediction complete: {n_sim} simulation(s), "
            f"{len(pp_data)} trials each, seed={posterior_predictive_seed}"
        )
    else:
        # Retain the original LLVM path for non-GPU fits.
        comp_pp = make_stab_flex(**sf_params)

        cueStimulusInterval_pp = get_node(comp_pp, "Cue Stimulus Interval")
        controlExecution_pp    = get_node(comp_pp, "Task Activations [C1, C2]")
        thresholdMechanism_pp  = get_node(comp_pp, "Threshold Mechanism")
        decisionMaker_pp       = get_node(comp_pp, "DDM")

        repeated_pp_data = pd.concat([pp_data] * n_sim, ignore_index=True)
        taskSequence_pp = repeated_pp_data[["T1", "T2"]].to_numpy()
        stimulusSequence_pp = repeated_pp_data[["S1", "S2", "S3", "S4"]].to_numpy()
        correctResponseSequence_pp = repeated_pp_data["correct_response"].to_numpy()
        inputs_pp = make_input_dict(
            comp_pp, taskSequence_pp, stimulusSequence_pp, correctResponseSequence_pp
        )

        conditions = repeated_pp_data["sequence"].unique()

        def _select(param_base):
            col_with_cond = f"{param_base}[{conditions[0]}]"
            if col_with_cond in df.columns:
                return np.select(
                    [repeated_pp_data["sequence"] == cond for cond in conditions],
                    [df[f"{param_base}[{cond}]"] for cond in conditions],
                )
            return np.full(len(repeated_pp_data), df[param_base].iloc[0])

        gain_array = _select("Task Activations [C1, C2].gain")
        prep_time_array = _select("Cue Stimulus Interval.slope")
        threshold_array = _select("Threshold Mechanism.intercept")
        collapse_array = _select("Threshold Mechanism.offset-integrator_function")
        ndt_array = _select("DDM.non_decision_time")

        gain_mech = pnl.ControlMechanism(name="Gain Mech", control_signals=("gain", controlExecution_pp), modulation=pnl.OVERRIDE)
        prep_time_mech = pnl.ControlMechanism(name="Prep Time Mech", control_signals=("slope", cueStimulusInterval_pp), modulation=pnl.OVERRIDE)
        threshold_mech = pnl.ControlMechanism(name="Threshold Mech", control_signals=("intercept", thresholdMechanism_pp), modulation=pnl.OVERRIDE)
        collapse_mech = pnl.ControlMechanism(name="Collapse Mech", control_signals=("offset-integrator_function", thresholdMechanism_pp), modulation=pnl.OVERRIDE)
        ndt_mech = pnl.ControlMechanism(name="NDT Mech", control_signals=("non_decision_time", decisionMaker_pp), modulation=pnl.OVERRIDE)
        comp_pp.add_nodes([
            gain_mech,
            prep_time_mech,
            threshold_mech,
            collapse_mech,
            ndt_mech,
        ])

        inputs_pp["Gain Mech"] = [[np.array([v])] for v in gain_array]
        inputs_pp["Prep Time Mech"] = [[np.array([v])] for v in prep_time_array]
        inputs_pp["Threshold Mech"] = [[np.array([v])] for v in threshold_array]
        inputs_pp["Collapse Mech"] = [[np.array([v])] for v in collapse_array]
        inputs_pp["NDT Mech"] = [[np.array([v])] for v in ndt_array]

        comp_pp.run(inputs_pp, execution_mode=pnl.ExecutionMode.LLVMRun)
        simulated_outcomes = np.squeeze(np.array(comp_pp.results))[:, 1:]

    sim_data = pd.DataFrame(
        simulated_outcomes,
        columns=["decision", "response_time"],
    ).assign(
        sequence=repeated_pp_data["sequence"].to_numpy(),
        task_transition=repeated_pp_data["task_transition"].to_numpy(),
        congruence=repeated_pp_data["congruence"].to_numpy(),
        subject_nr=actual_subject_id,
        posterior_predictive_replicate=np.repeat(
            np.arange(n_sim), len(pp_data)
        ),
    )

    posterior_output = output_dir / (
        "expectation_3.2_real_sequences_single_csi_leak12"
        f"_posterior_predict_sub{actual_subject_id}{output_suffix}.csv"
    )
    sim_data.to_csv(posterior_output, index=False)
    print(
        "Simulation Complete! Posterior-predictive duration: "
        f"{datetime.now() - posterior_predictive_start}"
    )
