"""Problem and run configuration for the Dask PEC MLE prototype.
"""

# --- Observed-data / simulation sizing ---
NUM_TRIALS = 50            # observed data trials (stimulus presentations)
NUM_ESTIMATES = 1000       # simulations per likelihood evaluation
INITIAL_SEED = 42          # fixed simulation seed (with CRN -> reproducible)

# --- Dask / parallelism ---
N_WORKERS = 4              # Dask workers (one evaluation each, in parallel)
WORKER_CORES = 4           # LLVM threads per worker (estimate-level parallelism)
# Keep this fixed when sweeping N_WORKERS so every run follows the same CMA-ES
# trajectory. Set it >= N_WORKERS if you want every worker busy every round.
OPTIMIZER_POPSIZE = 16     # CMA-ES population / evals per ask-tell round
N_ROUNDS = 30              # CMA-ES generations / ask-tell rounds


def total_evals():
    """Likelihood evaluations implied by the fixed-round budget."""
    return N_ROUNDS * OPTIMIZER_POPSIZE

# --- SLURM / network ---
SLURM_INTERFACE = "ib0"     # network interface for scheduler<->worker (InfiniBand)

# --- dask-jobqueue worker jobs -- TODO(remove-jobqueue) ---
# Everything below this line serves only the deprecated dask-jobqueue mode
# (one SLURM job per worker); dask-srun launches workers inside the existing
# allocation and needs none of it. Delete with the mode.
SLURM_PARTITION = "cpu"
WORKER_MEMORY = "8GB"       # memory per worker job
# Della auto-assigns the QOS from the *walltime* (an explicit --qos is
# overridden), and the QOS sets the per-user concurrent-JOB cap that one-job-
# per-worker scaling runs into:
#     <= 61 min -> test    (2 jobs/user)   <-- too few: blocks scaling
#     <= 24 h   -> short   (300 jobs, 300 cores/user)
#     <= 72 h   -> medium  (100 jobs, 250 cores/user)
#     <= 144 h  -> vlong   (40 jobs,  160 cores/user)
# So request just over an hour to land in `short`. The worker job still ends as
# soon as the fit finishes (Dask closes the cluster); walltime is only a ceiling.
WORKER_WALLTIME = "02:00:00"  # -> `short` QOS; driver alloc must outlast this

# --- Ground-truth DDM parameters used to synthesize the test data ---
TRUE_PARAMS = dict(
    starting_value=0.0, rate=0.3, noise=1.0,
    threshold=0.6, non_decision_time=0.15, time_step_size=0.001,
)

# --- Fit parameters: name -> (low, high) ---
# Insertion order defines the positional argument order expected by
# pec.log_likelihood(*params).
FIT_BOUNDS = {
    "rate": (-0.5, 0.5),
    "threshold": (0.5, 1.0),
    "non_decision_time": (0.0, 1.0),
}
