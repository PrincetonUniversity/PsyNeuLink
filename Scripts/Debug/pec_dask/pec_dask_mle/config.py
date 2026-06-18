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
