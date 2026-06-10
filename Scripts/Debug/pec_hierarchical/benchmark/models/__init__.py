"""Model providers for the PEC benchmark.

The harness is model-agnostic; each model is a module implementing this small
contract so adding a model = drop one file here + one REGISTRY line:

    TRUE_PARAMS  : dict[str, float]          # name -> true value (for recovery %)
    FIT_BOUNDS   : dict[str, (lo, hi)]       # name -> bounds; order = log_likelihood args
    INITIAL_SEED : int                       # PEC simulation seed (common random numbers)

    build_comp() -> Composition
    make_inputs(comp, num_trials) -> dict    # {input_node: data}, deterministic
    make_data(num_trials) -> DataFrame       # synthetic observed data (runs the model)
    build_pec(comp, data, num_estimates, optimization_function=None) -> PEC

`make_inputs` must be deterministic in num_trials so the driver's data and each
worker's locally-rebuilt inputs match exactly (only the data + scalars cross the
wire; comp/inputs/pec are rebuilt per worker via the provider).
"""

import warnings

# Same third-party FutureWarning silence as pec_dask_mle (runs before pnl import).
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"functools\.partial will be a method descriptor.*",
)

from . import ddm  # noqa: E402
from . import stabflex  # noqa: E402

REGISTRY = {
    "ddm": ddm,
    "stabflex": stabflex,
}


def get(name):
    if name not in REGISTRY:
        raise ValueError(f"unknown model '{name}'; available: {list(REGISTRY)}")
    return REGISTRY[name]
