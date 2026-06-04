"""Synthetic "observed" data generation -- the test harness.

In a real experiment this is replaced by actual subject data; here we generate
data from known ``TRUE_PARAMS`` so we can check whether the fit recovers them
(see ../CONCEPTS.md section 10).
"""

import numpy as np
import pandas as pd
from psyneulink.core.globals.utilities import set_global_seed

from . import config
from .factory import build_model


def make_data():
    """Generate synthetic DDM data on the driver.

    Returns ``(data_df, trial_inputs)`` where ``data_df`` has columns
    ``decision`` (categorical) and ``response_time``.
    """
    set_global_seed(0)  # reproducible synthetic data

    comp, decision = build_model()
    rng = np.random.default_rng(12345)
    trial_inputs = rng.choice([5.0, -5.0], size=(config.NUM_TRIALS, 1), p=[0.1, 0.9])
    comp.run(inputs={decision: trial_inputs})

    data = pd.DataFrame(
        np.squeeze(np.array(comp.results)), columns=["decision", "response_time"]
    )
    data["decision"] = data["decision"].astype("category")
    return data, trial_inputs
