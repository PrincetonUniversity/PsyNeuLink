import os

import numpy as np

from psyneulink.core.components.functions.integratorfunctions import NavarroAndFuss
from psyneulink.library.components.mechanisms.processing.integrator.ddm import DDM

# Get location of this script so we can write txt files to this directory
# regardless of the working directory. I feel like there must be a better way to
# do this?
__location__ = os.path.dirname(os.path.realpath(__file__))

def gen_matlab_ddm_test_data(non_degenerate_only=False):
    """
    This function generates CSV files that contain samples of random DDM simulation
    parameters and their results evaluated using the NavarroAndFuss() Function.
    NavarroAndFuss() depends on MATLAB and its functionality has been implemented
    completely in BogaczEtAl Function. These text files are used to run compatibility
    checks against the old code without needing MATLAB runtime.

    :param non_degenerate_only: Whether to sample degenerate cases; where
    difussion is close to zero, starting point is close to threshold, cases where
    answer is deterministic, etc.
    :return:
    """
    NF = DDM(
        name='DDM',
        function=NavarroAndFuss()
    )

    NUM_CHECKS = 5000
    res = np.zeros(shape=(NUM_CHECKS, 15))
    rng = np.random.RandomState(100)
    for i in range(NUM_CHECKS):

        if non_degenerate_only:
            r_stim = 1
            r_drift_rate = (rng.rand() - .5) * 1.5
            r_threshold = 1.5 + rng.rand()
            r_bias = rng.uniform(0.01, 0.98)
            r_starting_point = r_threshold * (2 * r_bias - 1)
            r_t0 = rng.rand() * 4
            r_noise = rng.uniform(0.2, 0.5)
        else:
            r_stim = rng.uniform(-5, 5)
            r_drift_rate = rng.uniform(-5, 5)
            r_threshold = rng.uniform(0, 10)
            r_starting_point = rng.uniform(-r_threshold, r_threshold)
            r_bias = (r_starting_point + r_threshold) / (2 * r_threshold)
            r_t0 = rng.uniform(0, 3)
            r_noise = rng.uniform(0, 1)

        NF.function_object.drift_rate=r_stim*r_drift_rate
        NF.function_object.threshold=r_threshold
        NF.function_object.starting_point=r_bias
        NF.function_object.t0=r_t0
        NF.function_object.noise=r_noise

        results_nf = NF.execute(r_stim)
        res[i, :] = np.concatenate(
            ([r_stim, r_drift_rate, r_threshold, r_starting_point, r_bias, r_t0, r_noise], np.squeeze(results_nf[1:6])))

    if non_degenerate_only:
        np.savetxt(os.path.join(__location__, 'matlab_ddm_code_ground_truth_non_degenerate.csv'), res)
    else:
        np.savetxt(os.path.join(__location__, 'matlab_ddm_code_ground_truth.csv'), res)


if __name__ == "__main__":
    gen_matlab_ddm_test_data(non_degenerate_only=True)
    gen_matlab_ddm_test_data(non_degenerate_only=False)