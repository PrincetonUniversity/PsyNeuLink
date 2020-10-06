import numpy as np
import os

from psyneulink.core.components.functions.distributionfunctions import DriftDiffusionAnalytical
from psyneulink.library.components.mechanisms.processing.integrator.ddm import DDM

# Get location of this script so we can load the txt files present in it regardless of the working
# directory. I feel like there must be a better way to do this?
__location__ = os.path.dirname(os.path.realpath(__file__))

def check_drift_diffusion_analytical(B, data, degenerate_cases=False):
    """
    Helper function to check a DriftDiffusionAnalytical Function against a set of data. Format of the
    data follows the following column ordering:

    stim, drift_rate, threshold, starting_point, bias, t0, noise, mean ER, mean RT,
    correct RT mean, correct RT variance, correct RT skew

    See gen_matlab_ddm_test_data.py script to generate more test data in this form. This script has since
    been deleted, see commit 7c67ca0f2.

    :param B:
    :param data:
    :return:
    """
    NUM_CHECKS = data.shape[0]
    for i in range(NUM_CHECKS):
        r_stim, r_drift_rate, r_threshold, r_starting_point, r_bias, r_t0, r_noise = data[i, 0:7].tolist()
        ground_truth = data[i,7:]

        B.function.drift_rate.base = r_drift_rate
        B.function.threshold.base = r_threshold
        B.function.starting_point.base = r_starting_point
        B.function.t0.base = r_t0
        B.function.noise.base = r_noise

        results_b = B.execute(r_stim)

        # Lets drop the singleton dimension
        results_b = np.squeeze(results_b)

        # Check that all components of the results are close, skip the first one since it is stochastic and should
        # depended on the others. Not the best approach but trouble with getting the same random seeds requires it for
        # now. If we are doing degenerate cases, then don't check conditional moments, these can vary wildly because
        # implementation differences of coth and csch between Python and MATLAB
        if degenerate_cases:
            assert np.allclose(results_b[1:6], ground_truth[0:5], atol=1e-10, equal_nan=True)
        else:
            assert np.allclose(results_b[1:], ground_truth, atol=1e-10, equal_nan=True)

def test_drift_difussion_analytical_shenhav_compat_mode():

    # Create a DriftDiffusionAnalytical Function, make sure to set shenav_et_al_compat_mode=True to get exact behavior
    # of old MATLAB code (Matlab/DDMFunctions/ddmSimFRG.m)
    B = DDM(
        name='DDM',
        function=DriftDiffusionAnalytical(shenhav_et_al_compat_mode=True)
    )

    # Load a CSV containing random sampled test values
    data = np.loadtxt(os.path.join(__location__, 'matlab_ddm_code_ground_truth.csv'))

    check_drift_diffusion_analytical(B, data, degenerate_cases=True)

def test_drift_difussion_analytical():

    # Create a DriftDiffusionAnalytical Function, setting shenav_et_al_compat_mode=False (the defualt) should only
    # really change degenerate input cases, this test tests non-degenerate cases.
    B = DDM(
        name='DDM',
        function=DriftDiffusionAnalytical()
    )

    # Load a CSV containing random sampled test values
    data = np.loadtxt(os.path.join(__location__, 'matlab_ddm_code_ground_truth_non_degenerate.csv'))

    check_drift_diffusion_analytical(B, data, degenerate_cases=True)
