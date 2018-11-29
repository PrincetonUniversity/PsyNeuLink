import pytest
import os
import numpy as np

from psyneulink.core.components.functions.function import BogaczEtAl
from psyneulink.library.components.mechanisms.processing.integrator.ddm import DDM

# Get location of this script so we can load the txt files present in it regardless of the working
# directory. I feel like there must be a better way to do this?
__location__ = os.path.dirname(os.path.realpath(__file__))

def check_drift_diffusion_analytical(B, data):
    """
    Helper function to check a BogaczEtAl Function against a set of data. Format of the
    data follows the following column ordering:

    stim, drift_rate, threshold, starting_point, bias, t0, noise, mean ER, mean RT,
    correct RT mean, correct RT variance, correct RT skew

    See gen_matlab_ddm_test_data.py script to generate more test data in this form.

    :param B:
    :param data:
    :return:
    """
    NUM_CHECKS = data.shape[0]
    for i in range(NUM_CHECKS):
        r_stim, r_drift_rate, r_threshold, r_starting_point, r_bias, r_t0, r_noise = data[i, 0:7].tolist()
        ground_truth = data[i,7:]

        B.function_object.drift_rate = r_drift_rate
        B.function_object.threshold = r_threshold
        B.function_object.starting_point = r_starting_point
        B.function_object.t0 = r_t0
        B.function_object.noise = r_noise

        results_b = B.execute(r_stim)

        ABS_TOL = 1e-10
        assert np.isclose(results_b[1], ground_truth[0], atol=ABS_TOL, equal_nan=True)
        assert np.isclose(results_b[2], ground_truth[1], atol=ABS_TOL, equal_nan=True)
        assert np.isclose(results_b[3], ground_truth[2], atol=ABS_TOL, equal_nan=True)
        assert np.isclose(results_b[4], ground_truth[3], atol=ABS_TOL, equal_nan=True)
        assert np.isclose(results_b[5], ground_truth[4], atol=ABS_TOL, equal_nan=True)


def test_drift_difussion_analytical_shenhav_compat_mode():

    # Create a BogaczEtAl Function, make sure to set shenav_et_al_compat_mode=True to get exact behavior
    # of old MATLAB code (Matlab/DDMFunctions/ddmSimFRG.m)
    B = DDM(
        name='DDM',
        function=BogaczEtAl(shenhav_et_al_compat_mode=True)
    )

    # Load a CSV containing random sampled test values
    data = np.loadtxt(os.path.join(__location__, 'matlab_ddm_code_ground_truth.csv'))

    check_drift_diffusion_analytical(B, data)

def test_drift_difussion_analytical():

    # Create a BogaczEtAl Function, setting shenav_et_al_compat_mode=False (the defualt) should only
    # really change degenerate input cases, this test tests non-degenerate cases.
    B = DDM(
        name='DDM',
        function=BogaczEtAl()
    )

    # Load a CSV containing random sampled test values
    data = np.loadtxt(os.path.join(__location__, 'matlab_ddm_code_ground_truth_non_degenerate.csv'))

    check_drift_diffusion_analytical(B, data)
