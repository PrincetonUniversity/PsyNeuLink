import numpy as np
import pytest

from psyneulink.core.components.functions.distributionfunctions import DriftDiffusionAnalytical, NavarroAndFuss
from psyneulink.library.components.mechanisms.processing.integrator.ddm import DDM

@pytest.mark.skip(reason="Requires MATLAB engine for NavarroAndFuss, NavarroAndFuss is deprecated as well.")
def test_nf_vs_bogacz():
    """
    This test compares the NavarroAndFuss() and bogaczEtAl (renamed DriftDiffusionAnalytical) against eachother.
    """
    NF = DDM(
        name='DDM',
        function=NavarroAndFuss()
    )

    # Create a DriftDiffusionAnalytical Function, make sure to set shenav_et_al_compat_mode=True to get exact behavior
    # of old MATLAB code (Matlab/DDMFunctions/ddmSimFRG.m)
    B = DDM(
        name='DDM',
        function=DriftDiffusionAnalytical(shenhav_et_al_compat_mode=True)
    )

    NUM_CHECKS = 2500
    rng = np.random.RandomState(100)
    for i in range(NUM_CHECKS):
        r_stim = rng.uniform(-5, 5)
        r_drift_rate = rng.uniform(-5,5)
        r_threshold = rng.uniform(0, 10)
        r_starting_point = rng.uniform(-r_threshold, r_threshold)
        r_bias = (r_starting_point + r_threshold) / (2 * r_threshold)
        r_t0 = rng.uniform(0, 3)
        r_noise = rng.uniform(0,1)

        # These print statements can be useful for debugging. Disable in general though.
        # print("\n{} of {}".format(i, NUM_CHECKS))
        # print("\tr_stim = {}".format(r_stim))
        # print("\tr_drift_rate = {}".format(r_drift_rate))
        # print("\tr_threshold = {}".format(r_threshold))
        # print("\tr_starting_point = {}".format(r_starting_point))
        # print("\tr_bias = {}".format(r_bias))
        # print("\tr_noise = {}".format(r_noise))
        # print("\tr_t0 = {}".format(r_t0))
        # print("ddm_params.z = {threshold}; ddm_params.c = {noise}; ddm_params.T0 = {t0}; [meanERs,meanRTs,meanDTs,condRTs,condVarRTs,condSkewRTs] = ddmSimFRG({drift_rate},{bias}, ddm_params, 1)".format(
        #        threshold=r_threshold, noise=r_noise, bias=r_bias, drift_rate=r_stim * r_drift_rate, t0=r_t0))
        NF.function.drift_rate=r_stim*r_drift_rate # NavarroAndFuss doesn't multiply stimulus to drift, do it here.
        NF.function.threshold=r_threshold
        NF.function.starting_point=r_bias
        NF.function.t0=r_t0
        NF.function.noise=r_noise

        B.function.drift_rate = r_drift_rate
        B.function.threshold = r_threshold
        B.function.starting_point = r_starting_point
        B.function.t0 = r_t0
        B.function.noise = r_noise

        results_nf = NF.execute(r_stim)
        results_b = B.execute(r_stim)

        # Check that all components of the results are close, skip the first one since it is stochastic and should
        # depended on the others. Not the best approach but trouble with getting the same random seeds requires it for
        # now.
        #assert np.allclose(results_b[1:], results_nf[1:], atol=1e-10, equal_nan=True)
        assert np.isclose(results_b[1], results_nf[1], atol=1e-10, equal_nan=True)
        assert np.isclose(results_b[2], results_nf[2], atol=1e-10, equal_nan=True)
        assert np.isclose(results_b[3], results_nf[3], atol=1e-10, equal_nan=True)
        assert np.isclose(results_b[4], results_nf[4], atol=1e-10, equal_nan=True)
        assert np.isclose(results_b[5], results_nf[5], atol=1e-10, equal_nan=True)

        # Comment out tests for some moments, in degenerate cases because of different
        # implementations of trig functions thes can be very different between MATLAB and
        # Python it seems.
        #assert np.isclose(results_b[6], results_nf[6], atol=1e-10, equal_nan=True)
        #assert np.isclose(results_b[7], results_nf[7], atol=1e-10, equal_nan=True)
        #assert np.isclose(results_b[8], results_nf[8], atol=1e-10, equal_nan=True)
        #assert np.isclose(results_b[9], results_nf[9], atol=1e-10, equal_nan=True)

